"""Command line interface."""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

import typer

from . import __version__
from .config import Config, load_config
from .logging import configure

app = typer.Typer(
    add_completion=False,
    help="Context-management proxy for Claude Code.",
    no_args_is_help=True,
)

ConfigOpt = typer.Option(None, "--config", "-c", help="Path to ctxproxy.yaml")


@app.command()
def serve(
    config_path: Path = ConfigOpt,
    host: str = typer.Option(None, help="Override server.host"),
    port: int = typer.Option(None, help="Override server.port"),
    log_level: str = typer.Option(None, help="Override server.log_level"),
    reload: bool = typer.Option(
        False, "--reload", help="Restart on source or config change (development)"
    ),
) -> None:
    """Run the proxy."""
    import os

    import uvicorn

    from .app import create_app
    from .config import resolve_config_path

    config = _load(config_path)
    if host:
        config.server.host = host
    if port:
        config.server.port = port
    if log_level:
        config.server.log_level = log_level

    configure(config.server.log_level, config.server.log_format)

    if reload:
        # The reloader spawns a subprocess and re-imports, so it needs an
        # import string. Pin the config path through the environment — the
        # subprocess does not inherit our parsed config, and without this it
        # would re-resolve against its own cwd.
        os.environ["CTXPROXY_CONFIG"] = str(resolve_config_path(config_path).resolve())
        package_dir = Path(__file__).parent
        uvicorn.run(
            "ctxproxy.app:app_factory",
            factory=True,
            reload=True,
            reload_dirs=[str(package_dir), str(Path(os.environ["CTXPROXY_CONFIG"]).parent)],
            # Default reload only watches *.py; config edits should count too.
            reload_includes=["*.py", "*.yaml", "*.yml"],
            host=config.server.host,
            port=config.server.port,
            log_config=None,
            access_log=False,
        )
        return

    uvicorn.run(
        create_app(config),
        host=config.server.host,
        port=config.server.port,
        log_config=None,
        access_log=False,
    )


@app.command()
def doctor(config_path: Path = ConfigOpt) -> None:
    """Validate configuration and probe every backend.

    Run this before pointing Claude Code at the proxy — most "it just fails"
    reports are a missing API key or a `context_window` that does not match the
    model actually being served.
    """
    configure("info", "text")
    config = _load(config_path)

    typer.secho(f"ctxproxy {__version__}", bold=True)
    typer.echo(f"state dir: {config.server.state_dir}")
    typer.echo("")

    ok = True

    typer.secho("Backends", bold=True)
    for backend in config.backends:
        key = backend.resolve_api_key()
        if key:
            credential = f"set ({backend.api_key_env or 'inline'})"
        elif backend.api_key_env:
            credential = f"MISSING — ${backend.api_key_env} is not set"
            ok = False
        else:
            credential = "none (will forward the client's)"
        typer.echo(f"  {backend.name:<16} {backend.kind:<10} {backend.base_url}")
        typer.echo(f"  {'':<16} credential: {credential}")
    typer.echo("")

    typer.secho("Profiles", bold=True)
    for profile in config.profiles:
        budget = (
            profile.context_window - profile.output_reserve - profile.safety_buffer
        )
        flag = "" if budget > 0 else "  <-- NEGATIVE BUDGET"
        if budget <= 0:
            ok = False
        typer.echo(
            f"  {profile.match:<24} -> {profile.backend:<14} "
            f"window={profile.context_window:<9} usable={budget}{flag}"
        )
        if profile.native_anthropic and profile.context_window > 200_000:
            typer.secho(
                f"    note: {profile.context_window} needs 1M-context entitlement; "
                f"if requests fail, set 200000",
                fg=typer.colors.YELLOW,
            )
        if not profile.native_anthropic and profile.supports_count_tokens:
            typer.secho(
                "    warning: non-native profile with supports_count_tokens: true — "
                "set it to false unless the gateway really serves that endpoint",
                fg=typer.colors.YELLOW,
            )
    typer.echo("")

    typer.secho("Connectivity", bold=True)
    asyncio.run(_probe(config))

    typer.echo("")
    if ok:
        typer.secho("Configuration looks valid.", fg=typer.colors.GREEN, bold=True)
    else:
        typer.secho("Problems found — see above.", fg=typer.colors.RED, bold=True)
        raise typer.Exit(1)


async def _probe(config: Config) -> None:
    import httpx

    for backend in config.backends:
        url = backend.base_url.rstrip("/")
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                response = await client.get(url, follow_redirects=True)
            typer.secho(
                f"  {backend.name:<16} reachable (HTTP {response.status_code})",
                fg=typer.colors.GREEN,
            )
        except Exception as exc:  # noqa: BLE001
            typer.secho(
                f"  {backend.name:<16} UNREACHABLE: {type(exc).__name__}: {exc}",
                fg=typer.colors.RED,
            )


@app.command()
def sessions(
    config_path: Path = ConfigOpt,
    limit: int = typer.Option(20, help="Rows to show"),
) -> None:
    """List tracked sessions and what compaction has saved."""
    configure("warning", "text")
    config = _load(config_path)

    from .store.file import FileLedgerStore

    store = FileLedgerStore(config.server.state_dir)
    ledgers = asyncio.run(store.list_all())

    if not ledgers:
        typer.echo("No sessions recorded yet.")
        return

    typer.echo(
        f"{'SESSION':<28} {'TURNS':>6} {'COMPACT':>8} {'SAVED':>10} {'DRIFT':>7}  UPDATED"
    )
    for ledger in ledgers[:limit]:
        drift = ledger.median_token_drift
        drift_col = f"{drift:.2f}x" if drift else "-"
        typer.echo(
            f"{ledger.session_key[:28]:<28} {ledger.turns_seen:>6} "
            f"{ledger.compaction_count:>8} {ledger.total_tokens_saved:>10} "
            f"{drift_col:>7}  {ledger.updated_at:%Y-%m-%d %H:%M}"
        )

    worst = [item.median_token_drift for item in ledgers if item.median_token_drift]
    if worst and max(worst) > 1.10:
        typer.secho(
            f"\nDRIFT > 1.10x: token estimates run low, so compaction fires late. "
            f"Raise tokenizer.safety_multiplier to about {max(worst) * 1.05:.2f}.",
            fg=typer.colors.YELLOW,
        )

    total = sum(item.total_tokens_saved for item in ledgers)
    typer.echo(f"\n{len(ledgers)} session(s), {total:,} tokens reclaimed.")


@app.command()
def inspect(
    session_key: str = typer.Argument(..., help="Session key (see `ctxproxy sessions`)"),
    config_path: Path = ConfigOpt,
) -> None:
    """Print a session's rolling summary — what survived compaction."""
    configure("warning", "text")
    config = _load(config_path)

    from .store.file import FileLedgerStore

    store = FileLedgerStore(config.server.state_dir)
    ledger = asyncio.run(store.load(session_key))
    if ledger is None:
        typer.secho(f"No ledger for {session_key!r}", fg=typer.colors.RED)
        raise typer.Exit(1)

    typer.echo(json.dumps(ledger.stats(), indent=2))
    typer.echo("")
    typer.secho("--- rolling summary ---", bold=True)
    typer.echo(ledger.render_summary_block())


@app.command()
def archive(
    session_key: str = typer.Argument(..., help="Session key (see `ctxproxy sessions`)"),
    config_path: Path = ConfigOpt,
    show: int = typer.Option(0, help="Print the first N messages of each fold"),
) -> None:
    """Show the raw messages compaction folded away.

    The summary is what the model sees; this is what it was made from. Use it
    when a session behaves as though it forgot something.
    """
    configure("warning", "text")
    config = _load(config_path)

    from .store.archive import FoldArchive

    entries = FoldArchive(config.server.state_dir / "archive").read(session_key)
    if not entries:
        typer.echo(f"No archived folds for {session_key!r}.")
        typer.echo("Either it never compacted, or state was reset.")
        return

    for i, entry in enumerate(entries, 1):
        lo, hi = entry["original_range"]
        typer.secho(
            f"\nFold {i}: original messages [{lo}:{hi}] "
            f"({entry['message_count']} messages) at {entry['archived_at']}",
            bold=True,
        )
        for msg in entry["messages"][:show]:
            content = msg.get("content")
            text = content if isinstance(content, str) else json.dumps(content)[:300]
            typer.echo(f"  [{msg.get('role')}] {text[:300]}")

    total = sum(e["message_count"] for e in entries)
    typer.echo(f"\n{len(entries)} fold(s), {total} messages archived.")
    if not show:
        typer.echo("Pass --show N to print message bodies.")


@app.command()
def capture(
    session_key: str = typer.Argument(..., help="Session key (see `ctxproxy sessions`)"),
    config_path: Path = ConfigOpt,
    last: int = typer.Option(5, help="Show the most recent N exchanges"),
    full: bool = typer.Option(False, "--full", help="Print the whole upstream payload as JSON"),
) -> None:
    """Show what was actually sent upstream and what came back.

    Requires ``server.debug_capture_dir`` to have been set while the session
    ran. This is the ground truth for "did the proxy corrupt the request, or is
    the model just doing less than asked": the payload here is the exact JSON the
    backend received, translated out of the Anthropic request.
    """
    configure("warning", "text")
    config = _load(config_path)

    if config.server.debug_capture_dir is None:
        typer.secho(
            "server.debug_capture_dir is not set, so nothing was captured.\n"
            "Set it in ctxproxy.yaml, restart, reproduce the issue, then re-run this.",
            fg=typer.colors.YELLOW,
        )
        raise typer.Exit(1)

    from .store.capture import DebugCapture

    entries = DebugCapture(config.server.debug_capture_dir).read(session_key)
    if not entries:
        typer.echo(f"No captured exchanges for {session_key!r}.")
        return

    for i, entry in enumerate(entries[-last:], 1):
        typer.secho(
            f"\nExchange {i} [{entry.get('kind')}] {entry.get('captured_at')} "
            f"-> {entry.get('upstream_model')} "
            f"({entry.get('message_count')} msgs, max_tokens={entry.get('max_tokens')})",
            bold=True,
        )
        if full:
            typer.echo(json.dumps(entry.get("payload"), indent=2))
        else:
            for msg in entry.get("payload", {}).get("messages", []):
                role = msg.get("role")
                content = msg.get("content")
                text = content if isinstance(content, str) else json.dumps(content)
                tool = " +tool_calls" if msg.get("tool_calls") else ""
                typer.echo(f"  [{role}{tool}] {(text or '')[:200]}")

        response = entry.get("response") or {}
        if isinstance(response, dict):
            finish = response.get("finish_reason") or (
                (response.get("choices") or [{}])[0].get("finish_reason")
            )
            if finish:
                colour = typer.colors.RED if finish == "length" else typer.colors.GREEN
                typer.secho(f"  -> finish_reason={finish}", fg=colour)

    truncated = sum(
        1
        for e in entries
        if _finish_reason(e) == "length"
    )
    typer.echo(f"\n{len(entries)} exchange(s) captured; {truncated} ended in truncation (length).")
    if truncated:
        typer.secho(
            "Truncation (finish_reason=length) means the model hit its output cap "
            "mid-turn — a direct cause of completing only part of a request.",
            fg=typer.colors.YELLOW,
        )


def _finish_reason(entry: dict) -> str | None:
    response = entry.get("response") or {}
    if not isinstance(response, dict):
        return None
    return response.get("finish_reason") or (
        (response.get("choices") or [{}])[0].get("finish_reason")
    )


@app.command()
def reset(
    session_key: str = typer.Argument(None, help="Session to clear; omit for all"),
    config_path: Path = ConfigOpt,
) -> None:
    """Delete session state. The next turn starts from a clean ledger."""
    configure("warning", "text")
    config = _load(config_path)

    from .store.file import FileLedgerStore

    store = FileLedgerStore(config.server.state_dir)
    archive_dir = config.server.state_dir / "archive"
    if session_key:
        removed = asyncio.run(store.delete(session_key))
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in session_key)[:120]
        (archive_dir / f"{safe}.jsonl").unlink(missing_ok=True)
        typer.echo("Deleted." if removed else "No such session.")
    else:
        typer.confirm(f"Delete all ledgers in {store.directory}?", abort=True)
        count = len(list(store.directory.glob("*.json")))
        for path in store.directory.glob("*.json"):
            path.unlink()
        if archive_dir.is_dir():
            for path in archive_dir.glob("*.jsonl"):
                path.unlink()
        typer.echo(f"Deleted {count} ledger(s) and their archives.")


@app.command()
def init(
    path: Path = typer.Option(Path("ctxproxy.yaml"), "--path", "-p"),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing file"),
) -> None:
    """Write a starter config file."""
    template = Path(__file__).parent / "config.example.yaml"
    if path.exists() and not force:
        typer.secho(f"{path} already exists (use --force to overwrite)", fg=typer.colors.RED)
        raise typer.Exit(1)
    shutil.copy(template, path)
    typer.secho(f"Wrote {path}", fg=typer.colors.GREEN)
    typer.echo("Edit it, then run: ctxproxy doctor")


@app.command()
def version() -> None:
    """Print the version."""
    typer.echo(__version__)


def _load(path: Path | None) -> Config:
    try:
        return load_config(path)
    except FileNotFoundError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(1) from exc
    except Exception as exc:  # noqa: BLE001
        typer.secho(f"Invalid config: {exc}", fg=typer.colors.RED)
        raise typer.Exit(1) from exc


if __name__ == "__main__":
    app()
