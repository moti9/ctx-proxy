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

    typer.echo(f"{'SESSION':<30} {'TURNS':>6} {'COMPACT':>8} {'SAVED':>10}  UPDATED")
    for ledger in ledgers[:limit]:
        typer.echo(
            f"{ledger.session_key[:30]:<30} {ledger.turns_seen:>6} "
            f"{ledger.compaction_count:>8} {ledger.total_tokens_saved:>10}  "
            f"{ledger.updated_at:%Y-%m-%d %H:%M}"
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
def reset(
    session_key: str = typer.Argument(None, help="Session to clear; omit for all"),
    config_path: Path = ConfigOpt,
) -> None:
    """Delete session state. The next turn starts from a clean ledger."""
    configure("warning", "text")
    config = _load(config_path)

    from .store.file import FileLedgerStore

    store = FileLedgerStore(config.server.state_dir)
    if session_key:
        removed = asyncio.run(store.delete(session_key))
        typer.echo("Deleted." if removed else "No such session.")
    else:
        typer.confirm(f"Delete all ledgers in {store.directory}?", abort=True)
        count = len(list(store.directory.glob("*.json")))
        for path in store.directory.glob("*.json"):
            path.unlink()
        typer.echo(f"Deleted {count} ledger(s).")


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
