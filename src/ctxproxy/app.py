"""FastAPI application factory."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .config import Config
from .errors import ProxyError
from .routes import router
from .state import AppState

log = logging.getLogger(__name__)


async def _prune(state, ttl_hours: int, archive_ttl_hours: int | None = None) -> None:
    """Delete expired ledgers and their archives.

    Archives are pruned alongside ledgers rather than on their own schedule:
    an archive without its ledger is unreachable (`ctxproxy archive` is keyed
    by session), so outliving one is pure waste.
    """
    try:
        ledgers = await state.store.prune(ttl_hours)
        archives = await asyncio.to_thread(
            state.archive.prune, archive_ttl_hours or ttl_hours
        )
        if ledgers or archives:
            log.info(
                "retention: removed %d ledger(s) (>%dh) and %d archive(s) (>%dh)",
                ledgers,
                ttl_hours,
                archives,
                archive_ttl_hours or ttl_hours,
            )
    except Exception:  # noqa: BLE001 — housekeeping must never take the proxy down
        log.exception("retention sweep failed")


async def _prune_periodically(state, config: Config) -> None:
    interval = config.server.prune_interval_hours * 3600
    while True:
        await asyncio.sleep(interval)
        await _prune(
            state,
            config.server.session_ttl_hours,
            config.server.effective_archive_ttl_hours,
        )


def create_app(config: Config) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state = AppState(config)
        app.state.ctx = state

        await _prune(
            state,
            config.server.session_ttl_hours,
            config.server.effective_archive_ttl_hours,
        )

        # Startup-only pruning leaves a proxy that runs for weeks accumulating
        # state forever, which is exactly the long-session case this exists for.
        sweeper = (
            asyncio.create_task(_prune_periodically(state, config))
            if config.server.prune_interval_hours > 0
            else None
        )

        log.info(
            "ctxproxy listening on http://%s:%d — %d backend(s), %d profile(s)",
            config.server.host,
            config.server.port,
            len(config.backends),
            len(config.profiles),
        )
        for profile in config.profiles:
            log.info(
                "  profile %-24s -> %-16s window=%-9d native=%s",
                profile.match,
                profile.backend,
                profile.context_window,
                profile.native_anthropic,
            )

        try:
            yield
        finally:
            if sweeper is not None:
                sweeper.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await sweeper
            await state.aclose()

    app = FastAPI(
        title="ctxproxy",
        description="Context-management proxy for Claude Code",
        version="0.1.0",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
    )

    @app.exception_handler(ProxyError)
    async def _proxy_error(_: Request, exc: ProxyError) -> JSONResponse:
        return exc.to_response()

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        # Anything reaching here is a proxy bug. Claude Code parses the
        # Anthropic error envelope, so returning FastAPI's default
        # {"detail": ...} would surface as an unhelpful generic failure.
        log.exception("unhandled error")
        return JSONResponse(
            status_code=500,
            content={
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": f"[ctxproxy] internal error: {type(exc).__name__}: {exc}",
                },
            },
        )

    app.include_router(router)
    return app


def app_factory() -> FastAPI:
    """Entry point for `uvicorn --reload`.

    The reloader runs the app in a subprocess and re-imports on every change,
    so it needs an import string rather than a constructed app. Config is
    re-read here rather than captured, which is what makes edits to
    ctxproxy.yaml take effect on reload too — not just source changes.

    A config that fails to load must NOT kill the worker. Letting it raise
    leaves the port closed, so every client request fails at the TCP level —
    and a client seeing connection-refused reports it as the *model* being
    unavailable, which sends you looking at the wrong system entirely. Staying
    up and answering with the actual parse error is far easier to diagnose, and
    fixing the file triggers another reload that recovers automatically.
    """
    from .config import load_config
    from .logging import configure

    try:
        config = load_config()
    except Exception as exc:  # noqa: BLE001
        configure("info", "text")
        log.error("CONFIG INVALID — serving errors until it is fixed:\n%s", exc)
        return _degraded_app(str(exc))

    configure(config.server.log_level, config.server.log_format)
    return create_app(config)


def _degraded_app(reason: str) -> FastAPI:
    """A stand-in that stays bound to the port and explains itself."""
    app = FastAPI(title="ctxproxy (config error)", docs_url=None, redoc_url=None)

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={"status": "config_error", "detail": reason},
        )

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
    async def catch_all(path: str) -> JSONResponse:  # noqa: ARG001
        return JSONResponse(
            status_code=503,
            content={
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": (
                        "[ctxproxy] configuration is invalid, so no requests can be "
                        f"served. This is a proxy problem, not a model outage. "
                        f"Fix ctxproxy.yaml and it will reload automatically.\n\n{reason}"
                    ),
                },
            },
        )

    return app
