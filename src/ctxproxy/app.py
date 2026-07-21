"""FastAPI application factory."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .config import Config
from .errors import ProxyError
from .routes import router
from .state import AppState

log = logging.getLogger(__name__)


def create_app(config: Config) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state = AppState(config)
        app.state.ctx = state

        pruned = await state.store.prune(config.server.session_ttl_hours)
        if pruned:
            log.info("pruned %d expired session ledger(s)", pruned)

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
    """
    from .config import load_config
    from .logging import configure

    config = load_config()
    configure(config.server.log_level, config.server.log_format)
    return create_app(config)
