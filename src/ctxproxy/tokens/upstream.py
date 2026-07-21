"""Token counting delegated to a native Anthropic upstream."""

from __future__ import annotations

import logging

from ..types_anthropic import MessagesRequest
from .base import TokenCounter

log = logging.getLogger(__name__)


class UpstreamCounter(TokenCounter):
    """Calls the backend's /v1/messages/count_tokens; falls back on failure.

    A fallback is mandatory rather than nice-to-have: if the gateway does not
    serve count_tokens (a very common LiteLLM/vLLM gap), a hard dependency here
    means the proxy stops managing context at exactly the moment it is needed.
    """

    def __init__(self, backend, upstream_model: str, fallback: TokenCounter) -> None:
        super().__init__()
        self._backend = backend
        self._upstream_model = upstream_model
        self._fallback = fallback
        self._degraded = False

    def count_text(self, text: str) -> int:
        return self._fallback.count_text(text)

    async def count_request(self, request: MessagesRequest) -> int:
        if self._degraded:
            return await self._fallback.count_request(request)

        try:
            return await self._backend.count_tokens(request, self._upstream_model)
        except Exception as exc:  # noqa: BLE001 — any failure must degrade, not raise
            self._degraded = True
            log.warning(
                "upstream count_tokens failed (%s); using local counting for the "
                "rest of this process. Set supports_count_tokens: false in the "
                "profile to silence this.",
                exc,
            )
            return await self._fallback.count_request(request)
