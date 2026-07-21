"""Backend interface and shared HTTP plumbing."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Mapping

import httpx

from ..config import BackendConfig
from ..errors import UpstreamError
from ..types_anthropic import MessagesRequest

log = logging.getLogger(__name__)

# Client headers that must never be forwarded verbatim — they describe the
# hop between Claude Code and us, not between us and the upstream.
HOP_BY_HOP = frozenset(
    {
        "host",
        "content-length",
        "connection",
        "keep-alive",
        "transfer-encoding",
        "upgrade",
        "accept-encoding",
        "te",
        "trailer",
        "proxy-authorization",
        "proxy-authenticate",
    }
)

# Auth headers we replace with the backend's own credentials.
AUTH_HEADERS = frozenset({"x-api-key", "authorization", "api-key"})


class Backend(ABC):
    """An upstream that speaks (or is made to speak) the Anthropic Messages API."""

    def __init__(self, config: BackendConfig) -> None:
        self.config = config
        self._client = httpx.AsyncClient(
            base_url=config.base_url.rstrip("/"),
            timeout=httpx.Timeout(
                config.timeout_s,
                connect=config.connect_timeout_s,
            ),
            follow_redirects=True,
        )

    @property
    def name(self) -> str:
        return self.config.name

    @abstractmethod
    async def complete(
        self,
        request: MessagesRequest,
        upstream_model: str,
        client_headers: Mapping[str, str],
    ) -> dict:
        """Non-streaming call. Returns an Anthropic-shaped message object."""

    @abstractmethod
    def stream(
        self,
        request: MessagesRequest,
        upstream_model: str,
        client_headers: Mapping[str, str],
        input_tokens: int = 0,
    ) -> AsyncIterator[bytes]:
        """Streaming call. Yields Anthropic-shaped SSE bytes."""

    @abstractmethod
    async def count_tokens(self, request: MessagesRequest, upstream_model: str) -> int: ...

    async def text_completion(self, request: MessagesRequest, upstream_model: str) -> str:
        """Convenience used by the summariser: run a request, return the text."""
        response = await self.complete(request, upstream_model, {})
        return "".join(
            block.get("text", "")
            for block in response.get("content", [])
            if block.get("type") == "text"
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    # -- shared helpers ----------------------------------------------------- #

    def _forwarded_headers(self, client_headers: Mapping[str, str]) -> dict[str, str]:
        """Client headers minus hop-by-hop, auth, and disallowed beta flags."""
        out: dict[str, str] = {}
        for key, value in client_headers.items():
            lowered = key.lower()
            if lowered in HOP_BY_HOP or lowered in AUTH_HEADERS:
                continue
            if lowered == "anthropic-beta":
                if filtered := self._filter_beta(value):
                    out["anthropic-beta"] = filtered
                continue
            if lowered == "content-type":
                continue
            out[key] = value

        out["content-type"] = "application/json"
        out.update(self.config.extra_headers)
        return out

    def _filter_beta(self, header_value: str) -> str:
        """Drop beta flags this upstream does not understand.

        Claude Code attaches Anthropic experimental flags on every request.
        Non-Anthropic providers reject unknown ones outright, which surfaces as
        a confusing "invalid beta flag" failure on every single call.
        """
        flags = [f.strip() for f in header_value.split(",") if f.strip()]
        allowed = [f for f in flags if self.config.allows_beta(f)]
        if dropped := set(flags) - set(allowed):
            log.debug("backend %s: dropped beta flags %s", self.name, sorted(dropped))
        return ",".join(allowed)

    async def _post_json(self, path: str, payload: dict, headers: dict) -> dict:
        try:
            response = await self._client.post(path, json=payload, headers=headers)
        except httpx.RequestError as exc:
            raise UpstreamError(502, f"{type(exc).__name__}: {exc}", self.name) from exc

        if response.status_code >= 400:
            raise UpstreamError(response.status_code, _body(response), self.name)
        return response.json()


def _body(response: httpx.Response):
    try:
        return response.json()
    except ValueError:
        return response.text
