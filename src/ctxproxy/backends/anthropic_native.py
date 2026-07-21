"""Native Anthropic backend — passthrough with header sanitisation."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping

import httpx

from ..errors import UpstreamError
from ..types_anthropic import MessagesRequest
from .base import Backend

ANTHROPIC_VERSION = "2023-06-01"


class AnthropicBackend(Backend):
    """Forwards to an Anthropic-compatible endpoint unchanged.

    Everything Anthropic-specific — cache_control, thinking blocks, server-side
    context_management, signatures — passes through untouched. This is the path
    where we want to do the *least* work: on a native upstream the model can
    compact server-side, which beats anything we can do client-side.
    """

    async def complete(
        self,
        request: MessagesRequest,
        upstream_model: str,
        client_headers: Mapping[str, str],
    ) -> dict:
        payload = self._payload(request, upstream_model, stream=False)
        return await self._post_json("/v1/messages", payload, self._headers(client_headers))

    async def stream(
        self,
        request: MessagesRequest,
        upstream_model: str,
        client_headers: Mapping[str, str],
        input_tokens: int = 0,  # noqa: ARG002 — upstream reports real counts itself
    ) -> AsyncIterator[bytes]:
        payload = self._payload(request, upstream_model, stream=True)
        headers = self._headers(client_headers)
        headers["accept"] = "text/event-stream"

        try:
            async with self._client.stream(
                "POST", "/v1/messages", json=payload, headers=headers
            ) as response:
                if response.status_code >= 400:
                    body = await response.aread()
                    raise UpstreamError(
                        response.status_code, body.decode(errors="replace"), self.name
                    )
                # Raw passthrough: the upstream already speaks our wire format,
                # so re-parsing would only add latency and a chance to corrupt it.
                async for chunk in response.aiter_raw():
                    yield chunk
        except httpx.RequestError as exc:
            raise UpstreamError(502, f"{type(exc).__name__}: {exc}", self.name) from exc

    async def count_tokens(self, request: MessagesRequest, upstream_model: str) -> int:
        payload = {
            "model": upstream_model,
            "messages": [m.model_dump(exclude_none=True) for m in request.messages],
        }
        if request.system is not None:
            payload["system"] = request.system
        if request.tools:
            payload["tools"] = request.tools

        data = await self._post_json(
            "/v1/messages/count_tokens", payload, self._headers({})
        )
        return int(data.get("input_tokens", 0))

    # -- internals ---------------------------------------------------------- #

    def _payload(self, request: MessagesRequest, upstream_model: str, *, stream: bool) -> dict:
        payload = request.model_dump(exclude_none=True)
        payload["model"] = upstream_model
        payload["stream"] = stream
        return payload

    def _headers(self, client_headers: Mapping[str, str]) -> dict[str, str]:
        headers = self._forwarded_headers(client_headers)
        headers.setdefault("anthropic-version", ANTHROPIC_VERSION)

        if key := self.config.resolve_api_key():
            headers["x-api-key"] = key
        else:
            # No configured credential: forward the client's. This is the
            # common local setup, where the developer's own key is already in
            # ANTHROPIC_API_KEY and the proxy has none of its own.
            for name in ("x-api-key", "authorization"):
                if value := _find(client_headers, name):
                    headers[name] = value
        return headers


def _find(headers: Mapping[str, str], name: str) -> str | None:
    for key, value in headers.items():
        if key.lower() == name:
            return value
    return None
