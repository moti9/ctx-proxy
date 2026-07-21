"""OpenAI-compatible backend — LiteLLM, vLLM, or any /v1/chat/completions gateway."""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator, Callable, Mapping

import httpx

from ..errors import UpstreamError
from ..translate.request import anthropic_to_openai
from ..translate.response import openai_to_anthropic
from ..translate.sse import OpenAIStreamTranslator, parse_sse_line, sse_event
from ..types_anthropic import MessagesRequest
from .base import Backend

log = logging.getLogger(__name__)

# Fields the proxy owns; openai_extra_params must not overwrite them.
_RESERVED = frozenset({"model", "messages", "stream", "tools"})


class OpenAICompatBackend(Backend):
    """Translates in both directions so Claude Code never sees the difference.

    Anthropic-specific request features that have no OpenAI equivalent
    (cache_control, thinking config, context_management, beta flags) are dropped
    during translation rather than forwarded — sending them is what produces the
    "invalid beta flag" and unknown-field errors on non-Anthropic upstreams.
    """

    async def complete(
        self,
        request: MessagesRequest,
        upstream_model: str,
        client_headers: Mapping[str, str],
    ) -> dict:
        payload = anthropic_to_openai(
            request,
            model=upstream_model,
            max_tokens_field=self.config.openai_max_tokens_field,
            include_usage=False,
        )
        payload = self._with_extra_params(payload)
        data = await self._post_json(
            "/v1/chat/completions", payload, self._headers(client_headers)
        )
        result = openai_to_anthropic(
            data,
            model=request.model,
            reasoning_output=self.config.openai_reasoning_output,
        )

        if _is_empty_turn(result):
            # Observed occasionally from vLLM: HTTP 200, finish_reason "stop",
            # but content and reasoning_content are both empty. The client just
            # sees the assistant say nothing, which is near-impossible to
            # diagnose after the fact — so make it visible here.
            log.warning(
                "backend %s returned an empty turn (finish_reason=%s, usage=%s) — "
                "upstream produced no content",
                self.name,
                (data.get("choices") or [{}])[0].get("finish_reason"),
                data.get("usage"),
            )

        return result

    async def stream(
        self,
        request: MessagesRequest,
        upstream_model: str,
        client_headers: Mapping[str, str],
        input_tokens: int = 0,
        on_usage: Callable[[int], None] | None = None,
    ) -> AsyncIterator[bytes]:
        payload = anthropic_to_openai(
            request,
            model=upstream_model,
            max_tokens_field=self.config.openai_max_tokens_field,
            include_usage=self.config.openai_stream_usage,
        )
        payload = self._with_extra_params(payload)
        headers = self._headers(client_headers)
        headers["accept"] = "text/event-stream"

        translator = OpenAIStreamTranslator(
            model=request.model,
            message_id=f"msg_{uuid.uuid4().hex[:24]}",
            input_tokens=input_tokens,
            reasoning_output=self.config.openai_reasoning_output,
        )

        try:
            async with self._client.stream(
                "POST", "/v1/chat/completions", json=payload, headers=headers
            ) as response:
                if response.status_code >= 400:
                    body = await response.aread()
                    raise UpstreamError(
                        response.status_code, body.decode(errors="replace"), self.name
                    )

                for event in translator.start():
                    yield event

                async for line in response.aiter_lines():
                    chunk = parse_sse_line(line)
                    if chunk is None:
                        continue
                    for event in translator.handle_chunk(chunk):
                        yield event

                for event in translator.finish():
                    yield event

                # Claude Code streams almost everything, so this is where
                # token-drift samples actually come from.
                if on_usage and translator.upstream_input_tokens:
                    on_usage(translator.upstream_input_tokens)

        except UpstreamError:
            raise
        except httpx.RequestError as exc:
            # The stream is already open from the client's perspective, so an
            # HTTP-level failure would hang it. Terminate the SSE properly
            # instead: an error event is recoverable, a truncated stream is not.
            log.warning("stream transport failure on %s: %s", self.name, exc)
            for event in translator.error(f"{type(exc).__name__}: {exc}"):
                yield event
            yield sse_event("message_stop", {"type": "message_stop"})

    async def count_tokens(self, request: MessagesRequest, upstream_model: str) -> int:
        raise NotImplementedError(
            f"backend {self.name!r} is OpenAI-compatible and has no count_tokens "
            f"endpoint; set supports_count_tokens: false on the profile"
        )

    def _with_extra_params(self, payload: dict) -> dict:
        extra = self.config.openai_extra_params
        if not extra:
            return payload
        if clobbered := _RESERVED & extra.keys():
            log.warning(
                "backend %s: ignoring openai_extra_params %s — the proxy owns those fields",
                self.name,
                sorted(clobbered),
            )
        return {**payload, **{k: v for k, v in extra.items() if k not in _RESERVED}}

    def _headers(self, client_headers: Mapping[str, str]) -> dict[str, str]:
        headers = self._forwarded_headers(client_headers)
        # Anthropic-only headers are meaningless here and some gateways reject
        # unknown ones outright.
        headers.pop("anthropic-beta", None)
        headers.pop("anthropic-version", None)

        if key := self.config.resolve_api_key():
            headers["authorization"] = f"Bearer {key}"
        else:
            for name in ("authorization", "x-api-key"):
                if value := _find(client_headers, name):
                    headers["authorization"] = (
                        value if value.lower().startswith("bearer ") else f"Bearer {value}"
                    )
                    break
        return headers


def _find(headers: Mapping[str, str], name: str) -> str | None:
    for key, value in headers.items():
        if key.lower() == name:
            return value
    return None


def _is_empty_turn(message: dict) -> bool:
    """True when a translated response carries no usable content."""
    blocks = message.get("content") or []
    if not blocks:
        return True
    return all(
        block.get("type") == "text" and not (block.get("text") or "").strip()
        for block in blocks
    )
