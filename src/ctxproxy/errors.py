"""Anthropic-shaped error envelopes.

Claude Code parses upstream errors. Returning a FastAPI default
``{"detail": ...}`` body makes it report something unhelpful, so every error we
surface is wrapped in the Anthropic error shape it expects.
"""

from __future__ import annotations

import re
from typing import Any

from fastapi.responses import JSONResponse

# Substrings that mean "the prompt did not fit", across Anthropic, OpenAI,
# vLLM, LiteLLM and the common gateways. Matched case-insensitively against the
# full upstream error body.
_OVERFLOW_PATTERNS = (
    r"context[_ ]window",
    r"context length",
    r"maximum context",
    r"model_context_window_exceeded",
    r"prompt is too long",
    r"too many tokens",
    r"reduce the length of the messages",
    r"input length and `max_tokens` exceed",
    r"string too long",
    r"exceeds the maximum",
)

_OVERFLOW_RE = re.compile("|".join(_OVERFLOW_PATTERNS), re.IGNORECASE)


class ProxyError(Exception):
    """Base for errors we surface to the client in Anthropic shape."""

    status_code = 500
    error_type = "api_error"

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)

    def to_response(self) -> JSONResponse:
        return JSONResponse(
            status_code=self.status_code,
            content={
                "type": "error",
                "error": {"type": self.error_type, "message": self.message},
            },
        )


class ConfigurationError(ProxyError):
    status_code = 400
    error_type = "invalid_request_error"


class UpstreamError(ProxyError):
    """Non-2xx from the backend. Body is forwarded as-is when it is already
    Anthropic-shaped, so client-side error handling keeps working."""

    def __init__(self, status_code: int, body: Any, backend: str) -> None:
        self.status_code = status_code
        self.body = body
        self.backend = backend
        super().__init__(f"upstream {backend} returned {status_code}")

    @property
    def text(self) -> str:
        if isinstance(self.body, str):
            return self.body
        return repr(self.body)

    def is_context_overflow(self) -> bool:
        return bool(_OVERFLOW_RE.search(self.text))

    def is_retryable_upstream(self) -> bool:
        """A capacity/availability failure worth failing over to another model.

        Deliberately narrow: 4xx client errors (bad request, auth, not found)
        would fail the same way on any backend, so failing over would only hide
        the real problem. These statuses mean "this deployment can't serve you
        right now", which is exactly when another model can.
        """
        return self.status_code in (429, 502, 503, 529)

    def to_response(self) -> JSONResponse:
        if isinstance(self.body, dict) and self.body.get("type") == "error":
            return JSONResponse(status_code=self.status_code, content=self.body)
        return JSONResponse(
            status_code=self.status_code,
            content={
                "type": "error",
                "error": {
                    "type": _type_for_status(self.status_code),
                    "message": f"[ctxproxy->{self.backend}] {self.text}",
                },
            },
        )


def _type_for_status(status: int) -> str:
    return {
        400: "invalid_request_error",
        401: "authentication_error",
        403: "permission_error",
        404: "not_found_error",
        413: "request_too_large",
        429: "rate_limit_error",
        529: "overloaded_error",
    }.get(status, "api_error")


def is_context_overflow_text(text: str) -> bool:
    return bool(_OVERFLOW_RE.search(text))
