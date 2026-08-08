"""HTTP surface: the Anthropic Messages API, plus operational endpoints."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .config import UnknownModelError
from .context.budget import compute_budget
from .errors import ConfigurationError, ProxyError, UpstreamError
from .logging import log_event
from .session import derive_session_key
from .state import AppState
from .types_anthropic import CountTokensRequest, MessagesRequest

log = logging.getLogger(__name__)

router = APIRouter()

# Below this, fixed per-request overhead swamps the tokenizer signal.
_MIN_TOKENS_FOR_DRIFT = 5_000


def _state(request: Request) -> AppState:
    return request.app.state.ctx


@router.post("/v1/messages")
async def create_message(http_request: Request):
    state = _state(http_request)
    body = await http_request.json()

    try:
        req = MessagesRequest.model_validate(body)
    except Exception as exc:  # noqa: BLE001
        return ConfigurationError(f"malformed request body: {exc}").to_response()

    try:
        profile = state.config.profile_for(req.model)
    except UnknownModelError as exc:
        return ConfigurationError(str(exc)).to_response()

    upstream_model = profile.resolve_upstream_model(req.model)
    session = derive_session_key(http_request.headers, req)
    counter = state.counter_for(profile, upstream_model)
    summarizer = state.summarizer_for(profile, req.model)
    backend = state.backend_for(profile)
    headers = dict(http_request.headers)

    async def reduce(target_ratio: float | None = None):
        # Re-validate from the untouched body: the pipeline mutates the request
        # in place, so a retry must start from the client's original payload,
        # not from the already-reduced one.
        fresh = MessagesRequest.model_validate(body)
        return await state.manager.process(
            fresh,
            profile=profile,
            session_key=session.value,
            counter=counter,
            summarizer=summarizer,
            target_ratio=target_ratio,
        )

    try:
        result, _ledger = await reduce()
    except ProxyError as exc:
        return exc.to_response()

    prepared = _prepare(result.request, profile)
    policy = state.config.policy

    if session.source == "fingerprint" and _first_sighting(state, session.value):
        # Header-based identity is exact; the fingerprint is a fallback. Say so
        # once per session so a silently-missing header is visible rather than
        # something you discover from a lost rolling summary hours later.
        log_event(
            log,
            "no session header; identifying by conversation fingerprint",
            session=session.value,
            seen_headers=sorted(
                h for h in http_request.headers if "session" in h.lower() or "agent" in h.lower()
            ) or ["none"],
        )

    log_event(
        log,
        "dispatch",
        session=session.value,
        source=session.source,
        model=req.model,
        upstream=upstream_model,
        backend=backend.name,
        tokens=result.tokens_after,
        saved=result.saved,
        stream=req.stream,
    )

    # -- streaming ---------------------------------------------------------- #
    if req.stream:
        observed: list[int] = []
        try:
            first, stream = await _open_stream(
                backend, prepared, upstream_model, headers, result.tokens_after,
                observed.append, session_key=session.value,
            )
        except UpstreamError as exc:
            if not (exc.is_context_overflow() and policy.overflow_retry_enabled):
                _log_upstream_error(session.value, exc)
                return exc.to_response()

            log_event(
                log,
                "upstream reported overflow; forcing reduction and retrying once",
                level=logging.WARNING,
                session=session.value,
                tokens=result.tokens_after,
            )
            try:
                retry, _ = await reduce(target_ratio=policy.overflow_retry_ratio)
                prepared = _prepare(retry.request, profile)
                first, stream = await _open_stream(
                    backend, prepared, upstream_model, headers, retry.tokens_after,
                    observed.append, session_key=session.value,
                )
            except (UpstreamError, ProxyError) as retry_exc:
                if isinstance(retry_exc, UpstreamError):
                    _log_upstream_error(session.value, retry_exc, retried=True)
                return retry_exc.to_response()

        return StreamingResponse(
            _replay(first, stream, state, session.value, result.tokens_after, observed),
            media_type="text/event-stream",
            headers={"cache-control": "no-cache", "x-ctxproxy-session": session.value},
        )

    # -- non-streaming ------------------------------------------------------ #
    try:
        payload = await backend.complete(
            prepared, upstream_model, headers, session_key=session.value
        )
    except UpstreamError as exc:
        if not (exc.is_context_overflow() and policy.overflow_retry_enabled):
            _log_upstream_error(session.value, exc)
            return exc.to_response()

        log_event(
            log,
            "upstream reported overflow; forcing reduction and retrying once",
            level=logging.WARNING,
            session=session.value,
        )
        try:
            retry, _ = await reduce(target_ratio=policy.overflow_retry_ratio)
            payload = await backend.complete(
                _prepare(retry.request, profile), upstream_model, headers,
                session_key=session.value,
            )
        except (UpstreamError, ProxyError) as retry_exc:
            if isinstance(retry_exc, UpstreamError):
                _log_upstream_error(session.value, retry_exc, retried=True)
            return retry_exc.to_response()

    await _record_drift(state, session.value, result.tokens_after, payload)
    return JSONResponse(payload, headers={"x-ctxproxy-session": session.value})


@router.post("/v1/messages/count_tokens")
async def count_tokens(http_request: Request):
    """Report what the request will cost *after* our reduction.

    This endpoint is load-bearing for behaviour, not just display. Claude Code
    drives its own auto-compact off these numbers, so reporting the raw
    pre-reduction count makes the client compact on top of us — double
    compaction, double loss. Reporting the post-reduction figure keeps the
    client's HUD honest and leaves context management in one place.
    """
    state = _state(http_request)
    body = await http_request.json()

    try:
        req = CountTokensRequest.model_validate(body)
    except Exception as exc:  # noqa: BLE001
        return ConfigurationError(f"malformed request body: {exc}").to_response()

    try:
        profile = state.config.profile_for(req.model)
    except UnknownModelError as exc:
        return ConfigurationError(str(exc)).to_response()

    upstream_model = profile.resolve_upstream_model(req.model)
    counter = state.counter_for(profile, upstream_model)

    messages_request = req.to_messages_request()
    actual = await counter.count_request(messages_request)
    budget = compute_budget(profile, state.config.policy, messages_request)

    # Above the trigger the next /v1/messages call is guaranteed to reduce to
    # at most `target_at`, so that is the honest figure to report.
    reported = actual if actual <= budget.trigger_at else budget.target_at

    if reported != actual:
        log_event(
            log,
            "count_tokens capped to post-reduction estimate",
            model=req.model,
            actual=actual,
            reported=reported,
        )

    return JSONResponse({"input_tokens": reported})


@router.get("/health")
async def health(http_request: Request):
    state = _state(http_request)
    return {
        "status": "ok",
        "backends": [b.name for b in state.config.backends],
        "profiles": [
            {
                "match": p.match,
                "backend": p.backend,
                "native_anthropic": p.native_anthropic,
                "context_window": p.context_window,
            }
            for p in state.config.profiles
        ],
    }


@router.get("/stats")
async def stats(http_request: Request):
    state = _state(http_request)
    ledgers = await state.store.list_all()
    return {
        "sessions": len(ledgers),
        "total_tokens_saved": sum(item.total_tokens_saved for item in ledgers),
        "total_compactions": sum(item.compaction_count for item in ledgers),
        "detail": [item.stats() for item in ledgers[:50]],
    }


@router.get("/v1/models")
async def models(http_request: Request):
    state = _state(http_request)
    return {
        "data": [
            {"id": p.match, "type": "model", "display_name": p.match}
            for p in state.config.profiles
        ]
    }


# --------------------------------------------------------------------------- #


def _log_upstream_error(session_key: str, exc: UpstreamError, *, retried: bool = False) -> None:
    """Make a forwarded upstream failure visible in ctxproxy's own logs.

    Without this the proxy passes a 429/5xx straight through to the client and
    logs nothing, so an upstream outage looks like ctxproxy "saw nothing" — the
    exact confusion that sends you debugging the wrong system. Capacity/
    availability statuses get an explicit hint that the cause is the gateway,
    not the proxy or the context pipeline.
    """
    hint = ""
    if exc.status_code in (429, 502, 503, 529):
        hint = (
            " | this is a gateway capacity/availability response (e.g. "
            "'no deployments available'), not a ctxproxy or context error — "
            "it is the upstream, and usually clears on its own"
        )
    log_event(
        log,
        "upstream error forwarded to client",
        level=logging.WARNING,
        session=session_key,
        backend=exc.backend,
        status=exc.status_code,
        retried=retried,
        detail=exc.text[:300] + hint,
    )


def _first_sighting(state, session_key: str) -> bool:
    seen = getattr(state, "_seen_sessions", None)
    if seen is None:
        seen = state._seen_sessions = set()
    if session_key in seen:
        return False
    seen.add(session_key)
    return True


async def _record_drift(state, session_key: str, estimated: int, payload: dict) -> None:
    """Compare our token estimate against what the backend actually charged.

    Our counter is an approximation — cl100k is not the tokenizer most
    non-Anthropic models use. Under-counting is the dangerous direction: it
    delays compaction until the request no longer fits the real window. This
    turns that risk from a guess into a number you can act on.
    """
    actual = ((payload or {}).get("usage") or {}).get("input_tokens") or 0
    if actual <= 0 or estimated <= 0:
        return

    # Backends add a fixed overhead our count cannot see — the chat template
    # and any built-in system prompt. On a small request that constant
    # dominates and the ratio is meaningless (a 10-token prompt billed at 570
    # reads as 57x drift, which says nothing about the tokenizer). Only sample
    # once the conversation is large enough for the constant to wash out.
    if estimated < _MIN_TOKENS_FOR_DRIFT:
        return

    ledger = await state.store.load(session_key)
    if ledger is None:
        return

    ratio = ledger.record_token_drift(estimated, actual)
    await state.store.save(ledger)
    if ratio is None:
        return

    median = ledger.median_token_drift or ratio
    if median > 1.10 and len(ledger.token_drift_samples) >= 5:
        log_event(
            log,
            "token estimates are running low; compaction may fire too late",
            level=logging.WARNING,
            session=session_key,
            estimated=estimated,
            actual=actual,
            median_ratio=round(median, 3),
            hint=f"raise tokenizer.safety_multiplier to about {median * 1.05:.2f}",
        )
    else:
        log_event(
            log,
            "token drift",
            level=logging.DEBUG,
            session=session_key,
            estimated=estimated,
            actual=actual,
            ratio=round(ratio, 3),
        )


def _prepare(request: MessagesRequest, profile) -> MessagesRequest:
    """Final per-backend adjustments before dispatch."""
    if profile.native_anthropic and profile.supports_server_compaction:
        # Belt and braces: we have already reduced client-side, but letting the
        # upstream compact server-side too means the model itself writes any
        # further summary — strictly better than anything we can do here.
        request.context_management = request.context_management or {
            "edits": [{"type": "clear_tool_uses_20250919"}]
        }
    elif not profile.native_anthropic:
        # Meaningless downstream, and rejected by some gateways.
        request.context_management = None
        if not profile.supports_prompt_caching:
            _strip_cache_control(request)
    return request


def _strip_cache_control(request: MessagesRequest) -> None:
    for message in request.messages:
        blocks = message.blocks()
        if any("cache_control" in b for b in blocks):
            message.content = [
                {k: v for k, v in b.items() if k != "cache_control"} for b in blocks
            ]
    if isinstance(request.system, list):
        request.system = [
            {k: v for k, v in b.items() if k != "cache_control"} for b in request.system
        ]


async def _open_stream(
    backend, request, upstream_model, headers, input_tokens, on_usage=None,
    *, session_key=None,
):
    """Start a stream and pull the first chunk.

    Pulling eagerly is what makes retry-on-overflow possible: the upstream
    status is checked before any byte reaches the client, so a context-overflow
    rejection can still be recovered from. Once bytes are flushed, it is too
    late.
    """
    stream = backend.stream(
        request, upstream_model, headers, input_tokens, on_usage,
        session_key=session_key,
    )
    try:
        first = await stream.__anext__()
    except StopAsyncIteration:
        first = None
    return first, stream


async def _replay(
    first: bytes | None,
    stream,
    state=None,
    session_key: str = "",
    estimated: int = 0,
    observed: list[int] | None = None,
) -> AsyncIterator[bytes]:
    if first is not None:
        yield first
    async for chunk in stream:
        yield chunk

    # The usage trailer only lands once the stream is fully consumed.
    if state is not None and observed:
        await _record_drift(
            state, session_key, estimated, {"usage": {"input_tokens": observed[-1]}}
        )
