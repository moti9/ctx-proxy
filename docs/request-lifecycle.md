# Request lifecycle

This page traces one `POST /v1/messages` request from the HTTP boundary to the
upstream backend and back.

## 1. HTTP entry: `routes.create_message`

Claude Code sends a standard Anthropic Messages request. ctxproxy parses it into
`MessagesRequest` (`types_anthropic.py`).

```python
req = MessagesRequest.model_validate(body)
```

Unknown block types are preserved verbatim because the models use
`extra="allow"`. This lets Claude Code adopt new block types without breaking
the proxy.

## 2. Resolve the profile

```python
profile = state.config.profile_for(req.model)
upstream_model = profile.resolve_upstream_model(req.model)
```

Profiles are matched in order by glob against `req.model`. The first match wins,
so catch-alls belong last. The profile tells us:

- Which backend to use.
- Whether the backend is native Anthropic.
- The real `context_window`.
- How to count tokens.
- Which model writes summaries.

## 3. Resolve the session

```python
session = derive_session_key(http_request.headers, req)
```

`session.py` prefers the `X-Claude-Code-Session-Id` header. If that is missing,
it fingerprints the stable prefix of the conversation (system prompt + first
plain user turn + model name). Subagents get separate ledgers keyed by both
session id and agent id.

## 4. Build helpers

```python
counter = state.counter_for(profile, upstream_model)
summarizer = state.summarizer_for(profile, req.model)
backend = state.backend_for(profile)
```

These objects live for the lifetime of the process and are cached per profile.
Counter caching is especially important: token counts are memoised per block, so
long sessions do not suffer O(n²) recounting.

## 5. Run the context pipeline

```python
result, ledger = await state.manager.process(
    request, profile=profile, session_key=session.value, ...
)
```

`ContextManager.process` (`context/manager.py`) does the real work:

1. Load or create the `SessionLedger`.
2. Note the task statement if this is the first turn.
3. Clamp `max_tokens` to the backend's `max_output_tokens` ceiling if set.
4. Reapply a previous fold if the conversation fingerprint still matches.
5. Count tokens.
6. If under the trigger, return the untouched request.
7. If over, run reduction strategies until under target.
8. Persist the ledger.

See [Context reduction](./context-reduction.md) for details on strategies and the
Protected Set.

## 6. Final per-profile preparation

```python
prepared = _prepare(result.request, profile)
```

- **Native Anthropic + server compaction enabled:** attach
  `context_management.edits` with `clear_tool_uses_20250919` so Anthropic can
  also compact server-side.
- **Non-native:** strip `context_management`; strip `cache_control` if prompt
  caching is unsupported.

## 7. Dispatch

### Streaming

```python
first, stream = await _open_stream(backend, prepared, upstream_model, headers, tokens)
return StreamingResponse(_replay(first, stream), media_type="text/event-stream")
```

The proxy eagerly reads the first chunk. If the upstream rejects with a context
overflow, the proxy can still retry before any byte reaches the client.

### Non-streaming

```python
payload = await backend.complete(prepared, upstream_model, headers)
return JSONResponse(payload)
```

## 8. Overflow retry

If the backend returns an error that looks like a context overflow and the
policy enables it:

```python
retry, _ = await reduce(target_ratio=policy.overflow_retry_ratio)
```

The request is reduced much more aggressively (default 40% of usable budget)
and retried once. This turns a hard failure into a transparent recovery.

## 9. Response shaping

- **Native Anthropic:** response bytes pass through unchanged.
- **OpenAI-compatible:** the backend returns Anthropic-shaped JSON or SSE thanks
to the translators in `translate/`. The client never sees OpenAI formats.

Both paths attach the response header `x-ctxproxy-session: <session-key>`.

## Flow diagram

```
Claude Code
    │
    ▼
routes.create_message
    │
    ├── parse MessagesRequest
    ├── profile_for(model)
    ├── derive_session_key(headers, request)
    └── counter / summarizer / backend
    │
    ▼
ContextManager.process
    │
    ├── load/create ledger
    ├── reapply fold (if valid)
    ├── count tokens
    ├── under trigger? → passthrough
    └── over? → reduce
    │
    ▼
_prepare(request, profile)
    │
    ▼
backend.stream / backend.complete
    │
    ├── context overflow? → reduce + retry once
    └── return Anthropic-shaped response
    │
    ▼
Claude Code
```

## Count tokens endpoint

`POST /v1/messages/count_tokens` is load-bearing. Claude Code uses it to drive
its own HUD and auto-compact. The proxy returns the count **after** the
reduction it would apply, so the client does not compact on top of the proxy.

```python
reported = actual if actual <= budget.trigger_at else budget.target_at
```

This means the client's context HUD sits stable around the target ratio.
