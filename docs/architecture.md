# Architecture

ctxproxy is a thin HTTP proxy that speaks the Anthropic Messages API on the
client side and routes to either an Anthropic-native endpoint or an
OpenAI-compatible gateway on the upstream side. Its only job beyond forwarding
is to keep long conversations inside the upstream's real context window without
silently losing important state.

## Big picture

```
                        ANTHROPIC_BASE_URL=http://127.0.0.1:4000
   ┌──────────────┐                              ┌────────────────────┐
   │ Claude Code  │ ── POST /v1/messages ──────▶ │     ctxproxy       │
   │ CLI + VSCode │ ── POST /v1/messages/       │                    │
   │              │    count_tokens             │  • profiles        │
   │              │ ◀──── SSE stream ────────── │  • token ledger    │
   └──────────────┘                              │  • reduction       │
                                                 │  • beta sanitise   │
                                                 └─────────┬──────────┘
                                                           │
                    native Anthropic ──────────────────────┼────── OpenAI-compatible
                    (passthrough, server-side compact)     │       (translation, proxy compact)
                                                           │
                                              ┌────────────┴────────────┐
                                              ▼                         ▼
                                       Anthropic API             LiteLLM / vLLM / gateway
```

## Core principles

1. **Context management belongs at the transport layer.** Claude Code cannot
correctly compact against a custom base URL because it assumes wrong window sizes
and token-count endpoints. The proxy owns the real profile per model.

2. **Cheapest reduction first.** Old tool output is evicted before reasoning is
summarized. Summarization is lossy and expensive, so it is the last resort.

3. **Never lose the thread.** A Protected Set declares which messages are
ineligible for compaction: the task statement, recent turns, unresolved errors,
the current plan, and user pins.

4. **Stable folding.** Because Claude Code resends the entire conversation every
turn, the proxy records how many original messages were folded into the summary.
On later turns it replays that fold deterministically and only summarizes new
material.

## Module map

| Layer | Modules | Responsibility |
|-------|---------|----------------|
| HTTP | `routes.py`, `app.py` | Receive Anthropic requests, return Anthropic responses. |
| Config | `config.py` | Profiles, backends, policy, validation. |
| Wiring | `state.py` | Long-lived backends, counters, summarizers, store. |
| Identity | `session.py` | Stable session key from header or content fingerprint. |
| Pipeline | `context/manager.py`, `context/strategies.py` | Decide when and how to reduce. |
| Safety | `context/protect.py` | Compute the Protected Set and safe cut points. |
| Memory | `context/ledger.py`, `store/file.py` | Rolling summary + durable persistence. |
| Summarise | `context/summarizer.py` | Chunked folding summarization. |
| Count | `tokens/base.py`, `tokens/local.py`, `tokens/upstream.py` | Fast token counting per block. |
| Dispatch | `backends/anthropic_native.py`, `backends/openai_compat.py` | Talk to upstreams. |
| Translate | `translate/request.py`, `translate/response.py`, `translate/sse.py` | Anthropic ↔ OpenAI. |

## The two code paths

### Native Anthropic

- Request passes through almost unchanged.
- Anthropic-specific features (`cache_control`, `thinking`, beta flags,
  `context_management`) are preserved.
- Token counting can be delegated to the upstream `count_tokens` endpoint.
- The proxy still reduces client-side as a safety net, but it can also ask
  Anthropic to compact server-side by attaching `context_management.edits`.

### OpenAI-compatible

- Anthropic request is translated to OpenAI Chat Completions.
- Anthropic-only features are stripped to avoid 400s.
- Token counting is local (`heuristic` or `tiktoken`) unless the gateway happens
to expose a compatible endpoint.
- Response and SSE stream are translated back to Anthropic format.

The `native_anthropic` flag on the profile selects between these two behaviors.

## Data flow per request

See [Request lifecycle](./request-lifecycle.md) for a turn-by-turn trace. At a
high level:

1. HTTP layer parses and validates the Anthropic request.
2. Resolve session, profile, backend, counter, and summarizer.
3. Reapply any previous fold if the conversation fingerprint still matches.
4. Count tokens.
5. If over budget, reduce in order: clear tool results, clear thinking, compact.
6. Persist the updated ledger.
7. Apply final profile-specific request adjustments.
8. Dispatch to backend; retry once on upstream context overflow.
9. Stream or return the upstream response.

## Key invariants

- Every `tool_use` block is paired with a `tool_result` block carrying the same
  `tool_use_id`. The proxy never splits a pair across a cut point.
- A message is summarized at most once over the life of a session.
- Token counts are memoised per block so repeated recounting is O(history) per
  turn, not O(history²).
- Ledgers are written atomically so a crash never leaves a half-written summary.
