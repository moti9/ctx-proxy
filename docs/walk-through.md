### ctxproxy codebase walkthrough

This is a Python proxy that sits between Claude Code and an LLM backend. Its job is to manage the context window: when a conversation gets too long, it compacts old messages into a rolling summary instead of letting the backend return a context-limit error.

---
1. What it is and why

The README explains the two problems it solves:

1. Claude Code's own auto-compact is unreliable — it only fires when free space hits zero and sometimes never fires.
2. Using a custom ANTHROPIC_BASE_URL breaks the assumptions Claude  size, missing token-count endpoint, rejected beta headers, etc.

ctxproxy fixes this by moving context management into the transportndow size, token counting, and compaction per backend/model.

High-level flow (from the README diagram):

Claude Code → ctxproxy → Anthropic API (native passthrough)
                    ↓
              LiteLLM / vLLM / OpenAI-compatible gateway (translati

---
2. Entry points

Package layout

src/ctxproxy/
├── __init__.py           # version
├── __main__.py           # `python -m ctxproxy` → cli.app
├── app.py                # FastAPI factory
├── cli.py                # Typer commands: serve, doctor, sessions, inspect, reset, init, version
├── config.py             # YAML config → Pydantic models
├── errors.py             # Anthropic-shaped error envelopes
├── logging.py            # structured log formatter
├── routes.py             # HTTP endpoints /v1/messages, /v1/messages/count_tokens, /health, /stats, /v1/models
├── session.py            # derive stable session identity
├── state.py              # AppState: backends, counters, summarizers, ledger store
├── types_anthropic.py    # permissive Anthropic wire models
├── backends/             # dispatch to upstream
│   ├── base.py
│   ├── anthropic_native.py
│   ├── openai_compat.py
│   └── __init__.py
├── context/              # the compaction pipeline
│   ├── budget.py
│   ├── ledger.py
│   ├── manager.py
│   ├── protect.py
│   ├── strategies.py
│   └── summarizer.py
├── store/                # ledger persistence
│   ├── base.py
│   └── file.py
├── tokens/               # token counting
│   ├── base.py
│   ├── local.py
│   └── upstream.py
└── translate/            # Anthropic ↔ OpenAI conversion
    ├── request.py
    ├── response.py
    └── sse.py

Running it

- ctxproxy serve — start the FastAPI server via uvicorn.
- ctxproxy serve --reload — reload on .py or .yaml changes.
- ctxproxy doctor — validate config and probe backends.
- ctxproxy init — copy config.example.yaml to ctxproxy.yaml.

The CLI is in cli.py; app.py builds the FastAPI app; routes.py defi

---
3. Configuration (config.py)

Everything is driven by ctxproxy.yaml, loaded into Pydantic models:

- BackendConfig — an upstream: name, kind (anthropic or openai), base_url, credentials, beta-flag allowlist, OpenAI-specific options.
- ModelProfile — per-model capabilities: match glob, backend, nativndow, output_reserve, safety_buffer, tokenizer, summarizer profile,etc.
- ReductionPolicy — trigger_ratio, target_ratio, keep-recent settin retry.
- ServerConfig — host/port, logging, state directory, session TTL.
- Config — the root; validates backend references and summarizer re

Key idea: native_anthropic is the master switch. If true, Anthropicugh; if false, they are translated/stripped.

---
4. Request lifecycle (routes.py)

POST /v1/messages

1. Parse the Anthropic MessagesRequest.
2. Resolve the profile for req.model.
3. Derive the session key from headers (or fingerprint the conversation prefix).
4. Get the token counter, summarizer, and backend from AppState.
5. Run ContextManager.process(...) to possibly reduce context.
6. Prepare final adjustments via _prepare():
  - Native Anthropic + server compaction → attach context_management.edits.
  - Non-native → strip context_management, optionally strip cache_c
7. Dispatch to backend:
  - Streaming: eagerly read the first chunk so overflow errors can h the client.
  - Non-streaming: await full response.
8. On upstream context-overflow with overflow_retry_enabled, force ratio and retry once.

POST /v1/messages/count_tokens

Returns the token count after reduction. This is intentional: if Cl it would compact on top of the proxy.

Operational endpoints

- GET /health — backends and profiles.
- GET /stats — aggregate session savings.
- GET /v1/models — list configured profiles.

---
5. Session identity (session.py)

Claude Code sends X-Claude-Code-Session-Id, but headers can be stripped. So identity is derived in two layers:

1. Header if present (preferred).
2. Fingerprint of the stable prefix (system prompt + first plain usise.

Subagents get separate ledgers by including the agent id in the keyhe parent conversation's summary.

---
6. Token counting (tokens/)

Goal: avoid O(n²) re-counting across long sessions. Every counter memoises per block by content hash.

- TokenCounter (base) — counts tools, system, messages, blocks; caches block counts.
- HeuristicCounter (local.py) — chars-per-token estimate, always avmistic.
- TiktokenCounter (local.py) — BPE counting via tiktoken (optional dependency).
- UpstreamCounter (upstream.py) — calls the backend's /v1/messages/cal counting on first failure.

AppState.counter_for(...) decides whether to wrap the local countersed on profile settings.

---
7. Context pipeline (context/)

This is the core algorithm.

manager.py — the orchestrator

Per request:

1. Load or create the SessionLedger.
2. Re-apply any previously recorded fold (deterministic, no model c
3. Count tokens.
4. If under trigger_at, pass through.
5. Otherwise, run strategies in order until under target_at.
6. Persist the ledger.

A key detail: _reapply_fold splices a prior summary back in if the ll matches. If the client rewound/compacted, the fold is invalidated and rebuilt.

budget.py — window arithmetic

usable = context_window - max(output_reserve, max_tokens) - safety_buffer
trigger_at = usable * trigger_ratio
target_at = usable * target_ratio

The gap between trigger and target is hysteresis so you don't compact every turn.

protect.py — the Protected Set

What compaction must never touch:

- Task statement — everything up to and including the first real user turn.
- Recent turns — snapped back to a safe cut point so tool pairs are
- Unresolved errors — errored tool_result and the assistant turn that produced it.
- Explicit pins — <ctx-pin> content.
- Current plan/todo — most recent plan tool call only.

Cut-point safety: safe_cut_points() only allows cuts where the message at the cut carries no tool_result; this guarantees every tool_use is answered by its
tool_result.

strategies.py — reduction order

Strategies run cheapest/least-lossy first:

1. clear_tool_results — replace old tool_result payloads with a plad intact. Old file dumps and command output are what actually fillwindows.
2. clear_thinking — drop thinking/redacted_thinking blocks, but onlackends (native signatures must be preserved).
3. compact — fold the oldest unprotected span into the rolling summary.

Each strategy updates the running token total incrementally (delta-based, not re-counting everything).

Compact._choose_cut() finds the smallest safe cut that brings tokens under target, then calls the summarizer to fold that span into the existing ledger summary.

If the summarizer fails, a deterministic mechanical digest is used — recovery must never fail.

ledger.py — durable session state

SessionLedger persists:

- task_statement — captured verbatim on first sight.
- summary — rolling structured summary.
- folded_through / fold_start / fold_signature — watermark for deterministic reapplication.
- files_touched, open_questions, decisions.
- compaction_count, total_tokens_saved, turns_seen, events.

The fold signature hashes the boundary messages + span length. If it stops matching, the fold is invalidated and rebuilt.

render_summary_block() produces the <conversation-summary> text spliced into messages.

summarizer.py — rolling-summary generation

- Sends a structured summarization prompt to a configured model/backend.
- Chunks the span so the summarizer never sees more than it can hol
- Folds each chunk sequentially into the existing summary (each message summarized once, not regenerated).
- Uses a different profile for summarization than the task model ifprofile).

---
8. Backends and translation (backends/, translate/)

backends/base.py

Abstract Backend with shared HTTP plumbing:

- Builds an httpx.AsyncClient.
- _forwarded_headers() strips hop-by-hop headers, replaces auth, fi
- _post_json() handles errors, raising UpstreamError.

backends/anthropic_native.py

Passthrough to Anthropic:

- Forwards /v1/messages and /v1/messages/count_tokens.
- Streams raw SSE bytes without reparsing.
- Preserves thinking blocks, cache control, signatures, server-side compaction.

backends/openai_compat.py

Translates Anthropic ↔ OpenAI:

- Request: anthropic_to_openai() in translate/request.py.
- Response: openai_to_anthropic() in translate/response.py.
- Streaming: OpenAIStreamTranslator in translate/sse.py.

Important behaviors:

- Drops cache_control, thinking, context_management, beta flags.
- Converts tool results into OpenAI role: "tool" messages.
- Re-renders reasoning models' reasoning_content as text (or drops, configurable).
- Handles OpenAI's streaming chunk format and emits proper Anthropi

translate/sse.py

This is subtle. Anthropic expects a strict event sequence (content__block_stop, message_delta, message_stop). OpenAI just emitsinterleaved deltas. The translator tracks open blocks and emits correctly shaped Anthropic SSE events.

---
9. Storage (store/)

- LedgerStore abstract interface.
- FileLedgerStore (file.py) — file-backed persistence:
  - Atomic writes via temp file + os.replace.
  - Disk I/O in worker threads (asyncio.to_thread).
  - Per-session asyncio lock so concurrent turns don't clobber each
  - Prunes old ledgers by TTL.

---
10. Logging (logging.py)

- Text or JSON formatter.
- log_event() attaches structured fields to log records.
- Captures every reduction decision with before/after tokens.

---
11. Tests

Test files:

- test_protect.py — Protected Set and cut-point safety (the correctness core).
- test_manager.py — full pipeline: reduction ordering, fold reappli, summarizer failure fallback, native-profile thinking preservation, max_tokens clamping.
- test_translate.py — Anthropic↔OpenAI request/response/SSE transla
- test_session.py, test_budget.py, test_routes.py, test_imports.py.

conftest.py provides fixtures and builders for conversations/tool exchanges/profiles.

---
12. How a request flows end-to-end

A concrete turn:

1. Claude Code sends POST /v1/messages with the full conversation.
2. routes.create_message() parses it, resolves profile/backend/session/counter/summarizer.
3. ContextManager.process():
  - Loads the ledger.
  - Reapplies previous fold if still valid.
  - Counts tokens.
  - If over budget, clears old tool results → clears thinking → comn into summary.
  - Saves ledger.
4. _prepare() applies final native/non-native adjustments.
5. Backend sends the request upstream.
6. If upstream says context overflow, force a deeper reduction and
7. Streamed or full response is returned to Claude Code, with x-ctxproxy-session header.