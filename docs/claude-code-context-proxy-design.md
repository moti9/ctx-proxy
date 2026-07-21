# Context-Management Proxy for Claude Code (Anthropic + LiteLLM)

A design for making long-running Claude Code sessions reliable across **both** Anthropic-native models and your own models served through LiteLLM — without hitting context-limit errors and without silently losing task context.

---

## 1. Why this breaks today

Two independent problems stack on top of each other.

**Problem A — Claude Code's own auto-compact is flaky (even against native Anthropic).**
- Compaction fires only when free space (minus a hardcoded ~33K buffer) hits zero, at roughly 83.5% of the window. It sometimes never fires and the session just stops at ~100% with "Context limit reached · /compact and continue."
- `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE` can *lower* the trigger but not raise it, is ignored when set in `settings.json`'s `env` block (must be exported in the shell), and has a history of not being reliably applied to the main conversation.

**Problem B — a custom `ANTHROPIC_BASE_URL` (LiteLLM / your backend) breaks the assumptions compaction depends on.**
- **Window size is wrong.** Behind a custom base URL, Claude Code does **not** send the `context-1m` beta header and assumes a 200K window regardless of the real model. If your model is 128K, compaction triggers too late (hard fail); if it's 1M, you leave capacity on the table.
- **Token accounting can be wrong or missing.** Claude Code relies on the Anthropic `POST /v1/messages/count_tokens` endpoint. If your gateway doesn't serve it, or returns bad numbers, the "how full am I" estimate is wrong and auto-compact misfires or never fires.
- **Beta headers get rejected.** Claude Code attaches Anthropic experimental beta flags on every request. Non-Anthropic providers reject unknown flags with "invalid beta flag" errors.
- **The compaction call itself can fail.** Client-side compaction summarizes the *whole* conversation in one shot. On a weaker or smaller-window backend, that summarization request can exceed the window and error — so recovery fails exactly when you need it.

**Takeaway:** don't try to make Claude Code's *client-side* compaction behave over a proxy. Move context management to the transport layer, where you control token accounting, window size, and the compaction strategy per provider.

---

## 2. Approach: a transparent Anthropic-Messages proxy (not a "plugin")

Claude Code only talks to whatever `ANTHROPIC_BASE_URL` points at. It has **no plugin hook** for rewriting the message array or managing context before a request goes out. The clean, supported interception point is therefore the **HTTP proxy layer**.

```
                 ANTHROPIC_BASE_URL = http://proxy:4000
   ┌──────────────┐        /v1/messages            ┌─────────────────────────┐
   │ Claude Code  │ ─────  /v1/messages/count_tokens ─▶ │  Context-Mgmt Proxy   │
   │ (CLI + VSCode)│ ◀─────  SSE stream back  ───────── │                         │
   └──────────────┘                                   │  • capability profiles  │
                                                       │  • token ledger/session │
                                                       │  • tiered compaction    │
                                                       │  • beta-header sanitize │
                                                       └───────────┬─────────────┘
                                          native Anthropic ────────┤────── LiteLLM / OpenAI-compatible
                                        (server-side compact)      │       (proxy-side polyfill)
                                                        ┌──────────┴──────────┐
                                                        ▼                     ▼
                                                 Anthropic API         Your models via LiteLLM
```

The front speaks the **Anthropic Messages API** so Claude Code is none the wiser. The back routes to Anthropic (native) or LiteLLM/your models, applying a **different compaction strategy per provider**.

---

## 3. Build vs. reuse — start with LiteLLM, extend only for the gaps

**LiteLLM already implements most of this.** It exposes `/v1/messages` + `/v1/messages/count_tokens`, and it natively polyfills Anthropic's `context_management` beta (`compact_20260112`, `clear_tool_uses_20250919`) across *all* providers, including non-Anthropic ones.

Two edit types matter:
- `clear_tool_uses_20250919` — replaces old `tool_result` contents with a placeholder, preserving message structure. Cheap, near-lossless, great first line of defense (stale file dumps and command output are what actually bloats agentic sessions).
- `compact_20260112` — collapses prior history into a single summary + the last user question. Heavier, lossy; use only when clearing isn't enough. Has a 50K-token minimum trigger enforced by the proxy.

**Critical LiteLLM config** (without it the polyfill is a silent no-op returning `summary_model_not_configured`):

```yaml
# proxy_server_config.yaml
general_settings:
  context_management_summary_model: claude-sonnet-4-5   # who writes summaries; any alias in model_list

model_list:
  - model_name: opus                      # Anthropic-native path
    litellm_params:
      model: anthropic/claude-opus-4-8
    model_info:
      max_input_tokens: 200000            # or 1000000 if entitled + beta enabled

  - model_name: our-coder                 # your model via LiteLLM
    litellm_params:
      model: openai/your-model            # or hosted_vllm/, bedrock/, etc.
      api_base: http://your-backend/v1
    model_info:
      max_input_tokens: 128000            # set the REAL window so thresholds are correct
```

Also strip unsupported beta flags per provider via LiteLLM's `anthropic_beta_headers_config.json` (or `LITELLM_ANTHROPIC_BETA_HEADERS_URL`) to kill "invalid beta flag" errors on non-Anthropic upstreams.

**Recommendation:** get the config path working first. It resolves the majority of cases. Build the custom layer in §4–§6 only if you need stronger *context-preservation guarantees* (deterministic "never summarize the task"), custom routing, or observability that LiteLLM's generic polyfill doesn't give you. When you do build it, prefer a **LiteLLM custom hook/callback** (a `CustomLogger` with a pre-call hook) over a separate service — you get LiteLLM's tokenizer, transformations, and beta handling for free and only add your ledger logic.

---

## 4. Per-model capability profiles

The whole design hinges on knowing what each backend can do. Drive everything off a profile:

```yaml
models:
  - alias: opus
    upstream: anthropic
    native_anthropic: true
    context_window: 200000
    supports_count_tokens: true
    supports_prompt_caching: true
    supports_server_compaction: true      # compact_20260112 handled upstream
    summarizer: opus

  - alias: our-coder
    upstream: litellm
    native_anthropic: false
    context_window: 128000                 # the REAL number, not 200K
    supports_count_tokens: false           # -> proxy counts via tokenizer
    supports_prompt_caching: false
    supports_server_compaction: false      # -> proxy polyfills
    tokenizer: cl100k_base                 # or the model's real tokenizer
    summarizer: opus                        # summarize with a strong model, run the task on the cheap one
```

`native_anthropic` is the master switch the user's message flagged: it decides whether you pass Anthropic-specific features through untouched or translate/strip them.

---

## 5. Compaction orchestration (the core loop)

Per incoming `/v1/messages` request:

```
1. Resolve session (X-Claude-Code-Session-Id) + model → capability profile.

2. Count input tokens:
     native + supports_count_tokens → trust upstream count_tokens (or track locally)
     else                           → tokenizer count (profile.tokenizer)

3. effective_window = context_window
                       − output_reserve   (e.g. requested max_tokens, default 32K)
                       − safety_buffer     (e.g. 8–16K for the summarization pass itself)

4. If tokens ≤ threshold (e.g. 0.75 × effective_window): pass through unchanged.

5. Else reduce, cheapest first, until under threshold:
     a. clear_tool_uses  — drop/placeholder stale tool_result bodies, keep last K tool uses
     b. still over?      — compact: summarize the collapsible span into ONE summary block
                            (always keep the Protected Set from §6)

6. Dispatch:
     native_anthropic  → attach context_management.edits and let Anthropic compact SERVER-SIDE
                          (best: no lossy client summarization, summarizer = the model itself)
     non-native        → run the reduction yourself (or delegate to LiteLLM's polyfill),
                          summarizer = profile.summarizer

7. On upstream error `model_context_window_exceeded` (or equivalent):
     force a compaction pass and RETRY ONCE, so long tasks continue instead of hard-failing.
```

Step 7 is what turns "context limit error" from a fatal stop into a transparent recovery. Step 5a before 5b is what keeps quality high — you evict junk (old command output) before you ever summarize reasoning.

---

## 6. Never lose the thread: the Protected Set + session ledger

"Without losing or missing the context" is a guarantee you have to build explicitly. Compaction is lossy by nature, so define what is **never** eligible for summarization:

- System prompts and `CLAUDE.md` / project memory
- The current task statement / active goal
- The most recent N user turns and assistant turns (verbatim)
- Active file paths, in-flight diffs, and **unresolved errors**
- Any todo/plan list the session is tracking
- Explicitly pinned messages

Everything else is collapsible. Maintain a **per-session ledger** (keyed by session id) holding a rolling structured summary — task, key decisions, files touched, open questions — plus the protected set. On each compaction you **fold new history into the existing summary** rather than re-summarizing from scratch or discarding. That rolling ledger is the actual memory that survives across compactions, and it's what you reconstruct from if a retry (step 7) fires.

This is also where you beat native Claude Code: its compaction is generic, so it can summarize away the one error you were mid-fixing. A task-aware protected set doesn't.

---

## 7. Transport details that bite

- **Streaming:** proxy SSE straight through. All context edits happen on the **request** side, so responses stream normally — Claude Code's UI stays live.
- **count_tokens:** you must implement `POST /v1/messages/count_tokens`. For native, proxy it. For others, compute with the model's tokenizer and, importantly, return the count **after** your edits would apply, so Claude Code's own HUD roughly agrees with reality.
- **Beta headers:** allowlist per provider; strip everything the upstream doesn't understand.
- **Prompt caching:** Anthropic cache-control blocks are meaningless to most non-Anthropic backends; drop them for `native_anthropic: false` to avoid wasted work or errors.
- **Attribution headers:** `X-Claude-Code-Session-Id` / `-Agent-Id` / `-Parent-Agent-Id` are your session + subagent keys — use them for the ledger and for per-session cost tracking.

---

## 8. Suggested stack & rollout

**Stack.** If you extend LiteLLM (recommended), the ledger + protected-set logic lives naturally as a Python `CustomLogger` pre-call hook — you inherit its tokenizer, provider transforms, and beta-header handling. A standalone proxy in Node/TS is viable too (closer to your day-to-day stack) but then you re-implement token counting and Anthropic↔OpenAI translation yourself.

**Rollout, lowest-effort first:**
1. **Config only.** Stand up LiteLLM with `context_management_summary_model` set, correct `max_input_tokens` per model, and beta-header sanitization. Point `ANTHROPIC_BASE_URL`/`ANTHROPIC_AUTH_TOKEN` at it. Verify `/v1/messages` and `/v1/messages/count_tokens` respond before launching Claude Code. **Test whether this alone fixes your long tasks — it often does.**
2. **Add the ledger hook.** Introduce the Protected Set + rolling session summary as a pre-call hook for the cases where generic summarization loses task state.
3. **Add retry-on-overflow (step 7)** and per-session observability (what got compacted, when, how many tokens saved).
4. Only if provider quirks demand it, split into a dedicated proxy service in front of LiteLLM.

**Client-side stopgaps while you build** (native Anthropic path, exported in the shell — not `settings.json`):
- `export CLAUDE_AUTOCOMPACT_PCT_OVERRIDE=70` to compact earlier (can only lower, not raise, the ~83% default).
- `export CLAUDE_CODE_AUTO_COMPACT_WINDOW=<real_window>` to correct the effective window for smaller models.
- Run `/context` to watch the ledger fill and `/compact` manually before big steps; don't set `DISABLE_COMPACT=1` (that removes the only safety net).

---

## Sources
- Anthropic — Compaction, Context editing, Context windows: https://platform.claude.com/docs/en/build-with-claude/compaction
- LiteLLM — Claude Code Context Management (polyfill + summary_model): https://docs.litellm.ai/docs/claude_code_context_management
- LiteLLM — Managing Anthropic Beta Headers: https://docs.litellm.ai/docs/tutorials/claude_code_beta_headers
- Claude Code docs: https://docs.claude.com/en/docs/claude-code/overview
