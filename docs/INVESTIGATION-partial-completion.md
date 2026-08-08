# Investigation: model completes only part of a multi-step instruction

**Status:** OPEN — diagnostic not yet built. Written as a handoff so a fresh
session (post-`/compact`) can execute without re-deriving anything.

**Symptom (user report):** Not a context-window error anymore. When given
several things to fix in one turn, the model does only a few of them, and
"seems to forget" mid-task.

**Branch:** `beta-v2` (working tree clean at time of writing). All prior fixes
(reasoning→thinking, archive, drift, fingerprint stability, degraded-config
mode) are already merged.

---

## The one rule for this investigation

**Do NOT assume it is a proxy bug.** There are two fundamentally different root
causes and they need different fixes:

1. **Proxy corrupts/truncates the conversation** → fix the proxy.
2. **kimi-k2-7-code-dev is simply a weaker model** than Claude/Sonnet and does
   fewer of N requested fixes → not fixable in the proxy; the answer is model
   choice, prompt shaping, or setting user expectations.

We have twice before recommended changes off too little evidence. **Build the
diagnostic first, look at real data, then fix.** Do not skip to a fix.

---

## STEP 1 (do this first): upstream-payload capture

There is currently **no way to see what actually reaches kimi**
(confirmed: no payload logging in `src/ctxproxy/backends/openai_compat.py`).
Without it we cannot tell corruption from model weakness. Build this first.

**What to add:** an opt-in debug capture that writes, per request, the exact
JSON payload sent upstream and the raw response/stream received. Gate it behind
a config flag so it is off by default (payloads contain full conversation text).

- Add `server.debug_capture_dir: Path | None = None` to `ServerConfig`
  (`src/ctxproxy/config.py`).
- In `OpenAICompatBackend.complete` / `.stream`
  (`src/ctxproxy/backends/openai_compat.py`), when the dir is set, dump:
  `{ts, session, upstream_model, payload, response_or_stream_text}` to a
  per-session JSONL file. Reuse the atomic-write / `asyncio.to_thread` pattern
  from `src/ctxproxy/store/archive.py`.
- Add a CLI: `ctxproxy capture <session>` to pretty-print the last N exchanges,
  mirroring `ctxproxy archive` in `src/ctxproxy/cli.py`.

**The decisive test once it exists:** run the failing multi-fix task, then diff
what Claude Code sent (Anthropic shape) against what we sent kimi (OpenAI
shape). If every instruction is present and correctly ordered upstream → it is
the model. If instructions are dropped/merged/reordered → it is us, and the
capture points straight at the offending translation path.

---

## Ranked hypotheses (check in this order once capture exists)

### H1 — Output truncation from the max_tokens clamp  [HIGH, cheap to confirm]
`src/ctxproxy/context/manager.py:61` clamps `max_tokens` to
`profile.max_output_tokens` (31000 for kimi). kimi is a **reasoning model**:
reasoning tokens count toward output. A "fix these 6 things" turn can exhaust
31000 on reasoning + partial answer and stop at `stop_reason: max_tokens` —
which looks exactly like "did only a few."

We do **not** currently surface `stop_reason` anywhere
(confirmed: no `stop_reason` handling in routes/openai_compat).

- Fix (visibility): log a WARNING whenever a response comes back
  `stop_reason == "max_tokens"` (non-streaming: `openai_to_anthropic` maps
  `finish_reason: length` → `max_tokens`; streaming: the translator already
  tracks `_stop_reason`). Surface it in `ctxproxy sessions` as a counter.
- Confirm: if the failing turns show `max_tokens`, this is a real cause.
- Mitigation options (discuss with user, don't auto-pick): raise the clamp if
  the gateway allows >31000; or set `reasoning_output`/effort lower to spend
  fewer reasoning tokens; or accept it and document.

### H2 — Translation drops/merges instruction content  [MED, capture reveals it]
Every request for the openai backend runs `anthropic_to_openai`
(`src/ctxproxy/translate/request.py`). Audit against a real captured payload:
- Multi-block user turns: `_user_content` — are all text blocks preserved?
- Mixed tool_result + text user turns: `_convert_message` splits tool_results
  into `role:"tool"` messages then forwards remaining text — verify the "also
  do X, Y, Z" text is not lost.
- Assistant turns with text + tool_use: verify text (the plan) is kept, not
  dropped when `tool_calls` present.
- System prompt: `system_blocks()` joined — verify multi-block system (Claude
  Code sends several, one with cache_control) is fully concatenated.

### H3 — Reduction firing when it should not  [LOW — likely not active]
At 215K usable, compaction should be rare. Confirm from logs it is NOT firing
on the failing sessions (`grep "reduction triggered\|applied\|fold" the log`).
If it never fires, H3 is ruled out and clear_tool_results/compact are innocent.
If it IS firing, re-examine the protected set (`src/ctxproxy/context/protect.py`)
— a multi-part instruction in an *old* user turn could be getting summarised.

### H4 — Model capability ceiling  [MED — the honest default if H1–H3 clear]
`kimi-k2-7-code-dev` is a ~7B code model. A small model doing 2–3 of 6 requested
edits is normal and not a bug. If capture shows a faithful upstream payload and
no truncation, **this is the answer** — report it plainly. Levers: a stronger
model for the main loop, or prompt-shaping (ask for a checklist + explicit
"complete ALL items" framing). Do not pretend a proxy change can fix model IQ.

---

## Guardrails (do not break working things)

- **Passthrough is the common path.** Small conversations must remain
  byte-for-byte untouched. Any change here is high-risk — cover with a test that
  asserts an unchanged request in == request out.
- Keep every change behind config, default off, when it adds overhead or writes
  data (capture dir is opt-in).
- Run `.venv/bin/python -m pytest -q` (currently 108 tests) and
  `.venv/bin/ruff check src tests` after each change. Do not commit red.
- The proxy on :4000 may be serving the user's real work. Prefer a second
  instance on :4001 for testing (see how earlier sessions did this), and do NOT
  edit `ctxproxy.yaml` while `--reload` is serving live work — a parse error now
  degrades gracefully (returns 503) but still interrupts in-flight requests.
- Commit docs/diagnostics separately from behavioural fixes.

---

## Fast start for the next session

```bash
cd /Users/moti.yadav/Desktop/selfexp/testzone/agent-mgr
git branch --show-current                 # expect beta-v2
.venv/bin/python -m pytest -q             # expect 108 passing
curl -s localhost:4000/health             # is a live instance serving real work?
grep -E "reduction|max_tokens|clamped" /tmp/ctxproxy-4000.log | tail   # H1/H3 quick signal
```

Then: build STEP 1 (capture) → run the user's failing task → read the capture →
decide between H1/H2/H3/H4 from evidence → fix only what the evidence supports.

## Key files
- `src/ctxproxy/translate/request.py` — Anthropic→OpenAI (H2)
- `src/ctxproxy/translate/response.py` / `sse.py` — response mapping, stop_reason (H1)
- `src/ctxproxy/context/manager.py:61` — max_tokens clamp (H1)
- `src/ctxproxy/backends/openai_compat.py` — where capture (STEP 1) goes
- `src/ctxproxy/context/protect.py` — protected set (H3)
- `src/ctxproxy/cli.py` — add `ctxproxy capture` command
