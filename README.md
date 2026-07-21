# ctxproxy

A context-management proxy for Claude Code. It sits between the CLI/VS Code
extension and your models, and makes long-running sessions survive the context
window — against **native Anthropic** and against **your own models** behind
LiteLLM, vLLM, or any OpenAI-compatible gateway.

```
              ANTHROPIC_BASE_URL=http://127.0.0.1:4000
┌──────────────┐                          ┌───────────────────────────┐
│ Claude Code  │ ── /v1/messages ───────▶ │        ctxproxy           │
│ CLI + VS Code│ ── /v1/.../count_tokens ▶│                           │
│              │ ◀──── SSE stream ─────── │  • capability profiles    │
└──────────────┘                          │  • token ledger / session │
                                          │  • tiered reduction       │
                                          │  • beta-header sanitising │
                                          └─────────┬─────────────────┘
                        native Anthropic ───────────┤─────── OpenAI-compatible
                        (passthrough, server-side   │        (full translation,
                         compaction preserved)      │         proxy-side compaction)
                                          ┌─────────┴─────────┐
                                          ▼                   ▼
                                   Anthropic API      LiteLLM / vLLM / your gateway
```

---

## Why this exists

Two independent problems stack up.

**Claude Code's own auto-compact is unreliable.** It fires only when free space
minus a hardcoded buffer hits zero. Sometimes it never fires and the session
just stops at `Context limit reached`.

**A custom `ANTHROPIC_BASE_URL` breaks the assumptions compaction depends on.**

| What breaks | Why | What ctxproxy does |
|---|---|---|
| Window size is wrong | Claude Code assumes ~200K regardless of the real model | `context_window` per profile — the real number |
| Token accounting is wrong or missing | Gateways often don't serve `/v1/messages/count_tokens` | Serves it itself; counts locally when the upstream can't |
| Beta headers rejected | Claude Code attaches Anthropic experimental flags to every call | Per-backend allowlist; unknown flags stripped |
| Compaction itself overflows | Client-side compaction summarises everything in one shot | Chunked, folding summarisation that can't exceed the window |
| Hard failure at the limit | No recovery path | Detects upstream overflow, force-reduces, retries once |

The fix is to move context management to the transport layer, where window size,
token accounting, and compaction strategy are all things you control per model.

---

## Quickstart

```bash
uv venv && uv pip install -e ".[tokenizers]"
ctxproxy init                             # writes ctxproxy.yaml from the example
code ctxproxy.yaml                        # set base_url + the REAL context_window
ctxproxy doctor                           # validate config, probe backends
ctxproxy serve                            # add --reload while iterating
```

`ctxproxy.yaml` is gitignored on purpose — it names your gateway and credential
setup. `config.example.yaml` is the committed template; each developer runs
`ctxproxy init` and edits their own copy. Session ledgers under `~/.ctxproxy/`
are ignored too: they contain rolling summaries of real conversations,
including source, file paths, and error text.

Then point Claude Code at it:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:4000
export ANTHROPIC_AUTH_TOKEN=<whatever your backend expects>
claude
```

For the VS Code extension, set the same two variables in the environment VS Code
is launched from (or in its terminal profile) and reload the window.

**Run `ctxproxy doctor` before you launch Claude Code.** Most "it just fails"
reports are a missing API key or a `context_window` that doesn't match the model
actually being served.

---

## How it works

Per incoming request:

1. **Resolve** session identity and the model's capability profile.
2. **Re-apply** any fold recorded on a previous turn — deterministic, no model call.
3. **Count** input tokens (upstream endpoint if available, local tokenizer otherwise).
4. **Under the trigger?** Pass through completely untouched.
5. **Over?** Reduce, cheapest strategy first, until under target.
6. **Persist** the session ledger.

### Reduction order

Ordering is the whole design. Junk is evicted before reasoning is ever summarised.

| # | Strategy | Loss | What it does |
|---|---|---|---|
| 1 | `clear_tool_results` | near-zero | Replaces stale `tool_result` payloads with a placeholder, keeping `tool_use_id` so pairing stays valid. Old file dumps and command output are what actually fill an agentic window, and the assistant already read them. |
| 2 | `clear_thinking` | zero | Drops reasoning blocks — **non-native backends only**, where they're dropped in translation anyway. Never touched on native Anthropic: they carry signatures the API validates. |
| 3 | `compact` | lossy | Folds the oldest unprotected span into the session's rolling summary. |

### The Protected Set

Compaction is lossy, so what's *ineligible* is declared explicitly:

- The task statement (everything up to and including the first real user turn)
- The most recent N turns, verbatim
- **Unresolved errors** — an errored `tool_result` and the call that produced it
- The current plan/todo state (only the latest; superseded plans are dropped)
- Anything the user pinned with `<ctx-pin>`

This is where it beats generic client-side compaction, which is task-agnostic
and can summarise away the exact error you're mid-way through fixing.

### The fold watermark

Claude Code is stateless — it resends the *entire* conversation every turn. So
after compacting, ctxproxy records how many original messages were folded, plus
a fingerprint of that span. On every later turn it replays the same fold
deterministically and only summarises material it hasn't seen before.

```
turn 1  reduction triggered   tokens=46327 pct=514.7 trigger=5400
turn 1  applied clear_tool_results  saved=19920  tokens=26407
turn 1  applied compact            saved=23693  tokens=2714
turn 2  fold reapplied  replaced=[1:69]  messages=8      ← no model call
```

Each message is summarised exactly once over the life of a session, and the
prompt prefix stays stable between compactions. If the client rewinds, branches,
or runs its own compaction, the fingerprint stops matching and the fold is
rebuilt rather than silently mis-spliced.

### Why `count_tokens` reports a capped number

Claude Code drives its own auto-compact off `/v1/messages/count_tokens`. If we
returned the raw pre-reduction count, the client would compact *on top of* us —
double compaction, double loss. Above the trigger we report the post-reduction
figure, which is what the next `/v1/messages` call is guaranteed to produce. In
practice this means the client's context HUD sits stable around the target ratio
and never fires its own compaction.

---

## Configuration

### Capability profiles

Everything hinges on these. First match wins, so keep the catch-all last.

```yaml
profiles:
  - match: "claude-*"              # glob against the request's `model`
    backend: anthropic
    native_anthropic: true         # the master switch
    context_window: 200000         # the REAL window
    output_reserve: 32000          # raised automatically to cover max_tokens
    safety_buffer: 12000           # headroom for the summarisation pass itself
    supports_count_tokens: true
    supports_prompt_caching: true
    supports_server_compaction: true
    tokenizer: {kind: upstream}

  - match: "our-coder"
    backend: internal
    upstream_model: internal-coder-v2   # what the gateway actually calls it
    native_anthropic: false
    context_window: 128000
    supports_count_tokens: false        # OpenAI-compatible has no such endpoint
    tokenizer: {kind: heuristic, chars_per_token: 3.5, safety_multiplier: 1.10}
    summarizer: "claude-*"              # run the task cheap, summarise strong
```

`native_anthropic` decides whether Anthropic-specific features (cache_control,
thinking blocks, `context_management`, beta flags) pass through untouched or get
translated and stripped.

`summarizer` points at another profile. Running the task on a cheap model while
writing summaries with a strong one is usually the right trade — a bad summary
poisons every subsequent turn.

### Reasoning models

Models like kimi-k2, DeepSeek-R1 and QwQ return their chain of thought in a
separate `reasoning_content` field. Anthropic has no client-visible equivalent,
so the proxy decides what becomes of it:

```yaml
backends:
  - name: internal
    openai_reasoning_output: thinking   # thinking | text | drop
```

| Mode | Effect |
|---|---|
| `thinking` | Anthropic `thinking` blocks — client shows collapsed reasoning. **Default.** |
| `text` | Inline with the answer. Visible, but noisy. |
| `drop` | Cleanest transcript, but the client sees nothing while the model reasons. |

This is a latency decision as much as a cosmetic one. Measured against a model
that reasons for several seconds before answering: `drop` gave **4867ms** to
first visible output, `thinking` gives **~200ms**. Suppressing reasoning means
suppressing *all* output during that window, which reads as the proxy hanging.

Whatever the setting, reasoning is still surfaced if a turn would otherwise be
empty — some models leave `content` empty and put the answer in
`reasoning_content`, so no setting can silence a response.

### Output caps

`max_output_tokens` is a hard ceiling separate from the context window. Claude
Code routinely asks for 32K+, and a backend capped lower **rejects the request
outright rather than clamping**. Set it whenever your model advertises one:

```yaml
context_window: 262144      # what fits going in
max_output_tokens: 31000    # what the backend will actually generate
```

### Retention

State lives under `server.state_dir` (default `~/.ctxproxy/sessions/`):

| What | Size | Read when |
|---|---|---|
| Ledgers `<session>.json` | ~1–30 KB | **Every request** — kept deliberately small |
| Archives `archive/<session>.jsonl` | up to `archive_max_mb` | Only by you, via `ctxproxy archive` |

```yaml
session_ttl_hours: 720       # ledgers — small, keep them a month
archive_ttl_hours: 168       # archives — large; omit to reuse session_ttl_hours
prune_interval_hours: 6      # sweep while running; 0 = startup only
archive_max_mb: 25           # per-session cap, oldest folds dropped first
```

Worth keeping these apart. Ledgers are kilobytes and stay useful for the life of
a session; archives are whole transcripts and are almost only ever read while
debugging something recent, so a long ledger TTL should not drag them along.

Worst case is `archive_max_mb` x sessions that compacted inside the **archive**
window. Set `archive_max_mb: 0` to disable archiving entirely if disk is tight.

### Policy

```yaml
policy:
  trigger_ratio: 0.75          # reduce above this fraction of budget
  target_ratio: 0.55           # reduce down to this
  keep_recent_turns: 8
  keep_recent_tool_results: 6
  overflow_retry_enabled: true
  overflow_retry_ratio: 0.40
```

The gap between `trigger_ratio` and `target_ratio` is **hysteresis**. Without
it, every turn lands just over the trigger again and you pay a summarisation
pass per turn.

Budget arithmetic:

```
usable    = context_window − max(output_reserve, request.max_tokens) − safety_buffer
trigger   = usable × trigger_ratio
target    = usable × target_ratio
```

### Tokenizers

| kind | When | Notes |
|---|---|---|
| `upstream` | Native Anthropic | Proxies `count_tokens`; falls back locally if the endpoint is missing |
| `tiktoken` | Non-native, accuracy matters | `uv pip install -e ".[tokenizers]"`. Not Anthropic's tokenizer — `safety_multiplier` absorbs the gap |
| `heuristic` | Always available | chars/ratio. Deliberately pessimistic: under-counting causes hard failures, over-counting just wastes a little window |

---

## Operations

```bash
ctxproxy serve --reload      # restart on source *or* ctxproxy.yaml change
ctxproxy doctor              # validate config, probe backends
ctxproxy sessions            # what compaction has saved, per session
ctxproxy inspect <key>       # a session's rolling summary — what survived
ctxproxy archive <key>       # the raw messages compaction folded away
ctxproxy reset [<key>]       # clear session state (ledger + archive)
curl localhost:4000/health   # profiles and backends
curl localhost:4000/stats    # aggregate savings
```

Every reduction decision is logged with before/after token counts:

```
reduction triggered  session=sess-A tokens=46327 pct=514.7 trigger=5400 target=3600
applied clear_tool_results  session=sess-A saved=19920 tokens=26407 detail=cleared 16 tool result(s)
applied compact  session=sess-A saved=23693 tokens=2714 detail=folded working[1:69] -> original[1:69]
```

Set `log_format: json` for machine-readable output.

### When a session seems to have forgotten something

`ctxproxy inspect` shows what the model can still see; `ctxproxy archive` shows
what was removed to get there. Comparing the two is the fastest way to tell a
summariser problem from a model problem.

### Tokenizer drift

Local token counting is an approximation — `cl100k` is not the tokenizer most
non-Anthropic models use, and **under-counting is the dangerous direction**: it
delays compaction until the request no longer fits the real window.

The proxy compares its estimate against the backend's reported `prompt_tokens`
and tracks a rolling median per session, shown in `ctxproxy sessions`:

```
SESSION                       TURNS  COMPACT      SAVED   DRIFT  UPDATED
```

Above `1.10x` it warns with a concrete `safety_multiplier` to set. Two things
worth knowing: samples under 5000 tokens are ignored (backends add fixed
chat-template overhead that swamps the ratio on small prompts), so **`DRIFT`
stays `-` until you have genuinely large turns** — that is correct, not broken.

---

## Troubleshooting

**Still hitting context limits.** `context_window` is almost certainly wrong for
the model actually being served. Check `ctxproxy doctor` output, then lower
`trigger_ratio`. If the numbers look right, your tokenizer is under-counting —
raise `safety_multiplier` or switch to `tiktoken`.

**"invalid beta flag" from a non-Anthropic upstream.** Set
`allowed_beta_flags: []` on that backend. `["*"]` (the default) passes
everything through, which is right only for genuine Anthropic endpoints.

**400 on `stream_options`.** Older self-hosted gateways reject it. Set
`openai_stream_usage: false` on the backend.

**Compaction fires every single turn.** `trigger_ratio` and `target_ratio` are
too close together. Widen the gap.

**Context feels lost after compaction.** Check `ctxproxy inspect <session>` to
see what the summary actually retained. Raise `keep_recent_turns`, point
`summarizer` at a stronger model, or mark critical facts with `<ctx-pin>` in a
message — pinned content is never eligible for reduction.

**Sessions aren't being tracked (`fp_...` keys everywhere).** The
`X-Claude-Code-Session-Id` header isn't reaching the proxy. This is a graceful
degradation, not a failure — identity falls back to a fingerprint of the
conversation prefix, which is stable as the conversation grows. Worth checking
whether something upstream is stripping headers.

---

## Relationship to LiteLLM

These are complementary, not competing.

LiteLLM already polyfills Anthropic's `context_management` beta across providers
and is an excellent provider-translation layer. If you only need generic
compaction, configure it directly — set `context_management_summary_model` (the
polyfill is a silent no-op without it) and the correct `max_input_tokens` per
model, and try that first.

Reach for ctxproxy when you need what a generic polyfill can't give you:

- **Deterministic context-preservation guarantees** — the Protected Set means
  "never summarise the active error / the task statement / the current plan" is
  enforced, not hoped for.
- **A rolling ledger** that folds rather than re-summarising, so each message is
  compressed once and the prefix stays stable.
- **Retry-on-overflow**, turning a hard failure into a transparent recovery.
- **Per-session observability** — what got compacted, when, and how much it saved.

They compose: point a ctxproxy backend at LiteLLM (`kind: openai`,
`base_url: http://litellm:4000`) and let LiteLLM own provider translation while
ctxproxy owns context.

---

## Development

```bash
uv pip install -e ".[dev,tokenizers]"
pytest -q          # 105 tests
ruff check src tests
```

The tests worth knowing about:

- `test_protect.py` — cut-point safety. Splitting a `tool_use`/`tool_result` pair
  is a hard API 400, so this is the invariant that matters most.
- `test_manager.py` — reduction ordering, fold reapplication, divergence
  detection, summariser-failure fallback.
- `test_translate.py` — Anthropic ↔ OpenAI both ways, including the SSE state
  machine (every opened block closed exactly once).
- `test_imports.py` — each module imported in a fresh interpreter, because
  import cycles are order-dependent and only bite one entry point.

### Layout

```
config.py              capability profiles, policy, backends
session.py             session identity (header, else content fingerprint)
tokens/                counting: upstream, tiktoken, heuristic
context/
  budget.py            window arithmetic
  protect.py           the Protected Set + cut-point safety
  strategies.py        clear_tool_results, clear_thinking, compact
  ledger.py            rolling summary + fold watermark
  summarizer.py        chunked folding summarisation
  manager.py           the pipeline
backends/              anthropic_native (passthrough), openai_compat (translating)
translate/             request, response, and SSE conversion
store/                 file-backed ledger persistence (atomic writes)
```
