# Context reduction

Context reduction is the reason ctxproxy exists. This page explains the budget
math, the Protected Set, the reduction strategies, and how folding is made
stable across turns.

## Budget math

From `context/budget.py`:

```
output_reserve = max(profile.output_reserve, request.max_tokens)
usable          = profile.context_window - output_reserve - profile.safety_buffer
trigger_at      = usable * policy.trigger_ratio
target_at       = usable * policy.target_ratio
```

- `context_window` is the **real** upstream window. Getting this wrong is the
  most common cause of hard failures.
- `output_reserve` ensures generation headroom. It is at least the requested
  `max_tokens`.
- `safety_buffer` covers counting error and the summarization pass itself.
- The gap between `trigger_at` and `target_at` is **hysteresis**. Without it,
  every turn would land just above the trigger and pay for a summarization pass.

If the request is at or below `trigger_at`, it passes through untouched. If it
is above, the proxy reduces until it is at or below `target_at`.

## The Protected Set

Compaction is lossy, so `context/protect.py` declares what is ineligible.

| Category | What is protected | Why |
|----------|-------------------|-----|
| Task | Everything up to and including the first plain user turn | This is what the session is for. |
| Recent turns | The last `keep_recent_turns` messages, snapped to a safe cut point | Keeps the live conversation intact. |
| Unresolved errors | Errored `tool_result` and the assistant turn that produced it | Summarizing away the error you are fixing is catastrophic. |
| Pins | Messages containing `<ctx-pin>` | Explicit user guarantees. |
| Current plan | The most recent todo/plan tool call | Live state; superseded plans are ignored. |

### Cut-point safety

The API requires every `tool_use` block to be answered by a `tool_result` block
with the same `tool_use_id` in the immediately following message. Therefore a
cut is safe only when the message at the cut carries no `tool_result` blocks.

`snap_back_to_safe_cut()` snaps the recent-turn boundary backwards to the
nearest safe point so the tail never splits a tool pair.

## Reduction strategies

Strategies run in the order configured by `policy.strategies`. The default is:

```yaml
strategies:
  - clear_tool_results
  - clear_thinking
  - compact
```

They are implemented in `context/strategies.py`.

### 1. clear_tool_results

Replace stale `tool_result` payloads with a placeholder while keeping the
`tool_use_id`. The most recent `keep_recent_tool_results` are exempt.

This is near-lossless: old file dumps and command output are what actually fill
an agentic context window, and the assistant already read them.

### 2. clear_thinking

Drop `thinking` and `redacted_thinking` blocks. This is a no-op for native
Anthropic because those blocks carry signatures the API validates. For
non-native backends the blocks are dropped in translation anyway, so clearing
them here corrects the proxy's own token accounting.

### 3. compact

Fold the oldest unprotected span into the session's rolling summary. This is the
only lossy step and runs only when the cheaper strategies did not free enough.

`Compact._choose_cut()`:

1. Estimates the summary cost from the existing ledger.
2. Walks messages from the protected head end forward until enough tokens would
   be freed.
3. Snaps the cut forward (or backward) to the nearest safe cut point.
4. Invokes the summarizer on that span.

The summary is then spliced into the conversation as a single user message
containing `<conversation-summary>...</conversation-summary>`.

## Stable folding

Because Claude Code resends the full conversation every turn, ctxproxy records
the fold as a watermark in the ledger:

- `fold_start` — original index where the summary begins.
- `folded_through` — original index of the first message *not* folded.
- `fold_signature` — hash of the boundary messages and span length.

On the next turn, `_reapply_fold()` in `context/manager.py` splices the recorded
summary back in without calling the model. If the conversation diverges, the
signature mismatches and the fold is invalidated. The existing summary is kept
as prior context but the watermark is reset.

Benefits:

- Each message is summarized at most once.
- The prompt prefix is stable between compactions.
- Prompt caching is not constantly invalidated by re-summarizing.

## Fold watermark on divergence

```
turn 1  reduction triggered   tokens=46327 pct=514.7 trigger=5400 target=3600
turn 1  applied clear_tool_results  saved=19920  tokens=26407
turn 1  applied compact            saved=23693  tokens=2714
turn 2  fold reapplied  replaced=[1:69]  messages=8      ← no model call
```

## When reduction still fails

If tokens remain above `usable` after all strategies, the proxy logs a warning
and sends the request anyway. At that point the only knobs are:

- Lower `trigger_ratio` to start reducing earlier.
- Increase `safety_buffer` or `safety_multiplier` if the tokenizer undercounts.
- Reduce `keep_recent_turns` to allow more compaction.
- Ensure `context_window` matches the real model window.
