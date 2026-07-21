"""Reduction strategies, cheapest and least lossy first.

Ordering is the whole point. Stale `tool_result` payloads — file dumps, command
output, search results — are what actually fill an agentic context window, and
evicting them is near-lossless because the assistant already read them and acted.
Summarising is lossy and expensive, so it only runs once eviction is exhausted.

Every strategy maintains the running token total incrementally. Re-counting the
whole request after each edit would be O(n²) across a long session; token counts
are additive per block, so a delta is both cheaper and exact.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol

from ..config import ModelProfile, ReductionPolicy
from ..types_anthropic import (
    Message,
    MessagesRequest,
    ReductionEvent,
    block_text,
    message_text,
)
from .budget import Budget
from .ledger import SessionLedger
from .protect import ProtectedSet, safe_cut_points

log = logging.getLogger(__name__)

SummarizeFn = Callable[[list[Message], SessionLedger], Awaitable[str]]

# Every generated placeholder starts with this, so an already-cleared block is
# recognisable on later turns without adding a non-standard field to the block.
_CLEARED_PREFIX = "[cleared to reclaim context]"

_PATH_RE = re.compile(r"(?:^|[\s\"'`(])(/?(?:[\w.-]+/){1,}[\w.-]+\.\w{1,8})")


@dataclass
class ReductionContext:
    request: MessagesRequest
    profile: ModelProfile
    policy: ReductionPolicy
    budget: Budget
    counter: object  # TokenCounter — untyped to avoid a circular import
    ledger: SessionLedger
    protected: ProtectedSet
    summarize: SummarizeFn
    tokens: int

    # Maps an index in ``request.messages`` back to the client's original array.
    to_original: Callable[[int], int] = lambda i: i
    # Set by ``compact`` so the manager can persist the new watermark.
    new_fold: tuple[int, int] | None = None
    # The raw messages compact folded away — archived before they are lost.
    folded_span: list[Message] | None = None
    # Working index at which a previously folded summary sits, if spliced.
    fold_anchor: int | None = None

    @property
    def messages(self) -> list[Message]:
        return self.request.messages

    @property
    def under_target(self) -> bool:
        return self.tokens <= self.budget.target_at

    def count_block(self, block: dict) -> int:
        return self.counter.count_block(block)  # type: ignore[attr-defined]

    def count_text(self, text: str) -> int:
        return self.counter.count_text(text)  # type: ignore[attr-defined]


class Strategy(Protocol):
    name: str

    async def apply(self, ctx: ReductionContext) -> ReductionEvent | None: ...


# --------------------------------------------------------------------------- #
# 1. Clear stale tool results
# --------------------------------------------------------------------------- #


class ClearToolResults:
    """Replace old tool_result payloads with a placeholder.

    Message structure is preserved exactly — the block keeps its ``tool_use_id``
    so the tool_use/tool_result pairing the API validates stays intact. Only the
    payload goes.
    """

    name = "clear_tool_results"

    async def apply(self, ctx: ReductionContext) -> ReductionEvent | None:
        positions = _tool_result_positions(ctx.messages)
        if not positions:
            return None

        # What each cleared result was the answer to. A bare "[cleared]" tells
        # the model something is missing but not what, which invites it to
        # reconstruct the content from memory — naming the exact call and
        # saying it can be re-run makes re-reading the obvious move instead.
        calls = _tool_call_index(ctx.messages)

        # Exempt the most recent K results: the assistant is very likely still
        # reasoning about those.
        exempt = set(positions[-ctx.policy.keep_recent_tool_results :])

        tokens_before = ctx.tokens
        cleared = 0
        placeholder = ctx.policy.placeholder_text

        for msg_idx, block_idx in positions:
            if ctx.under_target:
                break
            if (msg_idx, block_idx) in exempt:
                continue
            if ctx.protected.protects(msg_idx):
                continue

            block = ctx.messages[msg_idx].blocks()[block_idx]
            tool_use_id = block.get("tool_use_id", "")
            if ctx.protected.protects_tool_result(tool_use_id):
                continue
            content = block.get("content")
            if isinstance(content, str) and content.startswith(_CLEARED_PREFIX):
                # Already cleared on an earlier turn. Detected by content rather
                # than a marker field: anything we add to the block is serialised
                # upstream, and strict backends reject unknown keys.
                continue

            before = ctx.count_block(block)
            replacement = {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": _placeholder_for(tool_use_id, calls, placeholder),
            }
            if block.get("is_error"):
                replacement["is_error"] = True
            after = ctx.count_block(replacement)

            if before - after < 32:
                # Not worth the prompt-cache invalidation.
                continue

            _replace_block(ctx.messages, msg_idx, block_idx, replacement)
            ctx.tokens -= before - after
            cleared += 1

        if not cleared:
            return None

        return ReductionEvent(
            strategy=self.name,
            tokens_before=tokens_before,
            tokens_after=ctx.tokens,
            messages_before=len(ctx.messages),
            messages_after=len(ctx.messages),
            detail=f"cleared {cleared} tool result(s)",
        )


# --------------------------------------------------------------------------- #
# 2. Clear thinking blocks
# --------------------------------------------------------------------------- #


class ClearThinking:
    """Drop reasoning blocks — non-native backends only.

    Deliberately a no-op against native Anthropic. Thinking blocks carry
    signatures the API validates, must be replayed unchanged on the same model,
    and removing them can trigger ordering/signature 400s. Against an
    OpenAI-compatible upstream they are dropped in translation anyway, so
    clearing them here is both free and a correction to our own token
    accounting, which would otherwise over-count what the backend actually sees.
    """

    name = "clear_thinking"

    async def apply(self, ctx: ReductionContext) -> ReductionEvent | None:
        if ctx.profile.native_anthropic:
            return None

        tokens_before = ctx.tokens
        removed = 0

        for idx, msg in enumerate(ctx.messages):
            if ctx.protected.protects(idx):
                continue
            blocks = msg.blocks()
            if not any(b.get("type") in ("thinking", "redacted_thinking") for b in blocks):
                continue

            kept = [b for b in blocks if b.get("type") not in ("thinking", "redacted_thinking")]
            dropped = len(blocks) - len(kept)
            if not kept:
                # An empty content array is rejected by the API.
                kept = [{"type": "text", "text": "[reasoning omitted]"}]

            freed = sum(
                ctx.count_block(b)
                for b in blocks
                if b.get("type") in ("thinking", "redacted_thinking")
            ) - sum(ctx.count_block(b) for b in kept if b not in blocks)

            ctx.messages[idx] = msg.with_blocks(kept)
            ctx.tokens -= max(0, freed)
            removed += dropped

            if ctx.under_target:
                break

        if not removed:
            return None

        return ReductionEvent(
            strategy=self.name,
            tokens_before=tokens_before,
            tokens_after=ctx.tokens,
            messages_before=len(ctx.messages),
            messages_after=len(ctx.messages),
            detail=f"dropped {removed} thinking block(s)",
        )


# --------------------------------------------------------------------------- #
# 3. Compact into the rolling summary
# --------------------------------------------------------------------------- #


class Compact:
    """Fold the oldest unprotected span into the session's rolling summary.

    Folds *into* the existing summary rather than re-summarising from scratch,
    so each message is summarised exactly once over the life of the session and
    the summary text stays stable between compactions.
    """

    name = "compact"

    async def apply(self, ctx: ReductionContext) -> ReductionEvent | None:
        messages = ctx.messages
        if len(messages) < ctx.policy.min_messages_to_compact:
            return None

        start = ctx.fold_anchor if ctx.fold_anchor is not None else ctx.protected.head_end
        limit = _first_protected_at_or_after(ctx.protected, start, ctx.protected.tail_start)

        if limit - start < 2:
            return None

        cut = self._choose_cut(ctx, start, limit)
        if cut is None:
            return None

        span = messages[start:cut]
        # When a prior summary is spliced at ``start`` it is the first element of
        # the span and gets folded back in by the summariser, which sees it as
        # prior context via the ledger.
        summary_text = await self._summarise(ctx, span)

        summary_msg = Message(
            role="user",
            content=[{"type": "text", "text": ctx.ledger.render_summary_block()}],
        )

        tokens_before = ctx.tokens
        messages_before = len(messages)

        removed_tokens = sum(_message_tokens(ctx, m) for m in span)
        added_tokens = _message_tokens(ctx, summary_msg)

        ctx.request.messages = messages[:start] + [summary_msg] + messages[cut:]
        ctx.tokens = ctx.tokens - removed_tokens + added_tokens

        # Record against the *client's* indices, not our working ones, so the
        # watermark still means something on the next turn.
        original_start = ctx.to_original(start)
        original_end = ctx.to_original(cut)
        ctx.new_fold = (original_start, original_end)
        ctx.folded_span = span

        ctx.ledger.summary = summary_text
        ctx.ledger.note_files(_extract_paths(span))
        ctx.fold_anchor = start

        return ReductionEvent(
            strategy=self.name,
            tokens_before=tokens_before,
            tokens_after=ctx.tokens,
            messages_before=messages_before,
            messages_after=len(ctx.request.messages),
            detail=f"folded working[{start}:{cut}] -> original[{original_start}:{original_end}]",
        )

    def _choose_cut(self, ctx: ReductionContext, start: int, limit: int) -> int | None:
        """Smallest safe cut that brings the request under target."""
        # Assume the summary costs about what it did last time, with a floor —
        # exact cost is unknowable before generating it, and under-estimating
        # here just means one more compaction later.
        est_summary = max(600, ctx.count_text(ctx.ledger.render_summary_block()))
        need_to_free = ctx.tokens - ctx.budget.target_at + est_summary
        if need_to_free <= 0:
            return None

        freed = 0
        naive_cut = start
        for idx in range(start, limit):
            freed += _message_tokens(ctx, ctx.messages[idx])
            naive_cut = idx + 1
            if freed >= need_to_free:
                break

        points = safe_cut_points(ctx.messages)
        forward = [p for p in points if naive_cut <= p <= limit]
        if forward:
            cut = min(forward)
        else:
            backward = [p for p in points if start < p <= limit]
            if not backward:
                return None
            cut = max(backward)

        return cut if cut - start >= 2 else None

    async def _summarise(self, ctx: ReductionContext, span: list[Message]) -> str:
        try:
            return await ctx.summarize(span, ctx.ledger)
        except Exception as exc:  # noqa: BLE001
            # Recovery must never fail. A mechanical digest keeps the thread
            # even when the summariser model is down or itself over budget.
            log.warning("summariser failed (%s); using deterministic digest", exc)
            return _deterministic_digest(span, ctx.ledger)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _tool_call_index(messages: list[Message]) -> dict[str, tuple[str, str]]:
    """Map tool_use_id -> (tool name, compact rendering of its input)."""
    index: dict[str, tuple[str, str]] = {}
    for msg in messages:
        for block in msg.blocks():
            if block.get("type") == "tool_use" and block.get("id"):
                index[block["id"]] = (
                    block.get("name", "tool"),
                    _render_input(block.get("input")),
                )
    return index


def _render_input(value: object, limit: int = 120) -> str:
    if not isinstance(value, dict) or not value:
        return ""
    parts = []
    for key, item in value.items():
        text = item if isinstance(item, str) else str(item)
        if len(text) > 60:
            text = text[:60] + "..."
        parts.append(f"{key}={text!r}")
    rendered = ", ".join(parts)
    return rendered[:limit] + ("..." if len(rendered) > limit else "")


def _placeholder_for(
    tool_use_id: str, calls: dict[str, tuple[str, str]], fallback: str
) -> str:
    """Placeholder naming the call, so the model re-runs instead of guessing."""
    call = calls.get(tool_use_id)
    if not call:
        return f"{_CLEARED_PREFIX} {fallback}"
    name, rendered = call
    signature = f"{name}({rendered})" if rendered else name
    return (
        f"{_CLEARED_PREFIX} This was the output of `{signature}`. "
        f"It was removed to stay within the context window, not because it was "
        f"unimportant. Re-run the tool if you need its contents — do not "
        f"reconstruct them from memory."
    )


def _tool_result_positions(messages: list[Message]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for msg_idx, msg in enumerate(messages):
        for block_idx, block in enumerate(msg.blocks()):
            if block.get("type") == "tool_result":
                out.append((msg_idx, block_idx))
    return out


def _replace_block(messages: list[Message], msg_idx: int, block_idx: int, block: dict) -> None:
    blocks = list(messages[msg_idx].blocks())
    blocks[block_idx] = block
    messages[msg_idx] = messages[msg_idx].with_blocks(blocks)


def _message_tokens(ctx: ReductionContext, msg: Message) -> int:
    from ..tokens.base import MESSAGE_OVERHEAD_TOKENS

    return MESSAGE_OVERHEAD_TOKENS + sum(ctx.count_block(b) for b in msg.blocks())


def _first_protected_at_or_after(protected: ProtectedSet, start: int, ceiling: int) -> int:
    """Where the oldest contiguous unprotected run ends.

    Compacting a span that straddles a protected message would violate the
    guarantee, so we only ever fold the run that begins at ``start``.
    """
    for idx in range(start, ceiling):
        if protected.protects(idx):
            return idx
    return ceiling


def _extract_paths(messages: list[Message]) -> list[str]:
    found: list[str] = []
    for msg in messages:
        for block in msg.blocks():
            if block.get("type") == "tool_use":
                for value in _walk_strings(block.get("input")):
                    found.extend(_PATH_RE.findall(value))
    seen: list[str] = []
    for path in found:
        if path not in seen:
            seen.append(path)
    return seen[:50]


def _walk_strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _walk_strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in _walk_strings(v)]
    return []


def _deterministic_digest(span: list[Message], ledger: SessionLedger) -> str:
    """Mechanical fallback summary. No model call, never fails."""
    tools: list[str] = []
    user_turns: list[str] = []
    errors: list[str] = []

    for msg in span:
        for block in msg.blocks():
            btype = block.get("type")
            if btype == "tool_use":
                name = block.get("name", "?")
                if name not in tools:
                    tools.append(name)
            elif btype == "tool_result" and block.get("is_error"):
                errors.append(block_text(block)[:200])
        if msg.role == "user":
            text = message_text(msg).strip()
            if text and not text.startswith("<conversation-summary>"):
                user_turns.append(text[:300])

    parts = [ledger.summary] if ledger.summary else []
    parts.append("## Prior activity (mechanical digest — summariser unavailable)")
    parts.append(f"- Messages folded: {len(span)}")
    if tools:
        parts.append(f"- Tools used: {', '.join(tools[:25])}")
    if user_turns:
        parts.append("- User requests in this span:")
        parts.extend(f"  - {t}" for t in user_turns[-10:])
    if errors:
        parts.append("- Errors encountered:")
        parts.extend(f"  - {e}" for e in errors[-5:])
    return "\n".join(parts)


ALL_STRATEGIES: dict[str, Strategy] = {
    ClearToolResults.name: ClearToolResults(),
    ClearThinking.name: ClearThinking(),
    Compact.name: Compact(),
}


def build_strategies(policy: ReductionPolicy) -> list[Strategy]:
    return [ALL_STRATEGIES[name] for name in policy.strategies]
