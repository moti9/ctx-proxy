"""The Protected Set — what compaction is never allowed to touch.

"Without losing context" is not something you get for free from a summariser;
it is a guarantee you build by declaring, up front, which parts of the
conversation are ineligible for reduction. This module is that declaration.

This is also where we beat generic client-side compaction: Claude Code's own
compaction is task-agnostic, so it can summarise away the very error you were
mid-way through fixing. A task-aware protected set does not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..config import ReductionPolicy
from ..types_anthropic import (
    Message,
    has_error_result,
    is_plain_user_turn,
    message_text,
    tool_result_ids,
)

# Explicit user pins. Anything wrapped in these survives every reduction.
PIN_PATTERN = re.compile(r"<ctx-pin>|ctxproxy:\s*pin|<!--\s*ctx-pin\s*-->", re.IGNORECASE)

# Claude Code's plan/todo tooling. The *current* plan is load-bearing state;
# superseded plans are noise, so only the most recent one is protected.
PLAN_TOOL_PATTERN = re.compile(r"todo|plan|task_?list", re.IGNORECASE)

# How far back an errored tool result still counts as "possibly unresolved".
ERROR_LOOKBACK_MESSAGES = 40


@dataclass
class ProtectedSet:
    """Message indices and tool-result ids that reduction must preserve."""

    indices: set[int] = field(default_factory=set)
    reasons: dict[int, str] = field(default_factory=dict)
    tool_result_ids: set[str] = field(default_factory=set)

    # Exclusive end of the protected head (task statement and preamble).
    head_end: int = 0
    # Inclusive start of the protected tail (recent verbatim turns).
    tail_start: int = 0

    def protects(self, index: int) -> bool:
        return index in self.indices

    def protects_tool_result(self, tool_use_id: str) -> bool:
        return tool_use_id in self.tool_result_ids

    def add(self, index: int, reason: str) -> None:
        self.indices.add(index)
        # First reason wins — it is the most specific one, since callers add in
        # priority order.
        self.reasons.setdefault(index, reason)

    def summary(self) -> str:
        by_reason: dict[str, int] = {}
        for reason in self.reasons.values():
            by_reason[reason] = by_reason.get(reason, 0) + 1
        return ", ".join(f"{k}:{v}" for k, v in sorted(by_reason.items()))


def classify(messages: list[Message], policy: ReductionPolicy) -> ProtectedSet:
    """Compute the protected set for a conversation."""
    protected = ProtectedSet()
    n = len(messages)
    if n == 0:
        return protected

    # 1. The task statement. Everything up to and including the first genuine
    #    user turn — this is what the whole session is *for*.
    head_end = 0
    for i, msg in enumerate(messages):
        if is_plain_user_turn(msg):
            head_end = i + 1
            break
    else:
        head_end = min(1, n)

    for i in range(head_end):
        protected.add(i, "task")
    protected.head_end = head_end

    # 2. Recent turns, verbatim. Snapped backwards to a safe cut point so we
    #    never protect a tool_result whose tool_use sits outside the window.
    raw_tail = max(head_end, n - policy.keep_recent_turns)
    tail_start = _snap_back_to_safe_cut(messages, raw_tail, floor=head_end)
    for i in range(tail_start, n):
        protected.add(i, "recent")
    protected.tail_start = tail_start

    # 3. Unresolved errors. An error you are mid-way through fixing is the
    #    single worst thing to summarise away.
    error_floor = max(head_end, n - ERROR_LOOKBACK_MESSAGES)
    for i in range(error_floor, n):
        if has_error_result(messages[i]):
            protected.add(i, "error")
            protected.tool_result_ids |= tool_result_ids(messages[i])
            # The assistant turn that issued the failing call gives the error
            # its meaning; without it the surviving result is uninterpretable.
            if i > 0:
                protected.add(i - 1, "error")

    # 4. Explicit pins.
    for i, msg in enumerate(messages):
        if PIN_PATTERN.search(message_text(msg)):
            protected.add(i, "pin")
            protected.tool_result_ids |= tool_result_ids(msg)

    # 5. The current plan/todo state (most recent only — earlier plans are
    #    superseded and actively misleading if preserved).
    for i in range(n - 1, -1, -1):
        if _carries_plan(messages[i]):
            protected.add(i, "plan")
            protected.tool_result_ids |= tool_result_ids(messages[i])
            break

    return protected


def _carries_plan(msg: Message) -> bool:
    for block in msg.blocks():
        if block.get("type") == "tool_use" and PLAN_TOOL_PATTERN.search(block.get("name", "")):
            return True
    return False


def safe_cut_points(messages: list[Message]) -> list[int]:
    """Indices at which the conversation may be split.

    A cut at index ``i`` separates ``messages[:i]`` from ``messages[i:]``. The
    only structural invariant the API enforces here is that every assistant
    ``tool_use`` block is answered by a ``tool_result`` block carrying the same
    id, in the message that immediately follows it.

    So a cut is safe exactly when ``messages[i]`` carries no ``tool_result``
    blocks — cutting there cannot orphan a pair, because there is no pair
    straddling the boundary. Note this admits assistant messages as cut points,
    not just user turns; restricting to user turns would be safe but needlessly
    coarse, and on tool-heavy agentic transcripts it can leave no viable cut at
    all.
    """
    points = [0]
    points.extend(
        i for i, msg in enumerate(messages) if i > 0 and not tool_result_ids(msg)
    )
    if len(messages) not in points:
        points.append(len(messages))
    return points


def _snap_back_to_safe_cut(messages: list[Message], index: int, *, floor: int) -> int:
    """Largest safe cut point <= index, never below floor."""
    candidates = [p for p in safe_cut_points(messages) if floor <= p <= index]
    return max(candidates) if candidates else floor


def snap_forward_to_safe_cut(messages: list[Message], index: int, *, ceiling: int) -> int:
    """Smallest safe cut point >= index, never above ceiling."""
    candidates = [p for p in safe_cut_points(messages) if index <= p <= ceiling]
    return min(candidates) if candidates else ceiling
