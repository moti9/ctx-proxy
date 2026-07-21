"""The Protected Set and cut-point safety — the correctness core."""

from __future__ import annotations

from conftest import (
    assistant,
    assistant_tool_call,
    build_conversation,
    tool_result,
    user,
)

from ctxproxy.config import ReductionPolicy
from ctxproxy.context.protect import classify, safe_cut_points
from ctxproxy.types_anthropic import Message


def test_task_statement_is_protected(policy):
    messages = build_conversation(exchanges=10)
    protected = classify(messages, policy)
    assert protected.protects(0)
    assert protected.reasons[0] == "task"


def test_recent_turns_are_protected(policy):
    messages = build_conversation(exchanges=10)
    protected = classify(messages, policy)
    assert protected.protects(len(messages) - 1)
    assert protected.tail_start < len(messages)


def test_errored_tool_result_and_its_call_are_protected(policy):
    messages = [
        user("fix the build"),
        assistant_tool_call("toolu_a", name="bash"),
        tool_result("toolu_a", "ok"),
        assistant("looks fine"),
        user("now run the tests"),
        assistant_tool_call("toolu_b", name="bash"),
        tool_result("toolu_b", "FAILED: 3 tests", is_error=True),
    ]
    protected = classify(messages, policy)
    assert protected.protects(6), "errored result must be protected"
    assert protected.protects(5), "the call that produced the error must be too"
    assert "toolu_b" in protected.tool_result_ids


def test_explicit_pin_is_protected(policy):
    messages = build_conversation(exchanges=12)
    messages[5] = user("<ctx-pin> the API base path is /v2/payments")
    protected = classify(messages, policy)
    assert protected.protects(5)
    assert protected.reasons[5] == "pin"


def test_only_the_latest_plan_is_protected(policy):
    messages = build_conversation(exchanges=12)
    messages[3] = Message(
        role="assistant",
        content=[{"type": "tool_use", "id": "t1", "name": "TodoWrite", "input": {}}],
    )
    messages[9] = Message(
        role="assistant",
        content=[{"type": "tool_use", "id": "t2", "name": "TodoWrite", "input": {}}],
    )
    protected = classify(messages, policy)
    assert protected.protects(9), "current plan is live state"
    assert protected.reasons.get(3) != "plan", "superseded plan must not be pinned"


# --------------------------------------------------------------------------- #
# Cut-point safety: violating this produces an API 400, not a soft failure.
# --------------------------------------------------------------------------- #


def test_cut_points_never_split_a_tool_pair():
    messages = build_conversation(exchanges=8)
    points = safe_cut_points(messages)

    for point in points:
        if point in (0, len(messages)):
            continue
        # No message at a cut point may carry a tool_result, because its
        # matching tool_use lives in the message immediately before it.
        blocks = messages[point].blocks()
        assert not any(b.get("type") == "tool_result" for b in blocks), (
            f"cut at {point} would orphan a tool_use/tool_result pair"
        )


def test_every_tool_result_index_is_excluded():
    messages = build_conversation(exchanges=6)
    points = set(safe_cut_points(messages))
    for i, msg in enumerate(messages):
        if any(b.get("type") == "tool_result" for b in msg.blocks()):
            assert i not in points


def test_assistant_messages_are_valid_cut_points():
    """Restricting cuts to user turns would leave tool-heavy transcripts uncuttable."""
    messages = [
        user("go"),
        assistant_tool_call("t1"),
        tool_result("t1", "data"),
        assistant("done"),
        assistant("more"),
    ]
    points = safe_cut_points(messages)
    assert 3 in points and 4 in points
    assert 2 not in points


def test_removing_a_safe_span_keeps_every_pair_intact():
    messages = build_conversation(exchanges=10)
    points = [p for p in safe_cut_points(messages) if 0 < p < len(messages)]
    start, end = points[1], points[5]

    survivors = messages[:start] + messages[end:]

    issued: set[str] = set()
    for msg in survivors:
        for block in msg.blocks():
            if block.get("type") == "tool_use":
                issued.add(block["id"])
            if block.get("type") == "tool_result":
                assert block["tool_use_id"] in issued, "orphaned tool_result"


def test_tail_snaps_back_to_a_safe_point():
    policy = ReductionPolicy(keep_recent_turns=3)
    messages = build_conversation(exchanges=10)
    protected = classify(messages, policy)
    blocks = messages[protected.tail_start].blocks()
    assert not any(b.get("type") == "tool_result" for b in blocks)
