"""End-to-end context pipeline behaviour."""

from __future__ import annotations

import json

import pytest
from conftest import (
    assistant,
    build_conversation,
    build_text_heavy_conversation,
    make_request,
    tool_result,
    user,
)

from ctxproxy.context.ledger import SessionLedger
from ctxproxy.context.manager import ContextManager
from ctxproxy.store.file import FileLedgerStore
from ctxproxy.tokens.local import HeuristicCounter
from ctxproxy.types_anthropic import Message


class FakeSummarizer:
    """Stands in for a model call. Records what it was asked to fold."""

    def __init__(self) -> None:
        self.calls: list[list[Message]] = []

    async def fold(self, span, ledger):
        self.calls.append(list(span))
        return f"{ledger.summary}\n## State\nFolded {len(span)} messages.".strip()


class FailingSummarizer:
    async def fold(self, span, ledger):
        raise RuntimeError("summariser upstream is down")


@pytest.fixture
def manager(config, tmp_path):
    return ContextManager(config, FileLedgerStore(tmp_path / "sessions"))


@pytest.fixture
def counter():
    return HeuristicCounter()


async def run(manager, request, profile, counter, summarizer, session="s1", **kw):
    return await manager.process(
        request,
        profile=profile,
        session_key=session,
        counter=counter,
        summarizer=summarizer,
        **kw,
    )


# --------------------------------------------------------------------------- #


async def test_small_conversation_passes_through_untouched(
    manager, openai_profile, counter
):
    request = make_request(build_conversation(exchanges=1, payload_size=50))
    before = [m.model_copy(deep=True) for m in request.messages]

    result, _ = await run(manager, request, openai_profile, counter, FakeSummarizer())

    assert not result.triggered
    assert result.events == []
    assert [m.model_dump() for m in request.messages] == [m.model_dump() for m in before]


async def test_large_conversation_is_reduced_below_target(
    manager, openai_profile, counter
):
    request = make_request(build_conversation(exchanges=20, payload_size=4000))
    result, _ = await run(manager, request, openai_profile, counter, FakeSummarizer())

    assert result.triggered
    assert result.events, "expected at least one strategy to fire"
    assert result.tokens_after < result.tokens_before
    assert result.tokens_after <= result.budget


async def test_cheap_strategy_runs_before_summarising(manager, openai_profile, counter):
    """Junk eviction must be attempted before any lossy summarisation."""
    request = make_request(build_conversation(exchanges=20, payload_size=4000))
    summarizer = FakeSummarizer()
    result, _ = await run(manager, request, openai_profile, counter, summarizer)

    assert result.events[0].strategy == "clear_tool_results"


async def test_reduction_preserves_tool_pairing(manager, openai_profile, counter):
    """The invariant that turns a mistake into a hard 400."""
    request = make_request(build_conversation(exchanges=25, payload_size=4000))
    await run(manager, request, openai_profile, counter, FakeSummarizer())

    issued: set[str] = set()
    for msg in request.messages:
        for block in msg.blocks():
            if block.get("type") == "tool_use":
                issued.add(block["id"])
            elif block.get("type") == "tool_result":
                assert block["tool_use_id"] in issued, (
                    f"orphaned tool_result {block['tool_use_id']}"
                )


async def test_reduction_keeps_the_first_message_a_user_turn(
    manager, openai_profile, counter
):
    request = make_request(build_conversation(exchanges=25, payload_size=4000))
    await run(manager, request, openai_profile, counter, FakeSummarizer())
    assert request.messages[0].role == "user"


async def test_protected_error_survives_reduction(manager, openai_profile, counter):
    messages = build_conversation(exchanges=20, payload_size=4000)
    messages.append(assistant("running tests"))
    messages.append(
        Message(
            role="assistant",
            content=[{"type": "tool_use", "id": "toolu_boom", "name": "bash", "input": {}}],
        )
    )
    messages.append(tool_result("toolu_boom", "AssertionError: idempotency key reused", True))

    request = make_request(messages)
    await run(manager, request, openai_profile, counter, FakeSummarizer())

    rendered = "\n".join(
        b.get("content", "") if isinstance(b.get("content"), str) else ""
        for m in request.messages
        for b in m.blocks()
    )
    assert "AssertionError: idempotency key reused" in rendered


async def test_pinned_message_survives_reduction(manager, openai_profile, counter):
    messages = build_conversation(exchanges=20, payload_size=4000)
    messages[5] = user("<ctx-pin> deploy target is eu-west-1, never us-east-1")

    request = make_request(messages)
    await run(manager, request, openai_profile, counter, FakeSummarizer())

    rendered = "\n".join(
        b.get("text", "") for m in request.messages for b in m.blocks()
    )
    assert "eu-west-1" in rendered


# --------------------------------------------------------------------------- #
# Fold reapplication — the property that makes compaction idempotent.
# --------------------------------------------------------------------------- #


async def test_fold_is_reapplied_without_resummarising(manager, openai_profile, counter):
    original = build_text_heavy_conversation(turns=30, size=3000)
    summarizer = FakeSummarizer()

    first, ledger = await run(
        manager, make_request(list(original)), openai_profile, counter, summarizer
    )
    assert first.triggered
    assert ledger.has_fold
    calls_after_first = len(summarizer.calls)

    # Next turn: client resends the same history plus two new messages.
    turn2 = list(original) + [assistant("done"), user("what is left?")]
    second, ledger2 = await run(
        manager, make_request(turn2), openai_profile, counter, summarizer
    )

    assert ledger2.folded_through == ledger.folded_through or ledger2.compaction_count >= 1
    rendered = "\n".join(b.get("text", "") for m in second.request.messages for b in m.blocks())
    assert "<conversation-summary>" in rendered
    assert len(summarizer.calls) <= calls_after_first + 1, (
        "must not re-summarise material already folded"
    )


async def test_fold_invalidated_when_history_diverges(manager, openai_profile, counter):
    original = build_text_heavy_conversation(turns=30, size=3000)
    summarizer = FakeSummarizer()

    _, ledger = await run(
        manager, make_request(list(original)), openai_profile, counter, summarizer
    )
    assert ledger.has_fold

    # Simulate the client compacting on its own / rewinding: history no longer
    # matches the recorded fold.
    diverged = [user("fresh start"), assistant("ok"), user("continue")]
    _, ledger2 = await run(
        manager, make_request(diverged), openai_profile, counter, summarizer
    )

    assert ledger2.folded_through == 0
    assert "carried forward" in ledger2.summary


async def test_summariser_failure_falls_back_to_a_digest(
    manager, openai_profile, counter
):
    """Recovery must never fail — that is the whole point of the retry path."""
    request = make_request(build_conversation(exchanges=25, payload_size=4000))
    result, ledger = await run(
        manager, request, openai_profile, counter, FailingSummarizer()
    )

    assert result.triggered
    assert result.tokens_after < result.tokens_before
    if ledger.compaction_count:
        assert "mechanical digest" in ledger.summary


async def test_ledger_records_task_and_savings(manager, openai_profile, counter):
    request = make_request(build_conversation(exchanges=20, payload_size=4000))
    _, ledger = await run(manager, request, openai_profile, counter, FakeSummarizer())

    assert "idempotency keys" in ledger.task_statement
    assert ledger.total_tokens_saved > 0
    assert ledger.turns_seen == 1


async def test_native_profile_leaves_thinking_blocks_alone(
    manager, native_profile, counter
):
    """Thinking blocks carry signatures; removing them risks a 400."""
    messages = build_conversation(exchanges=6, payload_size=200)
    messages.insert(
        3,
        Message(
            role="assistant",
            content=[{"type": "thinking", "thinking": "deep thoughts", "signature": "sig"}],
        ),
    )
    request = make_request(messages, model="claude-opus-4-8")
    await run(manager, request, native_profile, counter, FakeSummarizer())

    kinds = [b.get("type") for m in request.messages for b in m.blocks()]
    assert "thinking" in kinds


async def test_max_tokens_is_clamped_to_the_backend_ceiling(
    manager, openai_profile, counter
):
    """Claude Code asks for 32K+; a backend capped lower rejects rather than clamps."""
    openai_profile.max_output_tokens = 31_000
    request = make_request(build_conversation(1, 50), max_tokens=64_000)

    result, _ = await run(manager, request, openai_profile, counter, FakeSummarizer())

    assert request.messages is result.request.messages
    assert request.max_tokens == 31_000


async def test_max_tokens_below_the_ceiling_is_left_alone(
    manager, openai_profile, counter
):
    openai_profile.max_output_tokens = 31_000
    request = make_request(build_conversation(1, 50), max_tokens=4_096)
    await run(manager, request, openai_profile, counter, FakeSummarizer())
    assert request.max_tokens == 4_096


async def test_cleared_tool_result_names_the_call_and_says_to_re_run(
    manager, openai_profile, counter
):
    """A bare placeholder invites the model to confabulate the missing output."""
    request = make_request(build_conversation(exchanges=20, payload_size=4000))
    await run(manager, request, openai_profile, counter, FakeSummarizer())

    cleared = [
        b.get("content", "")
        for m in request.messages
        for b in m.blocks()
        if b.get("type") == "tool_result" and str(b.get("content", "")).startswith("[cleared")
    ]
    assert cleared, "expected at least one cleared tool result"
    sample = cleared[0]
    assert "read_file" in sample, "placeholder must name the call that produced it"
    assert "src/mod_" in sample, "placeholder must carry the call's arguments"
    assert "do not" in sample.lower() and "memory" in sample.lower()


async def test_dropped_anchors_are_reattached_to_the_summary():
    """A weak summariser silently generalises identifiers away; we re-attach them."""
    from ctxproxy.config import ModelProfile
    from ctxproxy.context.summarizer import Summarizer

    async def lazy_summariser(request):
        return "## State\nDid some work on the payments code."   # drops every path

    profile = ModelProfile(match="s", backend="internal", context_window=32000)
    summarizer = Summarizer(lazy_summariser, profile)

    span = build_conversation(exchanges=4, payload_size=100)
    span.append(tool_result("toolu_x", "PermissionError: /etc/secrets denied", True))
    ledger = SessionLedger(session_key="k")

    summary = await summarizer.fold(span, ledger)

    assert "src/mod_0.py" in summary, "file path must survive the summariser"
    assert "PermissionError" in summary, "error text must survive the summariser"
    assert "preserved verbatim" in summary


async def test_folded_messages_are_archived_before_being_lost(config, tmp_path, counter):
    """Compaction is the one irreversible step; the raw span must be recoverable."""
    from ctxproxy.store.archive import FoldArchive

    archive = FoldArchive(tmp_path / "archive")
    mgr = ContextManager(config, FileLedgerStore(tmp_path / "sessions"), archive)
    profile = config.profile_for("our-coder")

    request = make_request(build_text_heavy_conversation(turns=30, size=3000))
    _, ledger = await mgr.process(
        request,
        profile=profile,
        session_key="arch1",
        counter=counter,
        summarizer=FakeSummarizer(),
    )
    assert ledger.has_fold

    entries = archive.read("arch1")
    assert entries, "fold must be archived"
    assert entries[0]["message_count"] > 0
    assert entries[0]["original_range"][1] == ledger.folded_through
    # The originals must be recoverable verbatim, not just summarised.
    dumped = json.dumps(entries[0]["messages"])
    assert "idempotency" in dumped or "Step 0" in dumped


async def test_archive_failure_never_breaks_a_request(config, tmp_path, counter):
    class ExplodingArchive:
        async def append(self, *a, **kw):
            raise OSError("disk full")

    mgr = ContextManager(config, FileLedgerStore(tmp_path / "s"), ExplodingArchive())
    request = make_request(build_text_heavy_conversation(turns=30, size=3000))
    result, _ = await mgr.process(
        request,
        profile=config.profile_for("our-coder"),
        session_key="arch2",
        counter=counter,
        summarizer=FakeSummarizer(),
    )
    assert result.triggered


def test_token_drift_flags_undercounting():
    ledger = SessionLedger(session_key="d")
    for _ in range(6):
        ledger.record_token_drift(estimated=1000, actual=1250)
    assert ledger.median_token_drift == 1.25, "must detect that we count 25% low"
    assert ledger.stats()["token_drift_median"] == 1.25


def test_token_drift_ignores_nonsense_samples():
    ledger = SessionLedger(session_key="d")
    assert ledger.record_token_drift(0, 100) is None
    assert ledger.record_token_drift(100, 0) is None
    assert ledger.median_token_drift is None


def test_drift_ignores_small_requests():
    """Fixed chat-template overhead swamps the ratio on tiny prompts."""
    from ctxproxy.routes import _MIN_TOKENS_FOR_DRIFT

    assert _MIN_TOKENS_FOR_DRIFT >= 1000, (
        "threshold must be high enough that per-request overhead is negligible"
    )


async def test_archive_enforces_a_size_cap(tmp_path):
    """A long session must not grow its archive without bound."""
    from ctxproxy.store.archive import FoldArchive

    archive = FoldArchive(tmp_path / "arch", max_mb=1)
    span = build_conversation(exchanges=6, payload_size=20_000)  # ~big fold

    for _ in range(12):
        await archive.append("big", span=span, start=0, end=len(span), summary="s")

    path = tmp_path / "arch" / "big.jsonl"
    size_mb = path.stat().st_size / (1024 * 1024)
    assert size_mb <= 1.0, f"archive grew to {size_mb:.2f} MB despite a 1 MB cap"

    entries = archive.read("big")
    assert entries, "trimming must keep the most recent folds, not wipe the file"


async def test_archive_prune_removes_expired_files(tmp_path):
    import os
    import time

    from ctxproxy.store.archive import FoldArchive

    archive = FoldArchive(tmp_path / "arch")
    await archive.append("old", span=[user("x")], start=0, end=1, summary="s")

    path = tmp_path / "arch" / "old.jsonl"
    stale = time.time() - 100 * 3600
    os.utime(path, (stale, stale))

    assert archive.prune(ttl_hours=72) == 1
    assert not path.exists()
    assert archive.prune(ttl_hours=0) == 0, "ttl 0 must disable pruning"
