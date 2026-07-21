"""Session identity — losing this means losing the rolling summary."""

from __future__ import annotations

from conftest import assistant, build_conversation, make_request, user

from ctxproxy.session import derive_session_key


def test_header_is_used_when_present():
    key = derive_session_key(
        {"X-Claude-Code-Session-Id": "abc-123"}, make_request(build_conversation(1))
    )
    assert key.source == "header"
    assert "abc" in key.value


def test_subagents_get_their_own_ledger():
    """A subagent is a separate conversation; folding them together corrupts both."""
    headers = {"x-claude-code-session-id": "s1"}
    parent = derive_session_key(headers, make_request(build_conversation(1)))
    child = derive_session_key(
        {**headers, "x-claude-code-agent-id": "agent-7", "x-claude-code-parent-agent-id": "s1"},
        make_request(build_conversation(1)),
    )
    assert parent.value != child.value
    assert child.is_subagent and not parent.is_subagent


def test_fingerprint_is_used_when_the_header_is_absent():
    key = derive_session_key({}, make_request(build_conversation(1)))
    assert key.source == "fingerprint"
    assert key.value.startswith("fp_")


def test_fingerprint_is_stable_as_the_conversation_grows():
    """Claude Code resends full history each turn; identity must not drift."""
    base = build_conversation(exchanges=3)
    turn1 = derive_session_key({}, make_request(list(base)))
    turn2 = derive_session_key(
        {}, make_request(base + [assistant("more"), user("keep going")])
    )
    assert turn1.value == turn2.value


def test_fingerprint_differs_for_a_different_task():
    a = derive_session_key({}, make_request([user("refactor billing")]))
    b = derive_session_key({}, make_request([user("write a changelog")]))
    assert a.value != b.value


def test_fingerprint_differs_across_models():
    messages = build_conversation(1)
    a = derive_session_key({}, make_request(list(messages), model="our-coder"))
    b = derive_session_key({}, make_request(list(messages), model="claude-opus-4-8"))
    assert a.value != b.value
