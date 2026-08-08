"""Upstream-payload capture: the diagnostic for partial-completion debugging."""

from __future__ import annotations

import logging

import httpx
import respx
from conftest import build_conversation, make_request
from fastapi.testclient import TestClient

from ctxproxy.app import create_app
from ctxproxy.backends.openai_compat import _StreamTranscript
from ctxproxy.store.capture import DebugCapture

OPENAI_URL = "http://internal.test/v1/chat/completions"


def body(messages=None, model="our-coder", **kw) -> dict:
    request = make_request(messages or build_conversation(1, 50), model=model, **kw)
    return request.model_dump(exclude_none=True)


def chat_completion(text: str = "done", finish_reason: str = "stop") -> dict:
    return {
        "id": "chatcmpl-1",
        "choices": [
            {"index": 0, "message": {"content": text}, "finish_reason": finish_reason}
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 4},
    }


# --------------------------------------------------------------------------- #
# Unit
# --------------------------------------------------------------------------- #


async def test_capture_disabled_is_a_noop(tmp_path):
    cap = DebugCapture(None)
    assert cap.enabled is False
    await cap.record("s", kind="complete", upstream_model="m", payload={"messages": []})
    assert cap.read("s") == []
    # Nothing must be created anywhere.
    assert not list(tmp_path.iterdir())


async def test_capture_records_and_reads_back(tmp_path):
    cap = DebugCapture(tmp_path / "cap")
    assert cap.enabled is True
    payload = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    await cap.record("sess-1", kind="complete", upstream_model="m", payload=payload,
                     response={"ok": True})

    entries = cap.read("sess-1")
    assert len(entries) == 1
    assert entries[0]["payload"] == payload
    assert entries[0]["response"] == {"ok": True}
    assert entries[0]["message_count"] == 1


async def test_capture_enforces_a_size_cap(tmp_path):
    """A long session appends the whole growing conversation each turn; the file
    must stay bounded, keeping the newest exchanges."""
    cap = DebugCapture(tmp_path / "cap", max_mb=1)  # 1 MB cap -> trims to ~0.5 MB
    big = {"model": "m", "messages": [{"role": "user", "content": "x" * 50_000}]}
    for i in range(60):  # ~50KB each -> ~3MB total, well over the cap
        await cap.record("s", kind="complete", upstream_model="m",
                         payload={**big, "turn": i})

    path = tmp_path / "cap" / "s.jsonl"
    assert path.stat().st_size <= 1024 * 1024, "capture must be trimmed to the cap"
    entries = cap.read("s")
    assert entries, "recent exchanges must survive the trim"
    # Newest kept: the last turn recorded must still be present.
    assert entries[-1]["payload"]["turn"] == 59


def test_capture_prune_removes_old_files(tmp_path):
    import os
    import time

    cap = DebugCapture(tmp_path / "cap")
    path = tmp_path / "cap"
    path.mkdir()
    old = path / "old.jsonl"
    old.write_text('{"kind":"complete"}\n')
    # Backdate it well past any positive TTL.
    ancient = time.time() - 10 * 24 * 3600
    os.utime(old, (ancient, ancient))

    assert cap.prune(ttl_hours=168) == 1
    assert not old.exists()


def test_disabled_capture_prune_is_a_noop(tmp_path):
    assert DebugCapture(None).prune(ttl_hours=168) == 0


def test_stream_transcript_reassembles_a_response():
    t = _StreamTranscript()
    t.observe({"choices": [{"delta": {"reasoning_content": "thinking..."}}]})
    t.observe({"choices": [{"delta": {"content": "Hel"}}]})
    t.observe({"choices": [{"delta": {"content": "lo"}}]})
    t.observe({"choices": [{"delta": {"tool_calls": [{"function": {"name": "edit"}}]}}]})
    t.observe({"choices": [{"delta": {}, "finish_reason": "length"}],
               "usage": {"completion_tokens": 31}})

    result = t.result("max_tokens", 31)
    assert result["text"] == "Hello"
    assert result["tool_calls"] == ["edit"]
    assert result["finish_reason"] == "length"
    assert result["stop_reason"] == "max_tokens"
    assert result["reasoning_chars"] == len("thinking...")


# --------------------------------------------------------------------------- #
# End-to-end through the real backend
# --------------------------------------------------------------------------- #


def test_exchange_is_captured_with_the_exact_upstream_payload(config, tmp_path):
    """The whole point: the file holds what the backend got, not what the client sent."""
    config.server.debug_capture_dir = tmp_path / "cap"

    with TestClient(create_app(config)) as client, respx.mock:
        respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json=chat_completion()))
        client.post(
            "/v1/messages", json=body(), headers={"x-claude-code-session-id": "sess-x"}
        )

    entries = DebugCapture(tmp_path / "cap").read("sess-x")
    assert len(entries) == 1
    # The alias is resolved to the upstream model in the captured payload.
    assert entries[0]["payload"]["model"] == "internal-coder-v2"
    assert entries[0]["upstream_model"] == "internal-coder-v2"
    assert entries[0]["response"]["choices"][0]["finish_reason"] == "stop"


def test_capture_off_by_default_writes_nothing(config, tmp_path):
    """Passthrough guarantee: with capture off, no diagnostic files appear."""
    assert config.server.debug_capture_dir is None
    with TestClient(create_app(config)) as client, respx.mock:
        respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json=chat_completion()))
        client.post("/v1/messages", json=body(), headers={"x-claude-code-session-id": "s"})
    # Only the sessions ledger dir should exist under tmp_path, never a capture dir.
    assert not (tmp_path / "cap").exists()


def test_truncation_is_warned_and_captured(config, tmp_path, caplog):
    """finish_reason=length is the top suspect for partial completion — make it loud."""
    config.server.debug_capture_dir = tmp_path / "cap"

    with TestClient(create_app(config)) as client, respx.mock, \
            caplog.at_level(logging.WARNING, logger="ctxproxy.backends.openai_compat"):
        respx.post(OPENAI_URL).mock(
            return_value=httpx.Response(200, json=chat_completion("half", "length"))
        )
        client.post(
            "/v1/messages", json=body(), headers={"x-claude-code-session-id": "trunc"}
        )

    assert any("output cap" in r.message for r in caplog.records), "expected a truncation warning"
    entries = DebugCapture(tmp_path / "cap").read("trunc")
    assert entries[0]["response"]["choices"][0]["finish_reason"] == "length"


def test_streaming_exchange_is_captured(config, tmp_path):
    config.server.debug_capture_dir = tmp_path / "cap"
    chunks = (
        'data: {"choices":[{"delta":{"content":"par"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"tial"}}]}\n\n'
        'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n\n'
        "data: [DONE]\n\n"
    )
    with TestClient(create_app(config)) as client, respx.mock:
        respx.post(OPENAI_URL).mock(
            return_value=httpx.Response(
                200, text=chunks, headers={"content-type": "text/event-stream"}
            )
        )
        with client.stream(
            "POST", "/v1/messages", json=body(stream=True),
            headers={"x-claude-code-session-id": "strm"},
        ) as response:
            "".join(response.iter_text())

    entries = DebugCapture(tmp_path / "cap").read("strm")
    assert len(entries) == 1
    assert entries[0]["kind"] == "stream"
    assert entries[0]["response"]["text"] == "partial"
    assert entries[0]["response"]["finish_reason"] == "length"
