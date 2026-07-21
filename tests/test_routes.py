"""HTTP surface, with the upstream mocked."""

from __future__ import annotations

import json

import httpx
import pytest
import respx
from conftest import build_conversation, make_request
from fastapi.testclient import TestClient

from ctxproxy.app import create_app

OPENAI_URL = "http://internal.test/v1/chat/completions"
ANTHROPIC_URL = "https://api.anthropic.test/v1/messages"


@pytest.fixture
def client(config):
    with TestClient(create_app(config)) as test_client:
        yield test_client


def chat_completion(text: str = "done") -> dict:
    return {
        "id": "chatcmpl-1",
        "choices": [{"index": 0, "message": {"content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 4},
    }


def body(messages=None, model="our-coder", **kw) -> dict:
    request = make_request(messages or build_conversation(1, 50), model=model, **kw)
    return request.model_dump(exclude_none=True)


# --------------------------------------------------------------------------- #


def test_health_lists_profiles(client):
    payload = client.get("/health").json()
    assert payload["status"] == "ok"
    assert {p["match"] for p in payload["profiles"]} == {"claude-*", "our-coder"}


def test_unknown_model_returns_an_anthropic_shaped_error(config):
    config.profiles = [p for p in config.profiles if p.match != "our-coder"]
    with TestClient(create_app(config)) as client:
        response = client.post("/v1/messages", json=body(model="mystery-model"))
    assert response.status_code == 400
    payload = response.json()
    assert payload["type"] == "error"
    assert payload["error"]["type"] == "invalid_request_error"
    assert "no profile matches" in payload["error"]["message"]


def test_openai_backend_response_is_translated(client):
    with respx.mock:
        route = respx.post(OPENAI_URL).mock(
            return_value=httpx.Response(200, json=chat_completion("hi"))
        )
        response = client.post("/v1/messages", json=body())

    assert response.status_code == 200
    payload = response.json()
    assert payload["type"] == "message"
    assert payload["role"] == "assistant"
    assert payload["content"] == [{"type": "text", "text": "hi"}]
    assert payload["model"] == "our-coder"
    assert route.called

    # The upstream must receive the profile's upstream_model, not the alias.
    sent = route.calls[0].request
    assert b'"internal-coder-v2"' in sent.content


def test_native_backend_is_passed_through(client):
    upstream = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-4-8",
        "content": [{"type": "text", "text": "native"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 5, "output_tokens": 2},
    }
    with respx.mock:
        route = respx.post(ANTHROPIC_URL).mock(return_value=httpx.Response(200, json=upstream))
        response = client.post("/v1/messages", json=body(model="claude-opus-4-8"))

    assert response.status_code == 200
    assert response.json() == upstream
    assert route.called


def test_disallowed_beta_flags_are_stripped(config):
    """Unknown beta flags are what produce 'invalid beta flag' on non-Anthropic upstreams."""
    config.backend("anthropic").allowed_beta_flags = ["context-1m-2025-08-07"]

    with TestClient(create_app(config)) as client, respx.mock:
        route = respx.post(ANTHROPIC_URL).mock(
            return_value=httpx.Response(200, json={"content": [], "type": "message"})
        )
        client.post(
            "/v1/messages",
            json=body(model="claude-opus-4-8"),
            headers={"anthropic-beta": "context-1m-2025-08-07,some-unsupported-flag"},
        )

    forwarded = route.calls[0].request.headers.get("anthropic-beta")
    assert forwarded == "context-1m-2025-08-07"


def test_anthropic_only_headers_are_not_sent_to_openai_backends(client):
    with respx.mock:
        route = respx.post(OPENAI_URL).mock(
            return_value=httpx.Response(200, json=chat_completion())
        )
        client.post(
            "/v1/messages",
            json=body(),
            headers={"anthropic-beta": "whatever", "anthropic-version": "2023-06-01"},
        )

    headers = route.calls[0].request.headers
    assert "anthropic-beta" not in headers
    assert "anthropic-version" not in headers


def test_streaming_is_translated_to_anthropic_sse(client):
    chunks = (
        'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n'
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        "data: [DONE]\n\n"
    )
    with respx.mock:
        respx.post(OPENAI_URL).mock(
            return_value=httpx.Response(
                200, text=chunks, headers={"content-type": "text/event-stream"}
            )
        )
        with client.stream("POST", "/v1/messages", json=body(stream=True)) as response:
            assert response.status_code == 200
            payload = "".join(response.iter_text())

    for expected in ("message_start", "content_block_start", "content_block_delta",
                     "content_block_stop", "message_delta", "message_stop"):
        assert f"event: {expected}" in payload, f"missing {expected}"
    assert "Hel" in payload and "lo" in payload


def test_upstream_error_is_forwarded_with_status(client):
    with respx.mock:
        respx.post(OPENAI_URL).mock(
            return_value=httpx.Response(429, json={"error": {"message": "slow down"}})
        )
        response = client.post("/v1/messages", json=body())

    assert response.status_code == 429
    payload = response.json()
    assert payload["type"] == "error"
    assert payload["error"]["type"] == "rate_limit_error"


def test_context_overflow_triggers_one_forced_retry(client):
    """A context-limit rejection must become a transparent recovery, not a dead session."""
    responses = [
        httpx.Response(
            400,
            json={"error": {"message": "maximum context length is 32000 tokens"}},
        ),
        httpx.Response(200, json=chat_completion("recovered")),
    ]
    with respx.mock:
        route = respx.post(OPENAI_URL).mock(side_effect=responses)
        response = client.post(
            "/v1/messages",
            json=body(build_conversation(exchanges=20, payload_size=4000)),
        )

    assert route.call_count == 2, "expected exactly one retry"
    assert response.status_code == 200
    assert response.json()["content"][0]["text"] == "recovered"


def test_non_overflow_error_is_not_retried(client):
    with respx.mock:
        route = respx.post(OPENAI_URL).mock(
            return_value=httpx.Response(400, json={"error": {"message": "bad tool schema"}})
        )
        response = client.post("/v1/messages", json=body())

    assert route.call_count == 1
    assert response.status_code == 400


# --------------------------------------------------------------------------- #


def test_count_tokens_reports_actual_when_under_trigger(client):
    payload = client.post(
        "/v1/messages/count_tokens",
        json={"model": "our-coder", "messages": [{"role": "user", "content": "hello"}]},
    ).json()
    assert 0 < payload["input_tokens"] < 100


def test_count_tokens_is_capped_to_the_post_reduction_estimate(client):
    """Otherwise Claude Code compacts on top of us — double compaction, double loss."""
    big = build_conversation(exchanges=30, payload_size=4000)
    payload = client.post(
        "/v1/messages/count_tokens",
        json={
            "model": "our-coder",
            "messages": [m.model_dump(exclude_none=True) for m in big],
        },
    ).json()

    # usable = 32000 - 4000 - 2000 = 26000; target = 0.55 * 26000 = 14300
    assert payload["input_tokens"] == 14300


def test_stats_endpoint_reports_sessions(client):
    with respx.mock:
        respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json=chat_completion()))
        client.post("/v1/messages", json=body(), headers={"x-claude-code-session-id": "abc"})

    payload = client.get("/stats").json()
    assert payload["sessions"] >= 1


def test_extra_params_are_merged_into_the_upstream_payload(config):
    config.backend("internal").openai_extra_params = {"reasoning_effort": "low"}
    with TestClient(create_app(config)) as client, respx.mock:
        route = respx.post(OPENAI_URL).mock(
            return_value=httpx.Response(200, json=chat_completion())
        )
        client.post("/v1/messages", json=body())

    sent = json.loads(route.calls[0].request.content)
    assert sent["reasoning_effort"] == "low"


def test_extra_params_cannot_clobber_proxy_owned_fields(config):
    """A stray `messages` override would silently send the wrong conversation."""
    config.backend("internal").openai_extra_params = {
        "messages": [{"role": "user", "content": "hijacked"}],
        "model": "wrong-model",
        "temperature": 0.1,
    }
    with TestClient(create_app(config)) as client, respx.mock:
        route = respx.post(OPENAI_URL).mock(
            return_value=httpx.Response(200, json=chat_completion())
        )
        client.post("/v1/messages", json=body())

    sent = json.loads(route.calls[0].request.content)
    assert sent["model"] == "internal-coder-v2"
    assert "hijacked" not in json.dumps(sent["messages"])
    assert sent["temperature"] == 0.1
