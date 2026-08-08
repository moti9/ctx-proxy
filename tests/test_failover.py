"""Health-aware failover across a profile's fallback chain.

Fallbacks are references to full profiles, so each carries its own backend,
model, window, token caps and tokenizer — exercised here.
"""

from __future__ import annotations

import json

import httpx
import respx
from conftest import build_conversation, make_request
from fastapi.testclient import TestClient

from ctxproxy.app import create_app
from ctxproxy.config import (
    BackendConfig,
    Config,
    ModelProfile,
    ReductionPolicy,
    ServerConfig,
    TokenizerConfig,
)
from ctxproxy.health import HealthTracker

URL = "http://internal.test/v1/chat/completions"


def _profile(match: str, model: str, **kw) -> ModelProfile:
    return ModelProfile(
        match=match,
        backend="internal",
        upstream_model=model,
        native_anthropic=False,
        context_window=kw.get("context_window", 32000),
        output_reserve=kw.get("output_reserve", 4000),
        safety_buffer=2000,
        max_output_tokens=kw.get("max_output_tokens"),
        supports_count_tokens=False,
        supports_prompt_caching=False,
        tokenizer=TokenizerConfig(kind="heuristic"),
        fallbacks=kw.get("fallbacks", []),
    )


def failover_config(tmp_path, cooldown_s: float = 30.0, **fallback_kw) -> Config:
    return Config(
        server=ServerConfig(state_dir=tmp_path / "sessions", upstream_cooldown_s=cooldown_s),
        policy=ReductionPolicy(),
        backends=[BackendConfig(name="internal", kind="openai", base_url="http://internal.test")],
        profiles=[
            _profile("*", "primary-model", fallbacks=["fallback"]),
            _profile("fallback", "fallback-model", **fallback_kw),
        ],
    )


def chat_completion(text: str) -> dict:
    return {
        "id": "c1",
        "choices": [{"index": 0, "message": {"content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 4},
    }


def body(**kw) -> dict:
    return make_request(build_conversation(1, 50), model="anything", **kw).model_dump(
        exclude_none=True
    )


def model_of(call) -> str:
    return json.loads(call.request.content)["model"]


def route_primary_down(request: httpx.Request) -> httpx.Response:
    if json.loads(request.content)["model"] == "primary-model":
        return httpx.Response(
            429, json={"error": {"message": "No deployments available for selected model"}}
        )
    return httpx.Response(200, json=chat_completion("from-fallback"))


# --------------------------------------------------------------------------- #
# HealthTracker unit
# --------------------------------------------------------------------------- #


def test_health_marks_down_and_recovers_after_cooldown():
    h = HealthTracker(cooldown_s=30)
    clock = [1000.0]
    h._now = lambda: clock[0]
    t = ("internal", "primary-model")

    assert h.available(t)
    h.mark_down(t)
    assert not h.available(t)
    clock[0] += 29
    assert not h.available(t), "still cooling down"
    clock[0] += 2
    assert h.available(t), "cooldown expired -> re-probe"


def test_health_order_moves_cooling_targets_to_back():
    h = HealthTracker(cooldown_s=30)
    a, b, c = ("x", "a"), ("x", "b"), ("x", "c")
    h.mark_down(a)
    assert h.order([a, b, c]) == [b, c, a], "healthy first (priority kept), cooling last"


def test_health_order_all_down_keeps_priority():
    h = HealthTracker(cooldown_s=30)
    a, b = ("x", "a"), ("x", "b")
    h.mark_down(a)
    h.mark_down(b)
    assert h.order([a, b]) == [a, b], "all down -> re-probe in priority order"


def test_health_order_with_key():
    """Ordering a list of objects by a derived target key."""
    h = HealthTracker(cooldown_s=30)
    items = [{"t": ("x", "a")}, {"t": ("x", "b")}]
    h.mark_down(("x", "a"))
    assert h.order(items, key=lambda it: it["t"]) == [items[1], items[0]]


# --------------------------------------------------------------------------- #
# Route-level failover
# --------------------------------------------------------------------------- #


def test_failover_to_next_model_on_429(tmp_path):
    with TestClient(create_app(failover_config(tmp_path))) as client, respx.mock:
        route = respx.post(URL).mock(side_effect=route_primary_down)
        r = client.post("/v1/messages", json=body(), headers={"x-claude-code-session-id": "a"})

    assert r.status_code == 200
    assert r.json()["content"][0]["text"] == "from-fallback"
    assert [model_of(c) for c in route.calls] == ["primary-model", "fallback-model"]


def test_streaming_fails_over_to_next_model(tmp_path):
    chunks = (
        'data: {"choices":[{"delta":{"content":"hi from fb"}}]}\n\n'
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        "data: [DONE]\n\n"
    )

    def route(request):
        if json.loads(request.content)["model"] == "primary-model":
            return httpx.Response(429, json={"error": {"message": "No deployments available"}})
        return httpx.Response(200, text=chunks, headers={"content-type": "text/event-stream"})

    with TestClient(create_app(failover_config(tmp_path))) as client, respx.mock:
        respx.post(URL).mock(side_effect=route)
        with client.stream(
            "POST", "/v1/messages", json=body(stream=True),
            headers={"x-claude-code-session-id": "s"},
        ) as resp:
            assert resp.status_code == 200
            text = "".join(resp.iter_text())
    assert "hi from fb" in text


def test_fallback_uses_its_own_token_cap(tmp_path):
    """A fallback profile re-budgets with ITS OWN max_output_tokens, not the primary's."""
    cfg = failover_config(tmp_path, max_output_tokens=8000)  # fallback caps output at 8000
    with TestClient(create_app(cfg)) as client, respx.mock:
        route = respx.post(URL).mock(side_effect=route_primary_down)
        client.post(
            "/v1/messages", json=body(max_tokens=20000),
            headers={"x-claude-code-session-id": "a"},
        )

    sent = {model_of(c): json.loads(c.request.content)["max_tokens"] for c in route.calls}
    # Primary has no cap -> keeps 20000; fallback clamps to its own 8000.
    assert sent["primary-model"] == 20000
    assert sent["fallback-model"] == 8000, "fallback must apply its own token cap"


def test_cooldown_routes_straight_to_fallback_on_next_request(tmp_path):
    """After the primary fails, the next request must skip it (fast), not retry it."""
    with TestClient(create_app(failover_config(tmp_path))) as client, respx.mock:
        route = respx.post(URL).mock(side_effect=route_primary_down)
        client.post("/v1/messages", json=body(), headers={"x-claude-code-session-id": "a"})
        route.reset()
        r2 = client.post("/v1/messages", json=body(), headers={"x-claude-code-session-id": "a"})

    assert r2.status_code == 200
    assert model_of(route.calls[0]) == "fallback-model"
    assert "primary-model" not in [model_of(c) for c in route.calls]


def test_recovered_primary_is_used_again_after_cooldown(tmp_path):
    """Once the cooldown expires the top model is preferred again."""
    app = create_app(failover_config(tmp_path, cooldown_s=30))
    with TestClient(app) as client, respx.mock:
        route = respx.post(URL).mock(side_effect=route_primary_down)
        client.post("/v1/messages", json=body(), headers={"x-claude-code-session-id": "a"})

        app.state.ctx.health._now = lambda: 10_000_000.0  # fast-forward past cooldown
        route.reset()
        respx.post(URL).mock(return_value=httpx.Response(200, json=chat_completion("primary-back")))
        r = client.post("/v1/messages", json=body(), headers={"x-claude-code-session-id": "a"})

    assert r.json()["content"][0]["text"] == "primary-back"
    assert model_of(route.calls[0]) == "primary-model", "recovered primary preferred again"


def test_non_retryable_error_does_not_fail_over(tmp_path):
    """A 400 would fail identically on the fallback, so it must not fail over."""
    with TestClient(create_app(failover_config(tmp_path))) as client, respx.mock:
        route = respx.post(URL).mock(
            return_value=httpx.Response(400, json={"error": {"message": "bad tool schema"}})
        )
        r = client.post("/v1/messages", json=body(), headers={"x-claude-code-session-id": "a"})

    assert r.status_code == 400
    assert [model_of(c) for c in route.calls] == ["primary-model"], "must not try the fallback"


def test_all_targets_down_returns_last_error(tmp_path):
    with TestClient(create_app(failover_config(tmp_path))) as client, respx.mock:
        route = respx.post(URL).mock(
            return_value=httpx.Response(
                429, json={"error": {"message": "No deployments available"}}
            )
        )
        r = client.post("/v1/messages", json=body(), headers={"x-claude-code-session-id": "a"})

    assert r.status_code == 429
    assert [model_of(c) for c in route.calls] == ["primary-model", "fallback-model"]
