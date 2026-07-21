"""Anthropic <-> OpenAI translation."""

from __future__ import annotations

import json

from conftest import assistant_tool_call, make_request, tool_result, user

from ctxproxy.translate.request import anthropic_to_openai
from ctxproxy.translate.response import openai_to_anthropic
from ctxproxy.translate.sse import OpenAIStreamTranslator, parse_sse_line
from ctxproxy.types_anthropic import Message

# --------------------------------------------------------------------------- #
# Request direction
# --------------------------------------------------------------------------- #


def test_system_prompt_becomes_a_system_message():
    request = make_request([user("hi")], system="You are terse.")
    payload = anthropic_to_openai(request, model="m")
    assert payload["messages"][0] == {"role": "system", "content": "You are terse."}


def test_tool_use_becomes_tool_calls():
    request = make_request([user("go"), assistant_tool_call("t1", "read_file", path="a.py")])
    payload = anthropic_to_openai(request, model="m")
    call = payload["messages"][-1]["tool_calls"][0]
    assert call["id"] == "t1"
    assert call["function"]["name"] == "read_file"
    assert json.loads(call["function"]["arguments"]) == {"path": "a.py"}


def test_tool_result_becomes_a_tool_role_message():
    request = make_request(
        [user("go"), assistant_tool_call("t1"), tool_result("t1", "file contents")]
    )
    payload = anthropic_to_openai(request, model="m")
    last = payload["messages"][-1]
    assert last["role"] == "tool"
    assert last["tool_call_id"] == "t1"
    assert last["content"] == "file contents"


def test_tool_results_precede_remaining_user_content():
    """OpenAI requires tool messages to directly follow the assistant turn."""
    mixed = Message(
        role="user",
        content=[
            {"type": "tool_result", "tool_use_id": "t1", "content": "out"},
            {"type": "text", "text": "and now do the next thing"},
        ],
    )
    request = make_request([user("go"), assistant_tool_call("t1"), mixed])
    roles = [m["role"] for m in anthropic_to_openai(request, model="m")["messages"]]
    assert roles == ["user", "assistant", "tool", "user"]


def test_thinking_blocks_are_dropped():
    thinking = Message(
        role="assistant",
        content=[
            {"type": "thinking", "thinking": "hmm", "signature": "sig"},
            {"type": "text", "text": "answer"},
        ],
    )
    request = make_request([user("q"), thinking])
    payload = anthropic_to_openai(request, model="m")
    assert payload["messages"][-1]["content"] == "answer"
    assert "thinking" not in json.dumps(payload)


def test_tool_choice_mapping():
    for anthropic_choice, expected in [
        ({"type": "auto"}, "auto"),
        ({"type": "any"}, "required"),
        ({"type": "none"}, "none"),
    ]:
        request = make_request([user("x")], tool_choice=anthropic_choice)
        assert anthropic_to_openai(request, model="m")["tool_choice"] == expected

    request = make_request([user("x")], tool_choice={"type": "tool", "name": "f"})
    assert anthropic_to_openai(request, model="m")["tool_choice"] == {
        "type": "function",
        "function": {"name": "f"},
    }


def test_max_tokens_field_is_configurable():
    request = make_request([user("x")], max_tokens=99)
    assert anthropic_to_openai(request, model="m")["max_tokens"] == 99
    payload = anthropic_to_openai(request, model="m", max_tokens_field="max_completion_tokens")
    assert payload["max_completion_tokens"] == 99


def test_image_block_becomes_a_data_uri():
    image = Message(
        role="user",
        content=[
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"},
            }
        ],
    )
    payload = anthropic_to_openai(make_request([image]), model="m")
    part = payload["messages"][0]["content"][0]
    assert part["image_url"]["url"] == "data:image/png;base64,AAAA"


# --------------------------------------------------------------------------- #
# Response direction
# --------------------------------------------------------------------------- #


def test_response_maps_content_and_usage():
    payload = {
        "id": "chatcmpl-1",
        "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 3},
    }
    result = openai_to_anthropic(payload, model="our-coder")
    assert result["type"] == "message"
    assert result["content"] == [{"type": "text", "text": "hello"}]
    assert result["stop_reason"] == "end_turn"
    assert result["usage"]["input_tokens"] == 10
    assert result["usage"]["output_tokens"] == 3


def test_response_maps_tool_calls_and_stop_reason():
    payload = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {"name": "f", "arguments": '{"a":1}'},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
    }
    result = openai_to_anthropic(payload, model="m")
    assert result["stop_reason"] == "tool_use"
    block = result["content"][0]
    assert block["type"] == "tool_use"
    assert block["input"] == {"a": 1}


def test_malformed_tool_arguments_degrade_instead_of_raising():
    payload = {
        "choices": [
            {
                "message": {
                    "tool_calls": [{"id": "c", "function": {"name": "f", "arguments": "{oops"}}]
                },
                "finish_reason": "tool_calls",
            }
        ]
    }
    result = openai_to_anthropic(payload, model="m")
    assert result["content"][0]["input"] == {}


# --------------------------------------------------------------------------- #
# Streaming
# --------------------------------------------------------------------------- #


def decode(events: list[bytes]) -> list[dict]:
    out = []
    for raw in events:
        for line in raw.decode().splitlines():
            if parsed := parse_sse_line(line):
                out.append(parsed)
    return out


def run_stream(chunks: list[dict]) -> list[dict]:
    translator = OpenAIStreamTranslator(model="our-coder")
    events = list(translator.start())
    for chunk in chunks:
        events.extend(translator.handle_chunk(chunk))
    events.extend(translator.finish())
    return decode(events)


def test_text_stream_is_well_formed():
    events = run_stream(
        [
            {"choices": [{"delta": {"content": "Hel"}}]},
            {"choices": [{"delta": {"content": "lo"}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ]
    )
    types = [e["type"] for e in events]
    assert types == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    text = "".join(
        e["delta"]["text"] for e in events if e["type"] == "content_block_delta"
    )
    assert text == "Hello"


def test_every_opened_block_is_closed_exactly_once():
    events = run_stream(
        [
            {"choices": [{"delta": {"content": "thinking..."}}]},
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "id": "c1", "function": {"name": "f"}}
                            ]
                        }
                    }
                ]
            },
            {
                "choices": [
                    {"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"a":'}}]}}
                ]
            },
            {
                "choices": [
                    {"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "1}"}}]}}
                ]
            },
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        ]
    )
    opened = [e["index"] for e in events if e["type"] == "content_block_start"]
    closed = [e["index"] for e in events if e["type"] == "content_block_stop"]
    assert sorted(opened) == sorted(closed) == [0, 1]


def test_tool_block_carries_id_and_name_and_streams_arguments():
    events = run_stream(
        [
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "id": "c1", "function": {"name": "read"}}
                            ]
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": '{"p":"a.py"}'}}
                            ]
                        }
                    }
                ]
            },
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        ]
    )
    start = next(e for e in events if e["type"] == "content_block_start")
    assert start["content_block"] == {
        "type": "tool_use",
        "id": "c1",
        "name": "read",
        "input": {},
    }
    partial = "".join(
        e["delta"]["partial_json"]
        for e in events
        if e["type"] == "content_block_delta"
    )
    assert json.loads(partial) == {"p": "a.py"}
    assert events[-2]["delta"]["stop_reason"] == "tool_use"


def test_parallel_tool_calls_get_distinct_indices():
    events = run_stream(
        [
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "id": "c1", "function": {"name": "a"}}
                            ]
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 1, "id": "c2", "function": {"name": "b"}}
                            ]
                        }
                    }
                ]
            },
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        ]
    )
    starts = [e for e in events if e["type"] == "content_block_start"]
    assert [s["index"] for s in starts] == [0, 1]
    assert [s["content_block"]["name"] for s in starts] == ["a", "b"]


def test_usage_trailer_without_choices_is_tolerated():
    events = run_stream(
        [
            {"choices": [{"delta": {"content": "hi"}}]},
            {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 2}},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ]
    )
    message_delta = next(e for e in events if e["type"] == "message_delta")
    assert message_delta["usage"]["output_tokens"] == 2


def test_done_sentinel_and_comments_are_ignored():
    assert parse_sse_line("data: [DONE]") is None
    assert parse_sse_line(": keepalive") is None
    assert parse_sse_line("event: message_stop") is None
    assert parse_sse_line("data: {invalid") is None
    assert parse_sse_line('data: {"a": 1}') == {"a": 1}


# --------------------------------------------------------------------------- #
# Reasoning models (kimi-k2 and friends set supports_reasoning: true)
# --------------------------------------------------------------------------- #


def test_reasoning_is_rendered_as_text_by_default():
    payload = {
        "choices": [
            {
                "message": {"reasoning_content": "let me think", "content": "answer"},
                "finish_reason": "stop",
            }
        ]
    }
    result = openai_to_anthropic(payload, model="m")
    assert [b["text"] for b in result["content"]] == ["let me think", "answer"]


def test_reasoning_can_be_dropped():
    payload = {
        "choices": [
            {
                "message": {"reasoning_content": "let me think", "content": "answer"},
                "finish_reason": "stop",
            }
        ]
    }
    result = openai_to_anthropic(payload, model="m", reasoning_output="drop")
    assert [b["text"] for b in result["content"]] == ["answer"]


def test_dropping_reasoning_never_yields_empty_content():
    """Some reasoning models put the whole answer in reasoning_content."""
    payload = {
        "choices": [
            {"message": {"reasoning_content": "only this"}, "finish_reason": "stop"}
        ]
    }
    result = openai_to_anthropic(payload, model="m", reasoning_output="drop")
    assert len(result["content"]) == 1


def test_streaming_reasoning_can_be_dropped():
    translator = OpenAIStreamTranslator(model="m", reasoning_output="drop")
    events = list(translator.start())
    events.extend(translator.handle_chunk({"choices": [{"delta": {"reasoning_content": "hmm"}}]}))
    events.extend(translator.handle_chunk({"choices": [{"delta": {"content": "answer"}}]}))
    events.extend(translator.handle_chunk({"choices": [{"delta": {}, "finish_reason": "stop"}]}))

    text = "".join(
        e["delta"]["text"] for e in decode(events) if e["type"] == "content_block_delta"
    )
    assert text == "answer"


def test_drop_falls_back_to_reasoning_rather_than_an_empty_turn():
    """Some reasoning models leave `content` empty. Silence is worse than noise."""
    payload = {
        "choices": [
            {"message": {"reasoning_content": "the whole answer"}, "finish_reason": "stop"}
        ]
    }
    result = openai_to_anthropic(payload, model="m", reasoning_output="drop")
    assert result["content"] == [{"type": "text", "text": "the whole answer"}]


def test_streaming_drop_falls_back_when_no_content_arrives():
    translator = OpenAIStreamTranslator(model="m", reasoning_output="drop")
    events = list(translator.start())
    events.extend(translator.handle_chunk({"choices": [{"delta": {"reasoning_content": "all "}}]}))
    events.extend(translator.handle_chunk({"choices": [{"delta": {"reasoning_content": "of it"}}]}))
    events.extend(translator.handle_chunk({"choices": [{"delta": {}, "finish_reason": "stop"}]}))
    events.extend(translator.finish())

    decoded = decode(events)
    text = "".join(e["delta"]["text"] for e in decoded if e["type"] == "content_block_delta")
    assert text == "all of it"
    opened = [e["index"] for e in decoded if e["type"] == "content_block_start"]
    closed = [e["index"] for e in decoded if e["type"] == "content_block_stop"]
    assert opened == closed == [0]


def test_streaming_drop_stays_silent_when_real_content_exists():
    translator = OpenAIStreamTranslator(model="m", reasoning_output="drop")
    events = list(translator.start())
    events.extend(translator.handle_chunk({"choices": [{"delta": {"reasoning_content": "noise"}}]}))
    events.extend(translator.handle_chunk({"choices": [{"delta": {"content": "answer"}}]}))
    events.extend(translator.handle_chunk({"choices": [{"delta": {}, "finish_reason": "stop"}]}))
    events.extend(translator.finish())

    text = "".join(
        e["delta"]["text"] for e in decode(events) if e["type"] == "content_block_delta"
    )
    assert text == "answer"


def test_empty_turn_detection():
    from ctxproxy.backends.openai_compat import _is_empty_turn

    assert _is_empty_turn({"content": []})
    assert _is_empty_turn({"content": [{"type": "text", "text": ""}]})
    assert _is_empty_turn({"content": [{"type": "text", "text": "   "}]})
    assert not _is_empty_turn({"content": [{"type": "text", "text": "hi"}]})
    assert not _is_empty_turn(
        {"content": [{"type": "tool_use", "id": "t", "name": "f", "input": {}}]}
    )
