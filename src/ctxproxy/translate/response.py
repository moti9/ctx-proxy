"""OpenAI Chat Completions response -> Anthropic Messages response."""

from __future__ import annotations

import json
from typing import Any

FINISH_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "refusal",
}


def openai_to_anthropic(
    payload: dict[str, Any], *, model: str, reasoning_output: str = "text"
) -> dict[str, Any]:
    choices = payload.get("choices") or [{}]
    choice = choices[0]
    message = choice.get("message") or {}

    content: list[dict[str, Any]] = []

    # Reasoning models surface their chain of thought here. Anthropic has no
    # client-visible equivalent on a translated response, so it is either
    # rendered as text or dropped — see BackendConfig.openai_reasoning_output.
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    if reasoning and reasoning_output == "text":
        content.append({"type": "text", "text": str(reasoning)})

    if text := message.get("content"):
        content.append({"type": "text", "text": text})

    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        content.append(
            {
                "type": "tool_use",
                "id": call.get("id") or f"toolu_{len(content)}",
                "name": function.get("name", ""),
                "input": _parse_arguments(function.get("arguments")),
            }
        )

    if not content and reasoning:
        # "drop" mode, but reasoning is all there is. Some reasoning models
        # leave `content` empty and put the answer in `reasoning_content`;
        # honouring the drop here would return an empty turn, which reads as
        # the model having said nothing. Noise beats silence.
        content.append({"type": "text", "text": str(reasoning)})

    if not content:
        content.append({"type": "text", "text": ""})

    usage = payload.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}

    return {
        "id": payload.get("id") or "msg_ctxproxy",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": FINISH_REASON_MAP.get(choice.get("finish_reason") or "stop", "end_turn"),
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "cache_read_input_tokens": details.get("cached_tokens", 0),
            "cache_creation_input_tokens": 0,
        },
    }


def _parse_arguments(raw: Any) -> dict[str, Any]:
    """Tool arguments arrive as a JSON string and are not always valid.

    A malformed argument blob must not take the whole turn down — returning an
    empty object lets the assistant see a failed call and retry, which is
    recoverable; raising here is not.
    """
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except (json.JSONDecodeError, TypeError):
        return {}
