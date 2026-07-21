"""Anthropic Messages request -> OpenAI Chat Completions request."""

from __future__ import annotations

import json
from typing import Any

from ..types_anthropic import Block, MessagesRequest, block_text


def anthropic_to_openai(
    request: MessagesRequest,
    *,
    model: str,
    max_tokens_field: str = "max_tokens",
    include_usage: bool = True,
) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []

    system_text = "\n\n".join(
        t for t in (block_text(b) for b in request.system_blocks()) if t
    )
    if system_text:
        messages.append({"role": "system", "content": system_text})

    for msg in request.messages:
        messages.extend(_convert_message(msg.role, msg.blocks()))

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        max_tokens_field: request.max_tokens,
    }

    if request.temperature is not None:
        payload["temperature"] = request.temperature
    if request.top_p is not None:
        payload["top_p"] = request.top_p
    if request.stop_sequences:
        payload["stop"] = request.stop_sequences
    if request.tools:
        payload["tools"] = [_convert_tool(t) for t in request.tools]
    if request.tool_choice and (
        choice := _convert_tool_choice(request.tool_choice)
    ) is not None:
        payload["tool_choice"] = choice
    if request.stream:
        payload["stream"] = True
        if include_usage:
            payload["stream_options"] = {"include_usage": True}

    return payload


def _convert_message(role: str, blocks: list[Block]) -> list[dict[str, Any]]:
    """One Anthropic message can become several OpenAI messages.

    Anthropic packs tool results into the following *user* message; OpenAI wants
    each one as its own ``role: "tool"`` message. Those must be emitted before
    any remaining user content, because OpenAI requires every tool message to
    directly follow the assistant turn that requested it.
    """
    out: list[dict[str, Any]] = []

    if role == "user":
        tool_results = [b for b in blocks if b.get("type") == "tool_result"]
        for block in tool_results:
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id", ""),
                    "content": block_text(block) or "(no output)",
                }
            )

        rest = [b for b in blocks if b.get("type") != "tool_result"]
        if content := _user_content(rest):
            out.append({"role": "user", "content": content})
        return out

    if role == "system":
        text = "\n".join(t for t in (block_text(b) for b in blocks) if t)
        return [{"role": "system", "content": text}] if text else []

    # assistant
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    for block in blocks:
        btype = block.get("type")
        if btype == "text":
            if t := block.get("text"):
                text_parts.append(t)
        elif btype == "tool_use":
            tool_calls.append(
                {
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input") or {}),
                    },
                }
            )
        # thinking / redacted_thinking are intentionally dropped: they carry
        # Anthropic-specific signatures and have no OpenAI equivalent.

    message: dict[str, Any] = {"role": "assistant"}
    message["content"] = "\n".join(text_parts) if text_parts else None
    if tool_calls:
        message["tool_calls"] = tool_calls
    if message["content"] is None and not tool_calls:
        return []
    return [message]


def _user_content(blocks: list[Block]) -> Any:
    """Return a plain string when possible, else OpenAI's multipart array."""
    if not blocks:
        return None

    if all(b.get("type") == "text" for b in blocks):
        text = "\n".join(b.get("text", "") for b in blocks)
        return text or None

    parts: list[dict[str, Any]] = []
    for block in blocks:
        btype = block.get("type")
        if btype == "text":
            parts.append({"type": "text", "text": block.get("text", "")})
        elif btype == "image":
            if url := _image_url(block):
                parts.append({"type": "image_url", "image_url": {"url": url}})
        else:
            if text := block_text(block):
                parts.append({"type": "text", "text": text})
    return parts or None


def _image_url(block: Block) -> str | None:
    source = block.get("source") or {}
    if source.get("type") == "base64":
        media = source.get("media_type", "image/png")
        return f"data:{media};base64,{source.get('data', '')}"
    if source.get("type") == "url":
        return source.get("url")
    return None


def _convert_tool(tool: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.get("name", ""),
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema") or {"type": "object", "properties": {}},
        },
    }


def _convert_tool_choice(choice: dict[str, Any]) -> Any:
    match choice.get("type"):
        case "auto":
            return "auto"
        case "any":
            return "required"
        case "none":
            return "none"
        case "tool":
            return {"type": "function", "function": {"name": choice.get("name", "")}}
    return None
