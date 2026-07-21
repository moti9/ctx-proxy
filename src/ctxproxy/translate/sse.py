"""OpenAI streaming chunks -> Anthropic SSE events.

Claude Code drives its entire UI off the Anthropic event sequence, so the shape
has to be exact: an out-of-order ``content_block_start`` or a missing
``content_block_stop`` shows up as a frozen or garbled response rather than an
error, which is far harder to debug.

Anthropic keeps exactly one content block open at a time and indexes blocks
sequentially. OpenAI has no such notion — text and tool-call deltas simply
interleave in whatever order the model emits them. This translator owns that
impedance mismatch: it opens, closes, and indexes blocks so the emitted stream
is always well-formed regardless of upstream ordering.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .response import FINISH_REASON_MAP


def sse_event(event_type: str, data: dict[str, Any]) -> bytes:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n".encode()


@dataclass
class _ToolBlock:
    anthropic_index: int
    started: bool = False
    tool_id: str = ""
    name: str = ""


@dataclass
class OpenAIStreamTranslator:
    model: str
    message_id: str = "msg_ctxproxy"
    input_tokens: int = 0
    reasoning_output: str = "text"

    _started: bool = False
    _finished: bool = False
    _next_index: int = 0
    _open_index: int | None = None
    _text_index: int | None = None
    _tools: dict[int, _ToolBlock] = field(default_factory=dict)
    _stop_reason: str = "end_turn"
    _output_tokens: int = 0
    _dropped_reasoning: list[str] = field(default_factory=list)
    _emitted_content: bool = False

    # -- lifecycle ---------------------------------------------------------- #

    def start(self) -> list[bytes]:
        if self._started:
            return []
        self._started = True
        return [
            sse_event(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": self.message_id,
                        "type": "message",
                        "role": "assistant",
                        "model": self.model,
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {
                            "input_tokens": self.input_tokens,
                            "output_tokens": 0,
                        },
                    },
                },
            )
        ]

    def handle_chunk(self, chunk: dict[str, Any]) -> list[bytes]:
        out: list[bytes] = []

        if error := chunk.get("error"):
            return self.error(str(error.get("message") or error))

        if usage := chunk.get("usage"):
            self._output_tokens = usage.get("completion_tokens", self._output_tokens)
            if not self.input_tokens:
                self.input_tokens = usage.get("prompt_tokens", 0)

        choices = chunk.get("choices") or []
        if not choices:
            # Usage-only trailer emitted when stream_options.include_usage is on.
            return out

        choice = choices[0]
        delta = choice.get("delta") or {}

        if reason := choice.get("finish_reason"):
            self._stop_reason = FINISH_REASON_MAP.get(reason, "end_turn")

        # Reasoning-model preamble. Rendered as text, or buffered and dropped.
        for key in ("reasoning_content", "reasoning"):
            if value := delta.get(key):
                if self.reasoning_output == "text":
                    out.extend(self._text_delta(str(value)))
                else:
                    # Held rather than discarded: if the turn produces nothing
                    # else, finish() flushes this instead of emitting an empty
                    # message. See the same guard in response.openai_to_anthropic.
                    self._dropped_reasoning.append(str(value))

        if (content := delta.get("content")) is not None and content != "":
            self._emitted_content = True
            out.extend(self._text_delta(content))

        for call in delta.get("tool_calls") or []:
            self._emitted_content = True
            out.extend(self._tool_delta(call))

        return out

    def finish(self) -> list[bytes]:
        if self._finished:
            return []
        self._finished = True

        out: list[bytes] = []
        if not self._started:
            out.extend(self.start())

        if not self._emitted_content and self._dropped_reasoning:
            out.extend(self._text_delta("".join(self._dropped_reasoning)))

        out.extend(self._close_open_block())
        out.append(
            sse_event(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": self._stop_reason, "stop_sequence": None},
                    "usage": {"output_tokens": self._output_tokens},
                },
            )
        )
        out.append(sse_event("message_stop", {"type": "message_stop"}))
        return out

    def error(self, message: str) -> list[bytes]:
        self._finished = True
        return [
            sse_event(
                "error",
                {"type": "error", "error": {"type": "api_error", "message": message}},
            )
        ]

    # -- block management --------------------------------------------------- #

    def _text_delta(self, text: str) -> list[bytes]:
        out: list[bytes] = []
        if self._text_index is None or self._open_index != self._text_index:
            out.extend(self._close_open_block())
            self._text_index = self._next_index
            self._next_index += 1
            self._open_index = self._text_index
            out.append(
                sse_event(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": self._text_index,
                        "content_block": {"type": "text", "text": ""},
                    },
                )
            )
        out.append(
            sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": self._text_index,
                    "delta": {"type": "text_delta", "text": text},
                },
            )
        )
        return out

    def _tool_delta(self, call: dict[str, Any]) -> list[bytes]:
        out: list[bytes] = []
        # `index` identifies which parallel tool call this fragment belongs to.
        # Absent (some gateways omit it on the first fragment) it defaults to 0.
        oai_index = call.get("index", 0)
        function = call.get("function") or {}

        block = self._tools.get(oai_index)
        if block is None:
            block = _ToolBlock(anthropic_index=-1)
            self._tools[oai_index] = block

        if call.get("id"):
            block.tool_id = call["id"]
        if function.get("name"):
            block.name = function["name"]

        # Defer opening until we have a name — Anthropic requires it on the
        # content_block_start, and it can arrive a fragment later than the id.
        if not block.started:
            if not block.name:
                return out
            out.extend(self._close_open_block())
            block.anthropic_index = self._next_index
            self._next_index += 1
            block.started = True
            self._open_index = block.anthropic_index
            out.append(
                sse_event(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": block.anthropic_index,
                        "content_block": {
                            "type": "tool_use",
                            "id": block.tool_id or f"toolu_{block.anthropic_index}",
                            "name": block.name,
                            "input": {},
                        },
                    },
                )
            )
        elif self._open_index != block.anthropic_index:
            out.extend(self._close_open_block())
            self._open_index = block.anthropic_index

        if (arguments := function.get("arguments")) is not None and arguments != "":
            out.append(
                sse_event(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": block.anthropic_index,
                        "delta": {"type": "input_json_delta", "partial_json": arguments},
                    },
                )
            )
        return out

    def _close_open_block(self) -> list[bytes]:
        if self._open_index is None:
            return []
        index = self._open_index
        self._open_index = None
        return [
            sse_event("content_block_stop", {"type": "content_block_stop", "index": index})
        ]


def parse_sse_line(line: str) -> dict[str, Any] | None:
    """Extract a JSON payload from one SSE ``data:`` line.

    Returns ``None`` for blank lines, comments, ``event:`` lines, the
    ``[DONE]`` sentinel, and anything that is not valid JSON.
    """
    line = line.strip()
    if not line or line.startswith(":") or not line.startswith("data:"):
        return None
    payload = line[len("data:") :].strip()
    if not payload or payload == "[DONE]":
        return None
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None
