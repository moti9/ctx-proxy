"""Minimal, permissive models of the Anthropic Messages API wire format.

Design note: we deliberately do NOT model every block type strictly. Claude Code
evolves faster than this proxy will, and an over-strict schema turns a new block
type into a 422 instead of a passthrough. Everything unknown is preserved
verbatim via ``model_config = ConfigDict(extra="allow")`` and dict round-tripping.

The helpers at the bottom are the ones the context pipeline actually depends on.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# --------------------------------------------------------------------------- #
# Wire models
# --------------------------------------------------------------------------- #

Role = Literal["user", "assistant", "system"]

# Content blocks are kept as plain dicts. They are heterogeneous, frequently
# extended upstream, and every transformation we do is structural rather than
# semantic. A typed union here buys nothing and costs forward compatibility.
Block = dict[str, Any]


class Message(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Role
    content: str | list[Block]

    def blocks(self) -> list[Block]:
        """Content normalised to a block list. Does not mutate the message."""
        if isinstance(self.content, str):
            return [{"type": "text", "text": self.content}]
        return self.content

    def with_blocks(self, blocks: list[Block]) -> Message:
        """Copy carrying replacement blocks, preserving any extra fields."""
        return self.model_copy(update={"content": blocks})

    def block_types(self) -> set[str]:
        return {b.get("type", "") for b in self.blocks()}


class MessagesRequest(BaseModel):
    """A POST /v1/messages body."""

    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[Message]
    max_tokens: int = 4096
    system: str | list[Block] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: dict[str, Any] | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop_sequences: list[str] | None = None
    stream: bool = False
    metadata: dict[str, Any] | None = None
    thinking: dict[str, Any] | None = None
    output_config: dict[str, Any] | None = None
    context_management: dict[str, Any] | None = None

    def system_blocks(self) -> list[Block]:
        if self.system is None:
            return []
        if isinstance(self.system, str):
            return [{"type": "text", "text": self.system}]
        return self.system


class CountTokensRequest(BaseModel):
    """A POST /v1/messages/count_tokens body."""

    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[Message]
    system: str | list[Block] | None = None
    tools: list[dict[str, Any]] | None = None

    def to_messages_request(self, max_tokens: int = 0) -> MessagesRequest:
        return MessagesRequest(
            model=self.model,
            messages=self.messages,
            system=self.system,
            tools=self.tools,
            max_tokens=max_tokens,
        )


class Usage(BaseModel):
    model_config = ConfigDict(extra="allow")

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


# --------------------------------------------------------------------------- #
# Structural helpers used by the context pipeline
# --------------------------------------------------------------------------- #

# Blocks that carry bulk payload and are cheap to evict.
EVICTABLE_BLOCK_TYPES = frozenset({"tool_result"})

# Blocks the model emits that we must never reorder or partially remove for a
# native Anthropic backend (signature validation is order-sensitive).
SIGNED_BLOCK_TYPES = frozenset({"thinking", "redacted_thinking"})


def tool_use_ids(msg: Message) -> set[str]:
    """IDs of tool_use blocks this (assistant) message issues."""
    return {b["id"] for b in msg.blocks() if b.get("type") == "tool_use" and b.get("id")}


def tool_result_ids(msg: Message) -> set[str]:
    """IDs of tool_use blocks this (user) message answers."""
    return {
        b["tool_use_id"]
        for b in msg.blocks()
        if b.get("type") == "tool_result" and b.get("tool_use_id")
    }


def has_error_result(msg: Message) -> bool:
    """True if the message carries an errored tool result."""
    return any(
        b.get("type") == "tool_result" and b.get("is_error") is True for b in msg.blocks()
    )


def is_plain_user_turn(msg: Message) -> bool:
    """A genuine user turn, i.e. not a tool-result carrier.

    Conversation spans may only be cut at these points — cutting anywhere else
    risks orphaning a tool_use block from its tool_result, which the API rejects.
    """
    if msg.role != "user":
        return False
    return not tool_result_ids(msg)


def block_text(block: Block) -> str:
    """Best-effort text extraction for token counting and summarisation."""
    btype = block.get("type")
    if btype == "text":
        return block.get("text") or ""
    if btype == "thinking":
        return block.get("thinking") or ""
    if btype == "tool_use":
        return f"{block.get('name', '')} {_stringify(block.get('input'))}"
    if btype == "tool_result":
        return _stringify(block.get("content"))
    if btype in ("image", "document"):
        # Media is counted separately; text extraction returns nothing.
        return ""
    return _stringify(block)


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(_stringify(v) for v in value)
    if isinstance(value, dict):
        if value.get("type") == "text":
            return value.get("text") or ""
        return " ".join(f"{k} {_stringify(v)}" for k, v in value.items())
    return str(value)


def message_text(msg: Message) -> str:
    return "\n".join(t for t in (block_text(b) for b in msg.blocks()) if t)


class ReductionEvent(BaseModel):
    """One applied reduction, recorded for observability and the /stats endpoint."""

    strategy: str
    tokens_before: int
    tokens_after: int
    messages_before: int
    messages_after: int
    detail: str = ""

    @property
    def saved(self) -> int:
        return max(0, self.tokens_before - self.tokens_after)


class ReductionResult(BaseModel):
    """Outcome of running the context pipeline over one request."""

    request: MessagesRequest
    events: list[ReductionEvent] = Field(default_factory=list)
    tokens_before: int = 0
    tokens_after: int = 0
    budget: int = 0
    triggered: bool = False

    @property
    def saved(self) -> int:
        return max(0, self.tokens_before - self.tokens_after)
