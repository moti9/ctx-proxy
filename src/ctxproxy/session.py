"""Session identity.

Claude Code sends ``X-Claude-Code-Session-Id`` (and agent/parent-agent ids for
subagents), but we must not *depend* on it: header names have changed before,
the VS Code extension and CLI have differed, and a gateway in the path can strip
unknown headers. Losing the session id silently would mean losing the rolling
summary, which is the one thing this proxy promises not to lose.

So identity is derived in two layers:

1. If the header is present, use it. Exact, cheap, survives prompt edits.
2. Otherwise, fingerprint the stable *prefix* of the conversation. Claude Code
   resends the full history every turn, so the first plain user turn plus the
   system prompt is a stable identifier for the life of that conversation.

Subagents get their own ledger — they are separate conversations with their own
task statements, and folding them together would corrupt both summaries.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass

from .types_anthropic import MessagesRequest, is_plain_user_turn, message_text

SESSION_HEADER = "x-claude-code-session-id"
AGENT_HEADER = "x-claude-code-agent-id"
PARENT_AGENT_HEADER = "x-claude-code-parent-agent-id"

# Alternative spellings seen in the wild / across client versions.
_SESSION_HEADER_ALIASES = (
    SESSION_HEADER,
    "x-session-id",
    "anthropic-session-id",
)

_FINGERPRINT_CHARS = 4000


@dataclass(frozen=True)
class SessionKey:
    """Stable identifier for one conversation."""

    value: str
    source: str  # "header" | "fingerprint"
    agent_id: str | None = None
    parent_agent_id: str | None = None

    @property
    def is_subagent(self) -> bool:
        return self.parent_agent_id is not None

    def __str__(self) -> str:
        return self.value


def derive_session_key(headers: Mapping[str, str], request: MessagesRequest) -> SessionKey:
    lowered = {k.lower(): v for k, v in headers.items()}
    agent_id = lowered.get(AGENT_HEADER)
    parent_agent_id = lowered.get(PARENT_AGENT_HEADER)

    for alias in _SESSION_HEADER_ALIASES:
        if raw := lowered.get(alias):
            # Subagents share the parent's session id but are distinct
            # conversations, so the agent id is part of the key.
            value = f"{_safe(raw)}.{_safe(agent_id)}" if agent_id else _safe(raw)
            return SessionKey(value, "header", agent_id, parent_agent_id)

    return SessionKey(
        _fingerprint(request),
        "fingerprint",
        agent_id,
        parent_agent_id,
    )


def _fingerprint(request: MessagesRequest) -> str:
    """Hash the parts of a conversation that do not change as it grows.

    Uses the system prompt and the first plain user turn. Tool-result carriers
    are skipped because in a resumed session the first message can be a
    tool_result whose content varies.
    """
    hasher = hashlib.sha256()
    hasher.update(request.model.encode())

    system_text = "\n".join(b.get("text", "") for b in request.system_blocks())
    hasher.update(system_text[:_FINGERPRINT_CHARS].encode())

    for msg in request.messages:
        if is_plain_user_turn(msg):
            hasher.update(message_text(msg)[:_FINGERPRINT_CHARS].encode())
            break

    return f"fp_{hasher.hexdigest()[:24]}"


def _safe(value: str | None) -> str:
    """Sanitise for use as a filename component."""
    if not value:
        return "none"
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in value)[:64]
