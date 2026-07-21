"""The per-session ledger — the memory that survives compaction.

Claude Code is stateless: it resends the *entire* conversation every turn. That
has one very useful consequence. If we record how many of the original messages
we have already folded into a summary, we can reproduce that fold deterministically
on every subsequent turn, and only ever summarise material we have not seen
before.

Without this, each turn re-summarises the whole history from scratch: expensive,
lossy, and non-deterministic (the prefix changes every turn, so prompt caching
never hits). With it, the fold is stable between compactions and each message is
summarised exactly once.

The watermark is validated against a fingerprint of the folded span. If the
client rewinds, branches, or runs its own compaction, the fingerprint stops
matching and we rebuild rather than silently mis-splicing the conversation.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from ..types_anthropic import Message, ReductionEvent, message_text

MAX_TRACKED_EVENTS = 50
MAX_TRACKED_FILES = 100
MAX_SUMMARY_CHARS = 24_000


def _now() -> datetime:
    return datetime.now(UTC)


class SessionLedger(BaseModel):
    """Durable state for one conversation."""

    model_config = ConfigDict(extra="ignore")

    session_key: str
    model: str = ""
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    # The original task, captured verbatim on first sight. Never summarised.
    task_statement: str = ""

    # The rolling structured summary. Folded into, never regenerated.
    summary: str = ""

    # Number of original messages folded into `summary` so far. Indexes into
    # the message array as the client sends it.
    folded_through: int = 0

    # Fingerprint of the folded span, used to detect client-side divergence.
    fold_signature: str = ""

    # Where the fold begins (end of the protected head).
    fold_start: int = 0

    files_touched: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)

    compaction_count: int = 0
    total_tokens_saved: int = 0
    turns_seen: int = 0
    events: list[ReductionEvent] = Field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Fold bookkeeping
    # ------------------------------------------------------------------ #

    @property
    def has_fold(self) -> bool:
        return self.folded_through > self.fold_start and bool(self.summary)

    def record_fold(
        self,
        messages: list[Message],
        *,
        start: int,
        end: int,
        summary: str,
    ) -> None:
        self.fold_start = start
        self.folded_through = end
        self.fold_signature = fold_signature(messages, start, end)
        self.summary = summary[:MAX_SUMMARY_CHARS]
        self.compaction_count += 1
        self.touch()

    def fold_is_valid(self, messages: list[Message]) -> bool:
        """True if a previously recorded fold still applies to this history."""
        if not self.has_fold:
            return False
        if len(messages) < self.folded_through:
            return False
        return fold_signature(messages, self.fold_start, self.folded_through) == self.fold_signature

    def invalidate_fold(self, reason: str) -> None:
        """Drop the splice but keep the summary as prior context.

        The summary stays because it may hold the only surviving record of early
        work. It gets re-folded into the next compaction rather than discarded.
        """
        self.folded_through = 0
        self.fold_signature = ""
        self.fold_start = 0
        if self.summary and not self.summary.startswith("[carried forward"):
            self.summary = f"[carried forward after {reason}]\n{self.summary}"
        self.touch()

    # ------------------------------------------------------------------ #
    # Accumulators
    # ------------------------------------------------------------------ #

    def note_task(self, messages: list[Message]) -> None:
        if self.task_statement:
            return
        for msg in messages:
            if msg.role == "user":
                text = message_text(msg).strip()
                if text:
                    self.task_statement = text[:4000]
                    return

    def note_files(self, paths: list[str]) -> None:
        for path in paths:
            if path not in self.files_touched:
                self.files_touched.append(path)
        if len(self.files_touched) > MAX_TRACKED_FILES:
            del self.files_touched[: len(self.files_touched) - MAX_TRACKED_FILES]

    def record_event(self, event: ReductionEvent) -> None:
        self.events.append(event)
        self.total_tokens_saved += event.saved
        if len(self.events) > MAX_TRACKED_EVENTS:
            del self.events[: len(self.events) - MAX_TRACKED_EVENTS]
        self.touch()

    def touch(self) -> None:
        self.updated_at = _now()

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #

    def render_summary_block(self) -> str:
        """The text spliced into the conversation in place of folded messages."""
        parts: list[str] = [
            "<conversation-summary>",
            "The earlier portion of this conversation was compacted to stay within "
            "the context window. Everything below is a faithful record of that "
            "work. Treat it as established fact and continue from it.",
            "",
        ]
        if self.task_statement:
            parts += ["## Original task", self.task_statement.strip(), ""]
        if self.summary:
            parts += [self.summary.strip(), ""]
        if self.files_touched:
            parts += [
                "## Files touched",
                "\n".join(f"- {p}" for p in self.files_touched[-40:]),
                "",
            ]
        if self.open_questions:
            parts += [
                "## Open questions / unresolved",
                "\n".join(f"- {q}" for q in self.open_questions[-20:]),
                "",
            ]
        parts.append("</conversation-summary>")
        return "\n".join(parts)

    def stats(self) -> dict:
        return {
            "session_key": self.session_key,
            "model": self.model,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "turns_seen": self.turns_seen,
            "compactions": self.compaction_count,
            "tokens_saved": self.total_tokens_saved,
            "folded_through": self.folded_through,
            "summary_chars": len(self.summary),
            "files_touched": len(self.files_touched),
        }


def fold_signature(messages: list[Message], start: int, end: int) -> str:
    """Fingerprint a message span.

    Hashes the span length plus the boundary messages rather than every message
    in it: full hashing is O(history) per turn for no extra safety, since a
    client-side rewrite that leaves both boundaries and the length intact would
    not change what the summary means.
    """
    hasher = hashlib.blake2b(digest_size=16)
    hasher.update(f"{start}:{end}".encode())
    if 0 <= start < len(messages):
        hasher.update(message_text(messages[start])[:2000].encode())
    if 0 < end <= len(messages):
        hasher.update(message_text(messages[end - 1])[:2000].encode())
    return hasher.hexdigest()
