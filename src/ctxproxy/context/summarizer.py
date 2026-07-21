"""Rolling-summary generation.

Two properties matter more than summary quality here:

1. **It must not itself overflow.** Client-side compaction classically fails by
   sending the whole conversation to the summariser in one shot — on a smaller
   backend that request exceeds the window and errors, so recovery breaks at
   exactly the moment you need it. We chunk the span and fold sequentially, so
   the summariser never sees more than it can hold.
2. **It folds, it does not regenerate.** Each call takes the previous summary
   plus only the new material. Content is summarised once and then carried
   forward, rather than being re-compressed on every compaction until it decays
   into nothing.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from ..config import ModelProfile
from ..types_anthropic import Message, MessagesRequest, block_text, message_text
from .ledger import SessionLedger

log = logging.getLogger(__name__)

# Returns the assistant's text for a completed request. The caller binds the
# backend and upstream model when constructing this, so no profile is passed —
# the summariser only needs "run this and give me the text back".
CompleteFn = Callable[[MessagesRequest], Awaitable[str]]

SYSTEM_PROMPT = """\
You maintain the running ledger for a long software-engineering session whose \
earlier turns must be dropped to fit the model's context window. Your output \
IS the agent's memory of that work — anything you omit is permanently lost.

Rewrite the existing ledger to incorporate the new transcript. Preserve every \
durable fact; drop only redundancy and narration.

Output these sections, omitting any that would be empty. No preamble.

## State
Where the work currently stands. What is done, what is verified, what is in flight.

## Decisions
Choices made and the reason for each. These must survive: re-litigating a settled \
decision is the most common failure after compaction.

## Files
Paths created, modified, or studied, each with a few words on what changed.

## Interfaces
Function signatures, schemas, endpoints, env vars, commands, and config keys \
established so far. Record exact names — approximations cause broken references later.

## Unresolved
Open errors, failing tests, unanswered questions, known-broken states. Be specific: \
exact error text beats a paraphrase.

## Next
The immediate next step, if one was established.

Rules:
- Write factually and densely. No hedging, no "the user asked".
- Keep exact identifiers, paths, and error strings verbatim.
- Never invent progress that is not in the transcript.
- Carry forward unresolved items until the transcript shows them resolved.
"""


class Summarizer:
    def __init__(
        self,
        complete: CompleteFn,
        profile: ModelProfile,
        *,
        max_output_tokens: int = 4000,
    ) -> None:
        self._complete = complete
        self._profile = profile
        self._max_output_tokens = max_output_tokens

    @property
    def _chunk_chars(self) -> int:
        """Transcript characters per fold call.

        Derived from the summariser's own window so a small summariser model
        cannot be handed more than it can process.
        """
        usable = max(
            8_000,
            self._profile.context_window
            - self._max_output_tokens
            - self._profile.safety_buffer,
        )
        # Reserve ~40% of the window for the prompt scaffold, the prior ledger,
        # and tokenizer error.
        return int(usable * self._profile.tokenizer.chars_per_token * 0.6)

    async def fold(self, span: list[Message], ledger: SessionLedger) -> str:
        """Fold ``span`` into ``ledger.summary`` and return the new summary."""
        summary = ledger.summary
        chunks = _chunk_transcript(span, self._chunk_chars)

        if len(chunks) > 1:
            log.info("summarising %d message(s) across %d chunk(s)", len(span), len(chunks))

        for chunk in chunks:
            summary = await self._fold_once(summary, chunk, ledger)
        return summary

    async def _fold_once(self, prior: str, transcript: str, ledger: SessionLedger) -> str:
        sections = []
        if ledger.task_statement:
            sections.append(
                f"# Original task\n{ledger.task_statement[:2000]}"
            )
        sections.append(
            f"# Existing ledger\n{prior or '(none — this is the first compaction)'}"
        )
        sections.append(f"# New transcript to fold in\n{transcript}")
        sections.append(
            "Produce the updated ledger now, using the required sections."
        )

        request = MessagesRequest(
            model=self._profile.upstream_model or "",
            system=SYSTEM_PROMPT,
            max_tokens=self._max_output_tokens,
            messages=[Message(role="user", content="\n\n".join(sections))],
        )

        text = (await self._complete(request)).strip()
        if not text:
            raise RuntimeError("summariser returned empty output")
        return text


def _chunk_transcript(span: list[Message], max_chars: int) -> list[str]:
    """Render messages to text, split on message boundaries."""
    chunks: list[str] = []
    current: list[str] = []
    size = 0

    for msg in span:
        rendered = _render(msg)
        if size + len(rendered) > max_chars and current:
            chunks.append("\n\n".join(current))
            current, size = [], 0
        # A single message larger than the budget is hard-truncated at the
        # middle, keeping both ends — the head says what was attempted and the
        # tail usually carries the result or the error.
        if len(rendered) > max_chars:
            half = max_chars // 2
            rendered = (
                rendered[:half]
                + f"\n... [{len(rendered) - max_chars} characters elided] ...\n"
                + rendered[-half:]
            )
        current.append(rendered)
        size += len(rendered)

    if current:
        chunks.append("\n\n".join(current))
    return chunks or [""]


def _render(msg: Message) -> str:
    lines = [f"[{msg.role}]"]
    for block in msg.blocks():
        btype = block.get("type")
        if btype == "text":
            lines.append(block.get("text", ""))
        elif btype == "tool_use":
            lines.append(f"<tool-call name={block.get('name')}>{block_text(block)}</tool-call>")
        elif btype == "tool_result":
            flag = " error" if block.get("is_error") else ""
            lines.append(f"<tool-result{flag}>{block_text(block)}</tool-result>")
        elif btype in ("thinking", "redacted_thinking"):
            continue  # reasoning is not durable state
        else:
            text = block_text(block)
            if text:
                lines.append(f"<{btype}>{text}</{btype}>")
    return "\n".join(line for line in lines if line.strip()) or f"[{msg.role}] (empty)"


def render_span(span: list[Message]) -> str:
    """Exposed for tests and the `ctxproxy inspect` command."""
    return "\n\n".join(_render(m) for m in span)


__all__ = ["Summarizer", "render_span", "message_text"]
