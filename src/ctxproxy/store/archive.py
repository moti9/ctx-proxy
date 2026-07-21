"""Raw transcript archive for compacted spans.

Compaction is the one irreversible step in the pipeline: once a span is folded
into the summary, the original messages are gone from every subsequent request
and nothing can get them back. That is fine for the model — the summary is what
it needs — but it is bad for everyone else. When a session goes wrong three
hours in, "what did the summariser actually drop?" is the first question, and
without this it is unanswerable.

So the raw span is appended to a per-session JSONL file before it is discarded.
This is deliberately *not* part of the ledger: the ledger is deserialised on
every single request and must stay small, whereas this grows without bound and
is read only by a human running `ctxproxy archive`.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..types_anthropic import Message

log = logging.getLogger(__name__)


class FoldArchive:
    def __init__(self, directory: Path, max_mb: int = 25) -> None:
        self.directory = Path(directory).expanduser()
        self.max_bytes = max_mb * 1024 * 1024

    def _path(self, session_key: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in session_key)[:120]
        return self.directory / f"{safe}.jsonl"

    async def append(
        self,
        session_key: str,
        *,
        span: list[Message],
        start: int,
        end: int,
        summary: str,
    ) -> None:
        """Record one fold. Never raises — archiving must not fail a request."""
        try:
            await asyncio.to_thread(
                self._append_sync, session_key, span, start, end, summary
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("could not archive folded span for %s: %s", session_key, exc)

    def _append_sync(
        self,
        session_key: str,
        span: list[Message],
        start: int,
        end: int,
        summary: str,
    ) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        entry = {
            "archived_at": datetime.now(UTC).isoformat(),
            "original_range": [start, end],
            "message_count": len(span),
            "summary_written": summary,
            "messages": [m.model_dump(exclude_none=True) for m in span],
        }
        path = self._path(session_key)
        with path.open("a") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")
        self._enforce_cap(path)

    def _enforce_cap(self, path: Path) -> None:
        """Drop oldest folds once a session's archive exceeds the cap.

        Trims to half the cap rather than exactly to it, so a long session does
        not rewrite the whole file on every single fold.
        """
        if self.max_bytes <= 0 or path.stat().st_size <= self.max_bytes:
            return

        lines = path.read_text().splitlines()
        kept: list[str] = []
        size = 0
        for line in reversed(lines):          # newest folds are the useful ones
            size += len(line) + 1
            if size > self.max_bytes // 2:
                break
            kept.append(line)
        kept.reverse()

        dropped = len(lines) - len(kept)
        path.write_text("\n".join(kept) + ("\n" if kept else ""))
        log.info(
            "archive %s exceeded %d MB; dropped %d oldest fold(s)",
            path.name,
            self.max_bytes // (1024 * 1024),
            dropped,
        )

    def read(self, session_key: str) -> list[dict[str, Any]]:
        path = self._path(session_key)
        if not path.is_file():
            return []
        entries = []
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return entries

    def prune(self, ttl_hours: int) -> int:
        import time

        if ttl_hours <= 0 or not self.directory.is_dir():
            return 0
        cutoff = time.time() - ttl_hours * 3600
        removed = 0
        for path in self.directory.glob("*.jsonl"):
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
                    removed += 1
            except OSError:
                continue
        return removed
