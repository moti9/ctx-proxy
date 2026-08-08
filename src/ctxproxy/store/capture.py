"""Upstream-payload capture — the diagnostic that shows what the model got.

Everything else in this proxy reasons about what *should* be sent. When a
session behaves oddly — the model completing only part of a request, say — the
first unanswerable question is "what did the backend actually receive?" The
translated OpenAI payload never appears in a log, so corruption in translation
and plain model weakness look identical from the outside.

This records, per request, the exact JSON sent upstream and the response that
came back, to a per-session JSONL file. It is a debugging aid, never part of
serving a request: it is off unless ``server.debug_capture_dir`` is set, it runs
off the request path via a thread, and it never raises into the caller.

Payloads contain the full conversation, so this is opt-in by design and must not
be pointed at a shared or committed location.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class DebugCapture:
    def __init__(self, directory: Path | None) -> None:
        self.directory = Path(directory).expanduser() if directory else None

    @property
    def enabled(self) -> bool:
        return self.directory is not None

    def _path(self, session_key: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in session_key)[:120]
        return self.directory / f"{safe}.jsonl"  # type: ignore[union-attr]

    async def record(
        self,
        session_key: str,
        *,
        kind: str,
        upstream_model: str,
        payload: dict[str, Any],
        response: Any = None,
    ) -> None:
        """Record one exchange. Never raises — capture must not fail a request."""
        if not self.enabled:
            return
        try:
            await asyncio.to_thread(
                self._record_sync, session_key, kind, upstream_model, payload, response
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("could not capture upstream exchange for %s: %s", session_key, exc)

    def _record_sync(
        self,
        session_key: str,
        kind: str,
        upstream_model: str,
        payload: dict[str, Any],
        response: Any,
    ) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)  # type: ignore[union-attr]
        entry = {
            "captured_at": datetime.now(UTC).isoformat(),
            "kind": kind,  # "complete" | "stream"
            "upstream_model": upstream_model,
            "message_count": len(payload.get("messages") or []),
            "max_tokens": payload.get("max_tokens") or payload.get("max_completion_tokens"),
            "payload": payload,
            "response": response,
        }
        with self._path(session_key).open("a") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")

    def read(self, session_key: str) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        path = self._path(session_key)
        if not path.is_file():
            return []
        entries: list[dict[str, Any]] = []
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return entries
