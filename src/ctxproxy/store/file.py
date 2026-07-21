"""File-backed ledger store.

Single-developer deployment, so no locking beyond atomic replace. Writes go to a
temp file in the same directory and are renamed into place, which is atomic on
POSIX — a crash mid-write can never leave a half-written ledger that would
deserialise into a corrupt summary.

All disk I/O runs in a worker thread so it never blocks the event loop while a
response is streaming.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import time
from pathlib import Path

from ..context.ledger import SessionLedger
from .base import LedgerStore

log = logging.getLogger(__name__)


class FileLedgerStore(LedgerStore):
    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory).expanduser()
        self.directory.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, asyncio.Lock] = {}

    def _path(self, session_key: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in session_key)[:120]
        return self.directory / f"{safe}.json"

    def _lock(self, session_key: str) -> asyncio.Lock:
        # Serialises concurrent turns of the same session. Claude Code can have
        # a subagent and its parent in flight simultaneously; without this the
        # later write silently clobbers the earlier fold.
        if session_key not in self._locks:
            self._locks[session_key] = asyncio.Lock()
        return self._locks[session_key]

    async def load(self, session_key: str) -> SessionLedger | None:
        return await asyncio.to_thread(self._load_sync, self._path(session_key))

    @staticmethod
    def _load_sync(path: Path) -> SessionLedger | None:
        if not path.is_file():
            return None
        try:
            return SessionLedger.model_validate_json(path.read_text())
        except Exception as exc:  # noqa: BLE001
            log.warning("discarding unreadable ledger %s: %s", path.name, exc)
            return None

    async def save(self, ledger: SessionLedger) -> None:
        async with self._lock(ledger.session_key):
            await asyncio.to_thread(self._save_sync, self._path(ledger.session_key), ledger)

    @staticmethod
    def _save_sync(path: Path, ledger: SessionLedger) -> None:
        payload = ledger.model_dump_json(indent=2)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=".json")
        try:
            with os.fdopen(fd, "w") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    async def list_all(self) -> list[SessionLedger]:
        return await asyncio.to_thread(self._list_sync)

    def _list_sync(self) -> list[SessionLedger]:
        out: list[SessionLedger] = []
        for path in self.directory.glob("*.json"):
            if ledger := self._load_sync(path):
                out.append(ledger)
        out.sort(key=lambda item: item.updated_at, reverse=True)
        return out

    async def delete(self, session_key: str) -> bool:
        path = self._path(session_key)
        existed = path.is_file()
        path.unlink(missing_ok=True)
        return existed

    async def prune(self, ttl_hours: int) -> int:
        return await asyncio.to_thread(self._prune_sync, ttl_hours)

    def _prune_sync(self, ttl_hours: int) -> int:
        if ttl_hours <= 0:
            return 0
        cutoff = time.time() - ttl_hours * 3600
        removed = 0
        for path in self.directory.glob("*.json"):
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
                    removed += 1
            except OSError:
                continue
        # Sweep any temp files orphaned by a crash mid-write.
        for path in self.directory.glob(".tmp-*.json"):
            try:
                path.unlink()
            except OSError:
                continue
        return removed
