"""Ledger persistence interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..context.ledger import SessionLedger


class LedgerStore(ABC):
    @abstractmethod
    async def load(self, session_key: str) -> SessionLedger | None: ...

    @abstractmethod
    async def save(self, ledger: SessionLedger) -> None: ...

    @abstractmethod
    async def list_all(self) -> list[SessionLedger]: ...

    @abstractmethod
    async def delete(self, session_key: str) -> bool: ...

    @abstractmethod
    async def prune(self, ttl_hours: int) -> int: ...

    async def get_or_create(self, session_key: str, model: str) -> SessionLedger:
        ledger = await self.load(session_key)
        if ledger is None:
            ledger = SessionLedger(session_key=session_key, model=model)
        elif model and ledger.model != model:
            # Model switched mid-session (a /model command, or a fallback).
            # The summary is still valid prose, so keep it — only the model
            # label changes.
            ledger.model = model
        return ledger
