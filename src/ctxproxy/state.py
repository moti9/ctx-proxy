"""Process-wide wiring: backends, counters, summarisers, ledger store."""

from __future__ import annotations

import logging

from .backends import Backend, BackendRegistry
from .config import Config, ModelProfile
from .context.manager import ContextManager
from .context.summarizer import Summarizer
from .store.archive import FoldArchive
from .store.file import FileLedgerStore
from .tokens.base import TokenCounter, build_counter
from .tokens.upstream import UpstreamCounter
from .types_anthropic import MessagesRequest

log = logging.getLogger(__name__)


class AppState:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.registry = BackendRegistry(config.backends)
        self.store = FileLedgerStore(config.server.state_dir)
        self.archive = FoldArchive(config.server.state_dir / "archive")
        self.manager = ContextManager(config, self.store, self.archive)

        # Counters hold a memo cache keyed by block content, so they must
        # outlive a single request — that cache is what keeps token counting
        # off the O(n²) path across a long session.
        self._counters: dict[tuple[str, str], TokenCounter] = {}

    def backend_for(self, profile: ModelProfile) -> Backend:
        return self.registry.get(profile.backend)

    def counter_for(self, profile: ModelProfile, upstream_model: str) -> TokenCounter:
        key = (profile.match, upstream_model)
        if key in self._counters:
            return self._counters[key]

        local = build_counter(profile.tokenizer)

        if profile.tokenizer.kind == "upstream" or (
            profile.supports_count_tokens and profile.native_anthropic
        ):
            counter: TokenCounter = UpstreamCounter(
                self.backend_for(profile), upstream_model, local
            )
        else:
            counter = local

        self._counters[key] = counter
        return counter

    def summarizer_for(self, profile: ModelProfile, requested_model: str) -> Summarizer:
        """Build a summariser.

        Summarising with a strong model while running the task on a cheap one is
        explicitly supported — it is usually the right trade, since a bad summary
        poisons every subsequent turn.
        """
        summ_profile = self.config.summarizer_for(profile)
        model = _resolve_model_name(summ_profile, requested_model)
        backend = self.backend_for(summ_profile)

        async def complete(request: MessagesRequest) -> str:
            return await backend.text_completion(request, model)

        return Summarizer(complete, summ_profile)

    async def aclose(self) -> None:
        await self.registry.aclose()


def _resolve_model_name(profile: ModelProfile, requested: str) -> str:
    if profile.upstream_model:
        return profile.upstream_model
    if "*" not in profile.match:
        return profile.match
    return requested
