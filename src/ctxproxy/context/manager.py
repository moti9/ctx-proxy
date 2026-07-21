"""The context pipeline.

Per incoming request:

    1. resolve session + profile
    2. re-apply any previously recorded fold (deterministic, no model call)
    3. count input tokens
    4. if under the trigger -> pass through untouched
    5. otherwise reduce, cheapest strategy first, until under target
    6. persist the ledger

Step 2 is what keeps the prompt stable between compactions; step 5's ordering is
what keeps quality high.
"""

from __future__ import annotations

import logging

from ..config import Config, ModelProfile
from ..logging import log_event
from ..store.archive import FoldArchive
from ..store.base import LedgerStore
from ..tokens.base import TokenCounter
from ..types_anthropic import Message, MessagesRequest, ReductionResult
from .budget import compute_budget
from .ledger import SessionLedger
from .protect import classify
from .strategies import ReductionContext, build_strategies
from .summarizer import Summarizer

log = logging.getLogger(__name__)


class ContextManager:
    def __init__(
        self, config: Config, store: LedgerStore, archive: FoldArchive | None = None
    ) -> None:
        self.config = config
        self.store = store
        self.archive = archive
        self._strategies = build_strategies(config.policy)

    async def process(
        self,
        request: MessagesRequest,
        *,
        profile: ModelProfile,
        session_key: str,
        counter: TokenCounter,
        summarizer: Summarizer,
        target_ratio: float | None = None,
    ) -> tuple[ReductionResult, SessionLedger]:
        ledger = await self.store.get_or_create(session_key, request.model)
        ledger.turns_seen += 1
        ledger.note_task(request.messages)

        # Clamp before the budget is computed: output_reserve is derived from
        # max_tokens, so budgeting against a value the backend will reject would
        # reserve window we never actually need.
        if profile.max_output_tokens and request.max_tokens > profile.max_output_tokens:
            log_event(
                log,
                "clamped max_tokens to the backend's ceiling",
                session=session_key,
                requested=request.max_tokens,
                clamped_to=profile.max_output_tokens,
            )
            request.max_tokens = profile.max_output_tokens

        budget = compute_budget(
            profile, self.config.policy, request, target_ratio=target_ratio
        )

        original_messages = list(request.messages)
        fold_anchor, to_original = self._reapply_fold(request, ledger, original_messages)

        tokens = await counter.count_request(request)
        result = ReductionResult(
            request=request,
            tokens_before=tokens,
            tokens_after=tokens,
            budget=budget.usable,
        )

        if tokens <= budget.trigger_at:
            await self.store.save(ledger)
            log_event(
                log,
                "passthrough",
                session=session_key,
                tokens=tokens,
                pct=round(budget.pct(tokens), 1),
                budget=budget.usable,
            )
            return result, ledger

        result.triggered = True
        log_event(
            log,
            "reduction triggered",
            session=session_key,
            tokens=tokens,
            pct=round(budget.pct(tokens), 1),
            trigger=budget.trigger_at,
            target=budget.target_at,
        )

        ctx = ReductionContext(
            request=request,
            profile=profile,
            policy=self.config.policy,
            budget=budget,
            counter=counter,
            ledger=ledger,
            protected=classify(request.messages, self.config.policy),
            summarize=summarizer.fold,
            tokens=tokens,
            to_original=to_original,
            fold_anchor=fold_anchor,
        )

        for strategy in self._strategies:
            if ctx.under_target:
                break
            event = await strategy.apply(ctx)
            if event is None:
                continue

            result.events.append(event)
            ledger.record_event(event)
            log_event(
                log,
                f"applied {event.strategy}",
                session=session_key,
                saved=event.saved,
                tokens=ctx.tokens,
                detail=event.detail,
            )

            # The protected set is index-based, so any strategy that changes the
            # message array invalidates it.
            if event.messages_before != event.messages_after:
                ctx.protected = classify(request.messages, self.config.policy)

        if ctx.new_fold:
            start, end = ctx.new_fold
            # Archive the raw span BEFORE recording the fold: after this the
            # originals never appear in a request again.
            if self.archive is not None and ctx.folded_span:
                try:
                    await self.archive.append(
                        session_key,
                        span=ctx.folded_span,
                        start=start,
                        end=end,
                        summary=ledger.summary,
                    )
                except Exception as exc:  # noqa: BLE001
                    # Archiving is a debugging aid, not part of serving the
                    # request. FoldArchive guards itself, but the manager must
                    # not depend on any particular implementation doing so —
                    # losing the archive is bad, failing the turn is worse.
                    log_event(
                        log,
                        "could not archive folded span",
                        level=logging.WARNING,
                        session=session_key,
                        error=f"{type(exc).__name__}: {exc}",
                    )
            ledger.record_fold(
                original_messages, start=start, end=end, summary=ledger.summary
            )

        result.tokens_after = ctx.tokens

        if ctx.tokens > budget.usable:
            log_event(
                log,
                "still over budget after reduction",
                level=logging.WARNING,
                session=session_key,
                tokens=ctx.tokens,
                usable=budget.usable,
                hint="lower keep_recent_turns, or check context_window in the profile",
            )

        await self.store.save(ledger)
        return result, ledger

    # ------------------------------------------------------------------ #

    def _reapply_fold(
        self,
        request: MessagesRequest,
        ledger: SessionLedger,
        original: list[Message],
    ):
        """Splice a previously recorded fold back in, if it still applies.

        Returns the working index of the summary message (the fold anchor) and a
        function mapping working indices back to the client's original indices.
        """
        identity = (None, lambda i: i)

        if not ledger.has_fold:
            return identity

        if not ledger.fold_is_valid(original):
            log_event(
                log,
                "fold invalidated; conversation diverged",
                session=ledger.session_key,
                folded_through=ledger.folded_through,
                messages=len(original),
            )
            ledger.invalidate_fold("history diverged")
            return identity

        start, end = ledger.fold_start, ledger.folded_through
        summary_msg = Message(
            role="user",
            content=[{"type": "text", "text": ledger.render_summary_block()}],
        )
        request.messages = original[:start] + [summary_msg] + original[end:]

        def to_original(working_index: int) -> int:
            if working_index <= start:
                return working_index
            return end + (working_index - start - 1)

        log_event(
            log,
            "fold reapplied",
            session=ledger.session_key,
            replaced=f"[{start}:{end}]",
            messages=len(request.messages),
        )
        return start, to_original
