"""Per-target health tracking for failover.

When a profile has a fallback chain, the proxy needs to know which upstream
targets are currently worth trying. A target that just returned "no deployments
available" should be skipped for a short cooldown rather than retried on every
request — otherwise every turn during an outage pays that target's failure
latency before falling back.

The state is intentionally tiny and in-memory: a cooldown deadline per
(backend, model). It is per-process (a single developer's proxy), reset on
restart, and needs no locking — the event loop is single-threaded and every
operation here is a plain dict read/write.

Recovery is automatic and needs no probe thread: the cooldown simply expires, so
the next request re-includes the primary in priority order. If it succeeds the
cooldown is cleared; if it fails again it is pushed back out.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

Target = tuple[str, str]  # (backend name, upstream model)
T = TypeVar("T")


class HealthTracker:
    def __init__(self, cooldown_s: float = 30.0) -> None:
        self.cooldown_s = cooldown_s
        self._down_until: dict[Target, float] = {}

    def _now(self) -> float:
        return time.monotonic()

    def available(self, target: Target) -> bool:
        until = self._down_until.get(target)
        return until is None or self._now() >= until

    def mark_down(self, target: Target) -> None:
        self._down_until[target] = self._now() + self.cooldown_s

    def mark_up(self, target: Target) -> None:
        self._down_until.pop(target, None)

    def order(
        self, items: list[T], key: Callable[[T], Target] | None = None
    ) -> list[T]:
        """Priority order with currently-cooling-down items moved to the back.

        ``key`` maps an item to its (backend, model) target; it defaults to the
        identity, so a list of targets can be ordered directly, but a list of
        profiles can be ordered by keying on their resolved target. Relative
        priority is preserved within the healthy and cooling groups, so the
        primary is always preferred while healthy. If every item is cooling down
        the original order is returned unchanged, so the primary is still
        re-probed first rather than the proxy giving up.
        """
        k = key or (lambda x: x)  # type: ignore[return-value,assignment]
        healthy_idx = [i for i, it in enumerate(items) if self.available(k(it))]
        if not healthy_idx:
            return list(items)
        healthy = set(healthy_idx)
        cooling_idx = [i for i in range(len(items)) if i not in healthy]
        return [items[i] for i in healthy_idx + cooling_idx]
