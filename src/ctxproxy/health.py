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

Target = tuple[str, str]  # (backend name, upstream model)


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

    def order(self, targets: list[Target]) -> list[Target]:
        """Priority order with currently-cooling-down targets moved to the back.

        Relative priority among healthy targets (and among cooling-down ones) is
        preserved, so the primary is always preferred while it is healthy. If
        every target is cooling down the original order is returned unchanged, so
        the primary is still re-probed first rather than the proxy giving up.
        """
        healthy = [t for t in targets if self.available(t)]
        if not healthy:
            return list(targets)
        cooling = [t for t in targets if t not in healthy]
        return healthy + cooling
