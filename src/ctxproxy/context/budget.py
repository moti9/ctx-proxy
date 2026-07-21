"""Budget arithmetic.

This is where the "assumed 200K window" bug gets fixed. Every number comes from
the resolved profile, never from a client-side guess.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import ModelProfile, ReductionPolicy
from ..types_anthropic import MessagesRequest


@dataclass(frozen=True)
class Budget:
    window: int
    output_reserve: int
    safety_buffer: int
    usable: int
    trigger_at: int
    target_at: int

    def is_over(self, tokens: int) -> bool:
        return tokens > self.trigger_at

    def headroom(self, tokens: int) -> int:
        return self.usable - tokens

    def pct(self, tokens: int) -> float:
        return 100.0 * tokens / self.usable if self.usable > 0 else 0.0

    def describe(self) -> str:
        return (
            f"window={self.window} usable={self.usable} "
            f"trigger={self.trigger_at} target={self.target_at}"
        )


def compute_budget(
    profile: ModelProfile,
    policy: ReductionPolicy,
    request: MessagesRequest,
    *,
    target_ratio: float | None = None,
) -> Budget:
    """Derive the input-token budget for one request.

    ``output_reserve`` must be at least the request's own ``max_tokens``: the
    model is permitted to generate that much, and input + output have to fit
    inside the window together. The profile value acts as a floor for cases
    where the client understates max_tokens but thinking tokens expand.
    """
    output_reserve = max(profile.output_reserve, request.max_tokens)
    usable = profile.context_window - output_reserve - profile.safety_buffer

    if usable <= 0:
        # Misconfiguration (or a genuinely tiny model). Degrade to a positive
        # budget so the pipeline still reduces rather than dividing by zero;
        # the caller logs this loudly.
        usable = max(1, profile.context_window // 4)

    effective_target = target_ratio if target_ratio is not None else policy.target_ratio

    return Budget(
        window=profile.context_window,
        output_reserve=output_reserve,
        safety_buffer=profile.safety_buffer,
        usable=usable,
        trigger_at=int(usable * policy.trigger_ratio),
        target_at=int(usable * effective_target),
    )
