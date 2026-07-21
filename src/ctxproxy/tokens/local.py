"""Local token counters, for backends that cannot count for us."""

from __future__ import annotations

import math

from ..types_anthropic import Block
from .base import TokenCounter


class HeuristicCounter(TokenCounter):
    """Character-ratio estimate.

    Always available. Biased to over-count: the failure mode we care about is
    under-counting (compact too late -> hard 400), not over-counting (compact
    slightly early -> a bit of wasted window).
    """

    def __init__(
        self,
        chars_per_token: float = 3.5,
        safety_multiplier: float = 1.10,
        image_token_cost: int = 1600,
    ) -> None:
        super().__init__()
        if chars_per_token <= 0:
            raise ValueError("chars_per_token must be positive")
        self.chars_per_token = chars_per_token
        self.safety_multiplier = safety_multiplier
        self._image_token_cost = image_token_cost

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        return math.ceil(len(text) / self.chars_per_token * self.safety_multiplier)

    def image_tokens(self, block: Block) -> int:  # noqa: ARG002 — flat cost; see base
        return self._image_token_cost


class TiktokenCounter(TokenCounter):
    """BPE counting via tiktoken.

    Not the same tokenizer Anthropic uses, so counts are approximate — the
    safety multiplier absorbs the difference. Considerably better than the
    character heuristic on code, JSON, and non-English text, which is exactly
    what fills an agentic context window.
    """

    def __init__(
        self,
        encoding: str = "cl100k_base",
        safety_multiplier: float = 1.10,
        image_token_cost: int = 1600,
    ) -> None:
        super().__init__()
        import tiktoken  # raises ImportError -> caller falls back

        self._encoding = tiktoken.get_encoding(encoding)
        self.safety_multiplier = safety_multiplier
        self._image_token_cost = image_token_cost

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        return math.ceil(len(self._encoding.encode(text)) * self.safety_multiplier)

    def image_tokens(self, block: Block) -> int:  # noqa: ARG002 — flat cost; see base
        return self._image_token_cost
