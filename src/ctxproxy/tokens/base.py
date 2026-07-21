"""Token counting interface.

Counting the full history on every request is the hot path — a 500-message
conversation counted naively per turn is O(n²) work over a session. Every
implementation therefore counts *per block* and memoises by content hash, so
each block is counted exactly once no matter how many turns resend it.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from typing import Any

from ..types_anthropic import Block, MessagesRequest

# Per-message framing overhead (role markers, delimiters). Anthropic does not
# publish an exact figure; this is a deliberate over-estimate.
MESSAGE_OVERHEAD_TOKENS = 8

# Tool definitions are rendered into the prompt ahead of everything else.
TOOL_DEFINITION_OVERHEAD_TOKENS = 12


class TokenCounter(ABC):
    """Counts tokens for Anthropic-shaped payloads."""

    def __init__(self, cache_size: int = 20_000) -> None:
        self._cache: dict[str, int] = {}
        self._cache_size = cache_size

    @abstractmethod
    def count_text(self, text: str) -> int:
        """Token count for a bare string. Implementations need only this."""

    async def count_request(self, request: MessagesRequest) -> int:
        """Total input tokens for a request: tools + system + messages."""
        total = 0

        for tool in request.tools or []:
            total += self._cached(tool) + TOOL_DEFINITION_OVERHEAD_TOKENS

        for block in request.system_blocks():
            total += self.count_block(block)

        for msg in request.messages:
            total += MESSAGE_OVERHEAD_TOKENS
            for block in msg.blocks():
                total += self.count_block(block)

        return total

    def count_block(self, block: Block) -> int:
        if block.get("type") == "image":
            return self.image_tokens(block)
        return self._cached(block)

    def image_tokens(self, block: Block) -> int:  # noqa: ARG002
        """Overridden by counters that can measure media properly."""
        return 1600

    # -- internals ---------------------------------------------------------- #

    def _cached(self, obj: Any) -> int:
        key = _stable_hash(obj)
        hit = self._cache.get(key)
        if hit is not None:
            return hit

        from ..types_anthropic import block_text

        text = block_text(obj) if isinstance(obj, dict) else json.dumps(obj, sort_keys=True)
        value = self.count_text(text)

        if len(self._cache) >= self._cache_size:
            # Cheap eviction: the working set is a growing conversation, so
            # wholesale clearing costs one re-count and avoids LRU bookkeeping.
            self._cache.clear()
        self._cache[key] = value
        return value


def _stable_hash(obj: Any) -> str:
    try:
        payload = json.dumps(obj, sort_keys=True, default=str)
    except (TypeError, ValueError):
        payload = repr(obj)
    return hashlib.blake2b(payload.encode(), digest_size=16).hexdigest()


def build_counter(tokenizer_config: Any) -> TokenCounter:
    """Construct a counter from a TokenizerConfig, degrading gracefully.

    A missing tiktoken install must not take the proxy down — it falls back to
    the heuristic counter and logs loudly, because a running proxy with slightly
    pessimistic counts beats a proxy that will not start.
    """
    from .local import HeuristicCounter, TiktokenCounter

    kind = tokenizer_config.kind

    if kind == "tiktoken":
        try:
            return TiktokenCounter(
                encoding=tokenizer_config.encoding,
                safety_multiplier=tokenizer_config.safety_multiplier,
                image_token_cost=tokenizer_config.image_token_cost,
            )
        except ImportError:
            import logging

            logging.getLogger(__name__).warning(
                "tiktoken not installed (pip install 'ctxproxy[tokenizers]'); "
                "falling back to heuristic counting"
            )

    return HeuristicCounter(
        chars_per_token=tokenizer_config.chars_per_token,
        safety_multiplier=tokenizer_config.safety_multiplier,
        image_token_cost=tokenizer_config.image_token_cost,
    )
