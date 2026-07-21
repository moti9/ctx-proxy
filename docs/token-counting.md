# Token counting

Accurate, fast token counts are essential because every reduction decision is
based on them. This page describes how ctxproxy counts tokens and why it is
designed that way.

## Why counting is hard

- Claude Code relies on `/v1/messages/count_tokens` to drive its HUD and auto-
  compact.
- Custom gateways often do not serve that endpoint.
- Anthropic's tokenizer is not publicly available.
- Counting the full history every turn is O(n²) over a long session.

## Counter interface

`tokens/base.py` defines `TokenCounter`:

```python
class TokenCounter(ABC):
    @abstractmethod
    def count_text(self, text: str) -> int: ...

    async def count_request(self, request: MessagesRequest) -> int: ...
    def count_block(self, block: Block) -> int: ...
    def image_tokens(self, block: Block) -> int: ...
```

Implementations only need `count_text`. The base class handles:

- Summing tools + system + messages + per-message overhead.
- Memoising block counts by content hash.
- Treating image blocks as a flat cost.

## Memoisation

Every block is hashed with a stable JSON representation and its token count is
cached. Across turns, most blocks are identical (the conversation is resent in
full), so the hot-path cost becomes incremental instead of quadratic.

The cache is bounded. When it fills, it clears entirely — the working set is a
growing conversation, so wholesale clearing is cheaper than LRU bookkeeping.

## Counter implementations

### UpstreamCounter

`tokens/upstream.py` calls the backend's `/v1/messages/count_tokens` when
available. On any failure it degrades to a local counter for the rest of the
process and logs a warning.

Used when `profile.tokenizer.kind == "upstream"` or when the profile is native
Anthropic and claims to support count tokens.

### HeuristicCounter

`tokens/local.py` uses a character ratio:

```python
tokens = ceil(len(text) / chars_per_token * safety_multiplier)
```

Defaults:

- `chars_per_token = 3.5` (pessimistic vs the ~4.0 often quoted for English).
- `safety_multiplier = 1.10`.

It is deliberately conservative. Under-counting causes hard failures; over-
counting only wastes a little window.

### TiktokenCounter

Also in `tokens/local.py`. Uses OpenAI's `tiktoken` BPE encoder. Better than the
heuristic on code, JSON, and non-English text, but still not Anthropic's actual
tokenizer. The `safety_multiplier` absorbs the gap.

Requires the `tokenizers` extra:

```bash
uv pip install -e ".[tokenizers]"
```

## Choosing a tokenizer in config

```yaml
profiles:
  - match: "claude-*"
    tokenizer:
      kind: upstream

  - match: "our-coder"
    tokenizer:
      kind: tiktoken
      encoding: cl100k_base
      safety_multiplier: 1.10

  - match: "*"
    tokenizer:
      kind: heuristic
      chars_per_token: 3.5
      safety_multiplier: 1.15
```

| Kind | Use when | Notes |
|------|----------|-------|
| `upstream` | Native Anthropic | Most accurate; falls back locally if needed. |
| `tiktoken` | Non-native, accuracy matters | Needs optional dep; still approximate. |
| `heuristic` | Always works | Conservative default; no dependencies. |

## Debugging count discrepancies

Symptoms and fixes:

- **Still hitting context limits.** The tokenizer is likely under-counting.
  Raise `safety_multiplier` or switch to `tiktoken`.
- **Compaction fires too early.** `safety_multiplier` may be too high, or
  `chars_per_token` too low. Lower them cautiously.
- **`count_tokens` returns too small a number.** For non-native backends, ensure
  `supports_count_tokens` is `false` so the proxy counts locally.

## Images

Images are counted as a flat cost (`image_token_cost`, default 1600). Real
vision token counts depend on resolution and are not computed here.
