# Adding features

This page shows how to extend ctxproxy in the most common ways.

## Add a backend

1. Create a class in `backends/` inheriting from `backends/base.py::Backend`.
2. Implement `complete`, `stream`, and `count_tokens`.
3. Register it in `backends/__init__.py::build_backend`.
4. Add tests in `tests/test_backends_your_kind.py` (or extend `test_routes.py`).

Example skeleton:

```python
from collections.abc import AsyncIterator, Mapping

from ..types_anthropic import MessagesRequest
from .base import Backend

class MyBackend(Backend):
    async def complete(
        self, request: MessagesRequest, upstream_model: str, client_headers: Mapping[str, str]
    ) -> dict:
        payload = request.model_dump(exclude_none=True)
        payload["model"] = upstream_model
        return await self._post_json("/v1/messages", payload, self._headers(client_headers))

    def stream(
        self, request: MessagesRequest, upstream_model: str,
        client_headers: Mapping[str, str], input_tokens: int = 0
    ) -> AsyncIterator[bytes]:
        ...

    async def count_tokens(self, request: MessagesRequest, upstream_model: str) -> int:
        ...
```

Reuse `_forwarded_headers`, `_post_json`, and `_filter_beta` from the base class.

## Add a reduction strategy

1. Implement the `Strategy` protocol in `context/strategies.py`.
2. Add the class to `ALL_STRATEGIES`.
3. Optionally allow it in `ReductionPolicy` validation.
4. Add tests in `tests/test_manager.py`.

A strategy receives a `ReductionContext` and returns a `ReductionEvent` or `None`:

```python
class MyStrategy:
    name = "my_strategy"

    async def apply(self, ctx: ReductionContext) -> ReductionEvent | None:
        tokens_before = ctx.tokens
        # ... mutate ctx.request.messages and ctx.tokens ...
        if saved := tokens_before - ctx.tokens:
            return ReductionEvent(
                strategy=self.name,
                tokens_before=tokens_before,
                tokens_after=ctx.tokens,
                messages_before=len(ctx.messages),
                messages_after=len(ctx.request.messages),
                detail="did something",
            )
        return None
```

If your strategy changes the message list, the manager will recompute the
Protected Set.

## Add a tokenizer

1. Implement `tokens/base.py::TokenCounter`.
2. Wire it into `tokens/base.py::build_counter`.
3. Add a config option in `TokenizerConfig` if it needs settings.

```python
class MyCounter(TokenCounter):
    def count_text(self, text: str) -> int:
        return len(my_tokenize(text))
```

## Add a storage backend

1. Implement `store/base.py::LedgerStore`.
2. Swap it in `state.py`.

```python
class RedisLedgerStore(LedgerStore):
    async def load(self, session_key: str) -> SessionLedger | None: ...
    async def save(self, ledger: SessionLedger) -> None: ...
    async def list_all(self) -> list[SessionLedger]: ...
    async def delete(self, session_key: str) -> bool: ...
    async def prune(self, ttl_hours: int) -> int: ...
```

## Add a CLI command

Edit `cli.py` with a Typer command:

```python
@app.command()
def my_command(config_path: Path = ConfigOpt) -> None:
    config = _load(config_path)
    ...
```

## Add an HTTP endpoint

Add a router function in `routes.py`:

```python
@router.get("/my-endpoint")
async def my_endpoint(http_request: Request):
    state = _state(http_request)
    return {"ok": True}
```

## Testing new code

- Unit tests for pure logic go in `tests/test_<module>.py`.
- End-to-end pipeline tests use the fixtures in `conftest.py`.
- Translation/SSE tests use `respx` to mock HTTP.
- Always run `pytest -q` and `ruff check src tests` before committing.

## Keep the design intact

Before extending, check these invariants:

- Do not split `tool_use`/`tool_result` pairs across cut points.
- Do not drop thinking blocks on native Anthropic.
- Do not re-summarize already-folded history.
- Keep token counting additive and memoised.
- Keep ledger writes atomic.
