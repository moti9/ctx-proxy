# ctxproxy developer documentation

These docs are for anyone who needs to understand, modify, or operate ctxproxy.
They assume you have read the [README](../README.md) and know what the proxy does
at a high level.

## What is here

| Doc | Read this if you want to... |
|-----|----------------------------|
| [Getting started](./getting-started.md) | Install, run tests, and spin up the proxy locally. |
| [Architecture](./architecture.md) | Understand the big pieces and how they fit together. |
| [Request lifecycle](./request-lifecycle.md) | Trace a single `/v1/messages` call from HTTP to upstream. |
| [Context reduction](./context-reduction.md) | Learn how the proxy decides what to keep, clear, or summarize. |
| [Backends and translation](./backends-and-translation.md) | See how native Anthropic and OpenAI-compatible paths differ. |
| [Token counting](./token-counting.md) | Understand why token counts are accurate enough and fast enough. |
| [Session and storage](./session-and-storage.md) | Learn how sessions are identified and how the ledger persists. |
| [Configuration](./configuration.md) | Look up profiles, backends, policy knobs, and their interactions. |
| [Adding features](./adding-features.md) | Add a backend, reduction strategy, tokenizer, or storage backend. |
| [Debugging](./debugging.md) | Read logs, use `ctxproxy doctor`, inspect sessions, and fix tests. |
| [Glossary](./glossary.md) | Clarify terms like "fold", "Protected Set", and "ledger". |

## Where the code lives

```
src/ctxproxy/
├── cli.py              # Typer commands
├── app.py              # FastAPI factory
├── routes.py           # HTTP surface
├── state.py            # Process-wide wiring
├── config.py           # YAML → Pydantic
├── session.py          # Session identity
├── types_anthropic.py  # Wire models
├── backends/           # Upstream dispatch
├── context/            # Compaction pipeline
├── tokens/             # Token counting
├── translate/          # Anthropic ↔ OpenAI
└── store/              # Ledger persistence
```

## One-sentence guide to each module

- **`config.py`** — every decision starts here. The `native_anthropic` flag is the
  master switch.
- **`routes.py`** — HTTP entry point; validates, resolves, dispatches.
- **`state.py`** — holds long-lived objects: backends, counters, store, manager.
- **`context/manager.py`** — orchestrates reduction for each request.
- **`context/protect.py`** — decides what compaction can never touch.
- **`context/strategies.py`** — the three reduction tactics, cheapest first.
- **`context/ledger.py`** — durable per-session memory of folded history.
- **`context/summarizer.py`** — chunks and folds spans into a rolling summary.
- **`tokens/base.py`** — token counter interface with per-block memoisation.
- **`backends/openai_compat.py`** — translates Anthropic requests/responses/SSE.
- **`store/file.py`** — atomic, thread-safe, file-backed ledger persistence.

Start with [Getting started](./getting-started.md), then read
[Architecture](./architecture.md) and [Request lifecycle](./request-lifecycle.md).
