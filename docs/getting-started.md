# Getting started

This page covers setup, running tests, and starting the proxy locally.

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (used throughout this project)

## Install

```bash
uv venv
uv pip install -e ".[dev,tokenizers]"
```

The `dev` extra brings `pytest`, `ruff`, and test utilities. The `tokenizers`
extra brings `tiktoken` for better token counting on non-Anthropic backends.

## Run the tests

```bash
pytest -q
```

The suite is small but dense. Key files to know about:

- `tests/test_protect.py` — cut-point safety and the Protected Set.
- `tests/test_manager.py` — the full reduction pipeline.
- `tests/test_translate.py` — Anthropic ↔ OpenAI conversion and SSE state.
- `tests/test_imports.py` — each module imports cleanly in isolation.

## Lint

```bash
ruff check src tests
```

Line length is 100 characters.

## Configure

Copy the example config and edit it:

```bash
ctxproxy init
# or
ctxproxy init -p myproxy.yaml
```

At minimum you need one backend and one profile. See
[Configuration](./configuration.md) for a full reference.

## Start the server

```bash
ctxproxy serve
```

For development with reload on code or config changes:

```bash
ctxproxy serve --reload
```

Point Claude Code at it:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:4000
export ANTHROPIC_AUTH_TOKEN=<whatever your backend expects>
claude
```

## Validate before you connect

Always run `doctor` before pointing Claude Code at the proxy:

```bash
ctxproxy doctor
```

It checks credentials, budget arithmetic, and backend reachability. Most "it
just fails" reports come from a missing API key or a `context_window` that does
not match the model actually being served.

## Inspect state

While the proxy is running (or after it stops), you can inspect session state:

```bash
ctxproxy sessions           # list tracked sessions
ctxproxy inspect <key>      # print a session's rolling summary
ctxproxy reset [<key>]      # clear one or all sessions
```

Session ledgers live under `~/.ctxproxy/sessions/` by default.

## Common first-time mistakes

- **Forgetting `ctxproxy doctor`.** Run it. It catches wrong windows and missing
  credentials before Claude Code does.
- **Wrong `context_window`.** This is the single most common cause of hard
  failures. Set it to the real window of the model served by the backend.
- **Using a wildcard profile first.** Profiles are evaluated in order; the first
  match wins. Put catch-alls last.
- **Enabling `supports_count_tokens` on a LiteLLM/vLLM backend.** Unless the
  gateway actually serves Anthropic's `/v1/messages/count_tokens`, leave this
  `false` and use `heuristic` or `tiktoken` counting.
