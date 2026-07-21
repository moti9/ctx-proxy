# Backends and translation

ctxproxy supports two upstream kinds: native Anthropic and OpenAI-compatible
(anything that speaks `/v1/chat/completions`, including LiteLLM and vLLM). The
`native_anthropic` flag on the profile selects the behavior.

## Backend interface

All backends inherit from `backends/base.py`:

```python
class Backend(ABC):
    async def complete(self, request, upstream_model, client_headers) -> dict: ...
    def stream(self, request, upstream_model, client_headers, input_tokens) -> AsyncIterator[bytes]: ...
    async def count_tokens(self, request, upstream_model) -> int: ...
    async def text_completion(self, request, upstream_model) -> str: ...
```

`BackendRegistry` (`backends/__init__.py`) owns one backend instance per config
entry and closes their connection pools on shutdown.

## Native Anthropic path

`backends/anthropic_native.py` forwards requests with minimal change.

- Request payload is dumped from `MessagesRequest` with `model` and `stream`
  injected.
- Headers are forwarded minus hop-by-hop and auth headers; the backend's own key
  is applied.
- `anthropic-beta` is filtered by the backend's `allowed_beta_flags` allowlist.
- `count_tokens` calls the upstream endpoint.
- Streaming yields raw SSE bytes without reparsing.

Because Anthropic itself can compact server-side, the proxy attaches
`context_management.edits` when `supports_server_compaction` is true, then
leaves the rest to the model.

## OpenAI-compatible path

`backends/openai_compat.py` translates in both directions.

### Request translation (`translate/request.py`)

`anthropic_to_openai()` converts:

- Anthropic `messages` → OpenAI `messages`.
- `tool_result` blocks → `role: "tool"` messages.
- `tool_use` blocks → `tool_calls` on the assistant message.
- Anthropic `tools` → OpenAI function-style `tools`.
- `system` blocks → a leading `role: "system"` message.
- `thinking` / `redacted_thinking` blocks are dropped.
- `cache_control`, `context_management`, and beta flags are stripped.

The caller chooses whether to use `max_tokens` or `max_completion_tokens` via
`BackendConfig.openai_max_tokens_field`.

### Response translation (`translate/response.py`)

`openai_to_anthropic()` converts an OpenAI chat completion into an Anthropic
message object:

- `content` → text blocks.
- `reasoning_content` / `reasoning` → text blocks (configurable: `text` or `drop`).
- `tool_calls` → `tool_use` blocks with parsed JSON input.
- `finish_reason` → Anthropic `stop_reason`.
- `usage` → Anthropic usage shape.

If a reasoning model puts the answer only in `reasoning_content` and the policy
is `drop`, the proxy falls back to rendering it as text so the turn is not empty.

### Streaming translation (`translate/sse.py`)

OpenAI streaming emits interleaved deltas. Anthropic expects a strict event
sequence:

```
message_start
content_block_start
content_block_delta
...
content_block_stop
message_delta
message_stop
```

`OpenAIStreamTranslator` tracks open blocks and emits the right events. It
handles:

- Text deltas.
- Tool-call deltas, including deferred name arrival.
- Reasoning-content deltas.
- Empty or usage-only trailers.
- Transport errors after the stream has started.

This is one of the trickiest parts of the codebase because a malformed event
sequence manifests as a frozen or garbled Claude Code UI rather than a clean
error.

## Header handling

`_forwarded_headers()` in `backends/base.py` strips:

- Hop-by-hop headers (`host`, `content-length`, `connection`, etc.).
- Auth headers, which are replaced by the backend's own credentials.
- `content-type`, which is reset to `application/json`.
- Unknown beta flags, filtered against `allowed_beta_flags`.

For OpenAI backends, `anthropic-beta` and `anthropic-version` are also removed.

## Credentials

Each backend resolves its key in order:

1. `backend.api_key` (inline, do not commit).
2. `backend.api_key_env` environment variable.
3. The credential the client sent (forwarded for local/dev setups).

```yaml
backends:
  - name: anthropic
    kind: anthropic
    base_url: https://api.anthropic.com
    api_key_env: ANTHROPIC_API_KEY
```

If no key is configured, the proxy forwards whatever Claude Code provided in
`x-api-key` or `authorization`.
