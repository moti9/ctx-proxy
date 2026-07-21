# Configuration

ctxproxy is driven by a YAML config. This page is a reference for every knob.

## File location

Resolution order:

1. Path passed with `--config` / `-c`.
2. `$CTXPROXY_CONFIG` environment variable.
3. `./ctxproxy.yaml`
4. `./config.yaml`
5. `~/.ctxproxy/config.yaml`

Run `ctxproxy init` to create a starter `./ctxproxy.yaml` from
`config.example.yaml`.

Environment variables inside the YAML are expanded: `${VAR}` or `${VAR:-default}`.

## Top-level sections

```yaml
server:
  ...
policy:
  ...
backends:
  ...
profiles:
  ...
```

## `server`

| Key | Default | Description |
|-----|---------|-------------|
| `host` | `127.0.0.1` | Bind address. |
| `port` | `4000` | Bind port. |
| `log_level` | `info` | Python log level. |
| `log_format` | `text` | `text` or `json`. |
| `state_dir` | `~/.ctxproxy/sessions` | Ledger storage directory. |
| `session_ttl_hours` | `72` | Delete untouched ledgers after this long. |

## `policy`

| Key | Default | Description |
|-----|---------|-------------|
| `trigger_ratio` | `0.75` | Start reducing when usable context exceeds this. |
| `target_ratio` | `0.55` | Reduce down to this fraction. Gap = hysteresis. |
| `keep_recent_tool_results` | `6` | Recent tool results exempt from clearing. |
| `keep_recent_turns` | `8` | Recent conversation turns kept verbatim. |
| `min_messages_to_compact` | `12` | Never compact shorter conversations. |
| `strategies` | `[clear_tool_results, clear_thinking, compact]` | Order of reduction tactics. |
| `overflow_retry_enabled` | `true` | Retry once on upstream context overflow. |
| `overflow_retry_ratio` | `0.40` | Aggressive target ratio used for the retry. |
| `placeholder_text` | `[tool output cleared to reclaim context]` | Replacement for cleared tool results. |

Constraint: `0 < target_ratio < trigger_ratio <= 1.0`.

## `backends`

Each backend is an upstream destination.

| Key | Required | Description |
|-----|----------|-------------|
| `name` | yes | Identifier referenced by profiles. |
| `kind` | yes | `anthropic` or `openai`. |
| `base_url` | yes | Upstream base URL. |
| `api_key_env` | no | Environment variable holding the key. |
| `api_key` | no | Inline key. Avoid committing this. |
| `timeout_s` | `600` | Request timeout. |
| `connect_timeout_s` | `10` | Connection timeout. |
| `extra_headers` | `{}` | Extra headers sent with every request. |
| `allowed_beta_flags` | `["*"]` | Allowlist for `anthropic-beta`. `"*"` allows all. |
| `openai_max_tokens_field` | `max_tokens` | Use `max_completion_tokens` for newer OpenAI. |
| `openai_stream_usage` | `true` | Request usage trailer in streams. |
| `openai_reasoning_output` | `text` | Render reasoning as text or drop it. |

## `profiles`

Profiles match the incoming `model` field by glob. First match wins.

| Key | Default | Description |
|-----|---------|-------------|
| `match` | required | Glob pattern, e.g. `claude-*` or `*`. |
| `backend` | required | Name of a backend. |
| `upstream_model` | incoming model | Model name sent upstream. |
| `native_anthropic` | `true` | Master switch for Anthropic features. |
| `context_window` | `200000` | Real context window. |
| `output_reserve` | `32000` | Minimum headroom for response generation. |
| `max_output_tokens` | none | Hard backend ceiling for `max_tokens`. |
| `safety_buffer` | `12000` | Headroom for counting error and summarization. |
| `supports_count_tokens` | `true` | Backend serves count_tokens. |
| `supports_prompt_caching` | `true` | Backend supports cache_control blocks. |
| `supports_server_compaction` | `false` | Backend can compact server-side. |
| `tokenizer` | heuristic | See below. |
| `summarizer` | self | Match value of another profile used for summaries. |

## `tokenizer`

| Key | Default | Description |
|-----|---------|-------------|
| `kind` | `heuristic` | `upstream`, `tiktoken`, or `heuristic`. |
| `encoding` | `cl100k_base` | tiktoken encoding name. |
| `chars_per_token` | `3.5` | Heuristic ratio. |
| `safety_multiplier` | `1.10` | Inflation applied to local estimates. |
| `image_token_cost` | `1600` | Flat cost per image block. |

## Example configs

### Native Anthropic only

```yaml
backends:
  - name: anthropic
    kind: anthropic
    base_url: https://api.anthropic.com
    api_key_env: ANTHROPIC_API_KEY
    allowed_beta_flags: ["*"]

profiles:
  - match: "claude-*"
    backend: anthropic
    native_anthropic: true
    context_window: 200000
    supports_server_compaction: true
    tokenizer:
      kind: upstream
```

### Internal model via LiteLLM

```yaml
backends:
  - name: litellm
    kind: openai
    base_url: ${LITELLM_BASE_URL}
    api_key_env: LITELLM_API_KEY
    allowed_beta_flags: []
    openai_stream_usage: true

profiles:
  - match: "our-coder"
    backend: litellm
    upstream_model: internal-coder-v2
    native_anthropic: false
    context_window: 128000
    supports_count_tokens: false
    supports_prompt_caching: false
    tokenizer:
      kind: tiktoken
      encoding: cl100k_base
    summarizer: "claude-*"

  - match: "*"
    backend: litellm
    native_anthropic: false
    context_window: 128000
    supports_count_tokens: false
    supports_prompt_caching: false
    tokenizer:
      kind: heuristic
```

## Validation

`config.py` validates at load time:

- Every profile references a defined backend.
- Every `summarizer` references a defined profile `match`.
- Strategy names are known.
- Ratios satisfy `0 < target_ratio < trigger_ratio <= 1.0`.

`ctxproxy doctor` additionally checks credential presence and backend
reachability.
