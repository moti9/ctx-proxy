# Debugging

This page is a field guide for when something goes wrong.

## Before anything else

```bash
ctxproxy doctor
```

It checks:

- Config loads without validation errors.
- Backends are reachable.
- Profiles have positive budgets.
- Credentials are present.

Most failures traced to a missing env var or a `context_window` mismatch.

## Read the logs

Logs are structured. In text mode each reduction decision prints on one line:

```
10:23:14 I passthrough session=s1 tokens=1234 pct=12.3 budget=10000
10:24:05 I reduction triggered session=s1 tokens=46327 pct=514.7 trigger=5400 target=3600
10:24:05 I applied clear_tool_results session=s1 saved=19920 tokens=26407 detail=cleared 16 tool result(s)
10:24:06 I applied compact session=s1 saved=23693 tokens=2714 detail=folded working[1:69] -> original[1:69]
```

Use JSON mode for machine parsing:

```yaml
server:
  log_format: json
```

## Symptom: "Context limit reached" still happens

Possible causes:

1. `context_window` does not match the real model. Fix the profile.
2. Tokenizer undercounts. Raise `safety_multiplier` or switch to `tiktoken`.
3. `trigger_ratio` is too high. Lower it.
4. The upstream itself enforces a smaller effective window than advertised.

Check the logs for the line `still over budget after reduction`.

## Symptom: "invalid beta flag"

The upstream received a beta flag it does not understand. Set the backend's
`allowed_beta_flags` to the exact flags it supports, or `[]` to strip all of
them.

```yaml
backends:
  - name: internal
    kind: openai
    allowed_beta_flags: []
```

## Symptom: 400 on `stream_options`

Older gateways reject `stream_options`. Disable usage trailers:

```yaml
backends:
  - name: internal
    kind: openai
    openai_stream_usage: false
```

## Symptom: compaction fires every turn

`trigger_ratio` and `target_ratio` are too close. Widen the gap:

```yaml
policy:
  trigger_ratio: 0.75
  target_ratio: 0.55
```

## Symptom: context feels lost after compaction

1. Run `ctxproxy inspect <session>` to see what survived.
2. Raise `keep_recent_turns`.
3. Point `summarizer` at a stronger model.
4. Pin critical facts with `<ctx-pin>` in a message.

## Symptom: sessions show `fp_...` keys

The `X-Claude-Code-Session-Id` header is not reaching the proxy. Identity fell
back to a content fingerprint. This is graceful degradation, but check whether
something upstream is stripping headers.

## Inspect a session

```bash
ctxproxy sessions
ctxproxy inspect s1
```

The output includes:

- Stats (turns seen, compactions, tokens saved, summary size).
- The rendered `<conversation-summary>` block.

## Reset state

To start a session from scratch:

```bash
ctxproxy reset s1
```

To clear everything:

```bash
ctxproxy reset
```

## Run a single test

```bash
pytest tests/test_manager.py::test_large_conversation_is_reduced_below_target -q
```

## Enable verbose HTTP logging

The proxy sets `httpx` to WARNING by default. For upstream request/response
details, temporarily raise it:

```yaml
server:
  log_level: debug
```

## Check config loading

```python
from ctxproxy.config import load_config
cfg = load_config("./ctxproxy.yaml")
print(cfg.model_dump())
```

## Useful curl probes

```bash
curl http://localhost:4000/health
curl http://localhost:4000/stats
curl -X POST http://localhost:4000/v1/messages/count_tokens \
  -H "content-type: application/json" \
  -d '{"model":"our-coder","messages":[{"role":"user","content":"hi"}]}'
```
