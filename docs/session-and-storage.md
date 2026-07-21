# Session and storage

Claude Code is stateless: it resends the entire conversation every turn. ctxproxy
exploits that by keeping durable per-session state in a ledger. This page
explains session identity and how the ledger is stored.

## Session identity

`session.py` derives a stable key for each conversation.

### Preferred path: headers

Claude Code sends:

- `X-Claude-Code-Session-Id`
- `X-Claude-Code-Agent-Id`
- `X-Claude-Code-Parent-Agent-Id`

Aliases are also checked (`x-session-id`, `anthropic-session-id`) for backwards
compatibility.

If a session id is present, it is used. The agent id is appended for subagents so
that parent and subagent conversations do not share a ledger.

### Fallback: content fingerprint

If no session header is present, the proxy hashes the stable prefix:

- Model name.
- System prompt (first 4000 chars).
- First plain user turn (first 4000 chars).

This is stable for the life of a normal conversation. The fallback exists
because some gateways strip unknown headers and because Claude Code's header
names have changed before.

## Why identity matters

Losing the session key means losing the rolling summary. Without it, every turn
would re-summarize the whole history from scratch.

## The session ledger

`context/ledger.py` defines `SessionLedger`, the durable memory for one session.

### Core fields

| Field | Purpose |
|-------|---------|
| `session_key` | Identifier used as the filename. |
| `task_statement` | First plain user turn, captured verbatim. |
| `summary` | Rolling structured summary of folded history. |
| `fold_start` | Original index where the summary begins. |
| `folded_through` | Original index of first message not yet folded. |
| `fold_signature` | Hash detecting client-side divergence. |
| `files_touched` | Paths extracted from tool calls. |
| `open_questions` | Extracted unresolved items. |
| `decisions` | Extracted decisions. |
| `compaction_count` | How many times compact ran. |
| `total_tokens_saved` | Running savings tally. |
| `events` | Recent reduction events for observability. |

### Fold validation

`fold_is_valid()` recomputes the signature of the folded span. If it does not
match, the fold is invalidated. The summary is kept as prior context, but the
watermark is reset so the next compaction rebuilds from the new history.

`invalidate_fold()` prefixes the summary with `[carried forward after ...]` so
it is clear the conversation diverged.

### Summary rendering

`render_summary_block()` produces the `<conversation-summary>` user message that
replaces folded messages in-flight.

## Storage backend

`store/base.py` defines the abstract `LedgerStore`. The only implementation is
`store/file.py`.

### FileLedgerStore

- Directory: `server.state_dir` (default `~/.ctxproxy/sessions/`).
- Filename: sanitized session key with `.json` extension.
- Writes: temp file in the same directory, then atomic `os.replace`.
- Disk I/O: runs in `asyncio.to_thread` so it never blocks streaming.
- Concurrency: per-session `asyncio.Lock` prevents concurrent turns from
  clobbering the same ledger.
- Pruning: deletes ledgers older than `session_ttl_hours` at startup.

### Adding a different store

Implement `LedgerStore` and swap it in `state.py`:

```python
self.store = MyLedgerStore(config.server.state_dir)
```

## Operational commands

```bash
ctxproxy sessions          # list sessions
ctxproxy inspect <key>     # view summary and stats
ctxproxy reset <key>       # delete one session
ctxproxy reset             # delete all sessions
```

## Privacy note

Session ledgers contain conversation summaries, file paths, and error text. The
default store directory is under `~/.ctxproxy/` and is gitignored by convention.
