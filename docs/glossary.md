# Glossary

Terms used throughout ctxproxy and these docs.

## Backend

An upstream destination: Anthropic API or an OpenAI-compatible gateway such as
LiteLLM or vLLM. Defined in `backends` config and referenced by profiles.

## Budget

The usable input-token headroom derived from `context_window`, `output_reserve`,
and `safety_buffer`. Defined in `context/budget.py`.

## Capability profile

A `profiles` entry declaring what a model can do: window size, backend, tokenizer,
native Anthropic support, summarizer, etc.

## Fold / folding

Replacing a span of original messages with a summary message. The fold watermark
records which original messages were folded so the same splice can be reapplied
deterministically on later turns.

## Fold signature

A hash of the boundary messages and span length of a fold. Used to detect when
the client rewound, branched, or compacted on its own.

## Ledger

Durable per-session state stored in `SessionLedger`. Contains the rolling
summary, fold watermark, task statement, files touched, and statistics.

## Native Anthropic

A backend that speaks the Anthropic Messages API directly. Selected by
`native_anthropic: true` on a profile.

## OpenAI-compatible

A backend that speaks `/v1/chat/completions`. ctxproxy translates Anthropic
requests/responses to this format. Selected by `native_anthropic: false`.

## Protected Set

The set of message indices and tool-result ids that reduction is never allowed
to touch. Computed by `context/protect.py`.

## Reduction

Any operation that lowers the input-token count of a request. Includes clearing
tool results, clearing thinking blocks, and compaction.

## Reduction strategy

One tactic in the reduction pipeline. Configured by `policy.strategies` and
implemented in `context/strategies.py`.

## Rolling summary

The `summary` field of the ledger. It is folded into, not regenerated from
scratch, so each message is summarized at most once.

## Safe cut point

An index at which the conversation may be split without orphaning a
`tool_use`/`tool_result` pair.

## Session

One conversation. Identified by header or content fingerprint. Each session has
its own ledger.

## Session key

The stable identifier for a session, used as the ledger filename.

## Summarizer

The model/backend that writes the rolling summary. May differ from the task
model via the `summarizer` profile field.

## Token counter

Implementation of `tokens/base.py::TokenCounter`. Responsible for estimating
input tokens. May delegate to the upstream, use tiktoken, or fall back to a
heuristic.

## Trigger / target ratio

Fractions of the usable budget at which reduction starts (`trigger_ratio`) and
ends (`target_ratio`). The gap between them is hysteresis.

## Upstream model

The model name actually sent to the backend. May differ from the incoming model
name via `upstream_model` on a profile.
