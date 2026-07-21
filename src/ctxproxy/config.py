"""Configuration: capability profiles, backends, and reduction policy.

Everything the pipeline does is driven off a resolved ``ModelProfile``. The
single most important field is ``native_anthropic`` — it decides whether we pass
Anthropic-specific features through untouched or translate/strip them.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_CONFIG_PATHS = (
    Path("./ctxproxy.yaml"),
    Path("./config.yaml"),
    Path.home() / ".ctxproxy" / "config.yaml",
)

BackendKind = Literal["anthropic", "openai"]


class BackendConfig(BaseModel):
    """An upstream we can dispatch to."""

    model_config = ConfigDict(extra="forbid")

    name: str
    kind: BackendKind
    base_url: str
    api_key_env: str | None = None
    api_key: str | None = None
    timeout_s: float = 600.0
    connect_timeout_s: float = 10.0
    extra_headers: dict[str, str] = Field(default_factory=dict)

    # Beta flags this upstream understands. Anything Claude Code sends that is
    # not listed is stripped — this is what kills "invalid beta flag" errors on
    # non-Anthropic upstreams. ``["*"]`` allows everything through.
    allowed_beta_flags: list[str] = Field(default_factory=lambda: ["*"])

    # OpenAI-compatible upstreams differ on this one; vLLM and older gateways
    # still want `max_tokens`, newer OpenAI wants `max_completion_tokens`.
    openai_max_tokens_field: Literal["max_tokens", "max_completion_tokens"] = "max_tokens"

    # Ask for a usage trailer on streamed responses. Standard, but a few older
    # self-hosted gateways reject the unknown field — turn off if you see a 400
    # mentioning `stream_options`.
    openai_stream_usage: bool = True

    # What to do with `reasoning_content` from reasoning models. Anthropic has
    # no client-visible equivalent on a translated response, so the choice is
    # between rendering it as text (noisy — the whole chain of thought lands in
    # the transcript) and discarding it. "text" is the default because some
    # models leave `content` empty and put the answer in `reasoning_content`;
    # dropping it there would lose the response entirely.
    openai_reasoning_output: Literal["text", "drop"] = "text"

    def resolve_api_key(self) -> str | None:
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            return os.environ.get(self.api_key_env)
        return None

    def allows_beta(self, flag: str) -> bool:
        return "*" in self.allowed_beta_flags or flag in self.allowed_beta_flags


class TokenizerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # "upstream"  -> trust the backend's /v1/messages/count_tokens
    # "tiktoken"  -> local BPE count (requires the `tokenizers` extra)
    # "heuristic" -> chars/ratio; always available, deliberately conservative
    kind: Literal["upstream", "tiktoken", "heuristic"] = "heuristic"
    encoding: str = "cl100k_base"

    # Heuristic only. 3.5 chars/token is intentionally pessimistic vs the ~4.0
    # commonly quoted for English — code and JSON tokenize denser than prose,
    # and undercounting is what causes the hard failure we are trying to avoid.
    chars_per_token: float = 3.5

    # Applied to every local estimate. Guards against tokenizer mismatch between
    # us and the real backend. 1.0 disables.
    safety_multiplier: float = 1.10

    # Flat cost charged per image block when we cannot measure it.
    image_token_cost: int = 1600


class ModelProfile(BaseModel):
    """Per-model capability declaration. The whole design hinges on this."""

    model_config = ConfigDict(extra="forbid")

    # Matched against the incoming request's `model` field. Supports `*` globs.
    match: str
    backend: str

    # The model name to send upstream. Defaults to the incoming name.
    upstream_model: str | None = None

    native_anthropic: bool = True

    # The REAL window, not an assumed 200K. Getting this wrong in either
    # direction is the root cause of the bug this proxy exists to fix.
    context_window: int = 200_000

    # Reserved for the response. Defaults to the request's own max_tokens when
    # that is larger, so a big generation cannot overflow the window.
    output_reserve: int = 32_000

    # Hard ceiling the backend enforces on generation. Claude Code routinely
    # asks for 32K+, and a backend whose real cap is lower rejects the request
    # outright rather than clamping, so we clamp before dispatch. Leave unset
    # when the backend has no separate output cap.
    max_output_tokens: int | None = None

    # Headroom for the summarisation pass itself plus counting error.
    safety_buffer: int = 12_000

    supports_count_tokens: bool = True
    supports_prompt_caching: bool = True

    # When true, we attach `context_management.edits` and let the upstream
    # compact server-side rather than doing it ourselves (strictly better —
    # the summariser is the model itself and nothing is lost client-side).
    supports_server_compaction: bool = False

    tokenizer: TokenizerConfig = Field(default_factory=TokenizerConfig)

    # Which profile writes the rolling summary. Points at another profile's
    # `match` value. Defaults to self.
    summarizer: str | None = None

    def resolve_upstream_model(self, requested: str) -> str:
        return self.upstream_model or requested

    def matches(self, model: str) -> bool:
        return bool(re.fullmatch(_glob_to_regex(self.match), model))


def _glob_to_regex(pattern: str) -> str:
    return ".*".join(re.escape(part) for part in pattern.split("*"))


class ReductionPolicy(BaseModel):
    """When to reduce, how far, and in what order."""

    model_config = ConfigDict(extra="forbid")

    # Reduce once usable context exceeds this fraction of the budget.
    trigger_ratio: float = 0.75

    # Reduce *down to* this fraction. The gap between the two is hysteresis:
    # without it every subsequent turn re-triggers and you pay a summarisation
    # pass per turn.
    target_ratio: float = 0.55

    # Verbatim tool results kept at the tail before clearing kicks in.
    keep_recent_tool_results: int = 6

    # Conversation turns always kept verbatim (the Protected Set tail).
    keep_recent_turns: int = 8

    # Never compact below this many messages — a short conversation that is
    # over budget has a prompt problem, not a history problem.
    min_messages_to_compact: int = 12

    # Ordered. Unknown names are rejected at load time.
    strategies: list[str] = Field(
        default_factory=lambda: ["clear_tool_results", "clear_thinking", "compact"]
    )

    # On upstream context-overflow, force a pass at this ratio and retry once.
    overflow_retry_ratio: float = 0.40
    overflow_retry_enabled: bool = True

    placeholder_text: str = "[tool output cleared to reclaim context]"

    @model_validator(mode="after")
    def _check_ratios(self) -> ReductionPolicy:
        if not 0 < self.target_ratio < self.trigger_ratio <= 1.0:
            raise ValueError(
                f"require 0 < target_ratio ({self.target_ratio}) "
                f"< trigger_ratio ({self.trigger_ratio}) <= 1.0"
            )
        known = {"clear_tool_results", "clear_thinking", "compact"}
        unknown = set(self.strategies) - known
        if unknown:
            raise ValueError(f"unknown strategies: {sorted(unknown)}; known: {sorted(known)}")
        return self


class ServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    host: str = "127.0.0.1"
    port: int = 4000
    log_level: str = "info"
    log_format: Literal["json", "text"] = "text"

    # Where session ledgers live.
    state_dir: Path = Field(default_factory=lambda: Path.home() / ".ctxproxy" / "sessions")

    # Ledgers untouched for this long are pruned at startup.
    session_ttl_hours: int = 72


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    server: ServerConfig = Field(default_factory=ServerConfig)
    policy: ReductionPolicy = Field(default_factory=ReductionPolicy)
    backends: list[BackendConfig]
    profiles: list[ModelProfile]

    @model_validator(mode="after")
    def _check_refs(self) -> Config:
        if not self.profiles:
            raise ValueError("at least one profile is required")
        names = {b.name for b in self.backends}
        for p in self.profiles:
            if p.backend not in names:
                raise ValueError(
                    f"profile {p.match!r} references unknown backend {p.backend!r}; "
                    f"defined backends: {sorted(names)}"
                )
        matches = {p.match for p in self.profiles}
        for p in self.profiles:
            if p.summarizer and p.summarizer not in matches:
                raise ValueError(
                    f"profile {p.match!r} names summarizer {p.summarizer!r}, "
                    f"which is not a defined profile match"
                )
        return self

    def backend(self, name: str) -> BackendConfig:
        for b in self.backends:
            if b.name == name:
                return b
        raise KeyError(name)

    def profile_for(self, model: str) -> ModelProfile:
        """First matching profile wins; the last profile acts as the catch-all."""
        for p in self.profiles:
            if p.matches(model):
                return p
        raise UnknownModelError(model, [p.match for p in self.profiles])

    def profile_by_match(self, match: str) -> ModelProfile:
        for p in self.profiles:
            if p.match == match:
                return p
        raise KeyError(match)

    def summarizer_for(self, profile: ModelProfile) -> ModelProfile:
        if profile.summarizer:
            return self.profile_by_match(profile.summarizer)
        return profile


class UnknownModelError(Exception):
    def __init__(self, model: str, known: list[str]) -> None:
        self.model = model
        self.known = known
        super().__init__(
            f"no profile matches model {model!r}. Configured matches: {known}. "
            f"Add a catch-all profile with match: '*' to accept anything."
        )


def _expand_env(node: Any) -> Any:
    """Expand ${VAR} / ${VAR:-default} inside string values."""
    if isinstance(node, dict):
        return {k: _expand_env(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_expand_env(v) for v in node]
    if isinstance(node, str):
        return re.sub(
            r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}",
            lambda m: os.environ.get(m.group(1), m.group(2) or ""),
            node,
        )
    return node


def resolve_config_path(path: Path | None = None) -> Path:
    """First existing candidate: explicit path, $CTXPROXY_CONFIG, then defaults."""
    if path:
        candidates = [path]
    elif env_path := os.environ.get("CTXPROXY_CONFIG"):
        candidates = [Path(env_path)]
    else:
        candidates = list(DEFAULT_CONFIG_PATHS)

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "no config file found. Looked in: "
        + ", ".join(str(c) for c in candidates)
        + ". Run `ctxproxy init` to generate one."
    )


def load_config(path: Path | None = None) -> Config:
    resolved = resolve_config_path(path)
    raw = yaml.safe_load(resolved.read_text()) or {}
    return Config.model_validate(_expand_env(raw))
