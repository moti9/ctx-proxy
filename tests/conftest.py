from __future__ import annotations

import pytest

from ctxproxy.config import (
    BackendConfig,
    Config,
    ModelProfile,
    ReductionPolicy,
    ServerConfig,
    TokenizerConfig,
)
from ctxproxy.types_anthropic import Message, MessagesRequest


def user(text: str) -> Message:
    return Message(role="user", content=[{"type": "text", "text": text}])


def assistant(text: str) -> Message:
    return Message(role="assistant", content=[{"type": "text", "text": text}])


def assistant_tool_call(tool_id: str, name: str = "read_file", **inp) -> Message:
    return Message(
        role="assistant",
        content=[{"type": "tool_use", "id": tool_id, "name": name, "input": inp or {}}],
    )


def tool_result(tool_id: str, payload: str, is_error: bool = False) -> Message:
    block = {"type": "tool_result", "tool_use_id": tool_id, "content": payload}
    if is_error:
        block["is_error"] = True
    return Message(role="user", content=[block])


def tool_exchange(index: int, payload_size: int = 4000, is_error: bool = False):
    """One assistant tool call plus its result — the unit that fills a window."""
    tool_id = f"toolu_{index}"
    return [
        assistant_tool_call(tool_id, name="read_file", path=f"src/mod_{index}.py"),
        tool_result(tool_id, f"file {index} contents " + "x" * payload_size, is_error),
    ]


def build_conversation(exchanges: int = 20, payload_size: int = 4000) -> list[Message]:
    messages = [user("Refactor the payment module to support idempotency keys.")]
    for i in range(exchanges):
        messages.extend(tool_exchange(i, payload_size))
        messages.append(assistant(f"Analysed module {i}."))
        messages.append(user(f"Continue with step {i + 1}."))
    return messages


def build_text_heavy_conversation(turns: int = 30, size: int = 3000) -> list[Message]:
    """A conversation that tool-result clearing cannot rescue.

    All the weight is in plain prose turns, so the cheap strategies are no-ops
    and the pipeline is forced all the way down to compaction.
    """
    messages = [user("Refactor the payment module to support idempotency keys.")]
    for i in range(turns):
        messages.append(assistant(f"Step {i} analysis: " + "reasoning " * (size // 10)))
        messages.append(user(f"Continue with step {i + 1}: " + "context " * (size // 8)))
    return messages


def make_request(messages: list[Message], model: str = "our-coder", **kw) -> MessagesRequest:
    return MessagesRequest(
        model=model,
        messages=messages,
        max_tokens=kw.pop("max_tokens", 4096),
        **kw,
    )


@pytest.fixture
def policy() -> ReductionPolicy:
    return ReductionPolicy()


@pytest.fixture
def native_profile() -> ModelProfile:
    return ModelProfile(
        match="claude-*",
        backend="anthropic",
        native_anthropic=True,
        context_window=200_000,
        supports_server_compaction=True,
        tokenizer=TokenizerConfig(kind="heuristic"),
    )


@pytest.fixture
def openai_profile() -> ModelProfile:
    return ModelProfile(
        match="our-coder",
        backend="internal",
        upstream_model="internal-coder-v2",
        native_anthropic=False,
        context_window=32_000,
        output_reserve=4_000,
        safety_buffer=2_000,
        supports_count_tokens=False,
        supports_prompt_caching=False,
        tokenizer=TokenizerConfig(kind="heuristic"),
    )


@pytest.fixture
def config(tmp_path, native_profile, openai_profile) -> Config:
    return Config(
        server=ServerConfig(state_dir=tmp_path / "sessions"),
        policy=ReductionPolicy(),
        backends=[
            BackendConfig(
                name="anthropic", kind="anthropic", base_url="https://api.anthropic.test"
            ),
            BackendConfig(
                name="internal", kind="openai", base_url="http://internal.test"
            ),
        ],
        profiles=[native_profile, openai_profile],
    )
