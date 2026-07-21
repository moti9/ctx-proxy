from __future__ import annotations

from conftest import build_conversation, make_request

from ctxproxy.config import ModelProfile, ReductionPolicy
from ctxproxy.context.budget import compute_budget


def test_budget_subtracts_reserve_and_buffer(openai_profile, policy):
    request = make_request(build_conversation(2), max_tokens=1000)
    budget = compute_budget(openai_profile, policy, request)
    assert budget.usable == 32_000 - 4_000 - 2_000
    assert budget.trigger_at < budget.usable
    assert budget.target_at < budget.trigger_at


def test_output_reserve_grows_to_cover_requested_max_tokens(openai_profile, policy):
    """Input and output must fit in the window together."""
    request = make_request(build_conversation(2), max_tokens=20_000)
    budget = compute_budget(openai_profile, policy, request)
    assert budget.output_reserve == 20_000
    assert budget.usable == 32_000 - 20_000 - 2_000


def test_target_ratio_override_lowers_the_target(openai_profile, policy):
    request = make_request(build_conversation(2))
    normal = compute_budget(openai_profile, policy, request)
    forced = compute_budget(openai_profile, policy, request, target_ratio=0.40)
    assert forced.target_at < normal.target_at


def test_pathological_config_degrades_instead_of_dividing_by_zero(policy):
    profile = ModelProfile(
        match="tiny",
        backend="internal",
        context_window=8_000,
        output_reserve=8_000,
        safety_buffer=4_000,
    )
    budget = compute_budget(profile, policy, make_request(build_conversation(1)))
    assert budget.usable > 0


def test_ratio_validation_rejects_inverted_thresholds():
    import pytest

    with pytest.raises(ValueError, match="trigger_ratio"):
        ReductionPolicy(trigger_ratio=0.5, target_ratio=0.8)


def test_unknown_strategy_is_rejected_at_load_time():
    import pytest

    with pytest.raises(ValueError, match="unknown strategies"):
        ReductionPolicy(strategies=["clear_tool_results", "magic"])
