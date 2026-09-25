"""Tests for :mod:`protocore.runtime.context.budgets`."""
from __future__ import annotations

from protocore.contracts.runtime_constants import LoopConstants
from protocore.runtime.context.budgets import TokenBudgets, derive_budgets


def test_derive_budgets_pure() -> None:
    rc = LoopConstants(model_context_window=49_152)
    budgets1 = derive_budgets(rc)
    budgets2 = derive_budgets(rc)
    assert budgets1 == budgets2  # purity: same input → same output


def test_derive_budgets_scales_with_window() -> None:
    small = derive_budgets(LoopConstants(model_context_window=8_192))
    large = derive_budgets(LoopConstants(model_context_window=200_000))
    assert large.max_context > small.max_context
    assert large.history_budget_tokens > small.history_budget_tokens
    assert large.compaction_trigger_tokens > small.compaction_trigger_tokens


def test_fixed_overhead_leaves_history_room() -> None:
    rc = LoopConstants(model_context_window=49_152)
    budgets = derive_budgets(rc)
    assert budgets.history_budget_tokens > 0
    assert budgets.history_budget_tokens + budgets.fixed_overhead_tokens == budgets.max_context


def test_compaction_trigger_below_max() -> None:
    rc = LoopConstants(model_context_window=49_152)
    budgets = derive_budgets(rc)
    assert budgets.compaction_trigger_tokens < budgets.max_context


def test_tool_result_threshold_smaller_than_trigger() -> None:
    rc = LoopConstants(model_context_window=49_152)
    budgets = derive_budgets(rc)
    assert budgets.tool_result_truncation_threshold < budgets.compaction_trigger_tokens


def test_token_budgets_is_frozen_dataclass() -> None:
    rc = LoopConstants()
    budgets = derive_budgets(rc)
    assert isinstance(budgets, TokenBudgets)
    # FrozenInstanceError raised on mutation
    import dataclasses

    import pytest as _pytest

    with _pytest.raises(dataclasses.FrozenInstanceError):
        budgets.max_context = 0  # type: ignore[misc]


def test_the_trigger_sits_below_the_prompt_size_the_provider_still_accepts() -> None:
    """A server that reserves the output budget inside the window rejects any
    prompt above ``window - max output``. With the stock ratios on a 65 536
    window the ratio alone would put the trigger ABOVE that cliff, so proactive
    compaction could never fire before the rejection."""
    rc = LoopConstants(model_context_window=65_536)
    budgets = derive_budgets(rc)

    output_cap = int(rc.model_context_window * rc.llm_output_max_tokens_ratio)
    accepted_prompt_ceiling = (
        rc.model_context_window - output_cap - rc.request_context_safety_tokens
    )
    assert budgets.compaction_trigger_tokens < accepted_prompt_ceiling
    # And a whole turn below it, not merely one token.
    assert (
        accepted_prompt_ceiling - budgets.compaction_trigger_tokens
        >= int(rc.model_context_window * rc.compaction_trigger_turn_headroom_ratio)
    )
    # The ratio on its own would have been unreachable.
    assert int(rc.model_context_window * rc.compaction_trigger_ratio) > accepted_prompt_ceiling


def test_the_ratio_still_binds_when_it_is_the_lower_of_the_two() -> None:
    """The acceptable-prompt ceiling is a cap, not a replacement: a small
    trigger ratio stays in force."""
    rc = LoopConstants(model_context_window=65_536, compaction_trigger_ratio=0.2)
    budgets = derive_budgets(rc)
    assert budgets.compaction_trigger_tokens == int(65_536 * 0.2)


def test_the_emergency_cliff_stays_strictly_above_the_effective_trigger() -> None:
    for window in (2_048, 8_192, 49_152, 65_536, 200_000):
        budgets = derive_budgets(LoopConstants(model_context_window=window))
        assert budgets.compaction_emergency_tokens > budgets.compaction_trigger_tokens
        assert budgets.compaction_trigger_tokens > 0


def test_an_output_reserve_that_leaves_no_headroom_is_refused() -> None:
    import pytest as _pytest

    with _pytest.raises(ValueError, match="compaction_trigger_turn_headroom_ratio"):
        LoopConstants(
            llm_output_max_tokens_ratio=0.9,
            compaction_trigger_turn_headroom_ratio=0.15,
        )


def test_a_provider_that_does_not_reserve_output_keeps_that_share_of_the_window() -> None:
    """The output-reserve deduction answers a serving stack that counts the
    requested output against the same window as the prompt. A provider that
    sizes its input window independently gives that share back."""
    window = 65_536
    reserving = derive_budgets(LoopConstants(model_context_window=window))
    independent = derive_budgets(
        LoopConstants(
            model_context_window=window,
            provider_reserves_output_in_context_window=False,
        )
    )
    rc = LoopConstants(model_context_window=window)
    assert independent.compaction_trigger_tokens > reserving.compaction_trigger_tokens
    # Without the reserve the ceiling rises above the configured ratio, which
    # then binds — the operator gets back the trigger the ratio asks for.
    assert independent.compaction_trigger_tokens == int(window * 0.8)
    assert (
        window
        - rc.request_context_safety_tokens
        - int(window * rc.compaction_trigger_turn_headroom_ratio)
    ) > int(window * 0.8)


def test_the_reserve_is_deducted_by_default_so_a_reserving_server_is_covered() -> None:
    rc = LoopConstants(model_context_window=65_536)
    assert rc.provider_reserves_output_in_context_window is True
    budgets = derive_budgets(rc)
    assert budgets.compaction_trigger_tokens == (
        65_536
        - int(65_536 * rc.llm_output_max_tokens_ratio)
        - rc.request_context_safety_tokens
        - int(65_536 * rc.compaction_trigger_turn_headroom_ratio)
    )
