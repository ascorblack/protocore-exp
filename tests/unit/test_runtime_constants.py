"""Tests for :class:`LoopConstants` and formula derivation."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from protocore.contracts.runtime_constants import LoopConstants


def test_frozen_semantics() -> None:
    rc = LoopConstants()
    with pytest.raises(ValidationError):
        rc.compaction_trigger_ratio = 0.5  # type: ignore[misc]


def test_emergency_must_exceed_trigger() -> None:
    with pytest.raises(ValidationError):
        LoopConstants(
            compaction_trigger_ratio=0.95,
            compaction_emergency_ratio=0.9,
        )


def test_prompt_budgets_must_not_consume_window() -> None:
    with pytest.raises(ValidationError):
        LoopConstants(
            system_prompt_max_ratio=0.6,
            skill_index_budget_ratio=0.5,
        )


def test_compaction_trigger_derivation_lives_in_budgets() -> None:
    """DEAD-SURFACE 1 — ``compaction_thresholds`` was deleted; the compaction
    trigger is now derived solely by :func:`derive_budgets`. The trigger scales
    with ``model_context_window`` (the canonical input) just as before."""
    from protocore.runtime.context.budgets import derive_budgets

    small = derive_budgets(LoopConstants(model_context_window=32_768))
    large = derive_budgets(LoopConstants(model_context_window=200_000))
    # Under stock ratios the acceptable-prompt ceiling binds before the ratio
    # does, so the trigger is that ceiling rather than 0.8 of the window.
    rc_small = LoopConstants(model_context_window=32_768)
    assert small.compaction_trigger_tokens == (
        32_768
        - int(32_768 * rc_small.llm_output_max_tokens_ratio)
        - rc_small.request_context_safety_tokens
        - int(32_768 * rc_small.compaction_trigger_turn_headroom_ratio)
    )
    assert small.compaction_trigger_tokens < int(32_768 * 0.8)
    assert large.compaction_trigger_tokens > small.compaction_trigger_tokens


def test_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        LoopConstants(unknown_field=1)  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "removed_name, value",
    [
        ("terminal_answer_observed_ref_ledger_enabled", True),
        ("terminal_tool_malformed_args_recovery_enabled", True),
        ("terminal_tool_malformed_args_recovery_max_attempts", 1),
    ],
)
def test_removed_runtime_constants_are_rejected(
    removed_name: str, value: object
) -> None:
    with pytest.raises(ValidationError):
        LoopConstants.model_validate({removed_name: value})


def test_removed_runtime_constants_are_not_advertised() -> None:
    properties = LoopConstants.model_json_schema()["properties"]
    assert "terminal_answer_observed_ref_ledger_enabled" not in properties
    assert "terminal_tool_malformed_args_recovery_enabled" not in properties
    assert "terminal_tool_malformed_args_recovery_max_attempts" not in properties


def test_negative_window_rejected() -> None:
    with pytest.raises(ValidationError):
        LoopConstants(model_context_window=0)


def test_ratio_outside_unit_interval_rejected() -> None:
    with pytest.raises(ValidationError):
        LoopConstants(compaction_trigger_ratio=1.5)
    with pytest.raises(ValidationError):
        LoopConstants(compaction_trigger_ratio=0.0)


def test_tool_result_truncation_ratio_default_is_0_10() -> None:
    """Default bumped 0.05 → 0.10 for long_context."""
    rc = LoopConstants()
    assert rc.tool_result_truncation_ratio == 0.10


def test_tool_result_truncation_ratio_boundary_validation() -> None:
    """Sanity bounds: > 0.0 AND <= 0.5."""
    # Zero / negative rejected
    with pytest.raises(ValidationError):
        LoopConstants(tool_result_truncation_ratio=0.0)
    with pytest.raises(ValidationError):
        LoopConstants(tool_result_truncation_ratio=-0.01)
    # Above 0.5 rejected
    with pytest.raises(ValidationError):
        LoopConstants(tool_result_truncation_ratio=0.51)
    with pytest.raises(ValidationError):
        LoopConstants(tool_result_truncation_ratio=1.0)
    # Exact bound 0.5 accepted
    rc = LoopConstants(tool_result_truncation_ratio=0.5)
    assert rc.tool_result_truncation_ratio == 0.5
    # Just-above-zero accepted
    rc = LoopConstants(tool_result_truncation_ratio=0.01)
    assert rc.tool_result_truncation_ratio == 0.01


def test_finalization_gate_enabled_default_false() -> None:
    """Default FALSE (flipped 2026-06-05, acceptance BC1/BC3): the end-of-run
    deliverable-verification gate AND its model-facing ``<finalization_contract>``
    JSON TEMPLATE block are retired UNIVERSALLY. With the gate off the executor
    passes no contract block to the leader prompt and ``verify_declared_
    deliverables`` short-circuits to ``None`` — terminal completed/partial
    classification falls back to the A5 tool-errors heuristic, which works
    normally without any contract. A tenant that wants the gate back flips it
    True via the Constants page."""
    rc = LoopConstants()
    assert rc.finalization_gate_enabled is False


def test_finalization_gate_enabled_is_overridable() -> None:
    rc = LoopConstants(finalization_gate_enabled=True)
    assert rc.finalization_gate_enabled is True


# ---------------------------------------------------------------------------
# Tool-dispatch consecutive same-error cap
# ---------------------------------------------------------------------------


def test_tool_dispatch_consecutive_error_cap_default_4() -> None:
    """Default allows up to 3 retries (4th capped).

    The leader can retry an identical failed tool call up to 200 times
    in pathological runs. Default 4 lets the 1st-3rd identical errors
    through unchanged; the 4th surfaces as
    ``DispatchErrorKind.consecutive_error_cap`` with guidance.
    """
    rc = LoopConstants()
    assert rc.tool_dispatch_consecutive_error_cap == 4


def test_tool_dispatch_consecutive_error_cap_overridable() -> None:
    """Operator override path stays open via the constructor / dashboard."""
    rc = LoopConstants(tool_dispatch_consecutive_error_cap=8)
    assert rc.tool_dispatch_consecutive_error_cap == 8


def test_tool_dispatch_consecutive_error_cap_rejects_zero_and_negative() -> None:
    """``ge=2`` rejects ``0`` and negative values too."""
    with pytest.raises(ValidationError):
        LoopConstants(tool_dispatch_consecutive_error_cap=0)
    with pytest.raises(ValidationError):
        LoopConstants(tool_dispatch_consecutive_error_cap=-1)


# ---------------------------------------------------------------------------
# DAG tool-precondition mechanism
# ---------------------------------------------------------------------------


def test_tool_preconditions_enabled_default_false() -> None:
    """Default is OFF: the only observed live firing was a
 false-positive. Mechanism stays wired but inert until the
 ``file_path`` vs ``path`` alias resolution lands in
 ``resolve_precondition``."""
    rc = LoopConstants()
    assert rc.tool_preconditions_enabled is False


def test_tool_preconditions_enabled_is_overridable() -> None:
    """Operator kill-switch path stays open via the constructor / dashboard."""
    rc = LoopConstants(tool_preconditions_enabled=True)
    assert rc.tool_preconditions_enabled is True
    rc2 = LoopConstants(tool_preconditions_enabled=False)
    assert rc2.tool_preconditions_enabled is False


# ---------------------------------------------------------------------------
# Session-memory fold — write budget vs carry cap
# ---------------------------------------------------------------------------


#: Worst ratio measured between the provider's own token count and the
#: character-ratio estimate the carry cap is expressed in. Bare sha256 digests
#: are the worst case at ~3.4x, base64 ~3.0x, mixed UUID/URL/token content
#: ~2.7x; Latin prose measures ~1.6x and Cyrillic prose ~1.3x. The WORST figure
#: governs, because the material the summary prompt orders copied verbatim is
#: exactly the material that measures worst, and a session's content class is
#: not known in advance.
_WORST_ESTIMATOR_UNDERCOUNT = 3.4

#: Share of the output budget that must remain AFTER re-emitting a capped
#: summary, so a fold has room to add the new run's facts rather than only
#: reproducing what it was handed.
_MIN_DELTA_ALLOWANCE = 0.20


def test_kb_unknown_name_is_rejected() -> None:
    """``extra='forbid'`` — a stale producer using a dropped name fails loudly."""
    with pytest.raises(ValidationError):
        LoopConstants.model_validate({"kb_max_wiki_bytes": 1})
