"""Formula-derived token budgets per

Pure function. Same :class:`LoopConstants` snapshot yields the same
:class:`TokenBudgets` — cross-pod deterministic. No module-level cache.

All budgets are **derived** from a canonical input
(:attr:`LoopConstants.model_context_window`) and per-section ratios.
The dashboard surfaces ratios as canonical inputs and the derived values
as read-only computed fields.
"""
from __future__ import annotations

from dataclasses import dataclass

from protocore.contracts.runtime_constants import LoopConstants


@dataclass(frozen=True, slots=True)
class TokenBudgets:
    """Per-layer derived token budgets for one turn.

 """

    max_context: int
    """Hard upper bound (provider's context window)."""

    compaction_trigger_tokens: int
    """When current_tokens > this, compaction runs before the LLM call.

    This is the EFFECTIVE trigger, not ``model_context_window *
    compaction_trigger_ratio``. The ratio is one of two bounds; the other is
    the largest prompt the provider will still accept — the window less the
    output reserve, less the safety margin a fitted request keeps, less one
    turn's headroom — and the trigger is the lower of the two. A consumer that
    sizes anything from this value (a recovery seed budget, say) is reading the
    prompt size compaction actually aims at, which on a window whose output
    reserve is large is well below the ratio alone.
    """

    compaction_emergency_tokens: int
    """Emergency cliff: when current_tokens > this, a proactive
    ``force_compaction`` runs before the LLM call (both tiers, unconditional)
    instead of waiting for the provider to raise a context-window error.
    Derived from ``model_context_window * compaction_emergency_ratio`` and
    held strictly above ``compaction_trigger_tokens``: the ratios alone order
    them (the RC validator enforces ``compaction_trigger_ratio <
    compaction_emergency_ratio``), and integer truncation on a small window is
    the one way they could still meet."""

    tool_result_truncation_threshold: int
    """Tool results larger than this are blobbed (Tier 1)."""

    system_prompt_max_tokens: int
    skill_index_budget_tokens: int
    loaded_skills_budget_tokens: int
    tool_definitions_budget_tokens: int
    user_context_budget_tokens: int

    history_budget_tokens: int
    """Remainder available for conversation history after fixed overhead."""

    @property
    def fixed_overhead_tokens(self) -> int:
        """Sum of all fixed-overhead layers (system + skills + tools + user_ctx)."""
        return (
            self.system_prompt_max_tokens
            + self.skill_index_budget_tokens
            + self.loaded_skills_budget_tokens
            + self.tool_definitions_budget_tokens
            + self.user_context_budget_tokens
        )


def derive_budgets(rc: LoopConstants) -> TokenBudgets:
    """Compute :class:`TokenBudgets` from an RC snapshot.

    Pure function. The dashboard's RC editor surfaces ratios; this function
    is the single source of truth for derived values across every pod.
    """
    max_context = rc.model_context_window

    # The trigger has to be a prompt size the provider would still accept.
    # Where the serving stack counts the requested output against the same
    # window as the prompt — ``provider_reserves_output_in_context_window`` —
    # every request whose prompt exceeds ``window - max output`` is refused,
    # so that, and not the window, is the ceiling the trigger sits under. A
    # provider that sizes its input window independently of the requested
    # output gives that share back. Either way the request the runtime builds
    # keeps ``request_context_safety_tokens`` unused for provider-side framing,
    # and the check runs BEFORE a turn, so a turn's worth of headroom has to
    # remain or the very turn the trigger was meant to precede is the one that
    # overflows. A trigger above the ceiling is unreachable: the provider
    # rejects the request before the history ever grows into it, and proactive
    # compaction never runs at all.
    output_reserve = (
        int(max_context * rc.llm_output_max_tokens_ratio)
        if rc.provider_reserves_output_in_context_window
        else 0
    )
    accept_ceiling = (
        max_context
        - output_reserve
        - rc.request_context_safety_tokens
        - int(max_context * rc.compaction_trigger_turn_headroom_ratio)
    )
    compaction_trigger = max(1, min(int(max_context * rc.compaction_trigger_ratio), accept_ceiling))
    compaction_emergency = max(
        int(max_context * rc.compaction_emergency_ratio), compaction_trigger + 1
    )
    tool_result_threshold = int(max_context * rc.tool_result_truncation_ratio)
    system_prompt_max = int(max_context * rc.system_prompt_max_ratio)
    skill_index = int(max_context * rc.skill_index_budget_ratio)
    loaded_skills = int(max_context * rc.loaded_skills_ratio)
    tool_definitions = int(max_context * rc.tool_definitions_ratio)
    user_context = int(max_context * rc.user_context_ratio)

    fixed_overhead = (
        system_prompt_max
        + skill_index
        + loaded_skills
        + tool_definitions
        + user_context
    )
    history_budget = max_context - fixed_overhead

    return TokenBudgets(
        max_context=max_context,
        compaction_trigger_tokens=compaction_trigger,
        compaction_emergency_tokens=compaction_emergency,
        tool_result_truncation_threshold=tool_result_threshold,
        system_prompt_max_tokens=system_prompt_max,
        skill_index_budget_tokens=skill_index,
        loaded_skills_budget_tokens=loaded_skills,
        tool_definitions_budget_tokens=tool_definitions,
        user_context_budget_tokens=user_context,
        history_budget_tokens=history_budget,
    )


__all__ = ["TokenBudgets", "derive_budgets"]
