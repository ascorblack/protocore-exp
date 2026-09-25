"""``ContextManager`` — assembles the 8-layer context bundle + drives compaction.

Pure-ish: every call rebuilds budgets from the latest RC snapshot. No
module-level cache — horizontal scaling rule (no per-pod state for
correctness-affecting decisions).
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass

from protocore.contracts.blob import IBlobStore
from protocore.contracts.llm import ILLMProvider, LLMObservabilityContext
from protocore.contracts.prompts import IPromptTemplateProvider
from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.skills import SkillBundle
from protocore.contracts.types import Message, ToolDefinition
from protocore.logging_utils import get_logger
from protocore.runtime.context.budgets import TokenBudgets, derive_budgets
from protocore.runtime.context.compaction import (
    CompactionAttempt,
    CompactionExhaustedError,
    CompactionState,
    RequestRecorder,
    Tier1Result,
    Tier2Result,
    Tier3Result,
    TokenEstimator,
    estimate_history_tokens,
    run_tier1_truncation,
    run_tier2_summarisation,
    run_tier3_fold,
    tier1_has_work,
    tier2_has_work,
    tier3_has_work,
)
from protocore.runtime.token_counting import LanguageProfile, detect_profile

_logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ContextBundle:
    """Output of :meth:`ContextManager.build_context`.

    The bundle is what the loop forwards to
    :meth:`ILLMProvider.stream_with_tools`. ``system_prompt_sections`` is
    the assembled prefix; ``messages`` is the (possibly compacted) history.
    """

    system_prompt_sections: tuple[str, ...]
    tools: tuple[ToolDefinition, ...]
    messages: tuple[Message, ...]
    active_language: str
    budgets: TokenBudgets


def detect_active_language(latest_message: Message | None) -> str:
    """Return ``"ru"`` or ``"en"`` based on Cyrillic-ratio heuristic.

    Mirrors v1 detection rule (Cyrillic ratio > 30% → RU). Defaults to
    ``"en"`` for empty / undetectable content.
    """
    if latest_message is None:
        return "en"
    text = latest_message.text
    if not text:
        return "en"
    profile = detect_profile(text)
    if profile in (LanguageProfile.cyrillic_prose, LanguageProfile.cyrillic_in_json_escape):
        return "ru"
    return "en"


class ContextManager:
    """Builds per-turn :class:`ContextBundle` and drives compaction.

 The manager is stateless across calls
 with respect to context assembly — it reads :class:`LoopConstants`
 fresh and operates on the history list provided.

 Compaction state IS persisted on the :class:`QueryEngine` (retry
 counter, summarised-turn IDs) — passed in as :class:`CompactionState`.

 pin LRU state IS persisted on the
 manager so :meth:`pin_tool` can cap the per-run pin list at
 :attr:`LoopConstants.pinned_tool_max_count` and evict the
 least-recently-pinned entry on overflow. Pin state lives on the
 manager because the QueryEngine builds at-most-one ContextManager
 per run, so the pin list is naturally per-run scoped without
 needing extra plumbing on :class:`CompactionState`.
 """

    def __init__(
        self,
        *,
        rc: LoopConstants,
        blob_store: IBlobStore,
        compaction_llm: ILLMProvider,
        prompts: IPromptTemplateProvider | None = None,
    ) -> None:
        self._rc = rc
        self._blob_store = blob_store
        self._compaction_llm = compaction_llm
        # The provider that renders the summariser instructions. ``None`` is a
        # caller that wired none, and the tiers fall back to the templates
        # bundled with the package — the same fallback the engine makes, for
        # the same reason: every call site renders unconditionally instead of
        # carrying a branch for the host that configured nothing.
        self._prompts = prompts
        # LRU pin tracking. OrderedDict
        # gives us O(1) ``move_to_end`` for re-pins and FIFO eviction
        # on overflow. Stored as ``dict[str, None]`` because we only
        # care about names + their relative order; the actual tool
        # descriptors live in the :class:`ToolRegistry`.
        self._pinned_tools: OrderedDict[str, None] = OrderedDict()
        # One estimator per manager, and a manager is built once per run: two
        # runs sharing a process never consult each other's remembered
        # estimates, whatever object identity would have allowed.
        self._token_estimator = TokenEstimator()
        """This manager's own estimate cache — see :attr:`token_estimator`."""

    @property
    def token_estimator(self) -> TokenEstimator:
        """The estimate cache belonging to this run.

        One per manager, so a run's remembered estimates are its own. Callers
        that size THIS run's history reach for it rather than the module-level
        functions, whose cache is shared by every run in the process and whose
        capacity a long history therefore competes for.
        """
        return self._token_estimator

    @property
    def rc(self) -> LoopConstants:
        return self._rc

    def update_rc(self, rc: LoopConstants) -> None:
        """Take a refreshed snapshot: the loop recalibrated the token estimate mid-run."""
        self._rc = rc

    # ------------------------------------------------------------------
    # pin LRU
    # ------------------------------------------------------------------

    def pin_tool(self, name: str) -> str | None:
        """Pin a tool by name with LRU + cap enforcement.

 The ``ToolVisibilityPolicy.pinned`` set can grow unbounded without a cap —
 a latent KV-cache bloat risk (every pinned tool stays in the prompt prefix
 regardless of retrieval scoring). The cap is taken fresh from
 :attr:`LoopConstants.pinned_tool_max_count` so an operator
 tightening the value via the dashboard takes effect on the
 next pin call without a redeploy.

 Behaviour:

 * Re-pinning an already-pinned name moves it to the
 most-recently-used position (no eviction).
 * Pinning a new name when the list is already at the cap
 evicts the oldest entry and returns its name so the caller
 can update the visibility policy that consumes the list.
 * Pinning when the cap is at or below zero is a no-op
 (defensive — the RC field is ``gt=0`` so this only fires
 for adversarial test setups).
 """

        if not name:
            return None
        cap = max(0, int(self._rc.pinned_tool_max_count))
        if cap <= 0:
            return None
        if name in self._pinned_tools:
            self._pinned_tools.move_to_end(name)
            return None
        evicted: str | None = None
        if len(self._pinned_tools) >= cap:
            # popitem(last=False) returns the OLDEST insertion — that's
            # the LRU because every re-pin moves the entry to the end.
            evicted_name, _ = self._pinned_tools.popitem(last=False)
            evicted = evicted_name
        self._pinned_tools[name] = None
        return evicted

    def pinned_tool_names(self) -> tuple[str, ...]:
        """Return the current LRU-ordered pinned tool names.

 Oldest-first (insertion order with
 re-pin promoting to the end). Callers building
 :class:`ToolVisibilityPolicy` pass ``frozenset(...)`` because
 the policy field is a set; the order is preserved here so the
 caller MAY render or log the recency.
 """

        return tuple(self._pinned_tools.keys())

    def build_context(
        self,
        *,
        history: Sequence[Message],
        tools: Sequence[ToolDefinition],
        skills_loaded: Sequence[SkillBundle] = (),
        system_prompt_sections: Sequence[str] = (),
        skill_index_block: str = "",
    ) -> ContextBundle:
        """Assemble the 8-layer context bundle.

        Filtering / retrieval of tools is the caller's concern (the loop
        delegates that to :class:`IToolRegistry.compute_effective_surface`).

        ``skill_index_block`` is the pre-rendered ``<system-reminder>`` skill
        catalog produced by
        :func:`~protocore.runtime.skill_index.render_skills_catalog`. Injected
        at Layer 2 (skill catalog sits between the system prompt proper and
        Layer 3 loaded skill bodies).
        """
        budgets = derive_budgets(self._rc)
        latest = history[-1] if history else None
        language = detect_active_language(latest)

        # Render skill bodies as system-prompt prepends (Layer 3). Each body
        # is capped at ``loaded_skills_budget_tokens // max_skills_per_run``
        # to keep the prefix stable.
        prepended_skills: list[str] = []
        if skills_loaded:
            max_per_skill = max(
                1,
                budgets.loaded_skills_budget_tokens
                // max(1, self._rc.max_skills_per_run),
            )
            for bundle in list(skills_loaded)[: self._rc.max_skills_per_run]:
                body = bundle.body or ""
                # Token-cap each loaded skill body — soft truncation by char
                # count using the RC-tunable chars-per-token heuristic
                # (Latin-prose baseline).
                budget_chars = max_per_skill * self._rc.skill_body_chars_per_token
                if len(body) > budget_chars:
                    body = body[:budget_chars] + "…"
                prepended_skills.append(
                    f"<loaded-skill name=\"{bundle.manifest.name}\">\n{body}\n</loaded-skill>"
                )

        sections: list[str] = list(system_prompt_sections)
        if skill_index_block:
            sections.append(skill_index_block)
        sections.extend(prepended_skills)
        assembled_sections = tuple(sections)

        return ContextBundle(
            system_prompt_sections=assembled_sections,
            tools=tuple(tools),
            messages=tuple(history),
            active_language=language,
            budgets=budgets,
        )

    async def _fold(
        self,
        history: list[Message],
        compaction_state: CompactionState,
        model_name: str,
        observability: LLMObservabilityContext | None,
        protect_tail_from_index: int | None,
        record_request: RequestRecorder | None,
        *,
        keep_recent_turns: int | None = None,
        compact_seeded_history: bool = False,
    ) -> Tier3Result | None:
        """Tier 3, after Tier 2 in both cascades.

        A failure here never aborts the pass: Tier 1 and Tier 2 have already
        freed what they could, and the fold is the part that makes a long
        session's window shrink rather than the part that makes a request fit.
        Returning ``None`` — for a run with no compaction LLM, or for a fold
        the operator switched off — is how the caller tells "did not run" from
        "ran and folded nothing". A fold that raised did run, and reports one
        attempted span so the pass is not mistaken for one with nothing to do.
        """
        if self._compaction_llm is None or not self._rc.compaction_fold_enabled:
            return None
        try:
            return await run_tier3_fold(
                history=history,
                compaction_llm=self._compaction_llm,
                state=compaction_state,
                rc=self._rc,
                model_name=model_name,
                observability=observability,
                protect_tail_from_index=protect_tail_from_index,
                record_request=record_request,
                prompts=self._prompts,
                keep_recent_turns=keep_recent_turns,
                compact_seeded_history=compact_seeded_history,
            )
        except Exception as exc:
            _logger.warning("tier3 fold failed; skipping (err=%s)", exc)
            return Tier3Result(
                spans_folded=0, messages_folded=0, tokens_freed=0, spans_attempted=1
            )

    def _settle_pass(
        self,
        *,
        compaction_state: CompactionState,
        attempt: CompactionAttempt,
        reactive: bool,
        error: Exception | None,
        label: str,
    ) -> None:
        """Charge one finished pass to its retry budget — at most once.

        A pass ends in one of three ways:

        * **progress** — it freed tokens, or rewrote, summarised or folded
          anything. Both budgets are cleared, whichever profile made it: the
          history the next pass of either kind faces is a different one.
        * **nothing to do** — no tier raised and no tier found anything it was
          allowed to touch under this pass's profile: Tier 1 modified nothing,
          Tier 2 sent no unit and Tier 3 no span. A proactive pass over a
          history of seeded turns lands here by design. It spends nothing:
          there was no attempt to fail.
        * **failed** — a tier raised, or a summariser call was made and nothing
          came of it. One increment, however many tiers failed.

        A reactive pass (after a provider rejection) spends
        :attr:`CompactionState.reactive_retry_count`; every other pass spends
        :attr:`CompactionState.retry_count`. Both are bounded by
        :attr:`LoopConstants.compaction_failed_max_retries`, and breaching the
        bound raises :class:`CompactionExhaustedError` chained to the tier
        exception when there was one.
        """
        tier1, tier2, tier3 = attempt.tier1, attempt.tier2, attempt.tier3
        progress = (
            attempt.tokens_after < attempt.tokens_before
            or (tier1 is not None and tier1.messages_modified > 0)
            or (tier2 is not None and tier2.turns_summarised > 0)
            or (tier3 is not None and tier3.spans_folded > 0)
        )
        if progress:
            compaction_state.reset_retries()
            return
        tried = (
            error is not None
            or (tier2 is not None and tier2.units_attempted > 0)
            or (tier3 is not None and tier3.spans_attempted > 0)
        )
        if not tried:
            return
        if reactive:
            compaction_state.reactive_retry_count += 1
            spent = compaction_state.reactive_retry_count
        else:
            compaction_state.retry_count += 1
            spent = compaction_state.retry_count
        if spent > self._rc.compaction_failed_max_retries:
            raise CompactionExhaustedError(f"{label} exhausted retries") from error

    def has_proactive_work(
        self,
        history: list[Message],
        compaction_state: CompactionState,
        *,
        force: bool,
        protect_tail_from_index: int | None = None,
        llm_tiers: bool = True,
    ) -> bool:
        """Whether a proactive pass would find anything its profile may touch.

        Asked before the pass opens: a pass with nothing to do would still flip
        the run into ``COMPACTING``, fire the compaction hooks, write a usage
        row and a snapshot, and tell the client it is compacting — once an
        iteration, for as long as the estimate stays over the gate. The answer
        mirrors the tiers' own eligibility (the proactive profile: routine keep
        window, seeded history untouched), so a ``False`` here is exactly a
        pass that would have changed nothing and called nothing. ``force``
        selects :meth:`force_compaction`'s rules, under which units the
        failure census has written off are still eligible. ``llm_tiers=False``
        asks about Tier 1 alone, as a pass run with the same flag would.
        """
        budgets = derive_budgets(self._rc)
        if tier1_has_work(
            history,
            self._rc,
            budgets.tool_result_truncation_threshold,
            protect_tail_from_index=protect_tail_from_index,
        ):
            return True
        if self._compaction_llm is None or not llm_tiers:
            return False
        if tier2_has_work(
            history,
            compaction_state,
            self._rc,
            protect_tail_from_index=protect_tail_from_index,
            retry_failed_units=force,
        ):
            return True
        return tier3_has_work(
            history, self._rc, protect_tail_from_index=protect_tail_from_index
        )

    async def run_compaction(
        self,
        *,
        history: list[Message],
        compaction_state: CompactionState,
        tenant_id: str,
        model_name: str,
        observability: LLMObservabilityContext | None = None,
        protect_tail_from_index: int | None = None,
        record_request: RequestRecorder | None = None,
        llm_tiers: bool = True,
    ) -> CompactionAttempt:
        """Run Tier 1 truncation; fall through to Tier 2 if needed.

        ``llm_tiers=False`` runs Tier 1 alone — the proactive gates do that
        while their summariser tiers are suspended.

        Charges :attr:`CompactionState.retry_count` once per failed pass and
        nothing for a pass that found nothing to compact (see
        :meth:`_settle_pass`); raises :class:`CompactionExhaustedError` when
        :attr:`LoopConstants.compaction_failed_max_retries` is breached.

        ``protect_tail_from_index`` (set only by the per-iteration gate)
        exempts the current just-executed tool-result batch from BOTH tiers on
        top of ``compaction_keep_recent_turns``, so a >keep parallel batch's
        fresh, unconsumed results cannot be compacted in the same iteration
        they were produced.
        """
        budgets = derive_budgets(self._rc)
        tokens_before = self._token_estimator.estimate_history(history, self._rc)

        attempt = CompactionAttempt(tokens_before=tokens_before)

        try:
            tier1 = await run_tier1_truncation(
                history=history,
                blob_store=self._blob_store,
                tenant_id=tenant_id,
                rc=self._rc,
                truncation_threshold_tokens=budgets.tool_result_truncation_threshold,
                protect_tail_from_index=protect_tail_from_index,
            )
        except Exception as exc:
            # a failed Tier-1 pass freed nothing, but tokens_after
            # defaults to 0. Stamp the real current estimate before returning so
            # the caller's COMPACTION_COMPLETED event does not report a phantom
            # "full clear" (tokens_after=0 ≪ tokens_before).
            attempt.tokens_after = self._token_estimator.estimate_history(history, self._rc)
            self._settle_pass(
                compaction_state=compaction_state,
                attempt=attempt,
                reactive=False,
                error=exc,
                label="tier1 truncation",
            )
            return attempt

        attempt.tier1 = tier1
        compaction_state.blob_refs_created.extend(tier1.blob_refs_created)

        # Only fire Tier 2 if Tier 1 didn't clear enough. The bar is
        # ``compaction_trigger_tokens * routine_min_clear_ratio`` per the
        # cascade definition.
        min_clear_target = int(
            budgets.compaction_trigger_tokens
            * self._rc.compaction_routine_min_clear_ratio
        )
        tier_error: Exception | None = None
        if (
            llm_tiers
            and tier1.tokens_freed < min_clear_target
            and self._compaction_llm is not None
        ):
            try:
                tier2 = await run_tier2_summarisation(
                    history=history,
                    compaction_llm=self._compaction_llm,
                    state=compaction_state,
                    rc=self._rc,
                    model_name=model_name,
                    observability=observability,
                    protect_tail_from_index=protect_tail_from_index,
                    record_request=record_request,
                    prompts=self._prompts,
                    # Tier-2 only needs to make up the shortfall Tier-1 left
                    # against the min-clear target; bound its per-pass LLM-call
                    # count to that budget rather than summarising every eligible
                    # turn serially.
                    free_target_tokens=min_clear_target - tier1.tokens_freed,
                )
            except Exception as exc:
                tier_error = exc
                tier2 = Tier2Result(turns_summarised=0, tokens_freed=0)
            attempt.tier2 = tier2

        if llm_tiers:
            attempt.tier3 = await self._fold(
                history,
                compaction_state,
                model_name,
                observability,
                protect_tail_from_index,
                record_request,
            )

        tokens_after = self._token_estimator.estimate_history(history, self._rc)
        attempt.tokens_after = tokens_after

        self._settle_pass(
            compaction_state=compaction_state,
            attempt=attempt,
            reactive=False,
            error=tier_error,
            label="compaction",
        )
        return attempt

    async def force_compaction(
        self,
        *,
        history: list[Message],
        compaction_state: CompactionState,
        tenant_id: str,
        model_name: str,
        observability: LLMObservabilityContext | None = None,
        protect_tail_from_index: int | None = None,
        record_request: RequestRecorder | None = None,
        reactive: bool = False,
        llm_tiers: bool = True,
    ) -> CompactionAttempt:
        """Run BOTH Tier 1 + Tier 2 unconditionally for emergency recovery.

 Unlike :meth:`run_compaction` (which gates Tier 2 behind a
 "Tier 1 didn't free enough" check), this method always runs both
 tiers — the upstream provider has already signalled that the
 request exceeds the window, so we must free as much as possible
 before re-streaming.

 Raises :class:`CompactionExhaustedError` per :meth:`_settle_pass`. A
 proactive pass shares :meth:`run_compaction`'s budget; a reactive pass has
 its own, so proactive failures cannot use up the one profile that may still
 compact seeded history. A pass that found nothing it was allowed to touch
 spends neither.

 ``protect_tail_from_index`` (set only by the per-iteration
 emergency-cliff gate) exempts the current just-executed tool-result
 batch from BOTH tiers on top of the keep window. The reactive-413
 caller passes ``None`` (the provider already rejected the request, so
 the whole history is fair game and the most-recent batch was never
 wire-accepted).

 ``reactive`` distinguishes a provider rejection from the two proactive
 emergency gates (turn start, per iteration) that also land here on an
 estimate. Only a rejection switches to the emergency profile: the keep
 window shrinks to ``compaction_force_keep_recent_turns`` and turns
 seeded from earlier runs become eligible for lossy Tier 2/Tier 3
 replacement, every replacement keeping the seed tag. The proactive
 gates keep the routine window and leave seeds untouched.
 """
        budgets = derive_budgets(self._rc)
        tokens_before = self._token_estimator.estimate_history(history, self._rc)

        attempt = CompactionAttempt(tokens_before=tokens_before)

        # A provider rejection is stronger evidence than the routine estimate:
        # keep only the emergency tail and make old session seeds eligible for
        # lossy compaction. Seed provenance is carried onto every replacement
        # so the host still excludes prior-run content when it persists the
        # new run. A proactive emergency pass has no such proof and keeps the
        # routine profile.
        force_keep_recent_turns = (
            self._rc.compaction_force_keep_recent_turns if reactive else None
        )
        compact_seeded_history = reactive

        tier_error: Exception | None = None
        # Tier 1 always runs.
        try:
            tier1 = await run_tier1_truncation(
                history=history,
                blob_store=self._blob_store,
                tenant_id=tenant_id,
                rc=self._rc,
                truncation_threshold_tokens=budgets.tool_result_truncation_threshold,
                keep_recent_turns=force_keep_recent_turns,
                protect_tail_from_index=protect_tail_from_index,
            )
        except Exception as exc:
            tier_error = exc
            tier1 = Tier1Result(
                tokens_freed=0,
                blob_refs_created=(),
                messages_modified=0,
            )

        attempt.tier1 = tier1
        compaction_state.blob_refs_created.extend(tier1.blob_refs_created)

        # Tier 2 always runs in force mode, unless the caller has suspended
        # the summariser tiers (``llm_tiers=False``, proactive only).
        if self._compaction_llm is not None and llm_tiers:
            # Free aggressively but bounded — enough to bring the post-Tier-1
            # history back under the trigger threshold so the request can
            # re-stream, without summarising every eligible turn serially.
            tokens_after_tier1 = self._token_estimator.estimate_history(history, self._rc)
            free_target = tokens_after_tier1 - budgets.compaction_trigger_tokens
            try:
                tier2 = await run_tier2_summarisation(
                    history=history,
                    compaction_llm=self._compaction_llm,
                    state=compaction_state,
                    rc=self._rc,
                    model_name=model_name,
                    observability=observability,
                    protect_tail_from_index=protect_tail_from_index,
                    free_target_tokens=free_target if free_target > 0 else None,
                    record_request=record_request,
                    prompts=self._prompts,
                    keep_recent_turns=force_keep_recent_turns,
                    compact_seeded_history=compact_seeded_history,
                    # A forced pass runs when the alternative is the run
                    # ending, so it tries every unit — including the ones the
                    # routine gate has written off. A call that is probably
                    # wasted is cheaper than a run that cannot continue.
                    retry_failed_units=True,
                )
            except Exception as exc:
                if tier_error is None:
                    tier_error = exc
                tier2 = Tier2Result(turns_summarised=0, tokens_freed=0)
            attempt.tier2 = tier2

        if llm_tiers:
            attempt.tier3 = await self._fold(
                history,
                compaction_state,
                model_name,
                observability,
                protect_tail_from_index,
                record_request,
                keep_recent_turns=force_keep_recent_turns,
                compact_seeded_history=compact_seeded_history,
            )

        tokens_after = self._token_estimator.estimate_history(history, self._rc)
        attempt.tokens_after = tokens_after

        # ANY progress (tokens freed, or any Tier 1 / Tier 2 / Tier 3 rewrite)
        # clears the budget; a pass that tried and failed is charged once.
        self._settle_pass(
            compaction_state=compaction_state,
            attempt=attempt,
            reactive=reactive,
            error=tier_error,
            label="reactive force_compaction" if reactive else "force_compaction",
        )
        return attempt

    def current_prompt_tokens(
        self,
        history: Sequence[Message],
    ) -> int:
        """Calibrated estimate of the history available to this gate."""
        return self._token_estimator.estimate_history(history, self._rc)

    def needs_compaction(
        self,
        history: Sequence[Message],
        *,
        overhead_tokens: int = 0,
    ) -> bool:
        """Return ``True`` if the current prompt exceeds the trigger threshold.

        ``overhead_tokens`` is the part of the prompt the history does not
        carry — the system prompt and the tool definitions — in the same
        calibrated tokens. The trigger is sized as a whole prompt (the largest
        one the provider accepts, less a turn's headroom), so the history alone
        must not be held against it: with a large tool surface it would reach
        the trigger only after the whole request had passed the provider's
        ceiling, and compaction would first run on a refusal.
        """
        budgets = derive_budgets(self._rc)
        current = self.current_prompt_tokens(history) + max(0, overhead_tokens)
        return current > budgets.compaction_trigger_tokens

    def needs_emergency_compaction(
        self,
        history: Sequence[Message],
        *,
        overhead_tokens: int = 0,
    ) -> bool:
        """Return ``True`` if the current prompt exceeds the emergency cliff.

        Activates :attr:`LoopConstants.compaction_emergency_ratio`. When
        this is True the runtime should run :meth:`force_compaction` (both
        tiers, unconditional) proactively rather than waiting for the provider
        to raise a context-window-exceeded error. ``compaction_emergency_tokens``
        is strictly above ``compaction_trigger_tokens`` (the RC validator
        enforces ``compaction_trigger_ratio < compaction_emergency_ratio``).
        """
        budgets = derive_budgets(self._rc)
        current = self.current_prompt_tokens(history) + max(0, overhead_tokens)
        return current > budgets.compaction_emergency_tokens


__all__ = [
    "ContextBundle",
    "ContextManager",
    "detect_active_language",
    "estimate_history_tokens",
]
