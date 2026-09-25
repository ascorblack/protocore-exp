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
from protocore.contracts.tool_roles import EMPTY_TOOL_ROLE_MAP, ToolRoleMap
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
    floor_has_work,
    place_ledger,
    run_floor,
    run_tier1_truncation,
    run_tier2_summarisation,
    run_tier3_fold,
    tier1_has_work,
    tier2_has_work,
    tier3_has_work,
)
from protocore.runtime.context.ledger import Ledger, is_ledger, ledger_from_history
from protocore.runtime.token_counting import LanguageProfile, detect_profile, estimate_tokens

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
        tool_roles: ToolRoleMap = EMPTY_TOOL_ROLE_MAP,
    ) -> None:
        self._rc = rc
        # What the host's tools do, so the ledger can say a file was written
        # rather than merely named.
        self._tool_roles = tool_roles
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
        tenant_id: str = "",
        ledger: Ledger | None = None,
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
                blob_store=self._blob_store,
                tenant_id=tenant_id,
                ledger=ledger,
                roles=self._tool_roles,
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

        * **progress** — it freed tokens, or masked, summarised, folded or
          removed anything. Both budgets are cleared, whichever profile made
          it: the history the next pass of either kind faces is a different one.
        * **nothing to do** — no tier raised and no tier found anything it was
          allowed to touch under this pass's profile. It spends nothing.
        * **failed** — a tier raised, or a tier was tried and nothing came of
          it: the summariser calls all failed and the floor, if it ran, had
          nothing left to remove. One increment, however many tiers failed.

        With the floor in the cascade a pass that could remove anything always
        makes progress, so only a history already at its floor can spend the
        budget. A reactive pass (after a provider rejection) spends
        :attr:`CompactionState.reactive_retry_count`; every other pass spends
        :attr:`CompactionState.retry_count`. Both are bounded by
        :attr:`LoopConstants.compaction_failed_max_retries`, and breaching the
        bound raises :class:`CompactionExhaustedError` chained to the tier
        exception when there was one.
        """
        tier1, tier2, tier3, floor = attempt.tier1, attempt.tier2, attempt.tier3, attempt.floor
        progress = (
            attempt.tokens_after < attempt.tokens_before
            or (tier1 is not None and tier1.messages_modified > 0)
            or (tier2 is not None and tier2.turns_summarised > 0)
            or (tier3 is not None and tier3.spans_folded > 0)
            or (floor is not None and floor.messages_dropped > 0)
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
        mirrors the tiers' own eligibility under the proactive profile (routine
        keep window, seeded history untouched), the floor included, so a
        ``False`` here is exactly a history at its floor. ``force`` selects
        :meth:`force_compaction`'s rules, under which units the failure census
        has written off are still eligible; ``llm_tiers=False`` leaves the
        summariser tiers out, as a pass run with the same flag would.
        """
        budgets = derive_budgets(self._rc)
        if tier1_has_work(
            history,
            self._rc,
            budgets.tool_result_truncation_threshold,
            protect_tail_from_index=protect_tail_from_index,
            mask_by_age=True,
        ):
            return True
        if floor_has_work(history, self._rc, protect_tail_from_index=protect_tail_from_index):
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

    async def _run_pass(
        self,
        *,
        history: list[Message],
        compaction_state: CompactionState,
        tenant_id: str,
        model_name: str,
        observability: LLMObservabilityContext | None,
        protect_tail_from_index: int | None,
        record_request: RequestRecorder | None,
        llm_tiers: bool,
        overhead_tokens: int,
        reactive: bool,
        forced: bool,
        label: str,
    ) -> CompactionAttempt:
        """One pass of the cascade: mask, summarise, fold, floor — each only while room is still needed.

        The pass aims at the TARGET, ``compaction_trigger_tokens *
        compaction_target_ratio``, measured on the whole prompt: the history
        plus ``overhead_tokens`` (system prompt and tools). Each tier runs only
        while the prompt is above the target, cheapest first. If the prompt is
        still above the TRIGGER when the model tiers are done — the summariser
        failed, timed out, is suspended, or found nothing worth a call — the
        floor removes the oldest spans until the target is met or nothing
        removable is left. So every pass that is opened ends below the trigger
        or at the floor; a summary is an improvement on the floor, never a
        precondition for progress.

        The ledger is rebuilt from the history's own ledger state plus
        everything this pass took out of the window, and put back at the
        boundary between the compacted past and the kept present.
        """
        rc = self._rc
        budgets = derive_budgets(rc)
        overhead = max(0, overhead_tokens)
        trigger = budgets.compaction_trigger_tokens
        tokens_before = self._token_estimator.estimate_history(history, rc)
        # A pass is opened because the prompt is too large: the gate saw it
        # over the trigger, or the provider refused it. Aiming below the
        # current size by the same ratio as below the trigger means an opened
        # pass always has something to free, including when the evidence that
        # opened it (a refusal) says more than the estimate does.
        target = max(
            1,
            min(
                int(trigger * rc.compaction_target_ratio),
                int((tokens_before + overhead) * rc.compaction_target_ratio),
            ),
        )
        attempt = CompactionAttempt(
            tokens_before=tokens_before,
            prompt_before=tokens_before + overhead,
            trigger_tokens=trigger,
            target_tokens=target,
        )
        keep = rc.compaction_force_keep_recent_turns if reactive else None
        compact_seeded_history = reactive
        ledger = ledger_from_history(history)
        ledger_before = sum(
            self._token_estimator.estimate_message(message, rc) for message in history if is_ledger(message)
        )

        def need() -> int:
            # What the pass still has to free: the prompt over the target, plus
            # what the ledger has grown by — it is put back after the tiers, and
            # a pass that met the target before adding it would end above it.
            growth = max(0, estimate_tokens(ledger.render(rc), rc) - ledger_before) if not ledger.is_empty() else 0
            return self._token_estimator.estimate_history(history, rc) + overhead + growth - target

        tier_error: Exception | None = None
        try:
            attempt.tier1 = await run_tier1_truncation(
                history=history,
                blob_store=self._blob_store,
                tenant_id=tenant_id,
                rc=rc,
                truncation_threshold_tokens=budgets.tool_result_truncation_threshold,
                keep_recent_turns=keep,
                protect_tail_from_index=protect_tail_from_index,
                mask_by_age=True,
                free_target_tokens=max(0, need()),
                ledger=ledger,
            )
            compaction_state.blob_refs_created.extend(attempt.tier1.blob_refs_created)
        except Exception as exc:
            _logger.warning("compaction tier 1 failed; the pass continues (err=%r)", exc)
            tier_error = exc
            attempt.tier1 = Tier1Result(tokens_freed=0, blob_refs_created=(), messages_modified=0)

        if llm_tiers and self._compaction_llm is not None and need() > 0:
            try:
                attempt.tier2 = await run_tier2_summarisation(
                    history=history,
                    compaction_llm=self._compaction_llm,
                    state=compaction_state,
                    rc=rc,
                    model_name=model_name,
                    observability=observability,
                    protect_tail_from_index=protect_tail_from_index,
                    free_target_tokens=need(),
                    record_request=record_request,
                    prompts=self._prompts,
                    keep_recent_turns=keep,
                    compact_seeded_history=compact_seeded_history,
                    # A forced pass runs when the alternative is the run
                    # ending, so it tries every unit — including the ones the
                    # routine gate has written off.
                    retry_failed_units=forced,
                    blob_store=self._blob_store,
                    tenant_id=tenant_id,
                    ledger=ledger,
                    roles=self._tool_roles,
                )
            except Exception as exc:
                _logger.warning("compaction tier 2 failed; the pass continues (err=%r)", exc)
                tier_error = tier_error or exc
                attempt.tier2 = Tier2Result(turns_summarised=0, tokens_freed=0)

        if llm_tiers and need() > 0:
            attempt.tier3 = await self._fold(
                history,
                compaction_state,
                model_name,
                observability,
                protect_tail_from_index,
                record_request,
                keep_recent_turns=keep,
                compact_seeded_history=compact_seeded_history,
                tenant_id=tenant_id,
                ledger=ledger,
            )

        # The floor takes over where the model tiers stopped short: above the
        # trigger for a routine pass, above the target for a forced one — a
        # refusal, or an estimate past the emergency line, is not satisfied by
        # merely getting back under the gate.
        floor_line = target if forced else trigger
        if self._token_estimator.estimate_history(history, rc) + overhead > floor_line:
            # The ledger grows with what the floor removes, and it is put back
            # after the floor: room for all of it is part of the target.
            ledger_room = max(0, ledger.budget(rc) - ledger_before)
            try:
                attempt.floor = await run_floor(
                    history,
                    rc,
                    free_target_tokens=self._token_estimator.estimate_history(history, rc)
                    + overhead
                    + ledger_room
                    - target,
                    keep_recent_turns=keep,
                    protect_tail_from_index=protect_tail_from_index,
                    compact_seeded_history=compact_seeded_history,
                    ledger=ledger,
                    roles=self._tool_roles,
                    blob_store=self._blob_store,
                    tenant_id=tenant_id,
                )
            except Exception as exc:
                _logger.warning("compaction floor failed; the pass continues (err=%r)", exc)
                tier_error = tier_error or exc

        attempt.ledger_tokens = place_ledger(
            history, ledger, rc, protect_tail_from_index=protect_tail_from_index
        )
        attempt.tokens_after = self._token_estimator.estimate_history(history, rc)
        attempt.prompt_after = attempt.tokens_after + overhead
        if attempt.prompt_after <= target:
            attempt.outcome = "below_target"
        elif attempt.prompt_after <= trigger:
            attempt.outcome = "below_trigger"
        elif attempt.floor is not None and attempt.floor.reached:
            attempt.outcome = "at_floor"
        elif attempt.tokens_after < attempt.tokens_before:
            attempt.outcome = "above_trigger"
        else:
            attempt.outcome = "unchanged"
        _logger.warning(
            "DIAG compaction.pass label=%s outcome=%s prompt=%d->%d trigger=%d target=%d "
            "t1_freed=%d t1_aged=%d t2=%s/%s t2_fail=%s fold=%s floor=%s ledger=%d",
            label,
            attempt.outcome,
            attempt.prompt_before,
            attempt.prompt_after,
            trigger,
            target,
            attempt.tier1.tokens_freed if attempt.tier1 else 0,
            attempt.tier1.masked_by_age if attempt.tier1 else 0,
            attempt.tier2.turns_summarised if attempt.tier2 else "-",
            attempt.tier2.units_attempted if attempt.tier2 else "-",
            attempt.tier2.failures if attempt.tier2 else {},
            attempt.tier3.spans_folded if attempt.tier3 else "-",
            attempt.floor.messages_dropped if attempt.floor else "-",
            attempt.ledger_tokens,
        )
        self._settle_pass(
            compaction_state=compaction_state,
            attempt=attempt,
            reactive=reactive,
            error=tier_error,
            label=label,
        )
        return attempt

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
        overhead_tokens: int = 0,
    ) -> CompactionAttempt:
        """A routine pass: the cascade under the routine profile (see :meth:`_run_pass`).

        ``llm_tiers=False`` leaves the summariser tiers out — the proactive
        gates do that while they are suspended; masking and the floor still
        run. ``protect_tail_from_index`` (set only by the per-iteration gate)
        exempts the current just-executed batch on top of
        ``compaction_keep_recent_turns``.

        Charges :attr:`CompactionState.retry_count` once per failed pass and
        nothing for a pass that found nothing to compact (see
        :meth:`_settle_pass`); raises :class:`CompactionExhaustedError` when
        :attr:`LoopConstants.compaction_failed_max_retries` is breached.
        """
        return await self._run_pass(
            history=history,
            compaction_state=compaction_state,
            tenant_id=tenant_id,
            model_name=model_name,
            observability=observability,
            protect_tail_from_index=protect_tail_from_index,
            record_request=record_request,
            llm_tiers=llm_tiers,
            overhead_tokens=overhead_tokens,
            reactive=False,
            forced=False,
            label="compaction",
        )

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
        overhead_tokens: int = 0,
    ) -> CompactionAttempt:
        """An emergency pass: the same cascade, with units the census wrote off retried.

        ``reactive`` distinguishes a provider rejection from the two proactive
        emergency gates (turn start, per iteration) that also land here on an
        estimate. Only a rejection switches to the emergency profile: the keep
        window shrinks to ``compaction_force_keep_recent_turns`` and turns
        seeded from earlier runs become eligible for replacement, every
        replacement keeping the seed tag. The reactive-413 caller passes no
        ``protect_tail_from_index``: the provider already rejected the request,
        so the most recent batch was never read.

        A proactive pass shares :meth:`run_compaction`'s retry budget; a
        reactive pass has its own, so proactive failures cannot use up the one
        profile that may still compact seeded history.
        """
        return await self._run_pass(
            history=history,
            compaction_state=compaction_state,
            tenant_id=tenant_id,
            model_name=model_name,
            observability=observability,
            protect_tail_from_index=protect_tail_from_index,
            record_request=record_request,
            llm_tiers=llm_tiers,
            overhead_tokens=overhead_tokens,
            reactive=reactive,
            forced=True,
            label="reactive force_compaction" if reactive else "force_compaction",
        )


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
