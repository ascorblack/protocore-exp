# ruff: noqa: RUF001 — Bilingual RU+EN runtime nudge strings intentionally use Cyrillic characters.
"""The turn driver — the body of ONE turn of the agent loop.

Public entries: :func:`resume` (pick a stored run back up and drive it) and
:func:`resume_approved_tool` (execute a call that was waiting on approval);
:meth:`QueryEngine.run` opens a fresh turn.  Everything below them is private.

md` (full ASCII sequence).

The function is **pure** w.r.t. global state — all mutations flow through
the injected :class:`QueryEngine` (history, state, compaction state). It
is the **single place in core that knows about** :class:`ProviderDelta`;
outside this function the loop deals in :class:`TurnEvent`.

Lifecycle (one invocation = one turn):

 1. Stop-check + state → RUNNING (PENDING → RUNNING transition).
 2. Compaction check + run if needed.
 3. UserPromptSubmit hook fire.
 4. Build context bundle (system prompt + tools + history).
 5. message_start event.
 6. Stream provider deltas → translate to TurnEvent.
 7. Collect tool_calls; dispatch each with hooks.
 8. Recurse (open new LLM stream w/ tool_results) until no tool calls.
 9. message_stop + state → COMPLETED (or AWAITING / CANCELLED / FAILED).
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import re
import time
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Iterator,
    Mapping,
    Sequence,
)
from dataclasses import dataclass, replace
from functools import partial
from typing import TYPE_CHECKING, Any, Final

from protocore.contracts.agent_dispatch import IDelegationTool
from protocore.contracts.background import describe_finished_task
from protocore.contracts.evidence import ToolEvidenceContext
from protocore.contracts.hooks import HookActionKind
from protocore.contracts.interrupt import (
    InterruptDecision,
    InterruptKind,
    InterruptResolution,
    InterruptResolutionError,
    PendingInterrupt,
    find_interrupt_for_call,
    interrupts_of_kind,
    plan_resolution,
)
from protocore.contracts.llm import (
    CacheBreakpoint,
    LLMContextWindowExceeded,
    LLMError,
    LLMObservabilityContext,
    LLMRateLimitError,
    LLMRequest,
    LLMStreamEvent,
    LLMStreamIdleError,
    LLMTimeoutError,
    ProviderDelta,
    ProviderDeltaKind,
)
from protocore.contracts.middleware import (
    ILifecycleRegistry,
    LifecycleOutcome,
    LifecycleVerdict,
)
from protocore.contracts.observability import (
    RequestManifest,
    build_request_manifest,
    constants_digest,
)
from protocore.contracts.run_state import RunScopedState
from protocore.contracts.skills import (
    ISkillStore,
    SkillBundle,
    SkillIndexEntry,
    SkillNotFoundError,
)
from protocore.contracts.tool_chunking import (
    CHUNKABLE_CONTENT_FIELD,
    chunkable_content_mutation_names,
    is_chunkable_content_mutation,
)
from protocore.contracts.tool_roles import (
    WORKSPACE_INSPECTION_ROLES,
    WORKSPACE_MUTATION_ROLES,
    ToolArgumentSlot,
    ToolRole,
)
from protocore.contracts.tools import (
    SUBAGENT_DISPATCH_GROUP_METADATA_KEY,
    SUBAGENT_DISPATCH_ORDER_METADATA_KEY,
    SUBAGENT_TREE_PERMIT_METADATA_KEY,
    ToolContext,
)
from protocore.contracts.turn_policy import (
    TurnContext,
    TurnCoordinate,
    TurnDirective,
    TurnFlags,
    TurnPolicyOutcome,
)
from protocore.contracts.types import (
    PARTIAL_ASSISTANT_ATTEMPT_METADATA_KEY,
    SESSION_HISTORY_SEED_METADATA_KEY,
    SYNTHETIC_RECOVERY_CIRCUIT_BREAKER,
    SYNTHETIC_RECOVERY_GUARANTEED_TERMINAL,
    SYNTHETIC_RECOVERY_LONGFILE_CONTINUE,
    SYNTHETIC_RECOVERY_LONGFILE_SALVAGE,
    SYNTHETIC_RECOVERY_LONGFILE_TERMINAL_SEAL,
    SYNTHETIC_RECOVERY_METADATA_KEY,
    SYNTHETIC_RECOVERY_POST_TOOL_EMPTY_NUDGE,
    SYNTHETIC_RECOVERY_PRE_DISPATCH_TERMINAL_VERIFY,
    SYNTHETIC_RECOVERY_PRE_TERMINAL_SELF_VERIFY,
    SYNTHETIC_RECOVERY_PROSE_GATE_REPAIR,
    SYNTHETIC_RECOVERY_REASONING_CUT,
    SYNTHETIC_RECOVERY_TERMINAL_REPAIR,
    SYNTHETIC_RECOVERY_TERMINAL_TOOL_NUDGE,
    SYNTHETIC_RECOVERY_THINKING_CONTINUE,
    TERMINAL_TOOL_METADATA_KEY,
    TOOL_RESULT_CONSECUTIVE_CAP_ELIGIBLE_METADATA_KEY,
    ContentBlock,
    HookEvent,
    Message,
    MessageRole,
    StopReason,
    TextBlock,
    ToolCall,
    ToolDefinition,
    ToolResultBlock,
    ToolUseBlock,
)
from protocore.logging_utils import get_logger
from protocore.runtime import longfile_convergence as _longfile
from protocore.runtime import pending_reads as _pending_reads
from protocore.runtime import run_tool_preconditions as _preconditions
from protocore.runtime import soft_stop as _soft_stop
from protocore.runtime.answer_narration import leading_narration_span
from protocore.runtime.context.compaction import (
    CompactionExhaustedError,
    current_tool_batch_protect_index,
    estimate_history_tokens,
)
from protocore.runtime.context.manager import ContextBundle
from protocore.runtime.error_kinds import (
    INTERNAL_ERROR_KIND as INTERNAL_ERROR_KIND,  # re-exported: a caller that
    # reports a run's terminal kind reads it from the driver it drives.
)
from protocore.runtime.events import BlockVisibility, EventType, TurnEvent
from protocore.runtime.intent import (
    RESERVED,
    SETTLED,
    IntentRecord,
    assert_pause_matches,
    commit_intent,
    find_intent,
    mark_dispatched,
    mark_paused_ask_user,
    mark_pending_approval,
    orphaned_intents,
    settle_intent,
    settle_unknown,
    unknown_outcome_text,
)
from protocore.runtime.intent import (
    fingerprint_arguments as _intent_fingerprint,
)
from protocore.runtime.live_control import (
    QueuedPrompt,
    place_items,
    placed_as_user_message,
    restore_queued_prompts,
)
from protocore.runtime.llm.delta_bridge import (
    delta_to_turn_events,
    stream_events_to_provider_deltas,
)
from protocore.runtime.loop_guard import (
    canonical_tool_fingerprint,
    identical_tool_should_block,
    inspect_stream_repeat,
)
from protocore.runtime.loop_state import LoopState
from protocore.runtime.loop_strategies import select_strategy
from protocore.runtime.prompt_caching import apply_system_and_3
from protocore.runtime.result_eviction import evict_history_for_llm, tool_name_for_result
from protocore.runtime.run_work_budget import (
    SUBAGENT_RUN_BUDGET_SHORT,
    ChildRunGrant,
    RunWorkLedger,
)
from protocore.runtime.skill_index import (
    derive_skill_index_budget_tokens,
    render_skills_catalog,
)
from protocore.runtime.subagent_budget import SubagentTreeBudget, SubagentTreePermit
from protocore.runtime.token_counting import estimate_tokens
from protocore.runtime.tool_arguments import argument_names, string_argument
from protocore.runtime.tool_dispatch import (
    DISPATCH_POST_TOOL_OUTPUT_MODIFIED_METADATA_KEY,
    DISPATCH_REPLAY_ERROR_KIND_METADATA_KEY,
    DISPATCH_REPLAY_ERROR_MESSAGE_METADATA_KEY,
    DISPATCH_STRUCTURED_ERROR_METADATA_KEY,
    STRUCTURED_ERROR_FINALIZATION_RECOMMENDED_KEY,
    STRUCTURED_ERROR_REASON_KEY,
    DispatchErrorKind,
    DispatchOutcome,
    ToolDispatcher,
    _annotate_tool_result_event,
    _append_soft_cap_warning_to_content,
    _merge_soft_cap_warning_metadata,
    _record_tool_call_soft_cap_warning,
)
from protocore.runtime.tool_permission import ToolPermissionGate
from protocore.runtime.turn_policies import (
    RunCounter,
    TurnPolicyRegistry,
    UnsupportedTurnDirectiveError,
)
from protocore.runtime.turn_policies.answer_floor import AnswerFloorPolicy
from protocore.runtime.turn_policies.cancellation import CancellationPolicy
from protocore.runtime.turn_policies.compaction import PerIterationCompactionPolicy
from protocore.runtime.turn_policies.empty_completion import EmptyCompletionGuardPolicy
from protocore.runtime.turn_policies.empty_model_turn import EmptyModelTurnPolicy
from protocore.runtime.turn_policies.longfile import LongFileConvergencePolicy
from protocore.runtime.turn_policies.output_cap import (
    OutputCapRecoveryPolicy,
    TruncationSalvage,
)
from protocore.runtime.turn_policies.provider_failure import ProviderFailurePolicy
from protocore.runtime.turn_policies.repeat_guard import StreamLoopGuardPolicy
from protocore.runtime.turn_policies.run_ceilings import RunCeilingsPolicy
from protocore.runtime.turn_policies.sibling_walk import (
    dispatch_parking_holds,
    park_deferred_hold,
)
from protocore.runtime.turn_policies.sibling_walk import (
    prose_gate_just_injected as _prose_gate_injected,
)
from protocore.runtime.turn_policies.terminal_nudge import TerminalNudgePolicy
from protocore.runtime.turn_policies.terminal_tool_finish import (
    TerminalToolFinishPolicy,
)
from protocore.runtime.turn_policies.truncated_tool_call import (
    TruncatedToolCallRecoveryPolicy,
)

if TYPE_CHECKING:
    from protocore.contracts.hooks import HookResult
    from protocore.contracts.runtime_constants import LoopConstants
    from protocore.runtime.query_engine import QueryEngine


_logger = get_logger(__name__)


def _enter_soft_stop(engine: QueryEngine, *, cause: str) -> list[TurnEvent]:
    """Begin the run wind-down for ``cause``. The ONE entry point.

    Returns the events the caller must forward, or an empty list when the
    wind-down does not apply — it is disabled, or it is already running. An
    empty list is the caller's signal to take its own terminal path, which is
    exactly what a second bound reached DURING the wind-down should do: the run
    was already told to stop and is now out of the turns it was given to do it.

    Every bound routes here — the tool-call budget, the turn cap, the
    output-token budget, the deadline, an upstream that stopped answering.
    Before, each ended somewhere different: one appended a paragraph of advice
    to a tool result and let the agent keep working, one nudged then forged a
    synthetic answer, one granted a best-effort turn, one just stopped. The
    caller's remaining job after this returns is mechanical and the same
    everywhere: grant the wind-down its turns, persist, rebuild the context so
    the next stream sees the narrowed surface, and continue.
    """
    if not _soft_stop.is_enabled(engine):
        return []
    if _soft_stop.is_armed(engine):
        return []
    events = _soft_stop.enter(engine, cause_name=cause)
    if events:
        # The terminal-only guard shares this latch with the voluntary-finish
        # contract repair: it is what makes a blocked non-terminal dispatch
        # answer with an instruction to finalise rather than a bare denial.
        engine._terminal_only_active = True
    return events


def _soft_stop_turn_budget(engine: QueryEngine, assistant_message_idx: int) -> int:
    """Turns the wind-down gets, counted from where the run actually is."""
    return assistant_message_idx + engine.config.rc.soft_stop_max_turns


def _append_thinking_continue_prompt(engine: QueryEngine) -> None:
    """Ask a model that spent its whole round thinking to answer.

    The reasoning itself is already back in history; this is the short user
    turn that says what to do with it.
    """
    engine.history.append(
        Message(
            role=MessageRole.user,
            content_blocks=[TextBlock(text=engine.config.rc.continue_prompt_text)],
            metadata={
                SYNTHETIC_RECOVERY_METADATA_KEY: SYNTHETIC_RECOVERY_THINKING_CONTINUE
            },
        )
    )


def _step_reasoning_after_cut(engine: QueryEngine, round_: int) -> str | None:
    """Turn one knob down for the retry after a reasoning-only length cut.

    Round 1 lowers the effort to ``low``; round 2 switches thinking off. Each
    step is skipped when it would change nothing, and a step the run mode
    forbids (``deep`` keeps thinking on) is skipped the same way, so the value
    returned says what the retry actually differs by — ``None`` when nothing
    is left to change and the ladder is spent. The values as they stood before
    the first step are kept for :func:`_restore_reasoning_after_cut`.
    """
    if engine._reasoning_cut_saved is None:
        engine._reasoning_cut_saved = (
            engine._live_thinking_enabled,
            engine._live_reasoning_effort,
        )
    # The ladder is laid out from where the knobs stood BEFORE the first
    # step, so the second round finds its step where the first left it.
    saved_thinking, saved_effort = engine._reasoning_cut_saved
    thinking = engine.config.thinking_enabled if saved_thinking is None else saved_thinking
    effort = saved_effort or engine.config.reasoning_effort
    steps: list[tuple[str, Callable[[], None]]] = []
    if thinking and effort != "low":
        steps.append(
            ("reasoning_effort=low", partial(engine.apply_live_controls, reasoning_effort="low"))
        )
    if (
        thinking
        and engine.config.rc.reasoning_length_cut_disable_thinking
        and engine.config.run_mode != "deep"
    ):
        steps.append(
            ("thinking=off", partial(engine.apply_live_controls, thinking_enabled=False))
        )
    if round_ < 1 or round_ > len(steps):
        return None
    name, step = steps[round_ - 1]
    step()
    return name


def _restore_reasoning_after_cut(engine: QueryEngine) -> None:
    """Put thinking and effort back after the retries a cut round started."""
    saved = engine._reasoning_cut_saved
    if saved is None:
        return
    engine._live_thinking_enabled, engine._live_reasoning_effort = saved
    engine._reasoning_cut_saved = None


def _append_reasoning_cut_nudge(engine: QueryEngine) -> None:
    """Name the cut and ask for a shorter shape; the cut reasoning is not kept."""
    engine.history.append(
        Message(
            role=MessageRole.user,
            content_blocks=[
                TextBlock(text=engine.config.rc.reasoning_length_cut_nudge_text)
            ],
            metadata={SYNTHETIC_RECOVERY_METADATA_KEY: SYNTHETIC_RECOVERY_REASONING_CUT},
        )
    )


def _policy_reasoning_cut_event(
    engine: QueryEngine, round_: int, changed: str, reasoning_chars: int
) -> TurnEvent:
    """Say a cut retry is going out, what it changed, and what the cut round cost."""
    _logger.warning(
        "reasoning-only length cut in run %s: %s chars of reasoning and no answer; retry %s with %s",
        engine.config.run_id,
        reasoning_chars,
        round_,
        changed,
    )
    return TurnEvent(
        type=EventType.STATE_CHANGED,
        run_id=engine.config.run_id,
        payload={
            "from": engine.state.value,
            "to": engine.state.value,
            "reason": "reasoning_length_cut_retry",
            "round": round_,
            "changed": changed,
            "reasoning_content_chars": reasoning_chars,
        },
    )


def _append_post_tool_empty_nudge(engine: QueryEngine) -> None:
    """Correct a model that answered a tool result with silence.

    An API-valid pair: an empty assistant turn first, so the wire sequence
    stays tool → assistant → user and never tool → user, then the nudge. The
    empty turn is flagged as recovery scaffolding, or the marker it carries
    would later read as a real model answer that a backstop could submit.
    """
    rc = engine.config.rc
    engine.history.append(
        Message(
            role=MessageRole.assistant,
            content_blocks=[TextBlock(text=rc.post_tool_empty_nudge_assistant_text)],
            metadata={
                SYNTHETIC_RECOVERY_METADATA_KEY: SYNTHETIC_RECOVERY_POST_TOOL_EMPTY_NUDGE
            },
        )
    )
    engine.history.append(
        Message(
            role=MessageRole.user,
            content_blocks=[TextBlock(text=rc.post_tool_empty_nudge_user_text)],
            metadata={
                SYNTHETIC_RECOVERY_METADATA_KEY: SYNTHETIC_RECOVERY_POST_TOOL_EMPTY_NUDGE
            },
        )
    )


def _policy_continue_prompt_event(
    engine: QueryEngine, round_: int, reasoning_chars: int
) -> TurnEvent:
    """Say a continue prompt went in, and what the round produced instead."""
    return TurnEvent(
        type=EventType.STATE_CHANGED,
        run_id=engine.config.run_id,
        payload={
            "from": engine.state.value,
            "to": engine.state.value,
            "reason": "continue_prompt_injected",
            "round": round_,
            "reasoning_content_chars": reasoning_chars,
        },
    )


def _empty_rounds_spent(engine: QueryEngine) -> int:
    return engine._consecutive_empty_responses


def _charge_empty_round(engine: QueryEngine) -> int:
    engine._consecutive_empty_responses += 1
    return engine._consecutive_empty_responses


def _reset_empty_rounds(engine: QueryEngine) -> None:
    engine._consecutive_empty_responses = 0


def _policy_commit_usage(
    engine: QueryEngine,
    *,
    kind: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    success: bool,
) -> TurnEvent | None:
    """Record what an attempt spent, from a policy that saw it fail."""
    from protocore.runtime.correctness_bind import commit_usage

    return commit_usage(
        engine,
        kind=kind,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        success=success,
    )


def _policy_fallback_event(
    engine: QueryEngine,
    advanced_to: str,
    exc: BaseException,
    error_class: str | None,
) -> TurnEvent:
    """Say on the wire that the run moved to another provider, and why."""
    payload: dict[str, Any] = {
        "from": engine.state.value,
        "to": engine.state.value,
        "reason": "model_fallback_triggered",
        "fallback_model_id": advanced_to,
        "primary_error": str(exc),
    }
    if error_class is not None:
        payload["error_class"] = error_class
    return TurnEvent(
        type=EventType.STATE_CHANGED,
        run_id=engine.config.run_id,
        payload=payload,
    )


def _policy_transient_retry_event(
    engine: QueryEngine,
    kind: str,
    attempt: int,
    backoff_seconds: float,
    exc: BaseException,
) -> TurnEvent:
    """Say that the same endpoint will be tried again, and after how long."""
    return TurnEvent(
        type=EventType.STATE_CHANGED,
        run_id=engine.config.run_id,
        payload={
            "from": engine.state.value,
            "to": engine.state.value,
            "reason": "transient_llm_error_retry",
            "error_class": kind,
            "attempt": attempt,
            "backoff_seconds": backoff_seconds,
            "primary_error": str(exc),
        },
    )


def _policy_log_stream_crash(engine: QueryEngine, exc: BaseException) -> None:
    """Name the turn an unclassified crash landed on, in the run's log."""
    _logger.warning(
        "DIAG query.stream_crashed run=%s tenant=%s turn=%s exception=%s "
        "message=%s",
        engine.config.run_id,
        engine.config.tenant_id,
        engine.turn_id(),
        type(exc).__name__,
        exc,
        exc_info=exc,
    )


def _output_recoveries_spent(engine: QueryEngine) -> int:
    return engine._max_output_recovery_count


def _charge_output_recovery(engine: QueryEngine) -> int:
    engine._max_output_recovery_count += 1
    return engine._max_output_recovery_count


def _reset_output_recoveries(engine: QueryEngine) -> None:
    engine._max_output_recovery_count = 0


def _pin_terminal_backstop(engine: QueryEngine) -> None:
    """Mark the wind-down turns as the run's last, recovery-free ones."""
    engine._terminal_backstop_turn_active = True


def _run_chunkable_write_names(engine: QueryEngine) -> frozenset[str]:
    """The built-in writes of this run whose content can be sent in chunks."""
    return chunkable_content_mutation_names(engine.config.tool_roles)


def _policy_state_payload_event(
    engine: QueryEngine, payload: dict[str, Any]
) -> TurnEvent:
    """A state-change event carrying a payload the policy composed itself."""
    return TurnEvent(
        type=EventType.STATE_CHANGED,
        run_id=engine.config.run_id,
        payload=payload,
    )


def _transient_retries_spent(engine: QueryEngine) -> int:
    return engine._transient_stream_retry_count


def _charge_transient_retry(engine: QueryEngine) -> int:
    engine._transient_stream_retry_count += 1
    return engine._transient_stream_retry_count


def _reset_transient_retries(engine: QueryEngine) -> None:
    engine._transient_stream_retry_count = 0


def _truncation_recoveries_spent(engine: QueryEngine) -> int:
    return engine._tool_call_truncated_recovery_count


def _charge_truncation_recovery(engine: QueryEngine) -> int:
    engine._tool_call_truncated_recovery_count += 1
    return engine._tool_call_truncated_recovery_count


def _reset_truncation_recoveries(engine: QueryEngine) -> None:
    engine._tool_call_truncated_recovery_count = 0


def _policy_truncation_result_event(
    engine: QueryEngine, tool_call_id: str, message: str
) -> TurnEvent:
    """The error result a call that was never dispatched leaves on the wire.

    Shaped as an ordinary ``tool_result`` with a real ``tool_call_id`` binding
    rather than as a bare error, because every consumer of the stream already
    knows how to draw a failed call, and none of them knows how to draw a call
    that silently was not made.
    """
    return TurnEvent(
        type=EventType.TOOL_RESULT,
        run_id=engine.config.run_id,
        payload={
            "tool_call_id": tool_call_id,
            "success": False,
            "is_error": True,
            "error": {"kind": "tool_call_truncated", "message": message},
            "content_blocks": [{"type": "text", "text": message}],
        },
    )


def _policy_tool_use_message_stop(engine: QueryEngine) -> TurnEvent:
    """Close the assistant-message window a policy dispatched tools inside.

    Carries the usage figures the ordinary between-messages frame carries, so
    a reader cannot tell a window a policy closed from one the loop closed.
    """
    return TurnEvent(
        type=EventType.MESSAGE_STOP,
        run_id=engine.config.run_id,
        payload={
            "turn_id": engine.turn_id(),
            "stop_reason": "tool_use",
            "tokens_used": _tokens_used_payload(engine),
            "cache_hit_rate": engine.total_usage.this_turn_cache_hit_rate(),
        },
    )


def _post_tool_nudges_spent(engine: QueryEngine) -> int:
    return engine._post_tool_empty_nudge_count


def _charge_post_tool_nudge(engine: QueryEngine) -> int:
    engine._post_tool_empty_nudge_count += 1
    return engine._post_tool_empty_nudge_count


def _reset_post_tool_nudges(engine: QueryEngine) -> None:
    engine._post_tool_empty_nudge_count = 0


def _log_output_token_budget_exhausted(
    engine: QueryEngine, spent: int, budget: int, turn: int
) -> None:
    """Say that the run spent its output-token budget, and by how much."""
    _logger.warning(
        "DIAG query.run_output_token_budget_exhausted run=%s tenant=%s "
        "output_tokens=%d budget=%d turn=%d",
        engine.config.run_id,
        engine.config.tenant_id,
        spent,
        budget,
        turn,
    )


def _empty_completion_redrives_spent(engine: QueryEngine) -> int:
    """How many re-drives this run has already spent on an empty finish."""
    return engine._empty_completion_redrive_count


def _charge_empty_completion_redrive(engine: QueryEngine) -> None:
    """Spend one of them."""
    engine._empty_completion_redrive_count += 1


def _log_answer_floor_repair(
    engine: QueryEngine,
    pointer: tuple[str, int, int] | None,
    attempt: int,
) -> None:
    """Say which of the two answer-floor tests fired, and what it measured.

    Two ways to reach a repair, and they need opposite reading. The floor line
    says the answer was too short and repeats the threshold it was measured
    against; the pointer line says the answer was long enough and still
    delivered nothing, and carries the two sizes that make that case. When
    both hold, the pointer line is the one that explains the run.
    """
    rc = engine.config.rc
    if pointer is None:
        _logger.warning(
            "DIAG query.finalize_prose_gate.plain_stop_repair "
            "run=%s tenant=%s turn=%s floor=%d",
            engine.config.run_id,
            engine.config.tenant_id,
            engine.turn_id(),
            rc.finalize_prose_gate_min_chars,
        )
        return
    path, answer_chars, written_chars = pointer
    _logger.warning(
        "DIAG query.finalize_prose_gate.pointer_answer_repair "
        "run=%s tenant=%s turn=%s attempt=%d/%d "
        "answer_chars=%d written_chars=%d max_fraction=%.3f path=%s",
        engine.config.run_id,
        engine.config.tenant_id,
        engine.turn_id(),
        attempt,
        rc.finalize_prose_gate_pointer_max_repair_attempts,
        answer_chars,
        written_chars,
        rc.finalize_prose_gate_pointer_max_answer_fraction,
        path,
    )


def _spend_short_answer_repair(engine: QueryEngine) -> None:
    """Spend the run's single repair for an answer that was merely too short."""
    engine._finalize_prose_gate_used = True


def _policy_message_stop(engine: QueryEngine, stop_reason: str) -> TurnEvent:
    """The end-of-turn frame, on a policy that has ended the turn."""
    return TurnEvent(
        type=EventType.MESSAGE_STOP,
        run_id=engine.config.run_id,
        payload={"turn_id": engine.turn_id(), "stop_reason": stop_reason},
    )


def _policy_pair_orphan_tool_calls(engine: QueryEngine) -> None:
    """Pair every tool_use that will never receive a result.

    A terminal reached from inside a turn leaves the calls the loop never got
    to, and a snapshot carrying an orphan tool_use is not a transcript any
    consumer can read.
    """
    _synthesize_missing_tool_results(
        engine.history,
        error_content=engine.prompt_text("tool_result_interrupted"),
    )


def _record_nothing() -> None:
    """The default at a coordinate where there is no partial round to keep."""


def _policy_state_change(engine: QueryEngine, reason: str) -> TurnEvent:
    """A state-change event reporting no transition, on a policy's say-so."""
    return _emit_state_change(engine, engine.state, engine.state, reason=reason)


def _turn_at(
    engine: QueryEngine,
    flags: TurnFlags,
    coordinate: TurnCoordinate,
    *,
    turn_budget: int = 0,
    stored_stream_error: tuple[BaseException, str] | None = None,
    text_emitted: bool = False,
    reasoning_emitted: bool = False,
    reasoning_chars: int = 0,
    tool_calls_pending: bool = False,
    tool_results_ready: bool = False,
    record_partial_attempt: Callable[[], None] = _record_nothing,
    terminal_tool_finished: bool = False,
    dispatched_tools: bool = False,
    pending_tool_calls: Sequence[ToolCall] = (),
    finish_reason: str = "",
    stream_error: BaseException | None = None,
    cancel_checkpoint: Callable[[], AsyncIterator[TurnEvent]] | None = None,
    stream_repeat_guard: Callable[[], tuple[str, int] | None] | None = None,
) -> TurnContext:
    """One consultation of the turn policies, at ``coordinate``.

    The engine goes in as itself; the annotation on
    :attr:`~protocore.contracts.turn_policy.TurnContext.engine` is what keeps
    a policy to the fourteen names it is allowed to touch.
    """
    turn = TurnContext(
        engine=engine,
        flags=flags,
        coordinate=coordinate,
        turn_budget=turn_budget,
        stored_stream_error=stored_stream_error,
        text_emitted=text_emitted,
        reasoning_emitted=reasoning_emitted,
        reasoning_chars=reasoning_chars,
        tool_calls_pending=tool_calls_pending,
        tool_results_ready=tool_results_ready,
        terminal_tool_finished=terminal_tool_finished,
        dispatched_tools=dispatched_tools,
        record_partial_attempt=record_partial_attempt,
        pending_tool_calls=pending_tool_calls,
        finish_reason=finish_reason,
        stream_error=stream_error,
    )
    if cancel_checkpoint is not None:
        turn.cancel_checkpoint = cancel_checkpoint
    if stream_repeat_guard is not None:
        turn.stream_repeat_guard = stream_repeat_guard
    return turn


def _turn_budget_after(
    outcome: TurnPolicyOutcome, max_messages: int, flags: TurnFlags
) -> int:
    """The turn budget a policy's ``restart_turn`` leaves behind.

    A policy either replaces the budget outright — the wind-down does, because
    its whole point is a different, smaller one — or asks for one more message
    than the loop is currently allowed, which is what every bounded re-drive
    wants and what a bare ``max_messages + 1`` would get wrong on the second
    re-drive of the same turn.
    """
    if outcome.turn_budget is not None:
        return outcome.turn_budget
    if outcome.extra_turn:
        return max(max_messages, flags.assistant_message_idx + 1)
    return max_messages


def _effective_policies(engine: QueryEngine) -> TurnPolicyRegistry:
    """The policy set this run is driven by.

    A host installs policies to REPLACE a decision the core makes, not to
    remove the bounds the core keeps. The turn cap, the compaction gate and
    the guard on a finish that produced no answer live in policies now, and a
    set that simply took the core's place would leave the driver's ``while``
    with nothing to stop it — silently, because a registry that is missing a
    bound looks exactly like one that never had it. So an installed set is
    merged in by name: one of the host's displaces the core policy that
    answers to the same name, a name the core does not carry is added, and
    nothing is dropped.
    """
    installed = engine.turn_policies
    if installed is None:
        return _CORE_TURN_POLICIES
    return _CORE_TURN_POLICIES.merged_with(installed)


def _finish_taken_over(turn: TurnContext) -> bool:
    """Whether a finish policy took the completion over from the loop.

    The finish seams honour ``end_turn`` — a policy that ends the turn has
    already emitted its terminal events, and sealing the run COMPLETED over
    them would report a second, different ending. ``restart_turn`` they cannot
    honour, and a directive the loop cannot obey is refused out loud rather
    than dropped.
    """
    directive = turn.outcome.directive
    if directive is TurnDirective.restart_turn:
        raise UnsupportedTurnDirectiveError(turn.coordinate, directive)
    return directive is TurnDirective.end_turn


async def _emit_voluntary_completion(
    engine: QueryEngine,
) -> AsyncIterator[TurnEvent]:
    """Close a run the model finished of its own accord.

    Names the wind-down when one is running: ``stop_reason="soft_stop"`` rather
    than ``end_turn``, preceded by the ``soft_stop_finalized`` state change. The
    two are different facts and a consumer needs both — the run DID finish, with
    an answer, and it finished because it was made to.
    """
    finalized = _soft_stop.finalize(engine)
    if finalized is not None:
        yield finalized
    yield TurnEvent(
        type=EventType.MESSAGE_STOP,
        run_id=engine.config.run_id,
        payload={
            "turn_id": engine.turn_id(),
            "stop_reason": (
                _soft_stop.STOP_REASON
                if _soft_stop.is_armed(engine)
                else "end_turn"
            ),
            "tokens_used": _tokens_used_payload(engine),
            "cache_hit_rate": engine.total_usage.this_turn_cache_hit_rate(),
        },
    )


#: The seams a run passes through when the model finishes without calling a
#: tool, in the order it passes them. Declared once because the order IS the
#: behaviour: an answer is held to its floor before the finish that carries
#: it is allowed, and a nudge towards the terminal tool comes before both.
_FINISH_SEAMS: Final[tuple[TurnCoordinate, ...]] = (
    TurnCoordinate.turn_end,
    TurnCoordinate.finish_nudge,
    TurnCoordinate.answer_floor,
    TurnCoordinate.voluntary_finish,
)


async def _finalise_or_verify_first(
    engine: QueryEngine, flags: TurnFlags
) -> bool:
    """One bounded corrective turn before a terminal result closes the run.

    Every dispatch path reaches the same question when a call comes back
    terminal — is the run finished, or does it owe itself one check first? —
    and each of them used to answer it in its own words. ``True`` means the
    corrective turn went in, so the batch stops WITHOUT finalising and the
    caller grants it a message slot; the helper persists the snapshot itself
    on injection. ``False`` means the run is finished, and says so on the
    flags the turn shares with its policies.
    """
    if await _maybe_inject_pre_terminal_self_verify(engine):
        return True
    flags.terminal_tool_completed = True
    return False


async def _cancel_checkpoint(
    engine: QueryEngine,
    flags: TurnFlags,
    policies: TurnPolicyRegistry,
) -> AsyncIterator[TurnEvent]:
    """Ask whether the run has been told to stop, and end it if it has.

    Placed at every seam whose next step would be a NEW side effect. The
    caller reads :attr:`TurnFlags.terminal_yielded` afterwards and returns
    when it is set — the policy has already emitted the whole ending.
    """
    turn = _turn_at(engine, flags, TurnCoordinate.cancel_checkpoint)
    async for event in policies.apply(turn):
        yield event
    if turn.outcome.directive is TurnDirective.end_turn:
        flags.terminal_yielded = True


async def _complete_via_terminal_tool(
    engine: QueryEngine,
    flags: TurnFlags,
    policies: TurnPolicyRegistry,
) -> AsyncIterator[TurnEvent]:
    """Close a run the model ended by calling the tool that ends runs.

    Three dispatch paths reach this ending, and what the ending IS belongs to
    the policy consulted here rather than to whichever path arrived: the
    pairing of the calls the turn abandoned and the seal itself are one
    decision, made once. All this seam owns is refusing a directive it cannot
    obey — a finish is not a place a turn can be sent back from.
    """
    turn = _turn_at(
        engine,
        flags,
        TurnCoordinate.terminal_tool_finish,
        terminal_tool_finished=True,
    )
    async for event in policies.apply(turn):
        yield event
    _finish_taken_over(turn)


def _tool_call_budget_reached(engine: QueryEngine) -> bool:
    """True once this run has dispatched its whole cumulative tool-call budget.

    Counted off the tool-call ledger's ordinal, which advances once per
    dispatched call in transcript order on both the serial and the parallel
    path, and is persisted — so a run re-driven on another pod does not get a
    fresh budget. The leader and each subagent run in separate engines with
    separate ledgers, so a subagent's internal calls are its own.
    """
    cap = (
        engine.config.rc.subagent_tool_call_soft_cap
        if engine.config.parent_run_id is not None
        else engine.config.rc.leader_tool_call_soft_cap
    )
    if cap <= 0:
        return False
    return engine._tool_call_ledger_seq >= cap



def _llm_history(engine: QueryEngine) -> tuple[list[Message], list[str]]:
    """History view for the next LLM request (eviction never mutates persist)."""
    from protocore.runtime.compact_checkpoint import apply_checkpoint

    view, evicted = evict_history_for_llm(
        engine.history,
        engine.config.rc,
        engine.prompt_provider,
        engine._pinned_tool_result_ids,
        roles=engine.config.tool_roles,
    )
    view = apply_checkpoint(view, getattr(engine, "compact_checkpoint", None))
    if engine.config.rc.tool_result_split_enabled:
        from protocore.contracts.types import ToolResultBlock
        from protocore.runtime.tool_result_split import project_result_content

        split_view: list[Message] = []
        for message in view:
            new_blocks: list[ContentBlock] = []
            changed = False
            for block in message.content_blocks:
                if isinstance(block, ToolResultBlock):
                    projection = project_result_content(
                        block.content,
                        rc=engine.config.rc,
                        canonical_ref=block.canonical_ref,
                    )
                    if projection.is_shortened:
                        # The projection replaces the text in the VIEW only.
                        # ``engine.history`` still holds the whole value, which
                        # is what lets the next build shorten it differently —
                        # or not at all, once compaction has moved the value to
                        # a blob and left a reference in its place.
                        new_blocks.append(
                            block.model_copy(update={"content": projection.content})
                        )
                        changed = True
                        continue
                new_blocks.append(block)
            split_view.append(
                message.model_copy(update={"content_blocks": new_blocks})
                if changed
                else message
            )
        view = split_view
    return view, evicted


def _maybe_run_settled_event(engine: QueryEngine) -> TurnEvent | None:
    if not engine.config.rc.run_settled_enabled or engine._run_settled_emitted:
        return None
    if engine.state is LoopState.COMPACTING:
        return None
    engine._run_settled_emitted = True
    return TurnEvent(
        type=EventType.RUN_SETTLED,
        run_id=engine.config.run_id,
        payload={
            "turn_id": engine.turn_id(),
            "phase_was": engine.state.value,
            "will_continue": False,
            # The two facts about the run that neither ``status`` nor
            # ``stop_reason`` carries: was the user answered, and what did the
            # run actually call. Both ride the settle event because it fires
            # once per run, unlike ``message_stop`` which fires per round.
            "has_final_answer": engine.has_final_answer,
            "tool_calls": engine.tool_call_ledger,
            "tool_calls_truncated": engine.tool_call_ledger_truncated,
        },
    )


def _strip_stream_repeat(
    engine: QueryEngine,
    stream_result: Any,
) -> tuple[str, int] | None:
    """Take a repeated tail off the round in flight, and say what went.

    Rewriting the buffers is bookkeeping on a round the loop is holding; what
    a repetition MEANS for the run — a nudge, and past a bound the refusal of
    the calls that came with it — belongs to the policy this probe is offered
    to.
    """
    new_text, new_reason, hit = inspect_stream_repeat(
        stream_result.text_buffer,
        stream_result.reasoning_buffer,
        engine.config.rc,
    )
    if hit is None:
        return None
    stream_result._text_fragments = [new_text] if new_text else []
    stream_result._reasoning_fragments = [new_reason] if new_reason else []
    return str(hit.kind), int(hit.stripped_chars)


def _loop_guard_nudges_spent(engine: QueryEngine) -> int:
    return engine._loop_guard_nudge_count


def _charge_loop_guard_nudge(engine: QueryEngine) -> int:
    engine._loop_guard_nudge_count += 1
    return engine._loop_guard_nudge_count


def _reset_loop_guard_nudges(engine: QueryEngine) -> None:
    engine._loop_guard_nudge_count = 0


def _policy_loop_guard_event(
    engine: QueryEngine, kind: str, nudge_index: int, stripped_chars: int
) -> TurnEvent:
    return TurnEvent(
        type=EventType.LOOP_GUARD_FIRED,
        run_id=engine.config.run_id,
        payload={
            "kind": kind,
            "nudge_index": nudge_index,
            "stripped_chars": stripped_chars,
        },
    )


def _block_identical_tools(
    engine: QueryEngine,
    tool_calls: Sequence[ToolCall],
    nudge_index: int = 0,
) -> tuple[list[ToolCall], list[TurnEvent]]:
    """Split tool calls into executable vs blocked-identical, emitting results."""
    executable: list[ToolCall] = []
    events: list[TurnEvent] = []
    rc = engine.config.rc
    for call in tool_calls:
        fingerprint = canonical_tool_fingerprint(call.name, call.arguments)
        blocked = identical_tool_should_block(
            fingerprint, engine._identical_tool_counts, rc
        )
        engine._identical_tool_counts[fingerprint] = (
            engine._identical_tool_counts.get(fingerprint, 0) + 1
        )
        if not blocked:
            executable.append(call)
            continue
        engine.history.append(
            Message(
                role=MessageRole.tool,
                content_blocks=[
                    ToolResultBlock(
                        tool_call_id=call.id,
                        content=(
                            "identical tool call blocked by loop guard; "
                            "change arguments or stop repeating this call"
                        ),
                        is_error=True,
                        metadata={"loop_guard": "identical_tool"},
                    )
                ],
            )
        )
        events.append(
            TurnEvent(
                type=EventType.TOOL_RESULT,
                run_id=engine.config.run_id,
                payload={
                    "tool_call_id": call.id,
                    "success": False,
                    "is_error": True,
                    "content_blocks": [
                        {
                            "type": "text",
                            "text": "identical tool call blocked by loop guard",
                        }
                    ],
                },
            )
        )
        events.append(
            TurnEvent(
                type=EventType.LOOP_GUARD_FIRED,
                run_id=engine.config.run_id,
                payload={
                    "kind": "identical_tool",
                    "nudge_index": nudge_index,
                    "tool_call_id": call.id,
                    "tool_name": call.name,
                },
            )
        )
    return executable, events


def _rule_project_roots(engine: QueryEngine) -> tuple[str, ...]:
    raw = getattr(engine, "rule_project_roots", ()) or ()
    return tuple(str(item) for item in raw)


def _rule_file_tuples(engine: QueryEngine) -> list[tuple[str, str]]:
    files = getattr(engine, "rule_files", None)
    if not files:
        return []
    out: list[tuple[str, str]] = []
    for item in files:
        out.append((str(item[0]), str(item[1])))
    return out


async def _populate_discovered_rules(engine: QueryEngine) -> None:
    """Assign ``engine.discovered_rules`` from the workspace/project tree once."""
    if not engine.config.rc.rules_discovery_enabled:
        return
    if engine.discovered_rules:
        return
    from protocore.runtime.rules_activation import discover_agents_md

    files = _rule_file_tuples(engine)
    if not files:
        loader = getattr(engine, "list_rule_files", None)
        if callable(loader):
            loaded = loader()
            if inspect.isawaitable(loaded):
                loaded = await loaded
            files = [(str(item[0]), str(item[1])) for item in (loaded or [])]
    if not files:
        return
    engine.discovered_rules = discover_agents_md(
        files, engine.config.rc, project_roots=_rule_project_roots(engine)
    )


def _activate_rules_from_tool(engine: QueryEngine, tool_call: ToolCall) -> None:
    from protocore.runtime.rules_activation import activate_on_filesystem_touch, discover_agents_md

    roles = engine.config.tool_roles
    if not roles.has_any_role(
        tool_call.name, WORKSPACE_INSPECTION_ROLES | WORKSPACE_MUTATION_ROLES
    ):
        return
    args = tool_call.arguments if isinstance(tool_call.arguments, dict) else {}
    path = string_argument(args, ToolArgumentSlot.path, roles=roles) or str(
        args.get("pattern") or ""
    )
    if not path:
        return
    if engine.config.rc.rules_discovery_enabled and not engine.discovered_rules:
        files = _rule_file_tuples(engine)
        if files:
            engine.discovered_rules = discover_agents_md(
                files, engine.config.rc, project_roots=_rule_project_roots(engine)
            )
    before = list(engine.active_rule_paths)
    engine.active_rule_paths = activate_on_filesystem_touch(
        touched_path=path,
        tool_name=tool_call.name,
        discovered=list(engine.discovered_rules),
        already_active=engine.active_rule_paths,
        rc=engine.config.rc,
        roles=roles,
    )
    if engine.active_rule_paths != before:
        engine._pending_rules_activated = [
            path for path in engine.active_rule_paths if path not in before
        ]


#: Background tasks are enabled for this run and no pool was injected at all.
BACKGROUND_DETACHED_NO_POOL = "no_pool_bound"
#: A pool is bound but it does not yet speak for this session — the shape a
#: resumed run takes when the host has not re-attached the session's commands.
BACKGROUND_DETACHED_SESSION_NOT_REATTACHED = "session_not_reattached"
#: The pool speaks for the session but does not hold commands this run's own
#: durable state says were still running. The shape a cold resume takes against
#: a pool that keeps its records in memory: it came up empty and, from the
#: inside, empty and idle are the same picture. Only the run can tell them apart.
BACKGROUND_DETACHED_TASKS_NOT_READOPTED = "tasks_not_readopted"


@dataclass(frozen=True, slots=True)
class _BackgroundWakeOutcome:
    """What the wake check found, told apart from finding nothing.

    ``task_ids`` empty with ``detached_reason`` empty is the ordinary answer:
    the pool speaks for the session and nothing has finished. ``detached_reason``
    set is the answer that used to look identical from the caller's side and is
    not — nobody can say what this session's background commands are doing.

    The two are independent, and that matters: one stranded command must not
    silence the wakes of every healthy one beside it. A detached report is a
    statement about what the pool CANNOT answer, not a reason to stop asking it
    what it can.
    """

    task_ids: tuple[str, ...] = ()
    detached_reason: str = ""


async def _maybe_place_background_wakes(
    engine: QueryEngine,
) -> _BackgroundWakeOutcome:
    """Refresh the session pool and inject one batched wake turn if needed.

    Reports a detached pool rather than an empty result. The two used to be the
    same value, and that is the whole defect: a run resumed on a fresh process
    came up with no pool bound to its session, every wake check answered "nothing
    finished", and the agent waited on a command that had finished before the
    run was even resumed. Background work only ever reaches the agent through
    this function, so silence here is silence everywhere.

    A detached report does not stop the check. Whatever the pool can still
    answer is still asked for and still delivered — one stranded record must not
    take the wakes of every healthy command beside it.
    """
    if not engine.config.rc.background_tasks_enabled:
        return _BackgroundWakeOutcome()
    if engine.config.run_depth > 0:
        # A wake is a turn injected into a conversation, and only the run at
        # the root of the tree has one a person is reading. A child run woken
        # by its own finished work would spend turns of its own budget on it
        # and then report it to nobody, so the notification is a property of
        # depth 0 — the pool still holds the records, and the root still gets
        # its own.
        return _BackgroundWakeOutcome()
    pool = engine.background_pool
    if pool is None:
        return _BackgroundWakeOutcome(detached_reason=BACKGROUND_DETACHED_NO_POOL)
    detached_reason = ""
    if not await pool.ensure_session_attached(engine.config.session_id):
        detached_reason = BACKGROUND_DETACHED_SESSION_NOT_REATTACHED
    elif any(
        pool.get(task_id) is None for task_id in engine._resumed_background_task_ids
    ):
        # The pool vouches for the session, and the commands this run recorded
        # as running are not in it. A pool holding its records in memory has no
        # way to notice that — it came up empty and empty looks exactly like
        # idle — so the run's own durable state is the only witness there is.
        detached_reason = BACKGROUND_DETACHED_TASKS_NOT_READOPTED
    for task in list(pool.list(engine.config.session_id)):
        await pool.refresh(task.id)
    ids: list[str] = list(pool.drain_wakes(engine.config.session_id))
    if not ids:
        return _BackgroundWakeOutcome(detached_reason=detached_reason)
    described = []
    for task_id in ids:
        item = pool.get(task_id)
        if item is not None:
            described.append(describe_finished_task(item))
    text = "background tasks finished: " + "; ".join(described)
    engine.history.append(
        Message(
            role=MessageRole.user,
            content_blocks=[TextBlock(text=text)],
        )
    )
    persister = getattr(engine, "persist_session_history", None)
    if callable(persister):
        persister(engine)
    return _BackgroundWakeOutcome(
        task_ids=tuple(ids), detached_reason=detached_reason
    )


async def _background_wake_events(engine: QueryEngine) -> AsyncIterator[TurnEvent]:
    """Run the wake check and emit what it found, at one site for both callers.

    The two turn-opening paths ask the same question in the same place and used
    to carry their own copy of the answer; a second copy of a fire-once report
    behind a run-scoped latch is unreachable by construction, and unreachable
    code is not a second chance at anything.
    """
    background = await _maybe_place_background_wakes(engine)
    if background.task_ids:
        yield TurnEvent(
            type=EventType.BACKGROUND_WAKE,
            run_id=engine.config.run_id,
            payload={"task_ids": list(background.task_ids)},
        )
    if background.detached_reason and not engine._background_detach_reported:
        engine._background_detach_reported = True
        _logger.warning(
            "DIAG query.background_pool_detached run=%s tenant=%s session=%s reason=%s",
            engine.config.run_id,
            engine.config.tenant_id,
            engine.config.session_id,
            background.detached_reason,
        )
        yield TurnEvent(
            type=EventType.STATE_CHANGED,
            run_id=engine.config.run_id,
            payload={
                "from": engine.state.value,
                "to": engine.state.value,
                "reason": "background_tasks_detached",
                "detached_reason": background.detached_reason,
                "session_id": engine.config.session_id,
            },
        )


async def _reload_live_control(engine: QueryEngine) -> None:
    """Pull mid-run steer / follow-up / model from the bound live store."""
    reloader = getattr(engine, "reload_live_control", None)
    if reloader is None:
        return
    await reloader(engine)


async def _persist_live_control(engine: QueryEngine) -> None:
    persister = getattr(engine, "persist_live_control", None)
    if persister is None:
        return
    await persister(engine)


def _inject_queue_into_history(
    engine: QueryEngine,
    *,
    kind: str,
    raw_queue: list[dict[str, Any]],
    mode: str,
) -> TurnEvent | None:
    if not engine.config.rc.steer_follow_up_enabled or not raw_queue:
        return None
    items = [QueuedPrompt.from_dict(raw) for raw in raw_queue]
    placed, remaining = place_items(items, mode)  # type: ignore[arg-type]
    message = placed_as_user_message(placed)
    if message is None:
        return None
    engine.history.append(message)
    remaining_dicts = [item.to_dict() for item in remaining]
    if kind == "steer":
        engine._steer_queue = remaining_dicts
    else:
        engine._follow_up_queue = remaining_dicts
    return TurnEvent(
        type=EventType.QUEUE_UPDATE,
        run_id=engine.config.run_id,
        payload={
            "placed": [item.id for item in placed],
            "kind": kind,
            "remaining": len(remaining_dicts),
        },
    )


def _inject_steer_into_history(engine: QueryEngine) -> TurnEvent | None:
    return _inject_queue_into_history(
        engine,
        kind="steer",
        raw_queue=engine._steer_queue,
        mode=engine.config.rc.steer_default_mode,
    )


def _inject_follow_up_into_history(engine: QueryEngine) -> TurnEvent | None:
    return _inject_queue_into_history(
        engine,
        kind="follow_up",
        raw_queue=engine._follow_up_queue,
        mode=engine.config.rc.follow_up_default_mode,
    )


def _pin_keep_flag(engine: QueryEngine, tool_call: ToolCall) -> None:
    parsed = tool_call.arguments
    if isinstance(parsed, dict) and parsed.get("keep") is True:
        engine.pin_tool_result(tool_call.id)


# Heartbeat observability for the PRE-DISPATCH terminal-verify gate. The gate
# ran silently on the no-veto path, which hid mis-diagnoses. The heartbeat
# logs an UNCONDITIONAL line on every gate application so "did the gate run,
# and with what verdict?" is answerable from the executor log alone.
#
# ``observed`` (the size of the per-run observed-state collection a trigger
# may compare cited refs against) is a named field on the run's state, so the
# count is always available and never a guess: a run whose read tools recorded
# nothing has an empty one, which is a fact rather than an unknown. ``cited`` is
# computed purely from the un-submitted ``ToolCall`` arguments (the exact input
# the trigger reads: canonical ``refs`` slot, legacy ``sources`` alias), so it
# is always exact too.
_TERMINAL_ANSWER_REFS_KEY: Final[str] = "refs"
_TERMINAL_ANSWER_REFS_LEGACY_ALIAS: Final[str] = "sources"


def _observability_context(
    engine: QueryEngine,
    *,
    call_purpose: str,
    call_category: str,
) -> LLMObservabilityContext:
    return LLMObservabilityContext(
        tenant_id=engine.config.tenant_id,
        run_id=engine.config.run_id,
        parent_run_id=engine.config.parent_run_id,
        session_id=engine.config.session_id,
        agent_id=engine.config.subagent_id,
        call_purpose=call_purpose,
        call_category=call_category,
    )


# The request contract's own default, so the one temperature policy below
# states a value on every path without repeating a literal that already lives
# on the contract.
_DEFAULT_REQUEST_TEMPERATURE: Final[float] = float(
    LLMRequest.model_fields["temperature"].default
)


def build_llm_request(
    *,
    model: str,
    messages: Sequence[Message],
    max_tokens: int,
    tools: Sequence[ToolDefinition] = (),
    temperature: float | None = None,
    thinking_enabled: bool | None = None,
    reasoning_effort: str | None = None,
    forced_tool_choice: str | None = None,
    response_format: Mapping[str, Any] | None = None,
    cache_breakpoints: Sequence[CacheBreakpoint] | None = None,
    observability: LLMObservabilityContext | None = None,
) -> LLMRequest:
    """Assemble the one :class:`LLMRequest` every provider call is made of.

    Every call that leaves this runtime — the action stream, the deep loop's
    plan call, its prompted-JSON plan fallback, and the Tier-2 compaction
    summariser — is built here, so the four agree by construction on the three
    things they used to decide separately:

    * **the model**. Callers pass ``QueryEngine.effective_model_name``, which
      is the live override when one is set and the configured model otherwise.
      Resolving it per call site meant a mid-run model change moved only the
      action stream: one agent turn then spanned two models with nothing on the
      event stream saying so.
    * **the forced tool**. ``extra["forced_tool_choice"]`` is the single slot,
      and it carries the tool NAME; a provider adapter renders it into whatever
      native single-tool ``tool_choice`` shape its wire wants. Stating the same
      intent in two spellings meant a single reader could not tell whether a
      turn had been forced.
    * **the temperature**. Stated on every request: the caller's value, or the
      request contract's default when the caller has no opinion.

    ``thinking_enabled`` and ``reasoning_effort`` travel as a pair or not at
    all — the effort bounds the CoT, and thinking requested without it was
    measured to truncate the answer — so passing exactly one is a programming
    error. Omitting both is how a path (the plan fallback) ships only the knobs
    every provider accepts.

    Keys land in ``extra`` only when the caller supplies them: an adapter that
    does not recognise a key ignores it, but an absent key and a key holding a
    default are different requests, and the manifest of a call has to be able
    to tell them apart.
    """
    if (thinking_enabled is None) != (reasoning_effort is None):
        raise ValueError(
            "thinking_enabled and reasoning_effort must be given together: "
            "the effort bounds the requested chain of thought"
        )
    extra: dict[str, object] = {}
    if cache_breakpoints is not None:
        extra["cache_breakpoints"] = cache_breakpoints
    if thinking_enabled is not None and reasoning_effort is not None:
        extra["enable_thinking"] = thinking_enabled
        extra["reasoning_effort"] = reasoning_effort
    if forced_tool_choice is not None:
        extra["forced_tool_choice"] = forced_tool_choice
    if response_format is not None:
        extra["response_format"] = dict(response_format)
    return LLMRequest(
        model=model,
        messages=list(messages),
        tools=list(tools),
        max_tokens=max_tokens,
        temperature=(
            _DEFAULT_REQUEST_TEMPERATURE if temperature is None else temperature
        ),
        extra=extra,
        observability=observability,
    )


async def _manifest_request(
    engine: QueryEngine,
    request: LLMRequest,
    *,
    call_purpose: str,
) -> RequestManifest | None:
    """Record what this call is made of, before a single delta of it exists.

    The manifest is the answer to the one question a run cut off mid-stream
    could not answer: whether the request that produced the output somebody
    already saw is the request this build would produce again. So it is emitted
    HERE — after the outbound list is final and before the provider is asked —
    and the id is stamped on the run's snapshot at the same moment, without
    waiting to find out whether the host's store accepted anything.

    Inert with no sink configured: a run nobody is recording pays nothing, not
    even the hashing of its history.

    A sink that raises does NOT fail the run. Manifesting a request is
    evidence-keeping, and a store that is full or unreachable is the host's
    operational problem; ending an agent's turn over it would trade a missing
    record for a failed run. The id is stamped either way, so the snapshot
    still names the call — a reader that cannot then find the manifest learns
    that it was not kept, which is a different and more useful fact than a run
    with no reference at all.
    """
    sink = engine.config.request_manifest_sink
    if sink is None:
        return None
    chain = engine.provider_chain
    manifest, bodies = build_request_manifest(
        request=request,
        attempt_scope=f"{engine.config.run_id}/{engine.turn_id()}/{call_purpose}",
        constants_sha256=constants_digest(engine.config.rc),
        inline_value_max_bytes=(
            engine.config.rc.request_manifest_inline_value_max_bytes
        ),
        provider_chain_position=engine._provider_chain_advances,
        provider_chain_model=(
            chain.current_model_name() if chain is not None else None
        ),
    )
    engine.note_request_manifest(manifest)
    try:
        await sink.record_request_manifest(
            manifest=manifest,
            manifest_id=manifest.manifest_id,
            bodies=bodies,
        )
    except Exception as exc:
        _logger.warning(
            "request manifest sink failed for run=%s attempt=%s (err=%s); the "
            "run continues without the record",
            engine.config.run_id,
            manifest.attempt_id,
            exc,
        )
    return manifest


def _provider_call_category(engine: QueryEngine) -> str:
    if engine.config.parent_run_id is not None:
        return "subagent_call"
    return "agent_call"


def _tool_surface_advertised_payload(
    engine: QueryEngine,
    context: ContextBundle,
) -> dict[str, object]:
    """The exact tool list sent to the provider, and what each of those tools does.

    The roles ride along because the runtime is no longer the only reader that
    needs them. A client renders an approval card red for a command it is about
    to let a person authorise, dims a surface a plan-only run cannot write
    with, and picks an icon per tool — and each of those was, until now, its own
    copy of a list of names, kept in step with this deployment by nobody. A
    scope that renamed its shell tool got a destructive command drawn as
    something harmless. The map is the host's answer to that question, so it is
    the thing to publish, rather than leaving every reader to guess again.
    """

    policy = engine.effective_tool_policy
    roles = engine.config.tool_roles
    toolsearch_pins = frozenset(engine.context_manager.pinned_tool_names())
    forced_pins = frozenset(policy.forced_pinned)
    configured_pins = frozenset(policy.pinned) - toolsearch_pins
    tool_names = [tool.name for tool in context.tools]
    tools: list[dict[str, object]] = []
    for tool in context.tools:
        sources: list[str] = []
        if tool.name in toolsearch_pins:
            sources.append("toolsearch_pin")
        if tool.name in configured_pins:
            sources.append("configured_pin")
        if tool.name in forced_pins:
            sources.append("forced_pin")
        if not sources:
            sources.append("retrieved_or_visible")
        tools.append(
            {
                "name": tool.name,
                "description": tool.description,
                "sources": sources,
                "roles": sorted(role.value for role in roles.roles_of(tool.name)),
            }
        )
    return {
        "turn_id": engine.turn_id(),
        "tool_count": len(tool_names),
        "tool_names": tool_names,
        "toolsearch_pinned_tool_names": sorted(toolsearch_pins),
        "configured_pinned_tool_names": sorted(configured_pins),
        "forced_pinned_tool_names": sorted(forced_pins),
        "retrieval_top_k": engine.config.rc.tool_retrieval_top_k,
        "argument_names": {
            slot.value: list(names)
            for slot, names in sorted(roles.argument_aliases.items())
        },
        "tools": tools,
    }


async def resume(
    engine: QueryEngine,
    snapshot: dict[str, Any],
    *,
    approved_tool_call: ToolCall | None = None,
    message: Message | None = None,
    abandon_approval: bool = False,
    resolutions: Mapping[str, InterruptResolution] | None = None,
    allow_partial_resolution: bool = False,
) -> AsyncIterator[TurnEvent]:
    """Pick a stored run back up and drive it — the one public resume entry.

    A run that stopped is stopped in one of three ways, and each needs a
    different drive.  Before this entry existed the choice between them lived
    in the host: it rehydrated with
    :meth:`~protocore.runtime.query_engine.QueryEngine.resume_from_snapshot`
    and then picked one of the drivers itself, which meant the guarantee a
    resumed turn got — whether ``stop()`` could reach it, whether it left a
    pickup point behind — depended on which driver the host happened to reach
    for.  Here the choice is made from what the caller supplies, and whichever
    branch is taken the drive carries the same cancellation handle and the same
    closing snapshot.

    ``snapshot`` is restored first and restored strictly: the identity binding
    (run, tenant, session and the subagent lineage), the delivery mode and the
    schema are all settled before the first mutation, so a snapshot belonging
    to another run is refused with the engine untouched.  Nothing is driven
    when the restore refuses.

    The four drives:

    * ``resolutions`` — the run stopped on one or more typed interrupts and
      every one of them now has an answer.  This is the general form: a map
      from interrupt id to :class:`~protocore.contracts.interrupt.InterruptResolution`,
      which is what lets three calls parked together be approved, denied and
      corrected in ONE resume instead of three rounds of stop-ask-resume.  A
      map that leaves an open interrupt undecided is refused unless
      ``allow_partial_resolution`` says that is deliberate; a map naming an
      interrupt that is not open, or answering an approval with an answer, is
      refused outright.
    * ``approved_tool_call`` — the run stopped waiting for a decision on that
      call and the decision was *approve*.  The call is executed once, through
      :func:`resume_approved_tool`, which verifies it against the durable
      pending call rather than trusting the argument.
    * ``message`` — the run stopped waiting for input that has now arrived
      (an answered question, an operator's reply).  The message opens a fresh
      turn, as it would on a live engine.
    * neither — the run stopped mid-turn with its history already ending in the
      input it owes an answer to; the turn is simply re-driven.

    ``approved_tool_call``, ``message`` and ``resolutions`` are alternatives,
    not a sequence: a caller that supplies more than one has not decided which
    of several different things happened, and gets a :class:`ValueError`
    instead of the engine's guess.

    A run restored in ``AWAITING`` is a run that was waiting, and the state
    that recorded the wait is cleared here rather than left for the caller to
    clear: the loop refuses to open a turn from ``AWAITING``.

    A pending approval is not cleared with it.  A re-drive and an arriving
    message are both news that something happened elsewhere; neither is a
    decision on the call an operator was asked about.  Clearing the latch
    anyway leaves the call parked in history with no result, and the wire
    repair then fills the gap with a synthetic failure — telling the model
    that a call nobody approved was attempted and failed, which is the one
    thing the durable record exists to prevent.  So a non-approval drive over
    a pending approval is refused, naming the call.

    ``abandon_approval`` is how a caller says the decision will never come:
    the parked call is closed with a result saying it was never approved and
    never ran, which is true and leaves nothing for the repair to invent.  It
    is a deliberate act, not a default, because the alternative reading — the
    operator has not answered yet — is the common one.

    The approved-call drive stops when the call's result lands in history; it
    does not go on to answer it.  Producing that answer is a fresh turn against
    the restored history and belongs to the caller, which is the one that knows
    whether it still wants it.
    """
    if approved_tool_call is not None and message is not None:
        raise ValueError(
            "resume() takes an approved tool call or a message, not both: "
            "an approved call resumes the tool that was waiting, a message "
            "opens a new turn."
        )
    if approved_tool_call is not None and abandon_approval:
        raise ValueError(
            "resume() cannot both execute an approved tool call and abandon "
            "the approval it was waiting for."
        )
    if resolutions is not None and (
        approved_tool_call is not None or message is not None or abandon_approval
    ):
        raise ValueError(
            "resume() takes a resolution map or one of the single-answer "
            "arguments, not both: the map already says, per interrupt, which "
            "of approve, deny, answer and abandon happened."
        )
    if allow_partial_resolution and resolutions is None:
        raise ValueError(
            "allow_partial_resolution says which interrupts a resolution map "
            "may leave parked, and there is no map to say it about."
        )

    await engine.resume_from_snapshot(snapshot)

    if resolutions is not None:
        async for event in resume_interrupts(
            engine, resolutions, allow_partial=allow_partial_resolution
        ):
            yield event
        return

    if approved_tool_call is not None:
        async for event in resume_approved_tool(engine, approved_tool_call):
            yield event
        return

    if engine.state is LoopState.AWAITING:
        parked = engine.pending_interrupts
        answerable = tuple(
            item for item in parked if item.kind is not InterruptKind.approval
        )
        if abandon_approval:
            for item in parked:
                _abandon_pending_approval(engine, item.tool_call_id)
                engine.release_interrupt(item.interrupt_id)
        elif message is not None and len(parked) == len(answerable) == 1:
            # One question outstanding and the answer has arrived. It settles
            # the call that asked it — that is what the tool was waiting for —
            # and the same text then opens the turn that reacts to it, which is
            # what the caller asked for by passing a message rather than a
            # resolution map.
            waiting = parked[0]
            _settle_parked_call(
                engine,
                tool_call_id=waiting.tool_call_id,
                content=_message_text(message),
                is_error=False,
            )
            engine.release_interrupt(waiting.interrupt_id)
        elif parked:
            raise ValueError(
                "this run is waiting on "
                f"{[(item.interrupt_id, item.kind.value, item.tool_call_id) for item in parked]}"
                "; resuming without answering would leave those calls with no "
                "result, and the wire repair would then report calls nobody "
                "decided on, and questions the model really asked, to the "
                "model as failures. Pass resolutions to answer them all, "
                "approved_tool_call for the single-approval case, or "
                "abandon_approval=True to close them as never answered."
            )
        engine.transition_to(LoopState.RUNNING)

    async for event in engine.run(message):
        yield event


def _abandon_pending_approval(engine: QueryEngine, tool_call_id: str) -> None:
    """Close a parked call whose approval the caller says will never come.

    Writes the one true thing about it — it was never approved, so it never
    ran and changed nothing — into history, and settles the durable record
    that was holding it. Both halves matter: a record left standing keeps the
    call open forever, and a ``tool_use`` left unpaired is filled in on the
    wire with a synthetic failure, which says the opposite of the truth.
    """
    text = engine.config.rc.tool_result_approval_abandoned_placeholder
    _insert_tool_result_after_use(
        engine.history,
        tool_call_id=tool_call_id,
        content=text,
        is_error=False,
    )
    intent = find_intent(engine.open_intents, tool_call_id)
    if intent is not None:
        settle_intent(intent, result=text)
        _forget_intent(engine, intent)
    engine.forget_tool_name(tool_call_id)


def _query(engine: QueryEngine) -> AsyncIterator[TurnEvent]:
    """Drive one already-prepared turn through the public delivery boundary.

    Internal, and internal on purpose.  ``QueryEngine.run`` owns initial-message
    admission — the input message, the turn number, the run clock and the
    turn-start snapshot — and then consumes the same private generator
    directly.  This lower-level entry leaves that admission to its caller and
    skips two further obligations ``run`` holds: ``_current_turn_task`` is never
    bound, so :meth:`QueryEngine.stop` has no handle to cancel through and only
    its cooperative flag fires, and no turn-start or turn-end snapshot is
    persisted, so a turn driven here has no cross-pod pickup point.  A turn
    driven through it therefore gets weaker cancellation and durability than
    one driven through :meth:`QueryEngine.run` or :func:`resume`, which is why
    it is not part of the package's public surface: an entry offering less than
    its neighbours must not be the one a host reaches for first.

    What it does still owe, it pays: the per-turn counters and latches, which a
    caller cannot reach because they are private to the engine, it puts back
    itself via
    :meth:`~protocore.runtime.query_engine.QueryEngine._reset_per_turn_state`.
    Without that, a turn opened here on an engine nudged in an earlier turn
    began already finalising and deleted its own answer as post-answer
    narration.  It must also never become an alternate route around
    verification-gated reader delivery.

    Deliberately NOT an async generator.  ``run`` is one, so its reset lands on
    the first ``__anext__`` rather than at the call — harmless there because
    every live caller iterates immediately, but a caveat that stops being
    harmless once it holds at more than one entry.  Returning the generator
    instead of being one makes the reset here happen when ``_query`` is called,
    so there is still exactly one place where a built-but-not-yet-iterated turn
    carries last turn's state.
    """
    engine._reset_per_turn_state()
    return _projected_turn_events(engine)


async def _projected_turn_events(engine: QueryEngine) -> AsyncIterator[TurnEvent]:
    """The body of :func:`_query`, split out so the reset above stays eager."""
    async for event in _query_raw(engine):
        for projected in engine._project_public_turn_event(event):
            yield projected


def _hook_denied_stop(
    engine: QueryEngine, point: HookEvent, outcome: LifecycleOutcome
) -> TurnEvent:
    """The stop a non-``allow`` verdict at ``point`` ends the turn with."""
    return TurnEvent(
        type=EventType.MESSAGE_STOP,
        run_id=engine.config.run_id,
        payload={
            "stop_reason": "hook_denied",
            "hook": point.value,
            "verdict": outcome.verdict.value,
            "reason": outcome.reason,
            "decided_by": outcome.decided_by,
        },
    )


def _apply_context_transform(
    context: ContextBundle, payload: Mapping[str, Any]
) -> ContextBundle:
    """Rebuild the bundle from what the ``context_transform`` chain returned.

    Only the two fields a transform is allowed to speak about are read, and
    each only when it came back as the right shape: a handler that returns
    nothing, or returns a field it has no business setting, leaves the bundle
    exactly as it was.
    """
    sections = payload.get("system_prompt_sections")
    language = payload.get("active_language")
    updates: dict[str, Any] = {}
    if isinstance(sections, (list, tuple)) and all(
        isinstance(item, str) for item in sections
    ):
        replacement = tuple(sections)
        if replacement != context.system_prompt_sections:
            updates["system_prompt_sections"] = replacement
    if isinstance(language, str) and language and language != context.active_language:
        updates["active_language"] = language
    if not updates:
        return context
    return replace(context, **updates)


async def _query_raw(engine: QueryEngine) -> AsyncIterator[TurnEvent]:
    """Drive one turn and close it out at the ``run_finalize`` coordinate.

    The coordinate fires however the turn ended — a normal finish, an early
    return, or an exception on its way out — because a finalization seam that
    only runs when nothing went wrong is the one nobody can use for cleanup.
    The exception is held, the coordinate is dispatched, and then the exception
    continues on its way.
    """
    from protocore.runtime.correctness_bind import fire_lifecycle

    error: Exception | None = None
    try:
        async for event in _drive_turn(engine):
            yield event
    except Exception as exc:
        error = exc
    _finalize, finalize_evt = await fire_lifecycle(
        engine,
        HookEvent.run_finalize,
        {
            "run_id": engine.config.run_id,
            "state": engine.state.value,
            "error": str(error) if error is not None else None,
        },
    )
    if finalize_evt is not None:
        yield finalize_evt
    if error is not None:
        raise error


async def _drive_turn(engine: QueryEngine) -> AsyncIterator[TurnEvent]:
    """Drive one full turn of ``engine``. Yields :class:`TurnEvent` envelopes.

    See module docstring for the lifecycle. The function is a Python async
    generator — each ``yield`` is a stop-check checkpoint.
    """
    # ── 1. Stop check ────────────────────────────────────────────────
    if engine.stop_requested:
        # a run resumed with stop already requested may carry a
        # dangling tool_use from the interrupted turn in its rehydrated
        # history; pair it before the cancel terminal so the persisted
        # snapshot stays pairing-valid.
        _synthesize_missing_tool_results(
            engine.history,
            error_content=engine.prompt_text("tool_result_interrupted"),
        )
        restored = restore_queued_prompts(engine)
        await _persist_live_control(engine)
        from protocore.runtime.correctness_bind import commit_usage

        abort_evt = commit_usage(
            engine,
            kind="abort",
            input_tokens=0,
            output_tokens=0,
            success=False,
        )
        if abort_evt is not None:
            yield abort_evt
        yield _emit_state_change(
            engine,
            engine.state,
            LoopState.CANCELLED,
            reason="stop_before_start",
        )
        engine.transition_to(LoopState.CANCELLED)
        yield TurnEvent(
            type=EventType.MESSAGE_STOP,
            run_id=engine.config.run_id,
            payload={
                "turn_id": engine.turn_id(),
                "stop_reason": StopReason.cancelled.value,
                "restored_queue_text": "\n\n".join(restored),
            },
        )
        return

    # PENDING → RUNNING transition
    if engine.state is LoopState.PENDING:
        engine.transition_to(LoopState.RUNNING)
        yield TurnEvent(
            type=EventType.STATE_CHANGED,
            run_id=engine.config.run_id,
            payload={"from": LoopState.PENDING.value, "to": LoopState.RUNNING.value},
        )

    # Per-turn block index reset
    engine.reset_block_idx()
    engine.total_usage.reset_turn()
    persister = getattr(engine, "persist_session_history", None)
    if callable(persister):
        persister(engine)
    await _populate_discovered_rules(engine)
    # Before this turn drives anything, close out any call this run was in the
    # middle of when it last stopped. A run rehydrated on another pod has to
    # say something about a tool call it left in flight, and the only true
    # thing it can say is that the outcome was never recorded.
    if engine.open_intents:
        async for intent_evt in _settle_interrupted_tool_intents(engine):
            yield intent_evt

    from protocore.runtime.correctness_bind import fire_lifecycle

    run_start, run_start_evt = await fire_lifecycle(
        engine, HookEvent.run_start, {"run_id": engine.config.run_id}
    )
    if run_start_evt is not None:
        yield run_start_evt
    if not run_start.allowed:
        yield TurnEvent(
            type=EventType.MESSAGE_STOP,
            run_id=engine.config.run_id,
            payload={
                "stop_reason": "hook_denied",
                "hook": HookEvent.run_start.value,
                "reason": run_start.reason,
            },
        )
        return

    latest = engine.latest_user_message
    if (
        engine.config.rc.compaction_manual_enabled
        and latest is not None
        and latest.text.lstrip().startswith("/compact")
    ):
        from protocore.runtime.compact_checkpoint import build_checkpoint

        instructions = latest.text.lstrip()[len("/compact") :].strip()
        ckpt = build_checkpoint(
            engine.history,
            keep_recent_turns=engine.config.rc.compaction_keep_recent_turns,
            instructions=instructions,
            reason="manual",
            enabled=True,
            tracked_tool_names=engine.config.rc.compaction_tracked_tool_names,
        )
        if ckpt is not None:
            engine.compact_checkpoint = ckpt
            persister = getattr(engine, "persist_session_history", None)
            if callable(persister):
                persister(engine)
            from protocore.runtime.correctness_bind import commit_usage

            usage_evt = commit_usage(
                engine,
                kind="compaction",
                input_tokens=0,
                output_tokens=0,
                success=True,
            )
            if usage_evt is not None:
                yield usage_evt
            yield TurnEvent(
                type=EventType.COMPACT_CHECKPOINT,
                run_id=engine.config.run_id,
                payload=ckpt.to_dict(),
            )

    # ── 2. Compaction check ──────────────────────────────────────────
    # When history already exceeds the emergency cliff
    # (``model_context_window * compaction_emergency_ratio``) at turn-start,
    # run a proactive ``force_compaction`` (both tiers, unconditional) rather
    # than the routine gated pass, so the wire payload is aggressively shrunk
    # before the first stream. RC-gated kill-switch
    # (``compaction_emergency_proactive_enabled``, default on).
    _emergency_turn_start = (
        engine.config.rc.compaction_emergency_proactive_enabled
        and engine.needs_emergency_compaction()
    )
    if _emergency_turn_start or engine.needs_compaction():
        async for evt in _run_compaction(
            engine,
            force=_emergency_turn_start,
            reason="proactive_emergency" if _emergency_turn_start else "routine",
        ):
            yield evt
        # If compaction transitioned to FAILED, surface the terminal
        # message_stop now and bail.
        if engine.state is LoopState.FAILED:
            # a compaction-exhausted FAILED terminal can persist a
            # history whose last assistant turn (or a turn compaction kept)
            # carries a tool_use with no result; pair it before the snapshot.
            _synthesize_missing_tool_results(
                engine.history,
                error_content=engine.prompt_text("tool_result_interrupted"),
            )
            yield TurnEvent(
                type=EventType.MESSAGE_STOP,
                run_id=engine.config.run_id,
                payload={
                    "turn_id": engine.turn_id(),
                    "stop_reason": StopReason.error.value,
                },
            )
            return

    # ── 3. UserPromptSubmit hook ─────────────────────────────────────
    hook_result = await _safe_hook_invoke(
        engine,
        HookEvent.user_prompt_submit,
        {
            "run_id": engine.config.run_id,
            "tenant_id": engine.config.tenant_id,
            "message": engine.latest_user_message.model_dump(mode="json") if engine.latest_user_message else None,
        },
    )
    yield TurnEvent(
        type=EventType.HOOK_FIRED,
        run_id=engine.config.run_id,
        payload={
            "hook_event": HookEvent.user_prompt_submit.value,
            "outcome": "deny" if hook_result.action == HookActionKind.DENY else "success",
        },
    )
    if hook_result.action == HookActionKind.DENY:
        # a resumed history may already carry a dangling tool_use
        # when a UserPromptSubmit hook denies the turn; pair it before the
        # FAILED snapshot so a later resume is wire-valid.
        _synthesize_missing_tool_results(
            engine.history,
            error_content=engine.prompt_text("tool_result_interrupted"),
        )
        from_state = engine.state
        engine.transition_to(LoopState.FAILED)
        yield _emit_state_change(
            engine,
            from_state,
            LoopState.FAILED,
            reason=hook_result.reason or "hook_denied",
        )
        yield TurnEvent(
            type=EventType.ERROR,
            run_id=engine.config.run_id,
            payload={
                "kind": "hook_denied",
                "message": hook_result.reason or "blocked by policy",
            },
        )
        yield TurnEvent(
            type=EventType.MESSAGE_STOP,
            run_id=engine.config.run_id,
            payload={
                "turn_id": engine.turn_id(),
                "stop_reason": StopReason.error.value,
            },
        )
        return

    # ── 4. Build context bundle ──────────────────────────────────────
    # ``effective_tool_policy`` applies the RC core tool-surface floor so
    # the six core file tools survive the BM25 RU clip on the live engine path.
    tool_defs = list(
        engine.tools.compute_effective_surface(
            tenant_id=engine.config.tenant_id,
            policy=engine.effective_tool_policy,
            query=engine.latest_user_message.text if engine.latest_user_message else "",
            top_k=engine.config.rc.tool_retrieval_top_k,
        )
    )

    # Skill catalog: the account's enabled skills (plus project pins) rendered
    # into a stable ``<system-reminder>`` block, built ONCE per run (cached on
    # the engine) and placed in the static system-prompt prefix so it is
    # byte-identical across turns + the inner agent loop + every recovery
    # rebuild (keeps the prompt cache; no redundant per-turn store round-trip).
    skill_catalog_block = await _ensure_run_skill_catalog(engine)
    if engine.config.rc.rules_discovery_enabled and engine.active_rule_paths:
        from protocore.runtime.rules_activation import bodies_for_prompt

        rule_bodies = bodies_for_prompt(
            list(engine.discovered_rules),
            engine.active_rule_paths,
            engine.config.rc,
        )
        if rule_bodies:
            skill_catalog_block = (skill_catalog_block + "\n" if skill_catalog_block else "") + "\n".join(rule_bodies)
    # Loaded bundles: rebuilt EVERY turn from THIS turn's user message —
    # trigger-style `<command-name>NAME</command-name>` references load the
    # full skill body as Layer 3. Per-turn, NOT cached on the engine.
    if engine.skills is not None and engine.latest_user_message is not None:
        engine._skill_loaded_bundles = await _load_triggered_skill_bodies(
            engine, engine.skills, engine.latest_user_message.text
        )
    else:
        engine._skill_loaded_bundles = []

    llm_history, evicted_ids = _llm_history(engine)
    context = engine.context_manager.build_context(
        history=llm_history,
        tools=tool_defs,
        system_prompt_sections=engine.config.system_prompt_sections,
        skill_index_block=skill_catalog_block,
        skills_loaded=engine._skill_loaded_bundles,
    )
    if evicted_ids:
        yield TurnEvent(
            type=EventType.TOOL_RESULT_EVICTED,
            run_id=engine.config.run_id,
            payload={"tool_call_ids": evicted_ids},
        )

    # #1/#4 — open the FIRST assistant-message wire round HERE, after every
    # pre-loop terminal (``stop_before_start`` L~254, compaction-failed L~314,
    # hook-denied L~358 — all keep the LEGACY ``turn-{run}-{turn_count}`` id with
    # ``_wire_round_seq == 0``) and BEFORE the Deep loop-strategy 4b step below.
    # The Deep ``REASONING_STEP`` (``loop_strategies._reasoning_step_payload``)
    # reads ``engine.turn_id()``; the chat reducer keys the SGR plan placeholder
    # by that id and later MERGES the first ``message_start`` into it (reducer
    # ``reasoning_step`` → ``message_start``), so the plan frame and round 1's
    # ``message_start`` MUST share one id. ``begin_wire_round()`` advances the
    # round to 1 (so every frame of round 1 reads the suffixed id) and restarts
    # ``block_idx`` at 0. Subsequent rounds advance inside the assistant-message
    # loop (guarded to skip round 1). No pre-loop terminal emits a content block,
    # so the earlier ``reset_block_idx()`` + this restart are equivalent there.
    engine.begin_wire_round()

    # ── 4b. Loop-strategy pre-action step. The
    # SINGLE branch point on ``run_mode``: Direct contributes nothing (today's
    # auto-tool loop, byte-unchanged); Deep runs the stand-validated SGR step
    # (a forced ``plan`` tool with native CoT bounded by ``reasoning_effort``)
    # → emits exactly one ``REASONING_STEP`` event → records the plan into
    # history. The shared assistant-message loop below then drives the real
    # action with the FULL surface — dispatch / pairing / repair / loop
    # detection / terminal gate all stay shared. ──────────────────────────
    strategy = select_strategy(engine.config.run_mode)
    history_len_before_plan = len(engine.history)
    async for evt in strategy.prepare_turn(engine, context):
        yield evt
    if len(engine.history) != history_len_before_plan:
        # Deep recorded a planning turn — rebuild the context bundle so the
        # action stream sees it. The surface + skill blocks are unchanged
        # (same user message), so only the (now longer) history is refreshed.
        rebuilt_history, _ = _llm_history(engine)
        context = engine.context_manager.build_context(
            history=rebuilt_history,
            tools=tool_defs,
            system_prompt_sections=engine.config.system_prompt_sections,
            skill_index_block=skill_catalog_block,
            skills_loaded=engine._skill_loaded_bundles,
        )

    # ── 5-9. Stream one assistant message (recursive on tool_use) ────
    async for evt in _stream_one_assistant_message(engine, context):
        yield evt
        # NOTE: do NOT short-circuit here on ``stop_requested`` or
        # ``engine.is_terminal``. The inner generator is responsible for
        # draining its own pending events (e.g. an LLM-error path that
        # yields ``state_changed → FAILED`` followed by ``error`` and
        # ``message_stop``). Short-circuiting mid-drain would swallow
        # those events and leave the wire-format invariant broken.

    # If the inner generator has driven us to a terminal state already
    # (COMPLETED / FAILED / CANCELLED), there is nothing more to emit;
    # the engine.run() finally-block will persist the terminal snapshot.
    # Terminal states have empty outgoing edges so a stop-after-terminal
    # MUST NOT attempt another transition.
    if engine.is_terminal:
        settled = _maybe_run_settled_event(engine)
        if settled is not None:
            yield settled
        return

    # Stop requested but engine is not terminal — synthesise a
    # message_stop(cancelled) cleanly (RUNNING / AWAITING / COMPACTING
    # all allow → CANCELLED).
    if engine.stop_requested:
        # before the cancel terminal, pair any already-emitted
        # tool_use that never received a result (stop landed after the
        # tool_use block closed but before/while the dispatch loop ran).
        # The engine.run() finally-block persists this snapshot; a resume on
        # another pod must not replay a dangling tool_use into a 400.
        _synthesize_missing_tool_results(
            engine.history,
            error_content=engine.prompt_text("tool_result_interrupted"),
        )
        restored = restore_queued_prompts(engine)
        await _persist_live_control(engine)
        from_state = engine.state
        engine.transition_to(LoopState.CANCELLED)
        yield _emit_state_change(
            engine,
            from_state,
            LoopState.CANCELLED,
            reason="stop_requested",
        )
        yield TurnEvent(
            type=EventType.MESSAGE_STOP,
            run_id=engine.config.run_id,
            payload={
                "turn_id": engine.turn_id(),
                "stop_reason": StopReason.cancelled.value,
                "restored_queue_text": "\n\n".join(restored),
            },
        )
        return

    if engine.state is LoopState.AWAITING:
        return

    settled = _maybe_run_settled_event(engine)
    if settled is not None:
        yield settled

    # Final snapshot is persisted by ``engine.run()`` finally-block.


# ----------------------------------------------------------------------
# Helpers — compaction + state + streaming
# ----------------------------------------------------------------------


async def _run_compaction(
    engine: QueryEngine,
    *,
    force: bool = False,
    reason: str = "routine",
    protect_tail_from_index: int | None = None,
) -> AsyncIterator[TurnEvent]:
    """Drive one compaction attempt; emit started/completed events.

    ``force`` selects ``force_compaction`` (both tiers, unconditional) over
    the routine ``run_compaction`` — used by the proactive emergency-cliff
    branch at turn-start and per-iteration. ``reason`` is surfaced in the
    started/completed event payloads for telemetry (``"routine"`` /
    ``"proactive_emergency"`` / ``"proactive_per_iteration"`` /
    ``"proactive_per_iteration_emergency"``).

    ``protect_tail_from_index`` is set ONLY by the per-iteration gate to the
    index of the current assistant ``tool_use`` turn. Everything from that
    index to the end of history (the just-executed tool batch, any size) is
    exempted from compaction this iteration on top of the keep window, so a
    >keep parallel batch's fresh, unconsumed results are never
    blobbed/summarised before the next assistant stream consumes them. The
    turn-start gate and reactive-413 path pass ``None`` (no in-flight batch).
    """
    from protocore.runtime.context.budgets import derive_budgets

    from_state = engine.state
    engine.transition_to(LoopState.COMPACTING)
    yield _emit_state_change(
        engine,
        from_state,
        LoopState.COMPACTING,
        reason=f"compaction_triggered:{reason}",
    )

    # ``tokens_before`` MUST reflect the actual current token count in the
    # history (the value that crossed the trigger threshold). The
    # ``trigger_threshold`` field MUST surface the compaction-trigger
    # boundary (= model_context_window * compaction_trigger_ratio), NOT
    # the bare model_context_window. Previously both fields conflated to
    # the bare window which made the telemetry incoherent.
    rc = engine.context_manager._rc
    budgets = derive_budgets(rc)
    tokens_before_value = engine.context_manager.token_estimator.estimate_history(
        engine.history, rc
    )
    yield TurnEvent(
        type=EventType.COMPACTION_STARTED,
        run_id=engine.config.run_id,
        payload={
            "reason": reason,
            "tokens_before": tokens_before_value,
            "trigger_threshold": budgets.compaction_trigger_tokens,
            "emergency_threshold": budgets.compaction_emergency_tokens,
            "history_messages": len(engine.history),
            "holds_settled": bool(engine.config.rc.run_settled_enabled),
        },
    )

    compaction_call = (
        engine.context_manager.force_compaction
        if force
        else engine.context_manager.run_compaction
    )
    from protocore.runtime.correctness_bind import fire_lifecycle

    pre_compact, pre_compact_evt = await fire_lifecycle(
        engine, HookEvent.pre_compact, {"reason": reason}
    )
    if pre_compact_evt is not None:
        yield pre_compact_evt
    if not pre_compact.allowed:
        # Compaction is a transaction, and this is the seam that can refuse to
        # open it. Nothing was written, so there is nothing to roll back.
        yield TurnEvent(
            type=EventType.HOOK_FIRED,
            run_id=engine.config.run_id,
            payload={
                "hook": HookEvent.compaction_rollback.value,
                "decision": pre_compact.verdict.value,
                "reason": pre_compact.reason,
            },
        )
        return
    async def _record_summariser_request(request: LLMRequest) -> None:
        """Put the summariser's calls in the run's record alongside the turn's.

        By default the summariser talks to the same provider the run streams
        through, so a recording that held only the turn's calls would be short
        by one entry per summary — and a compaction is exactly the event that
        rewrites the transcript every later request is built from, which makes
        it the one a reader most needs explained.
        """
        await _manifest_request(engine, request, call_purpose="compaction_summary")

    try:
        attempt = await compaction_call(
            history=engine.history,
            compaction_state=engine.compaction_state,
            tenant_id=engine.config.tenant_id,
            model_name=engine.effective_model_name,
            observability=_observability_context(
                engine,
                call_purpose="structured",
                call_category="compaction",
            ),
            protect_tail_from_index=protect_tail_from_index,
            record_request=_record_summariser_request,
        )
    except CompactionExhaustedError as exc:
        # The transaction opened at ``pre_compact`` and cannot close on
        # success; say so at the coordinate that means exactly that.
        _rollback, rollback_evt = await fire_lifecycle(
            engine,
            HookEvent.compaction_rollback,
            {"reason": reason, "error": str(exc)},
        )
        if rollback_evt is not None:
            yield rollback_evt
        compacting_from = engine.state
        engine.transition_to(LoopState.FAILED)
        yield _emit_state_change(
            engine,
            compacting_from,
            LoopState.FAILED,
            reason=str(exc),
        )
        yield TurnEvent(
            type=EventType.ERROR,
            run_id=engine.config.run_id,
            payload={"kind": "compaction_exhausted", "message": str(exc)},
        )
        return

    yield TurnEvent(
        type=EventType.COMPACTION_COMPLETED,
        run_id=engine.config.run_id,
        payload={
            "reason": reason,
            "tokens_before": attempt.tokens_before,
            "tokens_after": attempt.tokens_after,
            "tier1_freed": attempt.tier1.tokens_freed if attempt.tier1 else 0,
            "tier2_summarised": attempt.tier2.turns_summarised if attempt.tier2 else 0,
            "tier3_folded": attempt.tier3.messages_folded if attempt.tier3 else 0,
            "blob_refs_created": (list(attempt.tier1.blob_refs_created) if attempt.tier1 else []),
        },
    )
    _commit, commit_evt = await fire_lifecycle(
        engine,
        HookEvent.compaction_commit,
        {
            "reason": reason,
            "tokens_before": int(getattr(attempt, "tokens_before", 0) or 0),
            "tokens_after": int(getattr(attempt, "tokens_after", 0) or 0),
        },
    )
    if commit_evt is not None:
        yield commit_evt
    _after_compact, after_compact_evt = await fire_lifecycle(
        engine, HookEvent.post_compact, {"reason": reason}
    )
    if after_compact_evt is not None:
        yield after_compact_evt
    from protocore.runtime.correctness_bind import commit_usage

    compact_usage = commit_usage(
        engine,
        kind="compaction",
        input_tokens=int(getattr(attempt, "tokens_before", 0) or 0),
        output_tokens=int(getattr(attempt, "tokens_after", 0) or 0),
        success=True,
    )
    if compact_usage is not None:
        yield compact_usage

    # The last real prompt measurement now describes a pre-compaction history
    # that no longer exists. Clear it so the gate does not re-fire on a stale
    # high-water mark: the freshly-shrunk history is re-measured by the cheap
    # estimate until the next LLM call reports a new ground-truth prompt size.
    engine.last_observed_prompt_tokens = 0

    # Snapshot after compaction completion
    # — executor pod crash between compaction and the next LLM call would
    # lose the freed-up history otherwise.
    await engine._persist_snapshot()

    compacting_from = engine.state
    engine.transition_to(LoopState.RUNNING)
    yield _emit_state_change(
        engine,
        compacting_from,
        LoopState.RUNNING,
        reason="compaction_completed",
    )


def _emit_state_change(
    engine: QueryEngine,
    from_state: LoopState,
    to_state: LoopState,
    *,
    reason: str,
) -> TurnEvent:
    """Build a ``state_changed`` :class:`TurnEvent`.

    Caller passes the explicit ``from_state``/``to_state`` pair so that
    payload accuracy is independent of when ``engine.transition_to`` is
    invoked relative to event emission. (Previously the helper read
    ``engine.state`` directly which made ``from`` self-referential when the
    transition had already been applied.)
    """
    return TurnEvent(
        type=EventType.STATE_CHANGED,
        run_id=engine.config.run_id,
        payload={
            "from": from_state.value,
            "to": to_state.value,
            "reason": reason,
        },
    )


async def _emit_dispatch_cancel_teardown(
    engine: QueryEngine,
) -> AsyncIterator[TurnEvent]:
    """route a tool-dispatch turn to the CANCELLED terminal.

    Shared teardown for every ``engine.stop_requested`` checkpoint in the
    dispatch prelude/loop of :func:`_stream_one_assistant_message` (cancel
    before the dispatch loop, cancel inside the awaited hook predicate, cancel
    between two tool calls in a serial OR parallel batch). The guarantee is
    "no NEW tool dispatch once a stop is observed": tools already dispatched
    before the stop keep their real results; any already-emitted-but-
    undispatched ``tool_use`` (the pending assistant tool_use blocks were
    appended to history BEFORE the dispatch loop) is paired here via
    :func:`_synthesize_missing_tool_results` (synthetic ``is_error`` results,
    idempotent — a call already paired is skipped) so the persisted snapshot
    stays pairing-valid for a resume on another pod. Yields the
    ``state_changed`` + terminal ``message_stop(cancelled)`` envelopes; the
    caller MUST ``return`` immediately after draining it. The outer ``query()``
    finally-guard then sees ``engine.is_terminal`` and emits nothing further.
    """
    _synthesize_missing_tool_results(
        engine.history,
        error_content=engine.prompt_text("tool_result_interrupted"),
    )
    from_state = engine.state
    engine.transition_to(LoopState.CANCELLED)
    yield _emit_state_change(
        engine,
        from_state,
        LoopState.CANCELLED,
        reason="stop_requested",
    )
    yield TurnEvent(
        type=EventType.MESSAGE_STOP,
        run_id=engine.config.run_id,
        payload={
            "turn_id": engine.turn_id(),
            "stop_reason": StopReason.cancelled.value,
        },
    )


#: Failure classes a DIFFERENT provider could plausibly serve. Each names a
#: property of the row that failed — its quota, its capacity, its key, its
#: balance, its catalogue — and never a property of the request, so the next row
#: gets a real chance rather than reproducing the same failure at a second
#: vendor's expense.
#:
#: Everything absent from this set keeps the recovery it already has, and the
#: omissions are the load-bearing part. A prompt that overflows the window is
#: fixed by compaction, which advancing would skip; an oversized body is fixed
#: by compressing it; a malformed request is fixed by the structured-output
#: ladder; an invalid reasoning payload is CAUSED by a model swap, so swapping
#: again compounds it; a policy refusal re-routed to a vendor that might accept
#: the content is a compliance decision and not a default. An unclassified
#: failure is excluded on principle — advancing on an error nobody has
#: characterised turns one broken run into N provider calls and N bills.
#:
#: Compared as plain strings against the verdict the adapter layer attaches to
#: the raised error. The taxonomy itself lives above core and core must not
#: import it; a string comparison keeps the boundary intact and still fails
#: closed, since a reason spelled differently simply is not in the set.
_PROVIDER_ADVANCE_REASONS: frozenset[str] = frozenset(
    {
        "auth",
        "auth_permanent",
        "billing",
        "model_not_found",
        "overloaded",
        "rate_limit",
        "server_error",
        "timeout",
    }
)


@dataclass(frozen=True, slots=True)
class _IdleStreamVerdict:
    """Core's own verdict on a stream that stopped speaking.

    The provider adapters pin their classification onto the errors they raise
    and core reads it duck-typed, without naming the taxonomy. The idle
    watchdog lives in core, so nothing above it can classify what it raises and
    core has to supply the verdict itself — in the same shape, so the reader
    below stays one function.
    """

    reason: str = "timeout"


def _classified_failure_reason(exc: BaseException) -> str:
    """The adapter's verdict on ``exc``, or ``""`` when it carries none.

    Read duck-typed. The provider adapters pin their classification onto the
    exception they raise; core reads it without naming the type, which is what
    lets the decision below stay in core beside the loop that acts on it while
    the taxonomy stays where the vendors are.
    """
    classified = getattr(exc, "classified", None)
    if classified is None:
        return ""
    reason = getattr(classified, "reason", None)
    if reason is None:
        return ""
    return str(getattr(reason, "value", reason))


def _reason_permits_provider_advance(exc: BaseException) -> bool:
    """May this failure be answered by moving to the next provider?

    :class:`LLMRateLimitError` and :class:`LLMTimeoutError` are their own
    answer — a provider adapter raises those two types only for a quota it hit
    or a stream that stopped producing, both of which belong to the endpoint and
    not to the request. They also arrive unclassified from a caller that raised
    them directly, and treating the type as the verdict keeps that working.

    Everything else is a :class:`LLMProviderError`, which is the adapters'
    catch-all: a 503 and a policy refusal reach this branch as the same Python
    type. Only the attached verdict separates them, so an unclassified one does
    not advance.
    """
    if isinstance(exc, LLMRateLimitError | LLMTimeoutError):
        return True
    return _classified_failure_reason(exc) in _PROVIDER_ADVANCE_REASONS


async def _advance_provider_chain(
    engine: QueryEngine,
    exc: BaseException,
    *,
    kind: str,
) -> str:
    """Move ``engine`` onto the next provider. Returns its name, or ``""``.

    ``""`` — the run stays where it is and the caller falls through to its
    existing recovery — covers every way the step is unavailable: no chain
    configured, a failure class this list must not answer, the advance budget
    spent, or no rung left. Collapsing them is deliberate: the caller's next
    branch is the same in all four cases, and the reason each one happened is
    already on the record via the chain's own accounting.

    The whole provider moves. Rebinding only ``config.model_name`` would leave
    the previous vendor's endpoint and key in place, holding a name that
    endpoint does not serve.
    """
    chain = engine.provider_chain
    if chain is None:
        return ""
    if not _reason_permits_provider_advance(exc):
        return ""
    if engine._provider_chain_advances >= engine.config.rc.llm_provider_chain_max_advances:
        return ""
    if not await chain.advance(reason=kind):
        return ""
    engine._provider_chain_advances += 1
    engine.llm = chain.current()
    engine.config = replace(engine.config, model_name=chain.current_model_name())
    return chain.current_model_name()


def _persist_partial_attempt_to_history(
    engine: QueryEngine,
    stream_result: _StreamAttemptResult,
) -> None:
    """Persist the partial text + completed tool calls from a failed stream
    attempt to ``engine.history`` so the durable snapshot matches what the
    SSE consumers already saw live.

    On the fallback-model retry, the ``LLMStreamIdleError`` backstop, and
    the ``LLMProviderError`` backstop / terminal paths, the failed
    attempt's deltas were forwarded to the wire before the exception
    was raised. The normal completion path appends the assistant turn
    to history at the end of the inner stream loop, but a failure exits
    the inner loop BEFORE that line — so the partial is silently
    dropped from ``engine.history``. The user sees it live; a reload
    does not. This helper closes the gap by appending the partial
    assistant turn (text + completed ``tool_calls`` +
    ``reasoning_buffer``) to history exactly the way the normal
    completion path does.

    No-op when ``stream_result`` carries no text, tool calls, or reasoning —
    preserves the pre-existing behaviour for an empty pre-fail attempt
    (e.g. provider error on the very first byte). Mirrors the
    normal-completion guard in :func:`_stream_one_assistant_message`
    (the ``if assistant_blocks:`` check that gates the history
    append).
    """
    assistant_blocks: list[ContentBlock] = []
    if stream_result.text_buffer:
        assistant_blocks.extend(
            _split_answer_text_blocks(
                stream_result.text_buffer, stream_result.narration_prefix_chars
            )
        )
    for tc in stream_result.tool_calls:
        assistant_blocks.append(
            ToolUseBlock(
                tool_call_id=tc.id,
                name=tc.name,
                arguments_json=json.dumps(tc.arguments, ensure_ascii=False),
            )
        )
    if not assistant_blocks and not stream_result.reasoning_buffer:
        return
    engine.history.append(
        Message(
            role=MessageRole.assistant,
            content_blocks=assistant_blocks,
            reasoning_content=stream_result.reasoning_buffer or None,
            metadata={PARTIAL_ASSISTANT_ATTEMPT_METADATA_KEY: True},
        )
    )


def _content_block_visibility(
    kind: ProviderDeltaKind,
    *,
    tool_interleaved: bool,
    narration_prefix: bool = False,
) -> BlockVisibility:
    """Where this content block belongs in the reader's prose stream.

    A forward-only stream cannot see its own future. At the moment a text block
    OPENS we genuinely do not know whether it will turn out to be the answer or
    narration, because that is decided by whether a tool call follows — and no
    signal available at open time predicts it. Shape does not: a person asking
    about a config format gets an answer that looks exactly like a model
    thinking aloud about one, so any content heuristic eats real answers.

    So this function does not guess. It reports facts that are already settled,
    and the caller evaluates them twice — once when the block opens, once when
    it closes:

    ``tool_interleaved``
        A tool call that CONTINUES the run has already started inside this same
        assistant message. That is proof by construction, not inference: a
        non-terminal tool call must have its result fed back, so this message is
        followed by at least one more, and no text sharing a message with such a
        call can be the run's final prose. Text here is narration between tool
        calls.

    ``narration_prefix``
        This block holds a leading run of process narration that was CUT OFF
        the answer following it — the one place the runtime does read text (see
        :mod:`protocore.runtime.answer_narration` for why it is narrow enough
        to be safe, and why it applies only to a run that delegated). It is a
        settled fact by the time it reaches here: the caller has already
        measured the run, proved a substantial answer follows it, and put that
        answer in a block of its own. Nothing was removed — the two blocks
        concatenate to the text the model wrote.

    At open time ``tool_interleaved`` is only True for text that comes AFTER
    the tool call it is interleaved with. The commoner shape — narration, THEN
    the tool call — is unprovable at open and provable the instant the tool
    starts, which is also the instant the block closes. The caller therefore
    emits the settled value on ``content_block_stop``; both flags only ever move
    from False to True, so the settled value is never weaker than the one
    already sent.

    Reasoning is unconditionally ``COLLAPSED``: it is the model's thinking by
    definition and never the reply, which is also how the durable transcript
    projects it.

    Where the signal is absent or ambiguous the answer is ``PUBLIC``. The two
    errors are not symmetric — a missed ``COLLAPSED`` on narration shows the
    user one bubble too many, while a false ``COLLAPSED`` on the answer takes
    the reply away entirely — so ambiguity resolves to the visible side every
    time.
    """

    if kind is ProviderDeltaKind.thinking:
        return BlockVisibility.COLLAPSED
    if tool_interleaved:
        return BlockVisibility.COLLAPSED
    if narration_prefix:
        return BlockVisibility.COLLAPSED
    return BlockVisibility.PUBLIC


def _answer_narration_split_active(engine: QueryEngine) -> bool:
    """Whether this run's assistant text may be split at its leading narration.

    Two facts, both about the RUN rather than the text. The tenant switch, and
    whether the run has actually handed work to a subagent — the narration is a
    delegating leader's habit, measured at 11-12 of 12 delegating runs against a
    minority without delegation, so a run that dispatched no subtask is left
    exactly as it was.
    """

    rc = engine.config.rc
    return (
        rc.delegated_answer_narration_split_enabled
        and rc.delegated_answer_narration_scan_chars > 0
        and engine._run_delegated
    )


def _settled_narration_split(
    engine: QueryEngine,
    text: str,
    *,
    complete: bool,
) -> tuple[str, str] | None:
    """Split ``text`` into (leading narration, answer), once that is knowable.

    ``None`` means "not yet" and can only be returned for a block still being
    streamed: either the run of narration sentences may still grow, or it has
    settled but not enough answer has arrived to clear
    ``delegated_answer_narration_min_answer_chars``. The caller keeps buffering.
    Both waits are bounded by the scan ceiling plus that floor, so the wire is
    never held for more than a fixed number of characters.

    A ``("", text)`` result is the ordinary one: no narration, or not enough
    answer behind it to be worth splitting. Nothing about the text changes —
    the caller emits it as the single block it always was.
    """

    rc = engine.config.rc
    span = leading_narration_span(
        text,
        scan_chars=rc.delegated_answer_narration_scan_chars,
        complete=complete,
    )
    if not span.settled:
        return None
    if not span.found:
        return ("", text)
    if len(text) - span.length < rc.delegated_answer_narration_min_answer_chars:
        # An answer that is narration and little else stays whole and visible.
        # Collapsing it would hand the reader an empty bubble, which is a worse
        # failure than the narration this exists to hide.
        return ("", text) if complete else None
    return (text[: span.length], text[span.length :])


def _text_delta_events(
    engine: QueryEngine,
    text: str,
    *,
    block_idx: int,
) -> Iterator[TurnEvent]:
    """Put held-back text on the wire, through the ordinary delta mapping.

    The buffered head is released as one delta rather than replayed as the many
    the provider sent. A reader appends deltas in order, so the rendered block
    is identical; the only visible difference is that the first characters of a
    split block arrive together instead of one token at a time.
    """

    yield from delta_to_turn_events(
        ProviderDelta(kind=ProviderDeltaKind.text, content=text),
        run_id=engine.config.run_id,
        turn_id=engine.turn_id(),
        block_idx=block_idx,
    )


def _split_answer_text_blocks(text: str, narration_prefix_chars: int) -> list[ContentBlock]:
    """Build the durable text block(s) for one assistant message's prose.

    Mirrors, exactly, the split the live stream already made — it is handed the
    stream's own cut point rather than re-deciding from the finished text, so a
    message cannot render one way live and another way after a reload. The two
    blocks concatenate to ``text``; ``Message.text`` joins every
    :class:`TextBlock`, so the model's view of its own turn is unchanged.
    """

    if 0 < narration_prefix_chars < len(text):
        return [
            TextBlock(
                text=text[:narration_prefix_chars],
                visibility=BlockVisibility.COLLAPSED,
            ),
            TextBlock(text=text[narration_prefix_chars:]),
        ]
    return [TextBlock(text=text)]


def _tool_call_continues_run(engine: QueryEngine, tool_name: str | None) -> bool:
    """True iff this tool call guarantees the run has another turn to come.

    Read by the block-visibility signal, which needs "a tool call happened" to
    mean "this message cannot hold the final prose". That inference holds for
    every ordinary tool — the result must go back to the model — but NOT for the
    run's terminal tool. Under the background terminal gate the model writes its
    answer and calls ``Finalize`` in the SAME message, and the gate call is
    stripped from every reader-facing view, so treating it like any other tool
    would collapse exactly the text the user came for and would contradict the
    durable transcript, which shows that message as prose and nothing else.

    An unnamed call is treated as terminal (i.e. not proof) whenever the run
    HAS a terminal tool: some providers withhold the name until the arguments
    finish streaming, and by then the text block is long closed. Losing a
    collapse on those providers is the cheap error. With no terminal tool
    configured no call can be the terminal one, so an unnamed call is proof
    like any other.
    """

    expected_terminal = engine.config.expected_terminal_tool
    if expected_terminal is None:
        return True
    return tool_name is not None and tool_name != expected_terminal


async def _stream_one_assistant_message(
    engine: QueryEngine,
    context: ContextBundle,
) -> AsyncIterator[TurnEvent]:
    """Drive the assistant-message tool-dispatch loop for one user input.

    Iterates: open LLM stream → emit deltas → dispatch tool_calls →
    re-open LLM stream with tool_results → … until either (a) the model
    emits ``finish`` with no tool_calls, (b) approval is pending, (c) the
    cap on assistant messages (``max_turns_per_run``) is hit, or (d) the
    LLM raises.

    The loop body was previously implemented as recursion; it now uses an
    explicit ``while``-loop with a depth counter, so that endless tool calls
    cannot exceed Python's recursion limit. Each iteration counts as one
    assistant message, and the cap on how many a run may open is a policy —
    one of the bounds :func:`_effective_policies` keeps in the set whatever a
    host installs, because the loop itself no longer carries one.
    """
    # The turn-local state the loop shares with its policies. Every field was
    # a local of this function; a policy that owns one now owns the write.
    flags = TurnFlags()
    policies = _effective_policies(engine)
    max_messages = engine.config.rc.max_turns_per_run
    current_context = context
    previous_tool_results_ready_at: float | None = None
    # When the forced backstop is armed from a typed provider/stream error,
    # the original exception + its terminal ``kind`` are stashed here. If the
    # best-effort forced turn (or any later turn) reaches a no-answer
    # completion path, the stored error is surfaced as the terminal LLM error
    # rather than silently completing with no answer. Cleared implicitly once
    # the terminal answer is produced (the success completion paths do not
    # consult it).
    stored_stream_error: tuple[BaseException, str] | None = None

    while True:
        await _reload_live_control(engine)
        async for _bg_evt in _background_wake_events(engine):
            yield _bg_evt
        # Every bound this run can reach is read here, before the message
        # opens: the tool-call budget, the output-token budget, and a
        # precondition that burnt its attempts.
        _turn = _turn_at(
            engine,
            flags,
            TurnCoordinate.turn_start,
            turn_budget=max_messages,
            stored_stream_error=stored_stream_error,
        )
        async for _policy_evt in policies.apply(_turn):
            yield _policy_evt
        max_messages = _turn_budget_after(_turn.outcome, max_messages, flags)
        if _turn.outcome.directive is TurnDirective.end_turn:
            return
        if _turn.outcome.directive is TurnDirective.restart_turn:
            if _turn.outcome.rebuild_context:
                current_context = await _rebuild_context_for_recovery(engine)
            continue

        # Hard cap on assistant messages within a single turn. The run-level
        # turn count only moves once per run — without this counter a model
        # emitting endless tool calls would blow the Python recursion stack
        # before the run-level cap could fire.
        flags.assistant_message_idx += 1

        _turn = _turn_at(
            engine,
            flags,
            TurnCoordinate.turn_budget,
            turn_budget=max_messages,
            stored_stream_error=stored_stream_error,
        )
        async for _policy_evt in policies.apply(_turn):
            yield _policy_evt
        max_messages = _turn_budget_after(_turn.outcome, max_messages, flags)
        if _turn.outcome.directive is TurnDirective.end_turn:
            return
        if _turn.outcome.directive is TurnDirective.restart_turn:
            if _turn.outcome.rebuild_context:
                current_context = await _rebuild_context_for_recovery(engine)
            continue

        # Reset per-message recovery state at every new assistant
        # message — the budget is per-message, not per-run.
        engine.reset_recovery_state()

        # #1/#4 — advance to this round's wire turn id + restart block_idx at 0
        # BEFORE emitting ``message_start``. ``engine.turn_id()`` now yields a
        # DISTINCT id per assistant-message round, and every frame this round
        # emits (``message_start`` below, the ``content_block_*`` / ``tool_use_*``
        # / ``tool_result`` frames inside ``_drive_one_stream``, and the
        # ``message_stop`` at the bottom of this iteration) reads the same id
        # because they all call ``engine.turn_id()`` — so no frame is orphaned
        # from its round. The FIRST round (``flags.assistant_message_idx == 1``) was
        # already opened just before the 4b loop-strategy step (so the Deep
        # ``REASONING_STEP`` and this message_start share one id); only rounds 2+
        # advance here. Pre-loop terminals keep the legacy (unsuffixed) id.
        if flags.assistant_message_idx > 1:
            engine.begin_wire_round()

        # ── message_start ───────────────────────────────────────────
        yield TurnEvent(
            type=EventType.MESSAGE_START,
            run_id=engine.config.run_id,
            payload={
                "turn_id": engine.turn_id(),
                "model": engine.effective_model_name,
                "role": "assistant",
            },
        )

        # ── Stream loop with reactive-413 recovery ────────────────────
        pending_tool_calls: list[ToolCall] = []
        history_tool_calls: list[ToolCall] = []
        text_buffer = ""
        reasoning_buffer = ""

        flags.terminal_yielded = False
        # Forced terminal backstop — set True when an exhaustion exit
        # inside the inner stream loop armed the forced terminal backstop
        # instead of going terminal. Checked right after the inner loop so
        # the OUTER loop iterates with the injected nudge in history (one
        # final bounded turn). Default behaviour (RC off) never sets this.
        flags.backstop_armed = False
        # An inner while-loop lets us re-stream after recovery
        # for ``LLMContextWindowExceeded`` (reactive compaction),
        # ``LLMProviderError`` with configured fallback model, or
        # ``finish_reason='length'`` within the round budget.
        while True:
            stream_result = _StreamAttemptResult()
            tool_results_ready_at = previous_tool_results_ready_at
            previous_tool_results_ready_at = None
            from protocore.runtime.correctness_bind import commit_usage, fire_lifecycle

            turn_start, turn_start_evt = await fire_lifecycle(
                engine,
                HookEvent.turn_start,
                {
                    "assistant_message_idx": flags.assistant_message_idx,
                    "history_len": len(engine.history),
                },
            )
            if turn_start_evt is not None:
                yield turn_start_evt
            if not turn_start.allowed:
                yield _hook_denied_stop(engine, HookEvent.turn_start, turn_start)
                return
            transform_out, transform_evt = await fire_lifecycle(
                engine,
                HookEvent.context_transform,
                {
                    "history_len": len(engine.history),
                    "system_prompt_sections": list(current_context.system_prompt_sections),
                    "active_language": current_context.active_language,
                },
            )
            if transform_evt is not None:
                yield transform_evt
            if not transform_out.allowed:
                yield _hook_denied_stop(engine, HookEvent.context_transform, transform_out)
                return
            # The rewrite a transform returns is the context this turn is
            # actually built from. A transform whose result went nowhere was
            # the older, worse version of this seam.
            current_context = _apply_context_transform(current_context, transform_out.payload)
            request_prepare, request_prepare_evt = await fire_lifecycle(
                engine,
                HookEvent.request_prepare,
                {
                    "model": engine.effective_model_name,
                    "message_count": len(current_context.messages),
                },
            )
            if request_prepare_evt is not None:
                yield request_prepare_evt
            if not request_prepare.allowed:
                yield _hook_denied_stop(engine, HookEvent.request_prepare, request_prepare)
                return
            try:
                try:
                    async for evt in _drive_one_stream(
                        engine,
                        current_context,
                        stream_result,
                        previous_tool_results_ready_at=tool_results_ready_at,
                    ):
                        yield evt
                    retrying = engine._transient_stream_retry_count > 0
                    usage_evt = commit_usage(
                        engine,
                        kind="retry" if retrying else "inference",
                        input_tokens=engine.total_usage.this_turn_input,
                        output_tokens=engine.total_usage.this_turn_output,
                        success=True,
                    )
                    if usage_evt is not None:
                        yield usage_evt
                    _received, received_evt = await fire_lifecycle(
                        engine,
                        HookEvent.response_received,
                        {
                            "finish_reason": stream_result.finish_reason,
                            "tool_call_count": len(stream_result.tool_calls),
                            "text_length": len(stream_result.text_buffer),
                        },
                    )
                    if received_evt is not None:
                        yield received_evt
                except Exception as request_exc:
                    # Every provider failure this turn can take passes through
                    # here on its way to the recovery branch that owns it, so
                    # the coordinate fires once per failure and changes none of
                    # the recovery that follows.
                    _errored, error_evt = await fire_lifecycle(
                        engine,
                        HookEvent.request_error,
                        {
                            "error": str(request_exc),
                            "error_type": type(request_exc).__name__,
                        },
                    )
                    if error_evt is not None:
                        yield error_evt
                    raise
            except Exception as exc:
                # Every way a stream attempt can fail arrives here, and what
                # the run does about it — another endpoint, the same one
                # again, a wind-down onto the evidence it already has, or a
                # terminal — is one ranked decision rather than five branches
                # that grew apart. The partial the reader already saw is put
                # in the transcript first, so whatever recovery is chosen
                # carries it forward.
                _persist_partial_attempt_to_history(engine, stream_result)
                _turn = _turn_at(engine, flags, TurnCoordinate.stream_failed,
                    stream_error=exc,
                )
                async for _policy_evt in policies.apply(_turn):
                    yield _policy_evt
                if _turn.stored_stream_error is not None:
                    stored_stream_error = _turn.stored_stream_error
                max_messages = _turn_budget_after(_turn.outcome, max_messages, flags)
                if _turn.outcome.directive is TurnDirective.restart_turn:
                    if _turn.outcome.rebuild_context:
                        current_context = await _rebuild_context_for_recovery(engine)
                    continue
                if _turn.outcome.directive is TurnDirective.end_turn:
                    if flags.terminal_yielded or flags.backstop_armed:
                        if _turn.outcome.rebuild_context:
                            current_context = await _rebuild_context_for_recovery(
                                engine
                            )
                        break
                    # Neither a terminal nor a wind-down: the run was closed on
                    # an answer it had already delivered.
                    return
                # Nobody claimed the failure. Ending here would be the quietest
                # possible outcome — no terminal, no state change, and the
                # exception dropped — so the failure goes back to the caller
                # as what it is.
                raise


            # ── The output budget ran out before the round finished ────
            # Mid-sentence or mid-way through a tool call's arguments: either
            # way this is not a finished turn, and neither shape may be
            # accepted as one. Nothing has been dispatched yet, which is what
            # makes the refusal of a half-written call possible at all.
            _turn = _turn_at(
                engine,
                flags,
                TurnCoordinate.output_truncated,
                pending_tool_calls=stream_result.tool_calls,
                finish_reason=stream_result.finish_reason or "",
                text_emitted=bool(stream_result.text_buffer),
                reasoning_emitted=bool(stream_result.reasoning_buffer),
                record_partial_attempt=partial(
                    _persist_partial_attempt_to_history, engine, stream_result
                ),
                cancel_checkpoint=partial(
                    _cancel_checkpoint, engine, flags, policies
                ),
            )
            async for _policy_evt in policies.apply(_turn):
                yield _policy_evt
            max_messages = _turn_budget_after(_turn.outcome, max_messages, flags)
            if _turn.outcome.directive is TurnDirective.restart_turn:
                if _turn.outcome.rebuild_context:
                    current_context = await _rebuild_context_for_recovery(engine)
                continue
            if _turn.outcome.directive is TurnDirective.end_turn:
                if flags.terminal_tool_completed:
                    async for _evt in _complete_via_terminal_tool(
                        engine, flags, policies
                    ):
                        yield _evt
                    return
                if flags.terminal_yielded or flags.backstop_armed:
                    if _turn.outcome.rebuild_context:
                        current_context = await _rebuild_context_for_recovery(engine)
                    break
                return

            # Normal completion — collect the outcome and exit the
            # inner recovery loop.
            pending_tool_calls = stream_result.tool_calls
            history_tool_calls = list(stream_result.tool_calls)
            text_buffer = stream_result.text_buffer
            reasoning_buffer = stream_result.reasoning_buffer
            # A round that came back usable is not yet a round the run will
            # act on: whether it is the model working or the model repeating
            # itself — and what to do with what it asked for either way — is
            # a decision with its own bounds, and it is made here.
            _turn = _turn_at(
                engine,
                flags,
                TurnCoordinate.stream_settled,
                pending_tool_calls=pending_tool_calls,
                finish_reason=stream_result.finish_reason or "",
                stream_repeat_guard=partial(
                    _strip_stream_repeat, engine, stream_result
                ),
            )
            async for _policy_evt in policies.apply(_turn):
                yield _policy_evt
            text_buffer = stream_result.text_buffer
            reasoning_buffer = stream_result.reasoning_buffer
            if _turn.outcome.tool_calls is not None:
                pending_tool_calls = list(_turn.outcome.tool_calls)
            stream_result.tool_calls = history_tool_calls
            break

        # The inner loop has stopped re-streaming, so this assistant message
        # is finished however it got here — that is what ``turn_end`` means.
        from protocore.runtime.correctness_bind import fire_lifecycle

        _turn_end, turn_end_evt = await fire_lifecycle(
            engine,
            HookEvent.turn_end,
            {
                "assistant_message_idx": flags.assistant_message_idx,
                "finish_reason": stream_result.finish_reason,
                "tool_call_count": len(pending_tool_calls),
                "terminal": flags.terminal_yielded,
            },
        )
        if turn_end_evt is not None:
            yield turn_end_evt

        if flags.terminal_yielded:
            return

        # An exhaustion exit inside the inner stream loop armed the forced
        # terminal backstop. Restart the OUTER loop so the next assistant
        # turn streams with the injected nudge + the terminal-only latch
        # active. The ``flags.terminal_nudge_used`` latch already prevents
        # this from looping more than once.
        if flags.backstop_armed:
            continue

        # A round that came back with nothing the run can use: reasoning and
        # no answer, or nothing at all right after tool results. Both are
        # recovered by re-running the round, and both are bounded.
        _turn = _turn_at(
            engine,
            flags,
            TurnCoordinate.empty_model_turn,
            text_emitted=bool(text_buffer),
            reasoning_emitted=bool(reasoning_buffer),
            reasoning_chars=len(reasoning_buffer),
            tool_calls_pending=bool(pending_tool_calls),
            tool_results_ready=tool_results_ready_at is not None,
            finish_reason=stream_result.finish_reason,
            record_partial_attempt=partial(
                _persist_partial_attempt_to_history, engine, stream_result
            ),
        )
        async for _policy_evt in policies.apply(_turn):
            yield _policy_evt
        max_messages = _turn_budget_after(_turn.outcome, max_messages, flags)
        if _turn.outcome.directive is TurnDirective.end_turn:
            return
        if _turn.outcome.directive is TurnDirective.restart_turn:
            if _turn.outcome.rebuild_context:
                current_context = await _rebuild_context_for_recovery(engine)
            continue

        # ── Append assistant message to history ─────────────────────
        assistant_blocks: list[ContentBlock] = []
        if text_buffer:
            assistant_blocks.extend(
                _split_answer_text_blocks(
                    text_buffer, stream_result.narration_prefix_chars
                )
            )
        for tc in history_tool_calls or pending_tool_calls:
            assistant_blocks.append(
                ToolUseBlock(
                    tool_call_id=tc.id,
                    name=tc.name,
                    arguments_json=json.dumps(tc.arguments, ensure_ascii=False),
                )
            )
        if assistant_blocks:
            assistant_metadata = (
                {PARTIAL_ASSISTANT_ATTEMPT_METADATA_KEY: True}
                if stream_result.finish_reason == "stop"
                and any(tc.args_partial_truncated for tc in pending_tool_calls)
                else {}
            )
            engine.history.append(
                Message(
                    role=MessageRole.assistant,
                    content_blocks=assistant_blocks,
                    reasoning_content=reasoning_buffer or None,
                    metadata=assistant_metadata,
                )
            )

        # ── A tool call the model started and never finished ────────
        # Nothing is dispatched here until the policy has had its say: it is
        # the last moment a call can be refused rather than run, and the
        # refusal it makes is what teaches the model to chunk.
        _turn = _turn_at(
            engine,
            flags,
            TurnCoordinate.tool_calls_ready,
            pending_tool_calls=pending_tool_calls,
            finish_reason=stream_result.finish_reason or "",
            cancel_checkpoint=partial(
                _cancel_checkpoint, engine, flags, policies
            ),
        )
        async for _policy_evt in policies.apply(_turn):
            yield _policy_evt
        if _turn.outcome.directive is TurnDirective.end_turn:
            if flags.terminal_tool_completed:
                async for _evt in _complete_via_terminal_tool(
                    engine, flags, policies
                ):
                    yield _evt
            return
        if _turn.outcome.directive is TurnDirective.restart_turn:
            previous_tool_results_ready_at = time.perf_counter()
            if _turn.outcome.rebuild_context:
                current_context = await _rebuild_context_for_recovery(engine)
            continue

        # ── No tool calls — end_turn (terminal) ─────────────────────
        if not pending_tool_calls:
            # an interrupt that landed while this assistant stream
            # was opening breaks the inner stream immediately (the per-delta
            # ``if engine.stop_requested: break`` at the bottom of
            # :func:`_drive_one_stream`) and yields an empty result (no text,
            # no tool_calls). Without this guard the empty turn falls through
            # to the success-class ``end_turn`` → COMPLETED below, scoring an
            # aborted turn as a clean answer (indistinguishable downstream).
            # Mirror the reference's FIRST post-stream abort check
            # (``aborted_streaming``/``aborted_tools``): route to the
            # CANCELLED terminal with ``stop_reason=cancelled`` instead. This
            # runs BEFORE the terminal-tool nudge so a cancelled run is never
            # nudged into one more turn. The outer ``query()`` finally-guard
            # then sees ``engine.is_terminal`` and emits nothing further.
            async for _evt in _cancel_checkpoint(engine, flags, policies):
                yield _evt
            if flags.terminal_yielded:
                return

            # The seams a finish passes through, in the order it passes
            # them: the turn ends; the model is nudged towards the tool that
            # ends runs (and the failure a wind-down was started for is
            # reported here if the wind-down wrote nothing); the answer it
            # gave is held to a floor; and only then is the finish itself
            # allowed. Walking the list rather than writing four near-copies
            # is what keeps the order a fact of the core rather than of
            # whichever copy was edited last.
            _restart_turn = False
            for _seam in _FINISH_SEAMS:
                _turn = _turn_at(
                    engine,
                    flags,
                    _seam,
                    stored_stream_error=stored_stream_error,
                    text_emitted=bool(text_buffer),
                    reasoning_emitted=bool(reasoning_buffer),
                )
                async for _policy_evt in policies.apply(_turn):
                    yield _policy_evt
                max_messages = _turn_budget_after(
                    _turn.outcome, max_messages, flags
                )
                if _turn.outcome.directive is TurnDirective.end_turn:
                    return
                if _turn.outcome.directive is TurnDirective.restart_turn:
                    if _turn.outcome.rebuild_context:
                        current_context = await _rebuild_context_for_recovery(
                            engine
                        )
                    _restart_turn = True
                    break
            if _restart_turn:
                continue

            settled = _maybe_run_settled_event(engine)
            if settled is not None:
                yield settled
            await _reload_live_control(engine)
            follow_evt = _inject_follow_up_into_history(engine)
            if follow_evt is not None:
                yield follow_evt
                await _persist_live_control(engine)
                engine._run_settled_emitted = False
                max_messages = max(max_messages, flags.assistant_message_idx + 1)
                current_context = await _rebuild_context_for_recovery(engine)
                continue
            engine.transition_to(LoopState.COMPLETED)
            return

        # ── (tool-dispatch path) — cancel before dispatch ──────
        # An interrupt that landed while THIS assistant stream was streaming
        # its tool_use(s) breaks the inner stream's per-delta stop-check
        # (``_drive_one_stream``) AFTER the ``tool_use_stop`` delta already
        # accumulated the call(s) into ``pending_tool_calls`` — so the no-tool
        # branch above does NOT fire, and without this guard the loop would
        # proceed to DISPATCH those tools, performing side effects AFTER the
        # run was cancelled. Mirror the no-tool branch's abort check here for
        # the with-tools path: route to the CANCELLED terminal
        # (``stop_reason=cancelled``) WITHOUT dispatching via the shared
        # :func:`_emit_dispatch_cancel_teardown` helper (synthesises the
        # missing ``is_error`` tool_results so the snapshot stays pairing-valid
        # for a resume on another pod). This is the FIRST of several
        # ``stop_requested`` checkpoints — the cancel guarantee is repeated
        # AFTER the awaited hook predicate and before EACH individual tool
        # dispatch, so a cancel that lands in any
        # await gap of the dispatch prelude/loop cannot dispatch a NEW tool.
        async for _evt in _cancel_checkpoint(engine, flags, policies):
            yield _evt
        if flags.terminal_yielded:
            return

        # ── Dispatch each tool call ─────────────────────────────────
        # Parallelise concurrent-safe read-only tools so a
        # single timed-out PCM read does not block sibling reads in the
        # same assistant turn. Sequential dispatch was responsible for
        # turning one 30s PCM stall into a 60s turn gap when the model
        # asked for two reads at once. Parallel-safe tools satisfy
        # ``tool.is_concurrent_safe AND not tool.is_destructive AND not
        # any-PreToolUse-hook-might-match`` (see
        # :func:`_is_parallel_safe_tool` + :func:`_pre_tool_use_match_predicate`);
        # destructive / approval-gated / hook-gated tools stay serial so
        # their causal order with sibling reads is preserved as the LLM
        # emitted them and the serial path's web-mode approval downgrade
        # (``query.py:_dispatch_tool`` web-mode branch) is honoured.
        #
        # History invariant: ``ToolResultBlock`` entries MUST land in
        # ``engine.history`` in the LLM-requested tool-call order, so
        # the parallel branch defers the history mutation via
        # :func:`_drain_dispatch_tool_deferred` + iterates the batch
        # results in the original order via
        # :func:`_apply_deferred_tool_history`. Event emission also
        # follows the original order: gather completion order is
        # discarded.
        flags.approval_pending = False
        flags.terminal_tool_completed = False
        # Set True when a bounded pre-terminal self-verify turn was injected
        # at a would-be-terminal site. It breaks the dispatch loop WITHOUT
        # finalising; flow then falls through to the
        # ``message_stop(tool_use)`` + context-rebuild path so the outer loop
        # re-drives one corrective turn. Default False = no self-verify.
        self_verify_injected = False
        # Every call of this message a gate held, in the order the model asked
        # for them. The turn announces the whole set once rather than one card
        # per call, and the first one parked is the one the envelope names.
        parked_calls: list[PendingInterrupt] = []

        # Pre-compute, once per turn, a predicate that tells us whether a
        # given tool name could be matched by any enabled ``PreToolUse`` hook
        # for the current tenant. If so, the tool MUST stay on the serial
        # dispatch path so the web-mode approval downgrade + first-pending
        # stop invariants live exclusively in :func:`_dispatch_tool`. The
        # predicate is awaited ONCE per turn (one ``IHookManager.list``
        # round-trip) before the batching loop walks the pending tool calls.
        hook_match_predicate = await _pre_tool_use_match_predicate(engine)

        # Re-check AFTER the awaited hook predicate.
        # ``_pre_tool_use_match_predicate`` is an ``await`` (one
        # ``IHookManager.list`` round-trip): a cancel that lands DURING that
        # await is invisible to the pre-loop guard above, yet it must still
        # abort BEFORE any dispatch. Without this checkpoint a cancel in that
        # await gap would fall through and dispatch the pending tools (side
        # effect after cancellation). Route to CANCELLED here instead.
        async for _evt in _cancel_checkpoint(engine, flags, policies):
            yield _evt
        if flags.terminal_yielded:
            return

        # Pre-compute eligibility for every call so we do not call into
        # the registry twice per call (this also keeps the partition
        # stable when the registry mutates mid-turn — defensive).
        #
        # When ``rc.parallel_read_tools_enabled`` is False the fan-out is
        # disabled entirely: every call is marked non-eligible so it
        # dispatches through the serial single-element path. Behaviour is
        # otherwise identical (rollback switch). Default True enables it.
        if engine.config.rc.parallel_read_tools_enabled:
            parallel_eligible = [
                _is_parallel_safe_tool(engine, tc, hook_match_predicate)
                for tc in pending_tool_calls
            ]
        else:
            parallel_eligible = [False] * len(pending_tool_calls)

        # Delegation fan-out eligibility — a SEPARATE partition from the read
        # fan-out above. Disabled entirely (every call marked ineligible → the
        # serial path) when the master gate is off OR the effective concurrency
        # cap resolves to ``< 2`` (a cap of 1 ⇒ sequential ⇒ the exact serial
        # path). A read-parallel-eligible call is never delegation-eligible
        # (delegation tools are NOT ``is_concurrent_safe``), so the two
        # partitions are disjoint by construction.
        subagent_concurrency_cap = max(1, engine.config.rc.max_concurrent_subagents)
        if subagent_concurrency_cap >= 2:
            delegation_eligible = [
                _is_delegation_parallel_safe(engine, tc, hook_match_predicate)
                for tc in pending_tool_calls
            ]
        else:
            delegation_eligible = [False] * len(pending_tool_calls)

        # Record that this run hands work to subagents, once, for the whole run.
        # Set from the RAW structural predicate rather than from
        # ``delegation_eligible`` above: that list answers "may these calls fan
        # out concurrently", which a concurrency cap of 1 or a matching hook
        # turns False without making the call any less a delegation. Set HERE
        # rather than at the dispatch seam because the parallel gather and the
        # serial path reach that seam through different code and only this line
        # sees both.
        if not engine._run_delegated and any(
            _tool_is_delegation(engine, tc) for tc in pending_tool_calls
        ):
            engine._run_delegated = True

        # Walk the pending list, batching adjacent parallel-safe runs
        # together so we preserve the original interleaving with any
        # serial tools (a [read, write, read] turn becomes
        # [parallel(read)] → [serial(write)] → [parallel(read)] which
        # respects the model's intended causal order between the write
        # and the second read).
        idx = 0
        n = len(pending_tool_calls)
        # Bound each gather batch at ``parallel_read_tools_max_fanout`` so a
        # turn that emits many parallel-safe reads chunks into ≤N-wide
        # sub-batches (each independently snapshot→gather→restore→replayed,
        # preserving LLM-requested order) instead of fanning out unbounded
        # against a backend that degrades under load.
        #
        # ``0`` is the value-preserving sentinel: UNLIMITED fan-out (one
        # unbounded gather per adjacent parallel-eligible run). Only chunk
        # when the configured cap is ``> 0``. ``max_fanout = n`` for the
        # unlimited case keeps the ``(idx - batch_start) < max_fanout``
        # window from ever closing a
        # batch early, so a whole adjacent parallel run fans out at once.
        configured_fanout = engine.config.rc.parallel_read_tools_max_fanout
        max_fanout = configured_fanout if configured_fanout > 0 else n
        while idx < n:
            # Re-check before EACH dispatch step. A cancel can land DURING a
            # prior tool's dispatch ``await`` (the
            # tool's own ``run()``, a PreToolUse hook, a snapshot persist) or
            # in an awaited parallel-batch ``gather``. The pre-loop guard +
            # the post-hook-predicate guard only cover the window BEFORE the
            # loop starts; this checkpoint guarantees the loop never dispatches
            # a NEW serial tool or a NEW parallel batch once a stop has been
            # observed mid-loop. Tools already dispatched this turn keep their
            # real results; the remaining undispatched ``tool_use`` blocks
            # (still in history, unpaired) are synthesised into ``is_error``
            # results by the shared teardown so the snapshot stays
            # pairing-valid, then the run routes to CANCELLED.
            if engine.stop_requested and parked_calls:
                # A cancelled run waits for nothing: the decisions this
                # same message parked earlier can no longer be acted on.
                for _held in parked_calls:
                    engine.release_interrupt(_held.interrupt_id)
                parked_calls.clear()
                flags.approval_pending = False
            async for _evt in _cancel_checkpoint(engine, flags, policies):
                yield _evt
            if flags.terminal_yielded:
                return
            if parallel_eligible[idx]:
                batch_start = idx
                while (
                    idx < n
                    and parallel_eligible[idx]
                    and (idx - batch_start) < max_fanout
                ):
                    idx += 1
                batch = pending_tool_calls[batch_start:idx]
                if len(batch) == 1:
                    # Single-element batch — fall through to the serial
                    # dispatcher so behaviour is identical to the
                    # pre-Wave-10 path (no asyncio.gather overhead,
                    # event ordering is trivially preserved).
                    held_count = len(parked_calls)
                    async for evt in dispatch_parking_holds(
                        engine,
                        batch[0],
                        dispatch=_dispatch_tool,
                        park=_park_pause_interrupt,
                        parked=parked_calls,
                    ):
                        yield evt
                    if len(parked_calls) > held_count:
                        flags.approval_pending = True
                        continue
                    if flags.approval_pending:
                        continue
                    if _history_tool_result_is_terminal(engine, batch[0].id):
                        # One bounded self-verify turn before finalising. If a
                        # corrective turn is injected, do NOT finalise: break
                        # the dispatch loop so the outer loop runs one more
                        # bounded turn with the correction in history. The
                        # helper persists the snapshot itself on injection so
                        # the correction + latch survive a crash/resume.
                        if await _finalise_or_verify_first(engine, flags):
                            max_messages = max(
                                max_messages, flags.assistant_message_idx + 1
                            )
                            self_verify_injected = True
                        break
                    continue

                # ≥2 parallel-safe tool calls — fan out under
                # ``asyncio.gather`` so each PCM/HTTP RPC waits in
                # parallel rather than serialising the timeouts.
                #
                # Take the transcript-order state BEFORE gather so it can be
                # put back afterwards and the state transitions replayed in the
                # LLM-requested order. Without that, the streaks would track
                # gather completion order rather than the order the run is a
                # record of — see :meth:`RunScopedState.transcript_state` and
                # :func:`_replay_dispatch_state` for the contract.
                transcript_state = engine.run_state.transcript_state()
                await _record_batch_tool_intents(engine, batch)
                results = await asyncio.gather(
                    *(_drain_dispatch_tool_deferred(engine, tc) for tc in batch),
                    return_exceptions=False,
                )
                # Put the streak/satisfaction state back to pre-gather. The
                # parallel mutations are discarded; the replay below applies the
                # deterministic transcript-order state transitions.
                engine.run_state.restore_transcript_state(transcript_state)
                _logger.warning(
                    "DIAG query.parallel_batch.entered run=%s tenant=%s "
                    "turn=%s batch_size=%d tools=%s",
                    engine.config.run_id,
                    engine.config.tenant_id,
                    engine.turn_id(),
                    len(batch),
                    ",".join(tc.name for tc in batch),
                )
                # Emit events + apply deferred history mutations in the
                # ORIGINAL tool-call order regardless of gather
                # completion order — preserves the LLM-facing invariant.
                batch_drained = False
                for tool_call, (events, outcome) in zip(batch, results, strict=True):
                    if batch_drained:
                        # A tool in this batch ended the turn, so what the
                        # ones behind it produced is no longer part of the
                        # run. They are concurrent-safe and non-destructive
                        # by construction, so dropping their results has no
                        # external effect.
                        engine.forget_tool_name(tool_call.id)
                        continue
                    if outcome is None:
                        # Defensive: dispatcher should always yield a
                        # final outcome. Mirror the warning emitted by
                        # the serial path so silent drops are visible.
                        _logger.warning(
                            "tool dispatcher returned no outcome for "
                            "call_id=%s (parallel batch)",
                            tool_call.id,
                        )
                        engine.forget_tool_name(tool_call.id)
                        continue
                    if outcome.approval_required:
                        # The hook match predicate should have steered any
                        # hook-gated tool onto the serial path, so this branch
                        # is a mid-turn hook race. Every held call of the
                        # batch is parked, in LLM order — the same guarantee
                        # the serial path gives, rather than the "first one
                        # only" it used to have.
                        _logger.warning(
                            "DIAG query.parallel_batch.approval_parked "
                            "run=%s tenant=%s turn=%s tool=%s call_id=%s",
                            engine.config.run_id,
                            engine.config.tenant_id,
                            engine.turn_id(),
                            tool_call.name,
                            tool_call.id,
                        )
                        flags.approval_pending = True
                        async for evt in park_deferred_hold(
                            engine,
                            tool_call.id,
                            events,
                            park=_park_pause_interrupt,
                            parked=parked_calls,
                        ):
                            yield evt
                        # No tool_result is appended for a call that has not
                        # run; the decision produces one, or the abandon does.
                        continue
                    # a terminal-only-blocked tool produced a
                    # SYNTHETIC outcome (success=False, is_error=True) WITHOUT
                    # any dispatcher invocation (see
                    # :func:`_drain_dispatch_tool_deferred`). The SERIAL
                    # ``_dispatch_tool`` terminal-only short-circuit appends the
                    # blocked tool_result and returns WITHOUT touching the
                    # durable consecutive-error streak. Running the normal
                    # replay below would take the error path and call
                    # ``_apply_consecutive_error_cap``, INCREMENTING the streak
                    # in parallel mode but not serial mode (and the rewrite
                    # would also overwrite the ``terminal_only`` error kind
                    # with ``execution``). Detect the blocked synthetic by
                    # re-checking the SAME predicate that produced it (no
                    # terminal result has been appended for these non-terminal
                    # blocked reads, so the predicate is still True), then
                    # emit the ORIGINAL blocked events + append history with the
                    # ORIGINAL outcome — bit-identical to the serial path, no
                    # streak mutation.
                    if _terminal_only_blocks(engine, tool_call):
                        for evt in events:
                            yield evt
                        # This synthetic terminal-only veto must NOT feed the
                        # Repeated-tool-error breaker (parity with the serial path,
                        # which returns before breaker tracking). Otherwise a
                        # finalize-gate veto could trip the breaker mid-gate.
                        _apply_deferred_tool_history(
                            engine,
                            tool_call,
                            outcome,
                            track_circuit_breaker=False,
                        )
                        continue
                    # Replay the streak + satisfaction state transitions
                    # against the REAL run state in LLM-requested order so
                    # both the visible transcript and next-turn caps follow
                    # transcript-correct counts regardless of gather
                    # completion order.
                    _logger.warning(
                        "DIAG query.parallel_batch.helper_replay run=%s "
                        "tenant=%s turn=%s tool=%s call_id=%s success=%s",
                        engine.config.run_id,
                        engine.config.tenant_id,
                        engine.turn_id(),
                        tool_call.name,
                        tool_call.id,
                        outcome.success,
                    )
                    adjusted_outcome = _replay_dispatch_state(
                        engine, tool_call, outcome
                    )
                    for evt in _rewrite_deferred_tool_result_events(
                        events, adjusted_outcome
                    ):
                        yield evt
                    _apply_deferred_tool_history(engine, tool_call, adjusted_outcome)
                    if not flags.approval_pending and _dispatch_outcome_is_terminal(
                        adjusted_outcome,
                        engine=engine,
                        tool_name=tool_call.name,
                    ):
                        # Nothing ends the run while a call of the same message
                        # is still held for a decision: the run is waiting, and
                        # a seal over a wait is a run nothing comes back for.
                        # ``batch_drained`` stops the rest of the batch either
                        # way. When a corrective turn is injected the turn is
                        # NOT finalised — the outer loop re-drives it — and the
                        # helper persists the snapshot itself on injection.
                        if await _finalise_or_verify_first(engine, flags):
                            max_messages = max(
                                max_messages, flags.assistant_message_idx + 1
                            )
                            self_verify_injected = True
                        batch_drained = True
                # One snapshot per batch instead of one per tool —
                # parity with the serial path's per-tool persist but
                # batched to amortise the PG/Redis round-trip.
                await engine._persist_snapshot()
                if flags.approval_pending or flags.terminal_tool_completed or self_verify_injected:
                    break
                continue

            if delegation_eligible[idx]:
                # Group the MAXIMAL run of adjacent delegation-eligible calls.
                delegation_start = idx
                delegation_end = idx
                while delegation_end < n and delegation_eligible[delegation_end]:
                    delegation_end += 1
                if delegation_end - delegation_start >= 2:
                    idx = delegation_end
                    batch = pending_tool_calls[delegation_start:delegation_end]
                    # ≥2 adjacent delegation calls — fan them out under a bounded
                    # semaphore so up to ``max_concurrent_subagents`` child runs
                    # execute concurrently and any excess serialise in waves. Each
                    # child goes through the SAME deferred dispatcher the read
                    # fan-out uses (history append deferred so results land in
                    # LLM-requested order); the leader still blocks until the whole
                    # group finishes (blocking join). Delegation tools are NOT
                    # ``is_concurrent_safe`` so this is a SEPARATE path from the
                    # read fan-out above — but the leader's dispatcher mutates the
                    # shared per-run state (consecutive-error streak,
                    # cumulative tool-call soft cap, satisfied-precondition set) per
                    # child, so we snapshot that transcript-order state BEFORE the
                    # gather, restore it AFTER, and replay the transitions in
                    # LLM-requested order — the same correctness contract the read
                    # path relies on.
                    #
                    # TWO NESTED BOUNDS apply here. This per-turn semaphore bounds
                    # the WIDTH of THIS leader turn's group only — a fresh one is
                    # minted per turn. On its own that composes MULTIPLICATIVELY
                    # across depth (each nested group carries its own independent
                    # semaphore, so a depth-2 tree of width W runs up to W*W
                    # children at once). The SECOND bound — ``budget``, a
                    # tree-wide SubagentTreeBudget shared by reference down the
                    # whole run tree — caps the ADDITIVE sum of concurrently
                    # executing children across every nested group, so depth no
                    # longer multiplies (see rc.max_concurrent_subagents_per_tree).
                    #
                    # The tree bound is deadlock-free by construction, which the
                    # naive "one shared semaphore held around each child's whole
                    # run" is NOT: that naive scheme wedges because a parent pins
                    # its permit for the child's ENTIRE blocking join while the
                    # child needs permits for its OWN grandchildren, so at the cap
                    # the held permits starve the nested acquires. The scheme here
                    # is release-while-awaiting-children (leaf-counting): a tree
                    # slot is acquired per child at the DISPATCH site
                    # (``_dispatch_subagent_under_semaphore``), and a run that is
                    # blocked awaiting its OWN children RELEASES its slot for the
                    # duration of that wait (below) and reacquires it after. So
                    # every slot holder is a run doing local work — none is blocked
                    # on descendants — and no holder needs a further slot to finish
                    # its current slice; the budget can never form an acquisition
                    # cycle. ``budget`` is resolved (minted at the first parallel
                    # fan-out from the RC, then found on the run state) so the
                    # whole parallel-dispatched subtree shares the SAME object;
                    # ``tree_permit`` is THIS run's own slot (None for a run that
                    # was never dispatched under the budget — e.g. the root, or a
                    # run reached only by serial delegation) — present only for a
                    # child that was itself dispatched under the budget.
                    semaphore = asyncio.Semaphore(subagent_concurrency_cap)
                    budget = _resolve_subagent_tree_budget(engine)
                    tree_permit = _resolve_subagent_tree_permit(engine)
                    transcript_state = engine.run_state.transcript_state()
                    # Stable id for THIS fan-out group, shared by every child and
                    # distinct across groups/turns (tool_call ids are unique per
                    # run). Lets the parent ledger scope same-path batch-order
                    # resolution per group so a later turn's group is never frozen
                    # out by an earlier one (see AttemptLedger.declare).
                    dispatch_group = batch[0].id
                    # Release this run's tree slot BEFORE blocking on its children
                    # and reacquire AFTER (finally, so a raising gather still
                    # reacquires) — the crux of the deadlock-free scheme. No-op for
                    # the root leader (holds no permit) and under the unlimited
                    # sentinel.
                    if tree_permit is not None:
                        await tree_permit.release_while_waiting()
                    await _record_batch_tool_intents(engine, batch)
                    try:
                        gathered = await asyncio.gather(
                            *(
                                _dispatch_subagent_under_semaphore(
                                    engine,
                                    tc,
                                    semaphore,
                                    budget,
                                    dispatch_order=batch_pos,
                                    dispatch_group=dispatch_group,
                                )
                                for batch_pos, tc in enumerate(batch)
                            ),
                            return_exceptions=True,
                        )
                    finally:
                        if tree_permit is not None:
                            await tree_permit.reacquire()
                    engine.run_state.restore_transcript_state(transcript_state)
                    _logger.warning(
                        "DIAG query.parallel_subagents.entered run=%s tenant=%s "
                        "turn=%s batch_size=%d cap=%d tools=%s",
                        engine.config.run_id,
                        engine.config.tenant_id,
                        engine.turn_id(),
                        len(batch),
                        subagent_concurrency_cap,
                        ",".join(tc.name for tc in batch),
                    )
                    # Normalise gather results in LLM order. A child that RAISED
                    # (defensive — the dispatcher normally returns structured
                    # error outcomes for unknown subagent_type / hook block /
                    # timeout) is converted into its own error outcome so siblings
                    # still complete. A ``BaseException`` that is NOT an
                    # ``Exception`` (``CancelledError`` / ``SystemExit`` /
                    # ``KeyboardInterrupt``) is re-raised so cancellation is never
                    # swallowed by ``return_exceptions=True``.
                    normalised_results: list[tuple[list[TurnEvent], DispatchOutcome | None]] = []
                    for tool_call, raw in zip(batch, gathered, strict=True):
                        if isinstance(raw, BaseException):
                            if not isinstance(raw, Exception):
                                raise raw
                            _logger.warning(
                                "DIAG query.parallel_subagents.child_raised run=%s "
                                "tenant=%s turn=%s tool=%s call_id=%s error=%s",
                                engine.config.run_id,
                                engine.config.tenant_id,
                                engine.turn_id(),
                                tool_call.name,
                                tool_call.id,
                                type(raw).__name__,
                            )
                            normalised_results.append(
                                _synthesize_delegation_error_result(
                                    engine, tool_call, raw
                                )
                            )
                        else:
                            normalised_results.append(raw)

                    # Emit events + apply deferred history in the ORIGINAL
                    # tool-call order regardless of gather completion order.
                    batch_drained = False
                    for tool_call, (events, outcome) in zip(
                        batch, normalised_results, strict=True
                    ):
                        if batch_drained:
                            # A child in this group ended the turn; what the
                            # groups behind it produced is no longer part of
                            # the run.
                            engine.forget_tool_name(tool_call.id)
                            continue
                        if outcome is None:
                            _logger.warning(
                                "tool dispatcher returned no outcome for "
                                "call_id=%s (parallel subagent batch)",
                                tool_call.id,
                            )
                            engine.forget_tool_name(tool_call.id)
                            continue
                        if outcome.approval_required:
                            # The eligibility predicate steers a hook-gated
                            # delegation call onto the serial path, so this is
                            # a gate registered mid-turn. Every held child is
                            # parked, in the order the model asked for them —
                            # a group answered one decision at a time would
                            # cost one stop, one snapshot and one redraw per
                            # child.
                            flags.approval_pending = True
                            async for evt in park_deferred_hold(
                                engine,
                                tool_call.id,
                                events,
                                park=_park_pause_interrupt,
                                parked=parked_calls,
                            ):
                                yield evt
                            continue
                        if _terminal_only_blocks(engine, tool_call):
                            # Synthetic terminal-only veto (no dispatcher
                            # invocation) — emit the blocked events + append
                            # history WITHOUT feeding the circuit breaker, exactly
                            # as the serial path returns before breaker tracking.
                            for evt in events:
                                yield evt
                            _apply_deferred_tool_history(
                                engine,
                                tool_call,
                                outcome,
                                track_circuit_breaker=False,
                            )
                            continue
                        # Replay the streak + satisfaction + soft-cap transitions
                        # against the run's own state in LLM-requested order so
                        # the transcript and next-turn caps follow transcript-order
                        # counts regardless of gather completion order.
                        adjusted_outcome = _replay_dispatch_state(
                            engine, tool_call, outcome
                        )
                        for evt in _rewrite_deferred_tool_result_events(
                            events, adjusted_outcome
                        ):
                            yield evt
                        _apply_deferred_tool_history(
                            engine, tool_call, adjusted_outcome
                        )
                        if (
                            not flags.approval_pending
                            and _dispatch_outcome_is_terminal(
                                adjusted_outcome,
                                engine=engine,
                                tool_name=tool_call.name,
                            )
                        ):
                            # Nothing seals the run while a child of the same
                            # message is still held for a decision. One bounded
                            # self-verify turn before finalising otherwise —
                            # parity with the read and serial paths.
                            if await _finalise_or_verify_first(engine, flags):
                                max_messages = max(
                                    max_messages, flags.assistant_message_idx + 1
                                )
                                self_verify_injected = True
                            batch_drained = True
                    # One snapshot per batch (parity with the read fan-out).
                    await engine._persist_snapshot()
                    if (
                        flags.approval_pending
                        or flags.terminal_tool_completed
                        or self_verify_injected
                    ):
                        break
                    continue
                # Group of exactly one delegation call → fall through to the exact
                # serial path below (no gather, byte-identical single-call
                # behaviour). ``idx`` is unchanged, so the serial block dispatches
                # this call and advances.

            # Serial path — single non-parallel-safe tool call. A run holding a
            # tree-budget slot that dispatches a single delegation call blocks on
            # the child's whole nested run; the tree slot is released around that
            # join INSIDE :func:`_dispatch_tool` (the single choke point every
            # serial-style delegation await funnels through), so this call site
            # needs no permit handling of its own.
            tool_call = pending_tool_calls[idx]
            idx += 1
            held_count = len(parked_calls)
            async for evt in dispatch_parking_holds(
                engine,
                tool_call,
                dispatch=_dispatch_tool,
                park=_park_pause_interrupt,
                parked=parked_calls,
            ):
                yield evt
            if len(parked_calls) > held_count:
                # Every call a gate holds is parked, not only the first one of
                # the message. Stopping at the first left the rest of the batch
                # to the wire repair: an operator was shown one card, approved
                # it, and resumed into a run that had quietly dropped the
                # calls behind it. The walk carries on so the whole message is
                # decided in one round of asking.
                flags.approval_pending = True
                continue
            if flags.approval_pending:
                # Something in this message is already held for a decision, so
                # nothing here may end the run: the terminal below and the
                # corrective re-drive both assume the turn is free to move on.
                continue
            #  — if ``_dispatch_tool`` vetoed this
            # terminal via the prose-gate (appended its non-terminal error
            # result + the corrective user turn), STOP draining the rest of this
            # assistant batch: do NOT dispatch later sibling tool calls AFTER
            # the injected user-repair turn (that would interleave a user
            # message between sibling tool results in the durable snapshot).
            # Mirror ``self_verify_injected`` — break WITHOUT finalising; the
            # outer loop re-drives one corrective turn.
            if _prose_gate_just_injected(engine):
                max_messages = max(max_messages, flags.assistant_message_idx + 1)
                self_verify_injected = True
                break
            if _history_tool_result_is_terminal(engine, tool_call.id):
                # One bounded self-verify turn before finalising (serial
                # path). The helper persists the snapshot itself on injection.
                if await _finalise_or_verify_first(engine, flags):
                    max_messages = max(
                        max_messages, flags.assistant_message_idx + 1
                    )
                    self_verify_injected = True
                break
        if flags.approval_pending:
            if parked_calls:
                # One announcement for the whole message: the run goes to
                # AWAITING once, with everything it is waiting on visible in a
                # single envelope, so a host draws its cards in one pass.
                engine.transition_to(LoopState.AWAITING)
                await engine._persist_snapshot()
                yield _interrupt_parked_event(engine, parked_calls[0])
            return

        if flags.terminal_tool_completed:
            # The model completed the run of its own accord by calling the
            # run-terminal tool. What that ending IS — the pairing of the
            # calls the turn abandoned, and the seal — belongs to the finish
            # policies, so every dispatch path that reaches it ends the same.
            async for _evt in _complete_via_terminal_tool(engine, flags, policies):
                yield _evt
            return

        # ── message_stop (tool_use) — between assistant messages ────
        previous_tool_results_ready_at = time.perf_counter()
        yield TurnEvent(
            type=EventType.MESSAGE_STOP,
            run_id=engine.config.run_id,
            payload={
                "turn_id": engine.turn_id(),
                "stop_reason": "tool_use",
                "tokens_used": _tokens_used_payload(engine),
                "cache_hit_rate": engine.total_usage.this_turn_cache_hit_rate(),
            },
        )

        # The results of the batch just dispatched are in history and the
        # next stream is about to be built from all of it — the seam where
        # the transcript grows, and so the seam the compaction gate sits at.
        _turn = _turn_at(engine, flags, TurnCoordinate.iteration_end)
        async for _policy_evt in policies.apply(_turn):
            yield _policy_evt
        max_messages = _turn_budget_after(_turn.outcome, max_messages, flags)
        if _turn.outcome.directive is TurnDirective.end_turn:
            return
        if _turn.outcome.directive is TurnDirective.restart_turn:
            if _turn.outcome.rebuild_context:
                current_context = await _rebuild_context_for_recovery(engine)
            continue

        _turn = _turn_at(engine, flags, TurnCoordinate.turn_end, dispatched_tools=True)
        async for _policy_evt in policies.apply(_turn):
            yield _policy_evt
        max_messages = _turn_budget_after(_turn.outcome, max_messages, flags)
        if _turn.outcome.directive is TurnDirective.end_turn:
            return
        if _turn.outcome.directive is TurnDirective.restart_turn:
            if _turn.outcome.rebuild_context:
                current_context = await _rebuild_context_for_recovery(engine)
            continue

        # ── Build next context (tool_results now in history) ────────
        await _reload_live_control(engine)
        async for _bg_evt in _background_wake_events(engine):
            yield _bg_evt
        steer_evt = _inject_steer_into_history(engine)
        if steer_evt is not None:
            yield steer_evt
            await _persist_live_control(engine)
        tool_defs = list(
            engine.tools.compute_effective_surface(
                tenant_id=engine.config.tenant_id,
                policy=engine.effective_tool_policy,
                query=engine.latest_user_message.text if engine.latest_user_message else "",
                top_k=engine.config.rc.tool_retrieval_top_k,
            )
        )
        next_history, _ = _llm_history(engine)
        # Reuse the run's once-built skill catalog (NOT rebuilt per iteration —
        # a per-turn rebuild would bust the cached system-prompt prefix).
        current_context = engine.context_manager.build_context(
            history=next_history,
            tools=tool_defs,
            system_prompt_sections=engine.config.system_prompt_sections,
            skill_index_block=await _ensure_run_skill_catalog(engine),
            skills_loaded=engine._skill_loaded_bundles,
        )
        # The rebuilt context (above) carries the injected continue message; the
        # forced tool_choice rides on a transient engine attr consumed by the
        # next stream. The ``max_messages`` bump above is the only other effect.
        # Loop iterates — opens the next assistant LLM stream.


class _StreamAttemptResult:
    """Mutable accumulator for one streaming attempt.

    Owned by :func:`_stream_one_assistant_message`; mutated by
    :func:`_drive_one_stream` so the caller can branch on
    ``finish_reason`` / ``tool_calls`` post-stream. Internal — never
    yielded to consumers.

    ``reasoning_buffer`` accumulates ``ProviderDeltaKind.thinking``
    content for detecting an assistant turn that emits reasoning_content
    but NO visible text AND NO tool_calls — the "thinking eats all tokens"
    failure mode — the loop injects a continue-prompt and re-streams up to
    ``rc.max_consecutive_empty_responses`` times before terminal.

    The text/reasoning content is accumulated into ``list[str]`` fragment
    lists and joined ONCE on read via the ``text_buffer`` /
    ``reasoning_buffer`` properties.
    Per-delta ``str += fragment`` is O(n^2) over the buffer length —
    a large Write/HTML generation streamed as thousands of tiny deltas
    turned the consumer into a quadratic CPU section on the single
    executor event loop, which (with unbounded concurrent runs)
    starved a neighbour run's pending provider socket read and produced
    a FALSE ``provider stream produced no data`` stall. These buffers
    are write-only until end-of-stream (every reader in
    :func:`_stream_one_assistant_message` runs AFTER the
    ``async for ... _drive_one_stream`` loop completes), so deferring
    the join is observationally identical.
    """

    __slots__ = (
        "_reasoning_fragments",
        "_text_fragments",
        "finish_reason",
        "narration_prefix_chars",
        "tool_calls",
    )

    def __init__(self) -> None:
        self.finish_reason: str | None = None
        self.tool_calls: list[ToolCall] = []
        self._text_fragments: list[str] = []
        self._reasoning_fragments: list[str] = []
        # Where the live stream cut this message's prose into a collapsed
        # leading narration and the answer behind it — an offset into
        # ``text_buffer``, 0 when nothing was split. Recorded by the stream so
        # the durable blocks are built from the SAME decision the reader
        # already saw, never from a second reading of the finished text.
        self.narration_prefix_chars: int = 0

    def append_text(self, fragment: str) -> None:
        if fragment:
            self._text_fragments.append(fragment)

    def append_reasoning(self, fragment: str) -> None:
        if fragment:
            self._reasoning_fragments.append(fragment)

    @property
    def text_buffer(self) -> str:
        return "".join(self._text_fragments)

    @property
    def reasoning_buffer(self) -> str:
        return "".join(self._reasoning_fragments)


async def _drive_one_stream(
    engine: QueryEngine,
    context: ContextBundle,
    result: _StreamAttemptResult,
    *,
    previous_tool_results_ready_at: float | None = None,
) -> AsyncIterator[TurnEvent]:
    """Drive one provider stream + emit per-delta TurnEvents.

 Mutates ``result`` in place so the caller can branch on
 ``finish_reason`` / ``tool_calls`` post-stream. Per the
 streaming body is a separate helper so the outer recovery loop in
 :func:`_stream_one_assistant_message` can wrap it in
 :class:`LLMContextWindowExceeded` / future recovery handlers.

 Prompt-cache breakpoint hints are added to :attr:`LLMRequest.extra`.
 The hints are produced by
 :func:`protocore.runtime.prompt_caching.apply_system_and_3` (pure
 function — same input → same output, no IO). The Anthropic adapter
 consumes them; OpenAI / vLLM ignore them.
 """
    rc = engine.config.rc
    max_output_tokens = max(
        1,
        int(context.budgets.max_context * rc.llm_output_max_tokens_ratio),
    )
    # The global per-message output cap is ``max_context * ratio`` (the KB
    # binding cap). Keep it so the Item-4 final-turn floor below can never
    # exceed it — the reserve claws tokens back DOWN within this cap, it is
    # NOT a global-cap raise.
    output_cap_before_band = max_output_tokens
    # AdaptiveSafetyBand. The band subtracts a
    # calibrated drift margin from the per-call output budget so the
    # prompt + max_tokens stay under the provider context window even
    # when the local token estimator misjudges Cyrillic-in-JSON-escape
    # inflation. The band is per-(provider, model); when no band is
    # wired (kill-switch off / test fixture) the helper returns 0 and
    # behaviour is identical to pre-A4.
    safety_band = _resolve_safety_band_value(engine)
    if safety_band > 0:
        max_output_tokens = max(1, max_output_tokens - safety_band)
    # Final-turn-specific output-token reserve. Floors the post-safety-band
    # budget on the terminal / forced-final turn so the model can emit
    # message + refs + outcome. Default-off ⇒ no-op; never raises the
    # global cap (bounded by ``output_cap_before_band``).
    max_output_tokens = _apply_terminal_synthesis_output_reserve(
        engine, max_output_tokens, output_cap_before_band
    )
    full_messages = _prepend_system_sections(
        context.system_prompt_sections,
        context.messages,
    )

    # UNCONDITIONAL tool_use<->tool_result pairing repair at the wire
    # boundary, immediately before LLMRequest assembly. Applied on every
    # request: forward-fills synthetic is_error tool_results for orphaned
    # tool_use, reverse-strips orphaned tool_results, and dedupes duplicate
    # ids — so the wire is robust to orphaning from ANY source (Tier-2
    # compaction dropping one side, resume-from-partial-batch, max_tokens
    # truncation), not just the cases the orphan PRODUCERS were patched for.
    # Runs BEFORE the cache-breakpoint computation so the breakpoint indices
    # address the final outbound list.
    full_messages = _repair_outbound_tool_pairing(
        full_messages,
        placeholder=engine.prompt_text("tool_result_pairing_repair"),
    )

    # vLLM-400 backstop — normalize any non-leading ``system`` message to
    # ``user`` at the SAME wire boundary as the pairing repair. vLLM 400s on a
    # ``system`` message past index 0 ("System message must be at the
    # beginning."); a mid-history Tier-2 compaction summary used to be
    # system-role (now fixed at source, but legacy persisted snapshots may
    # still carry one). The genuine system prefix at index 0 is untouched.
    full_messages, _converted_system = _normalize_outbound_system_messages(
        full_messages
    )
    if _converted_system and not getattr(
        engine, "_outbound_system_normalized_warned", False
    ):
        engine._outbound_system_normalized_warned = True  # type: ignore[attr-defined]
        _logger.warning(
            "normalized %d non-leading system message(s) to user role at the "
            "request boundary (vLLM-400 guard; likely a legacy compaction "
            "summary from a persisted snapshot)",
            _converted_system,
        )

    # Compute prompt-cache breakpoints once per provider call. The pure
    # function is cheap (O(n) over messages,
    # no allocation hot-spots) so we recompute every iteration of
    # ``_stream_one_assistant_message`` rather than thread cache state
    # through the engine. Hints land on
    # :attr:`LLMRequest.extra["cache_breakpoints"]` — adapters that
    # don't recognise the key ignore it (vLLM, OpenAI). Anthropic
    # translates each :class:`CacheBreakpoint` into a wire-format
    # ``cache_control`` block per its detected
    # :class:`CachePolicy`.
    cache_breakpoints = apply_system_and_3(list(full_messages))
    # The forced-tool slot. ``build_llm_request`` places it on
    # ``extra["forced_tool_choice"]``; the native-thinking axis it threads
    # alongside (``enable_thinking`` + ``reasoning_effort``, always as a pair
    # so CoT stays bounded) comes straight off the engine's live controls.
    forced_tool_choice: str | None = None
    # The forced-tool slot is a SINGLE slot carrying the tool NAME
    # the host vLLM/OpenAI-compatible adapter translates into a native
    # ``tool_choice={type:function, function:{name}}``; a provider/adapter that
    # does not recognise the key ignores it. Three independent mechanisms want
    # that slot, and in every case it may only ever name a tool THIS request
    # actually advertises — a forced choice for an unadvertised tool 400s the
    # whole request, and a compacted/BM25-clipped surface can drop a tool
    # any of them wanted.
    #
    # A run-level tool PRECONDITION wins the slot. The three are different
    # concerns — the precondition is a promise to the caller that this tool
    # runs before the agent answers, the convergence hint is an internal nudge
    # toward finishing a file, the read-back gate is a debt the agent took on
    # by delegating — and only the promise has a deadline. While a
    # precondition is outstanding the other two are not even READ, let alone
    # popped, so whatever they decided is still pending, and still forced, once
    # the preconditions are done.
    #
    # Convergence beats read-back for the same reason: its hint is transient
    # (it forces exactly the next stream, and a stream spent elsewhere is a
    # stream a half-written file spends unfinished), while the pending-read set
    # is durable and loses nothing by waiting a turn.
    precondition_tool = _preconditions.outstanding_tool(engine)
    if precondition_tool is not None:
        # Unlike the hint below, an unforceable precondition is charged as a
        # SPENT attempt rather than deferred: the caller was promised this
        # call, so a surface that never offers the tool has to end the run
        # rather than let it answer anyway.
        if any(
            getattr(t, "name", None) == precondition_tool for t in context.tools
        ):
            forced_tool_choice = precondition_tool
            _preconditions.charge_attempt(engine)
        else:
            _preconditions.charge_attempt(
                engine,
                error=(
                    f"{precondition_tool} was not advertised on this turn's "
                    "tool surface, so it could not be forced"
                ),
            )
    else:
        # The convergence driver's hint is consumed (popped) here so it forces
        # exactly the next stream, and degrades to a strong prose nudge —
        # still bounded — on an adapter that ignores the key. PEEK first, only
        # pop when the surface includes the tool: ``AppendFile`` /
        # ``FinalizeFile`` are NOT in ``tool_surface_forced_pins`` by default,
        # so a BM25-clipped surface can drop the forced tool while the continue
        # message + ``commit_forced_*`` charge are already settled — an
        # unconditional pop dropped the hint and the model never saw a
        # ``forced_tool_choice``. A future stream whose surface does include
        # the tool picks the hint back up.
        forced_tool = _longfile.peek_force_next_tool(engine)
        if (
            forced_tool is not None
            and any(getattr(t, "name", None) == forced_tool for t in context.tools)
        ):
            # Surface includes the tool — consume the hint exactly once.
            _longfile.take_force_next_tool(engine)
            forced_tool_choice = forced_tool
        else:
            # Nothing else wants the slot — a convergence hint that could not
            # be forced this turn has left it empty and kept its own state, so
            # taking it here costs that driver nothing.
            #
            # The read-back gate: while a tool result's declared files are
            # unread, force the read tool so the agent cannot answer out of the
            # one-line pointer it was handed. Same peek-then-charge discipline
            # as the hint above — the pending set is NOT touched when the read
            # tool is missing from this turn's surface, and no attempt is
            # charged for a stream the model was never offered it on.
            released_reads = _pending_reads.release_exhausted(engine)
            if released_reads:
                # The bound is spent: the files could not be opened. Say so on
                # the run's event stream and hand the surface back, rather than
                # forcing a read that will never land.
                yield _emit_state_change(
                    engine,
                    engine.state,
                    engine.state,
                    reason="pending_reads_released",
                )
            readback_tool = _pending_reads.peek_forced_tool(engine)
            if readback_tool is not None:
                # STRICT terminal-only finalisation permits exactly one
                # dispatch, the run's terminal tool; a forced read under it
                # surfaces a ``terminal_only`` error and burns the turn. The
                # pending set survives, so a run that leaves the latch still
                # owes its reads (mirrors the convergence driver's own bail).
                #
                # Both bail-outs are RECORDED rather than merely taken: an
                # uncharged, unforced turn is otherwise indistinguishable in a
                # log from one where nothing ever declared a read-back.
                if _terminal_only_enforced(engine):
                    _pending_reads.note_not_forced(
                        engine, reason="terminal_only_enforced"
                    )
                elif not any(
                    getattr(t, "name", None) == readback_tool for t in context.tools
                ):
                    _pending_reads.note_not_forced(
                        engine, reason="tool_not_on_surface"
                    )
                else:
                    _pending_reads.charge_forced_attempt(engine)
                    forced_tool_choice = readback_tool
    yield TurnEvent(
        type=EventType.TOOL_SURFACE_ADVERTISED,
        run_id=engine.config.run_id,
        payload=_tool_surface_advertised_payload(engine, context),
    )
    request = build_llm_request(
        model=engine.effective_model_name,
        messages=full_messages,
        tools=context.tools,
        max_tokens=max_output_tokens,
        thinking_enabled=engine.effective_thinking_enabled,
        reasoning_effort=engine.effective_reasoning_effort,
        forced_tool_choice=forced_tool_choice,
        cache_breakpoints=cache_breakpoints,
        observability=_observability_context(
            engine,
            call_purpose="run",
            call_category=_provider_call_category(engine),
        ),
    )

    block_idx = engine.next_block_idx()
    # Track the KIND of the currently-open content block, not a bare
    # boolean. ``thinking`` and ``text`` deltas MUST land in SEPARATE typed
    # blocks: the chat reducer renders each block by its
    # ``content_block_start`` kind and silently drops deltas whose type
    # does not match the open block, so a thinking-then-answer stream under
    # one kind=thinking block loses the visible answer live.
    # ``None`` means no block is open.
    open_block_kind: ProviderDeltaKind | None = None
    # Track whether the most recently allocated ``block_idx`` belongs to a
    # TOOL block (set by ``tool_use_start``, cleared by ``tool_use_stop``).
    # The OpenAI wire permits interleaved ``delta.content`` while a tool
    # call is buffered open, so a text/thinking reopen can land with
    # ``open_block_kind is None`` but ``block_idx`` pointing at the
    # tool block. The reopen path uses this flag to advance
    # ``block_idx`` BEFORE emitting ``content_block_start``, guaranteeing
    # every emitted content block carries a unique ``block_idx`` per
    # turn (no two ``content_block_start`` for the same idx — a wire
    # block-model violation; the chat reducer otherwise replaces the
    # ``tool_use`` placeholder with the text block).
    last_block_was_tool: bool = False
    # Track whether a tool call that CONTINUES the run has already started
    # inside this assistant message — the whole signal behind the visibility
    # marking on every content block below. Deliberately NOT ``last_block_was_tool``:
    # that flag is block-index bookkeeping and must keep counting the terminal
    # gate call (which allocates an idx like any other), whereas this one must
    # not, because the terminal call sits in the same message as the answer.
    # Monotonic within the message — a run that has dispatched a tool cannot
    # un-dispatch it — which is what lets ``content_block_stop`` settle a value
    # that ``content_block_start`` had to send conservatively.
    tool_interleaved: bool = False
    if previous_tool_results_ready_at is not None:
        gap_ms = (time.perf_counter() - previous_tool_results_ready_at) * 1000.0
        _logger.warning(
            "DIAG query.internal_turn_gap run=%s tenant=%s turn=%s "
            "gap_ms=%.1f context_messages=%d context_tools=%d history_messages=%d",
            engine.config.run_id,
            engine.config.tenant_id,
            engine.turn_id(),
            gap_ms,
            len(full_messages),
            len(context.tools),
            len(engine.history),
        )
    await _manifest_request(engine, request, call_purpose="run")

    upstream = engine.llm.stream_with_tools(request)

    # Decide ONCE, up-front, whether this turn's visible assistant TEXT is the
    # redundant post-nudge META narration that must be suppressed from live SSE
    # + durable history. The decision is stable for the whole turn
    # (``_terminal_only_active`` + the prior-answer/background-tool facts do
    # not change mid-stream), so no per-delta cost and NO buffering: a
    # suppressed ``text`` delta is dropped before it opens a block or appends
    # to ``text_buffer``. THINKING + ALL tool calls (Write/AppendFile/Finalize)
    # are NEVER affected — only the user-facing text leak. See
    # :func:`_suppress_terminal_only_meta_text`.
    suppress_meta_text = _suppress_terminal_only_meta_text(engine)

    # ── Leading-narration split, for a run that delegated ──
    # ``head_buffer`` holds the start of this message's FIRST text block off the
    # wire until it is known whether that block opens with process narration.
    # The hold is bounded (scan ceiling + answer floor characters) and its
    # failure mode is deliberately mild: ``result.append_text`` still runs on
    # every delta as it arrives, so the durable text is complete whatever
    # happens to the buffer, and every path that closes the block flushes it.
    # ``None`` means not buffering — the split is off, or this block's head is
    # already settled, or the message's text started in an earlier block.
    narration_split_active = _answer_narration_split_active(engine)
    head_buffer: str | None = None
    head_scan_available = narration_split_active

    def _emit_pending_head(*, complete: bool) -> Iterator[TurnEvent]:
        """Release the buffered head, splitting it when the verdict is in.

        With ``complete=True`` a verdict is always available, so this is the
        call every block-close path makes and no buffered text can be stranded.
        Mid-stream it returns nothing while the run of narration can still
        grow. Leaves ``open_block_kind`` alone: on a split it closes the
        narration block and opens the answer block itself, so the caller's own
        ``content_block_stop`` still closes exactly one open text block.
        """

        nonlocal head_buffer, block_idx
        if head_buffer is None:
            return
        split = _settled_narration_split(engine, head_buffer, complete=complete)
        if split is None:
            return
        narration, answer = split
        head_buffer = None
        if narration:
            yield from _text_delta_events(engine, narration, block_idx=block_idx)
            yield TurnEvent(
                type=EventType.CONTENT_BLOCK_STOP,
                run_id=engine.config.run_id,
                payload={
                    "turn_id": engine.turn_id(),
                    "block_idx": block_idx,
                    "visibility": _content_block_visibility(
                        ProviderDeltaKind.text,
                        tool_interleaved=tool_interleaved,
                        narration_prefix=True,
                    ).value,
                },
            )
            block_idx = engine.next_block_idx()
            yield TurnEvent(
                type=EventType.CONTENT_BLOCK_START,
                run_id=engine.config.run_id,
                payload={
                    "turn_id": engine.turn_id(),
                    "block_idx": block_idx,
                    "kind": ProviderDeltaKind.text.value,
                    "visibility": _content_block_visibility(
                        ProviderDeltaKind.text, tool_interleaved=tool_interleaved
                    ).value,
                },
            )
            # The durable blocks are built from this number, so history and the
            # wire carry one decision rather than two readings of the same text.
            result.narration_prefix_chars = len(narration)
        if answer:
            yield from _text_delta_events(engine, answer, block_idx=block_idx)

    async for delta in _iter_with_idle_watchdog(
        _as_provider_deltas(upstream),
        idle_timeout=rc.llm_stream_idle_timeout_seconds,
        stall_threshold=rc.llm_stream_stall_threshold_seconds,
        reasoning_idle_timeout=rc.llm_stream_reasoning_idle_timeout_seconds,
    ):
        if engine.stop_requested:
            break

        if delta.kind is ProviderDeltaKind.finish:
            if open_block_kind is not None:
                # Release anything the narration split is still holding first —
                # its verdict is final now, and the block cannot close over
                # text the reader never received.
                for evt in _emit_pending_head(complete=True):
                    yield evt
                # Every ``content_block_stop`` carries the SETTLED visibility of
                # the block it closes, which is the value a reader should keep.
                # Recomputing here rather than echoing what the matching
                # ``content_block_start`` sent is safe in one direction only,
                # and that is the direction we need: ``tool_interleaved`` never
                # goes back to False inside a message, so the settled value can
                # tighten to ``collapsed`` but can never relax an already-sent
                # ``collapsed`` back to ``public``.
                yield TurnEvent(
                    type=EventType.CONTENT_BLOCK_STOP,
                    run_id=engine.config.run_id,
                    payload={
                        "turn_id": engine.turn_id(),
                        "block_idx": block_idx,
                        "visibility": _content_block_visibility(
                            open_block_kind, tool_interleaved=tool_interleaved
                        ).value,
                    },
                )
                open_block_kind = None
            result.finish_reason = delta.finish_reason
            break

        if delta.kind is ProviderDeltaKind.usage and delta.usage:
            cache_read = int(
                delta.usage.get("cache_read_input_tokens", 0)
                or delta.usage.get("cache_read_tokens", 0)
                or delta.usage.get("cached_tokens", 0)
            )
            cache_creation = int(delta.usage.get("cache_creation_input_tokens", 0))
            input_tokens = int(delta.usage.get("input_tokens", 0))
            engine.total_usage.add(
                input_tokens=input_tokens,
                output_tokens=int(delta.usage.get("output_tokens", 0)),
                cache_read_tokens=cache_read,
                cache_creation_tokens=cache_creation,
            )
            # The same envelope, charged a SECOND time against the whole tree's
            # cumulative ledger. ``total_usage`` belongs to THIS engine and every
            # delegated child gets a fresh one, so it can never answer "how much
            # has this question cost in total" — the ledger is shared by
            # reference with every descendant and does.
            _charge_run_work_tokens(
                engine,
                input_tokens=input_tokens,
                output_tokens=int(delta.usage.get("output_tokens", 0)),
            )
            # Ground-truth prompt size the provider reported for this call.
            # Every provider adapter normalises the FULL prompt (prompt_tokens,
            # inclusive of any cache-read portion) into ``input_tokens``, so it
            # already reflects total context-window occupancy — do NOT add
            # ``cache_read`` on top (that subset is already inside input_tokens
            # and would double-count). Floors the compaction gate against the
            # char heuristic, which under-counts adversarial content 2-3x.
            if input_tokens > 0:
                engine.last_observed_prompt_tokens = input_tokens
                _calibrate_token_estimate(engine, request, input_tokens)
            # Feed the optional cache observer one observation per usage
            # envelope. Core cannot import the host's metrics module (import
            # boundary); the host injects a concrete ``CacheObserverProtocol``
            # via ``QueryEngineConfig.cache_observer``. ``cache_breakpoints`` is
            # the placement-hint list assembled above by
            # :func:`apply_system_and_3` — adapters that ignore the key
            # still surface the count.
            observer = engine.config.cache_observer
            if observer is not None:
                cache_breakpoints_hint = request.extra.get(
                    "cache_breakpoints", []
                )
                observer.record_run_cache_hit_rate(
                    tenant_id=engine.config.tenant_id,
                    cache_read_tokens=cache_read,
                    prompt_tokens=input_tokens,
                    cache_breakpoint_count=len(cache_breakpoints_hint),
                )
            continue

        if delta.kind in (ProviderDeltaKind.text, ProviderDeltaKind.thinking):
            # Drop the redundant post-nudge META TEXT entirely: no
            # ``content_block_start`` / delta on the wire and no
            # ``append_text`` into ``text_buffer`` (so it never reaches
            # durable history nor ``result_preview``). Applies to TEXT only;
            # ``thinking`` still flows. Any open block is left as-is and is
            # closed by the next kind-transition / tool_use / finish handler,
            # so block-index bookkeeping stays valid. Tool calls on this same
            # turn (Write / AppendFile / Finalize) are unaffected — handled
            # by their own branches below.
            if suppress_meta_text and delta.kind is ProviderDeltaKind.text:
                continue
            if open_block_kind is not None and open_block_kind is not delta.kind:
                # Kind transition (thinking→text or text→thinking) — close
                # the open block and start a fresh one so every block stays
                # single-kind end-to-end, mirroring the per-kind buffers
                # the durable history keeps.
                for evt in _emit_pending_head(complete=True):
                    yield evt
                yield TurnEvent(
                    type=EventType.CONTENT_BLOCK_STOP,
                    run_id=engine.config.run_id,
                    payload={
                        "turn_id": engine.turn_id(),
                        "block_idx": block_idx,
                        "visibility": _content_block_visibility(
                            open_block_kind, tool_interleaved=tool_interleaved
                        ).value,
                    },
                )
                open_block_kind = None
                block_idx = engine.next_block_idx()
            if open_block_kind is None:
                # If the most recently allocated block was a tool
                # (``tool_use_start`` happened and ``tool_use_stop`` has not
                # yet arrived), the current ``block_idx`` belongs to that
                # tool block. Allocate a fresh idx for this content block so
                # the emitted ``content_block_start`` does not collide with
                # the tool block's idx on the wire. Mirrors the
                # ``tool_use_stop`` advance below for the post-stop reopen
                # case.
                if last_block_was_tool:
                    block_idx = engine.next_block_idx()
                # Say where this block belongs the moment it opens, so a live
                # reader can place it without waiting for the run to end and
                # without inspecting its text. ``public`` unless the model has
                # already committed to a tool call in this same message — see
                # :func:`_content_block_visibility` for why nothing stronger is
                # knowable yet and why the matching stop re-states it.
                yield TurnEvent(
                    type=EventType.CONTENT_BLOCK_START,
                    run_id=engine.config.run_id,
                    payload={
                        "turn_id": engine.turn_id(),
                        "block_idx": block_idx,
                        "kind": delta.kind.value,
                        "visibility": _content_block_visibility(
                            delta.kind, tool_interleaved=tool_interleaved
                        ).value,
                    },
                )
                open_block_kind = delta.kind
                # Arm the narration buffer on the block that STARTS this
                # message's prose. Only that one: the cut point is recorded as
                # an offset into ``text_buffer``, which is exactly this block's
                # offset while the buffer is still empty, and a later text
                # block in the same message shares a row with the tool call
                # between them and is already collapsed by ``tool_interleaved``.
                if head_scan_available and delta.kind is ProviderDeltaKind.text:
                    head_buffer = ""
                    head_scan_available = False
            if delta.kind is ProviderDeltaKind.text:
                # Accumulate BEFORE anything is emitted, so the durable text is
                # whole no matter what the wire-side buffer does with it.
                result.append_text(delta.content or "")
                if head_buffer is not None:
                    head_buffer += delta.content or ""
                    for evt in _emit_pending_head(complete=False):
                        yield evt
                    continue
            elif delta.kind is ProviderDeltaKind.thinking:
                # Accumulate reasoning_content: a turn that emits ONLY
                # thinking with no text/tool_calls is the "thinking-tokens
                # trap" — the recovery branch in
                # _stream_one_assistant_message injects a continue-prompt
                # and re-streams. List append + join-on-read instead of
                # O(n^2) ``str +=``.
                result.append_reasoning(delta.content or "")
            for evt in delta_to_turn_events(
                delta,
                run_id=engine.config.run_id,
                turn_id=engine.turn_id(),
                block_idx=block_idx,
            ):
                yield evt
            continue

        if delta.kind is ProviderDeltaKind.tool_use_start:
            # Flip the interleaving flag BEFORE closing the open block. This is
            # the moment the ambiguity the stream started with resolves: text
            # that had to open ``public`` because nothing yet proved otherwise
            # is now proven to be narration, and its ``content_block_stop`` is
            # the first frame able to say so. Doing this after the stop would
            # publish the stale value and leave the live view disagreeing with
            # the durable transcript for the rest of the run.
            tool_interleaved = tool_interleaved or _tool_call_continues_run(
                engine, delta.tool_name
            )
            if open_block_kind is not None:
                for evt in _emit_pending_head(complete=True):
                    yield evt
                yield TurnEvent(
                    type=EventType.CONTENT_BLOCK_STOP,
                    run_id=engine.config.run_id,
                    payload={
                        "turn_id": engine.turn_id(),
                        "block_idx": block_idx,
                        "visibility": _content_block_visibility(
                            open_block_kind, tool_interleaved=tool_interleaved
                        ).value,
                    },
                )
                open_block_kind = None
            block_idx = engine.next_block_idx()
            # Mark the freshly-allocated block as a tool block so a
            # subsequent text/thinking reopen path (open_block_kind is None,
            # last allocation was a tool) advances to a fresh idx instead of
            # reusing this one.
            last_block_was_tool = True
            if delta.tool_call_id and delta.tool_name:
                engine.remember_tool_name(delta.tool_call_id, delta.tool_name)
            for evt in delta_to_turn_events(
                delta,
                run_id=engine.config.run_id,
                turn_id=engine.turn_id(),
                block_idx=block_idx,
            ):
                yield evt
            continue

        if delta.kind is ProviderDeltaKind.tool_use_input:
            for evt in delta_to_turn_events(
                delta,
                run_id=engine.config.run_id,
                turn_id=engine.turn_id(),
                block_idx=block_idx,
            ):
                yield evt
            continue

        if delta.kind is ProviderDeltaKind.tool_use_stop:
            for evt in delta_to_turn_events(
                delta,
                run_id=engine.config.run_id,
                turn_id=engine.turn_id(),
                block_idx=block_idx,
            ):
                yield evt
            if delta.tool_call_id:
                # Propagate the parser's truncation signal onto the
                # :class:`ToolCall` so the recovery branch in
                # :func:`_stream_one_assistant_message` can distinguish
                # "the model emitted a complete tool call then stopped" from
                # "the model ran out of output tokens mid-tool-call args".
                # Default is ``False``; only set by the vLLM SSE parser on
                # its synthetic ``tool_use_stop`` emitted from the
                # ``finish_reason="length"`` branch.
                #
                # ``args_partial_truncated`` additionally surfaces every
                # wire-level truncation signature (stage-4 brace balancing
                # fired) regardless of ``finish_reason``. The detection
                # branch in :func:`_stream_one_assistant_message` keys off
                # this flag when ``finish_reason="stop"`` arrives mid-call.
                result.tool_calls.append(
                    ToolCall(
                        id=delta.tool_call_id,
                        name=engine.tool_name_for(delta.tool_call_id),
                        arguments=delta.tool_input_final or {},
                        truncated_by_output_cap=delta.truncated_by_output_cap,
                        args_partial_truncated=delta.args_partial_truncated,
                    )
                )
            # Clear the tool-block flag and allocate a fresh
            # ``block_idx`` for any subsequent content block. The common
            # "tool then finish" path pays no wire cost (``finish`` closes
            # whatever block is open), but a post-stop text/thinking delta
            # would otherwise reopen on the tool's idx — second-order
            # block-model violation.
            last_block_was_tool = False
            block_idx = engine.next_block_idx()
            continue

    # Close a still-open content block on EVERY non-``finish`` exit. The
    # ``finish``-delta branch above closes its block inline before breaking;
    # the two OTHER exits do not: (1) the ``stop_requested`` break at
    # the top of the loop and (2) a clean iterator exhaustion with NO finish
    # delta (the OpenRouter SSE tail-loss shape — ``data: [DONE]`` / EOF).
    # Without this, the open ``content_block_start`` has no matching
    # ``content_block_stop`` and the chat reducer is left with a dangling block.
    # Idempotent: a no-op when the finish branch already closed the block.
    if open_block_kind is not None:
        for evt in _emit_pending_head(complete=True):
            yield evt
        yield TurnEvent(
            type=EventType.CONTENT_BLOCK_STOP,
            run_id=engine.config.run_id,
            payload={
                "turn_id": engine.turn_id(),
                "block_idx": block_idx,
                "visibility": _content_block_visibility(
                    open_block_kind, tool_interleaved=tool_interleaved
                ).value,
            },
        )
        open_block_kind = None


def _calibrate_token_estimate(engine: QueryEngine, request: LLMRequest, observed: int) -> None:
    """Scale the token heuristic to the size the provider just reported for this request.

    The heuristic sizes everything the tiers decide on — which units are worth
    a summariser call, what a summary may cost, whether a pass gained anything —
    and it runs short of the real tokenizer on JSON-heavy and non-Latin text.
    Measured against the provider's own count of the very prompt it was asked
    to size, the factor makes those decisions in the provider's tokens. The
    reported size covers the whole request, so the raw estimate it is compared
    with does too: the messages as sent (system prompt included) and the tool
    definitions. Moves are damped, and a change too small to matter is not
    written, so the estimate cache is not invalidated on every call.
    """
    rc = engine.config.rc
    if not rc.token_estimate_calibration_enabled or observed <= 0:
        return
    uncalibrated = rc.model_copy(update={"token_estimate_calibration": 1.0})
    raw = estimate_history_tokens(list(request.messages), uncalibrated)
    for tool in request.tools:
        dump = getattr(tool, "model_dump_json", None)
        raw += estimate_tokens(dump() if dump is not None else str(tool), uncalibrated)
    if raw <= 0:
        return
    measured = min(max(observed / raw, 1.0), 4.0)
    current = rc.token_estimate_calibration
    smoothed = round(current + (measured - current) * 0.5, 3)
    if abs(smoothed - current) < 0.02:
        return
    calibrated = rc.model_copy(update={"token_estimate_calibration": smoothed})
    engine.config = replace(engine.config, rc=calibrated)
    engine.context_manager.update_rc(calibrated)


async def _handle_context_window_exceeded(
    engine: QueryEngine,
    exc: LLMContextWindowExceeded,
) -> AsyncIterator[TurnEvent]:
    """Recover from a context-window overflow — force_compaction once and
    re-stream, else terminal.

    Tracked via ``engine._compaction_attempted_for_current_turn``: a second
    overflow within the same message drives terminal FAILED.
    """
    if engine._compaction_attempted_for_current_turn:
        # Already retried — go terminal LLM error.
        # Death-spiral guard via _emit_llm_terminal.
        async for evt in _emit_llm_terminal(engine, exc, kind="llm_context_window_exceeded"):
            yield evt
        return

    engine._compaction_attempted_for_current_turn = True
    from_state = engine.state
    engine.transition_to(LoopState.COMPACTING)
    yield _emit_state_change(engine, from_state, LoopState.COMPACTING, reason="reactive_413")

    from protocore.runtime.context.budgets import derive_budgets

    rc = engine.context_manager._rc
    budgets = derive_budgets(rc)
    tokens_before_value = engine.context_manager.token_estimator.estimate_history(
        engine.history, rc
    )
    yield TurnEvent(
        type=EventType.COMPACTION_STARTED,
        run_id=engine.config.run_id,
        payload={
            "reason": "reactive_413",
            "tokens_before": tokens_before_value,
            "trigger_threshold": budgets.compaction_trigger_tokens,
            "history_messages": len(engine.history),
            "holds_settled": bool(engine.config.rc.run_settled_enabled),
        },
    )

    try:
        attempt = await engine.context_manager.force_compaction(
            history=engine.history,
            compaction_state=engine.compaction_state,
            tenant_id=engine.config.tenant_id,
            model_name=engine.effective_model_name,
            observability=_observability_context(
                engine,
                call_purpose="structured",
                call_category="compaction",
            ),
        )
    except CompactionExhaustedError as inner_exc:
        # Death-spiral guard — set BEFORE the state transition.
        engine.skip_terminal_hooks = engine.config.rc.skip_terminal_hooks_on_llm_error
        compacting_from = engine.state
        engine.transition_to(LoopState.FAILED)
        yield _emit_state_change(
            engine,
            compacting_from,
            LoopState.FAILED,
            reason="reactive_413_compaction_exhausted",
        )
        yield TurnEvent(
            type=EventType.ERROR,
            run_id=engine.config.run_id,
            payload={
                "kind": "llm_context_window_exceeded",
                "message": str(inner_exc),
                "primary_error": str(exc),
            },
        )
        yield TurnEvent(
            type=EventType.MESSAGE_STOP,
            run_id=engine.config.run_id,
            payload={
                "turn_id": engine.turn_id(),
                "stop_reason": StopReason.error.value,
            },
        )
        return

    yield TurnEvent(
        type=EventType.COMPACTION_COMPLETED,
        run_id=engine.config.run_id,
        payload={
            "reason": "reactive_413",
            "tokens_before": attempt.tokens_before,
            "tokens_after": attempt.tokens_after,
            "tier1_freed": attempt.tier1.tokens_freed if attempt.tier1 else 0,
            "tier2_summarised": attempt.tier2.turns_summarised if attempt.tier2 else 0,
            "tier3_folded": attempt.tier3.messages_folded if attempt.tier3 else 0,
            "blob_refs_created": (list(attempt.tier1.blob_refs_created) if attempt.tier1 else []),
        },
    )
    # This path calls force_compaction directly (not via _run_compaction), so
    # clear the stale prompt-size floor here too — the pre-compaction history it
    # described no longer exists, and leaving it set would drive one spurious
    # compaction at the next turn-start before the next LLM call self-heals it.
    engine.last_observed_prompt_tokens = 0
    await engine._persist_snapshot()
    compacting_from = engine.state
    engine.transition_to(LoopState.RUNNING)
    yield _emit_state_change(
        engine,
        compacting_from,
        LoopState.RUNNING,
        reason="reactive_413_compaction_completed",
    )


async def _iter_with_idle_watchdog(
    source: AsyncIterator[ProviderDelta],
    *,
    idle_timeout: float,
    stall_threshold: float,
    reasoning_idle_timeout: float | None = None,
) -> AsyncIterator[ProviderDelta]:
    """Wrap ``source`` with a per-iteration idle/stall watchdog.

 Each ``__anext__`` is awaited with :func:`asyncio.wait_for` using
 ``idle_timeout`` seconds. On timeout the upstream is presumed hung
 and :class:`LLMStreamIdleError` is raised (the caller maps this to
 the terminal LLM error path).

 A best-effort warning is logged when a post-first-delta inter-delta gap
 crosses ``stall_threshold`` seconds (independent of the hard timeout);
 this surfaces as ``DIAG`` telemetry without aborting the run. A stream
 that recovers stays alive but is still observable in logs.

 Reasoning-aware extension: when the upstream has emitted at least one
 :class:`ProviderDeltaKind.thinking` delta and ``reasoning_idle_timeout``
 is set, every subsequent ``__anext__`` waits up to
 ``reasoning_idle_timeout`` seconds instead of ``idle_timeout``. Once a
 non-reasoning delta arrives (visible text, tool call, finish, usage)
 the budget reverts to the baseline. This catches the
 smoke-test-result-r3 stall pattern where Qwen3-235B over OpenRouter
 sat silent for ~90 s during second-turn reasoning before producing
 any visible token; the legacy 90 s hard cap aborted the stream before
 the model could finish thinking. The error message includes which
 threshold was active so operators can distinguish a true hang from a
 long reasoning gap when the timeout DOES eventually fire.
 """
    iterator = source.__aiter__()
    loop = asyncio.get_event_loop()
    last_delta_at: float | None = None
    in_reasoning_window = False

    while True:
        # Per-iteration budget — reasoning-extended only after a thinking
        # delta has been observed AND only until the next non-reasoning
        # delta lands. This keeps the baseline tight for normal
        # non-reasoning streams.
        active_timeout = (
            reasoning_idle_timeout
            if (in_reasoning_window and reasoning_idle_timeout is not None)
            else idle_timeout
        )
        try:
            delta = await asyncio.wait_for(
                iterator.__anext__(), timeout=active_timeout
            )
        except StopAsyncIteration:
            return
        except TimeoutError as exc:
            kind_hint = "reasoning" if in_reasoning_window else "baseline"
            idle_error = LLMStreamIdleError(
                f"LLM stream idle for >{active_timeout:.1f}s "
                f"(window={kind_hint}, last_delta_seen=thinking={in_reasoning_window}) "
                "— terminating"
            )
            # This watchdog is the only thing that raises the idle error, so it
            # is the only place a verdict on it can come from. Without one the
            # loop's chain step reads the failure as uncharacterised and leaves
            # the run on the provider that just went silent — the branch that
            # exists to move it would then be unreachable by construction. A
            # stream that stopped speaking is a statement about the endpoint,
            # exactly like a timeout, so it is classified as one and the next
            # provider gets its turn.
            object.__setattr__(idle_error, "classified", _IdleStreamVerdict())
            raise idle_error from exc

        now = loop.time()
        gap = None if last_delta_at is None else now - last_delta_at
        if gap is not None and gap > stall_threshold:
            _logger.warning(
                "DIAG llm_stream.stall gap_s=%.1f stall_threshold_s=%.1f "
                "active_timeout_s=%.1f window=%s",
                gap,
                stall_threshold,
                active_timeout,
                "reasoning" if in_reasoning_window else "baseline",
            )
        # Update the reasoning-window flag AFTER observing the delta so
        # the NEXT ``wait_for`` honours the new state. A thinking delta
        # opens the window; visible output / tool call / finish close it.
        # Transport-only progress heartbeats are intentionally transparent:
        # they reset the idle clock but must not collapse the longer
        # reasoning-aware window while the provider is still thinking.
        if delta.kind is ProviderDeltaKind.thinking:
            if not in_reasoning_window:
                _logger.warning(
                    "DIAG llm_stream.reasoning_window_open "
                    "idle_timeout_s=%.1f reasoning_idle_timeout_s=%s",
                    idle_timeout,
                    reasoning_idle_timeout,
                )
            in_reasoning_window = True
        elif delta.kind is ProviderDeltaKind.progress:
            pass
        elif in_reasoning_window:
            _logger.warning(
                "DIAG llm_stream.reasoning_window_close delta_kind=%s",
                delta.kind.value,
            )
            in_reasoning_window = False
        last_delta_at = now
        yield delta


async def _emit_llm_terminal(
    engine: QueryEngine,
    exc: BaseException,
    *,
    kind: str,
) -> AsyncIterator[TurnEvent]:
    """Drive the run terminal on an LLM-provider class error.

 Used on provider timeout (PTL post-retry), fallback-exhausted /
 max-output-exhausted, and idle watchdog paths. Emits the
 state_changed → FAILED → error → message_stop sequence.

 **Death-spiral guard** — when the terminal cause is an LLM-provider
 class error (LLMProviderError, LLMStreamIdleError,
 LLMContextWindowExceeded after retry, MaxOutputTokensExhausted)
 Stop / SessionEnd hooks MUST be skipped to prevent broken-provider
 runs from cascading through error-only hooks. The guard is engaged
 by setting ``engine.skip_terminal_hooks = True`` BEFORE the state
 transition so any synchronous downstream consumer (engine.run
 finally-block, the host SessionEnd dispatcher, hook-manager
 invoke) sees a consistent value.

 The guard is opt-out via
 :attr:`LoopConstants.skip_terminal_hooks_on_llm_error` for
 diagnostic deployments where Stop hooks SHOULD see the LLM error.

 The classifier verdict (when present) is surfaced in the ERROR event
 payload as ``classified_reason``. LLM adapters (Anthropic / OpenAI)
 attach a ``ClassifiedError`` to every raised :class:`LLMError` subclass
 via the dynamic ``classified`` attribute (FailoverReason taxonomy).
 Surfacing the reason on the bus lets the host-side
 ``RecoveryDispatcher`` route the failure to the right recovery action
 (compaction, fallback, backoff, terminate) without re-classifying. Core
 does not own the dispatch policy — it only forwards the verdict
 downstream.

 **A crash of this process is not a failure of the upstream.** Every
 caller reaches this function with a ``kind`` describing an LLM-class
 failure, and the loop's catch-all reaches it with whatever escaped the
 stream — a parser bug, a ``RecursionError``, an ``AttributeError``.
 Those used to be recorded as a PROVIDER error, which makes the
 provider-failure metric count our own bugs and points every subsequent
 investigation at the wrong system. So the ``kind`` is decided HERE, from
 the exception's own type, and anything outside the
 :class:`~protocore.contracts.llm.LLMError` family is reported as
 :data:`INTERNAL_ERROR_KIND` no matter what the call site asked for. The
 classification lives at the single point every terminal passes through,
 because a call site that has to remember to classify is a call site that
 will eventually forget.

 The exception is logged WITH ITS TRACEBACK. Without it the whole record
 of a crash is one line naming a kind and a message, and the place the
 run actually died is not recoverable from anything the system kept.
 """
    # A typed LLM failure keeps the caller's ``kind`` (the taxonomy the
    # recovery dispatcher routes on); anything else is this process crashing.
    if not isinstance(exc, LLMError):
        kind = INTERNAL_ERROR_KIND
        _logger.warning(
            "DIAG query.internal_error run=%s tenant=%s turn=%s "
            "exception=%s message=%s",
            engine.config.run_id,
            engine.config.tenant_id,
            engine.turn_id(),
            type(exc).__name__,
            exc,
            exc_info=exc,
        )
    # pair any already-emitted tool_use that never received a
    # result before driving the FAILED terminal. A query error can throw
    # after the assistant tool_use turn was appended to history but before
    # its dispatch ran; the engine.run() finally persists this snapshot and a
    # a resume must not replay a dangling tool_use.
    _synthesize_missing_tool_results(
        engine.history,
        error_content=engine.prompt_text("tool_result_interrupted"),
    )
    from_state = engine.state
    # Engage the death-spiral guard BEFORE the state transition. It exists to
    # keep a BROKEN-PROVIDER run from cascading through error-only Stop /
    # SessionEnd hooks; an internal crash is not that, and suppressing the hooks
    # there would hide the teardown of the one failure class a deployment most
    # wants to observe.
    engine.skip_terminal_hooks = (
        kind != INTERNAL_ERROR_KIND
        and engine.config.rc.skip_terminal_hooks_on_llm_error
    )
    engine.transition_to(LoopState.FAILED)
    yield _emit_state_change(
        engine,
        from_state,
        LoopState.FAILED,
        reason=kind,
    )
    # Surface the classifier verdict downstream.
    error_payload: dict[str, object] = {"kind": kind, "message": str(exc)}
    classified = getattr(exc, "classified", None)
    if classified is not None:
        reason = getattr(classified, "reason", None)
        if reason is not None:
            # ``FailoverReason`` is a ``StrEnum`` — its ``.value`` is the
            # canonical wire string. Tolerate plain-string for forward
            # compatibility (e.g. tests stubbing a classified verdict).
            error_payload["classified_reason"] = getattr(reason, "value", reason)
        retryable = getattr(classified, "retryable", None)
        if retryable is not None:
            error_payload["retryable"] = bool(retryable)
        should_compress = getattr(classified, "should_compress", None)
        if should_compress is not None:
            error_payload["should_compress"] = bool(should_compress)
        should_fallback = getattr(classified, "should_fallback", None)
        if should_fallback is not None:
            error_payload["should_fallback"] = bool(should_fallback)
    yield TurnEvent(
        type=EventType.ERROR,
        run_id=engine.config.run_id,
        payload=error_payload,
    )
    yield TurnEvent(
        type=EventType.MESSAGE_STOP,
        run_id=engine.config.run_id,
        payload={
            "turn_id": engine.turn_id(),
            "stop_reason": StopReason.error.value,
        },
    )


async def _rebuild_context_for_recovery(
    engine: QueryEngine,
) -> ContextBundle:
    """Rebuild :class:`ContextBundle` after a recovery step mutated history.

    Used by the reactive-413 (freshly compacted history) and max-output
    recovery (synthetic resume-prompt user message) paths. The skill catalog
    block is the run's once-built value (NOT rebuilt here) so a recovery
    rebuild cannot bust the cached system-prompt prefix mid-run.
    """
    tool_defs = list(
        engine.tools.compute_effective_surface(
            tenant_id=engine.config.tenant_id,
            policy=engine.effective_tool_policy,
            query=engine.latest_user_message.text if engine.latest_user_message else "",
            top_k=engine.config.rc.tool_retrieval_top_k,
        )
    )
    recovery_history, _ = _llm_history(engine)
    return engine.context_manager.build_context(
        history=recovery_history,
        tools=tool_defs,
        system_prompt_sections=engine.config.system_prompt_sections,
        skill_index_block=await _ensure_run_skill_catalog(engine),
        skills_loaded=engine._skill_loaded_bundles,
    )


def _is_parallel_safe_tool(
    engine: QueryEngine,
    tool_call: ToolCall,
    hook_match_predicate: Callable[[str], bool] | None = None,
) -> bool:
    """Return ``True`` iff ``tool_call`` can run concurrently with siblings.

 Eligibility predicate for the parallel-dispatch branch in
 :func:`_stream_one_assistant_message`. A tool is parallel-safe only
 when ALL of these conditions hold:

 * The registry has a tool registered under ``tool_call.name`` (so we
 can read its static metadata; missing tools must fall through to
 the serial dispatcher which produces the canonical "unknown tool"
 error envelope).
 * ``tool.is_concurrent_safe is True`` — adapter explicitly opts in.
 * ``tool.is_destructive is False`` — destructive ops (writes,
 deletes, ``pcm_answer``) MUST serialise so the LLM sees a stable
 causal order between read and mutate.
 * No enabled ``PreToolUse`` hook for this tenant could match the
 tool name. Hooks can mark ANY
 tool ``requires_approval`` via
 :meth:`ToolPermissionGate._project_hook_result`; if such a tool
 ran under :func:`asyncio.gather` the serial path's web-mode
 approval downgrade (``query.py:_dispatch_tool``) would be
 bypassed AND a multi-call batch could leak multiple pending
 approvals from one turn. Steering any hook-matchable tool back
 onto the serial path keeps both invariants intact.

 The combination guarantees no approval-gate surface either: the
 safety policy only requires approval on destructive/sandbox tools
 (which the destructive predicate already excludes), so a tool that
 is concurrent-safe, non-destructive, AND outside every PreToolUse
 hook matcher cannot yield :class:`EventType.TOOL_CALL_PENDING`. The
 parallel orchestrator relies on this invariant — the in-batch
 approval handler is a defensive fallback for the narrow race where
 a hook is registered mid-turn between the predicate snapshot and
 the dispatch.

 Notes:

 * The predicate uses ``getattr(..., default)`` so it tolerates
 registries that pre-date the parallel-dispatch contract (tests +
 legacy adapters that never set the ClassVar default to ``False``,
 which is the conservative serial behaviour).
 * ``hook_match_predicate=None`` is the legacy single-argument contract
 used by the existing unit tests: it skips the hook check and assumes no
 hook could match. Production callers in
 :func:`_stream_one_assistant_message` ALWAYS pass the predicate.
 """
    tool = engine.tools.get(tool_call.name)
    if tool is None:
        return False
    if not bool(getattr(tool, "is_concurrent_safe", False)):
        return False
    if bool(getattr(tool, "is_destructive", False)):
        return False
    if hook_match_predicate is not None and hook_match_predicate(tool_call.name):
        return False
    return True


def _is_delegation_parallel_safe(
    engine: QueryEngine,
    tool_call: ToolCall,
    hook_match_predicate: Callable[[str], bool] | None = None,
) -> bool:
    """Return ``True`` iff ``tool_call`` is a fan-out-eligible delegation call.

    Distinct from :func:`_is_parallel_safe_tool` (which governs concurrent-safe
    READ tools). A delegation call — the subagent-dispatch tool — is deliberately
    NOT ``is_concurrent_safe`` (each child spawns a full nested run), so it is
    excluded from the read fan-out and keeps its serial-path safety wiring. This
    predicate instead identifies the delegation tool GENERICALLY, through the
    delegation contract (:func:`_tool_is_delegation`), so core hardcodes no tool
    name, and permits fanning several ADJACENT
    delegation calls emitted in one assistant turn out under a bounded semaphore
    (see the delegation branch in :func:`_stream_one_assistant_message`).

    ``True`` only when ALL hold:

    * ``parallel_subagents_enabled`` — master gate (default on).
    * The registry has a tool under ``tool_call.name`` that satisfies the
      delegation contract, or that the host declared in the delegating role.
    * No enabled ``PreToolUse`` hook for this tenant could match the tool name
      (same predicate the read fan-out uses; a hook-gated delegation call MUST
      stay serial so its approval surface is honoured).

    The effective concurrency cap (``max_concurrent_subagents``) is applied by
    the caller: a cap resolving to ``< 2`` disables the fan-out so a single
    delegation call (or a cap of 1) runs on the exact serial path. A registry
    whose tools declare nothing conservatively stays serial.
    """
    if not engine.config.rc.parallel_subagents_enabled:
        return False
    if not _tool_is_delegation(engine, tool_call):
        return False
    if hook_match_predicate is not None and hook_match_predicate(tool_call.name):
        return False
    return True


def _delegation_tool(
    engine: QueryEngine, tool_call: ToolCall
) -> IDelegationTool | None:
    """The registered tool behind ``tool_call``, if it can answer the contract.

    Recognition is by CONTRACT (:class:`IDelegationTool`), not by a boolean
    attribute. A flag says a class was marked; it does not say the object can
    answer the two questions the loop has to ask before it dispatches — how many
    child runs this call starts, and whether the caller waits for them. Reading
    a flag meant anything carrying an attribute of that name was treated as
    spawning child runs, and that the real obligations were written down
    nowhere, so a host could set the flag and satisfy none of them.

    ``None`` for a tool the host declared in the delegating ROLE without
    implementing the contract: it still delegates
    (:func:`_tool_is_delegation`), and the two helpers below give it the
    conservative defaults — one child run, and a caller that waits.
    """
    tool = engine.tools.get(tool_call.name)
    if isinstance(tool, IDelegationTool):
        return tool
    return None


def _tool_is_delegation(engine: QueryEngine, tool_call: ToolCall) -> bool:
    """Return ``True`` iff ``tool_call`` targets a delegation (subagent) tool.

    The RAW structural check — the delegation contract on the registered tool,
    or the host's declaration of that name in the delegating role — WITHOUT the
    ``parallel_subagents_enabled`` gate or the hook-steering check that
    :func:`_is_delegation_parallel_safe` layers on top. Those gates decide
    whether adjacent delegation calls FAN OUT concurrently; they do NOT change
    the fact that a foreground delegation call blocks its caller on a full
    nested child run. A run holding a tree-budget slot must therefore release it
    around ANY such join — the concurrent gather AND a single hook-gated or
    gate-disabled serial dispatch — or it pins a slot while blocked on a
    descendant and the tree can wedge at the cap.
    """
    tool = engine.tools.get(tool_call.name)
    if tool is None:
        return False
    return isinstance(tool, IDelegationTool) or engine.config.tool_roles.has_role(
        tool_call.name, ToolRole.delegates_work
    )


def _delegation_child_run_count(engine: QueryEngine, tool_call: ToolCall) -> int:
    """How many child runs ONE delegation call starts. ``1`` unless it says.

    One delegation call is not one child run. A delegation tool whose arguments
    carry a LIST of tasks starts one full child run per element, and a tree
    charged once for that call is bounded by the number of CALLS rather than the
    number of runs — the cap it advertises off by the batch width.

    How the arguments map to runs lives in the tool's own schema, which core
    deliberately knows nothing about, so the tool is ASKED rather than parsed —
    ``child_run_count(arguments)``, one half of :class:`IDelegationTool`. A tool
    known to delegate only by its declared role, or a counter that answers with
    nonsense, reads as one.
    """
    tool = _delegation_tool(engine, tool_call)
    if tool is None:
        return 1
    try:
        return max(1, int(tool.child_run_count(tool_call.arguments)))
    except (TypeError, ValueError):
        return 1


def _delegation_is_background(engine: QueryEngine, tool_call: ToolCall) -> bool:
    """Whether this delegation call returns without waiting for its children.

    The other half of :class:`IDelegationTool`, and the question that decides
    whether the parent's turn and its slot in the tree budget are held for the
    whole descendant run. A tool that cannot answer is treated as waiting, which
    is the conservative reading: a caller assumed to be waiting keeps the bounds
    it always had, whereas one wrongly assumed to have returned would release a
    slot it still occupies and a turn it is still inside.
    """
    tool = _delegation_tool(engine, tool_call)
    if tool is None:
        return False
    try:
        return bool(tool.is_background_call(tool_call.arguments))
    except (TypeError, ValueError):
        return False


# Operator subset needed by :func:`_hook_matchers_could_match_tool` —
# core does not import the host matcher module per the
# core-vs-the host import boundary (see
# ``protocore/tests/test_core_import_boundary.py``).
def _hook_matchers_could_match_tool(
    matchers: Mapping[str, Any],
    tool_name: str,
) -> bool:
    """Return True iff ``matchers`` could match a payload with this tool name.

    Mirrors the documented matcher subset a host's hook matcher applies to
    the ``tool_name`` field only. We are intentionally CONSERVATIVE — when the
    matcher
    references a field other than ``tool_name`` (e.g. ``tool_input.path``)
    we cannot know without invoking the dispatcher whether the payload
    would match, so we return ``True`` (treat as "could match"). This
    keeps the parallel-safe set strictly smaller in ambiguity, which is
    the right side to err on for the approval-gate invariant.

    Empty matchers ⇒ matches everything ⇒ returns True.

    Supported operators on ``tool_name``:
    ``$eq``, ``$ne``, ``$in``, ``$nin``, ``$exists``, ``$regex`` plus
    the scalar shorthand for ``$eq``. Unknown operators return True
    (conservative).
    """
    if not matchers:
        return True
    for key, spec in matchers.items():
        if key != "tool_name":
            # Matcher references a non-tool_name payload field — we
            # cannot evaluate it statically without running the full
            # dispatcher. Conservatively assume it could match.
            return True
        if isinstance(spec, dict):
            for op, expected in spec.items():
                if op == "$eq":
                    if tool_name != expected:
                        return False
                elif op == "$ne":
                    if tool_name == expected:
                        return False
                elif op == "$in":
                    if not isinstance(expected, list) or tool_name not in expected:
                        return False
                elif op == "$nin":
                    if isinstance(expected, list) and tool_name in expected:
                        return False
                elif op == "$exists":
                    # tool_name always exists in PreToolUse payloads.
                    if not bool(expected):
                        return False
                elif op == "$regex":
                    if not isinstance(expected, str):
                        return False
                    try:
                        pattern = re.compile(f"^(?:{expected})$")
                    except re.error:
                        return False
                    if not pattern.match(tool_name):
                        return False
                else:
                    # Unknown operator — conservative: could match.
                    return True
        else:
            # Scalar shorthand: implicit $eq.
            if tool_name != spec:
                return False
    return True


async def _pre_tool_use_match_predicate(
    engine: QueryEngine,
) -> Callable[[str], bool] | None:
    """Build a per-turn predicate: "does any PreToolUse hook match tool_name N?".

 Called ONCE per turn from :func:`_stream_one_assistant_message` BEFORE
 the dispatch batching loop walks the pending tool calls. Returns:

 * ``None`` when the engine has no hook manager (legacy test wiring
 or :class:`InMemoryHookManager` with no specs registered) — the
 caller falls back to the simple destructive-only predicate.
 * A predicate ``f(tool_name) -> bool`` that returns ``True`` when
 some enabled ``PreToolUse`` hook MIGHT match. Mid-turn hook
 registration is the only race window; the predicate's snapshot
 is intentionally taken once per turn so the dispatch-loop
 assumptions stay stable for that turn.

 Hook list failures (PG outage, ``IHookManager.list`` raising) are
 isolated like the rest of the hook subsystem: the predicate
 conservatively returns ``True`` for every tool, which restores serial
 dispatch behaviour for the turn. Better to lose parallelism than to
 break the approval contract.
 """
    hook_manager = getattr(engine, "hooks", None)
    if hook_manager is None:
        return None
    try:
        hooks = list(
            await hook_manager.list(
                engine.config.tenant_id, event=HookEvent.pre_tool_use
            )
        )
    except Exception:
        _logger.warning(
            "DIAG query.pre_tool_use_predicate.list_failed tenant=%s — "
            "falling back to serial dispatch for the turn",
            engine.config.tenant_id,
            exc_info=True,
        )
        # Conservative fallback: assume every tool could be hook-gated.
        return lambda _tool_name: True
    enabled_matchers: list[Mapping[str, Any]] = [
        getattr(h, "matchers", {}) or {} for h in hooks if getattr(h, "enabled", True)
    ]
    if not enabled_matchers:
        # No hooks registered → no tool can be hook-gated → predicate
        # returns False for every tool (parallelisation freely allowed
        # subject to the other predicate clauses).
        return lambda _tool_name: False

    def _predicate(tool_name: str) -> bool:
        for matchers in enabled_matchers:
            if _hook_matchers_could_match_tool(matchers, tool_name):
                return True
        return False

    return _predicate


async def _record_batch_tool_intents(
    engine: QueryEngine,
    tool_calls: Sequence[ToolCall],
) -> None:
    """Make a whole parallel batch durable in one snapshot, before it runs.

    The serial path writes its record from inside the dispatch, at the last
    moment before the tool is touched. A batch cannot: each call runs in its
    own coroutine under a gather, and an await placed before the invoke
    decides which sibling reaches its tool first. So the batch is recorded
    together, here, one snapshot for all of it, before any of it is dispatched.

    Recorded, not marked in flight. Between this snapshot and the tool there
    is still a permission gate, a hook and a precondition check, any of which
    can refuse the call outright. A record that already said "dispatched"
    would, after a crash in that window, tell the model the outcome of a call
    the gate refused is unknown and its effects may be in place. Each call
    marks its own record as it passes the last of those checks, in memory,
    where no await can reorder the siblings.
    """
    recorded = False
    for tool_call in tool_calls:
        if find_intent(engine.open_intents, tool_call.id) is not None:
            continue
        intent = commit_intent(
            tool_name=tool_call.name,
            tool_call_id=tool_call.id,
            rc=engine.config.rc,
            arguments=tool_call.arguments,
            state=RESERVED,
            roles=engine.config.tool_roles,
        )
        if intent.repeat_is_safe:
            continue
        engine.open_intents.append(intent)
        recorded = True
    if recorded:
        await engine._persist_snapshot()


async def _drain_dispatch_tool_deferred(
    engine: QueryEngine,
    tool_call: ToolCall,
    *,
    dispatch_order: int | None = None,
    dispatch_group: str | None = None,
    tree_permit: SubagentTreePermit | None = None,
) -> tuple[list[TurnEvent], DispatchOutcome | None]:
    """Run :func:`_dispatch_tool` but DEFER history append + persist.

 Used by the parallel-dispatch branch. The standard
 :func:`_dispatch_tool` appends a :class:`ToolResultBlock` to
 ``engine.history`` and persists a snapshot as its final side effect.
 Under :func:`asyncio.gather` the order of those appends is
 non-deterministic, which would break the LLM-facing invariant that
 tool results appear in the same order the model requested them.

 This helper drains the dispatcher into a buffer (returning the
 events list + the dispatcher's final :class:`DispatchOutcome`)
 WITHOUT touching ``engine.history``. The caller then iterates the
 parallel batch in the original ``ToolCall`` order and appends the
 result blocks sequentially via :func:`_apply_deferred_tool_history`.
 """
    # Mirror the serial dispatcher's terminal-only guard for the parallel
    # batch path. Build a synthetic deferred outcome (no dispatcher
    # invocation) so the caller's ``_apply_deferred_tool_history`` append
    # still emits the structured error blocked envelope.
    if _terminal_only_blocks(engine, tool_call):
        error_message = _terminal_only_error_message(engine, tool_call.name)
        synthetic_event = TurnEvent(
            type=EventType.TOOL_RESULT,
            run_id=engine.config.run_id,
            payload={
                "tool_call_id": tool_call.id,
                "success": False,
                "is_error": True,
                "error": {
                    "kind": "terminal_only",
                    "message": error_message,
                },
                "content_blocks": [{"type": "text", "text": error_message}],
            },
        )
        synthetic_outcome = DispatchOutcome(
            tool_call=tool_call,
            success=False,
            content=error_message,
            is_error=True,
            error_kind=DispatchErrorKind.execution,
            metadata={},
        )
        return [synthetic_event], synthetic_outcome
    # Cumulative total-work guard, same synthetic shape as the terminal-only
    # veto above and for the same reason: no dispatcher invocation, so a tree
    # that has spent its delegation budget pays nothing more to be told so.
    #
    # ONE act, not a gate followed by a charge. Reserving is itself the
    # question — it grants all the child runs this call starts or none of
    # them, and names the budget that refused — so asking first and charging
    # second would be the same question twice with nothing able to run between
    # the two: there is no await here, so no sibling can spend the budget in
    # the gap. A second, separately-worded refusal for that impossible gap is
    # what this branch used to carry, and it was never reached.
    grant = _charge_child_run_start(engine, tool_call)
    if grant is not None and not grant.fully_granted:
        _logger.warning(
            "DIAG query.run_work_budget.delegation_refused run=%s tenant=%s "
            "tool=%s reason=%s %s",
            engine.config.run_id,
            engine.config.tenant_id,
            tool_call.name,
            grant.reason,
            _resolve_run_work_ledger(engine).spent_summary(),
        )
        return _run_work_refusal_dispatch(
            engine,
            tool_call,
            _run_work_refusal_text(engine, tool_call, grant.reason),
            grant.reason,
        )
    # Seed the run's satisfied set from the durable ``engine.history`` when the
    # state is fresh (cross-process re-drive). This MUST run before the
    # dispatcher reads that set on the parallel branch.
    _rehydrate_satisfied_from_history(engine)
    metadata: dict[str, Any] = {}
    # Runtime-internal names are skipped, so a forged operator ``run_metadata``
    # cannot shadow ``tool_call_id`` / ``protocore.*`` on the parallel-dispatch
    # path either.
    _merge_run_metadata_into(metadata, engine.run_state)
    # Carry the child's LLM-requested batch position + its fan-out group id so
    # the host runner can declare its deliverables into the parent ledger in
    # batch order (not gather completion order), scoped per group so a later
    # group's declaration is never frozen out by an earlier one. Set AFTER the
    # run-metadata merge so a forged ``run_metadata`` cannot shadow them; absent
    # (serial/single dispatch) leaves declaration order at the pre-existing
    # last-writer-wins behaviour.
    if dispatch_order is not None:
        metadata[SUBAGENT_DISPATCH_ORDER_METADATA_KEY] = dispatch_order
    if dispatch_group is not None:
        metadata[SUBAGENT_DISPATCH_GROUP_METADATA_KEY] = dispatch_group
    # Carry THIS child's tree-budget permit handle (an in-memory object, not
    # serialized — like the cancel Event on the run state) so the host
    # runner can put it on the child's run state and the child engine can
    # release-while-awaiting around its own nested delegation gather. Set AFTER
    # the run-metadata merge so a forged ``run_metadata`` cannot shadow it; absent
    # (serial/single dispatch) means the child holds no tree slot.
    if tree_permit is not None:
        metadata[SUBAGENT_TREE_PERMIT_METADATA_KEY] = tree_permit
    ctx = ToolContext(
        tenant_id=engine.config.tenant_id,
        run_id=engine.config.run_id,
        session_id=engine.config.session_id,
        work_scope=engine.config.work_session_id,
        evidence=ToolEvidenceContext(
            origin=engine._engine_evidence_origin(), admission_deferred=True
        ),
        run_state=engine.run_state,
        metadata=metadata,
    )

    dispatcher = _ensure_tool_dispatcher(engine)
    events: list[TurnEvent] = []
    outcome: DispatchOutcome | None = None
    # Same durable record as the serial path: a call in a parallel batch is
    # every bit as capable of dying in flight, and a delegation repeated after
    # a restart starts a whole second subtree.
    intent = find_intent(engine.open_intents, tool_call.id)
    if intent is None:
        intent = commit_intent(
            tool_name=tool_call.name,
            tool_call_id=tool_call.id,
            rc=engine.config.rc,
            arguments=tool_call.arguments,
            state=RESERVED,
            roles=engine.config.tool_roles,
        )
        engine.open_intents.append(intent)

    async def _mark_in_flight(_call: ToolCall) -> None:
        """Move this call's record past every gate that could still stop it.

        In memory only. The batch was made durable before the gather; writing
        a snapshot here would place an await ahead of the invoke and decide
        which sibling reaches its tool first, which is the ordering the batch
        write exists to avoid.
        """
        mark_dispatched(intent)

    async for item in dispatcher.dispatch(
        tool_call=tool_call,
        ctx=ctx,
        # The effective policy carries the RC core tool-surface floor so the
        # gate permits exactly what was advertised (advertise/dispatch
        # parity; see ToolPermissionGate.check Stage-1 whitelist). Without
        # this, the parallel branch would compute ``allowed=visible|pinned|
        # forced_pinned`` from the raw policy and deny a tool that the
        # per-turn surface already advertised via the floor.
        visibility_policy=engine.effective_tool_policy,
        # The declared tool set of the agent driving THIS engine, when it
        # declared one. Empty declaration ⇒ ``None`` ⇒ the gate's allow-list
        # stage stays off, exactly as before it was wired.
        subagent_whitelist=engine.effective_subagent_tool_allowlist,
        child_run=engine.config.parent_run_id is not None,
        timeout_seconds=engine.config.rc.tool_timeout_seconds,
        preapproved_tool_call_id=None,
        admit_evidence=lambda records, producer: engine.append_tool_evidence(
            records, producer=producer
        ),
        # No per-call snapshot here, unlike the serial path: this coroutine is
        # one of several under a gather, and an await placed before the tool
        # is invoked reorders which sibling starts first. The batch is made
        # durable once, before the gather, by the caller; this callback only
        # moves the already-durable record out of its reserved state.
        on_dispatch_start=_mark_in_flight,
        lifecycle=_lifecycle_registry(engine),
    ):
        if isinstance(item, DispatchOutcome):
            outcome = item
            break
        events.append(item)
    if outcome is not None and not outcome.approval_required:
        settle_intent(intent, result=str(outcome.content or "")[:200])
        _forget_intent(engine, intent)
    return events, outcome


def _synthesize_delegation_error_result(
    engine: QueryEngine,
    tool_call: ToolCall,
    exc: BaseException,
) -> tuple[list[TurnEvent], DispatchOutcome]:
    """Build an error ``(events, outcome)`` pair for a delegation child that RAISED.

    The concurrent delegation branch gathers child dispatches with
    ``return_exceptions=True`` so ONE child raising an unexpected exception does
    not cancel its siblings. (The dispatcher normally converts subagent failures
    — unknown ``subagent_type``, hook block, timeout — into structured
    ``success=False`` outcomes, so a raised exception is the defensive edge.)
    This helper converts the exception into the same shape a normal dispatch
    yields — a ``TOOL_RESULT`` event plus a ``success=False`` ``DispatchOutcome``
    — so the raising child still contributes its OWN error ``ToolResultBlock`` in
    LLM-requested order and the leader loop continues with the successful
    siblings, matching the single-child error contract.
    """
    detail = str(exc).strip()
    message = f"subagent dispatch failed: {type(exc).__name__}"
    if detail:
        message = f"{message}: {detail[:500]}"
    event = TurnEvent(
        type=EventType.TOOL_RESULT,
        run_id=engine.config.run_id,
        payload={
            "tool_call_id": tool_call.id,
            "success": False,
            "is_error": True,
            "error": {"kind": "execution", "message": message},
            "content_blocks": [{"type": "text", "text": message}],
        },
    )
    outcome = DispatchOutcome(
        tool_call=tool_call,
        success=False,
        content=message,
        is_error=True,
        error_kind=DispatchErrorKind.execution,
        metadata={},
    )
    return [event], outcome


def _resolve_subagent_tree_budget(engine: QueryEngine) -> SubagentTreeBudget:
    """Resolve the shared tree-wide subagent budget, minting it at first fan-out.

    The budget is ONE object per maximal parallel-dispatched subtree, shared by
    reference. When the run's state already carries it (a run whose ancestor
    already minted it — a child state inherits the SAME object the way
    ``cancel_event`` / ``root_run_id`` do), return that. Otherwise mint
    from ``rc.max_concurrent_subagents_per_tree`` and store it back on the state
    BEFORE any child is dispatched, so the first child state composed from it
    carries the budget downward. The minting run is simply the first
    to reach a parallel fan-out with no budget on its state — usually the root,
    but deeper if the root only ever delegates serially; either way, any two
    concurrently-executing runs share one budget (they branched at a common
    fan-out ancestor that minted it before dispatching them). When the engine has
    no run state (unit tests / degenerate callers) fall back to a local budget
    that still bounds THIS group but cannot propagate to descendants.
    """
    return engine.run_state.ensure_subagent_tree_budget(
        engine.config.rc.max_concurrent_subagents_per_tree
    )


def _resolve_run_work_ledger(engine: QueryEngine) -> RunWorkLedger:
    """Resolve the tree's CUMULATIVE work ledger, minting one if the state has none.

    The live path finds one already there: the host composition root mints
    it for the root run when it builds the run's state, and every descendant
    inherits that same object by reference. Minting here is the fallback for a
    caller that never composed one — it still bounds the caller's own subtree,
    which is the most a run with nothing shared can be held to.

    Unlike :func:`_resolve_subagent_tree_budget` this must NOT be minted lazily
    at the first parallel fan-out. A leader that emits one delegation call per
    turn never fans out, and that serial wave-after-wave shape is precisely the
    one an instantaneous concurrency cap cannot see.
    """
    return engine.run_state.ensure_run_work_ledger(engine.config.rc)


def _charge_run_work_tokens(
    engine: QueryEngine, *, input_tokens: int, output_tokens: int
) -> None:
    """Fold one LLM call's usage into the tree's cumulative token total.

    Called from every engine in the tree against the SHARED ledger. Deliberately
    resolves (and therefore mints) rather than reading best-effort: the very
    first LLM call of the root run happens BEFORE any delegation, and a ledger
    that only came into existence at the first dispatch would start the tree's
    token count part-way through.
    """
    _resolve_run_work_ledger(engine).charge_tokens(
        input_tokens=input_tokens, output_tokens=output_tokens
    )


def _run_work_delegation_refusal(
    engine: QueryEngine, tool_call: ToolCall
) -> tuple[str, str]:
    """``(message, reason)`` for a delegation the tree cannot afford, or ``("","")``.

    Empty for every non-delegation tool (the budget bounds delegation, not the
    leader's own work — a leader must always be able to finish its answer) and
    for a tree with budget left. The reason token comes back alongside the text
    so the caller can stamp it on the outcome without asking the ledger twice.

    The text is written for the model that is about to read it, and it has one
    job: make a RETRY look pointless. A leader that reads a refusal as transient
    spends its remaining turns re-issuing the same call and answers with nothing,
    which is a worse outcome than the unbounded delegation this bound replaces.
    So it states that the budget is cumulative, that it does not refill, that no
    further delegation in this run will be accepted, and what to do instead.
    """
    if not _tool_is_delegation(engine, tool_call):
        return "", ""
    requested = _delegation_child_run_count(engine, tool_call)
    reason = _resolve_run_work_ledger(engine).delegation_refusal_reason(requested)
    if not reason:
        return "", ""
    return _run_work_refusal_text(engine, tool_call, reason), reason


def _run_work_refusal_text(
    engine: QueryEngine, tool_call: ToolCall, reason: str
) -> str:
    """The message a leader reads when the tree cannot pay for ``tool_call``.

    Split out of :func:`_run_work_delegation_refusal` because the refusal is
    reached two ways — the gate ASKS the ledger, the charge is TOLD by it — and
    the advice must not depend on which one spoke. The advice differs by reason,
    not by site: exhaustion means stop delegating, a short budget means ask for
    fewer, and a caller that wrote its own text got one of those wrong.
    """
    ledger = _resolve_run_work_ledger(engine)
    requested = _delegation_child_run_count(engine, tool_call)
    if reason == SUBAGENT_RUN_BUDGET_SHORT:
        # Not exhaustion, and the opposite advice: the tree can still pay for
        # some of this, just not all of it at once. Saying "give up" here would
        # throw away work the budget covers, so the text names the number that
        # would be admitted and asks for a smaller call instead.
        return (
            f"Delegation budget too small for this call ({reason}: "
            f"{ledger.spent_summary()}). This call asked to start {requested} "
            f"child runs and the tree can still afford "
            f"{ledger.remaining_child_runs}. The budget is cumulative over the "
            "whole run and does NOT refill. Re-issue this call with at most "
            f"{ledger.remaining_child_runs} of the most important tasks, or "
            "finalize from what you already have."
        )
    return (
        f"Delegation budget exhausted ({reason}: {ledger.spent_summary()}). "
        "This budget is cumulative over the whole run and does NOT refill — no "
        "further delegation will be accepted, now or on any later turn, and "
        "retrying this call will fail identically. Any subagents already running "
        "will still return. Finalize your answer now from the results you "
        "already have; if something is missing, say what is missing rather than "
        "delegating again."
    )


def _run_work_refusal_dispatch(
    engine: QueryEngine, tool_call: ToolCall, message: str, reason: str
) -> tuple[list[TurnEvent], DispatchOutcome]:
    """Build the synthetic ``(events, outcome)`` pair for a refused delegation.

    Refused BEFORE the dispatcher runs, so an exhausted tree spends nothing —
    not a tool invocation, not a child run — to discover it is exhausted. Shaped
    exactly like the terminal-only veto next door: a ``TOOL_RESULT`` event plus
    an ``is_error`` outcome, so the leader sees an ordinary failed tool result in
    LLM-requested order and its loop continues normally.
    """
    event = TurnEvent(
        type=EventType.TOOL_RESULT,
        run_id=engine.config.run_id,
        payload={
            "tool_call_id": tool_call.id,
            "success": False,
            "is_error": True,
            "error": {"kind": "execution", "message": message},
            "content_blocks": [{"type": "text", "text": message}],
        },
    )
    outcome = DispatchOutcome(
        tool_call=tool_call,
        success=False,
        content=message,
        is_error=True,
        error_kind=DispatchErrorKind.execution,
        metadata={
            DISPATCH_STRUCTURED_ERROR_METADATA_KEY: {
                STRUCTURED_ERROR_FINALIZATION_RECOMMENDED_KEY: True,
                STRUCTURED_ERROR_REASON_KEY: reason,
            }
        },
    )
    return [event], outcome


def _charge_child_run_start(
    engine: QueryEngine, tool_call: ToolCall
) -> ChildRunGrant | None:
    """Charge the tree's cumulative ledger for the child runs about to start.

    Called at the places a child run actually begins — the concurrent delegation
    fan-out and the serial dispatch, the latter only once every seam that can
    still refuse the call has let it past — so a call that was refused, denied
    or parked for approval is never charged.

    The amount is what the TOOL says the call starts
    (:func:`_delegation_child_run_count`), not one per call: a batch that starts
    thirty-two child runs costs the tree thirty-two, or the cap bounds calls
    rather than runs and is off by the batch width.

    The charge deliberately does NOT live inside :func:`_tool_is_delegation`.
    That is a pure predicate, consulted several times per dispatch (fan-out
    classification, permit resolution) and on every tool call rather than every
    child run; charging there would debit the tree several times for one child
    — the same over-counting this bound exists to prevent, only from the inside.

    The grant comes back so the caller can act on a refusal: the ledger is the
    authority on whether the call is affordable, and a caller that ignored a
    zero grant would start work the tree did not pay for. Charging is keyed on
    the call id, so one call reaching here twice — dispatched, parked for
    approval, dispatched again — is charged once.
    """
    if not _tool_is_delegation(engine, tool_call):
        return None
    return _resolve_run_work_ledger(engine).reserve_child_runs(
        _delegation_child_run_count(engine, tool_call), call_id=tool_call.id
    )


def _resolve_subagent_tree_permit(engine: QueryEngine) -> SubagentTreePermit | None:
    """Return THIS run's own tree permit from the run's state, or None.

    Present only for a run that was itself dispatched under the budget (the host
    runner puts the child's permit on its fresh run state). The root
    leader — which was never dispatched as a subagent — holds none, so it returns
    None and neither releases nor reacquires a tree slot while awaiting children.
    """
    return engine.run_state.subagent_tree_permit


async def _dispatch_subagent_under_semaphore(
    engine: QueryEngine,
    tool_call: ToolCall,
    semaphore: asyncio.Semaphore,
    budget: SubagentTreeBudget,
    *,
    dispatch_order: int | None = None,
    dispatch_group: str | None = None,
) -> tuple[list[TurnEvent], DispatchOutcome | None]:
    """Drain one deferred delegation dispatch while holding ``semaphore``.

    The concurrent delegation branch gathers these so at most
    ``max_concurrent_subagents`` children execute at once; the rest wait on the
    semaphore and run in waves. History append stays deferred (see
    :func:`_drain_dispatch_tool_deferred`) so the caller can append results in
    LLM-requested order regardless of completion order. ``dispatch_order`` is the
    child's 0-based position in the LLM-requested batch and ``dispatch_group`` the
    fan-out group's stable id, forwarded so parent-ledger declarations resolve in
    batch order (scoped per group) rather than completion order.

    Beyond the local per-group ``semaphore`` (which bounds this group's width),
    each child also draws ONE slot from the tree-wide ``budget`` (which bounds the
    additive sum across all nested groups). The tree slot is acquired here, on the
    child's behalf, only AFTER the local semaphore is held — so a wave waiting for
    group width never sits on a scarce tree slot. The child's permit handle is
    threaded onto its dispatch metadata so the child engine can
    release-while-awaiting around its OWN nested gather, and the parent releases
    the slot once in ``finally`` after the child run returns. Acquisition order
    (local width THEN tree slot) is uniform across every dispatch site, so the two
    bounds cannot themselves deadlock against each other.
    """
    async with semaphore:
        permit = await budget.acquire()
        # The tree budget is an in-process object, so how many slots a fan-out
        # actually charged is otherwise readable only from inside the process
        # that holds it — and a completed run never writes it anywhere, because
        # only a run that PAUSES snapshots the count. That left the bound with
        # no evidence but wall-clock timing, which is not evidence. One line per
        # charged slot, naming the group and the capacity it was charged
        # against, so a fan-out of N children is N lines under one group id.
        _logger.warning(
            "DIAG query.subagent_tree_budget.charged run=%s tenant=%s group=%s "
            "order=%s capacity=%d charged=%d unlimited=%s",
            engine.config.run_id,
            engine.config.tenant_id,
            dispatch_group,
            dispatch_order,
            budget.capacity,
            budget.in_use,
            budget.unlimited,
        )
        try:
            return await _drain_dispatch_tool_deferred(
                engine,
                tool_call,
                dispatch_order=dispatch_order,
                dispatch_group=dispatch_group,
                tree_permit=permit,
            )
        finally:
            await permit.release()


# — the finalize-hint ``reason`` is provider-VISIBLE (it is appended to
# the tool-result content the model sees), so an UNTRUSTED future producer must
# never leak internal/tenant data or a huge string into the prompt through it.
# A reason token is echoed ONLY when it is a short, machine-token-shaped string;
# anything else is dropped from the model-visible line (the full reason still
# lives verbatim in ``outcome.metadata`` / logs). The signal
# ``transport_retry_budget_exhausted`` is a clean snake_case token and passes.
_FINALIZATION_REASON_MAX_LEN: Final[int] = 64
_FINALIZATION_REASON_TOKEN_RE: Final[re.Pattern[str]] = re.compile(
    r"\A[A-Za-z0-9][A-Za-z0-9 _.-]*\Z"
)


def _safe_finalization_reason_suffix(reason: object) -> str:
    """Return a bounded ``(reason: …)`` suffix, or ``""`` when not safe to echo.

 sanitiser: only a string that is non-empty, within
 :data:`_FINALIZATION_REASON_MAX_LEN`, and matches the machine-token allowlist
 :data:`_FINALIZATION_REASON_TOKEN_RE` (alnum + ``_ - . space``, no control
 chars / newlines / structural punctuation) is echoed into the provider-
 visible hint. Any other value (non-string, over-length, arbitrary prose,
 injected markup) yields ``""`` so it is dropped from the model-visible line.
 The unsanitised reason is still available in ``outcome.metadata`` and logs.
 """
    if not isinstance(reason, str):
        return ""
    if not reason or len(reason) > _FINALIZATION_REASON_MAX_LEN:
        return ""
    if not _FINALIZATION_REASON_TOKEN_RE.fullmatch(reason):
        return ""
    return f" (reason: {reason})"


def _tool_result_content_with_finalization_hint(outcome: DispatchOutcome) -> str:
    """Return the tool-result content, with a finalize hint when the tool gave up.

    A tool that exhausts its transport-retry budget raises with
    ``structured_error={"finalization_recommended": True, ...}``. The dispatch
    except-branch forwards that under
    :data:`DISPATCH_STRUCTURED_ERROR_METADATA_KEY` on
    :attr:`DispatchOutcome.metadata` and onto :attr:`ToolResultBlock.metadata`,
    but the OpenAI/vLLM wire serializer emits only
    ``{role, tool_call_id, content}`` — so the metadata signal NEVER reaches
    the model.

    Rather than change the serializer per-provider (and risk a fallback
    chain), surface a SANITIZED one-line hint in the tool-result ``content``
    itself, so it survives serialization for EVERY provider. The hint is a
    bounded budget-signal nudge appended ONCE after the existing error text.
    When the outcome carries no ``finalization_recommended`` structured error
    (the common case) the content is returned VERBATIM — bit-identical.

    Total — never raises; a malformed structured_error degrades to the raw
    content.
    """
    content = outcome.content
    metadata = outcome.metadata or {}
    structured_error = metadata.get(DISPATCH_STRUCTURED_ERROR_METADATA_KEY)
    if not isinstance(structured_error, dict):
        return content
    if structured_error.get(STRUCTURED_ERROR_FINALIZATION_RECOMMENDED_KEY) is not True:
        return content
    # : the reason is provider-visible — echo ONLY a bounded, token-shaped
    # value; anything untrusted/long/markup is dropped (full reason stays in
    # metadata/logs).
    reason_suffix = _safe_finalization_reason_suffix(
        structured_error.get(STRUCTURED_ERROR_REASON_KEY)
    )
    # Deliberately says nothing about WHICH budget ran out. Two producers reach
    # this line — a transport-retry give-up on a failing dependency, and a
    # delegation call refused because the run's cumulative work budget is spent —
    # and a sentence naming either one is a false statement about the other. The
    # reason token disambiguates, and the tool's own error text above it carries
    # the specifics.
    hint = (
        "[finalization-recommended] This tool gave up: a budget it depends on is "
        "exhausted" + reason_suffix + ". Do NOT keep retrying the same call — "
        "finalize your answer now on the best evidence already gathered."
    )
    # Append after the existing (sanitised) error text; keep one blank-line sep
    # when there is prior content so the hint reads as its own line.
    return f"{content}\n\n{hint}" if content else hint


def _result_block_from_outcome(
    tool_call_id: str, outcome: DispatchOutcome
) -> ToolResultBlock:
    """The transcript's projection of one dispatched call.

    Every path that records a result goes through here, so the transcript
    carries the same shape wherever the result came from: the text for the
    model, the value that text was projected from while nothing has stored it,
    and the two references that say where the whole value can be fetched from
    and which file it describes. Built separately at four call sites, those
    fields were dropped at three of them, and a dropped reference is
    indistinguishable from a result that never had one.
    """
    return ToolResultBlock(
        tool_call_id=tool_call_id,
        content=_tool_result_content_with_finalization_hint(outcome),
        is_error=outcome.is_error,
        metadata=outcome.metadata or {},
        canonical_content=(
            outcome.canonical_content if outcome.canonical_ref is None else None
        ),
        canonical_ref=outcome.canonical_ref,
        path=outcome.path,
    )


def _apply_deferred_tool_history(
    engine: QueryEngine,
    tool_call: ToolCall,
    outcome: DispatchOutcome,
    *,
    track_circuit_breaker: bool = True,
) -> None:
    """Append the deferred :class:`ToolResultBlock` + forget the call id.

 Mirrors the tail of :func:`_dispatch_tool` (history
 append + ``forget_tool_name``) without the ``_persist_snapshot``
 await so the caller can batch multiple appends and persist once.

 ``track_circuit_breaker``: the breaker is skipped when
 ``False``. The parallel-batch caller passes ``False`` for a TERMINAL-ONLY
 finalize-gate veto (a SYNTHETIC ``is_error`` outcome produced WITHOUT a
 dispatcher invocation), exactly as the serial ``_dispatch_tool`` returns
 BEFORE the post-dispatch breaker for that case. Counting a finalize-gate
 veto as a hard tool failure would let the breaker inject a corrective turn
 DURING the finalize-background gate (meta-leak violation).
 """
    # Parity with the serial path: a SUCCESSFUL chunkable write marks the
    # path "chunking started".
    if not outcome.is_error:
        _record_chunk_write_success(engine, tool_call)
    # Parity with the serial path: observe byte production
    # (Write/AppendFile are serialised, so this rarely carries a mutation,
    # but the observation must be identical on both dispatch paths).
    _longfile.observe_tool_result(
        engine, tool_call, outcome.content, is_error=outcome.is_error
    )
    # Parity with the serial path: fold the result into run-level tool-
    # precondition progress. A SUCCESSFUL call advances the entry whether the
    # model was forced into it or reached for the tool on its own — the
    # contract is that the tool ran.
    _preconditions.observe_tool_result(
        engine, tool_call, outcome.content, is_error=outcome.is_error
    )
    # Parity with the serial path: fold the result into the declared-file
    # read-back gate — a result declaring files the caller must open engages
    # it, a successful read releases what it opened.
    _pending_reads.observe_tool_result(
        engine, tool_call, outcome.metadata, is_error=outcome.is_error
    )
    engine.history.append(
        Message(
            role=MessageRole.tool,
            content_blocks=[
                _result_block_from_outcome(tool_call.id, outcome)
            ],
        )
    )
    # Parity with the serial path: track the consecutive same-tool/
    # same-error-class streak and, on a trip, append the bounded corrective
    # convergence turn (the caller persists once after the batch). Appended
    # AFTER this call's result block so ordering stays valid. SKIPPED for a
    # terminal-only finalize-gate veto (``track_circuit_breaker=False``) so the
    # gate cannot be misread as a hard tool failure.
    if track_circuit_breaker:
        circuit_breaker_corrective = _circuit_breaker_track_and_maybe_trip(
            engine, tool_call, outcome
        )
        if circuit_breaker_corrective is not None:
            engine.history.append(
                Message(
                    role=MessageRole.user,
                    content_blocks=[TextBlock(text=circuit_breaker_corrective)],
                    metadata={
                        SYNTHETIC_RECOVERY_METADATA_KEY: (
                            SYNTHETIC_RECOVERY_CIRCUIT_BREAKER
                        )
                    },
                )
            )
    engine.forget_tool_name(tool_call.id)


def _ingest_tool_evidence(
    engine: QueryEngine,
    outcome: DispatchOutcome,
) -> DispatchOutcome:
    """Append a successful outcome's typed evidence, or fail that outcome closed.

    The dispatcher never exposes evidence on the model-visible event/history
    path.  This is the sole query-runtime ingress, deliberately called only
    after serial or replayed dispatch status is final and before a snapshot.
    ``QueryEngine.append_tool_evidence`` validates a complete batch before it
    replaces its immutable ledger, so rejection cannot partially mutate it.
    """
    records = outcome.evidence_records
    if not records:
        return outcome
    if outcome.is_error or not outcome.success:
        # ``ToolResult`` rejects this shape at construction.  Keep the runtime
        # fail-closed if an alternate dispatcher ever constructs it directly.
        return replace(
            outcome,
            success=False,
            is_error=True,
            error_kind=DispatchErrorKind.execution,
            content="tool evidence is invalid on an unsuccessful dispatch",
            evidence_records=(),
        )
    producer = outcome.evidence_producer
    if producer is None:
        return replace(
            outcome,
            success=False,
            is_error=True,
            error_kind=DispatchErrorKind.execution,
            content="tool evidence has no registered producer binding",
            evidence_records=(),
        )
    try:
        engine.append_tool_evidence(records, producer=producer)
    except ValueError as exc:
        _logger.warning(
            "tool evidence rejected run=%s call_id=%s error=%s",
            engine.config.run_id,
            outcome.tool_call.id,
            type(exc).__name__,
        )
        return replace(
            outcome,
            success=False,
            is_error=True,
            error_kind=DispatchErrorKind.execution,
            content=f"tool evidence rejected: {exc}",
            evidence_records=(),
        )
    return outcome


def _dispatch_outcome_is_terminal(
    outcome: DispatchOutcome | None,
    *,
    engine: QueryEngine,
    tool_name: str,
) -> bool:
    """Return True when a successful tool outcome explicitly terminates the loop.

    Parallel-dispatch counterpart of :func:`_history_tool_result_is_terminal`,
    and applies the SAME expected-tool-name guard: when
    ``expected_terminal_tool`` is configured, a successful terminal-metadata
    outcome counts as terminal ONLY if ``tool_name`` matches the declared
    terminal tool. When ``expected_terminal_tool`` is None the behaviour is
    bit-identical to before (any successful terminal-metadata outcome
    counts) — no regression for a host backend or for the default.
    """
    if outcome is None or not outcome.success or outcome.is_error:
        return False
    metadata = outcome.metadata or {}
    if metadata.get(TERMINAL_TOOL_METADATA_KEY) is not True:
        return False
    expected = engine.config.expected_terminal_tool
    if expected is None:
        return True
    return tool_name == expected


# ---------------------------------------------------------------------------
# The run boundary. ``engine.history`` is a SESSION transcript, not a run's:
# cross-run history seeding prepends earlier runs of the same session verbatim.
# Every helper whose question is scoped to one run takes its messages from
# here. That is the rule, not an observation about the code: what enforces it
# is tests/unit/runtime/test_history_run_boundary.py, and that file states in
# its own docstring which shapes it cannot see.
# ---------------------------------------------------------------------------


def _this_run_messages(engine: QueryEngine) -> list[Message]:
    """Messages that belong to THIS run, in history order.

    ``engine.history`` also holds PRIOR-RUN turns the executor seeded into it
    (:data:`SESSION_HISTORY_SEED_METADATA_KEY`): the earlier runs of the same
    session, prepended verbatim — their prose, their tool calls and their tool
    results alike. Nothing else distinguishes them; ``Message`` carries no run
    id, so the seed tag IS the run boundary.

    A helper that asks a run-scoped question — did this run answer, did it
    write its deliverable, did it satisfy this precondition — takes its
    messages from here rather than from ``engine.history`` directly. The
    ordering is the point: a foreign turn is not in the sequence the helper
    iterates, so it is not something the helper can reach and then have to rule
    out. Testing provenance after the fact fails open on whatever the test
    forgot; drawing from a scoped sequence fails closed.

    Whole-history questions do NOT belong here. Wire-pairing repair, prompt
    assembly, token accounting and compaction are all about the transcript that
    goes to the provider, which is the whole of ``engine.history`` by
    definition. Pure / total — never raises.
    """
    return [
        message
        for message in engine.history
        if message.metadata.get(SESSION_HISTORY_SEED_METADATA_KEY) is not True
    ]


# Universal terminal-tool nudge.
#
# Any tenant declares its terminal tool name via
# ``QueryEngineConfig.expected_terminal_tool`` (routed from
# ``leader_config.expected_terminal_tool``). When set, the universal
# ``LoopConstants.terminal_tool_nudge_enabled`` knob gates the
# contract-repair nudge.


def _resolved_terminal_tool_name(engine: QueryEngine) -> str | None:
    """Return the configured terminal tool name for this engine.

    Resolution order:
      1. ``engine.config.expected_terminal_tool`` (per-tenant universal).
      2. ``None`` — unset; the nudge / terminal-only guard is disabled.
    """

    return engine.config.expected_terminal_tool or None


def _tool_name_for_call_id(
    engine: QueryEngine, tool_call_id: str
) -> str | None:
    """The tool name behind ``tool_call_id`` in this engine's history, or None."""

    return tool_name_for_result(engine.history, tool_call_id)


def _history_has_terminal_tool_result(engine: QueryEngine) -> bool:
    """Return True once THIS run has a successful terminal tool result.

    Two paths:
      * The classic ``TERMINAL_TOOL_METADATA_KEY``-flagged result satisfies
        every tenant, INCLUDING tenants with ``expected_terminal_tool``
        configured — a message-carrying terminal backend (e.g. ``pcm_answer``)
        sets the metadata key on terminal success.
      * When ``expected_terminal_tool`` is set (per-tenant generalisation),
        we additionally require the corresponding ``ToolUseBlock.name`` to
        match the configured tool name. This protects a tenant from mistaking
        a foreign tool's terminal-metadata flag for its own finalisation —
        the run is only "answered" through the declared terminal tool.

    Scoped to :func:`_this_run_messages`, and that scoping is load-bearing
    rather than tidy. This predicate is the "the run is already answered" arm
    of every gate that exists to stop a run ending unanswered — the terminal
    tool nudge and the run wind-down. Read over the whole session transcript it
    answers True
    for a run that has done nothing, because an EARLIER run of the session
    answered through the terminal tool and that turn was seeded into this one's
    history. Every one of those gates then declines to fire and the run ends
    with no answer at all, silently and in ``COMPLETED`` state. Stored history
    reaches this predicate the same way: rehydrated prior-run rows are tagged
    as seeds when they are loaded, so data written before this scoping existed
    is excluded here, but until it was, that data disarmed the next run.
    """

    expected = engine.config.expected_terminal_tool
    for message in reversed(_this_run_messages(engine)):
        for block in reversed(message.content_blocks):
            if not (
                isinstance(block, ToolResultBlock)
                and not block.is_error
                and block.metadata.get(TERMINAL_TOOL_METADATA_KEY) is True
            ):
                continue
            if expected is None:
                return True
            tool_name = _tool_name_for_call_id(engine, block.tool_call_id)
            if tool_name == expected:
                return True
    return False


def _terminal_tool_nudge_required(engine: QueryEngine) -> bool:
    """Return True iff the run needs the contract-repair terminal-tool nudge.

    Enable path: ``QueryEngineConfig.expected_terminal_tool`` is set AND
    ``rc.terminal_tool_nudge_enabled`` is True AND no terminal tool result
    is in history. Universal — keyed only on the per-tenant terminal-tool
    contract.
    """

    rc = engine.config.rc
    if _history_has_terminal_tool_result(engine):
        return False
    return (
        engine.config.expected_terminal_tool is not None
        and rc.terminal_tool_nudge_enabled
    )


def _suppress_terminal_only_meta_text(engine: QueryEngine) -> bool:
    """True iff the CURRENT terminal-only turn's visible assistant TEXT must be
 suppressed from live SSE + durable history.

 The terminal-tool nudge ALWAYS fires (write-first recovery + typed Finalize
 depend on it — a prose-only "Done, I created the file" with 0 tools must
 still be nudged into the actual Write + Finalize). The cost is that a weak
 model, on that post-nudge turn, self-narrates English 3rd-person ``META``
 prose ("The user asked … Let me finalize.") co-located with the ``Finalize``
 call. That redundant narration streams live and persists to durable
 ``session_messages`` + the memory fold + ``runs.result_preview``.

 Suppress ONLY that turn's TEXT — never the tool calls. ``Write`` /
 ``AppendFile`` (write-first recovery) AND ``Finalize`` pass through unchanged,
 so the file is still written and the typed-Finalize deliverables chip is
 preserved. Because the suppressed text is also kept out of ``text_buffer`` →
 out of ``engine.history``, even the UNfiltered ``_run_local_history`` →
 ``result_preview`` derivation is clean (the durable net then covers any path
 that bypasses this stream-suppression — pickup / reload).

 Suppress IFF ALL hold:
 * ``engine._terminal_only_active`` — we are in the post-nudge terminal-only
 turn. This latch is set by :func:`_append_terminal_tool_nudge` BEFORE
 that turn streams, so the decision is known up-front (NO buffering: the
 whole turn streams under one stable decision).
 * the resolved terminal tool is a BACKGROUND gate — its schema carries NO
 answer field (:func:`_terminal_tool_carries_answer_field` is False). For
 such a tool the user-facing answer can ONLY be prose, so the terminal-only
 turn's text is pure narration once an answer exists. A MESSAGE-CARRYING
 terminal (``pcm_answer`` / ``final_answer``) submits its answer via args
 and its visible text is the real answer surface — never suppressed.
 Unknown schema ⟹ exempt (fail-safe).
 * a PRIOR substantive answer already exists after the latest non-terminal
 work (:func:`_has_visible_assistant_prose_after_work`, floored at
 ``finalize_prose_gate_min_chars`` so a terse ``144`` counts). When NO
 prior answer exists — the terminal-only turn's text IS the answer (the
 genuinely-empty / first-answer-in-the-terminal-turn case) — it MUST stay
 visible, so this returns False and the net stays correct.

 Pure / side-effect free; cheap enough to evaluate once per stream attempt.
 """

    if not getattr(engine, "_terminal_only_active", False):
        return False
    terminal_tool = _resolved_terminal_tool_name(engine)
    if terminal_tool is None:
        return False
    # Only a BACKGROUND terminal (no answer-carrying field) routes its answer
    # through prose; a MESSAGE-CARRYING terminal answers via its args, so its
    # visible text is the real answer surface and must NOT be suppressed. Unknown
    # schema ⟹ exempt (keep the text) — fail-safe, multi-tenant.
    if _terminal_tool_carries_answer_field(engine, terminal_tool):
        return False
    return _has_visible_assistant_prose_after_work(
        engine, terminal_tool, engine.config.rc.finalize_prose_gate_min_chars
    )


def _apply_terminal_synthesis_output_reserve(
    engine: QueryEngine, max_output_tokens: int, output_cap: int
) -> int:
    """Final-turn-specific output-token floor.

    On the ACTUAL terminal / forced-final turn — i.e. the terminal-only nudge
    has fired (the durable ``engine._terminal_only_active`` latch, set by
    :func:`_append_terminal_tool_nudge`) OR the deadline backstop has fired
    (:func:`_terminal_deadline_reached`) — ensure the per-message output budget
    is at least ``rc.terminal_synthesis_output_reserve_tokens``, so the model
    has room to emit message + refs + outcome instead of being starved by the
    AdaptiveSafetyBand subtraction.

    Note: this previously keyed on :func:`_terminal_tool_nudge_required`,
    which is True on EVERY turn of a run that merely has
    ``expected_terminal_tool`` set + the nudge enabled + no terminal result
    yet — so the reserve floored the output budget on all turns, not only the
    final one. Keying on the ``_terminal_only_active`` latch restricts it
    to the genuine forced-final turn (after the nudge actually fired) +
    the deadline backstop.

    The floor is bounded by ``output_cap`` (the pre-safety-band global cap,
    ``max_context * llm_output_max_tokens_ratio``) so it can NEVER raise the
    global cap — it only reclaims tokens the safety band removed, and only on
    the final turn. ``reserve == 0`` (default) makes ``min(0, cap) == 0`` and
    the floor a no-op, so behaviour is bit-identical. Keys only on the
    generic terminal-tool contract; purely a budget number, no prompt
    wording.
    """

    reserve = engine.config.rc.terminal_synthesis_output_reserve_tokens
    if reserve <= 0:
        return max_output_tokens
    if not (
        getattr(engine, "_terminal_only_active", False)
        or _terminal_deadline_reached(engine)
    ):
        return max_output_tokens
    floor = min(reserve, output_cap)
    return max(max_output_tokens, floor)


def _history_has_file_write_result(engine: QueryEngine) -> bool:
    """Return True iff THIS run landed a successful file-write tool result.

    A file-write deliverable counts as produced when
    any tool named in ``rc.terminal_tool_nudge_file_write_tool_names``
    (default ``Write``/``AppendFile``) has a non-error tool_result in this
    run's messages. Used to decide whether the terminal-tool nudge should steer
    the model to write the deliverable FIRST. Core never hardcodes
    the host write-tool names — they come from the RC tuple so the check
    stays universal.

    Scoped to :func:`_this_run_messages`: the deliverable this run owes is one
    it writes here. A follow-up run in the same session inherits the earlier
    run's ``Write`` through cross-run history seeding, and over the whole
    transcript that earlier write would drop the write-first steer from the
    nudge for a run that has produced nothing — precisely the run that needs
    it most.
    """

    write_names = set(engine.config.rc.terminal_tool_nudge_file_write_tool_names)
    if not write_names:
        return False
    for message in _this_run_messages(engine):
        for block in message.content_blocks:
            if not (isinstance(block, ToolResultBlock) and not block.is_error):
                continue
            tool_name = _tool_name_for_call_id(engine, block.tool_call_id)
            if tool_name in write_names:
                return True
    return False


def _resolved_terminal_tool_nudge_text(engine: QueryEngine) -> str:
    """Resolve the message body for the terminal-tool nudge.

    The body comes from the ``terminal_tool_nudge`` template, which is handed
    the live terminal tool name so nothing has to hard-code it. An operator who
    wants different wording, or the same wording in another language, replaces
    the template — there is no second copy of the text in configuration to keep
    in step with it.

    When ``terminal_tool_nudge_write_first_enabled`` is set AND no file-write
    deliverable is in history yet (:func:`_history_has_file_write_result`), the
    body is PREFIXED with the ``terminal_tool_nudge_write_first`` template so a
    model that narrated "now let me write this file" and fired 0 tools is
    steered to the actual write tool first, not just the terminal tool. The
    prefix is bounded by the single-shot nudge latch (it never loops) and is a
    no-op for a run that has already written its deliverable.

    The wording is deliberately an internal control note rather than a
    second-person imperative. A weak model in a looping state used to
    PARAPHRASE the imperative straight into its visible answer; a note that is
    self-evidently not answer prose is harmless when echoed verbatim, and the
    functional trigger is unchanged — the run still finishes by calling the
    terminal tool with its best supported answer.
    """

    rc = engine.config.rc
    tool_name = _resolved_terminal_tool_name(engine) or "the configured terminal tool"
    body = engine.prompt_text("terminal_tool_nudge", terminal_tool=tool_name)
    if rc.terminal_tool_nudge_write_first_enabled and not _history_has_file_write_result(
        engine
    ):
        prefix = engine.prompt_text("terminal_tool_nudge_write_first")
        return f"{prefix}\n\n{body}"
    return body


def _terminal_deadline_reached(engine: QueryEngine) -> bool:
    """True iff the run's wall-clock budget is (nearly) spent.

    The budget is ``rc.agent_max_seconds`` measured from ``QueryEngine.run()``
    entry (``engine._run_started_monotonic``); the early-finalize fires once
    the elapsed time reaches ``agent_max_seconds - agent_deadline_finalize_
    slack_seconds`` so a final terminal-tool round-trip can still complete
    before an external trial / reaper kills the run.

    Returns False when the budget is disabled (``agent_max_seconds <= 0``)
    or the clock was never stamped (``_run_started_monotonic == 0.0``). A
    negative start is valid after snapshot resume when persisted wall-clock
    elapsed time exceeds the new process's monotonic uptime.
    """

    rc = engine.config.rc
    budget = rc.agent_max_seconds
    if budget <= 0.0:
        return False
    started = getattr(engine, "_run_started_monotonic", 0.0)
    if started == 0.0:
        return False
    slack = rc.agent_deadline_finalize_slack_seconds
    # Threshold floored at 0 so a slack >= budget still finalises promptly
    # rather than going negative.
    threshold = budget - slack
    if threshold < 0.0:
        threshold = 0.0
    return (time.monotonic() - started) >= threshold


async def _maybe_inject_pre_terminal_self_verify(engine: QueryEngine) -> bool:
    """Inject ONE bounded corrective turn before finalising.

    Called at every terminal-completion site BEFORE the loop treats a
    terminal-tool result as final. When ALL hold:

      * ``rc.pre_terminal_self_verify_enabled`` is True,
      * the per-run latch ``engine._pre_terminal_self_verify_used`` is unset,
      * the bounded counter is below
        ``rc.pre_terminal_self_verify_max_extra_turns``,
      * an host-supplied ``config.pre_terminal_self_verify_trigger``
        returns a non-empty corrective instruction,

    this appends ONE corrective user-role turn, latches (so it fires at most
    once per run), bumps the counter, and returns ``True`` so the caller
    does NOT finalise — the outer loop runs one more bounded turn in which
    the model can fix the cited-but-unobserved ref or perform the
    declared-but-missing mutation.

    Returns ``False`` (finalise as usual) when the
    feature is disabled, already used, over budget, or the trigger declines.
    The trigger predicate is tenant-supplied so the self-verify turn stays
    universal — core never inspects a specific terminal tool's payload.
    """

    rc = engine.config.rc
    if not rc.pre_terminal_self_verify_enabled:
        return False
    if getattr(engine, "_pre_terminal_self_verify_used", False):
        return False
    if (
        getattr(engine, "_self_verify_extra_turns_used", 0)
        >= rc.pre_terminal_self_verify_max_extra_turns
    ):
        return False
    trigger = engine.config.pre_terminal_self_verify_trigger
    if trigger is None:
        return False
    try:
        corrective = trigger(engine)
    except Exception as exc:  # pragma: no cover - defensive; never break finalise
        _logger.warning(
            "DIAG query.pre_terminal_self_verify.trigger_failed run=%s error=%s",
            engine.config.run_id,
            exc,
        )
        return False
    if not corrective:
        return False
    engine._pre_terminal_self_verify_used = True
    engine._self_verify_extra_turns_used = (
        getattr(engine, "_self_verify_extra_turns_used", 0) + 1
    )
    engine.history.append(
        Message(
            role=MessageRole.user,
            content_blocks=[TextBlock(text=corrective)],
            metadata={
                SYNTHETIC_RECOVERY_METADATA_KEY: (
                    SYNTHETIC_RECOVERY_PRE_TERMINAL_SELF_VERIFY
                )
            },
        )
    )
    # Persist the snapshot IMMEDIATELY after the corrective turn + latch
    # mutation, BEFORE the outer loop opens the next LLM call. Without this,
    # a crash or cross-pod resume between the injection and the next
    # persistence boundary loses BOTH the appended correction and the
    # ``_pre_terminal_self_verify_used`` latch — the resumed run would
    # re-fire the self-verify turn (latch lost) or finalise without the
    # correction (correction lost). The snapshot schema already carries the
    # latch + counter, so persisting here makes the at-most-once corrective
    # turn durable across a re-drive.
    await engine._persist_snapshot()
    _logger.warning(
        "DIAG query.pre_terminal_self_verify.injected run=%s tenant=%s turn=%s",
        engine.config.run_id,
        engine.config.tenant_id,
        engine.turn_id(),
    )
    return True


# Fallback corrective text when a regressed terminal turn must be re-vetoed
# but the host trigger returned no corrective for that turn. Uses the
# same vocabulary as the existing ``veto_error`` on the pre-dispatch veto
# path.
_TERMINAL_CANDIDATE_REVETO_FALLBACK = (
    "Your previous answer was withheld and this replacement is empty or much "
    "shorter than the answer you had already drafted. Do not shorten or drop "
    "your answer — re-send your full answer."
)


def _terminal_candidate_message(tool_call: ToolCall) -> str:
    """Return the stripped terminal-answer ``message`` body for ``tool_call``.

    Universal over the terminal-answer contract: the required free-text body
    of every terminal tool is the generic ``message`` argument (a host's own
    answer tool carries it too). Core never inspects
    a tenant-specific payload — only this generic field. Returns ``""`` when
    args are missing / not a mapping / the field is absent or non-string.
    """

    args = tool_call.arguments
    if not isinstance(args, dict):
        return ""
    message = args.get("message")
    if not isinstance(message, str):
        return ""
    return message.strip()


def _terminal_candidate_snapshot_args(tool_call: ToolCall) -> dict[str, Any]:
    """Return a JSON-serialisable copy of the terminal tool args.

    Stored on the durable per-run candidate so the preserved draft survives a
    cross-pod resume. Falls back to an empty mapping when args are not a
    mapping (a candidate is only ever recorded for a substantive ``message``
    body, so this is defensive).
    """

    args = tool_call.arguments
    if not isinstance(args, dict):
        return {}
    return dict(args)


def _terminal_candidate_hash(message: str) -> str:
    """Stable content hash of the preserved candidate body (audit only)."""

    return hashlib.sha256(message.encode("utf-8", errors="replace")).hexdigest()


def _resolve_terminal_candidate_corrective(
    engine: QueryEngine, tool_call: ToolCall, corrective: str | None
) -> str | None:
    """Candidate-answer preservation.

    Wraps the host pre-dispatch veto verdict (``corrective``) so that
    the first SUBSTANTIVE terminal-answer draft is not silently lost when a
    later repair turn regresses to an empty / too-short body.

    Gated entirely behind ``rc.terminal_candidate_preserve_enabled`` —
    default-off returns ``corrective`` unchanged, so the candidate is discarded
    exactly as today (bit-identical). When enabled:

    * **Preserve** — if the verdict is a veto (``corrective`` truthy) and the
      proposed body is substantive (``len(message) >=
      max(1, terminal_answer_min_message_chars)``) and no substantive candidate
      is already held, durably record ``{tool, args, veto_reason,
      candidate_hash, message_chars, substantive}`` on the engine. The verdict
      is returned unchanged (the veto still fires).
    * **Re-veto a regression** — if a substantive candidate is already held and
      the current body regresses (empty, or shorter than
      ``terminal_answer_min_message_chars`` when that floor is set), force the
      veto exactly once using the EXISTING corrective text (no new wording
      is synthesised; the model owns the corrected args). The
      one-shot is bounded by the durable ``_terminal_candidate_reveto_used``
      latch.
    * **Allow-through** — once the single re-veto repair credit is spent, a
      still-regressed body is allowed through (verdict forced to ``None``) so
      the run finalises on best evidence rather than looping.

    This function only mutates engine-side fields; the caller's existing
    ``await engine._persist_snapshot()`` on the veto path makes the write
    durable (horizontal / cross-pod safe). It NEVER mutates the answer body.
    """

    rc = engine.config.rc
    if not rc.terminal_candidate_preserve_enabled:
        return corrective

    message = _terminal_candidate_message(tool_call)
    min_chars = max(0, rc.terminal_answer_min_message_chars)
    substantive_floor = max(1, min_chars)
    is_substantive = len(message) >= substantive_floor

    saved = engine._terminal_candidate

    # ``isinstance`` inline (vs a separate ``saved_substantive`` bool) so mypy
    # narrows ``saved`` to ``dict`` for the ``saved.get(...)`` reads below —
    # behaviour-identical, fixes a pre-existing union-attr (None.get) flag.
    if isinstance(saved, dict) and bool(saved.get("substantive")):
        # A substantive draft was preserved earlier. A regression is an empty
        # body, or (when a floor is set) one shorter than the floor.
        regressed = (not message) or (min_chars > 0 and len(message) < min_chars)
        if not regressed:
            # The current body is itself substantive — defer to the normal
            # verdict; do not clobber the already-preserved candidate.
            return corrective
        if not engine._terminal_candidate_reveto_used:
            engine._terminal_candidate_reveto_used = True
            forced = corrective or _TERMINAL_CANDIDATE_REVETO_FALLBACK
            _logger.warning(
                "DIAG terminal_candidate.regressed run=%s tenant=%s "
                "action=reveto saved_chars=%s new_chars=%d",
                engine.config.run_id,
                engine.config.tenant_id,
                saved.get("message_chars"),
                len(message),
            )
            return forced
        # Repair credit already spent — allow the regressed body through so the
        # run finalises rather than looping.
        _logger.warning(
            "DIAG terminal_candidate.regressed run=%s tenant=%s "
            "action=allow saved_chars=%s new_chars=%d",
            engine.config.run_id,
            engine.config.tenant_id,
            saved.get("message_chars"),
            len(message),
        )
        return None

    # No substantive candidate held yet. Preserve the current body iff the
    # verdict is a veto AND the body is worth keeping.
    if corrective and is_substantive:
        engine._terminal_candidate = {
            "tool": tool_call.name,
            "args": _terminal_candidate_snapshot_args(tool_call),
            "veto_reason": "pre_dispatch_terminal_verify",
            "candidate_hash": _terminal_candidate_hash(message),
            "message_chars": len(message),
            "substantive": True,
        }
        _logger.warning(
            "DIAG terminal_candidate.preserved run=%s tenant=%s chars=%d hash=%s",
            engine.config.run_id,
            engine.config.tenant_id,
            len(message),
            engine._terminal_candidate["candidate_hash"],
        )
    return corrective


def _terminal_candidate_repair_applies(
    engine: QueryEngine, tool_call: ToolCall
) -> bool:
    """Gate for candidate-regression protection on the REPAIR turn, i.e. once
    the pre-dispatch-verify one-shot latch is already closed.

    The pre-dispatch terminal veto (:func:`_pre_dispatch_terminal_verify_applies`)
    is fire-at-most-once: it sets ``_pre_dispatch_terminal_verify_used`` on the
    FIRST veto, so on the model's corrected re-submission (the repair turn) the
    pre-dispatch gate is CLOSED and the candidate-regression check wired inside
    it never re-runs.

    This independent seam re-runs candidate-regression protection on the
    expected terminal tool EVEN WHEN the pre-dispatch latch is closed. ALL must
    hold:

      * ``rc.terminal_candidate_preserve_enabled`` is True (same kill-switch as
        the preservation seam — default-off is bit-identical),
      * a per-tenant terminal tool is declared (``config.expected_terminal_tool``)
        AND ``tool_call`` IS that tool (universal; never intercepts a
        non-terminal tool),
      * a SUBSTANTIVE candidate is already held
        (``engine._terminal_candidate["substantive"]``) — only true AFTER the
        first veto preserved one, which is also when the pre-dispatch latch is
        closed, so this branch and the pre-dispatch branch never both fire on
        one dispatch.

    The actual regress / re-veto-once / allow-through decision is delegated to
    the EXISTING :func:`_resolve_terminal_candidate_corrective` (keyed on the
    durable ``_terminal_candidate_reveto_used`` latch), so the repair seam and
    the preservation seam share one decision implementation and one latch.

    Cheap and side-effect-free so ``_dispatch_tool`` can call it on every
    dispatch; returns False when disabled or conditions not met.
    """

    rc = engine.config.rc
    if not rc.terminal_candidate_preserve_enabled:
        return False
    expected = engine.config.expected_terminal_tool
    if expected is None or tool_call.name != expected:
        return False
    saved = engine._terminal_candidate
    return isinstance(saved, dict) and bool(saved.get("substantive"))


# ---------------------------------------------------------------------------
# Universal prose-gate before a BACKGROUND terminal tool.
#
# The terminal tool (e.g. ``Finalize``) operates as a pure background gate: its
# answer field is removed and its tool_use / tool_result pair is filtered from
# the live stream + durable history, so the ONLY user-facing answer is the
# model's own visible assistant prose. A small empirical tail of runs call the
# terminal tool with NO substantive prose after their last real work; those
# would surface an empty answer. This gate vetoes such a dispatch ONCE and
# injects one bounded repair turn asking the model to write the answer as
# normal text first, then call the terminal tool. One-shot, snapshot-persisted.
#
# Ported from the host ``_has_visible_assistant_prose_after_work`` /
# ``_is_non_finalize_tool_activity`` predicate (it operated on engine history,
# which core owns) but driven off ``expected_terminal_tool`` — NOT a hardcoded
# ``Finalize`` name — so it stays universal.
# ---------------------------------------------------------------------------


def _strip_tool_name_prefix(name: str) -> str:
    """Strip a legacy ``tool:`` namespace prefix from a tool name (universal).

    Mirrors the host / chat ``tool:`` prefix stripping so a terminal tool
    advertised as ``tool:Finalize`` still matches the configured
    ``expected_terminal_tool`` name. Returns ``name`` unchanged when no prefix
    is present.
    """

    return name[5:] if name.startswith("tool:") else name


def _is_terminal_tool_name(name: object, terminal_tool: str) -> bool:
    """True iff ``name`` is the configured terminal tool (prefix-tolerant).

 Universal: matched against the per-tenant ``terminal_tool`` (the resolved
 ``expected_terminal_tool``), never a hardcoded tool name. Exact match after
 stripping any ``tool:`` prefix — never a substring/prefix match, so a tool
 like ``FinalizeFile`` is NOT mistaken for ``Finalize``.
 """

    return isinstance(name, str) and _strip_tool_name_prefix(name) == terminal_tool


def _is_non_terminal_tool_activity(block: object, terminal_tool: str) -> bool:
    """True for tool activity that is real work, NOT the terminal gate.

    Ported from the host ``_is_non_finalize_tool_activity`` but keyed on
    the configured ``terminal_tool``:

      * a ``ToolUseBlock`` is real work unless it is the terminal tool;
      * a ``ToolResultBlock`` is real work unless its named tool is the terminal
        tool, OR (when unnamed) it carries the terminal-metadata flag — an
        unnamed successful terminal result is still the gate, not user work.
    """

    if isinstance(block, ToolUseBlock):
        return not _is_terminal_tool_name(block.name, terminal_tool)
    if isinstance(block, ToolResultBlock):
        tool_name = block.metadata.get("tool_name")
        if isinstance(tool_name, str):
            return not _is_terminal_tool_name(tool_name, terminal_tool)
        # A successful terminal result without a name is still the terminal
        # gate, not user work. Named non-terminal results (the common path) are
        # handled above; unnamed non-terminal results remain visible work.
        return block.metadata.get(TERMINAL_TOOL_METADATA_KEY) is not True
    return False


def _has_visible_assistant_prose_after_work(
    engine: QueryEngine, terminal_tool: str, min_chars: int
) -> bool:
    """Whether the run already has substantive user-facing prose after work.

 Ported from the host ``_has_visible_assistant_prose_after_work`` (it
 operated on engine history, which core owns), generalised over
 ``terminal_tool`` + a substantive ``min_chars`` floor. Typed-terminal runs
 often look like::

 assistant tool(Bash) -> tool result -> assistant prose answer ->
 assistant tool(Finalize) -> tool result

 In that shape the prose IS the user-facing answer and the terminal payload
 is only the internal gate. A payload-only terminal (all visible prose
 occurred BEFORE the latest non-terminal work, or no substantive prose at
 all) returns False — that prose was progress narration, not the final
 answer — so the prose-gate fires.

 Reads :func:`_this_run_messages`, so the PRIOR-RUN turns cross-run history
 seeding prepends are not in the window at all — otherwise a prior run's
 seeded answer prose would falsely satisfy the gate and let a payload-only
 terminal finalise with an empty CURRENT-run answer. Same boundary as
 the host ``_run_local_history`` seed-strip.

 ``min_chars`` is the stripped-length floor a single assistant ``TextBlock``
 must reach to count as substantive (``finalize_prose_gate_min_chars``); a
 floor of 0 accepts any non-empty visible prose. Pure / total — never raises.
 """

    last_work_pos = -1
    latest_prose_pos = -1
    pos = 0
    # A floor of 0 means "any non-empty visible prose counts" (so we still
    # require at least 1 stripped char); a positive floor demands that many.
    substantive_floor = max(1, min_chars)
    for message in _this_run_messages(engine):
        for block in message.content_blocks:
            if _is_non_terminal_tool_activity(block, terminal_tool):
                last_work_pos = pos
            if (
                message.role is MessageRole.assistant
                and message.metadata.get(PARTIAL_ASSISTANT_ATTEMPT_METADATA_KEY)
                is not True
                and isinstance(block, TextBlock)
                and len(block.text.strip()) >= substantive_floor
            ):
                latest_prose_pos = pos
            pos += 1
    return latest_prose_pos > last_work_pos


def _preserve_completed_answer_on_stream_error(engine: QueryEngine) -> bool:
    """True iff a transient stream/provider error must NOT fail the run.

    Fires when the run has already produced a substantive user-facing
    assistant answer after its latest non-terminal work — the reply the user
    saw stream live. In that state a provider / idle error raised on a later
    harness-forced continuation turn (the terminal-tool nudge or a
    continue-prompt injection) is bolt-on: the answer is already delivered,
    so the run must complete on it rather than propagate the transient error
    as a FAILED terminal status. Gated by
    ``rc.preserve_completed_answer_on_stream_error`` (default on); the
    substantive floor reuses ``finalize_prose_gate_min_chars``. Pure / total.
    """

    rc = engine.config.rc
    if not getattr(rc, "preserve_completed_answer_on_stream_error", False):
        return False
    return _has_visible_assistant_prose_after_work(
        engine,
        _resolved_terminal_tool_name(engine) or "",
        rc.finalize_prose_gate_min_chars,
    )


async def _complete_run_on_preserved_answer(
    engine: QueryEngine, *, reason: str
) -> AsyncIterator[TurnEvent]:
    """Complete the run on an already-delivered answer after a stream error.

    Mirrors the voluntary-finish terminal (an ``end_turn`` ``message_stop``
    followed by a transition to ``COMPLETED``) so the run's terminal status
    reflects the substantive reply already in history rather than the
    transient provider / idle error raised on a harness-forced continuation
    turn. Emits a ``state_changed`` first so the reason is observable on the
    bus. The caller ``return``s immediately after draining these events.
    """

    from_state = engine.state
    yield _emit_state_change(engine, from_state, from_state, reason=reason)
    yield TurnEvent(
        type=EventType.MESSAGE_STOP,
        run_id=engine.config.run_id,
        payload={
            "turn_id": engine.turn_id(),
            "stop_reason": "end_turn",
            "tokens_used": _tokens_used_payload(engine),
            "cache_hit_rate": engine.total_usage.this_turn_cache_hit_rate(),
        },
    )
    engine.transition_to(LoopState.COMPLETED)


def _transient_retry_backoff_seconds(
    rc: LoopConstants, attempt: int, exc: BaseException
) -> float:
    """Backoff (seconds) before the ``attempt``-th transient-error retry.

    ``attempt`` is 1-based. The delay is an exponential term
    ``base * 2 ** (attempt - 1)`` bounded by the configured ceiling. When the
    classifier surfaced a server-stated ``Retry-After`` on the error it takes
    precedence (some providers pace 429s), but is still clamped by the same
    ceiling so worst-case latency stays bounded. A zero base with no
    ``Retry-After`` retries immediately. Pure / total — never raises.
    """
    base = rc.llm_transient_error_retry_backoff_base_seconds
    ceiling = rc.llm_transient_error_retry_backoff_max_seconds
    delay = base * (2 ** (attempt - 1)) if base > 0.0 else 0.0
    classified = getattr(exc, "classified", None)
    retry_after = (
        getattr(classified, "retry_after_seconds", None)
        if classified is not None
        else None
    )
    if isinstance(retry_after, (int, float)) and retry_after > 0:
        delay = max(delay, float(retry_after))
    if ceiling > 0.0:
        delay = min(delay, ceiling)
    return max(0.0, delay)


def _this_run_model_turns(engine: QueryEngine) -> list[Message]:
    """Assistant turns carrying THIS run's model's own words, in history order.

    ``engine.history`` is not a record of one run. Two kinds of assistant turn
    sit in it that the model did not say here, and both are indistinguishable
    from a real answer once their text has been read out of the message:

    * PRIOR-RUN turns the executor seeded into this run's history
      (:data:`SESSION_HISTORY_SEED_METADATA_KEY`). Cross-run history seeding
      prepends earlier runs of the same session verbatim, so the newest
      assistant prose in ``history`` is routinely a fluent, complete answer to
      a question THIS run was never asked.
    * runtime-synthesised recovery scaffolding
      (:data:`SYNTHETIC_RECOVERY_METADATA_KEY` — the empty-completion /
      post-tool ``(empty)`` placeholders, the guaranteed-terminal tool-use
      turn). The runtime's own words, not the model's.

    Callers that ask "what did the model say in this run?" take their messages
    from here rather than from ``engine.history`` directly, so a turn from
    another run is not something they can reach and then have to rule out.
    The run half of the exclusion is :func:`_this_run_messages` — this narrows
    it to the model's own words rather than restating it, so there is exactly
    one place that knows where a run begins, and helpers that must also see
    ``role=tool`` results (write accounting, precondition rehydration) share
    that place instead of deriving a second answer to the same question.
    Pure / total — never raises.
    """
    return [
        message
        for message in _this_run_messages(engine)
        if message.role is MessageRole.assistant
        and not message.metadata.get(SYNTHETIC_RECOVERY_METADATA_KEY)
        and message.metadata.get(PARTIAL_ASSISTANT_ATTEMPT_METADATA_KEY) is not True
    ]


def run_has_final_answer(engine: QueryEngine) -> bool:
    """True iff THIS run already produced a substantive visible assistant answer.

    Scans :func:`_this_run_model_turns` for an assistant ``TextBlock`` with
    non-whitespace text — so neither a seeded prior answer nor the guard's own
    re-drive placeholder can mask a genuinely unanswered current run. Distinct
    from :func:`_has_visible_assistant_prose_after_work` (which requires the
    prose to come AFTER the latest non-terminal work and applies the
    substantive-char floor): this asks only "is there ANY real visible answer
    in this run yet", the precondition for the empty-completion guard.
    Pure / total — never raises.
    """
    for message in _this_run_model_turns(engine):
        for block in message.content_blocks:
            if isinstance(block, TextBlock) and block.text.strip():
                return True
    return False


def _append_empty_completion_redrive_nudge(engine: QueryEngine) -> None:
    """Append an API-valid synthetic pair to re-drive after a bare-empty turn.

    The empty turn appended nothing to history (no text / tool / reasoning), so
    the tail is whatever preceded it. To keep the wire sequence valid on the
    re-drive (never ``tool -> user`` nor a bare double-user turn), append an
    empty-marker assistant turn followed by a corrective user nudge, both flagged
    :data:`SYNTHETIC_RECOVERY_METADATA_KEY` so neither is mistaken for a real
    model answer by :func:`run_has_final_answer` /
    :func:`_latest_durable_answer_text`. Reuses the existing empty-response
    recovery scaffolding text so no new literal is introduced.
    """
    rc = engine.config.rc
    engine.history.append(
        Message(
            role=MessageRole.assistant,
            content_blocks=[
                TextBlock(text=rc.post_tool_empty_nudge_assistant_text)
            ],
            metadata={
                SYNTHETIC_RECOVERY_METADATA_KEY: (
                    SYNTHETIC_RECOVERY_POST_TOOL_EMPTY_NUDGE
                )
            },
        )
    )
    engine.history.append(
        Message(
            role=MessageRole.user,
            content_blocks=[TextBlock(text=rc.continue_prompt_text)],
            metadata={
                SYNTHETIC_RECOVERY_METADATA_KEY: (
                    SYNTHETIC_RECOVERY_POST_TOOL_EMPTY_NUDGE
                )
            },
        )
    )


async def _emit_empty_completion_terminal(
    engine: QueryEngine,
) -> AsyncIterator[TurnEvent]:
    """Drive the run FAILED on a bare-empty end_turn that produced no answer.

    Used by the empty-completion guard once its bounded re-drive budget is
    exhausted: the model kept ending the turn with ``finish_reason='stop'`` and
    no visible text / tool call / reasoning, and the run never produced an answer
    or a terminal tool result. Sealing that as COMPLETED would report an empty
    turn as a clean answer, so the run is terminated FAILED with a self-evident
    ``no_answer_empty_completion`` reason. Unlike :func:`_emit_llm_terminal` this
    is NOT an LLM-provider class error, so the Stop / SessionEnd death-spiral
    guard (``engine.skip_terminal_hooks``) is left untouched — ordinary terminal
    hooks still run. Emits ``state_changed -> error -> message_stop`` mirroring
    the other terminal sites.
    """

    # Pair any orphan tool_use before the FAILED transition (defensive: the
    # bare-empty turn appended no tool_use, but mirror the other terminals so a
    # resumed snapshot never carries a dangling call).
    _synthesize_missing_tool_results(
        engine.history,
        error_content=engine.prompt_text("tool_result_interrupted"),
    )
    from_state = engine.state
    reason = "no_answer_empty_completion"
    engine.transition_to(LoopState.FAILED)
    yield _emit_state_change(engine, from_state, LoopState.FAILED, reason=reason)
    yield TurnEvent(
        type=EventType.ERROR,
        run_id=engine.config.run_id,
        payload={
            "kind": reason,
            "message": (
                "assistant ended the turn with no visible answer, no tool "
                "call and no reasoning, and the run produced no answer"
            ),
        },
    )
    yield TurnEvent(
        type=EventType.MESSAGE_STOP,
        run_id=engine.config.run_id,
        payload={
            "turn_id": engine.turn_id(),
            "stop_reason": StopReason.error.value,
        },
    )


async def _emit_tool_precondition_terminal(
    engine: QueryEngine,
) -> AsyncIterator[TurnEvent]:
    """Drive the run FAILED when a tool precondition ran out of attempts.

    The run asked for a tool to be called before the agent answered, the forced
    turns were spent, and it still has not been called successfully. Completing
    would report an answer produced without the thing the caller made a
    condition of it, so the run is terminated FAILED with a reason naming the
    tool and the last error the tool reported. Like
    :func:`_emit_empty_completion_terminal` this is not an LLM-provider class
    error, so ``engine.skip_terminal_hooks`` is left untouched and the ordinary
    terminal hooks still run. Emits ``state_changed -> error -> message_stop``
    mirroring the other terminal sites.
    """

    # Pair any orphan tool_use before the FAILED transition so a resumed
    # snapshot never carries a dangling call — the exhausting turn may well
    # have appended a tool_use whose result never came back.
    _synthesize_missing_tool_results(
        engine.history,
        error_content=engine.prompt_text("tool_result_interrupted"),
    )
    from_state = engine.state
    reason = "tool_precondition_unsatisfied"
    message = _preconditions.failure_message(engine)
    engine.transition_to(LoopState.FAILED)
    yield _emit_state_change(engine, from_state, LoopState.FAILED, reason=reason)
    yield TurnEvent(
        type=EventType.ERROR,
        run_id=engine.config.run_id,
        payload={"kind": reason, "message": message},
    )
    yield TurnEvent(
        type=EventType.MESSAGE_STOP,
        run_id=engine.config.run_id,
        payload={
            "turn_id": engine.turn_id(),
            "stop_reason": StopReason.error.value,
        },
    )


def _prose_gate_just_injected(engine: QueryEngine) -> bool:
    """True iff the LAST history turn is the
    prose-gate corrective the dispatch path just appended.

    Lets the serial dispatch loop detect a prose-gate veto WITHOUT a transient
    flag: the veto in :func:`_dispatch_tool` appends a NON-terminal error
    tool_result PLUS a synthetic user turn tagged
    :data:`SYNTHETIC_RECOVERY_PROSE_GATE_REPAIR` as the final message. When that
    is the tail, the loop breaks the batch (does NOT dispatch later sibling tool
    calls AFTER the user-repair turn) and re-drives. Pure / total."""

    return _prose_gate_injected(engine)


def _terminal_tool_carries_answer_field(
    engine: QueryEngine, terminal_tool: str
) -> bool:
    """Return True iff the terminal tool's schema carries the user answer itself.

 The prose-gate must fire ONLY for a BACKGROUND terminal tool (one that
 does NOT carry the answer in its args, so the user-facing answer can only
 be the model's prose). A MESSAGE-CARRYING terminal tool (``pcm_answer`` /
 ``final_answer`` / ``Finalize.answer``) legitimately
 answers via its args and emits no prose — vetoing it would withhold a
 valid answer submission. Reuses the SAME schema signal as the synthesiser
 (the answer-carrying argument names the host declared).

 Fails SAFE (returns ``True`` ⟹ EXEMPT ⟹ no prose-gate) for multi-tenant
 safety whenever the schema cannot be introspected: no core registry, the
 tool is unknown to core (a host-backend tool whose contract core does not
 hold), or the parameters are unreadable. Only a tool that core CAN resolve
 AND whose declared properties contain NONE of the answer-carrying names is
 treated as a background terminal (returns ``False``). Cheap / side-effect
 free; mirrors :func:`_terminal_tool_accepts_refs`.
 """

    registry = getattr(engine, "tools", None)
    getter = getattr(registry, "get", None)
    if getter is None:
        return True  # no core registry → exempt (cannot prove background)
    try:
        tool = getter(terminal_tool)
    except Exception:  # pragma: no cover - defensive
        return True
    if tool is None:
        return True  # tool unknown to core (host backend) → exempt
    try:
        properties = tool.definition.parameters.properties
    except Exception:  # pragma: no cover - defensive
        return True
    answer_names = argument_names(ToolArgumentSlot.answer, roles=engine.config.tool_roles)
    if not answer_names:
        # The host named no answer-carrying arguments, so nothing here can
        # prove this tool is a background one. Fail SAFE, exactly as an
        # unreadable schema does: exempt, no prose gate.
        return True
    return any(name in properties for name in answer_names)


def _finalize_prose_gate_applies(
    engine: QueryEngine, tool_call: ToolCall
) -> bool:
    """gate for the universal prose-gate veto.

    Returns True iff the prose-gate should VETO this terminal dispatch and
    inject one bounded prose-repair turn. ALL must hold:

      * ``rc.finalize_prose_gate_enabled`` is True,
      * a per-tenant terminal tool is declared (``config.expected_terminal_tool``)
        AND ``tool_call`` IS that tool — the gate never intercepts a
        non-terminal tool (reads / writes / exec dispatch unaffected),
      * whichever of the two tests this dispatch answers to still has budget:
        the payload-only case is bounded by the durable one-shot latch
        ``_finalize_prose_gate_used`` (fire-at-most-once across resume), the
        pointer case by its own
        ``rc.finalize_prose_gate_pointer_max_repair_attempts``
        (:func:`_pointer_answer_repair_budget_spent`). Two bounds and not one,
        because the two tests fail differently: a model shown a payload-only
        terminal and told to write prose first either writes it or does not, and
        a second veto on the same run has nothing new to say — while the pointer
        refusal was measured against a model that answers the correction with a
        second filing notice, where one attempt detects the failure and does not
        repair it,
      * the terminal tool is a BACKGROUND gate — its resolved input schema has
        NO answer-carrying field (:func:`_terminal_tool_carries_answer_field` is
        False). A MESSAGE-CARRYING terminal tool (``pcm_answer`` /
        ``Finalize.answer``) answers via its args and is
        EXEMPT; an unknown schema is EXEMPT too (multi-tenant safe), and
      * the run has NO substantive visible assistant prose after its latest
        non-terminal work tool (:func:`_has_visible_assistant_prose_after_work`
        is False) — i.e. this is a payload-only terminal — OR that prose is
        principally a POINTER to a file the run wrote and the user cannot open
        (:func:`_pointer_answer_evidence`, inert unless the deployment declares
        the workspace hidden). The second is the same failure as the first with
        enough characters on it to clear a length floor.

    Returns False when the gate is disabled or the conditions are not met.
    Cheap and side-effect-free so ``_dispatch_tool`` can call it on every
    dispatch without cost when the gate is closed.
    """

    rc = engine.config.rc
    if not rc.finalize_prose_gate_enabled:
        return False
    terminal_tool = _resolved_terminal_tool_name(engine)
    if terminal_tool is None or tool_call.name != terminal_tool:
        return False
    # A MESSAGE-CARRYING terminal tool answers via its args (no prose
    # expected); only a BACKGROUND terminal (no answer-carrying field in its
    # schema) must produce prose. Unknown schema ⟹ exempt (multi-tenant
    # safe). This keeps the gate from withholding a valid payload-only answer
    # from pcm_answer / Finalize.answer.
    if _terminal_tool_carries_answer_field(engine, terminal_tool):
        return False
    # Substantive visible prose after the latest real work ⟹ the answer already
    # exists ⟹ no repair turn (the common, healthy shape) — UNLESS that prose is
    # principally a pointer to a file the user cannot open, which clears any
    # length floor while delivering nothing (:func:`_pointer_answer_evidence`,
    # inert unless the deployment says the workspace is hidden).
    if _has_visible_assistant_prose_after_work(
        engine, terminal_tool, rc.finalize_prose_gate_min_chars
    ):
        if _pointer_answer_repair_budget_spent(engine):
            return False
        return _pointer_answer_evidence(engine) is not None
    return not getattr(engine, "_finalize_prose_gate_used", False)


def _plain_stop_answer_floor_applies(engine: QueryEngine) -> bool:
    """Whether a run completing on a plain stop still owes the user an answer.

    :func:`_finalize_prose_gate_applies` can only ever intercept a TERMINAL-TOOL
    dispatch. A run in which the model simply stops — ``finish_reason='stop'``,
    no tool call — never reaches that seam, and on a deployment that declares no
    terminal tool that is how essentially every run ends, so the gate never
    participates at all. Measured shape: a leader delegated correctly, its
    subagents wrote five result files, and the reply the user actually received
    was 97 characters long. Nothing was broken; the gate was simply somewhere
    else.

    This is the SAME gate at the other completion path. It reuses the floor
    (``finalize_prose_gate_min_chars``), the durable latch
    (``_finalize_prose_gate_used``) and the repair text, so the SHORT-ANSWER
    half of the mechanism still fires AT MOST ONCE per run: whichever path
    reaches it first spends the single shot for both, and a run can never
    oscillate between them.

    The POINTER half does not share that shot. It is a different test —
    :func:`_pointer_answer_evidence`, an answer long enough to clear any floor
    that still delivers nothing — and it is bounded on its own by
    ``rc.finalize_prose_gate_pointer_max_repair_attempts``. The two remain
    exclusive per firing (a run either has substantive prose after its work or
    it does not, and that is the branch below), so neither can consume the
    other's budget; what changed is that spending one no longer silences the
    other for the rest of the run. That entanglement had teeth: a run repaired
    once for a thin answer could then file a notice about a 13 KB document and
    the pointer test, holding a spent latch, would never look at it.

    ALL must hold:

      * ``rc.finalize_prose_gate_enabled`` — the same kill switch, so an
        operator who turns the gate off restores the prior behaviour on BOTH
        paths and BOTH tests, not one of them;
      * the bound belonging to the branch this run lands in still has room: the
        durable one-shot latch for the short-answer branch (fire-at-most-once
        across resume), the attempt budget for the pointer branch
        (:func:`_pointer_answer_repair_budget_spent`, likewise resume-safe);
      * the run produced SOME visible assistant answer
        (:func:`run_has_final_answer`). A run that produced none
        at all belongs to the empty-completion guard, which owns that turn with
        its own RC, its own multi-re-drive budget and a loud FAILED terminal
        once it is spent — strictly more than the one repair turn offered here.
        The two predicates are exact complements on this point, so the paths
        never both fire, and an operator who switches that guard off keeps the
        sealed-empty behaviour they asked for rather than quietly inheriting
        this one;
      * the run has NO substantive visible assistant prose after its latest
        non-terminal work (:func:`_has_visible_assistant_prose_after_work`).
        With no terminal tool configured the predicate is passed ``""``, which
        matches no tool name, so every tool result counts as real work — the
        correct reading when nothing is a terminal gate. There is ONE shape in
        which prose that clears the floor still leaves the user with nothing:
        an answer that is principally a POINTER to a file this run wrote, on a
        surface where the user cannot open it. That is
        :func:`_pointer_answer_evidence`, and it is the second way this
        predicate can say yes. It is inert unless the deployment declares the
        workspace hidden, so the floor's behaviour is unchanged by default.

    On the ANSWER-FIELD EXEMPTION. The terminal path exempts a terminal tool
    whose schema carries the answer in its own args
    (:func:`_terminal_tool_carries_answer_field`): such a tool IS the answer
    channel, and vetoing it would withhold a valid submission. Transplanted
    verbatim onto this path that condition is not merely inert but inverted —
    by definition NO terminal tool was called on the turn that stopped, so it
    would key on the schema of a tool this run never used, and would switch the
    floor OFF for precisely the tenants whose message-carrying terminal tool
    went uncalled: the runs that delivered nothing through any channel.

    What the exemption actually asks is "has the answer already reached the user
    by some route other than prose?", and on this path exactly one thing makes
    that true: a terminal tool result is ALREADY in history AND that tool
    carries the answer in its args. That state is reachable — the
    guaranteed-terminal backstop submits on the model's behalf and then falls
    through to this same completion — so the exemption is kept, but conditioned
    on a submission having actually happened rather than on the shape of a tool
    that might never have been called. A BACKGROUND terminal whose result is in
    history is deliberately NOT exempt: its args carry no answer, so prose
    remains the only user-facing surface and the floor still applies.

    Cheap and side-effect free.
    """

    rc = engine.config.rc
    if not rc.finalize_prose_gate_enabled:
        return False
    if not run_has_final_answer(engine):
        return False
    terminal_tool = _resolved_terminal_tool_name(engine)
    if (
        terminal_tool is not None
        and _history_has_terminal_tool_result(engine)
        and _terminal_tool_carries_answer_field(engine, terminal_tool)
    ):
        return False
    if not _run_did_non_terminal_work(engine, terminal_tool or ""):
        return False
    if _has_visible_assistant_prose_after_work(
        engine, terminal_tool or "", rc.finalize_prose_gate_min_chars
    ):
        if _pointer_answer_repair_budget_spent(engine):
            return False
        return _pointer_answer_evidence(engine) is not None
    return not getattr(engine, "_finalize_prose_gate_used", False)


def _run_did_non_terminal_work(engine: QueryEngine, terminal_tool: str) -> bool:
    """True iff this run called at least one non-terminal tool.

    The floor is a length test, and a length test cannot by itself tell a reply
    that COLLAPSED from one that is correctly brief. What separates them is
    whether there was anything to report: a run that searched, delegated or
    wrote files and then answered in a few dozen characters has under-reported
    its own work, while a run that answered a greeting without touching a tool
    has reported everything it had. Without this condition the floor fires on
    the second case too, and the repair turn's only possible effect is to pad a
    correct short answer up to the threshold.

    Reads :func:`_this_run_messages` for the same reason
    :func:`_has_visible_assistant_prose_after_work` does: the obligation
    belongs to the work THIS run did, not to what an earlier run of the session
    left in history.
    """

    for message in _this_run_messages(engine):
        for block in message.content_blocks:
            if _is_non_terminal_tool_activity(block, terminal_tool):
                return True
    return False


def _parse_write_call(
    engine: QueryEngine, arguments_json: str
) -> tuple[str, int] | None:
    """The ``(path, content_chars)`` a write call carries, or None.

    Reads the call's OWN arguments, which is where the content the run produced
    actually is: the tool result reports bytes and outcomes, the arguments hold
    the text. Core deliberately does not go looking on disk for it — it has no
    workspace, no filesystem contract and no business acquiring one, and the
    history it already owns answers the question.

    Returns None for anything that is not a write of textual content to a named
    path: unparseable arguments (they crossed a provider boundary), a missing or
    empty ``content``, or no resolvable path. A write whose path cannot be read
    is dropped rather than guessed at, because the pointer test's other half is
    "the answer names THIS file" and there is nothing to name.
    """

    try:
        arguments = json.loads(arguments_json)
    except (TypeError, ValueError):
        return None
    if not isinstance(arguments, Mapping):
        return None
    content = arguments.get(CHUNKABLE_CONTENT_FIELD)
    if not isinstance(content, str) or not content:
        return None
    for key in argument_names(ToolArgumentSlot.path, roles=engine.config.tool_roles):
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip(), len(content)
    return None


def _run_written_content_by_path(engine: QueryEngine) -> dict[str, int]:
    """Characters of content THIS run wrote, accumulated per target file.

    Counts a write only once its result landed without error: a rejected write
    produced nothing, and treating it as a deliverable would let a failed run
    demand a longer answer about a file that does not exist. The tools that
    count are ``rc.terminal_tool_nudge_file_write_tool_names`` — the same tuple
    the write-first nudge already uses, so a tenant with differently named write
    tools configures them in one place and no the host tool name is hardcoded
    in core.

    Per PATH, not per call: a chunked deliverable arrives as one ``Write``
    followed by several ``AppendFile`` calls, and what the run produced is their
    sum. Two different files stay two entries — a run that writes a 12 KB report
    and a 200-byte scratch note has produced one document, not one 12.2 KB
    document, and the pointer test asks its question of each separately.

    Reads :func:`_this_run_messages`, so the prior-run turns cross-run history
    seeding prepends are excluded for the same reason they are excluded
    everywhere else here: the obligation belongs to the work THIS run did.
    Runtime-synthesised writes are NOT excluded — a salvage write puts real
    content in the file whoever emitted the call.
    """

    write_names = set(engine.config.rc.terminal_tool_nudge_file_write_tool_names)
    if not write_names:
        return {}
    attempted: dict[str, tuple[str, int]] = {}
    landed: set[str] = set()
    for message in _this_run_messages(engine):
        for block in message.content_blocks:
            if isinstance(block, ToolUseBlock):
                if _strip_tool_name_prefix(block.name) not in write_names:
                    continue
                parsed = _parse_write_call(engine, block.arguments_json)
                if parsed is not None:
                    attempted[block.tool_call_id] = parsed
            elif isinstance(block, ToolResultBlock) and not block.is_error:
                landed.add(block.tool_call_id)
    written: dict[str, int] = {}
    for call_id, (path, content_chars) in attempted.items():
        if call_id in landed:
            written[path] = written.get(path, 0) + content_chars
    return written


def _visible_answer_after_work(engine: QueryEngine, terminal_tool: str) -> str:
    """The visible prose this run delivered as its ANSWER, joined.

    The same window :func:`_has_visible_assistant_prose_after_work` already
    calls the answer — every visible assistant text block after the run's latest
    non-terminal tool activity — read for its content rather than tested against
    a floor. Sharing the definition matters: this module has already ruled that
    prose emitted BEFORE the last real work is progress narration and not an
    answer, and a second, wider reading of "what the user got" here would let
    that narration pay for a reply that says nothing.

    Runtime-synthesised assistant turns (:data:`SYNTHETIC_RECOVERY_METADATA_KEY`
    — the ``(empty)`` placeholder, the guaranteed-terminal scaffolding) are the
    runtime's own words, not delivered content, and are excluded exactly as
    :func:`_latest_durable_answer_text` excludes them. Their tool activity is
    still work: a synthetic write moves the window forward like any other.
    """

    answer_parts: list[str] = []
    for message in _this_run_messages(engine):
        scaffolding = bool(message.metadata.get(SYNTHETIC_RECOVERY_METADATA_KEY))
        partial_attempt = (
            message.metadata.get(PARTIAL_ASSISTANT_ATTEMPT_METADATA_KEY) is True
        )
        for block in message.content_blocks:
            if _is_non_terminal_tool_activity(block, terminal_tool):
                # Work landed — everything said before it was narration about
                # work in progress, not the report on it.
                answer_parts.clear()
                continue
            if (
                message.role is MessageRole.assistant
                and not scaffolding
                and not partial_attempt
                and isinstance(block, TextBlock)
                and block.text.strip()
            ):
                answer_parts.append(block.text.strip())
    return "\n".join(answer_parts)


def _answer_names_file(answer: str, path: str) -> bool:
    """True iff ``answer`` refers the reader to ``path``.

    A plain substring test on the full path or on the bare filename, over
    POSIX-normalised, case-folded text — the model writes the path back in
    whatever form it likes (backquoted, inside a sentence, with or without the
    workspace prefix it wrote through), and every one of those forms contains
    one of the two.

    Deliberately permissive, because it is not the discriminating half of the
    pointer test: an answer that happens to contain the word ``notes`` while a
    file named ``notes`` exists still has to be a small fraction of that file's
    content before anything fires. What this rules out is the run that wrote a
    document and then answered about something else entirely, where refusing the
    answer would be refusing it for a file it never claimed to be reporting.
    """

    haystack = answer.replace("\\", "/").casefold()
    needle = path.strip().replace("\\", "/").rstrip("/")
    while needle.startswith("./"):
        needle = needle[2:]
    needle = needle.casefold()
    if not needle:
        return False
    if needle in haystack:
        return True
    basename = needle.rsplit("/", 1)[-1]
    return bool(basename) and basename in haystack


def _pointer_answer_evidence(engine: QueryEngine) -> tuple[str, int, int] | None:
    """The file this run's answer merely points at: ``(path, answered, written)``.

    The failure a length floor cannot see. The model is asked for an article,
    writes 13 KB of one into a workspace file, and reports back: *"Готово.
    Статья … сохранена в ``workspace/article_fem_elasticity.md`` (13 031 байт,
    ~960 слов). Структура статьи: 1. Введение …"*. Measured on a live stand that
    is the outcome of roughly two runs in three, and the reply is 1 200-1 900
    characters long — so it clears any floor an operator would set, the
    substantive-answer machinery never fires, and the run is scored a success.
    For a user with a file browser it IS a success. For a chat user it is an
    empty reply about a file they cannot open.

    So the test is not "how long is the answer" but "how does the answer compare
    to what the run produced": a reply that names a file this run wrote and
    carries a small fraction of what went into it is a filing notice, whatever
    its absolute length. Both quantities are in the run's own history — the
    write call's arguments hold the content, the assistant text is the answer.

    Returns None (nothing fires) unless ALL hold:

      * ``rc.workspace_visible_to_user`` is False. Where the user can open the
        workspace, a path IS an answer and this whole mechanism is inert — which
        is the default, so no deployment inherits it by upgrading;
      * both knobs are live —
        ``finalize_prose_gate_pointer_max_answer_fraction`` > 0 and
        ``finalize_prose_gate_pointer_min_written_chars`` > 0. Either at zero
        disables the pointer test without touching the length floor;
      * some single file (:func:`_run_written_content_by_path`) received at
        least ``_min_written_chars`` of content across this run's successful
        writes. A scratch file, a config, a patch — anything below that floor —
        can be pointed at freely;
      * the answer (:func:`_visible_answer_after_work`) names that file, and
      * the answer is shorter than ``_max_answer_fraction`` of what was written
        into it.

    When several files qualify the largest is reported, so the DIAG names the
    deliverable rather than whichever entry happened to be first.

    WHAT THIS LETS THROUGH, on purpose. A real summary that also says where the
    file lives passes: at the default fifth, a 13 KB article answered with 3 KB
    of its substance is an answer, and the run keeps it. A run whose answer
    carries the content and ALSO wrote it to a file passes, because the answer
    is not small. A run that wrote nothing substantial passes however terse its
    reply — that is the plain length floor's business, not this one's. And the
    user who explicitly asked for a file still gets the file: nothing here
    touches the write, only the claim that mentioning it was an answer.

    The remaining false positive is a run that delivers the content as prose
    FIRST and only afterwards writes it to a file and files a notice about it —
    the answer window then holds just the notice. It costs that run one repair
    turn, in exchange for the two-in-three failure this catches, and the repair
    text asks for a summary rather than a re-paste, so the second reply is not a
    duplicate of the first.

    Pure / total — reads history and RC, never the filesystem.
    """

    rc = engine.config.rc
    if rc.workspace_visible_to_user:
        return None
    max_fraction = rc.finalize_prose_gate_pointer_max_answer_fraction
    min_written = rc.finalize_prose_gate_pointer_min_written_chars
    if max_fraction <= 0.0 or min_written <= 0:
        return None
    written = _run_written_content_by_path(engine)
    if not written:
        return None
    answer = _visible_answer_after_work(
        engine, _resolved_terminal_tool_name(engine) or ""
    )
    if not answer:
        return None
    answer_chars = len(answer)
    evidence: tuple[str, int, int] | None = None
    for path, written_chars in written.items():
        if written_chars < min_written:
            continue
        if answer_chars >= max_fraction * written_chars:
            continue
        if not _answer_names_file(answer, path):
            continue
        if evidence is None or written_chars > evidence[2]:
            evidence = (path, answer_chars, written_chars)
    return evidence


def _pointer_answer_repair_budget_spent(engine: QueryEngine) -> bool:
    """True when this run may not spend another turn on the pointer refusal.

    The bound is on the run and never resets. Its neighbour
    :mod:`protocore.runtime.pending_reads` resets its counter on a productive
    read, because there "productive" is a fact: the declared file was opened or
    it was not. Here there is no such fact to reset on — the pointer test IS the
    definition of progress, and while it still says "pointer" the mechanism has,
    by its own reading, achieved nothing. A counter that reset on partial
    progress would let a model that grows its filing notice by fifty characters
    a turn hold the budget open indefinitely: an unbounded loop reached by a
    strictly improving sequence, which is the one outcome worse than the failure
    being repaired.

    Cheap (one comparison) and side-effect free, so the predicates that call it
    stay cheap; the warning that says the budget ran out lives in
    :func:`_release_pointer_answer_repair`, which is a separate call precisely
    so that this one can be asked freely.

    A budget of 0 makes this True from the start, which switches the pointer
    refusal off while leaving the plain length floor untouched — a third kill
    switch alongside the two knobs :func:`_pointer_answer_evidence` already
    honours.
    """

    return (
        engine._pointer_answer_repair_attempts
        >= engine.config.rc.finalize_prose_gate_pointer_max_repair_attempts
    )


def _charge_pointer_answer_repair(engine: QueryEngine) -> int:
    """Spend one of the pointer refusal's attempts; returns the new count.

    Charged when a repair turn is actually INJECTED, at either seam, and never
    otherwise — a turn the refusal wanted and could not take (an empty repair
    text degrades the whole gate to a no-op) costs nothing, exactly as an
    unforceable read costs :func:`pending_reads.charge_forced_attempt` nothing.

    ON A REPAIR THAT PARTLY WORKS — a second answer that grows but is still
    principally a pointer. It consumes an attempt, like any other. What is
    bounded here is the number of times the runtime asks, not the number of
    times it is refused, and the cost being bounded is paid on asking: a full
    answer regeneration, appended to the context the next answer is written
    from. Crediting partial progress back would also require a threshold for
    "enough progress" that nothing in the run can supply — the only measure
    available is the pointer test itself, which is still returning "pointer".

    Nothing is lost by that strictness, because partial progress is already
    banked where it counts. :func:`_visible_answer_after_work` reads the whole
    window since the last real work, so a second reply ACCUMULATES on to the
    first rather than replacing it: a model that adds real substance each turn
    climbs toward ``finalize_prose_gate_pointer_max_answer_fraction`` and the
    refusal stops firing of its own accord, budget unspent. The run that burns
    all three attempts is the run that added nothing across all three.
    """

    engine._pointer_answer_repair_attempts += 1
    return engine._pointer_answer_repair_attempts


def _release_pointer_answer_repair(engine: QueryEngine) -> None:
    """Say, once, that a run is finishing on an answer the refusal rejected.

    Called where the gate declined to fire, so it reports the outcome rather
    than predicting it: reaching here with the budget spent AND the evidence
    still standing means the last repair turn did not work either and the user
    is about to receive a filing notice. That is the fact an operator needs in
    order to decide whether the budget is too small or the mechanism is not
    worth its turns, and it is exactly the fact a line emitted at charge time
    could not carry — at that moment the last attempt has not been answered yet.

    The distinction it draws is the point. A run that spent every attempt and
    then succeeded ends silent here, because the evidence is gone; only the
    failure speaks. The latch keeps it to one line per run, so a resumed or
    multi-dispatch run cannot turn one outcome into a stream of them.

    Cheap on the paths that call it: a run whose pointer refusal never engaged
    returns on the first comparison, which is what lets this sit on the
    per-dispatch path without being felt.
    """

    if engine._pointer_answer_repair_attempts == 0:
        return
    if engine._pointer_answer_repair_released:
        return
    if not _pointer_answer_repair_budget_spent(engine):
        return
    evidence = _pointer_answer_evidence(engine)
    if evidence is None:
        return
    path, answer_chars, written_chars = evidence
    engine._pointer_answer_repair_released = True
    _logger.warning(
        "DIAG query.finalize_prose_gate.pointer_answer_budget_spent "
        "run=%s tenant=%s turn=%s attempts=%d/%d answer_chars=%d "
        "written_chars=%d path=%s",
        engine.config.run_id,
        engine.config.tenant_id,
        engine.turn_id(),
        engine._pointer_answer_repair_attempts,
        engine.config.rc.finalize_prose_gate_pointer_max_repair_attempts,
        answer_chars,
        written_chars,
        path,
    )


def _append_answer_floor_repair_turn(engine: QueryEngine) -> None:
    """Append the ONE bounded repair turn the plain-stop answer floor grants.

    The same synthetic user turn the dispatch-path veto appends, carrying the
    same :data:`SYNTHETIC_RECOVERY_PROSE_GATE_REPAIR` marker so every consumer
    that already strips runtime scaffolding out of durable history, the memory
    fold and ``result_preview`` keeps doing so with no change.

    No error ``tool_result`` precedes it here. The terminal path has a vetoed
    dispatch to answer for; this path has nothing to veto — the model stopped of
    its own accord. The tail of history is therefore either this turn's
    assistant message or, when the stop carried no content at all, the last
    tool result, and a user turn is a valid successor to both (the dispatch-path
    veto appends exactly that after its own tool result).
    """

    engine.history.append(
        Message(
            role=MessageRole.user,
            content_blocks=[
                TextBlock(text=engine.prompt_text("finalize_prose_gate_repair"))
            ],
            metadata={
                SYNTHETIC_RECOVERY_METADATA_KEY: (
                    SYNTHETIC_RECOVERY_PROSE_GATE_REPAIR
                )
            },
        )
    )


def _count_terminal_answer_cited_refs(tool_call: ToolCall) -> int:
    """Count cited refs on an un-submitted terminal call.

    PURE / total — reads the canonical ``refs`` slot off ``tool_call.arguments``
    (falling back to the legacy ``sources`` alias only when ``refs`` is
    absent/empty), counting non-empty string entries. Returns 0 for any
    unexpected shape; never raises. Used only for the heartbeat DIAG —
    no behavioural effect.
    """

    arguments = tool_call.arguments
    if not isinstance(arguments, Mapping):
        return 0
    for key in (_TERMINAL_ANSWER_REFS_KEY, _TERMINAL_ANSWER_REFS_LEGACY_ALIAS):
        raw = arguments.get(key)
        if isinstance(raw, list):
            count = sum(
                1 for item in raw if isinstance(item, str) and item.strip()
            )
            if count:
                return count
    return 0


def _pre_dispatch_terminal_verify_applies(
    engine: QueryEngine, tool_call: ToolCall
) -> bool:
    """Gate for the PRE-DISPATCH terminal-tool veto.

    Returns True iff the pre-dispatch verify seam should consult its
    host-supplied trigger for ``tool_call``. ALL must hold:

      * ``rc.pre_dispatch_terminal_verify_enabled`` is True,
      * a per-tenant terminal tool is declared
        (``config.expected_terminal_tool``) AND ``tool_call`` IS that tool —
        the seam never intercepts a non-terminal tool, so reads/writes/exec
        dispatch unaffected,
      * the durable per-run latch ``_pre_dispatch_terminal_verify_used`` is
        unset (fire-at-most-once across resume), and
      * the shared corrective-turn budget
        (``rc.pre_terminal_self_verify_max_extra_turns`` vs
        ``_self_verify_extra_turns_used``) is not exhausted,
      * a host trigger is actually wired.

    Returns False (no interception) otherwise.
    Cheap and side-effect-free so ``_dispatch_tool`` can call it on every
    dispatch without the per-call trigger cost when the gate is closed.
    """

    rc = engine.config.rc
    if not getattr(rc, "pre_dispatch_terminal_verify_enabled", False):
        return False
    expected = engine.config.expected_terminal_tool
    if expected is None or tool_call.name != expected:
        return False
    if getattr(engine, "_pre_dispatch_terminal_verify_used", False):
        return False
    if (
        getattr(engine, "_self_verify_extra_turns_used", 0)
        >= rc.pre_terminal_self_verify_max_extra_turns
    ):
        return False
    return engine.config.pre_dispatch_terminal_verify_trigger is not None


def _resolve_pre_dispatch_terminal_veto(
    engine: QueryEngine, tool_call: ToolCall
) -> str | None:
    """Consult the pre-dispatch trigger; return the corrective message if the
    terminal dispatch must be VETOED, else ``None``.

    Assumes :func:`_pre_dispatch_terminal_verify_applies` already returned
    True. Invokes the host-supplied predicate with the engine + the
    UN-SUBMITTED ``ToolCall`` (so the predicate can inspect the answer the
    model is about to send against the observed-ref ledger). A trigger
    exception is swallowed (never break dispatch) and treated as "no veto".
    The caller owns the latch/counter mutation + history injection so this
    stays a pure read.
    """

    trigger = engine.config.pre_dispatch_terminal_verify_trigger
    if trigger is None:  # pragma: no cover - guarded by _applies
        return None
    try:
        corrective = trigger(engine, tool_call)
    except Exception as exc:  # pragma: no cover - defensive; never break dispatch
        _logger.warning(
            "DIAG query.pre_dispatch_terminal_verify.trigger_failed "
            "run=%s error=%s",
            engine.config.run_id,
            exc,
        )
        return None
    if not corrective:
        return None
    return corrective


def _append_terminal_tool_nudge(engine: QueryEngine) -> None:
    """Append the contract-repair nudge message + arm the terminal-only latch."""

    engine.history.append(
        Message(
            role=MessageRole.user,
            content_blocks=[TextBlock(text=_resolved_terminal_tool_nudge_text(engine))],
            metadata={
                SYNTHETIC_RECOVERY_METADATA_KEY: (
                    SYNTHETIC_RECOVERY_TERMINAL_TOOL_NUDGE
                )
            },
        )
    )
    # Flip the terminal-only guard latch once the nudge has been emitted so
    # subsequent dispatch calls reject every non-terminal tool with a
    # structured error pointing at finalisation. The latch persists for the
    # remainder of the run; it is implicitly cleared when
    # ``_history_has_terminal_tool_result`` returns True (the predicate
    # consulted at dispatch time).
    engine._terminal_only_active = True


def _truncated_call_state_path(engine: QueryEngine, tool_call: ToolCall) -> str | None:
    """The REAL file-target of a truncated mutation call, or ``None``.

    The state-only twin of :func:`_truncated_call_path`: it NEVER returns the
    ``"the target file"`` display placeholder. Only this resolved path may be
    fed into convergence state (``note_truncated_mutation`` / active-path
    handoff) — latching the placeholder would poison ``_longfile_active_path``
    so a later real Write to the actual file is ignored as off-path.
    """
    return string_argument(
        tool_call.arguments, ToolArgumentSlot.path, roles=engine.config.tool_roles
    )


def _truncated_call_path(engine: QueryEngine, tool_call: ToolCall) -> str:
    """Best-effort extract the file-target of a truncated mutation call (display).

    The chunk-recovery message names the file path so the model resumes the
    SAME file. Even a truncated call usually preserves its
    target (``path`` / ``file_path`` is the first, small field; the output cap
    cut the large ``content`` after it — exactly the prod ``{"path": ...}``
    shape). Falls back to a generic placeholder when no target is recoverable
    (the cut landed before it, or the call surfaced as a raw envelope). This is
    the MODEL-VISIBLE display path; convergence STATE must use
    :func:`_truncated_call_state_path` (which never returns the placeholder).
    """
    return _truncated_call_state_path(engine, tool_call) or "the target file"


def _truncated_call_paths(engine: QueryEngine, tool_calls: list[ToolCall]) -> list[str]:
    """Ordered, de-duplicated list of recoverable paths for telemetry."""
    seen: list[str] = []
    for tc in tool_calls:
        path = _truncated_call_path(engine, tc)
        if path not in seen:
            seen.append(path)
    return seen


def _is_content_mutation_truncation(engine: QueryEngine, tool_call: ToolCall) -> bool:
    """True iff a truncated call is a chunkable file-content mutation.

 The Write->AppendFile->FinalizeFile chunk recovery only makes sense for
 a KNOWN chunkable content-mutation tool. Delegates to the ONE
 shared predicate
 (:func:`protocore.contracts.tool_chunking.is_chunkable_content_mutation`)
 that the host LLM client uses too, so both layers route identically:
 the call's tool must REQUIRE ``content`` (the body the cap cut) AND be
 explicitly flagged (``ToolParameterSchema.chunkable_content_mutation``) OR
 declared by the host as byte-producing. The call must also
 be missing ``content`` (the cut-body shape). This EXCLUDES a tool like
 ``Read`` (no ``content``) and — the fix — a dynamic/tenant tool that
 merely declares a ``content`` field without opting in: such a call gets the
 generic "re-issue with complete arguments" resume instead. Schema lookup uses
 the engine's tool registry (in-process; no boundary violation). An unknown
 tool is treated as non-content (safe default — generic resume).
 """
    args = tool_call.arguments
    if not isinstance(args, dict):
        return False
    if "content" in args:
        # ``content`` already present → not the cut-content shape.
        return False
    required, chunkable_flag = _tool_content_schema(engine, tool_call.name)
    return is_chunkable_content_mutation(
        tool_name=tool_call.name,
        required=required,
        chunkable_flag=chunkable_flag,
        roles=engine.config.tool_roles,
    )


def _tool_content_schema(
    engine: QueryEngine, tool_name: str
) -> tuple[list[str], bool | None]:
    """Best-effort ``(required, chunkable_content_mutation)`` for a tool.

    Reads the registered tool's ``definition.parameters`` (the same typed
    schema the host client indexes). Returns ``([], None)`` when the tool
    is unknown or its schema cannot be read — the shared predicate then treats
    it as non-content (generic resume).
    """
    if not tool_name:
        return ([], None)
    tool = engine.tools.get(tool_name)
    if tool is None:
        return ([], None)
    try:
        params = tool.definition.parameters
        required = list(getattr(params, "required", []) or [])
        chunkable_flag = getattr(params, "chunkable_content_mutation", None)
    except AttributeError:
        return ([], None)
    return (required, chunkable_flag if isinstance(chunkable_flag, bool) else None)


def _salvage_truncated_content(tool_call: ToolCall) -> str:
    """The recoverable partial ``content`` body of a truncated chunkable write.

    Mirrors the stand's ``case_e._salvage_partial`` intent
    against the PROD shape. In prod the host SSE parser already stage-4
    brace-balances the cut args and ``json.loads``-es them, so a truncated Write
    whose body was cut MID-string surfaces here as ``arguments['content']`` — an
    already-VALID (just early-terminated) Python ``str``. No raw-JSON trim is
    needed (that belongs at the raw layer; ``arguments`` never carries raw text —
    Returns that string, or ``''`` when ``content`` is absent / not a
    non-empty str (the "cut before any body" shape — nothing to salvage; the
    caller then steers a smaller first chunk instead of writing an empty
    file). Only a VALID partial is ever returned — no corrupt content is
    dispatched.
    """
    args = tool_call.arguments
    if not isinstance(args, dict):
        return ""
    content = args.get("content")
    if isinstance(content, str) and content:
        return content
    return ""


def _build_truncation_chunk_recovery_text(
    engine: QueryEngine, truncated_tool_calls: list[ToolCall]
) -> str:
    """Build the recovery message for one or more truncated tool calls.

 A chunkable file-content mutation (``path`` present, ``content`` cut) gets
 the structured Write->AppendFile->FinalizeFile message naming the PATH + the
 per-call ``write_chunk_token_budget`` (the 0/4 -> 4/4 chunking protocol).
 Any OTHER truncated call (e.g. a non-content tool whose args were cut) gets
 the generic ``tool_call_truncation_resume`` template so it is not misrouted
 into an irrelevant file-chunk workflow. Both EN and RU
 halves are emitted for the chunk message (EN first), per the
 multilingual rule.

 the "continue with AppendFile" directive is emitted ONLY when a
 chunk has ACTUALLY been written for this path (``path in
 engine._mid_chunked_write_paths``, populated by a SUCCESSFUL Write/AppendFile
 dispatch). A repeat truncation BEFORE any chunk landed keeps the
 first-message ``Write(header)`` protocol (never tells the model to
 ``AppendFile`` a non-existent file) but LOWERS the header budget per prior
 no-success prompt for that path (so the next header attempt is smaller than
 the one that just truncated). This function does NOT add to
 ``_mid_chunked_write_paths`` — only a real successful write does (see
 :func:`_record_chunk_write_success`).
 """
    rc = engine.config.rc
    sections: list[str] = []
    for tc in truncated_tool_calls:
        if not _is_content_mutation_truncation(engine, tc):
            # Non-content truncation — generic resume, no file-chunk protocol.
            sections.append(
                engine.prompt_text("tool_call_truncation_resume", tool_name=tc.name)
            )
            continue
        path = _truncated_call_path(engine, tc)
        already_chunking = path in engine._mid_chunked_write_paths
        if already_chunking:
            # A chunk has really been written → steer to AppendFile, full budget.
            chunk_budget = rc.write_chunk_token_budget
            mid_note_en = rc.truncation_chunk_recovery_mid_chunked_note_en.format(
                path=path
            )
            mid_note_ru = rc.truncation_chunk_recovery_mid_chunked_note_ru.format(
                path=path
            )
        else:
            # No chunk written yet → keep Write(header); LOWER the header budget
            # by one division step per prior no-success prompt for this path.
            prior_prompts = engine._truncation_recovery_prompt_counts.get(path, 0)
            chunk_budget = _lowered_header_budget(rc, prior_prompts)
            engine._truncation_recovery_prompt_counts[path] = prior_prompts + 1
            mid_note_en = ""
            mid_note_ru = ""
        en = rc.truncation_chunk_recovery_message_en.format(
            tool_name=tc.name,
            path=path,
            chunk_budget_tokens=chunk_budget,
            mid_chunked_note_en=mid_note_en,
        )
        ru = rc.truncation_chunk_recovery_message_ru.format(
            tool_name=tc.name,
            path=path,
            chunk_budget_tokens=chunk_budget,
            mid_chunked_note_ru=mid_note_ru,
        )
        sections.append(f"{en}\n\n{ru}")
    return "\n\n".join(sections)


def _lowered_header_budget(rc: LoopConstants, prior_prompts: int) -> int:
    """Header chunk budget for a repeat truncation before any chunk was written.

 Integer-divide ``write_chunk_token_budget`` by
 ``truncation_chunk_recovery_repeat_budget_divisor`` once per prior no-success
 recovery prompt for the path, floored at
 ``truncation_chunk_recovery_min_chunk_token_budget`` (itself clamped to the
 base budget). ``prior_prompts == 0`` → the full base budget (first
 truncation). A divisor of 1 (or a min == base) yields a constant budget.
 """
    base = rc.write_chunk_token_budget
    floor = min(rc.truncation_chunk_recovery_min_chunk_token_budget, base)
    divisor = rc.truncation_chunk_recovery_repeat_budget_divisor
    budget = base
    for _ in range(max(0, prior_prompts)):
        if divisor > 1:
            budget //= divisor
        if budget <= floor:
            return floor
    return max(budget, floor)


def _record_chunk_write_success(engine: QueryEngine, tool_call: ToolCall) -> None:
    """Mark a path as "chunking started" after a SUCCESSFUL chunkable write.

 Called from the successful-dispatch path for a Write/AppendFile that
 carries a ``content`` body and a resolvable target
 path. Adds the path to ``engine._mid_chunked_write_paths`` (so a later repeat
 truncation of that path gets the "continue with AppendFile" directive) and
 clears its no-success prompt count. A non-chunkable tool, or a write whose
 target cannot be resolved, is a no-op.
 """
    name = tool_call.name
    if name not in chunkable_content_mutation_names(engine.config.tool_roles):
        # Only a host tool declared as byte-producing advances the "chunking
        # started" state; a per-tenant flagged tool drives recovery wording but
        # its append-resume semantics are tool-specific, so it is not tracked.
        return
    args = tool_call.arguments
    if not isinstance(args, dict) or "content" not in args:
        return
    path = _truncated_call_path(engine, tool_call)
    if path == "the target file":
        return
    engine._mid_chunked_write_paths.add(path)
    engine._truncation_recovery_prompt_counts.pop(path, None)


async def _salvage_truncated_write_to_disk(
    engine: QueryEngine,
    tool_call: ToolCall,
    salvaged_content: str,
) -> AsyncIterator[TurnEvent]:
    """land a truncated write's recovered partial on disk.

 The stand-validated salvage path ported to prod: the model's
 truncated Write/AppendFile carried a partial ``content`` body that the SSE
 parser recovered; this dispatches that VALID partial as a CLEAN synthetic
 write so bytes land on disk + are byte-tracked, after which the truncation-
 gated convergence driver can engage on the genuine target (long-en-004's
 stand 3/3). Without this prod discards the partial → 0 bytes land → the
 driver stays inert on the real file.

 Pairing-safe + snapshot-safe:
 * a NEW synthetic ``ToolCall`` (fresh id) is built with the resolved path
 + salvaged content; its tool name is remembered;
 * a MATCHING assistant ``ToolUseBlock`` is appended FIRST (flagged
 synthetic-recovery scaffold) so ``_dispatch_tool`` — which appends ONLY
 the tool RESULT — never produces an orphan result;
 * ``note_truncated_mutation`` runs BEFORE dispatch so the STICKY
 truncation latch + active-path handoff are set, and ``_dispatch_tool``'s
 post-dispatch ``observe_tool_result`` (which lands the bytes + clears the
 transient ``_longfile_last_mutation_truncated``) and snapshot persist
 capture the latch atomically with the bytes;
 * the salvage REUSES the model's OWN tool name (``tool_call.name``), which
 the caller's classifier already constrains to a built-in chunkable write
 (``Write`` / ``AppendFile``). The model's declared op IS the write
 semantics: a ``Write`` is a from-scratch REPLACE (so re-emitting a full
 truncated ``Write`` after a prior chunk landed overwrites, NOT
 duplicates, the prefix — the documented Write-spiral shape) and an
 ``AppendFile`` is a CONTINUE (so salvaging it as a fresh ``Write`` would
 destroy prior content created via Bash/sandbox or under a different
 active path). Inferring the op from the persisted active-file size
 (``>0 → AppendFile``) instead ignored that name and corrupted both shapes;
 * dispatched ``preapproved=True`` (runtime-internal salvage of the model's
 OWN content — never a new approval surface).

 Yields the dispatch events. The caller treats it as ONE recovery round
 (the shared ``_max_output_recovery_count`` is already debited by the branch).
 """
    state_path = _truncated_call_state_path(engine, tool_call)
    if state_path is None:
        # Defensive: the caller's classifier already requires a resolvable path,
        # so this never fires for a salvage_job — but never dispatch without one.
        return
    # Reuse the model's OWN tool name (the caller already constrains it to the
    # built-in chunkable allowlist — ``Write`` / ``AppendFile``). The declared op
    # is the authoritative write semantics: a ``Write`` REPLACES from scratch (so
    # a re-emitted full truncated Write overwrites — never duplicates — a prefix
    # that landed in a prior round) and an ``AppendFile`` CONTINUES (so it is
    # never rewritten into a whole-file Write that would wipe content created via
    # Bash/sandbox or under a different active path). Inferring from the persisted
    # active-file size corrupted both shapes.
    salvage_name = tool_call.name
    # A per-run monotonic counter makes the synthetic id UNIQUE across
    # multiple salvages in one run. ``turn_id()`` advances per
    # assistant-message round, not per salvage, so two salvages in one round
    # share a ``turn_id`` and deriving from it alone would collide → the
    # outbound pairing repair would drop the later duplicate. Snapshot-persisted.
    engine._longfile_salvage_seq += 1
    synthetic_id = (
        f"toolu-longfile-salvage-{engine.config.run_id}-{engine._longfile_salvage_seq}"
    )
    salvage_call = ToolCall(
        id=synthetic_id,
        name=salvage_name,
        arguments={"path": state_path, "content": salvaged_content},
    )
    engine.remember_tool_name(synthetic_id, salvage_name)
    # Append the matching assistant tool_use FIRST (scaffold-flagged) so the
    # tool result ``_dispatch_tool`` appends is paired in durable history.
    engine.history.append(
        Message(
            role=MessageRole.assistant,
            content_blocks=[
                ToolUseBlock(
                    tool_call_id=synthetic_id,
                    name=salvage_name,
                    arguments_json=json.dumps(
                        salvage_call.arguments, ensure_ascii=False
                    ),
                )
            ],
            metadata={
                SYNTHETIC_RECOVERY_METADATA_KEY: SYNTHETIC_RECOVERY_LONGFILE_SALVAGE
            },
        )
    )
    # Set the truncation latch + active-path handoff BEFORE dispatch so the
    # snapshot ``_dispatch_tool`` persists carries it atomically with the bytes.
    _longfile.note_truncated_mutation(engine, state_path)
    # Also pre-set the transient truncated-tail
    # flag BEFORE the dispatch (and pass ``keep_truncated_tail=True`` to
    # ``_dispatch_tool`` so the post-``observe_tool_result`` clear is
    # suppressed for this synthetic-recovery call). The in-``_dispatch_tool``
    # persist now captures the flag as set, atomically with the recovered
    # bytes — closing the persist-then-re-assert window that previously let a
    # pod kill between the two persists resume with a salvaged >= floor
    # half-file evaluating ``plausibly_complete=True`` and forcing a
    # PREMATURE FinalizeFile on the next stall.
    engine._longfile_last_mutation_truncated = True
    yield _emit_state_change(
        engine,
        engine.state,
        engine.state,
        reason="longfile_truncation_salvaged_write",
    )
    async for evt in _dispatch_tool(
        engine,
        salvage_call,
        preapproved=True,
        synthetic_recovery=True,
        synthetic_recovery_kind=SYNTHETIC_RECOVERY_LONGFILE_SALVAGE,
        keep_truncated_tail=True,
    ):
        yield evt
    # Re-assert the truncated-tail flag AFTER the dispatch. The
    # synthetic Write lands the recovered FIRST chunk and ``_dispatch_tool``'s
    # post-dispatch ``observe_tool_result`` would normally CLEAR the transient
    # ``_longfile_last_mutation_truncated`` on a successful byte-landing (it
    # reads the landing as a clean write). The ``keep_truncated_tail=True``
    # above suppresses that clear, so this final re-assert is a belt-and-
    # braces backstop (covers any future observe path that forgets the flag)
    # and keeps the persist in lock-step with the in-dispatch persist.
    # Re-asserting here keeps ``plausibly_complete`` False so the driver forces
    # AppendFile (continue), not FinalizeFile (seal). This does NOT wedge
    # finalization: a later GENUINE clean (non-truncated) AppendFile clears the
    # flag again via ``observe_tool_result`` → the finalize path is reachable.
    # Placed AFTER the dispatch loop (the LAST mutation of the flag this turn) and
    # snapshot-persisted so a cross-pod resume sees the re-set, not the cleared
    # value the in-dispatch persist captured. The sticky engage gate
    # (``_longfile_truncated_paths``, set by ``note_truncated_mutation`` above) is
    # already correct; this only fixes the append-vs-finalize decision.
    engine._longfile_last_mutation_truncated = True
    await engine._persist_snapshot()


async def _maybe_seal_longfile_at_voluntary_finish(
    engine: QueryEngine,
) -> AsyncIterator[TurnEvent]:
    """seal a truncation-gated unsealed file at a
    VOLUNTARY run completion seam.

    The max-turns terminal seal (``terminal_seal_required`` exhaustion block)
    covers only budget exhaustion. A run can ALSO complete VOLUNTARILY — the
    model calls the run-terminal tool, or finishes with a prose ``end_turn`` —
    leaving a truncation-gated file unsealed (after a 004-shape recovery the
    model appends chunks then ends the run without FinalizeFile). This helper is
    called at both voluntary seams BEFORE the run completes; when
    :func:`longfile_convergence.terminal_seal_required` is True it dispatches a
    SYNTHETIC ``FinalizeFile`` for ``engine._longfile_active_path`` so the file
    is sealed and ``_longfile_finalized`` flips to True.

    Mirrors the salvage synthetic-dispatch machinery
    (:func:`_salvage_truncated_write_to_disk`): a matching assistant
    ``ToolUseBlock`` is appended FIRST (scaffold-flagged) so ``_dispatch_tool``
    — which appends only the tool RESULT — never produces an orphan; the call is
    dispatched ``preapproved=True`` (a runtime-internal deterministic seal of a
    known path, never a new approval surface); and the post-dispatch
    ``observe_tool_result`` feeds the engine so ``_longfile_finalized`` flips.
    NO LLM call, NO extra turn.

    Zero-collateral: ``terminal_seal_required`` already gates on the RC
    kill-switch, the truncation latch (never fires for a non-truncated
    file), not-already-finalized, the HARD empty/below-floor finalize guard, and
    the forced-finalize budget — so an ordinary run is bit-identical. The
    one-shot ``_longfile_voluntary_seal_used`` latch caps it to ONCE per run, and
    ``commit_forced_finalize`` charges the seal against the finalize budget.

    The sealing tool must be on the run's tool surface for dispatch — if the
    host named none, or the tool is not in the registry, the seal
    is skipped SILENTLY (no crash): a tenant without the chunk-protocol tools
    simply cannot be sealed this way, and that is correct (the run completes as
    before). Yields any dispatch events; no-op (yields nothing) when not eligible.
    """
    if engine._longfile_voluntary_seal_used:
        return
    if not _longfile.terminal_seal_required(engine):
        return
    # Once the STRICT terminal-only latch is in force (deadline reached,
    # ``expected_terminal_tool`` configured) the only allowed dispatch is
    # the resolved terminal tool. ``FinalizeFile`` is NOT the
    # expected terminal tool, so attempting a seal would short-circuit
    # through ``_dispatch_tool`` → ``_terminal_only_blocks`` with a
    # ``terminal_only`` is_error result: forced budget charged
    # (``commit_forced_finalize`` below), one-shot
    # ``_longfile_voluntary_seal_used`` latch consumed, the synthetic
    # assistant tool_use is in history, AND the file is left unsealed —
    # every seal-related side effect fires except the seal itself. Skip
    # cleanly so the deadline path can drive the model to its
    # ``expected_terminal_tool`` and complete.
    if _terminal_only_enforced(engine):
        _logger.warning(
            "DIAG query.longfile_seal.skipped_terminal_only run=%s tenant=%s "
            "path=%s",
            engine.config.run_id,
            engine.config.tenant_id,
            engine._longfile_active_path,
        )
        return
    path = engine._longfile_active_path
    if not path:
        # Defensive: ``terminal_seal_required`` implies a truncation-gated active
        # path, but never dispatch FinalizeFile without a concrete target.
        return
    # The sealing tool must be advertised on this run's tool surface to
    # dispatch. A run whose tenant has no chunk-protocol tools simply cannot be
    # sealed — skip (the run completes as it would have without the seal).
    seal_tool_name = _longfile.sealing_tool_name(engine)
    if seal_tool_name is None or engine.tools.get(seal_tool_name) is None:
        return

    engine._longfile_voluntary_seal_used = True
    engine._longfile_salvage_seq += 1
    synthetic_id = (
        f"toolu-longfile-seal-{engine.config.run_id}-{engine._longfile_salvage_seq}"
    )
    seal_call = ToolCall(
        id=synthetic_id,
        name=seal_tool_name,
        arguments={"path": path},
    )
    engine.remember_tool_name(synthetic_id, seal_tool_name)
    # Append the matching assistant tool_use FIRST (scaffold-flagged) so the
    # tool result ``_dispatch_tool`` appends is paired in durable history.
    engine.history.append(
        Message(
            role=MessageRole.assistant,
            content_blocks=[
                ToolUseBlock(
                    tool_call_id=synthetic_id,
                    name=seal_tool_name,
                    arguments_json=json.dumps(seal_call.arguments, ensure_ascii=False),
                )
            ],
            metadata={
                SYNTHETIC_RECOVERY_METADATA_KEY: (
                    SYNTHETIC_RECOVERY_LONGFILE_TERMINAL_SEAL
                )
            },
        )
    )
    # Charge the seal against the forced-finalize budget so a model that already
    # exhausted it (e.g. via the stall-driver) does not double-seal — respected
    # alongside the one-shot latch above (``terminal_seal_required`` already
    # checked budget remaining; this keeps the accounting honest).
    _longfile.commit_forced_finalize(engine)
    yield _emit_state_change(
        engine,
        engine.state,
        engine.state,
        reason="longfile_terminal_seal_voluntary_finish",
    )
    async for evt in _dispatch_tool(
        engine,
        seal_call,
        preapproved=True,
        synthetic_recovery=True,
        synthetic_recovery_kind=SYNTHETIC_RECOVERY_LONGFILE_TERMINAL_SEAL,
    ):
        yield evt
    # ``_dispatch_tool``'s post-dispatch ``observe_tool_result`` flips
    # ``_longfile_finalized=True`` on the successful FinalizeFile result + the
    # in-dispatch snapshot persists it. Persist once more so the
    # ``_longfile_voluntary_seal_used`` latch (set before the dispatch) is
    # durable for a cross-pod resume even if the dispatch result was a no-op.
    await engine._persist_snapshot()


async def _maybe_drive_longfile_convergence(
    engine: QueryEngine,
) -> AsyncIterator[TurnEvent | bool]:
    """the end-of-turn convergence step (-).

 Called ONCE at every completed-assistant-turn boundary (both the
 tool-call-turn-end seam and the prose/no-tool-turn seam). It:

 1. advances the stall clock for the just-completed turn
 (:func:`longfile_convergence.register_completed_turn` — a turn that added
 no bytes increments ``turns_since_last_byte_adding_mutation``);
 2. asks the pure decision module whether to FORCE the next tool
 (:func:`longfile_convergence.decide_next_forced_tool`);
 3. if a forced tool is decided, injects the INCOMPLETE continue
 message (with the on-disk-tail anchor), records the forced ``tool_choice``
 for the next stream, charges the per-kind forced-round budget, persists
 the snapshot (cross-pod safe), and emits a ``state_changed`` event.

 Yields any emitted :class:`TurnEvent` then a FINAL ``bool`` sentinel: True
 iff a forced action was issued (the caller must ``continue`` the outer loop
 to open the forced stream). No-op (yields ``False``) when the driver is
 disabled, no stall/plateau is detected, or all forced-round budgets are
 spent — so the happy path and disabled-RC path are bit-identical to pre-FEAT.
 """
    # The longfile convergence driver forces the next assistant stream's
    # ``tool_choice`` to ``AppendFile``/``FinalizeFile``, neither of which is
    # the resolved ``expected_terminal_tool``. Once the STRICT
    # terminal-only latch is in force (deadline reached, ``expected_terminal_tool``
    # configured) the only allowed dispatch is that terminal tool, so the
    # forced tool_choice would surface a ``terminal_only`` is_error in the
    # NEXT iteration's ``_dispatch_tool`` call — forced budget charged
    # (``commit_forced_append`` / ``commit_forced_finalize``), INCOMPLETE
    # continue message appended, snapshot persisted, and a wasted LLM turn
    # burns a stream read. Bail BEFORE the budget/continue/persist side
    # effects so the deadline path can drive the model to its
    # ``expected_terminal_tool`` and complete. Stall clock still advances
    # below so the bookkeeping is honest.
    if _terminal_only_enforced(engine):
        _longfile.register_completed_turn(engine)
        _logger.warning(
            "DIAG query.longfile_convergence.skipped_terminal_only run=%s "
            "tenant=%s",
            engine.config.run_id,
            engine.config.tenant_id,
        )
        yield False
        return
    _longfile.register_completed_turn(engine)
    forced = _longfile.decide_next_forced_tool(engine)
    if forced is None:
        yield False
        return
    forced_name = _longfile.forced_tool_name(engine, forced)
    if forced_name is None:
        # The decision stands but this run has no single tool to carry it: the
        # host declared none for the role, or declared two. Charge nothing and
        # inject nothing — a continue message whose forced choice cannot be set
        # is the drop the forcing exists to prevent.
        yield False
        return

    # Inject the INCOMPLETE continue message — bilingual, tail-anchored,
    # never "safe on disk". The forced continue/seal directive rides
    # on the native ``tool_choice`` set below (the message is the continuation
    # hint, the forcing is the active ingredient).
    continue_text = _longfile.build_continue_message(engine)
    engine.history.append(
        Message(
            role=MessageRole.user,
            content_blocks=[TextBlock(text=continue_text)],
            metadata={
                SYNTHETIC_RECOVERY_METADATA_KEY: SYNTHETIC_RECOVERY_LONGFILE_CONTINUE
            },
        )
    )
    _longfile.set_force_next_tool(engine, forced_name)
    if forced is ToolRole.appends_path:
        _longfile.commit_forced_append(engine)
    else:
        _longfile.commit_forced_finalize(engine)
    await engine._persist_snapshot()
    yield _emit_state_change(
        engine,
        engine.state,
        engine.state,
        reason=(
            "longfile_forced_append"
            if forced is ToolRole.appends_path
            else "longfile_forced_finalize"
        ),
    )
    yield True


def _terminal_only_enforced(engine: QueryEngine) -> bool:
    """Return True iff a non-terminal tool dispatch must now be REJECTED.

    Armed by the wind-down, and by nothing else. The narrowed surface already
    means a withdrawn tool is neither advertised nor admitted by the permission
    gate, so this guard is not what stops the call — it is what the model reads
    when it tries one anyway, off a stale schema in its own context. A bare
    "not permitted" would leave it guessing; :func:`_terminal_only_error_message`
    names the tool it should be calling instead.

    The predicate used to key on a deadline-specific latch, which meant the
    strictness followed one of five ways a run could be cut short. It follows
    all five now, because they are all the same wind-down.
    """
    return _soft_stop.tools_withdrawn(engine)


def _terminal_only_blocks(
    engine: QueryEngine, tool_call: ToolCall
) -> bool:
    """Return True iff ``tool_call`` should be rejected because terminal
    finalisation is in effect.

    Predicate is True iff ALL of:
      * STRICT terminal-only finalisation is in force for this tenant
        (:func:`_terminal_only_enforced`: the durable deadline-finalize
        latch). The best-effort error backstop turn is intentionally NOT
        blocked.
      * the terminal nudge latch ``engine._terminal_only_active``
        is set (the per-turn loop flipped it when the deadline-finalize /
        contract-repair nudge fired)
      * a terminal tool name actually resolves
        (:func:`_resolved_terminal_tool_name` is not None; honours the
        per-tenant ``expected_terminal_tool``). With no resolvable terminal
        tool there is nothing to force, so the guard never blocks.
      * no successful terminal tool result is in history yet
      * the tool being dispatched is NOT that resolved terminal tool name.

    The single allowed tool while the latch is active is exactly
    ``_resolved_terminal_tool_name(engine)``.
    """
    if not _terminal_only_enforced(engine):
        return False
    if not getattr(engine, "_terminal_only_active", False):
        return False
    if _history_has_terminal_tool_result(engine):
        return False
    terminal_tool_name = _resolved_terminal_tool_name(engine)
    if terminal_tool_name is None:
        return False
    return tool_call.name != terminal_tool_name


def _terminal_only_error_message(
    engine: QueryEngine, blocked_tool_name: str
) -> str:
    """Structured, model-visible error surfaced when the terminal-only
    guard blocks a non-terminal tool dispatch.

    The message names BOTH the blocked tool and the resolved terminal tool
    so the model gets an actionable instruction (call the terminal tool
    now) rather than a silent drop. Keyed on the generic
    ``expected_terminal_tool`` slot (NOT on any benchmark token); the guard
    only fires once a terminal tool resolves, so the name is always present.
    """
    terminal_tool_name = (
        _resolved_terminal_tool_name(engine) or "the configured terminal tool"
    )
    return (
        f"tool '{blocked_tool_name}' execution blocked: [terminal-only mode: "
        f"deadline reached — call {terminal_tool_name} now with your "
        "best-evidence answer]"
    )


def _history_tool_result_is_terminal(
    engine: QueryEngine,
    tool_call_id: str,
) -> bool:
    """Find the just-appended tool result and inspect its terminal metadata.

    When ``expected_terminal_tool`` is configured, a successful
    terminal-metadata result counts as terminal ONLY if the originating
    tool call's name matches the declared terminal tool — the same
    expected-tool-name guard as :func:`_history_has_terminal_tool_result`.
    This keeps the two helper families consistent so a NON-expected tool
    that returns terminal metadata can never end an
    ``expected_terminal_tool`` tenant's run as "answered" (which would
    discard a stored stream error / surface a false-positive completion).
    When ``expected_terminal_tool`` is None the behaviour is bit-identical
    to before (any successful terminal-metadata result counts) — no
    regression for a host backend or for the default.
    """
    for message in reversed(engine.history):
        for block in reversed(message.content_blocks):
            if (
                isinstance(block, ToolResultBlock)
                and block.tool_call_id == tool_call_id
            ):
                if (
                    block.is_error
                    or block.metadata.get(TERMINAL_TOOL_METADATA_KEY) is not True
                ):
                    return False
                expected = engine.config.expected_terminal_tool
                if expected is None:
                    return True
                return _tool_name_for_call_id(engine, tool_call_id) == expected
    return False


def _rewrite_deferred_tool_result_events(
    events: list[TurnEvent],
    outcome: DispatchOutcome,
) -> list[TurnEvent]:
    """Return buffered events with ``TOOL_RESULT`` text matching ``outcome``.

    Parallel dispatch executes tools under ``asyncio.gather`` but yields results
    in LLM order. Consecutive-error cap decisions are transcript-order state, so
    replay may rewrite an error outcome after gather. The SSE event must carry
    the same text that we append to history.
    """

    rewritten: list[TurnEvent] = []
    for evt in events:
        if (
            evt.type is not EventType.TOOL_RESULT
            or evt.payload.get("tool_call_id") != outcome.tool_call.id
        ):
            rewritten.append(evt)
            continue

        payload = dict(evt.payload)
        payload["success"] = outcome.success
        blocks = payload.get("content_blocks")
        if isinstance(blocks, list):
            new_blocks: list[Any] = []
            for block in blocks:
                if isinstance(block, dict):
                    new_block = dict(block)
                    if new_block.get("type") == "text":
                        new_block["text"] = outcome.content
                    new_blocks.append(new_block)
                else:
                    new_blocks.append(block)
            payload["content_blocks"] = new_blocks

        error = payload.get("error")
        if not outcome.success:
            new_error = dict(error) if isinstance(error, dict) else {}
            if outcome.error_kind is not None:
                new_error["kind"] = outcome.error_kind.value
            new_error["message"] = outcome.content
            payload["error"] = new_error
        else:
            payload.pop("error", None)
        if outcome.metadata:
            payload["metadata"] = dict(outcome.metadata)
        else:
            payload.pop("metadata", None)

        rewritten.append(evt.model_copy(update={"payload": payload}))
    return rewritten


# The dispatcher mutates per-run state mid-execution (the consecutive-error
# streak, the transport-down streak, the string_type streak, the satisfied
# preconditions, the soft-cap counts). Under ``asyncio.gather`` those mutations
# happen in gather completion order, not in the order the model asked for the
# calls, so the final state depends on which tool finished first. The run state
# copies that group of cells out before the gather and puts it back after, and
# the replay below re-applies the transitions in transcript order so the next
# turn's caps fire on the correct count.


def _replay_dispatch_state(
    engine: QueryEngine,
    tool_call: ToolCall,
    outcome: DispatchOutcome,
) -> DispatchOutcome:
    """Apply transcript-order state transitions for one (tool_call, outcome).

    Called by the parallel-dispatch orchestrator in LLM-requested order after
    :meth:`RunScopedState.restore_transcript_state` has put the run's state back
    the way it was before the gather. Delegates to the same classmethods on
    :class:`ToolDispatcher` that the serial dispatch path runs so the
    cap / satisfaction semantics stay defined in ONE place
    (:mod:`protocore.runtime.tool_dispatch`).

    Skips silently when:
    * ``outcome`` is ``None`` (defensive — dispatcher must yield one).
    * ``outcome.approval_required`` (the caller already handled this
      before invoking us — replay would be wrong because the gate
      short-circuited the per-tool execution).

    Returns the transcript-correct outcome. When gathered dispatches hit
    consecutive-error caps in completion order, replay can produce a different
    surfaced error for this tool in LLM order. The caller must use the returned
    outcome for both SSE event text and history append.
    """
    if outcome is None or outcome.approval_required:
        return outcome

    # Gathered calls deliberately defer evidence admission so the immutable
    # ledger follows LLM order rather than completion order.  This is the
    # replay's first state transition: an admission failure becomes a normal
    # dispatch failure before any real error-streak, dependency, history, SSE,
    # or snapshot side effect can observe success.
    outcome = _ingest_tool_evidence(engine, outcome)

    # Lazy import to keep the module-level import graph clean and
    # avoid any chance of a circular import via
    # ``tool_dispatch.py`` re-exporting.
    from protocore.runtime.tool_dispatch import DispatchErrorKind, ToolDispatcher

    ctx = ToolContext(
        tenant_id=engine.config.tenant_id,
        run_id=engine.config.run_id,
        session_id=engine.config.session_id,
        work_scope=engine.config.work_session_id,
        evidence=ToolEvidenceContext(origin=engine._engine_evidence_origin()),
        run_state=engine.run_state,
        metadata=_build_replay_metadata(engine),
    )

    # Cumulative tool-call soft cap — count THIS executed tool call in
    # transcript order. The concurrent gather did NOT count it (the count is a
    # transcript-order cell restored to its pre-gather value above),
    # so the increment happens HERE, once per replayed call, in LLM-requested
    # order — mirroring the serial path's per-call count. Advisory only; the
    # warning (if any) is appended to the surfaced outcome via ``_finalize``.
    # Ledger this call in LLM-requested order. The gather that executed it
    # completed in whatever order the tools finished; the replay is where
    # transcript order is re-established, so it is where the ordinal is taken.
    engine.record_tool_call(tool_call.name, ok=bool(outcome.success))

    if outcome.success:
        ToolDispatcher._reset_consecutive_error_streak(ctx)
        tool = engine.tools.get(tool_call.name)
        if tool is not None:
            ToolDispatcher._record_precondition_satisfaction(
                tool=tool,
                tool_name=tool_call.name,
                arguments=tool_call.arguments,
                ctx=ctx,
            )
        return outcome

    # Error path — replay the consecutive-error cap from the original
    # pre-rewrite `(kind, message)` tuple so the restored state matches the
    # transcript-order serial path even when the gathered dispatch already hit
    # the cap locally.
    metadata = outcome.metadata or {}
    raw_kind = metadata.get(DISPATCH_REPLAY_ERROR_KIND_METADATA_KEY)
    raw_message = metadata.get(DISPATCH_REPLAY_ERROR_MESSAGE_METADATA_KEY)
    kind: DispatchErrorKind | None = None
    if isinstance(raw_kind, str):
        try:
            kind = DispatchErrorKind(raw_kind)
        except ValueError:
            kind = None
    if kind is None:
        kind = outcome.error_kind or DispatchErrorKind.execution
    message = raw_message if isinstance(raw_message, str) else outcome.content
    final_kind, final_message = _ensure_tool_dispatcher(engine)._apply_consecutive_error_cap(
        ctx,
        tool_call.name,
        kind,
        message,
        emit_diagnostics=False,
    )
    if bool(metadata.get(DISPATCH_POST_TOOL_OUTPUT_MODIFIED_METADATA_KEY)):
        # The dispatcher has already run PostToolUse against the surfaced
        # dispatch output. Replay owns transcript-order helper state, but it
        # must not undo hook-level redaction or other output modifications.
        return outcome
    if final_kind == outcome.error_kind and final_message == outcome.content:
        return outcome
    return replace(
        outcome,
        content=final_message,
        is_error=True,
        error_kind=final_kind,
    )


# the per-run ``run_metadata`` envelope is OPERATOR-supplied via the
# public ``POST /v1/runs.metadata`` API. It must never be allowed to shadow a
# RUNTIME-INTERNAL ``ToolContext.metadata`` key: the authoritative
# ``tool_call_id`` consumed by tool-result correlation / subagent-parent edges /
# answer-RPC binding, and any ``protocore.*`` control key such as
# synthetic-recovery / suppress-grounding, carry runtime trust. The merge below
# copies only NON-internal envelope keys.
_TOOL_CALL_ID_METADATA_KEY: Final[str] = "tool_call_id"
_RUNTIME_INTERNAL_METADATA_PREFIX: Final[str] = "protocore."


def _is_runtime_internal_metadata_key(key: str) -> bool:
    """Return ``True`` for a key that operator-supplied ``run_metadata`` must
 NOT be able to set/shadow on ``ToolContext.metadata`` .

 Covers the authoritative ``tool_call_id`` and every ``protocore.*``
 runtime-internal control key.
 """
    return key == _TOOL_CALL_ID_METADATA_KEY or key.startswith(
        _RUNTIME_INTERNAL_METADATA_PREFIX
    )


def _merge_run_metadata_into(
    metadata: dict[str, Any], state: RunScopedState
) -> None:
    """Merge the run's operator-supplied envelope onto ``metadata`` in place,
 skipping runtime-internal keys so a forgeable envelope cannot shadow trusted
 runtime state.
 """
    for key, value in state.run_metadata.items():
        if _is_runtime_internal_metadata_key(key):
            continue
        metadata[key] = value


def _build_replay_metadata(engine: QueryEngine) -> dict[str, Any]:
    """Build the ``ToolContext.metadata`` dict the replay's context carries.

    Mirrors the metadata construction in
    :func:`_drain_dispatch_tool_deferred` (and :func:`_dispatch_tool`) so the
    replay's ``ToolContext`` exposes the same per-run envelope the dispatcher
    would have seen — runtime-internal names skipped, so an operator's envelope
    cannot shadow trusted state on the replay path either.
    """
    # Seed the cross-process re-drive satisfaction set so a precondition check
    # on the replay path sees the same set the live recording produced.
    _rehydrate_satisfied_from_history(engine)
    metadata: dict[str, Any] = {}
    _merge_run_metadata_into(metadata, engine.run_state)
    return metadata


def _rehydrate_satisfied_from_history(engine: QueryEngine) -> None:
    """Seed the run's satisfied set from ``engine.history`` when it is empty.

 The run's state is composed per process and is NOT carried in the
 engine snapshot. On a cross-process re-drive a fresh process sees an
 empty satisfied set even when ``engine.history`` already contains
 a long transcript of ``AppendFile(foo)``/``Write(...)``/etc.
 Without rehydration a follow-up ``FinalizeFile(foo)`` call would
 be blocked with ``[PRECONDITION NOT MET: AppendFile:foo]`` even
 though the prereq is right there in the durable transcript.

 Reads :data:`engine.history` and writes the rebuilt set onto the run's
 state only when that set is empty — a populated one always wins, since
 in-process dispatches have already recorded the live satisfaction entries.

 The history is the only cross-process-durable record for a resumed run,
 so it is the canonical source of truth on the re-drive path.

 Rebuilt from :func:`_this_run_messages`, not from the whole
 transcript. A tool precondition is a statement about what THIS run
 has already done, and cross-run history seeding puts an earlier
 run's ``AppendFile(report.md)`` into this run's history — over the
 whole transcript that earlier call would authorise this run's
 ``FinalizeFile(report.md)`` without this run having appended
 anything, which is the same class of error as a failed call
 authorising a dependent one.
 """
    state = engine.run_state
    if state.satisfied_preconditions:
        return
    # Late import to avoid a circular import: tool_preconditions does
    # not import from this module, but the engine → query → preconditions
    # direction is cleaner at call-site.
    from protocore.runtime.tool_preconditions import record_satisfaction

    run_messages = _this_run_messages(engine)
    if not run_messages:
        return
    # A tool-use block merely records the model's request.  Rehydrating it as
    # satisfaction would let a failed tool (including evidence rejection)
    # authorize a dependent call after the state is rebuilt.  Pair the request
    # with its durable non-error result instead.
    pending_calls: dict[str, tuple[str, dict[str, Any]]] = {}
    rebuilt: set[str] = set()
    for message in run_messages:
        for block in message.content_blocks:
            if isinstance(block, ToolUseBlock):
                arguments: dict[str, Any] = {}
                try:
                    decoded = json.loads(block.arguments_json)
                except (TypeError, ValueError):
                    decoded = None
                if isinstance(decoded, dict):
                    arguments = decoded
                pending_calls[block.tool_call_id] = (block.name, arguments)
            elif isinstance(block, ToolResultBlock) and not block.is_error:
                call = pending_calls.pop(block.tool_call_id, None)
                if call is not None:
                    tool_name, arguments = call
                    record_satisfaction(
                        tool_name=tool_name,
                        arguments=arguments,
                        satisfied=rebuilt,
                    )
    if rebuilt:
        state.satisfied_preconditions = rebuilt


# ---------------------------------------------------------------------------
# Repeated-tool-error circuit breaker
# ---------------------------------------------------------------------------


def _resolve_max_consecutive_tool_errors(engine: QueryEngine) -> int:
    """Read ``max_consecutive_tool_errors`` from the RC snapshot.

    Defensive fallback mirrors :meth:`ToolDispatcher._resolve_consecutive_error_cap`:
    a corrupted/absent value degrades to the Pydantic default so the breaker can
    never trip on the very first error. The RC has a ``ge=2`` validator, but a
    sub-2 value here would mean "trip on first error" — clamp it out.
    """
    raw = getattr(engine.config.rc, "max_consecutive_tool_errors", 3)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return 3
    return value if value >= 2 else 3


def _circuit_breaker_corrective_text(tool_name: str) -> str:
    """Bilingual corrective convergence turn for a circuit-broken tool.

    Frames the disablement as a fact ("the tool is unavailable for the rest of
    this run") and instructs the model to ANSWER from the conversation instead
    of retrying — forcing convergence. EN+RU (multilingual mandatory).
    """

    return (
        f"The '{tool_name}' tool repeatedly failed with the same error and has "
        "been disabled for the rest of this run — do NOT call it again. Answer "
        "the user's request now using what you already know from this "
        "conversation; if you cannot, say so plainly and finish. | "
        f"Инструмент '{tool_name}' многократно завершался одной и той же "
        "ошибкой и отключён до конца этого запуска — больше не вызывайте его. "
        "Ответьте на запрос пользователя сейчас, используя то, что уже известно "
        "из этого диалога; если это невозможно, прямо сообщите об этом и "
        "завершите работу."
    )


def _circuit_breaker_track_and_maybe_trip(
    engine: QueryEngine,
    tool_call: ToolCall,
    outcome: DispatchOutcome,
) -> str | None:
    """Track the consecutive same-tool/same-error-class streak and trip the
    hard circuit breaker once it crosses ``max_consecutive_tool_errors``.

    Called after every NON-approval dispatch outcome. On a SUCCESS the streak is
    cleared. On an ERROR the ``(tool_name, error_kind)`` streak increments; when
    it reaches the cap the tool is added to ``engine._circuit_broken_tools``
    (removed from the surface AND denied at dispatch via
    ``effective_tool_policy.blocked``) and — at most once per tool — a corrective
    convergence message is returned for the caller to inject as a bounded
    synthetic user turn.

    Returns the corrective text to inject, or ``None`` (no trip this dispatch,
    or the tool was already broken+notified). The in-flight streak lives on the
    engine (``_circuit_breaker_streak``) so it is snapshot-persisted across a
    cross-pod resume.
    """

    tool_name = tool_call.name

    # SUCCESS — reset the streak unconditionally. A success of ANY tool
    # breaks the "consecutive" chain: ``Read(err) → List(ok) → Read(err) →
    # Read(err)`` must NOT trip at cap 3, because the Read failures were not
    # consecutive. (Matches the dispatcher's own ``_reset_consecutive_error_
    # streak`` on a successful tool result.)
    if not outcome.is_error:
        engine._circuit_breaker_streak = None
        return None

    # Cap-INELIGIBLE soft error — a tool result that opted out of
    # consecutive-error capping (``consecutive_error_cap_eligible=False``): an
    # ordinary Bash nonzero exit used as DATA (``grep -q`` no-match, ``test``
    # false, ``diff`` difference). These are NOT "repeat the same failing call"
    # signals, so — exactly like the dispatcher's soft-error path — they must
    # neither ADVANCE nor RESET the hard-breaker streak. Reuse the SAME strict-
    # bool eligibility flag the dispatcher honours (absent ⇒ eligible).
    raw_eligible = (outcome.metadata or {}).get(
        TOOL_RESULT_CONSECUTIVE_CAP_ELIGIBLE_METADATA_KEY, True
    )
    cap_eligible = raw_eligible if isinstance(raw_eligible, bool) else True
    if not cap_eligible:
        return None

    # Already broken — keep it blocked (the surface/gate already deny it) and do
    # NOT re-inject the corrective turn (the latch lives on the engine so it
    # survives cross-pod resume).
    if tool_name in engine._circuit_broken_tools:
        return None

    error_class = outcome.error_kind.value if outcome.error_kind is not None else "error"

    state_raw = engine._circuit_breaker_streak
    last_tool: str | None = None
    last_class: str | None = None
    count = 0
    if isinstance(state_raw, dict):
        if isinstance(state_raw.get("tool_name"), str):
            last_tool = state_raw["tool_name"]
        if isinstance(state_raw.get("error_class"), str):
            last_class = state_raw["error_class"]
        raw_count = state_raw.get("count", 0)
        if isinstance(raw_count, int) and raw_count >= 0:
            count = raw_count

    if last_tool == tool_name and last_class == error_class:
        count += 1
    else:
        count = 1

    engine._circuit_breaker_streak = {
        "tool_name": tool_name,
        "error_class": error_class,
        "count": count,
    }

    cap = _resolve_max_consecutive_tool_errors(engine)
    if count < cap:
        return None

    # TRIP — hard-stop the tool for the rest of the run and inject ONE corrective
    # convergence turn.
    engine._circuit_broken_tools.add(tool_name)
    _logger.warning(
        "DIAG query.circuit_breaker.tripped run=%s tenant=%s tool=%s "
        "error_class=%s count=%d cap=%d",
        engine.config.run_id,
        engine.config.tenant_id,
        tool_name,
        error_class,
        count,
        cap,
    )
    if tool_name in engine._circuit_breaker_notified_tools:
        return None
    engine._circuit_breaker_notified_tools.add(tool_name)
    return _circuit_breaker_corrective_text(tool_name)


def _message_text(message: Message) -> str:
    """The plain text of a message, as the reply to a question the tool asked."""
    return "\n".join(
        block.text for block in message.content_blocks if isinstance(block, TextBlock)
    )


def _park_pause_interrupt(
    engine: QueryEngine,
    tool_call_id: str,
    *,
    event: TurnEvent | None = None,
    tool_name: str = "",
) -> PendingInterrupt:
    """Record what the loop is now waiting for, reading the kind off the pause.

    Both pauses announce themselves with the same ``tool_call_pending``
    envelope; only its contents say which of the two happened. An envelope
    tagged as an ask carries a question a tool already asked and is waiting on;
    anything else is a call parked at a gate that has not run. The loop used to
    treat both as the second, so an answer arriving for the first was refused
    as "not the pending approval", and the operator's card described a call as
    awaiting a decision when what it awaited was a reply.
    """
    payload = dict(event.payload) if event is not None and event.payload else {}
    name = tool_name or str(payload.get("tool_name") or "")
    if payload.get("ask_user") or payload.get("kind") == "ask_user":
        return engine.mark_awaiting_answer(tool_call_id, tool_name=name, payload=payload)
    return engine.mark_pending_approval(tool_call_id, tool_name=name, payload=payload)


def _interrupt_parked_event(
    engine: QueryEngine, interrupt: PendingInterrupt
) -> TurnEvent:
    """Tell the host what the run is waiting for, and for how many things.

    The whole open set travels beside the one just parked: a host that renders
    a card per wait can draw a parked batch in one pass instead of learning
    about the second and third calls only when it tries to resume past them.
    """
    return TurnEvent(
        type=EventType.INTERRUPT_PARKED,
        run_id=engine.config.run_id,
        payload={
            "turn_id": engine.turn_id(),
            "interrupt": interrupt.to_dict(),
            "pending_interrupts": [
                item.to_dict() for item in engine.pending_interrupts
            ],
        },
    )


def _tool_call_from_history(engine: QueryEngine, tool_call_id: str) -> ToolCall:
    """Rebuild the parked call from the transcript that recorded it.

    The transcript is the authority here rather than anything the caller sends
    with the resolution: the resolution names an interrupt, and what that
    interrupt parked is whatever the assistant actually asked for. A caller
    that could also supply the call would be a caller that could substitute one.
    """
    for message in engine.history:
        for block in message.content_blocks:
            if isinstance(block, ToolUseBlock) and block.tool_call_id == tool_call_id:
                try:
                    arguments = json.loads(block.arguments_json or "{}")
                except (TypeError, ValueError):
                    arguments = {}
                return ToolCall(
                    id=tool_call_id,
                    name=block.name,
                    arguments=arguments if isinstance(arguments, dict) else {},
                )
    raise InterruptResolutionError(
        f"the call {tool_call_id!r} an interrupt parked is not in this run's "
        "history; the snapshot and the interrupt describe different runs"
    )


def _apply_updated_input(
    engine: QueryEngine,
    tool_call: ToolCall,
    updated_input: dict[str, Any],
) -> ToolCall:
    """Run the approved call with the arguments a person corrected.

    Three things have to agree afterwards or the correction is a lie somewhere:
    the call that is dispatched, the ``tool_use`` block the model is shown, and
    the durable record written before the call. The block is rewritten so the
    transcript says what was really run rather than what was proposed, and the
    record's fingerprint is rewritten with it so the pause check does not
    refuse the very call the operator just fixed.
    """
    corrected = ToolCall(id=tool_call.id, name=tool_call.name, arguments=dict(updated_input))
    arguments_json = json.dumps(corrected.arguments, ensure_ascii=False)
    for index, message in enumerate(engine.history):
        rewritten = [
            (
                block.model_copy(update={"arguments_json": arguments_json})
                if isinstance(block, ToolUseBlock)
                and block.tool_call_id == tool_call.id
                else block
            )
            for block in message.content_blocks
        ]
        if rewritten != list(message.content_blocks):
            engine.history[index] = message.model_copy(
                update={"content_blocks": rewritten}
            )
    intent = find_intent(engine.open_intents, tool_call.id)
    if intent is not None:
        intent.arguments_fingerprint = _intent_fingerprint(
            corrected.name, corrected.arguments
        )
    return corrected


def _settle_parked_call(
    engine: QueryEngine,
    *,
    tool_call_id: str,
    content: str,
    is_error: bool,
) -> None:
    """Close a parked call with the result a person's decision produced.

    Both halves are needed. A ``tool_use`` left unpaired is filled in on the
    wire with a synthetic failure, which says the opposite of what a denial or
    an answer means; and a durable record left standing keeps the call open
    forever, so the next resume finds a wait nobody is going to answer.
    """
    _insert_tool_result_after_use(
        engine.history,
        tool_call_id=tool_call_id,
        content=content,
        is_error=is_error,
    )
    intent = find_intent(engine.open_intents, tool_call_id)
    if intent is not None:
        settle_intent(intent, result=content)
        _forget_intent(engine, intent)
    engine.forget_tool_name(tool_call_id)


def _forget_intent(engine: QueryEngine, intent: IntentRecord) -> None:
    """Drop a record whose call is durably answered by history."""
    engine.open_intents = [
        item for item in engine.open_intents if item is not intent
    ]


def _assert_pause_envelope_matches_intent(
    intent: IntentRecord,
    buffered: list[TurnEvent],
) -> None:
    """Refuse a pause whose envelope contradicts the record it pauses.

    Both the approval park and the ask-user park announce themselves with a
    ``tool_call_pending`` envelope naming the call, the tool and the input.
    When that envelope and the record written before the dispatch disagree,
    there is no principled way to choose between them: one of them describes
    the call an operator is about to approve, the other describes something
    else, and executing either is a coin flip on which. So the run stops
    instead, loudly, with both readings in the message.
    """
    for evt in buffered:
        if evt.type is not EventType.TOOL_CALL_PENDING:
            continue
        payload = evt.payload or {}
        call_id = payload.get("tool_call_id")
        if not isinstance(call_id, str):
            continue
        assert_pause_matches(
            intent,
            tool_call_id=call_id,
            tool_name=str(payload.get("tool_name", intent.tool_name)),
            arguments=payload.get("tool_input"),
        )


def _durable_dispatch_start(
    engine: QueryEngine,
    intent: IntentRecord,
) -> Callable[[ToolCall], Awaitable[None]] | None:
    """Build the callback that makes the intent durable before the tool runs.

    Returns ``None`` for a tool whose repeat is harmless. Durability here buys
    exactly one thing — not applying a side effect twice — and a tool that only
    reads state has no side effect to apply twice, so paying for a snapshot
    write per call to buy nothing is the wrong trade on the hottest path in the
    loop.
    """
    if intent.repeat_is_safe:
        return None

    async def _start(_call: ToolCall) -> None:
        mark_dispatched(intent)
        await engine._persist_snapshot()

    return _start


async def _settle_interrupted_tool_intents(
    engine: QueryEngine,
) -> AsyncIterator[TurnEvent]:
    """Close out records left by a run that stopped while a tool was in flight.

    Runs at the top of every turn, which is where a run picked up on another
    pod first gets the chance to say something true about what it was doing
    when it died.

    A record still reading ``DISPATCHED`` with no result anywhere in history
    describes a call that was handed to a tool and never came back. The tool
    may well have done its work — written the file, sent the request — and the
    only honest thing to tell the model is that the outcome was never recorded.
    Telling it the call FAILED, which is what a synthetic error result says,
    invites precisely the repeat that doubles the effect.

    Records parked at an approval gate or waiting on a user's answer are never
    given an outcome here: neither has an unknown one, and neither may be
    executed by a resume. They are still dropped once history answers them —
    the answer to a paused question arrives as a tool result appended by the
    layer that collected it, and a record whose result is in history has
    nothing left to say. Keeping it would put one more entry in every snapshot
    the run writes from then on, for the life of the run.
    """
    resolved: set[str] = set()
    for message in engine.history:
        for block in message.content_blocks:
            if isinstance(block, ToolResultBlock):
                resolved.add(block.tool_call_id)

    stale = orphaned_intents(list(engine.open_intents), resolved_tool_call_ids=resolved)
    settled_by_history = [
        item
        for item in engine.open_intents
        if item.state != SETTLED and item.tool_call_id in resolved
    ]
    for item in settled_by_history:
        _forget_intent(engine, item)
    if not stale:
        if settled_by_history:
            await engine._persist_snapshot()
        return

    for item in stale:
        text = unknown_outcome_text(item, engine.config.rc)
        _insert_tool_result_after_use(
            engine.history,
            tool_call_id=item.tool_call_id,
            content=text,
            is_error=False,
        )
        settle_unknown(item)
        item.reported = True
        _logger.warning(
            "DIAG query.tool_intent.outcome_unknown run=%s tool=%s call_id=%s "
            "idempotency_key=%s repeat_safe=%s",
            engine.config.run_id,
            item.tool_name,
            item.tool_call_id,
            item.idempotency_key,
            item.repeat_is_safe,
        )
        yield TurnEvent(
            type=EventType.TOOL_RESULT,
            run_id=engine.config.run_id,
            payload={
                "tool_call_id": item.tool_call_id,
                "content": text,
                "is_error": False,
                "outcome": "unknown",
                "idempotency_key": item.idempotency_key,
            },
        )
        if engine.config.rc.intent_settlement_enabled:
            from protocore.runtime.correctness_bind import mark_intent_recovery

            for rec_evt in mark_intent_recovery(engine, item):
                yield rec_evt
    if engine.config.rc.intent_settlement_enabled:
        from protocore.runtime.correctness_bind import persist_correctness

        persist_correctness(engine)
    await engine._persist_snapshot()


def _insert_tool_result_after_use(
    history: list[Message],
    *,
    tool_call_id: str,
    content: str,
    is_error: bool,
) -> bool:
    """Place a tool result directly after the assistant turn that called it.

    Appending at the tail instead would leave the pair out of order whenever a
    recovery or user message already follows the call, and a provider rejects
    that outright.
    """
    for index, message in enumerate(history):
        if message.role is not MessageRole.assistant:
            continue
        if not any(
            isinstance(block, ToolUseBlock) and block.tool_call_id == tool_call_id
            for block in message.content_blocks
        ):
            continue
        history.insert(
            index + 1,
            Message(
                role=MessageRole.tool,
                content_blocks=[
                    ToolResultBlock(
                        tool_call_id=tool_call_id,
                        content=content,
                        is_error=is_error,
                    )
                ],
            ),
        )
        return True
    return False


async def _dispatch_tool(
    engine: QueryEngine,
    tool_call: ToolCall,
    *,
    preapproved: bool = False,
    synthetic_recovery: bool = False,
    synthetic_recovery_kind: str = SYNTHETIC_RECOVERY_GUARANTEED_TERMINAL,
    keep_truncated_tail: bool = False,
) -> AsyncIterator[TurnEvent]:
    """Execute one tool_call with hooks. Yields events. Appends tool_result.

 md`. Delegates the 7-step
 dispatch lifecycle to :class:`ToolDispatcher`; performs engine-side
 state mutations (history append, snapshot, tool-name cleanup) here
 so they stay observable from the engine's perspective.

 Event ordering invariant (matches existing tests):

 (LLM-stream) tool_use_start → tool_use_input_delta → tool_use_stop
 (dispatcher) hook_fired(pre) →
 (tool_call_pending | hook_fired(post) + tool_result)

 ``tool_transport_starting`` is emitted by the **host's transport**
 (never by core) on cold start only.
 """
    _pin_keep_flag(engine, tool_call)
    # Terminal-only finalisation guard. Once the terminal-answer nudge has
    # fired and no terminal tool result is in history yet, every
    # non-terminal dispatch is short-circuited with a structured error
    # pointing at finalisation. The model still chooses message / outcome /
    # refs — the runtime never synthesises the answer. See
    # :func:`_terminal_only_blocks` for the full predicate.
    if _terminal_only_blocks(engine, tool_call):
        error_message = _terminal_only_error_message(engine, tool_call.name)
        yield TurnEvent(
            type=EventType.TOOL_RESULT,
            run_id=engine.config.run_id,
            payload={
                "tool_call_id": tool_call.id,
                "success": False,
                "is_error": True,
                "error": {
                    "kind": "terminal_only",
                    "message": error_message,
                },
                "content_blocks": [{"type": "text", "text": error_message}],
            },
        )
        engine.history.append(
            Message(
                role=MessageRole.tool,
                content_blocks=[
                    ToolResultBlock(
                        tool_call_id=tool_call.id,
                        content=error_message,
                        is_error=True,
                    )
                ],
            )
        )
        engine.forget_tool_name(tool_call.id)
        await engine._persist_snapshot()
        return

    # Cumulative total-work guard — the serial twin of the one on the deferred
    # parallel path. Both dispatch routes must refuse, or a leader whose fan-out
    # was disabled (a single delegation call, a hook-gated one, or
    # ``parallel_subagents_enabled`` off) would keep delegating past the budget
    # on exactly the serial shape the cumulative bound exists to catch.
    run_work_refusal, run_work_reason = _run_work_delegation_refusal(
        engine, tool_call
    )
    if run_work_refusal:
        _logger.warning(
            "DIAG query.run_work_budget.delegation_refused run=%s tenant=%s "
            "tool=%s reason=%s %s",
            engine.config.run_id,
            engine.config.tenant_id,
            tool_call.name,
            run_work_reason,
            _resolve_run_work_ledger(engine).spent_summary(),
        )
        refusal_events, refusal_outcome = _run_work_refusal_dispatch(
            engine, tool_call, run_work_refusal, run_work_reason
        )
        for evt in refusal_events:
            yield evt
        engine.history.append(
            Message(
                role=MessageRole.tool,
                content_blocks=[
                    _result_block_from_outcome(tool_call.id, refusal_outcome)
                ],
            )
        )
        engine.forget_tool_name(tool_call.id)
        await engine._persist_snapshot()
        return

    #  — universal PROSE-GATE before a BACKGROUND terminal
    # tool. The terminal tool is a pure background gate: its answer field is
    # removed and its tool_use / tool_result pair is filtered from the stream +
    # durable history, so the ONLY user-facing answer is the model's own
    # visible assistant prose. When this run is about to latch the terminal
    # tool's result but produced NO substantive visible prose after its latest
    # real-work tool (:func:`_finalize_prose_gate_applies`), we VETO the
    # dispatch ONCE — exactly like the pre-dispatch / candidate-repair seams:
    # append a NON-terminal error tool_result (so
    # ``_history_tool_result_is_terminal`` returns False and the loop does NOT
    # finalise) plus ONE bounded corrective user turn asking the model to write
    # the final answer as normal text and THEN call the terminal tool, charge
    # the bound this veto answers to (durable across resume), persist, and
    # return so the outer loop re-drives one corrective turn. Unlike the
    # pre-dispatch verify seam this needs NO the host trigger (the
    # visible-prose predicate is a pure check on the engine history core owns)
    # and does NOT debit the self-verify budget — it carries its own.
    # TWO bounds, not one: a payload-only terminal spends the gate's single
    # durable latch, while a terminal whose only prose is a filing notice
    # spends one attempt of the pointer refusal's own budget, which is larger
    # because the first correction was measured to be ignored. The repair text
    # is the OPPOSITE of the terminal-tool nudge ('write the answer as normal
    # text first'), so the two never contradict. Default-on
    # (``finalize_prose_gate_enabled``); ``_applies`` returns False for the
    # healthy prose-then-terminal shape and once the relevant bound is spent, so
    # an uncorrected terminal eventually finalises rather than looping.
    if _finalize_prose_gate_applies(engine, tool_call):
        repair_text = engine.prompt_text("finalize_prose_gate_repair")
        # An empty repair text would inject an empty user turn — degrade to a
        # no-op (let the terminal dispatch through) instead. Nothing is latched
        # or charged in that case, mirroring "the gate did not fire".
        if repair_text:
            # Read the pointer evidence BEFORE the veto's own tool_result and
            # repair turn land: that error result reads as real work, which
            # closes the answer window the measurement is taken over.
            pointer = _pointer_answer_evidence(engine)
            # Same split as the plain-stop seam: the payload-only veto spends
            # the gate's single shot, the pointer veto one attempt of the
            # pointer refusal's own budget. A run may therefore be vetoed for a
            # filing notice after it was already vetoed for having no prose at
            # all — two different failures, each answered on its own terms.
            _attempt = 0
            if pointer is None:
                engine._finalize_prose_gate_used = True
            else:
                _attempt = _charge_pointer_answer_repair(engine)
            veto_error = (
                f"tool '{tool_call.name}' submission withheld: write your final "
                "answer to the user as a normal assistant message FIRST, then "
                f"call {tool_call.name} to end the run. It was NOT submitted."
            )
            yield TurnEvent(
                type=EventType.TOOL_RESULT,
                run_id=engine.config.run_id,
                payload={
                    "tool_call_id": tool_call.id,
                    "success": False,
                    "is_error": True,
                    "error": {
                        "kind": "finalize_prose_gate",
                        "message": veto_error,
                    },
                    "content_blocks": [{"type": "text", "text": veto_error}],
                },
            )
            # Non-terminal error tool_result for the vetoed call (keeps the
            # assistant/tool message pairing valid; ``_history_tool_result_is_
            # terminal`` returns False so the loop does NOT finalise) ...
            engine.history.append(
                Message(
                    role=MessageRole.tool,
                    content_blocks=[
                        ToolResultBlock(
                            tool_call_id=tool_call.id,
                            content=veto_error,
                            is_error=True,
                        )
                    ],
                )
            )
            # ... followed by the bounded prose-repair user turn (write the
            # answer as normal text first, THEN call the terminal tool).
            engine.history.append(
                Message(
                    role=MessageRole.user,
                    content_blocks=[TextBlock(text=repair_text)],
                    metadata={
                        SYNTHETIC_RECOVERY_METADATA_KEY: (
                            SYNTHETIC_RECOVERY_PROSE_GATE_REPAIR
                        )
                    },
                )
            )
            engine.forget_tool_name(tool_call.id)
            # Persist the snapshot IMMEDIATELY after the corrective turn +
            # latch mutation so a crash / cross-pod resume between the
            # injection and the next persistence boundary cannot lose the
            # latch (and re-veto) or the correction.
            await engine._persist_snapshot()
            # A prose-less terminal and a terminal whose prose was only a
            # pointer are the same veto and different failures; the second
            # carries the sizes that identify it.
            if pointer is None:
                _logger.warning(
                    "DIAG query.finalize_prose_gate.vetoed run=%s tenant=%s "
                    "turn=%s tool=%s",
                    engine.config.run_id,
                    engine.config.tenant_id,
                    engine.turn_id(),
                    tool_call.name,
                )
            else:
                _pointer_path, _answer_chars, _written_chars = pointer
                _rc = engine.config.rc
                _logger.warning(
                    "DIAG query.finalize_prose_gate.pointer_answer_vetoed "
                    "run=%s tenant=%s turn=%s tool=%s attempt=%d/%d "
                    "answer_chars=%d written_chars=%d max_fraction=%.3f path=%s",
                    engine.config.run_id,
                    engine.config.tenant_id,
                    engine.turn_id(),
                    tool_call.name,
                    _attempt,
                    _rc.finalize_prose_gate_pointer_max_repair_attempts,
                    _answer_chars,
                    _written_chars,
                    _rc.finalize_prose_gate_pointer_max_answer_fraction,
                    _pointer_path,
                )
            return

    # Reached when the prose gate let this dispatch through. If the pointer
    # refusal is out of attempts and the answer it objected to is still standing,
    # the terminal tool is about to seal the run on it — the same giving-up the
    # plain-stop completion reports, at the seam where a terminal tool exists to
    # reach it first. Costs one integer comparison on every other dispatch.
    _release_pointer_answer_repair(engine)

    # PRE-DISPATCH terminal-tool verify seam. For a terminal tool whose
    # external side effect (an answer-submission RPC) fires inside its own
    # ``run()``, the post-dispatch self-verify turn is POST-SUBMIT and cannot
    # repair the answer. This gate consults the host-supplied predicate
    # BEFORE the dispatcher runs the tool, and only for the configured
    # ``expected_terminal_tool``. If the predicate
    # vetoes (returns a corrective message), the terminal tool is NEVER
    # dispatched (no RPC fires): we append a non-terminal error tool_result
    # (so ``_history_tool_result_is_terminal`` returns False and the loop
    # does NOT finalise — this is also why the truncated-tool-call
    # terminal-completion site cannot bypass the check, R8A MEDIUM#1) plus
    # ONE bounded corrective user turn, latch (durable, fire-at-most-once),
    # debit the shared self-verify budget, persist, and return so the outer
    # loop re-drives one corrective turn. Default-off: ``_applies`` returns
    # False for every tenant that has not opted in, so behaviour is
    # bit-identical to the gate-disabled path.
    # Candidate-regression protection on the REPAIR turn. The pre-dispatch
    # terminal veto below is fire-at-most-once (it latches
    # ``_pre_dispatch_terminal_verify_used`` on the first veto), so the
    # model's corrected re-submission never re-enters that gate. Without this
    # independent seam the preserved-candidate regression check therefore
    # never runs on the repair turn and a regressed body dispatches unguarded.
    # This branch re-runs that check (via the SAME
    # ``_resolve_terminal_candidate_corrective`` decision + the SAME
    # ``_terminal_candidate_reveto_used`` one-shot latch) keyed only on a
    # held substantive candidate, independent of the pre-dispatch latch. It
    # only ever fires AFTER the first veto preserved a candidate (which is
    # also when the pre-dispatch latch is already closed), so it and the
    # pre-dispatch branch never both fire on one dispatch. Default-off (RC)
    # makes ``_applies`` return False. The core never synthesises the answer
    # body — it only re-vetoes once or allows through + finalises.
    if _terminal_candidate_repair_applies(engine, tool_call):
        repair_corrective = _resolve_terminal_candidate_corrective(
            engine, tool_call, None
        )
        if repair_corrective:
            veto_error = (
                f"tool '{tool_call.name}' submission withheld: pre-submission "
                "verification found a problem with this answer. It was NOT "
                "submitted. Review the correction below, fix your answer, then "
                f"call {tool_call.name} again."
            )
            yield TurnEvent(
                type=EventType.TOOL_RESULT,
                run_id=engine.config.run_id,
                payload={
                    "tool_call_id": tool_call.id,
                    "success": False,
                    "is_error": True,
                    "error": {
                        "kind": "terminal_candidate_repair_reveto",
                        "message": veto_error,
                    },
                    "content_blocks": [{"type": "text", "text": veto_error}],
                },
            )
            # Non-terminal error tool_result for the vetoed call (keeps the
            # assistant/tool message pairing valid; ``_history_tool_result_is_
            # terminal`` returns False so the loop does NOT finalise) ...
            engine.history.append(
                Message(
                    role=MessageRole.tool,
                    content_blocks=[
                        ToolResultBlock(
                            tool_call_id=tool_call.id,
                            content=veto_error,
                            is_error=True,
                        )
                    ],
                )
            )
            # ... followed by the bounded corrective user turn so the model
            # knows WHAT to fix before re-submitting.
            engine.history.append(
                Message(
                    role=MessageRole.user,
                    content_blocks=[TextBlock(text=repair_corrective)],
                    metadata={
                        SYNTHETIC_RECOVERY_METADATA_KEY: (
                            SYNTHETIC_RECOVERY_TERMINAL_REPAIR
                        )
                    },
                )
            )
            engine.forget_tool_name(tool_call.id)
            await engine._persist_snapshot()
            _logger.warning(
                "DIAG query.terminal_candidate_repair.revetoed run=%s "
                "tenant=%s turn=%s tool=%s",
                engine.config.run_id,
                engine.config.tenant_id,
                engine.turn_id(),
                tool_call.name,
            )
            return
        # No veto: either the repair body is itself substantive (not a
        # regression) or the one-shot repair credit was already spent on a
        # prior turn (which persisted the latch then). Neither case mutates new
        # engine state here, so the body dispatches normally below and the run
        # finalises on best evidence.

    if _pre_dispatch_terminal_verify_applies(engine, tool_call):
        corrective = _resolve_pre_dispatch_terminal_veto(engine, tool_call)
        # Preserve the first substantive draft across the veto and re-veto a
        # regressed (empty/1-char) repair turn once. Default-off (RC) returns
        # ``corrective`` unchanged, so the branch below is bit-identical when
        # disabled. Any engine-side candidate/latch mutation is made durable
        # by the ``await engine._persist_snapshot()`` already on this veto
        # path.
        corrective = _resolve_terminal_candidate_corrective(
            engine, tool_call, corrective
        )
        # UNCONDITIONAL heartbeat on every gate APPLICATION, regardless of
        # veto outcome. The gate's prior DIAG lived only inside
        # ``if corrective:`` below, so the no-veto path was silent. This
        # line makes "did the gate run, and what did it decide?" observable
        # from the executor log alone. LOG-ONLY: no mutation; only reached
        # when ``_pre_dispatch_terminal_verify_applies`` already returned
        # True. ``verdict`` = ``veto`` when a corrective was produced
        # (terminal submission withheld below) else ``no_veto``. ``cited``
        # is exact.
        _logger.warning(
            "DIAG query.pre_dispatch_terminal_verify.applied run=%s "
            "verdict=%s cited=%d",
            engine.config.run_id,
            "veto" if corrective else "no_veto",
            _count_terminal_answer_cited_refs(tool_call),
        )
        if corrective:
            engine._pre_dispatch_terminal_verify_used = True
            engine._self_verify_extra_turns_used = (
                getattr(engine, "_self_verify_extra_turns_used", 0) + 1
            )
            veto_error = (
                f"tool '{tool_call.name}' submission withheld: pre-submission "
                "verification found a problem with this answer. It was NOT "
                "submitted. Review the correction below, fix your answer, then "
                f"call {tool_call.name} again."
            )
            yield TurnEvent(
                type=EventType.TOOL_RESULT,
                run_id=engine.config.run_id,
                payload={
                    "tool_call_id": tool_call.id,
                    "success": False,
                    "is_error": True,
                    "error": {
                        "kind": "pre_dispatch_terminal_verify",
                        "message": veto_error,
                    },
                    "content_blocks": [{"type": "text", "text": veto_error}],
                },
            )
            # Non-terminal error tool_result for the vetoed call (keeps the
            # assistant/tool message pairing valid) ...
            engine.history.append(
                Message(
                    role=MessageRole.tool,
                    content_blocks=[
                        ToolResultBlock(
                            tool_call_id=tool_call.id,
                            content=veto_error,
                            is_error=True,
                        )
                    ],
                )
            )
            # ... followed by the bounded corrective user turn so the model
            # knows WHAT to fix before re-submitting.
            engine.history.append(
                Message(
                    role=MessageRole.user,
                    content_blocks=[TextBlock(text=corrective)],
                    metadata={
                        SYNTHETIC_RECOVERY_METADATA_KEY: (
                            SYNTHETIC_RECOVERY_PRE_DISPATCH_TERMINAL_VERIFY
                        )
                    },
                )
            )
            engine.forget_tool_name(tool_call.id)
            await engine._persist_snapshot()
            _logger.warning(
                "DIAG query.pre_dispatch_terminal_verify.vetoed run=%s "
                "tenant=%s turn=%s tool=%s",
                engine.config.run_id,
                engine.config.tenant_id,
                engine.turn_id(),
                tool_call.name,
            )
            return

    dispatcher = _ensure_tool_dispatcher(engine)
    # Seed the run's satisfied set from the durable ``engine.history`` when the
    # state is fresh (cross-process re-drive). The dispatcher's
    # :meth:`_check_tool_preconditions` reads that set, so this MUST run first.
    _rehydrate_satisfied_from_history(engine)
    metadata: dict[str, Any] = {}
    # Merge the run's operator-supplied envelope onto ``ToolContext.metadata``
    # so tools can read the values it carries. The merge skips RUNTIME-INTERNAL
    # names (the authoritative ``tool_call_id`` and any ``protocore.*`` control
    # key) so a forged envelope cannot shadow trusted runtime state. The
    # authoritative ``tool_call_id`` is then set by the dispatcher from the real
    # ``tool_call.id``.
    _merge_run_metadata_into(metadata, engine.run_state)
    # Flag the SYNTHETIC dispatch so a backend MAY default a required terminal
    # field (e.g. ``outcome``) ONLY for the runtime-synthesised last-resort
    # guaranteed-terminal answer, never for a model-emitted one.
    # ``synthetic_recovery_kind`` names WHICH scaffold: guaranteed-terminal
    # (default) vs the longfile salvage write. A salvage file-write must NOT
    # masquerade as a guaranteed-terminal scaffold; the metadata key is
    # unforgeable runtime state a backend may branch on. Set by core LAST
    # (after the run_metadata merge) so a forged ``run_metadata`` value cannot
    # shadow it; ALSO stripped from incoming run_metadata in
    # the host's own run-metadata sanitiser.
    # ``False`` for every normal tool call ⟹ no key set ⟹ bit-identical.
    if synthetic_recovery:
        metadata[SYNTHETIC_RECOVERY_METADATA_KEY] = synthetic_recovery_kind
    ctx = ToolContext(
        tenant_id=engine.config.tenant_id,
        run_id=engine.config.run_id,
        session_id=engine.config.session_id,
        work_scope=engine.config.work_session_id,
        evidence=ToolEvidenceContext(origin=engine._engine_evidence_origin()),
        run_state=engine.run_state,
        metadata=metadata,
    )

    # Buffer events so we can suppress
    # the dispatcher's ``TOOL_CALL_PENDING`` envelope when the web-mode
    # approval kill-switch (``LoopConstants.approval_gate_web_enabled``)
    # is off. The dispatcher yields events BEFORE the final
    # :class:`DispatchOutcome`, so we cannot inspect the verdict without
    # holding them back. The buffer is bounded by the dispatcher contract
    # (at most a single ``HOOK_FIRED(pre_tool_use)`` + a single
    # ``TOOL_CALL_PENDING`` ahead of the approval outcome) and is released
    # in-order on the happy path.
    # SINGLE CHOKE POINT for the tree-budget release-around-child-join. A run
    # holding a tree slot that dispatches a DELEGATION tool blocks on the child's
    # ENTIRE nested run inside ``dispatcher.dispatch`` below — a "permit holder
    # blocked on a descendant", which wedges the tree at the cap unless the slot
    # is released across the join. EVERY serial-style delegation await funnels
    # through :func:`_dispatch_tool` (the single-call serial path AND both
    # truncation-recovery sibling loops), so releasing HERE covers them all at
    # one site. (The >=2 parallel branch releases around its own gather and does
    # NOT pass through here.) Gate: a delegation tool AND this run actually
    # holding a slot; otherwise ``None`` ⇒ no-op for reads / non-permit runs.
    # A BACKGROUND delegation is deliberately excluded. The gate is "a
    # delegation call this run BLOCKS on", and a background spawn returns as
    # soon as its children are launched: there is no join to release around,
    # the parent goes on doing local work, and a permit holder doing local work
    # is exactly what the budget's invariant wants held. Releasing there would
    # hand away a slot the parent is still using and reacquire it a moment
    # later, and the parent would be pinning tree capacity on behalf of a child
    # it is not waiting for.
    dispatch_tree_permit = (
        _resolve_subagent_tree_permit(engine)
        if _tool_is_delegation(engine, tool_call)
        and not _delegation_is_background(engine, tool_call)
        else None
    )
    outcome: DispatchOutcome | None = None
    buffered: list[TurnEvent] = []
    # Release BEFORE the child join. The dispatch loop below only CONSUMES
    # (buffers) events — it never ``yield``s during the child join, so a caller
    # cannot break in mid-join; the only non-normal exits are task cancellation
    # or an exception propagating out of ``dispatcher.dispatch``. On that
    # teardown control never reaches the reacquire after the loop, so the slot
    # is deliberately LEFT
    # released and the parent-final idempotent ``release()`` reconciles it —
    # teardown never blocks on a reacquire that might have no free slot. On
    # NORMAL completion/break the reacquire restores the slot before the
    # local-work post-processing (soft caps, history append) runs.
    if dispatch_tree_permit is not None:
        await dispatch_tree_permit.release_while_waiting()
    # The durable record of this call. Written before anything is dispatched
    # and kept whatever happens next, because after a crash the difference
    # between "parked at a gate", "in flight", "asking the user" and "done" is
    # not recoverable from history alone — history shows only a ``tool_use``
    # with nothing after it, and all four look identical from there.
    intent = find_intent(engine.open_intents, tool_call.id)
    if intent is not None:
        # A record that is already settled means this exact call has an answer.
        # Re-issuing the tool would apply its effect a second time for an
        # answer that is already known, so the recorded one is returned.
        if intent.state == SETTLED:
            yield TurnEvent(
                type=EventType.TOOL_RESULT,
                run_id=engine.config.run_id,
                payload={
                    "tool_call_id": tool_call.id,
                    "content": (
                        intent.result
                        if intent.result is not None
                        else unknown_outcome_text(intent, engine.config.rc)
                    ),
                    "is_error": False,
                },
            )
            return
        # Everything that resumes a parked call arrives here carrying its own
        # copy of what is being resumed. When that copy and the record disagree
        # about which call it is, neither is preferred: one of them describes a
        # call nobody approved, and picking wrong runs it.
        assert_pause_matches(
            intent,
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            arguments=tool_call.arguments,
        )
    else:
        intent = commit_intent(
            tool_name=tool_call.name,
            tool_call_id=tool_call.id,
            rc=engine.config.rc,
            arguments=tool_call.arguments,
            roles=engine.config.tool_roles,
        )
        engine.open_intents.append(intent)
        if engine.config.rc.intent_settlement_enabled:
            yield TurnEvent(
                type=EventType.INTENT_COMMITTED,
                run_id=engine.config.run_id,
                payload=intent.to_dict(),
            )
    # Whether a call may run is decided at this coordinate and nowhere else.
    # It used to sit inside the intent-ledger branch, which meant a host that
    # registered a deny got it only when an unrelated ledger switch happened to
    # be on — a permission that depends on a bookkeeping toggle is not one.
    from protocore.runtime.correctness_bind import fire_lifecycle

    pre_tool, pre_tool_evt = await fire_lifecycle(
        engine,
        HookEvent.pre_tool_use,
        {"tool_name": tool_call.name, "arguments": tool_call.arguments},
    )
    if pre_tool_evt is not None:
        yield pre_tool_evt
    if pre_tool.verdict in (LifecycleVerdict.deny, LifecycleVerdict.fail_run):
        return
    if pre_tool.verdict is LifecycleVerdict.require_approval:
        engine.mark_pending_approval(
            tool_call.id,
            tool_name=tool_call.name,
            payload={"approval_token": pre_tool.approval_token},
        )
        await engine._persist_snapshot()
        yield TurnEvent(
            type=EventType.TOOL_CALL_PENDING,
            run_id=engine.config.run_id,
            payload={
                "tool_call_id": tool_call.id,
                "requires_approval": True,
                "approval_token": pre_tool.approval_token,
            },
        )
        return
    # Serial twin of the charge on the deferred path, and it stands HERE rather
    # than at the top of the dispatch: every seam above can still stop the call
    # — a hook that denies it, a hook that parks it for approval — and a call
    # that never reaches the dispatcher starts no child run to pay for. The
    # parked call comes back through this same function when its approval lands
    # and is charged then, once, because the ledger keys the charge on the call
    # id and the first pass never reached this line.
    #
    # Unlike the deferred path, the gate and this charge are NOT one act here:
    # the seams between them suspend — the tree permit is released across the
    # join and the typed ``before_tool`` hook runs host code — so the budget
    # this call was gated against can be spent by a sibling before the charge
    # lands. That is why the short-grant refusal below is a live branch rather
    # than the impossible one the deferred path used to carry, and why the
    # refusal text comes from the shared builder: at this point the reason may
    # be a SHORT budget, whose advice is "ask for fewer", not "stop".
    serial_grant = _charge_child_run_start(engine, tool_call)
    if serial_grant is not None and not serial_grant.fully_granted:
        _logger.warning(
            "DIAG query.run_work_budget.delegation_refused run=%s tenant=%s "
            "tool=%s reason=%s %s",
            engine.config.run_id,
            engine.config.tenant_id,
            tool_call.name,
            serial_grant.reason,
            _resolve_run_work_ledger(engine).spent_summary(),
        )
        refusal_events, refusal_outcome = _run_work_refusal_dispatch(
            engine,
            tool_call,
            _run_work_refusal_text(engine, tool_call, serial_grant.reason),
            serial_grant.reason,
        )
        for evt in refusal_events:
            yield evt
        engine.history.append(
            Message(
                role=MessageRole.tool,
                content_blocks=[
                    _result_block_from_outcome(tool_call.id, refusal_outcome)
                ],
            )
        )
        engine.forget_tool_name(tool_call.id)
        if dispatch_tree_permit is not None:
            await dispatch_tree_permit.reacquire()
        await engine._persist_snapshot()
        return
    async for item in dispatcher.dispatch(
        tool_call=tool_call,
        ctx=ctx,
        # The effective policy carries the RC core tool-surface floor so the
        # gate permits exactly what was advertised (advertise/dispatch
        # parity; see ToolPermissionGate.check Stage-1 whitelist).
        visibility_policy=engine.effective_tool_policy,
        # The declared tool set of the agent driving THIS engine, when it
        # declared one. Empty declaration ⇒ ``None`` ⇒ the gate's allow-list
        # stage stays off, exactly as before it was wired.
        subagent_whitelist=engine.effective_subagent_tool_allowlist,
        child_run=engine.config.parent_run_id is not None,
        timeout_seconds=engine.config.rc.tool_timeout_seconds,
        preapproved_tool_call_id=tool_call.id if preapproved else None,
        admit_evidence=lambda records, producer: engine.append_tool_evidence(
            records, producer=producer
        ),
        # Make the record durable at the last moment before the tool is
        # touched — past every gate that could still stop the call, and before
        # anything the tool does can be lost with the process.
        on_dispatch_start=_durable_dispatch_start(engine, intent),
        lifecycle=_lifecycle_registry(engine),
    ):
        if isinstance(item, DispatchOutcome):
            outcome = item
            break
        buffered.append(item)
    if outcome is not None and outcome.approval_required:
        # The gate parked the call before the tool was reached. The record must
        # say so, or a resumed run reads "dispatched" and tells the model the
        # outcome of a call that never happened is unknown.
        _assert_pause_envelope_matches_intent(intent, buffered)
        mark_pending_approval(intent)
    elif outcome is not None and outcome.ask_user_required:
        # The tool ran, asked the user something, and is waiting for the
        # answer. That answer is the result, and the layer that collects it
        # builds the result block when it arrives — so this path appends
        # nothing to history and settles nothing. A resumed run reads the
        # record and knows to wait rather than to declare the outcome unknown.
        _assert_pause_envelope_matches_intent(intent, buffered)
        mark_paused_ask_user(intent, pause_payload=outcome.ask_user_payload)
        # The wait is recorded as what it is. Parking it as an approval — which
        # is what a single latch could only ever do — sends the answer to a
        # question to the door that runs unapproved tools, and sends an
        # operator's decision to the door that writes replies into the
        # transcript.
        engine.mark_awaiting_answer(
            tool_call.id,
            tool_name=tool_call.name,
            payload=dict(outcome.ask_user_payload or {}),
        )
        if dispatch_tree_permit is not None:
            await dispatch_tree_permit.reacquire()
        for evt in buffered:
            yield evt
        engine.forget_tool_name(tool_call.id)
        await engine._persist_snapshot()
        return
    elif outcome is not None:
        settle_intent(intent, result=str(outcome.content or "")[:200])
        from protocore.runtime.correctness_bind import (
            commit_usage,
            fire_lifecycle,
            persist_correctness,
        )

        persist_correctness(engine)
        _after_tool, after_tool_evt = await fire_lifecycle(
            engine,
            HookEvent.post_tool_use,
            {"tool_name": tool_call.name, "ok": not bool(getattr(outcome, "is_error", False))},
        )
        if after_tool_evt is not None:
            yield after_tool_evt
        tool_usage = commit_usage(
            engine,
            kind="tool",
            input_tokens=0,
            output_tokens=0,
            success=not bool(getattr(outcome, "is_error", False)),
            operation_id=intent.operation_id,
        )
        if tool_usage is not None:
            yield tool_usage
    if dispatch_tree_permit is not None:
        await dispatch_tree_permit.reacquire()

    if outcome is None:
        # Defensive: dispatcher always yields a DispatchOutcome.
        _logger.warning(
            "tool dispatcher returned no outcome for call_id=%s",
            tool_call.id,
        )
        for evt in buffered:
            yield evt
        engine.forget_tool_name(tool_call.id)
        return

    if (
        outcome.approval_required
        and not engine.config.rc.approval_gate_web_enabled
    ):
        # Approval-gate kill-switch.
        # User constraint (verbatim): "система approval не должна быть
        # включена для web режима, она существует для дальнейшего создания
        # cli инструмента (в web все комманды итак выполняются в
        # изолированном sandbox)". Web-mode sandbox isolation +
        # ``dangerous_commands.py`` deny patterns ARE the safety boundary
        # so we transparently re-dispatch with ``preapproved=True`` and
        # suppress the ``TOOL_CALL_PENDING`` envelope. Forward any other
        # buffered events (e.g. ``HOOK_FIRED(pre_tool_use)``) so telemetry
        # still reflects the hook fire.
        _logger.warning(
            "approval.downgrade run=%s tool=%s reason='web_mode_default_off'",
            engine.config.run_id,
            tool_call.name,
        )
        for evt in buffered:
            if evt.type is EventType.TOOL_CALL_PENDING:
                continue
            yield evt
        # Re-run the dispatcher; the gate honours
        # ``skip_pre_tool_approval`` for this call id so the second pass
        # treats approval as already satisfied and the tool executes. This is the
        # pass that actually runs a web-mode-default-off delegation child, so it
        # gets the SAME release-around-join treatment as the primary loop above
        # (the primary pass reacquired after yielding the approval outcome).
        outcome = None
        buffered = []
        if dispatch_tree_permit is not None:
            await dispatch_tree_permit.release_while_waiting()
        async for item in dispatcher.dispatch(
            tool_call=tool_call,
            ctx=ctx,
            # Effective policy on the approval re-dispatch path too (parity).
            visibility_policy=engine.effective_tool_policy,
            # …and the same declared-tool allow-list, so an approval downgrade
            # cannot be a way past the declaration.
            subagent_whitelist=engine.effective_subagent_tool_allowlist,
            child_run=engine.config.parent_run_id is not None,
            timeout_seconds=engine.config.rc.tool_timeout_seconds,
            preapproved_tool_call_id=tool_call.id,
            admit_evidence=lambda records, producer: engine.append_tool_evidence(
                records, producer=producer
            ),
            on_dispatch_start=_durable_dispatch_start(engine, intent),
            lifecycle=_lifecycle_registry(engine),
        ):
            if isinstance(item, DispatchOutcome):
                outcome = item
                break
            buffered.append(item)
        if dispatch_tree_permit is not None:
            await dispatch_tree_permit.reacquire()
        if outcome is None:
            _logger.warning(
                "tool dispatcher returned no outcome on re-dispatch call_id=%s",
                tool_call.id,
            )
            for evt in buffered:
                yield evt
            engine.forget_tool_name(tool_call.id)
            return

    if outcome.approval_required:
        # Engine state transition + persistence handled by the caller
        # (`_stream_one_assistant_message`) so AWAITING semantics are
        # owned by the outer loop. We just stop here.
        for evt in buffered:
            yield evt
        return

    _activate_rules_from_tool(engine, tool_call)
    newly = list(getattr(engine, "_pending_rules_activated", []) or [])
    if newly:
        engine._pending_rules_activated = []
        yield TurnEvent(
            type=EventType.RULES_ACTIVATED,
            run_id=engine.config.run_id,
            payload={"paths": newly},
        )

    # Preserve the ORIGINAL, un-annotated tool-result body for the longfile
    # byte-parser before any soft-cap annotation runs. The soft-cap warning is
    # appended to ``outcome.content`` as free text, which makes the JSON the
    # model reads invalid; ``_longfile.observe_tool_result`` →
    # ``_parse_byte_result`` ``json.loads``-es this body, so it MUST see the
    # raw JSON or a real Write/AppendFile becomes invisible (frozen tracked
    # size, false stall → spurious forced appends mid-production). This mirrors
    # the host invariant that ``AppendFileOutput.next_step`` rides INSIDE
    # the JSON so the byte-parser stays valid.
    byte_result_content = outcome.content

    # Two advisory soft caps, both warn-only (they NEVER alter dispatch
    # success/error): the per-tool subagent cap (from the Agent tool's
    # ``tool_call_limits``) and the cumulative all-tools cap (the leader's own
    # tool calls, or a subagent's own — chosen by ``parent_run_id``). Collect
    # whichever fired and append each to the result the model reads.
    soft_cap_warnings: list[dict[str, Any]] = []
    per_tool_warning = await _record_tool_call_soft_cap_warning(
        ctx=ctx,
        tool_name=tool_call.name,
    )
    if per_tool_warning is not None:
        soft_cap_warnings.append(per_tool_warning)

    # Ledger the dispatch itself, in transcript order, alongside the counter
    # that shares this seam. Recorded from the DISPATCH rather than read back
    # out of history because compaction rewrites the turn and drops the names.
    engine.record_tool_call(tool_call.name, ok=not outcome.is_error)

    if soft_cap_warnings:
        annotated_content = outcome.content
        annotated_metadata: dict[str, Any] = dict(outcome.metadata or {})
        for _warning in soft_cap_warnings:
            annotated_content = _append_soft_cap_warning_to_content(
                annotated_content,
                _warning,
            )
            annotated_metadata = _merge_soft_cap_warning_metadata(
                annotated_metadata,
                _warning,
            )
        outcome = replace(
            outcome,
            content=annotated_content,
            metadata=annotated_metadata,
        )
        _primary_warning = soft_cap_warnings[-1]
        buffered = [
            _annotate_tool_result_event(
                evt,
                warning=_primary_warning,
                content=annotated_content,
                metadata=annotated_metadata,
            )
            for evt in buffered
        ]

    # Query owns the only ledger mutation point.  It admits the dispatcher-bound
    # evidence before flushing SSE, history, or any success-only state change.
    buffered = _rewrite_deferred_tool_result_events(buffered, outcome)

    # Flush any events that were buffered ahead of the final outcome.
    for evt in buffered:
        yield evt

    # A SUCCESSFUL chunkable write (Write/AppendFile) marks the path
    # "chunking started" so a later repeat truncation of that path gets
    # the "continue with AppendFile" directive (and not before).
    if not outcome.is_error:
        _record_chunk_write_success(engine, tool_call)

    #  — observe the byte production reported by THIS tool
    # result (Write/AppendFile size from the result payload) so the stall
    # detector tracks turns-since-last-byte-adding-mutation + the running file
    # size. No-op when the driver is disabled or the tool added no bytes.
    # Feed the ORIGINAL JSON body (``byte_result_content``), NOT
    # ``outcome.content`` — the latter may carry the soft-cap warning appended
    # as free text, which makes ``json.loads`` fail and silently hides the
    # write from the convergence driver.
    # Thread ``keep_truncated_tail`` so a
    # synthetic-recovery salvage dispatch (the longfile
    # salvage) preserves the truncated-tail flag. The post-``observe_tool_result``
    # persist that captures the dispatch result carries the flag as set by the
    # caller BEFORE the dispatch, so a pod kill between the in-dispatch persist
    # and a hypothetical post-dispatch re-assert cannot land on a half-file
    # with the flag cleared.
    _longfile.observe_tool_result(
        engine,
        tool_call,
        byte_result_content,
        is_error=outcome.is_error,
        keep_truncated=keep_truncated_tail,
    )

    # Fold the result into run-level tool-precondition progress. Fed
    # ``outcome.content`` (not ``byte_result_content``) because the failure
    # reason quotes the error the MODEL saw, warnings and all. A SUCCESSFUL
    # call advances the entry whether it was forced or voluntary.
    _preconditions.observe_tool_result(
        engine, tool_call, outcome.content, is_error=outcome.is_error
    )

    # Fold the result into the declared-file read-back gate. Fed
    # ``outcome.metadata`` — the soft-cap annotation above MERGES into that
    # dict rather than replacing it, so a tool's own declaration survives a
    # warned turn. A result that declares files the caller must open engages
    # the gate; a successful read releases the paths it opened.
    _pending_reads.observe_tool_result(
        engine, tool_call, outcome.metadata, is_error=outcome.is_error
    )

    # Append tool_result block to history + persist snapshot.
    engine.history.append(
        Message(
            role=MessageRole.tool,
            content_blocks=[
                _result_block_from_outcome(tool_call.id, outcome)
            ],
        )
    )
    # Repeated-tool-error circuit breaker. Track the consecutive
    # same-tool/same-error-class streak; once a tool that can never succeed
    # (e.g. a ``/project`` tool on a non-project session) crosses
    # ``max_consecutive_tool_errors``, it is hard-disabled for the rest of the
    # run (via ``effective_tool_policy.blocked``) and ONE bounded corrective
    # convergence turn is injected so the model answers/finalises instead of
    # storming. The synthetic-recovery marker keeps the corrective turn out of
    # the durable transcript (runtime scaffolding) while the model still sees it
    # on the next turn. No-op for every healthy run / single tool error.
    circuit_breaker_corrective = _circuit_breaker_track_and_maybe_trip(
        engine, tool_call, outcome
    )
    if circuit_breaker_corrective is not None:
        engine.history.append(
            Message(
                role=MessageRole.user,
                content_blocks=[TextBlock(text=circuit_breaker_corrective)],
                metadata={
                    SYNTHETIC_RECOVERY_METADATA_KEY: SYNTHETIC_RECOVERY_CIRCUIT_BREAKER
                },
            )
        )
    # The result is in history now, and history is where a settled call is
    # durably recorded. The record has done its work; keeping one per call for
    # the life of the run would grow every snapshot without answering a
    # question history cannot already answer.
    settle_intent(intent, result=str(outcome.content or "")[:200])
    _forget_intent(engine, intent)
    engine.forget_tool_name(tool_call.id)
    await engine._persist_snapshot()


async def resume_interrupts(
    engine: QueryEngine,
    resolutions: Mapping[str, InterruptResolution],
    *,
    allow_partial: bool = False,
) -> AsyncIterator[TurnEvent]:
    """Answer everything this run is waiting on, in one drive.

    The general form of picking a paused run back up, and the only one that can
    express a batch. Three dangerous calls in one assistant message park three
    interrupts and stop the run once; this answers all three — approve the
    first with corrected arguments, deny the second, abandon the third — and
    the results land in history in the order the model asked for them.

    The map is checked in full BEFORE anything is executed. A map that names an
    interrupt this run is not waiting on, or answers an approval with an
    answer, or leaves an open interrupt undecided, is refused with nothing
    done: a resume that ran the first two resolutions and then discovered the
    third was nonsense would have already dispatched tools on the strength of a
    decision set that turned out to be incoherent.

    ``allow_partial`` is how a caller says it means to leave the rest parked —
    an operator who decided two of three and will come back to the last one. It
    relaxes what the map must cover and nothing else: whether the run is still
    waiting at the end is read off what is still parked, so a partial resume
    that happens to answer everything finishes normally, and a full one whose
    approved call parks a fresh interrupt goes back to waiting.

    A cancelled run answers nothing and runs nothing. This entry does not go
    through the turn loop, so it does not pass the stop checkpoint that opens
    one; without its own check, a run an operator cancelled while it waited
    would run the very tool it was waiting on the moment the approval landed.
    """
    plan = plan_resolution(
        engine.pending_interrupts,
        resolutions,
        allow_partial=allow_partial,
        now_ms=int(time.time() * 1000),
    )
    async with engine.driving_turn():
        if engine.stop_requested:
            for interrupt, _ in plan:
                engine.release_interrupt(interrupt.interrupt_id)
            if engine.state is LoopState.AWAITING:
                engine.transition_to(LoopState.RUNNING)
            async for evt in _emit_dispatch_cancel_teardown(engine):
                yield evt
            return

        # The run is being driven again, so it is RUNNING for the length of the
        # drive — whether or not anything will still be parked at the end of
        # it. Deciding the state from the flag instead left a caller that
        # answered every interrupt under ``allow_partial`` driving out of
        # AWAITING, and the closing transition to COMPLETED is not one AWAITING
        # has; the run then died mid-way with its decisions already written and
        # its snapshot already persisted, recording the very shape — AWAITING
        # with nothing open — the witness rule exists to forbid.
        if engine.state is LoopState.AWAITING:
            engine.transition_to(LoopState.RUNNING)
            await engine._persist_snapshot()

        for interrupt, resolution in plan:
            # Released before the decision is acted on, not after. Executing an
            # approved call re-enters the dispatch path, and a dispatch that
            # still saw this call parked would read the run as waiting for the
            # very thing it is in the middle of doing.
            engine.release_interrupt(interrupt.interrupt_id)

            if resolution.decision is InterruptDecision.abandon:
                _abandon_pending_approval(engine, interrupt.tool_call_id)
                continue

            if resolution.decision is InterruptDecision.deny:
                _settle_parked_call(
                    engine,
                    tool_call_id=interrupt.tool_call_id,
                    content=engine.config.rc.tool_result_approval_denied_placeholder,
                    is_error=False,
                )
                continue

            if resolution.decision is InterruptDecision.answer:
                # The answer IS the result of the call that asked for it. It is
                # written where the tool's own result would have gone, so the
                # model reads a question it asked and the reply it got, rather
                # than an unexplained user turn arriving beside an unpaired
                # call.
                _settle_parked_call(
                    engine,
                    tool_call_id=interrupt.tool_call_id,
                    content=resolution.answer or "",
                    is_error=False,
                )
                continue

            if _history_has_tool_result(engine, interrupt.tool_call_id):
                # Already answered by a previous resume that got this far and
                # then lost its process. Re-running the tool here would apply
                # its effect a second time for a result already in history.
                continue
            tool_call = _tool_call_from_history(engine, interrupt.tool_call_id)
            if resolution.updated_input is not None:
                tool_call = _apply_updated_input(
                    engine, tool_call, resolution.updated_input
                )
            async for evt in _dispatch_tool(engine, tool_call, preapproved=True):
                yield evt

        if engine.pending_interrupts:
            # Still waiting, and the state has to keep saying so — a run left
            # in RUNNING with interrupts open is a run nothing will come back
            # for. What is still parked decides this, not what the caller
            # asked for: a partial resume that happened to answer everything is
            # finished, and a full resume whose approved call parked a fresh
            # interrupt is not.
            if engine.state is LoopState.RUNNING:
                engine.transition_to(LoopState.AWAITING)
            await engine._persist_snapshot()
            yield TurnEvent(
                type=EventType.MESSAGE_STOP,
                run_id=engine.config.run_id,
                payload={
                    "turn_id": engine.turn_id(),
                    "stop_reason": "tool_use",
                    "tokens_used": _tokens_used_payload(engine),
                    "cache_hit_rate": engine.total_usage.this_turn_cache_hit_rate(),
                },
            )
            return
        await engine._persist_snapshot()
        yield TurnEvent(
            type=EventType.MESSAGE_STOP,
            run_id=engine.config.run_id,
            payload={
                "turn_id": engine.turn_id(),
                "stop_reason": "tool_use",
                "tokens_used": _tokens_used_payload(engine),
                "cache_hit_rate": engine.total_usage.this_turn_cache_hit_rate(),
            },
        )
        if not engine.is_terminal:
            engine.transition_to(LoopState.COMPLETED)


async def resume_approved_tool(
    engine: QueryEngine,
    tool_call: ToolCall,
) -> AsyncIterator[TurnEvent]:
    """Execute one previously-pending, now-approved tool call.

    The live approval route records approval outside core, while the engine
    has already returned from ``run()`` in ``AWAITING`` state. This explicit
    resume path verifies that the requested call exactly matches a durable
    pending ``ToolUseBlock``, skips only the already-approved pre-tool gate for
    that call id, then uses the normal dispatcher execution/post-hook/result
    path so the real ``ToolResultBlock`` lands in history before finalization.

    It resolves exactly the one call it is given, and one of a batch is a
    legal thing to be given. An assistant message with three gated calls parks
    three approvals; approving the second of them runs the second and leaves
    the other two parked, with the run back in ``AWAITING`` because it is
    still waiting for the decisions nobody has made yet. Answering several at
    once — approve, deny, correct — is what :func:`resume_interrupts` is for;
    this entry is the single-decision case of the same thing and agrees with
    it about what a partially answered run looks like afterwards.

    Replays are idempotent: if a matching ``ToolResultBlock`` already exists,
    the tool is not invoked again and no duplicate result is appended.

    A cancelled run does not execute its pending call. This entry does not go
    through the turn loop, so it does not pass the stop checkpoint that opens
    one; without its own check, a run an operator cancelled while it waited for
    an approval would run the very tool it was waiting on the moment the
    approval landed — a command, a write, a whole delegated subtree — after the
    cancel had been recorded and restored.
    """
    async with engine.driving_turn():
        if _history_has_tool_result(engine, tool_call.id):
            engine.clear_pending_approval(tool_call.id)
            return

        if engine.stop_requested:
            engine.clear_pending_approval(tool_call.id)
            if engine.state is LoopState.AWAITING:
                engine.transition_to(LoopState.RUNNING)
            async for evt in _emit_dispatch_cancel_teardown(engine):
                yield evt
            return

        _assert_history_has_matching_pending_tool_use(engine, tool_call)

        if engine.state is LoopState.AWAITING:
            engine.transition_to(LoopState.RUNNING)
            await engine._persist_snapshot()

        async for evt in _dispatch_tool(engine, tool_call, preapproved=True):
            yield evt

        engine.clear_pending_approval(tool_call.id)
        if engine.pending_interrupts:
            # This approval was one of a batch, and the rest are still waiting
            # for a person. The run goes back to AWAITING rather than on to
            # COMPLETED: a run left RUNNING with interrupts open is a run
            # nothing will come back for, and completing it would abandon the
            # calls still parked behind it without saying so. Read off what is
            # still parked, exactly as the resolution-map drive reads it.
            if engine.state is LoopState.RUNNING:
                engine.transition_to(LoopState.AWAITING)
            await engine._persist_snapshot()
            yield TurnEvent(
                type=EventType.MESSAGE_STOP,
                run_id=engine.config.run_id,
                payload={
                    "turn_id": engine.turn_id(),
                    "stop_reason": "tool_use",
                    "tokens_used": _tokens_used_payload(engine),
                    "cache_hit_rate": engine.total_usage.this_turn_cache_hit_rate(),
                },
            )
            return
        yield TurnEvent(
            type=EventType.MESSAGE_STOP,
            run_id=engine.config.run_id,
            payload={
                "turn_id": engine.turn_id(),
                "stop_reason": "tool_use",
                "tokens_used": _tokens_used_payload(engine),
                "cache_hit_rate": engine.total_usage.this_turn_cache_hit_rate(),
            },
        )
        if not engine.is_terminal:
            engine.transition_to(LoopState.COMPLETED)


def _history_has_tool_result(engine: QueryEngine, tool_call_id: str) -> bool:
    for message in engine.history:
        for block in message.content_blocks:
            if isinstance(block, ToolResultBlock) and block.tool_call_id == tool_call_id:
                return True
    return False


def _parsed_arguments(arguments_json: str) -> object:
    """The values a stored ``tool_use`` block stands for, not its bytes.

    History keeps the arguments as the text the model emitted; a host that
    carries an approval back through its own transport re-serialises them, and
    a transport that sorts keys (or drops insignificant whitespace) hands back
    the same call spelled differently. Comparing the spellings refuses a call
    nobody changed, so the comparison is made on parsed values. Text that is
    not JSON at all has no values to compare and stands for itself.
    """
    try:
        return json.loads(arguments_json or "{}")
    except (TypeError, ValueError):
        return arguments_json


def _assert_history_has_matching_pending_tool_use(
    engine: QueryEngine,
    tool_call: ToolCall,
) -> None:
    """Refuse a call this run is not holding an approval for.

    The membership test is against the whole parked set, not against "the"
    pending approval. One assistant message can park three gated calls at
    once, and a host that approves the second of them is answering a decision
    this run really is waiting on; reading the batch through a view that
    reports a single id says ``None`` whenever more than one is parked, and
    the honest approval was then refused as a mismatch. What stays refused is
    a call that is not parked at all, or one parked as a different kind of
    wait — approving a question runs a tool whose answer, not whose approval,
    was being waited for.

    The arguments are compared as canonical values — the same digest the
    durable intent record carries — and never as serialized text. A host that
    round-trips the approved call through its own transport is free to
    re-serialise it (sorted keys, different spacing); that is a spelling of
    the same call, and refusing it strands a run whose approval was honest.
    An argument that really differs still changes the digest and is still
    refused.
    """
    parked = find_interrupt_for_call(engine.pending_interrupts, tool_call.id)
    if parked is None or parked.kind is not InterruptKind.approval:
        expected = [
            item.tool_call_id
            for item in interrupts_of_kind(engine.pending_interrupts, InterruptKind.approval)
        ]
        raise ValueError(
            f"approved tool call is not a parked approval: expected one of {expected!r}, got {tool_call.id!r}"
        )
    expected_fingerprint = _intent_fingerprint(tool_call.name, tool_call.arguments)
    for message in engine.history:
        for block in message.content_blocks:
            if not isinstance(block, ToolUseBlock):
                continue
            if block.tool_call_id != tool_call.id:
                continue
            if block.name != tool_call.name:
                raise ValueError(f"approved tool call does not match pending tool name: {tool_call.id}")
            if _intent_fingerprint(block.name, _parsed_arguments(block.arguments_json)) != (
                expected_fingerprint
            ):
                raise ValueError(f"approved tool call does not match pending tool input: {tool_call.id}")
            return
    raise ValueError(f"approved tool call is not pending in history: {tool_call.id}")


def _tokens_used_payload(engine: QueryEngine) -> dict[str, int]:
    """Build the ``tokens_used`` block surfaced in ``message_stop``.

    Per the prompt-caching playbook recommendation #1, includes the
    cache split (cache_read / cache_creation) so downstream metrics
    (Prometheus / dashboard) can compute hit rate without re-reading
    the run state.
    """
    usage = engine.total_usage
    return {
        "input": usage.this_turn_input,
        "output": usage.this_turn_output,
        "total": usage.this_turn_total(),
        "cache_read": usage.this_turn_cache_read,
        "cache_creation": usage.this_turn_cache_creation,
    }


def _prepend_system_sections(
    sections: tuple[str, ...],
    messages: tuple[Message, ...],
) -> list[Message]:
    """Concatenate sections into a single system :class:`Message` prefix.

    Returns ``list(messages)`` unchanged when ``sections`` is empty so the
    test invariant "no system block added unless there is content" holds.
    Sections are joined with a blank line to preserve readability across
    the skill-index, loaded-skill, and operator-supplied blocks.
    """
    if not sections:
        return list(messages)
    body = "\n\n".join(s for s in sections if s)
    if not body:
        return list(messages)
    system_msg = Message(
        role=MessageRole.system,
        content_blocks=[TextBlock(text=body)],
    )
    return [system_msg, *messages]


# ----------------------------------------------------------------------
# tool_use <-> tool_result pairing repair
# ----------------------------------------------------------------------
#
# Enforces pairing UNCONDITIONALLY at the API boundary AND synthesises
# missing tool_results on abnormal turn exit. Anthropic / OpenAI / vLLM
# all reject a request whose assistant ``tool_use`` has no matching
# ``tool_result`` (or a ``tool_result`` with no ``tool_use``, or duplicate
# ids) with HTTP 400.
#
# Protocore's wire model: a ``tool_use`` is a :class:`ToolUseBlock` on an
# ``assistant`` message; a ``tool_result`` is a :class:`ToolResultBlock`
# on a ``tool``-role message (one block per message). The pairing key is
# ``tool_call_id`` on both sides.


def _repair_outbound_tool_pairing(
    messages: list[Message],
    *,
    placeholder: str,
) -> list[Message]:
    """Return a pairing-valid, ADJACENCY-correct copy of ``messages`` .

    The wire-boundary backstop, run UNCONDITIONALLY on the outbound message
    list right before :class:`LLMRequest` assembly — independent of whether
    compaction ran this turn. It repairs orphaning from ANY source (Tier-2
    compaction dropping one side, resume-from-partial-batch, max_tokens
    truncation, a teardown that appended a result out of position). Four
    repairs:

    1. **Forward-fill** — for an assistant ``ToolUseBlock`` whose
       ``tool_call_id`` has no matching ``ToolResultBlock`` anywhere in the
       list, emit a synthetic ``is_error=True`` tool-role
       :class:`ToolResultBlock` (content ``placeholder``) immediately after
       the orphan's assistant message.
    2. **Reposition** — every real ``ToolResultBlock`` is emitted DIRECTLY
       after the assistant message that carries its ``tool_use`` (in
       tool_use order). The Anthropic wire requires the result to be the
       immediately-following turn; a result that drifted out of position
       (e.g. a teardown that appended it after an intervening user/recovery
       message) would otherwise still 400. The OpenAI wire is id-keyed and
       tolerates either, so repositioning is safe for both.
    3. **Reverse-strip** — drop any ``ToolResultBlock`` whose
       ``tool_call_id`` has no matching ``ToolUseBlock`` (and any standalone
       tool message left empty after its block was repositioned).
    4. **Dedupe** — keep only the first occurrence of each ``tool_use`` id
       and each ``tool_result`` id (the CC-1212 duplicate-id deadlock).

    Pure function: ``messages`` is not mutated; a new list of (possibly
    new) :class:`Message` objects is returned. ``Message`` /
    :class:`ToolResultBlock` are frozen, so unchanged messages are reused
    by reference.
    """
    # Pass 1 — index every tool_use id (first occurrence) and the FIRST
    # real ToolResultBlock seen per id (anywhere in the list), so pass 2 can
    # reposition the real result directly after its tool_use regardless of
    # where it currently sits. Later duplicate results are dropped.
    tool_use_ids: set[str] = set()
    result_block_by_id: dict[str, ToolResultBlock] = {}
    for message in messages:
        for block in message.content_blocks:
            if isinstance(block, ToolUseBlock):
                tool_use_ids.add(block.tool_call_id)
            elif isinstance(block, ToolResultBlock):
                if block.tool_call_id not in result_block_by_id:
                    result_block_by_id[block.tool_call_id] = block

    result: list[Message] = []
    seen_tool_use_ids: set[str] = set()
    # tool_use ids whose result we have already emitted (repositioned real or
    # synthetic) so a standalone tool message / duplicate never re-emits it.
    emitted_result_ids: set[str] = set()

    for message in messages:
        if message.role is MessageRole.assistant:
            # Dedupe duplicate tool_use blocks within / across assistant
            # turns; preserve the FIRST occurrence.
            new_blocks: list[ContentBlock] = []
            kept_tool_use_ids: list[str] = []
            changed = False
            for block in message.content_blocks:
                if isinstance(block, ToolUseBlock):
                    if block.tool_call_id in seen_tool_use_ids:
                        changed = True
                        continue
                    seen_tool_use_ids.add(block.tool_call_id)
                    kept_tool_use_ids.append(block.tool_call_id)
                new_blocks.append(block)
            if changed:
                # An assistant turn that was ALL duplicate tool_use would now
                # be empty — drop it (its surviving sibling already carries
                # the id). Otherwise rebuild with the surviving blocks.
                if not new_blocks:
                    continue
                message = message.model_copy(update={"content_blocks": new_blocks})
            result.append(message)
            # Emit each kept tool_use's result immediately after this turn,
            # in tool_use order: the repositioned real result if one exists
            # anywhere, else a synthetic forward-fill.
            for call_id in kept_tool_use_ids:
                if call_id in emitted_result_ids:
                    continue
                real = result_block_by_id.get(call_id)
                block_to_emit = (
                    real
                    if real is not None
                    else ToolResultBlock(
                        tool_call_id=call_id,
                        content=placeholder,
                        is_error=True,
                    )
                )
                result.append(
                    Message(
                        role=MessageRole.tool,
                        content_blocks=[block_to_emit],
                    )
                )
                emitted_result_ids.add(call_id)
            continue

        # Non-assistant message — strip every ToolResultBlock (real results
        # are re-emitted in position above; orphaned/duplicate ones are
        # dropped). Tool messages carry exactly one block today, but iterate
        # defensively in case that changes; non-result blocks are preserved.
        stripped_blocks: list[ContentBlock] = []
        changed = False
        for block in message.content_blocks:
            if isinstance(block, ToolResultBlock):
                changed = True
                continue
            stripped_blocks.append(block)
        if not changed:
            result.append(message)
            continue
        if not stripped_blocks:
            # Whole message was tool_result(s) — already re-emitted / dropped.
            continue
        result.append(message.model_copy(update={"content_blocks": stripped_blocks}))

    return result


def _normalize_outbound_system_messages(
    messages: list[Message],
) -> tuple[list[Message], int]:
    """Convert every non-leading ``system`` message to ``user`` role (vLLM-400).

    vLLM (and several OpenAI-compatible servers) reject any request whose
    message array carries a ``system`` message at an index OTHER than 0 with
    HTTP 400 ``"System message must be at the beginning."``. The genuine system
    prefix produced by :func:`_prepend_system_sections` always sits at index 0
    and is left untouched; any system message AFTER it — historically a Tier-2
    compaction summary (``context/compaction.run_tier2_summarisation``), now
    fixed at source to be USER-role, but legacy persisted snapshots may still
    rehydrate one mid-array — is converted to a ``user``-role copy with the SAME
    content blocks + metadata preserved (so the
    ``COMPACTION_SUMMARY_METADATA_KEY`` flag and ``<compacted-turn>`` wrapper
    survive and downstream summary recognition keeps working).

    Defense-in-depth backstop, run UNCONDITIONALLY at the request-assembly
    boundary right after :func:`_repair_outbound_tool_pairing`. Pure function:
    ``messages`` is not mutated; unchanged
    messages are reused by reference (``Message`` is frozen). Returns the new
    list + the count of converted messages so the caller can log once per run.
    """
    converted = 0
    out: list[Message] = []
    for idx, message in enumerate(messages):
        if idx != 0 and message.role is MessageRole.system:
            # system/user share the "at most one content block" validator, so a
            # role flip is always valid here.
            out.append(message.model_copy(update={"role": MessageRole.user}))
            converted += 1
        else:
            out.append(message)
    return out, converted


def _synthesize_missing_tool_results(
    history: list[Message],
    *,
    error_content: str,
) -> int:
    """Insert synthetic ``is_error`` tool_results for orphaned tool_use .

    Mutates ``history`` in place so a cancel / LLM-error teardown leaves a
    pairing-valid AND ordered persisted snapshot: every assistant
    ``ToolUseBlock`` whose ``tool_call_id`` has no matching
    ``ToolResultBlock`` gets a tool-role :class:`ToolResultBlock`
    (content ``error_content``, ``is_error=True``) inserted IMMEDIATELY after
    the assistant message that carries the orphan — not appended at the tail,
    which (if an intervening user/recovery message already follows the
    orphan) would persist an out-of-order pair the Anthropic wire rejects.
    Ensures an interrupted/crashed run rehydrated on another pod does not
    replay a
    dangling tool_use into a provider 400.

    Idempotent: a tool_use already paired (real OR a prior synthetic result
    anywhere in history) is skipped, so calling this on every teardown path
    never double-inserts. Returns the number of synthetic results inserted
    (0 when nothing to do); ``history`` is left untouched (same objects) when
    the return is 0.
    """
    resolved_ids: set[str] = set()
    for message in history:
        for block in message.content_blocks:
            if isinstance(block, ToolResultBlock):
                resolved_ids.add(block.tool_call_id)

    # Walk in order; immediately after each assistant message, insert a
    # synthetic result for every orphaned tool_use id it introduces (in
    # first-occurrence order). ``resolved_ids`` accumulates so a duplicate
    # orphan id across turns is paired exactly once.
    rebuilt: list[Message] = []
    inserted = 0
    for message in history:
        rebuilt.append(message)
        if message.role is not MessageRole.assistant:
            continue
        for block in message.content_blocks:
            if (
                isinstance(block, ToolUseBlock)
                and block.tool_call_id not in resolved_ids
            ):
                resolved_ids.add(block.tool_call_id)
                rebuilt.append(
                    Message(
                        role=MessageRole.tool,
                        content_blocks=[
                            ToolResultBlock(
                                tool_call_id=block.tool_call_id,
                                content=error_content,
                                is_error=True,
                            )
                        ],
                    )
                )
                inserted += 1

    if inserted:
        history[:] = rebuilt
    return inserted


# ----------------------------------------------------------------------
# Skill catalog (built once per run) + <command-name> triggers (per turn)
# ----------------------------------------------------------------------

# Trigger pattern in user-authored text: ``<command-name>NAME</command-name>``
# (case-sensitive name match against the skill name).
_COMMAND_NAME_PATTERN = re.compile(
    r"<command-name>([^<\s][^<]*?)</command-name>",
    re.IGNORECASE,
)


async def _ensure_run_skill_catalog(engine: QueryEngine) -> str:
    """Return the run's skill catalog block, building it at most ONCE per run.

    The catalog is a ``<system-reminder>`` listing the account's ENABLED
    skills (plus any project pins) as ``name: description`` lines,
    alphabetical. Over-budget → deterministic names-only degrade, decided
    once here. Empty string when no skills resolve or no store is wired.

    The result is cached on ``engine._skill_catalog_block`` (a per-run
    sentinel, ``None`` until built) so the ``store.list`` + render +
    token-count cost is paid only on the first turn of a run; later turns
    reuse the byte-identical block — which both preserves the cached
    system-prompt prefix and avoids a redundant DB/LLM round-trip every
    turn. If the enabled-skill set changes mid-run the run keeps turn-1's
    catalog (acceptable per the once-per-run intent — no invalidation).

    Failures on any step are isolated with a WARNING log; the run continues
    without the catalog rather than failing.
    """
    if (
        engine._skill_catalog_block is not None
        and not engine.config.rc.skills_hot_reload_enabled
    ):
        return engine._skill_catalog_block

    store = engine.skills
    if store is None:
        engine._skill_catalog_block = ""
        _report_skill_catalog_drift(engine, "")
        return ""

    rc = engine.config.rc

    try:
        entries = list(await store.list(engine.config.account_id))
    except Exception:
        _logger.warning(
            "DIAG skill_catalog.list_failed run_id=%s",
            engine.config.run_id,
            exc_info=True,
        )
        entries = []

    # Force-include project pins even if the enabled-list dropped them.
    entries = await _merge_pinned_skills(engine, store, entries)

    budget_tokens = derive_skill_index_budget_tokens(
        model_context_window=rc.model_context_window,
        skill_index_budget_ratio=rc.skill_index_budget_ratio,
    )

    # Token counter — use the engine's LLM provider if available.
    async def _count(text: str) -> int:
        try:
            return int(engine.llm.count_tokens(text))
        except Exception:
            # Conservative heuristic: 4 chars/token (Latin-prose baseline).
            return max(1, len(text) // 4)

    try:
        block = await render_skills_catalog(
            entries,
            token_counter=_count,
            budget_tokens=budget_tokens,
        )
    except Exception:
        _logger.warning(
            "DIAG skill_catalog.render_failed run_id=%s",
            engine.config.run_id,
            exc_info=True,
        )
        block = ""

    engine._skill_catalog_block = block
    _report_skill_catalog_drift(engine, block)
    return block


def _report_skill_catalog_drift(engine: QueryEngine, block: str) -> None:
    """Say so when a resumed run rebuilds a different catalog block.

    The block is the head of the cached prompt prefix, and it is rebuilt from
    the store on whichever process picks the run up. If the enabled-skill set
    moved between the two, the new block differs, the cached prefix is invalid
    for the rest of the run, and nothing about that is visible: the run simply
    costs more from here on. The digest the previous process recorded is the
    only thing that can tell, so it is compared once, here, and the answer is
    logged either way it goes wrong.
    """
    expected = engine._resumed_skill_catalog_sha256
    engine._resumed_skill_catalog_sha256 = None
    if expected is None:
        return
    rebuilt = hashlib.sha256(block.encode("utf-8")).hexdigest()
    if rebuilt == expected:
        return
    _logger.warning(
        "DIAG skill_catalog.rebuilt_differently run_id=%s expected=%s rebuilt=%s; "
        "the cached prompt prefix is invalid for the rest of this run",
        engine.config.run_id,
        expected,
        rebuilt,
    )


async def _merge_pinned_skills(
    engine: QueryEngine,
    store: ISkillStore,
    entries: list[SkillIndexEntry],
) -> list[SkillIndexEntry]:
    """Force-include project-pinned skills into the catalog entry list.

    A project's pinned skills are surfaced-by-default: the enabled-skill list
    is the catalog baseline, and each pin that is missing from it is fetched
    and appended (one :meth:`ISkillStore.list_enabled_subset` round-trip for
    the missing names).

    Pin = surfaced, NOT a visibility restriction: existing entries are
    returned unchanged. A missing/unknown pin (deleted skill) is silently
    skipped — surfacing is best-effort and never fails the run. Empty pin set
    (the common, non-project path) returns ``entries`` untouched.

    The MISSING-pin fetch uses :meth:`ISkillStore.list_enabled_subset` (not
    the whitelist-oriented ``list_subset``, which ignores the ``enabled``
    flag): a skill the operator disabled in the account-wide bank stays OFF
    the leader's catalog even when a project still holds a stale pin for its
    name (disable gates beat pins). ``store.list`` already returns only
    enabled skills, so an already-present entry is enabled by construction.
    """

    pinned_names = engine.config.pinned_skill_names
    if not pinned_names:
        return entries

    present = {entry.name for entry in entries}
    missing = [name for name in pinned_names if name not in present]
    if not missing:
        return entries

    try:
        fetched = await store.list_enabled_subset(engine.config.account_id, missing)
    except Exception:
        _logger.warning(
            "DIAG skill_catalog.pinned_subset_failed run_id=%s",
            engine.config.run_id,
            exc_info=True,
        )
        return entries

    pinned_set = set(pinned_names)
    seen_missing: set[str] = set()
    forced: list[SkillIndexEntry] = []
    for entry in fetched:
        if entry.name not in pinned_set or entry.name in seen_missing:
            continue
        seen_missing.add(entry.name)
        forced.append(entry)
    return entries + forced


async def _load_triggered_skill_bodies(
    engine: QueryEngine,
    store: ISkillStore,
    user_text: str,
) -> list[SkillBundle]:
    """Match ``<command-name>NAME</command-name>`` references → load bodies.

    Rebuilt EVERY turn from the current turn's user message (NOT cached on
    the engine like the run-stable catalog block) so a trigger in a later
    turn still force-loads its skill body. Honors the same
    ``max_skills_per_run`` cap as the manifest-driven loaded skills; missing
    or disabled skills are silently skipped.

    Resolution order:
      1. ``store.load(account_id, name)`` — works for stores that accept a
         name or UUID directly.
      2. Fallback to ``list_subset`` + UUID lookup — covers stores where
         ``load`` requires a UUID (e.g. InMemorySkillStore).
    """
    if not user_text:
        return []
    matches = _COMMAND_NAME_PATTERN.findall(user_text)
    if not matches:
        return []

    # Dedup while preserving order.
    seen: set[str] = set()
    unique: list[str] = []
    for name in matches:
        normalised = name.strip()
        if not normalised or normalised in seen:
            continue
        seen.add(normalised)
        unique.append(normalised)

    rc = engine.config.rc
    out: list[SkillBundle] = []
    for skill_name in unique[: rc.max_skills_per_run]:
        bundle = await _resolve_skill_bundle(engine, store, skill_name)
        if bundle is not None:
            out.append(bundle)
    return out


async def _resolve_skill_bundle(
    engine: QueryEngine,
    store: ISkillStore,
    skill_name: str,
) -> SkillBundle | None:
    """Resolve a ``<command-name>`` reference to a :class:`SkillBundle`.

    Tries ``store.load`` first (works for stores that accept a name or
    UUID) then falls back to ``list_subset`` + UUID-based ``load``. Failures
    are logged and yield ``None`` (the loop drops the trigger silently).

    The skill bank is account-wide, so every lookup keys on
    ``config.account_id`` — NOT ``tenant_id``, which is the run's scope and is
    a different key wherever a host keeps more than one scope per account. This
    is the
    skill-chaining path (e.g. web → frontend-design), so a
    scope-keyed lookup here would silently drop every chained skill body.
    """
    account_id = engine.config.account_id

    # Path A: direct load. Most stores accept the bare name or a UUID.
    try:
        return await store.load(account_id, skill_name)
    except SkillNotFoundError:
        pass
    except KeyError:
        # InMemorySkillStore raises KeyError on dict miss.
        pass
    except Exception:
        _logger.warning(
            "DIAG skill_index.trigger_load_failed name=%s account_id=%s",
            skill_name,
            account_id,
            exc_info=True,
        )
        return None

    # Path B: resolve via list_subset → fetch by id.
    try:
        entries = await store.list_subset(account_id, [skill_name])
    except Exception:
        _logger.warning(
            "DIAG skill_index.trigger_subset_failed name=%s account_id=%s",
            skill_name,
            account_id,
            exc_info=True,
        )
        return None

    if not entries:
        _logger.warning(
            "DIAG skill_index.trigger_not_found name=%s account_id=%s",
            skill_name,
            account_id,
        )
        return None

    try:
        return await store.load(account_id, entries[0].id)
    except Exception:
        _logger.warning(
            "DIAG skill_index.trigger_load_by_id_failed name=%s account_id=%s",
            skill_name,
            account_id,
            exc_info=True,
        )
        return None


def _ensure_tool_dispatcher(engine: QueryEngine) -> ToolDispatcher:
    """Lazily build a :class:`ToolDispatcher` for ``engine`` if absent.

 Engines instantiated before the dispatcher API existed (or in tests
 that bypass engine factories) won't carry a dispatcher reference.
 We construct a default one bound to the engine's registry +
 hook manager using the default :class:`ToolPermissionGate` chain.

 When the run carries a ``tool_error_counter``
 (a :class:`~protocore.contracts.run.IRunToolErrorCounter`), pass it through
 so every dispatch error path increments the active run's error count. A run
 wired without one gets a counter-less dispatcher: no telemetry, behaviour
 unchanged.
 """
    existing = getattr(engine, "_tool_dispatcher", None)
    if isinstance(existing, ToolDispatcher):
        return existing
    dispatcher = ToolDispatcher(
        registry=engine.tools,
        permission_gate=ToolPermissionGate(roles=engine.config.tool_roles),
        hook_manager=engine.hooks,
        tool_error_counter=engine.run_state.tool_error_counter,
        roles=engine.config.tool_roles,
        resilience_classifier=engine.config.resilience_classifier,
    )
    engine._tool_dispatcher = dispatcher  # type: ignore[attr-defined]
    return dispatcher


def _lifecycle_registry(engine: QueryEngine) -> ILifecycleRegistry | None:
    """The registry the tool dispatcher wraps its invocation in, if any."""
    from protocore.runtime.correctness_bind import lifecycle_registry

    return lifecycle_registry(engine)


async def _safe_hook_invoke(
    engine: QueryEngine,
    event: HookEvent,
    payload: dict[str, object],
) -> HookResult:
    """Invoke hooks; isolate failures (logged WARNING, treated as allow)."""
    from protocore.contracts.hooks import HookResult as _HookResult

    try:
        return await engine.hooks.invoke(event, payload, engine.config.tenant_id)
    except Exception:
        _logger.warning(
            "hook invoke raised for event=%s; isolating",
            event.value,
            exc_info=True,
        )
        return _HookResult(action=HookActionKind.ALLOW, reason="hook dispatch failed")


async def _as_provider_deltas(
    upstream: AsyncIterator[ProviderDelta | LLMStreamEvent],
) -> AsyncIterator[ProviderDelta]:
    """Adapt either a :class:`ProviderDelta` or :class:`LLMStreamEvent` stream.

 The vLLM adapter (the host) emits :class:`ProviderDelta` natively.
 The in-memory mock emits :class:`LLMStreamEvent` — translate
 via :func:`stream_events_to_provider_deltas`.
 """
    # Peek the first item to decide. `upstream` may be either an async
    # generator (synchronous call returning AsyncIterator) or an awaitable
    # returning an AsyncIterator. Normalise.
    if inspect.iscoroutine(upstream):
        upstream = await upstream

    first = None
    async for item in upstream:
        first = item
        break

    if first is None:
        return

    if isinstance(first, ProviderDelta):
        yield first
        async for item in upstream:
            if isinstance(item, ProviderDelta):
                yield item
        return

    if isinstance(first, LLMStreamEvent):
        first_evt: LLMStreamEvent = first

        # Chain the first event back into a synthetic stream.
        async def _chain() -> AsyncIterator[LLMStreamEvent]:
            yield first_evt
            async for tail_item in upstream:
                if isinstance(tail_item, LLMStreamEvent):
                    yield tail_item

        async for delta in stream_events_to_provider_deltas(_chain()):
            yield delta
        return

    # Unknown stream item — log + bail.
    _logger.warning("unknown LLM stream item type: %s", type(first).__name__)


# ---------------------------------------------------------------------------
# AdaptiveSafetyBand helpers
# ---------------------------------------------------------------------------


def _resolve_safety_band_value(engine: QueryEngine) -> int:
    """Read the current AdaptiveSafetyBand value off the run's state.

    Returns 0 when:
      - ``LoopConstants.adaptive_safety_band_enabled`` is False (kill-switch).
      - The run carries no band (test fixture / leader engine without the
        host wiring).
      - The band lookup raises (defensive — telemetry plane must never
        block the LLM call).
    """
    rc = engine.config.rc
    if not rc.adaptive_safety_band_enabled:
        return 0
    band = engine.run_state.adaptive_safety_band
    if band is None:
        return 0
    try:
        current = int(band.current())
    except Exception:
        _logger.warning(
            "adaptive_safety_band.current() raised — assuming band=0",
            exc_info=True,
        )
        return 0
    return max(0, current)


__all__ = ["resume", "resume_approved_tool", "resume_interrupts"]


#: The core's own policy set, in the order :data:`TURN_POLICY_ORDER` fixes.
#: Built here rather than in the registry module because a policy is handed the
#: loop primitives it borrows — the dispatcher, the event emitters — and those
#: live in this module.
_CORE_TURN_POLICIES: Final[TurnPolicyRegistry] = TurnPolicyRegistry(
    (
        LongFileConvergencePolicy(
            seal=_maybe_seal_longfile_at_voluntary_finish,
            drive=_maybe_drive_longfile_convergence,
        ),
        AnswerFloorPolicy(
            applies=_plain_stop_answer_floor_applies,
            pointer=_pointer_answer_evidence,
            charge_pointer=_charge_pointer_answer_repair,
            release_pointer=_release_pointer_answer_repair,
            spend_short_answer_repair=_spend_short_answer_repair,
            append_repair=_append_answer_floor_repair_turn,
            log=_log_answer_floor_repair,
            state_change=_policy_state_change,
        ),
        RunCeilingsPolicy(
            tool_call_budget_reached=_tool_call_budget_reached,
            deadline_reached=_terminal_deadline_reached,
            has_terminal_tool_result=_history_has_terminal_tool_result,
            has_final_answer=run_has_final_answer,
            enter_wind_down=_enter_soft_stop,
            wind_down_budget=_soft_stop_turn_budget,
            llm_terminal=_emit_llm_terminal,
            precondition_exhausted=_preconditions.is_exhausted,
            precondition_terminal=_emit_tool_precondition_terminal,
            wind_down_armed=_soft_stop.is_armed,
            wind_down_finalize=_soft_stop.finalize,
            pair_orphans=_policy_pair_orphan_tool_calls,
            transition_event=_emit_state_change,
            message_stop=_policy_message_stop,
            log_output_budget_exhausted=_log_output_token_budget_exhausted,
        ),
        EmptyModelTurnPolicy(
            empty_rounds=RunCounter(
                read=_empty_rounds_spent,
                charge=_charge_empty_round,
                reset=_reset_empty_rounds,
            ),
            post_tool_nudges=RunCounter(
                read=_post_tool_nudges_spent,
                charge=_charge_post_tool_nudge,
                reset=_reset_post_tool_nudges,
            ),
            append_continue_prompt=_append_thinking_continue_prompt,
            append_post_tool_nudge=_append_post_tool_empty_nudge,
            continue_prompt_event=_policy_continue_prompt_event,
            cut_step=_step_reasoning_after_cut,
            cut_restore=_restore_reasoning_after_cut,
            append_cut_nudge=_append_reasoning_cut_nudge,
            cut_retry_event=_policy_reasoning_cut_event,
            enter_wind_down=_enter_soft_stop,
            wind_down_budget=_soft_stop_turn_budget,
            llm_terminal=_emit_llm_terminal,
            state_change=_policy_state_change,
        ),
        EmptyCompletionGuardPolicy(
            has_terminal_tool_result=_history_has_terminal_tool_result,
            has_final_answer=run_has_final_answer,
            redrives_spent=_empty_completion_redrives_spent,
            charge_redrive=_charge_empty_completion_redrive,
            append_redrive_nudge=_append_empty_completion_redrive_nudge,
            empty_terminal=_emit_empty_completion_terminal,
            voluntary_completion=_emit_voluntary_completion,
            state_change=_policy_state_change,
        ),
        PerIterationCompactionPolicy(
            compact=_run_compaction,
            protect_index=current_tool_batch_protect_index,
            pair_orphans=_policy_pair_orphan_tool_calls,
            message_stop=_policy_message_stop,
        ),
        TerminalNudgePolicy(
            required=_terminal_tool_nudge_required,
            append=_append_terminal_tool_nudge,
            state_change=_policy_state_change,
        ),
        CancellationPolicy(teardown=_emit_dispatch_cancel_teardown),
        OutputCapRecoveryPolicy(
            recoveries=RunCounter(
                read=_output_recoveries_spent,
                charge=_charge_output_recovery,
                reset=_reset_output_recoveries,
            ),
            salvage=TruncationSalvage(
                state_path=_truncated_call_state_path,
                partial_content=_salvage_truncated_content,
                land_partial=_salvage_truncated_write_to_disk,
                recovery_text=_build_truncation_chunk_recovery_text,
                is_content_mutation=_is_content_mutation_truncation,
                paths=_truncated_call_paths,
                driver_enabled=_longfile.is_enabled,
                chunkable_names=_run_chunkable_write_names,
                note_truncated=_longfile.note_truncated_mutation,
                register_turn=_longfile.register_completed_turn,
            ),
            dispatch=_dispatch_tool,
            park=_park_pause_interrupt,
            parked_event=_interrupt_parked_event,
            result_is_terminal=_history_tool_result_is_terminal,
            pair_orphans=_policy_pair_orphan_tool_calls,
            wind_down=_enter_soft_stop,
            wind_down_budget=_soft_stop_turn_budget,
            pin_backstop=_pin_terminal_backstop,
            llm_terminal=_emit_llm_terminal,
            state_event=_policy_state_payload_event,
        ),
        TerminalToolFinishPolicy(pair_orphans=_policy_pair_orphan_tool_calls),
        StreamLoopGuardPolicy(
            nudges=RunCounter(
                read=_loop_guard_nudges_spent,
                charge=_charge_loop_guard_nudge,
                reset=_reset_loop_guard_nudges,
            ),
            block_identical=_block_identical_tools,
            guard_event=_policy_loop_guard_event,
        ),
        ProviderFailurePolicy(
            advance_chain=_advance_provider_chain,
            context_overflow=_handle_context_window_exceeded,
            has_preserved_answer=_preserve_completed_answer_on_stream_error,
            has_terminal_tool_result=_history_has_terminal_tool_result,
            has_final_answer=run_has_final_answer,
            preserved_finish=_complete_run_on_preserved_answer,
            wind_down=_enter_soft_stop,
            wind_down_budget=_soft_stop_turn_budget,
            llm_terminal=_emit_llm_terminal,
            fallback_event=_policy_fallback_event,
            retry_event=_policy_transient_retry_event,
            commit_usage=_policy_commit_usage,
            backoff=_transient_retry_backoff_seconds,
            retries=RunCounter(
                read=_transient_retries_spent,
                charge=_charge_transient_retry,
                reset=_reset_transient_retries,
            ),
            log_crash=_policy_log_stream_crash,
        ),
        TruncatedToolCallRecoveryPolicy(
            recoveries=RunCounter(
                read=_truncation_recoveries_spent,
                charge=_charge_truncation_recovery,
                reset=_reset_truncation_recoveries,
            ),
            llm_terminal=_emit_llm_terminal,
            dispatch=_dispatch_tool,
            park=_park_pause_interrupt,
            parked_event=_interrupt_parked_event,
            result_is_terminal=_history_tool_result_is_terminal,
            pair_orphans=_policy_pair_orphan_tool_calls,
            tool_use_stop=_policy_tool_use_message_stop,
            recovery_result_event=_policy_truncation_result_event,
        ),
    )
)
