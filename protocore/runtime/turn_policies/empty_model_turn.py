"""A round that came back with nothing usable in it.

Reasoning-only rounds cut by the output cap are retried without retaining the
incomplete reasoning: effort is lowered, then thinking is disabled when the run
mode permits it. A reasoning-only round the model ended itself is given the
reasoning back plus a short prompt to continue. A round with nothing at all,
arriving straight after tool results, is a model treating the tool result as
the last word: it is handed an API-valid pair — an empty assistant turn so the
sequence never goes tool → user, and a corrective nudge — and re-run.

All are bounded. Reasoning-only length cuts and model-ended reasoning use
separate counters, so one shape cannot spend the other's budget. Past the
bound the thinking traps wind the run down and, if they cannot, end it. The
post-tool nudge simply stops and lets the ordinary end-of-turn policies decide.

A round that produced anything at all clears the counters, so a single early
empty response does not permanently consume either budget.

None of this applies while the run's terminal call is being forced after a
delivered answer. The round that came back empty there was a request for one
tool call, and every recovery above would put words in front of the model — a
"continue" under a finished answer reads as an invitation to write another one.
Such a round is the forcing's to retry, on its own budget, so it passes through
untouched.
"""
from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any

from protocore.contracts.llm import LLMProviderError
from protocore.contracts.turn_policy import (
    TurnContext,
    TurnCoordinate,
    TurnDirective,
)
from protocore.runtime import soft_stop as _soft_stop
from protocore.runtime.events import TurnEvent
from protocore.runtime.turn_policies import (
    HistoryAppender,
    RunCounter,
    RunPredicate,
    StateChangeEmitter,
)
from protocore.runtime.turn_policies.run_ceilings import (
    TerminalEmitter,
    WindDownBudget,
    WindDownEntry,
)

#: The event that says a continue prompt went in, with the round it was and
#: how much reasoning the model had produced instead of an answer.
ContinuePromptEvent = Callable[[Any, int, int], TurnEvent]
ReasoningCutStep = Callable[[Any, int], str | None]
ReasoningCutRestore = Callable[[Any], None]
ReasoningCutEvent = Callable[[Any, int, str, int], TurnEvent]


class EmptyModelTurnPolicy:
    """Recover a round that produced nothing the run can use."""

    name = "empty_model_turn"
    coordinates = frozenset({TurnCoordinate.empty_model_turn})

    __slots__ = (
        "_append_continue_prompt",
        "_append_post_tool_nudge",
        "_append_reasoning_cut_nudge",
        "_continue_prompt_event",
        "_empty_rounds",
        "_enter_wind_down",
        "_forcing_terminal_call",
        "_llm_terminal",
        "_post_tool_nudges",
        "_reasoning_cut_event",
        "_reasoning_cut_restore",
        "_reasoning_cut_rounds",
        "_reasoning_cut_step",
        "_state_change",
        "_wind_down_budget",
    )

    def __init__(
        self,
        *,
        empty_rounds: RunCounter,
        reasoning_cut_rounds: RunCounter,
        post_tool_nudges: RunCounter,
        append_continue_prompt: HistoryAppender,
        append_post_tool_nudge: HistoryAppender,
        continue_prompt_event: ContinuePromptEvent,
        reasoning_cut_step: ReasoningCutStep,
        reasoning_cut_restore: ReasoningCutRestore,
        append_reasoning_cut_nudge: HistoryAppender,
        reasoning_cut_event: ReasoningCutEvent,
        enter_wind_down: WindDownEntry,
        wind_down_budget: WindDownBudget,
        llm_terminal: TerminalEmitter,
        state_change: StateChangeEmitter,
        forcing_terminal_call: RunPredicate,
    ) -> None:
        self._forcing_terminal_call = forcing_terminal_call
        self._empty_rounds = empty_rounds
        self._reasoning_cut_rounds = reasoning_cut_rounds
        self._post_tool_nudges = post_tool_nudges
        self._append_continue_prompt = append_continue_prompt
        self._append_post_tool_nudge = append_post_tool_nudge
        self._continue_prompt_event = continue_prompt_event
        self._reasoning_cut_step = reasoning_cut_step
        self._reasoning_cut_restore = reasoning_cut_restore
        self._append_reasoning_cut_nudge = append_reasoning_cut_nudge
        self._reasoning_cut_event = reasoning_cut_event
        self._enter_wind_down = enter_wind_down
        self._wind_down_budget = wind_down_budget
        self._llm_terminal = llm_terminal
        self._state_change = state_change

    async def apply(self, turn: TurnContext) -> AsyncIterator[TurnEvent]:
        engine = turn.engine
        if self._forcing_terminal_call(engine):
            return
        bound = engine.rc.max_consecutive_empty_responses
        empty_of_everything = (
            not turn.text_emitted
            and not turn.tool_calls_pending
            and not turn.reasoning_emitted
        )

        if (
            turn.finish_reason == "length"
            and not turn.text_emitted
            and not turn.tool_calls_pending
            and turn.reasoning_emitted
        ):
            async for event in self._length_cut(turn):
                yield event
            if turn.outcome.directive is not TurnDirective.proceed:
                return
        elif (
            not turn.text_emitted
            and not turn.tool_calls_pending
            and turn.reasoning_emitted
            and bound > 0
        ):
            async for event in self._thinking_trap(turn, bound):
                yield event
            if turn.outcome.directive is not TurnDirective.proceed:
                return

        # The trap either did not engage or was recovered from; either way the
        # count starts again, so a later turn that trips it gets a fresh one.
        self._empty_rounds.reset(engine)
        self._reasoning_cut_rounds.reset(engine)
        self._reasoning_cut_restore(engine)

        if (
            engine.rc.resilience_post_tool_empty_nudge_enabled
            and empty_of_everything
            and turn.tool_results_ready
            and bound > 0
        ):
            if self._post_tool_nudges.charge(engine) <= bound:
                self._append_post_tool_nudge(engine)
                turn.outcome.directive = TurnDirective.restart_turn
                turn.outcome.rebuild_context = True
                turn.outcome.reason = "post_tool_empty_nudge"
                yield self._state_change(engine, "post_tool_empty_nudge")
                return
            # Spent. Let the turn finish through the ordinary end-of-turn
            # path, which has its own answer for a run with nothing to show.
            # The count is NOT given back, so this cannot be looped on.

        if not empty_of_everything:
            self._post_tool_nudges.reset(engine)

    async def _thinking_trap(
        self, turn: TurnContext, bound: int
    ) -> AsyncIterator[TurnEvent]:
        """The model spent its output budget thinking and said nothing."""
        engine = turn.engine
        round_ = self._empty_rounds.charge(engine)
        # Keep the reasoning the round did produce: the next request re-injects
        # it, and some providers require that it be there.
        turn.record_partial_attempt()
        if round_ <= bound:
            self._append_continue_prompt(engine)
            turn.outcome.directive = TurnDirective.restart_turn
            turn.outcome.rebuild_context = True
            turn.outcome.reason = "continue_prompt_injected"
            yield self._continue_prompt_event(engine, round_, turn.reasoning_chars)
            return
        # Round after round of thinking and nothing consumable. Wind the run
        # down so it is asked for the best answer its evidence supports rather
        # than producing none at all. The count is deliberately not reset by
        # the per-message recovery reset, so another empty round during the
        # wind-down comes back here and falls through to the terminal below
        # under the original reason.
        async for event in self._nothing_consumable(
            turn,
            reason="thinking_eats_all_tokens",
            kind="thinking_eats_all_tokens",
            detail=(
                "consecutive empty responses with reasoning_content exceeded "
                "rc.max_consecutive_empty_responses"
            ),
        ):
            yield event

    async def _length_cut(self, turn: TurnContext) -> AsyncIterator[TurnEvent]:
        """Retry a reasoning-only length cut without retaining partial thought."""
        engine = turn.engine
        round_ = self._reasoning_cut_rounds.charge(engine)
        if round_ <= engine.rc.reasoning_length_cut_retries:
            control_change = self._reasoning_cut_step(engine, round_)
            if control_change is not None:
                if round_ == 1:
                    self._append_reasoning_cut_nudge(engine)
                turn.outcome.directive = TurnDirective.restart_turn
                turn.outcome.rebuild_context = True
                turn.outcome.reason = "reasoning_length_cut_retry"
                yield self._reasoning_cut_event(
                    engine, round_, control_change, turn.reasoning_chars
                )
                return
        self._reasoning_cut_restore(engine)
        async for event in self._nothing_consumable(
            turn,
            reason="reasoning_length_cut",
            kind="reasoning_length_cut",
            detail=(
                "reasoning-only length-limited responses exhausted "
                "rc.reasoning_length_cut_retries"
            ),
        ):
            yield event

    async def _nothing_consumable(
        self,
        turn: TurnContext,
        *,
        reason: str,
        kind: str,
        detail: str,
    ) -> AsyncIterator[TurnEvent]:
        engine = turn.engine
        # The endpoint answered every one of these rounds; what failed is the
        # model's output. The provider-error notice would tell the model the
        # endpoint was unreachable, and it would pass that on as the reason.
        events = self._enter_wind_down(
            engine, cause=_soft_stop.CAUSE_MODEL_NO_PROGRESS, detail=f"{kind}: {detail}"
        )
        if events:
            turn.outcome.directive = TurnDirective.restart_turn
            turn.outcome.turn_budget = self._wind_down_budget(
                engine, turn.flags.assistant_message_idx
            )
            turn.outcome.rebuild_context = True
            turn.outcome.reason = _soft_stop.CAUSE_MODEL_NO_PROGRESS
            for event in events:
                yield event
            await engine.persist_snapshot()
            return
        turn.outcome.directive = TurnDirective.end_turn
        turn.outcome.reason = reason
        async for event in self._llm_terminal(
            engine,
            LLMProviderError(detail),
            kind=kind,
        ):
            yield event


__all__ = [
    "ContinuePromptEvent",
    "EmptyModelTurnPolicy",
    "ReasoningCutEvent",
    "ReasoningCutRestore",
    "ReasoningCutStep",
]
