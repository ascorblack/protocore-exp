"""A round that came back with nothing usable in it.

Three shapes, one subject, and they are told apart by what the empty round DID
carry and by how it ended.

A round with reasoning and nothing else, ended by the output cap
(``finish_reason='length'``), is a model that thought past its budget. What it
produced is incomplete by construction and is not kept: no provider can
continue a reasoning block it did not finish (Anthropic and Gemini sign only
complete ones, DeepSeek discards ``reasoning_content`` on input outside a tool
call it completed), and a kept fragment only grows the next prompt by the size
of the cut and makes the next cut more likely. The retry sends the same prompt
with one knob changed — the effort lowered, then thinking off — and one nudge
that names the cause and asks for a shorter shape, bounded by
``reasoning_length_cut_retries``; the knobs are restored on the next round that
produces anything.

A round with reasoning and nothing else that the model ended itself is the
older trap of small reasoning models: it is given the reasoning back plus a
short prompt to continue, and re-run. A round with nothing at all, arriving
straight after tool results, is a model treating the tool result as the last
word: it is handed an API-valid pair — an empty assistant turn so the sequence
never goes tool → user, and a corrective nudge — and re-run.

All are bounded, and the two thinking shapes share one count, so a model that
alternates between them cannot double its budget. Past the bound they part
company: the thinking traps wind the run down and, if they cannot, end it,
because a model that has produced nothing consumable for several rounds will
not produce something on the next one. The post-tool nudge simply stops and
lets the turn finish, where the ordinary end-of-turn policies get their say.

A round that produced anything at all clears the counters, so a single early
empty response does not permanently consume either budget.
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
#: Change one knob for the retry after a reasoning-only length cut: the round
#: (1-based) says which step of the ladder it is. Returns what changed, or
#: ``None`` when that step would change nothing — the ladder is then spent.
CutStep = Callable[[Any, int], str | None]
#: Put the knobs a cut retry changed back to their configured values.
CutRestore = Callable[[Any], None]
#: The event that says a cut retry is going out: the round, what changed, how
#: much reasoning the cut round had produced.
CutRetryEvent = Callable[[Any, int, str, int], TurnEvent]


class EmptyModelTurnPolicy:
    """Recover a round that produced nothing the run can use."""

    name = "empty_model_turn"
    coordinates = frozenset({TurnCoordinate.empty_model_turn})

    __slots__ = (
        "_append_continue_prompt",
        "_append_cut_nudge",
        "_append_post_tool_nudge",
        "_continue_prompt_event",
        "_cut_restore",
        "_cut_retry_event",
        "_cut_step",
        "_empty_rounds",
        "_enter_wind_down",
        "_llm_terminal",
        "_post_tool_nudges",
        "_state_change",
        "_wind_down_budget",
    )

    def __init__(
        self,
        *,
        empty_rounds: RunCounter,
        post_tool_nudges: RunCounter,
        append_continue_prompt: HistoryAppender,
        append_post_tool_nudge: HistoryAppender,
        continue_prompt_event: ContinuePromptEvent,
        cut_step: CutStep,
        cut_restore: CutRestore,
        append_cut_nudge: HistoryAppender,
        cut_retry_event: CutRetryEvent,
        enter_wind_down: WindDownEntry,
        wind_down_budget: WindDownBudget,
        llm_terminal: TerminalEmitter,
        state_change: StateChangeEmitter,
    ) -> None:
        self._empty_rounds = empty_rounds
        self._post_tool_nudges = post_tool_nudges
        self._append_continue_prompt = append_continue_prompt
        self._append_post_tool_nudge = append_post_tool_nudge
        self._continue_prompt_event = continue_prompt_event
        self._cut_step = cut_step
        self._cut_restore = cut_restore
        self._append_cut_nudge = append_cut_nudge
        self._cut_retry_event = cut_retry_event
        self._enter_wind_down = enter_wind_down
        self._wind_down_budget = wind_down_budget
        self._llm_terminal = llm_terminal
        self._state_change = state_change

    async def apply(self, turn: TurnContext) -> AsyncIterator[TurnEvent]:
        engine = turn.engine
        bound = engine.rc.max_consecutive_empty_responses
        empty_of_everything = (
            not turn.text_emitted
            and not turn.tool_calls_pending
            and not turn.reasoning_emitted
        )

        if (
            not turn.text_emitted
            and not turn.tool_calls_pending
            and turn.reasoning_emitted
            and turn.finish_reason == "length"
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
        # count starts again, so a later turn that trips it gets a fresh one,
        # and the knobs a cut retry turned go back to where they were.
        self._empty_rounds.reset(engine)
        self._cut_restore(engine)

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

    async def _length_cut(self, turn: TurnContext) -> AsyncIterator[TurnEvent]:
        """The output cap cut the round while the model was still reasoning.

        The cut reasoning is not recorded: the retry sends the prompt the cut
        round saw, byte for byte, so a cached prefix survives and the prompt
        does not grow by the size of every failed attempt. One knob changes per
        round; a round with no knob left to turn is the end of the ladder.
        """
        engine = turn.engine
        round_ = self._empty_rounds.charge(engine)
        changed = (
            self._cut_step(engine, round_)
            if round_ <= engine.rc.reasoning_length_cut_retries
            else None
        )
        if changed is not None:
            if round_ == 1:
                self._append_cut_nudge(engine)
            turn.outcome.directive = TurnDirective.restart_turn
            turn.outcome.rebuild_context = True
            turn.outcome.reason = "reasoning_length_cut_retry"
            yield self._cut_retry_event(engine, round_, changed, turn.reasoning_chars)
            return
        async for event in self._nothing_consumable(
            turn,
            kind="reasoning_length_cut",
            message=(
                "the output cap cut every round while the model was still "
                f"reasoning; {round_ - 1} retries with lowered effort and "
                "thinking off changed nothing (rc.reasoning_length_cut_retries)"
            ),
        ):
            yield event

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
        async for event in self._nothing_consumable(
            turn,
            kind="thinking_eats_all_tokens",
            message=(
                "consecutive empty responses with reasoning_content exceeded "
                "rc.max_consecutive_empty_responses"
            ),
        ):
            yield event

    async def _nothing_consumable(
        self, turn: TurnContext, *, kind: str, message: str
    ) -> AsyncIterator[TurnEvent]:
        """Round after round of thinking and nothing consumable.

        Wind the run down so it is asked for the best answer its evidence
        supports rather than producing none at all. The count is deliberately
        not reset by the per-message recovery reset, so another empty round
        during the wind-down comes back here and falls through to the terminal
        below under the original reason.
        """
        engine = turn.engine
        events = self._enter_wind_down(engine, cause=_soft_stop.CAUSE_PROVIDER_ERROR)
        if events:
            turn.outcome.directive = TurnDirective.restart_turn
            turn.outcome.turn_budget = self._wind_down_budget(
                engine, turn.flags.assistant_message_idx
            )
            turn.outcome.rebuild_context = True
            turn.outcome.reason = _soft_stop.CAUSE_PROVIDER_ERROR
            for event in events:
                yield event
            await engine.persist_snapshot()
            return
        turn.outcome.directive = TurnDirective.end_turn
        turn.outcome.reason = kind
        async for event in self._llm_terminal(engine, LLMProviderError(message), kind=kind):
            yield event


__all__ = [
    "ContinuePromptEvent",
    "CutRestore",
    "CutRetryEvent",
    "CutStep",
    "EmptyModelTurnPolicy",
]
