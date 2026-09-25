"""The push towards the run's terminal tool.

Two different runs reach the end of a turn without having called it, and they
need opposite things.

A run that has written its answer owes the call and nothing else. It is not
told so — it is made to: every request from here on is constrained to a tool
call (see :mod:`protocore.runtime.forced_terminal` for the modes), nothing is
appended to the transcript, nothing asks the model to "continue", and no text
from a forced turn reaches the reader. A request that comes back without the
call is retried on a bounded budget, and when the budget is spent the run
completes on the answer it already delivered.

The forcing steps aside when the model has to act rather than seal: a gate
refused the call and asked for a better answer, a message arrived for the
model, or the any-tool request that follows a missing-work refusal was answered
with work. Stepping aside for a question spends from the same budget; stepping
aside for work does not spend again, because the request that produced the
work was already charged. What the model writes then is visible, and the answer
it ends on is forced again.

A run that has written nothing is different: the terminal call cannot be forced
on it, because the answer it still owes is prose and a forced tool call cannot
write any. That run is told, once, in words, and gets one more message to write
the answer. Once, because a second telling never produced a different answer
and a run that keeps being told burns its budget on being told. When it does
write the answer, the first case takes over.

The latch that spends the single telling is turn-local state the loop can still
see; the forcing's own state is the run's, snapshot-persisted, and everything
else about the decision is here.
"""
from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

from protocore.contracts.turn_policy import (
    TurnContext,
    TurnCoordinate,
    TurnDirective,
)
from protocore.runtime.events import TurnEvent
from protocore.runtime.forced_terminal import (
    MODE_ANY_TOOL,
    MODE_TERMINAL,
    REASON_EXHAUSTED,
    REASON_FORCED,
    REASON_RETRY,
    REASON_UNAVAILABLE,
)
from protocore.runtime.turn_policies import (
    HistoryAppender,
    RunPredicate,
    StateChangeEmitter,
)


@dataclass(frozen=True, slots=True)
class ForcedTerminalCall:
    """What the policy may ask and do about a terminal call it forces."""

    #: The run's answer is written after its latest work.
    answer_delivered: RunPredicate
    #: The terminal tool is a tool this run can still call.
    tool_registered: RunPredicate
    #: Start forcing; True when the forcing was not already on.
    arm: Callable[[Any], bool]
    #: The forcing is on.
    armed: RunPredicate
    #: Lift the forcing without giving back what it spent.
    release: HistoryAppender
    #: The transcript ends in a message asking the model something.
    question_pending: RunPredicate
    #: The last forced turn called a tool other than the terminal one.
    working_again: RunPredicate
    #: The terminal tool refused its last call because work is missing
    #: (the refusal carries the needs-work marker), for the first time.
    work_requested: RunPredicate
    #: The host wants the one-time write-first telling before a seal, and the
    #: run has written no file with a write tool it has.
    write_first_before_sealing: RunPredicate
    #: A run-level precondition owns the next request's forced slot.
    slot_taken: RunPredicate
    #: Spend one attempt.
    charge: Callable[[Any], int]
    #: Set the mode of the request about to open (``None``: not forced).
    set_mode: Callable[[Any, str | None], None]
    #: Every attempt the run may spend is spent.
    exhausted: RunPredicate
    #: Complete the run on the delivered answer, saying why.
    complete: Callable[[Any, str], AsyncIterator[TurnEvent]]


class TerminalNudgePolicy:
    """Force the terminal call after an answer; ask for the answer before one."""

    name = "terminal_nudge"
    coordinates = frozenset({TurnCoordinate.turn_start, TurnCoordinate.finish_nudge})

    __slots__ = ("_append", "_forced", "_required", "_state_change")

    def __init__(
        self,
        *,
        required: RunPredicate,
        append: HistoryAppender,
        state_change: StateChangeEmitter,
        forced: ForcedTerminalCall,
    ) -> None:
        self._required = required
        self._append = append
        self._state_change = state_change
        self._forced = forced

    async def apply(self, turn: TurnContext) -> AsyncIterator[TurnEvent]:
        if turn.coordinate is TurnCoordinate.turn_start:
            async for event in self._before_the_request(turn):
                yield event
            return
        async for event in self._at_the_finish(turn):
            yield event

    async def _before_the_request(self, turn: TurnContext) -> AsyncIterator[TurnEvent]:
        """Choose the forced request's mode, lift the forcing, or end the run.

        Every forced request and every lifting for a question spends from one
        budget, and the budget is read before anything else. A lifting for
        work spends nothing itself, but it can only follow a charged any-tool
        request (or a precondition turn its own budget bounds) — so neither a
        gate that keeps refusing the call nor a model that keeps resuming work
        can cycle through here for free.
        """
        engine = turn.engine
        forced = self._forced
        if not forced.armed(engine):
            return
        if not self._required(engine):
            # The call landed; nothing is owed.
            forced.release(engine)
            return
        if forced.exhausted(engine):
            turn.outcome.directive = TurnDirective.end_turn
            turn.outcome.reason = REASON_EXHAUSTED
            async for event in forced.complete(engine, REASON_EXHAUSTED):
                yield event
            return
        if not forced.tool_registered(engine):
            # The terminal tool can no longer be called at all; the answer
            # stands, and nothing is gained by asking for more of it.
            turn.outcome.directive = TurnDirective.end_turn
            turn.outcome.reason = REASON_UNAVAILABLE
            async for event in forced.complete(engine, REASON_UNAVAILABLE):
                yield event
            return
        if forced.question_pending(engine):
            # A gate refused the call and says why, or someone asked the model
            # something. A forced call can answer neither, so this request goes
            # out free and its text is the reader's; the answer it produces
            # arms the forcing again on what is left of the budget.
            forced.charge(engine)
            forced.release(engine)
            return
        if forced.working_again(engine):
            # The model answered a required call with work rather than the
            # terminal call. Not charged here: the request that produced the
            # work was a charged any-tool request — the only forced mode that
            # admits another tool — or a precondition's turn, which its own
            # budget bounds. The work is the model's to finish, and its next
            # answer arms the forcing again.
            forced.release(engine)
            return
        if forced.slot_taken(engine):
            forced.set_mode(engine, None)
            return
        mode = MODE_ANY_TOOL if forced.work_requested(engine) else MODE_TERMINAL
        forced.charge(engine)
        forced.set_mode(engine, mode)

    async def _at_the_finish(self, turn: TurnContext) -> AsyncIterator[TurnEvent]:
        engine = turn.engine
        if not self._required(engine):
            return
        forced = self._forced
        seal = forced.armed(engine) or forced.answer_delivered(engine)
        if (
            seal
            and not forced.armed(engine)
            and not turn.flags.terminal_nudge_used
            and forced.write_first_before_sealing(engine)
        ):
            # The host asked for the write-first telling to come before the
            # seal: the prose may only announce a file the run never wrote, and
            # the telling is what gets it written. Once, like any telling; the
            # answer the model ends on after it is sealed by force.
            seal = False
        if seal:
            if not forced.tool_registered(engine):
                # The run can never make the call it owes, so there is nothing
                # to force and nothing to wait for: the answer stands.
                turn.outcome.directive = TurnDirective.end_turn
                turn.outcome.reason = REASON_UNAVAILABLE
                async for event in forced.complete(engine, REASON_UNAVAILABLE):
                    yield event
                return
            first = forced.arm(engine)
            reason = REASON_FORCED if first else REASON_RETRY
            turn.outcome.directive = TurnDirective.restart_turn
            turn.outcome.extra_turn = True
            turn.outcome.rebuild_context = True
            turn.outcome.reason = reason
            yield self._state_change(engine, reason)
            return
        if turn.flags.terminal_nudge_used:
            return
        turn.flags.terminal_nudge_used = True
        self._append(engine)
        turn.outcome.directive = TurnDirective.restart_turn
        turn.outcome.extra_turn = True
        turn.outcome.rebuild_context = True
        turn.outcome.reason = "terminal_tool_nudge"
        yield self._state_change(engine, "terminal_tool_nudge")


__all__ = ["ForcedTerminalCall", "TerminalNudgePolicy"]
