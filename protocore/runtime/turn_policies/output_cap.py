"""A round the output budget cut short, and the two shapes that takes.

The model runs out of output tokens mid-sentence, or mid-way through writing
a tool call's arguments. Neither is a finished turn, and both used to be
answered by branches that grew inside the driver until they no longer agreed
with each other.

**Cut mid-prose.** The partial goes into the transcript and the model is
asked to resume from where it stopped. Nothing is lost and nothing is
repeated.

**Cut mid-call.** The call is never dispatched: its arguments were closed by
the parser, not by the model, so running it would write a truncated file and
the model would never learn why. What happens instead depends on what the cut
left behind. When the call was a chunkable write whose partial body did
survive, those bytes are landed with a clean synthetic write — the work is
real, and the file it belongs to is now known to be one being written in
pieces. When nothing survived, the call gets a recovery instruction naming
its path and a smaller budget for the first chunk, so the retry fits under
the cap, lands bytes, and the convergence driver can take over from there.
Either way the siblings the model DID finish in the same message are
dispatched: the truncation is one call's problem.

Both shapes debit the same per-message count, because they are the same
budget being spent twice. Past it the run winds down — a model that packed an
oversized call on every round is asked for a compact answer on a surface with
nothing else on it — and the count is deliberately left spent, so the
wind-down cannot re-enter the recovery it just exhausted.
"""
from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass
from typing import Any

from protocore.contracts.llm import MaxOutputTokensExhausted
from protocore.contracts.turn_policy import (
    TurnContext,
    TurnCoordinate,
    TurnDirective,
)
from protocore.contracts.types import (
    SYNTHETIC_RECOVERY_MAX_OUTPUT_CONTINUE,
    SYNTHETIC_RECOVERY_METADATA_KEY,
    SYNTHETIC_RECOVERY_TRUNCATION_CONTINUE,
    Message,
    MessageRole,
    TextBlock,
    ToolCall,
)
from protocore.runtime import soft_stop as _soft_stop
from protocore.runtime.events import EventType, TurnEvent
from protocore.runtime.loop_state import LoopState
from protocore.runtime.turn_policies import RunCounter
from protocore.runtime.turn_policies.run_ceilings import (
    TerminalEmitter,
    WindDownBudget,
    WindDownEntry,
)
from protocore.runtime.turn_policies.sibling_walk import (
    CallParker,
    ToolDispatcher,
    dispatch_parking_holds,
    prose_gate_just_injected,
)
from protocore.runtime.turn_policies.truncated_tool_call import (
    ParkedAnnouncer,
    ResultIsTerminal,
)

#: The text a run is nudged with when the cap cut it mid-sentence.
RESUME_PROMPT = (
    "Resume directly from where you left off, without preamble or repetition."
)


@dataclass(frozen=True, slots=True)
class TruncationSalvage:
    """What the run knows about a write the output cap cut in half.

    Grouped rather than passed one by one because these five answer a single
    question — what, if anything, of this call can be kept — and a policy that
    took them separately would let a caller supply four of them.
    """

    #: The path this call was writing to, or ``None`` when it names none.
    state_path: Callable[[Any, ToolCall], str | None]
    #: The partial body the parser recovered, or empty when it recovered none.
    partial_content: Callable[[ToolCall], str]
    #: Land that partial with a clean synthetic write.
    land_partial: Callable[[Any, ToolCall, str], AsyncIterator[TurnEvent]]
    #: The instruction a call with nothing to salvage is answered with.
    recovery_text: Callable[..., str]
    #: Whether this truncation is a content mutation at all.
    is_content_mutation: Callable[[Any, ToolCall], bool]
    #: The paths a set of truncated calls was writing to, for the record.
    paths: Callable[..., list[str]]
    #: Whether the convergence driver is on for this run at all.
    driver_enabled: Callable[[Any], bool]
    #: The built-in writes whose content can be sent in chunks.
    chunkable_names: Callable[[Any], frozenset[str]]
    #: Remember that this path is a large file the model is part-way through.
    note_truncated: Callable[[Any, str | None], None]
    #: Advance the stall clock for a recovery turn that landed no bytes.
    register_turn: Callable[[Any], None]


#: A state-change event carrying a payload the policy composed itself.
StatePayloadEvent = Callable[[Any, dict[str, Any]], TurnEvent]

#: Pin the recovery count as spent, so a wind-down cannot re-enter recovery.
BackstopPin = Callable[[Any], None]


class OutputCapRecoveryPolicy:
    """Answer a round the output budget cut short, whichever way it cut it."""

    name = "output_cap_recovery"
    coordinates = frozenset({TurnCoordinate.output_truncated})

    __slots__ = (
        "_dispatch",
        "_llm_terminal",
        "_pair_orphans",
        "_park",
        "_parked_event",
        "_pin_backstop",
        "_recoveries",
        "_result_is_terminal",
        "_salvage",
        "_state_event",
        "_wind_down",
        "_wind_down_budget",
    )

    def __init__(
        self,
        *,
        recoveries: RunCounter,
        salvage: TruncationSalvage,
        dispatch: ToolDispatcher,
        park: CallParker,
        parked_event: ParkedAnnouncer,
        result_is_terminal: ResultIsTerminal,
        pair_orphans: Callable[[Any], None],
        wind_down: WindDownEntry,
        wind_down_budget: WindDownBudget,
        pin_backstop: BackstopPin,
        llm_terminal: TerminalEmitter,
        state_event: StatePayloadEvent,
    ) -> None:
        self._recoveries = recoveries
        self._salvage = salvage
        self._dispatch = dispatch
        self._park = park
        self._parked_event = parked_event
        self._result_is_terminal = result_is_terminal
        self._pair_orphans = pair_orphans
        self._wind_down = wind_down
        self._wind_down_budget = wind_down_budget
        self._pin_backstop = pin_backstop
        self._llm_terminal = llm_terminal
        self._state_event = state_event

    async def apply(self, turn: TurnContext) -> AsyncIterator[TurnEvent]:
        cut_calls = [
            call for call in turn.pending_tool_calls if call.truncated_by_output_cap
        ]
        if cut_calls:
            # The trigger is the provider-agnostic truncation signal under ANY
            # finish reason, not only a length cap: a content-less write cut at
            # the cap can arrive reported as an ordinary tool-use finish, and a
            # gate that read the finish reason missed it and dispatched the
            # broken call.
            async for event in self._cut_mid_call(turn, cut_calls):
                yield event
            return

        if (
            turn.finish_reason in ("length", "")
            and not turn.pending_tool_calls
            and not turn.engine.stop_requested
            and not (turn.reasoning_emitted and not turn.text_emitted)
        ):
            # A round that is reasoning and nothing else is not cut mid-prose:
            # there is no prose to resume, and the reasoning is not kept. The
            # empty-model-turn policy answers for that round.
            # An empty finish reason is folded in here. Some providers end the
            # stream cleanly with no finish delta at all, and reading that as a
            # normal completion let a mid-sentence partial be persisted as the
            # run's final answer. A finish-less, call-less stream is an
            # incomplete turn, and gets the same bounded resume a length cap
            # does. A cancel also leaves the finish reason empty, which is why
            # a stopped run is excluded: an interrupted turn is not recovered.
            async for event in self._cut_mid_prose(turn):
                yield event

    # -- cut mid-sentence ---------------------------------------------------

    async def _cut_mid_prose(self, turn: TurnContext) -> AsyncIterator[TurnEvent]:
        engine = turn.engine
        turn.record_partial_attempt()
        if self._recoveries.read(engine) >= engine.rc.max_output_recovery_rounds:
            async for event in self._exhausted(
                turn, "max_output_recovery_rounds exhausted"
            ):
                yield event
            return
        round_index = self._recoveries.charge(engine)
        engine.history.append(
            Message(
                role=MessageRole.user,
                content_blocks=[TextBlock(text=RESUME_PROMPT)],
                metadata={
                    SYNTHETIC_RECOVERY_METADATA_KEY: (
                        SYNTHETIC_RECOVERY_MAX_OUTPUT_CONTINUE
                    )
                },
            )
        )
        yield self._state_event(
            engine,
            {
                "from": engine.state.value,
                "to": engine.state.value,
                "reason": "max_output_token_recovery",
                "round": round_index,
            },
        )
        turn.outcome.directive = TurnDirective.restart_turn
        turn.outcome.rebuild_context = True
        turn.outcome.reason = "max_output_token_recovery"

    # -- cut mid-call -------------------------------------------------------

    async def _cut_mid_call(
        self, turn: TurnContext, cut_calls: Sequence[ToolCall]
    ) -> AsyncIterator[TurnEvent]:
        engine = turn.engine
        turn.record_partial_attempt()
        if self._recoveries.read(engine) >= engine.rc.max_output_recovery_rounds:
            # The model packed an oversized call — a large list argument, say —
            # and ran out of budget on every round of trying again.
            async for event in self._exhausted(
                turn,
                "max_output_recovery_rounds exhausted (mid-tool-call truncation)",
            ):
                yield event
            return
        round_index = self._recoveries.charge(engine)

        salvage_jobs = self._salvageable(engine, cut_calls)
        salvaged_ids = {call.id for call, _ in salvage_jobs}

        async for event in self._dispatch_the_siblings(turn, cut_calls):
            yield event
        if turn.outcome.directive is not TurnDirective.proceed:
            return

        async for event in self._land_the_partials(turn, salvage_jobs):
            yield event
        if turn.outcome.directive is not TurnDirective.proceed:
            return

        if salvage_jobs:
            # The original truncated calls are never dispatched. Pairing them
            # with the interrupted-result placeholder keeps the provider's own
            # attempt reload-safe, while the clean synthetic write beside it
            # records the mutation that actually reached disk.
            self._pair_orphans(engine)

        unsalvaged = [call for call in cut_calls if call.id not in salvaged_ids]
        resume_text = ""
        if unsalvaged:
            resume_text = self._salvage.recovery_text(engine, unsalvaged)
            for call in unsalvaged:
                if self._salvage.is_content_mutation(engine, call):
                    # A chunkable write cut before any body left no bytes but
                    # IS a large file in progress: latching its real path means
                    # the smaller-chunk retry engages the driver as soon as it
                    # lands one.
                    self._salvage.note_truncated(
                        engine, self._salvage.state_path(engine, call)
                    )
        # A recovery turn that salvaged nothing added no bytes and restarts
        # without reaching the end-of-turn seam, so the stall clock is advanced
        # here. A salvage that DID land bytes has already reset it, which makes
        # this a correct no-op there.
        self._salvage.register_turn(engine)
        if resume_text:
            engine.history.append(
                Message(
                    role=MessageRole.user,
                    content_blocks=[TextBlock(text=resume_text)],
                    metadata={
                        SYNTHETIC_RECOVERY_METADATA_KEY: (
                            SYNTHETIC_RECOVERY_TRUNCATION_CONTINUE
                        )
                    },
                )
            )
        payload: dict[str, Any] = {
            "from": engine.state.value,
            "to": engine.state.value,
            "reason": "tool_call_truncation_recovery",
            "round": round_index,
            "tools": [call.name for call in cut_calls],
            "paths": self._salvage.paths(engine, cut_calls),
        }
        if salvage_jobs:
            payload["salvaged_paths"] = self._salvage.paths(
                engine, [call for call, _ in salvage_jobs]
            )
        yield self._state_event(engine, payload)
        turn.outcome.directive = TurnDirective.restart_turn
        turn.outcome.rebuild_context = True
        turn.outcome.reason = "tool_call_truncation_recovery"

    def _salvageable(
        self, engine: Any, cut_calls: Sequence[ToolCall]
    ) -> list[tuple[ToolCall, str]]:
        """The cut calls whose partial body is worth landing, and that body.

        Four conditions, all of them: the convergence driver is on, the tool is
        a built-in chunkable write (salvage dispatches a built-in write, so a
        tenant tool flagged as one keeps its own recovery rather than being
        silently rewritten), a real target path resolves, and a non-empty
        partial body was recovered. Anything else keeps the recovery
        instruction — a pathless call with content must never be dropped.
        """
        if not self._salvage.driver_enabled(engine):
            return []
        chunkable = self._salvage.chunkable_names(engine)
        jobs: list[tuple[ToolCall, str]] = []
        for call in cut_calls:
            if call.name not in chunkable:
                continue
            if self._salvage.state_path(engine, call) is None:
                continue
            partial = self._salvage.partial_content(call)
            if partial:
                jobs.append((call, partial))
        return jobs

    async def _dispatch_the_siblings(
        self, turn: TurnContext, cut_calls: Sequence[ToolCall]
    ) -> AsyncIterator[TurnEvent]:
        """Run the calls the model finished alongside the one it did not.

        The cap ends the stream, so every complete call precedes the truncated
        tail one and dispatching them here preserves the order the model asked
        for. Skipping them would leave their calls unanswered, and the wire
        repair would then show the model synthetic errors for work it had
        actually finished.
        """
        engine = turn.engine
        cut_ids = {call.id for call in cut_calls}
        parked: list[Any] = []
        for call in turn.pending_tool_calls:
            if call.id in cut_ids:
                continue
            if parked and engine.stop_requested:
                # A cancelled run waits for nothing: the decisions this same
                # message parked a moment ago can no longer be acted on, and
                # the teardown below is about to answer their calls with
                # synthetic errors — a wait witnessing an answered call is a
                # question the host can never close.
                for held_wait in parked:
                    engine.release_interrupt(held_wait.interrupt_id)
                parked.clear()
                turn.flags.approval_pending = False
            async for event in turn.cancel_checkpoint():
                yield event
            if turn.flags.terminal_yielded:
                turn.outcome.directive = TurnDirective.end_turn
                turn.outcome.reason = "stop_requested"
                return
            held_count = len(parked)
            async for event in dispatch_parking_holds(
                engine, call, dispatch=self._dispatch, park=self._park, parked=parked
            ):
                yield event
            if len(parked) > held_count:
                continue
            if parked:
                continue
            if prose_gate_just_injected(engine):
                # The gate refused this call and asked for the answer in prose.
                # Dispatching the siblings behind it would file their results
                # after that question, so the batch stops here and the
                # corrective round carries what is left.
                return
            if self._result_is_terminal(engine, call.id):
                turn.flags.terminal_tool_completed = True
                turn.outcome.directive = TurnDirective.end_turn
                turn.outcome.reason = "terminal_tool_completed"
                return
        if parked:
            turn.flags.approval_pending = True
            engine.transition_to(LoopState.AWAITING)
            await engine.persist_snapshot()
            yield self._parked_event(engine, parked[0])
            turn.outcome.directive = TurnDirective.end_turn
            turn.outcome.reason = "approval_pending"

    async def _land_the_partials(
        self, turn: TurnContext, salvage_jobs: Sequence[tuple[ToolCall, str]]
    ) -> AsyncIterator[TurnEvent]:
        """Write each salvageable partial to disk with a clean synthetic call.

        The synthetic write is pre-approved — it is the run recovering the
        model's OWN content, not a new request — so the gate never fires and
        the park below is defensive. The stop checkpoint before each one is
        not: the write mutates a workspace, and one performed after a cancel
        is a side effect nobody asked for.
        """
        engine = turn.engine
        for call, partial in salvage_jobs:
            async for event in turn.cancel_checkpoint():
                yield event
            if turn.flags.terminal_yielded:
                turn.outcome.directive = TurnDirective.end_turn
                turn.outcome.reason = "stop_requested"
                return
            held = None
            async for event in self._salvage.land_partial(engine, call, partial):
                if event.type is EventType.TOOL_CALL_PENDING:
                    held = self._park(
                        engine,
                        str(event.payload.get("tool_call_id", "")),
                        event=event,
                    )
                    yield event
                    continue
                yield event
            if held is not None:
                engine.transition_to(LoopState.AWAITING)
                await engine.persist_snapshot()
                yield self._parked_event(engine, held)
                turn.flags.approval_pending = True
                turn.outcome.directive = TurnDirective.end_turn
                turn.outcome.reason = "approval_pending"
                return

    # -- the count is spent -------------------------------------------------

    async def _exhausted(
        self, turn: TurnContext, message: str
    ) -> AsyncIterator[TurnEvent]:
        engine = turn.engine
        wound = self._wind_down(engine, cause=_soft_stop.CAUSE_OUTPUT_TOKEN_BUDGET)
        if wound:
            turn.outcome.turn_budget = self._wind_down_budget(
                engine, turn.flags.assistant_message_idx
            )
            # The count is left spent on purpose. Refreshing it per message
            # would hand the wind-down turns a whole new recovery budget and
            # let them re-enter the loop they were started to escape; with it
            # spent, a re-truncation during the wind-down reaches this branch
            # again and the terminal below takes over.
            self._pin_backstop(engine)
            for event in wound:
                yield event
            await engine.persist_snapshot()
            turn.flags.backstop_armed = True
            turn.outcome.directive = TurnDirective.end_turn
            turn.outcome.rebuild_context = True
            turn.outcome.reason = "output_length_wind_down"
            return
        async for event in self._llm_terminal(
            engine,
            MaxOutputTokensExhausted(message),
            kind="output_length_exhausted",
        ):
            yield event
        turn.flags.terminal_yielded = True
        turn.outcome.directive = TurnDirective.end_turn
        turn.outcome.reason = "output_length_exhausted"


__all__ = ["RESUME_PROMPT", "OutputCapRecoveryPolicy", "TruncationSalvage"]
