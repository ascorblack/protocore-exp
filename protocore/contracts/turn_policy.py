"""The turn-policy contract — where a product decision about a turn may live.

The driver of one assistant turn does two different jobs. One is **mechanics**:
open a stream, translate deltas into events, dispatch the calls the model
asked for, close the round. The other is **policy**: decide that this run has
spent its budget, that an empty answer earns one more try, that a file left
half-written must be sealed before the run is allowed to finish. Mechanics is
the same for every run. Policy is a product opinion, and every opinion that
was ever added to the loop was added by growing a branch inside it.

This module is the seam that stops that. A policy is an object. It declares
the **coordinates** — named seams of a turn — at which it wants to be
consulted, it is consulted in an order the core owns rather than in the order
someone edited the file, and it answers with events to forward and one
:class:`TurnDirective` saying what the loop does next. The loop keeps the
mechanics and loses the opinions.

Three shapes make that work:

:class:`ITurnState`
    Everything a policy may read or change on the run. It is a deliberately
    small structural view of the engine — a policy that needs something not
    named here is a policy that is reaching into the loop's insides, and the
    review that adds the name is the point at which that gets noticed.
:class:`TurnFlags`
    The turn-local state policies share with the loop. Before this existed
    these were bare locals of one very long function, which is why a policy
    could not be moved out of it: the read would leave with the policy and the
    write would stay behind.
:class:`TurnPolicyOutcome`
    What the loop must do next, said once, in a vocabulary of three directives
    instead of an inline ``continue`` a reader has to trace to a loop header.

A policy never yields a terminal event and then returns ``proceed``: the
directive is the whole statement about control flow, and the loop trusts it.
"""
from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover - typing only
    from protocore.contracts.interrupt import PendingInterrupt
    from protocore.contracts.runtime_constants import LoopConstants
    from protocore.contracts.types import Message, ToolCall
    from protocore.runtime.events import TurnEvent
    from protocore.runtime.loop_state import LoopState
    from protocore.runtime.usage import TokenUsage


class TurnCoordinate(StrEnum):
    """The seams of one turn at which policies are consulted.

    A coordinate is a place in the turn with a meaning, not a line number:
    "the turn is about to open" reads the same after the loop is rewritten,
    and a policy registered there keeps working.
    """

    #: Before the turn opens its stream — where the bounds of a run are read.
    turn_start = "turn_start"
    #: Just after this assistant message is counted, where the turn cap bites.
    turn_budget = "turn_budget"
    #: The stream closed having produced nothing usable.
    empty_model_turn = "empty_model_turn"
    #: The round ended because the output budget ran out, mid-sentence or
    #: mid-way through a tool call's arguments.
    output_truncated = "output_truncated"
    #: A stream attempt raised instead of finishing — the seam where a run
    #: decides between another endpoint, another attempt, and giving up.
    stream_failed = "stream_failed"
    #: An assistant turn completed, whether or not it called tools.
    turn_end = "turn_end"
    #: The round came back usable, before anything is read off it — where a
    #: run decides whether what came back is the model working or the model
    #: repeating itself.
    stream_settled = "stream_settled"
    #: The round's tool calls are in the transcript and none has been
    #: dispatched — the last moment a call can be refused rather than run.
    tool_calls_ready = "tool_calls_ready"
    #: The model finished with prose and no tool call, before anything is sealed.
    finish_nudge = "finish_nudge"
    #: A finish that carries an answer, before that answer is accepted.
    answer_floor = "answer_floor"
    #: The run is about to complete because the model chose to finish.
    voluntary_finish = "voluntary_finish"
    #: The run is about to complete because a terminal tool ended the turn.
    terminal_tool_finish = "terminal_tool_finish"
    #: The bottom of a dispatch iteration, before the next one opens.
    iteration_end = "iteration_end"
    #: Any seam where the turn's next step would be a NEW side effect, and a
    #: stop that landed in the await before it must be seen first.
    cancel_checkpoint = "cancel_checkpoint"


class TurnDirective(StrEnum):
    """What the loop does after a policy has been consulted."""

    #: Nothing happened. Carry on exactly as if the policy were not installed.
    proceed = "proceed"
    #: Start another assistant message. The loop applies the turn budget and
    #: rebuilds the context the policy asked for, then continues.
    restart_turn = "restart_turn"
    #: The turn is over. The policy has already emitted its terminal events.
    end_turn = "end_turn"


@runtime_checkable
class ITurnState(Protocol):
    """The run, as a policy is allowed to see it.

    Seventeen names. Reads are reads of the live run; the five callables that
    change something (:attr:`history` append, :meth:`transition_to`,
    :meth:`mark_pending_approval`, :meth:`release_interrupt`,
    :meth:`forget_tool_name`) and the durable :meth:`persist_snapshot` are the
    whole of what a policy can do to it.
    """

    #: The transcript. Policies append to it; nothing else about it is theirs.
    history: list[Message]
    #: Where the run is in its lifecycle.
    state: LoopState
    #: Iterations the per-iteration compaction gate still stands down for, after a pass that freed nothing.
    compaction_backoff_left: int
    #: The prompt's size when that backoff was set; growth past
    #: ``compaction_no_gain_backoff_growth_ratio`` of it ends the backoff early.
    compaction_backoff_prompt_tokens: int

    @property
    def rc(self) -> LoopConstants:
        """The run's constants — every threshold a policy reads."""

    @property
    def stop_requested(self) -> bool:
        """Whether cancellation has been asked for."""

    @property
    def is_terminal(self) -> bool:
        """Whether the run has already reached a terminal state."""

    @property
    def total_usage(self) -> TokenUsage:
        """What the run has spent so far."""

    def transition_to(self, new_state: LoopState) -> None:
        """Move the run to ``new_state``, or raise if that is illegal."""

    def turn_id(self) -> str:
        """The wire identity of the round in flight."""

    def prompt_text(self, name: str, /, **context: Any) -> str:
        """Render the named prompt template for this run."""

    async def persist_snapshot(self) -> None:
        """Write the run's durable state, so a resume sees what just happened."""

    def mark_pending_approval(
        self,
        tool_call_id: str,
        *,
        tool_name: str = "",
        payload: dict[str, Any] | None = None,
    ) -> PendingInterrupt:
        """Park a call at an approval gate as a typed, resumable wait."""

    def needs_compaction(self) -> bool:
        """Whether history has grown past the compaction threshold."""

    def needs_emergency_compaction(self) -> bool:
        """Whether history has grown past the cliff where compaction is forced."""

    def forget_tool_name(self, tool_call_id: str) -> None:
        """Drop the remembered name of a call that will never produce a result."""

    def release_interrupt(self, interrupt_id: str) -> None:
        """Drop a wait nobody will ever answer.

        The counterpart of :meth:`mark_pending_approval`: a policy that parked
        calls and then found the run ending — cancelled, most of all — has to
        take the questions back, because the calls they name are about to be
        answered by the ending instead.
        """


@dataclass(slots=True)
class TurnFlags:
    """Turn-local state that both the loop and its policies read.

    Every field here was a local variable of the turn driver. A policy that
    owns one of them owns the write; the loop keeps the read it needs to run
    the mechanics, and the flag is the only thing the two share.
    """

    #: How many assistant messages this turn has opened.
    assistant_message_idx: int = 0
    #: Whether a terminal answer has already gone out on this round.
    terminal_yielded: bool = False
    #: Whether the forced best-effort turn after a stream failure is armed.
    backstop_armed: bool = False
    #: Whether a dispatched call is parked at an approval gate.
    approval_pending: bool = False
    #: Whether a terminal tool ended the turn.
    terminal_tool_completed: bool = False
    #: Whether the one nudge towards the terminal tool has been spent.
    terminal_nudge_used: bool = False


@dataclass(slots=True)
class TurnPolicyOutcome:
    """What one consulted policy asks the loop to do next."""

    #: The control-flow statement. Everything else here refines it.
    directive: TurnDirective = TurnDirective.proceed
    #: Grant one more assistant message than the budget currently allows.
    extra_turn: bool = False
    #: Replace the turn budget outright (a wind-down sets its own).
    turn_budget: int | None = None
    #: Rebuild the context before the next assistant message opens.
    rebuild_context: bool = False
    #: Why, for the log — never for control flow.
    reason: str = ""
    #: The calls the turn will actually dispatch, when a policy refused some
    #: of what the model asked for. ``None`` means "whatever the round
    #: produced" — the loop must be able to tell that apart from an empty
    #: list, which is a policy saying the run runs none of them.
    tool_calls: Sequence[ToolCall] | None = None


def _do_nothing() -> None:
    """The default for a callback the loop did not offer at this coordinate."""


async def _no_events() -> AsyncIterator[TurnEvent]:
    """The default for an event source the loop did not offer here."""
    return
    yield  # pragma: no cover - the default source has nothing to say


def _nothing_repeated() -> tuple[str, int] | None:
    """The default for the repeat probe the loop did not offer here."""
    return None


@dataclass(slots=True)
class TurnContext:
    """One consultation: the run, the turn's flags, and what just happened."""

    #: The run, narrowed to what a policy may touch.
    engine: ITurnState
    #: The turn-local state shared with the loop.
    flags: TurnFlags
    #: Which seam this consultation is at.
    coordinate: TurnCoordinate
    #: The answer. Each policy writes into a fresh one, so none reads the
    #: policy before it; what the loop reads when the consultation is over is
    #: the accumulation — one directive, from the policy that took the turn,
    #: over the grants every policy at the coordinate asked for.
    outcome: TurnPolicyOutcome = field(default_factory=TurnPolicyOutcome)
    #: Whether the round that just closed put visible text in the transcript.
    text_emitted: bool = False
    #: Whether it produced reasoning.
    reasoning_emitted: bool = False
    #: Whether a terminal tool is what ended the turn.
    terminal_tool_finished: bool = False
    #: The number of assistant messages this turn is currently allowed.
    turn_budget: int = 0
    #: A typed provider failure the turn stored to surface if it ends with no
    #: answer, carried with the terminal kind it should be reported under.
    stored_stream_error: tuple[BaseException, str] | None = None
    #: Whether the round that just closed produced tool calls to dispatch.
    tool_calls_pending: bool = False
    #: The calls themselves, in the order the model asked for them. Empty at
    #: every coordinate but the one that sits between the round and its
    #: dispatch, where a policy may still refuse one.
    pending_tool_calls: Sequence[ToolCall] = ()
    #: How the round ended, as the provider reported it.
    finish_reason: str = ""
    #: What the stream attempt raised, at the seam that answers for it.
    stream_error: BaseException | None = None
    #: How much reasoning it produced, in characters.
    reasoning_chars: int = 0
    #: Whether this round followed tool results — the shape a model that
    #: mistakes a tool result for the last word answers emptily in.
    tool_results_ready: bool = False
    #: Keep what the round produced before it is retried rather than accepted.
    #: A policy that sends the round round again calls this first, or what the
    #: model did produce is lost between attempts.
    record_partial_attempt: Callable[[], None] = _do_nothing
    #: Ask the installed policies whether the run has been told to stop, at a
    #: seam inside a policy whose next step would be a NEW side effect. Drains
    #: to nothing at every coordinate the loop does not offer it at; a policy
    #: that used it reads :attr:`TurnFlags.terminal_yielded` afterwards.
    cancel_checkpoint: Callable[[], AsyncIterator[TurnEvent]] = _no_events
    #: Strip a repeated tail off the round the loop just took, and say what
    #: was stripped: the kind of repetition and how many characters went. The
    #: rewrite of the buffers is the loop's — where the round is held — and
    #: what a repetition MEANS for the run is the policy's.
    stream_repeat_guard: Callable[[], tuple[str, int] | None] = _nothing_repeated
    #: Whether the round that just closed dispatched tools. The loop rebuilds
    #: the context on its own way round from one that did, so a policy that
    #: asks it to restart there would skip what that rebuild is bracketed by.
    dispatched_tools: bool = False


class ITurnPolicy(Protocol):
    """One product decision about a turn, consulted at named seams."""

    @property
    def name(self) -> str:
        """The policy's identity, and its place in the core's fixed order."""

    @property
    def coordinates(self) -> frozenset[TurnCoordinate]:
        """The seams at which this policy wants to be consulted."""

    def apply(self, turn: TurnContext) -> AsyncIterator[TurnEvent]:
        """Run at ``turn.coordinate``, yielding events the loop forwards.

        Whatever the loop must do next is written to ``turn.outcome``; the
        events are yielded as they happen rather than collected, because a
        policy that dispatches a tool must not hold its deltas back until it
        is finished.
        """


__all__ = [
    "ITurnPolicy",
    "ITurnState",
    "TurnContext",
    "TurnCoordinate",
    "TurnDirective",
    "TurnFlags",
    "TurnPolicyOutcome",
]
