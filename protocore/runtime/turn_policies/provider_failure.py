"""What a run does when the provider it is talking to fails mid-stream.

Five failure classes reach this policy and they are answered by ranking three
recoveries, not by five separate opinions about the same question:

1. **A different endpoint.** A rate limit, a timeout, a stall and most
   provider errors say something about the endpoint rather than about the
   request, so a healthy sibling on the run's chain beats every other
   recovery — including sleeping on the sick one, which is a bet that this
   endpoint recovers and the chain exists precisely for the runs where that
   bet is wrong. Whether a failure is that kind of failure is a
   classification the adapter attached to it; an unclassified error never
   moves the chain.
2. **The same endpoint, later.** Only for the two classes that are
   transient by definition, bounded per consecutive-failure streak, and only
   once the chain has nothing left to offer.
3. **The answer the run already has.** The model has whatever evidence it
   gathered and the partial it produced is in the transcript, so one narrowed
   turn usually turns that into an answer. The original error is stashed
   while that runs: a wind-down that produces nothing must still report the
   provider failure rather than completing silently.

Two of the classes are outside that ranking. A context window overflow is
about the request, so it is answered by shrinking the request. And an
unclassified crash is not a provider failure at all — arming a best-effort
turn over one would let a later terminal complete "successfully" and swallow
the crash, so it stays terminal and the original error is always surfaced.

Whatever the failure, an answer the run has already delivered outranks it: a
transient error on a forced continuation turn must not bury a reply the
reader has already seen.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from protocore.contracts.llm import (
    LLMContextWindowExceeded,
    LLMProviderError,
    LLMRateLimitError,
    LLMStreamIdleError,
    LLMTimeoutError,
)
from protocore.contracts.turn_policy import (
    TurnContext,
    TurnCoordinate,
    TurnDirective,
)
from protocore.logging_utils import get_logger
from protocore.runtime import soft_stop as _soft_stop
from protocore.runtime.error_kinds import INTERNAL_ERROR_KIND
from protocore.runtime.events import TurnEvent
from protocore.runtime.turn_policies import RunCounter
from protocore.runtime.turn_policies.run_ceilings import (
    TerminalEmitter,
    WindDownBudget,
    WindDownEntry,
)

#: Step the run onto the next provider of its chain; the new provider's name,
#: or empty when the chain did not move.
ChainAdvance = Callable[..., Awaitable[str]]

#: Shrink the request that overflowed, in place.
ContextOverflowRecovery = Callable[..., AsyncIterator[TurnEvent]]

#: Close the run on an answer it already delivered.
PreservedAnswerFinish = Callable[..., AsyncIterator[TurnEvent]]

#: Whether such an answer exists.
HasPreservedAnswer = Callable[[Any], bool]

#: Say on the wire that the run moved to another provider.
FallbackAnnouncer = Callable[..., TurnEvent]

#: Say on the wire that the run will try the same provider again.
RetryAnnouncer = Callable[..., TurnEvent]

#: Record what the failed attempt spent.
UsageCommit = Callable[..., TurnEvent | None]

#: How long to wait before the ``attempt``-th retry.
BackoffSeconds = Callable[[Any, int, BaseException], float]

#: Write down that the turn crashed, naming the turn it crashed on.
CrashLogger = Callable[[Any, BaseException], None]


_logger = get_logger(__name__)

class ProviderFailurePolicy:
    """Rank the recoveries a failed stream attempt has, and take the best."""

    name = "provider_failure"
    coordinates = frozenset(
        {
            TurnCoordinate.stream_failed,
            TurnCoordinate.stream_settled,
            TurnCoordinate.finish_nudge,
        }
    )

    __slots__ = (
        "_advance_chain",
        "_backoff",
        "_commit_usage",
        "_context_overflow",
        "_fallback_event",
        "_has_final_answer",
        "_has_preserved_answer",
        "_has_terminal_tool_result",
        "_llm_terminal",
        "_log_crash",
        "_preserved_finish",
        "_retries",
        "_retry_event",
        "_wind_down",
        "_wind_down_budget",
    )

    def __init__(
        self,
        *,
        advance_chain: ChainAdvance,
        context_overflow: ContextOverflowRecovery,
        has_preserved_answer: HasPreservedAnswer,
        has_terminal_tool_result: HasPreservedAnswer,
        has_final_answer: HasPreservedAnswer,
        preserved_finish: PreservedAnswerFinish,
        wind_down: WindDownEntry,
        wind_down_budget: WindDownBudget,
        llm_terminal: TerminalEmitter,
        fallback_event: FallbackAnnouncer,
        retry_event: RetryAnnouncer,
        commit_usage: UsageCommit,
        backoff: BackoffSeconds,
        retries: RunCounter,
        log_crash: CrashLogger,
    ) -> None:
        self._advance_chain = advance_chain
        self._context_overflow = context_overflow
        self._has_preserved_answer = has_preserved_answer
        self._has_terminal_tool_result = has_terminal_tool_result
        self._has_final_answer = has_final_answer
        self._preserved_finish = preserved_finish
        self._wind_down = wind_down
        self._wind_down_budget = wind_down_budget
        self._llm_terminal = llm_terminal
        self._fallback_event = fallback_event
        self._retry_event = retry_event
        self._commit_usage = commit_usage
        self._backoff = backoff
        self._retries = retries
        self._log_crash = log_crash

    async def apply(self, turn: TurnContext) -> AsyncIterator[TurnEvent]:
        if turn.coordinate is TurnCoordinate.stream_settled:
            # A stream came back whole, so the consecutive-failure streak this
            # policy retries against is broken: a later, independent blip in
            # the same run gets the whole in-place budget rather than the
            # remainder of an unrelated one. The count is read and charged
            # here; giving it back is the same decision and belongs beside
            # them, not in the loop that happens to notice the clean round.
            self._retries.reset(turn.engine)
            return

        if turn.coordinate is TurnCoordinate.finish_nudge:
            async for event in self._stored_failure_is_the_outcome(turn):
                yield event
            return

        exc = turn.stream_error
        if exc is None:
            return
        _logger.warning(
            "stream failed in run %s: %s: %s",
            turn.engine.config.run_id,
            type(exc).__name__,
            str(exc)[:400],
        )

        if isinstance(exc, LLMContextWindowExceeded):
            async for event in self._shrink_the_request(turn, exc):
                yield event
            return

        if isinstance(exc, LLMStreamIdleError):
            # A stream that went quiet is as often a queue as a hang: the same
            # request is tried again, bounded, before the run winds down on it.
            async for event in self._recover(
                turn, exc, kind="llm_stream_idle", retryable=True, wind_down_when_stuck=True
            ):
                yield event
            return

        if isinstance(exc, LLMRateLimitError | LLMTimeoutError):
            # The only two classes that are transient by definition, and so
            # the only two that get the same endpoint a second time. Nothing
            # winds down after them: a bounded retry that ran out has already
            # spent the time a wind-down would need.
            kind = (
                "llm_rate_limit"
                if isinstance(exc, LLMRateLimitError)
                else "llm_timeout"
            )
            async for event in self._recover(turn, exc, kind=kind, retryable=True):
                yield event
            return

        if isinstance(exc, LLMProviderError):
            # The adapters' catch-all: a 5xx, a dropped connection, a refused
            # request. The first two pass on a retry and the third costs one
            # more call against a cached prompt, so the bounded retry comes
            # before the wind-down here too.
            async for event in self._recover(
                turn, exc, kind="llm_provider_error", retryable=True, wind_down_when_stuck=True
            ):
                yield event
            return

        async for event in self._crash(turn, exc):
            yield event

    async def _stored_failure_is_the_outcome(
        self, turn: TurnContext
    ) -> AsyncIterator[TurnEvent]:
        """Report the failure a wind-down was started for, if it wrote nothing.

        The stored error is the run's outcome ONLY when the wind-down it began
        produced neither a terminal tool result nor a final answer. A wind-down
        that got the model to write its answer did the job it exists for, and
        re-raising the upstream failure over that answer would throw the
        recovery away and report a run that answered as a run that failed.
        """
        stored = turn.stored_stream_error
        engine = turn.engine
        if stored is None:
            return
        if self._has_terminal_tool_result(engine) or self._has_final_answer(engine):
            return
        exc, kind = stored
        async for event in self._llm_terminal(engine, exc, kind=kind):
            yield event
        turn.outcome.directive = TurnDirective.end_turn
        turn.outcome.reason = kind

    # -- the request was the problem ----------------------------------------

    async def _shrink_the_request(
        self, turn: TurnContext, exc: LLMContextWindowExceeded
    ) -> AsyncIterator[TurnEvent]:
        engine = turn.engine
        async for event in self._context_overflow(engine, exc):
            yield event
        if engine.is_terminal:
            turn.flags.terminal_yielded = True
            turn.outcome.directive = TurnDirective.end_turn
            turn.outcome.reason = "llm_context_window_exceeded"
            return
        turn.outcome.directive = TurnDirective.restart_turn
        turn.outcome.rebuild_context = True
        turn.outcome.reason = "context_window_recovered"

    # -- the endpoint was the problem ---------------------------------------

    async def _recover(
        self,
        turn: TurnContext,
        exc: BaseException,
        *,
        kind: str,
        retryable: bool = False,
        wind_down_when_stuck: bool = False,
    ) -> AsyncIterator[TurnEvent]:
        engine = turn.engine

        advanced_to = await self._advance_chain(engine, exc, kind=kind)
        if advanced_to:
            # The partial the reader already saw live is in the transcript,
            # and the rebuild the loop does next is what puts it in front of
            # the replacement provider. Without it the new provider answers a
            # conversation that does not contain the output on the screen.
            yield self._fallback_event(
                engine,
                advanced_to,
                exc,
                None if kind == "llm_provider_error" else kind,
            )
            turn.outcome.directive = TurnDirective.restart_turn
            turn.outcome.rebuild_context = True
            turn.outcome.reason = "model_fallback_triggered"
            return

        if retryable:
            async for event in self._retry_in_place(turn, exc, kind=kind):
                yield event
            if turn.outcome.directive is not TurnDirective.proceed:
                return

        wound = (
            self._wind_down(engine, cause=_soft_stop.CAUSE_PROVIDER_ERROR)
            if wind_down_when_stuck
            else []
        )
        if wound:
            turn.stored_stream_error = (exc, kind)
            turn.outcome.turn_budget = self._wind_down_budget(
                engine, turn.flags.assistant_message_idx
            )
            for event in wound:
                yield event
            await engine.persist_snapshot()
            turn.flags.backstop_armed = True
            turn.outcome.directive = TurnDirective.end_turn
            turn.outcome.rebuild_context = True
            turn.outcome.reason = "provider_error_wind_down"
            return

        async for event in self._terminal(turn, exc, kind=kind):
            yield event

    async def _retry_in_place(
        self, turn: TurnContext, exc: BaseException, *, kind: str
    ) -> AsyncIterator[TurnEvent]:
        """Try the same endpoint again, bounded by the failure streak.

        The bound is per consecutive-failure streak rather than per run: a
        stream that succeeds gives the whole budget back, so an unrelated blip
        later in the same run is not charged for this one.
        """
        engine = turn.engine
        if self._retries.read(engine) >= engine.rc.llm_transient_error_retry_max_attempts:
            return
        spent = self._commit_usage(
            engine,
            kind="inference",
            input_tokens=engine.total_usage.this_turn_input,
            output_tokens=engine.total_usage.this_turn_output,
            success=False,
        )
        if spent is not None:
            yield spent
        attempt = self._retries.charge(engine)
        delay = self._backoff(engine.rc, attempt, exc)
        yield self._retry_event(engine, kind, attempt, delay, exc)
        # Persisted before the pause by the attempt that failed, so a crash
        # during the backoff does not lose what the reader already saw.
        if delay > 0:
            await asyncio.sleep(delay)
        turn.outcome.directive = TurnDirective.restart_turn
        turn.outcome.rebuild_context = True
        turn.outcome.reason = "transient_llm_error_retry"

    # -- nothing recovered it -----------------------------------------------

    async def _terminal(
        self, turn: TurnContext, exc: BaseException, *, kind: str
    ) -> AsyncIterator[TurnEvent]:
        engine = turn.engine
        if self._has_preserved_answer(engine):
            # A forced continuation turn failed over a reply the reader has
            # already been given. Reporting the failure would bury it.
            async for event in self._preserved_finish(
                engine, reason="stream_error_completed_answer_preserved"
            ):
                yield event
            turn.outcome.directive = TurnDirective.end_turn
            turn.outcome.reason = "stream_error_completed_answer_preserved"
            return
        async for event in self._llm_terminal(engine, exc, kind=kind):
            yield event
        turn.flags.terminal_yielded = True
        turn.outcome.directive = TurnDirective.end_turn
        turn.outcome.reason = kind

    async def _crash(
        self, turn: TurnContext, exc: BaseException
    ) -> AsyncIterator[TurnEvent]:
        # Written here as well as by the terminal emitter: this record names
        # the turn the crash landed on, and it is written even if driving the
        # terminal then raises in turn.
        self._log_crash(turn.engine, exc)
        async for event in self._llm_terminal(
            turn.engine, exc, kind=INTERNAL_ERROR_KIND
        ):
            yield event
        turn.flags.terminal_yielded = True
        turn.outcome.directive = TurnDirective.end_turn
        turn.outcome.reason = INTERNAL_ERROR_KIND


__all__ = ["ProviderFailurePolicy"]
