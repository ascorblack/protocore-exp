"""Compaction between the tool results and the next stream.

The results of the batch just dispatched are in history, and the next
assistant stream is about to be built from all of it. Without a check here the
turn rebuilds its payload from the full transcript every iteration and grows
until an upstream context-length error stops it — a failure that arrives as a
provider error and reads as one, long after the growth that caused it.

So the gate runs where the growth happens. It protects the batch that was just
produced: those results have not been read by the model yet, and a window that
only keeps the trailing few messages would summarise the fifth-from-last of a
wide parallel batch in the same iteration that produced it.

A pass here is proactive: nothing has been rejected yet. A pass with nothing
to do is not opened, and one that exhausts the retry budget suspends the
summariser tiers for a bounded stretch instead of ending the run — the request
goes out, Tier 1 keeps running, and a real rejection is recovered by the
reactive path.
"""
from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any

from protocore.contracts.turn_policy import TurnContext, TurnCoordinate
from protocore.contracts.types import Message
from protocore.runtime.events import EventType, TurnEvent

#: Run compaction over the run's history, forced or not, under a named reason.
Compactor = Callable[..., AsyncIterator[TurnEvent]]

#: Where the in-flight tool batch starts, so compaction leaves it alone.
BatchProtectIndex = Callable[[list[Message]], int | None]

#: The calibrated size of the run's prompt as it stands.
PromptTokens = Callable[[Any], int]


class PerIterationCompactionPolicy:
    """Compact between a dispatched batch and the stream that will read it."""

    name = "per_iteration_compaction"
    coordinates = frozenset({TurnCoordinate.iteration_end})

    __slots__ = ("_compact", "_prompt_tokens", "_protect_index")

    def __init__(
        self,
        *,
        compact: Compactor,
        protect_index: BatchProtectIndex,
        prompt_tokens: PromptTokens,
    ) -> None:
        self._compact = compact
        self._protect_index = protect_index
        self._prompt_tokens = prompt_tokens

    async def apply(self, turn: TurnContext) -> AsyncIterator[TurnEvent]:
        engine = turn.engine
        rc = engine.rc
        if not rc.compaction_per_iteration_enabled:
            return
        emergency = (
            rc.compaction_emergency_proactive_enabled
            and engine.needs_emergency_compaction()
        )
        if not emergency and engine.compaction_backoff_left > 0:
            grown_past = engine.compaction_backoff_prompt_tokens * (
                1.0 + rc.compaction_no_gain_backoff_growth_ratio
            )
            if self._prompt_tokens(engine) >= grown_past:
                # The gain reading the backoff rests on describes a history
                # that is no longer there: new content arrived, and it may be
                # exactly what the pass can shed.
                engine.compaction_backoff_left = 0
            else:
                # The last routine pass changed nothing the next one could
                # improve on; paying for it every iteration is the crawl the
                # backoff exists to stop.
                engine.compaction_backoff_left -= 1
                return
        if not emergency and not engine.needs_compaction():
            return
        before = after = 0
        async for event in self._compact(
            engine,
            force=emergency,
            reason=(
                "proactive_per_iteration_emergency"
                if emergency
                else "proactive_per_iteration"
            ),
            protect_tail_from_index=self._protect_index(engine.history),
        ):
            if event.type is EventType.COMPACTION_COMPLETED:
                before = int(event.payload.get("tokens_before") or 0)
                after = int(event.payload.get("tokens_after") or 0)
            yield event
        if not emergency and before > 0 and (before - after) < rc.compaction_min_gain_ratio * before:
            engine.compaction_backoff_left = rc.compaction_no_gain_backoff_iterations
            engine.compaction_backoff_prompt_tokens = after


__all__ = ["BatchProtectIndex", "Compactor", "PerIterationCompactionPolicy", "PromptTokens"]
