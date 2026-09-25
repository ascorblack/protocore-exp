"""A unit the summariser keeps failing on is left alone, and only that unit.

The pass used to be all-or-nothing: any failed call discarded everything the
other units in the batch had produced, so one oversized turn could keep a run
from shedding a single token, and the next pass paid for the same failure
again. The failures are counted per unit instead, and the units beside them
commit.
"""
from __future__ import annotations

import json
from typing import Any

import pytest

from protocore.contracts.llm import (
    LLMContextWindowExceeded,
    LLMRequest,
    LLMResponse,
    StopReason,
)
from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.types import Message, MessageRole, TextBlock
from protocore.runtime.context.compaction import (
    CompactionState,
    run_tier2_summarisation,
)
from protocore.tests_support.adapters import InMemoryLLMProvider


def _rc(**overrides: Any) -> LoopConstants:
    base: dict[str, Any] = {
        "model_context_window": 4_096,
        "compaction_keep_recent_turns": 1,
        "compaction_protect_first_user_turn": False,
        "compaction_summary_min_unit_tokens": 0,
        "compaction_summariser_parallelism": 4,
        # The census is per unit: one unit, one call.
        "compaction_summary_group_max_tokens": 0,
    }
    base.update(overrides)
    return LoopConstants(**base)


def _unit(marker: str) -> Message:
    return Message(
        role=MessageRole.assistant,
        content_blocks=[TextBlock(text=f"{marker} did a long thing " * 40)],
    )


def _history(*markers: str) -> list[Message]:
    messages = [_unit(marker) for marker in markers]
    messages.append(Message(role=MessageRole.user, content_blocks=[TextBlock(text="recent")]))
    return messages


class _Raising(InMemoryLLMProvider):
    """Every call is refused because the unit does not fit the summariser."""

    def __init__(self) -> None:
        super().__init__()
        self.text_calls = 0

    async def complete_text(self, request: LLMRequest) -> LLMResponse:
        self.text_calls += 1
        raise LLMContextWindowExceeded("this unit does not fit")


class _Flaky(InMemoryLLMProvider):
    """Every call fails the way a rate limit or a recycled pod fails."""

    def __init__(self) -> None:
        super().__init__()
        self.text_calls = 0

    async def complete_text(self, request: LLMRequest) -> LLMResponse:
        self.text_calls += 1
        raise RuntimeError("429 Too Many Requests")


class _EmptyReply(InMemoryLLMProvider):
    """Every reply parses but carries no summary — what a cut envelope looks like."""

    def __init__(self) -> None:
        super().__init__()
        self.text_calls = 0

    async def complete_text(self, request: LLMRequest) -> LLMResponse:
        self.text_calls += 1
        return LLMResponse(
            message=Message(role=MessageRole.assistant, content_blocks=[TextBlock(text="")]),
            stop_reason=StopReason.max_tokens,
        )


class _NoSmaller(InMemoryLLMProvider):
    """Every summary comes back larger than the unit it would replace."""

    def __init__(self) -> None:
        super().__init__()
        self.text_calls = 0

    async def complete_text(self, request: LLMRequest) -> LLMResponse:
        self.text_calls += 1
        return LLMResponse(
            message=Message(
                role=MessageRole.assistant,
                content_blocks=[
                    TextBlock(text=json.dumps({"summary": "verbose restatement " * 200}))
                ],
            ),
            stop_reason=StopReason.end_turn,
        )


class _OneUnitFails(InMemoryLLMProvider):
    """Refuses the unit whose text carries ``marker``; summarises the rest."""

    def __init__(self, marker: str) -> None:
        super().__init__()
        self._marker = marker
        self.text_calls = 0

    async def complete_text(self, request: LLMRequest) -> LLMResponse:
        self.text_calls += 1
        if self._marker in request.messages[-1].text:
            raise LLMContextWindowExceeded("this unit never fits")
        return LLMResponse(
            message=Message(
                role=MessageRole.assistant,
                content_blocks=[TextBlock(text=json.dumps({"summary": "short"}))],
            ),
            stop_reason=StopReason.end_turn,
        )


@pytest.mark.asyncio
async def test_a_unit_the_summariser_cannot_fit_is_left_alone_after_the_limit() -> None:
    rc = _rc(compaction_summary_failed_unit_max_attempts=2)
    history = _history("alpha")
    state = CompactionState()
    llm = _Raising()

    for _ in range(4):
        result = await run_tier2_summarisation(
            history=history, compaction_llm=llm, state=state, rc=rc, model_name="mock"
        )
        assert result.turns_summarised == 0

    # Two passes paid for the failure; the two after them did not.
    assert llm.text_calls == 2
    assert list(state.failed_anchor_keys.values()) == [2]
    assert history[-1].text == "recent"


@pytest.mark.asyncio
async def test_a_reply_with_no_summary_in_it_counts_as_a_failed_call() -> None:
    """Deliberate: an unreadable reply is the shape a cut-off reply takes, and
    repeating it on the same unit produces the same cut in the same place."""
    rc = _rc(compaction_summary_failed_unit_max_attempts=1)
    history = _history("alpha")
    state = CompactionState()
    llm = _EmptyReply()

    for _ in range(3):
        await run_tier2_summarisation(
            history=history, compaction_llm=llm, state=state, rc=rc, model_name="mock"
        )

    assert llm.text_calls == 1
    assert list(state.failed_anchor_keys.values()) == [1]


@pytest.mark.asyncio
async def test_a_summary_that_is_merely_no_smaller_is_not_held_against_the_unit() -> None:
    """The other half of the same decision: a summary that came back too big is
    a fact about this unit's size, not evidence the call cannot complete, so it
    is discarded without counting against the unit."""
    rc = _rc(compaction_summary_failed_unit_max_attempts=1)
    history = _history("alpha")
    state = CompactionState()
    llm = _NoSmaller()

    for _ in range(3):
        result = await run_tier2_summarisation(
            history=history, compaction_llm=llm, state=state, rc=rc, model_name="mock"
        )
        assert result.turns_summarised == 0

    assert llm.text_calls == 3
    assert state.failed_anchor_keys == {}


@pytest.mark.asyncio
async def test_one_failed_unit_does_not_discard_what_the_others_produced() -> None:
    rc = _rc()
    history = _history("alpha", "bravo", "charlie")
    state = CompactionState()
    llm = _OneUnitFails("bravo")

    result = await run_tier2_summarisation(
        history=history, compaction_llm=llm, state=state, rc=rc, model_name="mock"
    )

    assert result.turns_summarised == 2
    assert result.tokens_freed > 0
    assert len(state.failed_anchor_keys) == 1
    # The failed unit is still in the history, whole.
    assert any("bravo did a long thing" in message.text for message in history)
    assert not any("alpha did a long thing" in message.text for message in history)


@pytest.mark.asyncio
async def test_the_failure_census_survives_a_snapshot_round_trip() -> None:
    rc = _rc(compaction_summary_failed_unit_max_attempts=2)
    history = _history("alpha")
    state = CompactionState()
    llm = _Raising()

    await run_tier2_summarisation(
        history=history, compaction_llm=llm, state=state, rc=rc, model_name="mock"
    )
    assert list(state.failed_anchor_keys.values()) == [1]

    # What the engine snapshot carries and hands back.
    serialised = dict(state.failed_anchor_keys)
    rehydrated = CompactionState(
        failed_anchor_keys={str(k): int(v) for k, v in serialised.items()}
    )

    await run_tier2_summarisation(
        history=history, compaction_llm=llm, state=rehydrated, rc=rc, model_name="mock"
    )
    assert list(rehydrated.failed_anchor_keys.values()) == [2]

    calls_so_far = llm.text_calls
    await run_tier2_summarisation(
        history=history, compaction_llm=llm, state=rehydrated, rc=rc, model_name="mock"
    )
    assert llm.text_calls == calls_so_far


@pytest.mark.asyncio
async def test_a_transport_failure_is_not_held_against_the_unit() -> None:
    """A rate limit, a 5xx or a recycled summariser pod says nothing about the
    unit. Counting one would let a single blip across a parallel batch retire
    several units for the rest of the run — and across every resume, since the
    census rides the snapshot."""
    rc = _rc(compaction_summary_failed_unit_max_attempts=1)
    history = _history("alpha", "bravo", "charlie")
    state = CompactionState()
    llm = _Flaky()

    for _ in range(3):
        await run_tier2_summarisation(
            history=history, compaction_llm=llm, state=state, rc=rc, model_name="mock"
        )

    assert state.failed_anchor_keys == {}
    # Every pass still tried every unit.
    assert llm.text_calls == 9


@pytest.mark.asyncio
async def test_a_forced_pass_tries_the_units_the_routine_gate_has_written_off() -> None:
    """A forced pass runs when the alternative is the run ending, so it ignores
    the census rather than inheriting a verdict reached under lighter pressure."""
    rc = _rc(compaction_summary_failed_unit_max_attempts=1)
    history = _history("alpha")
    state = CompactionState()
    llm = _Raising()

    await run_tier2_summarisation(
        history=history, compaction_llm=llm, state=state, rc=rc, model_name="mock"
    )
    assert list(state.failed_anchor_keys.values()) == [1]

    # The routine gate now skips it.
    calls_after_first = llm.text_calls
    await run_tier2_summarisation(
        history=history, compaction_llm=llm, state=state, rc=rc, model_name="mock"
    )
    assert llm.text_calls == calls_after_first

    # The forced pass does not.
    await run_tier2_summarisation(
        history=history,
        compaction_llm=llm,
        state=state,
        rc=rc,
        model_name="mock",
        retry_failed_units=True,
    )
    assert llm.text_calls == calls_after_first + 1


@pytest.mark.asyncio
async def test_a_census_entry_whose_unit_has_left_the_history_is_forgotten() -> None:
    rc = _rc(compaction_summary_failed_unit_max_attempts=5)
    history = _history("alpha")
    state = CompactionState()

    await run_tier2_summarisation(
        history=history, compaction_llm=_Raising(), state=state, rc=rc, model_name="mock"
    )
    assert len(state.failed_anchor_keys) == 1

    # The unit is gone — folded away, dropped at a checkpoint, replaced.
    history[:] = [Message(role=MessageRole.user, content_blocks=[TextBlock(text="recent")])]
    await run_tier2_summarisation(
        history=history, compaction_llm=_Raising(), state=state, rc=rc, model_name="mock"
    )

    assert state.failed_anchor_keys == {}
