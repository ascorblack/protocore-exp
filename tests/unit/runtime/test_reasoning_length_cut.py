"""A round the output cap cut while the model was still reasoning.

The cut reasoning is not kept, the retry sends the same prompt with one knob
turned down, the ladder is bounded, and the knobs come back once a round
produces something.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import replace

import pytest

from protocore.contracts.llm import LLMRequest, ProviderDelta, ProviderDeltaKind
from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.types import (
    SYNTHETIC_RECOVERY_METADATA_KEY,
    SYNTHETIC_RECOVERY_REASONING_CUT,
    Message,
    MessageRole,
    TextBlock,
)
from protocore.runtime.events import EventType, TurnEvent
from protocore.runtime.loop_state import LoopState


class _CutWhileReasoningLLM:
    """Reasoning only and ``finish_reason='length'`` for the first N calls, then an answer."""

    def __init__(self, cut_rounds: int, answer: str = "42") -> None:
        self._cut_rounds = cut_rounds
        self._answer = answer
        self.calls: list[LLMRequest] = []

    async def stream_with_tools(  # type: ignore[no-untyped-def]
        self, request: LLMRequest
    ) -> AsyncIterator[ProviderDelta]:
        self.calls.append(request)
        if len(self.calls) <= self._cut_rounds:
            yield ProviderDelta(kind=ProviderDeltaKind.thinking, content="let me think " * 50)
            yield ProviderDelta(kind=ProviderDeltaKind.finish, finish_reason="length")
            return
        yield ProviderDelta(kind=ProviderDeltaKind.text, content=self._answer)
        yield ProviderDelta(kind=ProviderDeltaKind.finish, finish_reason="stop")

    async def complete_structured(self, request, schema):  # type: ignore[no-untyped-def]
        from protocore.contracts.llm import LLMResponse
        from protocore.contracts.types import StopReason

        return LLMResponse(
            message=Message(role=MessageRole.assistant, content_blocks=[]),
            stop_reason=StopReason.end_turn,
        )

    def count_tokens(self, text, model=None) -> int:  # type: ignore[no-untyped-def]
        return max(1, len(text) // 4)


def _wire_sizes(llm: _CutWhileReasoningLLM) -> list[int]:
    return [sum(len(block.text) for m in call.messages for block in m.content_blocks if isinstance(block, TextBlock)) for call in llm.calls]


def _retry_events(events: list[TurnEvent]) -> list[TurnEvent]:
    return [e for e in events if e.type is EventType.STATE_CHANGED and e.payload.get("reason") == "reasoning_length_cut_retry"]


async def _run(engine, llm) -> list[TurnEvent]:  # type: ignore[no-untyped-def]
    engine.llm = llm
    events: list[TurnEvent] = []
    async for evt in engine.run(Message(role=MessageRole.user, content_blocks=[TextBlock(text="what is 6*7?")])):
        events.append(evt)
    return events


@pytest.mark.asyncio
async def test_cut_reasoning_is_not_kept_and_the_retry_turns_effort_down(engine_factory, in_memory_runtime) -> None:
    engine = engine_factory(rc=LoopConstants(model_context_window=4_096))
    engine.apply_live_controls(thinking_enabled=True, reasoning_effort="high")
    llm = _CutWhileReasoningLLM(cut_rounds=1)
    events = await _run(engine, llm)

    assert engine.state is LoopState.COMPLETED
    assert len(llm.calls) == 2
    # Nothing from the cut round went back on the wire: no assistant message carries its reasoning.
    assert not [m for m in engine.history if m.role is MessageRole.assistant and m.reasoning_content]
    # The retry differs by the nudge and the knob, not by the cut reasoning.
    assert llm.calls[0].extra["reasoning_effort"] == "high"
    assert llm.calls[1].extra["reasoning_effort"] == "low"
    assert llm.calls[1].extra["enable_thinking"] is True
    nudges = [m for m in engine.history if m.metadata.get(SYNTHETIC_RECOVERY_METADATA_KEY) == SYNTHETIC_RECOVERY_REASONING_CUT]
    assert len(nudges) == 1
    retries = _retry_events(events)
    assert [(e.payload["round"], e.payload["changed"]) for e in retries] == [(1, "reasoning_effort=low")]
    assert retries[0].payload["reasoning_content_chars"] > 0
    # Once a round produced something the knobs are back where the operator set them.
    assert engine.effective_reasoning_effort == "high"
    assert engine.effective_thinking_enabled is True


@pytest.mark.asyncio
async def test_second_retry_switches_thinking_off_and_the_prompt_does_not_grow(engine_factory, in_memory_runtime) -> None:
    engine = engine_factory(rc=LoopConstants(model_context_window=4_096))
    engine.apply_live_controls(thinking_enabled=True, reasoning_effort="high")
    llm = _CutWhileReasoningLLM(cut_rounds=2)
    events = await _run(engine, llm)

    assert engine.state is LoopState.COMPLETED
    assert len(llm.calls) == 3
    assert llm.calls[2].extra["enable_thinking"] is False
    assert [e.payload["changed"] for e in _retry_events(events)] == ["reasoning_effort=low", "thinking=off"]
    sizes = _wire_sizes(llm)
    # The second retry carries exactly what the first did: the one nudge and no reasoning.
    assert sizes[1] == sizes[2]
    assert sizes[1] - sizes[0] == len(engine.config.rc.reasoning_length_cut_nudge_text)
    assert engine.effective_thinking_enabled is True
    assert engine.effective_reasoning_effort == "high"


@pytest.mark.asyncio
async def test_ladder_spent_ends_the_run_with_its_own_kind(engine_factory, in_memory_runtime) -> None:
    engine = engine_factory(rc=LoopConstants(model_context_window=4_096, soft_stop_enabled=False))
    engine.apply_live_controls(thinking_enabled=True, reasoning_effort="high")
    llm = _CutWhileReasoningLLM(cut_rounds=10)
    events = await _run(engine, llm)

    assert engine.state is LoopState.FAILED
    # The original, effort low, thinking off — and no fourth attempt.
    assert len(llm.calls) == 3
    errors = [e for e in events if e.type is EventType.ERROR]
    assert errors and errors[-1].payload["kind"] == "reasoning_length_cut"


@pytest.mark.asyncio
async def test_a_step_that_changes_nothing_is_skipped(engine_factory, in_memory_runtime) -> None:
    """Thinking already off: no knob to turn, so the first cut is the last."""
    engine = engine_factory(rc=LoopConstants(model_context_window=4_096, soft_stop_enabled=False))
    llm = _CutWhileReasoningLLM(cut_rounds=10)
    events = await _run(engine, llm)

    assert engine.state is LoopState.FAILED
    assert len(llm.calls) == 1
    assert _retry_events(events) == []


@pytest.mark.asyncio
async def test_retries_zero_disables_the_ladder(engine_factory, in_memory_runtime) -> None:
    engine = engine_factory(rc=LoopConstants(model_context_window=4_096, reasoning_length_cut_retries=0, soft_stop_enabled=False))
    engine.apply_live_controls(thinking_enabled=True, reasoning_effort="high")
    llm = _CutWhileReasoningLLM(cut_rounds=10)
    await _run(engine, llm)

    assert engine.state is LoopState.FAILED
    assert len(llm.calls) == 1
    assert engine.effective_reasoning_effort == "high"


@pytest.mark.asyncio
async def test_thinking_stays_on_in_deep_mode(engine_factory, in_memory_runtime) -> None:
    engine = engine_factory(rc=LoopConstants(model_context_window=4_096, soft_stop_enabled=False))
    engine.config = replace(engine.config, run_mode="deep", thinking_enabled=True)
    engine.apply_live_controls(thinking_enabled=True, reasoning_effort="high")
    llm = _CutWhileReasoningLLM(cut_rounds=10)
    events = await _run(engine, llm)

    assert engine.state is LoopState.FAILED
    assert [e.payload["changed"] for e in _retry_events(events)] == ["reasoning_effort=low"]
    assert all(call.extra["enable_thinking"] is True for call in llm.calls)
