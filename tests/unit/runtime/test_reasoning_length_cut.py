"""Reasoning-only output cuts use a bounded, non-growing recovery ladder."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import replace

import pytest

from protocore.contracts.llm import (
    LLMProviderError,
    LLMRequest,
    ProviderDelta,
    ProviderDeltaKind,
)
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
    def __init__(
        self, cut_rounds: int, answer: str = "42", cut_finish_reason: str = "length"
    ) -> None:
        self._cut_rounds = cut_rounds
        self._answer = answer
        self._cut_finish_reason = cut_finish_reason
        self.calls: list[LLMRequest] = []

    async def stream_with_tools(self, request: LLMRequest) -> AsyncIterator[ProviderDelta]:
        self.calls.append(request)
        if len(self.calls) <= self._cut_rounds:
            yield ProviderDelta(
                kind=ProviderDeltaKind.thinking,
                content="let me think " * 50,
            )
            if self._cut_finish_reason:
                yield ProviderDelta(
                    kind=ProviderDeltaKind.finish,
                    finish_reason=self._cut_finish_reason,
                )
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
    return [
        sum(
            len(block.text)
            for message in call.messages
            for block in message.content_blocks
            if isinstance(block, TextBlock)
        )
        for call in llm.calls
    ]


def _retry_events(events: list[TurnEvent]) -> list[TurnEvent]:
    return [
        event
        for event in events
        if event.type is EventType.STATE_CHANGED and event.payload.get("reason") == "reasoning_length_cut_retry"
    ]


async def _run(engine, llm) -> list[TurnEvent]:  # type: ignore[no-untyped-def]
    engine.llm = llm
    events: list[TurnEvent] = []
    message = Message(
        role=MessageRole.user,
        content_blocks=[TextBlock(text="what is 6*7?")],
    )
    async for event in engine.run(message):
        events.append(event)
    return events


@pytest.mark.asyncio
async def test_cut_reasoning_is_discarded_and_effort_is_lowered(engine_factory, in_memory_runtime) -> None:
    engine = engine_factory(rc=LoopConstants(model_context_window=4_096))
    engine.apply_live_controls(thinking_enabled=True, reasoning_effort="high")
    llm = _CutWhileReasoningLLM(cut_rounds=1)
    events = await _run(engine, llm)

    assert engine.state is LoopState.COMPLETED
    assert len(llm.calls) == 2
    assert not [
        message for message in engine.history if message.role is MessageRole.assistant and message.reasoning_content
    ]
    assert llm.calls[0].extra["reasoning_effort"] == "high"
    assert llm.calls[1].extra["reasoning_effort"] == "low"
    assert llm.calls[1].extra["enable_thinking"] is True
    nudges = [
        message
        for message in engine.history
        if message.metadata.get(SYNTHETIC_RECOVERY_METADATA_KEY) == SYNTHETIC_RECOVERY_REASONING_CUT
    ]
    assert len(nudges) == 1
    retries = _retry_events(events)
    assert [(event.payload["round"], event.payload["changed"]) for event in retries] == [(1, "reasoning_effort=low")]
    assert retries[0].payload["reasoning_content_chars"] > 0
    assert engine.effective_reasoning_effort == "high"
    assert engine.effective_thinking_enabled is True


@pytest.mark.asyncio
async def test_second_retry_disables_thinking_without_growing_the_prompt(engine_factory, in_memory_runtime) -> None:
    engine = engine_factory(rc=LoopConstants(model_context_window=4_096))
    engine.apply_live_controls(thinking_enabled=True, reasoning_effort="high")
    llm = _CutWhileReasoningLLM(cut_rounds=2)
    events = await _run(engine, llm)

    assert engine.state is LoopState.COMPLETED
    assert len(llm.calls) == 3
    assert llm.calls[2].extra["enable_thinking"] is False
    assert [event.payload["changed"] for event in _retry_events(events)] == [
        "reasoning_effort=low",
        "thinking=off",
    ]
    sizes = _wire_sizes(llm)
    assert sizes[1] == sizes[2]
    assert sizes[1] - sizes[0] == len(engine.config.rc.reasoning_length_cut_nudge_text)
    assert engine.effective_thinking_enabled is True
    assert engine.effective_reasoning_effort == "high"


@pytest.mark.asyncio
async def test_spent_ladder_ends_with_a_specific_failure(engine_factory, in_memory_runtime) -> None:
    engine = engine_factory(rc=LoopConstants(model_context_window=4_096, soft_stop_enabled=False))
    engine.apply_live_controls(thinking_enabled=True, reasoning_effort="high")
    llm = _CutWhileReasoningLLM(cut_rounds=10)
    events = await _run(engine, llm)

    assert engine.state is LoopState.FAILED
    assert len(llm.calls) == 3
    errors = [event for event in events if event.type is EventType.ERROR]
    assert errors[-1].payload["kind"] == "reasoning_length_cut"
    snapshot = engine.snapshot()
    assert snapshot["reasoning_recovery_effort"] is None
    assert snapshot["reasoning_recovery_thinking_enabled"] is None
    assert snapshot["reasoning_length_cut_count"] == 0


@pytest.mark.asyncio
async def test_spent_ladder_gets_one_default_wind_down_attempt_then_fails(
    engine_factory, in_memory_runtime
) -> None:
    engine = engine_factory(
        rc=LoopConstants(model_context_window=4_096, reasoning_length_cut_retries=2)
    )
    engine.apply_live_controls(thinking_enabled=True, reasoning_effort="high")
    llm = _CutWhileReasoningLLM(cut_rounds=10)
    events = await _run(engine, llm)

    assert engine.state is LoopState.FAILED
    assert len(llm.calls) == 4
    assert [event.payload["round"] for event in _retry_events(events)] == [1, 2]
    assert [
        event
        for event in events
        if event.type is EventType.STATE_CHANGED
        and event.payload.get("reason") == "soft_stop_notified"
    ]
    assert engine.effective_reasoning_effort == "high"
    assert engine.effective_thinking_enabled is True
    snapshot = engine.snapshot()
    assert snapshot["reasoning_length_cut_count"] == 0
    assert snapshot["reasoning_recovery_effort"] is None
    assert snapshot["reasoning_recovery_thinking_enabled"] is None


@pytest.mark.asyncio
async def test_spent_ladder_winds_down_as_a_model_that_stopped_progressing(
    engine_factory, in_memory_runtime
) -> None:
    """The endpoint answered every round, so the wind-down must not say it failed.

    Under the provider-error cause the notice told the model the endpoint was
    unreachable, and the model's closing message passed that on to the operator
    as the reason the run stopped.
    """
    from protocore.runtime import soft_stop as _soft_stop

    engine = engine_factory(
        rc=LoopConstants(model_context_window=4_096, reasoning_length_cut_retries=2)
    )
    engine.apply_live_controls(thinking_enabled=True, reasoning_effort="high")
    llm = _CutWhileReasoningLLM(cut_rounds=3, answer="best available answer")
    events = await _run(engine, llm)

    notified = [
        event
        for event in events
        if event.type is EventType.STATE_CHANGED
        and event.payload.get("reason") == "soft_stop_notified"
    ]
    assert [event.payload.get("soft_stop_cause") for event in notified] == [
        _soft_stop.CAUSE_MODEL_NO_PROGRESS
    ]
    notice = next(
        block.text
        for message in llm.calls[-1].messages
        if message.metadata.get(SYNTHETIC_RECOVERY_METADATA_KEY)
        == _soft_stop.SYNTHETIC_RECOVERY_SOFT_STOP
        for block in message.content_blocks
        if isinstance(block, TextBlock)
    )
    assert "model endpoint failed" not in notice
    assert "stopped making progress" in notice


@pytest.mark.asyncio
async def test_spent_ladder_can_answer_on_its_one_wind_down_attempt(
    engine_factory, in_memory_runtime
) -> None:
    engine = engine_factory(
        rc=LoopConstants(model_context_window=4_096, reasoning_length_cut_retries=2)
    )
    engine.apply_live_controls(thinking_enabled=True, reasoning_effort="high")
    llm = _CutWhileReasoningLLM(cut_rounds=3, answer="best available answer")
    events = await _run(engine, llm)

    assert engine.state is LoopState.COMPLETED
    assert len(llm.calls) == 4
    assert [event.payload["round"] for event in _retry_events(events)] == [1, 2]
    reasons = [event.payload.get("reason") for event in events]
    assert "soft_stop_notified" in reasons
    assert "soft_stop_finalized" in reasons
    stops = [event for event in events if event.type is EventType.MESSAGE_STOP]
    assert stops[-1].payload["stop_reason"] == "soft_stop"
    assert engine.effective_reasoning_effort == "high"
    assert engine.effective_thinking_enabled is True
    snapshot = engine.snapshot()
    assert snapshot["reasoning_length_cut_count"] == 0
    assert snapshot["reasoning_recovery_effort"] is None
    assert snapshot["reasoning_recovery_thinking_enabled"] is None


@pytest.mark.asyncio
async def test_a_step_that_changes_nothing_is_skipped(engine_factory, in_memory_runtime) -> None:
    engine = engine_factory(rc=LoopConstants(model_context_window=4_096, soft_stop_enabled=False))
    llm = _CutWhileReasoningLLM(cut_rounds=10)
    events = await _run(engine, llm)

    assert engine.state is LoopState.FAILED
    assert len(llm.calls) == 1
    assert _retry_events(events) == []


@pytest.mark.asyncio
async def test_zero_retries_disables_the_ladder(engine_factory, in_memory_runtime) -> None:
    engine = engine_factory(
        rc=LoopConstants(
            model_context_window=4_096,
            reasoning_length_cut_retries=0,
            soft_stop_enabled=False,
        )
    )
    engine.apply_live_controls(thinking_enabled=True, reasoning_effort="high")
    llm = _CutWhileReasoningLLM(cut_rounds=10)
    await _run(engine, llm)

    assert engine.state is LoopState.FAILED
    assert len(llm.calls) == 1
    assert engine.effective_reasoning_effort == "high"


@pytest.mark.asyncio
async def test_deep_mode_never_disables_thinking(engine_factory, in_memory_runtime) -> None:
    engine = engine_factory(rc=LoopConstants(model_context_window=4_096, soft_stop_enabled=False))
    engine.config = replace(engine.config, run_mode="deep", thinking_enabled=True)
    engine.apply_live_controls(thinking_enabled=True, reasoning_effort="high")
    llm = _CutWhileReasoningLLM(cut_rounds=10)
    events = await _run(engine, llm)

    assert engine.state is LoopState.FAILED
    assert [event.payload["changed"] for event in _retry_events(events)] == ["reasoning_effort=low"]
    assert all(call.extra["enable_thinking"] is True for call in llm.calls)


@pytest.mark.asyncio
async def test_finishless_reasoning_uses_existing_output_recovery(
    engine_factory, in_memory_runtime
) -> None:
    engine = engine_factory(rc=LoopConstants(model_context_window=4_096))
    engine.apply_live_controls(thinking_enabled=True, reasoning_effort="high")
    llm = _CutWhileReasoningLLM(cut_rounds=1, cut_finish_reason="")
    events = await _run(engine, llm)

    assert engine.state is LoopState.COMPLETED
    assert _retry_events(events) == []
    assert any(
        event.payload.get("reason") == "max_output_token_recovery"
        for event in events
    )


class _MixedThinkingFinishLLM(_CutWhileReasoningLLM):
    async def stream_with_tools(
        self, request: LLMRequest
    ) -> AsyncIterator[ProviderDelta]:
        self.calls.append(request)
        if len(self.calls) <= 2:
            yield ProviderDelta(kind=ProviderDeltaKind.thinking, content="thinking")
            yield ProviderDelta(
                kind=ProviderDeltaKind.finish,
                finish_reason="stop" if len(self.calls) == 1 else "length",
            )
            return
        yield ProviderDelta(kind=ProviderDeltaKind.text, content="done")
        yield ProviderDelta(kind=ProviderDeltaKind.finish, finish_reason="stop")


@pytest.mark.asyncio
async def test_model_ended_thinking_does_not_spend_reasoning_cut_rung(
    engine_factory, in_memory_runtime
) -> None:
    engine = engine_factory(rc=LoopConstants(model_context_window=4_096))
    engine.apply_live_controls(thinking_enabled=True, reasoning_effort="high")
    events = await _run(engine, _MixedThinkingFinishLLM(cut_rounds=0))

    retries = _retry_events(events)
    assert [(event.payload["round"], event.payload["changed"]) for event in retries] == [
        (1, "reasoning_effort=low")
    ]
    assert engine._reasoning_length_cut_count == 0


@pytest.mark.asyncio
async def test_snapshot_resume_restores_controls_after_success(engine_factory) -> None:
    from protocore.runtime.query import _step_reasoning_after_cut

    source = engine_factory(run_id="cut-run", tenant_id="cut-tenant", session_id="cut-session")
    source.apply_live_controls(thinking_enabled=True, reasoning_effort="high")
    assert _step_reasoning_after_cut(source, 1) == "reasoning_effort=low"
    source._reasoning_length_cut_count = 1

    resumed = engine_factory(
        run_id="cut-run", tenant_id="cut-tenant", session_id="cut-session"
    )
    await resumed.resume_from_snapshot(source.snapshot())
    assert resumed._reasoning_length_cut_count == 1
    assert resumed.effective_reasoning_effort == "low"
    await _run(resumed, _CutWhileReasoningLLM(cut_rounds=0))

    assert resumed._reasoning_length_cut_count == 0
    assert resumed.effective_reasoning_effort == "high"
    assert resumed.effective_thinking_enabled is True


@pytest.mark.asyncio
async def test_snapshot_resume_then_rearm_restores_controls(engine_factory) -> None:
    from protocore.runtime.query import _step_reasoning_after_cut

    source = engine_factory(run_id="cut-run", tenant_id="cut-tenant", session_id="cut-session")
    source.apply_live_controls(thinking_enabled=True, reasoning_effort="high")
    assert _step_reasoning_after_cut(source, 1) == "reasoning_effort=low"
    source._reasoning_length_cut_count = 1
    source.transition_to(LoopState.RUNNING)
    source.transition_to(LoopState.FAILED)

    resumed = engine_factory(
        run_id="cut-run", tenant_id="cut-tenant", session_id="cut-session"
    )
    await resumed.resume_from_snapshot(source.snapshot())
    resumed.rearm()

    assert resumed._reasoning_length_cut_count == 0
    assert resumed.effective_reasoning_effort == "high"
    assert resumed.effective_thinking_enabled is True


@pytest.mark.asyncio
async def test_live_reload_does_not_persist_recovery_overrides(
    engine_factory, in_memory_runtime
) -> None:
    engine = engine_factory(rc=LoopConstants(model_context_window=4_096))
    reloads: list[tuple[bool | None, str | None]] = []

    async def reload_live_control(current) -> None:  # type: ignore[no-untyped-def]
        current.apply_live_controls(thinking_enabled=True, reasoning_effort="high")
        reloads.append(
            (current._live_thinking_enabled, current._live_reasoning_effort)
        )

    engine.reload_live_control = reload_live_control
    llm = _CutWhileReasoningLLM(cut_rounds=2)
    await _run(engine, llm)

    assert reloads
    assert [call.extra["reasoning_effort"] for call in llm.calls] == [
        "high",
        "low",
        "low",
    ]
    assert [call.extra["enable_thinking"] for call in llm.calls] == [
        True,
        True,
        False,
    ]
    assert engine._live_reasoning_effort == "high"
    assert engine._live_thinking_enabled is True
    snapshot = engine.snapshot()
    assert snapshot["live_reasoning_effort"] == "high"
    assert snapshot["live_thinking_enabled"] is True
    assert snapshot["reasoning_recovery_effort"] is None
    assert snapshot["reasoning_recovery_thinking_enabled"] is None


class _ErrorAfterCutLLM(_CutWhileReasoningLLM):
    async def stream_with_tools(
        self, request: LLMRequest
    ) -> AsyncIterator[ProviderDelta]:
        if self.calls:
            raise LLMProviderError("provider unavailable")
        async for delta in super().stream_with_tools(request):
            yield delta


@pytest.mark.asyncio
async def test_provider_error_after_retry_clears_recovery_overrides(
    engine_factory, in_memory_runtime
) -> None:
    engine = engine_factory(
        rc=LoopConstants(model_context_window=4_096, soft_stop_enabled=False)
    )
    engine.apply_live_controls(thinking_enabled=True, reasoning_effort="high")
    await _run(engine, _ErrorAfterCutLLM(cut_rounds=1))

    assert engine.state is LoopState.FAILED
    assert engine._reasoning_recovery_effort is None
    assert engine._reasoning_recovery_thinking_enabled is None
    assert engine.snapshot()["reasoning_length_cut_count"] == 0


class _CancelAfterCutLLM(_CutWhileReasoningLLM):
    def __init__(self, engine) -> None:  # type: ignore[no-untyped-def]
        super().__init__(cut_rounds=1)
        self._engine = engine

    async def stream_with_tools(
        self, request: LLMRequest
    ) -> AsyncIterator[ProviderDelta]:
        if self.calls:
            self._engine.stop()
        async for delta in super().stream_with_tools(request):
            yield delta


@pytest.mark.asyncio
async def test_cancel_after_retry_clears_recovery_overrides(
    engine_factory, in_memory_runtime
) -> None:
    engine = engine_factory(rc=LoopConstants(model_context_window=4_096))
    engine.apply_live_controls(thinking_enabled=True, reasoning_effort="high")
    await _run(engine, _CancelAfterCutLLM(engine))

    assert engine.state is LoopState.CANCELLED
    assert engine._reasoning_recovery_effort is None
    assert engine._reasoning_recovery_thinking_enabled is None
    assert engine.snapshot()["reasoning_length_cut_count"] == 0


def test_rearm_restores_controls_after_a_failed_recovery(engine_factory) -> None:
    engine = engine_factory(rc=LoopConstants(model_context_window=4_096))
    engine.apply_live_controls(thinking_enabled=True, reasoning_effort="high")
    from protocore.runtime.query import _step_reasoning_after_cut

    assert _step_reasoning_after_cut(engine, 1) == "reasoning_effort=low"
    engine.transition_to(LoopState.RUNNING)
    engine.transition_to(LoopState.FAILED)
    engine.rearm()

    assert engine._reasoning_recovery_effort is None
    assert engine._reasoning_recovery_thinking_enabled is None
    assert engine.effective_thinking_enabled is True
    assert engine.effective_reasoning_effort == "high"
