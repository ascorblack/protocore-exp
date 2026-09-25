"""Tests for agentic-core hardening.

Covers reactive-413 compaction, fallback-model recovery, max-output-token
recovery, death-spiral guard, and continue-prompt fallback.
"""
from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import pytest

from protocore.contracts.llm import (
    LLMContextWindowExceeded,
    LLMProviderError,
    LLMRequest,
    LLMStreamEvent,
    LLMStreamIdleError,
)
from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.types import (
    PARTIAL_ASSISTANT_ATTEMPT_METADATA_KEY,
    SESSION_HISTORY_SEED_METADATA_KEY,
    HookEvent,
    Message,
    MessageRole,
    StopReason,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from protocore.prompts import bundled_prompt_provider
from protocore.runtime.events import EventType, TurnEvent
from protocore.runtime.loop_state import LoopState

from ._tool_fixtures import MockTool

# ----------------------------------------------------------------------
# Provider-chain doubles
# ----------------------------------------------------------------------


class _FakeProviderChain:
    """Minimal chain over one scripted LLM presented under several names.

    Every rung hands back the SAME provider object so a scripted failure
    sequence keeps running across a swap; only the model identity changes.
    """

    def __init__(self, provider: object, names: list[str]) -> None:
        self._provider = provider
        self._names = names
        self._index = 0
        self._attempted: list[tuple[str, str]] = []

    def current(self) -> object:
        return self._provider

    def current_model_name(self) -> str:
        return self._names[self._index]

    async def advance(self, *, reason: str) -> bool:
        if self._index + 1 >= len(self._names):
            return False
        self._attempted.append((self._names[self._index], reason))
        self._index += 1
        return True

    def attempted(self) -> list[tuple[str, str]]:
        return list(self._attempted)


def _attach_chain(engine, llm: object, *names: str) -> _FakeProviderChain:
    """Wire ``llm`` onto ``engine`` behind a chain listing ``names`` in order."""
    chain = _FakeProviderChain(llm, list(names))
    engine.llm = llm
    engine.provider_chain = chain
    return chain


class _Verdict:
    """Stand-in for the classification the adapter layer pins on an error."""

    def __init__(self, reason: str) -> None:
        self.reason = reason


def _classified(exc: BaseException, reason: str) -> BaseException:
    """Attach a classification to ``exc`` the way a provider adapter does."""
    object.__setattr__(exc, "classified", _Verdict(reason))
    return exc


# ----------------------------------------------------------------------
# Programmable LLM mocks for the recovery tests.
# ----------------------------------------------------------------------


class _ScriptedFailureLLM:
    """LLM mock that raises a specified exception on each call.

    Used to drive the recovery branches deterministically. After the
    list of exceptions is exhausted the mock falls back to emitting a
    healthy text turn.
    """

    def __init__(
        self,
        exceptions: list[BaseException | None],
        recovery_text: str = "recovered",
        recovery_stop: StopReason = StopReason.end_turn,
    ) -> None:
        self._exceptions = exceptions
        self._call_idx = 0
        self._recovery_text = recovery_text
        self._recovery_stop = recovery_stop
        self.calls: list[LLMRequest] = []

    async def stream_with_tools(  # type: ignore[no-untyped-def]
        self, request: LLMRequest
    ) -> AsyncIterator[LLMStreamEvent]:
        self.calls.append(request)
        idx = self._call_idx
        self._call_idx += 1
        if idx < len(self._exceptions):
            exc = self._exceptions[idx]
            if exc is not None:
                raise exc
        # Successful recovery turn.
        yield LLMStreamEvent(name="message_start", payload={})
        yield LLMStreamEvent(name="content_block_start", payload={"kind": "text"})
        yield LLMStreamEvent(
            name="content_block_delta",
            payload={"text": self._recovery_text},
        )
        yield LLMStreamEvent(name="content_block_stop", payload={})
        yield LLMStreamEvent(
            name="message_stop",
            payload={"stop_reason": self._recovery_stop.value},
        )

    async def complete_structured(self, request, schema):  # type: ignore[no-untyped-def]
        from protocore.contracts.llm import LLMResponse

        return LLMResponse(
            message=Message(
                role=MessageRole.assistant,
                content_blocks=[TextBlock(text="brief summary")],
            ),
            stop_reason=StopReason.end_turn,
        )

    def count_tokens(self, text, model=None) -> int:  # type: ignore[no-untyped-def]
        return max(1, len(text) // 4)


class _FirstCallOverflowLLM:
    """Reject one request, then delegate later internal iterations."""

    def __init__(self, delegate) -> None:  # type: ignore[no-untyped-def]
        self._delegate = delegate
        self.calls: list[LLMRequest] = []

    async def stream_with_tools(  # type: ignore[no-untyped-def]
        self, request: LLMRequest
    ) -> AsyncIterator[LLMStreamEvent]:
        self.calls.append(request)
        if len(self.calls) == 1:
            raise LLMContextWindowExceeded("request exceeded context")
        async for event in self._delegate.stream_with_tools(request):
            yield event

    async def complete_structured(self, request, schema):  # type: ignore[no-untyped-def]
        return await self._delegate.complete_structured(request, schema)

    def count_tokens(self, text, model=None) -> int:  # type: ignore[no-untyped-def]
        return int(self._delegate.count_tokens(text, model))


# ----------------------------------------------------------------------
# Partial-attempt persistence on fallback + backstop paths
# ----------------------------------------------------------------------


class _PartialTextThenFailLLM:
    """LLM mock that streams text deltas then raises a typed error.

    Drives the fallback-model and backstop branches with a *visible* partial
    answer in the SSE stream. The recovery / terminal arm must
    persist the partial text into ``engine.history`` so reload
    shows the same content the live stream showed the user.

    ``recovery_text`` is emitted by the SECOND (post-fail) call
    so a successful fallback / backstop produces a non-empty
    follow-up answer; a terminal-on-fail path never reaches the
    second call.
    """

    def __init__(
        self,
        exception: BaseException,
        partial_text: str,
        recovery_text: str = "fallback-answer",
    ) -> None:
        self._exception = exception
        self._partial_text = partial_text
        self._recovery_text = recovery_text
        self._call_idx = 0
        self.calls: list[LLMRequest] = []

    async def stream_with_tools(  # type: ignore[no-untyped-def]
        self, request: LLMRequest
    ) -> AsyncIterator[LLMStreamEvent]:
        self.calls.append(request)
        idx = self._call_idx
        self._call_idx += 1
        if idx == 0:
            # Stream partial text — these deltas are forwarded to
            # SSE consumers live, but never make it to
            # ``engine.history`` on a fail before the fix.
            yield LLMStreamEvent(name="message_start", payload={})
            yield LLMStreamEvent(
                name="content_block_start", payload={"kind": "text"}
            )
            yield LLMStreamEvent(
                name="content_block_delta",
                payload={"text": self._partial_text},
            )
            yield LLMStreamEvent(name="content_block_stop", payload={})
            raise self._exception
        # Subsequent calls: healthy end_turn with the recovery text.
        yield LLMStreamEvent(name="message_start", payload={})
        yield LLMStreamEvent(
            name="content_block_start", payload={"kind": "text"}
        )
        yield LLMStreamEvent(
            name="content_block_delta",
            payload={"text": self._recovery_text},
        )
        yield LLMStreamEvent(name="content_block_stop", payload={})
        yield LLMStreamEvent(
            name="message_stop", payload={"stop_reason": "end_turn"}
        )

    async def complete_structured(self, request, schema):  # type: ignore[no-untyped-def]
        from protocore.contracts.llm import LLMResponse

        return LLMResponse(
            message=Message(
                role=MessageRole.assistant,
                content_blocks=[TextBlock(text="brief summary")],
            ),
            stop_reason=StopReason.end_turn,
        )

    def count_tokens(self, text, model=None) -> int:  # type: ignore[no-untyped-def]
        return max(1, len(text) // 4)


# ----------------------------------------------------------------------
# Context-window-overflow recovery
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_context_window_exceeded_triggers_force_compaction(
    engine_factory, in_memory_runtime
) -> None:
    """``LLMContextWindowExceeded`` first time → force_compaction + re-stream.

    The mock raises PTL on call 0, then emits a healthy text turn on
    call 1. The loop must:

    1. Catch the PTL.
    2. Transition RUNNING → COMPACTING.
    3. Emit ``compaction_started(reason="reactive_413")``.
    4. Force-compaction completes (Tier 1 + Tier 2).
    5. Transition COMPACTING → RUNNING.
    6. Re-open the LLM stream.
    7. Terminate cleanly in COMPLETED.
    """
    rc = LoopConstants(
        model_context_window=4096,
        compaction_keep_recent_turns=1,
    )
    engine = engine_factory(rc=rc)
    failing_llm = _ScriptedFailureLLM(
        exceptions=[LLMContextWindowExceeded("simulated PTL")],
        recovery_text="recovered",
    )
    engine.llm = failing_llm  # type: ignore[assignment]
    # ContextManager uses compaction_llm; rebind to the same mock.
    engine.context_manager._compaction_llm = failing_llm  # type: ignore[attr-defined]
    engine.compaction_llm = failing_llm  # type: ignore[assignment]

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    events: list[TurnEvent] = []
    async for evt in engine.run(user_msg):
        events.append(evt)

    # The PTL → compaction → re-stream cycle ran.
    compaction_started = [
        e for e in events if e.type is EventType.COMPACTION_STARTED
    ]
    compaction_completed = [
        e for e in events if e.type is EventType.COMPACTION_COMPLETED
    ]
    assert compaction_started, "expected compaction_started on reactive_413"
    assert compaction_completed, "expected compaction_completed on reactive_413"
    # Overflow-recovery reason surfaced on the events.
    assert compaction_started[0].payload.get("reason") == "reactive_413"
    assert compaction_completed[0].payload.get("reason") == "reactive_413"

    # Engine ends COMPLETED — second LLM call succeeded.
    assert engine.state is LoopState.COMPLETED
    assert len(failing_llm.calls) == 2


@pytest.mark.asyncio
async def test_context_window_retry_halves_the_rejected_output_cap(
    engine_factory, in_memory_runtime
) -> None:
    rc = LoopConstants(
        model_context_window=65_536,
        llm_output_max_tokens_ratio=0.125,
        compaction_keep_recent_turns=1,
    )
    engine = engine_factory(rc=rc)
    llm = _ScriptedFailureLLM(
        exceptions=[LLMContextWindowExceeded("request exceeded context")],
    )
    engine.llm = llm  # type: ignore[assignment]
    engine.context_manager._compaction_llm = llm  # type: ignore[attr-defined]
    engine.compaction_llm = llm  # type: ignore[assignment]

    async for _ in engine.run(
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    ):
        pass

    assert engine.state is LoopState.COMPLETED
    assert [request.max_tokens for request in llm.calls] == [8_192, 4_096]


@pytest.mark.asyncio
async def test_context_window_retry_ceiling_uses_the_fitted_wire_cap(
    engine_factory, in_memory_runtime, monkeypatch: pytest.MonkeyPatch
) -> None:
    rc = LoopConstants(
        model_context_window=65_536,
        llm_output_max_tokens_ratio=0.125,
        compaction_keep_recent_turns=1,
    )
    engine = engine_factory(rc=rc)
    llm = _ScriptedFailureLLM(
        exceptions=[LLMContextWindowExceeded("request exceeded context")],
    )
    engine.llm = llm  # type: ignore[assignment]
    engine.context_manager._compaction_llm = llm  # type: ignore[attr-defined]
    engine.compaction_llm = llm  # type: ignore[assignment]
    prompt_estimates = iter(
        [
            rc.model_context_window - rc.request_context_safety_tokens - 3_000,
            1,
        ]
    )
    monkeypatch.setattr(
        "protocore.runtime.request_budget.estimate_request_prompt_tokens",
        lambda request, constants: next(prompt_estimates),
    )

    async for _ in engine.run(
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    ):
        pass

    assert engine.state is LoopState.COMPLETED
    assert [request.max_tokens for request in llm.calls] == [3_000, 1_500]


@pytest.mark.asyncio
async def test_context_window_retry_uses_provider_reported_prompt_size(
    engine_factory, in_memory_runtime
) -> None:
    rc = LoopConstants(
        model_context_window=65_536,
        request_context_safety_tokens=2_048,
        llm_output_max_tokens_ratio=0.25,
        compaction_keep_recent_turns=1,
    )
    engine = engine_factory(rc=rc)
    llm = _ScriptedFailureLLM(
        exceptions=[
            LLMContextWindowExceeded(
                "request exceeded context",
                context_window=65_536,
                input_tokens=58_475,
                requested_output_tokens=7_062,
            )
        ],
    )
    engine.llm = llm  # type: ignore[assignment]
    engine.context_manager._compaction_llm = llm  # type: ignore[attr-defined]
    engine.compaction_llm = llm  # type: ignore[assignment]

    events = [
        event
        async for event in engine.run(
            Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
        )
    ]

    assert engine.state is LoopState.COMPLETED
    assert [request.max_tokens for request in llm.calls] == [16_384, 3_531]
    assert not any(event.type is EventType.COMPACTION_STARTED for event in events)
    assert any(
        event.payload.get("reason") == "context_overflow_corrective_retry"
        for event in events
    )


@pytest.mark.asyncio
async def test_one_token_provider_overflow_retries_before_no_progress_compaction(
    engine_factory, in_memory_runtime
) -> None:
    rc = LoopConstants(
        model_context_window=65_536,
        request_context_safety_tokens=2_048,
        llm_output_max_tokens_ratio=0.125,
        compaction_keep_recent_turns=1,
    )
    engine = engine_factory(rc=rc)
    llm = _ScriptedFailureLLM(
        exceptions=[
            LLMContextWindowExceeded(
                "provider total exceeded the window by one token",
                context_window=65_536,
                input_tokens=63_490,
                requested_output_tokens=2_047,
            )
        ],
    )
    engine.llm = llm  # type: ignore[assignment]
    engine.context_manager._compaction_llm = llm  # type: ignore[attr-defined]
    engine.compaction_llm = llm  # type: ignore[assignment]

    events = [
        event
        async for event in engine.run(
            Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
        )
    ]

    assert engine.state is LoopState.COMPLETED
    assert [request.max_tokens for request in llm.calls] == [8_192, 1_023]
    assert not any(event.type is EventType.COMPACTION_STARTED for event in events)


@pytest.mark.asyncio
async def test_failed_direct_correction_then_compacts_once(
    engine_factory, in_memory_runtime
) -> None:
    rc = LoopConstants(
        model_context_window=65_536,
        request_context_safety_tokens=2_048,
        llm_output_max_tokens_ratio=0.125,
        compaction_keep_recent_turns=1,
    )
    engine = engine_factory(rc=rc)
    llm = _ScriptedFailureLLM(
        exceptions=[
            LLMContextWindowExceeded(
                "measured overflow",
                context_window=65_536,
                input_tokens=60_000,
                requested_output_tokens=8_192,
            ),
            LLMContextWindowExceeded("corrective request still overflowed"),
        ],
    )
    engine.llm = llm  # type: ignore[assignment]
    engine.context_manager._compaction_llm = llm  # type: ignore[attr-defined]
    engine.compaction_llm = llm  # type: ignore[assignment]

    events = [
        event
        async for event in engine.run(
            Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
        )
    ]

    assert engine.state is LoopState.COMPLETED
    assert [request.max_tokens for request in llm.calls] == [8_192, 3_488, 1_744]
    assert sum(event.type is EventType.COMPACTION_STARTED for event in events) == 1


@pytest.mark.asyncio
async def test_compacted_overflow_gets_one_measured_corrective_retry(
    engine_factory, in_memory_runtime
) -> None:
    rc = LoopConstants(
        model_context_window=65_536,
        request_context_safety_tokens=2_048,
        llm_output_max_tokens_ratio=0.25,
        compaction_keep_recent_turns=1,
    )
    engine = engine_factory(rc=rc)
    llm = _ScriptedFailureLLM(
        exceptions=[
            LLMContextWindowExceeded("pre-compaction overflow"),
            LLMContextWindowExceeded(
                "compacted request overflowed",
                context_window=65_536,
                input_tokens=58_475,
                requested_output_tokens=7_062,
            ),
        ],
    )
    engine.llm = llm  # type: ignore[assignment]
    engine.context_manager._compaction_llm = llm  # type: ignore[attr-defined]
    engine.compaction_llm = llm  # type: ignore[assignment]

    events = [
        event
        async for event in engine.run(
            Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
        )
    ]

    assert engine.state is LoopState.COMPLETED
    assert [request.max_tokens for request in llm.calls] == [16_384, 8_192, 3_531]
    assert any(
        event.payload.get("reason") == "context_overflow_corrective_retry"
        for event in events
    )


@pytest.mark.asyncio
async def test_context_window_retry_ceiling_does_not_leak_to_the_next_tool_iteration(
    engine_factory, in_memory_runtime
) -> None:
    rc = LoopConstants(
        model_context_window=65_536,
        llm_output_max_tokens_ratio=0.125,
        compaction_keep_recent_turns=1,
    )
    engine = engine_factory(rc=rc)
    delegate = in_memory_runtime["llm"]
    delegate.queue_tool_call_response(
        tool_call_id="toolu_retry_boundary",
        tool_name="Read",
        tool_input={"v": "hello"},
    )
    delegate.queue_response(text="done")
    llm = _FirstCallOverflowLLM(delegate)
    engine.llm = llm  # type: ignore[assignment]
    engine.context_manager._compaction_llm = llm  # type: ignore[attr-defined]
    engine.compaction_llm = llm  # type: ignore[assignment]
    in_memory_runtime["tools"].register(MockTool(tool_name="Read"))

    async for _ in engine.run(
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])
    ):
        pass

    assert engine.state is LoopState.COMPLETED
    assert [request.max_tokens for request in llm.calls] == [8_192, 4_096, 8_192]


@pytest.mark.asyncio
async def test_context_window_retry_persists_streamed_partial_attempt(
    engine_factory, in_memory_runtime
) -> None:
    rc = LoopConstants(
        model_context_window=4096,
        compaction_keep_recent_turns=1,
        context_overflow_retry_max_attempts=1,
    )
    engine = engine_factory(rc=rc)
    llm = _PartialTextThenFailLLM(
        exception=LLMContextWindowExceeded("stream exceeded context"),
        partial_text="visible before 413",
        recovery_text="recovered after compaction",
    )
    engine.llm = llm  # type: ignore[assignment]
    engine.context_manager._compaction_llm = llm  # type: ignore[attr-defined]
    engine.compaction_llm = llm  # type: ignore[assignment]

    async for _ in engine.run(
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    ):
        pass

    attempts = [m for m in engine.history if m.role is MessageRole.assistant]
    assert len(llm.calls) == 2
    assert [m.text for m in attempts] == [
        "visible before 413",
        "recovered after compaction",
    ]
    assert attempts[0].metadata[PARTIAL_ASSISTANT_ATTEMPT_METADATA_KEY] is True
    assert PARTIAL_ASSISTANT_ATTEMPT_METADATA_KEY not in attempts[1].metadata


@pytest.mark.asyncio
async def test_unclassified_stream_failure_persists_streamed_partial_attempt(
    engine_factory, in_memory_runtime
) -> None:
    engine = engine_factory(rc=LoopConstants(model_context_window=4096))
    llm = _PartialTextThenFailLLM(
        exception=RuntimeError("parser crashed"),
        partial_text="visible before parser failure",
    )
    engine.llm = llm  # type: ignore[assignment]

    async for _ in engine.run(
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    ):
        pass

    attempts = [
        message
        for message in engine.history
        if message.metadata.get(PARTIAL_ASSISTANT_ATTEMPT_METADATA_KEY) is True
    ]
    assert engine.state is LoopState.FAILED
    assert len(attempts) == 1
    assert attempts[0].text == "visible before parser failure"


@pytest.mark.asyncio
async def test_partial_attempt_marker_survives_snapshot_resume(
    engine_factory, in_memory_runtime
) -> None:
    source = engine_factory(rc=LoopConstants(model_context_window=4096))
    source.history.append(
        Message(
            role=MessageRole.assistant,
            content_blocks=[TextBlock(text="durable incomplete prefix")],
            metadata={PARTIAL_ASSISTANT_ATTEMPT_METADATA_KEY: True},
        )
    )
    resumed = engine_factory(rc=LoopConstants(model_context_window=4096))

    await resumed.resume_from_snapshot(source.snapshot())

    restored = resumed.history[-1]
    assert restored.text == "durable incomplete prefix"
    assert restored.metadata == {PARTIAL_ASSISTANT_ATTEMPT_METADATA_KEY: True}


@pytest.mark.asyncio
async def test_context_window_exceeded_second_failure_is_terminal(
    engine_factory, in_memory_runtime
) -> None:
    """Second :class:`LLMContextWindowExceeded` in same message → FAILED.

    The recovery budget is one attempt; a second PTL in the same
    message drives terminal FAILED.
    """
    rc = LoopConstants(
        model_context_window=65_536,
        llm_output_max_tokens_ratio=0.125,
        compaction_keep_recent_turns=1,
        context_overflow_retry_max_attempts=1,
    )
    engine = engine_factory(rc=rc)
    failing_llm = _ScriptedFailureLLM(
        exceptions=[
            LLMContextWindowExceeded("PTL #1"),
            LLMContextWindowExceeded("PTL #2"),
        ],
    )
    engine.llm = failing_llm  # type: ignore[assignment]
    engine.context_manager._compaction_llm = failing_llm  # type: ignore[attr-defined]
    engine.compaction_llm = failing_llm  # type: ignore[assignment]

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    events: list[TurnEvent] = []
    async for evt in engine.run(user_msg):
        events.append(evt)

    # Terminal FAILED.
    assert engine.state is LoopState.FAILED
    # Final error kind is the PTL.
    error_evts = [e for e in events if e.type is EventType.ERROR]
    assert error_evts
    assert error_evts[-1].payload["kind"] == "llm_context_window_exceeded"
    assert [request.max_tokens for request in failing_llm.calls] == [8_192, 4_096]


@pytest.mark.asyncio
async def test_context_window_corrective_retry_is_bounded(
    engine_factory, in_memory_runtime
) -> None:
    rc = LoopConstants(
        model_context_window=65_536,
        request_context_safety_tokens=2_048,
        llm_output_max_tokens_ratio=0.125,
        compaction_keep_recent_turns=1,
        context_overflow_retry_max_attempts=2,
    )
    engine = engine_factory(rc=rc)
    failing_llm = _ScriptedFailureLLM(
        exceptions=[
            LLMContextWindowExceeded("pre-compaction overflow"),
            LLMContextWindowExceeded(
                "compacted request overflowed",
                context_window=65_536,
                input_tokens=62_000,
                requested_output_tokens=4_096,
            ),
            LLMContextWindowExceeded(
                "corrective request overflowed",
                context_window=65_536,
                input_tokens=64_100,
                requested_output_tokens=1_488,
            ),
        ],
    )
    engine.llm = failing_llm  # type: ignore[assignment]
    engine.context_manager._compaction_llm = failing_llm  # type: ignore[attr-defined]
    engine.compaction_llm = failing_llm  # type: ignore[assignment]

    async for _ in engine.run(
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    ):
        pass

    assert engine.state is LoopState.FAILED
    assert [request.max_tokens for request in failing_llm.calls] == [8_192, 4_096, 1_488]


@pytest.mark.asyncio
async def test_lower_bound_overflow_repeatedly_reduces_cap_after_one_compaction(
    engine_factory, in_memory_runtime
) -> None:
    rc = LoopConstants(
        model_context_window=65_536,
        llm_output_max_tokens_ratio=0.125,
        compaction_keep_recent_turns=1,
    )
    engine = engine_factory(rc=rc)
    failing_llm = _ScriptedFailureLLM(
        exceptions=[
            LLMContextWindowExceeded(
                "prompt contains at least a lower bound",
                context_window=65_536,
                requested_output_tokens=cap,
            )
            for cap in (8_192, 4_096, 2_048)
        ],
    )
    engine.llm = failing_llm  # type: ignore[assignment]
    engine.context_manager._compaction_llm = failing_llm  # type: ignore[attr-defined]
    engine.compaction_llm = failing_llm  # type: ignore[assignment]

    events = [
        event
        async for event in engine.run(
            Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
        )
    ]

    assert engine.state is LoopState.COMPLETED
    assert [request.max_tokens for request in failing_llm.calls] == [
        8_192,
        4_096,
        2_048,
        1_024,
    ]
    assert sum(event.type is EventType.COMPACTION_STARTED for event in events) == 1


@pytest.mark.asyncio
async def test_a_proactive_pass_is_followed_by_exactly_one_reactive_pass_after_overflow(
    engine_factory, in_memory_runtime, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A proactive gate with nothing to do does not use up the reactive attempt.

    The two passes run different profiles: the proactive one keeps the routine
    window and leaves seeded history alone, the reactive one may shrink it.
    Here the proactive profile finds nothing eligible, so its pass is never
    opened. After the provider's rejection the reactive pass still runs once;
    a second rejection is answered by the output-cap ladder alone.
    """
    rc = LoopConstants(
        model_context_window=65_536,
        llm_output_max_tokens_ratio=0.125,
        compaction_keep_recent_turns=1,
    )
    engine = engine_factory(rc=rc)
    failing_llm = _ScriptedFailureLLM(
        exceptions=[
            LLMContextWindowExceeded(
                "prompt contains at least a lower bound",
                context_window=65_536,
                requested_output_tokens=cap,
            )
            for cap in (8_192, 4_096)
        ],
    )
    engine.llm = failing_llm  # type: ignore[assignment]
    engine.context_manager._compaction_llm = failing_llm  # type: ignore[attr-defined]
    engine.compaction_llm = failing_llm  # type: ignore[assignment]
    monkeypatch.setattr(engine, "needs_emergency_compaction", lambda: True)

    events = [
        event
        async for event in engine.run(
            Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
        )
    ]

    assert engine.state is LoopState.COMPLETED
    # Neither pass found anything its profile may touch: nothing to do is not
    # a failure, so neither budget was spent.
    assert engine.compaction_state.retry_count == 0
    assert engine.compaction_state.reactive_retry_count == 0
    assert [request.max_tokens for request in failing_llm.calls] == [
        8_192,
        4_096,
        2_048,
    ]
    assert [
        event.payload.get("reason")
        for event in events
        if event.type is EventType.COMPACTION_STARTED
    ] == ["reactive_413"]


class _DeadSummariser(_ScriptedFailureLLM):
    """A summariser whose every call fails in transport."""

    def __init__(self) -> None:
        super().__init__(exceptions=[])
        self.summary_calls = 0

    async def complete_structured(self, request, schema):  # type: ignore[no-untyped-def]
        self.summary_calls += 1
        raise RuntimeError("summariser unavailable")


@pytest.mark.asyncio
async def test_an_exhausted_compaction_budget_hands_the_turn_back_to_the_cap_ladder(
    engine_factory, in_memory_runtime
) -> None:
    """A reactive pass out of budget must not end a run the cap ladder can still save.

    Reached for real: seeded history the reactive profile tries to summarise,
    a summariser that fails every call, and a reactive budget already spent by
    earlier messages' failed passes — the count a resumed snapshot carries.
    """
    rc = LoopConstants(
        model_context_window=65_536,
        llm_output_max_tokens_ratio=0.125,
        compaction_keep_recent_turns=1,
    )
    engine = engine_factory(rc=rc)
    failing_llm = _ScriptedFailureLLM(
        exceptions=[
            LLMContextWindowExceeded(
                "prompt contains at least a lower bound",
                context_window=65_536,
                requested_output_tokens=cap,
            )
            for cap in (8_192, 4_096)
        ],
    )
    summariser = _DeadSummariser()
    engine.llm = failing_llm  # type: ignore[assignment]
    engine.context_manager._compaction_llm = summariser  # type: ignore[attr-defined]
    engine.compaction_llm = summariser  # type: ignore[assignment]
    seed = {SESSION_HISTORY_SEED_METADATA_KEY: True}
    engine.history.extend(
        [
            Message(
                role=MessageRole.user,
                content_blocks=[TextBlock(text="prior task " * 1_200)],
                metadata=seed,
            ),
            Message(
                role=MessageRole.assistant,
                content_blocks=[TextBlock(text="prior answer " * 1_200)],
                metadata=seed,
            ),
        ]
    )
    engine.compaction_state.reactive_retry_count = rc.compaction_failed_max_retries

    events = [
        event
        async for event in engine.run(
            Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
        )
    ]

    assert engine.state is LoopState.COMPLETED
    assert summariser.summary_calls > 0
    assert [request.max_tokens for request in failing_llm.calls] == [8_192, 4_096, 2_048]
    reasons = [
        event.payload.get("reason")
        for event in events
        if event.type is EventType.STATE_CHANGED
    ]
    assert "reactive_413_compaction_exhausted" not in reasons
    assert "reactive_413_compaction_exhausted_retry" in reasons


@pytest.mark.asyncio
async def test_lower_bound_overflow_stops_at_configured_reduction_bound(
    engine_factory, in_memory_runtime
) -> None:
    rc = LoopConstants(
        model_context_window=65_536,
        llm_output_max_tokens_ratio=0.125,
        compaction_keep_recent_turns=1,
        context_overflow_retry_max_attempts=3,
    )
    engine = engine_factory(rc=rc)
    failing_llm = _ScriptedFailureLLM(
        exceptions=[
            LLMContextWindowExceeded(
                "prompt contains at least a lower bound",
                context_window=65_536,
                requested_output_tokens=cap,
            )
            for cap in (8_192, 4_096, 2_048, 1_024)
        ],
    )
    engine.llm = failing_llm  # type: ignore[assignment]
    engine.context_manager._compaction_llm = failing_llm  # type: ignore[attr-defined]
    engine.compaction_llm = failing_llm  # type: ignore[assignment]

    events = [
        event
        async for event in engine.run(
            Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
        )
    ]

    assert engine.state is LoopState.FAILED
    assert [request.max_tokens for request in failing_llm.calls] == [
        8_192,
        4_096,
        2_048,
        1_024,
    ]
    assert sum(event.type is EventType.COMPACTION_STARTED for event in events) == 1


@pytest.mark.asyncio
async def test_lower_bound_overflow_can_reduce_the_full_output_cap_ladder(
    engine_factory, in_memory_runtime
) -> None:
    caps = [8_192, 4_096, 2_048, 1_024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
    rc = LoopConstants(
        model_context_window=65_536,
        llm_output_max_tokens_ratio=0.125,
        compaction_keep_recent_turns=1,
    )
    engine = engine_factory(rc=rc)
    failing_llm = _ScriptedFailureLLM(
        exceptions=[
            LLMContextWindowExceeded(
                "prompt contains at least a lower bound",
                context_window=65_536,
                requested_output_tokens=cap,
            )
            for cap in caps
        ],
    )
    engine.llm = failing_llm  # type: ignore[assignment]
    engine.context_manager._compaction_llm = failing_llm  # type: ignore[attr-defined]
    engine.compaction_llm = failing_llm  # type: ignore[assignment]

    async for _ in engine.run(
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    ):
        pass

    assert engine.state is LoopState.FAILED
    assert [request.max_tokens for request in failing_llm.calls] == caps


# ----------------------------------------------------------------------
# ContextManager.force_compaction unit coverage
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_force_compaction_runs_both_tiers_unconditionally(
    in_memory_runtime,
) -> None:
    """``force_compaction`` MUST run Tier 1 + Tier 2 even when Tier 1 freed enough.

    Verifies the divergence from :meth:`run_compaction` (which gates
    Tier 2 behind a "Tier 1 didn't free enough" check) — reactive-413
    must free as much as possible.
    """
    from protocore.runtime.context.compaction import CompactionState
    from protocore.runtime.context.manager import ContextManager

    rc = LoopConstants(
        model_context_window=128,
        request_context_safety_tokens=0,
        compaction_keep_recent_turns=1,
    )
    failing_llm = _ScriptedFailureLLM(exceptions=[None])  # never raises
    ctx_mgr = ContextManager(
        rc=rc,
        blob_store=in_memory_runtime["blobs"],
        compaction_llm=failing_llm,
    )

    history = [
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="x" * 200)]),
        Message(
            role=MessageRole.assistant,
            content_blocks=[TextBlock(text="y" * 200)],
        ),
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="z" * 200)]),
        Message(
            role=MessageRole.assistant,
            content_blocks=[TextBlock(text="last" * 50)],
        ),
    ]
    state = CompactionState()

    attempt = await ctx_mgr.force_compaction(
        history=history,
        compaction_state=state,
        tenant_id="t-test",
        model_name="m-test",
    )

    # Both tiers attempted.
    assert attempt.tier1 is not None
    assert attempt.tier2 is not None


@pytest.mark.asyncio
async def test_force_compaction_shrinks_seeded_cross_run_history(
    in_memory_runtime,
) -> None:
    """Reactive compaction may shrink old seeds without making them persistable."""
    from protocore.runtime.context.compaction import CompactionState
    from protocore.runtime.context.manager import ContextManager

    rc = LoopConstants(
        model_context_window=32_768,
        request_context_safety_tokens=0,
        compaction_keep_recent_turns=4,
        compaction_fold_min_messages=2,
        compaction_fold_min_tokens=0,
        compaction_fold_keep_operator_turns=0,
    )
    seed = {SESSION_HISTORY_SEED_METADATA_KEY: True}
    history = [
        Message(
            role=MessageRole.user,
            content_blocks=[TextBlock(text="prior task " * 1_200)],
            metadata=seed,
        ),
        Message(
            role=MessageRole.assistant,
            content_blocks=[TextBlock(text="prior answer " * 1_200)],
            metadata=seed,
        ),
        Message(
            role=MessageRole.assistant,
            content_blocks=[TextBlock(text="ok")],
            metadata=seed,
        ),
        Message(
            role=MessageRole.user,
            content_blocks=[TextBlock(text="another prior task " * 1_200)],
            metadata=seed,
        ),
        Message(
            role=MessageRole.assistant,
            content_blocks=[TextBlock(text="done")],
            metadata=seed,
        ),
        Message(
            role=MessageRole.user,
            content_blocks=[TextBlock(text="CURRENT TASK VERBATIM")],
            metadata={"current": "task"},
        ),
        Message(
            role=MessageRole.assistant,
            content_blocks=[
                ToolUseBlock(
                    tool_call_id="current-call",
                    name="read",
                    arguments_json='{"path":"current"}',
                )
            ],
            metadata={"current": "assistant"},
        ),
        Message(
            role=MessageRole.tool,
            content_blocks=[
                ToolResultBlock(tool_call_id="current-call", content="current result")
            ],
            metadata={"current": "tool"},
        ),
    ]
    current_tail_before = [message.model_dump_json() for message in history[-3:]]
    llm = in_memory_runtime["llm"]
    for text in ("seed one", "seed two", "seed three", "folded seed"):
        llm.queue_response(text=text)
    manager = ContextManager(
        rc=rc,
        blob_store=in_memory_runtime["blobs"],
        compaction_llm=llm,
    )

    attempt = await manager.force_compaction(
        history=history,
        compaction_state=CompactionState(),
        tenant_id="tenant",
        model_name="model",
        reactive=True,
    )

    assert attempt.tokens_after < attempt.tokens_before
    assert [message.model_dump_json() for message in history[-3:]] == current_tail_before
    assert history[-3].text == "CURRENT TASK VERBATIM"
    assert history[-3].metadata == {"current": "task"}
    assert all(
        message.metadata.get(SESSION_HISTORY_SEED_METADATA_KEY) is True
        for message in history[:-3]
    )
    persistable = [
        message
        for message in history
        if message.metadata.get(SESSION_HISTORY_SEED_METADATA_KEY) is not True
    ]
    assert persistable == history[-3:]


def _seeded_incident_history() -> list[Message]:
    """Five prior-run seed turns and the current task, as measured on a client stand."""
    seed = {SESSION_HISTORY_SEED_METADATA_KEY: True}
    return [
        Message(
            role=MessageRole.user,
            content_blocks=[TextBlock(text="prior task " * 1_200)],
            metadata=seed,
        ),
        Message(
            role=MessageRole.assistant,
            content_blocks=[TextBlock(text="prior answer " * 1_200)],
            metadata=seed,
        ),
        Message(role=MessageRole.assistant, content_blocks=[TextBlock(text="ok")], metadata=seed),
        Message(
            role=MessageRole.user,
            content_blocks=[TextBlock(text="another prior task " * 1_200)],
            metadata=seed,
        ),
        Message(
            role=MessageRole.assistant,
            content_blocks=[TextBlock(text="prior answer two " * 1_200)],
            metadata=seed,
        ),
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="CURRENT TASK VERBATIM")]),
    ]


@pytest.mark.asyncio
async def test_reactive_compaction_shrinks_the_incident_shape_under_default_constants(
    in_memory_runtime,
) -> None:
    """Stock constants: a provider rejection must free tokens from seed-only history."""
    from protocore.runtime.context.compaction import CompactionState
    from protocore.runtime.context.manager import ContextManager

    rc = LoopConstants(model_context_window=65_536)
    history = _seeded_incident_history()
    llm = in_memory_runtime["llm"]
    for text in ("seed one", "seed two", "seed three", "seed four", "seed five"):
        llm.queue_response(text=text)
    manager = ContextManager(
        rc=rc, blob_store=in_memory_runtime["blobs"], compaction_llm=llm
    )

    attempt = await manager.force_compaction(
        history=history,
        compaction_state=CompactionState(),
        tenant_id="tenant",
        model_name="model",
        reactive=True,
    )

    assert attempt.tokens_after < attempt.tokens_before // 4
    assert history[-1].text == "CURRENT TASK VERBATIM"
    assert history[-1].metadata.get(SESSION_HISTORY_SEED_METADATA_KEY) is not True
    assert all(
        message.metadata.get(SESSION_HISTORY_SEED_METADATA_KEY) is True
        for message in history[:-1]
    )


@pytest.mark.asyncio
async def test_proactive_force_compaction_keeps_seeds_and_routine_window(
    in_memory_runtime,
) -> None:
    """Without a provider rejection the emergency profile must not apply."""
    from protocore.runtime.context.compaction import CompactionState
    from protocore.runtime.context.manager import ContextManager

    rc = LoopConstants(model_context_window=65_536)
    history = _seeded_incident_history()
    before = [message.model_dump_json() for message in history]
    llm = in_memory_runtime["llm"]
    llm.queue_response(text="must not be used")
    manager = ContextManager(
        rc=rc, blob_store=in_memory_runtime["blobs"], compaction_llm=llm
    )

    await manager.force_compaction(
        history=history,
        compaction_state=CompactionState(),
        tenant_id="tenant",
        model_name="model",
    )

    assert llm.calls == ()
    assert [message.model_dump_json() for message in history] == before


@pytest.mark.asyncio
async def test_proactive_force_compaction_keeps_the_routine_window_for_a_fresh_batch(
    in_memory_runtime,
) -> None:
    """A proactive emergency pass must not shed an unconsumed parallel tool batch."""
    from protocore.runtime.context.compaction import CompactionState
    from protocore.runtime.context.manager import ContextManager

    rc = LoopConstants(model_context_window=8_192)
    history = [
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="task")]),
        Message(
            role=MessageRole.assistant,
            content_blocks=[
                ToolUseBlock(tool_call_id=f"call-{n}", name="read", arguments_json="{}")
                for n in range(3)
            ],
        ),
        *(
            Message(
                role=MessageRole.tool,
                content_blocks=[
                    ToolResultBlock(tool_call_id=f"call-{n}", content="fresh result " * 2_000)
                ],
            )
            for n in range(3)
        ),
    ]
    manager = ContextManager(rc=rc, blob_store=in_memory_runtime["blobs"], compaction_llm=None)

    attempt = await manager.force_compaction(
        history=history,
        compaction_state=CompactionState(),
        tenant_id="tenant",
        model_name="model",
    )

    assert attempt.tier1 is not None
    assert attempt.tier1.messages_modified == 0
    assert attempt.tier1.blob_refs_created == ()


class _WindowBoundLLM:
    """Refuses any request whose estimated prompt plus output cap exceeds the window.

    That is what a server that reserves the output budget inside its context
    window does. The rejection reports only a lower bound for the prompt, as
    such servers do, so recovery cannot measure its way out of it.
    """

    def __init__(self, rc: LoopConstants, summariser: object) -> None:
        self._rc = rc
        self._summariser = summariser
        self.attempts: list[tuple[int, int, bool]] = []

    async def stream_with_tools(  # type: ignore[no-untyped-def]
        self, request: LLMRequest
    ) -> AsyncIterator[LLMStreamEvent]:
        from protocore.runtime.context.compaction import estimate_history_tokens

        prompt = estimate_history_tokens(list(request.messages), self._rc)
        cap = request.max_tokens
        window = self._rc.model_context_window
        if prompt + cap > window:
            self.attempts.append((prompt, cap, False))
            raise LLMContextWindowExceeded(
                f"prompt contains at least {window} input tokens",
                context_window=window,
                requested_output_tokens=cap,
            )
        self.attempts.append((prompt, cap, True))
        yield LLMStreamEvent(name="message_start", payload={})
        yield LLMStreamEvent(name="content_block_start", payload={"kind": "text"})
        yield LLMStreamEvent(name="content_block_delta", payload={"text": "ПРИНЯТО"})
        yield LLMStreamEvent(name="content_block_stop", payload={})
        yield LLMStreamEvent(name="message_stop", payload={"stop_reason": "end_turn"})

    async def complete_structured(self, request, schema):  # type: ignore[no-untyped-def]
        return await self._summariser.complete_structured(request, schema)  # type: ignore[attr-defined]


def _filler_of_about(tokens: int, rc: LoopConstants) -> str:
    from protocore.runtime.context.compaction import estimate_tokens

    unit = "deadbeef0007 "
    text = unit
    while estimate_tokens(text, rc) < tokens:
        text = text + unit * max(1, (tokens - estimate_tokens(text, rc)) // 4)
    return text


@pytest.mark.asyncio
async def test_a_proactive_pass_that_freed_nothing_does_not_forfeit_reactive_compaction(
    engine_factory, in_memory_runtime
) -> None:
    """The proactive window fires on seed-only history, frees nothing, and must not
    stand in for the reactive pass the provider's rejection is owed.

    Three turns of ~24k tokens each in one session: two are seeded from prior
    runs, the third is the request. The routine profile leaves seeds alone, so
    the proactive pass at turn start is a no-op; the provider then rejects the
    request, and only the reactive profile can shrink the seeds.
    """
    from protocore.tests_support.adapters import InMemoryLLMProvider

    rc = LoopConstants(
        model_context_window=65_536,
        request_context_safety_tokens=2_048,
        llm_output_max_tokens_ratio=0.25,
    )
    summariser = InMemoryLLMProvider()
    for _ in range(8):
        summariser.queue_response(
            text=json.dumps({"summary": "прежний ход: длинный заполнитель, ответ ПРИНЯТО"}, ensure_ascii=False)
        )
    llm = _WindowBoundLLM(rc, summariser)
    engine = engine_factory(rc=rc)
    engine.llm = llm  # type: ignore[assignment]
    engine.compaction_llm = summariser  # type: ignore[assignment]
    engine.context_manager._compaction_llm = summariser  # type: ignore[attr-defined]

    seed = {SESSION_HISTORY_SEED_METADATA_KEY: True}
    big = _filler_of_about(24_000, rc)
    engine.history.extend(
        [
            Message(role=MessageRole.user, content_blocks=[TextBlock(text=big + " один")], metadata=seed),
            Message(role=MessageRole.assistant, content_blocks=[TextBlock(text="ПРИНЯТО")], metadata=seed),
            Message(role=MessageRole.user, content_blocks=[TextBlock(text=big + " два")], metadata=seed),
            Message(role=MessageRole.assistant, content_blocks=[TextBlock(text="ПРИНЯТО")], metadata=seed),
        ]
    )

    events = [
        evt
        async for evt in engine.run(
            Message(role=MessageRole.user, content_blocks=[TextBlock(text=big + " три")])
        )
    ]

    reasons = [
        evt.payload.get("reason")
        for evt in events
        if evt.type is EventType.COMPACTION_STARTED
    ]
    assert "reactive_413" in reasons
    assert engine.state is LoopState.COMPLETED
    assert llm.attempts[-1][2] is True
    assert len(summariser.calls) >= 1


@pytest.mark.asyncio
async def test_force_compaction_exhaustion_raises(
    in_memory_runtime,
) -> None:
    """Repeated force_compaction failures raise :class:`CompactionExhaustedError`."""
    from protocore.runtime.context.compaction import (
        CompactionExhaustedError,
        CompactionState,
    )
    from protocore.runtime.context.manager import ContextManager

    rc = LoopConstants(
        model_context_window=128,
        request_context_safety_tokens=0,
        compaction_keep_recent_turns=1,
        compaction_failed_max_retries=1,
    )

    class _BrokenLLM:
        async def stream_with_tools(self, request):  # type: ignore[no-untyped-def]
            raise NotImplementedError
            yield

        async def complete_structured(self, request, schema):  # type: ignore[no-untyped-def]
            raise RuntimeError("summariser is dead")

        def count_tokens(self, text, model=None) -> int:  # type: ignore[no-untyped-def]
            return max(1, len(text) // 4)

    ctx_mgr = ContextManager(
        rc=rc,
        blob_store=in_memory_runtime["blobs"],
        compaction_llm=_BrokenLLM(),
    )

    history = [
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")]),
        # Large enough to be worth a summariser call: a unit below the floor
        # is never sent, and a pass that sends nothing has nothing to fail.
        Message(
            role=MessageRole.assistant,
            content_blocks=[TextBlock(text="response one " * 400)],
        ),
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="more")]),
        Message(
            role=MessageRole.assistant,
            content_blocks=[TextBlock(text="response two")],
        ),
    ]
    state = CompactionState()

    with pytest.raises(CompactionExhaustedError):
        for _ in range(5):
            await ctx_mgr.force_compaction(
                history=history,
                compaction_state=state,
                tenant_id="t-test",
                model_name="m-test",
            )


# ----------------------------------------------------------------------
# Engine recovery state
# ----------------------------------------------------------------------


def test_reset_recovery_state_resets_compaction_attempted(
    engine_factory,
) -> None:
    """``reset_recovery_state`` clears ``_compaction_attempted_for_current_turn``."""
    engine = engine_factory()
    engine._compaction_attempted_for_current_turn = True
    engine._last_fitted_request_max_tokens = 8_192
    engine._context_overflow_retry_max_tokens = 4_096
    engine._context_overflow_corrective_retry_count = 1

    engine.reset_recovery_state()

    assert engine._compaction_attempted_for_current_turn is False
    assert engine._last_fitted_request_max_tokens is None
    assert engine._context_overflow_retry_max_tokens is None
    assert engine._context_overflow_corrective_retry_count == 0


def test_context_window_retry_cap_only_applies_inside_the_recovery_boundary(
    engine_factory,
) -> None:
    from protocore.runtime.query import _apply_context_overflow_retry_output_cap

    engine = engine_factory(
        rc=LoopConstants(context_overflow_retry_output_ratio=0.5)
    )
    engine._compaction_attempted_for_current_turn = True
    engine._context_overflow_retry_max_tokens = 4_096

    assert _apply_context_overflow_retry_output_cap(engine, 8_192) == 4_096

    engine.reset_recovery_state()

    assert _apply_context_overflow_retry_output_cap(engine, 8_192) == 8_192


def test_new_engine_has_recovery_flags_reset(engine_factory) -> None:
    """Fresh :class:`QueryEngine` has recovery flags at false / zero."""
    engine = engine_factory()
    assert engine._compaction_attempted_for_current_turn is False
    assert engine._last_fitted_request_max_tokens is None
    assert engine._context_overflow_retry_max_tokens is None
    assert engine._context_overflow_corrective_retry_count == 0
    assert engine._max_output_recovery_count == 0
    assert engine._provider_chain_advances == 0


# ----------------------------------------------------------------------
# Fallback model on LLMProviderError
# ----------------------------------------------------------------------


class _LengthFinishLLM:
    """LLM mock that emits text + finish_reason='length' the first N calls.

    After ``length_rounds`` rounds the mock emits a healthy end_turn so
    the test can verify the recovery loop terminated successfully.
    """

    def __init__(
        self,
        length_rounds: int,
        partial_text: str = "[partial]",
        final_text: str = "done",
    ) -> None:
        self._length_rounds = length_rounds
        self._call_idx = 0
        self._partial_text = partial_text
        self._final_text = final_text
        self.calls: list[LLMRequest] = []

    async def stream_with_tools(  # type: ignore[no-untyped-def]
        self, request: LLMRequest
    ) -> AsyncIterator[LLMStreamEvent]:
        self.calls.append(request)
        idx = self._call_idx
        self._call_idx += 1

        if idx < self._length_rounds:
            yield LLMStreamEvent(name="message_start", payload={})
            yield LLMStreamEvent(name="content_block_start", payload={"kind": "text"})
            yield LLMStreamEvent(
                name="content_block_delta",
                payload={"text": self._partial_text},
            )
            yield LLMStreamEvent(name="content_block_stop", payload={})
            yield LLMStreamEvent(
                name="message_stop", payload={"stop_reason": "max_tokens"}
            )
            return

        # Final round — healthy end_turn.
        yield LLMStreamEvent(name="message_start", payload={})
        yield LLMStreamEvent(name="content_block_start", payload={"kind": "text"})
        yield LLMStreamEvent(
            name="content_block_delta", payload={"text": self._final_text}
        )
        yield LLMStreamEvent(name="content_block_stop", payload={})
        yield LLMStreamEvent(
            name="message_stop", payload={"stop_reason": "end_turn"}
        )

    async def complete_structured(self, request, schema):  # type: ignore[no-untyped-def]
        from protocore.contracts.llm import LLMResponse

        return LLMResponse(
            message=Message(role=MessageRole.assistant, content_blocks=[]),
            stop_reason=StopReason.end_turn,
        )

    def count_tokens(self, text, model=None) -> int:  # type: ignore[no-untyped-def]
        return max(1, len(text) // 4)


@pytest.mark.asyncio
async def test_provider_error_with_chain_swaps_provider(
    engine_factory, in_memory_runtime
) -> None:
    """A 5xx with a rung left → step down + re-stream."""
    rc = LoopConstants(model_context_window=4096)
    engine = engine_factory(rc=rc)
    failing_llm = _ScriptedFailureLLM(
        exceptions=[_classified(LLMProviderError("primary 5xx"), "server_error")],
    )
    _attach_chain(engine, failing_llm, "primary-model", "fallback-model-x")

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    events: list[TurnEvent] = []
    async for evt in engine.run(user_msg):
        events.append(evt)

    assert engine._provider_chain_advances == 1
    assert engine.config.model_name == "fallback-model-x"
    fallback_evts = [
        e
        for e in events
        if e.type is EventType.STATE_CHANGED
        and e.payload.get("reason") == "model_fallback_triggered"
    ]
    assert fallback_evts, "expected model_fallback_triggered state_changed"
    assert fallback_evts[0].payload["fallback_model_id"] == "fallback-model-x"
    assert engine.state is LoopState.COMPLETED


@pytest.mark.asyncio
async def test_provider_error_without_fallback_is_terminal(
    engine_factory, in_memory_runtime
) -> None:
    """No fallback configured → :class:`LLMProviderError` is terminal FAILED.

    Wind-down off: with it on the failure buys one narrowed turn to still
    deliver an answer, which is a different behaviour with its own coverage.
    This pins the terminal a deployment gets with the wind-down disabled. The
    in-place retries are off too, so the one call this counts is the one the
    chain had rather than the ladder that precedes the terminal.
    """
    rc = LoopConstants(
        model_context_window=4096,
        soft_stop_enabled=False,
        llm_transient_error_retry_max_attempts=0,
    )
    engine = engine_factory(rc=rc)
    failing_llm = _ScriptedFailureLLM(
        exceptions=[LLMProviderError("provider down")],
    )
    engine.llm = failing_llm  # type: ignore[assignment]

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    async for _ in engine.run(user_msg):
        pass

    assert engine.state is LoopState.FAILED
    assert len(failing_llm.calls) == 1


@pytest.mark.asyncio
async def test_last_rung_failure_is_terminal(
    engine_factory, in_memory_runtime
) -> None:
    """The final rung ALSO fails → terminal FAILED. No third attempt.

    Wind-down off, and the in-place retries too, so the call count measures
    the chain and nothing else.
    """
    rc = LoopConstants(
        model_context_window=4096,
        soft_stop_enabled=False,
        llm_transient_error_retry_max_attempts=0,
    )
    engine = engine_factory(rc=rc)
    failing_llm = _ScriptedFailureLLM(
        exceptions=[
            _classified(LLMProviderError("primary fail"), "server_error"),
            _classified(LLMProviderError("fallback fail"), "server_error"),
        ],
    )
    _attach_chain(engine, failing_llm, "primary-model", "fallback-model-x")

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    async for _ in engine.run(user_msg):
        pass

    assert engine.state is LoopState.FAILED
    assert len(failing_llm.calls) == 2


@pytest.mark.asyncio
async def test_chain_step_persists_partial_text_to_history(
    engine_factory, in_memory_runtime
) -> None:
    """A chain step must persist the failed attempt's partial text to
    history before swapping to the next provider.

    The failed attempt's text deltas are forwarded to SSE consumers
    live, but pre-fix the assistant turn is NEVER appended to
    ``engine.history`` on a fail. The user sees the failed text in
    the live stream and then the next provider's text in a new turn window,
    while a reload (history-only) shows only the second —
    divergent live-vs-snapshot, with the first attempt's text
    silently dropped from durable state.
    """
    rc = LoopConstants(model_context_window=4096)
    engine = engine_factory(rc=rc)
    llm = _PartialTextThenFailLLM(
        exception=_classified(LLMProviderError("primary 5xx"), "server_error"),
        partial_text="partial-from-primary",
        recovery_text="fallback-answer",
    )
    _attach_chain(engine, llm, "primary-model", "fallback-model-x")

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    async for _ in engine.run(user_msg):
        pass

    assert engine.state is LoopState.COMPLETED
    assert len(llm.calls) == 2
    # History must carry BOTH the failed attempt's text and the
    # fallback's text as two assistant messages (in stream order).
    assistant_msgs = [m for m in engine.history if m.role is MessageRole.assistant]
    assert len(assistant_msgs) == 2
    assert "".join(b.text for b in assistant_msgs[0].content_blocks) == (
        "partial-from-primary"
    )
    assert (
        assistant_msgs[0].metadata[PARTIAL_ASSISTANT_ATTEMPT_METADATA_KEY]
        is True
    )
    assert "".join(b.text for b in assistant_msgs[1].content_blocks) == (
        "fallback-answer"
    )
    assert PARTIAL_ASSISTANT_ATTEMPT_METADATA_KEY not in assistant_msgs[1].metadata


@pytest.mark.asyncio
async def test_provider_error_backstop_persists_partial_text_to_history(
    engine_factory, in_memory_runtime
) -> None:
    """The partial the user already saw is in history before the wind-down runs.

    ``LLMProviderError`` with no fallback starts the wind-down. Without this,
    the partial is silently dropped and the wind-down's turn streams its answer
    into a history carrying no record of what the user had already been shown —
    so the live view and a reload disagree about the same turn.
    """
    rc = LoopConstants(
        model_context_window=4096,
        terminal_tool_nudge_enabled=True,
        llm_transient_error_retry_max_attempts=0,
    )
    engine = engine_factory(rc=rc, expected_terminal_tool="answer")
    llm = _PartialTextThenFailLLM(
        exception=LLMProviderError("provider down"),
        partial_text="partial-streamed-to-user",
    )
    engine.llm = llm  # type: ignore[assignment]

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    events: list[TurnEvent] = []
    async for evt in engine.run(user_msg):
        events.append(evt)

    windup_evts = [
        e
        for e in events
        if e.type is EventType.STATE_CHANGED
        and e.payload.get("reason") == "soft_stop_notified"
        and e.payload.get("soft_stop_cause") == "provider_error"
    ]
    assert windup_evts, "expected the wind-down to start on the provider error"
    # Partial text is now in history as a prior assistant turn,
    # BEFORE the wind-down's own (second) assistant turn.
    assistant_msgs = [m for m in engine.history if m.role is MessageRole.assistant]
    assert any(
        "partial-streamed-to-user" in "".join(b.text for b in m.content_blocks)
        for m in assistant_msgs
    )
    partial = next(
        m
        for m in assistant_msgs
        if "partial-streamed-to-user" in "".join(b.text for b in m.content_blocks)
    )
    assert partial.metadata[PARTIAL_ASSISTANT_ATTEMPT_METADATA_KEY] is True


@pytest.mark.asyncio
async def test_stream_idle_backstop_persists_partial_text_to_history(
    engine_factory, in_memory_runtime
) -> None:
    """``LLMStreamIdleError`` + the forced backstop must persist the partial
    text the user already saw live.

    Mirrors the provider-error backstop test for the watchdog exit path.
    """
    rc = LoopConstants(
        model_context_window=4096,
        # See :func:`_terminal_tool_nudge_required` — the backstop rides
        # on the universal terminal-tool nudge gate. Without this the
        # test reduces to the terminal-on-fail path.
        terminal_tool_nudge_enabled=True,
    )
    engine = engine_factory(rc=rc, expected_terminal_tool="answer")
    llm = _PartialTextThenFailLLM(
        exception=LLMStreamIdleError("stream stalled"),
        partial_text="partial-then-idle",
    )
    engine.llm = llm  # type: ignore[assignment]

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    async for _ in engine.run(user_msg):
        pass

    assistant_msgs = [m for m in engine.history if m.role is MessageRole.assistant]
    assert any(
        "partial-then-idle" in "".join(b.text for b in m.content_blocks)
        for m in assistant_msgs
    )


# ----------------------------------------------------------------------
# Max-output-token recovery loop
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_max_output_recovery_succeeds_within_budget(
    engine_factory, in_memory_runtime
) -> None:
    """``finish_reason='length'`` once → recovery prompt + re-stream → success."""
    rc = LoopConstants(model_context_window=4096, max_output_recovery_rounds=3)
    engine = engine_factory(rc=rc)
    llm = _LengthFinishLLM(length_rounds=1, partial_text="abc", final_text="xyz")
    engine.llm = llm  # type: ignore[assignment]

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    events: list[TurnEvent] = []
    async for evt in engine.run(user_msg):
        events.append(evt)

    recovery_evts = [
        e
        for e in events
        if e.type is EventType.STATE_CHANGED
        and e.payload.get("reason") == "max_output_token_recovery"
    ]
    assert recovery_evts, "expected max_output_token_recovery event"
    assert recovery_evts[0].payload["round"] == 1

    assert engine.state is LoopState.COMPLETED
    assert len(llm.calls) == 2

    assistant_msgs = [m for m in engine.history if m.role is MessageRole.assistant]
    assert assistant_msgs[0].text == "abc"
    assert assistant_msgs[0].metadata[PARTIAL_ASSISTANT_ATTEMPT_METADATA_KEY] is True
    assert assistant_msgs[-1].text == "xyz"
    assert PARTIAL_ASSISTANT_ATTEMPT_METADATA_KEY not in assistant_msgs[-1].metadata

    # History must carry the synthesised "Resume directly" user message.
    user_msgs = [m for m in engine.history if m.role is MessageRole.user]
    resume_msgs = [m for m in user_msgs if "Resume directly" in m.text]
    assert resume_msgs, "expected synthesised 'Resume directly' user message"


@pytest.mark.asyncio
async def test_max_output_recovery_exhaustion_is_terminal(
    engine_factory, in_memory_runtime
) -> None:
    """Beyond ``max_output_recovery_rounds`` → terminal FAILED.

    Wind-down off so the call count measures the recovery budget alone.
    """
    rc = LoopConstants(
        model_context_window=4096,
        max_output_recovery_rounds=2,
        soft_stop_enabled=False,
    )
    engine = engine_factory(rc=rc)
    llm = _LengthFinishLLM(length_rounds=10, partial_text="!", final_text="ok")
    engine.llm = llm  # type: ignore[assignment]

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    events: list[TurnEvent] = []
    async for evt in engine.run(user_msg):
        events.append(evt)

    assert engine.state is LoopState.FAILED
    error_evts = [e for e in events if e.type is EventType.ERROR]
    assert error_evts
    assert error_evts[-1].payload["kind"] == "output_length_exhausted"
    # Original + 2 recovery rounds = 3.
    assert len(llm.calls) == 3
    attempts = [
        message
        for message in engine.history
        if message.metadata.get(PARTIAL_ASSISTANT_ATTEMPT_METADATA_KEY) is True
    ]
    assert len(attempts) == len(llm.calls)
    assert [message.text for message in attempts] == ["!", "!", "!"]


@pytest.mark.asyncio
async def test_max_output_recovery_zero_disables_recovery(
    engine_factory, in_memory_runtime
) -> None:
    """``max_output_recovery_rounds=0`` → length finish immediately terminal.

    Wind-down off — "immediately" is the property under test.
    """
    rc = LoopConstants(
        model_context_window=4096,
        max_output_recovery_rounds=0,
        soft_stop_enabled=False,
    )
    engine = engine_factory(rc=rc)
    llm = _LengthFinishLLM(length_rounds=5, partial_text="!", final_text="ok")
    engine.llm = llm  # type: ignore[assignment]

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    events: list[TurnEvent] = []
    async for evt in engine.run(user_msg):
        events.append(evt)

    assert engine.state is LoopState.FAILED
    error_evts = [e for e in events if e.type is EventType.ERROR]
    assert error_evts
    assert error_evts[-1].payload["kind"] == "output_length_exhausted"
    assert len(llm.calls) == 1
    attempts = [
        message
        for message in engine.history
        if message.metadata.get(PARTIAL_ASSISTANT_ATTEMPT_METADATA_KEY) is True
    ]
    assert len(attempts) == 1


# ----------------------------------------------------------------------
# RC validation
# ----------------------------------------------------------------------


def test_new_rc_fields_have_correct_defaults_recovery() -> None:
    """Recovery RC fields default to spec-mandated values."""
    rc = LoopConstants()
    assert rc.max_output_recovery_rounds == 3
    assert rc.llm_provider_chain_max_advances == 2


# ----------------------------------------------------------------------
# Idle / stall detection
# ----------------------------------------------------------------------


class _IdleStreamLLM:
    """LLM mock whose stream never yields after message_start.

    Sleeps 60s between deltas so the watchdog times out.
    """

    def __init__(self) -> None:
        self.calls: list[LLMRequest] = []

    async def stream_with_tools(  # type: ignore[no-untyped-def]
        self, request: LLMRequest
    ) -> AsyncIterator[LLMStreamEvent]:
        self.calls.append(request)
        yield LLMStreamEvent(name="message_start", payload={})
        await asyncio.sleep(60)
        yield LLMStreamEvent(name="message_stop", payload={"stop_reason": "stop"})

    async def complete_structured(self, request, schema):  # type: ignore[no-untyped-def]
        from protocore.contracts.llm import LLMResponse

        return LLMResponse(
            message=Message(role=MessageRole.assistant, content_blocks=[]),
            stop_reason=StopReason.end_turn,
        )

    def count_tokens(self, text, model=None) -> int:  # type: ignore[no-untyped-def]
        return max(1, len(text) // 4)


@pytest.mark.asyncio
async def test_stream_idle_timeout_terminates_iter() -> None:
    """A stream that hangs > ``idle_timeout`` raises :class:`LLMStreamIdleError`.

    Uses the watchdog helper directly (rather than going through the full
    engine) so the test is fast — the underlying logic is the same.
    """
    from protocore.contracts.llm import ProviderDelta, ProviderDeltaKind
    from protocore.runtime.query import _iter_with_idle_watchdog

    async def _hanging_stream() -> AsyncIterator:
        yield ProviderDelta(kind=ProviderDeltaKind.text, content="hi")
        await asyncio.sleep(10)  # exceeds 0.1s idle timeout

    items = []
    with pytest.raises(LLMStreamIdleError):
        async for delta in _iter_with_idle_watchdog(
            _hanging_stream(),
            idle_timeout=0.1,
            stall_threshold=0.05,
        ):
            items.append(delta)
            if len(items) >= 10:
                break
    # We received the first delta then timed out.
    assert len(items) == 1


@pytest.mark.asyncio
async def test_stream_stall_threshold_logs_warning_without_aborting(
    caplog,
) -> None:
    """Stall threshold > delta gap → WARNING log only, no abort."""
    import logging

    from protocore.contracts.llm import ProviderDelta, ProviderDeltaKind
    from protocore.runtime.query import _iter_with_idle_watchdog

    async def _slow_stream() -> AsyncIterator:
        yield ProviderDelta(kind=ProviderDeltaKind.text, content="first")
        await asyncio.sleep(0.15)
        yield ProviderDelta(kind=ProviderDeltaKind.text, content="second")

    caplog.set_level(logging.WARNING)
    items = []
    async for delta in _iter_with_idle_watchdog(
        _slow_stream(),
        idle_timeout=2.0,
        stall_threshold=0.05,
    ):
        items.append(delta)

    assert len(items) == 2
    stall_logs = [
        rec for rec in caplog.records if "stall" in rec.message.lower()
    ]
    assert stall_logs, "expected a stall WARNING log entry"


@pytest.mark.asyncio
async def test_stream_stall_threshold_does_not_label_first_delta_gap(
    caplog,
) -> None:
    """The stall metric is inter-delta, not time-to-first-delta."""
    import logging

    from protocore.contracts.llm import ProviderDelta, ProviderDeltaKind
    from protocore.runtime.query import _iter_with_idle_watchdog

    async def _slow_first_delta() -> AsyncIterator:
        await asyncio.sleep(0.08)
        yield ProviderDelta(kind=ProviderDeltaKind.text, content="first")

    caplog.set_level(logging.WARNING)
    items = []
    async for delta in _iter_with_idle_watchdog(
        _slow_first_delta(),
        idle_timeout=0.2,
        stall_threshold=0.03,
    ):
        items.append(delta)

    assert len(items) == 1
    stall_logs = [
        rec for rec in caplog.records if "llm_stream.stall" in rec.message
    ]
    assert stall_logs == []


@pytest.mark.asyncio
async def test_progress_delta_resets_idle_watchdog_without_reasoning_window() -> None:
    """Provider transport progress resets idle without opening reasoning budget."""

    from protocore.contracts.llm import ProviderDelta, ProviderDeltaKind
    from protocore.runtime.query import _iter_with_idle_watchdog

    async def _stream_with_progress() -> AsyncIterator:
        yield ProviderDelta(kind=ProviderDeltaKind.progress)
        await asyncio.sleep(0.05)
        yield ProviderDelta(kind=ProviderDeltaKind.progress)
        await asyncio.sleep(0.05)
        yield ProviderDelta(kind=ProviderDeltaKind.text, content="done")

    items = []
    async for delta in _iter_with_idle_watchdog(
        _stream_with_progress(),
        idle_timeout=0.08,
        stall_threshold=0.03,
        reasoning_idle_timeout=1.0,
    ):
        items.append(delta)

    assert [d.kind for d in items] == [
        ProviderDeltaKind.progress,
        ProviderDeltaKind.progress,
        ProviderDeltaKind.text,
    ]


@pytest.mark.asyncio
async def test_progress_delta_preserves_open_reasoning_window() -> None:
    """Transport progress must not collapse a reasoning-aware idle budget."""

    from protocore.contracts.llm import ProviderDelta, ProviderDeltaKind
    from protocore.runtime.query import _iter_with_idle_watchdog

    async def _stream_with_reasoning_progress() -> AsyncIterator:
        yield ProviderDelta(kind=ProviderDeltaKind.thinking, content="thinking")
        await asyncio.sleep(0.02)
        yield ProviderDelta(kind=ProviderDeltaKind.progress)
        await asyncio.sleep(0.06)
        yield ProviderDelta(kind=ProviderDeltaKind.text, content="done")

    items = []
    async for delta in _iter_with_idle_watchdog(
        _stream_with_reasoning_progress(),
        idle_timeout=0.03,
        stall_threshold=0.01,
        reasoning_idle_timeout=0.2,
    ):
        items.append(delta)

    assert [d.kind for d in items] == [
        ProviderDeltaKind.thinking,
        ProviderDeltaKind.progress,
        ProviderDeltaKind.text,
    ]


@pytest.mark.asyncio
async def test_stream_idle_drives_terminal_via_engine(
    engine_factory, in_memory_runtime
) -> None:
    """Engine path: hung LLM → ``LLMStreamIdleError`` → FAILED."""
    rc = LoopConstants(
        model_context_window=4096,
        llm_stream_idle_timeout_seconds=0.2,
        llm_stream_stall_threshold_seconds=0.05,
    )
    engine = engine_factory(rc=rc)
    engine.llm = _IdleStreamLLM()  # type: ignore[assignment]

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    events: list[TurnEvent] = []
    async for evt in engine.run(user_msg):
        events.append(evt)

    assert engine.state is LoopState.FAILED
    error_evts = [e for e in events if e.type is EventType.ERROR]
    assert error_evts
    assert error_evts[-1].payload["kind"] == "llm_stream_idle"


def test_stall_threshold_must_be_below_idle_timeout() -> None:
    """RC validator rejects stall_threshold >= idle_timeout."""
    with pytest.raises(ValueError):
        LoopConstants(
            llm_stream_idle_timeout_seconds=30.0,
            llm_stream_stall_threshold_seconds=30.0,
        )
    with pytest.raises(ValueError):
        LoopConstants(
            llm_stream_idle_timeout_seconds=30.0,
            llm_stream_stall_threshold_seconds=60.0,
        )


def test_new_rc_fields_have_correct_defaults() -> None:
    """The idle/stall RC fields default to spec-mandated values."""
    rc = LoopConstants()
    assert rc.llm_stream_idle_timeout_seconds == 90.0
    assert rc.llm_stream_stall_threshold_seconds == 5.0


# ----------------------------------------------------------------------
# Reasoning-aware idle watchdog
# ----------------------------------------------------------------------


def test_reasoning_idle_timeout_default_is_300s() -> None:
    """Default extended budget is 5 minutes."""
    rc = LoopConstants()
    assert rc.llm_stream_reasoning_idle_timeout_seconds == 300.0


def test_reasoning_idle_timeout_must_be_ge_idle_timeout() -> None:
    """RC validator rejects reasoning_idle_timeout < idle_timeout."""
    with pytest.raises(ValueError):
        LoopConstants(
            llm_stream_idle_timeout_seconds=60.0,
            llm_stream_stall_threshold_seconds=10.0,
            llm_stream_reasoning_idle_timeout_seconds=30.0,
        )


@pytest.mark.asyncio
async def test_reasoning_window_extends_idle_budget() -> None:
    """A thinking delta widens the per-iteration wait_for to the extended budget.

    Without the extension the test stream would time out at 0.1 s; with
    the extension it gets 1.0 s and a non-reasoning delta arrives in
    between, so the watchdog never raises.
    """

    from protocore.contracts.llm import ProviderDelta, ProviderDeltaKind
    from protocore.runtime.query import _iter_with_idle_watchdog

    async def _stream_with_reasoning() -> AsyncIterator:
        # First delta: a thinking chunk opens the reasoning window.
        yield ProviderDelta(kind=ProviderDeltaKind.thinking, content="...")
        # Reasoning-gap > baseline timeout (0.1 s) but < extended (1.0 s).
        await asyncio.sleep(0.3)
        yield ProviderDelta(kind=ProviderDeltaKind.text, content="answer")

    items = []
    async for delta in _iter_with_idle_watchdog(
        _stream_with_reasoning(),
        idle_timeout=0.1,
        stall_threshold=0.05,
        reasoning_idle_timeout=1.0,
    ):
        items.append(delta)

    assert [d.kind for d in items] == [
        ProviderDeltaKind.thinking,
        ProviderDeltaKind.text,
    ]


@pytest.mark.asyncio
async def test_non_reasoning_delta_closes_window() -> None:
    """After a non-thinking delta the watchdog reverts to the baseline budget."""

    from protocore.contracts.llm import ProviderDelta, ProviderDeltaKind
    from protocore.runtime.query import _iter_with_idle_watchdog

    async def _stream_with_close() -> AsyncIterator:
        # Open the reasoning window.
        yield ProviderDelta(kind=ProviderDeltaKind.thinking, content="...")
        # Close it with a visible-content delta.
        yield ProviderDelta(kind=ProviderDeltaKind.text, content="visible")
        # Next gap > baseline AND < extended; baseline should now apply.
        await asyncio.sleep(0.3)
        yield ProviderDelta(kind=ProviderDeltaKind.text, content="should_not_arrive")

    items = []
    with pytest.raises(LLMStreamIdleError):
        async for delta in _iter_with_idle_watchdog(
            _stream_with_close(),
            idle_timeout=0.1,
            stall_threshold=0.05,
            reasoning_idle_timeout=1.0,
        ):
            items.append(delta)

    # Two deltas arrived before the baseline-budget timeout fired on the
    # third gap.
    assert len(items) == 2
    assert items[1].kind is ProviderDeltaKind.text


@pytest.mark.asyncio
async def test_reasoning_window_timeout_message_carries_window_hint(
    caplog,
) -> None:
    """Idle timeout in reasoning window reports ``window=reasoning``."""

    from protocore.contracts.llm import ProviderDelta, ProviderDeltaKind
    from protocore.runtime.query import _iter_with_idle_watchdog

    async def _stream_hang_in_reasoning() -> AsyncIterator:
        yield ProviderDelta(kind=ProviderDeltaKind.thinking, content="...")
        await asyncio.sleep(2.0)  # exceeds extended budget

    with pytest.raises(LLMStreamIdleError) as excinfo:
        async for _ in _iter_with_idle_watchdog(
            _stream_hang_in_reasoning(),
            idle_timeout=0.1,
            stall_threshold=0.05,
            reasoning_idle_timeout=0.3,
        ):
            pass

    assert "window=reasoning" in str(excinfo.value)


# ----------------------------------------------------------------------
# Death-spiral guard
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_death_spiral_guard_set_on_provider_error(
    engine_factory, in_memory_runtime
) -> None:
    """Terminal :class:`LLMProviderError` MUST set ``engine.skip_terminal_hooks``.

    Wind-down off so the failure IS terminal: with it on the run gets a narrowed
    turn to still answer, and a run that answers never reaches the guard. The
    in-place retries are off so the first failure is also the last one.
    """
    rc = LoopConstants(
        model_context_window=4096,
        soft_stop_enabled=False,
        llm_transient_error_retry_max_attempts=0,
    )
    engine = engine_factory(rc=rc)
    engine.llm = _ScriptedFailureLLM(  # type: ignore[assignment]
        exceptions=[LLMProviderError("burst error")],
    )

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    async for _ in engine.run(user_msg):
        pass

    assert engine.skip_terminal_hooks is True
    assert engine.state is LoopState.FAILED


@pytest.mark.asyncio
async def test_death_spiral_guard_set_on_stream_idle(
    engine_factory, in_memory_runtime
) -> None:
    """Terminal :class:`LLMStreamIdleError` MUST set ``engine.skip_terminal_hooks``."""
    rc = LoopConstants(
        model_context_window=4096,
        llm_stream_idle_timeout_seconds=0.2,
        llm_stream_stall_threshold_seconds=0.05,
    )
    engine = engine_factory(rc=rc)
    engine.llm = _IdleStreamLLM()  # type: ignore[assignment]

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    async for _ in engine.run(user_msg):
        pass

    assert engine.skip_terminal_hooks is True
    assert engine.state is LoopState.FAILED


@pytest.mark.asyncio
async def test_death_spiral_guard_set_on_max_output_exhaustion(
    engine_factory, in_memory_runtime
) -> None:
    """``output_length_exhausted`` terminal path MUST set the guard."""
    rc = LoopConstants(
        model_context_window=4096,
        max_output_recovery_rounds=0,
    )
    engine = engine_factory(rc=rc)
    engine.llm = _LengthFinishLLM(  # type: ignore[assignment]
        length_rounds=5, partial_text="!", final_text="ok"
    )

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    async for _ in engine.run(user_msg):
        pass

    assert engine.skip_terminal_hooks is True
    assert engine.state is LoopState.FAILED


@pytest.mark.asyncio
async def test_death_spiral_guard_set_on_post_retry_ptl(
    engine_factory, in_memory_runtime
) -> None:
    """``LLMContextWindowExceeded`` after the recovery retry MUST set the guard."""
    rc = LoopConstants(
        model_context_window=4096,
        compaction_keep_recent_turns=1,
        context_overflow_retry_max_attempts=1,
    )
    engine = engine_factory(rc=rc)
    failing_llm = _ScriptedFailureLLM(
        exceptions=[
            LLMContextWindowExceeded("PTL #1"),
            LLMContextWindowExceeded("PTL #2"),
        ],
    )
    engine.llm = failing_llm  # type: ignore[assignment]
    engine.context_manager._compaction_llm = failing_llm  # type: ignore[attr-defined]
    engine.compaction_llm = failing_llm  # type: ignore[assignment]

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    async for _ in engine.run(user_msg):
        pass

    assert engine.skip_terminal_hooks is True
    assert engine.state is LoopState.FAILED


@pytest.mark.asyncio
async def test_death_spiral_guard_disabled_via_rc(
    engine_factory, in_memory_runtime
) -> None:
    """``rc.skip_terminal_hooks_on_llm_error=False`` keeps the flag False.

    Diagnostic mode — Stop / SessionEnd hooks SHOULD see the failure
    (e.g. error-classifier hooks).
    """
    rc = LoopConstants(
        model_context_window=4096,
        skip_terminal_hooks_on_llm_error=False,
        soft_stop_enabled=False,
        llm_transient_error_retry_max_attempts=0,
    )
    engine = engine_factory(rc=rc)
    engine.llm = _ScriptedFailureLLM(  # type: ignore[assignment]
        exceptions=[LLMProviderError("burst error")],
    )

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    async for _ in engine.run(user_msg):
        pass

    assert engine.state is LoopState.FAILED
    assert engine.skip_terminal_hooks is False


def test_fresh_engine_skip_terminal_hooks_is_false(engine_factory) -> None:
    """``QueryEngine.skip_terminal_hooks`` defaults to ``False``."""
    engine = engine_factory()
    assert engine.skip_terminal_hooks is False


def test_new_rc_field_skip_terminal_hooks_default_true() -> None:
    """``rc.skip_terminal_hooks_on_llm_error`` defaults to ``True``."""
    rc = LoopConstants()
    assert rc.skip_terminal_hooks_on_llm_error is True


# ----------------------------------------------------------------------
# Continue-prompt fallback (thinking-tokens trap)
# ----------------------------------------------------------------------


class _ThinkingOnlyLLM:
    """LLM mock that emits ONLY thinking content (no text, no tool_calls).

    Used to script the thinking-tokens trap: small reasoning models
    (Qwen / DeepSeek / Kimi) burn their output budget on chain-of-thought
    and emit ``finish_reason='stop'`` with empty content but a populated
    ``reasoning_content``.

    The mock alternates: first ``empty_rounds`` calls return thinking-only,
    then the next call returns a healthy text turn so the recovery loop
    has something to converge on.
    """

    def __init__(
        self,
        *,
        empty_rounds: int,
        recovery_text: str = "ok after recovery",
        thinking_chunks: tuple[str, ...] = ("hm, ", "let me think... "),
    ) -> None:
        self._empty_rounds = empty_rounds
        self._recovery_text = recovery_text
        self._thinking_chunks = thinking_chunks
        self._call_idx = 0
        self.calls: list[LLMRequest] = []

    async def stream_with_tools(  # type: ignore[no-untyped-def]
        self, request: LLMRequest
    ) -> AsyncIterator[LLMStreamEvent]:
        self.calls.append(request)
        idx = self._call_idx
        self._call_idx += 1
        yield LLMStreamEvent(name="message_start", payload={})
        if idx < self._empty_rounds:
            # Thinking-only stream.
            yield LLMStreamEvent(
                name="content_block_start", payload={"kind": "thinking"}
            )
            for chunk in self._thinking_chunks:
                yield LLMStreamEvent(
                    name="content_block_delta",
                    payload={"text": chunk, "kind": "thinking"},
                )
            yield LLMStreamEvent(name="content_block_stop", payload={})
            yield LLMStreamEvent(
                name="message_stop",
                payload={"stop_reason": StopReason.end_turn.value},
            )
            return
        # Healthy recovery text turn.
        yield LLMStreamEvent(name="content_block_start", payload={"kind": "text"})
        yield LLMStreamEvent(
            name="content_block_delta",
            payload={"text": self._recovery_text, "kind": "text"},
        )
        yield LLMStreamEvent(name="content_block_stop", payload={})
        yield LLMStreamEvent(
            name="message_stop",
            payload={"stop_reason": StopReason.end_turn.value},
        )

    async def complete_structured(self, request, schema):  # type: ignore[no-untyped-def]
        from protocore.contracts.llm import LLMResponse

        return LLMResponse(
            message=Message(role=MessageRole.assistant, content_blocks=[]),
            stop_reason=StopReason.end_turn,
        )

    def count_tokens(self, text, model=None) -> int:  # type: ignore[no-untyped-def]
        return max(1, len(text) // 4)


@pytest.mark.asyncio
async def test_continue_prompt_recovers_from_thinking_only_response(
    engine_factory, in_memory_runtime
) -> None:
    """2 empty + 1 healthy → recovery converges, run COMPLETED."""
    rc = LoopConstants(
        model_context_window=4_096,
        max_consecutive_empty_responses=3,
    )
    engine = engine_factory(rc=rc)
    llm = _ThinkingOnlyLLM(empty_rounds=2, recovery_text="recovered!")
    engine.llm = llm  # type: ignore[assignment]

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    events: list[TurnEvent] = []
    async for evt in engine.run(user_msg):
        events.append(evt)

    assert engine.state is LoopState.COMPLETED, (
        f"expected COMPLETED, got {engine.state}"
    )
    # 3 calls total: 2 thinking-only + 1 recovery.
    assert len(llm.calls) == 3

    # Continue-prompt state_changed events fired twice.
    continue_events = [
        e
        for e in events
        if e.type is EventType.STATE_CHANGED
        and e.payload.get("reason") == "continue_prompt_injected"
    ]
    assert len(continue_events) == 2
    assert continue_events[0].payload["round"] == 1
    assert continue_events[1].payload["round"] == 2
    # Reasoning content length surfaced for telemetry.
    assert continue_events[0].payload["reasoning_content_chars"] > 0
    thinking_attempts = [
        message
        for message in engine.history
        if message.role is MessageRole.assistant and message.reasoning_content
    ]
    assert len(thinking_attempts) == 2
    assert all(
        message.metadata[PARTIAL_ASSISTANT_ATTEMPT_METADATA_KEY] is True
        for message in thinking_attempts
    )


@pytest.mark.asyncio
async def test_continue_prompt_budget_exhaustion_terminal(
    engine_factory, in_memory_runtime
) -> None:
    """4 empty rounds with budget=3 → terminal FAILED on the 4th.

    After 3 continue-prompt injections, the 4th empty response burns
    the budget and the engine transitions FAILED with the
    ``thinking_eats_all_tokens`` kind.
    """
    # Wind-down off so the attempt count measures the empty-response budget.
    rc = LoopConstants(
        model_context_window=4_096,
        max_consecutive_empty_responses=3,
        soft_stop_enabled=False,
    )
    engine = engine_factory(rc=rc)
    # 4 empty rounds — never recovers.
    llm = _ThinkingOnlyLLM(empty_rounds=10)
    engine.llm = llm  # type: ignore[assignment]

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    events: list[TurnEvent] = []
    async for evt in engine.run(user_msg):
        events.append(evt)

    assert engine.state is LoopState.FAILED
    error_evts = [e for e in events if e.type is EventType.ERROR]
    assert error_evts
    assert error_evts[-1].payload["kind"] == "thinking_eats_all_tokens"
    attempts = [
        message
        for message in engine.history
        if message.metadata.get(PARTIAL_ASSISTANT_ATTEMPT_METADATA_KEY) is True
        and message.reasoning_content
    ]
    assert len(attempts) == len(llm.calls) == 4


@pytest.mark.asyncio
async def test_continue_prompt_disabled_when_budget_zero(
    engine_factory, in_memory_runtime
) -> None:
    """``max_consecutive_empty_responses=0`` → no recovery, immediate end_turn.

    Backwards-compatible escape hatch — a tenant that prefers the v1
    behaviour (empty response = end_turn) can flip the RC to 0.
    """
    rc = LoopConstants(
        model_context_window=4_096,
        max_consecutive_empty_responses=0,
    )
    engine = engine_factory(rc=rc)
    llm = _ThinkingOnlyLLM(empty_rounds=5)
    engine.llm = llm  # type: ignore[assignment]

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    events: list[TurnEvent] = []
    async for evt in engine.run(user_msg):
        events.append(evt)

    # Recovery disabled → single call → end_turn → COMPLETED.
    assert engine.state is LoopState.COMPLETED
    assert len(llm.calls) == 1
    continue_events = [
        e
        for e in events
        if e.type is EventType.STATE_CHANGED
        and e.payload.get("reason") == "continue_prompt_injected"
    ]
    assert continue_events == []


def test_new_rc_field_max_consecutive_empty_responses_default() -> None:
    """``rc.max_consecutive_empty_responses`` defaults to 3."""
    rc = LoopConstants()
    assert rc.max_consecutive_empty_responses == 3


def test_new_rc_field_continue_prompt_text_default() -> None:
    """``rc.continue_prompt_text`` defaults to a non-empty string."""
    rc = LoopConstants()
    assert rc.continue_prompt_text
    # Generic English nudge — multilingual deployments override.
    assert "continue" in rc.continue_prompt_text.lower()


def test_fresh_engine_consecutive_empty_responses_is_zero(engine_factory) -> None:
    """``QueryEngine._consecutive_empty_responses`` initialises to 0."""
    engine = engine_factory()
    assert engine._consecutive_empty_responses == 0


# ----------------------------------------------------------------------
# Mid-tool-call truncation recovery
#
# Covers the production bug where ``finish_reason='length'`` arrives
# while a tool_call buffer was still streaming. The vLLM SSE parser
# silently closes the open args via stage-4 brace balancing; without
# the recovery branch the loop would dispatch the partial call (e.g.
# silently truncated Write to a 1500-line file) and the model would
# never know to continue. A resume-nudge re-stream is bounded by
# ``rc.max_output_recovery_rounds``; the nudge text is configurable
# via RC and a ``tool_call_truncation_recovery`` state-change event
# is emitted for telemetry.
# ----------------------------------------------------------------------


class _MidToolCallTruncatedLLM:
    """LLM mock emitting :class:`ProviderDelta` natively for mid-tool-call truncation.

    On the first ``truncated_rounds`` calls it streams:

    1. ``tool_use_start`` (Write tool).
    2. One ``tool_use_input`` delta carrying partial args text.
    3. ``tool_use_stop`` with ``truncated_by_output_cap=True`` (the SSE
       parser's signal that stage-4 brace balancing closed an open
       object because ``finish_reason='length'`` arrived).
    4. ``finish`` with ``finish_reason='length'``.

    After ``truncated_rounds`` the mock emits a clean text turn so the
    test can verify the loop recovered. The native ``ProviderDelta``
    stream bypasses the legacy ``LLMStreamEvent`` bridge — the loop's
    :func:`_as_provider_deltas` peeker detects the first item type and
    forwards directly.
    """

    def __init__(
        self,
        truncated_rounds: int,
        tool_name: str = "Write",
        partial_args: dict[str, object] | None = None,
        final_text: str = "all done",
    ) -> None:
        self._truncated_rounds = truncated_rounds
        self._call_idx = 0
        self._tool_name = tool_name
        self._partial_args = partial_args or {
            "path": "/workspace/big.md",
            "content": "incomplete tex",
        }
        self._final_text = final_text
        self.calls: list[LLMRequest] = []

    async def stream_with_tools(  # type: ignore[no-untyped-def]
        self, request: LLMRequest
    ) -> AsyncIterator[ProviderDelta]:
        self.calls.append(request)
        idx = self._call_idx
        self._call_idx += 1

        if idx < self._truncated_rounds:
            tool_call_id = f"toolu_trunc_{idx}"
            yield ProviderDelta(
                kind=ProviderDeltaKind.tool_use_start,
                tool_call_id=tool_call_id,
                tool_name=self._tool_name,
            )
            yield ProviderDelta(
                kind=ProviderDeltaKind.tool_use_input,
                tool_call_id=tool_call_id,
                tool_input_delta='{"path": "/workspace/big.md", "content": "incomplete tex',
            )
            yield ProviderDelta(
                kind=ProviderDeltaKind.tool_use_stop,
                tool_call_id=tool_call_id,
                tool_input_final=dict(self._partial_args),
                is_block_end=True,
                truncated_by_output_cap=True,
            )
            yield ProviderDelta(
                kind=ProviderDeltaKind.finish,
                finish_reason="length",
            )
            return

        # Final round — healthy text + end_turn.
        yield ProviderDelta(
            kind=ProviderDeltaKind.text,
            content=self._final_text,
        )
        yield ProviderDelta(
            kind=ProviderDeltaKind.finish,
            finish_reason="stop",
        )

    async def complete_structured(self, request, schema):  # type: ignore[no-untyped-def]
        from protocore.contracts.llm import LLMResponse

        return LLMResponse(
            message=Message(
                role=MessageRole.assistant,
                content_blocks=[TextBlock(text=self._final_text)],
            ),
            stop_reason=StopReason.end_turn,
        )

    def count_tokens(self, text, model=None) -> int:  # type: ignore[no-untyped-def]
        return max(1, len(text) // 4)


# ProviderDelta-emitting LLM needs ProviderDelta imported in scope.
from protocore.contracts.llm import ProviderDelta, ProviderDeltaKind  # noqa: E402


@pytest.mark.asyncio
async def test_finish_reason_length_text_only_uses_existing_recovery(
    engine_factory, in_memory_runtime
) -> None:
    """REGRESSION — text-only ``finish_reason='length'`` still drives the
    max-output-token recovery path.

    The mid-tool-call recovery branch must only fire when at least one tool
    call carries ``truncated_by_output_cap=True``; pure text truncation has
    none, so the older ``max_output_token_recovery`` reason is emitted.
    """
    rc = LoopConstants(model_context_window=4096, max_output_recovery_rounds=3)
    engine = engine_factory(rc=rc)
    engine.llm = _LengthFinishLLM(  # type: ignore[assignment]
        length_rounds=1, partial_text="abc", final_text="xyz"
    )

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    events: list[TurnEvent] = []
    async for evt in engine.run(user_msg):
        events.append(evt)

    text_only_evts = [
        e
        for e in events
        if e.type is EventType.STATE_CHANGED
        and e.payload.get("reason") == "max_output_token_recovery"
    ]
    tool_truncation_evts = [
        e
        for e in events
        if e.type is EventType.STATE_CHANGED
        and e.payload.get("reason") == "tool_call_truncation_recovery"
    ]
    assert text_only_evts, "expected text-only max-output recovery to fire"
    assert not tool_truncation_evts, (
        "truncation-recovery branch must NOT fire when no tool_calls carry the flag"
    )
    assert engine.state is LoopState.COMPLETED


@pytest.mark.asyncio
async def test_finish_reason_length_mid_tool_call_sets_truncated_flag(
    engine_factory, in_memory_runtime
) -> None:
    """Slice A — ToolCall preserves ``truncated_by_output_cap=True``.

    Drives one round of mid-tool-call truncation then a clean recovery.
    The truncation-recovery branch consumes the flag to decide whether to
    re-prompt or dispatch; this test asserts the wire-level signal is
    threaded all the way from the SSE parser delta to the :class:`ToolCall`
    that the loop reasons over.
    """
    rc = LoopConstants(model_context_window=4096, max_output_recovery_rounds=3)
    engine = engine_factory(rc=rc)
    llm = _MidToolCallTruncatedLLM(truncated_rounds=1)
    engine.llm = llm  # type: ignore[assignment]

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    async for _ in engine.run(user_msg):
        pass

    # Recovery fired exactly once and the run completed cleanly.
    assert engine._max_output_recovery_count == 1
    assert engine.state is LoopState.COMPLETED
    # Two LLM calls: one truncated round + one clean recovery round.
    assert len(llm.calls) == 2


@pytest.mark.asyncio
async def test_query_recovers_truncated_tool_call_via_resume_nudge(
    engine_factory, in_memory_runtime
) -> None:
    """Slice B — recovery branch appends history + emits the expected event.

 Verifies the full Slice B contract:

 * A ``tool_call_truncation_recovery`` STATE_CHANGED event is emitted
 with the truncated tool name in the payload.
 * History gains a synthetic user-role nudge whose text was rendered
 from ``rc.tool_call_truncation_resume_prompt`` with the tool name
 substituted via ``{tool_name}``.
 * The recovery round counter advances on the engine.

 The mock's truncated Write carries a partial ``content`` body, which the
 salvage path would otherwise write to disk instead of
 re-prompting. This test pins the LEGACY generic-resume-nudge mechanism, so
 the convergence driver is DISABLED here (salvage is part of the driver); the
 NEW salvage behaviour is covered in test_longfile_convergence_loop.py.
 """
    rc = LoopConstants(
        model_context_window=4096,
        max_output_recovery_rounds=3,
        longfile_convergence_enabled=False,
    )
    engine = engine_factory(rc=rc)
    llm = _MidToolCallTruncatedLLM(truncated_rounds=1)
    engine.llm = llm  # type: ignore[assignment]

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    events: list[TurnEvent] = []
    async for evt in engine.run(user_msg):
        events.append(evt)

    recovery_evts = [
        e
        for e in events
        if e.type is EventType.STATE_CHANGED
        and e.payload.get("reason") == "tool_call_truncation_recovery"
    ]
    assert len(recovery_evts) == 1
    assert recovery_evts[0].payload["round"] == 1
    assert recovery_evts[0].payload["tools"] == ["Write"]

    # Synthesised user nudge present and naming the truncated tool.
    user_msgs = [m for m in engine.history if m.role is MessageRole.user]
    nudges = [m for m in user_msgs if "Write" in m.text and "truncated" in m.text.lower()]
    assert nudges, "expected resume nudge naming the truncated tool"
    # Default template includes the explicit COMPLETE-arguments instruction.
    assert "COMPLETE" in nudges[0].text


@pytest.mark.asyncio
async def test_resume_nudge_appends_partial_assistant_message_to_history(
    engine_factory, in_memory_runtime
) -> None:
    """Slice B detail — partial assistant turn lands in history before nudge.

    The order matters: the next LLM call must see the partial tool_use
    block in its history before the user-role resume nudge so the model
    can recall what it already started emitting. Verifies both blocks
    are present and ordered correctly.

    Pins the LEGACY generic-resume path (driver disabled), since the mock's
    truncated Write carries a partial ``content`` body that the salvage path would
    otherwise dispatch to disk instead of re-prompting.
    """
    rc = LoopConstants(
        model_context_window=4096,
        max_output_recovery_rounds=3,
        longfile_convergence_enabled=False,
    )
    engine = engine_factory(rc=rc)
    llm = _MidToolCallTruncatedLLM(truncated_rounds=1)
    engine.llm = llm  # type: ignore[assignment]

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    async for _ in engine.run(user_msg):
        pass

    # Walk history: must contain assistant turn with a ToolUseBlock for
    # the truncated Write call, followed by a user-role nudge.
    assistant_turns_with_tool_use = [
        m
        for m in engine.history
        if m.role is MessageRole.assistant
        and any(isinstance(b, ToolUseBlock) for b in m.content_blocks)
    ]
    assert assistant_turns_with_tool_use, (
        "expected partial assistant turn carrying the truncated ToolUseBlock"
    )
    partial_turn = assistant_turns_with_tool_use[0]
    assert partial_turn.metadata[PARTIAL_ASSISTANT_ATTEMPT_METADATA_KEY] is True
    tool_blocks = [b for b in partial_turn.content_blocks if isinstance(b, ToolUseBlock)]
    assert tool_blocks[0].name == "Write"
    # Ordering — the nudge user turn comes AFTER the partial assistant.
    partial_idx = engine.history.index(partial_turn)
    later = engine.history[partial_idx + 1 :]
    nudge = next(
        (
            m
            for m in later
            if m.role is MessageRole.user and "truncated" in m.text.lower()
        ),
        None,
    )
    assert nudge is not None, "resume nudge must follow the partial assistant turn"


@pytest.mark.asyncio
async def test_max_output_recovery_rounds_cap_enforced(
    engine_factory, in_memory_runtime
) -> None:
    """Slice C — beyond ``max_output_recovery_rounds`` → terminal FAILED.

    The mid-tool-call branch shares the per-message counter with the
    text-only length-truncation branch. With ``max_output_recovery_rounds=2``
    and the mock truncating 10 rounds in a row, the engine MUST go terminal
    after exactly 2 recovery rounds (3 LLM calls total).
    """
    # Wind-down off so the call count measures the recovery budget alone.
    rc = LoopConstants(
        model_context_window=4096,
        max_output_recovery_rounds=2,
        soft_stop_enabled=False,
    )
    engine = engine_factory(rc=rc)
    llm = _MidToolCallTruncatedLLM(truncated_rounds=10)
    engine.llm = llm  # type: ignore[assignment]

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    events: list[TurnEvent] = []
    async for evt in engine.run(user_msg):
        events.append(evt)

    assert engine.state is LoopState.FAILED
    error_evts = [e for e in events if e.type is EventType.ERROR]
    assert error_evts
    assert error_evts[-1].payload["kind"] == "output_length_exhausted"
    # Original truncation + 2 recovery rounds = 3 calls.
    assert len(llm.calls) == 3
    attempts = [
        message
        for message in engine.history
        if message.metadata.get(PARTIAL_ASSISTANT_ATTEMPT_METADATA_KEY) is True
    ]
    assert len(attempts) == len(llm.calls)
    original_ids = [
        block.tool_call_id
        for message in attempts
        for block in message.content_blocks
        if isinstance(block, ToolUseBlock)
    ]
    assert original_ids == ["toolu_trunc_0", "toolu_trunc_1", "toolu_trunc_2"]


@pytest.mark.asyncio
async def test_telemetry_counter_increments_on_recovery(
    engine_factory, in_memory_runtime
) -> None:
    """Slice C observability — one recovery event per successful re-prompt.

    The ``tool_call_truncation_recovery`` STATE_CHANGED event is the
    core-side counter (one event per fired branch). The
    ``round`` payload counts cumulatively per message, so two
    consecutive truncations should produce two events with rounds 1
    and 2 respectively.
    """
    rc = LoopConstants(model_context_window=4096, max_output_recovery_rounds=3)
    engine = engine_factory(rc=rc)
    llm = _MidToolCallTruncatedLLM(truncated_rounds=2)
    engine.llm = llm  # type: ignore[assignment]

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    events: list[TurnEvent] = []
    async for evt in engine.run(user_msg):
        events.append(evt)

    recovery_evts = [
        e
        for e in events
        if e.type is EventType.STATE_CHANGED
        and e.payload.get("reason") == "tool_call_truncation_recovery"
    ]
    assert len(recovery_evts) == 2
    assert [e.payload["round"] for e in recovery_evts] == [1, 2]
    assert engine.state is LoopState.COMPLETED


def test_tool_call_truncated_by_output_cap_field_default_false() -> None:
    """Slice A contract — :class:`ToolCall.truncated_by_output_cap` defaults False."""
    from protocore.contracts.types import ToolCall

    tc = ToolCall(name="Read", arguments={"path": "x"})
    assert tc.truncated_by_output_cap is False


def test_the_generic_resume_template_names_the_truncated_tool() -> None:
    """The resume text is rendered with the name of the call that was cut.

    A template that ignored ``tool_name`` would tell the model to re-issue
    something it cannot identify, so the rendered text has to carry the name.
    The wording also states the cap and asks for COMPLETE arguments, which is
    the reference an operator overriding the template writes against.
    """
    rendered = bundled_prompt_provider().render(
        "tool_call_truncation_resume", {"tool_name": "Write"}
    )
    assert "Write" in rendered
    assert "COMPLETE" in rendered
    assert "truncated" in rendered.lower()


def test_rc_max_output_recovery_rounds_doc_mentions_shared_budget() -> None:
    """RC field documentation surfaces the shared text + tool budget.

    The text-only and mid-tool-call truncation branches share one
    per-message counter (``engine._max_output_recovery_count``).
    The RC field description must document this so tenants who tune
    it understand they are budgeting BOTH recovery paths together.
    """
    field_info = LoopConstants.model_fields["max_output_recovery_rounds"]
    description = field_info.description or ""
    assert "shared" in description.lower() or "share" in description.lower()


# ----------------------------------------------------------------------
# Truncated tool call on ``finish_reason="stop"``
# ----------------------------------------------------------------------
#
# Production bug: Write tool call streamed ``{`` and then
# ``finish_reason="stop"`` arrived. The vLLM SSE parser stage-4
# brace-balanced ``{`` to ``{}`` and emitted ``tool_use_stop`` with
# ``truncated_by_output_cap=False`` (only ``length`` triggers the
# length-truncation flag). The loop dispatched ``Write({})`` which fails
# on the required-path/content validation, but the agent never received a
# meaningful error and could not retry with chunked output.
#
# Fix: the SSE parser now sets ``args_partial_truncated=True`` whenever
# stage-4 had to synthesise braces (regardless of ``finish_reason``). The
# loop's recovery branch keys off this flag in combination with
# ``finish_reason="stop"`` to emit a synthetic error tool_result naming
# the recovery strategy (chunked writes) and re-stream so the agent can
# retry.
# ----------------------------------------------------------------------


class _TruncatedToolCallStopLLM:
    """LLM mock emitting the stop-truncation wire signature on the first call.

    Round 1 emits ``tool_use_start`` (Write) + a single
    ``tool_use_input`` carrying ``{`` then ``tool_use_stop`` with
    ``args_partial_truncated=True`` AND ``finish_reason="stop"``. This
    is the exact wire signature the vLLM SSE parser produces for the
    long-en-004 / plan-ru-001 failures.

    Round 2 emits a clean text turn so the loop has something to
    converge on after the recovery branch fires.
    """

    def __init__(
        self,
        truncated_rounds: int = 1,
        tool_name: str = "Write",
        final_text: str = "ok",
    ) -> None:
        self._truncated_rounds = truncated_rounds
        self._call_idx = 0
        self._tool_name = tool_name
        self._final_text = final_text
        self.calls: list[LLMRequest] = []

    async def stream_with_tools(  # type: ignore[no-untyped-def]
        self, request: LLMRequest
    ) -> AsyncIterator[ProviderDelta]:
        self.calls.append(request)
        idx = self._call_idx
        self._call_idx += 1

        if idx < self._truncated_rounds:
            tool_call_id = f"toolu_stop_trunc_{idx}"
            yield ProviderDelta(
                kind=ProviderDeltaKind.tool_use_start,
                tool_call_id=tool_call_id,
                tool_name=self._tool_name,
            )
            yield ProviderDelta(
                kind=ProviderDeltaKind.tool_use_input,
                tool_call_id=tool_call_id,
                tool_input_delta="{",
            )
            # Brace-balanced empty dict; mirrors the parser's stage-4
            # recovery output. ``truncated_by_output_cap=False`` —
            # this is the stop-finish variant (not the length-finish one).
            yield ProviderDelta(
                kind=ProviderDeltaKind.tool_use_stop,
                tool_call_id=tool_call_id,
                tool_input_final={},
                is_block_end=True,
                truncated_by_output_cap=False,
                args_partial_truncated=True,
            )
            yield ProviderDelta(
                kind=ProviderDeltaKind.finish,
                finish_reason="stop",
            )
            return

        # Recovery round — healthy text + end_turn.
        yield ProviderDelta(
            kind=ProviderDeltaKind.text,
            content=self._final_text,
        )
        yield ProviderDelta(
            kind=ProviderDeltaKind.finish,
            finish_reason="stop",
        )

    async def complete_structured(self, request, schema):  # type: ignore[no-untyped-def]
        from protocore.contracts.llm import LLMResponse

        return LLMResponse(
            message=Message(
                role=MessageRole.assistant,
                content_blocks=[TextBlock(text=self._final_text)],
            ),
            stop_reason=StopReason.end_turn,
        )

    def count_tokens(self, text, model=None) -> int:  # type: ignore[no-untyped-def]
        return max(1, len(text) // 4)


class _CleanCompleteToolCallLLM:
    """LLM mock emitting a CLEAN tool call (no truncation flag).

    Round 1 calls Write with valid args. Round 2 emits a healthy text
    turn so the loop completes. Used to verify the stop-truncation detection
    branch does NOT misfire on legitimate tool calls.
    """

    def __init__(
        self,
        tool_call_args: dict[str, str] | None = None,
        final_text: str = "ok",
    ) -> None:
        self._tool_call_args = tool_call_args or {
            "path": "/workspace/clean.txt",
            "content": "complete",
        }
        self._final_text = final_text
        self._call_idx = 0
        self.calls: list[LLMRequest] = []

    async def stream_with_tools(  # type: ignore[no-untyped-def]
        self, request: LLMRequest
    ) -> AsyncIterator[ProviderDelta]:
        self.calls.append(request)
        idx = self._call_idx
        self._call_idx += 1

        if idx == 0:
            tool_call_id = "toolu_clean"
            yield ProviderDelta(
                kind=ProviderDeltaKind.tool_use_start,
                tool_call_id=tool_call_id,
                tool_name="Write",
            )
            yield ProviderDelta(
                kind=ProviderDeltaKind.tool_use_input,
                tool_call_id=tool_call_id,
                tool_input_delta=json.dumps(self._tool_call_args),
            )
            yield ProviderDelta(
                kind=ProviderDeltaKind.tool_use_stop,
                tool_call_id=tool_call_id,
                tool_input_final=dict(self._tool_call_args),
                is_block_end=True,
                truncated_by_output_cap=False,
                args_partial_truncated=False,
            )
            yield ProviderDelta(
                kind=ProviderDeltaKind.finish,
                finish_reason="tool_use",
            )
            return

        yield ProviderDelta(
            kind=ProviderDeltaKind.text,
            content=self._final_text,
        )
        yield ProviderDelta(
            kind=ProviderDeltaKind.finish,
            finish_reason="stop",
        )

    async def complete_structured(self, request, schema):  # type: ignore[no-untyped-def]
        from protocore.contracts.llm import LLMResponse

        return LLMResponse(
            message=Message(
                role=MessageRole.assistant,
                content_blocks=[TextBlock(text=self._final_text)],
            ),
            stop_reason=StopReason.end_turn,
        )

    def count_tokens(self, text, model=None) -> int:  # type: ignore[no-untyped-def]
        return max(1, len(text) // 4)


# ``json`` is referenced inside the mock above.


@pytest.mark.asyncio
async def test_truncated_tool_call_surfaces_error(
    engine_factory, in_memory_runtime
) -> None:
    """Tool call with ``args_partial_truncated`` + ``finish_reason="stop"``
    yields an error tool result with the recovery message.

    The wire signature: ``input_partial='{'``, ``input_complete=False``,
    ``finish_reason="stop"`` (parser stage-4 brace-balanced the open
    object). The runtime must NOT dispatch the broken call. Instead it
    emits a synthetic ``TOOL_RESULT`` envelope with ``success=False`` +
    ``error.kind="tool_call_truncated"`` and persists a matching
    ``ToolResultBlock(is_error=True)`` to history so the agent reads
    the recovery instructions on the next turn.
    """
    rc = LoopConstants(model_context_window=4096)
    engine = engine_factory(rc=rc)
    llm = _TruncatedToolCallStopLLM(truncated_rounds=1)
    engine.llm = llm  # type: ignore[assignment]

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    events: list[TurnEvent] = []
    async for evt in engine.run(user_msg):
        events.append(evt)

    # A synthetic TOOL_RESULT carrying ``is_error=true`` was emitted
    # for the truncated call, naming the recovery strategy.
    tool_result_evts = [e for e in events if e.type is EventType.TOOL_RESULT]
    assert len(tool_result_evts) == 1
    payload = tool_result_evts[0].payload
    assert payload["success"] is False
    assert payload["error"]["kind"] == "tool_call_truncated"
    body = payload["content_blocks"][0]["text"]
    assert "truncated" in body.lower()
    # Recovery message names the chunked-write strategy + Edit append.
    assert "Edit" in body
    assert "Write" in body

    # The synthetic ToolResultBlock landed in history so the next LLM
    # call sees the recovery instruction. ``is_error=True`` is the
    # canonical flag the dispatcher would set on a hard failure.
    tool_result_blocks = [
        b
        for m in engine.history
        if m.role is MessageRole.tool
        for b in m.content_blocks
        if isinstance(b, ToolResultBlock)
    ]
    assert tool_result_blocks
    assert tool_result_blocks[0].is_error is True
    assert "truncated" in tool_result_blocks[0].content.lower()

    truncated_attempt = next(
        message
        for message in engine.history
        if message.role is MessageRole.assistant
        and any(isinstance(block, ToolUseBlock) for block in message.content_blocks)
    )
    assert (
        truncated_attempt.metadata[PARTIAL_ASSISTANT_ATTEMPT_METADATA_KEY] is True
    )

    # Run completed cleanly — the recovery round consumed the agent's
    # second turn (text-only "ok") so the engine reaches COMPLETED.
    assert engine.state is LoopState.COMPLETED
    assert len(llm.calls) == 2


@pytest.mark.asyncio
async def test_complete_tool_call_does_not_trigger_truncation_error(
    engine_factory, in_memory_runtime
) -> None:
    """Sanity check: clean tool calls (no truncation flag) MUST NOT
    fire the stop-truncation recovery branch.

    A normally-completing Write call with full args should reach
    :func:`_dispatch_tool` and surface its tool_result from the
    dispatcher. The synthetic ``tool_call_truncated`` envelope must
    not appear.
    """
    rc = LoopConstants(model_context_window=4096)
    engine = engine_factory(rc=rc)
    llm = _CleanCompleteToolCallLLM()
    engine.llm = llm  # type: ignore[assignment]

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    events: list[TurnEvent] = []
    async for evt in engine.run(user_msg):
        events.append(evt)

    # No truncation envelopes — clean call.
    truncation_evts = [
        e
        for e in events
        if e.type is EventType.TOOL_RESULT
        and (e.payload.get("error") or {}).get("kind") == "tool_call_truncated"
    ]
    assert not truncation_evts, (
        "truncation-recovery branch fired on a clean tool call — false positive"
    )

    # The tool was actually attempted via the dispatcher (Write tool is
    # not wired into in_memory runtime, so it surfaces as
    # ``tool_unknown`` from the dispatcher — that is the EXPECTED clean
    # path, not the truncation branch).
    tool_result_evts = [e for e in events if e.type is EventType.TOOL_RESULT]
    assert tool_result_evts
    # The dispatcher's failure kind is NOT ``tool_call_truncated``.
    for evt in tool_result_evts:
        kind = (evt.payload.get("error") or {}).get("kind") or ""
        assert kind != "tool_call_truncated"


def test_tool_call_args_partial_truncated_field_default_false() -> None:
    """Contract: :class:`ToolCall.args_partial_truncated` defaults to ``False``.

    The runtime depends on this default so any tool call not produced
    by the vLLM SSE parser's stage-4 path (Anthropic, mocks, tests)
    does not accidentally trip the stop-truncation detection branch.
    """
    from protocore.contracts.types import ToolCall

    tc = ToolCall(name="Read", arguments={"path": "x"})
    assert tc.args_partial_truncated is False


def test_provider_delta_args_partial_truncated_field_default_false() -> None:
    """Contract: :class:`ProviderDelta.args_partial_truncated` defaults to ``False``."""
    from protocore.contracts.llm import ProviderDelta, ProviderDeltaKind

    delta = ProviderDelta(kind=ProviderDeltaKind.tool_use_stop)
    assert delta.args_partial_truncated is False


def test_rc_tool_call_max_input_chunk_bytes_default_1024() -> None:
    """Contract: the soft chunk-size hint defaults to 1024 chars.

 The recovery message embeds this value as the per-chunk target the
 agent should aim for when splitting large writes. Operators can
 raise it for tenants that use models with larger output budgets;
 1024 chars mirrors ~20 lines of typical code or markdown and almost
 always fits in one tool call without re-truncating for typical models.
 """
    rc = LoopConstants()
    assert rc.tool_call_max_input_chunk_bytes == 1024


# ---------------------------------------------------------------------------
# Stop-truncation recovery fixes
#
# - mixed-batch dispatch preserves ``TOOL_CALL_PENDING`` semantics
# - per-message recovery budget (``tool_call_max_truncation_recoveries_per_message``)
# - bilingual RC-templated recovery message
# - mixed-batch (truncated + clean tool calls in same turn)
# ---------------------------------------------------------------------------


def test_rc_tool_call_max_truncation_recoveries_per_message_default_4() -> None:
    """Contract: per-message budget defaults to 4 recovery rounds.

 The budget bounds the stop-truncation recovery branch so a model stuck
 in a ``{`` + stop loop cannot burn through ``max_turns_per_run`` slots.
 The budget of 4 gives the model headroom to split a ~3-chunk write after
 the first two re-emit attempts trigger the more-directive recovery message.
 """
    rc = LoopConstants()
    assert rc.tool_call_max_truncation_recoveries_per_message == 4


def test_both_recovery_halves_spend_every_variable_they_are_given() -> None:
    """Both halves of the bilingual recovery message use all five variables.

    The loop renders the two halves with the same context and concatenates
    them EN + RU per the multilingual rule. A half that dropped one of the
    numbers would leave the model without the measurement it needs to size
    the next chunk — and, unlike a missing variable, an unused one is silent.
    So each is given a value that cannot occur by accident and looked for in
    the output.
    """
    context = {
        "tool_name": "Writeable",
        "partial_length": 4242,
        "chunk_bytes": 5353,
        "chunk_bytes_lines": 6464,
        "chunk_count_estimate": 7575,
    }
    prompts = bundled_prompt_provider()
    for name in (
        "tool_call_truncation_recovery_en",
        "tool_call_truncation_recovery_ru",
    ):
        rendered = prompts.render(name, dict(context))
        for value in context.values():
            assert str(value) in rendered, f"{name} never spends {value}"


def test_new_engine_has_tool_call_truncated_recovery_count_at_zero(
    engine_factory,
) -> None:
    """Fresh :class:`QueryEngine` starts the stop-truncation budget counter at zero."""
    engine = engine_factory()
    assert engine._tool_call_truncated_recovery_count == 0


def test_reset_recovery_state_does_not_reset_tool_call_truncated_recovery_count(
    engine_factory,
) -> None:
    """``reset_recovery_state`` does NOT touch the stop-truncation budget counter.

    The truncation recovery branch lives in the OUTER per-message loop and
    ``continue``-s a fresh assistant message every round. If
    ``reset_recovery_state`` (which runs at the top of every outer
    iteration) reset this counter the budget guard would never fire.
    The counter is reset on ``engine.run()`` entry instead, matching
    the ``_consecutive_empty_responses`` lifecycle — consecutive truncations
    across recovery rounds exhaust the budget.
    """
    engine = engine_factory()
    engine._tool_call_truncated_recovery_count = 2

    engine.reset_recovery_state()

    # Untouched — per-run lifecycle, not per-message.
    assert engine._tool_call_truncated_recovery_count == 2


@pytest.mark.asyncio
async def test_engine_run_resets_tool_call_truncated_recovery_count(
    engine_factory, in_memory_runtime
) -> None:
    """``engine.run()`` resets the stop-truncation budget counter on entry.

    Mirrors the ``_consecutive_empty_responses`` lifecycle — per-run,
    not per-message. A fresh ``engine.run()`` call gets a fresh budget
    even if the previous turn exhausted it.
    """
    engine = engine_factory()
    engine._tool_call_truncated_recovery_count = 99

    # Engage the run via a fresh clean LLM so the truncation branch
    # never fires. The counter must reset to 0 on entry regardless.
    llm = _CleanCompleteToolCallLLM()
    engine.llm = llm  # type: ignore[assignment]

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    async for _evt in engine.run(user_msg):
        pass

    assert engine._tool_call_truncated_recovery_count == 0


@pytest.mark.asyncio
async def test_truncated_tool_call_recovery_message_is_bilingual(
    engine_factory, in_memory_runtime
) -> None:
    """Synthetic ``tool_call_truncated`` recovery message includes BOTH
    English and Russian halves.

    Production is RU+EN. The recovery message is the agent-facing nudge
    that teaches it the chunked-write pattern; emitting only one language
    would leave half the production traffic without recovery guidance.
    """
    rc = LoopConstants(model_context_window=4096)
    engine = engine_factory(rc=rc)
    llm = _TruncatedToolCallStopLLM(truncated_rounds=1)
    engine.llm = llm  # type: ignore[assignment]

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    events: list[TurnEvent] = []
    async for evt in engine.run(user_msg):
        events.append(evt)

    tool_result_evts = [e for e in events if e.type is EventType.TOOL_RESULT]
    assert len(tool_result_evts) == 1
    body = tool_result_evts[0].payload["content_blocks"][0]["text"]
    # Directive fragments MUST appear in
    # both halves so the agent gets a clear "stop retrying the same
    # call, split into smaller chunks" instruction in either language.
    # EN half fragments.
    assert "TRUNCATED after" in body
    assert "Action: emit a NEW" in body
    assert "Do NOT retry the same" in body
    # RU half fragments (symmetric). The noqa silences ruff RUF001 on
    # the lines where Cyrillic capitals visually resemble Latin
    # capitals (genuinely Russian text, per the multilingual rule).
    assert "ОБРЕЗАН после" in body
    assert "Действие: эмитируйте НОВЫЙ" in body
    assert "НЕ повторяйте тот же" in body  # noqa: RUF001
    # The RC-templated chunk-byte ceiling is interpolated (default 1024 chars).
    assert "1024" in body
    # Strategy guidance (Write + Edit) still present per the original
    # contract pinned by ``test_truncated_tool_call_surfaces_error``.
    assert "Write" in body
    assert "Edit" in body


@pytest.mark.asyncio
async def test_recovery_message_includes_chunk_lines_and_count_estimate(
    engine_factory, in_memory_runtime
) -> None:
    """Directive message includes concrete line-count proxy and chunk-count
    estimate.

    Models tend to re-emit the same oversized Write when the recovery message
    lacks concrete numeric guidance. The placeholders are:

    * ``{chunk_bytes_lines} = max(1, chunk_bytes // 50)`` — a
      lines-per-chunk proxy Qwen handles better than raw byte counts.
    * ``{chunk_count_estimate} = max(2, (10240 // chunk_bytes) + 1)``
      — concrete chunk ceiling for a 10 KB target so the agent can
      plan multiple turns.

    For the default ``chunk_bytes=1024`` these resolve to ``20`` lines
    and ``11`` chunks respectively (``1024 // 50 = 20``,
    ``10240 // 1024 + 1 = 11``). Both digits MUST appear in the
    synthetic TOOL_RESULT body — otherwise the format() call dropped
    the placeholders silently and the model sees only abstract guidance.
    """
    rc = LoopConstants(model_context_window=4096)
    engine = engine_factory(rc=rc)
    llm = _TruncatedToolCallStopLLM(truncated_rounds=1)
    engine.llm = llm  # type: ignore[assignment]

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    events: list[TurnEvent] = []
    async for evt in engine.run(user_msg):
        events.append(evt)

    tool_result_evts = [e for e in events if e.type is EventType.TOOL_RESULT]
    assert len(tool_result_evts) == 1
    body = tool_result_evts[0].payload["content_blocks"][0]["text"]

    # Compute expected values from the RC defaults — same formulas the
    # loop uses at the format() call site so this test pins the
    # contract, not the literal numbers.
    expected_lines = max(1, rc.tool_call_max_input_chunk_bytes // 50)
    expected_chunks = max(2, (10240 // rc.tool_call_max_input_chunk_bytes) + 1)

    # Default RC: 1024 chars → 20 lines / 11 chunks. Both halves of the
    # bilingual message carry both placeholders so each digit appears
    # at least twice in the concatenated body.
    assert str(expected_lines) in body, (
        f"chunk_bytes_lines={expected_lines} missing from recovery body — "
        f"placeholder dropped from format() call"
    )
    assert str(expected_chunks) in body, (
        f"chunk_count_estimate={expected_chunks} missing from recovery body — "
        f"placeholder dropped from format() call"
    )

    # No literal ``{chunk_bytes_lines}`` / ``{chunk_count_estimate}``
    # tokens leaked through — silent placeholder leakage would mean
    # the format() call used the wrong kwargs.
    assert "{chunk_bytes_lines}" not in body
    assert "{chunk_count_estimate}" not in body
    # And the directive name-of-tool placeholder also resolved.
    assert "{tool_name}" not in body
    assert "Write" in body  # default _TruncatedToolCallStopLLM uses Write


@pytest.mark.asyncio
async def test_recovery_message_chunk_lines_and_count_scale_with_rc_override(
    engine_factory, in_memory_runtime
) -> None:
    """Operators tuning ``chunk_bytes`` see
    the line-count proxy and chunk-count estimate scale accordingly.

    The derived placeholders MUST be computed at format-time from the
    live ``rc.tool_call_max_input_chunk_bytes`` value — not hard-coded.
    A tenant that raises the chunk size for a larger-output model
    should see the lines/chunks estimate scale accordingly.

    With ``chunk_bytes=2048``: lines=40 (``2048 // 50``),
    chunks=6 (``10240 // 2048 + 1``).
    """
    rc = LoopConstants(
        model_context_window=4096,
        tool_call_max_input_chunk_bytes=2048,
    )
    engine = engine_factory(rc=rc)
    llm = _TruncatedToolCallStopLLM(truncated_rounds=1)
    engine.llm = llm  # type: ignore[assignment]

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    events: list[TurnEvent] = []
    async for evt in engine.run(user_msg):
        events.append(evt)

    first_tool_result = next(
        e for e in events if e.type is EventType.TOOL_RESULT
    )
    body = first_tool_result.payload["content_blocks"][0]["text"]
    # 2048-byte override → 40 lines / 6 chunks.
    assert "40" in body
    assert "6" in body
    # New chunk_bytes also embedded (operator-visible feedback).
    assert "2048" in body


@pytest.mark.asyncio
async def test_truncated_tool_call_budget_exhaustion_terminates(
    engine_factory, in_memory_runtime
) -> None:
    """Budget exhaustion → terminal LLMProviderError.

    A model that keeps emitting ``{`` + ``finish_reason="stop"`` would
    otherwise consume one outer loop iteration per round until
    ``max_turns_per_run`` fires. The per-message budget caps this at
    ``tool_call_max_truncation_recoveries_per_message`` (default 2)
    and surfaces a terminal LLM error so the run fails fast.
    """
    rc = LoopConstants(
        model_context_window=4096,
        # Allow plenty of headroom on the outer cap so the budget guard
        # is what stops the run, not ``max_turns_per_run``.
        max_turns_per_run=50,
        tool_call_max_truncation_recoveries_per_message=1,
    )
    engine = engine_factory(rc=rc)
    # Emit truncated tool calls forever — never a clean turn.
    llm = _TruncatedToolCallStopLLM(truncated_rounds=99)
    engine.llm = llm  # type: ignore[assignment]

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    events: list[TurnEvent] = []
    async for evt in engine.run(user_msg):
        events.append(evt)

    # First round consumed the budget (1 allowed). Second round saw the
    # same truncation but hit the budget guard → terminal LLM error.
    error_evts = [e for e in events if e.type is EventType.ERROR]
    assert error_evts, "budget exhaustion should surface an ERROR event"
    assert error_evts[-1].payload["kind"] == "tool_call_truncated_exhausted"
    assert "exhausted" in error_evts[-1].payload["message"].lower()

    # Run failed terminally — not COMPLETED, not AWAITING.
    assert engine.state is LoopState.FAILED
    # Exactly 2 LLM calls — one to consume the budget, one to trip it.
    # Anything more would mean the guard did NOT stop the loop fast enough.
    assert len(llm.calls) == 2


class _MixedBatchTruncatedAndCleanLLM:
    """Mixed batch: ONE truncated + ONE clean tool call in the same turn.

    Round 1: ``[truncated_Write, clean_Read]`` — the model emits a
    Write call with stage-4 brace-balanced empty args (stop-truncation
    signature) AND a Read call with full args, both before
    ``finish_reason="stop"``. This is the mixed-batch scenario the
    fix preserves dispatch semantics for and the M-6 test
    exercises end-to-end.

    Round 2: clean text turn so the loop can converge.
    """

    def __init__(self) -> None:
        self._call_idx = 0
        self.calls: list[LLMRequest] = []

    async def stream_with_tools(  # type: ignore[no-untyped-def]
        self, request: LLMRequest
    ) -> AsyncIterator:
        from protocore.contracts.llm import ProviderDelta, ProviderDeltaKind

        self.calls.append(request)
        idx = self._call_idx
        self._call_idx += 1

        if idx == 0:
            # Truncated Write call — stop-finish with args_partial_truncated.
            yield ProviderDelta(
                kind=ProviderDeltaKind.tool_use_start,
                tool_call_id="toolu_trunc_write",
                tool_name="Write",
            )
            yield ProviderDelta(
                kind=ProviderDeltaKind.tool_use_input,
                tool_call_id="toolu_trunc_write",
                tool_input_delta="{",
            )
            yield ProviderDelta(
                kind=ProviderDeltaKind.tool_use_stop,
                tool_call_id="toolu_trunc_write",
                tool_input_final={},
                is_block_end=True,
                truncated_by_output_cap=False,
                args_partial_truncated=True,
            )
            # Clean Read call — full args, no truncation flag.
            yield ProviderDelta(
                kind=ProviderDeltaKind.tool_use_start,
                tool_call_id="toolu_clean_read",
                tool_name="Read",
            )
            yield ProviderDelta(
                kind=ProviderDeltaKind.tool_use_input,
                tool_call_id="toolu_clean_read",
                tool_input_delta='{"path": "/workspace/clean.txt"}',
            )
            yield ProviderDelta(
                kind=ProviderDeltaKind.tool_use_stop,
                tool_call_id="toolu_clean_read",
                tool_input_final={"path": "/workspace/clean.txt"},
                is_block_end=True,
                truncated_by_output_cap=False,
                args_partial_truncated=False,
            )
            yield ProviderDelta(
                kind=ProviderDeltaKind.finish,
                finish_reason="stop",
            )
            return

        # Recovery round — healthy end_turn.
        yield ProviderDelta(
            kind=ProviderDeltaKind.text,
            content="ok",
        )
        yield ProviderDelta(
            kind=ProviderDeltaKind.finish,
            finish_reason="stop",
        )

    async def complete_structured(self, request, schema):  # type: ignore[no-untyped-def]
        from protocore.contracts.llm import LLMResponse

        return LLMResponse(
            message=Message(
                role=MessageRole.assistant,
                content_blocks=[TextBlock(text="ok")],
            ),
            stop_reason=StopReason.end_turn,
        )

    def count_tokens(self, text, model=None) -> int:  # type: ignore[no-untyped-def]
        return max(1, len(text) // 4)


@pytest.mark.asyncio
async def test_truncated_tool_call_mixed_batch_dispatches_both(
    engine_factory, in_memory_runtime
) -> None:
    """Mixed batch — ONE truncated + ONE clean tool call in the same turn.

    Assertions:
    1. The truncated Write call gets a synthetic
       ``tool_call_truncated`` ``TOOL_RESULT`` (success=False).
    2. The clean Read call still flows through ``_dispatch_tool`` and
       surfaces a real ``TOOL_RESULT`` (not the synthetic one) — in
       this in-memory runtime Read is unregistered so it surfaces as
       a dispatcher error, but the kind is NOT
       ``tool_call_truncated``.
    3. Exactly ONE ``MESSAGE_STOP(tool_use)`` event covers the whole
       turn (the synthetic-result MESSAGE_STOP is placed AFTER the
       non-truncated dispatch loop so all TOOL_RESULTs stay inside
       the same assistant-message window).
    4. The run completes normally — both tool_results land in history
       and the recovery round consumes the second LLM call.
    """
    rc = LoopConstants(model_context_window=4096)
    engine = engine_factory(rc=rc)
    llm = _MixedBatchTruncatedAndCleanLLM()
    engine.llm = llm  # type: ignore[assignment]

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    events: list[TurnEvent] = []
    async for evt in engine.run(user_msg):
        events.append(evt)

    tool_result_evts = [e for e in events if e.type is EventType.TOOL_RESULT]
    # Both tool calls produced a result. The truncated path emits the
    # synthetic envelope; the clean path emits whatever the dispatcher
    # returns (in-memory runtime → tool_unknown error result).
    assert len(tool_result_evts) == 2

    truncated_evts = [
        e
        for e in tool_result_evts
        if (e.payload.get("error") or {}).get("kind") == "tool_call_truncated"
    ]
    other_evts = [
        e
        for e in tool_result_evts
        if (e.payload.get("error") or {}).get("kind") != "tool_call_truncated"
    ]
    assert len(truncated_evts) == 1, (
        "truncated Write should produce exactly one synthetic tool_result"
    )
    assert truncated_evts[0].payload["tool_call_id"] == "toolu_trunc_write"
    assert len(other_evts) == 1, (
        "clean Read should still flow through _dispatch_tool"
    )
    assert other_evts[0].payload["tool_call_id"] == "toolu_clean_read"

    # M-5 — exactly one MESSAGE_STOP(tool_use) for this turn.
    tool_use_stops = [
        e
        for e in events
        if e.type is EventType.MESSAGE_STOP
        and e.payload.get("stop_reason") == "tool_use"
    ]
    assert len(tool_use_stops) == 1, (
        "mixed-batch turn should emit exactly one MESSAGE_STOP(tool_use)"
    )

    # The synthetic ToolResultBlock landed in history.
    from protocore.contracts.types import ToolResultBlock

    tool_result_blocks = [
        b
        for m in engine.history
        if m.role is MessageRole.tool
        for b in m.content_blocks
        if isinstance(b, ToolResultBlock)
    ]
    # Two blocks — one synthetic (truncated Write) + one from
    # _dispatch_tool (clean Read).
    assert len(tool_result_blocks) == 2

    # Run completed cleanly — round 2 (text-only "ok") flushed the loop.
    assert engine.state is LoopState.COMPLETED
    assert len(llm.calls) == 2


def _seeded_prior_turns() -> list[Message]:
    seed = {SESSION_HISTORY_SEED_METADATA_KEY: True}
    return [
        Message(
            role=MessageRole.user,
            content_blocks=[TextBlock(text="prior task " * 1_200)],
            metadata=seed,
        ),
        Message(
            role=MessageRole.assistant,
            content_blocks=[TextBlock(text="prior answer " * 1_200)],
            metadata=seed,
        ),
    ]


def _compaction_traces(events: list[TurnEvent]) -> list[TurnEvent]:
    return [
        event
        for event in events
        if event.type in (EventType.COMPACTION_STARTED, EventType.COMPACTION_COMPLETED)
        or (event.type is EventType.USAGE_COMMITTED and event.payload.get("kind") == "compaction")
        or (
            event.type is EventType.STATE_CHANGED
            and event.payload.get("to") == LoopState.COMPACTING.value
        )
    ]


@pytest.mark.asyncio
async def test_a_proactive_pass_with_nothing_eligible_is_never_opened(
    engine_factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Over the cliff on seeded history alone: no transaction, no trace of one."""
    engine = engine_factory(rc=LoopConstants(model_context_window=65_536))
    summariser = _DeadSummariser()
    engine.llm = _ScriptedFailureLLM(exceptions=[])  # type: ignore[assignment]
    engine.context_manager._compaction_llm = summariser  # type: ignore[attr-defined]
    engine.compaction_llm = summariser  # type: ignore[assignment]
    engine.history.extend(_seeded_prior_turns())
    monkeypatch.setattr(engine, "needs_emergency_compaction", lambda: True)
    persisted: list[int] = []
    real_persist = engine._persist_snapshot

    async def _count_persist() -> None:
        persisted.append(len(engine.history))
        await real_persist()

    monkeypatch.setattr(engine, "_persist_snapshot", _count_persist)
    pre_compact_fired: list[object] = []
    import protocore.runtime.correctness_bind as correctness_bind

    real_fire = correctness_bind.fire_lifecycle

    async def _watch_fire(engine_: Any, event: HookEvent, payload: Any) -> Any:
        if event is HookEvent.pre_compact:
            pre_compact_fired.append(payload)
        return await real_fire(engine_, event, payload)

    monkeypatch.setattr(correctness_bind, "fire_lifecycle", _watch_fire)

    events = [
        event
        async for event in engine.run(
            Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
        )
    ]

    assert engine.state is LoopState.COMPLETED
    assert _compaction_traces(events) == []
    assert pre_compact_fired == []
    assert summariser.summary_calls == 0
    assert engine.compaction_state.retry_count == 0
    assert engine._idle_compaction_probe is not None


def test_an_idle_probe_is_not_repeated_until_the_history_changes(
    engine_factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    from protocore.runtime.query import _proactive_pass_is_idle

    engine = engine_factory(
        rc=LoopConstants(model_context_window=65_536, compaction_keep_recent_turns=1)
    )
    engine.history.extend(_seeded_prior_turns())
    asked: list[bool] = []
    real_probe = engine.context_manager.has_proactive_work

    def _count_probe(*args: Any, **kwargs: Any) -> bool:
        asked.append(kwargs["force"])
        result: bool = real_probe(*args, **kwargs)
        return result

    monkeypatch.setattr(engine.context_manager, "has_proactive_work", _count_probe)

    for _ in range(3):
        assert _proactive_pass_is_idle(engine, force=True, protect_tail_from_index=None, llm_tiers=True)
    assert asked == [True]

    # Another profile is another question.
    assert _proactive_pass_is_idle(engine, force=False, protect_tail_from_index=None, llm_tiers=True)
    assert asked == [True, False]

    # New constants are another question too: a recalibrated estimate can move
    # a unit across the summariser floor without the history changing.
    engine.set_token_estimate_calibration(2.0, model_name="m")
    assert _proactive_pass_is_idle(engine, force=False, protect_tail_from_index=None, llm_tiers=True)
    assert asked == [True, False, False]

    # A changed history is asked about again — here it now holds a large
    # current-run turn the proactive profile may summarise.
    engine.history.append(
        Message(role=MessageRole.assistant, content_blocks=[TextBlock(text="current " * 2_000)])
    )
    engine.history.append(Message(role=MessageRole.user, content_blocks=[TextBlock(text="go on")]))
    assert not _proactive_pass_is_idle(engine, force=True, protect_tail_from_index=None, llm_tiers=True)
    assert asked == [True, False, False, True]
    assert engine._idle_compaction_probe is None


@pytest.mark.asyncio
async def test_proactive_exhaustion_suspends_proactive_compaction_instead_of_failing(
    engine_factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing was rejected, so the request goes out and the run completes."""
    rc = LoopConstants(model_context_window=65_536, compaction_keep_recent_turns=1)
    engine = engine_factory(rc=rc)
    summariser = _DeadSummariser()
    llm = _ScriptedFailureLLM(exceptions=[])
    engine.llm = llm  # type: ignore[assignment]
    engine.context_manager._compaction_llm = summariser  # type: ignore[attr-defined]
    engine.compaction_llm = summariser  # type: ignore[assignment]
    # Current-run turns: the proactive profile may summarise them, and every
    # summariser call fails.
    engine.history.extend(
        [message.model_copy(update={"metadata": {}}) for message in _seeded_prior_turns()]
    )
    engine.compaction_state.retry_count = rc.compaction_failed_max_retries
    monkeypatch.setattr(engine, "needs_emergency_compaction", lambda: True)

    events = [
        event
        async for event in engine.run(
            Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
        )
    ]

    assert engine.state is LoopState.COMPLETED
    assert summariser.summary_calls > 0
    assert len(llm.calls) == 1
    assert not [event for event in events if event.type is EventType.ERROR]
    reasons = [
        event.payload.get("reason")
        for event in events
        if event.type is EventType.STATE_CHANGED
    ]
    assert reasons.count("compaction_exhausted_proactive_suspended") == 1
    assert engine.proactive_compaction_suspended is True

    # Suspended: the next gate is not opened, and makes no summariser call.
    calls_before = summariser.summary_calls
    from protocore.runtime.query import _run_compaction

    engine.state = LoopState.RUNNING
    assert [event async for event in _run_compaction(engine, force=True)] == []
    assert summariser.summary_calls == calls_before


@pytest.mark.asyncio
async def test_a_provider_rejection_lifts_the_proactive_suspension(
    engine_factory,
) -> None:
    rc = LoopConstants(
        model_context_window=65_536,
        llm_output_max_tokens_ratio=0.125,
        compaction_keep_recent_turns=1,
    )
    engine = engine_factory(rc=rc)
    engine.llm = _ScriptedFailureLLM(  # type: ignore[assignment]
        exceptions=[
            LLMContextWindowExceeded(
                "prompt contains at least a lower bound",
                context_window=65_536,
                requested_output_tokens=8_192,
            )
        ],
    )
    engine._proactive_suspension_gates_left = 3
    engine._proactive_suspension_prompt_tokens = 10**9

    engine.compaction_backoff_left = 3

    events = [
        event
        async for event in engine.run(
            Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
        )
    ]

    assert engine.state is LoopState.COMPLETED
    assert "reactive_413" in [
        event.payload.get("reason")
        for event in events
        if event.type is EventType.COMPACTION_STARTED
    ]
    assert engine.compaction_backoff_left == 0
    assert engine.proactive_compaction_suspended is False


@pytest.mark.asyncio
async def test_rearm_starts_both_retry_budgets_over(engine_factory) -> None:
    engine = engine_factory(rc=LoopConstants(model_context_window=65_536))
    engine.llm = _ScriptedFailureLLM(exceptions=[])  # type: ignore[assignment]
    [
        event
        async for event in engine.run(
            Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
        )
    ]
    engine.compaction_state.retry_count = 2
    engine.compaction_state.reactive_retry_count = 2
    engine.compaction_state.summarised_turn_ids.add("kept")
    engine._proactive_suspension_gates_left = 3
    engine._proactive_suspension_prompt_tokens = 10**9

    engine.rearm()

    assert engine.compaction_state.retry_count == 0
    assert engine.compaction_state.reactive_retry_count == 0
    assert engine.compaction_state.summarised_turn_ids == {"kept"}
    assert engine.proactive_compaction_suspended is False


def test_a_suspension_expires_after_the_configured_gate_visits(engine_factory) -> None:
    from protocore.runtime.query import (
        _proactive_llm_tiers_allowed,
        _suspend_proactive_compaction,
    )

    engine = engine_factory(
        rc=LoopConstants(
            model_context_window=65_536, compaction_proactive_suspension_iterations=2
        )
    )
    engine.history.extend(_seeded_prior_turns())
    _suspend_proactive_compaction(engine)

    assert [_proactive_llm_tiers_allowed(engine) for _ in range(3)] == [False, False, True]
    assert engine.proactive_compaction_suspended is False


def test_a_suspension_expires_once_the_prompt_has_grown(engine_factory) -> None:
    from protocore.runtime.query import (
        _proactive_llm_tiers_allowed,
        _suspend_proactive_compaction,
    )

    engine = engine_factory(
        rc=LoopConstants(
            model_context_window=65_536,
            compaction_proactive_suspension_iterations=100,
            compaction_proactive_suspension_growth_ratio=0.25,
        )
    )
    engine.history.extend(_seeded_prior_turns())
    _suspend_proactive_compaction(engine)
    assert _proactive_llm_tiers_allowed(engine) is False

    engine.history.append(
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="more " * 2_000)])
    )

    assert _proactive_llm_tiers_allowed(engine) is True
    assert engine.proactive_compaction_suspended is False


@pytest.mark.asyncio
async def test_tier1_keeps_running_while_the_summariser_tiers_are_suspended(
    engine_factory,
) -> None:
    from protocore.runtime.query import _run_compaction, _suspend_proactive_compaction

    engine = engine_factory(
        rc=LoopConstants(model_context_window=65_536, compaction_keep_recent_turns=1)
    )
    summariser = _DeadSummariser()
    engine.context_manager._compaction_llm = summariser  # type: ignore[attr-defined]
    engine.compaction_llm = summariser  # type: ignore[assignment]
    engine.history.extend(
        [
            Message(role=MessageRole.user, content_blocks=[TextBlock(text="task")]),
            Message(
                role=MessageRole.assistant,
                content_blocks=[
                    ToolUseBlock(tool_call_id="c1", name="Read", arguments_json="{}")
                ],
            ),
            Message(
                role=MessageRole.tool,
                content_blocks=[ToolResultBlock(tool_call_id="c1", content="r" * 60_000)],
            ),
            Message(role=MessageRole.assistant, content_blocks=[TextBlock(text="prior " * 800)]),
            Message(role=MessageRole.user, content_blocks=[TextBlock(text="go on")]),
        ]
    )
    _suspend_proactive_compaction(engine)
    engine.transition_to(LoopState.RUNNING)

    events = [event async for event in _run_compaction(engine, force=True)]

    assert [e.type for e in events].count(EventType.COMPACTION_COMPLETED) == 1
    completed = next(e for e in events if e.type is EventType.COMPACTION_COMPLETED)
    assert completed.payload["tier1_freed"] > 0
    assert summariser.summary_calls == 0
    assert engine.proactive_compaction_suspended is True


@pytest.mark.asyncio
async def test_the_no_gain_backoff_skips_iterations_in_a_real_run(
    engine_factory, in_memory_runtime, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The backoff survives the per-message reset, so the gate really stands down."""
    from protocore.runtime.context.compaction import CompactionAttempt

    rc = LoopConstants(
        model_context_window=65_536,
        compaction_no_gain_backoff_iterations=2,
        compaction_min_gain_ratio=0.03,
    )
    engine = engine_factory(rc=rc)
    llm = in_memory_runtime["llm"]
    for i in range(5):
        llm.queue_tool_call_response(
            tool_call_id=f"call-{i}", tool_name="Read", tool_input={"v": str(i)}
        )
    llm.queue_response(text="done")
    in_memory_runtime["tools"].register(MockTool(tool_name="Read"))
    monkeypatch.setattr(engine, "needs_compaction", lambda: True)
    monkeypatch.setattr(engine, "needs_emergency_compaction", lambda: False)
    engine.context_manager.has_proactive_work = (  # type: ignore[method-assign]
        lambda *_args, **_kwargs: True
    )

    async def _no_gain(**_kwargs: Any) -> CompactionAttempt:
        return CompactionAttempt(tokens_before=1_000, tokens_after=1_000)

    engine.context_manager.run_compaction = _no_gain  # type: ignore[method-assign]

    events = [
        event
        async for event in engine.run(
            Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])
        )
    ]

    assert engine.state is LoopState.COMPLETED
    per_iteration = [
        event
        for event in events
        if event.type is EventType.COMPACTION_STARTED
        and event.payload.get("reason") == "proactive_per_iteration"
    ]
    # Five tool iterations: a pass, two skipped, a pass, one skipped.
    assert len(per_iteration) == 2
