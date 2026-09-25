"""The terminal call is forced once the run's answer is delivered.

A run whose contract ends in a terminal tool, and whose model wrote the answer
as prose and stopped, owes only the call. These tests hold the loop to the
property that every request after that point names the terminal tool in
``extra['forced_tool_choice']``, appends nothing to the transcript, never asks
the model to continue, and — when the forcing is spent — completes on the
answer already delivered rather than letting the model write another one.
"""
from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import replace
from typing import Any

import pytest

from protocore.contracts.llm import LLMRequest, LLMStreamEvent
from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.tools import ToolContext
from protocore.contracts.turn_policy import TurnCoordinate, TurnDirective, TurnFlags
from protocore.contracts.types import (
    SYNTHETIC_RECOVERY_METADATA_KEY,
    SYNTHETIC_RECOVERY_PROSE_GATE_REPAIR,
    TERMINAL_REFUSAL_NEEDS_WORK_METADATA_KEY,
    TERMINAL_TOOL_METADATA_KEY,
    TERMINAL_TOOL_STATUS_COMPLETED,
    TERMINAL_TOOL_STATUS_METADATA_KEY,
    Message,
    MessageRole,
    StopReason,
    TextBlock,
    ToolResult,
    ToolResultBlock,
    ToolUseBlock,
)
from protocore.runtime import forced_terminal
from protocore.runtime.events import EventType, TurnEvent
from protocore.runtime.loop_state import LoopState
from protocore.runtime.query import _CORE_TURN_POLICIES, _turn_at, resume
from protocore.runtime.query_engine import QueryEngine
from protocore.runtime.turn_policies.terminal_nudge import (
    ForcedTerminalCall,
    TerminalNudgePolicy,
)
from protocore.tests_support.adapters import (
    InMemoryBlobStore,
    InMemoryEventStream,
    InMemoryHookManager,
    InMemoryLLMProvider,
    InMemorySkillStore,
    InMemoryToolRegistry,
)

from ._tool_fixtures import MockTool
from .test_pointer_answer_floor import (
    ARTICLE,
    ARTICLE_PATH,
    FILING_NOTICE,
    SUBSTANTIVE_ANSWER,
    _prose_stream,
)

ANSWER = "PostgreSQL keeps row versions in the table; InnoDB keeps them in undo."


class _FinalizeTool(MockTool):
    """A background terminal tool: it records the call and ends the run."""

    async def invoke(
        self,
        context: ToolContext,
        arguments: dict[str, Any],
    ) -> ToolResult:
        self.calls.append(dict(arguments))
        return ToolResult(
            tool_call_id="",
            content="finalized",
            is_error=False,
            metadata={
                TERMINAL_TOOL_METADATA_KEY: True,
                TERMINAL_TOOL_STATUS_METADATA_KEY: TERMINAL_TOOL_STATUS_COMPLETED,
            },
        )


def _text_stream(text: str) -> list[LLMStreamEvent]:
    return [
        LLMStreamEvent(name="message_start", payload={}),
        LLMStreamEvent(name="content_block_start", payload={"kind": "text"}),
        LLMStreamEvent(name="content_block_delta", payload={"text": text, "kind": "text"}),
        LLMStreamEvent(name="content_block_stop", payload={}),
        LLMStreamEvent(
            name="message_stop", payload={"stop_reason": StopReason.end_turn.value}
        ),
    ]


def _reasoning_only_stream(stop_reason: StopReason) -> list[LLMStreamEvent]:
    return [
        LLMStreamEvent(name="message_start", payload={}),
        LLMStreamEvent(name="content_block_start", payload={"kind": "thinking"}),
        LLMStreamEvent(
            name="content_block_delta",
            payload={"text": "The user asked earlier about installing it...", "kind": "thinking"},
        ),
        LLMStreamEvent(name="content_block_stop", payload={}),
        LLMStreamEvent(name="message_stop", payload={"stop_reason": stop_reason.value}),
    ]


def _rc(**overrides: Any) -> LoopConstants:
    values: dict[str, Any] = {
        "model_context_window": 8_192,
        "terminal_tool_nudge_enabled": True,
    }
    values.update(overrides)
    return LoopConstants(**values)


def _engine(
    engine_factory: Any,
    in_memory_runtime: dict[str, Any],
    *,
    rc: LoopConstants,
    register_finalize: bool = True,
) -> tuple[QueryEngine, _FinalizeTool]:
    engine = engine_factory(rc=rc, expected_terminal_tool="Finalize")
    finalize = _FinalizeTool(tool_name="Finalize", description="End the run")
    if register_finalize:
        in_memory_runtime["tools"].register(finalize)
    in_memory_runtime["tools"].register(MockTool(tool_name="WebSearch", description="Search"))
    # The run thinks; the forced call must not.
    engine._live_thinking_enabled = True
    return engine, finalize


async def _run(engine: QueryEngine, text: str = "Compare them.") -> list[TurnEvent]:
    user = Message(role=MessageRole.user, content_blocks=[TextBlock(text=text)])
    return [event async for event in engine.run(user)]


def _reasons(events: Sequence[TurnEvent]) -> list[Any]:
    return [
        event.payload.get("reason")
        for event in events
        if event.type is EventType.STATE_CHANGED
    ]


def _request_texts(request: LLMRequest) -> list[str]:
    return [
        block.text
        for message in request.messages
        for block in message.content_blocks
        if isinstance(block, TextBlock)
    ]


def _answers(engine: QueryEngine) -> list[str]:
    return [
        block.text
        for message in engine.history
        if message.role is MessageRole.assistant
        for block in message.content_blocks
        if isinstance(block, TextBlock) and block.text.strip()
    ]


def _assert_forced_finalize(request: LLMRequest, rc: LoopConstants) -> None:
    assert request.extra.get("forced_tool_choice") == "Finalize"
    # Thinking is off on the forced call: the only output it can have is the
    # call's arguments.
    assert request.extra.get("enable_thinking") is False
    texts = _request_texts(request)
    assert rc.continue_prompt_text not in texts
    assert rc.reasoning_length_cut_nudge_text not in texts
    assert not any("has not been called yet" in text for text in texts)
    # Nothing was appended after the delivered answer.
    assert request.messages[-1].role is MessageRole.assistant
    assert _request_texts(request)[-1] == ANSWER


@pytest.mark.asyncio
async def test_a_text_answer_is_followed_by_a_forced_terminal_call(
    engine_factory, in_memory_runtime
) -> None:
    rc = _rc()
    engine, finalize = _engine(engine_factory, in_memory_runtime, rc=rc)
    llm: InMemoryLLMProvider = in_memory_runtime["llm"]
    llm._scripted_streams.append(_text_stream(ANSWER))
    llm.queue_tool_call_response(
        tool_call_id="fin-1",
        tool_name="Finalize",
        tool_input={"declared_deliverables": []},
    )

    events = await _run(engine)

    assert len(llm.calls) == 2
    first, second = llm.calls
    assert "forced_tool_choice" not in first.extra
    assert first.extra.get("enable_thinking") is True
    _assert_forced_finalize(second, rc)
    reasons = _reasons(events)
    assert forced_terminal.REASON_FORCED in reasons
    assert "terminal_tool_nudge" not in reasons
    assert finalize.calls == [{"declared_deliverables": []}]
    assert _answers(engine) == [ANSWER]
    assert engine.state is LoopState.COMPLETED


@pytest.mark.asyncio
@pytest.mark.parametrize("stop_reason", [StopReason.end_turn, StopReason.max_tokens])
async def test_reasoning_only_after_the_answer_retries_the_forced_call(
    engine_factory, in_memory_runtime, stop_reason: StopReason
) -> None:
    """The shape that produced two answers: the call was asked for, the model
    only reasoned, and the runtime said "continue". Now the round is simply
    forced again."""
    rc = _rc()
    engine, finalize = _engine(engine_factory, in_memory_runtime, rc=rc)
    llm: InMemoryLLMProvider = in_memory_runtime["llm"]
    llm._scripted_streams.append(_text_stream(ANSWER))
    llm._scripted_streams.append(_reasoning_only_stream(stop_reason))
    llm.queue_tool_call_response(
        tool_call_id="fin-1",
        tool_name="Finalize",
        tool_input={"declared_deliverables": []},
    )

    events = await _run(engine)

    assert len(llm.calls) == 3
    for request in llm.calls[1:]:
        _assert_forced_finalize(request, rc)
    reasons = _reasons(events)
    assert forced_terminal.REASON_RETRY in reasons
    assert "continue_prompt_injected" not in reasons
    assert "reasoning_length_cut_retry" not in reasons
    assert "max_output_token_recovery" not in reasons
    assert len(finalize.calls) == 1
    assert _answers(engine) == [ANSWER]
    assert engine.state is LoopState.COMPLETED


@pytest.mark.asyncio
async def test_spent_forcing_completes_on_the_delivered_answer(
    engine_factory, in_memory_runtime
) -> None:
    rc = _rc(terminal_tool_forced_max_attempts=2)
    engine, finalize = _engine(engine_factory, in_memory_runtime, rc=rc)
    llm: InMemoryLLMProvider = in_memory_runtime["llm"]
    llm._scripted_streams.append(_text_stream(ANSWER))
    llm._scripted_streams.append(_reasoning_only_stream(StopReason.end_turn))
    llm._scripted_streams.append(_reasoning_only_stream(StopReason.max_tokens))
    # A provider that would answer again if asked once more.
    llm._scripted_streams.append(_text_stream("A second answer to an earlier question."))

    events = await _run(engine)

    assert len(llm.calls) == 3
    for request in llm.calls[1:]:
        _assert_forced_finalize(request, rc)
    assert forced_terminal.REASON_EXHAUSTED in _reasons(events)
    assert finalize.calls == []
    assert _answers(engine) == [ANSWER]
    assert engine.state is LoopState.COMPLETED
    stops = [e for e in events if e.type is EventType.MESSAGE_STOP]
    assert stops[-1].payload["stop_reason"] == "end_turn"


@pytest.mark.asyncio
async def test_text_on_a_forced_turn_is_never_a_second_answer(
    engine_factory, in_memory_runtime
) -> None:
    """A provider that ignores the forced choice and writes prose instead: the
    prose reaches neither the reader nor the transcript, and the forced call
    is retried."""
    rc = _rc(terminal_tool_forced_max_attempts=1)
    engine, finalize = _engine(engine_factory, in_memory_runtime, rc=rc)
    llm: InMemoryLLMProvider = in_memory_runtime["llm"]
    second = "Answering the earlier question again: run apt install."
    llm._scripted_streams.append(_text_stream(ANSWER))
    llm._scripted_streams.append(_text_stream(second))

    events = await _run(engine)

    assert len(llm.calls) == 2
    streamed = "".join(
        str(e.payload["delta"].get("text", ""))
        for e in events
        if e.type is EventType.CONTENT_BLOCK_DELTA
        and isinstance(e.payload.get("delta"), dict)
        and e.payload["delta"].get("type") == "text_delta"
    )
    assert second not in streamed
    assert ANSWER in streamed
    assert _answers(engine) == [ANSWER]
    assert finalize.calls == []
    assert engine.state is LoopState.COMPLETED


@pytest.mark.asyncio
async def test_an_argument_error_is_forced_again_by_name(
    engine_factory, in_memory_runtime
) -> None:
    rc = _rc(terminal_tool_forced_max_attempts=2)
    engine = engine_factory(rc=rc, expected_terminal_tool="Finalize")
    failing = MockTool(
        tool_name="Finalize",
        description="End the run",
        response_content="declared_deliverables is required",
        response_is_error=True,
    )
    in_memory_runtime["tools"].register(failing)
    llm: InMemoryLLMProvider = in_memory_runtime["llm"]
    llm._scripted_streams.append(_text_stream(ANSWER))
    for index in range(3):
        llm.queue_tool_call_response(
            tool_call_id=f"fin-{index}", tool_name="Finalize", tool_input={}
        )

    events = await _run(engine)

    assert len(llm.calls) == 3
    # An error that is not marked as missing work is the call's own fault, so
    # the model is made to call the tool again — by name, never with a free
    # choice of tools that could start new research under the answer.
    for request in llm.calls[1:]:
        assert request.extra.get("forced_tool_choice") == "Finalize"
        assert "tool_choice_required" not in request.extra
    assert forced_terminal.REASON_EXHAUSTED in _reasons(events)
    # A failed terminal call is not work: the answer before it still counts,
    # so the prose gate never asks for another one.
    assert not any(
        message.metadata.get(SYNTHETIC_RECOVERY_METADATA_KEY)
        == SYNTHETIC_RECOVERY_PROSE_GATE_REPAIR
        for message in engine.history
    )
    assert _answers(engine) == [ANSWER]
    assert engine.state is LoopState.COMPLETED


@pytest.mark.asyncio
async def test_zero_attempts_completes_without_another_request(
    engine_factory, in_memory_runtime
) -> None:
    rc = _rc(terminal_tool_forced_max_attempts=0)
    engine, _ = _engine(engine_factory, in_memory_runtime, rc=rc)
    llm: InMemoryLLMProvider = in_memory_runtime["llm"]
    llm._scripted_streams.append(_text_stream(ANSWER))
    llm._scripted_streams.append(_text_stream("never requested"))

    events = await _run(engine)

    assert len(llm.calls) == 1
    assert forced_terminal.REASON_EXHAUSTED in _reasons(events)
    assert engine.state is LoopState.COMPLETED


@pytest.mark.asyncio
async def test_an_unregistered_terminal_tool_completes_on_the_answer(
    engine_factory, in_memory_runtime
) -> None:
    rc = _rc()
    engine, _ = _engine(engine_factory, in_memory_runtime, rc=rc, register_finalize=False)
    llm: InMemoryLLMProvider = in_memory_runtime["llm"]
    llm._scripted_streams.append(_text_stream(ANSWER))
    llm._scripted_streams.append(_text_stream("never requested"))

    events = await _run(engine)

    assert len(llm.calls) == 1
    assert forced_terminal.REASON_UNAVAILABLE in _reasons(events)
    assert _answers(engine) == [ANSWER]
    assert engine.state is LoopState.COMPLETED


@pytest.mark.asyncio
async def test_no_answer_yet_is_asked_for_in_words_then_forced(
    engine_factory, in_memory_runtime
) -> None:
    """Below the answer floor there is nothing to seal, and a forced call cannot
    write prose — so the run is told once, and the answer it then writes is
    followed by the forced call."""
    rc = _rc(finalize_prose_gate_min_chars=20, terminal_tool_nudge_write_first_enabled=False)
    engine, finalize = _engine(engine_factory, in_memory_runtime, rc=rc)
    llm: InMemoryLLMProvider = in_memory_runtime["llm"]
    llm._scripted_streams.append(_text_stream("Working on it."))
    llm._scripted_streams.append(_text_stream(ANSWER))
    llm.queue_tool_call_response(
        tool_call_id="fin-1",
        tool_name="Finalize",
        tool_input={"declared_deliverables": []},
    )

    events = await _run(engine)

    assert len(llm.calls) == 3
    reasons = _reasons(events)
    assert reasons.index("terminal_tool_nudge") < reasons.index(forced_terminal.REASON_FORCED)
    assert "forced_tool_choice" not in llm.calls[1].extra
    assert any("has not been called yet" in text for text in _request_texts(llm.calls[1]))
    forced = llm.calls[2]
    assert forced.extra.get("forced_tool_choice") == "Finalize"
    assert forced.extra.get("enable_thinking") is False
    # The one telling stays where it was; nothing was added after the answer.
    assert forced.messages[-1].role is MessageRole.assistant
    assert _request_texts(forced)[-1] == ANSWER
    assert len(finalize.calls) == 1
    assert engine.state is LoopState.COMPLETED


@pytest.mark.asyncio
async def test_thinking_can_be_kept_on_the_forced_call(
    engine_factory, in_memory_runtime
) -> None:
    rc = _rc(terminal_tool_forced_thinking_enabled=True)
    engine, _ = _engine(engine_factory, in_memory_runtime, rc=rc)
    llm: InMemoryLLMProvider = in_memory_runtime["llm"]
    llm._scripted_streams.append(_text_stream(ANSWER))
    llm.queue_tool_call_response(
        tool_call_id="fin-1",
        tool_name="Finalize",
        tool_input={"declared_deliverables": []},
    )

    await _run(engine)

    assert llm.calls[1].extra.get("forced_tool_choice") == "Finalize"
    assert llm.calls[1].extra.get("enable_thinking") is True


@pytest.mark.asyncio
async def test_the_wind_down_answer_is_sealed_by_a_forced_call(
    engine_factory, in_memory_runtime
) -> None:
    """A run cut short by its turn cap is told in words to write its answer
    (the answer is prose, which no forced call can produce) and the answer it
    writes is then sealed by force rather than by a second request in words."""
    rc = _rc(max_turns_per_run=1)
    engine, finalize = _engine(engine_factory, in_memory_runtime, rc=rc)
    llm: InMemoryLLMProvider = in_memory_runtime["llm"]
    llm.queue_tool_call_response(
        tool_call_id="s-1", tool_name="WebSearch", tool_input={"query": "oltp"}
    )
    llm._scripted_streams.append(_text_stream(ANSWER))
    llm.queue_tool_call_response(
        tool_call_id="fin-1",
        tool_name="Finalize",
        tool_input={"declared_deliverables": []},
    )

    events = await _run(engine)

    reasons = _reasons(events)
    assert "soft_stop_notified" in reasons
    assert forced_terminal.REASON_FORCED in reasons
    assert "terminal_tool_nudge" not in reasons
    _assert_forced_finalize(llm.calls[-1], rc)
    assert len(finalize.calls) == 1
    assert engine.state is LoopState.COMPLETED


@pytest.mark.asyncio
async def test_a_prose_gate_refusal_lifts_the_forcing(
    engine_factory, in_memory_runtime
) -> None:
    """When the gate refuses the call because the answer is not really there,
    the next request must let the model write."""
    rc = _rc()
    engine, _ = _engine(engine_factory, in_memory_runtime, rc=rc)
    engine.history.append(
        Message(role=MessageRole.assistant, content_blocks=[TextBlock(text=ANSWER)])
    )
    forced_terminal.arm(engine)
    engine.history.append(
        Message(
            role=MessageRole.user,
            content_blocks=[TextBlock(text="Write the answer first.")],
            metadata={SYNTHETIC_RECOVERY_METADATA_KEY: SYNTHETIC_RECOVERY_PROSE_GATE_REPAIR},
        )
    )

    turn = _turn_at(engine, TurnFlags(), TurnCoordinate.turn_start, turn_budget=10)
    async for _ in _CORE_TURN_POLICIES.apply(turn):
        pass

    assert turn.outcome.directive is TurnDirective.proceed

    assert forced_terminal.is_armed(engine) is False
    # Stepping aside is not free, or a gate that keeps refusing could cycle.
    assert forced_terminal.attempts_spent(engine) == 1


def test_forcing_pins_the_terminal_tool_to_the_surface(
    engine_factory, in_memory_runtime
) -> None:
    engine, _ = _engine(engine_factory, in_memory_runtime, rc=_rc())
    assert "Finalize" not in engine.effective_tool_policy.forced_pinned
    forced_terminal.arm(engine)
    assert "Finalize" in engine.effective_tool_policy.forced_pinned


@pytest.mark.asyncio
async def test_forcing_survives_a_snapshot_round_trip(
    engine_factory, in_memory_runtime
) -> None:
    engine, _ = _engine(engine_factory, in_memory_runtime, rc=_rc())
    forced_terminal.arm(engine)
    forced_terminal.charge_attempt(engine)
    snapshot = engine.snapshot()

    restored = engine_factory(rc=_rc(), expected_terminal_tool="Finalize")
    await restored.resume_from_snapshot(snapshot)

    assert forced_terminal.is_armed(restored) is True
    assert forced_terminal.attempts_spent(restored) == 1


@pytest.mark.asyncio
async def test_a_blocked_terminal_tool_completes_on_the_answer(
    engine_factory, in_memory_runtime
) -> None:
    """A terminal tool the circuit breaker has stopped cannot carry a forced
    call, so no forced request is sent at all: the run completes on its
    answer."""
    rc = _rc()
    engine, finalize = _engine(engine_factory, in_memory_runtime, rc=rc)
    engine._circuit_broken_tools.add("Finalize")
    llm: InMemoryLLMProvider = in_memory_runtime["llm"]
    llm._scripted_streams.append(_text_stream(ANSWER))
    llm._scripted_streams.append(_text_stream("Another answer."))
    llm._scripted_streams.append(_text_stream("never requested"))

    events = await _run(engine)

    assert len(llm.calls) == 1
    assert forced_terminal.REASON_UNAVAILABLE in _reasons(events)
    assert _answers(engine) == [ANSWER]
    assert finalize.calls == []
    assert engine.state is LoopState.COMPLETED


def _streamed_text(events: Sequence[TurnEvent]) -> str:
    return "".join(
        str(e.payload["delta"].get("text", ""))
        for e in events
        if e.type is EventType.CONTENT_BLOCK_DELTA
        and isinstance(e.payload.get("delta"), dict)
        and e.payload["delta"].get("type") == "text_delta"
    )


def _pointer_engine(
    engine_factory: Any, in_memory_runtime: dict[str, Any]
) -> tuple[QueryEngine, _FinalizeTool, InMemoryLLMProvider]:
    """A run whose workspace the reader cannot open, so a filing notice is
    refused as an answer by the prose gate."""
    rc = _rc(model_context_window=1_048_576, workspace_visible_to_user=False)
    engine = engine_factory(rc=rc, expected_terminal_tool="Finalize")
    finalize = _FinalizeTool(tool_name="Finalize", description="End the run")
    registry = in_memory_runtime["tools"]
    registry.register(finalize)
    registry.register(
        MockTool(
            tool_name="Write",
            description="Write a file",
            parameters_schema={"path": {"type": "string"}, "content": {"type": "string"}},
        )
    )
    llm: InMemoryLLMProvider = in_memory_runtime["llm"]
    llm.queue_tool_call_response(
        tool_call_id="w1",
        tool_name="Write",
        tool_input={"path": ARTICLE_PATH, "content": ARTICLE},
    )
    llm._scripted_streams.append(_prose_stream(FILING_NOTICE))
    llm.queue_tool_call_response(tool_call_id="f1", tool_name="Finalize", tool_input={})
    return engine, finalize, llm


@pytest.mark.asyncio
async def test_a_refused_forced_call_delivers_the_corrected_answer(
    engine_factory, in_memory_runtime
) -> None:
    """The gate refuses the forced call because the answer is only a pointer.
    The model is then free to write the real answer, which is streamed, kept,
    and sealed by a forced call."""
    engine, finalize, llm = _pointer_engine(engine_factory, in_memory_runtime)
    llm._scripted_streams.append(_prose_stream(SUBSTANTIVE_ANSWER))
    llm.queue_tool_call_response(tool_call_id="f2", tool_name="Finalize", tool_input={})

    events = await _run(engine, "write the article")

    assert len(llm.calls) == 5
    # The corrective's request is free: not forced, and its text is shown.
    assert "forced_tool_choice" not in llm.calls[3].extra
    assert llm.calls[4].extra.get("forced_tool_choice") == "Finalize"
    assert SUBSTANTIVE_ANSWER[:40] in _streamed_text(events)
    assert any(SUBSTANTIVE_ANSWER[:40] in answer for answer in _answers(engine))
    assert len(finalize.calls) == 1
    assert engine.state is LoopState.COMPLETED


@pytest.mark.asyncio
async def test_a_gate_that_keeps_refusing_cannot_loop_the_forcing(
    engine_factory, in_memory_runtime
) -> None:
    """Refusal, release and re-arm all spend one budget, so a model that never
    produces the call ends in a bounded number of requests."""
    engine, _, llm = _pointer_engine(engine_factory, in_memory_runtime)
    for _ in range(40):
        llm._scripted_streams.append(_prose_stream(SUBSTANTIVE_ANSWER))

    await _run(engine, "write the article")

    assert len(llm.calls) <= 3 + engine.config.rc.terminal_tool_forced_max_attempts
    assert engine.state is LoopState.COMPLETED


@pytest.mark.asyncio
async def test_a_refused_call_lets_the_model_do_the_missing_work(
    engine_factory, in_memory_runtime
) -> None:
    """Prose that only announces a file is sealed like an answer, and the file
    it declares is what the terminal tool can refuse. After a refusal the next
    request requires a tool call of any kind, so the model writes the file;
    that is work, so the forcing steps aside, and the answer the model writes
    after it is visible and sealed."""
    rc = _rc()
    engine = engine_factory(rc=rc, expected_terminal_tool="Finalize")
    engine._live_thinking_enabled = True
    written: list[str] = []

    class _CheckingFinalize(_FinalizeTool):
        async def invoke(self, context: ToolContext, arguments: dict[str, Any]) -> ToolResult:
            if not written:
                self.calls.append(dict(arguments))
                return ToolResult(
                    tool_call_id="",
                    content="report.md does not exist",
                    is_error=True,
                    metadata={TERMINAL_REFUSAL_NEEDS_WORK_METADATA_KEY: True},
                )
            return await super().invoke(context, arguments)

    finalize = _CheckingFinalize(tool_name="Finalize", description="End the run")
    write = MockTool(
        tool_name="Write",
        description="Write a file",
        on_invoke=lambda arguments: written.append(str(arguments.get("path"))),
    )
    registry = in_memory_runtime["tools"]
    registry.register(finalize)
    registry.register(write)
    llm: InMemoryLLMProvider = in_memory_runtime["llm"]
    announcement = "Now let me write the report to report.md."
    declared = {"declared_deliverables": [{"path": "report.md"}]}
    llm._scripted_streams.append(_text_stream(announcement))
    llm.queue_tool_call_response(tool_call_id="f1", tool_name="Finalize", tool_input=declared)
    llm.queue_tool_call_response(
        tool_call_id="w1", tool_name="Write", tool_input={"path": "report.md", "content": "c"}
    )
    llm._scripted_streams.append(_text_stream(ANSWER))
    llm.queue_tool_call_response(tool_call_id="f2", tool_name="Finalize", tool_input=declared)

    events = await _run(engine)

    assert len(llm.calls) == 5
    assert llm.calls[1].extra.get("forced_tool_choice") == "Finalize"
    assert llm.calls[2].extra.get("tool_choice_required") is True
    assert "forced_tool_choice" not in llm.calls[2].extra
    assert llm.calls[2].extra.get("enable_thinking") is False
    assert written == ["report.md"]
    # The work resumed, so the next request is free and its answer is shown.
    assert "tool_choice_required" not in llm.calls[3].extra
    assert "forced_tool_choice" not in llm.calls[3].extra
    assert ANSWER in _streamed_text(events)
    assert llm.calls[4].extra.get("forced_tool_choice") == "Finalize"
    assert _answers(engine) == [announcement, ANSWER]
    assert len(finalize.calls) == 2
    assert engine.state is LoopState.COMPLETED


@pytest.mark.asyncio
async def test_a_verification_refusal_of_announced_work_gets_the_work_done(
    engine_factory, in_memory_runtime
) -> None:
    """The same announcement under a host check that refuses the call before
    it runs: the corrective is answered in the open, with the file written."""
    base = engine_factory(rc=_rc(pre_dispatch_terminal_verify_enabled=True))

    def verify(_engine: QueryEngine, _call: Any) -> str:
        return "report.md was declared but never written; write it, then call Finalize."

    engine = QueryEngine(
        config=replace(
            base.config,
            expected_terminal_tool="Finalize",
            pre_dispatch_terminal_verify_trigger=verify,
        ),
        llm_provider=in_memory_runtime["llm"],
        tool_registry=in_memory_runtime["tools"],
        event_stream=in_memory_runtime["events"],
        hook_manager=in_memory_runtime["hooks"],
        skill_store=in_memory_runtime["skills"],
        blob_store=in_memory_runtime["blobs"],
    )
    finalize = _FinalizeTool(tool_name="Finalize", description="End the run")
    write = MockTool(tool_name="Write", description="Write a file")
    in_memory_runtime["tools"].register(finalize)
    in_memory_runtime["tools"].register(write)
    llm: InMemoryLLMProvider = in_memory_runtime["llm"]
    llm._scripted_streams.append(_text_stream("Now let me write the report to report.md."))
    llm.queue_tool_call_response(tool_call_id="f1", tool_name="Finalize", tool_input={})
    llm.queue_tool_call_response(
        tool_call_id="w1", tool_name="Write", tool_input={"path": "report.md", "content": "c"}
    )
    llm._scripted_streams.append(_text_stream(ANSWER))
    llm.queue_tool_call_response(tool_call_id="f2", tool_name="Finalize", tool_input={})

    events = await _run(engine)

    assert len(llm.calls) == 5
    assert "forced_tool_choice" not in llm.calls[2].extra
    assert "forced_tool_choice" not in llm.calls[3].extra
    assert write.calls == [{"path": "report.md", "content": "c"}]
    assert ANSWER in _streamed_text(events)
    assert llm.calls[4].extra.get("forced_tool_choice") == "Finalize"
    assert _answers(engine)[-1] == ANSWER
    assert len(finalize.calls) == 1
    assert engine.state is LoopState.COMPLETED


@pytest.mark.asyncio
async def test_a_provider_ignoring_the_forced_choice_cannot_run_other_tools(
    engine_factory, in_memory_runtime
) -> None:
    rc = _rc(terminal_tool_forced_max_attempts=2)
    engine, finalize = _engine(engine_factory, in_memory_runtime, rc=rc)
    search = MockTool(tool_name="Research", description="Research more")
    in_memory_runtime["tools"].register(search)
    llm: InMemoryLLMProvider = in_memory_runtime["llm"]
    llm._scripted_streams.append(_text_stream(ANSWER))
    llm.queue_tool_call_response(
        tool_call_id="r1",
        tool_name="Research",
        tool_input={"query": "earlier question"},
        text_prefix="Let me also look up your earlier question.",
    )
    llm.queue_tool_call_response(
        tool_call_id="r2", tool_name="Research", tool_input={"query": "again"}
    )
    llm._scripted_streams.append(_text_stream("never requested"))

    events = await _run(engine)

    assert len(llm.calls) == 3
    assert all(
        request.extra.get("forced_tool_choice") == "Finalize" for request in llm.calls[1:]
    )
    assert search.calls == []
    assert "earlier question" not in _streamed_text(events)
    # Not even announced on the live stream: a reader never sees a call that
    # will not run.
    assert not any(e.payload.get("tool_call_id") in {"r1", "r2"} for e in events)
    assert not any(
        isinstance(block, ToolUseBlock) and block.name == "Research"
        for message in engine.history
        for block in message.content_blocks
    )
    assert _answers(engine) == [ANSWER]
    assert finalize.calls == []
    assert engine.state is LoopState.COMPLETED


@pytest.mark.asyncio
async def test_a_verification_refusal_lets_the_model_fix_the_answer(
    engine_factory, in_memory_runtime
) -> None:
    """A host check refuses the first forced call with a correction. The model
    answers the correction in the open, and the fixed answer is sealed."""
    base = engine_factory(rc=_rc(pre_dispatch_terminal_verify_enabled=True))

    def verify(_engine: QueryEngine, _call: Any) -> str:
        return "The figure for MySQL is wrong; correct it, then call Finalize."

    engine = QueryEngine(
        config=replace(
            base.config,
            expected_terminal_tool="Finalize",
            pre_dispatch_terminal_verify_trigger=verify,
        ),
        llm_provider=in_memory_runtime["llm"],
        tool_registry=in_memory_runtime["tools"],
        event_stream=in_memory_runtime["events"],
        hook_manager=in_memory_runtime["hooks"],
        skill_store=in_memory_runtime["skills"],
        blob_store=in_memory_runtime["blobs"],
    )
    finalize = _FinalizeTool(tool_name="Finalize", description="End the run")
    in_memory_runtime["tools"].register(finalize)
    llm: InMemoryLLMProvider = in_memory_runtime["llm"]
    fixed = "Corrected: InnoDB keeps old row versions in its undo log."
    llm._scripted_streams.append(_text_stream(ANSWER))
    llm.queue_tool_call_response(
        tool_call_id="f1", tool_name="Finalize", tool_input={"declared_deliverables": []}
    )
    llm._scripted_streams.append(_text_stream(fixed))
    llm.queue_tool_call_response(
        tool_call_id="f2", tool_name="Finalize", tool_input={"declared_deliverables": []}
    )

    events = await _run(engine)

    assert len(llm.calls) == 4
    assert llm.calls[1].extra.get("forced_tool_choice") == "Finalize"
    assert "forced_tool_choice" not in llm.calls[2].extra
    assert "correct it" in _request_texts(llm.calls[2])[-1]
    assert fixed in _streamed_text(events)
    assert llm.calls[3].extra.get("forced_tool_choice") == "Finalize"
    assert _answers(engine)[-1] == fixed
    assert len(finalize.calls) == 1
    assert engine.state is LoopState.COMPLETED


@pytest.mark.asyncio
async def test_a_message_arriving_mid_forcing_is_answered_in_the_open(
    engine_factory, in_memory_runtime
) -> None:
    engine, _ = _engine(engine_factory, in_memory_runtime, rc=_rc())
    engine.history.append(
        Message(role=MessageRole.assistant, content_blocks=[TextBlock(text=ANSWER)])
    )
    forced_terminal.arm(engine)
    engine.history.append(
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="And for SQLite?")])
    )

    turn = _turn_at(engine, TurnFlags(), TurnCoordinate.turn_start, turn_budget=10)
    async for _ in _CORE_TURN_POLICIES.apply(turn):
        pass

    assert forced_terminal.is_armed(engine) is False
    assert forced_terminal.request_mode(engine) is None
    assert forced_terminal.attempts_spent(engine) == 1


@pytest.mark.asyncio
async def test_exhaustion_is_read_before_the_forcing_steps_aside(
    engine_factory, in_memory_runtime
) -> None:
    engine, _ = _engine(engine_factory, in_memory_runtime, rc=_rc())
    engine.history.append(
        Message(role=MessageRole.assistant, content_blocks=[TextBlock(text=ANSWER)])
    )
    engine.transition_to(LoopState.RUNNING)
    forced_terminal.arm(engine)
    forced_terminal.spend_all(engine)
    engine.history.append(
        Message(
            role=MessageRole.user,
            content_blocks=[TextBlock(text="Write the answer first.")],
            metadata={SYNTHETIC_RECOVERY_METADATA_KEY: SYNTHETIC_RECOVERY_PROSE_GATE_REPAIR},
        )
    )

    turn = _turn_at(engine, TurnFlags(), TurnCoordinate.turn_start, turn_budget=10)
    events = [event async for event in _CORE_TURN_POLICIES.apply(turn)]

    assert turn.outcome.directive is TurnDirective.end_turn
    assert forced_terminal.REASON_EXHAUSTED in _reasons(events)
    assert engine.state is LoopState.COMPLETED


@pytest.mark.asyncio
async def test_a_precondition_turn_does_not_spend_the_forcing(
    engine_factory, in_memory_runtime
) -> None:
    calls: list[str] = []
    policy = TerminalNudgePolicy(
        required=lambda _engine: True,
        append=lambda _engine: None,
        state_change=lambda _engine, _reason: TurnEvent(
            type=EventType.STATE_CHANGED, run_id="r", payload={}
        ),
        forced=ForcedTerminalCall(
            answer_delivered=lambda _engine: True,
            tool_registered=lambda _engine: True,
            arm=lambda _engine: True,
            armed=lambda _engine: True,
            release=lambda _engine: calls.append("release"),
            question_pending=lambda _engine: False,
            working_again=lambda _engine: False,
            work_requested=lambda _engine: False,
            write_first_before_sealing=lambda _engine: False,
            slot_taken=lambda _engine: True,
            charge=_record_charge(calls),
            set_mode=lambda _engine, mode: calls.append(f"mode={mode}"),
            exhausted=lambda _engine: False,
            complete=_no_completion,
        ),
    )
    engine, _ = _engine(engine_factory, in_memory_runtime, rc=_rc())
    turn = _turn_at(engine, TurnFlags(), TurnCoordinate.turn_start, turn_budget=10)
    async for _ in policy.apply(turn):
        pass

    assert calls == ["mode=None"]


def _record_charge(calls: list[str]) -> Any:
    def charge(_engine: Any) -> int:
        calls.append("charge")
        return len(calls)

    return charge


async def _no_completion(_engine: Any, _reason: str) -> AsyncIterator[TurnEvent]:
    raise AssertionError("the run must not complete here")
    yield  # pragma: no cover


@pytest.mark.asyncio
async def test_a_run_resumed_mid_forcing_stays_forced(
    engine_factory, in_memory_runtime
) -> None:
    """A run picked up from a snapshot taken while its terminal call was being
    forced continues forcing it; the re-drive is not a free request under the
    delivered answer."""
    rc = _rc()
    engine = engine_factory(rc=rc, expected_terminal_tool="Finalize")
    refusing = MockTool(
        tool_name="Finalize",
        description="End the run",
        response_content="declared_deliverables is required",
        response_is_error=True,
    )
    in_memory_runtime["tools"].register(refusing)
    llm: InMemoryLLMProvider = in_memory_runtime["llm"]
    llm._scripted_streams.append(_text_stream(ANSWER))
    llm.queue_tool_call_response(tool_call_id="f1", tool_name="Finalize", tool_input={})

    user = Message(role=MessageRole.user, content_blocks=[TextBlock(text="Compare them.")])
    async for event in engine.run(user):
        if event.type is EventType.TOOL_RESULT:
            break
    snapshot = engine.snapshot()
    assert snapshot["terminal_call_forced"] is True

    registry = InMemoryToolRegistry()
    finalize = _FinalizeTool(tool_name="Finalize", description="End the run")
    registry.register(finalize)
    latch_seen: list[bool] = []

    class _LatchWatchingLLM(InMemoryLLMProvider):
        def stream_with_tools(self, request: LLMRequest):  # type: ignore[no-untyped-def]
            latch_seen.append(resumed._terminal_only_active)
            return super().stream_with_tools(request)

    resumed_llm = _LatchWatchingLLM()
    resumed_llm.queue_tool_call_response(
        tool_call_id="f2", tool_name="Finalize", tool_input={"declared_deliverables": []}
    )
    resumed = QueryEngine(
        config=engine.config,
        llm_provider=resumed_llm,
        tool_registry=registry,
        event_stream=InMemoryEventStream(),
        hook_manager=InMemoryHookManager(),
        skill_store=InMemorySkillStore(),
        blob_store=InMemoryBlobStore(),
    )
    async for _ in resume(resumed, snapshot):
        pass

    assert len(resumed_llm.calls) == 1
    request = resumed_llm.calls[0]
    # The re-drive is a forced request, never a free one under the answer.
    assert (
        request.extra.get("forced_tool_choice") == "Finalize"
        or request.extra.get("tool_choice_required") is True
    )
    assert request.extra.get("enable_thinking") is False
    assert forced_terminal.attempts_spent(resumed) >= 2
    # The terminal-only latch that goes with a forcing came back with it.
    assert latch_seen == [True]
    assert len(finalize.calls) == 1
    assert _answers(resumed) == [ANSWER]
    assert resumed.state is LoopState.COMPLETED


class _WorkingModel(InMemoryLLMProvider):
    """A model that answers every free request with a new answer and every
    required one with more work — the shape that turns each work turn into
    another visible answer."""

    def __init__(self) -> None:
        super().__init__()
        self.answers_written = 0

    async def stream_with_tools(self, request: LLMRequest):  # type: ignore[no-untyped-def]
        self._calls.append(request)
        forced = request.extra.get("forced_tool_choice")
        if forced or request.extra.get("tool_choice_required"):
            name = forced or "Research"
            args: dict[str, Any] = {"query": "more"} if name == "Research" else {}
            call_id = f"c{len(self._calls)}"
            events = [
                LLMStreamEvent(name="message_start", payload={}),
                LLMStreamEvent(name="tool_use_start", payload={"tool_call_id": call_id, "tool_name": name}),
                LLMStreamEvent(name="tool_use_stop", payload={"tool_call_id": call_id, "final_input": args}),
                LLMStreamEvent(name="message_stop", payload={"stop_reason": StopReason.tool_use.value}),
            ]
        else:
            self.answers_written += 1
            events = _text_stream(f"Answer number {self.answers_written}, written out in full.")
        for event in events:
            yield event


@pytest.mark.asyncio
@pytest.mark.parametrize("needs_work", [False, True])
@pytest.mark.parametrize("breaker_cap", [3, 50])
async def test_a_tool_that_keeps_refusing_is_bounded_in_answers(
    engine_factory, in_memory_runtime, needs_work: bool, breaker_cap: int
) -> None:
    """A terminal tool that refuses every call. Refusals for the call itself
    never open a work turn, so the reader sees one answer. Refusals for missing
    work open exactly one, so the reader sees at most the answer and its one
    correction — whatever the budget."""
    rc = _rc(terminal_tool_forced_max_attempts=10, max_consecutive_tool_errors=breaker_cap)
    engine = engine_factory(rc=rc, expected_terminal_tool="Finalize")
    engine.llm = _WorkingModel()  # type: ignore[assignment]
    refusing = MockTool(
        tool_name="Finalize",
        description="End the run",
        response_content="refused",
        response_is_error=True,
        response_metadata={TERMINAL_REFUSAL_NEEDS_WORK_METADATA_KEY: True} if needs_work else {},
    )
    research = MockTool(tool_name="Research", description="Research more")
    in_memory_runtime["tools"].register(refusing)
    in_memory_runtime["tools"].register(research)

    events = await _run(engine)

    assert engine.state is LoopState.COMPLETED
    # Ended by the budget or by the breaker taking the tool away, never by
    # the model writing its way out.
    assert {forced_terminal.REASON_EXHAUSTED, forced_terminal.REASON_UNAVAILABLE} & set(
        _reasons(events)
    )
    visible = [a for a in _answers(engine) if a.startswith("Answer number")]
    assert len(visible) == (2 if needs_work else 1)
    assert len(research.calls) == (1 if needs_work else 0)
    assert len(engine.llm.calls) <= 2 + 2 * rc.terminal_tool_forced_max_attempts  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_the_write_first_nudge_can_come_before_the_seal(
    engine_factory, in_memory_runtime
) -> None:
    """With the host switch on, prose that may only announce a file gets the
    write-first telling instead of a seal; the file is written, and the answer
    the model ends on is then sealed by force."""
    rc = _rc(
        terminal_tool_nudge_write_first_before_forcing=True,
        terminal_tool_nudge_write_first_enabled=True,
    )
    engine, finalize = _engine(engine_factory, in_memory_runtime, rc=rc)
    write = MockTool(tool_name="Write", description="Write a file")
    in_memory_runtime["tools"].register(write)
    llm: InMemoryLLMProvider = in_memory_runtime["llm"]
    llm._scripted_streams.append(_text_stream("Now let me write the report to report.md."))
    llm.queue_tool_call_response(
        tool_call_id="w1", tool_name="Write", tool_input={"path": "report.md", "content": "c"}
    )
    llm._scripted_streams.append(_text_stream(ANSWER))
    llm.queue_tool_call_response(
        tool_call_id="f1", tool_name="Finalize", tool_input={"declared_deliverables": []}
    )

    events = await _run(engine)

    reasons = _reasons(events)
    assert reasons.index("terminal_tool_nudge") < reasons.index(forced_terminal.REASON_FORCED)
    assert "forced_tool_choice" not in llm.calls[1].extra
    assert write.calls == [{"path": "report.md", "content": "c"}]
    assert llm.calls[3].extra.get("forced_tool_choice") == "Finalize"
    assert _answers(engine)[-1] == ANSWER
    assert len(finalize.calls) == 1
    assert engine.state is LoopState.COMPLETED


@pytest.mark.asyncio
async def test_the_write_first_switch_is_inert_once_a_file_is_written(
    engine_factory, in_memory_runtime
) -> None:
    rc = _rc(
        terminal_tool_nudge_write_first_before_forcing=True,
        terminal_tool_nudge_write_first_enabled=True,
    )
    engine, finalize = _engine(engine_factory, in_memory_runtime, rc=rc)
    in_memory_runtime["tools"].register(MockTool(tool_name="Write", description="Write a file"))
    llm: InMemoryLLMProvider = in_memory_runtime["llm"]
    llm.queue_tool_call_response(
        tool_call_id="w1", tool_name="Write", tool_input={"path": "report.md", "content": "c"}
    )
    llm._scripted_streams.append(_text_stream(ANSWER))
    llm.queue_tool_call_response(
        tool_call_id="f1", tool_name="Finalize", tool_input={"declared_deliverables": []}
    )

    events = await _run(engine)

    assert "terminal_tool_nudge" not in _reasons(events)
    assert llm.calls[2].extra.get("forced_tool_choice") == "Finalize"
    assert len(finalize.calls) == 1


@pytest.mark.asyncio
async def test_a_repeated_call_on_a_forced_turn_never_reaches_the_loop_guard(
    engine_factory, in_memory_runtime
) -> None:
    """A provider ignores the forced choice and repeats a call the run already
    made. The call is dropped before the repeat guard sees it, so no result is
    filed for a call the transcript does not hold, the answer is not demoted,
    and the run seals on it."""
    rc = _rc(loop_guard_enabled=True, loop_guard_identical_tool_limit=1)
    engine, finalize = _engine(engine_factory, in_memory_runtime, rc=rc)
    search = MockTool(tool_name="CatalogSearch", description="Search the catalogue")
    in_memory_runtime["tools"].register(search)
    llm: InMemoryLLMProvider = in_memory_runtime["llm"]
    llm.queue_tool_call_response(tool_call_id="s1", tool_name="CatalogSearch", tool_input={"q": "x"})
    llm._scripted_streams.append(_text_stream(ANSWER))
    llm.queue_tool_call_response(tool_call_id="s2", tool_name="CatalogSearch", tool_input={"q": "x"})
    llm.queue_tool_call_response(
        tool_call_id="f1", tool_name="Finalize", tool_input={"declared_deliverables": []}
    )

    events = await _run(engine)

    uses = [
        block.tool_call_id
        for message in engine.history
        for block in message.content_blocks
        if isinstance(block, ToolUseBlock)
    ]
    results = [
        block.tool_call_id
        for message in engine.history
        for block in message.content_blocks
        if isinstance(block, ToolResultBlock)
    ]
    assert uses == ["s1", "f1"]
    assert results == ["s1", "f1"]
    assert not any(e.payload.get("tool_call_id") == "s2" for e in events)
    assert len(search.calls) == 1
    assert len(llm.calls) == 4
    assert _answers(engine) == [ANSWER]
    assert len(finalize.calls) == 1
    assert engine.state is LoopState.COMPLETED
