"""The run wind-down: what a stop that actually stops looks like.

The mechanism this replaced counted tool calls and, on reaching the budget,
appended a paragraph to the tool result asking the agent to wrap up — a
paragraph that said, in the same breath, "this is advisory — tools still run".
In the run it was written for it fired once at call eighty and the agent made
another eighteen calls. Five other bounds each ended somewhere else, on their
own flag, with their own semantics.

So the assertions here are about REMOVAL, not about wording. The load-bearing
one is :func:`test_the_withdrawal_beats_the_pinned_floor`: the tool surface is
built through a floor whose entire job is to keep tools visible past every clip
and every whitelist, so a withdrawal that does not outrank that floor is a
withdrawal the model never experiences. What is inspected is the tool list on
the ``LLMRequest`` — what the model is actually shown — and never the policy the
run was configured with.
"""
from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

import pytest

from protocore.contracts.llm import (
    LLMProviderError,
    LLMRequest,
    LLMResponse,
    LLMStreamEvent,
)
from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.tools import Tool, ToolContext
from protocore.contracts.types import (
    SYNTHETIC_RECOVERY_METADATA_KEY,
    TERMINAL_TOOL_METADATA_KEY,
    Message,
    MessageRole,
    StopReason,
    TextBlock,
    ToolDefinition,
    ToolParameterSchema,
    ToolResult,
)
from protocore.runtime import soft_stop as _soft_stop
from protocore.runtime.events import EventType, TurnEvent
from protocore.runtime.loop_state import LoopState
from protocore.runtime.query_engine import QueryEngine, QueryEngineConfig
from protocore.runtime.tool_registry import ToolRegistry
from protocore.tests_support.adapters import (
    InMemoryBlobStore,
    InMemoryEventStream,
    InMemoryHookManager,
    InMemorySkillStore,
)
from tests._fixtures.tool_roles import CONVENTIONAL_TOOL_ROLES

TERMINAL_TOOL = "Finalize"


# ----------------------------------------------------------------------
# Fixtures — a REAL registry, because the floor is what has to be beaten
# ----------------------------------------------------------------------


class _NamedTool(Tool):
    """A tool that records its calls. Name-parameterised so a run can hold several."""

    def __init__(self, name: str) -> None:
        self._name = name
        self.calls: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self._name,
            description=f"the {self._name} tool",
            parameters=ToolParameterSchema(properties={"x": {"type": "string"}}),
        )

    async def invoke(self, context: ToolContext, arguments: dict[str, Any]) -> ToolResult:
        self.calls.append(dict(arguments))
        return ToolResult(tool_call_id="", content=f"{self._name}-ok", is_error=False)


class _FinalizeTool(Tool):
    """Message-carrying terminal tool — auto-exempt from the prose gate."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return TERMINAL_TOOL

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=TERMINAL_TOOL,
            description="end the run",
            parameters=ToolParameterSchema(properties={"message": {"type": "string"}}),
        )

    async def invoke(self, context: ToolContext, arguments: dict[str, Any]) -> ToolResult:
        self.calls.append(dict(arguments))
        return ToolResult(
            tool_call_id="",
            content="finalized",
            is_error=False,
            metadata={TERMINAL_TOOL_METADATA_KEY: True},
        )


def _build_engine(
    *,
    rc: LoopConstants,
    llm: object,
    tools: list[Tool],
    expected_terminal_tool: str | None = TERMINAL_TOOL,
) -> QueryEngine:
    registry = ToolRegistry(tools)
    return QueryEngine(
        config=QueryEngineConfig(
            tool_roles=CONVENTIONAL_TOOL_ROLES,
            run_id="run-soft-stop",
            tenant_id="tenant-soft-stop",
            session_id="sess-soft-stop",
            model_name="qwen3.6-35b-a3b",
            rc=rc,
            expected_terminal_tool=expected_terminal_tool,
        ),
        llm_provider=llm,  # type: ignore[arg-type]
        tool_registry=registry,  # type: ignore[arg-type]
        event_stream=InMemoryEventStream(),
        hook_manager=InMemoryHookManager(),
        skill_store=InMemorySkillStore(),
        blob_store=InMemoryBlobStore(),
    )


class _ScriptedLLM:
    """Emits a scripted sequence of turns, repeating the last one forever."""

    def __init__(self, turns: list[dict[str, Any]]) -> None:
        self._turns = turns
        self.calls: list[LLMRequest] = []

    async def stream_with_tools(  # type: ignore[no-untyped-def]
        self, request: LLMRequest
    ) -> AsyncIterator[LLMStreamEvent]:
        self.calls.append(request)
        turn = self._turns[min(len(self.calls) - 1, len(self._turns) - 1)]
        yield LLMStreamEvent(name="message_start", payload={})
        if turn.get("text"):
            yield LLMStreamEvent(
                name="content_block_delta", payload={"text": turn["text"]}
            )
        if turn.get("tool"):
            call_id = f"toolu_{len(self.calls)}"
            yield LLMStreamEvent(
                name="tool_use_start",
                payload={"tool_call_id": call_id, "tool_name": turn["tool"]},
            )
            yield LLMStreamEvent(
                name="tool_use_input_delta",
                payload={
                    "tool_call_id": call_id,
                    "partial_input_json": json.dumps(turn.get("args", {})),
                },
            )
            yield LLMStreamEvent(
                name="tool_use_stop",
                payload={"tool_call_id": call_id, "final_input": turn.get("args", {})},
            )
            yield LLMStreamEvent(
                name="message_stop", payload={"stop_reason": StopReason.tool_use.value}
            )
        else:
            yield LLMStreamEvent(
                name="message_stop", payload={"stop_reason": StopReason.end_turn.value}
            )

    async def complete_structured(self, request, schema):  # type: ignore[no-untyped-def]
        return LLMResponse(
            message=Message(role=MessageRole.assistant, content_blocks=[]),
            stop_reason=StopReason.end_turn,
        )

    def count_tokens(self, text, model=None) -> int:  # type: ignore[no-untyped-def]
        return max(1, len(text) // 4)


class _FailingLLM:
    """Raises ``exc`` on the first call, then behaves like ``fallback``."""

    def __init__(self, exc: BaseException, fallback: _ScriptedLLM) -> None:
        self._exc = exc
        self._fallback = fallback
        self.calls: list[LLMRequest] = []

    async def stream_with_tools(  # type: ignore[no-untyped-def]
        self, request: LLMRequest
    ) -> AsyncIterator[LLMStreamEvent]:
        self.calls.append(request)
        if len(self.calls) == 1:
            if False:  # pragma: no cover — generator protocol marker
                yield LLMStreamEvent(name="never", payload={})
            raise self._exc
        async for evt in self._fallback.stream_with_tools(request):
            yield evt

    async def complete_structured(self, request, schema):  # type: ignore[no-untyped-def]
        return await self._fallback.complete_structured(request, schema)

    def count_tokens(self, text, model=None) -> int:  # type: ignore[no-untyped-def]
        return self._fallback.count_tokens(text, model)


def _user(text: str = "do the work") -> Message:
    return Message(role=MessageRole.user, content_blocks=[TextBlock(text=text)])


def _reasons(events: list[TurnEvent]) -> list[str]:
    return [
        str(e.payload.get("reason"))
        for e in events
        if e.type is EventType.STATE_CHANGED
    ]


def _final_stop(events: list[TurnEvent]) -> TurnEvent:
    stops = [
        e
        for e in events
        if e.type is EventType.MESSAGE_STOP
        and e.payload.get("stop_reason") != "tool_use"
    ]
    assert stops, [e.payload for e in events if e.type is EventType.MESSAGE_STOP]
    return stops[-1]


def _advertised(request: LLMRequest) -> set[str]:
    return {t.name for t in request.tools}


# ----------------------------------------------------------------------
# The five steps, in order, on the bus
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_wind_down_is_five_observable_steps_in_order() -> None:
    """Notify → withdraw → (prose gate) → finalize → stop, each on the bus.

    "The soft stop fired" has to be a thing a log can be asked about. Inferring
    it from an agent's behaviour is what made the old mechanism impossible to
    audit: it left no trace except a paragraph inside a tool result.
    """
    rc = LoopConstants(model_context_window=4_096, leader_tool_call_soft_cap=1)
    finalize = _FinalizeTool()
    llm = _ScriptedLLM(
        [
            {"tool": "Read", "args": {"x": "a"}},
            {"text": "Here is what I found: the value is 42."},
            {"tool": TERMINAL_TOOL, "args": {"message": "done"}},
        ]
    )
    engine = _build_engine(rc=rc, llm=llm, tools=[_NamedTool("Read"), finalize])

    events = [evt async for evt in engine.run(_user())]

    reasons = _reasons(events)
    assert reasons.index("soft_stop_notified") < reasons.index(
        "soft_stop_tools_withdrawn"
    )
    assert "soft_stop_finalized" in reasons
    assert reasons.index("soft_stop_tools_withdrawn") < reasons.index(
        "soft_stop_finalized"
    )
    assert _final_stop(events).payload["stop_reason"] == "soft_stop"
    assert engine.state is LoopState.COMPLETED


@pytest.mark.asyncio
async def test_a_run_that_answered_under_the_wind_down_completes() -> None:
    rc = LoopConstants(model_context_window=4_096, leader_tool_call_soft_cap=1)
    llm = _ScriptedLLM(
        [
            {"tool": "Read", "args": {"x": "a"}},
            {"text": "The answer the user asked for."},
        ]
    )
    engine = _build_engine(rc=rc, llm=llm, tools=[_NamedTool("Read"), _FinalizeTool()])

    events = [evt async for evt in engine.run(_user())]

    stop = _final_stop(events)
    assert stop.payload["stop_reason"] == "soft_stop"
    assert stop.payload["has_final_answer"] is True
    assert stop.payload["soft_stop_cause"] == _soft_stop.CAUSE_TOOL_CALL_BUDGET
    assert engine.state is LoopState.COMPLETED


@pytest.mark.asyncio
async def test_a_wind_down_that_produced_no_answer_does_not_complete() -> None:
    """A run given turns to answer in, that did not answer, is not a success."""
    rc = LoopConstants(
        model_context_window=4_096,
        leader_tool_call_soft_cap=1,
        soft_stop_max_turns=1,
    )
    # The model keeps calling the tool it no longer has; it never writes prose.
    llm = _ScriptedLLM([{"tool": "Read", "args": {"x": "a"}}])
    engine = _build_engine(rc=rc, llm=llm, tools=[_NamedTool("Read"), _FinalizeTool()])

    events = [evt async for evt in engine.run(_user())]

    stop = _final_stop(events)
    assert stop.payload["stop_reason"] == "soft_stop"
    assert stop.payload["has_final_answer"] is False
    assert engine.state is LoopState.FAILED


@pytest.mark.asyncio
async def test_the_notification_lands_in_history_as_the_runtimes_own_words() -> None:
    """Marked synthetic, so it cannot be mistaken for the model answering."""
    rc = LoopConstants(model_context_window=4_096, leader_tool_call_soft_cap=1)
    llm = _ScriptedLLM([{"tool": "Read", "args": {"x": "a"}}, {"text": "done"}])
    engine = _build_engine(rc=rc, llm=llm, tools=[_NamedTool("Read"), _FinalizeTool()])

    async for _ in engine.run(_user()):
        pass

    notices = [
        m
        for m in engine.history
        if m.metadata.get(SYNTHETIC_RECOVERY_METADATA_KEY)
        == _soft_stop.SYNTHETIC_RECOVERY_SOFT_STOP
    ]
    assert len(notices) == 1
    assert notices[0].role is MessageRole.user


@pytest.mark.asyncio
async def test_the_notification_is_bilingual_and_names_the_bound_that_was_hit() -> None:
    """A model told to wrap up in a language the conversation is not in switches
    languages before it wraps up. And "the run is closing" is not actionable
    without which budget ran out."""
    rc = LoopConstants(model_context_window=4_096, leader_tool_call_soft_cap=1)
    llm = _ScriptedLLM([{"tool": "Read", "args": {"x": "a"}}, {"text": "done"}])
    engine = _build_engine(rc=rc, llm=llm, tools=[_NamedTool("Read"), _FinalizeTool()])

    async for _ in engine.run(_user()):
        pass

    notice = next(
        m
        for m in engine.history
        if m.metadata.get(SYNTHETIC_RECOVERY_METADATA_KEY)
        == _soft_stop.SYNTHETIC_RECOVERY_SOFT_STOP
    )
    text = "".join(b.text for b in notice.content_blocks)
    assert "Write your final response" in text
    assert "Напишите финальный ответ" in text
    assert _soft_stop.CAUSE_TOOL_CALL_BUDGET in text
    assert "{cause}" not in text


# ----------------------------------------------------------------------
# The withdrawal — composition of the surface, not the wording of a hint
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_withdrawal_beats_the_pinned_floor() -> None:
    """The one assertion the whole mechanism rests on.

    ``tool_surface_forced_pins`` exists precisely to keep tools on the surface
    past every clip and every whitelist — on the stand ``Agent`` and the six
    core file tools sit in it. A withdrawal expressed anywhere the floor can
    re-admit is a withdrawal the model never experiences, and a test written
    against the CONFIGURED policy would pass while the live run failed. So this
    reads the tool list off the request the provider was actually sent.
    """
    rc = LoopConstants(
        model_context_window=4_096,
        leader_tool_call_soft_cap=1,
        tool_surface_forced_pins=("Agent", "Read"),
    )
    llm = _ScriptedLLM(
        [
            {"tool": "Read", "args": {"x": "a"}},
            {"text": "the answer"},
        ]
    )
    engine = _build_engine(
        rc=rc,
        llm=llm,
        tools=[_NamedTool("Read"), _NamedTool("Agent"), _FinalizeTool()],
    )

    async for _ in engine.run(_user()):
        pass

    # Before: the floor is doing its job.
    assert {"Read", "Agent"} <= _advertised(llm.calls[0])
    # After: nothing but the terminal tool, floor or no floor.
    assert _advertised(llm.calls[1]) == {TERMINAL_TOOL}


@pytest.mark.asyncio
async def test_a_withdrawn_tool_is_also_refused_at_dispatch() -> None:
    """Advertising and dispatch read the same policy, so they cannot disagree.

    A model working from a stale schema in its own context will try a tool that
    is no longer on the surface. It must not run.
    """
    rc = LoopConstants(
        model_context_window=4_096,
        leader_tool_call_soft_cap=1,
        soft_stop_max_turns=1,
    )
    read = _NamedTool("Read")
    llm = _ScriptedLLM([{"tool": "Read", "args": {"x": "a"}}])
    engine = _build_engine(rc=rc, llm=llm, tools=[read, _FinalizeTool()])

    async for _ in engine.run(_user()):
        pass

    # One call before the withdrawal; every attempt after it was refused.
    assert read.calls == [{"x": "a"}]


@pytest.mark.asyncio
async def test_with_no_terminal_tool_the_surface_is_emptied_entirely() -> None:
    """Such a run ends by writing its answer, so nothing on the surface is right.

    An empty whitelist reads as "no restriction" in the policy model — the exact
    opposite — which is why the emptiness is stated explicitly rather than left
    to fall out of an empty set.
    """
    rc = LoopConstants(model_context_window=4_096, leader_tool_call_soft_cap=1)
    llm = _ScriptedLLM([{"tool": "Read", "args": {"x": "a"}}, {"text": "the answer"}])
    engine = _build_engine(
        rc=rc,
        llm=llm,
        tools=[_NamedTool("Read"), _NamedTool("Agent")],
        expected_terminal_tool=None,
    )

    async for _ in engine.run(_user()):
        pass

    assert _advertised(llm.calls[0])  # something was advertised before
    assert _advertised(llm.calls[1]) == set()
    assert engine.state is LoopState.COMPLETED


@pytest.mark.asyncio
async def test_the_artifact_sealer_survives_the_withdrawal() -> None:
    """A run cut short mid-file has that file on disk, unsealed.

    Removing the one tool that can close it would throw the work away in the
    name of stopping cleanly.
    """
    rc = LoopConstants(
        model_context_window=4_096,
        leader_tool_call_soft_cap=1,
        longfile_convergence_enabled=True,
    )
    llm = _ScriptedLLM([{"tool": "Read", "args": {"x": "a"}}, {"text": "the answer"}])
    engine = _build_engine(
        rc=rc,
        llm=llm,
        tools=[_NamedTool("Read"), _NamedTool("FinalizeFile"), _FinalizeTool()],
    )
    # An in-flight, truncation-gated, past-the-floor, unsealed artifact.
    engine._longfile_active_path = "/workspace/big.py"
    engine._longfile_active_file_bytes = 12_000
    engine._longfile_truncated_paths.add("/workspace/big.py")

    async for _ in engine.run(_user()):
        pass

    assert _advertised(llm.calls[1]) == {TERMINAL_TOOL, "FinalizeFile"}


@pytest.mark.asyncio
async def test_a_run_with_no_open_artifact_does_not_keep_the_sealer() -> None:
    rc = LoopConstants(
        model_context_window=4_096,
        leader_tool_call_soft_cap=1,
        longfile_convergence_enabled=True,
    )
    llm = _ScriptedLLM([{"tool": "Read", "args": {"x": "a"}}, {"text": "the answer"}])
    engine = _build_engine(
        rc=rc,
        llm=llm,
        tools=[_NamedTool("Read"), _NamedTool("FinalizeFile"), _FinalizeTool()],
    )

    async for _ in engine.run(_user()):
        pass

    assert _advertised(llm.calls[1]) == {TERMINAL_TOOL}


@pytest.mark.asyncio
async def test_the_tool_result_the_model_reads_is_left_alone() -> None:
    """The stop is a change to the surface, not a paragraph glued to a result.

    Appending to a tool result is what the old mechanism did, and it also broke
    every consumer that parses that body as JSON.
    """
    rc = LoopConstants(model_context_window=4_096, leader_tool_call_soft_cap=1)
    llm = _ScriptedLLM([{"tool": "Read", "args": {"x": "a"}}, {"text": "answer"}])
    engine = _build_engine(rc=rc, llm=llm, tools=[_NamedTool("Read"), _FinalizeTool()])

    events = [evt async for evt in engine.run(_user())]

    result = next(e for e in events if e.type is EventType.TOOL_RESULT)
    assert result.payload["content_blocks"][0]["text"] == "Read-ok"


# ----------------------------------------------------------------------
# Every bound takes the same path
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_turn_cap_takes_the_wind_down() -> None:
    rc = LoopConstants(model_context_window=4_096, max_turns_per_run=1)
    llm = _ScriptedLLM([{"tool": "Read", "args": {"x": "a"}}, {"text": "the answer"}])
    engine = _build_engine(rc=rc, llm=llm, tools=[_NamedTool("Read"), _FinalizeTool()])

    events = [evt async for evt in engine.run(_user())]

    assert "soft_stop_notified" in _reasons(events)
    causes = {
        e.payload.get("soft_stop_cause")
        for e in events
        if e.type is EventType.STATE_CHANGED
        and e.payload.get("reason") == "soft_stop_notified"
    }
    assert causes == {_soft_stop.CAUSE_MAX_TURNS}
    assert _advertised(llm.calls[-1]) == {TERMINAL_TOOL}


@pytest.mark.asyncio
async def test_the_output_token_budget_takes_the_wind_down() -> None:
    rc = LoopConstants(
        model_context_window=4_096, run_max_output_tokens_budget=1
    )
    llm = _ScriptedLLM([{"tool": "Read", "args": {"x": "a"}}, {"text": "the answer"}])
    engine = _build_engine(rc=rc, llm=llm, tools=[_NamedTool("Read"), _FinalizeTool()])
    engine.total_usage.output_tokens = 999

    events = [evt async for evt in engine.run(_user())]

    causes = {
        e.payload.get("soft_stop_cause")
        for e in events
        if e.type is EventType.STATE_CHANGED
        and e.payload.get("reason") == "soft_stop_notified"
    }
    assert causes == {_soft_stop.CAUSE_OUTPUT_TOKEN_BUDGET}


@pytest.mark.asyncio
async def test_the_wall_clock_deadline_takes_the_wind_down() -> None:
    rc = LoopConstants(
        model_context_window=4_096,
        agent_max_seconds=1.0,
        agent_deadline_finalize_slack_seconds=0.0,
    )
    llm = _ScriptedLLM([{"text": "the answer"}])
    engine = _build_engine(rc=rc, llm=llm, tools=[_NamedTool("Read"), _FinalizeTool()])
    # Started long enough ago that the budget is spent at the first check.
    engine._run_started_monotonic = time.monotonic() - 60.0

    events = [evt async for evt in engine.run(_user())]

    causes = {
        e.payload.get("soft_stop_cause")
        for e in events
        if e.type is EventType.STATE_CHANGED
        and e.payload.get("reason") == "soft_stop_notified"
    }
    assert causes == {_soft_stop.CAUSE_DEADLINE}


@pytest.mark.asyncio
async def test_a_provider_failure_is_retried_before_anything_else() -> None:
    """One failed stream and a good one after it is a run that answers normally."""
    rc = LoopConstants(
        model_context_window=4_096, llm_transient_error_retry_backoff_base_seconds=0.0
    )
    recovered = _ScriptedLLM([{"text": "Here is the answer after the retry."}])
    llm = _FailingLLM(LLMProviderError("provider down"), recovered)
    engine = _build_engine(rc=rc, llm=llm, tools=[_NamedTool("Read"), _FinalizeTool()])

    events = [evt async for evt in engine.run(_user())]

    retries = [
        e
        for e in events
        if e.type is EventType.STATE_CHANGED
        and e.payload.get("reason") == "transient_llm_error_retry"
    ]
    assert len(retries) == 1
    assert not [
        e
        for e in events
        if e.type is EventType.STATE_CHANGED
        and e.payload.get("reason") == "soft_stop_notified"
    ]
    assert engine.state is LoopState.COMPLETED


@pytest.mark.asyncio
async def test_a_provider_failure_takes_the_wind_down() -> None:
    """The upstream stopped answering; the evidence gathered so far has not.

    Terminating here throws away a run that may already have everything it
    needs to answer — which is what the incident this was written for did.
    The retries come first; the wind-down is what follows when they are spent.
    """
    rc = LoopConstants(model_context_window=4_096, llm_transient_error_retry_max_attempts=0)
    recovered = _ScriptedLLM([{"text": "Here is the answer despite the failure."}])
    llm = _FailingLLM(LLMProviderError("provider down"), recovered)
    engine = _build_engine(rc=rc, llm=llm, tools=[_NamedTool("Read"), _FinalizeTool()])

    events = [evt async for evt in engine.run(_user())]

    causes = {
        e.payload.get("soft_stop_cause")
        for e in events
        if e.type is EventType.STATE_CHANGED
        and e.payload.get("reason") == "soft_stop_notified"
    }
    assert causes == {_soft_stop.CAUSE_PROVIDER_ERROR}
    assert engine.state is LoopState.COMPLETED
    assert _final_stop(events).payload["has_final_answer"] is True


@pytest.mark.asyncio
async def test_a_provider_failure_the_wind_down_cannot_rescue_still_reports_it() -> None:
    """The original error is surfaced, not buried under a silent no-answer stop."""
    rc = LoopConstants(
        model_context_window=4_096,
        soft_stop_max_turns=1,
        llm_transient_error_retry_backoff_base_seconds=0.0,
    )

    class _AlwaysFails:
        def __init__(self) -> None:
            self.calls: list[LLMRequest] = []

        async def stream_with_tools(  # type: ignore[no-untyped-def]
            self, request: LLMRequest
        ) -> AsyncIterator[LLMStreamEvent]:
            self.calls.append(request)
            if False:  # pragma: no cover
                yield LLMStreamEvent(name="never", payload={})
            raise LLMProviderError("provider down")

        async def complete_structured(self, request, schema):  # type: ignore[no-untyped-def]
            raise LLMProviderError("provider down")

        def count_tokens(self, text, model=None) -> int:  # type: ignore[no-untyped-def]
            return max(1, len(text) // 4)

    engine = _build_engine(rc=rc, llm=_AlwaysFails(), tools=[_FinalizeTool()])

    events = [evt async for evt in engine.run(_user())]

    errors = [e for e in events if e.type is EventType.ERROR]
    assert errors
    assert errors[-1].payload["kind"] == "llm_provider_error"
    assert engine.state is LoopState.FAILED


@pytest.mark.asyncio
async def test_only_one_wind_down_runs_however_many_bounds_are_hit() -> None:
    """Two bounds reached in one run is one stop, not two.

    A second entry would re-notify, re-emit both state changes, and grant a
    second budget of turns — which is how "the run is closing" stops meaning
    anything.
    """
    rc = LoopConstants(
        model_context_window=4_096,
        leader_tool_call_soft_cap=1,
        max_turns_per_run=2,
        soft_stop_max_turns=2,
    )
    llm = _ScriptedLLM([{"tool": "Read", "args": {"x": "a"}}])
    engine = _build_engine(rc=rc, llm=llm, tools=[_NamedTool("Read"), _FinalizeTool()])

    events = [evt async for evt in engine.run(_user())]

    assert _reasons(events).count("soft_stop_notified") == 1
    assert _reasons(events).count("soft_stop_tools_withdrawn") == 1


# ----------------------------------------------------------------------
# The prose requirement is the existing gate, not a second one
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_prose_gate_still_refuses_a_terminal_call_with_no_answer() -> None:
    """Step three is the gate that already exists. The wind-down does not bypass it.

    A payload-only terminal call under a wind-down would end the run with the
    budget spent and nothing the user can read — the exact outcome the whole
    mechanism exists to prevent.
    """

    class _PayloadOnlyFinalize(_FinalizeTool):
        @property
        def definition(self) -> ToolDefinition:
            return ToolDefinition(
                name=TERMINAL_TOOL,
                description="end the run",
                parameters=ToolParameterSchema(
                    properties={"declared_deliverables": {"type": "array"}}
                ),
            )

    rc = LoopConstants(
        model_context_window=4_096,
        leader_tool_call_soft_cap=1,
        finalize_prose_gate_enabled=True,
    )
    finalize = _PayloadOnlyFinalize()
    llm = _ScriptedLLM(
        [
            {"tool": "Read", "args": {"x": "a"}},
            {"tool": TERMINAL_TOOL, "args": {"declared_deliverables": []}},
            {"text": "The answer, written out properly this time."},
        ]
    )
    engine = _build_engine(rc=rc, llm=llm, tools=[_NamedTool("Read"), finalize])

    async for _ in engine.run(_user()):
        pass

    # The payload-only terminal call was refused — the tool never ran — and the
    # model answered in prose on the turn it was given instead.
    assert finalize.calls == []
    assert engine._finalize_prose_gate_used is True
    assert engine.has_final_answer is True
    assert engine.state is LoopState.COMPLETED


# ----------------------------------------------------------------------
# The switch, and surviving a resume
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_switch_off_restores_the_bare_terminal() -> None:
    rc = LoopConstants(
        model_context_window=4_096,
        max_turns_per_run=1,
        soft_stop_enabled=False,
    )
    llm = _ScriptedLLM([{"tool": "Read", "args": {"x": "a"}}])
    engine = _build_engine(rc=rc, llm=llm, tools=[_NamedTool("Read"), _FinalizeTool()])

    events = [evt async for evt in engine.run(_user())]

    assert "soft_stop_notified" not in _reasons(events)
    assert _final_stop(events).payload["stop_reason"] == StopReason.max_turns.value
    assert engine.state is LoopState.FAILED


def test_a_wind_down_cannot_be_entered_twice() -> None:
    rc = LoopConstants(model_context_window=4_096)
    engine = _build_engine(
        rc=rc, llm=_ScriptedLLM([{"text": "x"}]), tools=[_FinalizeTool()]
    )

    first = _soft_stop.enter(engine, cause_name=_soft_stop.CAUSE_MAX_TURNS)
    second = _soft_stop.enter(engine, cause_name=_soft_stop.CAUSE_DEADLINE)

    assert len(first) == 2
    assert second == []
    assert _soft_stop.cause(engine) == _soft_stop.CAUSE_MAX_TURNS


def test_an_unknown_cause_is_refused() -> None:
    """The cause rides every event and the run row; a typo would be invisible."""
    rc = LoopConstants(model_context_window=4_096)
    engine = _build_engine(
        rc=rc, llm=_ScriptedLLM([{"text": "x"}]), tools=[_FinalizeTool()]
    )

    with pytest.raises(ValueError):
        _soft_stop.enter(engine, cause_name="ran_out_of_patience")


@pytest.mark.asyncio
async def test_a_resumed_run_stays_wound_down() -> None:
    """Otherwise the re-drive hands back every tool the stop just took away."""
    rc = LoopConstants(model_context_window=4_096)
    engine = _build_engine(
        rc=rc,
        llm=_ScriptedLLM([{"text": "x"}]),
        tools=[_NamedTool("Read"), _FinalizeTool()],
    )
    _soft_stop.enter(engine, cause_name=_soft_stop.CAUSE_MAX_TURNS)
    snapshot = engine.snapshot()

    resumed = _build_engine(
        rc=rc,
        llm=_ScriptedLLM([{"text": "x"}]),
        tools=[_NamedTool("Read"), _FinalizeTool()],
    )
    assert _soft_stop.is_armed(resumed) is False
    await resumed.resume_from_snapshot(snapshot)

    assert _soft_stop.tools_withdrawn(resumed) is True
    advertised = {
        d.name
        for d in resumed.tools.compute_effective_surface(
            tenant_id=resumed.config.tenant_id,
            policy=resumed.effective_tool_policy,
        )
    }
    assert advertised == {TERMINAL_TOOL}
