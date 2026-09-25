"""Tests for :func:`protocore.runtime.query.query` — pure async generator."""
from __future__ import annotations

import pytest

from protocore.contracts.tools import Tool
from protocore.contracts.types import Message, MessageRole, StopReason, TextBlock
from protocore.runtime.events import EventType, TurnEvent
from protocore.runtime.loop_state import LoopState


@pytest.mark.asyncio
async def test_simple_text_turn_emits_minimum_event_sequence(
    engine_factory, in_memory_runtime
) -> None:
    """Single turn, no tools — model emits text only → COMPLETED."""
    engine = engine_factory()
    in_memory_runtime["llm"].queue_response(
        text="Hello there!",
        stop_reason=StopReason.end_turn,
        input_tokens=10,
        output_tokens=3,
    )

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    events: list[TurnEvent] = []
    async for evt in engine.run(user_msg):
        events.append(evt)

    types = [e.type.value for e in events]
    assert "message_start" in types
    assert "message_stop" in types
    # Stop reason recorded
    stop_evt = next(e for e in events if e.type is EventType.MESSAGE_STOP)
    assert stop_evt.payload["stop_reason"] == "end_turn"
    assert engine.state is LoopState.COMPLETED


@pytest.mark.asyncio
async def test_root_leader_request_observability_has_no_agent_sentinel(
    engine_factory, in_memory_runtime
) -> None:
    engine = engine_factory(run_id="run-root")
    in_memory_runtime["llm"].queue_response(text="ok")

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    async for _ in engine.run(user_msg):
        pass

    request = in_memory_runtime["llm"].calls[0]
    assert request.observability is not None
    assert request.observability.agent_id is None
    assert request.observability.call_category == "agent_call"


@pytest.mark.asyncio
async def test_empty_response_terminates_via_guard(
    engine_factory, in_memory_runtime
) -> None:
    """No queued response — the model returns silence on every turn.

    The empty-completion guard (default-on) grants a bounded re-drive and then,
    since the fixture keeps returning silence, terminates the run FAILED with a
    ``no_answer_empty_completion`` reason instead of sealing a silent empty
    COMPLETED. The async generator still terminates cleanly, emitting
    ``message_start`` / ``message_stop``.
    """
    engine = engine_factory()
    # No queued responses → mock returns end_turn with empty content, forever.

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    events: list[TurnEvent] = []
    async for evt in engine.run(user_msg):
        events.append(evt)

    types = [e.type.value for e in events]
    assert "message_start" in types
    assert "message_stop" in types
    assert engine.state is LoopState.FAILED
    assert any(
        e.type is EventType.ERROR
        and e.payload.get("kind") == "no_answer_empty_completion"
        for e in events
    )


@pytest.mark.asyncio
async def test_stop_before_start_yields_cancelled(
    engine_factory, in_memory_runtime
) -> None:
    """If engine.stop() called before first turn, message_stop(cancelled) emitted."""
    engine = engine_factory()
    in_memory_runtime["llm"].queue_response(text="should not be seen")
    engine.stop()  # request stop before run() starts

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    events = [evt async for evt in engine.run(user_msg)]

    stop_evts = [e for e in events if e.type is EventType.MESSAGE_STOP]
    assert len(stop_evts) == 1
    assert stop_evts[0].payload["stop_reason"] == "cancelled"
    assert engine.state is LoopState.CANCELLED


@pytest.mark.asyncio
async def test_state_changed_event_emitted_on_transition(
    engine_factory, in_memory_runtime
) -> None:
    """PENDING → RUNNING transition surfaces a state_changed event."""
    engine = engine_factory()
    in_memory_runtime["llm"].queue_response(text="ok")

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    events = [evt async for evt in engine.run(user_msg)]

    state_events = [e for e in events if e.type is EventType.STATE_CHANGED]
    # First state_changed is pending → running.
    assert any(
        e.payload.get("from") == "pending" and e.payload.get("to") == "running"
        for e in state_events
    )


@pytest.mark.asyncio
async def test_history_appends_assistant_message(
    engine_factory, in_memory_runtime
) -> None:
    engine = engine_factory()
    in_memory_runtime["llm"].queue_response(text="Reply text")

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    async for _ in engine.run(user_msg):
        pass

    # history starts with user msg, then assistant
    assert engine.history[0].role is MessageRole.user
    assert any(m.role is MessageRole.assistant for m in engine.history)


@pytest.mark.asyncio
async def test_content_block_events_emitted_in_order(
    engine_factory, in_memory_runtime
) -> None:
    """For a text-only turn the block_start/delta/stop ordering holds."""
    engine = engine_factory()
    in_memory_runtime["llm"].queue_response(text="streaming text here")

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    events = [evt async for evt in engine.run(user_msg)]
    types = [e.type for e in events]

    # message_start before any content_block; content_block_start before
    # content_block_delta; content_block_stop appears before message_stop.
    assert types.index(EventType.MESSAGE_START) < types.index(EventType.CONTENT_BLOCK_DELTA)
    assert types.index(EventType.CONTENT_BLOCK_DELTA) < types.index(EventType.MESSAGE_STOP)


@pytest.mark.asyncio
async def test_persists_snapshot_on_terminal(
    engine_factory, in_memory_runtime
) -> None:
    """Engine emits state_snapshot durable Event at terminal."""
    engine = engine_factory()
    in_memory_runtime["llm"].queue_response(text="ok")

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    async for _ in engine.run(user_msg):
        pass

    # InMemoryEventStream stores under (tenant, run_id)
    stream = in_memory_runtime["events"].stream_for(
        engine.config.tenant_id, engine.config.run_id
    )
    snapshot_events = [e for e in stream if e.name == "state_snapshot"]
    assert len(snapshot_events) >= 1
    # snapshot payload should carry tenant_id
    assert snapshot_events[-1].payload.get("tenant_id") == engine.config.tenant_id


class _RecordingTool(Tool):
    """Tool stub that records its invocation count + arguments.

    Returns the recorded count in the result so the test can verify
    dispatch ordering.
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self._next_error: Exception | None = None

    def fail_next(self, exc: Exception) -> None:
        self._next_error = exc

    @property
    def name(self) -> str:
        return "Recording"

    @property
    def definition(self):  # type: ignore[no-untyped-def]
        from protocore.contracts.types import ToolDefinition, ToolParameterSchema
        return ToolDefinition(
            name="Recording",
            description="Recording tool",
            parameters=ToolParameterSchema(
                properties={"v": {"type": "string"}},
            ),
        )

    async def invoke(self, context, arguments):  # type: ignore[no-untyped-def]
        from protocore.contracts.types import ToolResult
        self.calls.append(dict(arguments))
        if self._next_error is not None:
            err = self._next_error
            self._next_error = None
            raise err
        return ToolResult(
            tool_call_id="",
            content=f"call-{len(self.calls)}",
            is_error=False,
        )


class _NoopTool(Tool):
    """Tool stub used by the recursion-bound regression test.

    Returns an empty :class:`ToolResult` and does not raise — keeps the
    loop iterating against the scripted endless-tool-call LLM mock.
    """

    @property
    def name(self) -> str:
        return "Noop"

    @property
    def definition(self):  # type: ignore[no-untyped-def]
        from protocore.contracts.types import ToolDefinition, ToolParameterSchema
        return ToolDefinition(
            name="Noop",
            description="Noop tool",
            parameters=ToolParameterSchema(),
        )

    async def invoke(self, context, arguments):  # type: ignore[no-untyped-def]
        from protocore.contracts.types import ToolResult
        return ToolResult(tool_call_id="", content="ok", is_error=False)


class _TerminalAnswerTool(Tool):
    def __init__(self) -> None:
        self.calls: list[dict] = []

    @property
    def name(self) -> str:
        return "pcm_answer"

    @property
    def definition(self):  # type: ignore[no-untyped-def]
        from protocore.contracts.types import ToolDefinition, ToolParameterSchema

        return ToolDefinition(
            name="pcm_answer",
            description="Submit the final answer",
            parameters=ToolParameterSchema(
                properties={
                    "message": {"type": "string"},
                    "outcome": {"type": "string"},
                    "refs": {"type": "array"},
                },
                required=["message", "outcome"],
            ),
        )

    async def invoke(self, context, arguments):  # type: ignore[no-untyped-def]
        from protocore.contracts.types import TERMINAL_TOOL_METADATA_KEY, ToolResult

        self.calls.append(dict(arguments))
        return ToolResult(
            tool_call_id="",
            content="submitted",
            is_error=False,
            metadata={TERMINAL_TOOL_METADATA_KEY: True},
        )


@pytest.mark.asyncio
async def test_endless_tool_calls_terminate_at_max_turns(
    engine_factory, in_memory_runtime
) -> None:
    """If the LLM emits an endless tool_use stream the loop MUST terminate.

    Regression: original implementation recursed on every tool dispatch,
    so an endless tool stream blew the Python recursion limit before
    ``max_turns_per_run`` had a chance to fire. The loop now caps at
    ``max_turns_per_run`` assistant messages and emits
    ``stop_reason=max_turns`` cleanly.
    """
    from protocore.contracts.runtime_constants import LoopConstants

    rc = LoopConstants(model_context_window=4_096, max_turns_per_run=3)
    engine = engine_factory(rc=rc)
    in_memory_runtime["tools"].register(_NoopTool())
    in_memory_runtime["llm"].set_default_tool_call(
        tool_name="Noop",
        tool_input={"x": 1},
    )

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="loop")])
    events: list[TurnEvent] = []
    async for evt in engine.run(user_msg):
        events.append(evt)
        # Hard safety to prevent test hangs if the bound regresses.
        if len(events) > 10_000:
            pytest.fail("loop did not terminate within 10000 events")

    stop_evts = [e for e in events if e.type is EventType.MESSAGE_STOP]
    # The turn cap is reached, which starts the wind-down; the wind-down is
    # given its turns, the model spends them calling the same tool, and the run
    # closes on ``soft_stop`` rather than ``end_turn`` / ``tool_use``.
    assert stop_evts[-1].payload["stop_reason"] == "soft_stop"
    # budget exhaustion is a resource-exhaustion stop, NOT a
    # successful completion. The engine reaches the FAILED (failure-class)
    # terminal so downstream success/failure accounting does not score a
    # budget-exhausted run green.
    assert engine.state is LoopState.FAILED


@pytest.mark.asyncio
async def test_max_turns_transitions_to_failed_before_message_stop_yield(
    engine_factory, in_memory_runtime
) -> None:
    """The FAILED transition MUST be observable AT the max_turns stop yield.

 Regression (follow-up): the max-turns exit yielded
 ``MESSAGE_STOP(stop_reason=max_turns)`` and only then called
 ``engine.transition_to(LoopState.FAILED)``. ``query`` is consumed
 directly by the host executor; while the consumer processes the
 yielded event the generator is suspended, so the executor's terminal
 mirror (``if engine.state is LoopState.FAILED: terminal_event_kind =
 "error"``) read ``RUNNING`` — the late transition only ran on the next
 pull, which is ``StopAsyncIteration``. Every budget-exhausted run was
 then finalised as ``completed``. The transition must land BEFORE the
 yield (the contract every other FAILED/CANCELLED site follows), and a
 ``state_changed(to=failed)`` envelope must be on the wire.
 """
    from protocore.contracts.runtime_constants import LoopConstants

    rc = LoopConstants(model_context_window=4_096, max_turns_per_run=2)
    engine = engine_factory(rc=rc)
    in_memory_runtime["tools"].register(_NoopTool())
    in_memory_runtime["llm"].set_default_tool_call(
        tool_name="Noop",
        tool_input={"x": 1},
    )

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="loop")])
    state_at_max_turns_stop: LoopState | None = None
    events: list[TurnEvent] = []
    async for evt in engine.run(user_msg):
        events.append(evt)
        if (
            evt.type is EventType.MESSAGE_STOP
            and evt.payload.get("stop_reason") in ("max_turns", "soft_stop")
        ):
            # Mirror the executor: read engine.state while the generator is
            # suspended at this exact yield.
            state_at_max_turns_stop = engine.state
        if len(events) > 10_000:
            pytest.fail("loop did not terminate within 10000 events")

    assert state_at_max_turns_stop is LoopState.FAILED
    # The failure-class terminal surfaces a state_changed(failed) envelope
    # like every other FAILED/CANCELLED site.
    assert any(
        e.type is EventType.STATE_CHANGED
        and e.payload.get("to") == "failed"
        and e.payload.get("reason") in ("max_turns_exhausted", "soft_stop_exhausted")
        for e in events
    )


@pytest.mark.asyncio
async def test_terminal_nudge_recovers_plain_text_final(
    engine_factory, in_memory_runtime
) -> None:
    from protocore.contracts.llm import LLMStreamEvent
    from protocore.contracts.runtime_constants import LoopConstants

    #  — the prose-gate stays at its DEFAULT: ``pcm_answer``
    # is a MESSAGE-CARRYING terminal (schema declares ``message``), so the
    # schema-conditioned gate exempts it automatically (no RC override needed).
    rc = LoopConstants(
        model_context_window=4_096,
        max_turns_per_run=1,
        terminal_tool_nudge_enabled=True,
    )
    engine = engine_factory(rc=rc, expected_terminal_tool="pcm_answer")
    tool = _TerminalAnswerTool()
    in_memory_runtime["tools"].register(tool)
    in_memory_runtime["llm"]._scripted_streams.append(
        [
            LLMStreamEvent(name="message_start", payload={}),
            LLMStreamEvent(name="content_block_start", payload={"kind": "text"}),
            LLMStreamEvent(
                name="content_block_delta",
                payload={"text": "14", "kind": "text"},
            ),
            LLMStreamEvent(name="content_block_stop", payload={}),
            LLMStreamEvent(name="message_stop", payload={"stop_reason": "end_turn"}),
        ]
    )
    in_memory_runtime["llm"].queue_tool_call_response(
        tool_call_id="toolu_answer",
        tool_name="pcm_answer",
        tool_input={"message": "14", "outcome": "OUTCOME_OK", "refs": []},
    )

    user_msg = Message(
        role=MessageRole.user,
        content_blocks=[TextBlock(text="answer only")],
    )
    events = [evt async for evt in engine.run(user_msg)]

    reasons = [
        evt.payload.get("reason")
        for evt in events
        if evt.type is EventType.STATE_CHANGED
    ]
    # The plain-text answer is followed by a request that forces the terminal
    # tool rather than one that asks for it in words.
    assert "terminal_tool_forced" in reasons
    assert "terminal_tool_nudge" not in reasons
    calls = in_memory_runtime["llm"].calls
    assert len(calls) == 2
    assert calls[1].extra.get("forced_tool_choice") == "pcm_answer"
    assert tool.calls == [{"message": "14", "outcome": "OUTCOME_OK", "refs": []}]
    assert engine.state is LoopState.COMPLETED


@pytest.mark.asyncio
async def test_terminal_nudge_is_disabled_without_expected_terminal_tool(
    engine_factory, in_memory_runtime
) -> None:
    from protocore.contracts.runtime_constants import LoopConstants

    # No ``expected_terminal_tool`` configured -> the nudge never fires.
    rc = LoopConstants(
        model_context_window=4_096, terminal_tool_nudge_enabled=True
    )
    engine = engine_factory(rc=rc, expected_terminal_tool=None)
    in_memory_runtime["tools"].register(_TerminalAnswerTool())
    in_memory_runtime["llm"].queue_response(text="14")

    user_msg = Message(
        role=MessageRole.user,
        content_blocks=[TextBlock(text="answer only")],
    )
    events = [evt async for evt in engine.run(user_msg)]

    reasons = [
        evt.payload.get("reason")
        for evt in events
        if evt.type is EventType.STATE_CHANGED
    ]
    assert "terminal_tool_nudge" not in reasons
    assert len(in_memory_runtime["llm"].calls) == 1
    assert engine.state is LoopState.COMPLETED


@pytest.mark.asyncio
async def test_the_turn_cap_gives_the_model_room_to_finish(
    engine_factory, in_memory_runtime
) -> None:
    """Hitting the cap starts the wind-down, which grants turns to answer in.

    Without that, a run that reached its turn budget mid-work ended with
    whatever it had produced by then — usually nothing the user could read,
    despite the evidence being in hand.
    """
    from protocore.contracts.runtime_constants import LoopConstants

    # prose-gate at DEFAULT: ``pcm_answer`` is a MESSAGE-CARRYING terminal
    # (schema declares ``message``) ⟹ auto-exempt.
    rc = LoopConstants(model_context_window=4_096, max_turns_per_run=1)
    engine = engine_factory(rc=rc, expected_terminal_tool="pcm_answer")
    in_memory_runtime["tools"].register(_NoopTool())
    tool = _TerminalAnswerTool()
    in_memory_runtime["tools"].register(tool)
    in_memory_runtime["llm"].queue_tool_call_response(
        tool_call_id="toolu_noop",
        tool_name="Noop",
        tool_input={},
    )
    in_memory_runtime["llm"].queue_tool_call_response(
        tool_call_id="toolu_answer",
        tool_name="pcm_answer",
        tool_input={"message": "done", "outcome": "OUTCOME_OK", "refs": []},
    )

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])
    events = [evt async for evt in engine.run(user_msg)]

    reasons = [
        evt.payload.get("reason")
        for evt in events
        if evt.type is EventType.STATE_CHANGED
    ]
    assert "soft_stop_notified" in reasons
    assert "soft_stop_tools_withdrawn" in reasons
    assert len(in_memory_runtime["llm"].calls) == 2
    assert tool.calls == [{"message": "done", "outcome": "OUTCOME_OK", "refs": []}]
    assert engine.state is LoopState.COMPLETED


# ---------------------------------------------------------------------------
# Terminal-only finalisation guard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_terminal_only_latch_persists_across_snapshot_resume(
    engine_factory,
) -> None:
    """Regression — the terminal-only latch is per-engine memory. If an
    executor pod restarts mid-run and resumes from snapshot, the new
    engine MUST still reject a non-terminal dispatch instead of silently
    re-allowing tool use after the nudge already fired.

    Drives the latch via :func:`_append_terminal_tool_nudge` on the
    original engine, snapshots it, instantiates a fresh engine via
    ``engine_factory`` (per-pod-restart analogue), ``resume_from_snapshot``s,
    and checks :func:`_terminal_only_blocks` still returns True for a
    non-terminal tool call. The wind-down is what makes terminal-only strict,
    so it is started here too — and it has to survive the resume for the same
    reason the latch does."""
    from protocore.contracts.runtime_constants import LoopConstants
    from protocore.contracts.types import ToolCall
    from protocore.runtime import soft_stop as _soft_stop
    from protocore.runtime.query import (
        _append_terminal_tool_nudge,
        _terminal_only_blocks,
    )

    rc = LoopConstants(
        model_context_window=4_096,
        max_turns_per_run=1,
        terminal_tool_nudge_enabled=True,
    )
    original = engine_factory(rc=rc, expected_terminal_tool="pcm_answer")
    _append_terminal_tool_nudge(original)
    assert original._terminal_only_active is True
    _soft_stop.enter(original, cause_name=_soft_stop.CAUSE_DEADLINE)

    snapshot = original.snapshot()
    assert snapshot.get("terminal_only_active") is True
    assert snapshot.get("soft_stop_cause") == _soft_stop.CAUSE_DEADLINE

    # Fresh engine instance — analogue of an executor pod that took over
    # the run after a restart. ``resume_from_snapshot`` MUST rehydrate
    # the latch so a non-terminal dispatch is still blocked.
    resumed = engine_factory(rc=rc, expected_terminal_tool="pcm_answer")
    assert resumed._terminal_only_active is False
    await resumed.resume_from_snapshot(snapshot)
    assert resumed._terminal_only_active is True

    non_terminal = ToolCall(name="Noop", arguments={})
    assert _terminal_only_blocks(resumed, non_terminal) is True
    terminal = ToolCall(name="pcm_answer", arguments={})
    assert _terminal_only_blocks(resumed, terminal) is False

    # Sanity: a snapshot with no latch field present must resume with the
    # latch defaulting to False.
    legacy_snapshot = dict(snapshot)
    legacy_snapshot.pop("terminal_only_active", None)
    legacy_resumed = engine_factory(rc=rc, expected_terminal_tool="pcm_answer")
    await legacy_resumed.resume_from_snapshot(legacy_snapshot)
    assert legacy_resumed._terminal_only_active is False


@pytest.mark.asyncio
async def test_compaction_uses_rc_for_summary_caps(
    engine_factory, in_memory_runtime
) -> None:
    """Compaction Tier 2 LLMRequest MUST source caps from RC, not magic numbers.

 Regression: ``compaction.py`` hardcoded ``max_tokens=512`` and
 ``temperature=0.2``. Both MUST be sourced from RC fields (no inline magic
 numbers): the request cap from the summary budget, which is capped by
 ``compaction_summary_max_output_tokens``, and the temperature from
 ``compaction_summary_temperature``. This test inspects the
 ``InMemoryLLMProvider.calls`` view to verify the Tier 2 path picked them up.
 """
    from protocore.contracts.runtime_constants import LoopConstants
    from protocore.runtime.context.carrier import request_max_tokens

    rc = LoopConstants(
        model_context_window=64,
        request_context_safety_tokens=0,
        compaction_trigger_ratio=0.5,
        compaction_keep_recent_turns=1,
        # Custom caps — values different from defaults so we can assert.
        compaction_summary_max_output_tokens=64,
        compaction_summary_min_output_tokens=32,
        compaction_summary_temperature=0.5,
    )

    engine = engine_factory(rc=rc)
    in_memory_runtime["llm"].queue_response(text="ok")  # primary turn
    big_text = "z" * 1024
    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text=big_text)])
    async for _ in engine.run(user_msg):
        pass

    # Verify the compaction LLM saw a request with the RC-sourced caps.
    summariser_calls = [
        req for req in in_memory_runtime["llm"].calls
        if req.messages and req.messages[0].role is MessageRole.system
    ]
    # Tier 2 may not fire if Tier 1 freed enough; gate the assertion on
    # whether the summariser was invoked.
    for req in summariser_calls:
        assert req.max_tokens <= request_max_tokens(rc.compaction_summary_max_output_tokens)
        assert req.temperature == 0.5


@pytest.mark.asyncio
async def test_compaction_started_payload_uses_correct_threshold(
    engine_factory, in_memory_runtime
) -> None:
    """COMPACTION_STARTED payload MUST surface the trigger boundary, not the window.

 Regression: previously ``trigger_threshold`` was set to
 ``rc.model_context_window`` (e.g. 49152) which is the upper bound,
 not the trigger. Telemetry dashboards expected the actual trigger
 threshold (= window * compaction_trigger_ratio) per
 ``streaming-events.md``. Additionally ``tokens_before`` was
 missing entirely.
 """
    from protocore.contracts.runtime_constants import LoopConstants

    rc = LoopConstants(
        model_context_window=64,
        request_context_safety_tokens=0,
        compaction_trigger_ratio=0.5,
        compaction_keep_recent_turns=1,
    )
    expected_threshold = int(rc.model_context_window * rc.compaction_trigger_ratio)

    engine = engine_factory(rc=rc)
    in_memory_runtime["llm"].queue_response(text="ok")
    big_text = "y" * 1024
    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text=big_text)])

    events: list[TurnEvent] = []
    async for evt in engine.run(user_msg):
        events.append(evt)

    started = [e for e in events if e.type is EventType.COMPACTION_STARTED]
    assert started, "expected COMPACTION_STARTED event"
    payload = started[0].payload
    # trigger_threshold = compaction_trigger_tokens, NOT model_context_window.
    assert payload["trigger_threshold"] == expected_threshold
    assert payload["trigger_threshold"] != rc.model_context_window
    # tokens_before reflects actual current token count > threshold.
    assert payload["tokens_before"] >= expected_threshold


@pytest.mark.asyncio
async def test_compaction_completion_persists_snapshot(
    engine_factory, in_memory_runtime
) -> None:
    """A successful compaction MUST persist a snapshot afterwards.

 5 a snapshot
 fires after every (a) tool_result append, (b) compaction completion,
 (c) message_stop. Without (b) an executor crash between compaction
 and the next LLM call would lose the freed-up history.
 """
    from protocore.contracts.runtime_constants import LoopConstants

    # Tiny window so a single long message triggers compaction.
    rc = LoopConstants(
        model_context_window=64,
        request_context_safety_tokens=0,
        compaction_trigger_ratio=0.5,  # 32 tokens trigger
        compaction_keep_recent_turns=1,
        compaction_failed_max_retries=2,
    )
    engine = engine_factory(rc=rc)
    in_memory_runtime["llm"].queue_response(text="ok")

    # Long user message — well above 32-token compaction trigger via
    # the 4-chars-per-token Latin baseline.
    big_text = "x" * 1024
    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text=big_text)])

    events: list[TurnEvent] = []
    async for evt in engine.run(user_msg):
        events.append(evt)

    # Verify compaction COMPLETED was emitted.
    completed = [e for e in events if e.type is EventType.COMPACTION_COMPLETED]
    assert completed, "expected COMPACTION_COMPLETED event"

    # Verify multiple state_snapshot durable events were emitted —
    # specifically at least one BEFORE the final terminal one (i.e.
    # the post-compaction snapshot).
    stream = in_memory_runtime["events"].stream_for(
        engine.config.tenant_id, engine.config.run_id
    )
    snapshot_events = [e for e in stream if e.name == "state_snapshot"]
    # Initial (engine.run pre-query snapshot) + post-compaction +
    # final (finally-block) = at least 3.
    assert len(snapshot_events) >= 3, (
        f"expected ≥3 state_snapshot events (run-start + post-compaction + "
        f"finally), got {len(snapshot_events)}"
    )


@pytest.mark.asyncio
async def test_max_turns_stop_reason_uses_enum(
    engine_factory, in_memory_runtime
) -> None:
    """``max_turns`` MUST be a member of :class:`StopReason` (not bare str).

 Regression: ``query.py`` emitted ``stop_reason="max_turns"`` but the
 enum lacked the member; consumers parsing ``StopReason(value)`` then
 raised ``ValueError``. Per the streaming-events.md taxonomy
 (``stop_reason ∈ {end_turn, tool_use, max_tokens, max_turns,
 cancelled, error}``) the enum MUST list ``max_turns``.
 """
    # The enum carries the member.
    assert StopReason.max_turns.value == "max_turns"

    from protocore.contracts.runtime_constants import LoopConstants

    # The wind-down is off here on purpose: with it on the run closes on
    # ``soft_stop``, which is a different member and has its own coverage. This
    # is the raw exhaustion terminal, which is what a deployment with the
    # wind-down disabled still gets.
    rc = LoopConstants(
        model_context_window=4_096, max_turns_per_run=2, soft_stop_enabled=False
    )
    engine = engine_factory(rc=rc)
    in_memory_runtime["tools"].register(_NoopTool())
    in_memory_runtime["llm"].set_default_tool_call(
        tool_name="Noop",
        tool_input={"x": 1},
    )

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="loop")])
    events: list[TurnEvent] = []
    async for evt in engine.run(user_msg):
        events.append(evt)
        if len(events) > 10_000:
            pytest.fail("loop did not terminate")

    stop_evts = [e for e in events if e.type is EventType.MESSAGE_STOP]
    final_reason = stop_evts[-1].payload["stop_reason"]
    assert final_reason == StopReason.max_turns.value
    # Round-trip through the enum constructor.
    assert StopReason(final_reason) is StopReason.max_turns


@pytest.mark.asyncio
async def test_max_turns_exhaustion_is_failure_class_terminal(
    engine_factory, in_memory_runtime
) -> None:
    """budget exhaustion routes to a FAILURE-class terminal.

    Max-turns exhaustion surfaces as ``subtype:'error_max_turns', is_error:true``
    — a NON-success terminal, distinct from a genuine ``completed`` end_turn.
    Protocore previously collapsed the budget-exhaustion exit to
    ``LoopState.COMPLETED`` (the same success-class state a real end_turn uses),
    so the host finalisation / dashboards / eval rigs scored a green run where
    the model
    never finished. The exit MUST instead reach ``LoopState.FAILED`` while
    KEEPING ``stop_reason=max_turns``.

    The genuine end_turn and terminal-tool-completed sites are unaffected —
    those continue to reach COMPLETED (covered by
    ``test_simple_text_turn_emits_minimum_event_sequence`` and the nudge
    recovery tests).
    """
    from protocore.contracts.runtime_constants import LoopConstants

    # Wind-down off: this pins the RAW exhaustion terminal. The wind-down's own
    # exhaustion terminal is failure-class for the same reason and is covered
    # with the rest of the wind-down.
    rc = LoopConstants(
        model_context_window=4_096, max_turns_per_run=2, soft_stop_enabled=False
    )
    engine = engine_factory(rc=rc)
    in_memory_runtime["tools"].register(_NoopTool())
    in_memory_runtime["llm"].set_default_tool_call(
        tool_name="Noop",
        tool_input={"x": 1},
    )

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="loop")])
    events: list[TurnEvent] = []
    async for evt in engine.run(user_msg):
        events.append(evt)
        if len(events) > 10_000:
            pytest.fail("loop did not terminate")

    stops = [e for e in events if e.type is EventType.MESSAGE_STOP]
    # stop_reason stays max_turns — the distinction is preserved on the wire.
    assert stops[-1].payload["stop_reason"] == StopReason.max_turns.value
    # The state-machine terminal is failure-class, not success-class.
    assert engine.state is LoopState.FAILED
    assert engine.is_terminal
    assert engine.state is not LoopState.COMPLETED


@pytest.mark.asyncio
async def test_cancel_after_complete_is_noop(
    engine_factory, in_memory_runtime
) -> None:
    """``engine.stop_requested`` after a terminal-state inner loop MUST NOT raise.

    Regression: outer ``query()`` polled ``engine.stop_requested`` after
    ``_stream_one_assistant_message`` had already driven the engine to
    ``COMPLETED``. Terminal states have empty outgoing edges so the
    subsequent ``engine.transition_to(CANCELLED)`` would raise
    ``InvalidStateTransitionError`` and bubble as a 500.

    Scenario: caller requests stop mid-content. Inner stream breaks out,
    appends the assistant message, emits MESSAGE_STOP(end_turn),
    transitions to COMPLETED, then returns. Outer's ``async for`` body
    runs after the final yield with ``stop_requested == True`` AND
    ``engine.state == COMPLETED``. The guard MUST notice the terminal
    state and not attempt another transition.
    """
    engine = engine_factory()
    in_memory_runtime["llm"].queue_response(text="hello")

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    # Request stop AFTER the first content_block_delta arrives — this puts
    # ``stop_requested`` into effect while the inner stream is still
    # iterating. The inner will see the flag at its next delta-loop check,
    # break, then naturally drive to COMPLETED via the no-tool-calls path.
    events: list[TurnEvent] = []
    triggered = False
    async for evt in engine.run(user_msg):
        events.append(evt)
        if not triggered and evt.type is EventType.CONTENT_BLOCK_DELTA:
            engine.stop()
            triggered = True

    # MUST NOT raise InvalidStateTransitionError. The terminal-state
    # guard preserves the COMPLETED state — the late stop is a no-op.
    assert engine.state in {LoopState.COMPLETED, LoopState.CANCELLED}


@pytest.mark.asyncio
async def test_state_changed_payload_has_correct_from_field(
    engine_factory, in_memory_runtime
) -> None:
    """Every ``state_changed`` event MUST surface the actual prior state.

    Regression: prior implementation read ``engine.state`` after
    ``transition_to`` had already been applied, producing self-referential
    ``from == to`` payloads for transitions that aren't PENDING → RUNNING
    (e.g. RUNNING → CANCELLED via stop-before-start, RUNNING → FAILED via
    LLM error, COMPACTING → RUNNING via successful compaction).
    """
    engine = engine_factory()
    in_memory_runtime["llm"].queue_response(text="ok")
    engine.stop()  # request stop BEFORE run() — should hit the very first guard

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    events = [evt async for evt in engine.run(user_msg)]

    state_evts = [e for e in events if e.type is EventType.STATE_CHANGED]
    # Should have at least the PENDING → CANCELLED transition (the
    # "stop_before_start" path emits state_changed BEFORE the transition).
    assert state_evts, "expected at least one state_changed event"
    cancel_evt = next(
        (e for e in state_evts if e.payload.get("to") == "cancelled"),
        None,
    )
    assert cancel_evt is not None, "expected a state_changed → cancelled event"
    # The CRITICAL invariant: `from` must NOT equal `to`.
    assert cancel_evt.payload["from"] != cancel_evt.payload["to"]
    # And specifically: the prior state was PENDING.
    assert cancel_evt.payload["from"] == "pending"


# ----------------------------------------------------------------------
# LOW #11 — coverage tests for under-covered paths.
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_dispatch_flow(engine_factory, in_memory_runtime) -> None:
    """Full tool-dispatch sequence: start → input_delta → stop → result.

 Verifies the event ordering invariants from ``streaming-events.md``: every ``tool_use_start`` is followed by exactly one matching
 ``tool_use_stop`` AND a ``tool_result`` with the matching
 ``tool_call_id``. Also verifies the assistant message is appended to
 history with a ``ToolUseBlock`` carrying the structured arguments.
 """
    from protocore.contracts.types import ToolUseBlock

    engine = engine_factory()
    tool = _RecordingTool()
    in_memory_runtime["tools"].register(tool)
    in_memory_runtime["llm"].queue_tool_call_response(
        tool_call_id="toolu_abc",
        tool_name="Recording",
        tool_input={"v": "hello"},
        text_prefix="Let me call the tool.",
    )
    # After the tool result, a follow-up plain end_turn response.
    in_memory_runtime["llm"].queue_response(text="done")

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])
    events: list[TurnEvent] = []
    async for evt in engine.run(user_msg):
        events.append(evt)

    # Ordering: tool_use_start → tool_use_input_delta → tool_use_stop → tool_result.
    types = [e.type for e in events]
    starts = [i for i, t in enumerate(types) if t is EventType.TOOL_USE_START]
    inputs = [i for i, t in enumerate(types) if t is EventType.TOOL_USE_INPUT_DELTA]
    stops = [i for i, t in enumerate(types) if t is EventType.TOOL_USE_STOP]
    results = [i for i, t in enumerate(types) if t is EventType.TOOL_RESULT]
    assert starts and inputs and stops and results
    assert starts[0] < inputs[0] < stops[0] < results[0]

    # tool_call_id flows through every event.
    tool_use_start = events[starts[0]]
    assert tool_use_start.payload["tool_call_id"] == "toolu_abc"
    tool_result = events[results[0]]
    assert tool_result.payload["tool_call_id"] == "toolu_abc"

    # Tool actually invoked with structured args.
    assert tool.calls == [{"v": "hello"}]

    # History carries an assistant message with ToolUseBlock + a tool result.
    assistant_msgs = [m for m in engine.history if m.role is MessageRole.assistant]
    assert assistant_msgs
    tool_use_blocks = [
        b for b in assistant_msgs[0].content_blocks if isinstance(b, ToolUseBlock)
    ]
    assert tool_use_blocks
    assert tool_use_blocks[0].tool_call_id == "toolu_abc"

    # Final state is COMPLETED.
    assert engine.state is LoopState.COMPLETED


@pytest.mark.asyncio
async def test_compaction_mid_turn(engine_factory, in_memory_runtime) -> None:
    """Context fills past trigger → compaction runs → next LLM call proceeds.

    Verifies the full compaction wiring: state transitions
    (RUNNING → COMPACTING → RUNNING), compaction_started/completed
    events, the post-compaction snapshot, and that the subsequent LLM
    request proceeds normally to ``end_turn``.
    """
    from protocore.contracts.runtime_constants import LoopConstants

    rc = LoopConstants(
        model_context_window=4_096,
        compaction_trigger_ratio=0.5,
        compaction_keep_recent_turns=1,
    )
    engine = engine_factory(rc=rc)
    in_memory_runtime["llm"].queue_response(text='{"summary":"old answer"}')
    in_memory_runtime["llm"].queue_response(text="post-compaction reply")

    engine.history.append(
        Message(
            role=MessageRole.assistant,
            content_blocks=[TextBlock(text="a" * 10_000)],
        )
    )
    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="continue")])
    events: list[TurnEvent] = []
    async for evt in engine.run(user_msg):
        events.append(evt)

    # compaction_started → compaction_completed.
    started = [e for e in events if e.type is EventType.COMPACTION_STARTED]
    completed = [e for e in events if e.type is EventType.COMPACTION_COMPLETED]
    assert started and completed
    # state_changed events surface RUNNING → COMPACTING and back.
    state_evts = [e for e in events if e.type is EventType.STATE_CHANGED]
    transitions = [(e.payload["from"], e.payload["to"]) for e in state_evts]
    assert ("running", "compacting") in transitions
    assert ("compacting", "running") in transitions
    # The follow-up LLM call completed normally.
    final_stop = [e for e in events if e.type is EventType.MESSAGE_STOP][-1]
    assert final_stop.payload["stop_reason"] == "end_turn"
    assert engine.state is LoopState.COMPLETED


@pytest.mark.asyncio
async def test_recursion_depth_guarded(engine_factory, in_memory_runtime) -> None:
    """Endless tool calls MUST terminate at ``max_turns_per_run``.

    Duplicate coverage for ``test_endless_tool_calls_terminate_at_max_turns``
    using a slightly different RC (low cap) — explicitly verifies the
    LOW #11 description: "endless tool calls terminate at max_turns".
    """
    from protocore.contracts.runtime_constants import LoopConstants

    rc = LoopConstants(
        model_context_window=4_096, max_turns_per_run=4, soft_stop_enabled=False
    )
    engine = engine_factory(rc=rc)
    in_memory_runtime["tools"].register(_NoopTool())
    in_memory_runtime["llm"].set_default_tool_call(
        tool_name="Noop",
        tool_input={"x": 1},
    )

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="x")])
    events: list[TurnEvent] = []
    async for evt in engine.run(user_msg):
        events.append(evt)
        if len(events) > 10_000:
            pytest.fail("loop did not terminate")
    # Exactly one MESSAGE_STOP per assistant message + one final max_turns stop.
    stops = [e for e in events if e.type is EventType.MESSAGE_STOP]
    assert stops[-1].payload["stop_reason"] == StopReason.max_turns.value
    # max_turns exhaustion is failure-class, not COMPLETED.
    assert engine.state is LoopState.FAILED


@pytest.mark.asyncio
async def test_mid_stream_cancel(engine_factory, in_memory_runtime) -> None:
    """``engine.stop_requested`` mid-LLM-stream MUST drive a graceful cancel.

    Verifies the inner-stream stop-check branch in
    ``_stream_one_assistant_message`` (``if engine.stop_requested: break``)
    and the outer ``query()`` cancel-guard (which now respects the
    terminal-state predicate added by Fix HIGH #3).
    """
    engine = engine_factory()
    in_memory_runtime["llm"].queue_response(text="streaming text here")

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    events: list[TurnEvent] = []
    triggered = False
    async for evt in engine.run(user_msg):
        events.append(evt)
        if not triggered and evt.type is EventType.CONTENT_BLOCK_DELTA:
            engine.stop()
            triggered = True

    # MUST NOT raise InvalidStateTransitionError. Engine reaches a
    # terminal state (either CANCELLED or COMPLETED depending on race).
    assert engine.is_terminal


@pytest.mark.asyncio
async def test_stop_at_restream_top_finalizes_cancelled_not_end_turn(
    engine_factory, in_memory_runtime
) -> None:
    """an interrupt landing as the next stream opens MUST finalize
    as CANCELLED, never as a success-class end_turn.

    When stop is requested while a fresh assistant stream is opening, the
    inner stream breaks immediately and yields an empty result (no text, no
    tool_calls, no reasoning). The loop then fell through to the no-tool
    ``end_turn`` branch and transitioned to ``LoopState.COMPLETED`` — an empty
    model turn scored as a clean completion, indistinguishable downstream
    from a genuine answer. The reference fires its abort check as the FIRST
    post-stream action (``aborted_streaming``). The fix re-checks
    ``engine.stop_requested`` at the top of the no-tool end_turn branch and
    routes to the CANCELLED path (``stop_reason=cancelled``).

    This makes the previously racy ``test_mid_stream_cancel`` outcome
    deterministic for the stop-before-restream timing.
    """
    from collections.abc import AsyncIterator

    from protocore.contracts.llm import LLMRequest, LLMStreamEvent

    class _StopBeforeStreamLLM:
        """Requests stop, then opens a stream that yields zero usable deltas.

        The engine reference is injected after construction so the provider
        can flip ``engine.stop()`` exactly as the stream begins — modelling an
        operator interrupt that lands at the top of a re-stream.
        """

        def __init__(self) -> None:
            self.engine = None
            self._calls: list[LLMRequest] = []

        @property
        def calls(self) -> list[LLMRequest]:
            return self._calls

        async def stream_with_tools(
            self, request: LLMRequest
        ) -> AsyncIterator[LLMStreamEvent]:
            self._calls.append(request)
            # Interrupt observed exactly as the stream opens.
            assert self.engine is not None
            self.engine.stop()
            # message_start arrives, but the per-delta stop-check breaks the
            # inner loop before any content/finish → empty result.
            yield LLMStreamEvent(name="message_start", payload={})
            yield LLMStreamEvent(name="content_block_start", payload={"kind": "text"})
            yield LLMStreamEvent(
                name="content_block_delta", payload={"text": "ignored"}
            )
            yield LLMStreamEvent(
                name="message_stop", payload={"stop_reason": "end_turn"}
            )

        async def complete_structured(self, request, schema):  # type: ignore[no-untyped-def]
            raise RuntimeError("unused")

        def count_tokens(self, text, model=None) -> int:  # type: ignore[no-untyped-def]
            return max(1, len(text) // 4)

    engine = engine_factory()
    stop_llm = _StopBeforeStreamLLM()
    stop_llm.engine = engine
    engine.llm = stop_llm  # type: ignore[assignment]

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    events: list[TurnEvent] = []
    async for evt in engine.run(user_msg):
        events.append(evt)
        if len(events) > 10_000:
            pytest.fail("loop did not terminate")

    # The terminal state is the cancellation state, NOT the success state.
    assert engine.state is LoopState.CANCELLED
    assert engine.state is not LoopState.COMPLETED
    stops = [e for e in events if e.type is EventType.MESSAGE_STOP]
    assert stops, "expected a terminal MESSAGE_STOP"
    # The final stop_reason is cancelled, never end_turn.
    assert stops[-1].payload["stop_reason"] == StopReason.cancelled.value
    assert stops[-1].payload["stop_reason"] != "end_turn"


@pytest.mark.asyncio
async def test_llm_exception_handler(engine_factory, in_memory_runtime) -> None:
    """An untyped exception out of the stream drives RUNNING → FAILED + error.

    The provider adapter raises ``LLMProviderError`` and friends for upstream
    failures; a bare ``RuntimeError`` escaping the stream is a bug on THIS side,
    so the error event carries ``internal_error`` rather than claiming the
    upstream failed. ``stop_reason`` is unaffected.
    """
    from collections.abc import AsyncIterator

    from protocore.contracts.llm import LLMRequest, LLMStreamEvent

    class _ExplodingLLM:
        """LLM mock whose ``stream_with_tools`` raises immediately."""

        @property
        def calls(self):  # type: ignore[no-untyped-def]
            return []

        async def stream_with_tools(  # type: ignore[no-untyped-def]
            self, request: LLMRequest
        ) -> AsyncIterator[LLMStreamEvent]:
            if False:  # pragma: no cover — pure marker for generator protocol
                yield LLMStreamEvent(name="never", payload={})
            raise RuntimeError("upstream provider exploded")

        async def complete_structured(self, request, schema):  # type: ignore[no-untyped-def]
            raise RuntimeError("unused")

        def count_tokens(self, text, model=None) -> int:  # type: ignore[no-untyped-def]
            return max(1, len(text) // 4)

    engine = engine_factory()
    # Swap the LLM provider out for the exploder.
    engine.llm = _ExplodingLLM()  # type: ignore[assignment]

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    events: list[TurnEvent] = []
    async for evt in engine.run(user_msg):
        events.append(evt)

    error_evts = [e for e in events if e.type is EventType.ERROR]
    assert error_evts, (
        f"expected error event; types seen: "
        f"{[e.type.value for e in events]}; state: {engine.state}"
    )
    assert error_evts[0].payload["kind"] == "internal_error"
    final_stop = [e for e in events if e.type is EventType.MESSAGE_STOP][-1]
    assert final_stop.payload["stop_reason"] == StopReason.error.value
    assert engine.state is LoopState.FAILED
