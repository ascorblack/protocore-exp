"""Outbound tool_use/tool_result pairing repair.

Two complementary defenses:

* Outbound repair runs UNCONDITIONALLY on the outbound message list
 immediately before :class:`LLMRequest` assembly: it forward-fills a
 synthetic ``is_error`` tool_result for any orphaned ``tool_use``,
 reverse-strips orphaned ``tool_result`` blocks, and dedupes duplicate ids.
 The wire always satisfies tool_use<->tool_result pairing regardless of how
 the orphan arose (compaction, resume-from-partial-batch, max_tokens
 truncation).

* Missing-tool-result synthesis synthesises a tool-role ``is_error``
 tool_result on the cancel / LLM-error teardown for any already-emitted
 ``tool_use`` with no matching result, so a persisted/resumed history is
 always pairing-valid even if the dispatch loop never ran the call.
"""
from __future__ import annotations

from typing import Any

import pytest

from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.tools import Tool
from protocore.contracts.types import (
    COMPACTION_SUMMARY_METADATA_KEY,
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
from protocore.runtime.query import (
    _normalize_outbound_system_messages,
    _prepend_system_sections,
    _repair_outbound_tool_pairing,
    _synthesize_missing_tool_results,
)


def _tool_use_ids(messages: list[Message]) -> list[str]:
    return [
        b.tool_call_id
        for m in messages
        for b in m.content_blocks
        if isinstance(b, ToolUseBlock)
    ]


def _tool_result_ids(messages: list[Message]) -> list[str]:
    return [
        b.tool_call_id
        for m in messages
        for b in m.content_blocks
        if isinstance(b, ToolResultBlock)
    ]


# ---------------------------------------------------------------------------
# pure pairing-repair pass
# ---------------------------------------------------------------------------


def test_forward_fill_synthetic_for_orphaned_tool_use() -> None:
    """An assistant tool_use with no following tool_result gets a synthetic one."""
    placeholder = "PLACEHOLDER-MISSING"
    messages = [
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")]),
        Message(
            role=MessageRole.assistant,
            content_blocks=[
                ToolUseBlock(tool_call_id="call_1", name="Read", arguments_json="{}")
            ],
        ),
        # No tool_result for call_1 — e.g. compaction dropped it.
    ]
    repaired = _repair_outbound_tool_pairing(messages, placeholder=placeholder)

    results = [
        b
        for m in repaired
        for b in m.content_blocks
        if isinstance(b, ToolResultBlock)
    ]
    assert len(results) == 1
    assert results[0].tool_call_id == "call_1"
    assert results[0].is_error is True
    assert results[0].content == placeholder
    # The synthetic result lands AFTER its tool_use (forward direction).
    assert _tool_use_ids(repaired) == ["call_1"]
    assert _tool_result_ids(repaired) == ["call_1"]


def test_forward_fill_inserted_immediately_after_assistant() -> None:
    """The synthetic tool_result is positioned right after the orphan's turn."""
    messages = [
        Message(
            role=MessageRole.assistant,
            content_blocks=[
                ToolUseBlock(tool_call_id="c1", name="Read", arguments_json="{}")
            ],
        ),
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="next")]),
    ]
    repaired = _repair_outbound_tool_pairing(messages, placeholder="x")
    # assistant(tool_use c1) -> tool(result c1) -> user(next)
    assert repaired[0].role is MessageRole.assistant
    assert repaired[1].role is MessageRole.tool
    assert isinstance(repaired[1].content_blocks[0], ToolResultBlock)
    assert repaired[1].content_blocks[0].tool_call_id == "c1"
    assert repaired[2].role is MessageRole.user


def test_reverse_strip_orphaned_tool_result() -> None:
    """A tool_result with no matching tool_use is removed entirely."""
    messages = [
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")]),
        # Orphaned tool_result — its tool_use was dropped by compaction.
        Message(
            role=MessageRole.tool,
            content_blocks=[
                ToolResultBlock(tool_call_id="ghost", content="stale", is_error=False)
            ],
        ),
        Message(
            role=MessageRole.assistant,
            content_blocks=[TextBlock(text="done")],
        ),
    ]
    repaired = _repair_outbound_tool_pairing(messages, placeholder="x")
    assert _tool_result_ids(repaired) == []
    # The empty tool message is dropped (no surviving blocks).
    assert all(m.role is not MessageRole.tool for m in repaired)
    # Non-tool messages are preserved.
    assert [m.role for m in repaired] == [MessageRole.user, MessageRole.assistant]


def test_dedupe_duplicate_tool_use_ids() -> None:
    """A duplicated tool_use id keeps only the first occurrence."""
    messages = [
        Message(
            role=MessageRole.assistant,
            content_blocks=[
                ToolUseBlock(tool_call_id="dup", name="Read", arguments_json="{}"),
                ToolUseBlock(tool_call_id="dup", name="Read", arguments_json="{}"),
            ],
        ),
        Message(
            role=MessageRole.tool,
            content_blocks=[
                ToolResultBlock(tool_call_id="dup", content="ok", is_error=False)
            ],
        ),
    ]
    repaired = _repair_outbound_tool_pairing(messages, placeholder="x")
    assert _tool_use_ids(repaired) == ["dup"]
    assert _tool_result_ids(repaired) == ["dup"]


def test_dedupe_duplicate_tool_result_ids() -> None:
    """A duplicated tool_result id keeps only the first occurrence."""
    messages = [
        Message(
            role=MessageRole.assistant,
            content_blocks=[
                ToolUseBlock(tool_call_id="c1", name="Read", arguments_json="{}")
            ],
        ),
        Message(
            role=MessageRole.tool,
            content_blocks=[
                ToolResultBlock(tool_call_id="c1", content="first", is_error=False)
            ],
        ),
        Message(
            role=MessageRole.tool,
            content_blocks=[
                ToolResultBlock(tool_call_id="c1", content="second", is_error=False)
            ],
        ),
    ]
    repaired = _repair_outbound_tool_pairing(messages, placeholder="x")
    assert _tool_result_ids(repaired) == ["c1"]
    # The first (not the duplicate) survives.
    surviving = next(
        b
        for m in repaired
        for b in m.content_blocks
        if isinstance(b, ToolResultBlock)
    )
    assert surviving.content == "first"


def test_out_of_order_tool_result_repositioned_after_tool_use() -> None:
    """A tool_result separated from its tool_use is moved to be adjacent.

    The Anthropic wire requires the tool_result to be the message
    immediately following the assistant ``tool_use`` turn. A result that
    drifted (e.g. a teardown appended it after an intervening user message)
    must be repositioned, not just left present.
    """
    messages = [
        Message(
            role=MessageRole.assistant,
            content_blocks=[
                ToolUseBlock(tool_call_id="c1", name="Read", arguments_json="{}")
            ],
        ),
        # Intervening user/recovery message between tool_use and its result.
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="interrupted")]),
        Message(
            role=MessageRole.tool,
            content_blocks=[
                ToolResultBlock(tool_call_id="c1", content="late", is_error=True)
            ],
        ),
    ]
    repaired = _repair_outbound_tool_pairing(messages, placeholder="x")
    # assistant(tool_use c1) -> tool(result c1) -> user(interrupted)
    assert [m.role for m in repaired] == [
        MessageRole.assistant,
        MessageRole.tool,
        MessageRole.user,
    ]
    assert isinstance(repaired[1].content_blocks[0], ToolResultBlock)
    assert repaired[1].content_blocks[0].tool_call_id == "c1"
    # The real result content is preserved (not replaced by a synthetic).
    assert repaired[1].content_blocks[0].content == "late"
    # Exactly one result for c1.
    assert _tool_result_ids(repaired) == ["c1"]


def test_well_paired_history_is_unchanged() -> None:
    """A history that already satisfies pairing is returned structurally intact."""
    messages = [
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")]),
        Message(
            role=MessageRole.assistant,
            content_blocks=[
                ToolUseBlock(tool_call_id="c1", name="Read", arguments_json="{}")
            ],
        ),
        Message(
            role=MessageRole.tool,
            content_blocks=[
                ToolResultBlock(tool_call_id="c1", content="ok", is_error=False)
            ],
        ),
        Message(role=MessageRole.assistant, content_blocks=[TextBlock(text="done")]),
    ]
    repaired = _repair_outbound_tool_pairing(messages, placeholder="x")
    assert _tool_use_ids(repaired) == ["c1"]
    assert _tool_result_ids(repaired) == ["c1"]
    # No synthetic placeholder leaked in.
    for m in repaired:
        for b in m.content_blocks:
            if isinstance(b, ToolResultBlock):
                assert b.content == "ok"
                assert b.is_error is False


# ---------------------------------------------------------------------------
# integration: repair runs at the wire boundary before LLMRequest
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_orphaned_tool_use_in_history_repaired_before_request(
    engine_factory: Any,
    in_memory_runtime: dict[str, Any],
) -> None:
    """A dangling tool_use in resumed history is repaired on the next stream.

    Simulates a resume-from-partial-batch / crash scenario: history ends with
    an assistant ``tool_use`` and NO tool_result (the dispatch never ran). The
    next ``query()`` stream must NOT forward that orphan; the request the
    provider sees must carry a synthetic ``is_error`` tool_result.
    """
    engine = engine_factory()
    # Pre-seed history with a dangling tool_use (no matching tool_result).
    engine.history.append(
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="task")])
    )
    engine.history.append(
        Message(
            role=MessageRole.assistant,
            content_blocks=[
                ToolUseBlock(
                    tool_call_id="dangling_1", name="Read", arguments_json="{}"
                )
            ],
        )
    )
    in_memory_runtime["llm"].queue_response(text="recovered")

    # A fresh user turn re-opens the stream over the rehydrated history.
    user_msg = Message(
        role=MessageRole.user, content_blocks=[TextBlock(text="continue")]
    )
    async for _ in engine.run(user_msg):
        pass

    request = in_memory_runtime["llm"].calls[0]
    tool_use_ids = _tool_use_ids(list(request.messages))
    tool_result_ids = _tool_result_ids(list(request.messages))
    assert "dangling_1" in tool_use_ids
    # Every tool_use on the wire has a matching tool_result.
    for tu in tool_use_ids:
        assert tu in tool_result_ids, (
            f"tool_use {tu} reached the wire with no matching tool_result"
        )
    # The synthetic repair is an error result.
    synthetic = next(
        b
        for m in request.messages
        for b in m.content_blocks
        if isinstance(b, ToolResultBlock) and b.tool_call_id == "dangling_1"
    )
    assert synthetic.is_error is True


# ---------------------------------------------------------------------------
# cancel / error teardown synthesises missing tool_results
# ---------------------------------------------------------------------------


def test_synthesize_missing_tool_results_appends_error_block() -> None:
    """``_synthesize_missing_tool_results`` mutates history to satisfy pairing."""
    history = [
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")]),
        Message(
            role=MessageRole.assistant,
            content_blocks=[
                ToolUseBlock(tool_call_id="c1", name="Read", arguments_json="{}"),
                ToolUseBlock(tool_call_id="c2", name="Read", arguments_json="{}"),
            ],
        ),
    ]
    inserted = _synthesize_missing_tool_results(history, error_content="Interrupted")
    assert inserted == 2
    result_ids = _tool_result_ids(history)
    assert result_ids == ["c1", "c2"]
    for m in history:
        for b in m.content_blocks:
            if isinstance(b, ToolResultBlock):
                assert b.is_error is True
                assert b.content == "Interrupted"


def test_synthesize_inserts_after_owning_assistant_not_at_tail() -> None:
    """Synthetic result lands right after its tool_use, before any later turn.

    Appending at the tail when an intervening user/recovery message already
    follows the orphan would persist an out-of-order pair the Anthropic wire
    rejects; the result must be inserted immediately after the assistant.
    """
    history = [
        Message(
            role=MessageRole.assistant,
            content_blocks=[
                ToolUseBlock(tool_call_id="c1", name="Read", arguments_json="{}")
            ],
        ),
        # An intervening message already follows the orphan.
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="later turn")]),
    ]
    inserted = _synthesize_missing_tool_results(history, error_content="Interrupted")
    assert inserted == 1
    assert [m.role for m in history] == [
        MessageRole.assistant,
        MessageRole.tool,
        MessageRole.user,
    ]
    assert isinstance(history[1].content_blocks[0], ToolResultBlock)
    assert history[1].content_blocks[0].tool_call_id == "c1"
    assert history[1].content_blocks[0].is_error is True


def test_synthesize_skips_already_paired() -> None:
    """No synthetic block is added when a tool_use already has a result."""
    history = [
        Message(
            role=MessageRole.assistant,
            content_blocks=[
                ToolUseBlock(tool_call_id="c1", name="Read", arguments_json="{}")
            ],
        ),
        Message(
            role=MessageRole.tool,
            content_blocks=[
                ToolResultBlock(tool_call_id="c1", content="ok", is_error=False)
            ],
        ),
    ]
    inserted = _synthesize_missing_tool_results(history, error_content="Interrupted")
    assert inserted == 0
    assert _tool_result_ids(history) == ["c1"]


@pytest.mark.asyncio
async def test_cancel_teardown_pairs_dangling_tool_use(
    engine_factory: Any,
    in_memory_runtime: dict[str, Any],
) -> None:
    """A cancel after a tool_use is emitted leaves a pairing-valid history.

    The model emits a tool_use; the run is cancelled before the dispatch loop
    completes. The persisted history must end with a tool_result for the
    dangling tool_use so a resume on another pod does not 400.
    """
    engine = engine_factory()
    # Pre-seed: assistant emitted a tool_use, no result yet, then cancel.
    engine.history.append(
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="task")])
    )
    engine.history.append(
        Message(
            role=MessageRole.assistant,
            content_blocks=[
                ToolUseBlock(
                    tool_call_id="mid_call", name="Read", arguments_json="{}"
                )
            ],
        )
    )
    engine.stop()

    events: list[TurnEvent] = []
    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="x")])
    async for evt in engine.run(user_msg):
        events.append(evt)

    # Cancel terminal reached.
    stops = [e for e in events if e.type is EventType.MESSAGE_STOP]
    assert stops
    assert stops[-1].payload["stop_reason"] == StopReason.cancelled.value
    assert engine.state is LoopState.CANCELLED

    # History is pairing-valid: the dangling tool_use now has a result.
    tool_use_ids = _tool_use_ids(list(engine.history))
    tool_result_ids = _tool_result_ids(list(engine.history))
    assert "mid_call" in tool_use_ids
    assert "mid_call" in tool_result_ids
    synthetic = next(
        b
        for m in engine.history
        for b in m.content_blocks
        if isinstance(b, ToolResultBlock) and b.tool_call_id == "mid_call"
    )
    assert synthetic.is_error is True


class _SideEffectTool(Tool):
    """Tool stub that records every invocation as an observable side effect.

    Used to prove a cancel that lands AFTER a tool_use closed but BEFORE the
    dispatch loop ran does NOT actually invoke the tool. No
    ``is_concurrent_safe``/``is_destructive`` attributes ⇒ the loop takes the
    serial dispatch path (``getattr(..., False)``).
    """

    def __init__(self) -> None:
        self.invocations: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "SideEffect"

    @property
    def definition(self):  # type: ignore[no-untyped-def]
        from protocore.contracts.types import ToolDefinition, ToolParameterSchema

        return ToolDefinition(
            name="SideEffect",
            description="Performs an observable side effect when dispatched",
            parameters=ToolParameterSchema(properties={"v": {"type": "string"}}),
        )

    async def invoke(self, context, arguments):  # type: ignore[no-untyped-def]
        from protocore.contracts.types import ToolResult

        self.invocations.append(dict(arguments))
        return ToolResult(tool_call_id="", content="did-side-effect", is_error=False)


@pytest.mark.asyncio
async def test_cancel_after_tooluse_close_does_not_dispatch(
    engine_factory: Any,
    in_memory_runtime: dict[str, Any],
) -> None:
    """(tool-dispatch path) — a stop requested AFTER the assistant
    turn produced a tool_use but BEFORE the dispatch loop ran MUST NOT invoke
    the tool, MUST end the run CANCELLED, and MUST leave a pairing-valid
    history.

    Prior to the fix, the ``stop_requested`` re-check existed only in the
    no-tool ``end_turn`` branch. When the model emitted a tool_use and a
    cancel landed mid-stream, the loop still fell through to the dispatch loop
    and ran the tool — performing a side effect AFTER cancellation. The fix
    re-checks ``engine.stop_requested`` immediately before the dispatch loop
    (mirroring the no-tool branch), synthesises the missing tool_results, and
    routes to the CANCELLED terminal without dispatching.
    """
    from collections.abc import AsyncIterator

    from protocore.contracts.llm import LLMRequest, LLMStreamEvent

    class _ToolUseThenStopLLM:
        """Emits a COMPLETE tool_use, then flips ``engine.stop()``.

        The tool_call is fully accumulated (``tool_use_stop`` delta) before
        stop is observed, so ``result.tool_calls`` is non-empty when the inner
        stream breaks on the per-delta stop check at the top of the next loop
        iteration. ``finish_reason`` stays ``None``.
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
            assert self.engine is not None
            yield LLMStreamEvent(name="message_start", payload={})
            yield LLMStreamEvent(
                name="tool_use_start",
                payload={"tool_call_id": "mid_tool", "tool_name": "SideEffect"},
            )
            yield LLMStreamEvent(
                name="tool_use_input_delta",
                payload={
                    "tool_call_id": "mid_tool",
                    "partial_input_json": '{"v": "x"}',
                },
            )
            # tool_use closes — the consumer appends the ToolCall here.
            yield LLMStreamEvent(
                name="tool_use_stop",
                payload={"tool_call_id": "mid_tool", "final_input": {"v": "x"}},
            )
            # Operator interrupt lands AFTER the tool_use closed. The per-delta
            # stop-check at the top of the next loop iteration breaks the inner
            # stream before this finish delta is processed → tool_call survives,
            # finish_reason stays None.
            self.engine.stop()
            yield LLMStreamEvent(
                name="message_stop",
                payload={"stop_reason": StopReason.tool_use.value},
            )

        async def complete_structured(self, request, schema):  # type: ignore[no-untyped-def]
            raise RuntimeError("unused")

        def count_tokens(self, text, model=None) -> int:  # type: ignore[no-untyped-def]
            return max(1, len(text) // 4)

    engine = engine_factory()
    tool = _SideEffectTool()
    in_memory_runtime["tools"].register(tool)
    stop_llm = _ToolUseThenStopLLM()
    stop_llm.engine = engine
    engine.llm = stop_llm  # type: ignore[assignment]

    events: list[TurnEvent] = []
    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])
    async for evt in engine.run(user_msg):
        events.append(evt)
        if len(events) > 10_000:
            pytest.fail("loop did not terminate")

    # (1) No tool side effect was performed AFTER cancellation.
    assert tool.invocations == []
    # (2) The run ended CANCELLED, never COMPLETED.
    assert engine.state is LoopState.CANCELLED
    stops = [e for e in events if e.type is EventType.MESSAGE_STOP]
    assert stops
    assert stops[-1].payload["stop_reason"] == StopReason.cancelled.value
    # (3) No TOOL_RESULT for a dispatched tool was emitted on the wire (the
    # synthetic pairing result lives in history, not as a TOOL_RESULT event).
    tool_results = [e for e in events if e.type is EventType.TOOL_RESULT]
    assert tool_results == []
    # (4) History is pairing-valid: the dangling tool_use now has a synthetic
    # is_error result so a resume on another pod does not 400.
    assert "mid_tool" in _tool_use_ids(list(engine.history))
    assert "mid_tool" in _tool_result_ids(list(engine.history))
    synthetic = next(
        b
        for m in engine.history
        for b in m.content_blocks
        if isinstance(b, ToolResultBlock) and b.tool_call_id == "mid_tool"
    )
    assert synthetic.is_error is True


class _StopOnInvokeTool(Tool):
    """Serial tool whose ``invoke`` flips ``engine.stop()`` on its FIRST call.

    Records every invocation. Has no ``is_concurrent_safe``/``is_destructive``
    attribute ⇒ ``getattr(..., False)`` ⇒ the loop takes the SERIAL dispatch
    path. Used to prove a cancel that lands BETWEEN two serial tool calls in
    one assistant turn does NOT dispatch the second call (cancel in an await
    gap mid-dispatch-loop).
    """

    def __init__(self, engine: Any) -> None:
        self._engine = engine
        self.invocations: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "StopOnInvoke"

    @property
    def definition(self):  # type: ignore[no-untyped-def]
        from protocore.contracts.types import ToolDefinition, ToolParameterSchema

        return ToolDefinition(
            name="StopOnInvoke",
            description="Requests stop on its first dispatch",
            parameters=ToolParameterSchema(properties={"v": {"type": "string"}}),
        )

    async def invoke(self, context, arguments):  # type: ignore[no-untyped-def]
        from protocore.contracts.types import ToolResult

        self.invocations.append(dict(arguments))
        # The interrupt lands DURING the first tool's dispatch — i.e. inside
        # the dispatch-loop await gap, AFTER the pre-loop ``stop_requested``
        # re-check has already passed. Without the per-iteration guard the
        # loop would advance to the second call and dispatch it.
        self._engine.stop()
        return ToolResult(tool_call_id="", content="ok", is_error=False)


def _two_tool_call_stub(*tool_names: str):  # type: ignore[no-untyped-def]
    """Build an LLM stub that emits N COMPLETE tool_use blocks then end_turn.

    No ``engine.stop()`` is called inside the stream — the stop is driven
    later (by a tool's ``invoke`` or a hook), so the pre-dispatch-loop
    ``stop_requested`` re-check passes and the cancel only becomes visible
    once the loop is already iterating.
    """
    from collections.abc import AsyncIterator

    from protocore.contracts.llm import LLMRequest, LLMStreamEvent

    class _MultiToolLLM:
        def __init__(self) -> None:
            self._calls: list[LLMRequest] = []

        @property
        def calls(self) -> list[LLMRequest]:
            return self._calls

        async def stream_with_tools(
            self, request: LLMRequest
        ) -> AsyncIterator[LLMStreamEvent]:
            self._calls.append(request)
            yield LLMStreamEvent(name="message_start", payload={})
            for i, tool_name in enumerate(tool_names):
                call_id = f"call_{i}"
                yield LLMStreamEvent(
                    name="tool_use_start",
                    payload={"tool_call_id": call_id, "tool_name": tool_name},
                )
                yield LLMStreamEvent(
                    name="tool_use_input_delta",
                    payload={
                        "tool_call_id": call_id,
                        "partial_input_json": '{"v": "x"}',
                    },
                )
                yield LLMStreamEvent(
                    name="tool_use_stop",
                    payload={"tool_call_id": call_id, "final_input": {"v": "x"}},
                )
            yield LLMStreamEvent(
                name="message_stop",
                payload={"stop_reason": StopReason.tool_use.value},
            )

        async def complete_structured(self, request, schema):  # type: ignore[no-untyped-def]
            raise RuntimeError("unused")

        def count_tokens(self, text, model=None) -> int:  # type: ignore[no-untyped-def]
            return max(1, len(text) // 4)

    return _MultiToolLLM()


def _single_tool_stop_in_stream_stub(tool_name: str):  # type: ignore[no-untyped-def]
    """LLM stub: emit ONE complete tool_use, then flip ``engine.stop()``.

    The tool_call is fully accumulated (``tool_use_stop``) before stop is
    observed, so ``result.tool_calls`` is non-empty when the inner stream
    breaks on the per-delta stop check — exercising the pre-dispatch-loop guard
    of the MAIN dispatch path. ``engine`` is bound via the ``engine`` attribute
    after construction (mirrors ``_ToolUseThenStopLLM``).
    """
    from collections.abc import AsyncIterator

    from protocore.contracts.llm import LLMRequest, LLMStreamEvent

    class _SingleToolStopLLM:
        def __init__(self) -> None:
            self.engine: Any = None
            self._calls: list[LLMRequest] = []

        @property
        def calls(self) -> list[LLMRequest]:
            return self._calls

        async def stream_with_tools(
            self, request: LLMRequest
        ) -> AsyncIterator[LLMStreamEvent]:
            self._calls.append(request)
            assert self.engine is not None
            yield LLMStreamEvent(name="message_start", payload={})
            yield LLMStreamEvent(
                name="tool_use_start",
                payload={"tool_call_id": "mid_tool", "tool_name": tool_name},
            )
            yield LLMStreamEvent(
                name="tool_use_input_delta",
                payload={"tool_call_id": "mid_tool", "partial_input_json": '{"v": "x"}'},
            )
            yield LLMStreamEvent(
                name="tool_use_stop",
                payload={"tool_call_id": "mid_tool", "final_input": {"v": "x"}},
            )
            # Interrupt lands AFTER the tool_use closed; the per-delta stop-check
            # at the top of the next iteration breaks the inner stream.
            self.engine.stop()
            yield LLMStreamEvent(
                name="message_stop",
                payload={"stop_reason": StopReason.tool_use.value},
            )

        async def complete_structured(self, request, schema):  # type: ignore[no-untyped-def]
            raise RuntimeError("unused")

        def count_tokens(self, text, model=None) -> int:  # type: ignore[no-untyped-def]
            return max(1, len(text) // 4)

    return _SingleToolStopLLM()


@pytest.mark.asyncio
async def test_cancel_between_serial_tool_calls_does_not_dispatch_remainder(
    engine_factory: Any,
    in_memory_runtime: dict[str, Any],
) -> None:
    """A cancel that lands DURING the dispatch loop (between two tool calls in
    a multi-call batch) MUST NOT dispatch the remaining call(s).

    The model emits TWO serial tool calls in one assistant turn. The first
    tool's ``invoke`` flips ``engine.stop()``. The pre-dispatch-loop
    ``stop_requested`` re-check has already passed (stream did not stop), so
    the bug is observable only with a per-iteration guard: without it the loop
    advances to the second call and dispatches it (side effect after cancel).
    With the fix the second tool is NOT invoked, the run ends CANCELLED, and
    history stays pairing-valid (the second, undispatched tool_use gets a
    synthetic ``is_error`` result).
    """
    engine = engine_factory()
    tool = _StopOnInvokeTool(engine)
    side_effect = _SideEffectTool()
    in_memory_runtime["tools"].register(tool)
    in_memory_runtime["tools"].register(side_effect)
    # First call → StopOnInvoke (stops), second call → SideEffect (must NOT run).
    engine.llm = _two_tool_call_stub("StopOnInvoke", "SideEffect")  # type: ignore[assignment]

    events: list[TurnEvent] = []
    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])
    async for evt in engine.run(user_msg):
        events.append(evt)
        if len(events) > 10_000:
            pytest.fail("loop did not terminate")

    # (1) The first tool ran once; the SECOND tool was NEVER dispatched.
    assert tool.invocations == [{"v": "x"}]
    assert side_effect.invocations == []
    # (2) The run ended CANCELLED, never COMPLETED.
    assert engine.state is LoopState.CANCELLED
    stops = [e for e in events if e.type is EventType.MESSAGE_STOP]
    assert stops
    assert stops[-1].payload["stop_reason"] == StopReason.cancelled.value
    # (3) History is pairing-valid for BOTH tool_use blocks.
    tool_use_ids = _tool_use_ids(list(engine.history))
    tool_result_ids = _tool_result_ids(list(engine.history))
    assert "call_0" in tool_use_ids and "call_1" in tool_use_ids
    assert "call_0" in tool_result_ids and "call_1" in tool_result_ids
    # The second (undispatched) call_1 result is a synthetic is_error block.
    synthetic = next(
        b
        for m in engine.history
        for b in m.content_blocks
        if isinstance(b, ToolResultBlock) and b.tool_call_id == "call_1"
    )
    assert synthetic.is_error is True


@pytest.mark.asyncio
async def test_cancel_in_hook_predicate_await_gap_does_not_dispatch(
    engine_factory: Any,
    in_memory_runtime: dict[str, Any],
) -> None:
    """A cancel that lands DURING the awaited PreToolUse hook predicate
    (between the pre-loop ``stop_requested`` re-check and the dispatch loop)
    MUST NOT dispatch any tool.

    The hook manager's ``list`` is patched to flip ``engine.stop()`` while
    being awaited — exactly the await gap the prior fix did not re-check.
    Without an after-predicate re-check the loop dispatches the tool; with it
    the run ends CANCELLED with no dispatch and a pairing-valid history.
    """
    engine = engine_factory()
    side_effect = _SideEffectTool()
    in_memory_runtime["tools"].register(side_effect)
    engine.llm = _two_tool_call_stub("SideEffect")  # type: ignore[assignment]

    hook_manager = in_memory_runtime["hooks"]
    original_list = hook_manager.list

    async def _stopping_list(*args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
        # The interrupt lands while the per-turn hook predicate is awaited.
        engine.stop()
        return await original_list(*args, **kwargs)

    hook_manager.list = _stopping_list  # type: ignore[assignment]

    events: list[TurnEvent] = []
    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])
    async for evt in engine.run(user_msg):
        events.append(evt)
        if len(events) > 10_000:
            pytest.fail("loop did not terminate")

    # (1) No tool was dispatched after the cancel landed in the await gap.
    assert side_effect.invocations == []
    # (2) The run ended CANCELLED.
    assert engine.state is LoopState.CANCELLED
    stops = [e for e in events if e.type is EventType.MESSAGE_STOP]
    assert stops
    assert stops[-1].payload["stop_reason"] == StopReason.cancelled.value
    # (3) History is pairing-valid: the undispatched tool_use has a synthetic
    # is_error result.
    assert "call_0" in _tool_use_ids(list(engine.history))
    assert "call_0" in _tool_result_ids(list(engine.history))
    synthetic = next(
        b
        for m in engine.history
        for b in m.content_blocks
        if isinstance(b, ToolResultBlock) and b.tool_call_id == "call_0"
    )
    assert synthetic.is_error is True


def _truncated_plus_clean_stub(*clean_tool_names: str):  # type: ignore[no-untyped-def]
    """LLM stub: ONE truncated Write + N clean tool calls in one turn.

    Round 0 emits a Write call with the stop-truncation signature
    (``args_partial_truncated=True`` on ``tool_use_stop``, brace-balanced
    empty args) followed by ``clean_tool_names`` each with full args, then
    ``finish_reason="stop"``. This drives the truncated-tool RECOVERY dispatch
    path in :func:`_stream_one_assistant_message` (a SECOND dispatch entry
    point, distinct from the main dispatch loop): the truncated Write is paired
    with a synthetic error result and the clean calls flow through
    ``_dispatch_tool``. Round 1+ is a healthy text end_turn so an un-cancelled
    run can converge. No ``engine.stop()`` is driven inside the stream — the
    cancel is injected later (snapshot persist or a tool's ``invoke``).
    """
    from collections.abc import AsyncIterator

    from protocore.contracts.llm import (
        LLMRequest,
        ProviderDelta,
        ProviderDeltaKind,
    )

    class _TruncatedPlusCleanLLM:
        def __init__(self) -> None:
            self._call_idx = 0
            self._calls: list[LLMRequest] = []

        @property
        def calls(self) -> list[LLMRequest]:
            return self._calls

        async def stream_with_tools(
            self, request: LLMRequest
        ) -> AsyncIterator[ProviderDelta]:
            self._calls.append(request)
            idx = self._call_idx
            self._call_idx += 1
            if idx == 0:
                # Truncated Write — stop-finish with brace-balanced empty args
                # + ``args_partial_truncated`` on stop.
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
                # Clean (non-truncated) calls — full args.
                for i, tool_name in enumerate(clean_tool_names):
                    call_id = f"clean_{i}"
                    yield ProviderDelta(
                        kind=ProviderDeltaKind.tool_use_start,
                        tool_call_id=call_id,
                        tool_name=tool_name,
                    )
                    yield ProviderDelta(
                        kind=ProviderDeltaKind.tool_use_input,
                        tool_call_id=call_id,
                        tool_input_delta='{"v": "x"}',
                    )
                    yield ProviderDelta(
                        kind=ProviderDeltaKind.tool_use_stop,
                        tool_call_id=call_id,
                        tool_input_final={"v": "x"},
                        is_block_end=True,
                        truncated_by_output_cap=False,
                        args_partial_truncated=False,
                    )
                yield ProviderDelta(
                    kind=ProviderDeltaKind.finish,
                    finish_reason="stop",
                )
                return
            # Recovery round — healthy end_turn so an un-cancelled run ends.
            yield ProviderDelta(kind=ProviderDeltaKind.text, content="ok")
            yield ProviderDelta(
                kind=ProviderDeltaKind.finish, finish_reason="stop"
            )

        async def complete_structured(self, request, schema):  # type: ignore[no-untyped-def]
            raise RuntimeError("unused")

        def count_tokens(self, text, model=None) -> int:  # type: ignore[no-untyped-def]
            return max(1, len(text) // 4)

    return _TruncatedPlusCleanLLM()


@pytest.mark.asyncio
async def test_cancel_in_truncated_recovery_does_not_dispatch_non_truncated(
    engine_factory: Any,
    in_memory_runtime: dict[str, Any],
) -> None:
    """A cancel that lands DURING the truncated-tool recovery (the per-call
    synthetic-result yields + the snapshot persist that precede the
    non-truncated dispatch loop) MUST NOT dispatch the non-truncated call.

    The model emits ONE truncated Write + ONE clean SideEffect call in the same
    turn. The recovery branch synthesises the Write error result, then
    ``await engine._persist_snapshot()`` — patched here to flip ``engine.stop()``
    — immediately before the non-truncated dispatch loop. Before the fix that
    loop dispatched SideEffect anyway (side effect AFTER cancellation). With the
    per-call ``stop_requested`` re-check the recovery path routes through
    ``_emit_dispatch_cancel_teardown``: SideEffect is NOT invoked, the run ends
    CANCELLED, and history is pairing-valid (the undispatched ``clean_0``
    tool_use gets a synthetic ``is_error`` result; the truncated Write keeps its
    own synthetic result, idempotently un-doubled).
    """
    engine = engine_factory()
    side_effect = _SideEffectTool()
    in_memory_runtime["tools"].register(side_effect)
    engine.llm = _truncated_plus_clean_stub("SideEffect")  # type: ignore[assignment]

    # Flip stop the instant the truncated Write's synthetic TOOL_RESULT lands on
    # the wire (yielded inside the recovery synthesis, AFTER the assistant turn
    # with both tool_use blocks is in history, BEFORE the snapshot persist + the
    # non-truncated dispatch loop). This is exactly the recovery-yield gap the
    # cancel guard must cover; a real cancel lands here cross-pod.
    events: list[TurnEvent] = []
    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])
    async for evt in engine.run(user_msg):
        events.append(evt)
        if (
            evt.type is EventType.TOOL_RESULT
            and evt.payload.get("tool_call_id") == "toolu_trunc_write"
        ):
            engine.stop()
        if len(events) > 10_000:
            pytest.fail("loop did not terminate")

    # (1) The non-truncated SideEffect tool was NEVER dispatched after cancel.
    assert side_effect.invocations == []
    # (2) The run ended CANCELLED, never COMPLETED.
    assert engine.state is LoopState.CANCELLED
    stops = [e for e in events if e.type is EventType.MESSAGE_STOP]
    assert stops
    assert stops[-1].payload["stop_reason"] == StopReason.cancelled.value
    # (3) Only ONE LLM call happened — the recovery round never ran because the
    # turn was cancelled inside the recovery synthesis.
    assert len(engine.llm.calls) == 1  # type: ignore[attr-defined]
    # (4) History is pairing-valid for BOTH tool_use blocks.
    tool_use_ids = _tool_use_ids(list(engine.history))
    tool_result_ids = _tool_result_ids(list(engine.history))
    assert "toolu_trunc_write" in tool_use_ids and "clean_0" in tool_use_ids
    assert "toolu_trunc_write" in tool_result_ids and "clean_0" in tool_result_ids
    # The undispatched clean_0 result is a synthetic is_error block.
    synthetic_clean = next(
        b
        for m in engine.history
        for b in m.content_blocks
        if isinstance(b, ToolResultBlock) and b.tool_call_id == "clean_0"
    )
    assert synthetic_clean.is_error is True
    # The truncated Write keeps exactly ONE result (idempotent — the teardown
    # synthesiser skipped the already-paired call, no double insert).
    write_results = [
        b
        for m in engine.history
        for b in m.content_blocks
        if isinstance(b, ToolResultBlock) and b.tool_call_id == "toolu_trunc_write"
    ]
    assert len(write_results) == 1
    assert write_results[0].is_error is True


@pytest.mark.asyncio
async def test_cancel_between_recovery_dispatches_does_not_dispatch_remainder(
    engine_factory: Any,
    in_memory_runtime: dict[str, Any],
) -> None:
    """A cancel that lands BETWEEN two non-truncated dispatches in the
    truncated-tool recovery path MUST NOT dispatch the remaining call.

    The model emits ONE truncated Write + TWO clean calls (StopOnInvoke,
    SideEffect). The truncated Write is paired with its synthetic error result;
    the first clean call (StopOnInvoke) dispatches and its ``invoke`` flips
    ``engine.stop()``; the per-iteration re-check then routes the second clean
    call (SideEffect) through the cancel teardown WITHOUT dispatching it.
    """
    engine = engine_factory()
    stopper = _StopOnInvokeTool(engine)
    side_effect = _SideEffectTool()
    in_memory_runtime["tools"].register(stopper)
    in_memory_runtime["tools"].register(side_effect)
    engine.llm = _truncated_plus_clean_stub(  # type: ignore[assignment]
        "StopOnInvoke", "SideEffect"
    )

    events: list[TurnEvent] = []
    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])
    async for evt in engine.run(user_msg):
        events.append(evt)
        if len(events) > 10_000:
            pytest.fail("loop did not terminate")

    # (1) The first clean tool ran once; the SECOND clean tool NEVER dispatched.
    assert stopper.invocations == [{"v": "x"}]
    assert side_effect.invocations == []
    # (2) The run ended CANCELLED.
    assert engine.state is LoopState.CANCELLED
    stops = [e for e in events if e.type is EventType.MESSAGE_STOP]
    assert stops
    assert stops[-1].payload["stop_reason"] == StopReason.cancelled.value
    # (3) History is pairing-valid for all three tool_use blocks.
    tool_result_ids = _tool_result_ids(list(engine.history))
    assert "toolu_trunc_write" in tool_result_ids
    assert "clean_0" in tool_result_ids  # StopOnInvoke — real result
    assert "clean_1" in tool_result_ids  # SideEffect — synthetic is_error
    synthetic_remainder = next(
        b
        for m in engine.history
        for b in m.content_blocks
        if isinstance(b, ToolResultBlock) and b.tool_call_id == "clean_1"
    )
    assert synthetic_remainder.is_error is True


# ---------------------------------------------------------------------------
# no-new-dispatch-after-stop invariant across ALL dispatch paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "scenario",
    [
        "serial_before_loop",
        "serial_between_calls",
        "hook_predicate_gap",
        "truncated_recovery_before_loop",
        "truncated_recovery_between_calls",
    ],
)
@pytest.mark.asyncio
async def test_no_new_dispatch_after_stop_invariant(
    scenario: str,
    engine_factory: Any,
    in_memory_runtime: dict[str, Any],
) -> None:
    """Cross-path invariant: once ``engine.stop_requested`` is observed, NO new
    tool is dispatched on ANY dispatch path, the run ends CANCELLED, and the
    persisted history stays pairing-valid.

    Each scenario injects the cancel in a different await/yield gap of a
    distinct dispatch entry point. ``guard_tool`` is the tool that must NEVER
    run after the cancel; the assertions are uniform across paths.
    """
    engine = engine_factory()
    guard_tool = _SideEffectTool()
    in_memory_runtime["tools"].register(guard_tool)

    if scenario == "serial_before_loop":
        # Cancel lands inside the stream, after the tool_use closed but before
        # the main dispatch loop runs (pre-loop guard).
        stub = _single_tool_stop_in_stream_stub("SideEffect")
        stub.engine = engine  # type: ignore[attr-defined]
        engine.llm = stub  # type: ignore[assignment]
    elif scenario == "serial_between_calls":
        # Cancel lands between two serial calls in the main dispatch loop.
        stopper = _StopOnInvokeTool(engine)
        in_memory_runtime["tools"].register(stopper)
        engine.llm = _two_tool_call_stub("StopOnInvoke", "SideEffect")  # type: ignore[assignment]
    elif scenario == "hook_predicate_gap":
        # Cancel lands while the per-turn PreToolUse hook predicate is awaited.
        engine.llm = _two_tool_call_stub("SideEffect")  # type: ignore[assignment]
        hook_manager = in_memory_runtime["hooks"]
        original_list = hook_manager.list

        async def _stopping_list(*args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
            engine.stop()
            return await original_list(*args, **kwargs)

        hook_manager.list = _stopping_list  # type: ignore[assignment]
    elif scenario == "truncated_recovery_before_loop":
        # Cancel lands inside the truncated-recovery synthesis (on the truncated
        # call's synthetic TOOL_RESULT yield) before the non-truncated dispatch
        # loop. The consumer loop below flips stop on that event.
        engine.llm = _truncated_plus_clean_stub("SideEffect")  # type: ignore[assignment]
    elif scenario == "truncated_recovery_between_calls":
        # Cancel lands between two non-truncated calls in the recovery path.
        stopper = _StopOnInvokeTool(engine)
        in_memory_runtime["tools"].register(stopper)
        engine.llm = _truncated_plus_clean_stub(  # type: ignore[assignment]
            "StopOnInvoke", "SideEffect"
        )
    else:  # pragma: no cover - defensive
        pytest.fail(f"unknown scenario {scenario}")

    events: list[TurnEvent] = []
    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])
    async for evt in engine.run(user_msg):
        events.append(evt)
        if (
            scenario == "truncated_recovery_before_loop"
            and evt.type is EventType.TOOL_RESULT
            and evt.payload.get("tool_call_id") == "toolu_trunc_write"
        ):
            engine.stop()
        if len(events) > 10_000:
            pytest.fail("loop did not terminate")

    # Invariant 1 — the guard tool was NEVER dispatched after the cancel.
    assert guard_tool.invocations == [], (
        f"scenario={scenario}: a NEW tool was dispatched after stop_requested"
    )
    # Invariant 2 — the run ended CANCELLED (never COMPLETED).
    assert engine.state is LoopState.CANCELLED, f"scenario={scenario}"
    stops = [e for e in events if e.type is EventType.MESSAGE_STOP]
    assert stops, f"scenario={scenario}"
    assert stops[-1].payload["stop_reason"] == StopReason.cancelled.value, (
        f"scenario={scenario}"
    )
    # Invariant 3 — every tool_use in history is paired (resume stays wire-valid).
    use_ids = set(_tool_use_ids(list(engine.history)))
    result_ids = set(_tool_result_ids(list(engine.history)))
    assert use_ids, f"scenario={scenario}: expected at least one tool_use"
    assert use_ids <= result_ids, (
        f"scenario={scenario}: orphaned tool_use {use_ids - result_ids}"
    )


@pytest.mark.asyncio
async def test_llm_error_teardown_pairs_dangling_tool_use(
    engine_factory: Any,
    in_memory_runtime: dict[str, Any],
) -> None:
    """``_emit_llm_terminal`` repairs a dangling tool_use before FAILED stop.

    Drives the engine to the LLM-error terminal with a dangling tool_use in
    history and asserts the persisted history is pairing-valid.
    """
    from protocore.contracts.llm import LLMProviderError
    from protocore.runtime.query import _emit_llm_terminal

    engine = engine_factory()
    engine.transition_to(LoopState.RUNNING)
    engine.history.append(
        Message(
            role=MessageRole.assistant,
            content_blocks=[
                ToolUseBlock(
                    tool_call_id="err_call", name="Read", arguments_json="{}"
                )
            ],
        )
    )

    events: list[TurnEvent] = []
    async for evt in _emit_llm_terminal(
        engine, LLMProviderError("boom"), kind="llm_provider_error"
    ):
        events.append(evt)

    assert engine.state is LoopState.FAILED
    tool_result_ids = _tool_result_ids(list(engine.history))
    assert "err_call" in tool_result_ids
    synthetic = next(
        b
        for m in engine.history
        for b in m.content_blocks
        if isinstance(b, ToolResultBlock) and b.tool_call_id == "err_call"
    )
    assert synthetic.is_error is True


@pytest.mark.asyncio
async def test_hook_denied_teardown_pairs_dangling_tool_use(
    engine_factory: Any,
    in_memory_runtime: dict[str, Any],
) -> None:
    """A UserPromptSubmit DENY leaves a pairing-valid history .

    A resumed history can already carry a dangling tool_use when the next
    turn's prompt-submit hook denies; the FAILED snapshot must be wire-valid.
    """
    from protocore.contracts.hooks import HookActionKind, HookResult
    from protocore.contracts.types import HookEvent

    engine = engine_factory()
    engine.history.append(
        Message(
            role=MessageRole.assistant,
            content_blocks=[
                ToolUseBlock(tool_call_id="denied_call", name="Read", arguments_json="{}")
            ],
        )
    )
    in_memory_runtime["hooks"].queue_action(
        HookEvent.user_prompt_submit,
        HookResult(action=HookActionKind.DENY, reason="blocked"),
    )

    events: list[TurnEvent] = []
    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])
    async for evt in engine.run(user_msg):
        events.append(evt)

    assert engine.state is LoopState.FAILED
    assert "denied_call" in _tool_result_ids(list(engine.history))


@pytest.mark.asyncio
async def test_routine_compaction_exhaustion_leaves_a_pairing_valid_history(
    engine_factory: Any,
    in_memory_runtime: dict[str, Any],
) -> None:
    """Routine exhaustion suspends compaction; whatever ends the run pairs the call.

    A proactive pass out of budget no longer fails the run by itself — nothing
    has been rejected yet — so the run goes on, and the terminal it reaches
    must still leave the dangling tool_use paired.
    """
    from protocore.runtime.context.compaction import CompactionExhaustedError

    engine = engine_factory(
        rc=LoopConstants(model_context_window=64, request_context_safety_tokens=0)
    )
    engine.history.append(
        Message(
            role=MessageRole.assistant,
            content_blocks=[
                ToolUseBlock(
                    tool_call_id="compact_call", name="Read", arguments_json="{}"
                )
            ],
        )
    )

    async def _boom(*args: Any, **kwargs: Any) -> Any:
        raise CompactionExhaustedError("no room")

    engine.context_manager.run_compaction = _boom  # type: ignore[method-assign]
    engine.context_manager.has_proactive_work = (  # type: ignore[method-assign]
        lambda *_args, **_kwargs: True
    )
    engine.needs_compaction = lambda: True  # type: ignore[method-assign,assignment]

    events: list[TurnEvent] = []
    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])
    async for evt in engine.run(user_msg):
        events.append(evt)

    reasons = [
        evt.payload.get("reason") for evt in events if evt.type is EventType.STATE_CHANGED
    ]
    assert "compaction_exhausted_proactive_suspended" in reasons
    assert engine.proactive_compaction_suspended is True
    assert engine.is_terminal
    assert "compact_call" in _tool_result_ids(list(engine.history))


# ---------------------------------------------------------------------------
# Plumbing — the placeholders are templates, not inline magic strings
# ---------------------------------------------------------------------------


def test_the_pairing_placeholders_render_from_templates() -> None:
    prompts = bundled_prompt_provider()
    assert prompts.render("tool_result_pairing_repair").strip()
    assert prompts.render("tool_result_interrupted").strip()


# ---------------------------------------------------------------------------
# vLLM-400 fix — non-leading system messages are normalized to user at the
# request-assembly boundary (Layer 2, defense-in-depth for legacy snapshots)
# ---------------------------------------------------------------------------


def test_boundary_converts_legacy_system_summary_to_user_role() -> None:
    """A legacy system-role compaction summary mid-array is converted to a
    user-role copy at the wire boundary; metadata + content blocks preserved.

    FAIL before the fix (no normalizer existed → the system message survived
    past index 0 and would 400 vLLM), PASS after.
    """
    legacy_summary = Message(
        role=MessageRole.system,
        content_blocks=[
            TextBlock(text="<compacted-turn id='x'>prior turns</compacted-turn>")
        ],
        metadata={COMPACTION_SUMMARY_METADATA_KEY: True},
    )
    messages = [
        Message(role=MessageRole.system, content_blocks=[TextBlock(text="SYSTEM PREFIX")]),
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="task")]),
        legacy_summary,
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="recent")]),
    ]
    normalized, converted = _normalize_outbound_system_messages(messages)

    assert converted == 1
    # No system message survives past index 0.
    assert [i for i, m in enumerate(normalized) if m.role is MessageRole.system] == [0]
    # The summary is now user-role, same content + metadata preserved.
    converted_msg = normalized[2]
    assert converted_msg.role is MessageRole.user
    assert converted_msg.text == "<compacted-turn id='x'>prior turns</compacted-turn>"
    assert converted_msg.metadata.get(COMPACTION_SUMMARY_METADATA_KEY) is True


def test_boundary_leaves_genuine_index0_system_prefix_untouched() -> None:
    """The genuine system prefix at index 0 is never converted; with no other
    system message the call is a no-op (count 0, list returned unchanged)."""
    prefix = Message(
        role=MessageRole.system, content_blocks=[TextBlock(text="GENUINE SYSTEM PREFIX")]
    )
    messages = [
        prefix,
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="task")]),
        Message(role=MessageRole.assistant, content_blocks=[TextBlock(text="ans")]),
    ]
    normalized, converted = _normalize_outbound_system_messages(messages)

    assert converted == 0
    assert normalized[0] is prefix  # same object, untouched
    assert normalized[0].role is MessageRole.system


def test_boundary_end_to_end_zero_nonfirst_system_messages() -> None:
    """End-to-end-ish: assemble a request from a history that contains a
    mid-array (user-role NEW + legacy system-role) summary via the real
    ``_prepend_system_sections`` path, then normalize — the outgoing message
    list has ZERO system messages beyond index 0."""
    history = (
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="task")]),
        # New user-role summary (already fixed at source).
        Message(
            role=MessageRole.user,
            content_blocks=[TextBlock(text="<compacted-turn id='a'>s1</compacted-turn>")],
            metadata={COMPACTION_SUMMARY_METADATA_KEY: True},
        ),
        Message(role=MessageRole.assistant, content_blocks=[TextBlock(text="work")]),
        # Legacy system-role summary from a persisted snapshot.
        Message(
            role=MessageRole.system,
            content_blocks=[TextBlock(text="<compacted-turn id='b'>s2</compacted-turn>")],
            metadata={COMPACTION_SUMMARY_METADATA_KEY: True},
        ),
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="recent")]),
    )
    assembled = _prepend_system_sections(("SYSTEM PREFIX",), history)
    normalized, converted = _normalize_outbound_system_messages(assembled)

    assert converted == 1  # only the legacy system-role one needed flipping
    non_first_system = [
        i for i, m in enumerate(normalized) if i != 0 and m.role is MessageRole.system
    ]
    assert non_first_system == []
    assert normalized[0].role is MessageRole.system
    assert normalized[0].text == "SYSTEM PREFIX"
