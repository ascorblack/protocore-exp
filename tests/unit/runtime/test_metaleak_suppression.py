
"""End-to-end loop regressions for the terminal-only META-text suppression
at the stream.

These drive the ACTUAL ``query.py`` loop (``engine.run``), not just the
predicate, to prove:

* the terminal-tool nudge STILL fires (write-first recovery + typed Finalize
  depend on it);
* the post-nudge terminal-only turn's redundant META text is NOT on the live
  SSE stream NOR in durable ``engine.history`` (so ``result_preview`` is clean),
  while the real prior answer survives;
* **BLOCKER regression** — a prose-only "Создай файл X" with 0 tools is still
  nudged into the actual Write (the file gets written); suppression never skips
  the nudge;
* a genuinely-empty run still surfaces the terminal-only turn's text as the
  answer (no prior answer ⟹ not suppressed).

The message-carrying-terminal path (``pcm_answer`` keeps both nudge + text) is
covered by ``test_query_async_gen.py::test_terminal_nudge_recovers_plain_text_final``.
"""
from __future__ import annotations

from typing import Any

import pytest

from protocore.contracts.llm import LLMStreamEvent
from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.tools import Tool
from protocore.contracts.types import (
    TERMINAL_REFUSAL_NEEDS_WORK_METADATA_KEY,
    TERMINAL_TOOL_METADATA_KEY,
    Message,
    MessageRole,
    StopReason,
    TextBlock,
    ToolDefinition,
    ToolParameterSchema,
    ToolResult,
)
from protocore.runtime.events import EventType, TurnEvent
from protocore.runtime.loop_state import LoopState
from protocore.runtime.query import _query as query

# ---------------------------------------------------------------------------
# Tool stubs: a recording Write (write-first recovery target) + a BACKGROUND
# Finalize gate (no answer field).
# ---------------------------------------------------------------------------


class _RecordingWriteTool(Tool):
    """A ``Write``-shaped tool that records every invocation (so the BLOCKER
    regression can assert the file actually gets written)."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "Write"

    @property
    def definition(self):  # type: ignore[no-untyped-def]
        return ToolDefinition(
            name="Write",
            description="Write a file",
            parameters=ToolParameterSchema(
                properties={
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                required=["path"],
            ),
        )

    async def invoke(self, context, arguments):  # type: ignore[no-untyped-def]
        self.calls.append(dict(arguments))
        return ToolResult(
            tool_call_id="",
            content="written",
            is_error=False,
            metadata={"tool_name": "Write"},
        )


class _BackgroundFinalizeTool(Tool):
    """A ``Finalize``-shaped BACKGROUND terminal: schema carries NO answer
    field, so the user-facing answer can only be model prose."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "Finalize"

    @property
    def definition(self):  # type: ignore[no-untyped-def]
        return ToolDefinition(
            name="Finalize",
            description="Background finalize gate",
            parameters=ToolParameterSchema(
                properties={"declared_deliverables": {"type": "array"}},
                required=[],
            ),
        )

    async def invoke(self, context, arguments):  # type: ignore[no-untyped-def]
        self.calls.append(dict(arguments))
        return ToolResult(
            tool_call_id="",
            content="{}",
            is_error=False,
            metadata={TERMINAL_TOOL_METADATA_KEY: True},
        )


def _finalize_rc() -> LoopConstants:
    """The live shape: typed-Finalize terminal + nudge armed + min_chars=1.

    Disable the write-first prefix by default so the plain background-Finalize
    tests are not affected by the deliverable-write steering; the BLOCKER test
    re-enables it explicitly."""

    return LoopConstants(
        model_context_window=4_096,
        terminal_tool_nudge_enabled=True,
        terminal_tool_nudge_write_first_enabled=False,
        finalize_prose_gate_min_chars=1,
        # Keep the prose-gate from injecting its own repair turn in these
        # scripted scenarios (we exercise the nudge path, not the gate).
        finalize_prose_gate_enabled=False,
    )


def _text_turn_stream(text: str) -> list[LLMStreamEvent]:
    """A no-tool text turn that ends with ``end_turn`` (drives the nudge site)."""
    return [
        LLMStreamEvent(name="message_start", payload={}),
        LLMStreamEvent(name="content_block_start", payload={"kind": "text"}),
        LLMStreamEvent(name="content_block_delta", payload={"text": text, "kind": "text"}),
        LLMStreamEvent(name="content_block_stop", payload={}),
        LLMStreamEvent(name="message_stop", payload={"stop_reason": StopReason.end_turn.value}),
    ]


def _meta_plus_finalize_stream(meta_text: str) -> list[LLMStreamEvent]:
    """The H1 turn-2 shape: visible TEXT co-located with the Finalize tool call.

    Used both for the META-leak case (a prior answer exists → text suppressed)
    and the genuinely-empty case (no prior answer → this text IS the answer →
    kept). The Finalize tool call is always present and always executes."""
    return [
        LLMStreamEvent(name="message_start", payload={}),
        LLMStreamEvent(name="content_block_start", payload={"kind": "text"}),
        LLMStreamEvent(
            name="content_block_delta", payload={"text": meta_text, "kind": "text"}
        ),
        LLMStreamEvent(name="content_block_stop", payload={}),
        LLMStreamEvent(
            name="tool_use_start",
            payload={"tool_call_id": "fin-1", "tool_name": "Finalize"},
        ),
        LLMStreamEvent(
            name="tool_use_input_delta",
            payload={"tool_call_id": "fin-1", "partial_input_json": "{}"},
        ),
        LLMStreamEvent(
            name="tool_use_stop",
            payload={"tool_call_id": "fin-1", "final_input": {}},
        ),
        LLMStreamEvent(
            name="message_stop", payload={"stop_reason": StopReason.tool_use.value}
        ),
    ]


def _streamed_text(events: list[TurnEvent]) -> str:
    """All visible assistant TEXT that reached the wire (content_block_delta).

    The text-delta payload nests the fragment under ``delta.text`` (a
    ``text_delta``), mirroring ``delta_to_turn_events``; thinking deltas use
    ``thinking_delta`` and are excluded."""
    out: list[str] = []
    for e in events:
        if e.type is not EventType.CONTENT_BLOCK_DELTA:
            continue
        delta = e.payload.get("delta")
        if isinstance(delta, dict) and delta.get("type") == "text_delta":
            txt = delta.get("text")
            if isinstance(txt, str):
                out.append(txt)
    return "".join(out)


def _history_text(engine) -> str:  # type: ignore[no-untyped-def]
    """All assistant TextBlock text in durable history."""
    out: list[str] = []
    for m in engine.history:
        if m.role is MessageRole.assistant:
            for block in m.content_blocks:
                if isinstance(block, TextBlock):
                    out.append(block.text)
    return "\n".join(out)


@pytest.mark.asyncio
async def test_meta_text_suppressed_live_and_durable_answer_kept(
    engine_factory, in_memory_runtime
) -> None:
    """Background Finalize + scripted prose "144" → the post-nudge META text is
    NOT in the live stream NOR durable history; "144" IS; the nudge fired and
    Finalize ran."""
    rc = _finalize_rc()
    engine = engine_factory(rc=rc, expected_terminal_tool="Finalize")
    finalize_tool = _BackgroundFinalizeTool()
    in_memory_runtime["tools"].register(finalize_tool)

    meta = "The user asked a simple math question. The answer is 144. Let me finalize."
    in_memory_runtime["llm"]._scripted_streams.append(_text_turn_stream("144"))
    in_memory_runtime["llm"]._scripted_streams.append(_meta_plus_finalize_stream(meta))

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="12*12?")])
    events = [evt async for evt in engine.run(user_msg)]

    # The answer was followed by a forced Finalize request, not a nudge.
    reasons = [
        e.payload.get("reason") for e in events if e.type is EventType.STATE_CHANGED
    ]
    assert "terminal_tool_forced" in reasons
    # Two LLM turns happened (answer + terminal-only).
    assert len(in_memory_runtime["llm"].calls) == 2
    # Finalize actually ran (background gate executed).
    assert len(finalize_tool.calls) == 1

    streamed = _streamed_text(events)
    durable = _history_text(engine)
    # The real answer survives both live and durable.
    assert "144" in streamed
    assert "144" in durable
    # The META narration is NOWHERE — not on the wire, not in history.
    assert "The user asked" not in streamed
    assert "Let me finalize" not in streamed
    assert "The user asked" not in durable
    assert "Let me finalize" not in durable
    assert engine.state is LoopState.COMPLETED


@pytest.mark.asyncio
async def test_blocker_prose_no_write_still_triggers_write(
    engine_factory, in_memory_runtime
) -> None:
    """"Создай файл index.html" answered with prose and 0 tools: the claim is
    sealed by a forced Finalize, the Finalize that declares the missing file
    is refused, and the refusal gets the file actually written. Suppression
    never hides the answer written after the work."""
    rc = _finalize_rc()
    engine = engine_factory(rc=rc, expected_terminal_tool="Finalize")
    write_tool = _RecordingWriteTool()

    class _CheckingFinalize(_BackgroundFinalizeTool):
        async def invoke(self, context, arguments):  # type: ignore[no-untyped-def]
            if not write_tool.calls:
                self.calls.append(dict(arguments))
                return ToolResult(
                    tool_call_id="",
                    content="index.html was declared but does not exist",
                    is_error=True,
                    metadata={TERMINAL_REFUSAL_NEEDS_WORK_METADATA_KEY: True},
                )
            return await super().invoke(context, arguments)

    finalize_tool = _CheckingFinalize()
    in_memory_runtime["tools"].register(write_tool)
    in_memory_runtime["tools"].register(finalize_tool)
    llm = in_memory_runtime["llm"]

    # Turn 1: prose claiming the file was created, but ZERO tools.
    llm._scripted_streams.append(_text_turn_stream("Done, I created index.html."))
    # Turn 2 (forced Finalize): the declared file is missing, so it is refused.
    llm.queue_tool_call_response(
        tool_call_id="fin-1",
        tool_name="Finalize",
        tool_input={"declared_deliverables": [{"path": "index.html"}]},
    )
    # Turn 3 (a tool is required): the model writes the file.
    llm.queue_tool_call_response(
        tool_call_id="w-1",
        tool_name="Write",
        tool_input={"path": "index.html", "content": "<html></html>"},
    )
    # Turn 4 (free again): the answer; turn 5 (forced): the seal.
    llm._scripted_streams.append(_text_turn_stream("index.html is written."))
    llm.queue_tool_call_response(
        tool_call_id="fin-2",
        tool_name="Finalize",
        tool_input={"declared_deliverables": [{"path": "index.html"}]},
    )

    user_msg = Message(
        role=MessageRole.user,
        content_blocks=[TextBlock(text="Создай файл index.html")],
    )
    events = [evt async for evt in engine.run(user_msg)]

    # THE BLOCKER ASSERTION: the file actually got written.
    assert len(write_tool.calls) == 1
    assert write_tool.calls[0]["path"] == "index.html"
    assert "index.html is written." in _streamed_text(events)
    # Finalize was refused once, then sealed the run.
    assert len(finalize_tool.calls) == 2
    assert engine.state is LoopState.COMPLETED


@pytest.mark.asyncio
async def test_genuinely_empty_answer_surfaces_terminal_turn_text(
    engine_factory, in_memory_runtime
) -> None:
    """No prior answer → the terminal-only turn's text IS the answer → it MUST
    stay visible (suppression returns False when no prior answer exists). The
    nudge fires; the answer the model emits in the terminal turn is preserved."""
    rc = _finalize_rc()
    engine = engine_factory(rc=rc, expected_terminal_tool="Finalize")
    finalize_tool = _BackgroundFinalizeTool()
    in_memory_runtime["tools"].register(finalize_tool)

    # Turn 1: NO substantive prose (empty assistant turn) → end_turn.
    in_memory_runtime["llm"]._scripted_streams.append(
        [
            LLMStreamEvent(name="message_start", payload={}),
            LLMStreamEvent(
                name="message_stop",
                payload={"stop_reason": StopReason.end_turn.value},
            ),
        ]
    )
    # Turn 2 (post-nudge): the model finally answers IN the terminal turn +
    # Finalize. This text is the ONLY answer → must be visible.
    answer = "Привет! Чем могу помочь?"
    in_memory_runtime["llm"]._scripted_streams.append(_meta_plus_finalize_stream(answer))

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="привет")])
    events = [evt async for evt in engine.run(user_msg)]

    reasons = [
        e.payload.get("reason") for e in events if e.type is EventType.STATE_CHANGED
    ]
    assert "terminal_tool_nudge" in reasons
    # The terminal-turn answer is preserved live + durable (NOT suppressed).
    assert answer in _streamed_text(events)
    assert answer in _history_text(engine)
    assert len(finalize_tool.calls) == 1
    assert engine.state is LoopState.COMPLETED


def _answer_plus_finalize_stream(text: str, call_id: str) -> list[LLMStreamEvent]:
    """``_meta_plus_finalize_stream`` with a caller-chosen Finalize call id.

    A second turn on the same engine must not reuse the first turn's tool call
    id — the outbound pairing repair keys on it."""
    return [
        LLMStreamEvent(name="message_start", payload={}),
        LLMStreamEvent(name="content_block_start", payload={"kind": "text"}),
        LLMStreamEvent(
            name="content_block_delta", payload={"text": text, "kind": "text"}
        ),
        LLMStreamEvent(name="content_block_stop", payload={}),
        LLMStreamEvent(
            name="tool_use_start",
            payload={"tool_call_id": call_id, "tool_name": "Finalize"},
        ),
        LLMStreamEvent(
            name="tool_use_input_delta",
            payload={"tool_call_id": call_id, "partial_input_json": "{}"},
        ),
        LLMStreamEvent(
            name="tool_use_stop", payload={"tool_call_id": call_id, "final_input": {}}
        ),
        LLMStreamEvent(
            name="message_stop", payload={"stop_reason": StopReason.tool_use.value}
        ),
    ]


@pytest.mark.asyncio
async def test_a_second_turn_answer_is_not_dropped_as_narration(
    engine_factory, in_memory_runtime
) -> None:
    """A turn that follows a nudged turn delivers its answer.

    The suppression above is scoped to the turn the nudge fired in: it exists
    to drop the redundant self-narration a weak model co-locates with the
    terminal call once it has ALREADY answered. Nothing about a later turn is
    narration — its text is the answer to a question the first turn never saw.

    Measured, not supposed: the first end-to-end run of a turn opened on an
    engine that had been nudged produced two model calls and no assistant
    message at all. The answer was written, it streamed, and it was deleted
    before anything could select it, because the latch that says "we are in the
    post-nudge terminal-only turn" outlived that turn."""
    rc = _finalize_rc()
    engine = engine_factory(rc=rc, expected_terminal_tool="Finalize")
    finalize_tool = _BackgroundFinalizeTool()
    in_memory_runtime["tools"].register(finalize_tool)

    # Turn 1 — the shape that arms the latch: an answer, then the nudge, then
    # the terminal-only turn whose narration is (correctly) suppressed.
    in_memory_runtime["llm"]._scripted_streams.append(_text_turn_stream("144"))
    in_memory_runtime["llm"]._scripted_streams.append(
        _meta_plus_finalize_stream("The user asked. The answer is 144. Let me finalize.")
    )
    async for _ in engine.run(
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="12*12?")])
    ):
        pass
    assert engine.state is LoopState.COMPLETED
    assert "144" in _history_text(engine)

    # Turn 2 — a new question on the same engine, answered in one message.
    # COMPLETED has no outgoing edge, so a caller that opens another turn on a
    # finished run puts the phase back itself; assigning it here is that caller,
    # not a shortcut around the state machine.
    engine.state = LoopState.RUNNING
    second_answer = "13*13 is 169."
    in_memory_runtime["llm"]._scripted_streams.append(
        _answer_plus_finalize_stream(second_answer, "fin-2")
    )
    second_events = [
        evt
        async for evt in engine.run(
            Message(role=MessageRole.user, content_blocks=[TextBlock(text="13*13?")])
        )
    ]

    # The answer reached the reader's stream AND durable history.
    assert second_answer in _streamed_text(second_events)
    assert second_answer in _history_text(engine)
    assert len(finalize_tool.calls) == 2
    assert engine.state is LoopState.COMPLETED


@pytest.mark.asyncio
async def test_a_new_turn_starts_with_the_terminal_only_latch_down(
    engine_factory, in_memory_runtime
) -> None:
    """``run()`` clears the latch at the turn boundary.

    Stated without the loop around it, so a future reader can see the rule
    itself rather than infer it from a scenario. No terminal tool is configured
    here, so the nudge cannot fire and re-arm: the latch can only be down
    afterwards because opening a turn put it down."""
    rc = _finalize_rc()
    engine = engine_factory(rc=rc, expected_terminal_tool=None)
    engine._terminal_only_active = True

    in_memory_runtime["llm"]._scripted_streams.append(_text_turn_stream("done"))
    async for _ in engine.run(
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    ):
        pass

    assert engine._terminal_only_active is False


@pytest.mark.asyncio
async def test_the_exported_turn_iterator_also_starts_with_the_latch_down(
    engine_factory, in_memory_runtime
) -> None:
    """The rule above belongs to the turn, not to one way of opening it.

    :func:`~protocore.runtime.query.query` is exported from
    ``protocore.runtime`` and drives the same private generator as
    ``QueryEngine.run``. It admits an already-prepared turn, so its caller
    supplies the input message and the turn number — but the per-turn state a
    turn must not inherit is private to the engine, so a caller has no way to
    lower this latch even if it knew it had to. Same setup as the test above:
    no terminal tool, so the nudge cannot fire and re-arm."""
    rc = _finalize_rc()
    engine = engine_factory(rc=rc, expected_terminal_tool=None)
    engine._terminal_only_active = True

    in_memory_runtime["llm"]._scripted_streams.append(_text_turn_stream("done"))
    engine.history.append(
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    )
    engine.turn_count += 1
    async for _ in query(engine):
        pass

    assert engine._terminal_only_active is False


@pytest.mark.asyncio
async def test_a_second_turn_opened_through_the_exported_iterator_answers(
    engine_factory, in_memory_runtime
) -> None:
    """The end-to-end shape of the regression, on the exported entry.

    Identical to
    :func:`test_a_second_turn_answer_is_not_dropped_as_narration` except that
    the second turn is opened through :func:`~protocore.runtime.query.query`
    instead of ``QueryEngine.run``. Both reach the same loop; only one used to
    lower the latch on the way in, so a turn opened through the exported
    iterator on a nudged engine began already finalising and deleted its own
    answer as post-answer narration."""
    rc = _finalize_rc()
    engine = engine_factory(rc=rc, expected_terminal_tool="Finalize")
    finalize_tool = _BackgroundFinalizeTool()
    in_memory_runtime["tools"].register(finalize_tool)

    # Turn 1 — arms the latch: an answer, then the nudge, then the
    # terminal-only turn whose narration is (correctly) suppressed.
    in_memory_runtime["llm"]._scripted_streams.append(_text_turn_stream("144"))
    in_memory_runtime["llm"]._scripted_streams.append(
        _meta_plus_finalize_stream("The user asked. The answer is 144. Let me finalize.")
    )
    async for _ in engine.run(
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="12*12?")])
    ):
        pass
    assert engine.state is LoopState.COMPLETED
    assert "144" in _history_text(engine)
    # The latch outlived the turn it was armed in — the precondition the
    # second turn has to survive.
    assert engine._terminal_only_active is True

    # Turn 2 — a new question, opened through the exported iterator. The
    # caller does the admission that entry leaves to it: the input message,
    # the turn number, and the phase (COMPLETED has no outgoing edge).
    engine.state = LoopState.RUNNING
    second_answer = "13*13 is 169."
    in_memory_runtime["llm"]._scripted_streams.append(
        _answer_plus_finalize_stream(second_answer, "fin-2")
    )
    engine.history.append(
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="13*13?")])
    )
    engine.turn_count += 1
    second_events = [evt async for evt in query(engine)]

    # The answer reached the reader's stream AND durable history.
    assert second_answer in _streamed_text(second_events)
    assert second_answer in _history_text(engine)
    assert len(finalize_tool.calls) == 2
    assert engine.state is LoopState.COMPLETED
