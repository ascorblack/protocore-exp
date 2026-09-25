"""Terminal-answer gate for a ``Finalize`` tool.

Core owns only the *mechanism* (the typed ``Finalize`` tool + its event ship in
a host-side Finalize tool). This proves the universal terminal-tool gate enforces
the §A.6 ``Finalize`` contract name end-to-end:

* a prose final attempt WITHOUT a prior ``Finalize`` is rejected/repaired — the
  loop injects the terminal nudge and re-drives, and the run only completes once
  ``Finalize`` returns a terminal result (mirrors stand Case D ``ru_file`` →
  "precondition gate forces ``Finalize``", ``leaked_prose_contract=False``);
* an analytic answer that goes straight to ``Finalize`` completes immediately
  (Case D ``ru_analytic``).

Wired via ``QueryEngineConfig.expected_terminal_tool="Finalize"`` +
``rc.terminal_tool_nudge_enabled``.
"""
from __future__ import annotations

from typing import Any

import pytest

from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.tools import ToolContext
from protocore.contracts.types import (
    SESSION_HISTORY_SEED_METADATA_KEY,
    SYNTHETIC_RECOVERY_METADATA_KEY,
    SYNTHETIC_RECOVERY_PROSE_GATE_REPAIR,
    TERMINAL_TOOL_METADATA_KEY,
    TERMINAL_TOOL_STATUS_COMPLETED,
    TERMINAL_TOOL_STATUS_METADATA_KEY,
    Message,
    MessageRole,
    TextBlock,
    ToolCall,
    ToolResult,
    ToolResultBlock,
    ToolUseBlock,
)
from protocore.runtime import soft_stop as _soft_stop
from protocore.runtime.loop_state import LoopState
from protocore.runtime.query import (
    _finalize_prose_gate_applies,
    _has_visible_assistant_prose_after_work,
    _history_has_terminal_tool_result,
    _is_non_terminal_tool_activity,
    _is_terminal_tool_name,
    _prose_gate_just_injected,
    _resolved_terminal_tool_name,
    _terminal_only_blocks,
    _terminal_only_enforced,
    _terminal_only_error_message,
    _terminal_tool_carries_answer_field,
    _terminal_tool_nudge_required,
)
from protocore.runtime.query_engine import QueryEngine, QueryEngineConfig
from protocore.tests_support.adapters import (
    InMemoryBlobStore,
    InMemoryEventStream,
    InMemoryHookManager,
    InMemoryLLMProvider,
    InMemorySkillStore,
    InMemoryToolRegistry,
)
from tests._fixtures.tool_roles import CONVENTIONAL_TOOL_ROLES

from ._tool_fixtures import MockTool


class _FinalizeTool(MockTool):
    """Stand-in for a host's ``Finalize`` tool — returns a terminal result.

    The real tool records the typed ``declared_deliverables`` contract +
    emits a typed event; here we only need the terminal-metadata flag so the
    core gate recognises a satisfied finalisation. No prose contract is ever
    produced (the answer flows through the tool, not raw text)."""

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


def _build_engine(
    *,
    llm: InMemoryLLMProvider,
    registry: InMemoryToolRegistry,
    rc: LoopConstants,
) -> QueryEngine:
    return QueryEngine(
        config=QueryEngineConfig(
            tool_roles=CONVENTIONAL_TOOL_ROLES,
            run_id="run-finalize",
            tenant_id="tenant-test",
            session_id="sess-test",
            model_name="qwen3.6-35b-a3b",
            rc=rc,
            expected_terminal_tool="Finalize",
        ),
        llm_provider=llm,
        tool_registry=registry,
        event_stream=InMemoryEventStream(),
        hook_manager=InMemoryHookManager(),
        skill_store=InMemorySkillStore(),
        blob_store=InMemoryBlobStore(),
    )


# ---------------------------------------------------------------------------
# Mechanism — the generic gate resolves + enforces the "Finalize" name
# ---------------------------------------------------------------------------


def test_finalize_gate_requires_nudge_before_finalize() -> None:
    rc = LoopConstants(model_context_window=4_096, terminal_tool_nudge_enabled=True)
    engine = _build_engine(
        llm=InMemoryLLMProvider(), registry=InMemoryToolRegistry(), rc=rc
    )
    assert _resolved_terminal_tool_name(engine) == "Finalize"
    # No Finalize result in history yet → the gate requires the nudge.
    assert _history_has_terminal_tool_result(engine) is False
    assert _terminal_tool_nudge_required(engine) is True


def test_finalize_terminal_only_blocks_non_finalize_after_latch() -> None:
    rc = LoopConstants(model_context_window=4_096, terminal_tool_nudge_enabled=True)
    engine = _build_engine(
        llm=InMemoryLLMProvider(), registry=InMemoryToolRegistry(), rc=rc
    )
    # Simulate the wind-down having started (the deadline was reached).
    _soft_stop.enter(engine, cause_name=_soft_stop.CAUSE_DEADLINE)
    engine._terminal_only_active = True
    assert _terminal_only_enforced(engine) is True

    prose_via_other_tool = ToolCall(name="Write", arguments={"path": "x"})
    finalize = ToolCall(name="Finalize", arguments={})
    # A non-Finalize tool is BLOCKED with a visible, actionable error.
    assert _terminal_only_blocks(engine, prose_via_other_tool) is True
    msg = _terminal_only_error_message(engine, prose_via_other_tool.name)
    assert "Write" in msg
    assert "Finalize" in msg
    # Finalize itself is the single allowed tool.
    assert _terminal_only_blocks(engine, finalize) is False


# ---------------------------------------------------------------------------
# Full loop — Case D parity (prose → gate forces Finalize; analytic → direct)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prose_attempt_is_repaired_into_finalize() -> None:
    """ru_file parity — the model proses to end without Finalize → the
 precondition gate forces a Finalize round-trip → the run completes via a
 TYPED Finalize result, never a leaked prose contract.

 ``finalize_prose_gate_enabled=False`` isolates the terminal-tool NUDGE
 mechanism under test here from the prose-gate (the model
 here proses, then calls a payload-only Finalize on the re-drive, which the
 prose-gate would otherwise also intercept — that path is covered by its own
 tests below)."""
    rc = LoopConstants(
        model_context_window=4_096,
        terminal_tool_nudge_enabled=True,
        finalize_prose_gate_enabled=False,
    )
    registry = InMemoryToolRegistry()
    finalize_tool = _FinalizeTool(tool_name="Finalize", description="Finish the run")
    registry.register(finalize_tool)
    registry.register(MockTool(tool_name="Write", description="Write a file"))

    from protocore.contracts.llm import LLMStreamEvent
    from protocore.contracts.types import StopReason

    llm = InMemoryLLMProvider()
    # call#1 — the model proses a final answer WITHOUT calling Finalize. Scripted
    # directly (NOT ``queue_response``) so it is observed as the FIRST turn:
    # ``stream_with_tools`` drains ``_scripted_streams`` in order, and
    # ``queue_tool_call_response`` ALSO appends to ``_scripted_streams`` — mixing
    # ``queue_response`` (a separate ``_queue``) with it would let the scripted
    # Finalize stream be popped FIRST, so the prose turn would never drive the
    # nudge (the prior stale form of this test). Driving both turns through
    # ``_scripted_streams`` proves the real prose → nudge → Finalize sequence.
    llm._scripted_streams.append(
        [
            LLMStreamEvent(name="message_start", payload={}),
            LLMStreamEvent(name="content_block_start", payload={"kind": "text"}),
            LLMStreamEvent(
                name="content_block_delta",
                payload={"text": "Here is your landing page: <html>...</html>", "kind": "text"},
            ),
            LLMStreamEvent(name="content_block_stop", payload={}),
            LLMStreamEvent(
                name="message_stop", payload={"stop_reason": StopReason.end_turn.value}
            ),
        ]
    )
    # call#2 (after the forced nudge) — the model calls Finalize properly.
    llm.queue_tool_call_response(
        tool_call_id="toolu_finalize",
        tool_name="Finalize",
        tool_input={
            "declared_deliverables": [
                {
                    "path": "index.html",
                    "kind": "file",
                    "required": True,
                    "summary": "the landing page",
                }
            ],
        },
    )

    engine = _build_engine(llm=llm, registry=registry, rc=rc)
    initial = Message(
        role=MessageRole.user,
        content_blocks=[TextBlock(text="напиши лендинг для кофейни")],
    )
    events = [evt async for evt in engine.run(initial)]

    # The forcing started BEFORE the Finalize turn (proves the prose attempt was the
    # first turn and did NOT complete the run on its own).
    reasons = [
        e.payload.get("reason")
        for e in events
        if e.type.value == "state_changed"
    ]
    assert "terminal_tool_forced" in reasons
    assert len(llm.calls) == 2, "prose turn + forced Finalize turn"
    # The second request names Finalize natively and adds nothing to the
    # transcript: the delivered answer is its last message.
    assert llm.calls[1].extra.get("forced_tool_choice") == "Finalize"
    assert llm.calls[1].messages[-1].role is MessageRole.assistant
    # The gate forced the Finalize round-trip.
    assert finalize_tool.calls, "the gate must force a Finalize call"
    assert engine.state is LoopState.COMPLETED

    # The run completed through a TYPED terminal Finalize result (no prose
    # contract leaked — the answer is the tool result, not raw model text).
    terminal_results = [
        block
        for message in engine.history
        for block in message.content_blocks
        if isinstance(block, ToolResultBlock)
        and block.metadata.get(TERMINAL_TOOL_METADATA_KEY) is True
    ]
    assert terminal_results, "a terminal Finalize result must be in history"
    assert _history_has_terminal_tool_result(engine) is True
    # The real prose answer (turn 1) survives — it was NOT a co-located meta turn,
    # so the prose-suppression logic does not touch it; the Finalize turn carried no text.
    assert any(
        isinstance(block, TextBlock) and "landing page" in block.text
        for message in engine.history
        for block in message.content_blocks
    )


@pytest.mark.asyncio
async def test_analytic_answer_finalizes_directly() -> None:
    """ru_analytic parity — the model calls Finalize straight away (no files,
 empty ``declared_deliverables``) and the run completes immediately with no
 nudge re-drive.

 ``finalize_prose_gate_enabled=False`` isolates the analytic-direct path: a
 truly prose-less analytic Finalize is exactly what the prose-gate intercepts (covered by its own tests below); here we assert the
 PRE-prose-gate direct-finalize behaviour."""
    rc = LoopConstants(
        model_context_window=4_096,
        terminal_tool_nudge_enabled=True,
        finalize_prose_gate_enabled=False,
    )
    registry = InMemoryToolRegistry()
    finalize_tool = _FinalizeTool(tool_name="Finalize", description="Finish the run")
    registry.register(finalize_tool)

    llm = InMemoryLLMProvider()
    llm.queue_tool_call_response(
        tool_call_id="toolu_finalize",
        tool_name="Finalize",
        tool_input={"declared_deliverables": [], "answer": "42 is the answer"},
    )
    # Guard: a follow-up text response should NOT be consumed (Finalize is
    # terminal and closes the loop immediately).
    llm.queue_response(text="should not be reached")

    engine = _build_engine(llm=llm, registry=registry, rc=rc)
    initial = Message(
        role=MessageRole.user,
        content_blocks=[TextBlock(text="сколько будет 6 умножить на 7?")],
    )
    async for _ in engine.run(initial):
        pass

    assert finalize_tool.calls == [
        {"declared_deliverables": [], "answer": "42 is the answer"}
    ]
    # The terminal Finalize closed the loop on the FIRST call (no nudge,
    # no second LLM round-trip).
    assert len(llm.calls) == 1
    assert engine.state is LoopState.COMPLETED


# ---------------------------------------------------------------------------
#  — prose-gate (background terminal must have visible prose)
# ---------------------------------------------------------------------------


def _build_prose_gate_engine(
    *, llm: InMemoryLLMProvider, registry: InMemoryToolRegistry, **rc_kwargs: Any
) -> QueryEngine:
    rc = LoopConstants(
        model_context_window=4_096,
        terminal_tool_nudge_enabled=False,
        **rc_kwargs,
    )
    return _build_engine(llm=llm, registry=registry, rc=rc)


def test_prose_gate_predicate_payload_only_vs_prose_after_work() -> None:
    """The prose predicate: prose AFTER the latest non-terminal work counts;
    prose only BEFORE the work (or none) does not."""
    rc = LoopConstants(model_context_window=4_096)
    engine = _build_engine(
        llm=InMemoryLLMProvider(), registry=InMemoryToolRegistry(), rc=rc
    )
    min_chars = rc.finalize_prose_gate_min_chars
    long_prose = "x" * (min_chars + 20)

    # Shape A — work tool result, then NO prose (payload-only terminal): the
    # predicate is False (no substantive prose after work) → gate would fire.
    engine.history = [
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="q")]),
        Message(
            role=MessageRole.assistant,
            content_blocks=[
                ToolUseBlock(tool_call_id="t1", name="Read", arguments_json="{}")
            ],
        ),
        Message(
            role=MessageRole.tool,
            content_blocks=[
                ToolResultBlock(tool_call_id="t1", content="file body", is_error=False)
            ],
        ),
    ]
    assert (
        _has_visible_assistant_prose_after_work(engine, "Finalize", min_chars)
        is False
    )

    # Shape B — append substantive assistant prose AFTER the work: True.
    engine.history.append(
        Message(role=MessageRole.assistant, content_blocks=[TextBlock(text=long_prose)])
    )
    assert (
        _has_visible_assistant_prose_after_work(engine, "Finalize", min_chars)
        is True
    )


def test_prose_gate_short_prose_below_floor_does_not_satisfy() -> None:
    """Prose shorter than ``finalize_prose_gate_min_chars`` does NOT count."""
    rc = LoopConstants(model_context_window=4_096, finalize_prose_gate_min_chars=100)
    engine = _build_engine(
        llm=InMemoryLLMProvider(), registry=InMemoryToolRegistry(), rc=rc
    )
    engine.history = [
        Message(role=MessageRole.assistant, content_blocks=[TextBlock(text="too short")]),
    ]
    assert (
        _has_visible_assistant_prose_after_work(engine, "Finalize", 100) is False
    )


@pytest.mark.asyncio
async def test_payload_only_finalize_triggers_prose_repair_then_finalizes() -> None:
    """A payload-only Finalize (no prose) is VETOED once → the model writes
    prose THEN Finalize on the repair turn → the run completes."""
    registry = InMemoryToolRegistry()
    finalize_tool = _FinalizeTool(tool_name="Finalize", description="Finish the run")
    registry.register(finalize_tool)

    llm = InMemoryLLMProvider()
    # call#1 — payload-only Finalize (NO prose at all).
    llm.queue_tool_call_response(
        tool_call_id="toolu_finalize_1",
        tool_name="Finalize",
        tool_input={"declared_deliverables": []},
    )
    # call#2 — after the prose-gate repair the model writes prose THEN Finalize.
    llm.queue_tool_call_response(
        tool_call_id="toolu_finalize_2",
        tool_name="Finalize",
        tool_input={"declared_deliverables": []},
        text_prefix="Here is the full answer to your question: 42." * 4,
    )

    engine = _build_prose_gate_engine(llm=llm, registry=registry)
    initial = Message(
        role=MessageRole.user, content_blocks=[TextBlock(text="what is the answer?")]
    )
    async for _ in engine.run(initial):
        pass

    # The first Finalize was VETOED (not dispatched); only the second (after
    # prose) reached the tool. Two LLM round-trips total.
    assert len(llm.calls) == 2
    assert len(finalize_tool.calls) == 1
    assert engine._finalize_prose_gate_used is True
    assert engine.state is LoopState.COMPLETED
    assert _history_has_terminal_tool_result(engine) is True
    # The repair turn is in history as a synthetic prose-gate nudge.
    assert any(
        m.metadata.get(SYNTHETIC_RECOVERY_METADATA_KEY)
        == SYNTHETIC_RECOVERY_PROSE_GATE_REPAIR
        for m in engine.history
    )
    # Substantive visible prose exists after the (vetoed) work.
    assert any(
        m.role is MessageRole.assistant
        and any(isinstance(b, TextBlock) and b.text.strip() for b in m.content_blocks)
        for m in engine.history
    )


@pytest.mark.asyncio
async def test_prose_gate_is_one_shot_second_payload_only_finalizes() -> None:
    """The gate fires AT MOST once: if the model emits a SECOND payload-only
    Finalize after the repair, the run finalizes (no infinite loop)."""
    registry = InMemoryToolRegistry()
    finalize_tool = _FinalizeTool(tool_name="Finalize", description="Finish the run")
    registry.register(finalize_tool)

    llm = InMemoryLLMProvider()
    # call#1 — payload-only Finalize → vetoed by the prose-gate (one-shot).
    llm.queue_tool_call_response(
        tool_call_id="toolu_finalize_1",
        tool_name="Finalize",
        tool_input={"declared_deliverables": []},
    )
    # call#2 — STILL payload-only (model ignored the repair): the latch is
    # already spent, so this Finalize dispatches and the run completes.
    llm.queue_tool_call_response(
        tool_call_id="toolu_finalize_2",
        tool_name="Finalize",
        tool_input={"declared_deliverables": []},
    )
    # Guard — a third response must NOT be consumed (the loop ends on call#2).
    llm.queue_response(text="should not be reached")

    engine = _build_prose_gate_engine(llm=llm, registry=registry)
    initial = Message(
        role=MessageRole.user, content_blocks=[TextBlock(text="q")]
    )
    async for _ in engine.run(initial):
        pass

    assert len(llm.calls) == 2  # exactly one repair re-drive, then finalize
    assert len(finalize_tool.calls) == 1  # only the second Finalize dispatched
    assert engine._finalize_prose_gate_used is True
    assert engine.state is LoopState.COMPLETED


@pytest.mark.asyncio
async def test_prose_then_finalize_does_not_trigger_gate() -> None:
    """The healthy shape (prose in the SAME turn as Finalize) finalizes on the
    first call — the gate does NOT fire."""
    registry = InMemoryToolRegistry()
    finalize_tool = _FinalizeTool(tool_name="Finalize", description="Finish the run")
    registry.register(finalize_tool)

    llm = InMemoryLLMProvider()
    llm.queue_tool_call_response(
        tool_call_id="toolu_finalize_1",
        tool_name="Finalize",
        tool_input={"declared_deliverables": []},
        text_prefix="Here is the complete final answer for the user. " * 3,
    )
    llm.queue_response(text="should not be reached")

    engine = _build_prose_gate_engine(llm=llm, registry=registry)
    initial = Message(role=MessageRole.user, content_blocks=[TextBlock(text="q")])
    async for _ in engine.run(initial):
        pass

    assert len(llm.calls) == 1
    assert len(finalize_tool.calls) == 1
    assert engine._finalize_prose_gate_used is False
    assert engine.state is LoopState.COMPLETED


@pytest.mark.asyncio
async def test_short_valid_answer_with_default_floor_does_not_trigger_gate() -> None:
    """Regression (chat finalize meta-leak, 2026-06-16): a terse-but-complete
    answer (e.g. ``144``) emitted in the SAME turn as a payload-only Finalize
    must NOT be vetoed/duplicated under the DEFAULT
    ``finalize_prose_gate_min_chars`` (now 1, decoupled from the 100-char
    analytic floor). The 100-char floor previously mis-classified ``144`` as
    'no prose' → veto → repair → the model re-emitted the identical short
    answer (the duplicate bubble). With the floor at 1 the gate sees the answer
    and finalizes on the FIRST call."""
    registry = InMemoryToolRegistry()
    finalize_tool = _FinalizeTool(tool_name="Finalize", description="Finish the run")
    registry.register(finalize_tool)

    llm = InMemoryLLMProvider()
    # ONE turn: a short real answer (`144`) + a payload-only Finalize together.
    llm.queue_tool_call_response(
        tool_call_id="toolu_finalize_1",
        tool_name="Finalize",
        tool_input={"declared_deliverables": []},
        text_prefix="144",
    )
    llm.queue_response(text="should not be reached")

    # NO finalize_prose_gate_min_chars override → exercises the new default (1).
    engine = _build_prose_gate_engine(llm=llm, registry=registry)
    assert engine.config.rc.finalize_prose_gate_min_chars == 1
    initial = Message(
        role=MessageRole.user, content_blocks=[TextBlock(text="12 x 12?")]
    )
    async for _ in engine.run(initial):
        pass

    # Gate never fired: a single LLM round-trip, a single Finalize, no repair,
    # no re-emit → no duplicate answer bubble.
    assert len(llm.calls) == 1
    assert len(finalize_tool.calls) == 1
    assert engine._finalize_prose_gate_used is False
    assert engine.state is LoopState.COMPLETED
    # The short answer is the ONLY assistant prose (not duplicated).
    answer_blocks = [
        b.text.strip()
        for m in engine.history
        if m.role is MessageRole.assistant
        for b in m.content_blocks
        if isinstance(b, TextBlock) and b.text.strip()
    ]
    assert answer_blocks == ["144"]
    # No synthetic prose-gate repair turn was injected.
    assert not any(
        m.metadata.get(SYNTHETIC_RECOVERY_METADATA_KEY)
        == SYNTHETIC_RECOVERY_PROSE_GATE_REPAIR
        for m in engine.history
    )


@pytest.mark.asyncio
async def test_prose_gate_disabled_via_rc_finalizes_payload_only() -> None:
    """With ``finalize_prose_gate_enabled=False`` a payload-only Finalize
    finalizes immediately (old behaviour)."""
    registry = InMemoryToolRegistry()
    finalize_tool = _FinalizeTool(tool_name="Finalize", description="Finish the run")
    registry.register(finalize_tool)

    llm = InMemoryLLMProvider()
    llm.queue_tool_call_response(
        tool_call_id="toolu_finalize_1",
        tool_name="Finalize",
        tool_input={"declared_deliverables": []},
    )
    llm.queue_response(text="should not be reached")

    engine = _build_prose_gate_engine(
        llm=llm, registry=registry, finalize_prose_gate_enabled=False
    )
    initial = Message(role=MessageRole.user, content_blocks=[TextBlock(text="q")])
    async for _ in engine.run(initial):
        pass

    assert len(llm.calls) == 1
    assert len(finalize_tool.calls) == 1
    assert engine._finalize_prose_gate_used is False
    assert engine.state is LoopState.COMPLETED


def _background_registry() -> InMemoryToolRegistry:
    """Registry with a BACKGROUND ``Finalize`` (MockTool default schema has NO
    message/answer/text field) so the schema-conditioned gate can fire."""
    registry = InMemoryToolRegistry()
    registry.register(_FinalizeTool(tool_name="Finalize", description="Finish"))
    return registry


def test_prose_gate_applies_predicate_gating() -> None:
    """``_finalize_prose_gate_applies`` honours: RC enable, terminal-tool match,
    one-shot latch, and the BACKGROUND-schema condition."""
    rc = LoopConstants(model_context_window=4_096)
    engine = _build_engine(
        llm=InMemoryLLMProvider(), registry=_background_registry(), rc=rc
    )
    # No prose anywhere + a BACKGROUND terminal schema → the gate fires.
    finalize_call = ToolCall(name="Finalize", arguments={"declared_deliverables": []})
    assert _finalize_prose_gate_applies(engine, finalize_call) is True
    # A non-terminal tool is never intercepted.
    assert _finalize_prose_gate_applies(engine, ToolCall(name="Read", arguments={})) is False
    # Latch spent → no second gate.
    engine._finalize_prose_gate_used = True
    assert _finalize_prose_gate_applies(engine, finalize_call) is False
    engine._finalize_prose_gate_used = False
    # RC disabled → no gate.
    rc_off = LoopConstants(
        model_context_window=4_096, finalize_prose_gate_enabled=False
    )
    engine_off = _build_engine(
        llm=InMemoryLLMProvider(), registry=_background_registry(), rc=rc_off
    )
    assert _finalize_prose_gate_applies(engine_off, finalize_call) is False


# ---------------------------------------------------------------------------
# The gate is SCHEMA-CONDITIONED: a MESSAGE-CARRYING terminal tool
# (answer in its args) and an UNKNOWN-schema terminal tool are EXEMPT, even
# with no assistant prose. Only a BACKGROUND terminal (no answer-carrying
# field) is gated.
# ---------------------------------------------------------------------------


class _MessageCarryingTerminalTool(MockTool):
    """A terminal tool that carries the answer in its ``message`` arg (the
    ``pcm_answer`` / ``final_answer`` / legacy ``Finalize.answer`` shape)."""

    @property
    def definition(self):  # type: ignore[override]
        from protocore.contracts.types import ToolDefinition, ToolParameterSchema

        return ToolDefinition(
            name=self.tool_name,
            description=self.description,
            parameters=ToolParameterSchema(
                properties={"message": {"type": "string"}}, required=["message"]
            ),
        )


def test_message_carrying_terminal_is_exempt_even_without_prose() -> None:
    """A MESSAGE-CARRYING terminal tool (schema declares ``message``)
    is EXEMPT from the prose-gate even when the run has no assistant prose: it
    legitimately answers via its args, so the gate must NOT withhold it."""
    rc = LoopConstants(model_context_window=4_096)
    registry = InMemoryToolRegistry()
    registry.register(
        _MessageCarryingTerminalTool(tool_name="Finalize", description="answer")
    )
    engine = _build_engine(
        llm=InMemoryLLMProvider(), registry=registry, rc=rc
    )
    # Schema carries an answer field → exempt; no prose, but gate does NOT fire.
    assert _terminal_tool_carries_answer_field(engine, "Finalize") is True
    call = ToolCall(name="Finalize", arguments={"message": "the answer"})
    assert _finalize_prose_gate_applies(engine, call) is False


def test_unknown_schema_terminal_is_exempt() -> None:
    """A terminal tool unknown to the core registry (a host-backend
    tool whose contract core does not hold) is EXEMPT for multi-tenant safety
    (we cannot prove it is background)."""
    rc = LoopConstants(model_context_window=4_096)
    engine = _build_engine(
        llm=InMemoryLLMProvider(),
        registry=InMemoryToolRegistry(),  # nothing registered
        rc=rc,
    )
    assert _terminal_tool_carries_answer_field(engine, "Finalize") is True
    call = ToolCall(name="Finalize", arguments={})
    assert _finalize_prose_gate_applies(engine, call) is False


@pytest.mark.asyncio
async def test_message_carrying_payload_only_finalizes_without_repair() -> None:
    """End-to-end — a message-carrying terminal called payload-only
    (no prose) finalizes on the FIRST call (no prose-gate repair re-drive)."""
    registry = InMemoryToolRegistry()

    class _MsgTerminal(_MessageCarryingTerminalTool):
        async def invoke(self, context, arguments):  # type: ignore[override]
            self.calls.append(dict(arguments))
            return ToolResult(
                tool_call_id="",
                content="done",
                is_error=False,
                metadata={
                    TERMINAL_TOOL_METADATA_KEY: True,
                    TERMINAL_TOOL_STATUS_METADATA_KEY: TERMINAL_TOOL_STATUS_COMPLETED,
                },
            )

    terminal = _MsgTerminal(tool_name="Finalize", description="answer")
    registry.register(terminal)

    llm = InMemoryLLMProvider()
    llm.queue_tool_call_response(
        tool_call_id="toolu_1",
        tool_name="Finalize",
        tool_input={"message": "the final answer"},
    )
    llm.queue_response(text="should not be reached")

    engine = _build_prose_gate_engine(llm=llm, registry=registry)
    async for _ in engine.run(
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="q")])
    ):
        pass

    assert len(llm.calls) == 1  # no repair re-drive
    assert terminal.calls == [{"message": "the final answer"}]
    assert engine._finalize_prose_gate_used is False
    assert engine.state is LoopState.COMPLETED


# ---------------------------------------------------------------------------
# — PRIOR-RUN seeded prose must NOT satisfy the current run's gate.
# ---------------------------------------------------------------------------


def test_seeded_prior_run_prose_does_not_satisfy_gate() -> None:
    """assistant prose tagged ``SESSION_HISTORY_SEED_METADATA_KEY``
    (a PRIOR-RUN turn the executor seeded into this run) is SKIPPED, so a
    payload-only terminal in the CURRENT run still trips the gate."""
    rc = LoopConstants(model_context_window=4_096, finalize_prose_gate_min_chars=10)
    engine = _build_engine(
        llm=InMemoryLLMProvider(), registry=_background_registry(), rc=rc
    )
    long_prose = "a complete prior-run answer that is plenty long" * 2
    engine.history = [
        # PRIOR RUN (seeded): Read -> answer prose. Both seed-tagged.
        Message(
            role=MessageRole.assistant,
            content_blocks=[
                ToolUseBlock(tool_call_id="s1", name="Read", arguments_json="{}")
            ],
            metadata={SESSION_HISTORY_SEED_METADATA_KEY: True},
        ),
        Message(
            role=MessageRole.tool,
            content_blocks=[
                ToolResultBlock(tool_call_id="s1", content="body", is_error=False)
            ],
            metadata={SESSION_HISTORY_SEED_METADATA_KEY: True},
        ),
        Message(
            role=MessageRole.assistant,
            content_blocks=[TextBlock(text=long_prose)],
            metadata={SESSION_HISTORY_SEED_METADATA_KEY: True},
        ),
        # CURRENT run: only the new user turn (no current-run prose).
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="new task")]),
    ]
    # The seeded prior-run prose is skipped → no CURRENT-run prose → predicate
    # is False → the gate STILL fires for a payload-only terminal.
    assert (
        _has_visible_assistant_prose_after_work(engine, "Finalize", 10) is False
    )
    finalize_call = ToolCall(name="Finalize", arguments={"declared_deliverables": []})
    assert _finalize_prose_gate_applies(engine, finalize_call) is True


# ---------------------------------------------------------------------------
# — a prose-gate veto on a same-turn [Finalize, <other tool>] shape must
# break the dispatch loop (no sibling dispatch after the corrective user turn).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prose_gate_veto_breaks_same_turn_sibling_dispatch() -> None:
    """when the model emits [Finalize(payload-only), <other tool>] in the
    SAME assistant turn, the prose-gate veto on Finalize must STOP the batch:
    the sibling tool is NOT dispatched after the injected corrective user turn."""
    registry = InMemoryToolRegistry()
    finalize_tool = _FinalizeTool(tool_name="Finalize", description="Finish")
    registry.register(finalize_tool)
    sibling = MockTool(tool_name="Sibling", description="a sibling tool")
    registry.register(sibling)

    llm = InMemoryLLMProvider()
    # call#1 — Finalize (payload-only) FIRST, then a sibling tool in the SAME
    # assistant message. Build the multi-tool stream manually.
    from protocore.contracts.llm import LLMStreamEvent

    llm._scripted_streams.append(
        [
            LLMStreamEvent(name="message_start", payload={}),
            LLMStreamEvent(
                name="tool_use_start",
                payload={"tool_call_id": "toolu_fin", "tool_name": "Finalize"},
            ),
            LLMStreamEvent(
                name="tool_use_input_delta",
                payload={"tool_call_id": "toolu_fin", "partial_input_json": "{}"},
            ),
            LLMStreamEvent(
                name="tool_use_stop",
                payload={"tool_call_id": "toolu_fin", "final_input": {}},
            ),
            LLMStreamEvent(
                name="tool_use_start",
                payload={"tool_call_id": "toolu_sib", "tool_name": "Sibling"},
            ),
            LLMStreamEvent(
                name="tool_use_input_delta",
                payload={"tool_call_id": "toolu_sib", "partial_input_json": "{}"},
            ),
            LLMStreamEvent(
                name="tool_use_stop",
                payload={"tool_call_id": "toolu_sib", "final_input": {}},
            ),
            LLMStreamEvent(
                name="message_stop", payload={"stop_reason": "tool_use"}
            ),
        ]
    )
    # call#2 — after the repair, the model writes prose then Finalize.
    llm.queue_tool_call_response(
        tool_call_id="toolu_fin_2",
        tool_name="Finalize",
        tool_input={"declared_deliverables": []},
        text_prefix="Here is the full final answer for the user, in prose. " * 2,
    )

    engine = _build_prose_gate_engine(llm=llm, registry=registry)
    async for _ in engine.run(
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="q")])
    ):
        pass

    # The sibling tool was NEVER actually dispatched (the veto broke the batch
    # BEFORE it ran) — this is the guarantee. The orphan sibling
    # ``tool_use`` is paired with a synthesized ``Interrupted`` placeholder
    # (wire-validity), NOT a real invocation.
    assert sibling.calls == [], "sibling must not run after the prose-gate veto"
    # Exactly one repair re-drive; Finalize ran once (the 2nd, prose-backed call).
    assert len(llm.calls) == 2
    assert len(finalize_tool.calls) == 1
    assert engine._finalize_prose_gate_used is True
    assert engine.state is LoopState.COMPLETED

    # Pairing-safe ordering (the invariant): the synthetic prose-gate
    # repair USER turn must come AFTER every tool_result of the vetoed assistant
    # batch — never interleaved BETWEEN sibling tool results. Locate the repair
    # turn and assert all batch tool_result rows precede it.
    repair_idx = next(
        i
        for i, m in enumerate(engine.history)
        if m.metadata.get(SYNTHETIC_RECOVERY_METADATA_KEY)
        == SYNTHETIC_RECOVERY_PROSE_GATE_REPAIR
    )
    batch_result_indices = [
        i
        for i, m in enumerate(engine.history[:repair_idx])
        if m.role is MessageRole.tool
        and any(
            isinstance(b, ToolResultBlock) and b.tool_call_id in {"toolu_fin", "toolu_sib"}
            for b in m.content_blocks
        )
    ]
    # Both batch results (sibling-orphan placeholder + vetoed-terminal error)
    # are present and BEFORE the repair turn (no user-turn interleave).
    assert len(batch_result_indices) == 2
    assert max(batch_result_indices) < repair_idx
    # The sibling's only result is a synthesized error placeholder, never a
    # success from an actual dispatch.
    sibling_results = [
        b
        for m in engine.history
        for b in m.content_blocks
        if isinstance(b, ToolResultBlock) and b.tool_call_id == "toolu_sib"
    ]
    assert len(sibling_results) == 1
    assert sibling_results[0].is_error is True


# ---------------------------------------------------------------------------
# Branch coverage — the prose-gate helper edge paths .
# ---------------------------------------------------------------------------


def test_is_terminal_tool_name_strips_prefix_and_exact_matches() -> None:
    """``_is_terminal_tool_name`` strips a ``tool:`` prefix, exact-matches, and
    never matches a non-string or a prefixed sibling (``FinalizeFile``)."""
    assert _is_terminal_tool_name("Finalize", "Finalize") is True
    assert _is_terminal_tool_name("tool:Finalize", "Finalize") is True
    # Exact match only — a longer sibling name must NOT match.
    assert _is_terminal_tool_name("FinalizeFile", "Finalize") is False
    assert _is_terminal_tool_name("tool:FinalizeFile", "Finalize") is False
    # Non-string is never a match.
    assert _is_terminal_tool_name(None, "Finalize") is False
    assert _is_terminal_tool_name(123, "Finalize") is False


def test_is_non_terminal_tool_activity_named_result_branches() -> None:
    """``_is_non_terminal_tool_activity`` — a ToolResultBlock with a ``tool_name``
    metadata that IS the terminal tool is the GATE (not work); a named
    non-terminal tool result IS work; an unnamed terminal-metadata result is the
    gate; a TextBlock is never tool activity."""
    # Named result == the terminal tool → NOT work (the gate).
    term_named = ToolResultBlock(
        tool_call_id="a", content="x", is_error=False, metadata={"tool_name": "Finalize"}
    )
    assert _is_non_terminal_tool_activity(term_named, "Finalize") is False
    # Named non-terminal tool result → IS work.
    work_named = ToolResultBlock(
        tool_call_id="b", content="x", is_error=False, metadata={"tool_name": "Read"}
    )
    assert _is_non_terminal_tool_activity(work_named, "Finalize") is True
    # Unnamed result carrying terminal metadata → the gate (NOT work).
    term_unnamed = ToolResultBlock(
        tool_call_id="c", content="x", is_error=False,
        metadata={TERMINAL_TOOL_METADATA_KEY: True},
    )
    assert _is_non_terminal_tool_activity(term_unnamed, "Finalize") is False
    # Unnamed non-terminal result → visible work.
    work_unnamed = ToolResultBlock(tool_call_id="d", content="x", is_error=False)
    assert _is_non_terminal_tool_activity(work_unnamed, "Finalize") is True
    # A terminal-tool ToolUseBlock is NOT work; a non-terminal one IS.
    assert _is_non_terminal_tool_activity(
        ToolUseBlock(tool_call_id="e", name="Finalize", arguments_json="{}"), "Finalize"
    ) is False
    assert _is_non_terminal_tool_activity(
        ToolUseBlock(tool_call_id="f", name="Read", arguments_json="{}"), "Finalize"
    ) is True
    # A TextBlock is never tool activity.
    assert _is_non_terminal_tool_activity(TextBlock(text="hi"), "Finalize") is False


def test_prose_gate_just_injected_empty_and_non_repair_tail() -> None:
    """``_prose_gate_just_injected`` — False on empty history, False when the
    tail is not the prose-gate repair turn, True when it is."""
    rc = LoopConstants(model_context_window=4_096)
    engine = _build_engine(
        llm=InMemoryLLMProvider(), registry=InMemoryToolRegistry(), rc=rc
    )
    # Empty history → False (covers the empty-history guard).
    engine.history = []
    assert _prose_gate_just_injected(engine) is False
    # A non-repair tail → False (a plain user turn).
    engine.history = [
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="q")])
    ]
    assert _prose_gate_just_injected(engine) is False
    # A user turn WITHOUT the prose-gate tag → False.
    engine.history = [
        Message(
            role=MessageRole.user,
            content_blocks=[TextBlock(text="x")],
            metadata={SYNTHETIC_RECOVERY_METADATA_KEY: "some_other_kind"},
        )
    ]
    assert _prose_gate_just_injected(engine) is False
    # The prose-gate repair turn as tail → True.
    engine.history.append(
        Message(
            role=MessageRole.user,
            content_blocks=[TextBlock(text="write prose first")],
            metadata={SYNTHETIC_RECOVERY_METADATA_KEY: SYNTHETIC_RECOVERY_PROSE_GATE_REPAIR},
        )
    )
    assert _prose_gate_just_injected(engine) is True


def test_terminal_tool_carries_answer_field_no_registry_is_exempt() -> None:
    """``_terminal_tool_carries_answer_field`` — an engine whose ``tools`` has no
    ``.get`` (no usable registry) fails SAFE → exempt (True)."""
    rc = LoopConstants(model_context_window=4_096)
    engine = _build_engine(
        llm=InMemoryLLMProvider(), registry=InMemoryToolRegistry(), rc=rc
    )
    # Simulate a missing/usable-less registry: ``tools`` without ``.get``.
    engine.tools = object()  # type: ignore[assignment]
    assert _terminal_tool_carries_answer_field(engine, "Finalize") is True


def test_prose_gate_disabled_predicate_short_circuits() -> None:
    """``_finalize_prose_gate_applies`` returns False immediately when the RC
    flag is off (covers the enable short-circuit) and when no terminal tool is
    configured."""
    rc_off = LoopConstants(
        model_context_window=4_096, finalize_prose_gate_enabled=False
    )
    engine_off = _build_engine(
        llm=InMemoryLLMProvider(), registry=_background_registry(), rc=rc_off
    )
    call = ToolCall(name="Finalize", arguments={"declared_deliverables": []})
    assert _finalize_prose_gate_applies(engine_off, call) is False

    # No expected_terminal_tool configured → never intercept.
    rc_on = LoopConstants(model_context_window=4_096)
    engine_no_term = QueryEngine(
        config=QueryEngineConfig(
            tool_roles=CONVENTIONAL_TOOL_ROLES,
            run_id="r",
            tenant_id="t",
            session_id="s",
            model_name="m",
            rc=rc_on,
            expected_terminal_tool=None,
        ),
        llm_provider=InMemoryLLMProvider(),
        tool_registry=_background_registry(),
        event_stream=InMemoryEventStream(),
        hook_manager=InMemoryHookManager(),
        skill_store=InMemorySkillStore(),
        blob_store=InMemoryBlobStore(),
    )
    assert _finalize_prose_gate_applies(engine_no_term, call) is False
