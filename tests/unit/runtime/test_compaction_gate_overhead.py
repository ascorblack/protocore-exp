"""The compaction gate sizes the whole prompt, not the history alone.

The trigger is a whole-prompt size: the largest prompt the provider accepts,
less a turn's headroom. The gate held the history's estimate against it. On a
deployment whose system prompt and tool definitions take a quarter of the
window — 153 tools and a long system prompt, 56k of a 256k window in the live
case — the history reached the trigger only once the request as a whole had
passed the provider's ceiling. Proactive compaction then never ran first: the
fit shrank the output cap turn after turn, and compaction came only after a
refusal.
"""
from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest

from protocore.contracts.llm import LLMRequest, LLMResponse, LLMStreamEvent
from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.tool_registry import ToolVisibilityPolicy
from protocore.contracts.tools import Tool, ToolContext
from protocore.contracts.types import (
    Message,
    MessageRole,
    StopReason,
    TextBlock,
    ToolDefinition,
    ToolParameterSchema,
    ToolResult,
)
from protocore.runtime.context.budgets import derive_budgets
from protocore.runtime.context.manager import ContextManager
from protocore.runtime.query_engine import QueryEngine, QueryEngineConfig
from protocore.runtime.tool_registry import ToolRegistry
from protocore.tests_support.adapters import (
    InMemoryBlobStore,
    InMemoryEventStream,
    InMemoryHookManager,
    InMemorySkillStore,
)
from tests._fixtures.tool_roles import CONVENTIONAL_TOOL_ROLES


class _WideTool(Tool):
    """A tool whose definition is long, the way a real deployment's are."""

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self._name,
            description=f"The {self._name} tool. " + "It does one careful thing. " * 60,
            parameters=ToolParameterSchema(properties={"x": {"type": "string"}}),
        )

    async def invoke(self, context: ToolContext, arguments: dict[str, Any]) -> ToolResult:
        return ToolResult(tool_call_id="", content="ok", is_error=False)


class _OneCallThenAnswer:
    def __init__(self) -> None:
        self.calls: list[LLMRequest] = []

    async def stream_with_tools(self, request: LLMRequest) -> AsyncIterator[LLMStreamEvent]:
        self.calls.append(request)
        yield LLMStreamEvent(name="message_start", payload={})
        if len(self.calls) == 1:
            yield LLMStreamEvent(
                name="tool_use_start", payload={"tool_call_id": "call-1", "tool_name": "Tool0"}
            )
            yield LLMStreamEvent(
                name="tool_use_input_delta",
                payload={"tool_call_id": "call-1", "partial_input_json": json.dumps({"x": "a"})},
            )
            yield LLMStreamEvent(
                name="tool_use_stop", payload={"tool_call_id": "call-1", "final_input": {"x": "a"}}
            )
            yield LLMStreamEvent(name="message_stop", payload={"stop_reason": "tool_use"})
            return
        yield LLMStreamEvent(name="content_block_delta", payload={"text": "done"})
        yield LLMStreamEvent(name="message_stop", payload={"stop_reason": "end_turn"})

    async def complete_structured(self, request, schema):  # type: ignore[no-untyped-def]
        return LLMResponse(
            message=Message(role=MessageRole.assistant, content_blocks=[]),
            stop_reason=StopReason.end_turn,
        )

    def count_tokens(self, text, model=None) -> int:  # type: ignore[no-untyped-def]
        return max(1, len(text) // 4)


def _rc() -> LoopConstants:
    return LoopConstants(
        model_context_window=32_000,
        llm_output_max_tokens_ratio=0.25,
        compaction_trigger_ratio=0.59,
    )


def _history(tokens: int) -> list[Message]:
    """Earlier turns of plain prose, about ``tokens`` in all."""
    history: list[Message] = []
    while sum(len(m.text) for m in history) < tokens * 4:
        history.append(
            Message(role=MessageRole.user, content_blocks=[TextBlock(text="a question " * 40)])
        )
        history.append(
            Message(role=MessageRole.assistant, content_blocks=[TextBlock(text="an answer " * 40)])
        )
    return history


def test_the_gate_adds_what_the_history_does_not_carry() -> None:
    rc = _rc()
    trigger = derive_budgets(rc).compaction_trigger_tokens
    manager = ContextManager(rc=rc, blob_store=InMemoryBlobStore(), compaction_llm=None)  # type: ignore[arg-type]
    history = _history(trigger // 2)

    assert not manager.needs_compaction(history)
    assert manager.needs_compaction(history, overhead_tokens=trigger)
    assert not manager.needs_emergency_compaction(history, overhead_tokens=0)


@pytest.mark.asyncio
async def test_after_a_request_the_engine_gate_counts_the_system_prompt_and_tools() -> None:
    # The gate between iterations would now compact this history — the floor
    # removes what the summariser cannot — and the claim here is about the
    # measurement the gate reads, so the gate itself is kept out of the run.
    rc = _rc().model_copy(update={"compaction_per_iteration_enabled": False})
    trigger = derive_budgets(rc).compaction_trigger_tokens
    tools = [_WideTool(f"Tool{n}") for n in range(20)]
    llm = _OneCallThenAnswer()
    engine = QueryEngine(
        config=QueryEngineConfig(
            tool_roles=CONVENTIONAL_TOOL_ROLES,
            run_id="run-gate",
            tenant_id="tenant-gate",
            session_id="sess-gate",
            model_name="m",
            rc=rc,
            system_prompt_sections=("You are an agent. " * 200,),
            tool_visibility_policy=ToolVisibilityPolicy(pinned={t.name for t in tools}),
        ),
        llm_provider=llm,  # type: ignore[arg-type]
        tool_registry=ToolRegistry(tools),  # type: ignore[arg-type]
        event_stream=InMemoryEventStream(),
        hook_manager=InMemoryHookManager(),
        skill_store=InMemorySkillStore(),
        blob_store=InMemoryBlobStore(),
    )
    engine.history = _history(trigger * 2 // 3)
    assert not engine.needs_compaction()  # nothing sent yet: the history alone

    async for _ in engine.run(
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])
    ):
        pass

    history_only = engine.context_manager.current_prompt_tokens(engine.history)
    assert history_only < trigger
    assert engine.request_overhead_tokens() > trigger - history_only
    assert engine.needs_compaction()
    assert engine.snapshot()["request_overhead_tokens_raw"] > 0
