"""Live-run guardrails: loop guard, eviction, settled, steer, lease."""
from __future__ import annotations

import json

import pytest

from protocore.constants import PROTOCOL_COMPACTED_TOOL_RESULT_V1
from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.types import (
    Message,
    MessageRole,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from protocore.prompts import bundled_prompt_provider
from protocore.runtime.events import EventType
from protocore.runtime.live_control import (
    enqueue,
    lease_key,
    new_queued_prompt,
    place_items,
    placed_as_user_message,
    restore_queued_prompts,
    validate_thinking_for_mode,
)
from protocore.runtime.loop_guard import (
    canonical_tool_fingerprint,
    identical_tool_should_block,
    inspect_stream_repeat,
    repeating_tail_cut,
)
from protocore.runtime.query import _query as query
from protocore.runtime.result_eviction import evict_history_for_llm
from protocore.tests_support.adapters import InMemoryLLMProvider


def _enabled(**overrides: object) -> LoopConstants:
    values: dict[str, object] = {
        "model_context_window": 4_096,
        "loop_guard_enabled": True,
        "loop_guard_nudge_max": 1,
        "loop_guard_repeat_min_chars": 8,
        "loop_guard_repeat_window_tokens": 4,
        "loop_guard_identical_tool_limit": 2,
        "result_eviction_enabled": True,
        "run_settled_enabled": True,
        "steer_follow_up_enabled": True,
    }
    values.update(overrides)
    return LoopConstants(**values)  # type: ignore[arg-type]


def test_repeating_tail_is_cut_and_not_kept() -> None:
    phrase = "same sentence forever"
    buffer = " ".join([phrase] * 6)
    kept, stripped = repeating_tail_cut(buffer, min_chars=8, window_tokens=3)
    assert stripped > 0
    assert kept.count(phrase) < buffer.count(phrase)
    assert not kept.endswith(phrase + " " + phrase)


def test_inspect_stream_repeat_thinking_only() -> None:
    thought = "looping thought forever"
    reasoning = " ".join([thought] * 6)
    text, reason, hit = inspect_stream_repeat(
        "",
        reasoning,
        _enabled(),
    )
    assert hit is not None
    assert hit.kind == "stream_repeat"
    assert text == ""
    assert reason.count(thought) < reasoning.count(thought)


def test_inspect_stream_repeat_off_is_noop() -> None:
    phrase = "same sentence forever"
    buffer = " ".join([phrase] * 6)
    text, _reason, hit = inspect_stream_repeat(
        buffer,
        "",
        LoopConstants(model_context_window=4096),
    )
    assert hit is None
    assert text == buffer


def test_identical_tool_fingerprint_and_limit() -> None:
    fp = canonical_tool_fingerprint("Bash", {"command": "echo hi"})
    assert fp == canonical_tool_fingerprint("Bash", '{"command":"echo hi"}')
    counts = {fp: 2}
    rc = _enabled(loop_guard_identical_tool_limit=2)
    assert identical_tool_should_block(fp, counts, rc)
    assert not identical_tool_should_block(
        fp, {}, LoopConstants(model_context_window=4096)
    )


def test_eviction_keeps_marked_and_persist_full() -> None:
    pages = []
    history: list[Message] = []
    for idx, keep in ((1, False), (2, True), (3, False)):
        call_id = f"r{idx}"
        pages.append(f"page {idx} body " + "x" * 20)
        history.append(
            Message(
                role=MessageRole.assistant,
                content_blocks=[
                    ToolUseBlock(
                        tool_call_id=call_id,
                        name="Read",
                        arguments_json=json.dumps({"path": f"f{idx}"}),
                    )
                ],
            )
        )
        history.append(
            Message(
                role=MessageRole.tool,
                content_blocks=[
                    ToolResultBlock(
                        tool_call_id=call_id,
                        content=pages[-1],
                        metadata={"keep": keep},
                    )
                ],
            )
        )
    persist = list(history)
    view, evicted = evict_history_for_llm(history, _enabled(), bundled_prompt_provider(), pinned_ids=())
    assert persist[1].content_blocks[0].content == pages[0]
    assert persist[3].content_blocks[0].content == pages[1]
    assert persist[5].content_blocks[0].content == pages[2]
    assert "r1" in evicted and "r3" in evicted
    assert "r2" not in evicted
    assert pages[1] in view[3].content_blocks[0].content
    assert pages[0] not in view[1].content_blocks[0].content
    assert "evicted tool result r1" in view[1].content_blocks[0].content


def test_eviction_leaves_compacted_placeholder() -> None:
    placeholder = f"{PROTOCOL_COMPACTED_TOOL_RESULT_V1}:SNAPSHOT|blob://x|abc|1|t"
    history = [
        Message(
            role=MessageRole.assistant,
            content_blocks=[
                ToolUseBlock(
                    tool_call_id="c1",
                    name="Read",
                    arguments_json="{}",
                )
            ],
        ),
        Message(
            role=MessageRole.tool,
            content_blocks=[ToolResultBlock(tool_call_id="c1", content=placeholder)],
        ),
    ]
    view, evicted = evict_history_for_llm(history, _enabled(), bundled_prompt_provider())
    assert evicted == []
    assert view[1].content_blocks[0].content == placeholder


def test_steer_places_after_tools_follow_up_stays() -> None:
    steer = [new_queued_prompt("steer", "do this now")]
    follow = [new_queued_prompt("follow_up", "later")]
    placed, rest = place_items(steer, "one-at-a-time")
    assert [p.text for p in placed] == ["do this now"]
    assert rest == []
    msg = placed_as_user_message(placed)
    assert msg is not None
    assert "do this now" in msg.text
    still, _ = place_items(follow, "one-at-a-time")
    # follow-up is not inserted until settled — caller decides when to place
    assert still[0].kind == "follow_up"
    with pytest.raises(ValueError, match="steer_follow_up_disabled"):
        enqueue([], new_queued_prompt("steer", "x"), LoopConstants())


def test_deep_rejects_thinking_off() -> None:
    validate_thinking_for_mode("deep", True)
    with pytest.raises(ValueError, match="deep_requires_thinking"):
        validate_thinking_for_mode("deep", False)


def test_lease_key_is_scoped() -> None:
    assert lease_key("scope-a", "sess-1", "main") == "pc:lease:scope-a:sess-1:main"


@pytest.mark.asyncio
async def test_query_stream_repeat_and_settled(
    engine_factory, in_memory_runtime: dict[str, object]
) -> None:
    phrase = "same sentence forever"
    repeating = " ".join([phrase] * 8)
    llm = in_memory_runtime["llm"]
    assert isinstance(llm, InMemoryLLMProvider)
    llm.queue_response(text=repeating)
    engine = engine_factory(rc=_enabled())
    engine.history.append(
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    )
    events = [evt async for evt in query(engine)]
    types = [evt.type for evt in events]
    assert EventType.LOOP_GUARD_FIRED in types
    assert EventType.RUN_SETTLED in types
    settled_idx = types.index(EventType.RUN_SETTLED)
    assert EventType.COMPACTION_STARTED not in types[settled_idx:]
    assistant = [m for m in engine.history if m.role is MessageRole.assistant]
    assert assistant
    stored = assistant[0].text
    assert stored.count(phrase) < repeating.count(phrase)


@pytest.mark.asyncio
async def test_query_compact_then_retry_settled_only_at_end(
    engine_factory, in_memory_runtime: dict[str, object]
) -> None:
    """run_settled must not fire between compaction and the retry stream."""
    llm = in_memory_runtime["llm"]
    assert isinstance(llm, InMemoryLLMProvider)
    llm.queue_response(text="ok after compact")
    rc = _enabled(
        model_context_window=64,
        request_context_safety_tokens=0,
        compaction_trigger_ratio=0.5,
        compaction_keep_recent_turns=1,
    )
    engine = engine_factory(rc=rc)
    user_msg = Message(
        role=MessageRole.user,
        content_blocks=[TextBlock(text="y" * 1024)],
    )
    engine.history.append(user_msg)
    events = [evt async for evt in query(engine)]
    types = [evt.type for evt in events]
    assert EventType.COMPACTION_STARTED in types
    assert EventType.RUN_SETTLED in types
    compact_idx = types.index(EventType.COMPACTION_STARTED)
    settled_idx = types.index(EventType.RUN_SETTLED)
    assert compact_idx < settled_idx
    assert EventType.RUN_SETTLED not in types[:compact_idx]
    # Compaction/retry window has no settled event.
    mid = types[compact_idx:settled_idx]
    assert EventType.RUN_SETTLED not in mid
    assert types.count(EventType.RUN_SETTLED) == 1


@pytest.mark.asyncio
async def test_query_identical_tools_stop_before_max(
    engine_factory, in_memory_runtime: dict[str, object]
) -> None:
    llm = in_memory_runtime["llm"]
    assert isinstance(llm, InMemoryLLMProvider)
    for i in range(6):
        llm.queue_tool_call_response(
            tool_call_id=f"t{i}",
            tool_name="Read",
            tool_input={"path": "same.txt"},
        )
    llm.queue_response(text="done")
    engine = engine_factory(
        rc=_enabled(loop_guard_identical_tool_limit=2, max_turns_per_run=20)
    )
    engine.history.append(
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="read it")])
    )
    events = [evt async for evt in query(engine)]
    guard = [e for e in events if e.type is EventType.LOOP_GUARD_FIRED]
    assert guard
    assert any(e.payload.get("kind") == "identical_tool" for e in guard)
    results = [
        b
        for m in engine.history
        for b in m.content_blocks
        if isinstance(b, ToolResultBlock)
    ]
    assert results
    assert all(isinstance(b, ToolResultBlock) for b in results)
    assert EventType.RUN_SETTLED in [e.type for e in events]


@pytest.mark.asyncio
async def test_query_eviction_three_pages_and_compact_placeholder(
    engine_factory, in_memory_runtime: dict[str, object]
) -> None:
    llm = in_memory_runtime["llm"]
    assert isinstance(llm, InMemoryLLMProvider)
    llm.queue_tool_call_response(
        tool_call_id="p1", tool_name="Read", tool_input={"path": "a", "keep": False}
    )
    llm.queue_tool_call_response(
        tool_call_id="p2", tool_name="Read", tool_input={"path": "b", "keep": True}
    )
    llm.queue_tool_call_response(
        tool_call_id="p3", tool_name="Read", tool_input={"path": "c", "keep": False}
    )
    llm.queue_response(text="summarised")
    engine = engine_factory(rc=_enabled())
    engine.history.append(
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="pages")])
    )
    # Seed persist with a compacted placeholder that must survive eviction.
    engine.history.append(
        Message(
            role=MessageRole.assistant,
            content_blocks=[
                ToolUseBlock(
                    tool_call_id="old",
                    name="Read",
                    arguments_json="{}",
                )
            ],
        )
    )
    engine.history.append(
        Message(
            role=MessageRole.tool,
            content_blocks=[
                ToolResultBlock(
                    tool_call_id="old",
                    content=f"{PROTOCOL_COMPACTED_TOOL_RESULT_V1}:SNAPSHOT|x|y|1|t",
                )
            ],
        )
    )
    events = [evt async for evt in query(engine)]
    persist_reads = [
        b.content
        for m in engine.history
        for b in m.content_blocks
        if isinstance(b, ToolResultBlock)
    ]
    assert any(PROTOCOL_COMPACTED_TOOL_RESULT_V1 in c for c in persist_reads)
    # Next LLM request after the first tool should have seen eviction of unmarked.
    assert llm.calls
    last_prompt = json.dumps(
        [m.model_dump(mode="json") for m in llm.calls[-1].messages]
    )
    assert "evicted tool result" in last_prompt or EventType.TOOL_RESULT_EVICTED in [
        e.type for e in events
    ]


@pytest.mark.asyncio
async def test_steer_lands_before_next_llm_follow_up_waits(
    engine_factory, in_memory_runtime: dict[str, object]
) -> None:
    llm = in_memory_runtime["llm"]
    assert isinstance(llm, InMemoryLLMProvider)
    llm.queue_tool_call_response(
        tool_call_id="once",
        tool_name="Read",
        tool_input={"path": "x"},
    )
    llm.queue_response(text="after steer")
    llm.queue_response(text="after follow")
    engine = engine_factory(rc=_enabled())
    engine.history.append(
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])
    )
    engine._steer_queue = [new_queued_prompt("steer", "steer now").to_dict()]
    engine._follow_up_queue = [new_queued_prompt("follow_up", "later work").to_dict()]
    events = [evt async for evt in query(engine)]
    types = [e.type for e in events]
    assert EventType.RUN_SETTLED in types
    steer_idx = next(
        i
        for i, e in enumerate(events)
        if e.type is EventType.QUEUE_UPDATE and e.payload.get("kind") == "steer"
    )
    first_settled = types.index(EventType.RUN_SETTLED)
    follow_idx = next(
        i
        for i, e in enumerate(events)
        if e.type is EventType.QUEUE_UPDATE and e.payload.get("kind") == "follow_up"
    )
    assert steer_idx < first_settled < follow_idx
    texts = " ".join(m.text for m in engine.history)
    assert "steer now" in texts
    assert "later work" in texts
    assert engine._follow_up_queue == []
    # Follow-up text is on a later provider call than the first tool turn.
    later_prompts = [
        json.dumps([m.model_dump(mode="json") for m in call.messages])
        for call in llm.calls[1:]
    ]
    assert any("later work" in body for body in later_prompts)


def test_cancel_restores_queued_text(engine_factory) -> None:
    engine = engine_factory(rc=_enabled())
    engine._steer_queue = [
        new_queued_prompt("steer", "please stop and do this").to_dict()
    ]
    engine._follow_up_queue = [new_queued_prompt("follow_up", "later").to_dict()]
    restored = restore_queued_prompts(engine)
    assert restored == ["please stop and do this", "later"]
    assert engine._steer_queue == []
    assert engine._follow_up_queue == []


@pytest.mark.asyncio
async def test_query_stop_drains_queue_via_restore(engine_factory) -> None:
    engine = engine_factory(rc=_enabled())
    engine._steer_queue = [new_queued_prompt("steer", "please stop").to_dict()]
    engine.stop()
    events = [evt async for evt in query(engine)]
    assert engine._steer_queue == []
    stop = next(e for e in events if e.type is EventType.MESSAGE_STOP)
    assert "please stop" in str(stop.payload.get("restored_queue_text") or "")


@pytest.mark.asyncio
async def test_reload_picks_up_steer_posted_during_tools(
    engine_factory, in_memory_runtime: dict[str, object]
) -> None:
    llm = in_memory_runtime["llm"]
    assert isinstance(llm, InMemoryLLMProvider)
    llm.queue_tool_call_response(
        tool_call_id="once",
        tool_name="Read",
        tool_input={"path": "x"},
    )
    llm.queue_response(text="after live steer")
    engine = engine_factory(rc=_enabled())
    engine.history.append(
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])
    )
    reloads = {"n": 0}

    async def reload(eng) -> None:
        reloads["n"] += 1
        if reloads["n"] >= 2:
            eng._steer_queue = [new_queued_prompt("steer", "from redis").to_dict()]

    engine.reload_live_control = reload
    events = [evt async for evt in query(engine)]
    assert reloads["n"] >= 2
    assert any(
        e.type is EventType.QUEUE_UPDATE and e.payload.get("kind") == "steer"
        for e in events
    )
    assert "from redis" in " ".join(m.text for m in engine.history)


@pytest.mark.asyncio
async def test_mid_run_model_and_thinking(
    engine_factory, in_memory_runtime: dict[str, object]
) -> None:
    llm = in_memory_runtime["llm"]
    assert isinstance(llm, InMemoryLLMProvider)
    llm.queue_response(text="ok")
    engine = engine_factory(rc=_enabled())
    engine.history.append(
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    )
    engine.apply_live_controls(model_name="other-model", thinking_enabled=True)
    _ = [evt async for evt in query(engine)]
    assert llm.calls
    assert llm.calls[-1].model == "other-model"
    assert llm.calls[-1].extra.get("enable_thinking") is True


@pytest.mark.asyncio
async def test_deep_apply_live_controls_rejects_thinking_off(engine_factory) -> None:
    engine = engine_factory(rc=_enabled())
    object.__setattr__(engine.config, "run_mode", "deep")
    with pytest.raises(ValueError, match="deep_requires_thinking"):
        engine.apply_live_controls(thinking_enabled=False)


@pytest.mark.asyncio
async def test_flags_off_do_not_emit_new_events(
    engine_factory, in_memory_runtime: dict[str, object]
) -> None:
    llm = in_memory_runtime["llm"]
    assert isinstance(llm, InMemoryLLMProvider)
    phrase = "same sentence forever"
    llm.queue_response(text=" ".join([phrase] * 8))
    engine = engine_factory(rc=LoopConstants(model_context_window=4096))
    engine.history.append(
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    )
    events = [evt async for evt in query(engine)]
    types = {evt.type for evt in events}
    assert EventType.LOOP_GUARD_FIRED not in types
    assert EventType.RUN_SETTLED not in types
    assistant = [m for m in engine.history if m.role is MessageRole.assistant]
    assert assistant
    assert phrase in assistant[0].text


@pytest.mark.asyncio
async def test_snapshot_resume_keeps_steer_queue(engine_factory) -> None:
    engine = engine_factory(rc=_enabled())
    engine._steer_queue = [new_queued_prompt("steer", "keep me").to_dict()]
    engine._pinned_tool_result_ids.add("abc")
    engine._trimmed_tool_result_ids = frozenset({"def"})
    snap = engine.snapshot()
    other = engine_factory(rc=_enabled())
    await other.resume_from_snapshot(snap)
    assert other._steer_queue[0]["text"] == "keep me"
    assert "abc" in other._pinned_tool_result_ids
    # The trim is sticky for the run, so a run resumed in another process must
    # rebuild the same view — a forgotten set cuts a different batch and moves
    # the prompt prefix on the first request after the resume.
    assert other._trimmed_tool_result_ids == frozenset({"def"})
