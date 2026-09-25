"""Context pressure: compaction, eviction, and the read-back debt.

What the model is shown is the only place any of this is observable from
outside — a compaction that ran but did not change the next request bought
nothing. So each scenario compares the requests before and after the
pressure, and reads the events the caller saw while it happened.
"""
from __future__ import annotations

from protocore.contracts.tool_roles import ToolRole, ToolRoleMap
from protocore.contracts.types import (
    COMPACTION_SUMMARY_METADATA_KEY,
    PENDING_READS_METADATA_KEY,
    Message,
    MessageRole,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from protocore.runtime.events import EventType

from .conftest import ScenarioFactory, ScriptedTool, default_rc

LONG = "x" * 6_000


def _compacting_rc(**overrides: object) -> object:
    values: dict[str, object] = {
        "model_context_window": 512,
        "request_context_safety_tokens": 0,
        "compaction_trigger_ratio": 0.5,
        "compaction_keep_recent_turns": 1,
    }
    values.update(overrides)
    return default_rc(**values)


async def test_a_history_over_the_trigger_is_compacted_before_the_answer(
    scenario: ScenarioFactory,
) -> None:
    run = scenario(rc=_compacting_rc())
    run.llm.queue_response(text="answer after compaction")

    produced = await run.run(LONG)

    types = [evt.type for evt in produced]
    assert EventType.COMPACTION_STARTED in types
    assert EventType.COMPACTION_COMPLETED in types


async def test_compaction_finishes_before_the_run_settles(
    scenario: ScenarioFactory,
) -> None:
    """A settle announced mid-compaction is a settle the reader cannot trust."""
    run = scenario(rc=_compacting_rc(run_settled_enabled=True))
    run.llm.queue_response(text="answer after compaction")

    produced = await run.run(LONG)

    types = [evt.type for evt in produced]
    assert EventType.RUN_SETTLED in types
    assert types.index(EventType.COMPACTION_STARTED) < types.index(
        EventType.RUN_SETTLED
    )
    assert EventType.COMPACTION_STARTED not in types[types.index(EventType.RUN_SETTLED):]


async def test_a_history_under_the_trigger_is_left_alone(
    scenario: ScenarioFactory,
) -> None:
    run = scenario(rc=default_rc(model_context_window=32_000))
    run.llm.queue_response(text="short answer")

    produced = await run.run("short question")

    assert EventType.COMPACTION_STARTED not in [evt.type for evt in produced]


async def test_the_proactive_switches_do_not_reach_the_turn_start_gate(
    scenario: ScenarioFactory,
) -> None:
    """Both proactive switches off still leaves the turn-start trigger armed.

    The two are deliberately not one switch, and a run that opens over the
    routine trigger is compacted whatever the proactive settings say. A
    scenario is the only place that distinction is visible: each flag on its
    own looks like it turns compaction off.
    """
    run = scenario(
        rc=_compacting_rc(
            compaction_per_iteration_enabled=False,
            compaction_emergency_proactive_enabled=False,
        )
    )
    run.llm.queue_response(text="straight through")

    produced = await run.run(LONG)

    assert EventType.COMPACTION_STARTED in [evt.type for evt in produced]


async def test_provider_count_from_an_old_request_does_not_trigger_compaction(
    scenario: ScenarioFactory,
) -> None:
    """A scalar from one wire envelope cannot size the next one by itself."""
    run = scenario(
        rc=default_rc(
            model_context_window=1_000,
            request_context_safety_tokens=0,
            compaction_trigger_ratio=0.5,
            compaction_keep_recent_turns=1,
        ),
        tools=[ScriptedTool(tool_name="Note")],
    )
    run.llm.queue_tool_call_response(
        tool_call_id="call-1",
        tool_name="Note",
        tool_input={},
        usage_input_tokens=900,
    )
    run.llm.queue_response(text="after the real number arrived")

    produced = await run.run("small question")

    assert EventType.COMPACTION_STARTED not in [evt.type for evt in produced]
    assert len(run.requests) == 2


async def test_a_prompt_over_the_cliff_is_compacted_unconditionally(
    scenario: ScenarioFactory,
) -> None:
    """Past the cliff the gate stops asking whether compaction is worth it.

    Below the cliff the per-iteration gate runs the ordinary compaction, which
    is allowed to decide there is nothing worth summarising. Above it the same
    gate forces both tiers, because the alternative is the provider refusing
    the next request outright — and the two are told apart from outside only
    by the reason the run reports.
    """
    run = scenario(
        rc=default_rc(
            model_context_window=1_000,
            request_context_safety_tokens=0,
            compaction_trigger_ratio=0.5,
            compaction_emergency_ratio=0.8,
            compaction_keep_recent_turns=1,
        ),
        tools=[ScriptedTool(tool_name="Note", content="x" * 1_700)],
    )
    # Two results: the second is the batch just produced and stays protected,
    # so the first is what the forced pass has to work on. A pass with nothing
    # eligible is not opened at all, so it would show no reason to assert on.
    for call_id in ("call-1", "call-2"):
        run.llm.queue_tool_call_response(
            tool_call_id=call_id,
            tool_name="Note",
            tool_input={},
        )
    run.llm.queue_response(text="after the cliff was cleared")

    produced = await run.run("small question")

    reasons = [
        evt.payload.get("reason")
        for evt in produced
        if evt.type is EventType.COMPACTION_STARTED
    ]
    assert "proactive_per_iteration_emergency" in reasons
    assert "reactive_413" not in reasons


async def test_the_cliff_switch_leaves_the_ordinary_gate_running(
    scenario: ScenarioFactory,
) -> None:
    """Turning the cliff off is not turning compaction off."""
    run = scenario(
        rc=default_rc(
            model_context_window=1_000,
            request_context_safety_tokens=0,
            compaction_trigger_ratio=0.5,
            compaction_emergency_ratio=0.8,
            compaction_emergency_proactive_enabled=False,
            compaction_keep_recent_turns=1,
        ),
        tools=[ScriptedTool(tool_name="Note", content="x" * 1_100)],
    )
    # The first result is the one outside the protected batch, so the pass
    # has something to work on and is opened.
    for call_id in ("call-1", "call-2"):
        run.llm.queue_tool_call_response(
            tool_call_id=call_id,
            tool_name="Note",
            tool_input={},
        )
    run.llm.queue_response(text="after the ordinary compaction")

    produced = await run.run("small question")

    reasons = [
        evt.payload.get("reason")
        for evt in produced
        if evt.type is EventType.COMPACTION_STARTED
    ]
    assert "proactive_per_iteration" in reasons
    assert "proactive_per_iteration_emergency" not in reasons


def _reader_roles() -> ToolRoleMap:
    """The host names one tool as the one that reads a file."""
    return ToolRoleMap.declare({"Read": [ToolRole.reads_path]})


async def test_a_declared_file_forces_the_read_tool_onto_the_next_request(
    scenario: ScenarioFactory,
) -> None:
    """A pointer is not an answer: the reader is forced before the model may reply."""
    tool = ScriptedTool(
        tool_name="Note",
        content="wrote reports/a.md",
        metadata={PENDING_READS_METADATA_KEY: ["reports/a.md"]},
    )
    run = scenario(
        rc=default_rc(pending_reads_enabled=True),
        tools=[tool, ScriptedTool(tool_name="Read", content="the file body")],
        tool_roles=_reader_roles(),
    )
    run.llm.queue_tool_call_response(
        tool_call_id="call-1", tool_name="Note", tool_input={}
    )
    run.llm.queue_response(text="I already know")

    await run.run("write it")

    assert run.requests[1].extra.get("forced_tool_choice") == "Read"


async def test_the_read_back_gate_is_inert_when_it_is_switched_off(
    scenario: ScenarioFactory,
) -> None:
    tool = ScriptedTool(
        tool_name="Note",
        content="wrote reports/a.md",
        metadata={PENDING_READS_METADATA_KEY: ["reports/a.md"]},
    )
    run = scenario(
        rc=default_rc(pending_reads_enabled=False),
        tools=[tool, ScriptedTool(tool_name="Read", content="the file body")],
    )
    run.llm.queue_tool_call_response(
        tool_call_id="call-1", tool_name="Note", tool_input={}
    )
    run.llm.queue_response(text="I already know")

    await run.run("write it")

    assert "forced_tool_choice" not in run.requests[1].extra


async def test_reading_the_declared_file_releases_the_gate(
    scenario: ScenarioFactory,
) -> None:
    tool = ScriptedTool(
        tool_name="Note",
        content="wrote reports/a.md",
        metadata={PENDING_READS_METADATA_KEY: ["reports/a.md"]},
    )
    run = scenario(
        rc=default_rc(pending_reads_enabled=True),
        tools=[tool, ScriptedTool(tool_name="Read", content="the file body")],
        tool_roles=_reader_roles(),
    )
    run.llm.queue_tool_call_response(
        tool_call_id="call-1", tool_name="Note", tool_input={}
    )
    run.llm.queue_tool_call_response(
        tool_call_id="call-2",
        tool_name="Read",
        tool_input={"file_path": "reports/a.md"},
    )
    run.llm.queue_response(text="now I have read it")

    await run.run("write it")

    assert run.requests[1].extra.get("forced_tool_choice") == "Read"
    assert "forced_tool_choice" not in run.requests[2].extra


def _summary_messages(run) -> list[object]:
    """Messages the run replaced old turns with, as a summariser writes them."""
    return [
        message
        for message in run.engine.history_snapshot()
        if message.metadata.get(COMPACTION_SUMMARY_METADATA_KEY)
    ]


async def test_pressure_a_first_pass_can_absorb_never_reaches_the_summariser(
    scenario: ScenarioFactory,
) -> None:
    """Compaction has two passes, and the cheap one goes first.

    The first pass drops what can be dropped without asking anybody: it costs
    no provider call and no waiting. Only when it has not freed enough does the
    run pay for a summary. A test that asserts nothing but "a compaction
    happened" cannot tell the two apart, so a run that silently stopped ever
    reaching the second pass would still look healthy.
    """
    run = scenario(rc=_compacting_rc())
    run.llm.queue_response(text="answer after compaction")

    produced = await run.run(LONG)

    assert EventType.COMPACTION_COMPLETED in [evt.type for evt in produced]
    assert _summary_messages(run) == []


async def test_pressure_the_first_pass_cannot_absorb_is_summarised(
    scenario: ScenarioFactory,
) -> None:
    """The second pass rewrites old turns into a summary, and it shows.

    The summary is a message in the transcript, tagged as one, and every later
    request is built from it — which is what makes this pass worth telling
    apart from the first.
    """
    run = scenario(
        rc=_compacting_rc(
            model_context_window=4_096,
            compaction_trigger_ratio=0.3,
            compaction_routine_min_clear_ratio=1.0,
        ),
    )
    for index in range(2):
        run.engine.history.append(
            Message(
                role=MessageRole.user,
                content_blocks=[TextBlock(text=f"question {index} " + "y" * 1_500)],
            )
        )
        run.engine.history.append(
            Message(
                role=MessageRole.assistant,
                content_blocks=[TextBlock(text=f"answer {index} " + "z" * 1_500)],
            )
        )
    for _ in range(4):
        run.llm.queue_response(text="the old turns, summarised")
    run.llm.queue_response(text="answer after compaction")

    produced = await run.run("and now answer")

    assert EventType.COMPACTION_COMPLETED in [evt.type for evt in produced]
    summaries = _summary_messages(run)
    assert summaries, "the second pass produced no summary"
    # And the run's own request was built from the summary, not the originals.
    last_request = run.requests[-1]
    shown = " ".join(
        block.text
        for message in last_request.messages
        for block in message.content_blocks
        if isinstance(block, TextBlock)
    )
    assert "the old turns, summarised" in shown
    originals = [f"question {i} " + "y" * 1_500 for i in range(2)]
    originals += [f"answer {i} " + "z" * 1_500 for i in range(2)]
    assert any(body not in shown for body in originals), (
        "every original turn survived: the summary replaced nothing"
    )


# ---------------------------------------------------------------------------
# What compaction is allowed to take, and what it must leave
# ---------------------------------------------------------------------------

WHOLE_VALUE = "the whole of what the tool found: " + ("y" * 6_000)


def _shedding_rc() -> object:
    """Enough pressure for the first tier to shed a result, and no more.

    A far tighter window folds the whole head away into a summary instead, and
    then there is no result block left to make a statement about. What is
    under test here is the tier that replaces one result's text — so the window
    is set where that tier alone relieves the pressure.
    """
    return default_rc(
        model_context_window=2_000,
        compaction_trigger_ratio=0.5,
        compaction_keep_recent_turns=1,
    )


def _seed_an_oversized_result(run: object, *, path: str | None = None) -> None:
    """Put a settled, oversized tool call in the history the next turn opens on.

    Compaction protects the batch the current turn just executed, so a result
    can only be shed on a LATER turn — which is exactly the situation this
    seeds, and the one every long run is in.
    """
    run.engine.history.extend(  # type: ignore[attr-defined]
        [
            Message(role=MessageRole.user, content_blocks=[TextBlock(text="look")]),
            Message(
                role=MessageRole.assistant,
                content_blocks=[
                    ToolUseBlock(
                        tool_call_id="call-1",
                        name="Note",
                        arguments_json='{"v": "1"}',
                    )
                ],
            ),
            Message(
                role=MessageRole.tool,
                content_blocks=[
                    ToolResultBlock(
                        tool_call_id="call-1", content=WHOLE_VALUE, path=path
                    )
                ],
            ),
        ]
    )


async def test_compaction_sheds_the_model_projection_and_keeps_the_value(
    scenario: ScenarioFactory,
) -> None:
    """The model stops paying for the text; the run does not lose the value.

    This is the whole point of separating the two. Before, the transcript held
    one string serving the model, the reader and the record at once, so making
    room meant destroying evidence — and a run could not say afterwards what a
    tool had actually returned. Now the projection is what shrinks, the value
    goes to the blob store, and the block that used to hold it says where it
    went.
    """
    run = scenario(rc=_shedding_rc(), tools=[ScriptedTool(tool_name="Note")])
    _seed_an_oversized_result(run)
    run.llm.queue_response(text="answered from the summary")

    await run.run("carry on")

    shed = [block for block in run.tool_results() if block.canonical_ref is not None]
    assert shed, "the oversized result should have been shed"
    block = shed[0]

    # The projection is gone from the transcript and from the wire…
    assert WHOLE_VALUE not in block.content
    assert WHOLE_VALUE not in "".join(run.request_texts(-1))
    # …and the canonical value is exactly where the block says it is.
    assert await run.blobs.get("tenant-scenario", block.canonical_ref) == (
        WHOLE_VALUE.encode("utf-8")
    )


async def test_a_shed_result_still_says_which_file_it_was_about(
    scenario: ScenarioFactory,
) -> None:
    """Shedding the text does not make the result stop describing a file.

    The path is what lets a later write say this result is out of date, and a
    compacted placeholder is exactly the result most likely to still be in the
    context when that write happens.
    """
    run = scenario(rc=_shedding_rc(), tools=[ScriptedTool(tool_name="Note")])
    _seed_an_oversized_result(run, path="/w/app.py")
    run.llm.queue_response(text="answered from the summary")

    await run.run("carry on")

    shed = [block for block in run.tool_results() if block.canonical_ref is not None]
    assert shed and shed[0].path == "/w/app.py"
