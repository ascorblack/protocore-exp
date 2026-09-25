"""Intra-run compaction tests.

Covers, with universal (non-eval-tuned) assertions:

* Per-iteration proactive compaction gate inside the inner tool loop:
  a long single run with many large tool results stays context-bounded
  (per-call prompt does NOT grow monotonically; compaction fires
  proactively) and the original task + recent context survive.
* Tier-1 sheds aged ``reasoning_content`` and may compact an over-budget
  frozen reference (bootstrap) block; recent window + task are protected.
* The previously-dead ``compaction_emergency_ratio`` is wired into
  ``derive_budgets`` and drives a proactive ``force_compaction``.
* ``summarised_turn_ids`` keys on a DURABLE content hash, so a
  snapshot→resume on a fresh engine does NOT re-summarise already-summarised
  turns (no churn, no nested ``<compacted-turn>``).
* A compacted tool-result placeholder carries the originating tool name +
  a head/tail preview and round-trips through the wire parser.
"""
from __future__ import annotations

import pytest

from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.types import (
    COMPACTION_REFERENCE_METADATA_KEY,
    COMPACTION_SUMMARY_METADATA_KEY,
    SESSION_HISTORY_SEED_METADATA_KEY,
    SYNTHETIC_RECOVERY_METADATA_KEY,
    SYNTHETIC_RECOVERY_TERMINAL_REPAIR,
    CompactionSourceRef,
    Message,
    MessageRole,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from protocore.runtime.context.budgets import derive_budgets
from protocore.runtime.context.compaction import (
    CompactionState,
    _build_summarisation_units,
    _foldable_indices,
    _is_compaction_summary,
    _is_plain_operator_turn,
    current_tool_batch_protect_index,
    estimate_history_tokens,
    estimate_message_tokens,
    run_tier1_truncation,
    run_tier2_summarisation,
)
from protocore.runtime.events import EventType, TurnEvent
from protocore.runtime.loop_state import LoopState
from protocore.runtime.wire_format import (
    is_compacted_placeholder,
    parse_compacted_placeholder,
    render_compacted_placeholder,
)
from protocore.tests_support.adapters import InMemoryBlobStore

from ._tool_fixtures import MockTool

#: The test-side half of a registry pin. ``test_history_run_boundary.py``
#: records which test holds each run-scope claim; the test records which claim
#: it holds, and the two must agree — so an entry cannot be quietly downgraded
#: to whole-transcript while the test written to hold it stays green.
PINNED_ENTRIES: dict[str, tuple[str, ...]] = {
    "test_first_user_turn_skips_seeded_prior_run_turns": (
        "protocore/runtime/context/compaction.py::_first_user_turn_index",
    ),
}


# ---------------------------------------------------------------------------
# emergency ratio is wired into derive_budgets
# ---------------------------------------------------------------------------
def test_emergency_tokens_derived_and_above_trigger() -> None:
    rc = LoopConstants(model_context_window=49_152)
    budgets = derive_budgets(rc)
    assert budgets.compaction_emergency_tokens == int(
        rc.model_context_window * rc.compaction_emergency_ratio
    )
    # The RC validator guarantees trigger_ratio < emergency_ratio, and the
    # trigger only ever moves DOWN from its ratio.
    assert budgets.compaction_emergency_tokens > budgets.compaction_trigger_tokens


def test_synthetic_user_recovery_is_not_an_operator_or_fold_source() -> None:
    rc = LoopConstants(
        compaction_fold_keep_operator_turns=1,
        compaction_protect_first_user_turn=True,
    )
    history = [
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="study animals")]),
        Message(
            role=MessageRole.system,
            content_blocks=[TextBlock(text="older compacted work")],
            metadata={COMPACTION_SUMMARY_METADATA_KEY: True},
        ),
        Message(
            role=MessageRole.user,
            content_blocks=[TextBlock(text="synthetic circus repair")],
            metadata={
                SYNTHETIC_RECOVERY_METADATA_KEY: SYNTHETIC_RECOVERY_TERMINAL_REPAIR
            },
        ),
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="focus on mammals")]),
        Message(
            role=MessageRole.system,
            content_blocks=[TextBlock(text="newer compacted work")],
            metadata={COMPACTION_SUMMARY_METADATA_KEY: True},
        ),
        Message(
            role=MessageRole.user,
            content_blocks=[TextBlock(text="synthetic terminal repair")],
            metadata={
                SYNTHETIC_RECOVERY_METADATA_KEY: SYNTHETIC_RECOVERY_TERMINAL_REPAIR
            },
        ),
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="compare mammals")]),
    ]

    assert _is_plain_operator_turn(history[3])
    assert not _is_plain_operator_turn(history[2])
    assert not _is_plain_operator_turn(history[5])
    assert history[-1].text == "compare mammals"
    foldable = _foldable_indices(history, len(history), rc)
    assert foldable == frozenset({1, 3, 4})
    assert 0 not in foldable
    assert 5 not in foldable
    assert 6 not in foldable


# ---------------------------------------------------------------------------
# placeholder enrichment + round-trip
# ---------------------------------------------------------------------------
def test_placeholder_carries_tool_name_and_preview_roundtrip() -> None:
    ref = CompactionSourceRef(
        blob_ref="blob://abc",
        sha256="deadbeef",
        original_tokens=1234,
        label="tool_result",
        tool_name="workspace_read_file",
        # Pipes + newlines + Cyrillic must survive (base64-encoded on the wire).
        preview="head | пример\nmiddle … tail",
    )
    placeholder = render_compacted_placeholder(ref, "SNAPSHOT")
    assert is_compacted_placeholder(placeholder)
    # Single-line, no raw pipe-collision: the frame is pipe-delimited and the
    # preview is base64, so exactly the 6 framing pipes appear.
    assert placeholder.count("|") == 6

    parsed = parse_compacted_placeholder(placeholder)
    assert parsed is not None
    out_ref, variant = parsed
    assert variant == "SNAPSHOT"
    assert out_ref.tool_name == "workspace_read_file"
    assert out_ref.preview == "head | пример\nmiddle … tail"
    assert out_ref.blob_ref == "blob://abc"
    assert out_ref.original_tokens == 1234


def test_legacy_five_field_placeholder_still_parses() -> None:
    # A legacy placeholder with no trailing tool_name/preview fields.
    legacy = "PROTOCOL_COMPACTED_TOOL_RESULT_V1:SNAPSHOT|blob://x|abc123|999|tool_result"
    parsed = parse_compacted_placeholder(legacy)
    assert parsed is not None
    ref, _variant = parsed
    assert ref.tool_name == ""
    assert ref.preview == ""
    assert ref.blob_ref == "blob://x"


@pytest.mark.asyncio
async def test_tier1_placeholder_enriched_from_originating_tool() -> None:
    rc = LoopConstants(model_context_window=4_096)
    blobs = InMemoryBlobStore()
    big = "LINE\n" * 4000
    history = [
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="task")]),
        Message(
            role=MessageRole.assistant,
            content_blocks=[
                ToolUseBlock(tool_call_id="t1", name="Bash", arguments_json='{"cmd":"ls"}')
            ],
        ),
        Message(
            role=MessageRole.tool,
            content_blocks=[ToolResultBlock(tool_call_id="t1", content=big)],
        ),
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="recent")]),
    ]
    budgets = derive_budgets(rc)
    await run_tier1_truncation(
        history=history,
        blob_store=blobs,
        tenant_id="t",
        rc=rc,
        truncation_threshold_tokens=budgets.tool_result_truncation_threshold,
        keep_recent_turns=1,
    )
    block = history[2].content_blocks[0]
    assert isinstance(block, ToolResultBlock)
    parsed = parse_compacted_placeholder(block.content)
    assert parsed is not None
    ref, _ = parsed
    assert ref.tool_name == "Bash"  # originating tool surfaced
    # The preview is for the model, so it is written where the model can read
    # it — the line under the frame — not base64-encoded inside it.
    assert ref.preview == ""
    readable = block.content.split("\n", 1)[1]
    assert "output of Bash" in readable
    assert "First and last lines:" in readable


# ---------------------------------------------------------------------------
# reasoning shed + bootstrap bound (recent window + task protected)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_sheds_aged_reasoning_content_keeps_recent() -> None:
    rc = LoopConstants(model_context_window=4_096)
    blobs = InMemoryBlobStore()
    aged_reasoning = "deep thinking " * 200
    recent_reasoning = "recent thinking " * 200
    history = [
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="task")]),
        Message(
            role=MessageRole.assistant,
            content_blocks=[TextBlock(text="aged answer")],
            reasoning_content=aged_reasoning,
        ),
        Message(
            role=MessageRole.assistant,
            content_blocks=[TextBlock(text="recent answer")],
            reasoning_content=recent_reasoning,
        ),
    ]
    budgets = derive_budgets(rc)
    result = await run_tier1_truncation(
        history=history,
        blob_store=blobs,
        tenant_id="t",
        rc=rc,
        truncation_threshold_tokens=budgets.tool_result_truncation_threshold,
        keep_recent_turns=1,  # only history[2] is "recent"
    )
    assert result.tokens_freed > 0
    # Aged reasoning shed, recent reasoning kept verbatim.
    assert history[1].reasoning_content is None
    assert history[2].reasoning_content == recent_reasoning


@pytest.mark.asyncio
async def test_reasoning_shed_kill_switch() -> None:
    rc = LoopConstants(
        model_context_window=4_096, compaction_shed_reasoning_enabled=False
    )
    blobs = InMemoryBlobStore()
    history = [
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="task")]),
        Message(
            role=MessageRole.assistant,
            content_blocks=[TextBlock(text="aged")],
            reasoning_content="thinking " * 200,
        ),
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="recent")]),
    ]
    budgets = derive_budgets(rc)
    await run_tier1_truncation(
        history=history,
        blob_store=blobs,
        tenant_id="t",
        rc=rc,
        truncation_threshold_tokens=budgets.tool_result_truncation_threshold,
        keep_recent_turns=1,
    )
    assert history[1].reasoning_content is not None  # untouched


@pytest.mark.asyncio
async def test_bounds_over_budget_reference_block() -> None:
    rc = LoopConstants(model_context_window=4_096)
    blobs = InMemoryBlobStore()
    big_bootstrap = "<environment_context>\n" + ("CONFIG LINE\n" * 4000)
    history = [
        Message(
            role=MessageRole.user,
            content_blocks=[TextBlock(text=big_bootstrap)],
            metadata={COMPACTION_REFERENCE_METADATA_KEY: True},
        ),
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="the task")]),
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="recent")]),
    ]
    budgets = derive_budgets(rc)
    result = await run_tier1_truncation(
        history=history,
        blob_store=blobs,
        tenant_id="t",
        rc=rc,
        truncation_threshold_tokens=budgets.tool_result_truncation_threshold,
        keep_recent_turns=1,
    )
    assert result.tokens_freed > 0
    # Bootstrap reference block compacted to a placeholder…
    ref_block = history[0].content_blocks[0]
    assert isinstance(ref_block, TextBlock)
    assert is_compacted_placeholder(ref_block.text)
    # …but the (untagged) task turn is untouched.
    assert history[1].content_blocks[0].text == "the task"


@pytest.mark.asyncio
async def test_untagged_user_turn_never_shed_as_reference() -> None:
    rc = LoopConstants(model_context_window=4_096)
    blobs = InMemoryBlobStore()
    big_task = "PLEASE DO THIS LONG THING\n" * 4000  # large but NOT a reference block
    history = [
        Message(role=MessageRole.user, content_blocks=[TextBlock(text=big_task)]),
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="recent")]),
    ]
    budgets = derive_budgets(rc)
    await run_tier1_truncation(
        history=history,
        blob_store=blobs,
        tenant_id="t",
        rc=rc,
        truncation_threshold_tokens=budgets.tool_result_truncation_threshold,
        keep_recent_turns=1,
    )
    block = history[0].content_blocks[0]
    assert isinstance(block, TextBlock)
    assert not is_compacted_placeholder(block.text)  # untagged → never shed


@pytest.mark.asyncio
async def test_sheds_summary_seed_block_via_reference_path() -> None:
    """Engagement test: a ``build_seed`` summary block
    (dual-tagged seed + reference) is actually blob-shed by the Tier-1
    reference path when over budget — proving the previously-immovable large
    user-text seed block now has a real shed path (and survives as a recoverable
    SNAPSHOT placeholder, not a destructive drop)."""
    from protocore.runtime.context.session_memory import (
        ArtifactLedger,
        SessionMemory,
        build_seed,
    )

    rc = LoopConstants(model_context_window=4_096)
    blobs = InMemoryBlobStore()
    mem = SessionMemory(
        running_summary="DECISION: port 8080. " + ("carried fact line. " * 4000),
        ledger=ArtifactLedger(files=["a.py"], content={"a.py": "x = 1\n" * 2000}),
        turn_index=3,
    )
    head = [Message(role=MessageRole.user, content_blocks=[TextBlock(text="ORIGINAL TASK")])]
    seed = build_seed(mem, [], head, 40_000, rc)
    # A fresh task turn after the seed (the recent window the guard keeps).
    history = [
        *seed,
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="NEW TASK now")]),
    ]

    budgets = derive_budgets(rc)
    result = await run_tier1_truncation(
        history=history,
        blob_store=blobs,
        tenant_id="t",
        rc=rc,
        truncation_threshold_tokens=budgets.tool_result_truncation_threshold,
        keep_recent_turns=1,
    )
    assert result.tokens_freed > 0

    def _text(m: Message) -> str:
        b = m.content_blocks[0]
        return b.text if isinstance(b, TextBlock) else ""

    shed = [m for m in history if is_compacted_placeholder(_text(m))]
    # At least the large running-summary seed block was shed to a placeholder.
    assert shed, (
        "the dual-tagged summary/ledger seed block must be shed by the "
        "Tier-1 reference path"
    )
    # The shed block kept its seed tag (finalization filter still excludes it)
    # and the reference tag (so it stays sheddable) and is recoverable.
    for m in shed:
        assert m.metadata.get(SESSION_HISTORY_SEED_METADATA_KEY) is True
        parsed = parse_compacted_placeholder(_text(m))
        assert parsed is not None  # round-trips → recall_artifact can recover it
    # The original task + the new task turn are NOT shed.
    assert any("ORIGINAL TASK" == _text(m) for m in history)
    assert any("NEW TASK now" == _text(m) for m in history)


# ---------------------------------------------------------------------------
# stable dedup key survives snapshot/resume (no re-summarise churn)
# ---------------------------------------------------------------------------
def _summary_history() -> list[Message]:
    # An aged assistant turn (idx 1) + a recent user turn (idx 2). The first
    # user turn (idx 0) is the protected task.
    return [
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="the task")]),
        Message(role=MessageRole.assistant, content_blocks=[TextBlock(text="aged answer " * 60)]),
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="recent")]),
    ]


@pytest.mark.asyncio
async def test_resume_does_not_resummarise() -> None:
    from protocore.tests_support.adapters import InMemoryLLMProvider

    rc = LoopConstants(model_context_window=4_096, compaction_keep_recent_turns=1)

    llm1 = InMemoryLLMProvider()
    llm1.queue_response(text="SUMMARY-1")
    history = _summary_history()
    state = CompactionState()
    r1 = await run_tier2_summarisation(
        history=history, compaction_llm=llm1, state=state, rc=rc, model_name="m"
    )
    assert r1.turns_summarised == 1
    summary_idx = next(
        i for i, m in enumerate(history)
        if m.metadata.get(COMPACTION_SUMMARY_METADATA_KEY) is True
    )
    assert history[summary_idx].text.startswith("<compacted-turn")

    # ── snapshot → resume on a FRESH engine (new Message object identities).
    persisted_ids = list(state.summarised_turn_ids)
    resumed_history = [Message.model_validate(m.model_dump(mode="json")) for m in history]
    resumed_state = CompactionState(summarised_turn_ids=set(persisted_ids))

    # Append a NEW aged assistant turn so there is fresh eligible work; the
    # already-summarised turn must NOT be re-summarised.
    resumed_history.insert(
        len(resumed_history) - 1,
        Message(role=MessageRole.assistant, content_blocks=[TextBlock(text="newly aged " * 60)]),
    )

    llm2 = InMemoryLLMProvider()
    llm2.queue_response(text="SUMMARY-2")
    llm2.queue_response(text="SUMMARY-EXTRA")  # would be used if it re-summarised
    r2 = await run_tier2_summarisation(
        history=resumed_history, compaction_llm=llm2, state=resumed_state, rc=rc, model_name="m"
    )
    # Exactly ONE summariser call — the new turn, not the already-summarised one.
    assert len(llm2.calls) == 1
    assert r2.turns_summarised == 1
    # No nested <compacted-turn> wrapper anywhere.
    for m in resumed_history:
        assert m.text.count("<compacted-turn") <= 1


@pytest.mark.asyncio
async def test_existing_summary_anchor_skipped() -> None:
    from protocore.tests_support.adapters import InMemoryLLMProvider

    rc = LoopConstants(model_context_window=4_096, compaction_keep_recent_turns=1)
    # A history whose aged turn is ALREADY a compaction summary.
    history = [
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="task")]),
        Message(
            role=MessageRole.system,
            content_blocks=[TextBlock(text="<compacted-turn id='x'>prior</compacted-turn>")],
            metadata={COMPACTION_SUMMARY_METADATA_KEY: True},
        ),
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="recent")]),
    ]
    llm = InMemoryLLMProvider()
    llm.queue_response(text="SHOULD-NOT-FIRE")
    result = await run_tier2_summarisation(
        history=history, compaction_llm=llm, state=CompactionState(), rc=rc, model_name="m"
    )
    # System summaries are excluded from units AND the anchor-skip guards it.
    assert result.turns_summarised == 0
    assert len(llm.calls) == 0


@pytest.mark.asyncio
async def test_tier2_drops_aged_synthetic_nudge_without_summarising_it() -> None:
    from protocore.tests_support.adapters import InMemoryLLMProvider

    marker = "INTERNAL RECOVERY MUST NOT BECOME USER INTENT " * 30
    history = [
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="real task")]),
        Message(
            role=MessageRole.assistant,
            content_blocks=[TextBlock(text="aged answer " * 40)],
        ),
        Message(
            role=MessageRole.user,
            content_blocks=[TextBlock(text=marker)],
            metadata={
                SYNTHETIC_RECOVERY_METADATA_KEY: SYNTHETIC_RECOVERY_TERMINAL_REPAIR
            },
        ),
        Message(
            role=MessageRole.assistant,
            content_blocks=[TextBlock(text="another aged answer " * 40)],
        ),
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="recent steer")]),
    ]
    llm = InMemoryLLMProvider()
    llm.queue_response(text="FIRST SUMMARY")
    llm.queue_response(text="SECOND SUMMARY")

    result = await run_tier2_summarisation(
        history=history,
        compaction_llm=llm,
        state=CompactionState(),
        rc=LoopConstants(model_context_window=4_096, compaction_keep_recent_turns=1),
        model_name="m",
    )

    # With the nudge gone the two answers are adjacent, and one span.
    assert result.turns_summarised == 1
    assert result.tokens_freed > 0
    assert all(marker not in message.text for message in history)
    assert history[0].text == "real task"
    prompts = "\n".join(message.text for call in llm.calls for message in call.messages)
    assert marker not in prompts


@pytest.mark.asyncio
async def test_tier2_cleanup_only_satisfies_target_without_provider_call() -> None:
    from protocore.tests_support.adapters import InMemoryLLMProvider

    rc = LoopConstants(model_context_window=4_096, compaction_keep_recent_turns=1)
    synthetic = Message(
        role=MessageRole.user,
        content_blocks=[TextBlock(text="discard internal recovery " * 20)],
        metadata={
            SYNTHETIC_RECOVERY_METADATA_KEY: SYNTHETIC_RECOVERY_TERMINAL_REPAIR
        },
    )
    history = [
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="real task")]),
        synthetic,
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="recent")]),
    ]
    llm = InMemoryLLMProvider()

    result = await run_tier2_summarisation(
        history=history,
        compaction_llm=llm,
        state=CompactionState(),
        rc=rc,
        model_name="m",
        free_target_tokens=1,
    )

    assert result.turns_summarised == 0
    assert result.tokens_freed == estimate_message_tokens(synthetic, rc)
    assert len(llm.calls) == 0
    assert [message.text for message in history] == ["real task", "recent"]


@pytest.mark.asyncio
async def test_tier2_provider_failure_keeps_the_deterministic_cleanup() -> None:
    """A failed summariser call costs its own unit and nothing else.

    Dropping an aged synthetic recovery nudge needs no provider, so it is not
    undone by a call that failed elsewhere in the pass: the pass commits what
    it actually achieved and counts the failure against the unit that caused it.
    """
    from protocore.tests_support.adapters import InMemoryLLMProvider

    class FailingSummaryLLM(InMemoryLLMProvider):
        async def complete_text(self, request):  # type: ignore[no-untyped-def]
            raise RuntimeError("summary unavailable")

    history = [
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="real task")]),
        Message(
            role=MessageRole.assistant,
            content_blocks=[TextBlock(text="aged answer " * 40)],
        ),
        Message(
            role=MessageRole.user,
            content_blocks=[TextBlock(text="internal recovery " * 20)],
            metadata={
                SYNTHETIC_RECOVERY_METADATA_KEY: SYNTHETIC_RECOVERY_TERMINAL_REPAIR
            },
        ),
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="recent")]),
    ]
    synthetic_tokens = estimate_message_tokens(history[2], LoopConstants(
        model_context_window=4_096, compaction_keep_recent_turns=1
    ))
    state = CompactionState()

    result = await run_tier2_summarisation(
        history=history,
        compaction_llm=FailingSummaryLLM(),
        state=state,
        rc=LoopConstants(model_context_window=4_096, compaction_keep_recent_turns=1),
        model_name="m",
    )

    # Nothing was summarised, but the nudge the runtime had put there is gone.
    assert result.turns_summarised == 0
    assert result.tokens_freed == synthetic_tokens
    assert [message.text for message in history] == [
        "real task",
        "aged answer " * 40,
        "recent",
    ]
    assert state.summarised_turn_ids == set()
    # A transport failure says nothing about the unit, so nothing is written
    # off: the next pass tries it again.
    assert state.failed_anchor_keys == {}


@pytest.mark.asyncio
async def test_tier2_record_failure_leaves_history_unchanged() -> None:
    from protocore.tests_support.adapters import InMemoryLLMProvider

    history = [
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="task")]),
        Message(
            role=MessageRole.assistant,
            content_blocks=[TextBlock(text="aged answer " * 40)],
        ),
        Message(
            role=MessageRole.user,
            content_blocks=[TextBlock(text="internal recovery " * 20)],
            metadata={
                SYNTHETIC_RECOVERY_METADATA_KEY: SYNTHETIC_RECOVERY_TERMINAL_REPAIR
            },
        ),
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="recent")]),
    ]
    before = [message.model_dump(mode="json") for message in history]
    llm = InMemoryLLMProvider()
    llm.queue_response(text="summary")

    async def fail_record(_request) -> None:  # type: ignore[no-untyped-def]
        raise RuntimeError("record unavailable")

    with pytest.raises(RuntimeError, match="record unavailable"):
        await run_tier2_summarisation(
            history=history,
            compaction_llm=llm,
            state=CompactionState(),
            rc=LoopConstants(
                model_context_window=4_096, compaction_keep_recent_turns=1
            ),
            model_name="m",
            record_request=fail_record,
        )

    assert [message.model_dump(mode="json") for message in history] == before


@pytest.mark.asyncio
async def test_tier2_prompt_failure_leaves_history_unchanged() -> None:
    from protocore.tests_support.adapters import InMemoryLLMProvider

    class FailingPrompts:
        def render(self, name, variables):  # type: ignore[no-untyped-def]
            raise RuntimeError("prompt unavailable")

    history = [
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="task")]),
        Message(
            role=MessageRole.assistant,
            content_blocks=[TextBlock(text="aged answer " * 40)],
        ),
        Message(
            role=MessageRole.user,
            content_blocks=[TextBlock(text="internal recovery " * 20)],
            metadata={
                SYNTHETIC_RECOVERY_METADATA_KEY: SYNTHETIC_RECOVERY_TERMINAL_REPAIR
            },
        ),
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="recent")]),
    ]
    before = [message.model_dump(mode="json") for message in history]

    with pytest.raises(RuntimeError, match="prompt unavailable"):
        await run_tier2_summarisation(
            history=history,
            compaction_llm=InMemoryLLMProvider(),
            state=CompactionState(),
            rc=LoopConstants(
                model_context_window=4_096, compaction_keep_recent_turns=1
            ),
            model_name="m",
            prompts=FailingPrompts(),  # type: ignore[arg-type]
        )

    assert [message.model_dump(mode="json") for message in history] == before


# ---------------------------------------------------------------------------
# the dedup key must distinguish DISTINCT turns with identical content
# (the documented content-missing Write-retry spiral: same name+arguments_json,
# fresh tool_call_id each time). Pre-fix the second identical unit collides with
# the first one's entry in ``summarised_turn_ids`` and is silently skipped, so
# Tier-2 frees almost nothing beyond the first copy.
# ---------------------------------------------------------------------------
def _spiral_history() -> list[Message]:
    """``[task, A(id-1), R(id-1), A(id-2), R(id-2), keep, keep]``.

    The two assistant turns re-emit the SAME tool call (identical ``name`` +
    ``arguments_json``) with DIFFERENT ``tool_call_id``s — the by-construction
    duplicate shape. Tool results are also byte-identical save for the id.
    """
    # Content is sized WELL ABOVE the no-net-gain floor (the empty
    # <compacted-turn> wrapper size) so the units stay Tier-2-eligible — this
    # test pins the dedup-key collision, not the size gate.
    args = '{"path":"big.py","content":"' + "x" * 4000 + '"}'
    return [
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="the task")]),
        Message(
            role=MessageRole.assistant,
            content_blocks=[ToolUseBlock(tool_call_id="w1", name="Write", arguments_json=args)],
        ),
        Message(
            role=MessageRole.tool,
            content_blocks=[ToolResultBlock(tool_call_id="w1", content="Field required: content. " + "ctx " * 400)],
        ),
        Message(
            role=MessageRole.assistant,
            content_blocks=[ToolUseBlock(tool_call_id="w2", name="Write", arguments_json=args)],
        ),
        Message(
            role=MessageRole.tool,
            content_blocks=[ToolResultBlock(tool_call_id="w2", content="Field required: content. " + "ctx " * 400)],
        ),
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="keep-a")]),
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="keep-b")]),
    ]


@pytest.mark.asyncio
async def test_duplicate_tool_call_turns_each_summarised() -> None:
    from protocore.tests_support.adapters import InMemoryLLMProvider

    # keep_recent=2 protects the two trailing user turns; the first user turn
    # (the task) is protected by compaction_protect_first_user_turn. That leaves
    # BOTH A/R units (indices {1,2} and {3,4}) eligible.
    rc = LoopConstants(
        model_context_window=4_096,
        compaction_keep_recent_turns=2,
        compaction_summary_group_max_tokens=0,
    )
    history = _spiral_history()
    llm = InMemoryLLMProvider()
    llm.queue_response(text="SUMMARY-1")
    llm.queue_response(text="SUMMARY-2")
    state = CompactionState()

    # Single pass must summarise BOTH duplicate units, not just the first.
    result = await run_tier2_summarisation(
        history=history, compaction_llm=llm, state=state, rc=rc, model_name="m"
    )
    assert result.turns_summarised == 2, (
        "both distinct duplicate-content tool-call turns must be summarised; "
        "pre-fix the second collides on the content hash and is skipped"
    )
    assert len(llm.calls) == 2
    # Both tool-result messages were dropped — no verbatim duplicate survives.
    assert not any(m.role is MessageRole.tool for m in history)
    summaries = [
        m for m in history if m.metadata.get(COMPACTION_SUMMARY_METADATA_KEY) is True
    ]
    assert len(summaries) == 2
    # Distinct dedup keys were banked for the two distinct turns.
    assert len(state.summarised_turn_ids) == 2
    # The protected task + trailing keep window survive verbatim.
    assert history[0].text == "the task"
    assert any(m.text == "keep-a" for m in history)
    assert any(m.text == "keep-b" for m in history)


# ---------------------------------------------------------------------------
# guardrail — original task protected from Tier-2
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_first_user_task_protected_from_summarisation() -> None:
    from protocore.tests_support.adapters import InMemoryLLMProvider

    rc = LoopConstants(model_context_window=4_096, compaction_keep_recent_turns=1)
    history = [
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="ORIGINAL TASK " * 20)]),
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="recent")]),
    ]
    llm = InMemoryLLMProvider()
    llm.queue_response(text="SUMMARY")
    result = await run_tier2_summarisation(
        history=history, compaction_llm=llm, state=CompactionState(), rc=rc, model_name="m"
    )
    assert result.turns_summarised == 0
    assert history[0].text.startswith("ORIGINAL TASK")  # verbatim


# ---------------------------------------------------------------------------
# Cross-run history seeding interaction with compaction
# ---------------------------------------------------------------------------
def test_first_user_turn_skips_seeded_prior_run_turns() -> None:
    """``_first_user_turn_index`` must return the NEW task, not a seeded turn.

    Seeded prior-run user turns precede the new task in history; the
    protect-first-user-turn guard must shield the new task (the last,
    non-seeded, non-reference user turn), not a seeded one.
    """
    from protocore.runtime.context.compaction import _first_user_turn_index

    history = [
        Message(
            role=MessageRole.user,
            content_blocks=[TextBlock(text="prior turn 1")],
            metadata={SESSION_HISTORY_SEED_METADATA_KEY: True},
        ),
        Message(
            role=MessageRole.assistant,
            content_blocks=[TextBlock(text="prior answer")],
            metadata={SESSION_HISTORY_SEED_METADATA_KEY: True},
        ),
        Message(
            role=MessageRole.user,
            content_blocks=[TextBlock(text="env ref")],
            metadata={COMPACTION_REFERENCE_METADATA_KEY: True},
        ),
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="THE NEW TASK")]),
    ]
    assert _first_user_turn_index(history) == 3
    assert history[3].text == "THE NEW TASK"


@pytest.mark.asyncio
async def test_seeded_turns_protected_from_tier2_summary() -> None:
    """Seeded prior-run turns must NOT be collapsed by lossy Tier-2.

    A Tier-2 ``<compacted-turn>`` system message would drop the
    ``SESSION_HISTORY_SEED`` tag the host finalization filter relies on,
    re-persisting prior-run content under the new run_id. So a span of seeded
    turns is protected from Tier-2 (Tier-1 shedding — which preserves the tag —
    still bounds them).
    """
    from protocore.tests_support.adapters import InMemoryLLMProvider

    rc = LoopConstants(model_context_window=4_096, compaction_keep_recent_turns=1)
    history = [
        Message(
            role=MessageRole.user,
            content_blocks=[TextBlock(text="SEEDED PRIOR TURN " * 20)],
            metadata={SESSION_HISTORY_SEED_METADATA_KEY: True},
        ),
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="THE NEW TASK")]),
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="recent")]),
    ]
    llm = InMemoryLLMProvider()
    llm.queue_response(text="SUMMARY")
    result = await run_tier2_summarisation(
        history=history, compaction_llm=llm, state=CompactionState(), rc=rc, model_name="m"
    )
    assert result.turns_summarised == 0
    # Seeded turn stays verbatim AND keeps its tag.
    assert history[0].text.startswith("SEEDED PRIOR TURN")
    assert history[0].metadata.get(SESSION_HISTORY_SEED_METADATA_KEY) is True


@pytest.mark.asyncio
async def test_tier2_keeps_mixed_provenance_tool_unit_atomic() -> None:
    """A summary cannot represent a tool pair with two persistence provenances."""
    from protocore.tests_support.adapters import InMemoryLLMProvider

    rc = LoopConstants(
        model_context_window=4_096,
        compaction_keep_recent_turns=1,
        compaction_protect_first_user_turn=False,
    )
    history = [
        Message(
            role=MessageRole.assistant,
            content_blocks=[
                ToolUseBlock(
                    tool_call_id="mixed",
                    name="read",
                    arguments_json='{"path":"old"}',
                )
            ],
            metadata={SESSION_HISTORY_SEED_METADATA_KEY: True},
        ),
        Message(
            role=MessageRole.tool,
            content_blocks=[
                ToolResultBlock(tool_call_id="mixed", content="large result " * 200)
            ],
        ),
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="current task")]),
    ]
    before = [message.model_dump_json() for message in history]
    llm = InMemoryLLMProvider()
    llm.queue_response(text="must not be used")

    result = await run_tier2_summarisation(
        history=history,
        compaction_llm=llm,
        state=CompactionState(),
        rc=rc,
        model_name="mock",
        compact_seeded_history=True,
    )

    assert result.turns_summarised == 0
    assert llm.calls == ()
    assert [message.model_dump_json() for message in history] == before


@pytest.mark.asyncio
async def test_tier1_shed_preserves_seed_tag_on_placeholder() -> None:
    """Tier-1 shedding of a LARGE seeded tool result MUST keep the seed tag.

    This is the ONLY bound for large seeded tool results — Tier-2 is barred
    from collapsing seeded turns (it would drop the
    ``SESSION_HISTORY_SEED`` tag), so Tier-1 has to shed them to a placeholder
    WITHOUT losing the tag. The tag survives because :func:`run_tier1_truncation`
    uses ``message.model_copy(update={"content_blocks": [...]})`` which leaves
    message-level ``metadata`` untouched. If a future Tier-1 change ever
    rebuilt the message (or passed ``update={"metadata": ...}``) and stripped
    the tag, the seeded turn would be re-persisted under the new ``run_id``
    (exponential ``session_messages`` growth) — this test fails loudly first.
    """
    rc = LoopConstants(model_context_window=4_096)
    blobs = InMemoryBlobStore()
    big = "SEED RESULT LINE\n" * 4000
    history = [
        Message(
            role=MessageRole.user,
            content_blocks=[TextBlock(text="prior task")],
            metadata={SESSION_HISTORY_SEED_METADATA_KEY: True},
        ),
        Message(
            role=MessageRole.assistant,
            content_blocks=[
                ToolUseBlock(tool_call_id="s1", name="Read", arguments_json='{"path":"x"}')
            ],
            metadata={SESSION_HISTORY_SEED_METADATA_KEY: True},
        ),
        Message(
            role=MessageRole.tool,
            content_blocks=[ToolResultBlock(tool_call_id="s1", content=big)],
            metadata={SESSION_HISTORY_SEED_METADATA_KEY: True},
        ),
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="the new task")]),
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="recent")]),
    ]
    budgets = derive_budgets(rc)
    result = await run_tier1_truncation(
        history=history,
        blob_store=blobs,
        tenant_id="t",
        rc=rc,
        truncation_threshold_tokens=budgets.tool_result_truncation_threshold,
        keep_recent_turns=1,
    )
    # The big seeded tool result was actually shed to a placeholder.
    assert result.messages_modified >= 1
    shed_block = history[2].content_blocks[0]
    assert isinstance(shed_block, ToolResultBlock)
    assert is_compacted_placeholder(shed_block.content)
    # CRITICAL: the seed tag survives shedding on the placeholder message, so
    # the finalization filter still excludes it from this run's persistence.
    assert history[2].metadata.get(SESSION_HISTORY_SEED_METADATA_KEY) is True


# ---------------------------------------------------------------------------
# Reference blocks — Tier-1-only shed surface, protected from Tier-2
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_reference_block_protected_from_tier2_summary() -> None:
    """An un-blobbed reference block must NOT be collapsed by lossy Tier-2.

    Reference blocks (the executor's ``<environment_context>``/
    ``<memory-context>`` bootstrap) are a Tier-1-ONLY shed surface — the Tier-1
    reference path blobs them to a RECOVERABLE placeholder when over budget. A
    Tier-2
    ``<compacted-turn>`` collapse is lossy and contradicts that design, so the
    block stays verbatim (and tagged) until Tier-1 sheds it.
    """
    from protocore.tests_support.adapters import InMemoryLLMProvider

    rc = LoopConstants(model_context_window=4_096, compaction_keep_recent_turns=1)
    history = [
        Message(
            role=MessageRole.user,
            content_blocks=[TextBlock(text="<environment_context>ENV " * 20 + "</environment_context>")],
            metadata={COMPACTION_REFERENCE_METADATA_KEY: True},
        ),
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="THE TASK")]),
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="recent")]),
    ]
    llm = InMemoryLLMProvider()
    llm.queue_response(text="SHOULD-NOT-FIRE")
    result = await run_tier2_summarisation(
        history=history, compaction_llm=llm, state=CompactionState(), rc=rc, model_name="m"
    )
    assert result.turns_summarised == 0
    assert len(llm.calls) == 0
    # Reference block stays verbatim AND keeps its tag (still Tier-1-sheddable).
    assert history[0].text.startswith("<environment_context>")
    assert history[0].metadata.get(COMPACTION_REFERENCE_METADATA_KEY) is True


@pytest.mark.asyncio
async def test_blobbed_reference_placeholder_not_summarised_by_tier2() -> None:
    """A Tier-1-blobbed reference placeholder must survive a Tier-2 pass.

    End-to-end regression for the Tier-1 reference-block recoverable-snapshot
    contract: Tier-1
    blobs an over-budget reference block to a ``PROTOCOL_COMPACTED…SNAPSHOT``
    placeholder — the ONLY in-history pointer to the blob. A subsequent Tier-2
    pass (force_compaction / routine) must NOT treat the placeholder as an
    eligible singleton unit: summarising it would burn one LLM call on the raw
    placeholder marker AND erase the blob ref from history, making the
    snapshot unrecoverable.
    """
    from protocore.tests_support.adapters import InMemoryLLMProvider

    rc = LoopConstants(model_context_window=4_096, compaction_keep_recent_turns=1)
    blobs = InMemoryBlobStore()
    big_ref = "<memory-context>\n" + "MEMORY LINE\n" * 4000 + "</memory-context>"
    history = [
        Message(
            role=MessageRole.user,
            content_blocks=[TextBlock(text=big_ref)],
            metadata={COMPACTION_REFERENCE_METADATA_KEY: True},
        ),
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="THE TASK")]),
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="recent")]),
    ]
    budgets = derive_budgets(rc)
    tier1 = await run_tier1_truncation(
        history=history,
        blob_store=blobs,
        tenant_id="t",
        rc=rc,
        truncation_threshold_tokens=budgets.tool_result_truncation_threshold,
        keep_recent_turns=1,
    )
    # Precondition: Tier-1 actually blob-shed the reference block.
    assert tier1.blob_refs_created
    placeholder_block = history[0].content_blocks[0]
    assert isinstance(placeholder_block, TextBlock)
    assert is_compacted_placeholder(placeholder_block.text)
    blob_ref = history[0].metadata["blob_ref"]

    llm = InMemoryLLMProvider()
    llm.queue_response(text="SHOULD-NOT-FIRE")
    tier2 = await run_tier2_summarisation(
        history=history, compaction_llm=llm, state=CompactionState(), rc=rc, model_name="m"
    )
    assert tier2.turns_summarised == 0
    # No token-burning summariser call on the raw placeholder marker.
    assert len(llm.calls) == 0
    # The placeholder — the only in-history pointer to the blob — survives.
    block = history[0].content_blocks[0]
    assert isinstance(block, TextBlock)
    assert is_compacted_placeholder(block.text)
    assert history[0].metadata.get("blob_ref") == blob_ref
    assert history[0].metadata.get(COMPACTION_REFERENCE_METADATA_KEY) is True


# ---------------------------------------------------------------------------
# per-iteration proactive gate (end-to-end query() loop)
# ---------------------------------------------------------------------------
def _prompt_estimate(req) -> int:
    """Sum estimated tokens across the messages on an LLMRequest."""
    rc = LoopConstants()
    return estimate_history_tokens(list(req.messages), rc)


async def _drive_long_tool_chain(
    engine_factory,
    in_memory_runtime,
    *,
    per_iteration_enabled: bool,
    iterations: int = 12,
    window: int = 4_096,
):
    # Small window so big tool outputs cross the trigger within the run.
    rc = LoopConstants(
        model_context_window=window,
        request_context_safety_tokens=0,
        compaction_per_iteration_enabled=per_iteration_enabled,
        # Keep the test deterministic: only Tier-1 (no summariser LLM).
        compaction_keep_recent_turns=2,
        # This fixture drives a long tool chain and then lets the scripted
        # responses run out (the mock then streams a bare-empty end_turn). That
        # empty-exhaustion path is incidental scaffolding for the compaction
        # assertions, so disable the empty-completion guard here — it is
        # exercised directly in test_query_transient_and_empty.
        empty_completion_guard_enabled=False,
    )
    engine = engine_factory(rc=rc)
    # A tool that returns a large body every call (forces inflation).
    big = "DATA " * 1200
    tool = MockTool(tool_name="ReadBig", description="read a big file", response_content=big)
    in_memory_runtime["tools"].register(tool)

    llm = in_memory_runtime["llm"]
    # Script ``iterations`` tool calls then a terminal text answer.
    for i in range(iterations):
        llm.queue_tool_call_response(
            tool_call_id=f"t{i}", tool_name="ReadBig", tool_input={"path": f"f{i}"}
        )
    llm.queue_response(text="final answer")

    user_msg = Message(
        role=MessageRole.user, content_blocks=[TextBlock(text="ORIGINAL TASK: read all the files")]
    )
    events: list[TurnEvent] = []
    async for evt in engine.run(user_msg):
        events.append(evt)
    return engine, llm, events, rc


@pytest.mark.asyncio
async def test_per_iteration_gate_bounds_prompt_and_fires_compaction(
    engine_factory, in_memory_runtime
) -> None:
    engine, llm, events, rc = await _drive_long_tool_chain(
        engine_factory, in_memory_runtime, per_iteration_enabled=True
    )

    # The run reaches a terminal state (no terminal LLMContextWindowExceeded).
    assert engine.state in (LoopState.COMPLETED, LoopState.RUNNING)

    # Proactive (non-reactive) compaction fired at least once.
    proactive = [
        e for e in events
        if e.type is EventType.COMPACTION_STARTED
        and str(e.payload.get("reason", "")).startswith("proactive_per_iteration")
    ]
    assert proactive, "per-iteration compaction never fired"

    # No reactive-413 compaction was needed.
    reactive = [
        e for e in events
        if e.type is EventType.COMPACTION_STARTED
        and e.payload.get("reason") == "reactive_413"
    ]
    assert not reactive

    # The per-call prompt estimate plateaus — the max never runs away to a
    # multiple of the trigger threshold (it is bounded near it).
    budgets = derive_budgets(rc)
    estimates = [_prompt_estimate(req) for req in llm.calls if req.messages]
    assert estimates
    assert max(estimates) <= budgets.compaction_emergency_tokens + budgets.tool_result_truncation_threshold

    # The original task survives verbatim in history.
    assert any(
        "ORIGINAL TASK: read all the files" in m.text for m in engine.history
    )


@pytest.mark.asyncio
async def test_per_iteration_gate_kill_switch_reverts_to_turn_start_only(
    engine_factory, in_memory_runtime
) -> None:
    _engine, llm, events, rc = await _drive_long_tool_chain(
        engine_factory, in_memory_runtime, per_iteration_enabled=False
    )
    # With the gate OFF, NO per-iteration proactive compaction fires (the
    # turn-start gate already ran once before the first stream and history is
    # rebuilt unbounded thereafter — the regression behaviour, kept available
    # as a kill-switch).
    per_iter = [
        e for e in events
        if e.type is EventType.COMPACTION_STARTED
        and str(e.payload.get("reason", "")).startswith("proactive_per_iteration")
    ]
    assert not per_iter

    # Contrast proof: with the gate OFF a provider-bound prompt grows past the
    # proactive trigger. The hard request ceiling still prevents an oversized
    # call and routes the next attempt through reactive compaction.
    budgets = derive_budgets(rc)
    requests = [request for request in llm.calls if request.messages]
    estimates = [_prompt_estimate(request) for request in requests]
    assert estimates
    assert max(estimates) > budgets.compaction_trigger_tokens
    assert all(
        estimate + request.max_tokens <= rc.model_context_window
        for estimate, request in zip(estimates, requests, strict=True)
    )


@pytest.mark.asyncio
async def test_observed_prompt_size_does_not_drive_history_only_compaction(
    engine_factory, in_memory_runtime
) -> None:
    """A count from an older wire envelope cannot size history by itself.

    The provider reports a deliberately enormous count on a tiny first action
    request. The following per-iteration gate has only history, not the full
    current request's model/tool/context envelope, so it must rely on the
    calibrated current-history estimate and leave the stale scalar out.
    """
    rc = LoopConstants(
        model_context_window=4_096,
        compaction_per_iteration_enabled=True,
        compaction_keep_recent_turns=50,
        compaction_failed_max_retries=10,
    )
    engine = engine_factory(rc=rc)
    tool = MockTool(tool_name="Ping", description="ping", response_content="pong")
    in_memory_runtime["tools"].register(tool)

    llm = in_memory_runtime["llm"]
    # The assistant makes one tool call; the provider reports a real prompt
    # size of 147_892 tokens for that call (2.25x the 4096 window here).
    llm.queue_tool_call_response(
        tool_call_id="t0",
        tool_name="Ping",
        tool_input={"path": "f0"},
        usage_input_tokens=147_892,
    )
    llm.queue_response(text="final answer")

    budgets = derive_budgets(rc)
    # Sanity: the char estimate of this history never approaches the trigger.
    user_msg = Message(
        role=MessageRole.user, content_blocks=[TextBlock(text="ping the server")]
    )
    events: list[TurnEvent] = []
    async for evt in engine.run(user_msg):
        events.append(evt)

    assert estimate_history_tokens(engine.history, rc) < budgets.compaction_trigger_tokens

    all_started = [e for e in events if e.type is EventType.COMPACTION_STARTED]
    assert all_started == []
    assert engine.state is LoopState.COMPLETED
    assert engine.last_observed_prompt_tokens == 147_892


# ---------------------------------------------------------------------------
# The per-iteration gate must NEVER compact the current
# iteration's just-executed tool batch, regardless of batch size, on top of
# the keep window. A >keep parallel batch's fresh, unconsumed results would
# otherwise be blobbed/summarised before the next assistant stream sees them.
# ---------------------------------------------------------------------------
def _large_parallel_batch_history(*, fanout: int, window: int) -> list[Message]:
    """Build: task → assistant turn emitting ``fanout`` parallel tool_use →
    ``fanout`` large tool results (one per call). The whole batch is the most
    recent thing in history (the just-executed iteration)."""
    big = "RESULT DATA LINE\n" * 4000  # each result is well over the threshold
    use_blocks = [
        ToolUseBlock(tool_call_id=f"p{i}", name="ReadBig", arguments_json=f'{{"f":{i}}}')
        for i in range(fanout)
    ]
    history: list[Message] = [
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="ORIGINAL TASK")]),
        # A couple of AGED prior turns so there IS eligible-to-compact history
        # outside the protected batch (otherwise the test would pass vacuously).
        Message(
            role=MessageRole.assistant,
            content_blocks=[ToolUseBlock(tool_call_id="old", name="ReadBig", arguments_json="{}")],
        ),
        Message(
            role=MessageRole.tool,
            content_blocks=[ToolResultBlock(tool_call_id="old", content="OLD DATA\n" * 4000)],
        ),
        # The CURRENT iteration's batch: one assistant turn with N parallel
        # tool_use blocks, followed by N tool-result messages.
        Message(role=MessageRole.assistant, content_blocks=list(use_blocks)),
        *[
            Message(
                role=MessageRole.tool,
                content_blocks=[ToolResultBlock(tool_call_id=f"p{i}", content=big)],
            )
            for i in range(fanout)
        ],
    ]
    return history


@pytest.mark.asyncio
async def test_tier1_protects_current_parallel_batch() -> None:
    """>keep parallel tool results in the current batch are NEVER blobbed by
    the per-iteration Tier-1 gate, even though they fall outside the keep
    window. The aged prior result (genuinely old) IS still compacted, proving
    the gate is otherwise active."""
    rc = LoopConstants(model_context_window=4_096, compaction_keep_recent_turns=4)
    blobs = InMemoryBlobStore()
    fanout = 6  # > keep_recent_turns (4): the 5th/6th-from-last are outside it
    history = _large_parallel_batch_history(fanout=fanout, window=4_096)
    batch_assistant_idx = 3  # index of the current parallel tool_use turn

    # The helper the per-iteration gate uses must point at the current batch.
    protect_idx = current_tool_batch_protect_index(history)
    assert protect_idx == batch_assistant_idx

    budgets = derive_budgets(rc)
    await run_tier1_truncation(
        history=history,
        blob_store=blobs,
        tenant_id="t",
        rc=rc,
        truncation_threshold_tokens=budgets.tool_result_truncation_threshold,
        protect_tail_from_index=protect_idx,
    )

    # EVERY tool result in the current batch is present and INTACT (the model
    # has not consumed them yet — they must survive for the next assistant turn).
    for offset in range(fanout):
        result_msg = history[batch_assistant_idx + 1 + offset]
        block = result_msg.content_blocks[0]
        assert isinstance(block, ToolResultBlock)
        assert not is_compacted_placeholder(block.content), (
            f"current-batch result {offset} was compacted in-iteration"
        )
        assert block.content.startswith("RESULT DATA LINE")

    # The current assistant tool_use turn is also untouched (still N tool_use).
    batch_assistant = history[batch_assistant_idx]
    assert sum(isinstance(b, ToolUseBlock) for b in batch_assistant.content_blocks) == fanout

    # Sanity: the genuinely AGED prior result (idx 2, before the batch) WAS
    # compacted — the gate is active, the protection is surgical not a no-op.
    aged_block = history[2].content_blocks[0]
    assert isinstance(aged_block, ToolResultBlock)
    assert is_compacted_placeholder(aged_block.content)


@pytest.mark.asyncio
async def test_without_protection_current_batch_would_be_compacted() -> None:
    """Contrast / regression proof: WITHOUT ``protect_tail_from_index`` (the
    pre-fix behaviour), the keep window alone leaves the >keep oldest results
    of the current batch eligible — they get compacted. This is exactly the
    context-loss the in-iteration protection prevents."""
    rc = LoopConstants(model_context_window=4_096, compaction_keep_recent_turns=4)
    blobs = InMemoryBlobStore()
    fanout = 6
    history = _large_parallel_batch_history(fanout=fanout, window=4_096)
    batch_assistant_idx = 3

    budgets = derive_budgets(rc)
    await run_tier1_truncation(
        history=history,
        blob_store=blobs,
        tenant_id="t",
        rc=rc,
        truncation_threshold_tokens=budgets.tool_result_truncation_threshold,
        # protect_tail_from_index intentionally NOT passed → keep-window only.
    )

    # With keep=4 and a fanout-6 batch, the 2 OLDEST batch results fall outside
    # the keep window and ARE compacted (the bug). At least one current-batch
    # result is lost — demonstrating why the protect index is required.
    compacted = [
        offset
        for offset in range(fanout)
        if is_compacted_placeholder(history[batch_assistant_idx + 1 + offset].content_blocks[0].content)
    ]
    assert compacted, "expected the unprotected keep-window path to lose batch results"


@pytest.mark.asyncio
async def test_tier2_protects_current_parallel_batch() -> None:
    """Tier-2 summarisation also exempts the current batch when the gate passes
    ``protect_tail_from_index`` — the batch's assistant turn + results are not
    summarised away before consumption."""
    from protocore.tests_support.adapters import InMemoryLLMProvider

    rc = LoopConstants(model_context_window=4_096, compaction_keep_recent_turns=4)
    fanout = 6
    history = _large_parallel_batch_history(fanout=fanout, window=4_096)
    protect_idx = current_tool_batch_protect_index(history)
    assert protect_idx == 3  # the current parallel tool_use turn

    llm = InMemoryLLMProvider()
    # Queue enough responses that re-summarising the batch WOULD succeed if it
    # were eligible — proving the skip is the protection, not a dry LLM.
    for _ in range(fanout + 2):
        llm.queue_response(text="SUMMARY")

    result = await run_tier2_summarisation(
        history=history,
        compaction_llm=llm,
        state=CompactionState(),
        rc=rc,
        model_name="m",
        protect_tail_from_index=protect_idx,
    )

    # Tier-2 mutates history in place (drops summarised members + replaces the
    # anchor), so assert by CONTENT not fixed index. The current batch's
    # assistant tool_use turn must survive intact (not collapsed into a
    # <compacted-turn> system summary), with all N parallel tool_use blocks.
    batch_assistants = [
        m
        for m in history
        if m.role is MessageRole.assistant
        and sum(isinstance(b, ToolUseBlock) for b in m.content_blocks) == fanout
    ]
    assert len(batch_assistants) == 1, "the protected parallel batch turn was lost/altered"
    # Every result body in the current batch survives verbatim (none summarised
    # away, none turned into a placeholder).
    surviving_results = [
        b.content
        for m in history
        if m.role is MessageRole.tool
        for b in m.content_blocks
        if isinstance(b, ToolResultBlock) and b.content.startswith("RESULT DATA LINE")
    ]
    assert len(surviving_results) == fanout
    # The only thing Tier-2 may summarise is the genuinely aged prior pair
    # (idx 1-2 at call time), never the protected batch.
    assert result.turns_summarised <= 1


def test_protect_index_none_when_no_tool_use_turn() -> None:
    """No assistant tool_use turn → no in-flight batch → ``None`` (callers fall
    back to the plain keep-window behaviour)."""
    history = [
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="task")]),
        Message(role=MessageRole.assistant, content_blocks=[TextBlock(text="plain answer")]),
    ]
    assert current_tool_batch_protect_index(history) is None
    assert current_tool_batch_protect_index([]) is None


# ---------------------------------------------------------------------------
# vLLM-400 fix — Tier-2 summary replacement is USER-role (Layer 1, source)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_tier2_summary_replacement_is_user_role_with_flag() -> None:
    """The Tier-2 summary turn is USER-role (vLLM 400s on mid-array system) and
    keeps the durable summary flag + ``<compacted-turn>`` wrapper.

    FAIL before the fix (the replacement was ``MessageRole.system``), PASS after.
    """
    from protocore.tests_support.adapters import InMemoryLLMProvider

    rc = LoopConstants(model_context_window=4_096, compaction_keep_recent_turns=1)
    llm = InMemoryLLMProvider()
    llm.queue_response(text="User asked X; assistant did Y.")
    history = [
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="the original task " * 8)]),
        Message(role=MessageRole.assistant, content_blocks=[TextBlock(text="aged answer " * 60)]),
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="recent")]),
    ]
    result = await run_tier2_summarisation(
        history=history, compaction_llm=llm, state=CompactionState(), rc=rc, model_name="m"
    )
    assert result.turns_summarised == 1

    summaries = [m for m in history if m.metadata.get(COMPACTION_SUMMARY_METADATA_KEY) is True]
    assert len(summaries) == 1
    summary = summaries[0]
    # The load-bearing assertion: a Tier-2 summary is NEVER system-role.
    assert summary.role is MessageRole.user
    assert summary.text.startswith("<compacted-turn")
    # And there is no system message anywhere in the mutated history.
    assert all(m.role is not MessageRole.system for m in history)


# ---------------------------------------------------------------------------
# vLLM-400 fix — legacy system-role summary mid-array is still recognised as a
# summary by the unit-builder (Layer 1, backward recognition of snapshots)
# ---------------------------------------------------------------------------
def test_legacy_system_role_summary_is_recognised_and_skipped_by_unit_builder() -> None:
    """A persisted-snapshot system-role ``<compacted-turn>`` mid-array is treated
    as an already-compacted summary: ``_is_compaction_summary`` matches it AND
    the unit-builder never folds it into a summarisation unit."""
    legacy_summary = Message(
        role=MessageRole.system,
        content_blocks=[TextBlock(text="<compacted-turn id='legacy'>prior</compacted-turn>")],
        metadata={COMPACTION_SUMMARY_METADATA_KEY: True},
    )
    assert _is_compaction_summary(legacy_summary) is True

    history = [
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="task")]),
        legacy_summary,
        Message(role=MessageRole.assistant, content_blocks=[TextBlock(text="aged " * 20)]),
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="recent")]),
    ]
    # eligible_upper excludes the trailing keep-window; index 1 (the legacy
    # summary) is inside the eligible region but must NOT become a unit.
    units = _build_summarisation_units(history, eligible_upper=3)
    summary_anchored = [u for u in units if 1 in u.indices or u.anchor_idx == 1]
    assert summary_anchored == [], "legacy system-role summary must not be summarised again"
