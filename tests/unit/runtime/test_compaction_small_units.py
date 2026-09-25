"""A history made only of small rounds can still be compacted.

The shape comes from a long-running agent loop: hundreds of rounds of one short
tool call and a short result each, every one below the operator's minimum unit
size, with earlier summaries and the run's own task in between. Tier 2 skipped
every unit under the floor and the fold only takes summaries and operator
turns, so no tier could take a token off such a history however large it grew:
each pass freed nothing and the run failed on the retry budget. Adjacent small
units are now summarised together.
"""
from __future__ import annotations

import json
import re
from typing import Any

import pytest

from protocore.contracts.llm import LLMRequest, LLMResponse, StopReason
from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.types import (
    COMPACTION_SUMMARY_METADATA_KEY,
    SESSION_HISTORY_SEED_METADATA_KEY,
    Message,
    MessageRole,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from protocore.runtime.context.compaction import (
    CompactionState,
    estimate_history_tokens,
    run_tier2_summarisation,
    tier2_has_work,
)
from protocore.runtime.context.manager import ContextManager
from protocore.tests_support.adapters import InMemoryBlobStore, InMemoryLLMProvider


class _Summariser(InMemoryLLMProvider):
    """Answers with a summary a little over half the length the prompt allows.

    A summariser that returns one short sentence whatever it is given would make
    any grouping look like a large gain; this one writes what it was asked for.
    """

    def __init__(self) -> None:
        super().__init__()
        self.prompts: list[str] = []

    async def complete_structured(
        self, request: LLMRequest, response_schema: dict[str, Any]
    ) -> LLMResponse:
        prompt = request.messages[-1].text
        self.prompts.append(prompt)
        allowed = int(re.search(r"about (\d+) characters", prompt).group(1))  # type: ignore[union-attr]
        summary = ("checked one case, exit 0; " * 400)[: int(allowed * 0.6)]
        return LLMResponse(
            message=Message(
                role=MessageRole.assistant,
                content_blocks=[TextBlock(text=json.dumps({"summary": summary}))],
            ),
            stop_reason=StopReason.end_turn,
        )


def _rc(**overrides: Any) -> LoopConstants:
    base: dict[str, Any] = {
        "model_context_window": 256_000,
        "llm_output_max_tokens_ratio": 0.256,
        "compaction_trigger_ratio": 0.59,
        "compaction_summary_min_unit_tokens": 1_500,
        "compaction_summary_max_output_tokens": 4_096,
        "compaction_protect_first_user_turn": True,
    }
    base.update(overrides)
    return LoopConstants(**base)


def _round(n: int, *, result_chars: int = 2_400, seeded: bool = False) -> list[Message]:
    """One round of the live shape: a short command and a few dozen lines back."""
    call_id = f"call-{n}"
    metadata = {SESSION_HISTORY_SEED_METADATA_KEY: True} if seeded else {}
    line = f"ok    case-{n:03d}  covered_by_the_checksums(...) -> True   entries 130 ok 128\n"
    return [
        Message(
            role=MessageRole.assistant,
            content_blocks=[
                ToolUseBlock(
                    tool_call_id=call_id,
                    name="Exec",
                    arguments_json=json.dumps({"command": f"python3 check.py --case {n}"}),
                )
            ],
            metadata=metadata,
        ),
        Message(
            role=MessageRole.tool,
            content_blocks=[
                ToolResultBlock(tool_call_id=call_id, content=(line * 60)[:result_chars])
            ],
            metadata=metadata,
        ),
    ]


def _summary(text: str) -> Message:
    return Message(
        role=MessageRole.user,
        content_blocks=[TextBlock(text=f"<compacted-turn id='k{len(text)}'>{text}</compacted-turn>")],
        metadata={COMPACTION_SUMMARY_METADATA_KEY: True},
    )


def _live_shaped_history(rounds: int) -> list[Message]:
    """Seeded earlier-run rounds, the run's task, then small rounds with the odd summary between."""
    history: list[Message] = []
    for n in range(3):
        history += _round(1_000 + n, seeded=True)
    history.append(
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="Loop iteration: run the checks.")])
    )
    for n in range(rounds):
        history += _round(n)
        if n % 25 == 24:
            history.append(_summary("Earlier rounds ran the checks one by one; all passed."))
    return history


def _pairing_is_whole(history: list[Message]) -> bool:
    uses = {
        block.tool_call_id
        for message in history
        for block in message.content_blocks
        if isinstance(block, ToolUseBlock)
    }
    results = {
        block.tool_call_id
        for message in history
        for block in message.content_blocks
        if isinstance(block, ToolResultBlock)
    }
    return uses == results


def test_every_round_of_the_live_shape_is_below_the_floor() -> None:
    """The precondition the failure rests on: not one round is worth a call by itself."""
    rc = _rc()
    history = _live_shaped_history(40)
    from protocore.runtime.context.compaction import estimate_message_tokens

    sizes = [
        estimate_message_tokens(history[i], rc) + estimate_message_tokens(history[i + 1], rc)
        for i in range(len(history) - 1)
        if history[i].role is MessageRole.assistant
    ]
    assert sizes and max(sizes) < rc.compaction_summary_min_unit_tokens


@pytest.mark.asyncio
async def test_a_history_of_small_rounds_is_compacted_in_groups() -> None:
    rc = _rc()
    history = _live_shaped_history(160)
    before = estimate_history_tokens(history, rc)
    llm = _Summariser()

    assert tier2_has_work(history, CompactionState(), rc)
    result = await run_tier2_summarisation(
        history, llm, CompactionState(), rc, model_name="m", free_target_tokens=None
    )

    after = estimate_history_tokens(history, rc)
    assert result.turns_summarised > 0
    assert after < before * 0.75
    # One call per group, not one per round.
    assert len(llm.prompts) < 160 // 2
    assert _pairing_is_whole(history)
    # The seeded rounds and the run's task are untouched.
    assert sum(1 for m in history if m.metadata.get(SESSION_HISTORY_SEED_METADATA_KEY)) == 6
    assert any(m.text == "Loop iteration: run the checks." for m in history)


@pytest.mark.asyncio
async def test_the_proactive_pass_makes_progress_where_it_used_to_exhaust() -> None:
    """The live failure: every pass freed nothing, and the third one raised."""
    rc = _rc(compaction_failed_max_retries=2)
    history = _live_shaped_history(160)
    manager = ContextManager(rc=rc, blob_store=InMemoryBlobStore(), compaction_llm=_Summariser())
    state = CompactionState()

    attempt = await manager.run_compaction(
        history=history, compaction_state=state, tenant_id="t", model_name="m"
    )

    assert attempt.tokens_after < attempt.tokens_before
    assert attempt.tier2 is not None and attempt.tier2.turns_summarised > 0
    assert state.retry_count == 0


@pytest.mark.asyncio
async def test_a_group_does_not_cross_a_summary_or_mix_provenance() -> None:
    rc = _rc(compaction_keep_recent_turns=1)
    history: list[Message] = []
    for n in range(4):
        history += _round(100 + n, seeded=True)
    for n in range(4):
        history += _round(n)
    history.append(_summary("An earlier summary."))
    for n in range(4, 8):
        history += _round(n)
    history.append(Message(role=MessageRole.user, content_blocks=[TextBlock(text="recent")]))
    llm = _Summariser()

    await run_tier2_summarisation(
        history, llm, CompactionState(), rc, model_name="m", compact_seeded_history=True
    )

    # Three groups: the seeded rounds, the rounds before the summary, the rounds after it.
    assert len(llm.prompts) == 3
    assert all("case-10" not in p or "case-000" not in p for p in llm.prompts)
    seeded_summaries = [
        m
        for m in history
        if m.metadata.get(COMPACTION_SUMMARY_METADATA_KEY)
        and m.metadata.get(SESSION_HISTORY_SEED_METADATA_KEY)
    ]
    assert len(seeded_summaries) == 1
    assert any(m.text == "<compacted-turn id='k19'>An earlier summary.</compacted-turn>" for m in history)
    assert _pairing_is_whole(history)


@pytest.mark.asyncio
async def test_a_group_stops_at_its_cap_and_zero_turns_grouping_off() -> None:
    history = _live_shaped_history(30)
    capped = _Summariser()
    await run_tier2_summarisation(
        list(history),
        capped,
        CompactionState(),
        _rc(compaction_summary_group_max_tokens=3_000),
        model_name="m",
    )
    wide = _Summariser()
    await run_tier2_summarisation(
        list(history),
        wide,
        CompactionState(),
        _rc(compaction_summary_group_max_tokens=12_000),
        model_name="m",
    )
    assert len(capped.prompts) > len(wide.prompts) > 0

    off = _Summariser()
    unchanged = list(history)
    await run_tier2_summarisation(
        unchanged,
        off,
        CompactionState(),
        _rc(compaction_summary_group_max_tokens=0),
        model_name="m",
    )
    assert off.prompts == []
    assert unchanged == history
