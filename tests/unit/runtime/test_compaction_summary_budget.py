"""A summary's budget follows what it replaces, and is held by clamping sections, not by cutting text.

The old carrier asked for a word count and then capped the one JSON string at
1,024 characters; whatever a unit held past that was gone, and a reply the
cap cut was never parsed at all. These tests pin what replaced it: a budget in
tokens proportional to the span, a request cap with room to overshoot, and a
clamp that cuts each section at a line (or word) boundary so one long section
cannot push the exact values out.
"""
from __future__ import annotations

import pytest

from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.types import Message, MessageRole, TextBlock
from protocore.runtime.context.carrier import (
    read_carrier,
    request_max_tokens,
    summary_output_budget,
)
from protocore.runtime.context.compaction import CompactionState, run_tier2_summarisation
from protocore.runtime.token_counting import estimate_tokens
from protocore.tests_support.adapters import InMemoryLLMProvider


def test_the_budget_is_a_share_of_what_the_summary_replaces() -> None:
    rc = LoopConstants(compaction_summary_ratio=0.2, compaction_summary_min_output_tokens=100)
    assert summary_output_budget(5_000, rc, ceiling=2_048) == 1_000


def test_a_small_span_keeps_the_floor_and_a_large_one_the_ceiling() -> None:
    rc = LoopConstants(compaction_summary_ratio=0.2, compaction_summary_min_output_tokens=256)
    assert summary_output_budget(300, rc, ceiling=2_048) == 256
    assert summary_output_budget(1_000_000, rc, ceiling=2_048) == 2_048
    # The floor never outranks the ceiling of the kind of call it is.
    assert summary_output_budget(300, rc, ceiling=100) == 100


def test_the_request_leaves_room_to_overshoot_the_budget() -> None:
    assert request_max_tokens(1_000) == 2_000


def test_an_overlong_section_is_cut_at_a_line_and_the_others_keep_their_share() -> None:
    rc = LoopConstants()
    progress = "\n".join(f"- step {i}: ran the check for shard {i} and it passed" for i in range(400))
    reply = f"## Progress\n{progress}\n## Facts and values\n- port 62114\n- lock /opt/kestrel/state/install.lock"

    carrier = read_carrier(reply, budget_tokens=300, rc=rc)

    assert carrier.clamped
    assert estimate_tokens(carrier.text, rc) <= 330
    assert "- port 62114" in carrier.text
    assert "- lock /opt/kestrel/state/install.lock" in carrier.text
    # Every kept line is a whole line of the reply.
    kept = [line for line in carrier.text.split("\n") if line.startswith("- step")]
    assert kept and all(line in progress.split("\n") for line in kept)


def test_a_single_overlong_line_is_cut_at_a_word_never_inside_one() -> None:
    rc = LoopConstants()
    words = " ".join(f"value-{i:04d}" for i in range(2_000))
    carrier = read_carrier(f"## Facts and values\n{words}", budget_tokens=120, rc=rc)
    body = carrier.text.split("\n", 1)[1]
    assert body.endswith(" …")
    assert all(token.startswith("value-") and len(token) == len("value-0000") for token in body[:-2].split())


@pytest.mark.asyncio
async def test_the_request_carries_the_budget_of_the_unit_it_summarises() -> None:
    rc = LoopConstants(
        model_context_window=65_536,
        compaction_keep_recent_turns=1,
        compaction_protect_first_user_turn=False,
    )
    llm = InMemoryLLMProvider()
    llm.queue_response(text="## Progress\nsummarised")
    unit = Message(role=MessageRole.assistant, content_blocks=[TextBlock(text="слово " * 3_000)])
    history = [unit, Message(role=MessageRole.user, content_blocks=[TextBlock(text="recent")])]

    await run_tier2_summarisation(
        history=history,
        compaction_llm=llm,
        state=CompactionState(),
        rc=rc,
        model_name="mock",
    )

    before = estimate_tokens(unit.text, rc)
    budget = summary_output_budget(before, rc, ceiling=rc.compaction_summary_max_output_tokens)
    assert llm.calls[0].max_tokens == request_max_tokens(budget)
    assert f"about {max(40, budget // 2)} words" in llm.calls[0].messages[0].text
