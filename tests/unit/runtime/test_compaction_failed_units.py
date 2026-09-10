"""A summary the cap would cut, and a unit the summariser keeps failing on."""

from __future__ import annotations

import pytest

from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.types import Message, MessageRole, TextBlock
from protocore.runtime.context.compaction import (
    CompactionState,
    _summary_word_budget,
    run_tier2_summarisation,
)
from protocore.tests_support.adapters import InMemoryLLMProvider


def test_the_word_budget_never_asks_for_more_than_the_cap_can_hold_in_any_script() -> None:
    rc = LoopConstants(model_context_window=4_096, compaction_summary_max_output_tokens=4096, compaction_summary_output_tokens_per_word=4)
    # A large unit: scaled by size it would be far above the ceiling.
    assert _summary_word_budget(200_000, rc) == 1024
    # A small unit keeps the floor.
    assert _summary_word_budget(100, rc) == rc.compaction_summary_min_words
    # Two tokens a word was the English figure; the constant is what sizes it now.
    rc_english = rc.model_copy(update={"compaction_summary_output_tokens_per_word": 2})
    assert _summary_word_budget(200_000, rc_english) == 2048


def _history() -> list[Message]:
    return [
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="hello there " * 40)]),
        Message(role=MessageRole.assistant, content_blocks=[TextBlock(text="hi back " * 40)]),
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="recent")]),
    ]


@pytest.mark.asyncio
async def test_a_unit_the_summariser_keeps_failing_on_is_left_alone_after_the_limit() -> None:
    class _Failing(InMemoryLLMProvider):
        calls = 0

        async def complete_structured(self, request, response_schema):  # type: ignore[no-untyped-def]
            _Failing.calls += 1
            raise RuntimeError("output truncated by max_tokens")

    rc = LoopConstants(model_context_window=4_096, compaction_keep_recent_turns=1, compaction_summary_failed_unit_max_attempts=2)
    history = _history()
    state = CompactionState()
    llm = _Failing()
    for _ in range(4):
        result = await run_tier2_summarisation(history=history, compaction_llm=llm, state=state, rc=rc, model_name="mock")
        assert result.turns_summarised == 0
    # Two passes paid for the failure; the next two did not.
    assert _Failing.calls == 2
    assert list(state.failed_anchor_keys.values()) == [2]
    assert history[2].text == "recent"
