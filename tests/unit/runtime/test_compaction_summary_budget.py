"""The word budget a summary is asked for is sized for the script it is written in."""
from __future__ import annotations

import json

import pytest

from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.types import Message, MessageRole, TextBlock
from protocore.runtime.context.compaction import (
    CompactionState,
    _summary_word_budget,
    run_tier2_summarisation,
)
from protocore.tests_support.adapters import InMemoryLLMProvider


def test_the_budget_never_exceeds_what_the_output_cap_holds_at_the_going_rate() -> None:
    rc = LoopConstants(
        model_context_window=4_096,
        compaction_summary_max_output_tokens=4_096,
        compaction_summary_output_tokens_per_word=4,
        # Raised out of the way so the OUTPUT cap is the ceiling under test.
        compaction_summary_string_max_chars=32_768,
    )
    # A large unit: scaled by its size the budget would be far above the ceiling.
    assert _summary_word_budget(200_000, rc) == (4_096 - 32) // 4


def test_a_small_unit_keeps_the_floor() -> None:
    rc = LoopConstants(model_context_window=4_096)
    assert _summary_word_budget(100, rc) == rc.compaction_summary_min_words


def test_the_per_word_cost_is_configurable_not_the_english_figure() -> None:
    rc = LoopConstants(
        model_context_window=4_096,
        compaction_summary_max_output_tokens=4_096,
        compaction_summary_string_max_chars=32_768,
    )
    cyrillic = _summary_word_budget(200_000, rc)
    english = _summary_word_budget(
        200_000, rc.model_copy(update={"compaction_summary_output_tokens_per_word": 2})
    )
    # Two tokens a word was the English figure; the default is four, so the
    # budget asked for on the same unit is half of what it used to be.
    assert rc.compaction_summary_output_tokens_per_word == 4
    assert english == (4_096 - 32) // 2
    assert cyrillic == (4_096 - 32) // 4


def test_the_stock_budget_fits_the_stock_output_cap_in_a_non_latin_script() -> None:
    rc = LoopConstants(model_context_window=65_536)
    budget = _summary_word_budget(1_000_000, rc)
    assert budget * rc.compaction_summary_output_tokens_per_word <= (
        rc.compaction_summary_max_output_tokens
    )


@pytest.mark.asyncio
async def test_the_summariser_prompt_states_the_budget_in_characters_too() -> None:
    """A model holds to a length it can count directly better than to a word
    count it has to estimate, and a reply the output cap cuts is discarded."""
    rc = LoopConstants(
        model_context_window=4_096,
        compaction_keep_recent_turns=1,
        compaction_protect_first_user_turn=False,
        compaction_summary_min_unit_tokens=0,
    )
    history = [
        Message(
            role=MessageRole.assistant,
            content_blocks=[TextBlock(text="ran a tool and read the output " * 40)],
        ),
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="recent")]),
    ]
    llm = InMemoryLLMProvider()
    llm.queue_response(text=json.dumps({"summary": "short"}))

    await run_tier2_summarisation(
        history=history,
        compaction_llm=llm,
        state=CompactionState(),
        rc=rc,
        model_name="mock",
    )

    assert llm.calls
    prompt = llm.calls[0].messages[0].text
    assert "characters" in prompt
    assert "cut off and discarded" in prompt
    assert '{"summary": "..."}' in prompt
    assert "a long tool result is not copied" in prompt
    # The character figure is the word budget at the configured rate.
    import re

    stated_words = int(re.search(r"at most (\d+) words", prompt).group(1))  # type: ignore[union-attr]
    stated_chars = int(re.search(r"about (\d+) characters", prompt).group(1))  # type: ignore[union-attr]
    assert stated_chars == stated_words * rc.compaction_summary_chars_per_word


def test_the_ceiling_leaves_room_for_the_json_around_the_words() -> None:
    """A budget that spends the output cap exactly is a summary cut one token
    short of closing its envelope — the very failure the ceiling exists for."""
    rc = LoopConstants(model_context_window=65_536)
    budget = _summary_word_budget(1_000_000, rc)
    words_cost = budget * rc.compaction_summary_output_tokens_per_word
    assert words_cost < rc.compaction_summary_max_output_tokens
    assert (
        words_cost + rc.compaction_summary_envelope_tokens
        <= rc.compaction_summary_max_output_tokens
    )
    # Stock values: 512 // 4 would have been 128 words, the whole cap.
    assert budget < rc.compaction_summary_max_output_tokens // (
        rc.compaction_summary_output_tokens_per_word
    )


def test_the_budget_never_asks_for_more_characters_than_the_grammar_accepts() -> None:
    """``compaction_summary_string_max_chars`` is the decode-time ``maxLength``
    on the summary string. A prompt asking for more is asking for a reply the
    grammar must cut."""
    rc = LoopConstants(
        model_context_window=65_536,
        compaction_summary_max_output_tokens=4_096,
    )
    budget = _summary_word_budget(1_000_000, rc)
    assert (
        budget * rc.compaction_summary_chars_per_word
        <= rc.compaction_summary_string_max_chars
    )


def test_an_envelope_allowance_that_eats_the_whole_output_cap_is_refused() -> None:
    with pytest.raises(ValueError, match="compaction_summary_envelope_tokens"):
        LoopConstants(
            compaction_summary_max_output_tokens=32,
            compaction_summary_envelope_tokens=32,
        )


def test_the_grammar_cap_outranks_the_floor_rather_than_being_overrun() -> None:
    """A budget too small to say much is recoverable; one the grammar must cut
    comes back unusable, so the cap wins even against the floor."""
    rc = LoopConstants(
        compaction_summary_min_words=25,
        compaction_summary_chars_per_word=6,
        compaction_summary_string_max_chars=60,
    )
    budget = _summary_word_budget(1_000_000, rc)
    assert budget == 10
    assert budget * rc.compaction_summary_chars_per_word <= 60
