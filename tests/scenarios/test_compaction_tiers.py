"""Compaction over a long conversation: what each tier takes, and what it refuses.

A window under pressure is shed in three passes. The first sheds tool-result
bytes to a blob. The second summarises old turns — but never a turn the
operator wrote, because an instruction is short enough that summarising it
frees nothing and specific enough that a paraphrase changes it. The third
folds what the second leaves behind: over a long session, one summary per
tool batch and every operator message become the whole window, and neither of
the passes below can take a byte off them.

Every assertion here is on what a host can see: the requests the providers
received, the events the caller iterated, and the transcript that survives a
snapshot.
"""
from __future__ import annotations

import asyncio
import json
import random
from typing import Any

import pytest

from protocore.contracts.llm import LLMRequest
from protocore.contracts.types import StopReason
from protocore.runtime.events import EventType
from protocore.tests_support.adapters import InMemoryLLMProvider

from .conftest import ScenarioFactory, ScriptedTool, default_rc


class Summariser(InMemoryLLMProvider):
    """The provider compaction talks to, with the fan-out it saw recorded.

    ``peak_in_flight`` is the widest the pass ever got. It is the only way to
    tell a bounded batch from an unbounded fan-out from outside: both make the
    same number of calls and both finish.
    """

    def __init__(self, summary: str = "the old turns, summarised") -> None:
        super().__init__()
        self.summary = summary
        self.peak_in_flight = 0
        self._in_flight = 0

    async def complete_structured(
        self, request: LLMRequest, response_schema: dict[str, Any]
    ) -> Any:
        self._in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self._in_flight)
        try:
            # A real call suspends; without a suspension point here every
            # coroutine in the batch would run to completion before the next
            # one started and the fan-out would be invisible.
            await asyncio.sleep(0)
            self.queue_response(text=json.dumps({"summary": self.summary}))
            return await super().complete_structured(request, response_schema)
        finally:
            self._in_flight -= 1

    @property
    def prompts(self) -> list[str]:
        """What the summariser was asked, one string per call."""
        return [call.messages[0].text for call in self.calls]


def _tiered_rc(**overrides: Any) -> Any:
    """Constants that put all three tiers within reach of a short scenario."""
    values: dict[str, Any] = {
        "model_context_window": 2_048,
        "compaction_trigger_ratio": 0.4,
        "compaction_emergency_ratio": 0.6,
        "compaction_keep_recent_turns": 2,
        "compaction_fold_min_messages": 4,
        "compaction_fold_min_tokens": 0,
        "compaction_fold_keep_operator_turns": 1,
        "compaction_fold_max_spans_per_pass": 4,
    }
    values.update(overrides)
    return default_rc(**values)


async def _long_conversation(
    run: Any,
    *,
    turns: int,
    tool: ScriptedTool | None = None,
    instruction: str = "step {index}: keep going",
) -> None:
    """Drive ``turns`` turns, each one an operator instruction and a reply.

    Between turns the engine is re-armed, which is what a host does when the
    same session asks a second question.
    """
    for index in range(turns):
        if index:
            run.engine.rearm()
        if tool is not None:
            run.llm.queue_tool_call_response(
                tool_call_id=f"call-{index}",
                tool_name=tool.name,
                tool_input={"v": str(index)},
            )
        run.llm.queue_response(
            text=f"answer {index}: " + "prose that is long enough to be worth summarising " * 8,
            stop_reason=StopReason.end_turn,
        )
        await run.run(instruction.format(index=index))


# ---------------------------------------------------------------------------
# All three tiers, over one conversation
# ---------------------------------------------------------------------------


async def test_a_long_conversation_crosses_all_three_compaction_tiers(
    scenario: ScenarioFactory,
) -> None:
    """Each tier's work shows up in the completion event the caller saw.

    The tiers are not alternatives: Tier 1 sheds tool bytes, Tier 2 summarises
    the turns around them, and Tier 3 folds the summaries Tier 2 leaves once
    there are enough of them. A conversation long enough reaches all three, and
    a reader watching the events can say which pass did what.
    """
    summariser = Summariser()
    tool = ScriptedTool(tool_name="Note", content="R" * 4_000)
    run = scenario(rc=_tiered_rc(), tools=[tool], compaction_provider=summariser)

    await _long_conversation(run, turns=14, tool=tool)

    completions = [
        evt.payload for evt in run.events_of(EventType.COMPACTION_COMPLETED)
    ]
    assert completions, "a conversation this long must have compacted at least once"
    assert any(payload["tier1_freed"] > 0 for payload in completions)
    assert any(payload["tier2_summarised"] > 0 for payload in completions)
    assert any(payload["tier3_folded"] > 0 for payload in completions)


async def test_the_fold_leaves_one_message_standing_for_many(
    scenario: ScenarioFactory,
) -> None:
    """What the fold buys is a shorter transcript, not merely a shorter tier list."""
    summariser = Summariser()
    tool = ScriptedTool(tool_name="Note", content="R" * 4_000)
    run = scenario(rc=_tiered_rc(), tools=[tool], compaction_provider=summariser)

    await _long_conversation(run, turns=14, tool=tool)

    folded = [
        payload["tier3_folded"]
        for payload in (evt.payload for evt in run.events_of(EventType.COMPACTION_COMPLETED))
        if payload["tier3_folded"] > 0
    ]
    assert folded and max(folded) >= 4


# ---------------------------------------------------------------------------
# What compaction refuses to paraphrase
# ---------------------------------------------------------------------------


async def test_an_operator_turn_is_never_paraphrased_away(
    scenario: ScenarioFactory,
) -> None:
    """An instruction either stands verbatim or is quoted, and never condensed.

    Two things are asserted together because either alone would pass on a
    transcript that had thrown the words away: the per-turn summariser was
    never handed an operator turn to condense, and every instruction is still
    readable somewhere — verbatim in the transcript, or as an exact quote in
    the fold that replaced it.
    """
    summariser = Summariser()
    tool = ScriptedTool(tool_name="Note", content="R" * 4_000)
    run = scenario(rc=_tiered_rc(), tools=[tool], compaction_provider=summariser)
    instruction = "step {index}: remove the model-name field from the header"

    await _long_conversation(run, turns=12, tool=tool, instruction=instruction)

    per_turn = [p for p in summariser.prompts if p.lstrip().startswith("<turn>")]
    folds = [p for p in summariser.prompts if p.lstrip().startswith("<turns>")]
    surviving = "\n".join(run.history_texts())
    quoted = "\n".join(folds)
    for index in range(12):
        words = instruction.format(index=index)
        # Never the thing the per-turn summariser was asked to condense.
        for prompt in per_turn:
            assert words not in prompt, f"turn {index} was handed to the summariser"
        assert words in surviving or f"[operator said] {words}" in quoted, (
            f"the operator's turn {index} is readable nowhere"
        )


@pytest.mark.parametrize("seed", [1, 7, 99, 2026])
async def test_every_identifier_reaches_the_summariser_verbatim(
    scenario: ScenarioFactory, seed: int
) -> None:
    """A summary that rounds an id is worse than no summary: it reads as fact.

    The core cannot make a model obey, but it owns the two halves that decide
    whether obeying is possible — that the exact characters go into the prompt,
    and that the prompt says to keep them. Seeded ids rather than one
    hand-picked string, because what this guards against is a class of value
    (a path, a port, a digest, a status code) and not one value.
    """
    rng = random.Random(seed)
    identifiers = [
        f"/srv/{rng.randrange(10**6):06d}/build.log",
        f"port {rng.randrange(1024, 65535)}",
        f"sha256:{rng.randrange(16**12):012x}",
        f"HTTP {rng.choice([400, 404, 409, 502])}",
        f"run-{rng.randrange(10**8):08d}",
    ]
    summariser = Summariser()
    run = scenario(rc=_tiered_rc(), compaction_provider=summariser)

    for index, identifier in enumerate(identifiers):
        if index:
            run.engine.rearm()
        run.llm.queue_response(
            text=f"answer {index}: touched {identifier}. " + "prose to make the turn worth summarising " * 12,
            stop_reason=StopReason.end_turn,
        )
        await run.run(f"step {index}: keep going")
    # A few more turns, so the early ones are well outside the keep window.
    for index in range(5, 12):
        run.engine.rearm()
        run.llm.queue_response(
            text=f"answer {index}: " + "prose to make the turn worth summarising " * 12,
            stop_reason=StopReason.end_turn,
        )
        await run.run(f"step {index}: keep going")

    per_turn = [p for p in summariser.prompts if p.lstrip().startswith("<turn>")]
    assert per_turn, "nothing was summarised, so nothing was proved"
    seen = "\n".join(summariser.prompts)
    for identifier in identifiers:
        assert identifier in seen, f"{identifier!r} never reached the summariser"
    assert all(
        "verbatim" in prompt and "never round, guess or substitute" in prompt for prompt in per_turn
    )


async def test_the_summariser_is_told_that_a_missing_result_is_an_unknown(
    scenario: ScenarioFactory,
) -> None:
    """Absence of evidence is not evidence of either outcome.

    A turn with no tool result and no confirmation used to be summarised as if
    the action had happened, or as if it had not; both are inventions, and the
    run acts on them afterwards.
    """
    summariser = Summariser()
    tool = ScriptedTool(tool_name="Note", content="R" * 4_000)
    run = scenario(rc=_tiered_rc(), tools=[tool], compaction_provider=summariser)

    await _long_conversation(run, turns=8, tool=tool)

    per_turn = [p for p in summariser.prompts if p.lstrip().startswith("<turn>")]
    assert per_turn
    assert all("state unknowns as unknown" in prompt for prompt in per_turn)
    folds = [p for p in summariser.prompts if p.lstrip().startswith("<turns>")]
    assert all("state unknown outcomes as unknown" in prompt for prompt in folds)


# ---------------------------------------------------------------------------
# What one pass is allowed to cost
# ---------------------------------------------------------------------------


async def test_summariser_calls_are_bounded_by_the_configured_width(
    scenario: ScenarioFactory,
) -> None:
    """A pass is a chain of seconds-long calls otherwise, with the run parked.

    Widening it is what stops a long history from sitting in COMPACTING for
    minutes; the cap is what stops the cure from being an unbounded fan-out at
    the provider.
    """
    summariser = Summariser()
    tool = ScriptedTool(tool_name="Note", content="R" * 4_000)
    run = scenario(
        rc=_tiered_rc(compaction_summariser_parallelism=2),
        tools=[tool],
        compaction_provider=summariser,
    )

    await _long_conversation(run, turns=12, tool=tool)

    assert summariser.peak_in_flight == 2


async def test_a_unit_too_small_to_shrink_is_never_sent(
    scenario: ScenarioFactory,
) -> None:
    """The net-gain guard discards such a summary — but only after paying for it."""
    summariser = Summariser()
    tool = ScriptedTool(tool_name="Note", content="R" * 4_000)
    run = scenario(
        rc=_tiered_rc(compaction_summary_min_unit_tokens=100_000),
        tools=[tool],
        compaction_provider=summariser,
    )

    await _long_conversation(run, turns=10, tool=tool)

    per_turn = [p for p in summariser.prompts if p.lstrip().startswith("<turn>")]
    assert per_turn == []
    # Compaction still ran — Tier 1 has bytes to shed whatever the floor says.
    assert run.events_of(EventType.COMPACTION_COMPLETED)


# ---------------------------------------------------------------------------
# Across a process boundary
# ---------------------------------------------------------------------------


async def test_a_cold_resume_does_not_re_summarise_what_was_already_summarised(
    scenario: ScenarioFactory,
) -> None:
    """The dedup key is content-addressed, so it survives the snapshot.

    A resumed run that summarised the same turns again would pay twice and
    decay the summary each time. What makes it not happen is in the snapshot,
    so the only honest test of it is a NEW engine restored from one.
    """
    summariser = Summariser()
    tool = ScriptedTool(tool_name="Note", content="R" * 4_000)
    first = scenario(rc=_tiered_rc(), tools=[tool], compaction_provider=summariser)

    await _long_conversation(first, turns=8, tool=tool)
    snapshot = first.engine.snapshot()
    already = set(snapshot["compaction"]["summarised_turn_ids"])
    assert already, "nothing was summarised, so the resume proves nothing"
    calls_before = len(summariser.calls)

    resumed = scenario(rc=_tiered_rc(), tools=[tool], compaction_provider=summariser)
    await resumed.engine.resume_from_snapshot(snapshot)
    resumed.engine.rearm()
    resumed.llm.queue_response(text="answer after the resume", stop_reason=StopReason.end_turn)
    await resumed.run("and now finish")

    # Whatever the resumed run summarised, it was not a turn already carrying a
    # key the snapshot brought back.
    resumed_snapshot = resumed.engine.snapshot()
    assert already <= set(resumed_snapshot["compaction"]["summarised_turn_ids"])
    fresh_prompts = [call.messages[0].text for call in summariser.calls[calls_before:]]
    assert all("<compacted-turn" not in prompt.split("</turn>")[0] for prompt in fresh_prompts)
