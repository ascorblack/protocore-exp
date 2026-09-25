"""The compaction contract, clause by clause (``docs/compaction.md``).

Each test names the clause it pins and, where there was one, the failure it
comes from: a summariser that returned an unclosed JSON object, one that
answered prose where JSON was due, one that was down, one that hung, a history
of small rounds no tier could shrink, a provider refusal after which the
summariser could not run because its own input did not fit.
"""
from __future__ import annotations

import asyncio
import json
import random
import re
from pathlib import Path
from typing import Any

import pytest

from protocore.contracts.llm import LLMRequest, LLMResponse
from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.tool_roles import ToolRole, ToolRoleMap
from protocore.contracts.types import (
    COMPACTION_SUMMARY_METADATA_KEY,
    SESSION_HISTORY_SEED_METADATA_KEY,
    Message,
    MessageRole,
    StopReason,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from protocore.runtime.context.compaction import (
    CompactionState,
    estimate_history_tokens,
    render_span_for_summary,
    run_tier1_truncation,
    run_tier2_summarisation,
    run_tier3_fold,
    summariser_input_cap,
)
from protocore.runtime.context.ledger import (
    LEDGER_METADATA_KEY,
    Ledger,
    distinct_lines,
    is_ledger,
    ledger_from_history,
)
from protocore.runtime.context.manager import ContextManager
from protocore.runtime.token_counting import estimate_tokens
from protocore.tests_support.adapters import InMemoryBlobStore, InMemoryLLMProvider
from tests._fixtures.compaction import planted

# ---------------------------------------------------------------------------
# doubles
# ---------------------------------------------------------------------------


class _Summariser(InMemoryLLMProvider):
    """Answers every call in a scripted way: ``ok``, ``echo``, ``drop``, ``empty``,
    ``json``, ``unterminated``, ``error``, ``hang``, or ``cut`` (a reply the output
    cap stopped), cycling through ``modes``."""

    def __init__(self, *modes: str) -> None:
        super().__init__()
        self.modes = list(modes) or ["ok"]
        self.seen: list[LLMRequest] = []

    async def complete_text(self, request: LLMRequest) -> LLMResponse:
        self.seen.append(request)
        mode = self.modes[(len(self.seen) - 1) % len(self.modes)]
        material = request.messages[-1].text
        stop = StopReason.end_turn
        if mode == "error":
            raise RuntimeError("502 bad gateway")
        if mode == "hang":
            await asyncio.sleep(60)
        if mode == "empty":
            text = ""
        elif mode == "json":
            text = json.dumps({"summary": "ran the checks; case 12 timed out"})
        elif mode == "unterminated":
            text = '{"summary": "Turn 1: ran check.py --case 7; exit 0 (UNKNOWN)."'
        elif mode == "echo":
            # Keeps every line of the material that carries a value, the way a
            # careful summariser keeps the facts it was shown.
            kept = [line for line in material.split("\n") if re.search(r"\d", line)][:60]
            text = "## Progress\nchecked the sections\n## Facts and values\n" + "\n".join(kept)
        elif mode == "drop":
            text = "## Progress\nchecked more sections; nothing notable"
        elif mode == "cut":
            text = "## Progress\nstep one done\nstep two was hal"
            stop = StopReason.max_tokens
        else:
            text = "## Progress\nchecked the sections\n## Open\nnext: continue"
        return LLMResponse(
            message=Message(role=MessageRole.assistant, content_blocks=[TextBlock(text=text)] if text else []),
            stop_reason=stop,
        )


def _round(n: int, *, output: str, tool: str = "Exec", error: bool = False) -> list[Message]:
    call = f"call-{n}"
    return [
        Message(
            role=MessageRole.assistant,
            content_blocks=[
                TextBlock(text=f"Checking section {n}."),
                ToolUseBlock(tool_call_id=call, name=tool, arguments_json=json.dumps({"command": f"inspect --section {n}"})),
            ],
        ),
        Message(role=MessageRole.tool, content_blocks=[ToolResultBlock(tool_call_id=call, content=output, is_error=error)]),
    ]


def _log(n: int, lines: int = 60) -> str:
    return "\n".join(
        f"2026-07-1{i % 10}T0{i % 10}:1{i % 6}:00Z INFO job ledger-compact-{n * 100 + i:05d} finished in {i * 7} ms"
        for i in range(lines)
    )


def _history(rounds: int, *, lines: int = 60) -> list[Message]:
    history = [Message(role=MessageRole.user, content_blocks=[TextBlock(text="Migrate the ledger. Never restart production without asking.")])]
    for n in range(rounds):
        history += _round(n, output=_log(n, lines))
    return history


def _manager(llm: Any, **rc: Any) -> ContextManager:
    values: dict[str, Any] = {"model_context_window": 32_768, "compaction_keep_recent_turns": 2}
    values.update(rc)
    return ContextManager(rc=LoopConstants(**values), blob_store=InMemoryBlobStore(), compaction_llm=llm)


def _pairing_is_whole(history: list[Message]) -> bool:
    uses = {b.tool_call_id for m in history for b in m.content_blocks if isinstance(b, ToolUseBlock)}
    results = {b.tool_call_id for m in history for b in m.content_blocks if isinstance(b, ToolResultBlock)}
    return uses == results


# ---------------------------------------------------------------------------
# Progress: every opened pass ends below the trigger or at the floor
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "modes",
    [("ok",), ("error",), ("empty",), ("json",), ("unterminated",), ("cut",), ("ok", "error", "empty")],
)
async def test_every_pass_ends_below_the_trigger_whatever_the_summariser_does(modes: tuple[str, ...]) -> None:
    llm = _Summariser(*modes)
    manager = _manager(llm)
    history = _history(60)
    state = CompactionState()
    trigger = manager.rc.model_context_window  # replaced below by the attempt's own figure

    attempt = await manager.run_compaction(
        history=history, compaction_state=state, tenant_id="t", model_name="m", overhead_tokens=8_000
    )

    trigger = attempt.trigger_tokens
    assert attempt.prompt_before > trigger
    assert attempt.prompt_after <= trigger or attempt.outcome == "at_floor"
    assert attempt.outcome in {"below_target", "below_trigger"}
    assert _pairing_is_whole(history)
    assert history[0].text.startswith("Migrate the ledger.")
    assert state.retry_count == 0


async def test_repeated_passes_with_a_dead_summariser_neither_loop_nor_fail() -> None:
    """The live incident: 75 passes of 0-3 tokens each, then the run failed.

    Every pass here meets a history that has grown by a few rounds and a
    summariser that is down. Each one still ends below the trigger, and the
    retry budget is never touched.
    """
    llm = _Summariser("error")
    manager = _manager(llm)
    history = _history(40)
    state = CompactionState()
    outcomes = []
    for more in range(30):
        history += _round(1_000 + more, output=_log(1_000 + more))
        # The gate, as the loop runs it: a pass is opened only over the trigger.
        if not manager.needs_compaction(history, overhead_tokens=8_000):
            continue
        attempt = await manager.run_compaction(
            history=history, compaction_state=state, tenant_id="t", model_name="m", overhead_tokens=8_000
        )
        outcomes.append(attempt.outcome)
        assert attempt.prompt_after <= attempt.trigger_tokens
    assert outcomes
    assert state.retry_count == 0
    assert _pairing_is_whole(history)
    assert "at_floor" not in outcomes


async def test_a_history_at_its_floor_is_not_opened_again() -> None:
    manager = _manager(_Summariser("error"))
    history = [
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="the task " * 400)]),
        Message(role=MessageRole.assistant, content_blocks=[TextBlock(text="working")]),
    ]
    assert not manager.has_proactive_work(history, CompactionState(), force=True)


# ---------------------------------------------------------------------------
# Tier order: masking first, with a pointer and the lines an output said once
# ---------------------------------------------------------------------------


async def test_old_outputs_are_masked_before_anything_is_summarised() -> None:
    llm = _Summariser("ok")
    manager = _manager(llm, compaction_mask_keep_recent_results=4)
    history = _history(30, lines=40)

    attempt = await manager.run_compaction(
        history=history, compaction_state=CompactionState(), tenant_id="t", model_name="m", overhead_tokens=15_000
    )

    assert attempt.tier1 is not None and attempt.tier1.masked_by_age > 0
    assert llm.seen, "masking alone does not reach the target here, so the summariser ran after it"


async def test_a_masked_output_says_what_it_was_and_where_it_went() -> None:
    rc = LoopConstants(model_context_window=32_768, compaction_mask_keep_recent_results=1)
    history = _history(4, lines=40)
    result = await run_tier1_truncation(
        history, InMemoryBlobStore(), "t", rc, truncation_threshold_tokens=10**9,
        keep_recent_turns=1, mask_by_age=True,
    )
    assert result.masked_by_age == 3
    masked = [b for m in history for b in m.content_blocks if isinstance(b, ToolResultBlock) and b.canonical_ref]
    readable = masked[0].content.split("\n", 1)[1]
    assert readable.startswith("[The output of Exec (")
    assert "was masked by compaction; the original is stored as" in readable
    assert "First and last lines:" in readable


def test_a_masked_output_keeps_the_lines_it_said_once() -> None:
    output = _log(3, 50).split("\n")
    output.insert(20, "admin.bind: 0.0.0.0:47031")
    kept = distinct_lines("\n".join(output), limit=6)
    assert kept == ["admin.bind: 0.0.0.0:47031"]


async def test_the_summariser_reads_a_masked_output_in_full() -> None:
    """A summary of a placeholder is a summary of nothing; the original is in the blob store."""
    blobs = InMemoryBlobStore()
    rc = LoopConstants(model_context_window=32_768, compaction_keep_recent_turns=1, compaction_protect_first_user_turn=False)
    history = [
        *_round(1, output=_log(1, 80) + "\nsigning key: kid-7F3A9C21"),
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="recent")]),
    ]
    await run_tier1_truncation(history, blobs, "t", rc, truncation_threshold_tokens=50, keep_recent_turns=1)
    assert "kid-7F3A9C21" in history[1].content_blocks[0].content  # kept as a distinct line
    llm = _Summariser("ok")
    await run_tier2_summarisation(
        history, llm, CompactionState(), rc, model_name="m", blob_store=blobs, tenant_id="t"
    )
    material = llm.seen[0].messages[-1].text
    assert "ledger-compact-00179" in material  # a repeated log line: only the original has it


# ---------------------------------------------------------------------------
# A failed or absent summary is a normal outcome; the input is always bounded
# ---------------------------------------------------------------------------


async def test_a_summariser_that_hangs_is_abandoned_at_its_deadline() -> None:
    llm = _Summariser("hang")
    rc = LoopConstants(
        model_context_window=32_768,
        compaction_keep_recent_turns=1,
        compaction_protect_first_user_turn=False,
        compaction_summary_timeout_seconds=0.05,
    )
    history = [*_history(6), Message(role=MessageRole.user, content_blocks=[TextBlock(text="recent")])]

    result = await asyncio.wait_for(
        run_tier2_summarisation(history, llm, CompactionState(), rc, model_name="m"), timeout=5
    )

    assert result.turns_summarised == 0
    assert result.failures == {"timeout": result.units_attempted}


def test_the_summariser_input_is_bounded_whatever_the_span_holds() -> None:
    """After a refusal the summariser is often the same model: its input must fit by construction."""
    rc = LoopConstants(model_context_window=32_768)
    cap = summariser_input_cap(rc)
    span = _round(1, output="x" * 400_000 + "\nthe last line") + _round(2, output=_log(2, 3_000))

    rendered = render_span_for_summary(span, rc, tool_names={}, originals={}, max_tokens=cap)

    assert estimate_tokens(rendered, rc) <= cap
    assert "the last line" in rendered  # head and tail survive, the middle goes
    assert "lines omitted" in rendered or "[cut]" in rendered


async def test_an_echoed_instruction_is_not_carried_forward() -> None:
    """The summary that presented the summariser's own instruction as the user's."""

    class _Echo(_Summariser):
        async def complete_text(self, request: LLMRequest) -> LLMResponse:
            self.seen.append(request)
            instruction = request.messages[0].text.split("\n")[0]
            return LLMResponse(
                message=Message(
                    role=MessageRole.assistant,
                    content_blocks=[TextBlock(text=f"## Progress\n{instruction}\nran the checks")],
                ),
                stop_reason=StopReason.end_turn,
            )

    rc = LoopConstants(model_context_window=32_768, compaction_keep_recent_turns=1, compaction_protect_first_user_turn=False)
    history = [*_history(4), Message(role=MessageRole.user, content_blocks=[TextBlock(text="recent")])]
    await run_tier2_summarisation(history, _Echo(), CompactionState(), rc, model_name="m")

    summary = next(m for m in history if m.metadata.get(COMPACTION_SUMMARY_METADATA_KEY))
    assert "You write the compaction summary" not in summary.text
    assert "ran the checks" in summary.text
    assert "written by the runtime, not by the user" in summary.text


# ---------------------------------------------------------------------------
# Facts from state: the ledger
# ---------------------------------------------------------------------------


async def test_the_ledger_carries_what_left_the_window_and_is_never_summarised() -> None:
    roles = ToolRoleMap.declare({"Write": [ToolRole.writes_path]})
    llm = _Summariser("drop")
    manager = ContextManager(
        rc=LoopConstants(model_context_window=32_768, compaction_keep_recent_turns=2),
        blob_store=InMemoryBlobStore(),
        compaction_llm=llm,
        tool_roles=roles,
    )
    history = _history(20)
    history.insert(9, Message(role=MessageRole.user, content_blocks=[TextBlock(text="Keep the cache origin at https://origin-7.example.net/assets.")]))
    history += [
        Message(
            role=MessageRole.assistant,
            content_blocks=[ToolUseBlock(tool_call_id="w1", name="Write", arguments_json=json.dumps({"path": "conf/pool.toml", "content": "x"}))],
        ),
        Message(role=MessageRole.tool, content_blocks=[ToolResultBlock(tool_call_id="w1", content="written")]),
    ]
    history += _history(30)[1:]
    state = CompactionState()

    for _ in range(3):
        await manager.force_compaction(history=history, compaction_state=state, tenant_id="t", model_name="m")

    ledgers = [m for m in history if is_ledger(m)]
    assert len(ledgers) == 1
    text = ledgers[0].text
    assert "wrote conf/pool.toml" in text
    # The operator's turn is either still there verbatim or quoted by the ledger.
    assert "https://origin-7.example.net/assets" in _context(history)
    assert all("<compaction-ledger>" not in request.messages[-1].text for request in llm.seen)
    # Never between a call and its result, and never the last word.
    position = history.index(ledgers[0])
    assert history[position + 1].role is not MessageRole.tool
    assert position < len(history) - 1
    assert _pairing_is_whole(history)


def test_the_ledger_is_rebuilt_from_its_own_state() -> None:
    ledger = Ledger()
    ledger.absorb(
        _round(1, output="lockfile /opt/kestrel/state/install.lock held by pid 4411\n" + _log(1, 20)),
        is_operator=lambda m: False,
        skip=lambda m: False,
    )
    rc = LoopConstants(model_context_window=32_768)
    message = Message(
        role=MessageRole.user,
        content_blocks=[TextBlock(text=ledger.render(rc))],
        metadata={LEDGER_METADATA_KEY: ledger.to_dict()},
    )
    restored = Message.model_validate(json.loads(message.model_dump_json()))
    again = ledger_from_history([restored])

    assert again.render(rc) == ledger.render(rc)
    assert "/opt/kestrel/state/install.lock" in again.render(rc)


async def test_operator_words_survive_a_fold_verbatim_by_code() -> None:
    """The fold used to be asked to quote the operator; a model paraphrases under pressure."""
    rule = "Never restart the primary; the owning department is Отдел расчётов-4."
    summary = lambda i: Message(  # noqa: E731
        role=MessageRole.user,
        content_blocks=[TextBlock(text=f"<compacted-turn id='k{i}'>\n## Progress\nstep {i} " + "done " * 80 + "\n</compacted-turn>")],
        metadata={COMPACTION_SUMMARY_METADATA_KEY: True},
    )
    history = [
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="the task")]),
        summary(0), summary(1),
        Message(role=MessageRole.user, content_blocks=[TextBlock(text=rule)]),
        summary(2), summary(3),
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="recent")]),
        Message(role=MessageRole.assistant, content_blocks=[TextBlock(text="working")]),
    ]
    rc = LoopConstants(
        model_context_window=32_768,
        compaction_keep_recent_turns=2,
        compaction_fold_min_tokens=0,
        compaction_fold_keep_operator_turns=1,
    )
    ledger = Ledger()
    result = await run_tier3_fold(
        history, _Summariser("drop"), CompactionState(), rc, model_name="m", ledger=ledger
    )

    assert result.spans_folded == 1
    assert all(m.text != rule for m in history)
    assert ledger.operator == [rule]


# ---------------------------------------------------------------------------
# Recursive folding over a long session: the planted-fact fixture
# ---------------------------------------------------------------------------


async def _fold_session(llm: Any, *, window: int = 40_000) -> tuple[list[Message], planted.Session]:
    session = planted.build()
    rc = LoopConstants(
        model_context_window=window,
        llm_output_max_tokens_ratio=0.2,
        compaction_keep_recent_turns=4,
        compaction_summary_min_unit_tokens=800,
    )
    manager = ContextManager(rc=rc, blob_store=InMemoryBlobStore(), compaction_llm=llm)
    state = CompactionState()
    history: list[Message] = []
    for chunk in session.chunks:
        history.extend(chunk)
        attempt = await manager.force_compaction(history=history, compaction_state=state, tenant_id="t", model_name="m")
        assert attempt.prompt_after <= attempt.trigger_tokens
        assert _pairing_is_whole(history)
    return history, session


def _context(history: list[Message]) -> str:
    return "\n".join(
        (getattr(b, "text", None) or getattr(b, "content", None) or getattr(b, "arguments_json", ""))
        for m in history
        for b in m.content_blocks
    )


async def test_a_summariser_that_keeps_nothing_still_leaves_the_shaped_values_in_the_ledger() -> None:
    """The worst summariser: every summary is one uninformative line."""
    history, session = await _fold_session(_Summariser("drop"))
    context = _context(history)
    carried = {label for label, value, *_ in session.graded if value in context}
    # Shape-recognisable values reach the carrier by code; the two with no
    # shape (a count with its unit, a Cyrillic name given in chat) are the
    # summary's job — the operator's own turn is quoted by the ledger.
    assert {"tenant UUID", "cache origin", "admin console port", "install lock path", "signing key id", "release tag"} <= carried
    assert len(carried) >= 10


async def test_a_summariser_that_keeps_the_facts_it_was_shown_carries_every_one() -> None:
    history, session = await _fold_session(_Summariser("echo"))
    context = _context(history)
    missing = [label for label, value, *_ in session.graded if value not in context]
    assert missing in ([], ["ledger shard count"])  # "19 shards": echo keeps lines with digits


# ---------------------------------------------------------------------------
# The long-loop shape that stopped making progress
# ---------------------------------------------------------------------------

_SHAPE = json.loads((Path(__file__).parents[2] / "_fixtures" / "compaction" / "long_loop_shape.json").read_text())


def _shaped_history() -> list[Message]:
    rng = random.Random(3)
    words = "ok check case entries covered exit elapsed sandbox passed result value table row".split()

    def text(n: int) -> str:
        out: list[str] = []
        while sum(len(w) + 1 for w in out) < n:
            out.append(rng.choice(words))
        return " ".join(out)[:n]

    history: list[Message] = []
    pending: list[str] = []
    for index, spec in enumerate(_SHAPE["messages"]):
        role = MessageRole(spec["role"])
        blocks: list[Any] = []
        for block in spec["blocks"]:
            if block[0] == "text":
                blocks.append(TextBlock(text=text(block[1]) or "."))
            elif block[0] == "call":
                call = f"c{index}-{len(blocks)}"
                pending.append(call)
                blocks.append(ToolUseBlock(tool_call_id=call, name=block[1], arguments_json=json.dumps({"command": text(max(0, block[2] - 16))})))
            elif block[0] == "result":
                blocks.append(ToolResultBlock(tool_call_id=pending.pop(0) if pending else f"orphan-{index}", content=text(block[1]), is_error=block[2]))
        if role is MessageRole.system or not blocks:
            continue
        metadata = {SESSION_HISTORY_SEED_METADATA_KEY: True} if spec["seeded"] else {}
        history.append(Message(role=role, content_blocks=blocks, metadata=metadata))
    return history


@pytest.mark.parametrize("modes", [("ok",), ("error",), ("unterminated", "empty", "json")])
async def test_the_long_loop_shape_is_compacted_below_the_trigger_in_one_pass(modes: tuple[str, ...]) -> None:
    """Hundreds of short rounds under 56k tokens of system prompt and tools.

    The live run gated on the whole prompt, could shrink only the history,
    found no unit worth a call and failed after 75 passes of 0-3 tokens. Here
    the same shape, sized as it was, under the constants it ran with: one pass
    brings the whole prompt under the trigger whatever the summariser does.
    """
    history = _shaped_history()
    rc = LoopConstants(
        model_context_window=256_000,
        llm_output_max_tokens_ratio=0.256,
        compaction_trigger_ratio=0.59,
        compaction_summary_min_unit_tokens=800,
        compaction_keep_recent_turns=4,
        # The factor the live run measured between the estimate and the count.
        token_estimate_calibration=1.135,
    )
    manager = ContextManager(rc=rc, blob_store=InMemoryBlobStore(), compaction_llm=_Summariser(*modes))
    # The rest of the live request — a 26 KB system prompt and 153 tool
    # definitions — is whatever makes the whole prompt the size the provider
    # counted: 151,847 tokens, just over the trigger.
    overhead = 151_847 - estimate_history_tokens(history, rc)
    assert 50_000 < overhead < 65_000

    attempt = await manager.run_compaction(
        history=history, compaction_state=CompactionState(), tenant_id="t", model_name="m", overhead_tokens=overhead
    )

    assert attempt.prompt_before > attempt.trigger_tokens
    assert attempt.prompt_after <= attempt.trigger_tokens
    assert attempt.outcome in {"below_target", "below_trigger"}
    assert _pairing_is_whole(history)
    assert all(m.metadata.get(SESSION_HISTORY_SEED_METADATA_KEY) for m in history[:3])
