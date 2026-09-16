"""Stale tool results are cut down; fresh ones, pins and persist are not."""
from __future__ import annotations

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
from protocore.runtime.stale_result_trim import trim_stale_results
from tests._fixtures.tool_roles import CONVENTIONAL_TOOL_ROLES

PROMPTS = bundled_prompt_provider()


def _rc(**overrides: object) -> LoopConstants:
    values: dict[str, object] = {
        "tool_result_stale_trim_enabled": True,
        "tool_result_fresh_count": 2,
        "tool_result_stale_max_chars": 100,
        "tool_result_stale_trim_batch_chars": 500,
    }
    values.update(overrides)
    return LoopConstants(**values)  # type: ignore[arg-type]


def _pair(call_id: str, size: int, **result_fields: object) -> list[Message]:
    """One call and its answer, the answer ``size`` characters long."""
    return [
        Message(
            role=MessageRole.assistant,
            content_blocks=[ToolUseBlock(tool_call_id=call_id, name="Read", arguments_json="{}")],
        ),
        Message(
            role=MessageRole.user,
            content_blocks=[
                ToolResultBlock(tool_call_id=call_id, content="x" * size, **result_fields)
            ],
        ),
    ]


def _results(view: list[Message]) -> dict[str, ToolResultBlock]:
    return {
        block.tool_call_id: block
        for message in view
        for block in message.content_blocks
        if isinstance(block, ToolResultBlock)
    }


def _turn(call_id: str, size: int) -> list[Message]:
    """A whole turn: the operator asks, one call goes out, one result answers it."""
    return [
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="and now this")]),
        *_pair(call_id, size),
    ]


def _transcript(count: int, size: int = 900, prefix: str = "call") -> list[Message]:
    history: list[Message] = []
    for index in range(count):
        history.extend(_turn(f"{prefix}_{index}", size))
    return history


def test_the_oldest_results_are_cut_and_the_newest_are_not() -> None:
    history = _transcript(5)
    view, trimmed = trim_stale_results(history, _rc(), PROMPTS)
    assert sorted(trimmed) == ["call_0", "call_1", "call_2"]
    results = _results(view)
    for call_id in ("call_0", "call_1", "call_2"):
        assert results[call_id].content.startswith("x" * 100)
        assert "call the tool again" in results[call_id].content
        assert results[call_id].metadata["stale_trimmed"] is True
    for call_id in ("call_3", "call_4"):
        assert results[call_id].content == "x" * 900
        assert "stale_trimmed" not in results[call_id].metadata


def test_the_pointer_says_how_much_went() -> None:
    view, _ = trim_stale_results(_transcript(4), _rc(), PROMPTS)
    assert "[trimmed 800 chars: this result is older than the last 2 tool calls" in (
        _results(view)["call_0"].content
    )


def test_persist_is_never_touched() -> None:
    history = _transcript(5)
    before = [message.model_dump() for message in history]
    trim_stale_results(history, _rc(), PROMPTS)
    assert [message.model_dump() for message in history] == before


def test_nothing_happens_below_the_batch_threshold() -> None:
    # Three trimmable results at 800 chars of excess each is 2400 — under a
    # batch of 10_000, so the prefix is left exactly where it was.
    history = _transcript(5)
    view, trimmed = trim_stale_results(
        history, _rc(tool_result_stale_trim_batch_chars=10_000), PROMPTS
    )
    assert trimmed == frozenset()
    assert view == list(history)


def test_short_results_are_left_alone_however_old() -> None:
    history = _transcript(2, size=50, prefix="short") + _transcript(3)
    view, trimmed = trim_stale_results(history, _rc(), PROMPTS)
    assert "short_0" not in trimmed
    assert _results(view)["short_0"].content == "x" * 50


def test_a_trimmed_result_stays_trimmed_after_the_batch_that_cut_it() -> None:
    history = _transcript(5)
    _, trimmed = trim_stale_results(history, _rc(), PROMPTS)
    # The next build has nothing new worth a batch of its own — the decision
    # still has to hold, or the whole result comes back and the prefix moves.
    view, still = trim_stale_results(
        history,
        _rc(tool_result_stale_trim_batch_chars=10_000),
        PROMPTS,
        already_trimmed=trimmed,
    )
    assert still == trimmed
    assert len(_results(view)["call_0"].content) < 900


def test_a_pinned_result_is_never_cut() -> None:
    history = _transcript(5)
    view, trimmed = trim_stale_results(history, _rc(), PROMPTS, pinned_ids={"call_0"})
    assert "call_0" not in trimmed
    assert _results(view)["call_0"].content == "x" * 900


def test_a_marked_result_is_never_cut() -> None:
    history = _pair("kept", 900, metadata={"retention": "pinned"}) + _transcript(4)
    view, trimmed = trim_stale_results(history, _rc(), PROMPTS)
    assert "kept" not in trimmed
    assert _results(view)["kept"].content == "x" * 900


def test_a_compacted_placeholder_is_never_rewritten() -> None:
    placeholder = PROTOCOL_COMPACTED_TOOL_RESULT_V1 + " " + "y" * 900
    history = [
        *_pair("folded", 0),
        *_transcript(4),
    ]
    history[1] = Message(
        role=MessageRole.user,
        content_blocks=[ToolResultBlock(tool_call_id="folded", content=placeholder)],
    )
    view, trimmed = trim_stale_results(history, _rc(), PROMPTS)
    assert "folded" not in trimmed
    assert _results(view)["folded"].content == placeholder


def test_the_batch_in_flight_is_never_cut() -> None:
    # Three calls issued together; the fresh window holds only two of them, and
    # the third is still an answer the model is reading right now.
    history = _transcript(4)
    history.append(
        Message(
            role=MessageRole.assistant,
            content_blocks=[
                ToolUseBlock(tool_call_id="a", name="Read", arguments_json="{}"),
                ToolUseBlock(tool_call_id="b", name="Read", arguments_json="{}"),
                ToolUseBlock(tool_call_id="c", name="Read", arguments_json="{}"),
            ],
        )
    )
    for call_id in ("a", "b", "c"):
        history.append(
            Message(
                role=MessageRole.user,
                content_blocks=[ToolResultBlock(tool_call_id=call_id, content="x" * 900)],
            )
        )
    _, trimmed = trim_stale_results(history, _rc(), PROMPTS)
    # call_3 belongs to the same turn as the batch — the operator has not
    # spoken since — so it is exempt too, outside the fresh window as it is.
    assert trimmed == {"call_0", "call_1", "call_2"}


def test_a_turn_of_many_rounds_keeps_every_result_it_made() -> None:
    # One question, ten sequential reads of a large file. The fresh window is
    # six; the other four are still this turn's own work, and an answer is
    # still being written from them.
    history: list[Message] = [
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="one question")])
    ]
    for index in range(10):
        history.extend(_pair(f"round_{index}", 20_000))
    view, trimmed = trim_stale_results(
        history,
        LoopConstants(tool_result_stale_trim_enabled=True),
        PROMPTS,
    )
    assert trimmed == frozenset()
    assert view == list(history)


def test_a_sticky_id_the_view_no_longer_carries_is_forgotten() -> None:
    history = _transcript(5)
    _, trimmed = trim_stale_results(history, _rc(), PROMPTS)
    # The checkpoint dropped the oldest turn; its id must not ride along in
    # the set the engine keeps and every snapshot writes.
    _, still = trim_stale_results(history[3:], _rc(), PROMPTS, already_trimmed=trimmed)
    assert "call_0" not in still
    assert "call_1" in still


def test_a_trimmed_result_fits_the_split_that_runs_after_it() -> None:
    # The two limits are independent: a deployment that keeps more head than
    # the split allows would otherwise have the split cut the trim pointer in
    # half, and the line saying how to get the result back with it.
    history = _transcript(5, size=4000)
    view, trimmed = trim_stale_results(
        history,
        _rc(
            tool_result_stale_max_chars=3000,
            tool_result_split_enabled=True,
            tool_result_content_max_chars=500,
        ),
        PROMPTS,
    )
    assert "call_0" in trimmed
    content = _results(view)["call_0"].content
    assert len(content) <= 500
    assert content.endswith("call the tool again if you need it]")


def test_the_switch_off_leaves_the_view_alone() -> None:
    history = _transcript(5)
    view, trimmed = trim_stale_results(
        history, _rc(tool_result_stale_trim_enabled=False), PROMPTS
    )
    assert view == list(history)
    assert trimmed == frozenset()


def test_prose_blocks_are_carried_through_untouched() -> None:
    history = [
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")]),
        *_transcript(5),
    ]
    view, _ = trim_stale_results(history, _rc(), PROMPTS)
    assert view[0] is history[0]


def test_a_pin_the_run_has_since_falsified_is_cut_like_any_other() -> None:
    # The pinned read describes a file the run rewrote two calls later. Holding
    # it whole keeps the version before the write in front of the model, which
    # is worse than handing back its head and saying to read the file again.
    history = [
        Message(
            role=MessageRole.assistant,
            content_blocks=[
                ToolUseBlock(
                    tool_call_id="read", name="Read", arguments_json='{"path": "a.py"}'
                )
            ],
        ),
        Message(
            role=MessageRole.user,
            content_blocks=[
                ToolResultBlock(tool_call_id="read", content="x" * 900, path="a.py")
            ],
        ),
        Message(
            role=MessageRole.assistant,
            content_blocks=[
                ToolUseBlock(
                    tool_call_id="write", name="Write", arguments_json='{"path": "a.py"}'
                )
            ],
        ),
        Message(
            role=MessageRole.user,
            content_blocks=[ToolResultBlock(tool_call_id="write", content="written")],
        ),
        *_transcript(4),
    ]
    kept, _ = trim_stale_results(history, _rc(), PROMPTS, pinned_ids={"read"})
    assert _results(kept)["read"].content == "x" * 900

    cut, trimmed = trim_stale_results(
        history, _rc(), PROMPTS, pinned_ids={"read"}, roles=CONVENTIONAL_TOOL_ROLES
    )
    assert "read" in trimmed
    assert "call the tool again" in _results(cut)["read"].content


def test_a_pin_over_a_file_nothing_rewrote_survives_the_roles_being_declared() -> None:
    history = _pair("kept", 900, path="untouched.py") + _transcript(4)
    view, trimmed = trim_stale_results(
        history, _rc(), PROMPTS, pinned_ids={"kept"}, roles=CONVENTIONAL_TOOL_ROLES
    )
    assert "kept" not in trimmed
    assert _results(view)["kept"].content == "x" * 900
