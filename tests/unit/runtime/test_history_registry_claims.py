"""The registry's run-scope reasons, each pinned by the behaviour it asserts.

``tests/unit/runtime/test_history_run_boundary.py`` enumerates every function
that reaches the session transcript, each with the reason its answer has the
scope it has. Those reasons were prose, and prose is not enforcement: one of
them —

    QueryEngine.latest_user_message: "the tail-most user turn; the new task is
    appended after the seed, so it is always this run's"

— became false under a one-word edit (``reversed(self.history)`` →
``self.history``) with the entire suite still green. That edit is a live
run-scope defect, not a style problem: ``latest_user_message`` feeds tool
retrieval and skill-body loading, so a run would fetch its tool surface and load
its skills for a PRIOR run's task.

Every entry in that registry whose reason claims "this reaches the whole
session and yet answers about ONE run" names one of the tests in this file, and
the guard fails if a named test stops existing. An entry whose reason is
instead "this question is about the whole session by definition" names nothing,
because there is no run-scope property to hold — that is stated at the entry
rather than left implied.
"""
from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest

from protocore.contracts.interrupt import InterruptResolutionError
from protocore.contracts.llm import LLMRequest, LLMStreamEvent
from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.types import (
    SESSION_HISTORY_SEED_METADATA_KEY,
    SYNTHETIC_RECOVERY_BACKGROUND_WAKE,
    SYNTHETIC_RECOVERY_METADATA_KEY,
    SYNTHETIC_RECOVERY_PROSE_GATE_REPAIR,
    TERMINAL_TOOL_METADATA_KEY,
    Message,
    MessageRole,
    StopReason,
    TextBlock,
    ToolCall,
    ToolResultBlock,
    ToolUseBlock,
)
from protocore.runtime.context.compaction import _session_history_seed_indices
from protocore.runtime.longfile_convergence import _active_file_tail
from protocore.runtime.query import (
    _apply_updated_input,
    _assert_history_has_matching_pending_tool_use,
    _history_has_tool_result,
    _history_tool_result_is_terminal,
    _prose_gate_just_injected,
    _run_produced_output,
    _tool_call_from_history,
    _tool_name_for_call_id,
)
from protocore.runtime.query_engine import QueryEngine
from protocore.runtime.result_eviction import tool_name_for_result

#: The other half of the pin. The registry names the test that holds each
#: run-scope claim; each test names, here, the entries it was written to hold.
#: ``test_every_pinning_test_still_pins_what_it_claims`` requires the two to
#: agree exactly, which is what stops an entry being downgraded to
#: whole-transcript while the test written to hold its claim stays green beside
#: it. A one-sided pin was one paraphrase deep: reclassifying an entry with its
#: prose untouched passed the whole suite.
PINNED_ENTRIES: dict[str, tuple[str, ...]] = {
    "test_latest_user_message_is_this_runs_task_not_a_seeded_one": (
        "protocore/runtime/query_engine.py::QueryEngine.latest_user_message",
        "protocore/runtime/query_engine.py::QueryEngine.run",
    ),
    "test_prose_gate_reads_the_tail_and_not_a_seeded_turn": (
        "protocore/runtime/turn_policies/sibling_walk.py::prose_gate_just_injected",
    ),
    "test_produced_output_ignores_a_seeded_prior_run": (
        "protocore/runtime/query.py::_run_produced_output",
    ),
    "test_call_id_lookups_resolve_the_call_they_are_asked_for": (
        "protocore/runtime/query.py::_tool_name_for_call_id",
        "protocore/runtime/query.py::_history_has_tool_result",
        "protocore/runtime/query.py::_history_tool_result_is_terminal",
    ),
    "test_interrupt_resolution_is_keyed_on_the_parked_call": (
        "protocore/runtime/query.py::_tool_call_from_history",
        "protocore/runtime/query.py::_apply_updated_input",
    ),
    "test_pending_tool_use_assertion_is_keyed_on_the_approved_call": (
        "protocore/runtime/query.py::_assert_history_has_matching_pending_tool_use",
    ),
    "test_active_file_tail_needs_this_runs_own_binding": (
        "protocore/runtime/longfile_convergence.py::_active_file_tail",
    ),
    "test_seed_indices_select_every_seeded_turn_and_nothing_else": (
        "protocore/runtime/context/compaction.py::_session_history_seed_indices",
    ),
}

_PRIOR_TASK = "Draft the Q2 incident retrospective."
_NEW_TASK = "Summarise the Q3 headcount plan for the Berlin office."


def _seeded(message: Message) -> Message:
    """Tag ``message`` the way the executor tags a prior run's turn."""
    return message.model_copy(
        update={
            "metadata": {
                **message.metadata,
                SESSION_HISTORY_SEED_METADATA_KEY: True,
            }
        }
    )


def _user(text: str) -> Message:
    return Message(role=MessageRole.user, content_blocks=[TextBlock(text=text)])


def _assistant(*blocks: Any) -> Message:
    return Message(role=MessageRole.assistant, content_blocks=list(blocks))


def _tool_turn(*blocks: Any) -> Message:
    return Message(role=MessageRole.tool, content_blocks=list(blocks))


class _PlainTextEndTurnLLM:
    """A turn of ordinary prose that ends without calling any tool."""

    def __init__(self, *, text: str = "Berlin grows by four in Q3.") -> None:
        self._text = text
        self.requests: list[LLMRequest] = []

    async def stream_with_tools(  # type: ignore[no-untyped-def]
        self, request: LLMRequest
    ) -> AsyncIterator[LLMStreamEvent]:
        self.requests.append(request)
        yield LLMStreamEvent(name="message_start", payload={})
        yield LLMStreamEvent(name="content_block_start", payload={"kind": "text"})
        yield LLMStreamEvent(name="content_block_delta", payload={"text": self._text})
        yield LLMStreamEvent(name="content_block_stop", payload={})
        yield LLMStreamEvent(
            name="message_stop", payload={"stop_reason": StopReason.end_turn.value}
        )

    async def complete_structured(self, request, schema):  # type: ignore[no-untyped-def]
        from protocore.contracts.llm import LLMResponse

        return LLMResponse(
            message=Message(role=MessageRole.assistant, content_blocks=[]),
            stop_reason=StopReason.end_turn,
        )

    def count_tokens(self, text, model=None) -> int:  # type: ignore[no-untyped-def]
        return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# QueryEngine.latest_user_message  +  QueryEngine.run
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_latest_user_message_is_this_runs_task_not_a_seeded_one(
    engine_factory,
) -> None:
    """The tail-most user turn is THIS run's task, through the real run path.

    Pins two registry reasons at once, because they are one claim in two
    halves: ``QueryEngine.run`` appends the new task AFTER the seeded prior-run
    turns, which is the only thing that makes ``latest_user_message``'s
    backward walk land on this run's question.

    ``latest_user_message`` is not decoration. It is the query that tool
    retrieval and skill-body loading run against, so walking the transcript
    forward instead of backward hands this run a PRIOR run's task and the run
    fetches the wrong tools and loads the wrong skills — silently, and in a
    perfectly healthy-looking COMPLETED run.
    """
    llm = _PlainTextEndTurnLLM()
    engine: QueryEngine = engine_factory(rc=LoopConstants(model_context_window=4_096))
    engine.llm = llm  # type: ignore[assignment]

    # Exactly what the executor splices in: a prior run of the same session,
    # prepended verbatim, every turn carrying the seed tag.
    engine.history[0:0] = [
        _seeded(_user(_PRIOR_TASK)),
        _seeded(_assistant(TextBlock(text="The retrospective is attached."))),
    ]

    async for _event in engine.run(_user(_NEW_TASK)):
        pass

    latest = engine.latest_user_message
    assert latest is not None
    assert latest.text == _NEW_TASK, (
        "the tail-most user turn must be this run's task; a forward walk "
        "returns the seeded prior run's, and tool retrieval and skill loading "
        "then run against a question the user did not ask"
    )
    assert latest.metadata.get(SESSION_HISTORY_SEED_METADATA_KEY) is not True

    # The premise, stated separately so its failure is legible: the new task is
    # positioned after the seed rather than merged into it.
    seed_positions = [
        index
        for index, message in enumerate(engine.history)
        if message.metadata.get(SESSION_HISTORY_SEED_METADATA_KEY) is True
    ]
    task_position = next(
        index
        for index, message in enumerate(engine.history)
        if message.role is MessageRole.user and message.text == _NEW_TASK
    )
    assert seed_positions and task_position > max(seed_positions)


# ---------------------------------------------------------------------------
# query.py::_prose_gate_just_injected
# ---------------------------------------------------------------------------


def test_prose_gate_reads_the_tail_and_not_a_seeded_turn(engine_factory) -> None:
    """The prose-gate probe fires only on the LAST turn.

    Its registry reason is "inspects the LAST message only; the seed is
    prepended, so the tail always belongs to this run". Widen it past the tail
    and a prior run's repair turn — durable, rehydrated, seeded back in —
    convinces the dispatch loop that THIS run was just vetoed, and the loop
    breaks a tool batch that nothing vetoed.
    """
    engine: QueryEngine = engine_factory()
    repair_turn = Message(
        role=MessageRole.user,
        content_blocks=[TextBlock(text="answer in prose before finalising")],
        metadata={SYNTHETIC_RECOVERY_METADATA_KEY: SYNTHETIC_RECOVERY_PROSE_GATE_REPAIR},
    )

    engine.history = [_seeded(repair_turn), _user(_NEW_TASK)]
    assert _prose_gate_just_injected(engine) is False

    engine.history = [_seeded(_user(_PRIOR_TASK)), _user(_NEW_TASK), repair_turn]
    assert _prose_gate_just_injected(engine) is True

    engine.history = []
    assert _prose_gate_just_injected(engine) is False


def test_produced_output_ignores_a_seeded_prior_run(engine_factory) -> None:
    """Work a PREVIOUS run did is not work this one can be asked to report on.

    Its registry reason is that the walk starts after the last caller message,
    which the seed always precedes. Widen it to the whole transcript and a
    session whose earlier run answered makes every later run look productive —
    so a run whose first request the endpoint refused is wound down and told to
    write a closing summary, and it summarises the previous run's work as its
    own.
    """
    engine: QueryEngine = engine_factory()
    prior_answer = _assistant(TextBlock(text="The retrospective is attached."))

    engine.history = [_seeded(_user(_PRIOR_TASK)), _seeded(prior_answer), _user(_NEW_TASK)]
    assert _run_produced_output(engine) is False

    engine.history = [*engine.history, _assistant(TextBlock(text="Reading the plan now."))]
    assert _run_produced_output(engine) is True


def test_produced_output_is_output_and_not_merely_a_turn(engine_factory) -> None:
    """Prose, a tool call or a tool result count; an empty turn does not.

    The predicate gates the wind-down, and a wind-down is a request for the
    best answer the evidence supports. An assistant turn carrying nothing but
    whitespace is not evidence of anything, and a synthetic recovery turn the
    runtime itself wrote is not a caller message, so it must not be mistaken
    for one and move the boundary past the work that precedes it.
    """
    engine: QueryEngine = engine_factory()

    engine.history = [_user(_NEW_TASK), _assistant(TextBlock(text="   "))]
    assert _run_produced_output(engine) is False

    engine.history = [
        _user(_NEW_TASK),
        _assistant(ToolUseBlock(tool_call_id="call-1", name="Read", arguments_json="{}")),
    ]
    assert _run_produced_output(engine) is True

    engine.history = [
        _user(_NEW_TASK),
        _assistant(ToolUseBlock(tool_call_id="call-1", name="Read", arguments_json="{}")),
        Message(
            role=MessageRole.tool,
            content_blocks=[
                ToolResultBlock(tool_call_id="call-1", content="42 lines", is_error=False)
            ],
        ),
        Message(
            role=MessageRole.user,
            content_blocks=[TextBlock(text="wrap up now")],
            metadata={SYNTHETIC_RECOVERY_METADATA_KEY: SYNTHETIC_RECOVERY_PROSE_GATE_REPAIR},
        ),
    ]
    assert _run_produced_output(engine) is True


def test_a_background_wake_message_does_not_move_the_output_boundary(engine_factory) -> None:
    """The wake-up the runtime writes when background tasks finish is scaffolding.

    It is a user-role message, so anchoring on "the caller's last message"
    would restart the boundary at it and judge a run that already read
    evidence to have produced nothing — the exact case the wind-down protects.
    """
    engine: QueryEngine = engine_factory()
    engine.history = [
        _user(_NEW_TASK),
        _assistant(ToolUseBlock(tool_call_id="call-1", name="Read", arguments_json="{}")),
        Message(
            role=MessageRole.tool,
            content_blocks=[
                ToolResultBlock(tool_call_id="call-1", content="42 lines", is_error=False)
            ],
        ),
        Message(
            role=MessageRole.user,
            content_blocks=[TextBlock(text="background tasks finished: build ok")],
            metadata={SYNTHETIC_RECOVERY_METADATA_KEY: SYNTHETIC_RECOVERY_BACKGROUND_WAKE},
        ),
    ]
    assert _run_produced_output(engine) is True


# ---------------------------------------------------------------------------
# query.py — the lookups keyed on ONE tool_call_id
# ---------------------------------------------------------------------------


def _transcript_with_two_runs_of_tool_calls() -> list[Message]:
    """A seeded prior run that answered, then this run's unfinished work.

    This run's own tool result is placed LAST on purpose: a lookup that stopped
    keying on the call id and simply took the tail-most result would return
    this run's non-terminal one for every question asked of it, so the prior
    run's terminal assertion below is what actually holds the keying in place.
    """
    return [
        _seeded(_user(_PRIOR_TASK)),
        _seeded(
            _assistant(
                ToolUseBlock(
                    tool_call_id="prior-final",
                    name="final_answer",
                    arguments_json=json.dumps({"message": "done"}),
                )
            )
        ),
        _seeded(
            _tool_turn(
                ToolResultBlock(
                    tool_call_id="prior-final",
                    content="submitted",
                    metadata={TERMINAL_TOOL_METADATA_KEY: True},
                )
            )
        ),
        _user(_NEW_TASK),
        _assistant(
            ToolUseBlock(
                tool_call_id="cur-append",
                name="AppendFile",
                arguments_json=json.dumps({"path": "report.md", "content": "x"}),
            )
        ),
        _tool_turn(
            ToolResultBlock(tool_call_id="cur-append", content="12 bytes written")
        ),
    ]


def test_call_id_lookups_resolve_the_call_they_are_asked_for(engine_factory) -> None:
    """A tool_call_id names one call, so these lookups span the whole session.

    Three registry entries rest on this — ``_tool_name_for_call_id``,
    ``_history_has_tool_result`` and ``_history_tool_result_is_terminal``.
    The name lookup itself is the one shared ``tool_name_for_result``; the
    engine-facing wrapper only supplies the history.

    The claim is not "the seed cannot be reached"; it is that reaching it is
    harmless because the id decides the answer. Both halves are asserted: each
    lookup finds a SEEDED call when asked for it by id, and never answers with
    a different call's result.
    """
    engine: QueryEngine = engine_factory(expected_terminal_tool="final_answer")
    engine.history = _transcript_with_two_runs_of_tool_calls()

    assert _tool_name_for_call_id(engine, "cur-append") == "AppendFile"
    assert _tool_name_for_call_id(engine, "prior-final") == "final_answer"
    assert _tool_name_for_call_id(engine, "never-issued") is None

    assert _history_has_tool_result(engine, "cur-append") is True
    assert _history_has_tool_result(engine, "prior-final") is True
    assert _history_has_tool_result(engine, "never-issued") is False

    # The discriminating pair: the terminal answer belongs to the id that was
    # asked for, not to whichever result the walk happens to reach first.
    assert _history_tool_result_is_terminal(engine, "prior-final") is True
    assert _history_tool_result_is_terminal(engine, "cur-append") is False
    assert _history_tool_result_is_terminal(engine, "never-issued") is False

    # The shared lookup, keyed the same way over the same transcript: every id
    # resolves to ITS OWN tool name, so a seeded call's name can never be the
    # one written into this run's compaction placeholder.
    assert tool_name_for_result(engine.history, "cur-append") == "AppendFile"
    assert tool_name_for_result(engine.history, "prior-final") == "final_answer"
    assert tool_name_for_result(engine.history, "never-issued") is None


def test_interrupt_resolution_is_keyed_on_the_parked_call(engine_factory) -> None:
    """Resolving an interrupt reaches the call its id names, and no other.

    Two registry entries rest on this. Both walk the whole transcript, and
    both are handed a tool_call_id — so the seeded prior run is exactly what
    could break them: rebuilding a prior run's call would dispatch a tool this
    run never asked for, and rewriting a prior run's arguments would edit a
    transcript that is already settled.
    """
    engine: QueryEngine = engine_factory()
    engine.history = _transcript_with_two_runs_of_tool_calls()

    rebuilt = _tool_call_from_history(engine, "cur-append")
    assert rebuilt.name == "AppendFile"
    assert rebuilt.arguments == {"path": "report.md", "content": "x"}

    # A seeded call resolves to ITS OWN call, never to this run's.
    seeded = _tool_call_from_history(engine, "prior-final")
    assert seeded.name == "final_answer"

    with pytest.raises(InterruptResolutionError, match="not in this run's history"):
        _tool_call_from_history(engine, "never-issued")

    corrected = _apply_updated_input(engine, rebuilt, {"path": "fixed.md"})
    assert corrected.arguments == {"path": "fixed.md"}
    # The correction landed on the call it was asked for, and the seeded call
    # beside it is untouched.
    assert _tool_call_from_history(engine, "cur-append").arguments == {"path": "fixed.md"}
    assert _tool_call_from_history(engine, "prior-final").arguments == {"message": "done"}


def test_pending_tool_use_assertion_is_keyed_on_the_approved_call(
    engine_factory,
) -> None:
    """The approval check matches a PARKED call id, name and argument payload.

    Its reason is a structural claim, and the seeded prior run is the thing
    that could break it: a prior run's approved call must neither satisfy this
    run's approval nor be mistaken for a mismatch. Membership is read off the
    open set, so a batch of parked calls admits each of them by name and a
    call that was never parked is still refused.
    """
    engine: QueryEngine = engine_factory()
    arguments = {"path": "report.md", "content": "x"}
    engine.history = _transcript_with_two_runs_of_tool_calls()
    engine.mark_pending_approval("cur-append")

    _assert_history_has_matching_pending_tool_use(
        engine, ToolCall(id="cur-append", name="AppendFile", arguments=arguments)
    )

    with pytest.raises(ValueError, match="not a parked approval"):
        _assert_history_has_matching_pending_tool_use(
            engine,
            ToolCall(id="prior-final", name="final_answer", arguments={"message": "done"}),
        )

    engine.mark_pending_approval("cur-append")
    with pytest.raises(ValueError, match="does not match pending tool input"):
        _assert_history_has_matching_pending_tool_use(
            engine,
            ToolCall(id="cur-append", name="AppendFile", arguments={"path": "other.md"}),
        )


# ---------------------------------------------------------------------------
# longfile_convergence.py::_active_file_tail
# ---------------------------------------------------------------------------


def test_active_file_tail_needs_this_runs_own_binding(engine_factory) -> None:
    """The continuation anchor is gated on a path only this run can bind.

    ``_active_file_tail`` walks the raw transcript, and its registry reason
    says that is safe because the walk cannot start until
    ``_longfile_active_path`` is bound, and only this run's own byte-adding
    tool call binds it. Drop the gate and a fresh run inherits a prior run's
    half-written file as its "continue from here" anchor.

    Be precise about which half this holds. It binds the path BY HAND and then
    checks the gate's effect, so it exercises no binder and cannot see that
    there are three: two in ``longfile_convergence.py`` and
    ``QueryEngine.resume_from_snapshot``, which restores the path from a
    durable snapshot. That third one is why the clause is worth reading
    twice — and it is true, because both resume call sites are the approval
    pickup path restoring the SAME run's snapshot. So the provenance clause
    holds today, and this test is not what holds it.
    """
    def _write(call_id: str, path: str, content: str, *, seed: bool) -> Message:
        turn = _assistant(
            ToolUseBlock(
                tool_call_id=call_id,
                name="Write" if seed else "AppendFile",
                arguments_json=json.dumps({"path": path, "content": content}),
            )
        )
        return _seeded(turn) if seed else turn

    engine: QueryEngine = engine_factory()

    # Unbound: a prior run's write to the very file this run will work on must
    # not become an anchor, because this run has not claimed that file yet.
    engine.history = [
        _seeded(_user(_PRIOR_TASK)),
        _write("prior-write", "report.md", "PRIOR RUN BODY", seed=True),
        _user(_NEW_TASK),
    ]
    assert engine._longfile_active_path is None
    assert _active_file_tail(engine, 64) == ""

    # Bound, but the only write in reach is to a DIFFERENT file: an explicit
    # path match is required, never a wildcard onto whatever was written last.
    engine.history = [
        _seeded(_user(_PRIOR_TASK)),
        _write("prior-notes", "notes.md", "PRIOR RUN NOTES", seed=True),
        _user(_NEW_TASK),
    ]
    engine._longfile_active_path = "report.md"
    assert _active_file_tail(engine, 64) == ""

    # Bound, and this run has written: its own bytes are the anchor.
    engine.history.append(
        _write("cur-append", "report.md", "THIS RUN BODY", seed=False)
    )
    assert _active_file_tail(engine, 64) == "THIS RUN BODY"


# ---------------------------------------------------------------------------
# context/compaction.py::_session_history_seed_indices
# ---------------------------------------------------------------------------


def test_seed_indices_select_every_seeded_turn_and_nothing_else() -> None:
    """Compaction's seed set is exactly the tagged turns, by position.

    This is the second place in the package that derives the run boundary, and
    it is declared rather than folded into ``_this_run_messages`` because
    compaction needs INDICES into the list it is handed — which a filtered copy
    cannot express. Declared means pinned: under-select and a lossy Tier-2
    summary collapses prior-run content into an untagged turn, which the
    service's finalization filter then re-persists under this run's id;
    over-select and this run's own turns become immovable under budget
    pressure.
    """
    history = [
        _seeded(_user(_PRIOR_TASK)),
        _seeded(_assistant(TextBlock(text="prior answer"))),
        _user(_NEW_TASK),
        _assistant(TextBlock(text="this run's answer")),
    ]
    assert _session_history_seed_indices(history) == frozenset({0, 1})
    assert _session_history_seed_indices([]) == frozenset()
    assert _session_history_seed_indices(history[2:]) == frozenset()
