"""What a reader is told about the tools, and what the store is told about the turn.

Both are contracts with the layer above the core, and both changed shape for
the same reason: the work of a round was growing with the size of the
conversation and with the size of the registry rather than with what the round
did. These drive the engine through its public entry points and assert on what
a host actually sees — the event stream, and the calls its session store
receives.
"""
from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from protocore.contracts.types import Message, MessageRole, TextBlock
from protocore.runtime.events import EventType
from protocore.runtime.history_persist import HistoryDelta, HistoryPersister
from protocore.runtime.tool_surface import forget_tool_surfaces, surface_descriptions

from .conftest import ScenarioFactory, ScriptedTool


@pytest.fixture(autouse=True)
def _forget_surfaces() -> Iterator[None]:
    # What a reader has been told is remembered for the life of the process, so
    # a scenario that asserts on a first advertisement says so explicitly.
    forget_tool_surfaces()
    yield
    forget_tool_surfaces()


def _message(text: str) -> Message:
    return Message(role=MessageRole.user, content_blocks=[TextBlock(text=text)])


def _adverts(scenario: Any) -> list[dict[str, Any]]:
    return [
        dict(evt.payload)
        for evt in scenario.events_of(EventType.TOOL_SURFACE_ADVERTISED)
    ]


async def test_a_reader_is_told_what_the_tools_do_once_and_which_tools_thereafter(
    scenario: ScenarioFactory,
) -> None:
    run = scenario(tools=[ScriptedTool(tool_name="Note", description="record a note")])
    run.llm.queue_response(text="the first answer")
    run.llm.queue_response(text="the second answer")

    await run.run("first")
    run.engine.rearm()
    await run.run("second")

    adverts = _adverts(run)
    assert len(adverts) == 2
    first, second = adverts
    assert first["tool_surface_described"] is True
    assert {entry["name"]: entry["description"] for entry in first["tools"]}[
        "Note"
    ] == "record a note"

    # The second advertisement names the same surface and does not repeat what
    # the reader was already told. Everything run-specific is still on it.
    assert second["tool_surface_digest"] == first["tool_surface_digest"]
    assert second["tool_surface_described"] is False
    assert all("description" not in entry for entry in second["tools"])
    assert [entry["name"] for entry in second["tools"]] == [
        entry["name"] for entry in first["tools"]
    ]
    assert all("roles" in entry and "sources" in entry for entry in second["tools"])

    # And a reader that has nothing for the digest can still be answered.
    assert surface_descriptions(second["tool_surface_digest"]) == {
        "Note": "record a note"
    }


async def test_another_reader_is_told_the_same_surface_for_itself(
    scenario: ScenarioFactory,
) -> None:
    """A session is the unit a host fans events out over, and each gets its own.

    A process-wide claim would assume the second session's client had been
    listening to the first session's stream.
    """
    one = scenario(
        session_id="sess-one",
        tools=[ScriptedTool(tool_name="Note", description="record a note")],
    )
    other = scenario(
        session_id="sess-two",
        run_id="run-other",
        tools=[ScriptedTool(tool_name="Note", description="record a note")],
    )

    one.llm.queue_response(text="an answer")
    other.llm.queue_response(text="an answer")

    await one.run("first")
    await other.run("first")

    assert _adverts(one)[0]["tool_surface_described"] is True
    described = _adverts(other)[0]
    assert described["tool_surface_described"] is True
    assert described["tool_surface_digest"] == _adverts(one)[0]["tool_surface_digest"]
    assert any("description" in entry for entry in described["tools"])


class _Store(HistoryPersister):
    """A session store that can write incrementally, and records how it was driven."""

    def __init__(self) -> None:
        self.rows: list[str] = []
        self.deltas: list[HistoryDelta] = []

    def persist_session_history(self, engine: Any) -> None:  # pragma: no cover
        raise AssertionError("a store with a delta method is never asked for the whole")

    def persist_history_delta(self, engine: Any, delta: HistoryDelta) -> None:
        self.deltas.append(delta)
        if delta.rewritten:
            self.rows = [message.text for message in delta.history]
        else:
            self.rows.extend(message.text for message in delta.appended)


async def test_the_store_is_told_what_the_turn_added(
    scenario: ScenarioFactory,
) -> None:
    run = scenario()
    run.llm.queue_response(text="the first answer")
    run.llm.queue_response(text="the second answer")
    store = _Store()
    run.engine.persist_history_delta = store.persist_history_delta  # type: ignore[attr-defined]

    await run.run("first")
    run.engine.rearm()
    await run.run("second")

    # The first hand-over of a run is a rewrite — nothing has said what the
    # store holds. The second names only what the turn before it added, and
    # the store's copy still agrees with the engine's history.
    assert [delta.rewritten for delta in store.deltas] == [True, False]
    assert store.deltas[1].appended
    assert len(store.deltas[1].appended) < len(store.deltas[1].history)
    assert "second" in [message.text for message in store.deltas[1].appended]
    # The store's copy is a prefix of the engine's history and agrees with it:
    # what the second turn goes on to say is handed over at the turn after it.
    assert store.rows == run.history_texts()[: len(store.rows)]
    assert "the first answer" in store.rows


async def test_a_resumed_run_does_not_append_onto_rows_it_cannot_vouch_for(
    scenario: ScenarioFactory,
) -> None:
    """A run picked up elsewhere writes the history whole before it grows it."""
    first = scenario()
    first.llm.queue_response(text="the first answer")
    await first.run("first")
    snapshot = first.engine.snapshot()

    resumed = scenario()
    resumed.llm.queue_response(text="the second answer")
    store = _Store()
    resumed.engine.persist_history_delta = store.persist_history_delta  # type: ignore[attr-defined]
    await resumed.engine.resume_from_snapshot(snapshot)
    resumed.engine.rearm()
    await resumed.run("second")

    assert store.deltas[0].rewritten is True
    assert "first" in store.rows
    assert "the first answer" in store.rows
    assert store.rows == resumed.history_texts()[: len(store.rows)]


async def test_a_reader_that_never_received_the_event_is_still_owed_the_descriptions(
    scenario: ScenarioFactory,
) -> None:
    """The claim is recorded once the event has been handed to the stream.

    A caller that stops iterating at the advertisement closes the run's
    generator there. Claiming while the payload was being built would have
    spent that reader's one description on an event the reader never got to
    act on, for the life of the process.
    """
    tools = [ScriptedTool(tool_name="Note", description="record a note")]
    abandoned = scenario(tools=tools)
    abandoned.llm.queue_response(text="the first answer")

    async for event in abandoned.engine.run(_message("first")):
        if event.type is EventType.TOOL_SURFACE_ADVERTISED:
            assert event.payload["tool_surface_described"] is True
            break
    abandoned.engine.stop()

    # The next run of the same session — the same reader — is still owed them.
    again = scenario(tools=tools, run_id="run-again")
    again.llm.queue_response(text="the second answer")
    await again.run("second")

    assert _adverts(again)[0]["tool_surface_described"] is True
