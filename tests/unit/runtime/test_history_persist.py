"""The session store is told what changed, and only what changed."""
from __future__ import annotations

from typing import Any

from protocore.contracts.types import Message, MessageRole, TextBlock
from protocore.runtime.history_persist import (
    HistoryDelta,
    HistoryPersister,
    persist_history,
)
from protocore.runtime.query_engine import QueryEngine


def _msg(text: str) -> Message:
    return Message(role=MessageRole.user, content_blocks=[TextBlock(text=text)])


class _RecordingStore(HistoryPersister):
    """A store that can write incrementally and remembers how it was driven."""

    def __init__(self) -> None:
        self.full_writes = 0
        self.deltas: list[HistoryDelta] = []

    def persist_session_history(self, engine: QueryEngine) -> None:
        self.full_writes += 1

    def persist_history_delta(self, engine: QueryEngine, delta: HistoryDelta) -> None:
        self.deltas.append(delta)


class _FullWriteOnlyStore(HistoryPersister):
    """A store that never learned about deltas."""

    def __init__(self) -> None:
        self.full_writes = 0

    def persist_session_history(self, engine: QueryEngine) -> None:
        self.full_writes += 1


def _attach(engine: QueryEngine, store: Any) -> None:
    engine.persist_session_history = store.persist_session_history  # type: ignore[attr-defined]
    delta = getattr(type(store), "persist_history_delta", None)
    if delta is not HistoryPersister.persist_history_delta:
        engine.persist_history_delta = store.persist_history_delta  # type: ignore[attr-defined]


def test_a_round_hands_over_only_the_messages_it_added(engine_factory: Any) -> None:
    engine = engine_factory()
    store = _RecordingStore()
    _attach(engine, store)

    engine.history.extend([_msg("one"), _msg("two")])
    persist_history(engine)
    engine.history.append(_msg("three"))
    persist_history(engine)
    engine.history.append(_msg("four"))
    persist_history(engine)

    # The first hand-over of a run is a rewrite: nobody has said what the store
    # holds. Every one after it names only what the round added.
    assert [d.rewritten for d in store.deltas] == [True, False, False]
    assert [len(d.appended) for d in store.deltas] == [0, 1, 1]
    assert [d.persisted_through for d in store.deltas] == [0, 2, 3]
    assert store.deltas[-1].appended[0].text == "four"
    assert store.full_writes == 0


def test_a_history_that_did_not_change_is_not_handed_over_at_all(
    engine_factory: Any,
) -> None:
    engine = engine_factory()
    store = _RecordingStore()
    _attach(engine, store)

    engine.history.append(_msg("one"))
    persist_history(engine)
    persist_history(engine)
    persist_history(engine)

    assert len(store.deltas) == 1


def test_a_rewritten_sequence_is_handed_over_whole(engine_factory: Any) -> None:
    engine = engine_factory()
    store = _RecordingStore()
    _attach(engine, store)

    engine.history.extend([_msg("one"), _msg("two"), _msg("three")])
    persist_history(engine)
    # What a compaction does: a span replaced by the summary that stands for it.
    engine.history[:2] = [_msg("summary of one and two")]
    persist_history(engine)

    rewrite = store.deltas[-1]
    assert rewrite.rewritten is True
    assert rewrite.appended == ()
    assert rewrite.persisted_through == 0
    assert [m.text for m in rewrite.history] == ["summary of one and two", "three"]


def test_a_shorter_history_is_a_rewrite_even_with_the_same_head(
    engine_factory: Any,
) -> None:
    engine = engine_factory()
    store = _RecordingStore()
    _attach(engine, store)

    kept = _msg("one")
    engine.history.extend([kept, _msg("two")])
    persist_history(engine)
    engine.history[:] = [kept]
    persist_history(engine)

    assert store.deltas[-1].rewritten is True


def test_a_host_that_dropped_its_rows_gets_the_history_again(
    engine_factory: Any,
) -> None:
    engine = engine_factory()
    store = _RecordingStore()
    _attach(engine, store)

    engine.history.append(_msg("one"))
    persist_history(engine)
    engine.history.append(_msg("two"))
    persist_history(engine)
    engine.forget_persisted_history()
    engine.history.append(_msg("three"))
    persist_history(engine)

    assert store.deltas[-1].rewritten is True
    assert len(store.deltas[-1].history) == 3


def test_a_store_without_the_delta_method_is_driven_as_before(
    engine_factory: Any,
) -> None:
    engine = engine_factory()
    store = _FullWriteOnlyStore()
    _attach(engine, store)

    engine.history.append(_msg("one"))
    persist_history(engine)
    engine.history.append(_msg("two"))
    persist_history(engine)

    assert store.full_writes == 2


def test_the_default_delta_method_asks_for_the_full_write(
    engine_factory: Any,
) -> None:
    engine = engine_factory()
    store = _FullWriteOnlyStore()
    engine.persist_history_delta = store.persist_history_delta  # type: ignore[attr-defined]

    engine.history.append(_msg("one"))
    persist_history(engine)

    assert store.full_writes == 1


def test_an_engine_with_no_store_attached_persists_nothing(
    engine_factory: Any,
) -> None:
    engine = engine_factory()
    engine.history.append(_msg("one"))

    persist_history(engine)

    assert engine.persisted_history_prefix is None


def test_an_empty_history_nobody_has_stored_is_not_handed_over(
    engine_factory: Any,
) -> None:
    engine = engine_factory()
    store = _RecordingStore()
    _attach(engine, store)

    persist_history(engine)

    assert store.deltas == []


def test_a_turn_boundary_keeps_what_the_store_already_holds() -> None:
    # A re-arm that forgot the marker would open every turn by rewriting the
    # whole history, which is the cost the marker exists to remove.
    assert "_persisted_history" in QueryEngine._REARM_PRESERVED_ATTRS
