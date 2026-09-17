"""What a session store is told when the working history changes.

A ReAct round adds messages to the end of a sequence it does not otherwise
touch. Telling the store "these two are new" is the whole of what it needs to
hear; handing it the other eight hundred again is a rewrite of everything the
session has ever said, once per round, and the cost of it grows with the
conversation rather than with what the conversation just did. Measured against
a store that writes a row per message, a 800-message history cost 63 ms and
1.2 MB of writes per round to record one new message.

The engine therefore remembers the prefix it last handed over and compares the
current history against it. :class:`~protocore.contracts.types.Message` is
frozen, so a message is replaced rather than edited whenever it changes: an
append leaves every earlier object exactly where it was, and a compaction, a
checkpoint or an eviction builds new ones. Identity comparison of the prefix
reports the difference without anything having to declare it, and the prefix is
held by reference, so the comparison costs a pointer per message and nothing is
kept alive that the history was not keeping alive anyway.

An engine that has never handed anything over remembers nothing rather than
remembering an empty history, and the two are not the same claim: a run picked
up on another process holds a history the store may already have most of, and
appending onto rows nobody can vouch for is how a transcript acquires a
duplicate. So the first hand-over of a run is a rewrite, and every one after it
is an append until the sequence itself changes.

Two shapes of store are supported, and a store says which it is by what it
attaches to the engine:

* ``persist_session_history(engine)`` — the full write. The only method a
  store must have, and what every store had before the delta existed.
* ``persist_history_delta(engine, delta)`` — the incremental write. A store
  that has it is handed a :class:`HistoryDelta` and writes only
  ``delta.appended``, unless ``delta.rewritten`` says the sequence itself
  changed and ``delta.history`` replaces what the store holds.

A store with only the first is called exactly as it was before. A store that
subclasses :class:`HistoryPersister` and overrides nothing inherits the same
behaviour through the default below.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from protocore.contracts.types import Message

if TYPE_CHECKING:  # pragma: no cover - typing only
    from protocore.runtime.query_engine import QueryEngine

__all__ = ["HistoryDelta", "HistoryPersister", "persist_history"]


@dataclass(frozen=True)
class HistoryDelta:
    """What changed in the working history since the store last saw it."""

    history: tuple[Message, ...]
    """The whole working history as it stands. Authoritative when
    :attr:`rewritten`; otherwise it is context, and writing it again would be
    the cost this delta exists to avoid."""

    appended: tuple[Message, ...]
    """The messages added since the last hand-over. Empty on a rewrite, because
    a rewrite is not an addition to anything."""

    persisted_through: int
    """How many messages of :attr:`history` the store was last told about. The
    store may use it as a positional check on its own tail; ``0`` on a rewrite,
    where nothing that came before is still true."""

    rewritten: bool
    """The sequence changed rather than grew — a compaction replaced a span
    with its summary, a checkpoint rebuilt it, a host evicted rows the engine
    had been told were durable. The store replaces what it holds with
    :attr:`history`."""


class HistoryPersister:
    """The store's side of the contract, for a host that would rather subclass.

    Nothing requires it: the engine looks for two attributes and does not care
    where they came from, so attaching two plain functions is as valid as
    subclassing this. What the class buys is the fallback — a store that can
    only write the history whole overrides :meth:`persist_session_history`
    alone and is driven exactly as it was before the delta existed, without
    having to know the delta exists.
    """

    def persist_session_history(self, engine: QueryEngine) -> None:
        """Write the engine's working history out in full."""
        raise NotImplementedError

    def persist_history_delta(self, engine: QueryEngine, delta: HistoryDelta) -> None:
        """Write what changed. Falls back to the full write."""
        self.persist_session_history(engine)


def persist_history(engine: QueryEngine) -> None:
    """Hand the store what changed in the working history, if anything did.

    The one call site for all three places the loop persists — the turn-start
    reset, the background-wake append and the manual checkpoint — so the
    prefix marker advances wherever the history is handed over and cannot be
    advanced by one path and not another.

    A call that would say nothing is not made at all. The engine reaches this
    function once per round whether or not the round changed anything, and a
    store woken to be told that its copy is already correct pays the full
    write to find that out.
    """
    incremental = getattr(engine, "persist_history_delta", None)
    persister = getattr(engine, "persist_session_history", None)
    if not callable(incremental) and not callable(persister):
        return
    persisted = engine.persisted_history_prefix
    history = tuple(engine.history)
    known: tuple[Message, ...] = () if persisted is None else persisted
    rewritten = (
        persisted is None
        or len(history) < len(known)
        or any(before is not now for before, now in zip(known, history, strict=False))
    )
    appended = () if rewritten else history[len(known) :]
    if not rewritten and not appended:
        return
    if persisted is None and not history:
        # Nothing has been said, and nothing was ever stored to correct.
        return
    engine.note_history_persisted(history)
    if callable(incremental):
        incremental(
            engine,
            HistoryDelta(
                history=history,
                appended=appended,
                persisted_through=0 if rewritten else len(known),
                rewritten=rewritten,
            ),
        )
    elif callable(persister):
        persister(engine)
