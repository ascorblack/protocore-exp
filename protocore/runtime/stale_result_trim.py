"""Cut down the tool results the run has already finished reading.

A long result is worth its size exactly once — in the turn the model asked for
it and the turn or two after, while it is still working from what it saw. After
that it is a page of text the request carries again on every call for the rest
of the run, and ten of them are a context window. Eviction
(:mod:`protocore.runtime.result_eviction`) answers the same problem by dropping
a result whole, which is right for a read of a file that is still on disk and
wrong for everything else: a search, an HTTP body, a tool whose output cannot be
asked for again cheaply. This is the middle answer — keep the head, say what was
cut, say how to get the rest back.

Three properties are what make it safe to leave on:

*Age, not size alone.* The newest :attr:`~LoopConstants.tool_result_fresh_count`
results are never touched, however long they are, and neither is any result
belonging to the turn in flight — every call made since the last message that
opened a turn, however many rounds that turn has run. A result the run is still
assembling an answer from is never the result this shortens, which is the bug a
size-only rule has and the reason the split projection is not simply turned up.
Only a turn the run has finished answering can lose anything.

*Batches, not drips.* Every rewrite of the view changes the prompt prefix, and a
changed prefix is a cache miss on the whole request. Trimming one result per turn
would pay that cost every turn and save a page each time. So nothing happens at
all until the trimmable excess crosses
:attr:`~LoopConstants.tool_result_stale_trim_batch_chars`, and then a batch of
results goes at once.

*Sticky.* Once trimmed, a result stays trimmed for the rest of the run: the ids
live on the engine, not in the text. Re-deciding per build would let a result
come back whole after the batch that cut it, and the prefix would flap.

Persist is never touched. Like eviction and the compaction checkpoint, this
rewrites the copy handed to the provider and leaves ``engine.history`` holding
the whole value, so the run can still be resumed, replayed or compacted against
a transcript that lost nothing.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence

from protocore.contracts.prompts import IPromptTemplateProvider
from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.tool_roles import EMPTY_TOOL_ROLE_MAP, ToolRoleMap
from protocore.contracts.types import (
    ContentBlock,
    Message,
    MessageRole,
    ToolResultBlock,
    ToolUseBlock,
)
from protocore.runtime.result_eviction import (
    is_compacted_placeholder,
    is_pinned_result,
    pins_invalidated_by_writes,
)


def _calls_of_the_turn_in_flight(history: Sequence[Message]) -> frozenset[str]:
    """The call ids of every tool call made since the turn in flight opened.

    "In flight" is read off the transcript rather than remembered. Walking back
    from the end, a user message that carries something other than tool results
    is the message that opened the current turn; every call after it belongs to
    that turn — the first of seven reads as much as the seventh. The fresh
    window alone does not cover this: a turn that runs more rounds than
    :attr:`~LoopConstants.tool_result_fresh_count` would otherwise have its own
    earlier results cut while it is still writing the answer they are for.

    A view with no such message is one long turn, and all of it is in flight.
    """
    ids: set[str] = set()
    for message in reversed(history):
        ids.update(
            block.tool_call_id
            for block in message.content_blocks
            if isinstance(block, ToolUseBlock)
        )
        if message.role is MessageRole.user and not any(
            isinstance(block, ToolResultBlock) for block in message.content_blocks
        ):
            break
    return frozenset(ids)


def _pointer(prompts: IPromptTemplateProvider, dropped: int, fresh_count: int) -> str:
    """The line that replaces what was cut: how much went and how to get it back."""
    return prompts.render(
        "result_stale_trim", {"dropped_chars": dropped, "fresh_count": fresh_count}
    )


def trim_stale_results(
    history: Sequence[Message],
    rc: LoopConstants,
    prompts: IPromptTemplateProvider,
    *,
    pinned_ids: Iterable[str] = (),
    already_trimmed: Iterable[str] = (),
    roles: ToolRoleMap = EMPTY_TOOL_ROLE_MAP,
) -> tuple[list[Message], frozenset[str]]:
    """Return the view with stale oversized results cut to their head.

    The second element is the sticky set as it stands after this build — the
    ids handed in plus whatever this build decided to cut. The caller stores it
    back on the engine; handing it in again is what keeps the prefix stable.
    """
    sticky = frozenset(str(x) for x in already_trimmed)
    if not rc.tool_result_stale_trim_enabled:
        return list(history), sticky

    limit = rc.tool_result_stale_max_chars
    #: The split projection runs over this same view immediately after the trim
    #: (:func:`protocore.runtime.query._llm_history`) and cuts at a limit of its
    #: own, appending its own pointer over the one written here. The two are
    #: independent knobs — nothing orders them — so when both are on, a
    #: shortened result is kept under the split's limit and passes through it
    #: untouched whichever way the deployment set them.
    split_limit = rc.tool_result_content_max_chars if rc.tool_result_split_enabled else None
    pinned = set(pinned_ids)
    in_flight = _calls_of_the_turn_in_flight(history)

    #: Every result the view carries, in transcript order. Compacted
    #: placeholders are not results any more — the value they stood for is
    #: already gone — so they are not counted into the fresh window either.
    results: list[ToolResultBlock] = [
        block
        for message in history
        for block in message.content_blocks
        if isinstance(block, ToolResultBlock) and not is_compacted_placeholder(block.content)
    ]
    fresh_count = max(rc.tool_result_fresh_count, 0)
    newest = results[len(results) - fresh_count :] if fresh_count else []
    fresh = frozenset(block.tool_call_id for block in newest)

    to_trim: set[str] = set()
    new_excess = 0
    #: Computed on first need and then reused. The question only arises for a
    #: result that is BOTH pinned and oversized, and the answer costs a pass
    #: over the transcript, so a run with no such result never pays for it.
    invalidated: frozenset[str] | None = None
    for block in results:
        if len(block.content) <= limit:
            continue
        # A pin is a standing request to keep this result in front of the
        # model. It outranks age: the run said this one is the exception —
        # unless the run has since rewritten the file the result describes, in
        # which case the pin is holding a page that is no longer true and
        # cutting it to its head is the kinder answer.
        if is_pinned_result(block, pinned, keep_marked=True):
            if invalidated is None:
                invalidated = pins_invalidated_by_writes(history, roles=roles)
            if is_pinned_result(
                block, pinned, keep_marked=True, invalidated_ids=invalidated
            ):
                continue
        if block.tool_call_id in sticky:
            to_trim.add(block.tool_call_id)
            continue
        if block.tool_call_id in fresh or block.tool_call_id in in_flight:
            continue
        new_excess += len(block.content) - limit
        to_trim.add(block.tool_call_id)

    if new_excess <= rc.tool_result_stale_trim_batch_chars:
        # Not enough to pay for the changed prefix. Whatever was already cut
        # stays cut; nothing new joins it this build.
        to_trim &= sticky
    #: Only ids the view still carries stay sticky. A result compaction has
    #: since replaced with a placeholder, or a checkpoint dropped entirely, is
    #: never coming back whole, so remembering it only grows the set the engine
    #: holds and every snapshot written from it.
    present = frozenset(block.tool_call_id for block in results)
    sticky &= present
    if not to_trim:
        return list(history), sticky

    rewritten: list[Message] = []
    for message in history:
        new_blocks: list[ContentBlock] = []
        changed = False
        for existing in message.content_blocks:
            if not isinstance(existing, ToolResultBlock) or existing.tool_call_id not in to_trim:
                new_blocks.append(existing)
                continue
            head = limit
            pointer = _pointer(prompts, len(existing.content) - head, fresh_count)
            if split_limit is not None and head + 1 + len(pointer) > split_limit:
                # Size the head against the longest the pointer can become:
                # cutting more only ever adds digits to the number it names.
                widest = _pointer(prompts, len(existing.content), fresh_count)
                head = max(split_limit - len(widest) - 1, 0)
                pointer = _pointer(prompts, len(existing.content) - head, fresh_count)
            new_blocks.append(
                existing.model_copy(
                    update={
                        "content": existing.content[:head] + "\n" + pointer,
                        "metadata": {**existing.metadata, "stale_trimmed": True},
                    }
                )
            )
            changed = True
        rewritten.append(
            message.model_copy(update={"content_blocks": new_blocks}) if changed else message
        )
    return rewritten, (sticky | to_trim) & present


__all__ = ["trim_stale_results"]
