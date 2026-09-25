"""Cut down the tool results the run has already finished reading.

A long result is worth its size exactly once — in the turn the model asked for
it and the turn or two after, while it is still working from what it saw. After
that it is a page of text the request carries again on every call for the rest
of the run, and ten of them are a context window. Eviction
(:mod:`protocore.runtime.result_eviction`) answers the same problem by dropping
a result whole, which is right for a read of a file that is still on disk and
wrong for everything else: a search, an HTTP body, a page fetched from a
catalogue, a tool whose output cannot be asked for again cheaply. This is the
middle answer — keep the head, keep what identifies the result, say what was
cut, say how to get the rest back.

Four properties are what make it safe to leave on:

*Age, not size alone.* Two things are never touched, however long they are: the
newest :attr:`~LoopConstants.tool_result_fresh_count` results, and every result
of the latest round — the last batch of tool calls in the view together with the
results answering it. That is the unit the model is about to read, and cutting
inside it is the bug a size-only rule has, and the reason the split projection
is not simply turned up. Everything older is eligible, including earlier rounds
of the run in flight: a run that reads twenty pages to answer one question is
exactly the run this is for, and the first of those pages is not what its answer
is being written from any more.

*Batches, not drips.* Every rewrite of the view changes the prompt prefix, and a
changed prefix is a cache miss on the whole request. Trimming one result per turn
would pay that cost every turn and save a page each time. So nothing happens at
all until the trimmable excess crosses
:attr:`~LoopConstants.tool_result_stale_trim_batch_chars`, and then a batch of
results goes at once.

*Sticky.* Once trimmed, a result stays trimmed for the rest of the run: the ids
live on the engine, not in the text. Re-deciding per build would let a result
come back whole after the batch that cut it, and the prefix would flap. The set
is part of the engine snapshot, so a run resumed in another process keeps the
prefix it had.

*Identity survives the body.* A result that carries the line telling the model
how to cite what it just read cannot lose that line to the trim: a citation the
model cannot see is a citation it invents. Every line of the cut part whose text
starts with one of
:attr:`~LoopConstants.tool_result_stale_trim_protected_prefixes` is carried over
verbatim, ahead of the pointer. The prefixes are configuration, so the rule says
nothing about which tools a deployment runs.

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
from protocore.contracts.types import ContentBlock, Message, ToolResultBlock, ToolUseBlock
from protocore.runtime.result_eviction import (
    is_compacted_placeholder,
    is_pinned_result,
    pins_invalidated_by_writes,
)


def _calls_of_the_latest_round(history: Sequence[Message]) -> frozenset[str]:
    """The call ids of the last batch of tool calls in the view.

    The round is read off the transcript rather than remembered: the last
    message carrying tool calls is the batch the model has just made, and a
    result answering one of those calls is being read right now even when the
    fresh window has already filled up with the rest of the batch. Earlier
    rounds of the same run are not protected by this — the fresh window is what
    covers them, and past it the run has moved on.
    """
    for message in reversed(history):
        ids = [
            block.tool_call_id
            for block in message.content_blocks
            if isinstance(block, ToolUseBlock)
        ]
        if ids:
            return frozenset(ids)
    return frozenset()


def _protected_prefixes(raw: str) -> tuple[str, ...]:
    """The configured line prefixes, in the order they were written."""
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _carried_lines(content: str, head: int, prefixes: Sequence[str]) -> list[str]:
    """Lines of the cut part that identify the result and must be kept.

    A line is carried when it is not wholly inside the head — a line that ends
    past the cut is a line the model would otherwise see truncated or not at
    all — and when its text starts with one of the configured prefixes.
    """
    if not prefixes:
        return []
    carried: list[str] = []
    offset = 0
    for line in content.splitlines(keepends=True):
        offset += len(line)
        text = line.strip()
        if offset > head and any(text.startswith(prefix) for prefix in prefixes):
            carried.append(text)
    return carried


def _shortened(
    content: str,
    head: int,
    prefixes: Sequence[str],
    prompts: IPromptTemplateProvider,
    fresh_count: int,
) -> str:
    """The head, the lines that identify the result, and the pointer."""
    carried = _carried_lines(content, head, prefixes)
    kept = min(head + sum(len(line) for line in carried), len(content))
    pointer = prompts.render(
        "result_stale_trim",
        {
            "kept_chars": kept,
            "dropped_chars": max(len(content) - kept, 0),
            "fresh_count": fresh_count,
        },
    )
    return "\n".join([content[:head], *carried, pointer])


def _widest_rest(
    content: str,
    prefixes: Sequence[str],
    prompts: IPromptTemplateProvider,
    fresh_count: int,
) -> int:
    """An upper bound on everything :func:`_shortened` adds after the head.

    Every line the content can carry over, plus the pointer rendered with the
    largest numbers it could ever name. Cutting a longer head only ever removes
    a carried line or takes a digit off one of the numbers, so a head sized
    against this bound cannot overflow the limit it was sized for.
    """
    carried = _carried_lines(content, 0, prefixes)
    pointer = prompts.render(
        "result_stale_trim",
        {
            "kept_chars": len(content),
            "dropped_chars": len(content),
            "fresh_count": fresh_count,
        },
    )
    return len("\n".join(["", *carried, pointer]))


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
    prefixes = _protected_prefixes(rc.tool_result_stale_trim_protected_prefixes)
    #: The split projection runs over this same view immediately after the trim
    #: (:func:`protocore.runtime.query._llm_history`) and cuts at a limit of its
    #: own, appending its own pointer over the one written here. The two are
    #: independent knobs — nothing orders them — so when both are on, a
    #: shortened result is kept under the split's limit and passes through it
    #: untouched whichever way the deployment set them.
    split_limit = rc.tool_result_content_max_chars if rc.tool_result_split_enabled else None
    pinned = set(pinned_ids)
    latest_round = _calls_of_the_latest_round(history)

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
        if block.tool_call_id in fresh or block.tool_call_id in latest_round:
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
            shortened = _shortened(existing.content, head, prefixes, prompts, fresh_count)
            if split_limit is not None and len(shortened) > split_limit:
                # Size the head against the widest the rest can become: every
                # line the content can carry over, and the pointer with its
                # largest numbers, since cutting more only adds digits to them.
                widest = _widest_rest(existing.content, prefixes, prompts, fresh_count)
                head = max(split_limit - widest, 0)
                shortened = _shortened(existing.content, head, prefixes, prompts, fresh_count)
            new_blocks.append(
                existing.model_copy(
                    update={
                        "content": shortened,
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
