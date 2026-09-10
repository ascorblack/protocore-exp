"""Two-tier compaction state + Tier 1 / Tier 2 implementations.

Tier 1 — tool-result truncation:
 Replace tool-result content > ``tool_result_truncation_threshold`` with
 the canonical ``PROTOCOL_COMPACTED_TOOL_RESULT_V1:SNAPSHOT`` placeholder
 (renderer = :mod:`protocore.runtime.wire_format`). Full payload stored
 in :class:`IBlobStore`. Loop oldest-first; stop when enough freed.

Tier 2 — old-turn summarisation:
 For turns older than ``compaction_keep_recent_turns``, call
 :meth:`ILLMProvider.complete_structured` with :func:`build_summary_schema`
 and replace the turn with a system message containing the summary. Strip
 injection patterns before sending to the summariser. An operator turn is
 never summarised: an instruction is short, and a summary of an instruction
 is where "remove the model-name field" becomes "the user asked for changes".

Tier 3 — folding what Tier 2 leaves behind:
 Tier 2 leaves one summary per tool batch and never touches a summary again,
 and it keeps every operator turn verbatim, so a long session ends up with a
 window made almost entirely of those two kinds of message. Tier 3 replaces
 each contiguous run of them with ONE consolidated summary in which the
 operator's own words survive as exact quotes. A fold is a summary like any
 other, so a later fold absorbs it once its neighbourhood has grown again.

Cyrillic-in-JSON-escape safety preserved via
:mod:`protocore.runtime.token_counting`.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import weakref
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from protocore.constants import MAX_TOKEN_ESTIMATE_CACHE_ENTRIES
from protocore.contracts.blob import IBlobStore
from protocore.contracts.llm import (
    ILLMProvider,
    LLMObservabilityContext,
    LLMRequest,
)
from protocore.contracts.prompts import IPromptTemplateProvider
from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.types import (
    COMPACTION_REFERENCE_METADATA_KEY,
    COMPACTION_SUMMARY_METADATA_KEY,
    SESSION_HISTORY_SEED_METADATA_KEY,
    CompactionSourceRef,
    ContentBlock,
    ImageRefBlock,
    Message,
    MessageRole,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from protocore.logging_utils import get_logger
from protocore.prompts import bundled_prompt_provider
from protocore.runtime.result_eviction import tool_names_by_call_id
from protocore.runtime.token_counting import estimate_tokens
from protocore.runtime.wire_format import (
    is_compacted_placeholder,
    render_compacted_placeholder,
)

_logger = get_logger(__name__)


# Anti-injection patterns applied ONLY to content sent to the summariser, never to user-visible text.
_INJECTION_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"(?i)ignore (?:previous|prior|all|above) instructions?"),
    re.compile(r"(?i)you are now"),
    re.compile(r"(?i)(?:^|\n)system:\s*"),
    re.compile(r"---\s*END OF CONVERSATION\s*---"),
    re.compile(r"(?i)return (?:exactly|only) this json"),
    re.compile(r"(?i)disregard (?:the |all )?(?:above|previous)"),
)

_INJECTION_REPLACEMENT: Final[str] = "[REDACTED-INJECTION-PATTERN]"


def build_summary_schema(rc: LoopConstants) -> dict[str, Any]:
    """Build the summariser JSON schema with the RC-driven ``maxLength``.

    The ``summary.maxLength`` cap is sourced from
    ``rc.compaction_summary_string_max_chars`` so dashboard tuning takes
    effect without a code change (no inline magic numbers). XGrammar enforces
    this at decode time.

    One field, because a declared field is not free. Under a response schema a
    model fills every field it is given whether or not the source material
    supports one, so a field nothing reads is paid for twice: once in the output
    budget (``compaction_summary_max_output_tokens``, which this shares with the
    summary that actually survives) and again in the risk that the model invents
    plausible content to fill it. This schema carried two such arrays of up to
    twenty strings each; nothing ever extracted them.
    """
    return {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "minLength": 1,
                "maxLength": rc.compaction_summary_string_max_chars,
            },
        },
        "required": ["summary"],
        "additionalProperties": False,
    }


class CompactionExhaustedError(RuntimeError):
    """Compaction failed beyond :attr:`LoopConstants.compaction_failed_max_retries`."""


@dataclass(frozen=True, slots=True)
class Tier1Result:
    """Outcome of Tier 1 — tool-result truncation."""

    tokens_freed: int
    blob_refs_created: tuple[str, ...]
    messages_modified: int


@dataclass(frozen=True, slots=True)
class Tier2Result:
    """Outcome of Tier 2 — old-turn summarisation."""

    turns_summarised: int
    tokens_freed: int


@dataclass(frozen=True, slots=True)
class Tier3Result:
    """Outcome of Tier 3 — folding runs of old summaries and operator turns."""

    spans_folded: int
    messages_folded: int
    tokens_freed: int


@dataclass(slots=True)
class CompactionAttempt:
    """One compaction attempt's combined outcome.

    Used by :class:`QueryEngine` to emit ``compaction_started`` /
    ``compaction_completed`` event payloads with full provenance.
    """

    tier1: Tier1Result | None = None
    tier2: Tier2Result | None = None
    tier3: Tier3Result | None = None
    tokens_before: int = 0
    tokens_after: int = 0


@dataclass(slots=True)
class CompactionState:
    """Per-engine compaction state — counts retries + tracks summarised IDs."""

    retry_count: int = 0
    summarised_turn_ids: set[str] = field(default_factory=set)
    blob_refs_created: list[str] = field(default_factory=list)
    failed_anchor_keys: dict[str, int] = field(default_factory=dict)
    """How many times the summariser failed on a unit, by anchor key; past the limit the unit is left alone."""

    def reset_retries(self) -> None:
        self.retry_count = 0


def _strip_injection_patterns(text: str) -> str:
    """Redact known injection patterns before sending to the summariser."""
    if not text:
        return text
    redacted = text
    for pattern in _INJECTION_PATTERNS:
        redacted = pattern.sub(_INJECTION_REPLACEMENT, redacted)
    return redacted


def _block_text_for_estimation(block: ContentBlock) -> str:
    """Return the estimation/summariser text for one content block.

 Exhaustive per :data:`~protocore.contracts.types.ContentBlock` kind so no
 block is ever silently dropped. Mirrors the reference rough estimator:

 * :class:`TextBlock` → ``text``
 * :class:`ThinkingBlock` → ``text``
 * :class:`ToolUseBlock` → ``name`` + ``arguments_json`` (the model-generated
 tool-call payload the provider re-serializes on the wire — )
 * :class:`ToolResultBlock` → ``content``
 * :class:`ImageRefBlock` → a compact serialized marker (image token weight
 is added separately as a flat constant by :func:`estimate_message_tokens`;
 this only gives the summariser a textual breadcrumb)
 * any other / future kind → ``model_dump_json`` serialized form so it is
 never counted as 0 (catch-all).
 """
    if isinstance(block, TextBlock):
        return block.text
    if isinstance(block, ThinkingBlock):
        return block.text
    if isinstance(block, ToolUseBlock):
        return f"{block.name}{block.arguments_json}"
    if isinstance(block, ToolResultBlock):
        return block.content
    if isinstance(block, ImageRefBlock):
        return f"[image:{block.mime_type}:{block.blob_ref}]"
    # Defensive catch-all for any future ContentBlock kind — serialize so the
    # estimate is never silently zero.
    return block.model_dump_json()


def _message_text_for_estimation(message: Message, rc: LoopConstants) -> str:
    """Return concatenated text content used for token estimation + summarising.

 Exhaustive across every content block kind PLUS
 :attr:`Message.reasoning_content` (re-emitted chain-of-thought for
 thinking-capable providers — ). The ``rc`` parameter is accepted for
 signature symmetry with :func:`estimate_message_tokens` and forward
 compatibility (per-kind text shaping may become RC-tunable); it is not used
 for plain text assembly today.
 """
    _ = rc  # reserved for future RC-tunable text shaping; keeps the call sites aligned
    parts: list[str] = [_block_text_for_estimation(block) for block in message.content_blocks]
    if message.reasoning_content:
        parts.append(message.reasoning_content)
    return "\n".join(part for part in parts if part)


def _token_estimate_signature(rc: LoopConstants) -> tuple[float, ...]:
    """The RC values a per-message estimate depends on, as a cache key part.

    Chars-per-token ratios and the flat image cost are dashboard-tunable and
    can be applied to a live process, so an estimate remembered under the old
    values must not be handed back under the new ones.
    """
    return (
        rc.token_count_chars_per_token_latin,
        rc.token_count_chars_per_token_cyrillic,
        rc.token_count_chars_per_token_cyrillic_json_escape,
        rc.token_count_chars_per_token_cjk,
        rc.token_count_chars_per_token_json_struct,
        rc.token_count_image_tokens,
        rc.token_estimate_calibration,
    )


def _estimate_message_tokens_uncached(message: Message, rc: LoopConstants) -> int:
    total = 0
    for block in message.content_blocks:
        if isinstance(block, ImageRefBlock):
            total += rc.token_count_image_tokens
            continue
        total += estimate_tokens(_block_text_for_estimation(block), rc)
    if message.reasoning_content:
        total += estimate_tokens(message.reasoning_content, rc)
    # In the provider's tokens, not the heuristic's: see LoopConstants.token_estimate_calibration.
    return round(total * rc.token_estimate_calibration)


class _CachedEstimate:
    """One remembered estimate, tied to the message object that produced it."""

    __slots__ = ("message", "signature", "tokens")

    def __init__(
        self,
        message: Message,
        signature: tuple[float, ...],
        tokens: int,
    ) -> None:
        self.message: Callable[[], Message | None] = weakref.ref(message)
        self.signature = signature
        self.tokens = tokens


class TokenEstimator:
    """Per-message token estimates, remembered for as long as the message lives.

    Every budget that sizes a history — compaction, session memory, the run
    accounting around a turn — re-estimates the whole sequence from scratch,
    and a history is re-estimated several times per turn. The estimate walks
    each message character by character, so a long history costs hundreds of
    milliseconds of uninterruptible work on the event loop every time, and the
    second walk over an unchanged message produces exactly the first answer.

    Caching by object is what makes this safe, and it rests on one property of
    :class:`Message`: its content is held in immutable sequences, so a message
    that is still the same object still has the same content. Frozen alone
    would not be enough — it stops the field being rebound, not a list behind
    it being appended to — which is why the blocks are held as a tuple rather
    than merely promised not to change. A compaction that rewrites history
    produces new objects, which simply are not in the cache. The entry holds a
    weak reference and is checked against the message it was made for, so a
    recycled address cannot return someone else's number, and an estimator
    kept for a whole process never keeps a history alive. The tunable parts of
    :class:`LoopConstants` are part of the key, because they can change
    under a running process.

    An estimator is not shared between runs by the components that own one:
    identity keys make a shared instance harmless, but a private one makes the
    isolation structural rather than incidental.

    Identity keys also settle what "shared" costs. Two runs cannot read each
    other's content through a shared estimator — a key is one run's message
    object and nothing else's — so what they contend for is capacity, not
    confidentiality: in a process running several runs at once, one long
    history evicts another's entries and both pay the full walk again. That is
    why a component holding a history of its own holds an estimator of its
    own.
    """

    def __init__(
        self,
        *,
        max_entries: int = MAX_TOKEN_ESTIMATE_CACHE_ENTRIES,
    ) -> None:
        self._max_entries = max_entries
        self._entries: OrderedDict[int, _CachedEstimate] = OrderedDict()

    def estimate_message(self, message: Message, rc: LoopConstants) -> int:
        """Token weight of one message, from the cache when it is still valid."""
        return self._estimate(message, rc, _token_estimate_signature(rc))

    def estimate_history(
        self,
        history: Sequence[Message],
        rc: LoopConstants,
    ) -> int:
        """Token weight of a sequence, paying only for messages not yet seen."""
        signature = _token_estimate_signature(rc)
        return sum(self._estimate(message, rc, signature) for message in history)

    def clear(self) -> None:
        """Forget every remembered estimate."""
        self._entries.clear()

    def __len__(self) -> int:
        """How many estimates are currently remembered."""
        return len(self._entries)

    def _estimate(
        self,
        message: Message,
        rc: LoopConstants,
        signature: tuple[float, ...],
    ) -> int:
        key = id(message)
        entry = self._entries.get(key)
        if (
            entry is not None
            and entry.signature == signature
            and entry.message() is message
        ):
            self._entries.move_to_end(key)
            return entry.tokens
        tokens = _estimate_message_tokens_uncached(message, rc)
        self._entries[key] = _CachedEstimate(message, signature, tokens)
        self._entries.move_to_end(key)
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)
        return tokens


_shared_estimator: Final[TokenEstimator] = TokenEstimator()


def estimate_message_tokens(message: Message, rc: LoopConstants) -> int:
    """Estimate the token weight of a single :class:`Message` exhaustively.

 The single source of truth for the cheap pre-flight estimate, shared by
 :func:`estimate_history_tokens` and the Tier-2 freed-token accounting
 below. Every content block contributes:

 * text-bearing blocks (text / thinking / tool_use / tool_result / unknown)
 via :func:`~protocore.runtime.token_counting.estimate_tokens` on their
 extracted text (tool_use args no longer count as 0);
 * :class:`ImageRefBlock` via the flat
 :attr:`LoopConstants.token_count_image_tokens` constant — image blocks
 carry only a blob ref, so a size-derived estimate is impossible ;
 * :attr:`Message.reasoning_content` via ``estimate_tokens`` .

    Answers from a process-wide :class:`TokenEstimator`, for callers that hold
    no history of their own to keep one for. What the runs sharing it share is
    capacity and nothing else: a key is one message object, so no run can read
    another's estimate, but a bounded cache split between several concurrent
    runs evicts entries a single run would have kept. A caller that sizes one
    run's history repeatedly — the loop, a host's per-run accounting — uses
    that run's :attr:`ContextManager.token_estimator` instead and does not
    compete for these slots.
    """
    return _shared_estimator.estimate_message(message, rc)


def estimate_history_tokens(
    history: Sequence[Message],
    rc: LoopConstants,
) -> int:
    """Sum :func:`estimate_message_tokens` over ``history``.

    The cheap pre-flight counter used before :meth:`ILLMProvider.count_tokens`
    (the authoritative endpoint) and by every budget that sizes a message
    sequence — compaction, session memory, the host's run accounting — so one
    estimate is shared by all of them.
    """
    return _shared_estimator.estimate_history(history, rc)


def _content_is_already_compacted(text: str) -> bool:
    return is_compacted_placeholder(text)


def _stable_turn_key(message: Message) -> str:
    """Return a DURABLE dedup key for one turn.

    The prior key was ``str(id(message))`` — Python object identity, which
    is reborn on every ``Message.model_validate`` (snapshot/resume), so the
    persisted ``summarised_turn_ids`` set matched nothing after a resume and
    every resume re-summarised the same old turns (churn + summary-of-summary
    decay + cross-pod non-determinism).

 This key is a SHA-256 over the message role + its canonical content text +
 the per-block ``tool_call_id`` of every ``tool_use`` / ``tool_result`` block,
 so it is identical for the same logical turn across processes/pods and
 stable across snapshot round-trips. It does NOT include ``created_at`` /
 ``metadata`` (which can drift) — only the wire-relevant content the
 summariser would consume.

 Why the ``tool_call_id`` is part of the key (the "collision is
 safe" claim was WRONG for one by-construction case): two DISTINCT turns can
 carry byte-identical content text. A model that re-emits the same tool call
e.g. the documented content-missing Write-retry spiral — produces the
 SAME ``name`` + ``arguments_json`` each time (``_block_text_for_estimation``
 folds neither the id) and, after Tier-1 sheds ``reasoning_content``, even
 aged copies converge further. Without the id in the key, every later
 identical turn collides with the first one's entry in
 ``state.summarised_turn_ids`` and is silently skipped by the anchor-skip
 guard in :func:`run_tier2_summarisation` — so its whole unit (the assistant
 ``tool_use`` turn AND its tool results) is never summarised and never
 dropped, on every pass, persistently. Tier-2 then frees almost nothing
 beyond the first copy and compaction can abort the run via
 :class:`CompactionExhaustedError` where summarising the duplicates would
 have freed the space. The provider assigns a fresh ``tool_call_id`` per
 emission, and that id is a persisted, snapshot-stable, cross-pod-deterministic
 field — so folding it in disambiguates distinct spiral turns while keeping
 the SAME turn's key identical across snapshot/resume (the A4 invariant
 below). Pure-text turns carry no ``tool_call_id`` and are unaffected.

 Invariant (why ``reasoning_content`` in the key is safe despite
 Tier-1 shedding it): the digest folds in ``reasoning_content`` (line
 below), and Tier-1's reasoning-shed (:func:`run_tier1_truncation`)
 replaces an aged assistant turn with a ``reasoning_content=None`` copy —
 which would change this key. The theoretical drift (a turn summarised with
 a "with-reasoning" key in an earlier pass, later shed, then re-checked with
 a "without-reasoning" key) is NON-REACHABLE by construction: once Tier-2
 summarises a turn it REPLACES that turn in-place with a system summary
 message. The original ``reasoning_content``-bearing assistant turn no
 longer exists in history, so Tier-1 can never shed it afterwards, and on
 the next pass the replacement is caught by :func:`_is_compaction_summary`
 (early skip) BEFORE this key is ever computed for it. The exact hash order
 of Tier-1-vs-Tier-2 is therefore irrelevant for already-summarised turns.
 This is load-bearing: a future change to the Tier-2 in-place-replacement
 pattern (e.g. keeping the original turn alongside the summary) would break
 the dedup invariant silently.
 """
    digest = hashlib.sha256()
    digest.update(message.role.value.encode("utf-8"))
    digest.update(b"\x00")
    for block in message.content_blocks:
        digest.update(_block_text_for_estimation(block).encode("utf-8"))
        digest.update(b"\x00")
        # distinguish DISTINCT turns that share byte-identical content
        # (the model re-emitting the same tool call with a fresh tool_call_id).
        # The id is a persisted, snapshot-stable, cross-pod-deterministic field;
        # text-only blocks carry none and are unaffected.
        if isinstance(block, (ToolUseBlock, ToolResultBlock)):
            digest.update(block.tool_call_id.encode("utf-8"))
            digest.update(b"\x00")
    if message.reasoning_content:
        digest.update(message.reasoning_content.encode("utf-8"))
    return digest.hexdigest()


def current_tool_batch_protect_index(history: list[Message]) -> int | None:
    """Return the index of the most recent assistant ``tool_use`` turn, or None.

    The per-iteration compaction gate (:mod:`protocore.runtime.query`) fires
    AFTER all tool
    results from the current assistant turn's just-executed batch have been
    appended to ``history``. Those results are freshly produced and the model
    has NOT yet consumed them. The ``compaction_keep_recent_turns`` window
    (default 4) only protects the trailing N messages, so a parallel batch of
    more than ``keep`` tool calls leaves the 5th-from-last (and earlier) fresh
    result inside the eligible zone — Tier-1 can blob it to a placeholder and
    Tier-2 can summarise it away in the SAME iteration, before the next
    assistant stream ever sees it.

    The protection point is the lowest index of the CURRENT batch: the most
    recent assistant turn that emits a :class:`ToolUseBlock`. Everything from
    that index to the end of history (the assistant tool_use turn plus every
    tool-result message answering it, regardless of batch size) is the
    in-flight, unconsumed batch and must be exempt from compaction this
    iteration. Returns ``None`` when no assistant ``tool_use`` turn exists
    (no batch to protect — e.g. a plain text turn), in which case callers fall
    back to the unmodified ``keep_recent_turns`` window.

    Only the per-iteration gate passes this; the turn-start gate keeps the
    pre-existing keep-window-only behaviour (no in-flight batch exists when a
    fresh turn begins).
    """
    for idx in range(len(history) - 1, -1, -1):
        message = history[idx]
        if message.role is MessageRole.assistant and any(
            isinstance(block, ToolUseBlock) for block in message.content_blocks
        ):
            return idx
    return None


def _effective_eligible_upper(
    history: list[Message],
    keep: int,
    protect_tail_from_index: int | None,
) -> int:
    """Compute ``eligible_upper`` honouring keep-window + current-batch guard.

    The base is ``max(0, len(history) - keep)`` (the trailing keep-window is
    never eligible). When ``protect_tail_from_index`` is set (per-iteration
    gate), the eligible region is additionally clamped so that
    NO message at or after that index is eligible — protecting the current
    just-executed tool-result batch on top of the keep window.
    """
    eligible_upper = max(0, len(history) - keep)
    if protect_tail_from_index is not None:
        eligible_upper = min(eligible_upper, max(0, protect_tail_from_index))
    return eligible_upper


def _is_compaction_summary(message: Message) -> bool:
    """Return ``True`` if ``message`` is an already-produced Tier-2 summary.

    Recognised by the durable ``COMPACTION_SUMMARY_METADATA_KEY`` flag (set
    on every summary this module produces) OR — defensively, for summaries
    produced before the flag existed — by a content body that is a
    ``<compacted-turn ...>`` wrapper, OR — for LEGACY persisted snapshots —
    by ``role is MessageRole.system`` (the role this module used for summaries
    before the vLLM-400 fix flipped them to ``MessageRole.user``; such
    snapshots may still rehydrate into the eligible region and MUST keep being
    recognised as already-compacted). Used to skip re-summarising a summary
    (idempotency under the per-iteration gate) and so the
    unit-builder never folds a summary into a new component.
    """
    if message.metadata.get(COMPACTION_SUMMARY_METADATA_KEY) is True:
        return True
    if message.role is MessageRole.system:
        return True
    text = message.text.strip()
    return text.startswith("<compacted-turn")


def _content_preview(text: str, max_chars: int) -> str:
    """Head/tail excerpt of ``text`` capped at ``max_chars``.

    Keeps a head and tail window (most-informative ends of a tool result —
    headers + final lines) joined by an ellipsis when the content is longer
    than the cap. Returns the full text when it already fits, and ``""`` when
    the cap is 0 (preview disabled). Newlines are collapsed to spaces so the
    preview stays a single readable line inside the pipe-delimited placeholder
    (the wire renderer base64-encodes it regardless, so this is purely for
    readability once decoded).
    """
    if max_chars <= 0 or not text:
        return ""
    flat = " ".join(text.split())
    if len(flat) <= max_chars:
        return flat
    head_len = max_chars // 2
    tail_len = max_chars - head_len
    return f"{flat[:head_len]}…{flat[-tail_len:]}"


async def run_tier1_truncation(
    history: list[Message],
    blob_store: IBlobStore,
    tenant_id: str,
    rc: LoopConstants,
    truncation_threshold_tokens: int,
    *,
    keep_recent_turns: int | None = None,
    protect_tail_from_index: int | None = None,
) -> Tier1Result:
    """Replace large tool-result blocks with blobbed wire-format placeholders.

    Mutates ``history`` in place — oldest first. Stops when nothing further
    can be freed (i.e. all remaining tool_results are below threshold or
    already compacted).

    Additional compaction passes (both RC-gated, default-on):

    * ``compaction_shed_reasoning_enabled`` — strip ``reasoning_content`` from
      assistant turns older than ``keep_recent_turns``. Re-emitted CoT is
      single-turn scaffolding the model never needs from prior turns and is
      otherwise uncompactable bloat on a small window.
    * ``compaction_bound_reference_blocks_enabled`` — compact an OVER-BUDGET
      frozen reference block (a non-tool single-text message tagged
      ``COMPACTION_REFERENCE_METADATA_KEY``, e.g. the executor's
      ``<environment_context>``/``<memory-context>`` bootstrap) to a blobbed
      placeholder, the same way large tool results are shed. The original task
      user turn is never tagged, so it is never shed here.

    The tool-result placeholder is enriched with the originating tool name +
    a short head/tail preview (RC-bounded by
    ``compaction_placeholder_preview_chars``) so a compacted result is
    recoverable rather than an information dead-end.

    Args:
        history: full message list (mutated in place).
        blob_store: target durable store for the original content.
        tenant_id: scoping for blob refs.
        rc: token-counting + RC fields.
        truncation_threshold_tokens: tool_result tokens above this get blobbed.
        keep_recent_turns: trailing turns to skip (anchor caching). ``None``
            defaults to :attr:`LoopConstants.compaction_keep_recent_turns`.
        protect_tail_from_index: When set, NO message at or after this index is
            eligible — protects the current iteration's just-executed
            tool-result batch (any batch size) on top of ``keep_recent_turns``.
            ``None`` keeps the keep-window-only behaviour (turn-start gate).
            See :func:`current_tool_batch_protect_index`.

    Returns:
        :class:`Tier1Result` with stats for telemetry.
    """
    if not history:
        return Tier1Result(tokens_freed=0, blob_refs_created=(), messages_modified=0)

    keep = rc.compaction_keep_recent_turns if keep_recent_turns is None else keep_recent_turns
    eligible_upper = _effective_eligible_upper(history, keep, protect_tail_from_index)

    preview_cap = rc.compaction_placeholder_preview_chars

    # Hoisted out of the eviction loop. Naming the tool behind a shed result
    # is a per-block question, but answering it per block walks the whole
    # transcript per block — quadratic exactly on the large histories
    # compaction exists for. The pass below only rewrites tool RESULT blocks
    # and reasoning_content, so no tool_use block moves under this map while
    # the loop runs.
    tool_names = tool_names_by_call_id(history)

    tokens_freed = 0
    refs_created: list[str] = []
    modified = 0

    for idx in range(eligible_upper):
        message = history[idx]

        # ── A2(1) — shed re-emitted reasoning_content on aged assistant turns.
        # Free the tokens BEFORE the tool-result branch's continue so a
        # tool-pairing assistant turn with bloated CoT is still trimmed even
        # when its tool result is the thing being blobbed.
        if (
            rc.compaction_shed_reasoning_enabled
            and message.role is MessageRole.assistant
            and message.reasoning_content
        ):
            freed = estimate_tokens(message.reasoning_content, rc)
            if freed > 0:
                history[idx] = message.model_copy(update={"reasoning_content": None})
                message = history[idx]
                tokens_freed += freed
                modified += 1

        if not message.content_blocks:
            continue

        block = message.content_blocks[0]

        # ── A2(2) — bound an over-budget FROZEN reference block (bootstrap).
        if (
            rc.compaction_bound_reference_blocks_enabled
            and message.role is not MessageRole.tool
            and message.metadata.get(COMPACTION_REFERENCE_METADATA_KEY) is True
            and isinstance(block, TextBlock)
            and not _content_is_already_compacted(block.text)
        ):
            ref_tokens = estimate_tokens(block.text, rc)
            if ref_tokens >= truncation_threshold_tokens:
                ref_bytes = block.text.encode("utf-8")
                ref_sha = hashlib.sha256(ref_bytes).hexdigest()
                ref_blob = await blob_store.put(
                    tenant_id=tenant_id,
                    content=ref_bytes,
                    content_type="text/plain; charset=utf-8",
                    metadata={"label": "reference_block", "tier": "tier1"},
                )
                ref_placeholder = render_compacted_placeholder(
                    CompactionSourceRef(
                        blob_ref=ref_blob.ref,
                        sha256=ref_sha,
                        original_tokens=ref_tokens,
                        label="reference_block",
                        tool_name="",
                        preview=_content_preview(block.text, preview_cap),
                    ),
                    "SNAPSHOT",
                )
                history[idx] = message.model_copy(
                    update={
                        "content_blocks": [TextBlock(text=ref_placeholder)],
                        "metadata": {**message.metadata, "compacted": True, "blob_ref": ref_blob.ref},
                    }
                )
                modified += 1
                refs_created.append(ref_blob.ref)
                tokens_freed += ref_tokens - estimate_tokens(ref_placeholder, rc)
            continue

        if message.role is not MessageRole.tool:
            continue
        # a tool-role message may carry MORE THAN ONE ToolResultBlock
        # — the ``Message`` validator caps only system/user at one block
        # (types.py comment that claims tool is capped is wrong), and
        # ``_build_summarisation_units`` explicitly models "a single tool-role
        # message may answer more than one assistant tool_use turn". Iterate
        # EVERY block, blob each over-threshold ToolResultBlock, and only
        # replace the message when at least one block was shed — the prior
        # code took ``block = message.content_blocks[0]`` and rebuilt with
        # ``content_blocks=[new_block]``, which (a) left blocks [1:] as
        # permanent uncompactable bloat and (b) DROPPED non-ToolResultBlock
        # siblings, corrupting the message.
        new_blocks: list[ContentBlock] = []
        msg_modified = False
        for block in message.content_blocks:
            if not isinstance(block, ToolResultBlock):
                new_blocks.append(block)
                continue
            if _content_is_already_compacted(block.content):
                new_blocks.append(block)
                continue
            original_tokens = estimate_tokens(block.content, rc)
            if original_tokens < truncation_threshold_tokens:
                new_blocks.append(block)
                continue

            # What gets stored is the CANONICAL value, not the text in front
            # of the model. A tool that handed back a short view of a long
            # result left the long result on the block; blobbing the view
            # instead would put a truncated copy behind a reference the
            # placeholder calls canonical.
            canonical_text = block.canonical_content or block.content
            # The canonical value is stored ONCE. A block that already names
            # where its value lives — a tool that stored its own output, a
            # result an earlier pass already shed — is not stored again: the
            # reference it carries is the canonical value's address, and
            # writing a second copy under a second address would leave two
            # answers to "what did this call return" with nothing to say which
            # is the value and which the projection.
            existing_ref = block.canonical_ref
            if existing_ref is not None:
                canonical_ref = existing_ref
                # The bytes behind that reference were written elsewhere and
                # this pass has not seen them. A digest of what is on the block
                # would describe a different string, so the placeholder says
                # nothing about the digest rather than something false.
                sha256 = ""
            else:
                content_bytes = canonical_text.encode("utf-8")
                sha256 = hashlib.sha256(content_bytes).hexdigest()
                blob_md = await blob_store.put(
                    tenant_id=tenant_id,
                    content=content_bytes,
                    content_type="text/plain; charset=utf-8",
                    metadata={
                        "tool_call_id": block.tool_call_id,
                        "label": "tool_result",
                        "tier": "tier1",
                    },
                )
                canonical_ref = blob_md.ref
                refs_created.append(canonical_ref)

            placeholder = render_compacted_placeholder(
                CompactionSourceRef(
                    blob_ref=canonical_ref,
                    sha256=sha256,
                    original_tokens=original_tokens,
                    label="tool_result",
                    tool_name=tool_names.get(block.tool_call_id, ""),
                    preview=_content_preview(block.content, preview_cap),
                ),
                "SNAPSHOT",
            )

            # What is shed is the PROJECTION. ``canonical_ref`` survives on
            # the block, and so does ``path``: a result whose text is now a
            # placeholder still describes the file it described, and a later
            # write must still be able to say it is out of date.
            new_blocks.append(
                ToolResultBlock(
                    tool_call_id=block.tool_call_id,
                    content=placeholder,
                    is_error=block.is_error,
                    metadata={**block.metadata, "compacted": True, "blob_ref": canonical_ref},
                    canonical_ref=canonical_ref,
                    path=block.path,
                )
            )
            msg_modified = True
            tokens_freed += original_tokens - estimate_tokens(placeholder, rc)

        if msg_modified:
            history[idx] = message.model_copy(update={"content_blocks": new_blocks})
            modified += 1

    return Tier1Result(
        tokens_freed=max(0, tokens_freed),
        blob_refs_created=tuple(refs_created),
        messages_modified=modified,
    )


def _tool_use_ids(message: Message) -> tuple[str, ...]:
    """Return the tool_call_ids of every :class:`ToolUseBlock` on ``message``."""
    return tuple(
        block.tool_call_id
        for block in message.content_blocks
        if isinstance(block, ToolUseBlock)
    )


def _tool_result_ids(message: Message) -> tuple[str, ...]:
    """Return the tool_call_ids of every :class:`ToolResultBlock` on ``message``."""
    return tuple(
        block.tool_call_id
        for block in message.content_blocks
        if isinstance(block, ToolResultBlock)
    )


@dataclass(slots=True)
class _SummarisationUnit:
    """One atomic compaction unit .

    ``indices`` are the history positions that MUST be summarised/skipped
    together — a tool-pairing connected component: every assistant ``tool_use``
    turn and every tool-role ``tool_result`` message that share a
    ``tool_call_id`` (transitively, e.g. parallel calls answered by a shared
    tool message) belong to the same component. ``anchor_idx`` is the lowest
    assistant index in the component; it becomes the system summary and ALL
    other members (other assistant turns + every matching tool-result message)
    are removed, so no side of any pair is ever orphaned. A plain turn (no tool
    blocks) is a singleton component.
    """

    anchor_idx: int
    indices: tuple[int, ...]


def _first_user_turn_index(history: list[Message]) -> int | None:
    """Return the index of the FIRST user-role turn (the original task), or None.

    This turn carries the verbatim task + constraints and must survive every
    compaction once the per-iteration gate makes compaction fire often. Skips
    runtime-prepended reference blocks (``COMPACTION_REFERENCE_METADATA_KEY``)
    — those are bootstrap context, not the user's task, and Tier-1 may shed
    them.

    Also skips PRIOR-RUN user turns the executor seeded into history
    (``SESSION_HISTORY_SEED_METADATA_KEY``). Those precede the new task in
    history, so without this skip the "first user turn" would be a seeded
    prior-run turn and the protect-first-user-turn guard would shield the wrong
    message, leaving the NEW task summarisable. With the skip, the guard
    correctly protects the new task — the first user turn that is neither a
    reference block nor a seeded prior-run turn.
    """
    for idx, message in enumerate(history):
        if (
            message.role is MessageRole.user
            and message.metadata.get(COMPACTION_REFERENCE_METADATA_KEY) is not True
            and message.metadata.get(SESSION_HISTORY_SEED_METADATA_KEY) is not True
            # vLLM-400 fix: Tier-2 summaries are now USER-role. A summary is
            # never the user's verbatim task, so it must not be picked as the
            # protect-first-user-turn target.
            and not _is_compaction_summary(message)
        ):
            return idx
    return None


def _session_history_seed_indices(history: list[Message]) -> frozenset[int]:
    """Return the history indices of executor-seeded prior-run turns.

    These are protected from Tier-2 summarisation so a lossy summary never
    silently collapses seeded prior-run content into an UNtagged
    ``<compacted-turn>`` system message — which would (a) defeat the host
    finalization filter that excludes seed-tagged turns from re-persistence and
    (b) re-write prior-run conversation under the new ``run_id``.

    Tier-1 still bounds these turns under budget pressure, by TWO mechanisms
    (both preserve the seed tag because ``model_copy`` keeps ``metadata``):

    * seeded ``role=tool`` results are blobbed by the Tier-1 main shed path
      (large seeded tool results → SNAPSHOT placeholders);
    * the synthetic running-summary + artifact-ledger seed blocks are
      ``role=user`` ``TextBlock`` messages that DUAL-TAG
      :data:`SESSION_HISTORY_SEED_METADATA_KEY` AND
      :data:`COMPACTION_REFERENCE_METADATA_KEY` (see
      ``session_memory._tag_seeded_reference``), so the Tier-1 A2(2)
      reference-block path blobs THEM to a recoverable placeholder when they
      exceed the truncation threshold. Without the reference dual-tag these
      large user-text blocks had NO shed path and were permanently immovable.

    Only the lossy Tier-2 collapse is withheld from ALL seed-tagged turns; the
    reference dual-tag does not weaken that (Tier-2 still skips them by seed tag).
    """
    return frozenset(
        idx
        for idx, message in enumerate(history)
        if message.metadata.get(SESSION_HISTORY_SEED_METADATA_KEY) is True
    )


def _compaction_reference_indices(history: list[Message]) -> frozenset[int]:
    """Return the history indices of frozen reference blocks (bootstrap context).

    Reference blocks (``COMPACTION_REFERENCE_METADATA_KEY``, e.g. the
    executor's ``<environment_context>``/``<memory-context>`` bootstrap) are a
    Tier-1-ONLY shed surface: the A2(2) path blobs an over-budget block to a
    RECOVERABLE ``PROTOCOL_COMPACTED…SNAPSHOT`` placeholder (the blob ref stays
    in history). They are protected from the lossy Tier-2 collapse in BOTH
    states:

    * un-blobbed (small) — a Tier-2 summary would lossily collapse frozen
      bootstrap context that the Tier-1-only shed design deliberately keeps
      verbatim until it is over budget;
    * blobbed — the placeholder message is the ONLY in-history pointer to the
      blob. Summarising it sends the raw placeholder marker to the summariser
      (one wasted LLM call) and replaces the message, erasing the blob ref and
      breaking the A2(2) recoverable-snapshot contract.

    The tag survives the A2(2) blob-shed (`run_tier1_truncation` spreads the
    existing ``message.metadata`` into the placeholder copy), so this single
    tag check covers the placeholder state too.
    """
    return frozenset(
        idx
        for idx, message in enumerate(history)
        if message.metadata.get(COMPACTION_REFERENCE_METADATA_KEY) is True
    )


def _build_summarisation_units(
    history: list[Message],
    eligible_upper: int,
    *,
    protected_indices: frozenset[int] = frozenset(),
) -> list[_SummarisationUnit]:
    """Partition the eligible region into atomic tool-pairing units .

    ``protected_indices`` are history positions that must never be summarised
    — any component containing one is skipped wholesale (so the original task
    user turn stays verbatim and no tool-pairing partner is half-dropped).

    Pairing is computed as CONNECTED COMPONENTS over the message graph so the
    rewrite can never half-drop a pair: a tool_call_id may be
    answered by more than one tool-role message, and a single tool-role message
    may answer more than one assistant ``tool_use`` turn (parallel calls). Two
    message indices are in the same component when they share any
    ``tool_call_id`` (assistant ``tool_use`` ↔ tool-role ``tool_result``).

    Rules:

    * A component is summarised/skipped atomically. It is eligible ONLY if EVERY
      member index is inside the eligible region (``< eligible_upper``); if any
      member is anchored in the kept-recent tail, the whole component is skipped
      — never half-summarised.
    * The anchor (summary target) is the LOWEST assistant index in the
      component; every other member (other assistant turns + ALL matching
      tool-result messages) is dropped. A component with NO assistant member
      (orphan tool results whose originator is not in history, e.g. already
      compacted away) is left untouched — we never synthesise or strip a bare
      result here.
    * Already-compacted summaries (``_is_compaction_summary``) are skipped and
      never join a component. This matches both new USER-role summaries (the
      vLLM-400 fix: a mid-history ``MessageRole.system`` 400s vLLM) and LEGACY
      ``MessageRole.system`` summaries rehydrated from persisted snapshots.
    """
    # Map every tool_call_id -> the tool-role result message indices answering
    # it (a list, NOT first-writer-wins: duplicates must all be grouped).
    result_indices_by_call: dict[str, list[int]] = {}
    # Map every tool_call_id -> the assistant message indices that emit it.
    tool_use_indices_by_call: dict[str, list[int]] = {}
    for idx in range(len(history)):
        msg = history[idx]
        if _is_compaction_summary(msg):
            continue
        if msg.role is MessageRole.tool:
            for call_id in _tool_result_ids(msg):
                result_indices_by_call.setdefault(call_id, []).append(idx)
        elif msg.role is MessageRole.assistant:
            for call_id in _tool_use_ids(msg):
                tool_use_indices_by_call.setdefault(call_id, []).append(idx)

    # Union-find over message indices linked by a shared tool_call_id.
    parent: dict[int, int] = {}

    def _find(i: int) -> int:
        parent.setdefault(i, i)
        root = i
        while parent[root] != root:
            root = parent[root]
        while parent[i] != root:
            parent[i], i = root, parent[i]
        return root

    def _union(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    # Seed every non-summary message index as its own node.
    for idx in range(len(history)):
        if not _is_compaction_summary(history[idx]):
            _find(idx)

    # Link assistant tool_use turns with the tool-role results that answer them.
    for call_id, use_indices in tool_use_indices_by_call.items():
        members = list(use_indices) + result_indices_by_call.get(call_id, [])
        for other in members[1:]:
            _union(members[0], other)
    # A duplicated result with no in-history originator still links its own
    # result messages together so they are treated as one (orphan) component.
    for call_id, res_indices in result_indices_by_call.items():
        if call_id in tool_use_indices_by_call:
            continue
        for other in res_indices[1:]:
            _union(res_indices[0], other)

    components: dict[int, list[int]] = {}
    for idx in range(len(history)):
        if _is_compaction_summary(history[idx]):
            continue
        components.setdefault(_find(idx), []).append(idx)

    units: list[_SummarisationUnit] = []
    for member_indices in components.values():
        member_indices.sort()
        # Atomicity: the whole component must live inside the eligible region.
        if any(member >= eligible_upper for member in member_indices):
            continue
        # A component touching a protected index (the original task user turn)
        # is skipped wholesale so the verbatim task survives and no
        # tool-pairing partner is half-dropped.
        if protected_indices and any(member in protected_indices for member in member_indices):
            continue
        assistant_members = [
            i for i in member_indices if history[i].role is MessageRole.assistant
        ]
        if not assistant_members:
            # Orphan tool result(s) / pure-non-assistant component. A standalone
            # plain user turn is its own component and IS summarisable; a bare
            # tool result with no originator must be left intact.
            non_tool_members = [
                i for i in member_indices if history[i].role is not MessageRole.tool
            ]
            if not non_tool_members:
                continue
            anchor_idx = non_tool_members[0]
        else:
            anchor_idx = assistant_members[0]
        units.append(
            _SummarisationUnit(anchor_idx=anchor_idx, indices=tuple(member_indices))
        )

    # Deterministic order: summarise components by their anchor position.
    units.sort(key=lambda u: u.anchor_idx)
    return units


def _wrap_compaction_summary(anchor_key: str, summary_text: str) -> str:
    """Build the ``<compacted-turn>`` replacement body for a summarised unit.

    Single source of truth for the wrapper so the no-net-gain floor
    (:func:`_compaction_wrapper_floor_tokens`) and the actual replacement stay
    byte-for-byte in sync.
    """
    return f"<compacted-turn id='{anchor_key}'>{summary_text}</compacted-turn>"


def _compaction_wrapper_floor_tokens(anchor_key: str, rc: LoopConstants) -> int:
    """Estimated token weight of an EMPTY ``<compacted-turn>`` wrapper.

    A unit can only shrink under Tier-2 if its current token estimate is
    strictly above this floor — the minimum size the replacement can ever be,
    reached when the summariser returns an empty body. A unit at or below the
    floor cannot be made smaller by summarising it; replacing it would GROW
    history (measured: a 1-token turn becomes a ~39-token wrapper) while a
    ``max(0, ...)`` clamp would hide that growth from ``tokens_freed``. So such
    units are skipped before any LLM call — this both prevents the inflation
    (which can push ``run_compaction``'s ``tokens_after`` above
    ``tokens_before`` and count a no-progress retry toward
    :class:`CompactionExhaustedError`) and avoids spending a ~5-11s summariser
    call that cannot help.
    """
    return estimate_tokens(_wrap_compaction_summary(anchor_key, ""), rc)


#: Called with every request the summariser is about to make, before it is
#: made. The turn's own provider calls are recorded this way; a compaction
#: rewrites the transcript every later request is built from, so leaving its
#: calls unrecorded makes a recording unreplayable from the first compaction on.
RequestRecorder = Callable[[LLMRequest], Awaitable[Any]]


@dataclass(frozen=True, slots=True)
class _SummaryOutcome:
    """What one summariser call produced.

    ``replacement`` is ``None`` for every way a call can fail to earn its
    keep — the provider raised, the reply carried no usable summary, or the
    summary came back no smaller than what it would replace. The caller commits
    nothing in that case and the original messages stay as they are.
    """

    anchor_key: str
    replacement: Message | None
    tokens_freed: int
    failed: bool = False
    """The call itself went wrong (raised, cut, or no summary in the reply), as opposed to a summary no smaller."""


def _is_plain_operator_turn(message: Message) -> bool:
    """Is this a turn the operator wrote?

    A user-role message that is not a summary, not a frozen reference block,
    not a seeded turn from an earlier run of the session, and carries no tool
    result. What is left is what a person typed: the task, a steer, a
    correction. Compaction treats those as the one kind of message it may not
    paraphrase, because an instruction is short enough that a summary of it
    frees nothing and specific enough that a paraphrase changes it.
    """
    if message.role is not MessageRole.user or _is_compaction_summary(message):
        return False
    if message.metadata.get(COMPACTION_REFERENCE_METADATA_KEY) is True:
        return False
    if message.metadata.get(SESSION_HISTORY_SEED_METADATA_KEY) is True:
        return False
    return not any(isinstance(block, ToolResultBlock) for block in message.content_blocks)


def _operator_turn_indices(history: list[Message]) -> tuple[int, ...]:
    """Positions of every operator turn, oldest first."""
    return tuple(idx for idx, message in enumerate(history) if _is_plain_operator_turn(message))


def _summary_from_response(raw: str, unit_label: str) -> str:
    """The ``summary`` string out of a structured summariser reply.

    ``complete_structured`` is invoked with :func:`build_summary_schema`
    (json_object), and the openai-compat provider's
    ``_structured_response_from_body`` returns the model's RAW content without
    parsing it, so ``response.message.text`` normally carries the whole
    ``{"summary": "..."}`` envelope. Detect that by the leading ``{`` (the
    schema is a top-level object), parse it strictly, and read the ``summary``
    string. Any other key is ignored rather than trusted: ``additionalProperties``
    is false, but a provider that does not enforce the grammar can still return
    one, and an unread key must never reach the history the summary replaces. A
    reply that is NOT a JSON object (a provider that pre-parses, or the
    in-memory double) is taken verbatim.

    An unusable reply returns ``""`` and is logged WITH ITS HEAD: "the
    summariser returned nothing" is not diagnosable, and the first two hundred
    characters are what tells a truncated envelope from a refusal.
    """
    if not raw:
        return ""
    if not raw.lstrip().startswith("{"):
        return raw
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        _logger.warning(
            "summariser reply for %s looked like JSON but failed to parse (err=%s): %r",
            unit_label,
            exc,
            raw[:200],
        )
        return ""
    if not isinstance(parsed, dict) or not isinstance(parsed.get("summary"), str):
        _logger.warning(
            "summariser reply for %s carries no 'summary' string: %r", unit_label, raw[:200]
        )
        return ""
    return str(parsed["summary"])


def _summary_word_budget(before_tokens: int, rc: LoopConstants) -> int:
    """Words the summary of a unit this size may spend.

    A fixed "in 1-3 sentences" asks for the same output whatever it is given,
    which is how a small unit comes back larger than the original and a large
    one comes back too thin to have kept anything. The budget scales with what
    is being replaced instead, with a floor so a short unit still has room for
    the identifiers that must survive verbatim.
    """
    scaled = before_tokens // rc.compaction_summary_tokens_per_word
    # The prompt must not ask for more than the reply may hold: a budget above the
    # output cap is a summary that is always truncated, never parsed and never
    # committed, so the largest units are exactly the ones that never shrink.
    # The per-word cost on the way out is a property of the script and the JSON
    # around the words, not of English: see compaction_summary_output_tokens_per_word.
    ceiling = rc.compaction_summary_max_output_tokens // rc.compaction_summary_output_tokens_per_word
    return max(rc.compaction_summary_min_words, min(scaled, ceiling))


async def _run_summariser(
    prompt: str,
    *,
    anchor_key: str,
    unit_label: str,
    before_tokens: int,
    max_output_tokens: int,
    compaction_llm: ILLMProvider,
    rc: LoopConstants,
    model_name: str,
    observability: LLMObservabilityContext | None,
    record_request: RequestRecorder | None,
) -> _SummaryOutcome:
    """One summariser exchange, from a built prompt to a committable replacement.

    Shared by the per-turn pass and the fold, because the two differ only in
    what they put in the prompt: the request is assembled by the same builder
    every other provider call goes through, recorded the same way, and held to
    the same net-gain rule — a summary at or above the size of what it replaces
    is discarded rather than committed, since committing it would GROW history
    while the freed-token clamp hid the growth.
    """
    # Local import — the shared request builder lives beside the action
    # stream, which imports this module, so the dependency is taken at call
    # time rather than at module import.
    from protocore.runtime.query import build_llm_request

    request = build_llm_request(
        model=model_name,
        messages=[Message(role=MessageRole.user, content_blocks=[TextBlock(text=prompt)])],
        tools=[],
        max_tokens=max_output_tokens,
        temperature=rc.compaction_summary_temperature,
        observability=observability,
    )
    if record_request is not None:
        await record_request(request)
    try:
        response = await compaction_llm.complete_structured(request, build_summary_schema(rc))
    except Exception as exc:
        _logger.warning("summariser failed for %s; skipping (err=%s)", unit_label, exc)
        return _SummaryOutcome(anchor_key=anchor_key, replacement=None, tokens_freed=0, failed=True)
    summary_text = _summary_from_response(response.message.text, unit_label)
    if not summary_text:
        return _SummaryOutcome(anchor_key=anchor_key, replacement=None, tokens_freed=0, failed=True)
    wrapped = _wrap_compaction_summary(anchor_key, summary_text)
    after_tokens = estimate_tokens(wrapped, rc)
    if after_tokens >= before_tokens:
        _logger.warning(
            "summary for %s is not smaller than the original (%s >= %s tokens); kept the original",
            unit_label,
            after_tokens,
            before_tokens,
        )
        return _SummaryOutcome(anchor_key=anchor_key, replacement=None, tokens_freed=0)
    # vLLM-400 fix: the summary replaces an aged turn IN THE MIDDLE of
    # history. vLLM rejects any ``system`` message past index 0 ("System
    # message must be at the beginning."), so the summary turn is USER-role.
    # It stays recognisable as a summary via the durable
    # ``COMPACTION_SUMMARY_METADATA_KEY`` flag + the ``<compacted-turn>``
    # wrapper (``_is_compaction_summary``); legacy persisted system-role
    # summaries remain recognised too. (The request-assembly boundary in
    # ``query._normalize_outbound_system_messages`` is the defense-in-depth
    # backstop for those legacy snapshots.)
    return _SummaryOutcome(
        anchor_key=anchor_key,
        replacement=Message(
            role=MessageRole.user,
            content_blocks=[TextBlock(text=wrapped)],
            metadata={COMPACTION_SUMMARY_METADATA_KEY: True},
        ),
        tokens_freed=before_tokens - after_tokens,
    )


async def _summarise_unit(
    unit_messages: list[Message],
    *,
    anchor_key: str,
    unit_label: str,
    before_tokens: int,
    compaction_llm: ILLMProvider,
    rc: LoopConstants,
    prompts: IPromptTemplateProvider,
    model_name: str,
    observability: LLMObservabilityContext | None,
    record_request: RequestRecorder | None,
) -> _SummaryOutcome:
    """Summarise ONE atomic unit — an assistant turn and the results answering it."""
    raw_text = "\n".join(_message_text_for_estimation(member, rc) for member in unit_messages).strip()
    if not raw_text:
        return _SummaryOutcome(anchor_key=anchor_key, replacement=None, tokens_freed=0)
    prompt = prompts.render(
        "compaction_turn_summary",
        {
            "turn": _strip_injection_patterns(raw_text),
            "max_words": (budget := _summary_word_budget(before_tokens, rc)),
            "max_chars": budget * 6,
        },
    )
    return await _run_summariser(
        prompt,
        anchor_key=anchor_key,
        unit_label=unit_label,
        before_tokens=before_tokens,
        max_output_tokens=rc.compaction_summary_max_output_tokens,
        compaction_llm=compaction_llm,
        rc=rc,
        model_name=model_name,
        observability=observability,
        record_request=record_request,
    )


async def run_tier2_summarisation(
    history: list[Message],
    compaction_llm: ILLMProvider,
    state: CompactionState,
    rc: LoopConstants,
    *,
    model_name: str,
    observability: LLMObservabilityContext | None = None,
    protect_tail_from_index: int | None = None,
    free_target_tokens: int | None = None,
    record_request: RequestRecorder | None = None,
    prompts: IPromptTemplateProvider | None = None,
) -> Tier2Result:
    """Summarise old turns via the compaction LLM.

 For each ATOMIC unit older than ``rc.compaction_keep_recent_turns`` that has
 not yet been summarised, call
 :meth:`ILLMProvider.complete_structured` with the summary schema and replace
 the unit's anchor turn in-place with a system message wrapping the summary.

 tool pairing is atomic: an assistant ``tool_use`` turn and the
 tool-role ``tool_result`` message(s) that answer it are summarised (the
 assistant turn becomes the summary, the result messages are removed) or
 skipped together. The function never replaces a tool-role result while
 leaving its originating ``ToolUseBlock`` (or vice versa), so the rewritten
 history always satisfies tool_use ↔ tool_result pairing and the next
 provider call cannot 400 on an orphan.

 Anti-injection: every turn body is stripped of
 :data:`_INJECTION_PATTERNS` before being included in the prompt.

 The dedup key is a DURABLE content hash (:func:`_stable_turn_key`), not
 ``str(id(obj))``, so a turn already summarised before a snapshot/resume is
 recognised after rehydration and is NOT re-summarised (no churn, no
 summary-of-summary decay, deterministic across pods). The produced summary
 system message is tagged with :data:`COMPACTION_SUMMARY_METADATA_KEY` and
 an already-summary anchor is
 skipped, so the per-iteration gate (A1) is idempotent.

 The original task user turn is protected from summarisation when
 ``rc.compaction_protect_first_user_turn`` is set, and EVERY operator turn is
 protected unconditionally — see :func:`_is_plain_operator_turn`. Those are
 what :func:`run_tier3_fold` condenses, with the operator's wording quoted
 rather than paraphrased.

 ``record_request`` is called with each summariser request before it is made,
 which is what puts the summariser on the same footing as the turn's own
 provider calls: a compaction is precisely the event that rewrites the
 transcript every later request is built from, so a recording that skipped it
 could not be replayed past the first one.

 When ``protect_tail_from_index`` is set (the per-iteration gate), the
 current iteration's just-executed tool batch (assistant ``tool_use`` turn +
 its results) is exempt from summarisation on top of the keep window, so a
 >keep parallel batch's fresh results cannot be summarised away before the
 next assistant stream consumes them. See
 :func:`current_tool_batch_protect_index`.

 No-net-gain floor — a unit whose combined token estimate is at or below
 :func:`_compaction_wrapper_floor_tokens` (the empty ``<compacted-turn>``
 wrapper) cannot shrink under summarisation; replacing it would only GROW
 history. Such units are skipped before any LLM call, so Tier-2 never inflates
 a small-turn-dominated history (which would push ``tokens_after`` above
 ``tokens_before`` and miscount a no-progress retry toward
 :class:`CompactionExhaustedError`) and never spends a summariser call that
 cannot free tokens.

 Bounded per-pass cost — ``free_target_tokens`` (when set by the caller) is
 the freed-token budget for this pass: once that many tokens have been freed
 no further batch is issued. The remaining eligible units are summarised on a
 later pass if still needed. Within a batch the calls go out together, up to
 ``rc.compaction_summariser_parallelism`` at a time: the calls are seconds
 long apiece and the run is parked in ``COMPACTING`` for the whole pass, so
 issuing them one after another was most of that wait.

 ``prompts`` renders the summariser instruction. ``None`` falls back to the
 templates bundled with the package, so a caller that wired no provider still
 gets the shipped wording rather than a literal built here.

 Mutates ``history`` in place.
 """
    if not history:
        return Tier2Result(turns_summarised=0, tokens_freed=0)

    keep = rc.compaction_keep_recent_turns
    eligible_upper = _effective_eligible_upper(history, keep, protect_tail_from_index)
    if eligible_upper == 0:
        return Tier2Result(turns_summarised=0, tokens_freed=0)

    protected: frozenset[int] = frozenset()
    if rc.compaction_protect_first_user_turn:
        first_user = _first_user_turn_index(history)
        if first_user is not None:
            protected = frozenset({first_user})

    # Every operator turn stays verbatim. An instruction sent mid-run — a
    # steer, a correction, a follow-up — is short, so summarising it frees
    # almost nothing, and it is specific, so a paraphrase of it is exactly the
    # loss this pass cannot afford: "remove the model-name field from the
    # header" becomes "the user asked for changes to the header" and the run
    # goes on to do something else. What condenses them instead is Tier 3,
    # which keeps their wording as quotes.
    protected = protected | frozenset(_operator_turn_indices(history))

    # Protect executor-seeded prior-run turns from the lossy Tier-2 collapse so
    # a summary never drops the SESSION_HISTORY_SEED tag (which the host
    # finalization filter relies on to avoid re-persisting prior-run
    # conversation under the new run_id). Tier-1 still bounds them: seeded tool
    # results via the main shed path, and the session summary/ledger seed blocks
    # via the reference path (they DUAL-TAG COMPACTION_REFERENCE — see
    # ``_session_history_seed_indices``).
    seed_indices = _session_history_seed_indices(history)
    if seed_indices:
        protected = protected | seed_indices

    # Reference blocks are a Tier-1-ONLY shed surface (A2(2) recoverable blob
    # path). Tier-2 must never collapse them — un-blobbed, that loses frozen
    # bootstrap context; blobbed, the placeholder is the only in-history
    # pointer to the blob and summarising it both wastes an LLM call on the
    # raw placeholder marker and erases the blob ref. See
    # ``_compaction_reference_indices``.
    reference_indices = _compaction_reference_indices(history)
    if reference_indices:
        protected = protected | reference_indices

    units = _build_summarisation_units(history, eligible_upper, protected_indices=protected)
    resolved_prompts = prompts if prompts is not None else bundled_prompt_provider()

    summarised = 0
    freed = 0
    # Replacement Message keyed by anchor index; indices to delete after the loop.
    replacements: dict[int, Message] = {}
    indices_to_drop: set[int] = set()

    # Which units are worth a call at all, decided before any call is made:
    # not already summarised, and big enough that a summary could come back
    # smaller than what it replaces.
    jobs: list[tuple[_SummarisationUnit, str, list[Message], int]] = []
    for unit in units:
        anchor = history[unit.anchor_idx]
        # A4 idempotency — never re-summarise an existing compaction summary
        # (would nest <compacted-turn> wrappers and decay the summary).
        if _is_compaction_summary(anchor):
            continue
        anchor_key = _stable_turn_key(anchor)
        if anchor_key in state.summarised_turn_ids:
            continue
        if state.failed_anchor_keys.get(anchor_key, 0) >= rc.compaction_summary_failed_unit_max_attempts:
            # The summariser has failed on this unit as often as it may: paying
            # again buys the same failure. The fold tier still gets its turn.
            continue
        # Exhaustive across EVERY member of the unit (assistant turn + its
        # tool results), so the summary preserves the tool exchange.
        unit_messages = [history[member] for member in unit.indices]
        before_tokens = sum(estimate_message_tokens(member, rc) for member in unit_messages)
        # No-net-gain floor — a unit at or below the empty-wrapper size cannot
        # shrink; replacing it would only GROW history (the inflation the
        # max(0, ...) freed clamp would mask). The operator's own minimum sits
        # on top of that floor: a summariser writes a sentence or three
        # whatever it is handed, so below some size the call is spent to
        # discover the summary is no smaller.
        floor = max(
            _compaction_wrapper_floor_tokens(anchor_key, rc),
            rc.compaction_summary_min_unit_tokens,
        )
        if before_tokens <= floor:
            continue
        jobs.append((unit, anchor_key, unit_messages, before_tokens))

    # Calls go out in small parallel batches, oldest unit first, and stop once
    # the pass has freed its budget. Sequentially this was one chain of
    # seconds-long calls with the run parked in COMPACTING for the length of
    # it; the cap is what keeps the alternative from being an unbounded
    # fan-out at the provider.
    batch_size = rc.compaction_summariser_parallelism
    for offset in range(0, len(jobs), batch_size):
        if free_target_tokens is not None and freed >= free_target_tokens:
            break
        batch = jobs[offset : offset + batch_size]
        outcomes = await asyncio.gather(
            *(
                _summarise_unit(
                    unit_messages,
                    anchor_key=anchor_key,
                    unit_label=f"tier2 unit anchor_idx={unit.anchor_idx}",
                    before_tokens=before_tokens,
                    compaction_llm=compaction_llm,
                    rc=rc,
                    prompts=resolved_prompts,
                    model_name=model_name,
                    observability=observability,
                    record_request=record_request,
                )
                for unit, anchor_key, unit_messages, before_tokens in batch
            )
        )
        for (unit, _key, _members, _before), outcome in zip(batch, outcomes, strict=True):
            if outcome.replacement is None:
                if outcome.failed:
                    state.failed_anchor_keys[outcome.anchor_key] = state.failed_anchor_keys.get(outcome.anchor_key, 0) + 1
                continue
            replacements[unit.anchor_idx] = outcome.replacement
            # Every non-anchor member of the unit (the matching tool results) is
            # removed so the dropped ToolUseBlock leaves no orphaned tool_result.
            indices_to_drop.update(member for member in unit.indices if member != unit.anchor_idx)
            state.summarised_turn_ids.add(outcome.anchor_key)
            summarised += 1
            freed += outcome.tokens_freed

    if replacements or indices_to_drop:
        rebuilt: list[Message] = []
        for idx in range(len(history)):
            if idx in indices_to_drop:
                continue
            rebuilt.append(replacements.get(idx, history[idx]))
        history[:] = rebuilt

    return Tier2Result(turns_summarised=summarised, tokens_freed=freed)


COMPACTION_FOLD_METADATA_KEY: Final[str] = "protocore.compaction_fold"
"""On a fold summary: how many messages it stands for, and how many were the operator's."""


def _foldable_indices(history: list[Message], eligible_upper: int, rc: LoopConstants) -> frozenset[int]:
    """Positions the fold may consolidate.

    A message qualifies when it is an old compaction summary or an old operator
    turn, and is none of the things every tier protects: the first user turn
    (this run's task), one of the ``compaction_fold_keep_operator_turns`` most
    recent operator turns, a turn seeded from an earlier run of the session, or
    a frozen reference block. The seed exclusion is not cosmetic — a fold that
    absorbed a seeded turn would drop the tag that separates this run's
    messages from the previous run's, which is the one thing distinguishing
    them.
    """
    first_user = _first_user_turn_index(history) if rc.compaction_protect_first_user_turn else None
    keep = rc.compaction_fold_keep_operator_turns
    operators = _operator_turn_indices(history)
    recent_operators = frozenset(operators[-keep:]) if keep else frozenset()
    protected = recent_operators | _session_history_seed_indices(history) | _compaction_reference_indices(history)
    return frozenset(
        idx
        for idx in range(min(eligible_upper, len(history)))
        if idx != first_user
        and idx not in protected
        and (_is_compaction_summary(history[idx]) or _is_plain_operator_turn(history[idx]))
    )


def _fold_spans(history: list[Message], eligible_upper: int, rc: LoopConstants) -> list[tuple[int, int]]:
    """Contiguous runs ``[start, end)`` of foldable messages, long and heavy enough to be worth a call.

    A run must be at least ``compaction_fold_min_messages`` long and
    ``compaction_fold_min_tokens`` big. Both bounds exist so a span that has
    already been folded is not folded again for nothing: one fold summary
    standing alone is neither long enough nor heavy enough to qualify.
    """
    foldable = _foldable_indices(history, eligible_upper, rc)
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for idx in range(len(history) + 1):
        if idx in foldable:
            if start is None:
                start = idx
            continue
        if start is not None:
            end = idx
            if end - start >= rc.compaction_fold_min_messages:
                tokens = sum(estimate_message_tokens(history[j], rc) for j in range(start, end))
                if tokens >= rc.compaction_fold_min_tokens:
                    spans.append((start, end))
            start = None
    return spans


def _fold_item_text(message: Message) -> str:
    """One span member as the fold prompt shows it.

    The two kinds are labelled apart because the instruction that follows
    treats them differently: an earlier summary may be condensed further, an
    operator's words must come back out as a quote.
    """
    text = message.text.strip()
    if _is_compaction_summary(message):
        inner = re.sub(r"^<compacted-turn[^>]*>", "", text).removesuffix("</compacted-turn>").strip()
        return f"[earlier summary] {inner}"
    return f"[operator said] {text}"


def _fold_anchor_key(members: Sequence[Message]) -> str:
    """A durable id for the span, derived from its members.

    Content-addressed like every other summary key, so the same span folded
    after a snapshot and resume produces the same id and the dedup set
    recognises it.
    """
    digest = hashlib.sha256("|".join(_stable_turn_key(m) for m in members).encode("utf-8")).hexdigest()
    return f"fold-{digest[:16]}"


async def _fold_span(
    members: list[Message],
    *,
    compaction_llm: ILLMProvider,
    rc: LoopConstants,
    prompts: IPromptTemplateProvider,
    model_name: str,
    observability: LLMObservabilityContext | None,
    record_request: RequestRecorder | None,
) -> _SummaryOutcome:
    """Fold ONE run of old summaries and operator turns into a single summary."""
    anchor_key = _fold_anchor_key(members)
    before_tokens = sum(estimate_message_tokens(member, rc) for member in members)
    operator_count = sum(1 for member in members if _is_plain_operator_turn(member))
    prompt = prompts.render(
        "compaction_fold_summary",
        {
            "items": "\n\n".join(_strip_injection_patterns(_fold_item_text(m)) for m in members),
            "item_count": len(members),
            "operator_count": operator_count,
            "max_words": rc.compaction_fold_summary_target_words,
        },
    )
    outcome = await _run_summariser(
        prompt,
        anchor_key=anchor_key,
        unit_label=f"tier3 span of {len(members)} messages",
        before_tokens=before_tokens,
        max_output_tokens=rc.compaction_fold_max_output_tokens,
        compaction_llm=compaction_llm,
        rc=rc,
        model_name=model_name,
        observability=observability,
        record_request=record_request,
    )
    if outcome.replacement is None:
        return outcome
    # A fold says what it stands for. Nothing in the loop branches on this; it
    # is what makes a window of folds readable afterwards, when the question is
    # how much of the session one message is now carrying.
    return _SummaryOutcome(
        anchor_key=outcome.anchor_key,
        replacement=outcome.replacement.model_copy(
            update={
                "metadata": {
                    **outcome.replacement.metadata,
                    COMPACTION_FOLD_METADATA_KEY: {
                        "messages": len(members),
                        "operator_turns": operator_count,
                    },
                }
            }
        ),
        tokens_freed=outcome.tokens_freed,
    )


async def run_tier3_fold(
    history: list[Message],
    compaction_llm: ILLMProvider,
    state: CompactionState,
    rc: LoopConstants,
    *,
    model_name: str,
    observability: LLMObservabilityContext | None = None,
    protect_tail_from_index: int | None = None,
    record_request: RequestRecorder | None = None,
    prompts: IPromptTemplateProvider | None = None,
) -> Tier3Result:
    """Fold runs of old summaries and old operator turns into one summary each.

    Tier 2 leaves one ``<compacted-turn>`` per tool batch and never touches a
    summary again, and it keeps every operator turn verbatim. Over a long
    session those two become the window: a run measured here reached 184
    summaries and 35 operator turns in 344 messages, and neither tier below
    this one could take a byte off it. This pass replaces each contiguous run
    of such messages in the eligible region (see :func:`_fold_spans`) with a
    single consolidated summary in which operator instructions survive as exact
    quotes. The result is a summary like any other — same wrapper, same
    metadata flag — so a later fold absorbs it once its neighbourhood has grown
    again.

    Bounded on purpose: at most ``rc.compaction_fold_max_spans_per_pass`` runs
    are folded, in batches of ``rc.compaction_summariser_parallelism``. A
    history with many foldable runs is compacted over several passes rather
    than in one long ``COMPACTING`` pause.

    Mutates ``history`` in place.
    """
    if not history or not rc.compaction_fold_enabled:
        return Tier3Result(spans_folded=0, messages_folded=0, tokens_freed=0)
    eligible_upper = _effective_eligible_upper(
        history, rc.compaction_keep_recent_turns, protect_tail_from_index
    )
    if eligible_upper == 0:
        return Tier3Result(spans_folded=0, messages_folded=0, tokens_freed=0)
    spans = _fold_spans(history, eligible_upper, rc)[: rc.compaction_fold_max_spans_per_pass]
    if not spans:
        return Tier3Result(spans_folded=0, messages_folded=0, tokens_freed=0)

    resolved_prompts = prompts if prompts is not None else bundled_prompt_provider()
    replacements: dict[int, Message] = {}
    drop: set[int] = set()
    folded_spans = 0
    folded_messages = 0
    freed = 0

    batch_size = rc.compaction_summariser_parallelism
    for offset in range(0, len(spans), batch_size):
        batch = spans[offset : offset + batch_size]
        outcomes = await asyncio.gather(
            *(
                _fold_span(
                    history[start:end],
                    compaction_llm=compaction_llm,
                    rc=rc,
                    prompts=resolved_prompts,
                    model_name=model_name,
                    observability=observability,
                    record_request=record_request,
                )
                for start, end in batch
            )
        )
        for (start, end), outcome in zip(batch, outcomes, strict=True):
            if outcome.replacement is None:
                continue
            replacements[start] = outcome.replacement
            drop.update(range(start + 1, end))
            state.summarised_turn_ids.add(outcome.anchor_key)
            folded_spans += 1
            folded_messages += end - start
            freed += outcome.tokens_freed

    if replacements:
        rebuilt: list[Message] = []
        for idx in range(len(history)):
            if idx in drop:
                continue
            rebuilt.append(replacements.get(idx, history[idx]))
        history[:] = rebuilt
    return Tier3Result(spans_folded=folded_spans, messages_folded=folded_messages, tokens_freed=freed)


__all__ = [
    "COMPACTION_FOLD_METADATA_KEY",
    "CompactionAttempt",
    "CompactionExhaustedError",
    "CompactionState",
    "Tier1Result",
    "Tier2Result",
    "Tier3Result",
    "TokenEstimator",
    "build_summary_schema",
    "run_tier1_truncation",
    "run_tier2_summarisation",
    "run_tier3_fold",
]
