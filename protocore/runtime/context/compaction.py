"""The compaction tiers: masking, summarising, folding, and the floor.

The contract these implement is written down in ``docs/compaction.md``; this
docstring only maps it onto the functions here.

Tier 1 — masking (:func:`run_tier1_truncation`). Tool outputs outside the
 protected tail are replaced, oldest first, by a placeholder that names the
 tool and points at the original in the :class:`IBlobStore`: any output larger
 than ``tool_result_truncation_threshold``, and — when the pass still needs
 room — any output older than the ``compaction_mask_keep_recent_results`` most
 recent. Needs no model and cannot fail for want of one.

Tier 2 — summarising (:func:`run_tier2_summarisation`). The oldest spans (an
 assistant turn with the results answering it, or a run of small ones joined)
 are replaced by a summary written by the compaction LLM as plain text under
 fixed headings (:mod:`protocore.runtime.context.carrier`). The summariser is
 shown masked outputs in full, read back from the blob store, within a bounded
 input. An operator turn is never summarised here.

Tier 2b — folding (:func:`run_tier3_fold`). Runs of old summaries and old
 operator turns are merged into one summary. The operator's wording does not
 depend on the merge: it is copied into the ledger by code first.

Floor (:func:`run_floor`). What the tiers above could not bring under the
 target is removed, oldest span first, and replaced by one runtime digest
 written by code. Nothing a model does can stop it.

Every tier records what it removes in the compaction ledger
(:mod:`protocore.runtime.context.ledger`) before it removes it.

Cyrillic-in-JSON-escape safety preserved via
:mod:`protocore.runtime.token_counting`.
"""
from __future__ import annotations

import asyncio
import hashlib
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
    LLMContextWindowExceeded,
    LLMObservabilityContext,
    LLMRequest,
)
from protocore.contracts.prompts import IPromptTemplateProvider
from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.tool_roles import EMPTY_TOOL_ROLE_MAP, ToolRoleMap
from protocore.contracts.types import (
    COMPACTION_REFERENCE_METADATA_KEY,
    COMPACTION_SUMMARY_METADATA_KEY,
    SESSION_HISTORY_SEED_METADATA_KEY,
    SYNTHETIC_RECOVERY_METADATA_KEY,
    CompactionSourceRef,
    ContentBlock,
    ImageRefBlock,
    Message,
    MessageRole,
    StopReason,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from protocore.logging_utils import get_logger
from protocore.prompts import bundled_prompt_provider
from protocore.runtime.context.carrier import (
    carrier_header,
    read_carrier,
    request_max_tokens,
    summary_output_budget,
)
from protocore.runtime.context.ledger import Ledger, distinct_lines, is_ledger, ledger_message
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


class CompactionExhaustedError(RuntimeError):
    """Compaction failed beyond :attr:`LoopConstants.compaction_failed_max_retries`."""


@dataclass(frozen=True, slots=True)
class Tier1Result:
    """Outcome of Tier 1 — masking tool outputs."""

    tokens_freed: int
    blob_refs_created: tuple[str, ...]
    messages_modified: int
    masked_by_age: int = 0
    """Outputs masked because they were old, not because they were large."""


@dataclass(frozen=True, slots=True)
class Tier2Result:
    """Outcome of Tier 2 — summarising the oldest spans."""

    turns_summarised: int
    tokens_freed: int
    units_attempted: int = 0
    """Summariser calls the pass issued. Zero with nothing summarised means the
    pass found no unit it was allowed to send — nothing to do, not a failure."""
    failures: dict[str, int] = field(default_factory=dict)
    """Calls that produced no summary, by kind: ``transport``, ``timeout``,
    ``empty``, ``too_large``, ``not_smaller``."""
    recovered: dict[str, int] = field(default_factory=dict)
    """Summaries kept from a reply that was not plain headed text, by how:
    ``json``, ``unterminated_json``, ``truncated``, ``unheaded``."""


@dataclass(frozen=True, slots=True)
class Tier3Result:
    """Outcome of Tier 2b — folding runs of old summaries and operator turns."""

    spans_folded: int
    messages_folded: int
    tokens_freed: int
    spans_attempted: int = 0
    """Fold calls the pass issued; zero means no span qualified."""
    failures: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FloorResult:
    """Outcome of the floor — spans removed without a model."""

    messages_dropped: int
    tokens_freed: int
    reached: bool
    """True when nothing more could be removed and the prompt is still above
    the target: the history is at its floor (head, ledger and kept tail)."""


@dataclass(slots=True)
class CompactionAttempt:
    """One compaction pass's combined outcome.

    Used by :class:`QueryEngine` to emit ``compaction_started`` /
    ``compaction_completed`` event payloads with full provenance.
    ``tokens_before``/``tokens_after`` size the history; ``prompt_before`` and
    ``prompt_after`` add what the request carries besides it (system prompt,
    tools), which is what the gate and the target are measured against.
    """

    tier1: Tier1Result | None = None
    tier2: Tier2Result | None = None
    tier3: Tier3Result | None = None
    floor: FloorResult | None = None
    tokens_before: int = 0
    tokens_after: int = 0
    prompt_before: int = 0
    prompt_after: int = 0
    trigger_tokens: int = 0
    target_tokens: int = 0
    ledger_tokens: int = 0
    outcome: str = ""
    """``below_target``, ``below_trigger``, ``floor`` (the floor removed spans
    and reached the target), ``at_floor`` (nothing left to remove, still above
    the trigger) or ``unchanged`` (no tier had anything to do)."""


@dataclass(slots=True)
class CompactionState:
    """Per-engine compaction state — counts retries + tracks summarised IDs."""

    retry_count: int = 0
    """Consecutive failed routine and proactive passes."""
    reactive_retry_count: int = 0
    """Consecutive failed reactive passes — the ones run after a provider
    rejection, under the profile that may also compact seeded history. Kept
    apart from :attr:`retry_count` because a proactive pass leaves that history
    alone: its failures say nothing about what the reactive profile can still
    free, and must not spend the budget the reactive profile is owed."""
    summarised_turn_ids: set[str] = field(default_factory=set)
    blob_refs_created: list[str] = field(default_factory=list)
    failed_anchor_keys: dict[str, int] = field(default_factory=dict)
    """How many passes the summariser has failed on a unit, by anchor key.

    Past ``compaction_summary_failed_unit_max_attempts`` the unit is not sent
    again: the failure is a property of the unit (too large to fit, or a reply
    the cap cuts every time), so paying for it again buys the same failure.
    Snapshotted with the rest of the state, because a run re-driven on another
    pod would otherwise start the same census from zero.
    """

    def reset_retries(self) -> None:
        """Clear both retry budgets.

        Any progress clears both: a pass that changed the history ends the
        run of consecutive failures for either profile, since the next pass of
        either kind faces a different history.
        """
        self.retry_count = 0
        self.reactive_retry_count = 0


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

    ``token_estimate_calibration`` is deliberately NOT here. It is a single
    multiplier over the whole partition, applied where the number is handed
    out, so what is cached does not depend on it. Keying on it split the cache
    down the middle: the calibrator sizes the same history uncalibrated to
    compare it against what the provider reported, and every such pass evicted
    the calibrated entries the loop had just paid for, and was evicted by them
    in turn — a measured 72% on the history estimate when the two alternate.
    """
    return (
        rc.token_count_chars_per_token_latin,
        rc.token_count_chars_per_token_cyrillic,
        rc.token_count_chars_per_token_cyrillic_json_escape,
        rc.token_count_chars_per_token_cjk,
        rc.token_count_chars_per_token_json_struct,
        rc.token_count_image_tokens,
    )


def _estimate_message_tokens_uncached(message: Message, rc: LoopConstants) -> int:
    """The heuristic's own count of one message, before calibration.

    The provider factor is applied by the callers below rather than here: it is
    a multiplier over this whole number, so folding it in would make the cached
    partition depend on a value that does not change the partition. See
    :func:`_token_estimate_signature`.
    """
    total = 0
    for block in message.content_blocks:
        if isinstance(block, ImageRefBlock):
            total += rc.token_count_image_tokens
            continue
        total += estimate_tokens(_block_text_for_estimation(block), rc)
    if message.reasoning_content:
        total += estimate_tokens(message.reasoning_content, rc)
    return total


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
        """Token weight of one message, from the cache when it is still valid.

        In the provider's tokens, not the heuristic's: see
        :attr:`LoopConstants.token_estimate_calibration`.
        """
        raw = self._estimate(message, rc, _token_estimate_signature(rc))
        return round(raw * rc.token_estimate_calibration)

    def estimate_history(
        self,
        history: Sequence[Message],
        rc: LoopConstants,
    ) -> int:
        """Token weight of a sequence, paying only for messages not yet seen."""
        signature = _token_estimate_signature(rc)
        calibration = rc.token_estimate_calibration
        return sum(
            round(self._estimate(message, rc, signature) * calibration)
            for message in history
        )

    def estimate_history_uncalibrated(
        self,
        history: Sequence[Message],
        rc: LoopConstants,
    ) -> int:
        """The same sum in the heuristic's own tokens.

        What the calibrator compares against the size the provider reported:
        multiplying by the factor first and dividing it back out would be the
        same arithmetic with a rounding error in it.
        """
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


def estimate_history_tokens_uncalibrated(
    history: Sequence[Message],
    rc: LoopConstants,
) -> int:
    """Sum the heuristic's own count over ``history``, before calibration.

    The calibrator's side of :attr:`LoopConstants.token_estimate_calibration`:
    it asks what the heuristic makes of the very request the provider has just
    reported a size for, and scales the factor by the ratio. Answered from the
    same cache the calibrated readings use, because the two differ only in a
    multiplier applied afterwards.
    """
    return _shared_estimator.estimate_history_uncalibrated(history, rc)


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
    (default 4; reactive recovery narrows it to
    ``compaction_force_keep_recent_turns``) only protects the trailing N
    messages, so a parallel batch of
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


def _tier1_sheds_reasoning(message: Message, rc: LoopConstants) -> bool:
    """An aged assistant turn whose re-emitted reasoning Tier 1 would drop."""
    return (
        rc.compaction_shed_reasoning_enabled
        and message.role is MessageRole.assistant
        and bool(message.reasoning_content)
    )


def _tier1_reference_candidate(message: Message, rc: LoopConstants) -> bool:
    """A frozen reference block Tier 1 owns — shed or not, no other branch applies."""
    if not message.content_blocks:
        return False
    block = message.content_blocks[0]
    return (
        rc.compaction_bound_reference_blocks_enabled
        and message.role is not MessageRole.tool
        and message.metadata.get(COMPACTION_REFERENCE_METADATA_KEY) is True
        and isinstance(block, TextBlock)
        and not _content_is_already_compacted(block.text)
    )


def _tier1_sheds_result(block: ContentBlock, rc: LoopConstants, threshold: int) -> bool:
    """A tool result large enough, and not yet shed, for Tier 1 to blob."""
    return (
        isinstance(block, ToolResultBlock)
        and not _content_is_already_compacted(block.content)
        and estimate_tokens(block.content, rc) >= threshold
    )


def _age_mask_candidates(
    history: list[Message],
    eligible_upper: int,
    rc: LoopConstants,
) -> list[int]:
    """Positions of tool messages whose outputs are old enough to mask, oldest first.

    Old means outside the ``compaction_mask_keep_recent_results`` most recent
    tool outputs of the whole history, as well as outside the protected tail;
    worth masking means at least ``compaction_mask_min_tokens`` — below that the
    placeholder costs nearly what it saves.
    """
    keep = rc.compaction_mask_keep_recent_results
    result_positions = [
        idx
        for idx, message in enumerate(history)
        if message.role is MessageRole.tool
        and any(isinstance(block, ToolResultBlock) for block in message.content_blocks)
    ]
    recent = frozenset(result_positions[-keep:]) if keep else frozenset()
    return [
        idx
        for idx in result_positions
        if idx < eligible_upper
        and idx not in recent
        and any(
            _tier1_sheds_result(block, rc, max(1, rc.compaction_mask_min_tokens))
            for block in history[idx].content_blocks
        )
    ]


def tier1_has_work(
    history: list[Message],
    rc: LoopConstants,
    truncation_threshold_tokens: int,
    *,
    keep_recent_turns: int | None = None,
    protect_tail_from_index: int | None = None,
    mask_by_age: bool = False,
) -> bool:
    """Whether :func:`run_tier1_truncation` would rewrite anything — without rewriting it."""
    if not history:
        return False
    keep = rc.compaction_keep_recent_turns if keep_recent_turns is None else keep_recent_turns
    eligible_upper = _effective_eligible_upper(history, keep, protect_tail_from_index)
    for message in history[:eligible_upper]:
        if _tier1_sheds_reasoning(message, rc) and estimate_tokens(
            message.reasoning_content or "", rc
        ) > 0:
            return True
        if _tier1_reference_candidate(message, rc):
            block = message.content_blocks[0]
            if (
                isinstance(block, TextBlock)
                and estimate_tokens(block.text, rc) >= truncation_threshold_tokens
            ):
                return True
            continue
        if message.role is MessageRole.tool and any(
            _tier1_sheds_result(block, rc, truncation_threshold_tokens)
            for block in message.content_blocks
        ):
            return True
    return mask_by_age and bool(_age_mask_candidates(history, eligible_upper, rc))


def _masked_text(ref: CompactionSourceRef, preview: str, distinct: Sequence[str]) -> str:
    """The placeholder a masked output leaves: a machine line, and lines the model can read.

    The first line is the wire-format frame other components parse; its
    preview field is left empty because base64 is not something a model
    reads. Then what was there and where it went, the output's first and last
    characters, and the lines it did not repeat (see
    :func:`~protocore.runtime.context.ledger.distinct_lines`) — the setting,
    the error, the result that a log of a thousand lines said once.
    """
    what = f"output of {ref.tool_name}" if ref.tool_name else "tool output"
    tail = f" First and last lines: {preview}" if preview else ""
    kept = "".join(f"\n  {line}" for line in distinct)
    lines_note = f"\nLines it did not repeat:{kept}" if kept else ""
    return (
        f"{render_compacted_placeholder(ref, 'SNAPSHOT')}\n"
        f"[The {what} ({ref.original_tokens} tokens) was masked by compaction; "
        f"the original is stored as {ref.blob_ref}.{tail}]{lines_note}"
    )


async def _mask_result(
    block: ToolResultBlock,
    *,
    tool_name: str,
    blob_store: IBlobStore,
    tenant_id: str,
    rc: LoopConstants,
) -> tuple[ToolResultBlock, int, str | None]:
    """One output replaced by its placeholder: the new block, tokens freed, the blob written (if any)."""
    original_tokens = estimate_tokens(block.content, rc)
    # What gets stored is the CANONICAL value, not the text in front of the
    # model. A tool that handed back a short view of a long result left the
    # long result on the block; blobbing the view instead would put a
    # truncated copy behind a reference the placeholder calls canonical.
    canonical_text = block.canonical_content or block.content
    # The canonical value is stored ONCE. A block that already names where its
    # value lives — a tool that stored its own output, a result an earlier
    # pass already shed — is not stored again: the reference it carries is the
    # canonical value's address, and a second copy under a second address
    # would leave two answers to "what did this call return".
    created: str | None = None
    if block.canonical_ref is not None:
        canonical_ref = block.canonical_ref
        # The bytes behind that reference were written elsewhere and this pass
        # has not seen them; a digest of what is on the block would describe a
        # different string, so the placeholder says nothing rather than
        # something false.
        sha256 = ""
    else:
        content_bytes = canonical_text.encode("utf-8")
        sha256 = hashlib.sha256(content_bytes).hexdigest()
        blob_md = await blob_store.put(
            tenant_id=tenant_id,
            content=content_bytes,
            content_type="text/plain; charset=utf-8",
            metadata={"tool_call_id": block.tool_call_id, "label": "tool_result", "tier": "tier1"},
        )
        canonical_ref = blob_md.ref
        created = canonical_ref
    text = _masked_text(
        CompactionSourceRef(
            blob_ref=canonical_ref,
            sha256=sha256,
            original_tokens=original_tokens,
            label="tool_result",
            tool_name=tool_name,
            preview="",
        ),
        _content_preview(block.content, rc.compaction_placeholder_preview_chars),
        distinct_lines(canonical_text, limit=rc.compaction_mask_distinct_lines),
    )
    # What is shed is the PROJECTION. ``canonical_ref`` survives on the block,
    # and so does ``path``: a result whose text is now a placeholder still
    # describes the file it described, and a later write must still be able to
    # say it is out of date.
    masked = ToolResultBlock(
        tool_call_id=block.tool_call_id,
        content=text,
        is_error=block.is_error,
        metadata={**block.metadata, "compacted": True, "blob_ref": canonical_ref},
        canonical_ref=canonical_ref,
        path=block.path,
    )
    return masked, original_tokens - estimate_tokens(text, rc), created


async def run_tier1_truncation(
    history: list[Message],
    blob_store: IBlobStore,
    tenant_id: str,
    rc: LoopConstants,
    truncation_threshold_tokens: int,
    *,
    keep_recent_turns: int | None = None,
    protect_tail_from_index: int | None = None,
    mask_by_age: bool = False,
    free_target_tokens: int | None = None,
    ledger: Ledger | None = None,
) -> Tier1Result:
    """Mask tool outputs, shed aged reasoning and bound reference blocks — no model involved.

    Mutates ``history`` in place, oldest first. Three passes over the region
    outside the kept tail:

    * ``compaction_shed_reasoning_enabled`` — strip ``reasoning_content`` from
      aged assistant turns: re-emitted chain-of-thought is single-turn
      scaffolding the model never needs from prior turns.
    * ``compaction_bound_reference_blocks_enabled`` — blob an over-budget frozen
      reference block (a non-tool single-text message tagged
      ``COMPACTION_REFERENCE_METADATA_KEY``, e.g. the executor's bootstrap).
      The original task user turn is never tagged, so it is never shed here.
    * every tool output at or above ``truncation_threshold_tokens`` is masked.

    Then, with ``mask_by_age``, outputs older than the
    ``compaction_mask_keep_recent_results`` most recent are masked oldest first
    until ``free_target_tokens`` have been freed (all of them when it is
    ``None``). A history of hundreds of short rounds has no single large
    output, and this is what lets it shrink without a model at all.

    Every masked output leaves a placeholder that names the tool, says how
    large it was, points at the stored original and shows its first and last
    lines; ``ledger`` (when given) records the output's exact values before it
    leaves the window.

    ``protect_tail_from_index`` exempts the current just-executed batch (any
    size) on top of ``keep_recent_turns``; see
    :func:`current_tool_batch_protect_index`.
    """
    if not history:
        return Tier1Result(tokens_freed=0, blob_refs_created=(), messages_modified=0)

    keep = rc.compaction_keep_recent_turns if keep_recent_turns is None else keep_recent_turns
    eligible_upper = _effective_eligible_upper(history, keep, protect_tail_from_index)
    preview_cap = rc.compaction_placeholder_preview_chars

    # Hoisted out of the loop: naming the tool behind a shed result per block
    # would walk the whole transcript per block — quadratic exactly on the
    # large histories compaction exists for. Only result blocks and
    # reasoning_content are rewritten below, so no tool_use block moves.
    tool_names = tool_names_by_call_id(history)

    tokens_freed = 0
    refs_created: list[str] = []
    modified: set[int] = set()

    async def mask_message(idx: int, threshold: int) -> int:
        message = history[idx]
        new_blocks: list[ContentBlock] = []
        freed_here = 0
        changed = False
        # A tool-role message may carry MORE THAN ONE ToolResultBlock (a single
        # tool message may answer several parallel calls). Every block is
        # considered and non-result siblings are kept.
        for block in message.content_blocks:
            if not isinstance(block, ToolResultBlock) or not _tier1_sheds_result(block, rc, threshold):
                new_blocks.append(block)
                continue
            name = tool_names.get(block.tool_call_id, "")
            if ledger is not None:
                ledger.absorb_result(block, tool_name=name)
            masked, freed, created = await _mask_result(
                block, tool_name=name, blob_store=blob_store, tenant_id=tenant_id, rc=rc
            )
            if created is not None:
                refs_created.append(created)
            new_blocks.append(masked)
            freed_here += freed
            changed = True
        if changed:
            history[idx] = message.model_copy(update={"content_blocks": new_blocks})
            modified.add(idx)
        return freed_here

    for idx in range(eligible_upper):
        message = history[idx]

        # Shed re-emitted reasoning_content on aged assistant turns, before the
        # tool-result branch, so a tool-pairing assistant turn with bloated CoT
        # is trimmed even when its result is what gets masked.
        if _tier1_sheds_reasoning(message, rc) and message.reasoning_content:
            freed = estimate_tokens(message.reasoning_content, rc)
            if freed > 0:
                history[idx] = message.model_copy(update={"reasoning_content": None})
                message = history[idx]
                tokens_freed += freed
                modified.add(idx)

        if not message.content_blocks:
            continue

        block = message.content_blocks[0]
        # Bound an over-budget FROZEN reference block (bootstrap).
        if _tier1_reference_candidate(message, rc) and isinstance(block, TextBlock):
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
                modified.add(idx)
                refs_created.append(ref_blob.ref)
                tokens_freed += ref_tokens - estimate_tokens(ref_placeholder, rc)
            continue

        if message.role is MessageRole.tool:
            tokens_freed += await mask_message(idx, truncation_threshold_tokens)

    masked_by_age = 0
    if mask_by_age:
        for idx in _age_mask_candidates(history, eligible_upper, rc):
            if free_target_tokens is not None and tokens_freed >= free_target_tokens:
                break
            freed = await mask_message(idx, max(1, rc.compaction_mask_min_tokens))
            if freed > 0:
                masked_by_age += 1
            tokens_freed += freed

    return Tier1Result(
        tokens_freed=max(0, tokens_freed),
        blob_refs_created=tuple(refs_created),
        messages_modified=len(modified),
        masked_by_age=masked_by_age,
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
            and not is_ledger(message)
        ):
            return idx
    return None


def _session_history_seed_indices(history: list[Message]) -> frozenset[int]:
    """Return the history indices of executor-seeded prior-run turns.

    Routine Tier-2 summarisation and Tier-3 folding protect these so a lossy
    summary never silently collapses seeded prior-run content into an UNtagged
    ``<compacted-turn>`` message — which would (a) defeat the host
    finalization filter that excludes seed-tagged turns from re-persistence and
    (b) re-write prior-run conversation under the new ``run_id``. Reactive
    recovery after a provider rejection (``compact_seeded_history=True``) may
    replace seed-only units and spans, and then copies the seed tag onto every
    replacement so the filter still holds.

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

    Only the lossy Tier-2 collapse is withheld from seed-tagged turns outside
    reactive recovery; the reference dual-tag does not weaken that (Tier-2
    still skips them by seed tag).
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


def _wrap_compaction_summary(anchor_key: str, summary_text: str, *, attributes: str = "") -> str:
    """Build the ``<compacted-turn>`` replacement body for a summarised span.

    Single source of truth for the wrapper so the no-net-gain floor
    (:func:`_compaction_wrapper_floor_tokens`) and the actual replacement stay
    byte-for-byte in sync.
    """
    extra = f" {attributes}" if attributes else ""
    return f"<compacted-turn id='{anchor_key}'{extra}>\n{summary_text}\n</compacted-turn>"


def _compaction_wrapper_floor_tokens(anchor_key: str, rc: LoopConstants) -> int:
    """Estimated token weight of an empty summary: the wrapper and its header.

    A unit at or below this cannot be made smaller by summarising it;
    replacing it would GROW history (measured: a 1-token turn becomes a
    ~39-token wrapper) while the freed-token clamp hid that growth. Such units
    are skipped, or joined with their neighbours, before any call.
    """
    header = carrier_header(messages=1, kind="summary", pointer="")
    return estimate_tokens(_wrap_compaction_summary(anchor_key, header), rc)


#: Called with every request the summariser is about to make, before it is
#: made. The turn's own provider calls are recorded this way; a compaction
#: rewrites the transcript every later request is built from, so leaving its
#: calls unrecorded makes a recording unreplayable from the first compaction on.
RequestRecorder = Callable[[LLMRequest], Awaitable[Any]]


@dataclass(frozen=True, slots=True)
class _SummaryOutcome:
    """What one summariser call produced.

    ``replacement`` is ``None`` for every way a call can fail to earn its
    keep — the provider raised or timed out, the reply carried no usable text,
    or the summary came back no smaller than what it would replace. The caller
    commits nothing in that case and the original messages stay as they are
    (the floor may still remove them), and ``repeatable_failure`` says whether
    trying again could plausibly differ.
    """

    anchor_key: str
    replacement: Message | None
    tokens_freed: int
    repeatable_failure: bool = False
    """The call failed for a reason that belongs to THIS UNIT, so repeating it
    produces the same failure: the request does not fit the summariser's own
    window, or the reply carried no usable text at all.

    Only this is counted against the unit. A transport failure — a rate limit,
    a 5xx, a socket reset, a deadline — says nothing about the unit and
    everything about the moment, and counting it would let one blip across a
    parallel batch retire several units permanently.
    """
    failure: str = ""
    """Why there is no replacement: ``transport``, ``timeout``, ``empty``,
    ``too_large`` or ``not_smaller``; empty on success."""
    recovered: str = ""
    """On success, how the text was recovered when the reply was not plain
    headed text (see :class:`~protocore.runtime.context.carrier.Carrier`)."""


def _is_plain_operator_turn(message: Message) -> bool:
    """Is this a turn the operator wrote?

    A user-role message that is not a summary, not the ledger, not a frozen
    reference block, not a seeded turn from an earlier run of the session, and
    carries no tool result. What is left is what a person typed: the task, a
    steer, a correction. Compaction treats those as the one kind of message it
    may not paraphrase, because an instruction is short enough that a summary
    of it frees nothing and specific enough that a paraphrase changes it.
    """
    if message.role is not MessageRole.user or _is_compaction_summary(message):
        return False
    if is_ledger(message):
        return False
    if message.metadata.get(SYNTHETIC_RECOVERY_METADATA_KEY):
        return False
    if message.metadata.get(COMPACTION_REFERENCE_METADATA_KEY) is True:
        return False
    if message.metadata.get(SESSION_HISTORY_SEED_METADATA_KEY) is True:
        return False
    return not any(isinstance(block, ToolResultBlock) for block in message.content_blocks)


def _operator_turn_indices(history: list[Message]) -> tuple[int, ...]:
    """Positions of every operator turn, oldest first."""
    return tuple(idx for idx, message in enumerate(history) if _is_plain_operator_turn(message))


def _is_compaction_artefact(message: Message) -> bool:
    """A summary or the ledger: text compaction wrote, never read back as source."""
    return _is_compaction_summary(message) or is_ledger(message)


def summariser_input_cap(rc: LoopConstants) -> int:
    """The most one summariser call is shown, in tokens.

    Bounded twice: by the configured ceiling, and by a quarter of the window,
    so the summariser — which by default is the same model — can run even
    right after that model refused a request as too long.
    """
    return max(512, min(rc.compaction_summariser_input_max_tokens, rc.model_context_window // 4))


def _clip_middle(text: str, max_tokens: int, rc: LoopConstants) -> str:
    """``text`` within ``max_tokens``: whole head and tail lines around an omission marker.

    Cuts only at line boundaries, so no value is split; a text that is one
    enormous line is cut at a word boundary instead.
    """
    if estimate_tokens(text, rc) <= max_tokens:
        return text
    lines = text.split("\n")
    if len(lines) == 1:
        words = text.split(" ")
        head: list[str] = []
        for word in words:
            if estimate_tokens(" ".join([*head, word]), rc) > max(1, max_tokens - 8):
                break
            head.append(word)
        return " ".join(head) + " … [cut]"
    head_lines: list[str] = []
    tail_lines: list[str] = []
    spent = estimate_tokens("… [000000 lines omitted] …", rc)
    lo, hi = 0, len(lines) - 1
    take_head = True
    head_open = tail_open = True
    # Alternate ends; an end whose next line does not fit is closed and the
    # other keeps going, so one enormous line does not cost the whole tail.
    while lo <= hi and (head_open or tail_open):
        if not (head_open if take_head else tail_open):
            take_head = not take_head
            continue
        line = lines[lo] if take_head else lines[hi]
        cost = estimate_tokens(line, rc) + 1
        if spent + cost > max_tokens:
            if take_head:
                head_open = False
            else:
                tail_open = False
            take_head = not take_head
            continue
        spent += cost
        if take_head:
            head_lines.append(line)
            lo += 1
        else:
            tail_lines.append(line)
            hi -= 1
        take_head = not take_head
    omitted = hi - lo + 1
    if omitted <= 0:
        return text
    return "\n".join([*head_lines, f"… [{omitted} lines omitted] …", *reversed(tail_lines)])


def render_span_for_summary(
    messages: Sequence[Message],
    rc: LoopConstants,
    *,
    tool_names: dict[str, str],
    originals: dict[str, str],
    max_tokens: int,
) -> str:
    """The span as the summariser reads it, within ``max_tokens``.

    One line of role and tool per block. A masked tool output is shown as the
    original it stands for (``originals``, read back from the blob store), so
    the summary is of what the tool said, not of a placeholder. When the span
    is larger than the cap, tool outputs are cut first — head and tail lines,
    progressively shorter — then everything else, and never mid-line.
    """
    entries: list[tuple[str, str, bool]] = []
    for message in messages:
        for block in message.content_blocks:
            if isinstance(block, TextBlock) and block.text.strip():
                entries.append((f"[{message.role.value}]", block.text.strip(), False))
            elif isinstance(block, ToolUseBlock):
                entries.append((f"[{message.role.value} calls {block.name}]", block.arguments_json or "{}", False))
            elif isinstance(block, ToolResultBlock):
                name = tool_names.get(block.tool_call_id, "")
                body = originals.get(block.tool_call_id) or block.canonical_content or block.content
                label = f"[output of {name or 'tool'}{', error' if block.is_error else ''}]"
                entries.append((label, body, True))
            elif isinstance(block, ImageRefBlock):
                entries.append((f"[{message.role.value}]", f"[image {block.mime_type}]", False))
    if not entries:
        return ""

    def render(result_cap: int, other_cap: int) -> str:
        return "\n".join(
            f"{label} {_clip_middle(body, result_cap if is_result else other_cap, rc)}"
            for label, body, is_result in entries
        )

    result_cap = other_cap = max_tokens
    text = render(result_cap, other_cap)
    while estimate_tokens(text, rc) > max_tokens and result_cap > 64:
        result_cap //= 2
        text = render(result_cap, other_cap)
    while estimate_tokens(text, rc) > max_tokens and other_cap > 64:
        other_cap //= 2
        text = render(result_cap, other_cap)
    return _clip_middle(text, max_tokens, rc)


async def _originals_for(
    messages: Sequence[Message],
    blob_store: IBlobStore | None,
    tenant_id: str,
) -> dict[str, str]:
    """The stored originals of the masked tool outputs in ``messages``, by call id."""
    if blob_store is None:
        return {}
    found: dict[str, str] = {}
    for message in messages:
        for block in message.content_blocks:
            if not isinstance(block, ToolResultBlock) or not _content_is_already_compacted(block.content):
                continue
            ref = block.canonical_ref or block.metadata.get("blob_ref")
            if not isinstance(ref, str) or not ref:
                continue
            try:
                found[block.tool_call_id] = (await blob_store.get(tenant_id, ref)).decode("utf-8", errors="replace")
            except Exception as exc:  # a lost original leaves the placeholder, not a failed pass
                _logger.warning("compaction could not read back %s: %s", ref, exc)
    return found


async def _store_originals(
    messages: Sequence[Message],
    blob_store: IBlobStore | None,
    tenant_id: str,
    *,
    label: str,
) -> str:
    """Keep the messages a summary replaces, and return where; ``""`` if they could not be kept.

    Best effort by design: the durable copy is what makes a summary
    reversible, but a store that is down must not stop the history from
    shrinking. The host's own transcript is the other copy.
    """
    if blob_store is None or not messages:
        return ""
    # Without ``created_at``: the reference ends up in the summary, the summary
    # in every later request, and a request must be the same bytes when the
    # same history is replayed. The host's own transcript keeps the times.
    payload = "\n".join(
        message.model_dump_json(exclude={"created_at"}) for message in messages
    ).encode("utf-8")
    try:
        stored = await blob_store.put(
            tenant_id=tenant_id,
            content=payload,
            content_type="application/x-ndjson",
            metadata={"label": label, "messages": len(messages)},
        )
    except Exception as exc:  # the originals are a convenience, the shrink is the contract
        _logger.warning("compaction could not store the originals of %s: %s", label, exc)
        return ""
    return f"blob {stored.ref}"


#: The line after the fenced material. Part of the instruction for the echo
#: scrub: a summary that repeats it is repeating the summariser's own prompt.
_SPAN_REQUEST_LINE: Final[str] = "Write the summary of the transcript above under the five headings."
_FOLD_REQUEST_LINE: Final[str] = "Merge the material above into one summary under the five headings."


def _budget_words(budget_tokens: int) -> int:
    # Two tokens a word is between English prose (about 1.3) and Cyrillic
    # (three to four); the section clamp, not the model's count, is what holds
    # the budget, so the figure only has to be in the right range.
    return max(40, budget_tokens // 2)


async def _run_summariser(
    system_prompt: str,
    material: str,
    *,
    request_line: str,
    anchor_key: str,
    unit_label: str,
    before_tokens: int,
    budget_tokens: int,
    kind: str,
    messages_count: int,
    pointer: str,
    compaction_llm: ILLMProvider,
    rc: LoopConstants,
    model_name: str,
    observability: LLMObservabilityContext | None,
    record_request: RequestRecorder | None,
) -> _SummaryOutcome:
    """One summariser exchange, from built instructions to a committable replacement.

    Shared by the per-span pass and the fold. The instruction is the SYSTEM
    message and the material is fenced as data in the user message; the reply
    is requested as plain text (:meth:`ILLMProvider.complete_text`), read
    tolerantly by :func:`~protocore.runtime.context.carrier.read_carrier`, and
    held to the net-gain rule — a summary at or above the size of what it
    replaces is discarded rather than committed. The call has its own
    deadline, ``compaction_summary_timeout_seconds``.
    """
    # Local import — the shared request builder lives beside the action
    # stream, which imports this module, so the dependency is taken at call
    # time rather than at module import.
    from protocore.runtime.query import build_llm_request
    from protocore.runtime.request_budget import fit_request_to_context

    request = build_llm_request(
        model=model_name,
        messages=[
            Message(role=MessageRole.system, content_blocks=[TextBlock(text=system_prompt)]),
            Message(
                role=MessageRole.user,
                content_blocks=[TextBlock(text=f"<transcript>\n{material}\n</transcript>\n\n{request_line}")],
            ),
        ],
        tools=[],
        max_tokens=request_max_tokens(budget_tokens),
        temperature=rc.compaction_summary_temperature,
        # A summary is written, not reasoned towards: a thinking model spends
        # the output cap on its reasoning and returns the summary cut short.
        # The effort travels with the flag by the builder's rule; with
        # thinking off it bounds nothing.
        thinking_enabled=False,
        reasoning_effort="low",
        observability=observability,
    )
    try:
        request = fit_request_to_context(request, rc)
    except LLMContextWindowExceeded as exc:
        _logger.warning("summariser request exceeds the context window for %s; skipping (err=%s)", unit_label, exc)
        return _SummaryOutcome(anchor_key, None, 0, repeatable_failure=True, failure="too_large")
    if record_request is not None:
        await record_request(request)
    try:
        response = await asyncio.wait_for(
            compaction_llm.complete_text(request), timeout=rc.compaction_summary_timeout_seconds
        )
    except LLMContextWindowExceeded as exc:
        _logger.warning("summariser rejected %s as too large; skipping (err=%s)", unit_label, exc)
        return _SummaryOutcome(anchor_key, None, 0, repeatable_failure=True, failure="too_large")
    except TimeoutError:
        _logger.warning(
            "summariser gave no answer for %s within %ss; skipping",
            unit_label,
            rc.compaction_summary_timeout_seconds,
        )
        return _SummaryOutcome(anchor_key, None, 0, failure="timeout")
    except Exception as exc:  # every provider failure is a failed summary, not a failed pass
        _logger.warning("summariser failed for %s; skipping (err=%s)", unit_label, exc)
        return _SummaryOutcome(anchor_key, None, 0, failure="transport")
    carrier = read_carrier(
        response.message.text,
        budget_tokens=budget_tokens,
        rc=rc,
        truncated=response.stop_reason is StopReason.max_tokens,
        instruction=f"{system_prompt}\n{request_line}",
    )
    if not carrier.text:
        _logger.warning(
            "summariser reply for %s carried no usable text (stop=%s): %r",
            unit_label,
            response.stop_reason.value,
            response.message.text[:200],
        )
        return _SummaryOutcome(anchor_key, None, 0, repeatable_failure=True, failure="empty")
    body = f"{carrier_header(messages=messages_count, kind=kind, pointer=pointer)}\n{carrier.text}"
    wrapped = _wrap_compaction_summary(anchor_key, body, attributes=f"messages='{messages_count}' kind='{kind}'")
    after_tokens = estimate_tokens(wrapped, rc)
    if after_tokens >= before_tokens:
        _logger.warning(
            "summary for %s is not smaller than the original (%s >= %s tokens); kept the original",
            unit_label,
            after_tokens,
            before_tokens,
        )
        return _SummaryOutcome(anchor_key, None, 0, failure="not_smaller")
    # vLLM-400 fix: the summary replaces an aged turn IN THE MIDDLE of
    # history. vLLM rejects any ``system`` message past index 0 ("System
    # message must be at the beginning."), so the summary turn is USER-role,
    # and its header says in words that the user did not write it.
    return _SummaryOutcome(
        anchor_key=anchor_key,
        replacement=Message(
            role=MessageRole.user,
            content_blocks=[TextBlock(text=wrapped)],
            metadata={COMPACTION_SUMMARY_METADATA_KEY: True},
        ),
        tokens_freed=before_tokens - after_tokens,
        recovered=carrier.recovered or ("" if carrier.headed else "unheaded"),
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
    tool_names: dict[str, str],
    blob_store: IBlobStore | None,
    tenant_id: str,
) -> _SummaryOutcome:
    """Summarise ONE span — an assistant turn and the results answering it, or a run of them."""
    originals = await _originals_for(unit_messages, blob_store, tenant_id)
    material = render_span_for_summary(
        unit_messages,
        rc,
        tool_names=tool_names,
        originals=originals,
        max_tokens=summariser_input_cap(rc),
    ).strip()
    if not material:
        return _SummaryOutcome(anchor_key, None, 0, failure="empty")
    budget = summary_output_budget(before_tokens, rc, ceiling=rc.compaction_summary_max_output_tokens)
    system_prompt = prompts.render("compaction_turn_summary", {"budget_words": _budget_words(budget)})
    pointer = await _store_originals(unit_messages, blob_store, tenant_id, label=f"summary {anchor_key[:16]}")
    return await _run_summariser(
        system_prompt,
        _strip_injection_patterns(material),
        request_line=_SPAN_REQUEST_LINE,
        anchor_key=anchor_key,
        unit_label=unit_label,
        before_tokens=before_tokens,
        budget_tokens=budget,
        kind="summary",
        messages_count=len(unit_messages),
        pointer=pointer,
        compaction_llm=compaction_llm,
        rc=rc,
        model_name=model_name,
        observability=observability,
        record_request=record_request,
    )


def _prune_failed_anchor_keys(state: CompactionState, history: list[Message]) -> None:
    """Forget the units the history no longer holds.

    A key whose anchor has been folded away, dropped at a checkpoint or
    replaced describes nothing that can be summarised again, and every one of
    them is written into every later snapshot. Runs on every exit from the
    pass, including the ones that never reach a summariser call, because that
    is exactly the shape a transcript takes once the fold has consumed it.
    """
    if not state.failed_anchor_keys:
        return
    present = {
        _stable_turn_key(message)
        for message in history
        if not _is_compaction_summary(message)
    }
    state.failed_anchor_keys = {
        key: count for key, count in state.failed_anchor_keys.items() if key in present
    }


@dataclass(frozen=True, slots=True)
class _Tier2Plan:
    """What a Tier 2 pass would do, worked out before any call is made."""

    synthetic_removed: int
    synthetic_tokens_freed: int
    jobs: list[tuple[_SummarisationUnit, str, list[Message], int, bool]]


def _plan_tier2(
    history: list[Message],
    state: CompactionState,
    rc: LoopConstants,
    *,
    keep_recent_turns: int | None,
    protect_tail_from_index: int | None,
    compact_seeded_history: bool,
    retry_failed_units: bool,
) -> _Tier2Plan | None:
    """Decide which units a Tier 2 pass would send, without sending any.

    ``history`` is the pass's working copy: aged recovery nudges are removed
    from it here, and the jobs' indices refer to it afterwards. ``None`` means
    the keep window leaves nothing eligible at all.
    """
    keep = rc.compaction_keep_recent_turns if keep_recent_turns is None else keep_recent_turns
    eligible_upper = _effective_eligible_upper(history, keep, protect_tail_from_index)
    if eligible_upper == 0:
        return None

    # Aged user-role recovery nudges are runtime control flow, not conversation
    # content. Sending them to the summariser lets it misattribute the runtime's
    # instruction to the operator; protecting them instead makes them immortal.
    # They have no tool-pairing role, so remove them deterministically before
    # building Tier-2 units while the recent tail remains untouched. Every
    # nudge is appended as the trailing message of the request it steers, so
    # even the one-message reactive keep window keeps the live nudge.
    synthetic_nudges = {
        idx
        for idx in range(eligible_upper)
        if history[idx].role is MessageRole.user
        and history[idx].metadata.get(SYNTHETIC_RECOVERY_METADATA_KEY)
    }
    synthetic_tokens_freed = sum(
        estimate_message_tokens(history[idx], rc) for idx in synthetic_nudges
    )
    if synthetic_nudges:
        history[:] = [
            message for idx, message in enumerate(history) if idx not in synthetic_nudges
        ]
        eligible_upper -= len(synthetic_nudges)

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

    # Routine compaction protects prior-run seeds. Reactive compaction may
    # summarise seed-only units, but every replacement retains the seed tag so
    # the host's finalization filter still excludes it from persistence.
    seed_indices = _session_history_seed_indices(history)
    if seed_indices and not compact_seeded_history:
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
    # The ledger is carried by code and rebuilt by code; a summary of it would
    # be a model's rewrite of the one record kept out of a model's hands.
    protected = protected | frozenset(idx for idx, message in enumerate(history) if is_ledger(message))

    units = _build_summarisation_units(history, eligible_upper, protected_indices=protected)
    # Which units are worth a call at all, decided before any call is made:
    # not already summarised, and big enough — alone, or joined with the small
    # units beside it (see ``_group_small_units``) — that a summary could come
    # back smaller than what it replaces.
    candidates: list[tuple[_SummarisationUnit, str, list[Message], int, bool, bool]] = []
    for unit in units:
        anchor = history[unit.anchor_idx]
        # A4 idempotency — never re-summarise an existing compaction summary
        # (would nest <compacted-turn> wrappers and decay the summary).
        if _is_compaction_summary(anchor):
            continue
        anchor_key = _stable_turn_key(anchor)
        if anchor_key in state.summarised_turn_ids:
            continue
        if (
            not retry_failed_units
            and state.failed_anchor_keys.get(anchor_key, 0)
            >= rc.compaction_summary_failed_unit_max_attempts
        ):
            # The summariser has failed on this unit as often as it may. Paying
            # again buys the same failure; the fold tier still gets its turn.
            # A forced pass sets ``retry_failed_units`` and ignores the census:
            # it runs when the alternative is the run ending, and a call that
            # is probably wasted is cheaper than that.
            continue
        # Exhaustive across EVERY member of the unit (assistant turn + its
        # tool results), so the summary preserves the tool exchange.
        unit_messages = [history[member] for member in unit.indices]
        seeded_members = [
            member.metadata.get(SESSION_HISTORY_SEED_METADATA_KEY) is True
            for member in unit_messages
        ]
        # A single replacement cannot preserve exact persistence provenance for
        # a mixed seed/current unit. Leave it intact rather than turning either
        # side into the other.
        if any(seeded_members) and not all(seeded_members):
            continue
        seed_only = all(seeded_members)
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
        candidates.append(
            (unit, anchor_key, unit_messages, before_tokens, seed_only, before_tokens > floor)
        )

    return _Tier2Plan(
        synthetic_removed=len(synthetic_nudges),
        synthetic_tokens_freed=synthetic_tokens_freed,
        jobs=_group_small_units(history, candidates, rc),
    )


def _group_small_units(
    history: list[Message],
    candidates: list[tuple[_SummarisationUnit, str, list[Message], int, bool, bool]],
    rc: LoopConstants,
) -> list[tuple[_SummarisationUnit, str, list[Message], int, bool]]:
    """The Tier 2 jobs: adjacent units joined into spans of up to ``compaction_summary_group_max_tokens``.

    A span, not a unit, is what a summary stands for. Summarising unit by unit
    asks the same sentence or three of every small tool exchange, and a run
    that works in many short rounds — one small tool call and a short result
    each, hundreds of times — used to have no unit worth a call at all: over a
    live history of 160 units of 300 to 1,400 tokens, not one was eligible,
    every pass freed nothing and the run failed on the retry budget. Masking
    makes every old round small, so the same holds of any long run.

    So adjacent units are joined, oldest first, until the next one would pass
    the cap; a unit larger than the cap is a span of its own. Units join only
    when nothing lies between them (a summary, an operator turn, a unit left
    alone for any reason ends the span), when every unit is contiguous in
    itself, and when they share a seed provenance, so the replacement can carry
    exactly one. Tool pairing stays whole because each unit is already a closed
    pairing component. A span is sent only when it clears the floor
    (``compaction_summary_min_unit_tokens``, and the empty-summary size under
    it); a smaller one waits until later rounds grow it. The span is keyed by
    its first anchor, so the failure census and the dedup set treat it as that
    unit. With ``compaction_summary_group_max_tokens`` at 0 every unit is its
    own span.
    """
    jobs: list[tuple[_SummarisationUnit, str, list[Message], int, bool]] = []
    group: list[tuple[_SummarisationUnit, str, list[Message], int, bool, bool]] = []
    group_cap = rc.compaction_summary_group_max_tokens

    def contiguous(unit: _SummarisationUnit) -> bool:
        return unit.indices == tuple(range(unit.indices[0], unit.indices[-1] + 1))

    def flush() -> None:
        if group:
            first_unit, first_key, _messages, _before, first_seed_only, _big = group[0]
            tokens = sum(member[3] for member in group)
            floor = max(
                _compaction_wrapper_floor_tokens(first_key, rc),
                rc.compaction_summary_min_unit_tokens,
            )
            if tokens > floor:
                indices = tuple(index for member in group for index in member[0].indices)
                jobs.append(
                    (
                        _SummarisationUnit(anchor_idx=first_unit.anchor_idx, indices=indices),
                        first_key,
                        [history[index] for index in indices],
                        tokens,
                        first_seed_only,
                    )
                )
        group.clear()

    for candidate in candidates:
        unit, _anchor_key, _unit_messages, before_tokens, seed_only, _over_floor = candidate
        if group_cap <= 0 or before_tokens >= group_cap or not contiguous(unit):
            flush()
            group.append(candidate)
            flush()
            continue
        if group and (
            unit.indices[0] != group[-1][0].indices[-1] + 1
            or seed_only != group[0][4]
            or sum(member[3] for member in group) + before_tokens > group_cap
        ):
            flush()
        group.append(candidate)
    flush()
    return jobs


def tier2_has_work(
    history: list[Message],
    state: CompactionState,
    rc: LoopConstants,
    *,
    keep_recent_turns: int | None = None,
    protect_tail_from_index: int | None = None,
    compact_seeded_history: bool = False,
    retry_failed_units: bool = False,
) -> bool:
    """Whether :func:`run_tier2_summarisation` would send or drop anything."""
    if not history:
        return False
    plan = _plan_tier2(
        list(history),
        state,
        rc,
        keep_recent_turns=keep_recent_turns,
        protect_tail_from_index=protect_tail_from_index,
        compact_seeded_history=compact_seeded_history,
        retry_failed_units=retry_failed_units,
    )
    return plan is not None and (plan.synthetic_removed > 0 or bool(plan.jobs))


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
    keep_recent_turns: int | None = None,
    compact_seeded_history: bool = False,
    retry_failed_units: bool = False,
    blob_store: IBlobStore | None = None,
    tenant_id: str = "",
    ledger: Ledger | None = None,
    roles: ToolRoleMap = EMPTY_TOOL_ROLE_MAP,
) -> Tier2Result:
    """Summarise old turns via the compaction LLM.

 For each ATOMIC unit older than ``keep_recent_turns`` (or
 ``rc.compaction_keep_recent_turns`` when no override is supplied) that has not
 yet been summarised, ask the compaction LLM for a plain-text summary under the
 carrier's fixed headings (:meth:`ILLMProvider.complete_text`) and replace the
 unit's anchor turn in-place with a user-role message wrapping it. With
 ``blob_store`` the summariser is shown masked outputs in full and the
 replaced messages are stored so the summary can point at them; with
 ``ledger`` their exact values are recorded by code before they go.
 ``compact_seeded_history`` is reserved for reactive overflow recovery: it
 makes seed-only units eligible and copies the seed tag to their replacement.
 Mixed seed/current units remain intact because one replacement cannot retain
 both persistence provenances exactly.

 ``retry_failed_units`` is set by the forced passes: they ignore
 ``CompactionState.failed_anchor_keys`` and try every eligible unit, because
 they run when the alternative is the run ending. The routine gate honours the
 census so a run does not buy the same failure once an iteration.

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
 user message is tagged with :data:`COMPACTION_SUMMARY_METADATA_KEY` and
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
        state.failed_anchor_keys = {}
        return Tier2Result(turns_summarised=0, tokens_freed=0)

    original_history = history
    history = list(history)
    summarised_turn_ids = set(state.summarised_turn_ids)
    failed_anchor_keys = dict(state.failed_anchor_keys)

    plan = _plan_tier2(
        history,
        state,
        rc,
        keep_recent_turns=keep_recent_turns,
        protect_tail_from_index=protect_tail_from_index,
        compact_seeded_history=compact_seeded_history,
        retry_failed_units=retry_failed_units,
    )
    if plan is None:
        _prune_failed_anchor_keys(state, history)
        return Tier2Result(turns_summarised=0, tokens_freed=0)
    jobs = plan.jobs
    resolved_prompts = prompts if prompts is not None else bundled_prompt_provider()

    summarised = 0
    freed = plan.synthetic_tokens_freed
    # Replacement Message keyed by anchor index; indices to delete after the loop.
    replacements: dict[int, Message] = {}
    indices_to_drop: set[int] = set()
    failures: dict[str, int] = {}
    recovered: dict[str, int] = {}
    tool_names = tool_names_by_call_id(history)

    # Calls go out in small parallel batches, oldest unit first, and stop once
    # the pass has freed its budget. Sequentially this was one chain of
    # seconds-long calls with the run parked in COMPACTING for the length of
    # it; the cap is what keeps the alternative from being an unbounded
    # fan-out at the provider.
    attempted = 0
    batch_size = rc.compaction_summariser_parallelism
    for offset in range(0, len(jobs), batch_size):
        if free_target_tokens is not None and freed >= free_target_tokens:
            break
        batch = jobs[offset : offset + batch_size]
        attempted += len(batch)
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
                    tool_names=tool_names,
                    blob_store=blob_store,
                    tenant_id=tenant_id,
                )
                for unit, anchor_key, unit_messages, before_tokens, _seed_only in batch
            )
        )
        for (unit, _key, members, _before, seed_only), outcome in zip(
            batch, outcomes, strict=True
        ):
            if outcome.failure:
                failures[outcome.failure] = failures.get(outcome.failure, 0) + 1
            if outcome.recovered:
                recovered[outcome.recovered] = recovered.get(outcome.recovered, 0) + 1
            if outcome.replacement is None:
                # A failed CALL is counted against its own unit and nothing
                # else: the units beside it in the batch are unaffected, and a
                # pass that lost one of five still commits the other four. An
                # all-or-nothing pass meant one oversized unit could keep a run
                # from shedding a single token.
                if outcome.repeatable_failure:
                    failed_anchor_keys[outcome.anchor_key] = (
                        failed_anchor_keys.get(outcome.anchor_key, 0) + 1
                    )
                continue
            replacement = outcome.replacement
            if seed_only:
                replacement = replacement.model_copy(
                    update={
                        "metadata": {
                            **replacement.metadata,
                            SESSION_HISTORY_SEED_METADATA_KEY: True,
                        }
                    }
                )
            replacements[unit.anchor_idx] = replacement
            if ledger is not None:
                ledger.absorb(
                    members,
                    roles=roles,
                    is_operator=_is_plain_operator_turn,
                    skip=_is_compaction_artefact,
                    tool_names=tool_names,
                )
            # Every non-anchor member of the unit (the matching tool results) is
            # removed so the dropped ToolUseBlock leaves no orphaned tool_result.
            indices_to_drop.update(member for member in unit.indices if member != unit.anchor_idx)
            summarised_turn_ids.add(outcome.anchor_key)
            summarised += 1
            freed += outcome.tokens_freed

    if replacements or indices_to_drop:
        rebuilt: list[Message] = []
        for idx in range(len(history)):
            if idx in indices_to_drop:
                continue
            rebuilt.append(replacements.get(idx, history[idx]))
        history[:] = rebuilt

    if history != original_history:
        original_history[:] = history
    state.summarised_turn_ids = summarised_turn_ids
    state.failed_anchor_keys = failed_anchor_keys
    _prune_failed_anchor_keys(state, history)

    return Tier2Result(
        turns_summarised=summarised,
        tokens_freed=freed,
        units_attempted=attempted,
        failures=failures,
        recovered=recovered,
    )


COMPACTION_FOLD_METADATA_KEY: Final[str] = "protocore.compaction_fold"
"""On a fold summary: how many messages it stands for, and how many were the operator's."""


def _foldable_indices(
    history: list[Message],
    eligible_upper: int,
    rc: LoopConstants,
    *,
    compact_seeded_history: bool = False,
) -> frozenset[int]:
    """Positions the fold may consolidate.

    A message qualifies when it is an old compaction summary or an old operator
    turn, and is none of the things every tier protects: the first user turn
    (this run's task), one of the ``compaction_fold_keep_operator_turns`` most
    recent operator turns, a turn seeded from an earlier run of the session, or
    a frozen reference block. The seed exclusion is not cosmetic — a fold that
    absorbed a seeded turn would drop the tag that separates this run's
    messages from the previous run's, which is the one thing distinguishing
    them. ``compact_seeded_history`` (reactive recovery only) lifts it for
    plain seeded user turns; :func:`_fold_spans` then never lets a seeded and
    a current message share one span, and the fold copies the tag onto a
    seed-only replacement.
    """
    first_user = _first_user_turn_index(history) if rc.compaction_protect_first_user_turn else None
    keep = rc.compaction_fold_keep_operator_turns
    operators = _operator_turn_indices(history)
    recent_operators = frozenset(operators[-keep:]) if keep else frozenset()
    protected = recent_operators | _compaction_reference_indices(history)
    if not compact_seeded_history:
        protected = protected | _session_history_seed_indices(history)

    def is_foldable(message: Message) -> bool:
        if _is_compaction_summary(message) or _is_plain_operator_turn(message):
            return True
        return (
            compact_seeded_history
            and message.role is MessageRole.user
            and message.metadata.get(SESSION_HISTORY_SEED_METADATA_KEY) is True
            and message.metadata.get(SYNTHETIC_RECOVERY_METADATA_KEY) is not True
            and not any(
                isinstance(block, ToolResultBlock) for block in message.content_blocks
            )
        )

    return frozenset(
        idx
        for idx in range(min(eligible_upper, len(history)))
        if idx != first_user
        and idx not in protected
        and is_foldable(history[idx])
    )


def _fold_spans(
    history: list[Message],
    eligible_upper: int,
    rc: LoopConstants,
    *,
    compact_seeded_history: bool = False,
) -> list[tuple[int, int]]:
    """Contiguous runs ``[start, end)`` of foldable messages, long and heavy enough to be worth a call.

    A run must be at least ``compaction_fold_min_messages`` long and
    ``compaction_fold_min_tokens`` big. Both bounds exist so a span that has
    already been folded is not folded again for nothing: one fold summary
    standing alone is neither long enough nor heavy enough to qualify.

    With ``compact_seeded_history`` a run is additionally split where seed
    provenance changes, so every span is either all seeded or all current and
    its replacement can carry exactly one provenance.
    """
    foldable = _foldable_indices(
        history,
        eligible_upper,
        rc,
        compact_seeded_history=compact_seeded_history,
    )
    spans: list[tuple[int, int]] = []
    start: int | None = None
    span_is_seeded: bool | None = None
    for idx in range(len(history) + 1):
        if idx in foldable:
            is_seeded = (
                history[idx].metadata.get(SESSION_HISTORY_SEED_METADATA_KEY) is True
            )
            if start is None:
                start = idx
                span_is_seeded = is_seeded
            elif is_seeded != span_is_seeded:
                if idx - start >= rc.compaction_fold_min_messages:
                    tokens = sum(
                        estimate_message_tokens(history[j], rc) for j in range(start, idx)
                    )
                    if tokens >= rc.compaction_fold_min_tokens:
                        spans.append((start, idx))
                start = idx
                span_is_seeded = is_seeded
            continue
        if start is not None:
            if idx - start >= rc.compaction_fold_min_messages:
                tokens = sum(
                    estimate_message_tokens(history[j], rc) for j in range(start, idx)
                )
                if tokens >= rc.compaction_fold_min_tokens:
                    spans.append((start, idx))
            start = None
            span_is_seeded = None
    return spans


def _fold_item_text(message: Message) -> str:
    """One span member as the fold prompt shows it.

    The kinds are labelled apart: an earlier summary may be condensed further,
    an operator's message is the operator's (its wording is carried verbatim
    by the ledger, so the merge only has to keep its meaning in the account).
    """
    text = message.text.strip()
    if _is_compaction_summary(message):
        inner = re.sub(r"^<compacted-turn[^>]*>", "", text).removesuffix("</compacted-turn>").strip()
        # The header line is the runtime's, not the summary's; folding it would
        # nest "compacted context" notes inside each other.
        if inner.startswith("[Compacted context:"):
            inner = inner.split("\n", 1)[1] if "\n" in inner else ""
        return f"[earlier summary]\n{inner.strip()}"
    if message.metadata.get(SESSION_HISTORY_SEED_METADATA_KEY) is True:
        return f"[earlier session turn]\n{text}"
    return f"[operator said]\n{text}"


def _fold_anchor_key(members: Sequence[Message]) -> str:
    """A durable id for the span, derived from its members.

    Content-addressed like every other summary key, so the same span folded
    after a snapshot and resume produces the same id and the dedup set
    recognises it.
    """
    digest = hashlib.sha256("|".join(_stable_turn_key(m) for m in members).encode("utf-8")).hexdigest()
    return f"fold-{digest[:16]}"


def _bounded_span(history: list[Message], start: int, end: int, rc: LoopConstants) -> int:
    """The end of the longest prefix of ``[start, end)`` the summariser may be shown whole.

    A fold's input is its members' text, and a long run of summaries can
    outgrow the summariser's input cap; the rest of the run waits for a later
    pass rather than being cut mid-item.
    """
    cap = summariser_input_cap(rc)
    spent = 0
    for idx in range(start, end):
        cost = estimate_tokens(_fold_item_text(history[idx]), rc) + 2
        if spent + cost > cap and idx - start >= 2:
            return idx
        spent += cost
    return end


async def _fold_span(
    members: list[Message],
    *,
    compaction_llm: ILLMProvider,
    rc: LoopConstants,
    prompts: IPromptTemplateProvider,
    model_name: str,
    observability: LLMObservabilityContext | None,
    record_request: RequestRecorder | None,
    seed_only: bool,
    blob_store: IBlobStore | None = None,
    tenant_id: str = "",
) -> _SummaryOutcome:
    """Fold ONE run of old summaries and operator turns into a single summary."""
    anchor_key = _fold_anchor_key(members)
    before_tokens = sum(estimate_message_tokens(member, rc) for member in members)
    operator_count = sum(1 for member in members if _is_plain_operator_turn(member))
    material = "\n\n".join(_strip_injection_patterns(_fold_item_text(m)) for m in members)
    material = _clip_middle(material, summariser_input_cap(rc), rc)
    budget = summary_output_budget(before_tokens, rc, ceiling=rc.compaction_fold_max_output_tokens)
    system_prompt = prompts.render(
        "compaction_fold_summary",
        {
            "budget_words": _budget_words(budget),
            "item_count": len(members),
            "operator_count": operator_count,
        },
    )
    pointer = await _store_originals(members, blob_store, tenant_id, label=f"fold {anchor_key}")
    outcome = await _run_summariser(
        system_prompt,
        material,
        request_line=_FOLD_REQUEST_LINE,
        anchor_key=anchor_key,
        unit_label=f"fold of {len(members)} messages",
        before_tokens=before_tokens,
        budget_tokens=budget,
        kind="fold",
        messages_count=len(members),
        pointer=pointer,
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
                    **({SESSION_HISTORY_SEED_METADATA_KEY: True} if seed_only else {}),
                }
            }
        ),
        tokens_freed=outcome.tokens_freed,
        recovered=outcome.recovered,
    )


def tier3_has_work(
    history: list[Message],
    rc: LoopConstants,
    *,
    protect_tail_from_index: int | None = None,
    keep_recent_turns: int | None = None,
    compact_seeded_history: bool = False,
) -> bool:
    """Whether :func:`run_tier3_fold` would find a span to fold."""
    if not history or not rc.compaction_fold_enabled or rc.compaction_fold_max_spans_per_pass < 1:
        return False
    keep = rc.compaction_keep_recent_turns if keep_recent_turns is None else keep_recent_turns
    eligible_upper = _effective_eligible_upper(history, keep, protect_tail_from_index)
    if eligible_upper == 0:
        return False
    return bool(
        _fold_spans(
            history,
            eligible_upper,
            rc,
            compact_seeded_history=compact_seeded_history,
        )
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
    keep_recent_turns: int | None = None,
    compact_seeded_history: bool = False,
    blob_store: IBlobStore | None = None,
    tenant_id: str = "",
    ledger: Ledger | None = None,
    roles: ToolRoleMap = EMPTY_TOOL_ROLE_MAP,
) -> Tier3Result:
    """Fold runs of old summaries and old operator turns into one summary each.

    Tier 2 leaves one summary per span and never touches a summary again, and
    it keeps every operator turn verbatim. Over a long session those two
    become the window: a run measured here reached 184 summaries and 35
    operator turns in 344 messages, and neither tier below this one could take
    a byte off it. This pass replaces each contiguous run of such messages in
    the eligible region (see :func:`_fold_spans`) with a single merged summary.
    The operator's turns are copied into ``ledger`` verbatim, by code, before
    the merge replaces them, so their wording does not depend on the model. The
    result is a summary like any other — same wrapper, same metadata flag — so
    a later fold absorbs it once its neighbourhood has grown again.

    Bounded on purpose: at most ``rc.compaction_fold_max_spans_per_pass`` runs
    are folded, each no larger than the summariser's input cap, in batches of
    ``rc.compaction_summariser_parallelism``.

    ``keep_recent_turns`` overrides ``rc.compaction_keep_recent_turns`` and
    ``compact_seeded_history`` admits seed-only spans; both are set only by
    reactive recovery after a provider rejection. A seed-only span's
    replacement keeps the seed tag.

    Mutates ``history`` in place.
    """
    if not history or not rc.compaction_fold_enabled:
        return Tier3Result(spans_folded=0, messages_folded=0, tokens_freed=0)
    keep = rc.compaction_keep_recent_turns if keep_recent_turns is None else keep_recent_turns
    eligible_upper = _effective_eligible_upper(history, keep, protect_tail_from_index)
    if eligible_upper == 0:
        return Tier3Result(spans_folded=0, messages_folded=0, tokens_freed=0)
    spans = [
        (start, _bounded_span(history, start, end, rc))
        for start, end in _fold_spans(
            history,
            eligible_upper,
            rc,
            compact_seeded_history=compact_seeded_history,
        )[: rc.compaction_fold_max_spans_per_pass]
    ]
    if not spans:
        return Tier3Result(spans_folded=0, messages_folded=0, tokens_freed=0)

    resolved_prompts = prompts if prompts is not None else bundled_prompt_provider()
    replacements: dict[int, Message] = {}
    drop: set[int] = set()
    folded_spans = 0
    folded_messages = 0
    freed = 0
    failures: dict[str, int] = {}

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
                    seed_only=all(
                        message.metadata.get(SESSION_HISTORY_SEED_METADATA_KEY) is True
                        for message in history[start:end]
                    ),
                    blob_store=blob_store,
                    tenant_id=tenant_id,
                )
                for start, end in batch
            )
        )
        for (start, end), outcome in zip(batch, outcomes, strict=True):
            if outcome.failure:
                failures[outcome.failure] = failures.get(outcome.failure, 0) + 1
            if outcome.replacement is None:
                continue
            if ledger is not None:
                ledger.absorb(
                    history[start:end],
                    roles=roles,
                    is_operator=_is_plain_operator_turn,
                    skip=_is_compaction_artefact,
                )
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
    return Tier3Result(
        spans_folded=folded_spans,
        messages_folded=folded_messages,
        tokens_freed=freed,
        spans_attempted=len(spans),
        failures=failures,
    )


COMPACTION_FLOOR_METADATA_KEY: Final[str] = "protocore.compaction_floor"
"""On a floor digest: how many messages it stands in for."""


def _floor_units(
    history: list[Message],
    rc: LoopConstants,
    *,
    keep_recent_turns: int | None,
    protect_tail_from_index: int | None,
    compact_seeded_history: bool,
) -> list[tuple[int, ...]]:
    """The spans the floor may remove, oldest first, as tuples of history positions.

    Everything outside the protected set: the kept tail and the in-flight
    batch, the first user turn (this run's task), the ledger, frozen reference
    blocks, and — outside reactive recovery — turns seeded from an earlier run.
    Earlier summaries, folds and operator turns are all removable here: an
    operator turn's words are in the ledger before it goes, and a summary is
    already the lossy copy. Tool pairing stays whole because a span is a
    closed pairing component, exactly as for Tier 2.
    """
    keep = rc.compaction_keep_recent_turns if keep_recent_turns is None else keep_recent_turns
    eligible_upper = _effective_eligible_upper(history, keep, protect_tail_from_index)
    if eligible_upper == 0:
        return []
    protected: set[int] = set(_compaction_reference_indices(history))
    protected.update(idx for idx, message in enumerate(history) if is_ledger(message))
    first_user = _first_user_turn_index(history)
    if first_user is not None:
        protected.add(first_user)
    if not compact_seeded_history:
        protected.update(_session_history_seed_indices(history))
    units = [
        unit.indices
        for unit in _build_summarisation_units(history, eligible_upper, protected_indices=frozenset(protected))
    ]
    for idx in range(eligible_upper):
        if idx not in protected and _is_compaction_summary(history[idx]):
            units.append((idx,))
    if compact_seeded_history:
        # A span mixing an earlier run's turns with this run's cannot carry one
        # provenance in its digest; it stays.
        units = [
            unit
            for unit in units
            if len({history[i].metadata.get(SESSION_HISTORY_SEED_METADATA_KEY) is True for i in unit}) == 1
        ]
    # Raw spans go before summaries, each oldest first. A summary is the
    # cheapest carrier of what it stands for, and the span a model just
    # summarised in this same pass would otherwise be the first thing removed;
    # a raw span's exact values are in the ledger before it goes.
    units.sort(key=lambda unit: (_is_compaction_summary(history[unit[0]]), unit[0]))
    return units


def floor_has_work(
    history: list[Message],
    rc: LoopConstants,
    *,
    keep_recent_turns: int | None = None,
    protect_tail_from_index: int | None = None,
    compact_seeded_history: bool = False,
) -> bool:
    """Whether :func:`run_floor` has anything it could remove."""
    return bool(
        _floor_units(
            history,
            rc,
            keep_recent_turns=keep_recent_turns,
            protect_tail_from_index=protect_tail_from_index,
            compact_seeded_history=compact_seeded_history,
        )
    )


def _floor_digest(members: Sequence[Message], *, pointer: str, rc: LoopConstants) -> str:
    """What the floor leaves in place of the spans it removed, written by code."""
    counts: dict[str, int] = {}
    last_note = ""
    for message in members:
        for block in message.content_blocks:
            if isinstance(block, ToolUseBlock):
                counts[block.name] = counts.get(block.name, 0) + 1
            elif (
                isinstance(block, TextBlock)
                and message.role is MessageRole.assistant
                and block.text.strip()
            ):
                last_note = block.text.strip()
    lines = [carrier_header(messages=len(members), kind="floor", pointer=pointer)]
    # Nothing here may depend on when the messages were written: a digest is
    # part of every later request, and a request must be the same bytes when
    # the same history is replayed.
    lines.append(
        "These messages were removed without a summary because the context had to shrink and "
        "no summary was available. Their exact values, files and operator instructions are in "
        "the compaction ledger."
    )
    if counts:
        lines.append("Tool calls: " + ", ".join(f"{name} x{count}" for name, count in sorted(counts.items(), key=lambda kv: -kv[1])))
    if last_note:
        lines.append("Last note in the removed span: " + _clip_middle(last_note, 120, rc).replace("\n", " "))
    return "\n".join(lines)


async def run_floor(
    history: list[Message],
    rc: LoopConstants,
    *,
    free_target_tokens: int,
    keep_recent_turns: int | None = None,
    protect_tail_from_index: int | None = None,
    compact_seeded_history: bool = False,
    ledger: Ledger | None = None,
    roles: ToolRoleMap = EMPTY_TOOL_ROLE_MAP,
    blob_store: IBlobStore | None = None,
    tenant_id: str = "",
) -> FloorResult:
    """Remove the oldest spans, without a model, until ``free_target_tokens`` are freed.

    The last tier and the one that cannot fail: it runs when the tiers above
    it — masking, summarising, folding — left the prompt above the trigger,
    whatever the reason (the summariser is down, timed out, returned nothing,
    or the history is made of pieces too small to be worth a call). Each span
    is recorded in ``ledger`` and stored in ``blob_store`` before it goes, and
    ONE digest written by code stands where the removed spans were (one per
    provenance when reactive recovery removes an earlier run's turns too).

    ``reached`` is set when there was nothing left to remove before the target
    was met: the history is then at its floor — the task, the ledger and the
    kept tail — and no pass can take it lower.
    """
    units = _floor_units(
        history,
        rc,
        keep_recent_turns=keep_recent_turns,
        protect_tail_from_index=protect_tail_from_index,
        compact_seeded_history=compact_seeded_history,
    )
    if free_target_tokens <= 0:
        return FloorResult(messages_dropped=0, tokens_freed=0, reached=False)
    chosen: list[tuple[int, ...]] = []
    freed = 0
    for unit in units:
        if freed >= free_target_tokens:
            break
        chosen.append(unit)
        freed += sum(estimate_message_tokens(history[i], rc) for i in unit)
    reached = freed < free_target_tokens
    if not chosen:
        return FloorResult(messages_dropped=0, tokens_freed=0, reached=True)
    dropped = sorted(i for unit in chosen for i in unit)
    tool_names = tool_names_by_call_id(history)
    groups: dict[bool, list[int]] = {}
    for idx in dropped:
        groups.setdefault(history[idx].metadata.get(SESSION_HISTORY_SEED_METADATA_KEY) is True, []).append(idx)
    replacements: dict[int, Message] = {}
    marker_tokens = 0
    for seeded, indices in groups.items():
        members = [history[i] for i in indices]
        if ledger is not None:
            ledger.absorb(
                members,
                roles=roles,
                is_operator=_is_plain_operator_turn,
                skip=_is_compaction_artefact,
                tool_names=tool_names,
            )
        pointer = await _store_originals(members, blob_store, tenant_id, label="floor")
        text = _wrap_compaction_summary(
            f"floor-{_fold_anchor_key(members)[5:]}",
            _floor_digest(members, pointer=pointer, rc=rc),
            attributes=f"messages='{len(members)}' kind='floor'",
        )
        metadata: dict[str, Any] = {
            COMPACTION_SUMMARY_METADATA_KEY: True,
            COMPACTION_FLOOR_METADATA_KEY: {"messages": len(members)},
        }
        if seeded:
            metadata[SESSION_HISTORY_SEED_METADATA_KEY] = True
        marker = Message(role=MessageRole.user, content_blocks=[TextBlock(text=text)], metadata=metadata)
        replacements[indices[0]] = marker
        marker_tokens += estimate_message_tokens(marker, rc)
    drop = set(dropped)
    history[:] = [
        replacements[idx] if idx in replacements else message
        for idx, message in enumerate(history)
        if idx not in drop or idx in replacements
    ]
    _logger.warning(
        "DIAG compaction.floor dropped=%d freed=%d target=%d reached=%s",
        len(dropped),
        freed - marker_tokens,
        free_target_tokens,
        reached,
    )
    return FloorResult(messages_dropped=len(dropped), tokens_freed=max(0, freed - marker_tokens), reached=reached)


def _ledger_position(history: list[Message], protect_tail_from_index: int | None) -> int:
    """Where the ledger goes: at the boundary between the compacted past and the raw present.

    After the newest summary, or after the newest masked output when nothing
    has been summarised yet, and never between an assistant's tool call and
    the results answering it, nor after the batch the model has not yet read
    (a trailing user message there would read as a new request).
    """
    limit = len(history) if protect_tail_from_index is None else min(len(history), protect_tail_from_index)
    position: int | None = None
    for idx in range(len(history) - 1, -1, -1):
        if _is_compaction_summary(history[idx]):
            position = idx + 1
            break
    if position is None:
        for idx in range(len(history) - 1, -1, -1):
            message = history[idx]
            if message.role is MessageRole.tool and any(
                isinstance(block, ToolResultBlock) and _content_is_already_compacted(block.content)
                for block in message.content_blocks
            ):
                position = idx + 1
                break
    if position is None:
        first_user = _first_user_turn_index(history)
        position = 0 if first_user is None else first_user + 1
    position = min(position, limit)
    while position < limit and history[position].role is MessageRole.tool:
        position += 1
    if position >= limit and limit < len(history):
        position = limit
    while 0 < position < len(history) and history[position].role is MessageRole.tool:
        position -= 1
    return position


def place_ledger(
    history: list[Message],
    ledger: Ledger,
    rc: LoopConstants,
    *,
    protect_tail_from_index: int | None = None,
) -> int:
    """Put the rendered ledger back into ``history``; return its size in tokens.

    Every earlier ledger message is removed and one is inserted, rebuilt from
    ``ledger``. When the result is the same text at the same place, the
    existing message object is kept, so a pass that recorded nothing new does
    not look like a rewrite to anything comparing the history by identity.
    """
    existing = [idx for idx, message in enumerate(history) if is_ledger(message)]
    fresh = ledger_message(ledger, rc)
    if fresh is None:
        if existing:
            history[:] = [message for message in history if not is_ledger(message)]
        return 0
    if len(existing) == 1:
        current = history[existing[0]]
        without = history[: existing[0]] + history[existing[0] + 1 :]
        position = _ledger_position(without, protect_tail_from_index)
        if position == existing[0] and current.text == fresh.text:
            return estimate_message_tokens(current, rc)
    if existing:
        history[:] = [message for message in history if not is_ledger(message)]
        if protect_tail_from_index is not None:
            protect_tail_from_index -= sum(1 for idx in existing if idx < protect_tail_from_index)
    history.insert(_ledger_position(history, protect_tail_from_index), fresh)
    return estimate_message_tokens(fresh, rc)


def compaction_event_payload(attempt: CompactionAttempt, *, reason: str) -> dict[str, Any]:
    """The ``compaction_completed`` payload: what each tier did and how the pass ended.

    The historical keys (``tokens_before``, ``tier2_summarised`` …) keep their
    meaning; the rest say per tier what was tried and what failed, so a pass
    that ended at the floor because the summariser was down reads as that, and
    not as a pass that merely freed fewer tokens than usual.
    """
    tier1, tier2, tier3, floor = attempt.tier1, attempt.tier2, attempt.tier3, attempt.floor
    return {
        "reason": reason,
        "outcome": attempt.outcome,
        "tokens_before": attempt.tokens_before,
        "tokens_after": attempt.tokens_after,
        "prompt_before": attempt.prompt_before,
        "prompt_after": attempt.prompt_after,
        "trigger_threshold": attempt.trigger_tokens,
        "target_tokens": attempt.target_tokens,
        "tier1_freed": tier1.tokens_freed if tier1 else 0,
        "tier1_masked_by_age": tier1.masked_by_age if tier1 else 0,
        "tier2_summarised": tier2.turns_summarised if tier2 else 0,
        "tier2_attempted": tier2.units_attempted if tier2 else 0,
        "tier2_failures": dict(tier2.failures) if tier2 else {},
        "tier2_recovered": dict(tier2.recovered) if tier2 else {},
        "tier3_folded": tier3.messages_folded if tier3 else 0,
        "tier3_failures": dict(tier3.failures) if tier3 else {},
        "floor_dropped": floor.messages_dropped if floor else 0,
        "floor_reached": bool(floor and floor.reached),
        "ledger_tokens": attempt.ledger_tokens,
        "blob_refs_created": list(tier1.blob_refs_created) if tier1 else [],
    }


__all__ = [
    "COMPACTION_FLOOR_METADATA_KEY",
    "COMPACTION_FOLD_METADATA_KEY",
    "CompactionAttempt",
    "CompactionExhaustedError",
    "CompactionState",
    "FloorResult",
    "Tier1Result",
    "Tier2Result",
    "Tier3Result",
    "TokenEstimator",
    "compaction_event_payload",
    "floor_has_work",
    "place_ledger",
    "render_span_for_summary",
    "run_floor",
    "run_tier1_truncation",
    "run_tier2_summarisation",
    "run_tier3_fold",
    "summariser_input_cap",
    "tier1_has_work",
    "tier2_has_work",
    "tier3_has_work",
]
