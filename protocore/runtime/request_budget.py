"""Hard context-window budgeting for fully assembled LLM requests.

Two ways to size a request. The estimate is local and free, and it can run
several times short of a real tokenizer on dense text. A provider that
implements :class:`~protocore.contracts.llm.IRequestTokenCounter` can say what
the request actually renders to; it is asked only when the estimate is close
enough to a limit for the difference to decide something, and what it says is
kept per request content so a retry of the same request is not counted twice.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, cast

from protocore.contracts.llm import LLMContextWindowExceeded, LLMRequest
from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.types import Message
from protocore.logging_utils import get_logger
from protocore.runtime.context.compaction import estimate_history_tokens_uncalibrated
from protocore.runtime.tool_surface import read_tool_surface, tool_surface_tokens

_logger = get_logger(__name__)

#: What a request's size does not depend on. The output cap and the sampling
#: temperature do not change the rendered prompt, and the observability
#: context never reaches the wire; leaving them out lets a refit of the same
#: prompt with a smaller cap reuse the count it already has.
_SIZE_INDEPENDENT_FIELDS: frozenset[str] = frozenset(
    {"max_tokens", "temperature", "observability"}
)

#: Request options that move from one iteration to the next without changing
#: what the prompt renders to. Prompt-cache breakpoints mark where a provider
#: may cache, and are re-placed on every request as the history grows; the
#: forced tool choice constrains decoding, not the template. Left in the digest,
#: either would make the tool definitions look new on every iteration.
_RENDER_NEUTRAL_EXTRA_KEYS: frozenset[str] = frozenset(
    {"cache_breakpoints", "forced_tool_choice", "tool_choice_required"}
)

RequestTokenCount = Callable[[LLMRequest], Awaitable[int | None]]


def estimate_request_prompt_tokens_uncalibrated(
    request: LLMRequest,
    rc: LoopConstants,
) -> int:
    """Estimate the complete request before adaptive calibration."""
    surface = read_tool_surface(request.tools)
    raw_tokens = estimate_history_tokens_uncalibrated(list(request.messages), rc)
    raw_tokens += tool_surface_tokens(surface, rc)
    return raw_tokens


def fit_max_tokens(
    *,
    prompt_tokens: int,
    requested_max_tokens: int,
    context_window: int,
    safety_tokens: int = 0,
) -> int:
    """Return the largest output cap after a provider-framing safety margin."""
    usable_window = context_window - safety_tokens
    if prompt_tokens >= usable_window:
        raise LLMContextWindowExceeded(
            "estimated prompt fills the usable model context window "
            f"({prompt_tokens} >= {usable_window}; context={context_window}, "
            f"safety={safety_tokens})"
        )
    return min(requested_max_tokens, usable_window - prompt_tokens)


def estimate_request_prompt_tokens(
    request: LLMRequest,
    rc: LoopConstants,
) -> int:
    """Estimate the current complete prompt with adaptive calibration."""
    return round(estimate_request_prompt_tokens_uncalibrated(request, rc) * rc.token_estimate_calibration)


def fit_request_to_context(
    request: LLMRequest,
    rc: LoopConstants,
) -> LLMRequest:
    """Clip one built request so its prompt and output fit the hard window."""
    prompt_tokens = estimate_request_prompt_tokens(request, rc)
    return _fit_to_prompt_tokens(request, rc, prompt_tokens)


def _fit_to_prompt_tokens(
    request: LLMRequest,
    rc: LoopConstants,
    prompt_tokens: int,
) -> LLMRequest:
    max_tokens = fit_max_tokens(
        prompt_tokens=prompt_tokens,
        requested_max_tokens=request.max_tokens,
        context_window=rc.model_context_window,
        safety_tokens=rc.request_context_safety_tokens,
    )
    if max_tokens == request.max_tokens:
        return request
    return request.model_copy(update={"max_tokens": max_tokens})


def request_token_counter(provider: object) -> RequestTokenCount | None:
    """The provider's exact request counter, or ``None`` when it has none.

    Looked up on the provider's CLASS rather than on the instance. A test double
    built on a mock answers every attribute an instance is asked for, and the
    capability must not appear on a provider because a mock invented it.
    """
    if getattr(type(provider), "count_request_tokens", None) is None:
        return None
    return cast(RequestTokenCount, provider.count_request_tokens)  # type: ignore[attr-defined]


def near_limit(estimate: int, limit: int, rc: LoopConstants) -> bool:
    """Whether ``estimate`` is close enough to ``limit`` to be worth counting."""
    return estimate >= limit * (1.0 - rc.exact_token_count_margin_ratio)


@dataclass(frozen=True, slots=True)
class CountAnchor:
    """The last full request the provider counted, as the next fit reads it."""

    model: str
    #: What the provider said that request rendered to.
    measured: int
    #: The content digest of every message that request carried.
    message_digests: frozenset[str]
    #: The digest of everything else that decides its size: model, tools, extra.
    frame_digest: str


class ExactTokenCountCache:
    """What one run has learned from counting: the counts, the last one, and failures.

    Counts are keyed by request content. The anchor is the last full request
    counted, which lets a later fit decide whether the content added since
    could have carried the prompt over its limit. ``backoff_until`` is the
    monotonic time before which counting is not attempted again after a count
    failed or timed out: an endpoint that hangs would otherwise cost the full
    timeout on every iteration.
    """

    __slots__ = ("_entries", "anchor", "backoff_until")

    def __init__(self) -> None:
        self._entries: OrderedDict[str, int] = OrderedDict()
        self.anchor: CountAnchor | None = None
        self.backoff_until: float = 0.0

    def get(self, key: str) -> int | None:
        count = self._entries.get(key)
        if count is not None:
            self._entries.move_to_end(key)
        return count

    def put(self, key: str, count: int, *, max_entries: int) -> None:
        self._entries[key] = count
        self._entries.move_to_end(key)
        while len(self._entries) > max_entries:
            self._entries.popitem(last=False)

    def __len__(self) -> int:
        return len(self._entries)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def message_digest(message: Message) -> str:
    """A digest of what one message renders to. Its creation time renders to nothing."""
    return _sha256(message.model_dump_json(exclude={"created_at"}))


@dataclass(frozen=True, slots=True)
class RequestDigests:
    """A request's content, digested per message and for the frame around them."""

    messages: tuple[str, ...]
    frame: str

    @property
    def key(self) -> str:
        return _sha256("\n".join((*self.messages, self.frame)))


def request_digests(request: LLMRequest) -> RequestDigests:
    """Digest ``request`` once, for the count cache and for the re-count decision."""
    exclude: dict[str, Any] = dict.fromkeys(("messages", *_SIZE_INDEPENDENT_FIELDS), True)
    exclude["extra"] = dict.fromkeys(_RENDER_NEUTRAL_EXTRA_KEYS, True)
    frame = request.model_dump_json(exclude=exclude)
    return RequestDigests(
        messages=tuple(message_digest(message) for message in request.messages),
        frame=_sha256(frame),
    )


def request_content_key(request: LLMRequest) -> str:
    """A digest of everything in ``request`` that decides its rendered size."""
    return request_digests(request).key


async def count_request_tokens_exactly(
    request: LLMRequest,
    provider: object,
    rc: LoopConstants,
    *,
    cache: ExactTokenCountCache | None = None,
    estimate: int | None = None,
    digests: RequestDigests | None = None,
) -> int | None:
    """Ask ``provider`` what ``request`` renders to; ``None`` when it cannot say.

    ``None`` covers a provider without the capability, the capability switched
    off, an endpoint that has no counting route, and a count that failed or did
    not arrive within :attr:`LoopConstants.exact_token_count_timeout_seconds`. A
    failure is logged; the others are the ordinary state of most providers and
    are not.

    A fresh count is logged beside ``estimate``, when one is given — that line
    is the drift between the heuristic and the provider. A count served from
    ``cache`` was logged when it was made and is not logged again.
    """
    if not rc.exact_token_count_enabled:
        return None
    counter = request_token_counter(provider)
    if counter is None:
        return None
    if cache is not None and time.monotonic() < cache.backoff_until:
        return None
    key: str | None
    try:
        key = (digests or request_digests(request)).key
    except (TypeError, ValueError):
        key = None
    if key is not None and cache is not None:
        cached = cache.get(key)
        if cached is not None:
            return cached
    try:
        measured = await asyncio.wait_for(
            counter(request), timeout=rc.exact_token_count_timeout_seconds
        )
    except Exception as exc:
        if cache is not None:
            cache.backoff_until = (
                time.monotonic() + rc.exact_token_count_failure_backoff_seconds
            )
        _logger.warning(
            "exact request token count failed for model=%s; using the estimate, "
            "and not counting again for %.0fs (err=%r)",
            request.model,
            rc.exact_token_count_failure_backoff_seconds,
            exc,
        )
        return None
    if measured is None:
        return None
    if isinstance(measured, bool) or not isinstance(measured, int) or measured <= 0:
        _logger.warning(
            "exact request token count for model=%s returned %r; using the estimate",
            request.model,
            measured,
        )
        return None
    if key is not None and cache is not None:
        cache.put(key, measured, max_entries=rc.exact_token_count_cache_max_entries)
    if estimate is not None:
        _logger.warning(
            "DIAG request_budget.exact_count model=%s estimate=%d measured=%d "
            "drift_ratio=%.3f",
            request.model,
            estimate,
            measured,
            measured / estimate if estimate > 0 else 0.0,
        )
    return measured


def worst_case_ratio(rc: LoopConstants) -> float:
    """The largest undercount the counting margin is sized to catch.

    A margin ``m`` catches an estimate that runs short by up to ``1 / (1 - m)``;
    the incremental trigger assumes new content is that dense, so the two rules
    protect against the same worst case.
    """
    margin = rc.exact_token_count_margin_ratio
    return math.inf if margin >= 1.0 else 1.0 / (1.0 - margin)


def unmeasured_raw_tokens(
    request: LLMRequest,
    digests: RequestDigests,
    anchor: CountAnchor,
    rc: LoopConstants,
) -> int:
    """The heuristic's size of what in ``request`` the last count did not see.

    Gross, message by message: every message whose content is not among the
    counted ones — appended, or rewritten by compaction or eviction — is new,
    whatever was removed alongside it. A net difference would let a summary that
    replaced a page of prose hide the dense tool result that arrived in the same
    breath. Removed messages are simply left out of the sum: they can only make
    the real prompt smaller than the count already says, so the bound stays an
    upper bound. A changed frame (tools, model, request options) adds the tool
    definitions back in whole.
    """
    unseen = [
        message
        for message, digest in zip(request.messages, digests.messages, strict=True)
        if digest not in anchor.message_digests
    ]
    added = estimate_history_tokens_uncalibrated(unseen, rc) if unseen else 0
    if digests.frame != anchor.frame_digest:
        added += tool_surface_tokens(read_tool_surface(request.tools), rc)
    return added


def _count_warranted(
    request: LLMRequest,
    digests: RequestDigests,
    estimate: int,
    limit: int,
    rc: LoopConstants,
    cache: ExactTokenCountCache | None,
) -> bool:
    """Whether this fit should ask the provider.

    Before the run has a count of a full request for this model, the margin rule
    decides. After it, the question is narrower: the last count is known, and
    only the content it did not see is unmeasured. If that content, sized at the
    worst undercount the margin assumes, still could not carry the prompt over
    the limit, the count would only confirm what the calibrated estimate already
    says, and it is not made. Prose-heavy turns then count about once near the
    edge instead of on every iteration, and a large block of dense content —
    the case the count exists for — crosses the bound at once, including when it
    arrives in a history that compaction has just rewritten.
    """
    anchor = cache.anchor if cache is not None else None
    if anchor is None or anchor.model != request.model:
        return True
    added = unmeasured_raw_tokens(request, digests, anchor, rc)
    if added == 0:
        return anchor.measured >= limit
    return anchor.measured + added * worst_case_ratio(rc) >= limit


@dataclass(frozen=True, slots=True)
class FittedRequest:
    """A request fitted to the window, and the sizes the fit was made from."""

    request: LLMRequest
    #: The heuristic's own count of the request, before calibration.
    raw_estimate: int
    #: The calibrated estimate.
    estimate: int
    #: What the provider said the request renders to, when it was asked.
    measured: int | None


async def fit_request_to_context_measured(
    request: LLMRequest,
    rc: LoopConstants,
    provider: object,
    *,
    cache: ExactTokenCountCache | None = None,
    on_measured: Callable[[int], None] | None = None,
) -> FittedRequest:
    """:func:`fit_request_to_context`, sized by the provider near the edge.

    The limit that matters for the fit is the prompt size at which the output
    cap starts being clipped: the usable window less the requested cap. Below
    ``limit * (1 - exact_token_count_margin_ratio)`` the estimate decides alone
    and nothing leaves the process; at or above it, a provider that can count
    the rendered request is asked, and its number replaces the estimate. The
    request itself is not changed by being counted: with no counter, or with a
    count that failed, the result is exactly what :func:`fit_request_to_context`
    returns.

    ``on_measured`` receives the count BEFORE the fit is attempted. A count
    that proves the request cannot fit is the strongest evidence the loop ever
    gets about its estimate, and the fit raises on exactly that count; handed
    over afterwards, it would be lost with the exception and the recovery that
    follows would size history with the undercount the count had just exposed.

    A refusal that rests on a count says so in its message, and deliberately
    carries no sizes: the rejection handler reads sizes as proof that a smaller
    output cap fits, and a prompt that alone fills the window is answered by
    compaction, never by a smaller cap.
    """
    estimate = estimate_request_prompt_tokens(request, rc)
    # Recovered from the calibrated figure rather than estimated a second time:
    # the two differ only by the factor, and a long history is not walked twice
    # per call for a number that is only read if the provider rejects it.
    raw = round(estimate / rc.token_estimate_calibration)
    measured: int | None = None
    clip_limit = (
        rc.model_context_window - rc.request_context_safety_tokens - request.max_tokens
    )
    if near_limit(estimate, clip_limit, rc):
        digests = request_digests(request)
        if _count_warranted(request, digests, estimate, clip_limit, rc, cache):
            measured = await count_request_tokens_exactly(
                request, provider, rc, cache=cache, estimate=estimate, digests=digests
            )
            if measured is not None and cache is not None:
                cache.anchor = CountAnchor(
                    model=request.model,
                    measured=measured,
                    message_digests=frozenset(digests.messages),
                    frame_digest=digests.frame,
                )
    if measured is None:
        fitted = _fit_to_prompt_tokens(request, rc, estimate)
    else:
        if on_measured is not None:
            on_measured(measured)
        try:
            fitted = _fit_to_prompt_tokens(request, rc, measured)
        except LLMContextWindowExceeded as exc:
            raise LLMContextWindowExceeded(
                "counted prompt fills the usable model context window "
                f"({measured} tokens; context={rc.model_context_window}, "
                f"safety={rc.request_context_safety_tokens})"
            ) from exc
    return FittedRequest(
        request=fitted,
        raw_estimate=raw,
        estimate=estimate,
        measured=measured,
    )


__all__ = [
    "CountAnchor",
    "ExactTokenCountCache",
    "FittedRequest",
    "RequestDigests",
    "RequestTokenCount",
    "count_request_tokens_exactly",
    "estimate_request_prompt_tokens",
    "estimate_request_prompt_tokens_uncalibrated",
    "fit_max_tokens",
    "fit_request_to_context",
    "fit_request_to_context_measured",
    "message_digest",
    "near_limit",
    "request_content_key",
    "request_digests",
    "request_token_counter",
    "unmeasured_raw_tokens",
    "worst_case_ratio",
]
