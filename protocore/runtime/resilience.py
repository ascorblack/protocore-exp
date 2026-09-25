"""Universal resilience runtime — classify-then-act over LLM + tool transport.

The active counterpart to :mod:`protocore.contracts.resilience`: pure
functions + a small policy object + the generic transport wrapper that
the host side binds to a concrete backend.

What lives here (all backend-agnostic, no backend/wire symbols):

* :func:`classify_transport_error` — neutral exception → error-class
 classifier. Recognises generic timeout/connection/rate-limit signatures
 and honours a classifier verdict attached by the backend
 (``getattr(exc, "classified")`` or a
 :class:`~protocore.contracts.resilience.ClassifiedLike`).
* :func:`transport_retry_after_seconds` — extract an OPTIONAL server-stated
 rate-limit reset / retry-after hint from a failure so the policy can
 honour it .
* :func:`reason_to_error_class` — collapse a rich the host verdict
 (LLM ``FailoverReason`` string, transport reason) onto the neutral
 :class:`ResilienceErrorClass`.
* :func:`decorrelated_jitter_backoff` — "decorrelated jitter" backoff
 (thread-safe via the caller's RNG); the universal backoff schedule.
* :class:`TokenBucketRetryBudget` — async helper around
 :class:`RetryBudgetState` that consumes/deposits under an injected lock
 (process-local-per-pod, N-pod safe; no module-level state).
* :func:`deadline_finalization_reserve_ok` — deadline-aware retry gate:
 refuse a retry that would eat the slice reserved for a full final answer
 (never raises the deadline; only ever gives up earlier).
* :class:`ResiliencePolicy` — combines classify → decision using the RC
 knobs. Same (error, RC, attempt) → same :class:`ResilienceDecision`.
* :func:`resilient_transport_call` — the universal decorator: drives an
 :class:`IToolTransport` with classify-then-act + budget + backoff +
 deadline reserve + the classify-don't-retry-mutations stance, and the
 rebuild-before-retry hook for the stale-connection class.

Defaults preserve current behaviour: with ``resilience_enabled=False`` the
policy returns the conservative "one retry on transient, else abort"
decisions that match the conservative call paths, and
:func:`resilient_transport_call` with budgeting off + immediate backoff is
bit-identical to a plain attempt-count retry loop.
"""
from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from protocore.contracts.resilience import (
    RETRYABLE_ERROR_CLASSES,
    IToolTransport,
    ResilienceAction,
    ResilienceDecision,
    ResilienceErrorClass,
    RetryBudgetState,
    ToolTransportError,
    ToolTransportRetryBudgetExhausted,
    ToolTransportTimeout,
    TransportCallSpec,
)

if TYPE_CHECKING:
    from protocore.contracts.runtime_constants import LoopConstants

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Neutral classification
# ---------------------------------------------------------------------------

# Generic transient transport signatures — matched on the lower-cased
# exception string so the classifier needs NO wire-format symbols. Mirrors
# the union a host transport predicate + generic timeout patterns use,
# stripped of any backend specifics.
_TRANSIENT_TIMEOUT_MARKERS: tuple[str, ...] = (
    "deadline exceeded",
    "deadline_exceeded",
    "timed out",
    "timeout",
    "temporarily unavailable",
    "unavailable",
    "read timed out",
    "body read timed out",
)

# Stale-connection signatures — a reset / aborted / broken-pipe class means
# the pooled socket is (likely) dead, so the right move is to REBUILD the
# client/connection before retrying rather than re-issue on the same socket
# (, mirroring the reference's ECONNRESET/EPIPE -> fresh-client path).
# Matched on the lower-cased message for non-``ConnectionError`` exceptions
# (the stdlib ``ConnectionError`` family is routed structurally; see
# :func:`classify_transport_error`).
_STALE_CONNECTION_MARKERS: tuple[str, ...] = (
    "connection reset",
    "connection aborted",
    "broken pipe",
    "econnreset",
    "epipe",
)

_RATE_LIMIT_MARKERS: tuple[str, ...] = (
    "rate limit",
    "rate_limit",
    "too many requests",
    "resource_exhausted",
    "throttl",
    "retry in",
    "resets at",
)

# Defensive upper bound on an HONOURED server-stated rate-limit reset, in
# seconds. : the reference honours the server reset VERBATIM (it
# deliberately bypasses the generic backoff cap — sleeping less than the
# stated window just re-issues into a still-closed limit), so the honoured
# reset is NOT clamped to ``resilience_backoff_max_seconds``; the
# deadline-reserve gate decides retry-vs-finalize. This cap is only a sanity
# guard against a garbage/hostile value (e.g. a multi-day "reset") pinning an
# in-loop sleep — it sits an order of magnitude above any legitimate
# rate-limit window and far above the finalization-reserve horizon, so it
# never shortens a real reset. 1 hour.
_MAX_HONOURED_RATE_LIMIT_RESET_SECONDS: float = 3600.0

# StrEnum-value strings the host LLM ``FailoverReason`` taxonomy uses,
# collapsed onto the neutral classes. Kept as a plain dict keyed on the
# string value so core never imports the host enum (the reason arrives
# as ``getattr(reason, "value", reason)`` — already a string).
_LLM_REASON_TO_CLASS: dict[str, ResilienceErrorClass] = {
    "rate_limit": ResilienceErrorClass.rate_limited,
    "long_context_tier": ResilienceErrorClass.context_overflow,
    "overloaded": ResilienceErrorClass.transient_retryable,
    "server_error": ResilienceErrorClass.transient_retryable,
    "timeout": ResilienceErrorClass.timeout_rebuild,
    "context_overflow": ResilienceErrorClass.context_overflow,
    "payload_too_large": ResilienceErrorClass.payload_too_large,
    "image_too_large": ResilienceErrorClass.payload_too_large,
    "auth": ResilienceErrorClass.deterministic_abort,
    "auth_permanent": ResilienceErrorClass.deterministic_abort,
    "billing": ResilienceErrorClass.deterministic_abort,
    "model_not_found": ResilienceErrorClass.deterministic_abort,
    "provider_policy_blocked": ResilienceErrorClass.deterministic_abort,
    "format_error": ResilienceErrorClass.deterministic_abort,
    "thinking_signature": ResilienceErrorClass.transient_retryable,
    "oauth_long_context_beta_forbidden": ResilienceErrorClass.deterministic_abort,
    "llama_cpp_grammar_pattern": ResilienceErrorClass.deterministic_abort,
    "unknown": ResilienceErrorClass.unknown,
}


def reason_to_error_class(reason: object) -> ResilienceErrorClass:
    """Collapse a host classifier ``reason`` onto a neutral class.

    Accepts a ``StrEnum`` (reads ``.value``) or a plain string. Unknown
    reasons map to :attr:`ResilienceErrorClass.unknown` (default-safe).
    """
    value = getattr(reason, "value", reason)
    if not isinstance(value, str):
        return ResilienceErrorClass.unknown
    return _LLM_REASON_TO_CLASS.get(value, ResilienceErrorClass.unknown)


def _classified_verdict(exc: BaseException) -> Any | None:
    """Return an attached classifier verdict, if any.

    Backends attach the verdict as the dynamic ``classified`` attribute
    (mirrors the LLM ``classified_to_exception`` pattern). Read
    structurally so core never imports the host type.
    """
    return getattr(exc, "classified", None)


def _coerce_positive_seconds(value: object) -> float | None:
    """Coerce a candidate reset/retry-after value to positive seconds.

    Returns ``None`` for missing / non-numeric / non-positive values so a
    caller can treat "no usable hint" uniformly.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        seconds = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if seconds <= 0.0:
        return None
    return seconds


def transport_retry_after_seconds(exc: BaseException) -> float | None:
    """Extract a server-stated reset / retry-after hint from an exception.

 : a backend signals when a rate-limit window reopens by setting
 :attr:`~protocore.contracts.resilience.ToolTransportError.retry_after_seconds`
 on the raised error, OR by attaching it to the classified verdict
 (``exc.classified.retry_after_seconds``). Read structurally (``getattr``)
 so core never imports the host classifier type. Returns the hint in
 seconds-from-now, or ``None`` when no usable (positive, numeric) hint is
 present. The error's own attribute wins over the verdict's when both are
 set (the raiser is closest to the wire).
 """
    direct = _coerce_positive_seconds(getattr(exc, "retry_after_seconds", None))
    if direct is not None:
        return direct
    verdict = _classified_verdict(exc)
    if verdict is not None:
        return _coerce_positive_seconds(
            getattr(verdict, "retry_after_seconds", None)
        )
    return None


def classify_transport_error(exc: BaseException) -> ResilienceErrorClass:
    """Classify a tool/VM transport exception into a neutral error class.

 Priority:

 1. An explicit neutral subclass (:class:`ToolTransportTimeout` →
 ``transient_retryable``; bare :class:`ToolTransportError` →
 ``rate_limited`` / ``timeout_rebuild`` / ``transient_retryable`` only
 when its message matches, else ``deterministic_abort``).
 2. An attached classifier verdict (``exc.classified.reason``) collapsed
 via :func:`reason_to_error_class`.
 3. Generic stdlib transient types: ``TimeoutError`` →
 ``transient_retryable``; the ``ConnectionError`` family
 (``ConnectionResetError`` = ECONNRESET, ``BrokenPipeError`` = EPIPE,
 etc.) → ``timeout_rebuild`` (rebuild the dead socket before retry,
 ).
 4. Message-pattern fallback (rate-limit markers → ``rate_limited``;
 stale-connection markers → ``timeout_rebuild``; timeout markers →
 ``transient_retryable``).
 5. Otherwise ``deterministic_abort`` — an unrecognised structural
 failure must surface immediately, NOT be masked as transient
 ("structural failures must not be masked as transient").
 """
    if isinstance(exc, ToolTransportRetryBudgetExhausted):
        # Already a give-up signal — surface as unknown so the policy
        # finalises rather than re-deciding a retry.
        return ResilienceErrorClass.unknown
    if isinstance(exc, ToolTransportTimeout):
        # A pure per-call deadline — retry on the same connection (a timeout
        # is not by itself a stale-socket signal; the message fallback below
        # still routes an explicit reset/broken-pipe message to rebuild).
        message = str(exc).lower()
        if _any_marker(message, _STALE_CONNECTION_MARKERS):
            return ResilienceErrorClass.timeout_rebuild
        return ResilienceErrorClass.transient_retryable

    verdict = _classified_verdict(exc)
    if verdict is not None:
        reason = getattr(verdict, "reason", None)
        if reason is not None:
            return reason_to_error_class(reason)

    message = str(exc).lower()
    if isinstance(exc, ToolTransportError):
        # A bare transport error: classify by message; structural otherwise.
        if _any_marker(message, _RATE_LIMIT_MARKERS):
            return ResilienceErrorClass.rate_limited
        if _any_marker(message, _STALE_CONNECTION_MARKERS):
            return ResilienceErrorClass.timeout_rebuild
        if _any_marker(message, _TRANSIENT_TIMEOUT_MARKERS):
            return ResilienceErrorClass.transient_retryable
        return ResilienceErrorClass.deterministic_abort

    # Stdlib structural types. A bare ``ConnectionError`` (and its
    # ECONNRESET/EPIPE subclasses) means the connection is dead → rebuild it
    # before retrying; a ``TimeoutError`` is a plain deadline → retry as-is.
    if isinstance(exc, ConnectionError):
        return ResilienceErrorClass.timeout_rebuild
    if isinstance(exc, TimeoutError):
        return ResilienceErrorClass.transient_retryable

    if _any_marker(message, _RATE_LIMIT_MARKERS):
        return ResilienceErrorClass.rate_limited
    if _any_marker(message, _STALE_CONNECTION_MARKERS):
        return ResilienceErrorClass.timeout_rebuild
    if _any_marker(message, _TRANSIENT_TIMEOUT_MARKERS):
        return ResilienceErrorClass.transient_retryable
    return ResilienceErrorClass.deterministic_abort


def _any_marker(haystack: str, markers: tuple[str, ...]) -> bool:
    return any(m in haystack for m in markers)


# ---------------------------------------------------------------------------
# Decorrelated-jitter backoff (AWS "Exponential Backoff And Jitter")
# ---------------------------------------------------------------------------


def decorrelated_jitter_backoff(
    *,
    attempt: int,
    base_seconds: float,
    max_seconds: float,
    previous_delay: float = 0.0,
    rng: random.Random | None = None,
) -> float:
    """Return the next decorrelated-jitter backoff delay, in seconds.

    AWS decorrelated jitter:
    ``sleep = min(max, uniform(base, prev*3))`` — gives exponential-ish
    growth without the thundering-herd lockstep of plain ``base*2**n``. The
    caller threads ``previous_delay`` so consecutive retries decorrelate.

    The FIRST retry (``previous_delay == 0``) seeds the decorrelation from
    ``base`` rather than collapsing the window to a single point: it draws
    from ``uniform(base, min(base*3, ceiling))``. Without this seed the
    first (and most common) retry would fire at EXACTLY ``base`` for every
    concurrent retrier — the precise lockstep decorrelated jitter exists to
    break. (Previously ``previous_delay=0`` -> ``upper=base``
    -> ``uniform(base, base)`` -> always ``base``.)

    Degenerate inputs preserve the prior behaviour:

    * ``base_seconds <= 0`` → returns 0.0 (immediate re-issue — exactly the
      ``backoff_base=0.0`` default that keeps the loop a plain retry loop).
    * ``attempt < 1`` → 0.0 (the first attempt never sleeps).
    """
    if attempt < 1 or base_seconds <= 0.0:
        return 0.0
    ceiling = max_seconds if max_seconds > 0.0 else base_seconds
    # Seed the decorrelation from ``base`` on the first retry so the window
    # is ``[base, base*3]`` (clamped to the ceiling) rather than a single
    # point; subsequent retries decorrelate off the prior sleep.
    effective_previous = previous_delay if previous_delay > 0.0 else base_seconds
    upper = max(base_seconds, effective_previous * 3.0)
    upper = min(upper, ceiling)
    lower = min(base_seconds, upper)
    r = rng if rng is not None else random
    delay = r.uniform(lower, upper)
    return max(0.0, min(delay, ceiling))


# ---------------------------------------------------------------------------
# Token-bucket failure-rate retry budget (async, lock-injected)
# ---------------------------------------------------------------------------


class TokenBucketRetryBudget:
    """Async failure-rate retry budget over a :class:`RetryBudgetState`.

    Holds one :class:`RetryBudgetState` per ``host_key`` in a plain dict
    and mutates it under an INJECTED lock so the authority is
    process-local-per-pod (sufficient for non-amplification; A3) with NO
    module-level state — the caller owns the dict + lock lifetime (e.g.
    per-run state guarded by the run's shared-state lock).

    With budgeting OFF the caller simply never constructs/consults this —
    the transport loop is then bit-identical to a plain attempt-count loop.
    """

    def __init__(
        self,
        *,
        max_tokens: float,
        token_ratio: float,
        suppress_below_ratio: float,
        lock: asyncio.Lock | None = None,
        buckets: dict[str, RetryBudgetState] | None = None,
    ) -> None:
        if max_tokens <= 0.0:
            raise ValueError("max_tokens must be > 0")
        self._max_tokens = max_tokens
        self._token_ratio = token_ratio
        self._suppress_below_ratio = suppress_below_ratio
        self._lock = lock if lock is not None else asyncio.Lock()
        self._buckets: dict[str, RetryBudgetState] = (
            buckets if buckets is not None else {}
        )

    def _bucket(self, host_key: str) -> RetryBudgetState:
        bucket = self._buckets.get(host_key)
        if bucket is None:
            bucket = RetryBudgetState.full(
                max_tokens=self._max_tokens,
                token_ratio=self._token_ratio,
                suppress_below_ratio=self._suppress_below_ratio,
            )
            self._buckets[host_key] = bucket
        return bucket

    async def consume_or_suppress(self, host_key: str) -> bool:
        """Try to spend one token on a retry.

        Returns ``True`` when the retry must be SUPPRESSED (bucket at/below
        the suppress threshold — give up on best evidence). Returns
        ``False`` after consuming one token (retry permitted). Mutation is
        under the injected lock so concurrent retriers on the same pod
        share one budget.

        Prefer the :meth:`would_suppress` (read-only probe) + :meth:`charge`
        (debit only on an actually-issued retry) pair in
        :func:`resilient_transport_call`: charging a token here BEFORE the
        policy decides retry-vs-finalize corrupts the accounting on every
        finalize path (policy disabled, deadline-reserve gate, honoured-reset
        finalize). This atomic check-and-consume is retained for any caller
        that genuinely issues a retry exactly when the verdict is "permitted".
        """
        async with self._lock:
            bucket = self._bucket(host_key)
            if bucket.should_suppress_retry():
                return True
            bucket.consume_retry()
            return False

    async def would_suppress(self, host_key: str) -> bool:
        """Read-only suppress verdict — does NOT consume a token.

        Returns ``True`` when the bucket is at/below the suppress threshold
        (the next retry must be suppressed — give up on best evidence). The
        caller threads this into :meth:`ResiliencePolicy.decide` as
        ``budget_suppressed`` and then calls :meth:`charge` ONLY when the
        decision is an actually-issued retry, so a finalize decision never
        debits the budget. Acquires the lock for a consistent read; the
        check-then-:meth:`charge` window is process-local-per-pod and bounded
        (non-amplification is sufficient — see class docstring).
        """
        async with self._lock:
            return self._bucket(host_key).should_suppress_retry()

    async def charge(self, host_key: str) -> None:
        """Debit one token for an ACTUALLY-ISSUED retry (floored at 0).

        Call this only after :meth:`ResiliencePolicy.decide` returns a real
        transport-retry action and the loop is committed to re-issuing — so a
        consumed token always corresponds to an issued retry.
        """
        async with self._lock:
            self._bucket(host_key).consume_retry()

    async def deposit_success(self, host_key: str) -> None:
        """Deposit ``token_ratio`` tokens on a successful call."""
        async with self._lock:
            self._bucket(host_key).deposit_success()

    def snapshot(self, host_key: str) -> RetryBudgetState | None:
        """Return the current bucket state (read-only; testing/telemetry)."""
        bucket = self._buckets.get(host_key)
        return bucket.model_copy() if bucket is not None else None


# ---------------------------------------------------------------------------
# Deadline-aware finalization reserve
# ---------------------------------------------------------------------------


def deadline_finalization_reserve_ok(
    *,
    remaining_seconds: float | None,
    reserve_seconds: float,
    next_attempt_cost_seconds: float,
) -> bool:
    """Whether a retry still leaves the reserved slice for a full answer.

    Deadline-aware retry (gRPC A6 "deadline applies across all attempts" +
    our deadline-reserve carry-forward): a retry must NEVER spend the
    deadline its caller reserved for the final answer. Returns ``True``
    only when ``remaining - next_attempt_cost >= reserve`` — i.e. issuing
    the retry still leaves ``reserve_seconds`` for finalisation. This only
    ever causes an EARLIER give-up; it never widens the deadline (A1/N3).

    #147 expired-vs-untracked split — ``remaining_seconds`` distinguishes
    two cases the pre-#147 code conflated on the single ``<= 0.0`` sentinel:

    * ``None`` — NO wall-clock budget is tracked (``agent_max_seconds <= 0``
      or the run clock was never stamped). The reserve gate is INERT →
      returns ``True`` (behaviour preserved for untracked callers).
    * ``<= 0.0`` — a TRACKED budget that is already EXPIRED. A retry now
      would eat the terminal-answer window, so the gate REFUSES it →
      returns ``False`` (the run finalises on best evidence instead of
      issuing a doomed post-deadline RPC). Before #147 this floored-to-0.0
      tracked-expired value fell through the untracked branch and FAILED
      OPEN, issuing more retries at the precise moment the budget was gone.
    """
    if remaining_seconds is None:
        return True
    if remaining_seconds <= 0.0:
        return False
    return (remaining_seconds - max(0.0, next_attempt_cost_seconds)) >= max(
        0.0, reserve_seconds
    )


# ---------------------------------------------------------------------------
# Classify-then-act policy
# ---------------------------------------------------------------------------

# error_class → the action used when a retry is PERMITTED (idempotent +
# within budget + reserve). The non-retry fallbacks (abort / finalize_now)
# are decided separately by the policy.
_RETRY_ACTION_FOR_CLASS: dict[ResilienceErrorClass, ResilienceAction] = {
    ResilienceErrorClass.transient_retryable: ResilienceAction.backoff_and_retry,
    ResilienceErrorClass.rate_limited: ResilienceAction.backoff_and_retry,
    ResilienceErrorClass.timeout_rebuild: ResilienceAction.rebuild_and_retry,
    ResilienceErrorClass.context_overflow: ResilienceAction.compress_and_retry,
    ResilienceErrorClass.payload_too_large: ResilienceAction.shrink_and_retry,
    ResilienceErrorClass.unknown: ResilienceAction.backoff_and_retry,
}


@dataclass(frozen=True, slots=True)
class _PolicyParams:
    """Resolved resilience knobs (from the RC snapshot, defaults preserve)."""

    enabled: bool
    backoff_base_seconds: float
    backoff_max_seconds: float
    deadline_reserve_seconds: float


def _resolve_policy_params(rc: LoopConstants | None) -> _PolicyParams:
    if rc is None:
        return _PolicyParams(
            enabled=False,
            backoff_base_seconds=0.0,
            backoff_max_seconds=0.0,
            deadline_reserve_seconds=0.0,
        )
    return _PolicyParams(
        enabled=bool(getattr(rc, "resilience_enabled", False)),
        backoff_base_seconds=float(
            getattr(rc, "resilience_backoff_base_seconds", 0.0) or 0.0
        ),
        backoff_max_seconds=float(
            getattr(rc, "resilience_backoff_max_seconds", 0.0) or 0.0
        ),
        deadline_reserve_seconds=float(
            getattr(rc, "resilience_deadline_reserve_seconds", 0.0) or 0.0
        ),
    )


class ResiliencePolicy:
    """Universal classify-then-act policy.

    Pure decision-maker: given a classified failure + the call's
    idempotency stance + the attempt index + the remaining wall-clock,
    return a :class:`ResilienceDecision`. Stateless and deterministic —
    same inputs → same decision — so it is trivially testable without any
    transport/LLM machinery.

    The token bucket + the actual sleeping live in
    :func:`resilient_transport_call`; the policy only DECIDES (action +
    advisory backoff + finalization hint).
    """

    def __init__(self, *, rc: LoopConstants | None = None) -> None:
        self._params = _resolve_policy_params(rc)

    @property
    def enabled(self) -> bool:
        return self._params.enabled

    def decide(
        self,
        *,
        error_class: ResilienceErrorClass,
        idempotent: bool,
        attempt: int,
        max_attempts: int,
        remaining_seconds: float | None = None,
        next_call_timeout_seconds: float = 0.0,
        previous_backoff_seconds: float = 0.0,
        server_reset_seconds: float | None = None,
        budget_suppressed: bool = False,
        rng: random.Random | None = None,
    ) -> ResilienceDecision:
        """Return the classify-then-act verdict for one failed attempt.

 ``attempt`` is 1-based (the attempt that just failed).
 ``budget_suppressed`` is the token-bucket verdict for THIS retry
 decision (the caller consults the bucket then passes the result —
 keeping the policy free of mutable state).

 ``server_reset_seconds`` is an OPTIONAL server-stated rate-limit
 reset / retry-after window (seconds from now) the backend surfaced on
 the failure . For the
 :attr:`~protocore.contracts.resilience.ResilienceErrorClass.rate_limited`
 class the policy prefers ``max(server_reset, jittered_backoff)`` with
 the server reset taken VERBATIM (NOT clamped to the generic backoff
 ceiling — sleeping less than the stated window just re-issues into the
 same closed limit; only a defensive absolute sanity cap applies). The
 deadline-reserve gate is what bounds a large reset (retry-vs-finalize).
 ``None`` (the default) keeps the plain jittered backoff. It is only
 honoured for ``rate_limited`` — a reset window is a rate-limit
 affordance, not a universal override of every class's backoff.

 ``remaining_seconds`` is the wall-clock budget left before the run
 deadline, or ``None`` when no budget is tracked (#147). The default
 is ``None`` so a caller that does not thread a deadline keeps the
 reserve gate inert; a tracked-but-expired budget arrives as ``<= 0.0``
 and the reserve gate then refuses the retry (finalize on best
 evidence) rather than fail-opening into a doomed post-deadline RPC.

 ``next_call_timeout_seconds`` is the wall-clock TIMEOUT budget of the
 next RPC the retry would issue (e.g. ``spec.timeout_ms / 1000.0``).
 The deadline reserve must reserve room for the PROJECTED next attempt —
 its backoff sleep PLUS its call timeout — not the previous sleep (which
 is 0.0 on the first retry and would let a retry eat the finalization
 slice). The policy computes the candidate backoff first, then gates the
 reserve on ``backoff + next-call timeout``.
 """
        p = self._params

        # Deterministic NO → abort (hand to fallback model upstream where
        # configured; the transport layer has no fallback, so abort). This
        # fires even when the policy is disabled — a structural failure must
        # surface unchanged, never be masked behind a finalize.
        if error_class is ResilienceErrorClass.deterministic_abort:
            return ResilienceDecision(
                error_class=error_class,
                action=ResilienceAction.abort,
                retryable=False,
                reason_detail="deterministic",
            )

        # ENFORCE ``resilience_enabled``. The policy stores
        # the flag; ``decide()`` must honour it so the policy is safe as a
        # universal primitive: a caller that constructs
        # ``ResiliencePolicy(rc=LoopConstants(resilience_enabled=False))``
        # and passes ``max_attempts > 1`` must NOT retry. When disabled, every
        # non-structural class finalises on best evidence (no retry / no
        # compress / no shrink). Defensive even though the live routing
        # already gates on ``rc.resilience_enabled`` before constructing the
        # policy + wrapper.
        if not p.enabled:
            return ResilienceDecision(
                error_class=error_class,
                action=ResilienceAction.finalize_now,
                retryable=False,
                finalization_recommended=True,
                reason_detail="resilience_disabled",
            )

        retry_action = _RETRY_ACTION_FOR_CLASS.get(
            error_class, ResilienceAction.backoff_and_retry
        )

        # context_overflow / payload_too_large are NOT same-request
        # transport retries — they need a COMPRESS / SHRINK first. Surface
        # the action with retryable=True so the caller performs the body
        # adjustment then re-attempts; the transport wrapper treats these
        # as non-transport-retryable (it does not own compaction).
        if error_class in (
            ResilienceErrorClass.context_overflow,
            ResilienceErrorClass.payload_too_large,
        ):
            return ResilienceDecision(
                error_class=error_class,
                action=retry_action,
                retryable=True,
                reason_detail=error_class.value,
            )

        # From here: a candidate transport-retryable class
        # (transient / rate_limited / timeout_rebuild / unknown).
        if error_class not in RETRYABLE_ERROR_CLASSES:
            return ResilienceDecision(
                error_class=error_class,
                action=ResilienceAction.abort,
                retryable=False,
                reason_detail=error_class.value,
            )

        # Classify-don't-retry mutations (D6): a non-idempotent call is
        # NEVER blind-retried — the runtime tells the model the commit
        # state via read-back; it does not re-issue. Surface finalize_now
        # so the caller stops cleanly rather than aborting hard.
        if not idempotent:
            return ResilienceDecision(
                error_class=error_class,
                action=ResilienceAction.finalize_now,
                retryable=False,
                finalization_recommended=True,
                reason_detail="non_idempotent_no_retry",
            )

        # Out of attempts → give up (finalise on best evidence).
        if attempt >= max_attempts:
            return ResilienceDecision(
                error_class=error_class,
                action=ResilienceAction.finalize_now,
                retryable=False,
                finalization_recommended=True,
                reason_detail="attempts_exhausted",
            )

        # Budget exhausted → suppress the retry, finalise on best evidence.
        if budget_suppressed:
            return ResilienceDecision(
                error_class=error_class,
                action=ResilienceAction.finalize_now,
                retryable=False,
                finalization_recommended=True,
                reason_detail="retry_budget_exhausted",
            )

        # Compute the candidate backoff BEFORE the reserve check: the
        # projected cost of the next attempt is its backoff sleep PLUS the
        # next RPC's call timeout. Computing it here lets the reserve gate
        # reserve the finalization slice against the REAL next-attempt cost
        # rather than the previous (often 0.0) sleep.
        backoff = decorrelated_jitter_backoff(
            attempt=attempt,
            base_seconds=p.backoff_base_seconds,
            max_seconds=p.backoff_max_seconds,
            previous_delay=previous_backoff_seconds,
            rng=rng,
        )

        # honour a server-stated rate-limit reset. When the host
        # told us exactly when the window reopens, sleeping a shorter generic
        # backoff just re-issues into the same closed limit; prefer
        # ``max(server_reset, jittered_backoff)`` with the server reset taken
        # VERBATIM (NOT clamped to the generic backoff ceiling — the reference
        # deliberately bypasses that cap for a stated reset, and the
        # deadline-reserve gate below is what decides retry-vs-finalize when
        # the reset is large). Only a defensive absolute sanity cap guards
        # against a garbage/hostile value. Only for ``rate_limited``: a reset
        # window is a rate-limit affordance, not a universal override.
        if (
            error_class is ResilienceErrorClass.rate_limited
            and server_reset_seconds is not None
            and server_reset_seconds > 0.0
        ):
            honoured_reset = min(
                server_reset_seconds, _MAX_HONOURED_RATE_LIMIT_RESET_SECONDS
            )
            backoff = max(backoff, honoured_reset)

        # Deadline reserve: refuse a retry that would eat the finalization
        # slice. Inert when no wall-clock budget is tracked. The next-attempt
        # cost is conservative — backoff + the next RPC's full timeout — so a
        # retry is refused unless ``remaining`` still covers BOTH the projected
        # next call AND the reserved finalization slice.
        next_attempt_cost_seconds = backoff + max(0.0, next_call_timeout_seconds)
        if not deadline_finalization_reserve_ok(
            remaining_seconds=remaining_seconds,
            reserve_seconds=p.deadline_reserve_seconds,
            next_attempt_cost_seconds=next_attempt_cost_seconds,
        ):
            return ResilienceDecision(
                error_class=error_class,
                action=ResilienceAction.finalize_now,
                retryable=False,
                finalization_recommended=True,
                reason_detail="deadline_reserve",
            )

        return ResilienceDecision(
            error_class=error_class,
            action=retry_action,
            retryable=True,
            backoff_seconds=backoff,
            reason_detail=error_class.value,
        )


# ---------------------------------------------------------------------------
# The universal transport decorator
# ---------------------------------------------------------------------------


async def _maybe_rebuild_transport(transport: IToolTransport) -> None:
    """Invoke the transport's optional ``rebuild`` hook .

    Called before a ``rebuild_and_retry`` retry so the next attempt runs on a
    fresh client/connection rather than a stale/dead socket. The hook is
    OPTIONAL: a transport without ``rebuild`` is a graceful no-op. Cancellation
    propagates (orchestrator teardown must not be swallowed); any other rebuild
    failure is best-effort — the retry still proceeds on the existing transport
    rather than aborting the whole call on a rebuild hiccup.
    """
    rebuild = getattr(transport, "rebuild", None)
    if rebuild is None or not callable(rebuild):
        return
    try:
        await rebuild()
    except (asyncio.CancelledError, GeneratorExit):
        raise
    except Exception:
        logger.warning(
            "resilient_transport_call transport.rebuild() raised; retrying on "
            "the existing transport (rebuild is best-effort)",
            exc_info=True,
        )


async def resilient_transport_call(
    transport: IToolTransport,
    spec: TransportCallSpec,
    *,
    policy: ResiliencePolicy,
    max_attempts: int,
    budget: TokenBucketRetryBudget | None = None,
    remaining_seconds_fn: Callable[[], float | None] | None = None,
    on_attempt: Callable[[int, ResilienceErrorClass, ResilienceDecision], None]
    | None = None,
    sleep: Callable[[float], Awaitable[None]] | None = None,
    rng: random.Random | None = None,
) -> Any:
    """Drive ``transport.invoke(spec)`` with classify-then-act resilience.

 The single universal transport-retry primitive. It:

 * issues the first attempt unconditionally (the budget gates RETRIES,
 never the first try);
 * on failure, classifies via :func:`classify_transport_error`, then
 asks ``policy`` for a decision (honouring idempotency, attempt
 count, the deadline reserve, and the token-bucket verdict);
 * on a stale-connection (``rebuild_and_retry``) decision, invokes the
 transport's optional ``rebuild`` hook before the retry ;
 * sleeps the advisory backoff (cancellable) and retries; on
 budget/deadline/attempt give-up raises
 :class:`ToolTransportRetryBudgetExhausted` so the caller finalises
 on best evidence;
 * NEVER re-issues a non-idempotent (mutating) spec — it raises the
 original error immediately for the caller to read-back the commit
 state (no idempotency key needed; read-back is the substitute).
 * deposits a success token on the winning attempt.

 ``max_attempts`` is the TOTAL attempt budget (>=1). With ``budget=None``
 and a base backoff of 0.0 this loop is bit-identical to a plain
 attempt-count retry loop, so the prior behaviour is exactly reproduced
 when the new RCs keep their defaults.
 """
    if max_attempts < 1:
        max_attempts = 1
    do_sleep = sleep if sleep is not None else asyncio.sleep
    previous_backoff = 0.0
    last_exc: BaseException | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            response = await transport.invoke(spec)
        except ToolTransportRetryBudgetExhausted:
            # An inner layer already gave up — propagate unchanged.
            raise
        except (asyncio.CancelledError, GeneratorExit):
            raise
        except Exception as exc:
            last_exc = exc
            error_class = classify_transport_error(exc)

            is_last_attempt = attempt >= max_attempts
            # Consult the budget for THIS retry decision only when a retry
            # would otherwise be issued (not the last attempt, idempotent,
            # candidate-retryable) — never spend a token on the final attempt
            # (no retry follows it).
            #
            # The suppress verdict is a READ-ONLY probe here; the token is
            # only DEBITED (``budget.charge`` below) once ``policy.decide()``
            # returns an actually-issued transport retry. Charging before
            # ``decide()`` corrupted the accounting on every finalize path
            # (policy disabled, deadline-reserve gate, honoured rate-limit
            # reset that finalizes) — a token spent with no retry issued.
            # ``budget_is_consultable`` is recomputed below to gate the charge
            # so it tracks the SAME predicate as the probe.
            budget_is_consultable = (
                budget is not None
                and not is_last_attempt
                and spec.idempotent
                and error_class in RETRYABLE_ERROR_CLASSES
            )
            budget_suppressed = False
            if budget_is_consultable:
                if budget is None:
                    raise AssertionError("consultable retry budget is missing") from None
                budget_suppressed = await budget.would_suppress(spec.host_key)

            # #147 — when no remaining-fn is wired the budget is UNTRACKED,
            # so pass ``None`` (reserve gate inert), NOT ``0.0`` (which now
            # means a tracked-but-EXPIRED budget → reserve gate refuses).
            remaining = (
                remaining_seconds_fn() if remaining_seconds_fn is not None else None
            )
            decision = policy.decide(
                error_class=error_class,
                idempotent=spec.idempotent,
                attempt=attempt,
                max_attempts=max_attempts,
                remaining_seconds=remaining,
                # The projected next-attempt cost is the next RPC's full
                # timeout (the policy adds the candidate backoff itself), NOT
                # the previous sleep (0.0 on the first retry → underestimate).
                next_call_timeout_seconds=spec.timeout_ms / 1000.0,
                previous_backoff_seconds=previous_backoff,
                # surface any server-stated rate-limit reset the
                # backend attached so the policy can prefer it over a generic
                # jittered backoff for the rate_limited class.
                server_reset_seconds=transport_retry_after_seconds(exc),
                budget_suppressed=budget_suppressed,
                rng=rng,
            )
            # The attempt observer is best-effort telemetry; a
            # throwing observer must NEVER abort the transport call. Guard it
            # (but let cancellation propagate so teardown is not swallowed).
            if on_attempt is not None:
                try:
                    on_attempt(attempt, error_class, decision)
                except (asyncio.CancelledError, GeneratorExit):
                    raise
                except Exception:
                    logger.warning(
                        "resilient_transport_call on_attempt observer raised; "
                        "ignoring (telemetry is best-effort)",
                        exc_info=True,
                    )

            if not decision.retryable:
                if decision.finalization_recommended:
                    raise ToolTransportRetryBudgetExhausted(
                        decision.reason_detail or "finalize",
                        host_key=spec.host_key,
                        method_class=spec.method_class,
                    ) from exc
                # abort — surface the structural failure unchanged.
                raise

            # Only transport-level retries actually re-issue here. COMPRESS
            # / SHRINK are owned by the LLM loop, not the transport wrapper;
            # if such a class reaches here (shouldn't for a transport spec),
            # surface it rather than blind-retry the same body.
            if decision.action in (
                ResilienceAction.compress_and_retry,
                ResilienceAction.shrink_and_retry,
            ):
                raise

            # The loop is now COMMITTED to re-issuing a real transport retry
            # (decision.retryable AND not a compress/shrink hand-off). Every
            # finalize path above (policy disabled, deadline reserve,
            # honoured-reset finalize, attempts exhausted, non-idempotent)
            # returned before this point and therefore never charges the bucket.

            # a stale-connection class (timeout_rebuild ->
            # rebuild_and_retry) must run on a FRESH client/connection. Invoke
            # the transport's optional rebuild() hook BEFORE the retry so the
            # next attempt does not re-issue on the dead socket (mirrors the
            # reference forcing a fresh client on ECONNRESET/EPIPE). Cancel
            # propagates; any other rebuild failure is best-effort (the retry
            # still proceeds on the existing transport).
            if decision.action is ResilienceAction.rebuild_and_retry:
                await _maybe_rebuild_transport(transport)

            previous_backoff = decision.backoff_seconds
            if previous_backoff > 0.0:
                # Do NOT suppress CancelledError here. A cancel that arrives
                # during the backoff sleep means the run was cancelled
                # (orchestrator teardown / deadline kill); it MUST propagate
                # so the loop stops immediately and never issues another RPC
                # after cancellation (post-cancel side effects through a
                # generic transport wrapper break cancellation semantics). The
                # sleep is the only await between attempts, so letting the
                # cancel out here is sufficient.
                await do_sleep(previous_backoff)

            # Debit the budget token HERE, AFTER the rebuild + backoff awaits
            # and IMMEDIATELY before the loop re-issues ``transport.invoke``
            # at the top of the next iteration, so a token is spent
            # if-and-only-if a retry is actually issued. Charging before the
            # awaits (the prior ordering) leaked a token whenever a
            # CancelledError landed during ``_maybe_rebuild_transport`` or
            # ``do_sleep`` — the retry never fired yet the per-host budget was
            # already debited. Both awaits above re-raise CancelledError
            # (rebuild via ``_maybe_rebuild_transport``; backoff via the bare
            # ``do_sleep``), so a cancel in either window now propagates BEFORE
            # this charge and
            # the bucket stays exact. Gated on the SAME ``budget_is_consultable``
            # predicate as the read-only ``would_suppress`` probe so the count
            # tracks the probe (probe semantics unchanged).
            if budget_is_consultable:
                if budget is None:
                    raise AssertionError("consultable retry budget is missing") from None
                await budget.charge(spec.host_key)
            continue
        else:
            if budget is not None:
                await budget.deposit_success(spec.host_key)
            return response

    # Loop exhausted without returning — re-raise the last error. (Reached
    # only when the final attempt failed AND the policy said retryable but
    # the loop ran out — defensive; the attempt-count check above normally
    # converts the last failure into finalize_now.)
    if last_exc is not None:
        raise last_exc
    raise ToolTransportError("resilient_transport_call exhausted with no error")


__all__ = [
    "ResiliencePolicy",
    "TokenBucketRetryBudget",
    "classify_transport_error",
    "deadline_finalization_reserve_ok",
    "decorrelated_jitter_backoff",
    "reason_to_error_class",
    "resilient_transport_call",
    "transport_retry_after_seconds",
]
