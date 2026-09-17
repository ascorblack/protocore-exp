"""Repeated token estimates of an unchanged history must be nearly free.

Every budget that sizes a message sequence re-estimates the whole thing, and a
history is sized several times per turn; the estimate itself walks every
character. Before these estimates were remembered, three consecutive counts of
a 300-message history cost 363, 346 and 355 ms — the second walk paid the full
price of the first for an answer it already had, on the event loop, while every
other run sharing that loop waited.

The tests below check both halves of the claim: that a repeat is cheap, and
that it is still correct when the messages or the tuning change.
"""
from __future__ import annotations

import time

import pytest

from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.types import ImageRefBlock, Message, MessageRole, TextBlock
from protocore.runtime.context.compaction import (
    TokenEstimator,
    estimate_history_tokens,
    estimate_history_tokens_uncalibrated,
    estimate_message_tokens,
)

HISTORY_LENGTH = 300
MIN_REPEAT_SPEEDUP = 20.0
MAX_APPEND_OVERHEAD = 2.0
# One append is a handful of microseconds; timing that few needs more than one
# reading to separate the measurement from the scheduler.
APPEND_ROUNDS = 5
# Mixed script on purpose: the estimate costs each character class separately,
# so a single-alphabet fixture would understate the work being skipped.
BODY = 'Строка русского текста, JSON {"k": 1}, latin words, 日本語 ' * 120


def _history(marker: str, length: int = HISTORY_LENGTH) -> list[Message]:
    return [
        Message(
            role=MessageRole.user if index % 2 == 0 else MessageRole.assistant,
            content_blocks=[TextBlock(text=f"{marker} {index} {BODY}")],
        )
        for index in range(length)
    ]


def _seconds(call: object) -> float:
    started = time.perf_counter()
    call()  # type: ignore[operator]
    return time.perf_counter() - started


def test_repeat_estimate_of_an_unchanged_history_is_far_cheaper() -> None:
    rc = LoopConstants()
    history = _history("repeat")
    estimator = TokenEstimator()

    first = _seconds(lambda: estimator.estimate_history(history, rc))
    second = _seconds(lambda: estimator.estimate_history(history, rc))
    third = _seconds(lambda: estimator.estimate_history(history, rc))

    expected = estimator.estimate_history(history, rc)
    assert expected > 0
    assert second * MIN_REPEAT_SPEEDUP <= first, (
        f"second estimate took {second * 1000:.2f} ms against a first of "
        f"{first * 1000:.2f} ms"
    )
    assert third * MIN_REPEAT_SPEEDUP <= first, (
        f"third estimate took {third * 1000:.2f} ms against a first of "
        f"{first * 1000:.2f} ms"
    )


def test_appending_one_message_costs_about_one_message() -> None:
    """The marginal cost of one more message is the cost of that message.

    Both sides of the comparison are measured with the estimator the run
    actually has, so the claim holds whether the character counting is done in
    Python or by the optional native extension: what is asserted is a ratio
    between two costs paid by the same implementation, never an absolute
    calibrated for one of them.

    Re-estimating a warm history is not free — every remembered message is
    still looked up — and that bookkeeping is a fixed cost the append did not
    cause. It is measured separately, on the same warm history with nothing
    added, and subtracted, leaving the marginal cost of the one new message.
    Each round is timed on its own history, and the cheapest round of each
    measurement is compared, so a scheduling hiccup inflates neither side.
    """
    rc = LoopConstants()
    marginal: list[float] = []
    alone: list[float] = []

    for round_index in range(APPEND_ROUNDS):
        history = _history(f"append {round_index}")
        estimator = TokenEstimator()
        estimator.estimate_history(history, rc)

        # The bookkeeping alone: the same warm history, nothing added.
        repeat = _seconds(
            lambda e=estimator, h=history: e.estimate_history(h, rc),  # type: ignore[misc]
        )
        added = Message(
            role=MessageRole.user,
            content_blocks=[TextBlock(text=f"added {round_index} {BODY}")],
        )
        grown = _seconds(
            lambda e=estimator, h=history, m=added: e.estimate_history(  # type: ignore[misc]
                [*h, m], rc
            ),
        )
        marginal.append(grown - repeat)

        # The same text, in a message no estimator has seen, priced alone.
        alone_message = Message(
            role=MessageRole.user,
            content_blocks=[TextBlock(text=f"added {round_index} {BODY}")],
        )
        alone.append(
            _seconds(
                lambda m=alone_message: TokenEstimator().estimate_message(  # type: ignore[misc]
                    m, rc
                ),
            )
        )

    cheapest_marginal = min(marginal)
    cheapest_alone = min(alone)
    assert cheapest_alone > 0
    assert cheapest_marginal <= cheapest_alone * MAX_APPEND_OVERHEAD, (
        f"growing a warm history cost {cheapest_marginal * 1000:.2f} ms above its own "
        f"repeat, against {cheapest_alone * 1000:.2f} ms for the one added message"
    )


def test_changed_tuning_is_not_answered_from_the_cache() -> None:
    """A cached estimate must not survive a change to the tuning it used."""
    history = _history("tuning", length=8)
    estimator = TokenEstimator()

    baseline = LoopConstants()
    before = estimator.estimate_history(history, baseline)
    retuned = baseline.model_copy(
        update={
            "token_count_chars_per_token_cyrillic": (
                baseline.token_count_chars_per_token_cyrillic * 2
            ),
        },
    )
    after = estimator.estimate_history(history, retuned)

    assert before != after
    assert after == estimator.estimate_history(history, retuned)
    assert before == estimator.estimate_history(history, baseline)


def test_the_calibration_factor_scales_an_answer_it_does_not_evict() -> None:
    """The calibrator's pass and the loop's must not evict each other.

    The factor is a multiplier over the whole partition, so an estimate made
    under one value answers for any other. Keying the cache on it split every
    history in two: the calibrator sizes the request uncalibrated to compare it
    against what the provider reported, and each pass threw away what the pass
    before it had paid for.
    """
    history = _history("calibrated", length=8)
    estimator = TokenEstimator()

    plain = LoopConstants(token_estimate_calibration=1.0)
    scaled = LoopConstants(token_estimate_calibration=1.5)

    raw = estimator.estimate_history_uncalibrated(history, plain)
    entries = len(estimator)
    calibrated = estimator.estimate_history(history, scaled)

    assert len(estimator) == entries
    assert raw == estimator.estimate_history(history, plain)
    assert calibrated == sum(
        round(estimator.estimate_message(message, plain) * 1.5) for message in history
    )
    assert calibrated > raw


def test_the_module_level_uncalibrated_reading_agrees_with_an_owned_estimator() -> None:
    history = _history("module-uncalibrated", length=6)
    estimator = TokenEstimator()
    rc = LoopConstants(token_estimate_calibration=1.3)

    assert estimate_history_tokens_uncalibrated(
        history, rc
    ) == estimator.estimate_history_uncalibrated(history, rc)


def test_image_token_tuning_is_part_of_the_key() -> None:
    message = Message(
        role=MessageRole.user,
        content_blocks=[ImageRefBlock(blob_ref="blob://synthetic/one")],
    )
    estimator = TokenEstimator()
    baseline = LoopConstants()
    retuned = baseline.model_copy(
        update={"token_count_image_tokens": baseline.token_count_image_tokens + 7},
    )
    assert estimator.estimate_message(message, baseline) != estimator.estimate_message(
        message,
        retuned,
    )


def test_two_runs_do_not_read_each_other_s_estimates() -> None:
    """Separate estimators share nothing, even for identical content."""
    rc = LoopConstants()
    first_run = _history("run", length=6)
    second_run = _history("run", length=6)
    assert all(
        a.content_blocks == b.content_blocks and a is not b
        for a, b in zip(first_run, second_run, strict=True)
    )

    first_estimator = TokenEstimator()
    second_estimator = TokenEstimator()
    first_estimator.estimate_history(first_run, rc)

    second_estimator.estimate_history(second_run, rc)
    assert second_estimator.estimate_history(
        second_run,
        rc,
    ) == first_estimator.estimate_history(first_run, rc)

    # Nothing the first estimator remembered is reachable through the second.
    fresh = TokenEstimator()
    assert fresh.estimate_history(first_run, rc) == first_estimator.estimate_history(
        first_run,
        rc,
    )


def test_a_rewritten_history_is_estimated_again() -> None:
    """Compaction builds new message objects, which are not cache hits."""
    rc = LoopConstants()
    original = _history("rewrite", length=4)
    estimator = TokenEstimator()
    before = estimator.estimate_history(original, rc)

    rewritten = [
        message.model_copy(
            update={"content_blocks": [TextBlock(text="summary placeholder")]},
        )
        for message in original
    ]
    after = estimator.estimate_history(rewritten, rc)
    assert after < before
    assert after == TokenEstimator().estimate_history(rewritten, rc)


def test_the_cache_is_bounded() -> None:
    rc = LoopConstants()
    bound = 4
    estimator = TokenEstimator(max_entries=bound)
    history = _history("bounded", length=bound * 3)
    total = estimator.estimate_history(history, rc)
    assert total == TokenEstimator().estimate_history(history, rc)
    assert len(estimator) == bound


def test_clearing_forgets_everything() -> None:
    rc = LoopConstants()
    history = _history("clear", length=4)
    estimator = TokenEstimator()
    before = estimator.estimate_history(history, rc)
    estimator.clear()
    assert estimator.estimate_history(history, rc) == before


@pytest.mark.parametrize("length", [0, 1, 5])
def test_module_functions_agree_with_an_owned_estimator(length: int) -> None:
    rc = LoopConstants()
    history = _history("module", length=length)
    estimator = TokenEstimator()
    assert estimate_history_tokens(history, rc) == estimator.estimate_history(
        history,
        rc,
    )
    for message in history:
        assert estimate_message_tokens(message, rc) == estimator.estimate_message(
            message,
            rc,
        )


def test_appending_to_a_message_is_not_possible_behind_the_cache() -> None:
    """The memoisation's premise, enforced rather than asserted.

    An estimate is remembered against the message object, on the grounds that
    the same object still holds the same content. ``frozen=True`` alone does
    not give that: it stops the field being rebound, not a list behind it
    being appended to. If it could be appended to, a message could grow by
    forty thousand characters and every budget sizing that history would go on
    reading the number from before.
    """
    message = Message(
        role=MessageRole.assistant,
        content_blocks=[TextBlock(text="hello world")],
    )
    with pytest.raises(AttributeError):
        message.content_blocks.append(TextBlock(text="x" * 40_000))  # type: ignore[union-attr]

    estimator = TokenEstimator()
    assert estimator.estimate_message(message, LoopConstants()) == TokenEstimator(
    ).estimate_message(message, LoopConstants())


def test_the_shared_cache_is_capacity_two_runs_compete_for_not_content_they_share() -> None:
    """What concurrent runs share through the module functions, exactly.

    They cannot read each other's numbers: a key is one run's message object.
    What they do share is the bound, so a long history from one run evicts
    another's entries and both pay the full walk again — which is the reason a
    component holding a run's history holds that run's estimator instead.
    """
    rc = LoopConstants()
    bound = 4
    shared = TokenEstimator(max_entries=bound)
    first_run = _history("first", length=bound)
    second_run = _history("second", length=bound)

    shared.estimate_history(first_run, rc)
    assert len(shared) == bound

    shared.estimate_history(second_run, rc)
    assert len(shared) == bound

    # The first run's entries are gone — evicted by a peer, not answered
    # wrongly — so its next sizing is a full walk again, and still correct.
    assert shared.estimate_history(first_run, rc) == TokenEstimator(
    ).estimate_history(first_run, rc)

    # A run that owns its estimator never enters that competition.
    owned = TokenEstimator(max_entries=bound)
    owned.estimate_history(first_run, rc)
    assert len(owned) == bound
    assert owned.estimate_history(first_run, rc) == estimate_history_tokens(
        first_run,
        rc,
    )
