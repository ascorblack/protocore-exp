"""A provider that can count a rendered request is asked near the edge, and only there.

The local estimate can run several times short of a real tokenizer on dense
text: hexadecimal filler was sized at well under two thirds of what the server
counted. These tests pin the three places the provider's own count now reaches
— the output-cap fit, the compaction gate, and the calibration factor, which
also learns from a rejection for length — and that a provider without the
capability sends exactly the requests it sent before.
"""

from __future__ import annotations

import asyncio
import logging
import re
import secrets
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock

import pytest

from protocore.contracts.llm import (
    IRequestTokenCounter,
    LLMContextWindowExceeded,
    LLMRequest,
    LLMResponse,
    LLMStreamEvent,
)
from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.types import (
    Message,
    MessageRole,
    StopReason,
    TextBlock,
    ToolDefinition,
    ToolParameterSchema,
    ToolResultBlock,
    ToolUseBlock,
)
from protocore.runtime.loop_state import LoopState
from protocore.runtime.query import _calibrate_near_compaction_trigger
from protocore.runtime.request_budget import (
    ExactTokenCountCache,
    count_request_tokens_exactly,
    estimate_request_prompt_tokens,
    estimate_request_prompt_tokens_uncalibrated,
    fit_request_to_context,
    fit_request_to_context_measured,
    near_limit,
    request_digests,
    request_token_counter,
)


def _msg(text: str, role: MessageRole = MessageRole.user) -> Message:
    return Message(role=role, content_blocks=[TextBlock(text=text)])


class _Provider:
    """A provider with no counting capability."""

    def __init__(self, exceptions: list[BaseException | None] | None = None) -> None:
        self._exceptions = list(exceptions or [])
        self.calls: list[LLMRequest] = []

    async def stream_with_tools(  # type: ignore[no-untyped-def]
        self, request: LLMRequest
    ) -> AsyncIterator[LLMStreamEvent]:
        self.calls.append(request)
        if self._exceptions:
            exc = self._exceptions.pop(0)
            if exc is not None:
                raise exc
        yield LLMStreamEvent(name="message_start", payload={})
        yield LLMStreamEvent(name="content_block_start", payload={"kind": "text"})
        yield LLMStreamEvent(name="content_block_delta", payload={"text": "done"})
        yield LLMStreamEvent(name="content_block_stop", payload={})
        yield LLMStreamEvent(
            name="message_stop", payload={"stop_reason": StopReason.end_turn.value}
        )

    async def complete_structured(  # type: ignore[no-untyped-def]
        self, request: LLMRequest, schema: dict[str, Any]
    ) -> LLMResponse:
        return LLMResponse(message=_msg("brief summary", MessageRole.assistant), stop_reason=StopReason.end_turn)

    async def complete_text(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(message=_msg("brief summary", MessageRole.assistant), stop_reason=StopReason.end_turn)

    def count_tokens(self, text: str, model: str | None = None) -> int:
        return len(text) // 4


class _CountingProvider(_Provider):
    """A provider that answers every count with ``count``, or fails with ``fail``."""

    def __init__(
        self,
        *,
        count: int | None = None,
        fail: BaseException | None = None,
        exceptions: list[BaseException | None] | None = None,
    ) -> None:
        super().__init__(exceptions)
        self.count = count
        self.fail = fail
        self.counted: list[LLMRequest] = []

    async def count_request_tokens(self, request: LLMRequest) -> int | None:
        self.counted.append(request)
        if self.fail is not None:
            raise self.fail
        return self.count


def _near_edge_request(rc: LoopConstants) -> LLMRequest:
    """A request whose estimate sits just inside the counting margin."""
    return LLMRequest(model="m", messages=[_msg("x" * 20_000)], max_tokens=1_000)


def _rc(**overrides: Any) -> LoopConstants:
    values: dict[str, Any] = {
        "model_context_window": 8_192,
        "request_context_safety_tokens": 0,
    }
    values.update(overrides)
    return LoopConstants(**values)


def test_the_capability_is_a_protocol_the_counting_double_satisfies() -> None:
    assert isinstance(_CountingProvider(count=1), IRequestTokenCounter)
    assert not isinstance(_Provider(), IRequestTokenCounter)


def test_a_mock_provider_is_not_mistaken_for_a_counter() -> None:
    assert request_token_counter(MagicMock()) is None
    assert request_token_counter(_Provider()) is None
    assert request_token_counter(_CountingProvider(count=1)) is not None


def test_near_limit_reads_the_margin_as_a_share_of_the_limit() -> None:
    rc = _rc(exact_token_count_margin_ratio=0.25)
    assert near_limit(750, 1_000, rc)
    assert not near_limit(749, 1_000, rc)
    assert near_limit(1_000, 1_000, _rc(exact_token_count_margin_ratio=0.0))
    assert not near_limit(999, 1_000, _rc(exact_token_count_margin_ratio=0.0))


@pytest.mark.asyncio
async def test_without_the_capability_the_fit_is_the_estimated_fit() -> None:
    rc = _rc()
    request = _near_edge_request(rc)
    fitted = await fit_request_to_context_measured(request, rc, _Provider())
    assert fitted.measured is None
    assert fitted.request.model_dump_json() == fit_request_to_context(request, rc).model_dump_json()


@pytest.mark.asyncio
async def test_far_from_the_edge_the_provider_is_not_asked() -> None:
    rc = _rc(model_context_window=1_000_000)
    provider = _CountingProvider(count=999_999)
    request = _near_edge_request(rc)
    fitted = await fit_request_to_context_measured(request, rc, provider)
    assert provider.counted == []
    assert fitted.measured is None
    assert fitted.request is request


@pytest.mark.asyncio
async def test_near_the_edge_the_count_replaces_the_estimate() -> None:
    rc = _rc()
    request = _near_edge_request(rc)
    estimate = estimate_request_prompt_tokens(request, rc)
    clip_limit = rc.model_context_window - request.max_tokens
    assert near_limit(estimate, clip_limit, rc) and estimate + request.max_tokens < rc.model_context_window
    measured = rc.model_context_window - 400
    provider = _CountingProvider(count=measured)

    fitted = await fit_request_to_context_measured(request, rc, provider)

    assert len(provider.counted) == 1
    assert fitted.measured == measured
    assert fitted.estimate == estimate
    # The estimate alone would have left the cap untouched; the count clips it.
    assert fit_request_to_context(request, rc).max_tokens == request.max_tokens
    assert fitted.request.max_tokens == 400


@pytest.mark.asyncio
async def test_a_count_that_fills_the_window_refuses_the_request_locally() -> None:
    rc = _rc()
    request = _near_edge_request(rc)
    provider = _CountingProvider(count=rc.model_context_window * 2)
    with pytest.raises(LLMContextWindowExceeded):
        await fit_request_to_context_measured(request, rc, provider)


@pytest.mark.asyncio
async def test_a_failed_count_falls_back_to_the_estimate_with_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    rc = _rc()
    request = _near_edge_request(rc)
    provider = _CountingProvider(fail=RuntimeError("tokenizer down"))
    with caplog.at_level(logging.WARNING, logger="protocore.runtime.request_budget"):
        fitted = await fit_request_to_context_measured(request, rc, provider)
    assert fitted.measured is None
    assert fitted.request.model_dump_json() == fit_request_to_context(request, rc).model_dump_json()
    assert any("tokenizer down" in record.getMessage() for record in caplog.records)
    assert all(record.levelno == logging.WARNING for record in caplog.records)


@pytest.mark.asyncio
async def test_an_unusable_count_is_ignored() -> None:
    rc = _rc()
    request = _near_edge_request(rc)
    for bad in (0, -5, True):
        provider = _CountingProvider(count=bad)  # type: ignore[arg-type]
        assert await count_request_tokens_exactly(request, provider, rc) is None


@pytest.mark.asyncio
async def test_the_switch_turns_counting_off() -> None:
    rc = _rc(exact_token_count_enabled=False)
    provider = _CountingProvider(count=10)
    assert await count_request_tokens_exactly(_near_edge_request(rc), provider, rc) is None
    assert provider.counted == []


@pytest.mark.asyncio
async def test_a_request_is_counted_once_whatever_its_output_cap() -> None:
    rc = _rc(exact_token_count_cache_max_entries=2)
    provider = _CountingProvider(count=123)
    cache = ExactTokenCountCache()
    request = _near_edge_request(rc)
    assert await count_request_tokens_exactly(request, provider, rc, cache=cache) == 123
    smaller = request.model_copy(update={"max_tokens": 10, "temperature": 0.1})
    assert await count_request_tokens_exactly(smaller, provider, rc, cache=cache) == 123
    assert len(provider.counted) == 1
    other = request.model_copy(update={"messages": [_msg("y" * 10)]})
    await count_request_tokens_exactly(other, provider, rc, cache=cache)
    third = request.model_copy(update={"messages": [_msg("z" * 10)]})
    await count_request_tokens_exactly(third, provider, rc, cache=cache)
    assert len(cache) == 2
    assert len(provider.counted) == 3


def _turn_rc(**overrides: Any) -> LoopConstants:
    values: dict[str, Any] = {
        "model_context_window": 65_536,
        "llm_output_max_tokens_ratio": 0.125,
        "compaction_keep_recent_turns": 1,
    }
    values.update(overrides)
    return LoopConstants(**values)


def _install(engine: Any, provider: _Provider) -> None:
    engine.llm = provider
    engine.context_manager._compaction_llm = provider
    engine.compaction_llm = provider


@pytest.mark.asyncio
async def test_a_provider_without_the_capability_sends_the_same_requests(
    engine_factory: Any,
) -> None:
    """Near the edge, a counter that has nothing to say changes nothing on the wire."""
    rc = _turn_rc(exact_token_count_margin_ratio=1.0)
    plain = _Provider()
    silent = _CountingProvider(count=None)
    requests: list[str] = []
    for provider in (plain, silent):
        engine = engine_factory(rc=rc)
        _install(engine, provider)
        async for _ in engine.run(_msg("f" * 4_000)):
            pass
        assert engine.state is LoopState.COMPLETED
        # Messages carry their creation time, which is the one thing two runs
        # cannot share; everything else on the wire must be the same.
        requests.append(
            re.sub(r'"created_at":"[^"]*"', "", provider.calls[0].model_dump_json())
        )
    assert silent.counted, "the margin of 1.0 asks on every request"
    assert requests[0] == requests[1]


@pytest.mark.asyncio
async def test_a_turn_near_the_edge_is_fitted_to_the_count_and_calibrates(
    engine_factory: Any,
) -> None:
    rc = _turn_rc(exact_token_count_margin_ratio=1.0)
    engine = engine_factory(rc=rc)
    provider = _CountingProvider(count=rc.model_context_window - rc.request_context_safety_tokens - 1_000)
    _install(engine, provider)

    async for _ in engine.run(_msg("0123456789abcdef" * 250)):
        pass

    assert engine.state is LoopState.COMPLETED
    assert provider.calls[0].max_tokens == 1_000
    # An exact count sets the factor outright rather than averaging towards it.
    assert engine.config.rc.token_estimate_calibration == 4.0
    assert engine.context_manager._rc.token_estimate_calibration == 4.0


@pytest.mark.asyncio
async def test_a_rejection_for_length_raises_calibration_to_the_proven_floor(
    engine_factory: Any,
) -> None:
    rc = _turn_rc()
    engine = engine_factory(rc=rc)
    provider = _Provider(exceptions=[LLMContextWindowExceeded("prompt contains at least N input tokens")])
    _install(engine, provider)

    async for _ in engine.run(_msg("0123456789abcdef" * 1_000)):
        pass

    rejected = provider.calls[0]
    raw = round(
        estimate_request_prompt_tokens(rejected, rc) / rc.token_estimate_calibration
    )
    floor = rc.model_context_window - rejected.max_tokens + 1
    expected = round(min(floor / raw, 4.0), 3)
    assert expected > 1.0
    assert engine.config.rc.token_estimate_calibration == expected


@pytest.mark.asyncio
async def test_a_quoted_prompt_size_is_used_as_the_floor(engine_factory: Any) -> None:
    rc = _turn_rc()
    engine = engine_factory(rc=rc)
    provider = _Provider(
        exceptions=[
            LLMContextWindowExceeded(
                "too long", context_window=65_536, input_tokens=60_000, requested_output_tokens=8_192
            )
        ]
    )
    _install(engine, provider)
    async for _ in engine.run(_msg("0123456789abcdef" * 1_000)):
        pass
    rejected = provider.calls[0]
    raw = round(estimate_request_prompt_tokens(rejected, rc) / rc.token_estimate_calibration)
    assert engine.config.rc.token_estimate_calibration == round(min(60_000 / raw, 4.0), 3)


@pytest.mark.asyncio
async def test_a_local_refusal_is_not_read_as_provider_evidence(
    engine_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    rc = _turn_rc()
    engine = engine_factory(rc=rc)
    provider = _Provider()
    _install(engine, provider)
    monkeypatch.setattr(
        "protocore.runtime.request_budget.estimate_request_prompt_tokens",
        lambda request, constants: constants.model_context_window,
    )
    async for _ in engine.run(_msg("hello")):
        pass
    assert provider.calls == []
    assert engine.config.rc.token_estimate_calibration == 1.0


@pytest.mark.asyncio
async def test_calibration_learned_from_a_rejection_survives_resume(
    engine_factory: Any,
) -> None:
    rc = _turn_rc()
    source = engine_factory(rc=rc)
    provider = _Provider(exceptions=[LLMContextWindowExceeded("too long")])
    _install(source, provider)
    async for _ in source.run(_msg("0123456789abcdef" * 1_000)):
        pass
    learned = source.config.rc.token_estimate_calibration
    assert learned > 1.0

    resumed = engine_factory(rc=rc)
    await resumed.resume_from_snapshot(source.snapshot())

    assert resumed.config.rc.token_estimate_calibration == learned


@pytest.mark.asyncio
async def test_the_gate_reads_the_count_near_the_trigger(engine_factory: Any) -> None:
    rc = _turn_rc(exact_token_count_margin_ratio=0.5)
    engine = engine_factory(rc=rc)
    provider = _CountingProvider(count=rc.model_context_window)
    _install(engine, provider)
    engine.history.append(_msg("0123456789abcdef" * 6_000))
    assert not engine.needs_compaction()

    await _calibrate_near_compaction_trigger(engine)

    assert len(provider.counted) == 1
    assert provider.counted[0].tools == []
    assert engine.needs_compaction()


@pytest.mark.asyncio
async def test_the_gate_asks_nothing_far_from_the_trigger(engine_factory: Any) -> None:
    rc = _turn_rc(exact_token_count_margin_ratio=0.1)
    engine = engine_factory(rc=rc)
    provider = _CountingProvider(count=rc.model_context_window)
    _install(engine, provider)
    engine.history.append(_msg("short"))

    await _calibrate_near_compaction_trigger(engine)

    assert provider.counted == []
    assert engine.config.rc.token_estimate_calibration == 1.0


class _FitOnlyCounter(_CountingProvider):
    """Counts only the requests the fit sizes — the built ones, which carry observability.

    The compaction gate counts the bare history; leaving it unanswered isolates
    what the fit does with a count that proves the request cannot fit.
    """

    def __init__(self, *, ratio: float, rc: LoopConstants) -> None:
        super().__init__()
        self.ratio = ratio
        self.rc = rc

    async def count_request_tokens(self, request: LLMRequest) -> int | None:
        if request.observability is None:
            return None
        self.counted.append(request)
        return round(estimate_request_prompt_tokens_uncalibrated(request, self.rc) * self.ratio)


@pytest.mark.asyncio
async def test_a_count_that_proves_overflow_calibrates_before_the_refusal(
    engine_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    rc = _turn_rc()
    engine = engine_factory(rc=rc)
    provider = _FitOnlyCounter(ratio=3.5, rc=rc)
    _install(engine, provider)
    compactions = 0
    force_compaction = engine.context_manager.force_compaction

    async def tracked(**kwargs: Any) -> Any:
        nonlocal compactions
        compactions += 1
        return await force_compaction(**kwargs)

    monkeypatch.setattr(engine.context_manager, "force_compaction", tracked)
    # Earlier turns of this session read dense tool output: hexadecimal, which
    # the heuristic undercounts by about 3.5x against a real tokenizer.
    engine.history.append(_msg("dump the device registers"))
    for i in range(12):
        engine.history.append(
            Message(
                role=MessageRole.assistant,
                content_blocks=[
                    ToolUseBlock(tool_call_id=f"t{i}", name="Bash", arguments_json='{"cmd":"xxd"}')
                ],
            )
        )
        engine.history.append(
            Message(
                role=MessageRole.tool,
                content_blocks=[ToolResultBlock(tool_call_id=f"t{i}", content=secrets.token_hex(3_500))],
            )
        )
    engine.history.append(_msg("done", MessageRole.assistant))

    async for _ in engine.run(_msg(secrets.token_hex(1_000))):
        pass

    counted = provider.counted[0]
    raw = estimate_request_prompt_tokens_uncalibrated(counted, rc)
    measured = round(raw * 3.5)
    assert measured >= rc.model_context_window - rc.request_context_safety_tokens
    # The factor is the counted ratio, set before the fit refused the request —
    # not the 1.0 the refusal used to leave behind.
    assert engine.config.rc.token_estimate_calibration == round(measured / raw, 3)
    # The refusal went to the same recovery a provider rejection gets.
    assert compactions >= 1
    # And the recovery, sized in the counted tokens, got the turn through.
    assert engine.state is LoopState.COMPLETED
    assert provider.calls


class _SlowCounter(_CountingProvider):
    async def count_request_tokens(self, request: LLMRequest) -> int | None:
        self.counted.append(request)
        await asyncio.sleep(3600)
        return 1


@pytest.mark.asyncio
async def test_a_counter_that_does_not_answer_in_time_falls_back(
    caplog: pytest.LogCaptureFixture,
) -> None:
    rc = _rc(exact_token_count_timeout_seconds=0.01)
    request = _near_edge_request(rc)
    with caplog.at_level(logging.WARNING, logger="protocore.runtime.request_budget"):
        fitted = await fit_request_to_context_measured(request, rc, _SlowCounter())
    assert fitted.measured is None
    assert fitted.request.model_dump_json() == fit_request_to_context(request, rc).model_dump_json()
    assert any("exact request token count failed" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_the_drift_line_is_written_once_per_count_not_per_cache_hit(
    caplog: pytest.LogCaptureFixture,
) -> None:
    rc = _rc()
    request = _near_edge_request(rc)
    provider = _CountingProvider(count=1_000)
    cache = ExactTokenCountCache()
    with caplog.at_level(logging.WARNING, logger="protocore.runtime.request_budget"):
        for _ in range(3):
            await fit_request_to_context_measured(request, rc, provider, cache=cache)
    drift = [r for r in caplog.records if "request_budget.exact_count" in r.getMessage()]
    assert len(provider.counted) == 1
    assert len(drift) == 1


@pytest.mark.asyncio
async def test_after_the_fit_has_counted_the_gate_does_not_count_again(engine_factory: Any) -> None:
    rc = _turn_rc()
    engine = engine_factory(rc=rc)
    provider = _CountingProvider(count=rc.model_context_window // 2)
    _install(engine, provider)
    engine.history.append(_msg("0123456789abcdef" * 6_000))

    await _calibrate_near_compaction_trigger(engine)
    assert len(provider.counted) == 1
    engine._exact_count_model = engine.effective_model_name
    await _calibrate_near_compaction_trigger(engine)
    assert len(provider.counted) == 1


def test_the_default_margin_covers_the_largest_undercount_calibration_can_express() -> None:
    rc = LoopConstants()
    ceiling = LoopConstants.model_fields["token_estimate_calibration"].metadata
    largest = max(getattr(m, "le", 0) or 0 for m in ceiling)
    assert rc.exact_token_count_margin_ratio >= 1 - 1 / largest


def _conversation(*parts: str) -> LLMRequest:
    return LLMRequest(model="m", messages=[_msg(part) for part in parts], max_tokens=1_000)


@pytest.mark.asyncio
async def test_after_a_count_small_additions_are_not_counted_again() -> None:
    rc = _rc(model_context_window=32_768)
    provider = _CountingProvider(count=6_000)
    cache = ExactTokenCountCache()
    parts = ["Прочитанный фрагмент книги. " * 900]
    await fit_request_to_context_measured(_conversation(*parts), rc, provider, cache=cache)
    assert len(provider.counted) == 1
    for _ in range(5):
        parts.append("Ещё абзац прозы из поиска. " * 20)
        await fit_request_to_context_measured(_conversation(*parts), rc, provider, cache=cache)
    assert len(provider.counted) == 1


@pytest.mark.asyncio
async def test_a_large_dense_addition_is_counted() -> None:
    rc = _rc(model_context_window=32_768)
    provider = _CountingProvider(count=6_000)
    cache = ExactTokenCountCache()
    parts = ["Прочитанный фрагмент книги. " * 900]
    await fit_request_to_context_measured(_conversation(*parts), rc, provider, cache=cache)
    parts.append(secrets.token_hex(16_000))
    provider.count = 30_000
    fitted = await fit_request_to_context_measured(_conversation(*parts), rc, provider, cache=cache)
    assert len(provider.counted) == 2
    assert fitted.measured == 30_000


@pytest.mark.asyncio
async def test_a_failed_count_backs_off_for_the_configured_period() -> None:
    rc = _rc(exact_token_count_timeout_seconds=0.01, exact_token_count_failure_backoff_seconds=3600)
    provider = _SlowCounter()
    cache = ExactTokenCountCache()
    request = _near_edge_request(rc)
    for _ in range(3):
        fitted = await fit_request_to_context_measured(request, rc, provider, cache=cache)
        assert fitted.measured is None
    assert len(provider.counted) == 1

    cache.backoff_until = 0.0
    await fit_request_to_context_measured(request, rc, provider, cache=cache)
    assert len(provider.counted) == 2


@pytest.mark.asyncio
async def test_a_zero_backoff_retries_on_the_next_request() -> None:
    rc = _rc(exact_token_count_failure_backoff_seconds=0)
    provider = _CountingProvider(fail=RuntimeError("down"))
    cache = ExactTokenCountCache()
    for _ in range(2):
        await fit_request_to_context_measured(_near_edge_request(rc), rc, provider, cache=cache)
    assert len(provider.counted) == 2


class _ContentAwareCounter(_CountingProvider):
    """Real size by content: prose 0.84x the heuristic, hexadecimal 3.5x."""

    def __init__(self, rc: LoopConstants) -> None:
        super().__init__()
        self.rc = rc

    async def count_request_tokens(self, request: LLMRequest) -> int | None:
        self.counted.append(request)
        return _real_size(request, self.rc)


def _real_size(request: LLMRequest, rc: LoopConstants) -> int:
    total = 0.0
    for message in request.messages:
        raw = estimate_request_prompt_tokens_uncalibrated(
            LLMRequest(model=request.model, messages=[message]), rc
        )
        text = message.content_blocks[0].text  # type: ignore[union-attr]
        dense = all(ch in "0123456789abcdef" for ch in text)
        total += raw * (3.5 if dense else 0.84)
    return round(total)


def _prose(i: int) -> Message:
    return _msg(f"paragraph {i} " + "plain library prose " * 420)


@pytest.mark.parametrize(
    ("kept", "hex_chars"),
    [
        # Compaction replaced most of the prose; the request got SMALLER by the
        # heuristic, and a hex result arrived in the same breath.
        (8, 80_000),
        # Fewer messages replaced, and the net growth is a few hundred tokens.
        (13, 60_000),
    ],
)
@pytest.mark.asyncio
async def test_dense_content_arriving_after_a_rewrite_is_counted(kept: int, hex_chars: int) -> None:
    rc = _rc(model_context_window=65_536)
    provider = _ContentAwareCounter(rc)
    cache = ExactTokenCountCache()
    before = LLMRequest(model="m", messages=[_prose(i) for i in range(20)], max_tokens=16_384)
    await fit_request_to_context_measured(before, rc, provider, cache=cache)
    assert len(provider.counted) == 1

    after = LLMRequest(
        model="m",
        messages=[
            _msg("Summary of the earlier reading."),
            *[_prose(i) for i in range(20 - kept, 20)],
            _msg(secrets.token_hex(hex_chars // 2)),
        ],
        max_tokens=16_384,
    )
    raw_before = estimate_request_prompt_tokens_uncalibrated(before, rc)
    raw_after = estimate_request_prompt_tokens_uncalibrated(after, rc)
    if kept == 8:
        assert raw_after < raw_before
    else:
        assert 0 < raw_after - raw_before < 1_000
    assert _real_size(after, rc) + after.max_tokens > rc.model_context_window

    try:
        await fit_request_to_context_measured(after, rc, provider, cache=cache)
    except LLMContextWindowExceeded:
        pass

    assert len(provider.counted) == 2


@pytest.mark.asyncio
async def test_a_rewrite_that_adds_nothing_unseen_is_not_counted_again() -> None:
    rc = _rc(model_context_window=65_536)
    provider = _ContentAwareCounter(rc)
    cache = ExactTokenCountCache()
    messages = [_prose(i) for i in range(20)]
    await fit_request_to_context_measured(
        LLMRequest(model="m", messages=messages, max_tokens=16_384), rc, provider, cache=cache
    )
    # Eviction dropped the oldest half; nothing new arrived.
    await fit_request_to_context_measured(
        LLMRequest(model="m", messages=messages[10:], max_tokens=16_384), rc, provider, cache=cache
    )
    assert len(provider.counted) == 1


@pytest.mark.asyncio
async def test_moving_cache_breakpoints_do_not_make_the_tools_look_new() -> None:
    rc = _rc(model_context_window=65_536)
    provider = _ContentAwareCounter(rc)
    cache = ExactTokenCountCache()
    tools = [
        ToolDefinition(
            name=f"tool_{i}",
            description="Searches the library. " * 200,
            parameters=ToolParameterSchema(type="object", properties={"q": {"type": "string"}}),
        )
        for i in range(12)
    ]
    messages = [_prose(i) for i in range(10)]
    for iteration in range(6):
        messages.append(_msg(f"search result {iteration} " + "short prose " * 20))
        request = LLMRequest(
            model="m",
            messages=list(messages),
            tools=tools,
            max_tokens=16_384,
            extra={
                "cache_breakpoints": [{"message_index": len(messages) - 1, "cache_control_type": "ephemeral"}],
                "forced_tool_choice": "tool_0" if iteration % 2 else None,
                "enable_thinking": False,
            },
        )
        await fit_request_to_context_measured(request, rc, provider, cache=cache)
    assert len(provider.counted) == 1


@pytest.mark.asyncio
async def test_a_rendering_option_still_makes_the_frame_new() -> None:
    rc = _rc(model_context_window=65_536)
    provider = _ContentAwareCounter(rc)
    cache = ExactTokenCountCache()
    messages = [_prose(i) for i in range(20)]
    for thinking in (False, True):
        request = LLMRequest(
            model="m", messages=messages, max_tokens=16_384, extra={"enable_thinking": thinking}
        )
        await fit_request_to_context_measured(request, rc, provider, cache=cache)
    assert request_digests(request).frame != request_digests(
        request.model_copy(update={"extra": {"enable_thinking": False}})
    ).frame
