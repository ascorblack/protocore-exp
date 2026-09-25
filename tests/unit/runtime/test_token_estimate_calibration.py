"""The token heuristic is scaled to the provider's count; an empty compaction pass backs the gate off."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import pytest

from protocore.contracts.llm import LLMRequest
from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.types import Message, MessageRole, TextBlock
from protocore.runtime.context.compaction import estimate_history_tokens, estimate_message_tokens
from protocore.runtime.events import EventType, TurnEvent
from protocore.runtime.loop_state import LoopState
from protocore.runtime.query import _calibrate_token_estimate
from protocore.runtime.request_budget import (
    estimate_request_prompt_tokens_uncalibrated,
    fit_request_to_context,
)
from protocore.runtime.turn_policies.compaction import PerIterationCompactionPolicy


def _msg(text: str) -> Message:
    return Message(role=MessageRole.user, content_blocks=[TextBlock(text=text)])


def test_calibration_scales_the_estimate_and_invalidates_the_cache() -> None:
    rc = LoopConstants()
    message = _msg("x" * 4000)
    base = estimate_message_tokens(message, rc)
    doubled = estimate_message_tokens(message, rc.model_copy(update={"token_estimate_calibration": 2.0}))
    assert base > 0 and doubled == 2 * base
    assert estimate_history_tokens([message, message], rc.model_copy(update={"token_estimate_calibration": 1.5})) == 2 * round(base * 1.5)


@dataclass
class _Manager:
    rc: LoopConstants
    updated: list[LoopConstants] = field(default_factory=list)

    def update_rc(self, rc: LoopConstants) -> None:
        self.rc = rc
        self.updated.append(rc)


@dataclass
class _Config:
    rc: LoopConstants


@dataclass
class _Engine:
    config: _Config
    context_manager: _Manager
    _token_estimate_calibration_baseline: float
    _token_estimate_calibration_model: str
    effective_model_name: str

    def set_token_estimate_calibration(
        self,
        factor: float,
        *,
        model_name: str,
    ) -> None:
        calibrated = self.config.rc.model_copy(
            update={"token_estimate_calibration": factor}
        )
        self.config.rc = calibrated
        self.context_manager.update_rc(calibrated)
        self._token_estimate_calibration_model = model_name


def _engine(rc: LoopConstants) -> _Engine:
    return _Engine(
        config=_Config(rc=rc),
        context_manager=_Manager(rc=rc),
        _token_estimate_calibration_baseline=rc.token_estimate_calibration,
        _token_estimate_calibration_model="m",
        effective_model_name="m",
    )


def test_calibration_moves_towards_the_provider_count_and_is_damped() -> None:
    rc = LoopConstants()
    request = LLMRequest(model="m", messages=[_msg("y" * 8000)], tools=[])
    raw = estimate_history_tokens(list(request.messages), rc)
    engine = _engine(rc)
    _calibrate_token_estimate(engine, request, observed=raw * 2)  # type: ignore[arg-type]
    factor = engine.config.rc.token_estimate_calibration
    assert 1.4 < factor < 1.6, factor  # half-way to 2.0 on the first observation
    assert engine.context_manager.rc is engine.config.rc
    _calibrate_token_estimate(engine, request, observed=raw * 2)  # type: ignore[arg-type]
    assert engine.config.rc.token_estimate_calibration > factor
    # A provider that counts fewer tokens than the heuristic never pulls the factor below 1.
    engine = _engine(rc)
    _calibrate_token_estimate(engine, request, observed=raw // 2)  # type: ignore[arg-type]
    assert engine.config.rc.token_estimate_calibration == 1.0 and engine.context_manager.updated == []


def test_calibration_can_be_switched_off() -> None:
    rc = LoopConstants(token_estimate_calibration_enabled=False)
    request = LLMRequest(model="m", messages=[_msg("y" * 8000)], tools=[])
    engine = _engine(rc)
    _calibrate_token_estimate(engine, request, observed=10**6)  # type: ignore[arg-type]
    assert engine.config.rc.token_estimate_calibration == 1.0


@pytest.mark.asyncio
async def test_same_model_resume_restores_conservative_calibration(
    engine_factory,
) -> None:
    rc = LoopConstants(
        model_context_window=1_000,
        request_context_safety_tokens=0,
        token_estimate_calibration=1.0,
    )
    source = engine_factory(model_name="m", rc=rc)
    request = LLMRequest(
        model="m",
        messages=[_msg("x" * 1_000)],
        max_tokens=500,
    )
    raw = estimate_request_prompt_tokens_uncalibrated(request, rc)
    _calibrate_token_estimate(source, request, observed=raw * 4)
    learned = source.config.rc.token_estimate_calibration
    resumed = engine_factory(model_name="m", rc=rc)

    await resumed.resume_from_snapshot(source.snapshot())

    assert resumed.config.rc.token_estimate_calibration == learned
    assert fit_request_to_context(request, resumed.config.rc).max_tokens < (
        fit_request_to_context(request, rc).max_tokens
    )


@pytest.mark.asyncio
async def test_different_model_resume_ignores_learned_calibration(
    engine_factory,
) -> None:
    rc = LoopConstants(token_estimate_calibration=1.25)
    source = engine_factory(model_name="m", rc=rc)
    request = LLMRequest(model="m", messages=[_msg("x" * 1_000)])
    raw = estimate_request_prompt_tokens_uncalibrated(request, rc)
    _calibrate_token_estimate(source, request, observed=raw * 4)
    snapshot = source.snapshot()
    snapshot["live_model_name"] = "other-model"
    resumed = engine_factory(model_name="m", rc=rc)

    await resumed.resume_from_snapshot(snapshot)

    assert resumed.effective_model_name == "other-model"
    assert resumed.config.rc.token_estimate_calibration == 1.25


@pytest.mark.asyncio
async def test_resume_ignores_learned_calibration_when_policy_is_disabled(
    engine_factory,
) -> None:
    source = engine_factory(model_name="m", rc=LoopConstants())
    source.set_token_estimate_calibration(3.0, model_name="m")
    destination_rc = LoopConstants(
        token_estimate_calibration=1.25,
        token_estimate_calibration_enabled=False,
    )
    resumed = engine_factory(model_name="m", rc=destination_rc)

    await resumed.resume_from_snapshot(source.snapshot())

    assert resumed.config.rc.token_estimate_calibration == 1.25
    assert resumed._token_estimate_calibration_model == "m"


@pytest.mark.asyncio
async def test_resume_never_lowers_a_raised_destination_baseline(
    engine_factory,
) -> None:
    source = engine_factory(model_name="m", rc=LoopConstants())
    source.set_token_estimate_calibration(2.0, model_name="m")
    resumed = engine_factory(
        model_name="m",
        rc=LoopConstants(token_estimate_calibration=3.0),
    )

    await resumed.resume_from_snapshot(source.snapshot())

    assert resumed.config.rc.token_estimate_calibration == 3.0


def test_live_model_switch_resets_learned_calibration(engine_factory) -> None:
    rc = LoopConstants(token_estimate_calibration=1.25)
    engine = engine_factory(model_name="m", rc=rc)
    request = LLMRequest(model="m", messages=[_msg("x" * 1_000)])
    raw = estimate_request_prompt_tokens_uncalibrated(request, rc)
    _calibrate_token_estimate(engine, request, observed=raw * 4)
    assert engine.config.rc.token_estimate_calibration > 1.25

    engine.apply_live_controls(model_name="other-model")

    assert engine.config.rc.token_estimate_calibration == 1.25
    assert engine._token_estimate_calibration_model == "other-model"


def test_late_usage_from_previous_model_does_not_replace_active_binding(
    engine_factory,
) -> None:
    rc = LoopConstants(token_estimate_calibration=1.25)
    engine = engine_factory(model_name="old-model", rc=rc)
    old_request = LLMRequest(
        model="old-model",
        messages=[_msg("x" * 1_000)],
        max_tokens=500,
    )
    engine.apply_live_controls(model_name="new-model")

    _calibrate_token_estimate(engine, old_request, observed=100_000)

    assert engine.effective_model_name == "new-model"
    assert engine.config.rc.token_estimate_calibration == 1.25
    assert engine._token_estimate_calibration_model == "new-model"
    next_request = old_request.model_copy(update={"model": "new-model"})
    assert fit_request_to_context(next_request, engine.config.rc).max_tokens == 500


class _GateEngine:
    def __init__(self, rc: LoopConstants) -> None:
        self.rc = rc
        self.history: list[Message] = []
        self.state = LoopState.RUNNING
        self.compaction_backoff_left = 0
        self.compaction_backoff_prompt_tokens = 0
        self.prompt_tokens = 59_900
        self.needs = True

    def needs_emergency_compaction(self) -> bool:
        return False

    def needs_compaction(self) -> bool:
        return self.needs


@pytest.mark.asyncio
async def test_empty_routine_pass_backs_the_gate_off() -> None:
    rc = LoopConstants(compaction_no_gain_backoff_iterations=2, compaction_min_gain_ratio=0.03)
    engine = _GateEngine(rc)
    calls: list[str] = []

    async def compact(eng: Any, *, force: bool, reason: str, protect_tail_from_index: int | None) -> AsyncIterator[TurnEvent]:
        calls.append(reason)
        yield TurnEvent(type=EventType.COMPACTION_COMPLETED, run_id="r", payload={"tokens_before": 60_000, "tokens_after": 59_900})

    policy = PerIterationCompactionPolicy(
        compact=compact, protect_index=lambda h: None, prompt_tokens=lambda e: e.prompt_tokens
    )

    async def run() -> list[TurnEvent]:
        turn = type("Turn", (), {"engine": engine, "outcome": type("O", (), {"directive": None, "reason": None})()})()
        return [e async for e in policy.apply(turn)]  # type: ignore[arg-type]

    assert len(await run()) == 1 and calls == ["proactive_per_iteration"]
    assert engine.compaction_backoff_left == 2
    assert await run() == [] and await run() == [] and calls == ["proactive_per_iteration"]  # two skipped iterations
    assert len(await run()) == 1 and len(calls) == 2  # then it is consulted again


@pytest.mark.asyncio
async def test_prompt_growth_ends_the_no_gain_backoff_early() -> None:
    """A new large result makes the old gain reading worthless; the gate runs again."""
    rc = LoopConstants(
        compaction_no_gain_backoff_iterations=6,
        compaction_min_gain_ratio=0.03,
        compaction_no_gain_backoff_growth_ratio=0.1,
    )
    engine = _GateEngine(rc)
    calls: list[str] = []

    async def compact(eng: Any, *, force: bool, reason: str, protect_tail_from_index: int | None) -> AsyncIterator[TurnEvent]:
        calls.append(reason)
        yield TurnEvent(type=EventType.COMPACTION_COMPLETED, run_id="r", payload={"tokens_before": 60_000, "tokens_after": 59_900})

    policy = PerIterationCompactionPolicy(
        compact=compact, protect_index=lambda h: None, prompt_tokens=lambda e: e.prompt_tokens
    )

    async def run() -> list[TurnEvent]:
        turn = type("Turn", (), {"engine": engine, "outcome": type("O", (), {"directive": None, "reason": None})()})()
        return [e async for e in policy.apply(turn)]  # type: ignore[arg-type]

    await run()
    assert engine.compaction_backoff_left == 6
    assert engine.compaction_backoff_prompt_tokens == 59_900

    engine.prompt_tokens = 60_500  # grew, but by less than the ratio
    assert await run() == [] and len(calls) == 1
    assert engine.compaction_backoff_left == 5

    engine.prompt_tokens = 66_000  # a large result arrived
    assert len(await run()) == 1 and len(calls) == 2
