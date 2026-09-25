from __future__ import annotations

import pytest

from protocore.contracts.llm import LLMContextWindowExceeded, LLMRequest
from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.types import (
    Message,
    MessageRole,
    TextBlock,
    ToolDefinition,
    ToolParameterSchema,
)
from protocore.runtime.context.compaction import estimate_history_tokens_uncalibrated
from protocore.runtime.events import EventType
from protocore.runtime.loop_state import LoopState
from protocore.runtime.request_budget import (
    estimate_request_prompt_tokens,
    fit_max_tokens,
    fit_request_to_context,
)
from protocore.runtime.tool_surface import read_tool_surface, tool_surface_tokens


@pytest.mark.parametrize(
    ("requested_max_tokens", "expected"),
    [(9, 9), (10, 10), (11, 10)],
)
def test_fit_max_tokens_accepts_boundary_and_clips_only_overflow(
    requested_max_tokens: int,
    expected: int,
) -> None:
    assert (
        fit_max_tokens(
            prompt_tokens=90,
            requested_max_tokens=requested_max_tokens,
            context_window=100,
        )
        == expected
    )


def test_fit_max_tokens_rejects_prompt_when_even_one_output_token_cannot_fit() -> None:
    with pytest.raises(LLMContextWindowExceeded):
        fit_max_tokens(
            prompt_tokens=100,
            requested_max_tokens=1,
            context_window=100,
        )


def test_fit_max_tokens_keeps_provider_framing_margin_unused() -> None:
    assert (
        fit_max_tokens(
            prompt_tokens=57_344,
            requested_max_tokens=8_192,
            context_window=65_536,
            safety_tokens=64,
        )
        == 8_128
    )


def test_default_margin_keeps_provider_framing_space_unused() -> None:
    assert (
        fit_max_tokens(
            prompt_tokens=57_344,
            requested_max_tokens=8_192,
            context_window=65_536,
            safety_tokens=LoopConstants().request_context_safety_tokens,
        )
        == 7_168
    )


def test_default_margin_absorbs_additive_provider_framing_drift() -> None:
    estimated_prompt_tokens = 56_832
    provider_prompt_lower_bound = 57_345
    fitted = fit_max_tokens(
        prompt_tokens=estimated_prompt_tokens,
        requested_max_tokens=8_192,
        context_window=65_536,
        safety_tokens=LoopConstants().request_context_safety_tokens,
    )

    assert fitted == 7_680
    assert provider_prompt_lower_bound + fitted <= 65_536


def test_request_fitting_wires_default_margin_into_the_hard_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = LLMRequest(
        model="test-model",
        messages=[Message(role=MessageRole.user, content_blocks=[TextBlock(text="hello")])],
        max_tokens=8_192,
    )
    monkeypatch.setattr(
        "protocore.runtime.request_budget.estimate_request_prompt_tokens",
        lambda request, constants: 56_832,
    )

    fitted = fit_request_to_context(
        request,
        LoopConstants(model_context_window=65_536),
    )

    assert fitted.max_tokens == 7_680
    assert 57_345 + fitted.max_tokens <= 65_536


def test_fit_max_tokens_rejects_prompt_that_fills_usable_window() -> None:
    with pytest.raises(LLMContextWindowExceeded):
        fit_max_tokens(
            prompt_tokens=65_472,
            requested_max_tokens=1,
            context_window=65_536,
            safety_tokens=64,
        )


def test_estimate_covers_calibrated_messages_and_tools() -> None:
    rc = LoopConstants(model_context_window=8_192, token_estimate_calibration=2.0)
    message = Message(
        role=MessageRole.user,
        content_blocks=[TextBlock(text="calibrated prompt")],
    )
    tool = ToolDefinition(
        name="Read",
        description="read a named resource",
        parameters=ToolParameterSchema(properties={"path": {"type": "string"}}),
    )
    request = LLMRequest(
        model="test-model",
        messages=[message],
        tools=[tool],
        max_tokens=100,
    )
    raw = estimate_history_tokens_uncalibrated([message], rc)
    raw += tool_surface_tokens(read_tool_surface([tool]), rc)

    assert estimate_request_prompt_tokens(request, rc) == round(raw * 2.0)


def test_fit_request_returns_copy_with_hard_ceiling() -> None:
    base_rc = LoopConstants(model_context_window=8_192)
    request = LLMRequest(
        model="test-model",
        messages=[Message(role=MessageRole.user, content_blocks=[TextBlock(text="hello")])],
        max_tokens=50,
    )
    prompt_tokens = estimate_request_prompt_tokens(request, base_rc)
    rc = base_rc.model_copy(
        update={
            "model_context_window": prompt_tokens + 10,
            "request_context_safety_tokens": 0,
        }
    )

    fitted = fit_request_to_context(request, rc)

    assert fitted.max_tokens == 10
    assert request.max_tokens == 50


def test_fit_request_applies_configured_provider_margin() -> None:
    base_rc = LoopConstants(model_context_window=8_192)
    request = LLMRequest(
        model="test-model",
        messages=[Message(role=MessageRole.user, content_blocks=[TextBlock(text="hello")])],
        max_tokens=50,
    )
    prompt_tokens = estimate_request_prompt_tokens(request, base_rc)
    rc = base_rc.model_copy(
        update={
            "model_context_window": prompt_tokens + 10,
            "request_context_safety_tokens": 7,
        }
    )

    assert fit_request_to_context(request, rc).max_tokens == 3


def test_constants_reject_margin_that_consumes_the_context_window() -> None:
    with pytest.raises(ValueError, match="request_context_safety_tokens"):
        LoopConstants(model_context_window=64, request_context_safety_tokens=64)


def test_hard_ceiling_clips_a_larger_terminal_reserve() -> None:
    base_rc = LoopConstants(model_context_window=8_192)
    reserved_request = LLMRequest(
        model="test-model",
        messages=[Message(role=MessageRole.user, content_blocks=[TextBlock(text="hello")])],
        max_tokens=40,
    )
    prompt_tokens = estimate_request_prompt_tokens(reserved_request, base_rc)
    rc = base_rc.model_copy(
        update={
            "model_context_window": prompt_tokens + 10,
            "request_context_safety_tokens": 0,
        }
    )

    fitted = fit_request_to_context(reserved_request, rc)

    assert fitted.max_tokens == 10


def test_history_only_gates_ignore_stale_provider_prompt_count(
    engine_factory,
) -> None:
    engine = engine_factory(rc=LoopConstants(model_context_window=4_096))
    engine.history.append(Message(role=MessageRole.user, content_blocks=[TextBlock(text="small")]))
    engine.last_observed_prompt_tokens = 100_000

    assert engine.needs_compaction() is False
    assert engine.needs_emergency_compaction() is False


@pytest.mark.asyncio
async def test_local_overflow_compacts_once_without_calling_provider(
    engine_factory,
    in_memory_runtime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rc = LoopConstants(model_context_window=4_096, compaction_keep_recent_turns=1)
    engine = engine_factory(rc=rc)
    force_compaction = engine.context_manager.force_compaction
    compaction_calls = 0

    async def tracked_force_compaction(**kwargs):  # type: ignore[no-untyped-def]
        nonlocal compaction_calls
        compaction_calls += 1
        return await force_compaction(**kwargs)

    monkeypatch.setattr(
        engine.context_manager,
        "force_compaction",
        tracked_force_compaction,
    )
    monkeypatch.setattr(
        "protocore.runtime.request_budget.estimate_request_prompt_tokens",
        lambda request, constants: (
            constants.model_context_window - constants.request_context_safety_tokens
        ),
    )

    events = [
        event async for event in engine.run(Message(role=MessageRole.user, content_blocks=[TextBlock(text="hello")]))
    ]

    llm = in_memory_runtime["llm"]
    assert compaction_calls == 1
    assert not llm.calls  # type: ignore[union-attr]
    assert engine.state is LoopState.FAILED
    assert [event.payload["kind"] for event in events if event.type is EventType.ERROR][
        -1
    ] == "llm_context_window_exceeded"
