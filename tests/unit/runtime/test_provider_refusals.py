"""A provider that refuses the request for good.

The live shape: the provider answered every attempt with a 400
``invalid_request_error`` — the client was too old for the model. A failure
the adapter marks as permanent, by the flag on the classification it attaches
or by the reason when it sets no flag, is not retried in place and does not
wind down against the endpoint that refused it; it may still step to the next
model of the chain, which is bounded and one-way.
"""
from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from protocore.contracts.llm import (
    LLMProviderError,
    LLMRateLimitError,
    LLMRequest,
    LLMStreamEvent,
)
from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.types import Message, MessageRole, StopReason, TextBlock
from protocore.runtime.events import EventType, TurnEvent
from protocore.runtime.loop_state import LoopState

_REFUSAL = (
    "claude: HTTP 400: Client 1.0 does not support this model; "
    "version 1.2 or newer is required. (client_version_too_old)"
)


class _Verdict:
    """The classification an adapter pins on the error it raises."""

    def __init__(self, reason: str, retryable: bool | None = None) -> None:
        self.reason = reason
        if retryable is not None:
            self.retryable = retryable


def _classified(
    exc: BaseException, reason: str, retryable: bool | None = None
) -> BaseException:
    object.__setattr__(exc, "classified", _Verdict(reason, retryable))
    return exc


class _ScriptedLLM:
    """Raise ``failures[i]`` on call ``i``; a call past the script answers."""

    def __init__(
        self,
        failures: list[BaseException | None],
        *,
        prose_first: str = "",
        text: str = "the answer",
    ) -> None:
        self._failures = failures
        self._prose_first = prose_first
        self._text = text
        self.calls: list[LLMRequest] = []

    async def stream_with_tools(  # type: ignore[no-untyped-def]
        self, request: LLMRequest
    ) -> AsyncIterator[LLMStreamEvent]:
        index = len(self.calls)
        self.calls.append(request)
        if index == 0 and self._prose_first:
            async for event in self._answer(self._prose_first):
                yield event
            return
        if index < len(self._failures) and self._failures[index] is not None:
            raise self._failures[index]  # type: ignore[misc]
        async for event in self._answer(self._text):
            yield event

    @staticmethod
    async def _answer(text: str) -> AsyncIterator[LLMStreamEvent]:
        yield LLMStreamEvent(name="message_start", payload={})
        yield LLMStreamEvent(name="content_block_start", payload={"kind": "text"})
        yield LLMStreamEvent(name="content_block_delta", payload={"text": text})
        yield LLMStreamEvent(name="content_block_stop", payload={})
        yield LLMStreamEvent(name="message_stop", payload={"stop_reason": "end_turn"})

    async def complete_structured(self, request, schema):  # type: ignore[no-untyped-def]
        from protocore.contracts.llm import LLMResponse

        return LLMResponse(
            message=Message(role=MessageRole.assistant, content_blocks=[]),
            stop_reason=StopReason.end_turn,
        )

    def count_tokens(self, text, model=None) -> int:  # type: ignore[no-untyped-def]
        return max(1, len(text) // 4)


class _Chain:
    """A one-way chain over one scripted provider under several model names."""

    def __init__(self, provider: object, names: list[str]) -> None:
        self._provider = provider
        self._names = names
        self._index = 0

    def current(self) -> object:
        return self._provider

    def current_model_name(self) -> str:
        return self._names[self._index]

    async def advance(self, *, reason: str) -> bool:
        if self._index + 1 >= len(self._names):
            return False
        self._index += 1
        return True

    def attempted(self) -> list[tuple[str, str]]:
        return []


def _rc(**overrides: object) -> LoopConstants:
    base: dict[str, object] = {
        "model_context_window": 4096,
        "llm_transient_error_retry_backoff_base_seconds": 0.0,
        "llm_transient_error_retry_backoff_max_seconds": 0.0,
        "llm_transient_error_retry_max_attempts": 2,
    }
    base.update(overrides)
    return LoopConstants(**base)  # type: ignore[arg-type]


def _user(text: str) -> Message:
    return Message(role=MessageRole.user, content_blocks=[TextBlock(text=text)])


def _assistant(text: str) -> Message:
    return Message(role=MessageRole.assistant, content_blocks=[TextBlock(text=text)])


async def _drive(engine, text: str = "what changed overnight?") -> list[TurnEvent]:
    return [event async for event in engine.run(_user(text))]


def _reasons(events: list[TurnEvent]) -> list[str]:
    return [
        str(e.payload.get("reason"))
        for e in events
        if e.type is EventType.STATE_CHANGED
    ]


def _errors(events: list[TurnEvent]) -> list[dict[str, object]]:
    return [e.payload for e in events if e.type is EventType.ERROR]


def _terminal_stop(events: list[TurnEvent]) -> dict[str, object]:
    stops = [e.payload for e in events if e.type is EventType.MESSAGE_STOP]
    assert stops, "the run never said it stopped"
    return stops[-1]


# -- a refusal is not a blip ---------------------------------------------


@pytest.mark.parametrize(
    "verdict",
    [
        _Verdict("format_error", retryable=False),
        _Verdict("model_not_found"),
        _Verdict("auth"),
        _Verdict("server_error", retryable=False),
    ],
    ids=["retryable-false", "model-not-found", "auth", "flag-wins-over-reason"],
)
@pytest.mark.asyncio
async def test_a_permanent_refusal_is_not_retried_and_fails_with_its_words(
    engine_factory, in_memory_runtime, verdict: _Verdict
) -> None:
    engine = engine_factory(rc=_rc())
    error = LLMProviderError(_REFUSAL)
    object.__setattr__(error, "classified", verdict)
    llm = _ScriptedLLM([error, error, error])
    engine.llm = llm  # type: ignore[assignment]

    events = await _drive(engine)

    assert len(llm.calls) == 1
    assert "transient_llm_error_retry" not in _reasons(events)
    assert "provider_error_wind_down" not in _reasons(events)
    assert engine.state is LoopState.FAILED
    errors = _errors(events)
    assert errors and "does not support this model" in str(errors[-1]["message"])
    assert _terminal_stop(events)["stop_reason"] == StopReason.error.value


@pytest.mark.parametrize(
    "error",
    [
        LLMRateLimitError("429 slow down"),
        _classified(LLMProviderError("HTTP 503: upstream down"), "server_error", True),
        _classified(LLMProviderError("HTTP 529: overloaded"), "overloaded"),
        LLMProviderError("transport error: connection reset"),
    ],
    ids=["rate-limit", "server-error", "overloaded", "unclassified"],
)
@pytest.mark.asyncio
async def test_a_transient_failure_is_still_retried(
    engine_factory, in_memory_runtime, error: BaseException
) -> None:
    engine = engine_factory(rc=_rc())
    llm = _ScriptedLLM([error])
    engine.llm = llm  # type: ignore[assignment]

    events = await _drive(engine)

    assert _reasons(events).count("transient_llm_error_retry") == 1
    assert len(llm.calls) == 2
    assert engine.state is LoopState.COMPLETED


@pytest.mark.asyncio
async def test_a_permanent_refusal_may_fall_back_once_per_rung_and_then_fails(
    engine_factory, in_memory_runtime
) -> None:
    """Each rung refuses; the run walks the chain once and stops, never loops."""
    engine = engine_factory(rc=_rc(llm_provider_chain_max_advances=5))
    refusals = [
        _classified(LLMProviderError(_REFUSAL), "model_not_found", False)
        for _ in range(6)
    ]
    llm = _ScriptedLLM(refusals)  # type: ignore[arg-type]
    chain = _Chain(llm, ["m0", "m1", "m2"])
    engine.llm = llm  # type: ignore[assignment]
    engine.provider_chain = chain

    events = await _drive(engine)

    assert len(llm.calls) == 3
    assert _reasons(events).count("model_fallback_triggered") == 2
    assert "transient_llm_error_retry" not in _reasons(events)
    assert engine.state is LoopState.FAILED
    assert engine.config.model_name == "m2"


@pytest.mark.asyncio
async def test_a_permanent_refusal_falls_back_to_a_rung_that_answers(
    engine_factory, in_memory_runtime
) -> None:
    engine = engine_factory(rc=_rc())
    llm = _ScriptedLLM([_classified(LLMProviderError(_REFUSAL), "model_not_found", False)])
    engine.llm = llm  # type: ignore[assignment]
    engine.provider_chain = _Chain(llm, ["m0", "m1"])

    await _drive(engine)

    assert engine.state is LoopState.COMPLETED
    assert engine.config.model_name == "m1"
    assert len(llm.calls) == 2
