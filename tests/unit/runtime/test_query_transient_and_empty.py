"""Transient LLM-error recovery (429 / timeout) + empty-completion guard.

Covers two runtime-loop hardening behaviours in
:mod:`protocore.runtime.query`:

* A 429 (:class:`LLMRateLimitError`) or a request/stream timeout
  (:class:`LLMTimeoutError`) must get the SAME recovery a generic provider
  error gets — a step down the run's provider chain when one is configured,
  otherwise a bounded in-place backoff-retry — and only go terminal FAILED once
  both are unavailable/exhausted (preserving any already-delivered answer).
* A bare-empty ``finish_reason='stop'`` turn (no text, no tool calls, no
  reasoning) with no answer yet in history must NOT seal a silent empty
  COMPLETED — it is re-driven a bounded number of times and then fails loudly.
"""
from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from protocore.contracts.llm import (
    LLMProviderError,
    LLMRateLimitError,
    LLMRequest,
    LLMStreamEvent,
    LLMTimeoutError,
)
from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.types import Message, MessageRole, StopReason, TextBlock
from protocore.runtime.events import EventType, TurnEvent
from protocore.runtime.loop_state import LoopState

from ._tool_fixtures import MockTool

# ----------------------------------------------------------------------
# LLM mocks
# ----------------------------------------------------------------------


class _ScriptedFailureLLM:
    """Raise ``exceptions[i]`` on call ``i``; once exhausted emit a healthy turn."""

    def __init__(
        self,
        exceptions: list[BaseException | None],
        recovery_text: str = "recovered",
    ) -> None:
        self._exceptions = exceptions
        self._recovery_text = recovery_text
        self._call_idx = 0
        self.calls: list[LLMRequest] = []

    async def stream_with_tools(  # type: ignore[no-untyped-def]
        self, request: LLMRequest
    ) -> AsyncIterator[LLMStreamEvent]:
        self.calls.append(request)
        idx = self._call_idx
        self._call_idx += 1
        if idx < len(self._exceptions):
            exc = self._exceptions[idx]
            if exc is not None:
                raise exc
        yield LLMStreamEvent(name="message_start", payload={})
        yield LLMStreamEvent(name="content_block_start", payload={"kind": "text"})
        yield LLMStreamEvent(
            name="content_block_delta", payload={"text": self._recovery_text}
        )
        yield LLMStreamEvent(name="content_block_stop", payload={})
        yield LLMStreamEvent(
            name="message_stop", payload={"stop_reason": "end_turn"}
        )

    async def complete_structured(self, request, schema):  # type: ignore[no-untyped-def]
        from protocore.contracts.llm import LLMResponse

        return LLMResponse(
            message=Message(role=MessageRole.assistant, content_blocks=[]),
            stop_reason=StopReason.end_turn,
        )

    def count_tokens(self, text, model=None) -> int:  # type: ignore[no-untyped-def]
        return max(1, len(text) // 4)


class _PartialThenTransientLLM:
    """Stream partial text, then raise ``exception``; later calls recover."""

    def __init__(
        self,
        exception: BaseException,
        partial_text: str,
        recovery_text: str = "recovered",
    ) -> None:
        self._exception = exception
        self._partial_text = partial_text
        self._recovery_text = recovery_text
        self._call_idx = 0
        self.calls: list[LLMRequest] = []

    async def stream_with_tools(  # type: ignore[no-untyped-def]
        self, request: LLMRequest
    ) -> AsyncIterator[LLMStreamEvent]:
        self.calls.append(request)
        idx = self._call_idx
        self._call_idx += 1
        if idx == 0:
            yield LLMStreamEvent(name="message_start", payload={})
            yield LLMStreamEvent(
                name="content_block_start", payload={"kind": "text"}
            )
            yield LLMStreamEvent(
                name="content_block_delta", payload={"text": self._partial_text}
            )
            yield LLMStreamEvent(name="content_block_stop", payload={})
            raise self._exception
        yield LLMStreamEvent(name="message_start", payload={})
        yield LLMStreamEvent(name="content_block_start", payload={"kind": "text"})
        yield LLMStreamEvent(
            name="content_block_delta", payload={"text": self._recovery_text}
        )
        yield LLMStreamEvent(name="content_block_stop", payload={})
        yield LLMStreamEvent(
            name="message_stop", payload={"stop_reason": "end_turn"}
        )

    async def complete_structured(self, request, schema):  # type: ignore[no-untyped-def]
        from protocore.contracts.llm import LLMResponse

        return LLMResponse(
            message=Message(role=MessageRole.assistant, content_blocks=[]),
            stop_reason=StopReason.end_turn,
        )

    def count_tokens(self, text, model=None) -> int:  # type: ignore[no-untyped-def]
        return max(1, len(text) // 4)


class _EmptyStreamLLM:
    """Emit an empty stream (no text/tool/reasoning) for the first
    ``empty_rounds`` calls, then a healthy text turn."""

    def __init__(self, empty_rounds: int, recovery_text: str = "here is the answer") -> None:
        self._empty_rounds = empty_rounds
        self._recovery_text = recovery_text
        self._call_idx = 0
        self.calls: list[LLMRequest] = []

    async def stream_with_tools(  # type: ignore[no-untyped-def]
        self, request: LLMRequest
    ) -> AsyncIterator[LLMStreamEvent]:
        self.calls.append(request)
        idx = self._call_idx
        self._call_idx += 1
        if idx < self._empty_rounds:
            # Bare empty completion — no content_block, no tool, no reasoning.
            yield LLMStreamEvent(name="message_start", payload={})
            yield LLMStreamEvent(
                name="message_stop", payload={"stop_reason": "end_turn"}
            )
            return
        yield LLMStreamEvent(name="message_start", payload={})
        yield LLMStreamEvent(name="content_block_start", payload={"kind": "text"})
        yield LLMStreamEvent(
            name="content_block_delta", payload={"text": self._recovery_text}
        )
        yield LLMStreamEvent(name="content_block_stop", payload={})
        yield LLMStreamEvent(
            name="message_stop", payload={"stop_reason": "end_turn"}
        )

    async def complete_structured(self, request, schema):  # type: ignore[no-untyped-def]
        from protocore.contracts.llm import LLMResponse

        return LLMResponse(
            message=Message(role=MessageRole.assistant, content_blocks=[]),
            stop_reason=StopReason.end_turn,
        )

    def count_tokens(self, text, model=None) -> int:  # type: ignore[no-untyped-def]
        return max(1, len(text) // 4)


class _ProseThenAlwaysTransientLLM:
    """Turn 0 emits a full prose answer (no terminal tool); every later call
    raises a transient error. Drives the ``preserve delivered answer on
    exhaustion`` path via the terminal-tool nudge continuation."""

    def __init__(self, exception: BaseException, prose: str) -> None:
        self._exception = exception
        self._prose = prose
        self._call_idx = 0
        self.calls: list[LLMRequest] = []

    async def stream_with_tools(  # type: ignore[no-untyped-def]
        self, request: LLMRequest
    ) -> AsyncIterator[LLMStreamEvent]:
        self.calls.append(request)
        idx = self._call_idx
        self._call_idx += 1
        if idx == 0:
            yield LLMStreamEvent(name="message_start", payload={})
            yield LLMStreamEvent(
                name="content_block_start", payload={"kind": "text"}
            )
            yield LLMStreamEvent(
                name="content_block_delta", payload={"text": self._prose}
            )
            yield LLMStreamEvent(name="content_block_stop", payload={})
            yield LLMStreamEvent(
                name="message_stop", payload={"stop_reason": "end_turn"}
            )
            return
        raise self._exception

    async def complete_structured(self, request, schema):  # type: ignore[no-untyped-def]
        from protocore.contracts.llm import LLMResponse

        return LLMResponse(
            message=Message(role=MessageRole.assistant, content_blocks=[]),
            stop_reason=StopReason.end_turn,
        )

    def count_tokens(self, text, model=None) -> int:  # type: ignore[no-untyped-def]
        return max(1, len(text) // 4)


class _FakeProviderChain:
    """Minimal :class:`IProviderChain` over one scripted LLM under many names.

    Every rung hands back the SAME provider object so a scripted failure
    sequence keeps running across a swap; only the model identity changes, which
    is what the loop writes into ``engine.config``.
    """

    def __init__(self, provider: object, names: list[str]) -> None:
        self._provider = provider
        self._names = names
        self._index = 0
        self._attempted: list[tuple[str, str]] = []

    def current(self) -> object:
        return self._provider

    def current_model_name(self) -> str:
        return self._names[self._index]

    async def advance(self, *, reason: str) -> bool:
        if self._index + 1 >= len(self._names):
            return False
        self._attempted.append((self._names[self._index], reason))
        self._index += 1
        return True

    def attempted(self) -> list[tuple[str, str]]:
        return list(self._attempted)


def _attach_chain(engine, llm: object, *names: str) -> _FakeProviderChain:
    """Wire ``llm`` onto ``engine`` behind a chain listing ``names`` in order."""
    chain = _FakeProviderChain(llm, list(names))
    engine.llm = llm
    engine.provider_chain = chain
    return chain


def _no_backoff_rc(**overrides: object) -> LoopConstants:
    """RC with zero backoff so retry tests never actually sleep."""
    base: dict[str, object] = {
        "model_context_window": 4096,
        "llm_transient_error_retry_backoff_base_seconds": 0.0,
        "llm_transient_error_retry_backoff_max_seconds": 0.0,
    }
    base.update(overrides)
    return LoopConstants(**base)  # type: ignore[arg-type]


async def _drive(engine, text: str = "hi") -> list[TurnEvent]:
    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text=text)])
    events: list[TurnEvent] = []
    async for evt in engine.run(user_msg):
        events.append(evt)
    return events


# ======================================================================
# Fix 1 — 429 / timeout get fallback + bounded retry, not terminal FAILED
# ======================================================================


@pytest.mark.asyncio
async def test_rate_limit_steps_down_provider_chain(engine_factory, in_memory_runtime) -> None:
    """A 429 with a second provider available steps down (not terminal)."""
    rc = _no_backoff_rc()
    engine = engine_factory(rc=rc)
    llm = _ScriptedFailureLLM(exceptions=[LLMRateLimitError("429 rate limited")])
    _attach_chain(engine, llm, "primary-model", "fallback-model-x")

    events = await _drive(engine)

    assert engine.state is LoopState.COMPLETED
    assert engine._provider_chain_advances == 1
    assert engine.config.model_name == "fallback-model-x"
    swap = [
        e
        for e in events
        if e.type is EventType.STATE_CHANGED
        and e.payload.get("reason") == "model_fallback_triggered"
    ]
    assert swap and swap[0].payload.get("error_class") == "llm_rate_limit"
    # The chain took priority over the in-place retry.
    assert engine._transient_stream_retry_count == 0
    assert len(llm.calls) == 2


@pytest.mark.asyncio
async def test_rate_limit_bounded_retry_then_success(engine_factory, in_memory_runtime) -> None:
    """A 429 with NO fallback retries in place and then succeeds."""
    rc = _no_backoff_rc(llm_transient_error_retry_max_attempts=2)
    engine = engine_factory(rc=rc)
    llm = _ScriptedFailureLLM(exceptions=[LLMRateLimitError("429")])
    engine.llm = llm  # type: ignore[assignment]

    events = await _drive(engine)

    assert engine.state is LoopState.COMPLETED
    assert len(llm.calls) == 2  # one failure + one successful retry
    retries = [
        e
        for e in events
        if e.type is EventType.STATE_CHANGED
        and e.payload.get("reason") == "transient_llm_error_retry"
    ]
    assert len(retries) == 1
    assert retries[0].payload.get("error_class") == "llm_rate_limit"
    assert retries[0].payload.get("attempt") == 1
    # Streak reset after the successful stream.
    assert engine._transient_stream_retry_count == 0


@pytest.mark.asyncio
async def test_chain_step_then_retry_composes(engine_factory, in_memory_runtime) -> None:
    """A 429 steps down the chain; a 429 on the last rung then retries in place."""
    rc = _no_backoff_rc(llm_transient_error_retry_max_attempts=1)
    engine = engine_factory(rc=rc)
    llm = _ScriptedFailureLLM(
        exceptions=[LLMRateLimitError("primary 429"), LLMRateLimitError("fallback 429")]
    )
    _attach_chain(engine, llm, "primary-model", "fallback-model-x")

    events = await _drive(engine)

    assert engine.state is LoopState.COMPLETED
    assert engine._provider_chain_advances == 1
    assert engine.config.model_name == "fallback-model-x"
    assert len(llm.calls) == 3  # primary 429 -> swap -> fallback 429 -> retry -> ok
    assert [
        e
        for e in events
        if e.type is EventType.STATE_CHANGED
        and e.payload.get("reason") == "model_fallback_triggered"
    ]
    assert [
        e
        for e in events
        if e.type is EventType.STATE_CHANGED
        and e.payload.get("reason") == "transient_llm_error_retry"
    ]


@pytest.mark.asyncio
async def test_timeout_retry_success_within_budget(engine_factory, in_memory_runtime) -> None:
    """A read-timeout with NO fallback recovers within the retry budget."""
    rc = _no_backoff_rc(llm_transient_error_retry_max_attempts=2)
    engine = engine_factory(rc=rc)
    llm = _ScriptedFailureLLM(
        exceptions=[LLMTimeoutError("read timeout"), LLMTimeoutError("read timeout")]
    )
    engine.llm = llm  # type: ignore[assignment]

    events = await _drive(engine)

    assert engine.state is LoopState.COMPLETED
    assert len(llm.calls) == 3  # two failures + one success
    retries = [
        e
        for e in events
        if e.type is EventType.STATE_CHANGED
        and e.payload.get("reason") == "transient_llm_error_retry"
    ]
    assert [r.payload.get("error_class") for r in retries] == ["llm_timeout", "llm_timeout"]
    assert [r.payload.get("attempt") for r in retries] == [1, 2]


@pytest.mark.asyncio
async def test_timeout_terminal_only_after_retries_exhausted(engine_factory, in_memory_runtime) -> None:
    """A read-timeout goes terminal FAILED only once retries are exhausted."""
    rc = _no_backoff_rc(llm_transient_error_retry_max_attempts=2)
    engine = engine_factory(rc=rc)
    llm = _ScriptedFailureLLM(
        exceptions=[LLMTimeoutError("t"), LLMTimeoutError("t"), LLMTimeoutError("t")]
    )
    engine.llm = llm  # type: ignore[assignment]

    await _drive(engine)

    assert engine.state is LoopState.FAILED
    # initial attempt + 2 retries, then terminal — no further call.
    assert len(llm.calls) == 3


@pytest.mark.asyncio
async def test_rate_limit_no_fallback_no_retry_is_terminal(engine_factory, in_memory_runtime) -> None:
    """With retries disabled and no fallback a 429 is terminal on the first hit."""
    rc = _no_backoff_rc(llm_transient_error_retry_max_attempts=0)
    engine = engine_factory(rc=rc)
    llm = _ScriptedFailureLLM(exceptions=[LLMRateLimitError("429")])
    engine.llm = llm  # type: ignore[assignment]

    await _drive(engine)

    assert engine.state is LoopState.FAILED
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_transient_retry_persists_partial_text(engine_factory, in_memory_runtime) -> None:
    """The failed attempt's partial text is persisted before the retry."""
    rc = _no_backoff_rc(llm_transient_error_retry_max_attempts=1)
    engine = engine_factory(rc=rc)
    llm = _PartialThenTransientLLM(
        exception=LLMTimeoutError("stall"),
        partial_text="partial-before-timeout",
        recovery_text="final-answer",
    )
    engine.llm = llm  # type: ignore[assignment]

    await _drive(engine)

    assert engine.state is LoopState.COMPLETED
    assistant_msgs = [m for m in engine.history if m.role is MessageRole.assistant]
    texts = [
        "".join(b.text for b in m.content_blocks if isinstance(b, TextBlock))
        for m in assistant_msgs
    ]
    assert "partial-before-timeout" in texts
    assert "final-answer" in texts


@pytest.mark.asyncio
async def test_transient_exhaustion_preserves_delivered_answer(
    engine_factory, in_memory_runtime
) -> None:
    """A transient error on the forced terminal call after a delivered answer
    completes on that answer instead of failing."""
    prose = "Here is the complete and substantive answer to your question."
    rc = _no_backoff_rc(
        llm_transient_error_retry_max_attempts=1,
        terminal_tool_nudge_enabled=True,
    )
    engine = engine_factory(rc=rc, expected_terminal_tool="final_answer")
    # A terminal tool the run can call, or there is no call to force and the
    # run completes on the answer without another request.
    in_memory_runtime["tools"].register(
        MockTool(tool_name="final_answer", description="Submit the answer")
    )
    llm = _ProseThenAlwaysTransientLLM(
        exception=LLMRateLimitError("429 on the forced continuation"),
        prose=prose,
    )
    engine.llm = llm  # type: ignore[assignment]

    await _drive(engine)

    assert engine.state is LoopState.COMPLETED
    # turn 0 prose + forced continuation (429 + one retry) all attempted.
    assert len(llm.calls) >= 3
    all_text = " ".join(
        b.text
        for m in engine.history
        if m.role is MessageRole.assistant
        for b in m.content_blocks
        if isinstance(b, TextBlock)
    )
    assert "substantive answer" in all_text


# ======================================================================
# Fix 2 — empty end_turn no longer seals a silent empty COMPLETED
# ======================================================================


@pytest.mark.asyncio
async def test_empty_completion_redrive_then_answer_completes(
    engine_factory, in_memory_runtime
) -> None:
    """An empty first turn is re-driven; a real answer then completes the run."""
    rc = LoopConstants(model_context_window=4096)  # guard default-on, 1 re-drive
    engine = engine_factory(rc=rc)
    llm = _EmptyStreamLLM(empty_rounds=1, recovery_text="the real answer text")
    engine.llm = llm  # type: ignore[assignment]

    events = await _drive(engine)

    assert engine.state is LoopState.COMPLETED
    assert len(llm.calls) == 2
    redrive = [
        e
        for e in events
        if e.type is EventType.STATE_CHANGED
        and e.payload.get("reason") == "empty_completion_redrive"
    ]
    assert len(redrive) == 1
    all_text = " ".join(
        b.text
        for m in engine.history
        if m.role is MessageRole.assistant
        for b in m.content_blocks
        if isinstance(b, TextBlock)
    )
    assert "the real answer text" in all_text


@pytest.mark.asyncio
async def test_empty_completion_exhausted_is_terminal_not_empty_completed(
    engine_factory, in_memory_runtime
) -> None:
    """A persistently-empty run fails loudly rather than sealing empty COMPLETED."""
    rc = LoopConstants(model_context_window=4096, empty_completion_guard_max_redrives=1)
    engine = engine_factory(rc=rc)
    llm = _EmptyStreamLLM(empty_rounds=5)  # never produces content
    engine.llm = llm  # type: ignore[assignment]

    events = await _drive(engine)

    assert engine.state is LoopState.FAILED
    assert len(llm.calls) == 2  # first empty + one re-drive, then terminal
    err = [
        e
        for e in events
        if e.type is EventType.ERROR
        and e.payload.get("kind") == "no_answer_empty_completion"
    ]
    assert err, "expected a no_answer_empty_completion error event"
    stops = [e for e in events if e.type is EventType.MESSAGE_STOP]
    assert stops and stops[-1].payload.get("stop_reason") == StopReason.error.value


@pytest.mark.asyncio
async def test_empty_completion_guard_disabled_seals_completed(
    engine_factory, in_memory_runtime
) -> None:
    """With the guard disabled the prior behaviour (empty COMPLETED) holds."""
    rc = LoopConstants(model_context_window=4096, empty_completion_guard_enabled=False)
    engine = engine_factory(rc=rc)
    llm = _EmptyStreamLLM(empty_rounds=5)
    engine.llm = llm  # type: ignore[assignment]

    await _drive(engine)

    assert engine.state is LoopState.COMPLETED
    assert len(llm.calls) == 1  # no re-drive; sealed on the first empty turn


@pytest.mark.asyncio
async def test_empty_guard_does_not_fire_on_healthy_answer(
    engine_factory, in_memory_runtime
) -> None:
    """A normal answered turn completes untouched (no empty-guard regression)."""
    rc = LoopConstants(model_context_window=4096)
    engine = engine_factory(rc=rc)
    llm = _ScriptedFailureLLM(exceptions=[], recovery_text="a perfectly good answer")
    engine.llm = llm  # type: ignore[assignment]

    events = await _drive(engine)

    assert engine.state is LoopState.COMPLETED
    assert len(llm.calls) == 1
    assert not [
        e
        for e in events
        if e.type is EventType.STATE_CHANGED
        and e.payload.get("reason") == "empty_completion_redrive"
    ]


# ======================================================================
# Provider-chain bounds, stickiness, and the failures that must NOT advance
# ======================================================================


class _Verdict:
    """Stand-in for the classification the adapter layer pins on an error."""

    def __init__(self, reason: str) -> None:
        self.reason = reason


def _classified(exc: BaseException, reason: str) -> BaseException:
    """Attach a classification to ``exc`` the way a provider adapter does."""
    object.__setattr__(exc, "classified", _Verdict(reason))
    return exc


@pytest.mark.asyncio
async def test_chain_advance_is_bounded(engine_factory, in_memory_runtime) -> None:
    """``llm_provider_chain_max_advances`` caps the walk, then the retry budget
    takes over and only then is the run terminal."""
    rc = _no_backoff_rc(
        llm_provider_chain_max_advances=1,
        llm_transient_error_retry_max_attempts=0,
    )
    engine = engine_factory(rc=rc)
    llm = _ScriptedFailureLLM(
        exceptions=[LLMRateLimitError("a"), LLMRateLimitError("b"), LLMRateLimitError("c")]
    )
    chain = _attach_chain(engine, llm, "m0", "m1", "m2")

    await _drive(engine)

    assert engine.state is LoopState.FAILED
    # One advance only, even though a third rung was available.
    assert engine._provider_chain_advances == 1
    assert engine.config.model_name == "m1"
    assert len(llm.calls) == 2
    assert chain._index == 1


@pytest.mark.asyncio
async def test_zero_advances_disables_stepping_entirely(
    engine_factory, in_memory_runtime
) -> None:
    """The kill switch: a configured chain that may not be walked."""
    rc = _no_backoff_rc(
        llm_provider_chain_max_advances=0,
        llm_transient_error_retry_max_attempts=0,
    )
    engine = engine_factory(rc=rc)
    llm = _ScriptedFailureLLM(exceptions=[LLMRateLimitError("429")])
    _attach_chain(engine, llm, "m0", "m1")

    await _drive(engine)

    assert engine.state is LoopState.FAILED
    assert engine._provider_chain_advances == 0
    assert engine.config.model_name != "m1"
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_a_demotion_is_sticky_for_the_rest_of_the_run(
    engine_factory, in_memory_runtime
) -> None:
    """Position 0 failing once must not be retried later in the same run, even
    though nothing proves it is still unhealthy."""
    rc = _no_backoff_rc(llm_provider_chain_max_advances=2)
    engine = engine_factory(rc=rc)
    # Fail once, then succeed forever — a chain that walked back would show up
    # as a second advance and a model name that returned to m0.
    llm = _ScriptedFailureLLM(exceptions=[LLMRateLimitError("429")])
    chain = _attach_chain(engine, llm, "m0", "m1", "m2")

    await _drive(engine)

    assert engine.state is LoopState.COMPLETED
    assert engine._provider_chain_advances == 1
    assert engine.config.model_name == "m1"

    # The per-message recovery reset must NOT hand position 0 back. It clears
    # the compaction and output-truncation budgets, which are per-message; a
    # demotion is not, and re-testing a provider that just failed costs the same
    # failed turn again.
    engine.reset_recovery_state()

    assert engine._provider_chain_advances == 1
    assert chain._index == 1
    assert chain.current_model_name() == "m1"


@pytest.mark.parametrize(
    "reason",
    [
        "context_overflow",
        "payload_too_large",
        "image_too_large",
        "format_error",
        "thinking_signature",
        "provider_policy_blocked",
        "long_context_tier",
        "oauth_long_context_beta_forbidden",
        "llama_cpp_grammar_pattern",
        "unknown",
    ],
)
@pytest.mark.asyncio
async def test_these_failures_never_advance_the_chain(
    engine_factory, in_memory_runtime, reason: str
) -> None:
    """Each of these has a recovery that already works, and advancing would
    skip it — compaction, compression, the structured-output ladder, stripping
    an invalidated reasoning payload, or a deliberate refusal to re-route
    content a provider declined."""
    rc = _no_backoff_rc(llm_provider_chain_max_advances=3)
    engine = engine_factory(rc=rc)
    llm = _ScriptedFailureLLM(
        exceptions=[_classified(LLMProviderError(f"{reason} failure"), reason)]
    )
    _attach_chain(engine, llm, "m0", "m1")

    await _drive(engine)

    assert engine._provider_chain_advances == 0
    assert engine.config.model_name != "m1"


@pytest.mark.parametrize(
    "reason",
    ["auth", "auth_permanent", "billing", "model_not_found", "overloaded", "server_error"],
)
@pytest.mark.asyncio
async def test_these_failures_do_advance_the_chain(
    engine_factory, in_memory_runtime, reason: str
) -> None:
    """Each names a property of the failing ROW — its key, its balance, its
    capacity, its catalogue — so the next row gets a real chance."""
    rc = _no_backoff_rc(llm_provider_chain_max_advances=3)
    engine = engine_factory(rc=rc)
    llm = _ScriptedFailureLLM(
        exceptions=[_classified(LLMProviderError(f"{reason} failure"), reason)]
    )
    _attach_chain(engine, llm, "m0", "m1")

    await _drive(engine)

    assert engine.state is LoopState.COMPLETED
    assert engine._provider_chain_advances == 1
    assert engine.config.model_name == "m1"


@pytest.mark.asyncio
async def test_an_unclassified_provider_error_does_not_advance(
    engine_factory, in_memory_runtime
) -> None:
    """``LLMProviderError`` is the adapters' catch-all. With no verdict attached
    there is no basis for spending another provider's quota."""
    rc = _no_backoff_rc(llm_provider_chain_max_advances=3)
    engine = engine_factory(rc=rc)
    llm = _ScriptedFailureLLM(exceptions=[LLMProviderError("something went wrong")])
    _attach_chain(engine, llm, "m0", "m1")

    await _drive(engine)

    assert engine._provider_chain_advances == 0


@pytest.mark.asyncio
async def test_no_chain_configured_leaves_recovery_untouched(
    engine_factory, in_memory_runtime
) -> None:
    """The superset gate in the loop: with no chain, a 429 gets exactly the
    bounded in-place retry it gets today."""
    rc = _no_backoff_rc(llm_transient_error_retry_max_attempts=2)
    engine = engine_factory(rc=rc)
    llm = _ScriptedFailureLLM(exceptions=[LLMRateLimitError("429")])
    engine.llm = llm  # type: ignore[assignment]

    events = await _drive(engine)

    assert engine.provider_chain is None
    assert engine.state is LoopState.COMPLETED
    assert engine._provider_chain_advances == 0
    retries = [
        e
        for e in events
        if e.type is EventType.STATE_CHANGED
        and e.payload.get("reason") == "transient_llm_error_retry"
    ]
    assert len(retries) == 1
