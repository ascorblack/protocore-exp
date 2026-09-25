"""Every provider call is assembled by one builder.

Four call paths reach a provider: the action stream, the deep loop's plan
call, the deep loop's prompted-JSON plan fallback, and the Tier-2 compaction
summariser. Each used to construct its own request, so they disagreed on how
the model was resolved (live override vs. frozen config), on how a forced tool
was spelled (``forced_tool_choice`` vs. a wire-shaped ``tool_choice``) and on
whether a temperature was stated at all. The builder states one only when the
caller has one; otherwise the request leaves it to the host.

These tests pin the assembled request on all four paths — the shape each one
had before the builder existed, so the consolidation is provably observable-
identical — and then pin the three properties the builder adds: one model
resolution, one forced-choice key, one temperature policy.
"""
from __future__ import annotations

from typing import Any

from protocore.contracts.llm import LLMRequest, LLMStreamEvent
from protocore.contracts.observability import request_digest
from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.types import (
    Message,
    MessageRole,
    StopReason,
    TextBlock,
)
from protocore.runtime.context.compaction import (
    CompactionState,
    run_tier2_summarisation,
)
from protocore.runtime.events import EventType
from protocore.runtime.loop_state import LoopState
from protocore.runtime.loop_strategies import PLAN_TOOL_NAME, DeepStrategy
from protocore.runtime.query import (
    _drive_one_stream,
    _StreamAttemptResult,
    build_llm_request,
)
from protocore.runtime.query_engine import QueryEngine, QueryEngineConfig
from protocore.tests_support.adapters import (
    InMemoryBlobStore,
    InMemoryEventStream,
    InMemoryHookManager,
    InMemoryLLMProvider,
    InMemorySkillStore,
    InMemoryToolRegistry,
)

from ._tool_fixtures import MockTool

MODEL = "test-model-a"
OVERRIDE_MODEL = "test-model-b"


def _build_engine(
    *,
    run_mode: str,
    llm: Any,
    rc: LoopConstants | None = None,
) -> QueryEngine:
    registry = InMemoryToolRegistry()
    for name in ("Read", "Write"):
        registry.register(MockTool(tool_name=name, description=f"{name} tool"))
    return QueryEngine(
        config=QueryEngineConfig(
            run_id="run-builder",
            tenant_id="tenant-builder",
            session_id="sess-builder",
            model_name=MODEL,
            rc=rc or LoopConstants(model_context_window=8_192),
            run_mode=run_mode,
            thinking_enabled=(run_mode == "deep"),
            reasoning_effort="low",
        ),
        llm_provider=llm,
        tool_registry=registry,
        event_stream=InMemoryEventStream(),
        hook_manager=InMemoryHookManager(),
        skill_store=InMemorySkillStore(),
        blob_store=InMemoryBlobStore(),
    )


def _scripted_action_llm() -> InMemoryLLMProvider:
    llm = InMemoryLLMProvider()
    llm.queue_response(text="done", stop_reason=StopReason.end_turn)
    return llm


async def _drive(engine: QueryEngine) -> None:
    initial = Message(
        role=MessageRole.user, content_blocks=[TextBlock(text="do the thing")]
    )
    [evt async for evt in engine.run(initial)]


class _StubClassified:
    """Duck-typed stand-in for a host adapter's classified-error verdict.

    Core reads ``.should_fallback`` off the attached object via ``getattr``
    (``loop_strategies._is_fallback_worthy``); the import boundary forbids
    importing a host classifier, so the duck shape is mirrored here.
    """

    def __init__(self, *, should_fallback: bool) -> None:
        self.should_fallback = should_fallback


def _fallback_worthy_error() -> Exception:
    from protocore.contracts.llm import LLMProviderError

    exc = LLMProviderError("forced tool rejected (HTTP 400)")
    object.__setattr__(exc, "classified", _StubClassified(should_fallback=True))
    return exc


def _plan_json_stream(plan_json: str) -> list[LLMStreamEvent]:
    return [
        LLMStreamEvent(name="message_start", payload={}),
        LLMStreamEvent(name="content_block_start", payload={"kind": "text"}),
        LLMStreamEvent(name="content_block_delta", payload={"text": plan_json}),
        LLMStreamEvent(name="content_block_stop", payload={}),
        LLMStreamEvent(
            name="message_stop", payload={"stop_reason": StopReason.end_turn.value}
        ),
    ]


# ---------------------------------------------------------------------------
# Characterisation — the four paths' assembled requests
# ---------------------------------------------------------------------------


async def test_action_stream_request_shape() -> None:
    llm = _scripted_action_llm()
    engine = _build_engine(run_mode="direct", llm=llm)
    await _drive(engine)

    request = llm.calls[0]
    assert request.model == MODEL
    assert request.temperature is None
    assert set(request.extra) == {
        "cache_breakpoints",
        "enable_thinking",
        "reasoning_effort",
    }
    assert request.extra["enable_thinking"] is False
    assert request.extra["reasoning_effort"] == "low"
    assert [t.name for t in request.tools] == ["Read", "Write"]
    obs = request.observability
    assert obs is not None
    assert obs.tenant_id == "tenant-builder"
    assert obs.run_id == "run-builder"
    assert obs.session_id == "sess-builder"
    assert obs.call_purpose == "run"
    assert obs.call_category == "agent_call"


async def test_action_request_ignores_stale_observed_prompt_count() -> None:
    llm = _scripted_action_llm()
    engine = _build_engine(
        run_mode="direct",
        llm=llm,
        rc=LoopConstants(model_context_window=4_096),
    )
    engine.last_observed_prompt_tokens = 4_000
    context = engine.context_manager.build_context(
        history=[
            Message(role=MessageRole.user, content_blocks=[TextBlock(text="act")])
        ],
        tools=[],
    )

    [
        event
        async for event in _drive_one_stream(
            engine,
            context,
            _StreamAttemptResult(),
        )
    ]

    assert llm.calls[0].max_tokens == 1_024


async def test_late_usage_from_old_model_keeps_new_model_calibration() -> None:
    class _SwitchAfterRequest:
        def __init__(self) -> None:
            self.engine: QueryEngine | None = None
            self.calls: list[LLMRequest] = []

        async def stream_with_tools(self, request: LLMRequest) -> Any:
            self.calls.append(request)
            if len(self.calls) == 1:
                assert self.engine is not None
                self.engine.apply_live_controls(model_name=OVERRIDE_MODEL)
                yield LLMStreamEvent(name="message_start", payload={})
                yield LLMStreamEvent(
                    name="usage",
                    payload={"input_tokens": 100_000},
                )
            yield LLMStreamEvent(
                name="message_stop",
                payload={"stop_reason": StopReason.end_turn.value},
            )

    llm = _SwitchAfterRequest()
    rc = LoopConstants(
        model_context_window=4_096,
        token_estimate_calibration=1.25,
    )
    engine = _build_engine(run_mode="direct", llm=llm, rc=rc)
    llm.engine = engine
    context = engine.context_manager.build_context(
        history=[
            Message(
                role=MessageRole.user,
                content_blocks=[TextBlock(text="x" * 5_000)],
            )
        ],
        tools=[],
    )

    for _ in range(2):
        [
            event
            async for event in _drive_one_stream(
                engine,
                context,
                _StreamAttemptResult(),
            )
        ]

    assert [request.model for request in llm.calls] == [MODEL, OVERRIDE_MODEL]
    assert engine.total_usage.input_tokens == 100_000
    assert engine.config.rc.token_estimate_calibration == 1.25
    assert engine._token_estimate_calibration_model == OVERRIDE_MODEL
    assert llm.calls[1].max_tokens == 1_024


async def test_action_fit_refreshes_calibration_after_surface_event() -> None:
    llm = _scripted_action_llm()
    rc = LoopConstants(
        model_context_window=4_096,
        token_estimate_calibration=1.25,
    )
    engine = _build_engine(run_mode="direct", llm=llm, rc=rc)
    engine.set_token_estimate_calibration(3.0, model_name=MODEL)
    context = engine.context_manager.build_context(
        history=[
            Message(
                role=MessageRole.user,
                content_blocks=[TextBlock(text="x" * 5_000)],
            )
        ],
        tools=[],
    )
    stream = _drive_one_stream(engine, context, _StreamAttemptResult())

    advertised = await anext(stream)
    assert advertised.type is EventType.TOOL_SURFACE_ADVERTISED
    engine.apply_live_controls(model_name=OVERRIDE_MODEL)
    [event async for event in stream]

    assert llm.calls[0].model == OVERRIDE_MODEL
    assert llm.calls[0].max_tokens == 1_024
    assert engine.config.rc.token_estimate_calibration == 1.25
    assert engine._token_estimate_calibration_model == OVERRIDE_MODEL


async def test_plan_request_shape() -> None:
    llm = InMemoryLLMProvider()
    llm.queue_tool_call_response(
        tool_call_id="toolu_plan",
        tool_name=PLAN_TOOL_NAME,
        tool_input={"plan": ["a"], "next_tool": "Write", "task_complete": True},
    )
    llm.queue_response(text="done", stop_reason=StopReason.end_turn)
    engine = _build_engine(run_mode="deep", llm=llm)
    await _drive(engine)

    request = llm.calls[0]
    assert request.model == MODEL
    assert request.temperature is None
    assert [t.name for t in request.tools] == [PLAN_TOOL_NAME]
    assert request.extra["enable_thinking"] is True
    assert request.extra["reasoning_effort"] == "low"
    obs = request.observability
    assert obs is not None
    assert obs.call_purpose == "deep_plan"
    assert obs.call_category == "planning"


async def test_plan_fallback_request_shape() -> None:
    base = InMemoryLLMProvider()
    base.queue_response(text="done", stop_reason=StopReason.end_turn)
    plan_json = '{"plan": ["a"], "next_tool": "Write", "task_complete": true}'
    captured: dict[str, LLMRequest] = {}

    class _RejectThenFallback:
        def __init__(self) -> None:
            self._call = 0

        async def stream_with_tools(self, request: LLMRequest) -> Any:
            self._call += 1
            if self._call == 1:
                yield LLMStreamEvent(name="message_start", payload={})
                raise _fallback_worthy_error()
            if self._call == 2:
                captured["fallback"] = request
                for evt in _plan_json_stream(plan_json):
                    yield evt
                return
            async for evt in base.stream_with_tools(request):
                yield evt

    engine = _build_engine(run_mode="deep", llm=_RejectThenFallback())
    await _drive(engine)

    request = captured["fallback"]
    assert request.model == MODEL
    assert request.temperature is None
    assert list(request.tools) == []
    assert request.extra == {"response_format": {"type": "json_object"}}
    obs = request.observability
    assert obs is not None
    assert obs.call_purpose == "deep_plan_fallback"
    assert obs.call_category == "planning"


async def test_compaction_summariser_request_shape() -> None:
    rc = LoopConstants(
        model_context_window=4_096,
        compaction_keep_recent_turns=1,
    )
    llm = InMemoryLLMProvider()
    llm.queue_response(text='{"summary": "they greeted each other"}')
    history = [
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="hello there " * 20)]),
        Message(
            role=MessageRole.assistant, content_blocks=[TextBlock(text="hi back " * 20)]
        ),
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="recent")]),
    ]
    await run_tier2_summarisation(
        history=history,
        compaction_llm=llm,
        state=CompactionState(),
        rc=rc,
        model_name=MODEL,
    )

    request = llm.calls[0]
    assert request.model == MODEL
    assert request.temperature == rc.compaction_summary_temperature
    assert list(request.tools) == []
    assert request.max_tokens == rc.compaction_summary_max_output_tokens
    assert request.extra == {}


async def test_compaction_summariser_skips_a_known_oversized_request() -> None:
    rc = LoopConstants(
        model_context_window=64,
        request_context_safety_tokens=0,
        compaction_keep_recent_turns=1,
    )
    llm = InMemoryLLMProvider()
    history = [
        Message(
            role=MessageRole.assistant,
            content_blocks=[TextBlock(text="old evidence " * 200)],
        ),
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="recent")]),
    ]
    original = list(history)

    await run_tier2_summarisation(
        history=history,
        compaction_llm=llm,
        state=CompactionState(),
        rc=rc,
        model_name=MODEL,
    )

    assert not llm.calls
    assert history == original


async def test_deep_plan_skips_a_known_oversized_request() -> None:
    llm = InMemoryLLMProvider()
    engine = _build_engine(
        run_mode="deep",
        llm=llm,
            rc=LoopConstants(model_context_window=64, request_context_safety_tokens=0),
    )
    tool = MockTool(tool_name="Read").definition
    context = engine.context_manager.build_context(
        history=[
            Message(
                role=MessageRole.user,
                content_blocks=[TextBlock(text="large prompt " * 200)],
            )
        ],
        tools=[tool],
    )

    events = [event async for event in DeepStrategy().prepare_turn(engine, context)]

    assert events == []
    assert not llm.calls


async def test_deep_plan_ignores_observed_tokens_from_an_action_request() -> None:
    llm = InMemoryLLMProvider()
    llm.queue_tool_call_response(
        tool_call_id="toolu_plan",
        tool_name=PLAN_TOOL_NAME,
        tool_input={"plan": ["a"], "next_tool": "Read", "task_complete": True},
    )
    engine = _build_engine(
        run_mode="deep",
        llm=llm,
        rc=LoopConstants(model_context_window=4_096),
    )
    engine.last_observed_prompt_tokens = 4_000
    context = engine.context_manager.build_context(
        history=[
            Message(role=MessageRole.user, content_blocks=[TextBlock(text="plan")])
        ],
        tools=[MockTool(tool_name="Read").definition],
    )

    [event async for event in DeepStrategy().prepare_turn(engine, context)]

    assert llm.calls[0].max_tokens == 1_024


async def test_deep_plan_json_fallback_skips_a_known_oversized_request() -> None:
    llm = InMemoryLLMProvider()
    engine = _build_engine(
        run_mode="deep",
        llm=llm,
            rc=LoopConstants(model_context_window=64, request_context_safety_tokens=0),
    )
    messages = [
        Message(
            role=MessageRole.user,
            content_blocks=[TextBlock(text="large fallback prompt " * 200)],
        )
    ]

    result = await DeepStrategy()._fetch_plan_fallback(
        engine,
        messages,
        ["Read"],
        False,
        max_tokens=32,
    )

    assert result is None
    assert not llm.calls


async def test_deep_plan_json_fallback_ignores_observed_action_tokens() -> None:
    llm = InMemoryLLMProvider()
    llm.queue_response(
        text='{"plan":["a"],"next_tool":"Read","task_complete":true}'
    )
    engine = _build_engine(
        run_mode="deep",
        llm=llm,
        rc=LoopConstants(model_context_window=4_096),
    )
    engine.last_observed_prompt_tokens = 4_000

    result = await DeepStrategy()._fetch_plan_fallback(
        engine,
        [Message(role=MessageRole.user, content_blocks=[TextBlock(text="plan")])],
        ["Read"],
        False,
        max_tokens=1_024,
    )

    assert result is not None
    assert llm.calls[0].max_tokens == 1_024


# ---------------------------------------------------------------------------
# The properties the single builder guarantees
# ---------------------------------------------------------------------------


async def test_plan_call_states_its_forced_tool_under_the_shared_key() -> None:
    """One spelling of a forced choice across every path.

    The plan call used to state its forced tool in the wire shape under
    ``extra['tool_choice']`` while the action stream used the bare-name
    ``extra['forced_tool_choice']``. Both reach a provider as the same
    single-tool choice, but only one of them can be read by a single reader.
    """
    llm = InMemoryLLMProvider()
    llm.queue_tool_call_response(
        tool_call_id="toolu_plan",
        tool_name=PLAN_TOOL_NAME,
        tool_input={"plan": ["a"], "next_tool": "Write", "task_complete": True},
    )
    llm.queue_response(text="done", stop_reason=StopReason.end_turn)
    engine = _build_engine(run_mode="deep", llm=llm)
    await _drive(engine)

    request = llm.calls[0]
    assert request.extra["forced_tool_choice"] == PLAN_TOOL_NAME
    assert "tool_choice" not in request.extra


async def test_live_model_override_reaches_the_plan_call() -> None:
    llm = InMemoryLLMProvider()
    llm.queue_tool_call_response(
        tool_call_id="toolu_plan",
        tool_name=PLAN_TOOL_NAME,
        tool_input={"plan": ["a"], "next_tool": "Write", "task_complete": True},
    )
    llm.queue_response(text="done", stop_reason=StopReason.end_turn)
    engine = _build_engine(run_mode="deep", llm=llm)
    engine.apply_live_controls(model_name=OVERRIDE_MODEL)
    await _drive(engine)

    assert [call.model for call in llm.calls] == [OVERRIDE_MODEL] * len(llm.calls)


async def test_live_model_override_reaches_the_plan_fallback() -> None:
    base = InMemoryLLMProvider()
    base.queue_response(text="done", stop_reason=StopReason.end_turn)
    plan_json = '{"plan": ["a"], "next_tool": "Write", "task_complete": true}'
    captured: dict[str, LLMRequest] = {}

    class _RejectThenFallback:
        def __init__(self) -> None:
            self._call = 0

        async def stream_with_tools(self, request: LLMRequest) -> Any:
            self._call += 1
            if self._call == 1:
                yield LLMStreamEvent(name="message_start", payload={})
                raise _fallback_worthy_error()
            if self._call == 2:
                captured["fallback"] = request
                for evt in _plan_json_stream(plan_json):
                    yield evt
                return
            async for evt in base.stream_with_tools(request):
                yield evt

    engine = _build_engine(run_mode="deep", llm=_RejectThenFallback())
    engine.apply_live_controls(model_name=OVERRIDE_MODEL)
    await _drive(engine)

    assert captured["fallback"].model == OVERRIDE_MODEL


async def test_live_model_override_reaches_the_compaction_summariser() -> None:
    """The summariser call is resolved the same way as every other call.

    It is issued against a separately injected provider, so nothing else in
    the run states which model it names; a live override that skipped it split
    one agent turn across two models with no event saying so.
    """
    from protocore.runtime.context.compaction import CompactionAttempt
    from protocore.runtime.query import _run_compaction

    llm = _scripted_action_llm()
    engine = _build_engine(run_mode="direct", llm=llm)
    engine.apply_live_controls(model_name=OVERRIDE_MODEL)
    engine.transition_to(LoopState.RUNNING)

    seen: dict[str, Any] = {}

    async def _record(**kwargs: Any) -> CompactionAttempt:
        seen.update(kwargs)
        return CompactionAttempt()

    engine.context_manager.run_compaction = _record  # type: ignore[method-assign]
    # The history here holds nothing to compact, and a pass with nothing to do
    # is never opened; the question is only what an opened pass is handed.
    engine.context_manager.has_proactive_work = (  # type: ignore[method-assign]
        lambda *_args, **_kwargs: True
    )
    [evt async for evt in _run_compaction(engine)]

    assert seen["model_name"] == OVERRIDE_MODEL


# ---------------------------------------------------------------------------
# Temperature policy — unset unless the caller states one
# ---------------------------------------------------------------------------


def test_the_request_contract_leaves_the_temperature_unset() -> None:
    request = LLMRequest(model=MODEL, messages=[])

    assert request.temperature is None


def test_builder_leaves_the_temperature_to_the_host_when_the_caller_has_none() -> None:
    request = build_llm_request(model=MODEL, messages=[], max_tokens=16)

    assert request.temperature is None


def test_builder_keeps_an_explicit_temperature_including_zero() -> None:
    assert build_llm_request(
        model=MODEL, messages=[], max_tokens=16, temperature=0.2
    ).temperature == 0.2
    assert build_llm_request(
        model=MODEL, messages=[], max_tokens=16, temperature=0.0
    ).temperature == 0.0


def test_an_unset_and_a_stated_temperature_are_different_requests() -> None:
    unset = build_llm_request(model=MODEL, messages=[], max_tokens=16)
    stated = build_llm_request(model=MODEL, messages=[], max_tokens=16, temperature=0.7)

    assert request_digest(unset) != request_digest(stated)
