"""What was sent, recorded before the answer exists — and played back.

A provider request is assembled from the history, the compaction checkpoint,
the pairing repair, the tool surface and the constants in force, and it used to
exist only in the stack frame that made the call. Nothing durable could tell a
run that behaved oddly because it was SENT something different from one that
got a different answer to the same thing, and a run cut off mid-stream left
behind a snapshot written before the request was even built while its
subscribers had already seen deltas.

The manifest closes that. It is built before the provider is asked, it is
addressed from the snapshot by id rather than copied into it, and it is the
input to a replay provider that re-drives a recorded run without an endpoint —
and refuses, loudly, when the run it is replaying no longer asks for the same
thing.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from protocore.contracts.llm import CacheBreakpoint, LLMRequest, LLMStreamEvent
from protocore.contracts.observability import (
    _MODEL_VISIBLE_MESSAGE_FIELDS,
    REQUEST_MANIFEST_SCHEMA_KEY,
    REQUEST_MANIFEST_SCHEMA_VERSION,
    ManifestSchemaError,
    RequestManifest,
    build_request_manifest,
    canonical_bytes,
    read_manifest_schema_version,
    request_digest,
)
from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.snapshot import RUN_SCOPED_STATE_SNAPSHOT_KEY
from protocore.contracts.types import (
    Message,
    MessageRole,
    StopReason,
    TextBlock,
)
from protocore.runtime.query_engine import QueryEngine, QueryEngineConfig
from protocore.tests_support.adapters import (
    InMemoryBlobStore,
    InMemoryEventStream,
    InMemoryHookManager,
    InMemoryLLMProvider,
    InMemoryRequestManifestSink,
    InMemorySkillStore,
    InMemoryToolRegistry,
    ReplayLLMProvider,
    ReplayMismatchError,
)

from ._tool_fixtures import MockTool

MODEL = "test-model-manifest"


class _OrderRecordingSink(InMemoryRequestManifestSink):
    """Records WHEN it was called, relative to the stream it describes."""

    def __init__(self, log: list[str], blobs: InMemoryBlobStore | None = None) -> None:
        super().__init__(blobs)
        self._log = log

    async def record_request_manifest(
        self,
        *,
        manifest: RequestManifest,
        manifest_id: str,
        bodies: Mapping[str, bytes],
    ) -> None:
        self._log.append("manifest")
        await super().record_request_manifest(
            manifest=manifest, manifest_id=manifest_id, bodies=bodies
        )


class _RaisingSink(InMemoryRequestManifestSink):
    """A host whose store is full, unreachable, or misconfigured."""

    async def record_request_manifest(
        self,
        *,
        manifest: RequestManifest,
        manifest_id: str,
        bodies: Mapping[str, bytes],
    ) -> None:
        raise RuntimeError("the store is not answering")


class _LoggingLLM(InMemoryLLMProvider):
    """Notes the moment its first delta leaves the provider."""

    def __init__(self, log: list[str]) -> None:
        super().__init__()
        self._log = log

    async def stream_with_tools(self, request: LLMRequest):  # type: ignore[no-untyped-def]
        first = True
        async for event in super().stream_with_tools(request):
            if first:
                self._log.append("delta")
                first = False
            yield event


def _build_engine(
    *,
    llm: Any,
    sink: Any = None,
    rc: LoopConstants | None = None,
    run_id: str = "run-manifest",
    tools: Sequence[str] = ("Read", "Write"),
) -> QueryEngine:
    registry = InMemoryToolRegistry()
    for name in tools:
        registry.register(MockTool(tool_name=name, description=f"{name} tool"))
    return QueryEngine(
        config=QueryEngineConfig(
            run_id=run_id,
            tenant_id="tenant-manifest",
            session_id="sess-manifest",
            model_name=MODEL,
            rc=rc or LoopConstants(model_context_window=8_192),
            request_manifest_sink=sink,
        ),
        llm_provider=llm,
        tool_registry=registry,
        event_stream=InMemoryEventStream(),
        hook_manager=InMemoryHookManager(),
        skill_store=InMemorySkillStore(),
        blob_store=InMemoryBlobStore(),
    )


async def _drive(engine: QueryEngine, text: str = "do the thing") -> None:
    initial = Message(role=MessageRole.user, content_blocks=[TextBlock(text=text)])
    [event async for event in engine.run(initial)]


def _answer_stream(text: str) -> list[LLMStreamEvent]:
    return [
        LLMStreamEvent(name="message_start", payload={}),
        LLMStreamEvent(name="content_block_start", payload={"kind": "text"}),
        LLMStreamEvent(name="content_block_delta", payload={"text": text}),
        LLMStreamEvent(name="content_block_stop", payload={}),
        LLMStreamEvent(
            name="message_stop", payload={"stop_reason": StopReason.end_turn.value}
        ),
    ]


# ── the manifest exists before the answer does ──────────────────────────────


async def test_the_manifest_is_emitted_before_the_first_delta() -> None:
    """The ordering is the whole point: a run killed mid-stream has already
    published what it asked for, so the deltas its subscribers saw can be
    matched against a request rather than against a guess."""
    log: list[str] = []
    llm = _LoggingLLM(log)
    llm.queue_response(text="done", stop_reason=StopReason.end_turn)
    sink = _OrderRecordingSink(log)

    await _drive(_build_engine(llm=llm, sink=sink))

    assert log[:2] == ["manifest", "delta"]
    assert len(sink.manifests) == 1


async def test_no_sink_means_no_manifest_is_built_at_all() -> None:
    """The machinery is inert for a host that does not want it — hashing a long
    history on every call is not free, and a run nobody records must not pay."""
    llm = InMemoryLLMProvider()
    llm.queue_response(text="done", stop_reason=StopReason.end_turn)
    engine = _build_engine(llm=llm, sink=None)

    await _drive(engine)

    assert engine.last_request_manifest is None
    assert engine.snapshot()["last_request_manifest"] is None


async def test_the_manifest_states_what_the_request_was_made_of() -> None:
    llm = InMemoryLLMProvider()
    llm.queue_response(text="done", stop_reason=StopReason.end_turn)
    sink = InMemoryRequestManifestSink()
    engine = _build_engine(llm=llm, sink=sink)

    await _drive(engine)

    manifest = sink.manifests[0]
    request = llm.calls[0]
    assert manifest.manifest_schema_version == REQUEST_MANIFEST_SCHEMA_VERSION
    assert manifest.model == MODEL == request.model
    assert manifest.max_tokens == request.max_tokens
    assert manifest.temperature == request.temperature
    # The action stream has no temperature of its own; the manifest records
    # that the host was left to decide, not a number nobody chose.
    assert manifest.temperature is None
    assert manifest.message_count == len(request.messages)
    assert manifest.tool_count == len(request.tools) == 2
    assert manifest.provider_chain_position == 0
    assert manifest.provider_chain_model is None
    assert manifest.request_sha256 == request_digest(request)
    assert manifest.identity["run_id"] == "run-manifest"
    assert manifest.identity["tenant_id"] == "tenant-manifest"
    # The tool definitions travel WHOLE — schemas included, which is the half
    # the tool-surface event never carried.
    tools = json.loads(await sink.body_of(manifest, "tools"))
    assert [item["name"] for item in tools] == ["Read", "Write"]
    assert all(item["parameters"]["properties"] is not None for item in tools)
    # And the constants are digested, so a prompt that moved because an
    # operator retuned a value is distinguishable from one the agent changed.
    assert len(manifest.constants_sha256) == 64


async def test_manifest_records_the_hard_fitted_output_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = InMemoryLLMProvider()
    llm.queue_response(text="done", stop_reason=StopReason.end_turn)
    sink = InMemoryRequestManifestSink()
    rc = LoopConstants(model_context_window=8_192)
    monkeypatch.setattr(
        "protocore.runtime.request_budget.estimate_request_prompt_tokens",
        lambda request, constants, **kwargs: (
            constants.model_context_window - constants.request_context_safety_tokens - 1
        ),
    )

    await _drive(_build_engine(llm=llm, sink=sink, rc=rc))

    assert llm.calls[0].max_tokens == 1
    assert sink.manifests[0].max_tokens == 1


async def test_the_attempt_id_is_stable_across_processes() -> None:
    """Derived from the run, the turn and the request's own digest — so the
    same request rebuilt on another pod carries the same id without anything
    having been persisted to mint it."""
    manifests: list[RequestManifest] = []
    for _ in range(2):
        llm = InMemoryLLMProvider()
        llm.queue_response(text="done", stop_reason=StopReason.end_turn)
        sink = InMemoryRequestManifestSink()
        await _drive(_build_engine(llm=llm, sink=sink))
        manifests.append(sink.manifests[0])

    assert manifests[0].attempt_id == manifests[1].attempt_id
    assert manifests[0].manifest_id == manifests[1].manifest_id
    assert manifests[0].attempt_id.startswith("run-manifest/")


# ── the snapshot addresses it, and never carries it ─────────────────────────


async def test_the_snapshot_addresses_the_manifest_by_id() -> None:
    llm = InMemoryLLMProvider()
    llm.queue_response(text="done", stop_reason=StopReason.end_turn)
    sink = InMemoryRequestManifestSink()
    engine = _build_engine(llm=llm, sink=sink)

    await _drive(engine)
    reference = engine.snapshot()["last_request_manifest"]

    assert reference == {
        "manifest_id": sink.manifests[0].manifest_id,
        REQUEST_MANIFEST_SCHEMA_KEY: REQUEST_MANIFEST_SCHEMA_VERSION,
        "attempt_id": sink.manifests[0].attempt_id,
    }
    # By id means BY ID: none of the request's content is in the snapshot's
    # reference, however small the request happened to be.
    assert "messages" not in reference
    assert "inline" not in json.dumps(reference)


async def test_a_sink_that_fails_does_not_fail_the_run() -> None:
    """Evidence-keeping is not the run's job to succeed at. The id is stamped
    anyway, so the snapshot still names the call — a reader that cannot find
    the manifest learns it was not kept, which is a more useful fact than a
    reference that was never written."""
    llm = InMemoryLLMProvider()
    llm.queue_response(text="done", stop_reason=StopReason.end_turn)
    engine = _build_engine(llm=llm, sink=_RaisingSink())

    await _drive(engine)

    reference = engine.last_request_manifest
    assert reference is not None
    assert reference["manifest_id"]


# ── the size policy ─────────────────────────────────────────────────────────


async def test_a_value_over_the_threshold_travels_as_a_hash_and_a_blob_ref() -> None:
    """Ordered messages and full tool definitions are megabytes on a long run.
    Over the threshold the manifest carries the digest and the length, and the
    body goes to the store the host already runs."""
    llm = InMemoryLLMProvider()
    llm.queue_response(text="done", stop_reason=StopReason.end_turn)
    sink = InMemoryRequestManifestSink()
    engine = _build_engine(
        llm=llm,
        sink=sink,
        rc=LoopConstants(
            model_context_window=8_192,
            request_manifest_inline_value_max_bytes=32,
        ),
    )

    await _drive(engine, text="a" * 500)

    manifest = sink.manifests[0]
    assert manifest.messages.inline is None
    assert manifest.messages.blob_ref is not None
    assert len(manifest.messages.sha256) == 64
    assert manifest.messages.byte_length > 32
    # The body is retrievable, and it is what the digest says it is.
    body = await sink.body_of(manifest, "messages")
    assert len(body) == manifest.messages.byte_length
    assert "a" * 500 in body.decode("utf-8")
    # And the ref does not move the id — where a host put the bytes is not part
    # of what was sent.
    assert sink.get(manifest.manifest_id) is manifest


async def test_the_threshold_changes_what_travels_not_what_was_sent() -> None:
    """The same request manifested under two thresholds yields the same digests
    — only the carriage differs."""
    request = LLMRequest(
        model=MODEL,
        messages=[
            Message(role=MessageRole.user, content_blocks=[TextBlock(text="x" * 400)])
        ],
        max_tokens=64,
    )
    inline, no_bodies = build_request_manifest(
        request=request,
        attempt_scope="run/turn/run",
        constants_sha256="c" * 64,
        inline_value_max_bytes=100_000,
    )
    offloaded, bodies = build_request_manifest(
        request=request,
        attempt_scope="run/turn/run",
        constants_sha256="c" * 64,
        inline_value_max_bytes=16,
    )

    assert inline.messages.sha256 == offloaded.messages.sha256
    assert inline.request_sha256 == offloaded.request_sha256
    assert inline.attempt_id == offloaded.attempt_id
    assert no_bodies == {}
    assert "messages" in bodies
    # The ids differ, and must: a manifest that says "the body is here" is a
    # different record from one that says "the body is in the store".
    assert inline.manifest_id != offloaded.manifest_id


# ── fail-closed on a version this build cannot read ─────────────────────────


async def test_an_unknown_manifest_schema_version_is_refused_on_resume() -> None:
    llm = InMemoryLLMProvider()
    llm.queue_response(text="done", stop_reason=StopReason.end_turn)
    sink = InMemoryRequestManifestSink()
    source = _build_engine(llm=llm, sink=sink)
    await _drive(source)

    snapshot = source.snapshot()
    snapshot["last_request_manifest"][REQUEST_MANIFEST_SCHEMA_KEY] = (
        REQUEST_MANIFEST_SCHEMA_VERSION + 1
    )

    resumed = _build_engine(llm=InMemoryLLMProvider(), sink=sink)
    with pytest.raises(ManifestSchemaError, match="schema version"):
        await resumed.resume_from_snapshot(snapshot)

    # Refused with the engine untouched — the same discipline every other
    # snapshot refusal follows.
    assert resumed.history == []
    assert resumed.turn_count == 0
    assert resumed.last_request_manifest is None


async def test_a_reference_with_no_version_is_refused() -> None:
    llm = InMemoryLLMProvider()
    llm.queue_response(text="done", stop_reason=StopReason.end_turn)
    source = _build_engine(llm=llm, sink=InMemoryRequestManifestSink())
    await _drive(source)

    snapshot = source.snapshot()
    del snapshot["last_request_manifest"][REQUEST_MANIFEST_SCHEMA_KEY]

    resumed = _build_engine(llm=InMemoryLLMProvider())
    with pytest.raises(ManifestSchemaError):
        await resumed.resume_from_snapshot(snapshot)


@pytest.mark.parametrize("version", [True, "1", 1.0, 0, -3])
def test_only_an_integer_version_in_range_is_read(version: Any) -> None:
    with pytest.raises(ManifestSchemaError):
        read_manifest_schema_version({REQUEST_MANIFEST_SCHEMA_KEY: version})


async def test_an_absent_manifest_is_not_a_refusal() -> None:
    """Retention belongs to the host, so a manifest that was never kept — or
    has aged out — is evidence this run does not have, not a run that cannot
    continue."""
    llm = InMemoryLLMProvider()
    llm.queue_response(text="done", stop_reason=StopReason.end_turn)
    source = _build_engine(llm=llm, sink=InMemoryRequestManifestSink())
    await _drive(source)
    snapshot = source.snapshot()
    snapshot["last_request_manifest"] = None

    resumed = _build_engine(llm=InMemoryLLMProvider())
    await resumed.resume_from_snapshot(snapshot)

    assert resumed.last_request_manifest is None
    assert resumed.turn_count == source.turn_count


async def test_a_payload_written_before_the_field_existed_still_resumes() -> None:
    """The upcaster's honest lift: a run recorded under the older schema has no
    manifest, and says so."""
    llm = InMemoryLLMProvider()
    llm.queue_response(text="done", stop_reason=StopReason.end_turn)
    source = _build_engine(llm=llm, sink=InMemoryRequestManifestSink())
    await _drive(source)
    snapshot = source.snapshot()
    del snapshot["last_request_manifest"]
    run_state_payload = snapshot.pop(RUN_SCOPED_STATE_SNAPSHOT_KEY)
    snapshot["run_work_ledger"] = run_state_payload["run_work_ledger"]
    snapshot["subagent_tree_budget"] = run_state_payload["subagent_tree_budget"]
    snapshot["schema_version"] = 2

    resumed = _build_engine(llm=InMemoryLLMProvider())
    await resumed.resume_from_snapshot(snapshot)

    assert resumed.last_request_manifest is None


# ── the recorded run plays back ─────────────────────────────────────────────


async def test_a_recorded_run_replays_and_yields_the_same_manifest() -> None:
    """The acceptance the whole unit is for: drive a run against a provider,
    keep what it asked for, then re-drive it against nothing but the recording
    and get the identical manifest back."""
    llm = InMemoryLLMProvider()
    llm.queue_response(text="the answer", stop_reason=StopReason.end_turn)
    recorded_sink = InMemoryRequestManifestSink()
    await _drive(_build_engine(llm=llm, sink=recorded_sink))

    replay = ReplayLLMProvider.from_sink(
        recorded_sink, [_answer_stream("the answer")]
    )
    replay_sink = InMemoryRequestManifestSink()
    await _drive(_build_engine(llm=replay, sink=replay_sink))

    assert replay.exhausted
    assert [m.manifest_id for m in replay_sink.manifests] == [
        m.manifest_id for m in recorded_sink.manifests
    ]
    assert replay_sink.manifests[0].attempt_id == (
        recorded_sink.manifests[0].attempt_id
    )


async def test_a_run_that_compacted_records_the_summariser_call_and_replays() -> None:
    """A compaction is a provider call too, and the one a reader most needs.

    The summariser talks to the run's own provider by default, so a recording
    that held only the turn's calls would be short by one entry per summary —
    and the replay, which pairs calls with recorded answers one for one, would
    run off the end of its own recording. Worse, a compaction is precisely the
    event that rewrites the transcript every later request is built from, so
    it is the call an incident most needs explained.
    """
    compacting_rc = LoopConstants(
        model_context_window=4_096,
        compaction_trigger_ratio=0.2,
        compaction_keep_recent_turns=1,
        # One summary per answer, so the recording holds one call per turn.
        compaction_summary_group_max_tokens=0,
    )
    summary = "the old turns, summarised"

    def _seeded(engine: QueryEngine) -> QueryEngine:
        # One question and three answers: an operator turn is never summarised,
        # so the eligible units are the assistant turns.
        engine.history.append(
            Message(
                role=MessageRole.user,
                content_blocks=[TextBlock(text="question " + "y" * 1_500)],
            )
        )
        for index in range(3):
            engine.history.append(
                Message(
                    role=MessageRole.assistant,
                    content_blocks=[TextBlock(text=f"answer {index} " + "z" * 1_500)],
                )
            )
        return engine

    llm = InMemoryLLMProvider()
    for _ in range(3):
        llm.queue_response(text=summary)
    llm.queue_response(text="the answer", stop_reason=StopReason.end_turn)
    recorded_sink = InMemoryRequestManifestSink()
    await _drive(
        _seeded(_build_engine(llm=llm, sink=recorded_sink, rc=compacting_rc)),
        "and now answer",
    )

    attempts = [manifest.attempt_id for manifest in recorded_sink.manifests]
    assert sum("compaction_summary" in attempt for attempt in attempts) == 3, attempts
    assert "/run/" in attempts[-1]

    replay = ReplayLLMProvider.from_sink(
        recorded_sink,
        [_answer_stream(summary)] * 3 + [_answer_stream("the answer")],
    )
    replay_sink = InMemoryRequestManifestSink()
    await _drive(
        _seeded(_build_engine(llm=replay, sink=replay_sink, rc=compacting_rc)),
        "and now answer",
    )

    assert replay.exhausted
    assert [m.manifest_id for m in replay_sink.manifests] == [
        m.manifest_id for m in recorded_sink.manifests
    ]


async def test_the_replay_refuses_a_request_the_recording_does_not_hold() -> None:
    """A run whose assembly has drifted must not be served the answer to the
    question it used to ask.

    The drift is produced the way a real one is — by re-driving the same input
    against a tool surface that has lost a tool — and the refusal is asserted
    at the provider boundary, which is where a host would see it: inside the
    loop it becomes an ordinary failed run, and a failed run is what the
    recording is meant to explain rather than what it is meant to be."""
    llm = InMemoryLLMProvider()
    llm.queue_response(text="the answer", stop_reason=StopReason.end_turn)
    recorded_sink = InMemoryRequestManifestSink()
    await _drive(_build_engine(llm=llm, sink=recorded_sink))

    # One tool fewer on the surface: a real drift, and exactly the kind that
    # used to be invisible.
    drifted_llm = InMemoryLLMProvider()
    drifted_llm.queue_response(text="the answer", stop_reason=StopReason.end_turn)
    await _drive(_build_engine(llm=drifted_llm, sink=None, tools=("Read",)))

    replay = ReplayLLMProvider.from_sink(
        recorded_sink, [_answer_stream("the answer")]
    )
    with pytest.raises(ReplayMismatchError, match="different question"):
        [
            event
            async for event in replay.stream_with_tools(drifted_llm.calls[0])
        ]


async def test_the_replay_refuses_a_call_past_the_end_of_the_recording() -> None:
    replay = ReplayLLMProvider.from_recording([])
    request = LLMRequest(model=MODEL, messages=[], max_tokens=8)

    with pytest.raises(ReplayMismatchError, match="another one"):
        [event async for event in replay.stream_with_tools(request)]


async def test_replaying_under_a_new_run_id_is_not_a_mismatch() -> None:
    """Correlation metadata is not part of what was asked: the same request
    made by two runs is the same request, or a recording could only ever be
    replayed by the run that made it."""
    llm = InMemoryLLMProvider()
    llm.queue_response(text="the answer", stop_reason=StopReason.end_turn)
    recorded_sink = InMemoryRequestManifestSink()
    await _drive(_build_engine(llm=llm, sink=recorded_sink, run_id="run-a"))

    replay = ReplayLLMProvider.from_sink(
        recorded_sink, [_answer_stream("the answer")]
    )
    await _drive(_build_engine(llm=replay, sink=None, run_id="run-b"))

    assert replay.exhausted


async def test_the_replay_serves_the_non_streaming_calls_too() -> None:
    events = _answer_stream("summarised")
    request = LLMRequest(model=MODEL, messages=[], max_tokens=8)
    replay = ReplayLLMProvider.from_recording([(request_digest(request), events)])

    response = await replay.complete_structured(request, {})

    assert response.stop_reason is StopReason.end_turn
    assert response.message.content_blocks[0].text == "summarised"


# --- what the digest is allowed to be taken over ----------------------------


def test_every_message_field_is_either_sent_or_deliberately_left_out() -> None:
    """A new field on ``Message`` has to be classified, not silently dropped.

    The set of fields a provider sees drives both the request digest and the
    manifest's record of the messages. Add a field the model does see and
    forget this set, and two materially different requests digest the same:
    the replay serves a recorded answer to a question the run did not ask, and
    two different calls collide on one manifest id. Nothing about that is
    visible at the call site, which is why the inventory is asserted here.

    The two exclusions are safe for reasons that would have to change before
    they could be included: ``created_at`` is a wall clock, so including it
    would make the same request digest differently in two processes, and
    ``metadata`` is the runtime's own annotation, documented on the model as
    not sent.
    """
    assert set(Message.model_fields) == _MODEL_VISIBLE_MESSAGE_FIELDS | {
        "created_at",
        "metadata",
    }


def test_a_value_with_no_canonical_form_is_refused_rather_than_stringified() -> None:
    """A digest may not depend on a repr, because a repr can carry an address.

    ``LLMRequest.extra`` takes anything, and the digest is what a replay
    matches on. A value serialised as ``str(value)`` digests differently in
    two processes whenever its repr carries an identity — so the same request
    would fail to match its own recording, and a repr that happens to be
    stable would be worse still: two different values could collide. Refusing
    at build time is the outcome that cannot reach a replay as a false match.
    """

    class _Opaque:
        pass

    with pytest.raises(TypeError, match="canonical serialisation"):
        canonical_bytes({"x": _Opaque()})


def test_a_dataclass_in_the_extras_travels_as_its_fields() -> None:
    """The one non-JSON value the core puts in ``extra`` has a real form.

    A cache breakpoint is a frozen dataclass, and a dataclass has a synthesised
    repr — deterministic, so stringifying it happened to work, but only by
    luck and only until one of them held something whose repr is not. Its
    fields are what it is, and they digest the same everywhere.
    """
    breakpoint_value = CacheBreakpoint(
        message_index=0,
        cache_control_type="ephemeral",
        rationale="system_prefix",
    )
    payload = json.loads(canonical_bytes({"cache": breakpoint_value}))
    assert payload["cache"] == {
        "message_index": 0,
        "cache_control_type": "ephemeral",
        "rationale": "system_prefix",
    }
