"""Observability contracts — optional hooks the core exposes to its host.

Prompt-caching metrics belong to whatever the host already runs for
observability, so the producer lives outside the core and is wired in
through an injectable :class:`CacheObserverProtocol` carried on
:class:`protocore.runtime.query_engine.QueryEngineConfig`.

The shape is intentionally minimal — one method, four kwargs. Adding
fields to the recorder requires a contract bump; removing fields would
break host implementations silently. Keep this stable.

The second contract here answers a different question: not "how did this call
perform" but "what exactly was sent". A provider request is assembled from the
history, the compaction checkpoint, the pairing repair, the tool surface and
the constants in force, and until now it existed only in the stack frame that
made the call. Nothing durable could answer whether a run that behaved oddly
was sent a different request or got a different answer to the same one, and
nothing could re-drive a recorded run without paying for the tokens again.

:class:`RequestManifest` is that record. The core builds it, computes its id
and hands it to an :class:`IRequestManifestSink`; where it is kept, and for how
long, is the host's decision — the core has neither a store nor a retention
policy, and acquiring one mid-run is exactly the obligation the loop must not
take on. The id is a SHA-256 over the manifest's own canonical serialisation,
so it is known before the host has written anything, and a run snapshot can
address a manifest by id rather than carry it by value.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from protocore.contracts.llm import LLMRequest
from protocore.contracts.types import Message, MessageRole


@runtime_checkable
class CacheObserverProtocol(Protocol):
    """Optional sink for per-LLM-call prompt-caching observations.

    Implementations live in the host (Prometheus histograms, for
    instance). The runtime calls this once per
    ``ProviderDeltaKind.usage`` envelope, after the engine's
    :class:`~protocore.runtime.usage.TokenUsage` has been updated.

    The protocol is :func:`typing.runtime_checkable` so tests can verify
    a concrete implementation's shape without depending on its module.
    """

    def record_run_cache_hit_rate(
        self,
        *,
        tenant_id: str,
        cache_read_tokens: int,
        prompt_tokens: int,
        cache_breakpoint_count: int,
    ) -> None:
        """Record a single LLM-call cache observation.

        Arguments
        ---------
        tenant_id:
            The tenant this LLM call belongs to. Used as the primary
            label dimension on the underlying Prometheus histograms.
        cache_read_tokens:
            Tokens served from the provider's prompt cache for this call
            (Anthropic ``cache_read_input_tokens`` / OpenAI+vLLM
            ``cached_tokens``).
        prompt_tokens:
            Total prompt tokens for this call (``input_tokens`` from the
            provider usage envelope). Combined with ``cache_read_tokens``
            this yields the hit rate the implementation may bucket.
        cache_breakpoint_count:
            Number of :class:`~protocore.contracts.llm.CacheBreakpoint`
            hints attached to the originating
            :class:`~protocore.contracts.llm.LLMRequest`. Surfaces the
            placement-strategy effect on cache success.

        Implementations MUST be cheap (lock-free counter / histogram
        observation). The runtime calls this on the hot streaming path
        and does NOT spawn a thread / task to defer the call.
        """
        ...


#: Where the manifest states its own schema version.
REQUEST_MANIFEST_SCHEMA_KEY = "manifest_schema_version"

#: The manifest schema this build writes and can read.
REQUEST_MANIFEST_SCHEMA_VERSION = 1

#: The manifest fields that carry a value which may be too large to inline.
#: Named once so a host filling blob refs and the id computation that must
#: ignore them cannot drift apart.
MANIFEST_VALUE_SLOTS: tuple[str, ...] = (
    "system_prompt",
    "messages",
    "tools",
    "extra",
)


class ManifestSchemaError(ValueError):
    """A manifest reference cannot be read as the schema this build understands.

    Subclasses :class:`ValueError` for the same reason
    :class:`~protocore.contracts.snapshot.SnapshotSchemaError` does: every
    refusal a resume raises is one, so a caller that already treats a refused
    payload as "leave the run alone" keeps doing so unchanged.
    """


def canonical_bytes(value: Any) -> bytes:
    """The one serialisation every digest in this module is taken over.

    Sorted keys, no insignificant whitespace, UTF-8. Two processes that build
    the same request must produce the same bytes, so nothing here may depend on
    dict ordering, locale or the repr of a model class. A value with no
    canonical form is refused rather than stringified: a digest that quietly
    included a memory address would differ between two processes and turn a
    correct replay into a mismatch, or — worse, when the repr happens to be
    stable — let two different calls collide on one identifier.
    """
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_json_default,
    ).encode("utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=repr)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    raise TypeError(
        f"{type(value).__name__} has no canonical serialisation, so a digest "
        "over it would depend on its repr — and a repr can carry a memory "
        "address, which differs between two processes building the same "
        "request. Give the value a JSON form before it reaches a manifest."
    )


def digest_of(value: Any) -> str:
    """SHA-256 of :func:`canonical_bytes`, as hex."""
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def constants_digest(constants: BaseModel) -> str:
    """The digest of the constants a request was assembled under.

    Typed as a bare model rather than as ``LoopConstants`` so this module
    stays free of the several-thousand-field import; every caller passes the
    run's constants snapshot.
    """
    return digest_of(constants.model_dump(mode="json"))


class ManifestValue(BaseModel):
    """One part of a request, carried whole or carried by digest.

    A manifest holds the ordered messages and the full tool definitions, which
    on a long run is megabytes per call. So a part longer than the configured
    threshold travels as its digest and its length, and its body goes to the
    host's blob store — which is content-addressed and idempotent, so the same
    body put twice costs one object.

    ``blob_ref`` is filled by the HOST, after it has stored the body, and is
    deliberately excluded from the manifest id: where a host chose to put the
    bytes is not part of what was sent, and letting it into the id would mean
    the same request had two ids depending on whether the store had answered.
    """

    model_config = ConfigDict(frozen=True)

    sha256: str
    byte_length: int
    #: The canonical serialisation itself, present only below the threshold.
    #: ``None`` means the body is not here — ask the store.
    inline: str | None = None
    #: Set by the host once it has stored an oversized body.
    blob_ref: str | None = None

    @property
    def is_inline(self) -> bool:
        return self.inline is not None


class RequestManifest(BaseModel):
    """Exactly what one provider call was made of.

    Every field is either a value small enough to read at a glance or the
    digest of one that is not. Two manifests are equal as records of a request
    exactly when their ids are equal, and the id is a function of this payload
    alone — no clock, no host state, no store.
    """

    model_config = ConfigDict(frozen=True)

    manifest_schema_version: int = REQUEST_MANIFEST_SCHEMA_VERSION
    #: Stable across processes: the same request, rebuilt after a resume,
    #: carries the same attempt id. Derived rather than minted, so nothing has
    #: to be persisted for a re-drive to line up with what was recorded.
    attempt_id: str
    #: The model actually asked, which is the live override when one is in
    #: force rather than whatever the run was configured with.
    model: str
    #: How far down the provider chain the run had been demoted, and the rung's
    #: own name. ``None`` for a run with no chain.
    provider_chain_position: int = 0
    provider_chain_model: str | None = None
    system_prompt: ManifestValue
    messages: ManifestValue
    tools: ManifestValue
    extra: ManifestValue
    max_tokens: int
    #: ``None`` when the caller left the temperature to the host.
    temperature: float | None
    tool_count: int
    message_count: int
    #: The constants the request was assembled under. A prompt that changed
    #: because an operator retuned a constant is otherwise indistinguishable
    #: from one that changed because the agent did something different.
    constants_sha256: str
    #: The digest of the request itself — model, messages, tools, budgets and
    #: extras. This is what a replay provider keys on, and it deliberately
    #: excludes the correlation identity below: the same request made by two
    #: runs is the same request.
    request_sha256: str
    #: Non-secret correlation identity: which tenant, run, session and agent
    #: this call belongs to and what it was for.
    identity: Mapping[str, str | None] = Field(default_factory=dict)

    @property
    def manifest_id(self) -> str:
        """SHA-256 of this manifest's canonical serialisation, as hex.

        Computed, never stored: a field holding the id would be part of the
        payload the id is taken over. Host blob refs are excluded (see
        :class:`ManifestValue`), so an id is available before the host has
        written anything and does not change once it has.
        """
        return hashlib.sha256(self.identity_bytes()).hexdigest()

    def identity_bytes(self) -> bytes:
        """The bytes :attr:`manifest_id` is taken over."""
        return canonical_bytes(
            self.model_dump(
                mode="json",
                exclude={slot: {"blob_ref"} for slot in MANIFEST_VALUE_SLOTS},
            )
        )

    def with_blob_refs(self, refs: Mapping[str, str]) -> RequestManifest:
        """A copy naming where the host stored each oversized body.

        ``refs`` is keyed by slot name — the keys of the ``bodies`` mapping the
        sink was handed. :attr:`manifest_id` is unchanged by this.
        """
        unknown = sorted(set(refs) - set(MANIFEST_VALUE_SLOTS))
        if unknown:
            raise ValueError(f"not manifest value slots: {unknown}")
        updates: dict[str, ManifestValue] = {}
        for slot, ref in refs.items():
            value: ManifestValue = getattr(self, slot)
            updates[slot] = value.model_copy(update={"blob_ref": ref})
        return self.model_copy(update=updates)


def _manifest_value(payload: Any, *, inline_max_bytes: int) -> tuple[ManifestValue, bytes | None]:
    """One part of a request as a manifest value, plus the body to store.

    The body is returned separately rather than attached, because the core does
    not store it: the host does, and the core must be able to compute the whole
    manifest without waiting on anything.
    """
    body = canonical_bytes(payload)
    sha = hashlib.sha256(body).hexdigest()
    if len(body) <= inline_max_bytes:
        return (
            ManifestValue(
                sha256=sha, byte_length=len(body), inline=body.decode("utf-8")
            ),
            None,
        )
    return ManifestValue(sha256=sha, byte_length=len(body)), body


#: The fields of a :class:`~protocore.contracts.types.Message` a provider
#: actually sees. ``created_at`` is a wall clock and ``metadata`` is the
#: runtime's own annotation, documented on the model as not sent to the model —
#: neither is part of the request, and including either would make the digest
#: of one request differ from the digest of the same request. That is not a
#: nicety: it would make every manifest unreproducible and every replay of a
#: recorded run a mismatch on the strength of a microsecond.
_MODEL_VISIBLE_MESSAGE_FIELDS = frozenset({"role", "content_blocks", "reasoning_content"})


def model_visible(message: Message) -> dict[str, Any]:
    """``message`` reduced to what a provider is actually shown."""
    return message.model_dump(mode="json", include=set(_MODEL_VISIBLE_MESSAGE_FIELDS))


def _system_prompt_payload(messages: Sequence[Message]) -> list[Any]:
    """The leading system messages, which are the rendered system prompt.

    Read off the request rather than taken from the caller so the manifest
    states what was SENT — a section the assembly dropped is absent here too.
    """
    leading: list[Message] = []
    for message in messages:
        if message.role is not MessageRole.system:
            break
        leading.append(message)
    return [model_visible(item) for item in leading]


def request_digest(request: LLMRequest) -> str:
    """The digest of a request's content, ignoring who made it.

    ``observability`` is excluded deliberately: it is correlation metadata —
    run and session ids — and including it would make every replay of a
    recorded run a mismatch purely because it is a different run.
    """
    return digest_of(
        {
            "model": request.model,
            "messages": [model_visible(item) for item in request.messages],
            "tools": [item.model_dump(mode="json") for item in request.tools],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "extra": request.extra,
        }
    )


def build_request_manifest(
    *,
    request: LLMRequest,
    attempt_scope: str,
    constants_sha256: str,
    inline_value_max_bytes: int,
    provider_chain_position: int = 0,
    provider_chain_model: str | None = None,
) -> tuple[RequestManifest, dict[str, bytes]]:
    """The manifest of ``request``, and the oversized bodies a host must store.

    Pure and free of I/O: the id is available immediately, so a snapshot can
    address the manifest before the host's store has been touched — and stays
    correct if that store is never reachable at all.

    ``attempt_scope`` names the run and turn the call belongs to; the attempt
    id is that scope plus the head of the request digest, which makes it stable
    across processes without anything being persisted to mint it.
    """
    if inline_value_max_bytes < 0:
        raise ValueError("inline_value_max_bytes cannot be negative")
    bodies: dict[str, bytes] = {}
    values: dict[str, ManifestValue] = {}
    payloads: dict[str, Any] = {
        "system_prompt": _system_prompt_payload(request.messages),
        "messages": [model_visible(item) for item in request.messages],
        "tools": [item.model_dump(mode="json") for item in request.tools],
        "extra": request.extra,
    }
    for slot in MANIFEST_VALUE_SLOTS:
        value, body = _manifest_value(
            payloads[slot], inline_max_bytes=inline_value_max_bytes
        )
        values[slot] = value
        if body is not None:
            bodies[slot] = body
    digest = request_digest(request)
    observability = request.observability
    identity: dict[str, str | None] = (
        {}
        if observability is None
        else {
            key: value
            for key, value in observability.model_dump(mode="json").items()
            if value is not None
        }
    )
    manifest = RequestManifest(
        attempt_id=f"{attempt_scope}/{digest[:16]}",
        model=request.model,
        provider_chain_position=provider_chain_position,
        provider_chain_model=provider_chain_model,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        tool_count=len(request.tools),
        message_count=len(request.messages),
        constants_sha256=constants_sha256,
        request_sha256=digest,
        identity=identity,
        system_prompt=values["system_prompt"],
        messages=values["messages"],
        tools=values["tools"],
        extra=values["extra"],
    )
    return manifest, bodies


def read_manifest_schema_version(reference: Mapping[str, Any]) -> int:
    """The manifest schema version ``reference`` declares, or raise.

    Fail-closed, and for the same reason the snapshot's own reader is: a
    manifest reference this build cannot place is one whose absent fields are
    indistinguishable from fields that were legitimately empty. An unreadable
    version is refused rather than read as far as it goes.

    ``bool`` is rejected explicitly — it is an ``int`` subclass, so a payload
    carrying ``manifest_schema_version: true`` would otherwise read as 1.

    A future version brings its own upcaster here, registered beside this
    reader: the manifest versions itself independently of the snapshot,
    because the snapshot addresses a manifest by id rather than copying it.
    """
    if not isinstance(reference, Mapping):
        raise ManifestSchemaError(
            f"manifest reference must be a mapping, got {type(reference).__name__}"
        )
    if REQUEST_MANIFEST_SCHEMA_KEY not in reference:
        raise ManifestSchemaError(
            f"manifest reference states no {REQUEST_MANIFEST_SCHEMA_KEY!r}; a "
            "reference has carried one since the first version that wrote any"
        )
    version: object = reference[REQUEST_MANIFEST_SCHEMA_KEY]
    if isinstance(version, bool) or not isinstance(version, int):
        raise ManifestSchemaError(
            f"manifest {REQUEST_MANIFEST_SCHEMA_KEY!r} must be an integer, got "
            f"{type(version).__name__}"
        )
    if version < 1 or version > REQUEST_MANIFEST_SCHEMA_VERSION:
        raise ManifestSchemaError(
            f"manifest schema version {version} is not one this build reads "
            f"(1..{REQUEST_MANIFEST_SCHEMA_VERSION}); no upcaster brings it "
            "forward"
        )
    return version


@runtime_checkable
class IRequestManifestSink(Protocol):
    """Where the core hands the record of a provider call it is about to make.

    One operation, which returns nothing and reads nothing back. The core does
    not learn where the manifest went, does not wait to find out whether it was
    kept, and never asks for it again — a run whose manifests were discarded is
    a run without that evidence, not a run that fails.

    ``bodies`` holds the oversized parts, keyed by the manifest field they
    belong to. A host stores each one — its blob store is content-addressed and
    idempotent, so re-storing a body it already has is free — and may keep
    :meth:`RequestManifest.with_blob_refs` of the manifest so the reference is
    recorded beside the digest. Retention is entirely the host's: the core
    neither deletes a body nor depends on one existing later.
    """

    async def record_request_manifest(
        self,
        *,
        manifest: RequestManifest,
        manifest_id: str,
        bodies: Mapping[str, bytes],
    ) -> None:
        """Accept one manifest, promptly. Must not raise on a full store.

        The core awaits this on the hot path, between the request being
        assembled and the provider being asked, and it is awaited in order so
        that the sequence of manifests is the sequence of calls — which is
        what makes a recorded run replayable. So the requirement is a hard one
        rather than advice: an implementation returns as soon as it has taken
        custody of the manifest. A host that stores bodies through a blob
        store, or anything else that can be slow or can stall, hands the work
        to its own queue and returns; it does not do that work here. A sink
        that blocks the event loop stalls every other run in the process, and
        one that hangs stalls this run with nothing to time it out.

        :class:`~protocore.conformance.request_manifest.RequestManifestSinkConformance`
        checks this against a host's own adapter.
        """
        ...


__all__ = [
    "MANIFEST_VALUE_SLOTS",
    "REQUEST_MANIFEST_SCHEMA_KEY",
    "REQUEST_MANIFEST_SCHEMA_VERSION",
    "CacheObserverProtocol",
    "IRequestManifestSink",
    "ManifestSchemaError",
    "ManifestValue",
    "RequestManifest",
    "build_request_manifest",
    "canonical_bytes",
    "constants_digest",
    "digest_of",
    "model_visible",
    "read_manifest_schema_version",
    "request_digest",
]
