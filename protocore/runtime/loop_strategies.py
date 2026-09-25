"""Agent loop strategies — Direct vs Deep (SGR) — for the harness modes.

Two orthogonal axes drive the harness:

* **loop strategy** ``{direct, deep}`` — this module;
* **native thinking** ``{off, on}`` (bounded by ``reasoning_effort``) — threaded
  onto ``LLMRequest.extra`` directly in :mod:`protocore.runtime.query`.

``query()`` branches **once** on ``engine.config.run_mode`` via
:func:`select_strategy` and calls :meth:`AgentLoop.prepare_turn` immediately
before the shared assistant-message loop
(:func:`protocore.runtime.query._stream_one_assistant_message`). Everything the
mature loop already owns — context build, tool dispatch, tool_use<->tool_result
pairing, recovery/repair, loop-detection, the terminal gate — stays SHARED; the
strategy only contributes the **pre-action** step.

* :class:`DirectStrategy` — no pre-action step (today's auto-tool loop,
  byte-unchanged).
* :class:`DeepStrategy` — the stand-validated SGR step: a forced ``plan`` tool
  call (native ``tool_choice=plan`` + native CoT bounded by
  ``reasoning_effort``) recorded as exactly one ``REASONING_STEP`` event, then
  the shared loop drives the real action with the full surface.

Measured against a vLLM endpoint: the forced
native ``plan`` tool RESPECTS the ``next_tool`` enum (unlike ``guided_json``,
which let the model invent ``create_html_file``); native CoT already carries the
"why", so the lean schema ``{plan, next_tool, task_complete}`` is the cheapest
enforcing shape (195 plan tokens vs 252 full). The plan-tool schema below is
lifted verbatim from ``cases/case_c_sgr.py::_plan_tool``.
"""
# ruff: noqa: RUF001 — Cyrillic prompt strings are intentional (bilingual RU+EN
# SGR plan-fallback instruction).
from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from protocore.contracts.llm import (
    LLMContextWindowExceeded,
    LLMError,
    ProviderDeltaKind,
)
from protocore.contracts.types import (
    Message,
    MessageRole,
    TextBlock,
    ToolDefinition,
    ToolParameterSchema,
    ToolResultBlock,
    ToolUseBlock,
)
from protocore.json_utils import (
    OutputParserException,
    parse_complete_json,
    strip_thinking,
    structured_json_candidates,
)
from protocore.logging_utils import get_logger
from protocore.runtime.context.manager import ContextBundle
from protocore.runtime.events import EventType, TurnEvent
from protocore.runtime.llm.delta_bridge import stream_events_to_provider_deltas

if TYPE_CHECKING:
    from protocore.contracts.llm import ProviderDelta
    from protocore.runtime.query_engine import QueryEngine

_logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# SGR plan tool — forced, native (ENFORCES on vLLM)
# ---------------------------------------------------------------------------

# PascalCase per the tool-naming convention (tools-initiative A1, 2026-06-06).
# Internal Deep-mode forced call only — never in the standing surface; the
# REASONING_STEP payload keys (``plan``/``next_tool``/``task_complete``) are
# field names, not this tool name, and stay unchanged.
PLAN_TOOL_NAME = "Plan"
PLAN_SUMMARY_MAX_LENGTH = 280
"""``reasoning_summary`` cap (chars) — measured at roughly +27 plan tokens."""

# Transient engine attribute carrying the reasoning_content captured during the
# plan stream, from ``_fetch_plan``/``_fetch_plan_fallback`` to
# ``_append_plan_turn`` (which records the planning-turn assistant Message). The
# established in-file idiom for per-turn scratch state on the engine (cf.
# ``engine._outbound_system_normalized_warned``); consumed-and-cleared on use.
_DEEP_PLAN_REASONING_ATTR = "_deep_plan_reasoning"

# Provider-robust plan fallback ------------------------------------------
#
# The PRIMARY plan elicitation is the forced native ``Plan`` tool
# (``tool_choice=Plan`` + strict ``additionalProperties:false`` schema). vLLM /
# qwen / OpenAI / OpenRouter accept it and it ENFORCES the ``next_tool`` enum
# as measured. Some OpenAI-compatible providers — notably DeepSeek's
# ``api.deepseek.com`` — REJECT a strict forced-tool / ``json_schema`` request
# with an HTTP 400, which the adapter classifies as a fallback-worthy
# :class:`~protocore.contracts.llm.LLMProviderError` (``classified.should_fallback``).
# Before this fix that error made :meth:`DeepStrategy._fetch_plan` return
# ``None`` → Deep silently degraded to Direct. The fallback below re-elicits the
# SAME ``{plan, next_tool, task_complete}`` object as PROMPTED JSON (no forced
# tool, no strict schema), which every OpenAI-compatible provider accepts. This
# stays UNIVERSAL: it is gated purely on the classified error (never a hardcoded
# provider name), the forced path is still primary, and the parsed dict flows
# through the UNCHANGED ``_validated_plan_args`` + REASONING_STEP emission.

# DeepSeek's ``json_object`` mode REQUIRES the literal word "json" in the prompt
# and rejects ``json_schema``; ``json_object`` + plain are the two rungs every
# such provider survives. Ordered most-to-least constrained; the first rung the
# provider accepts wins (mirrors ``openai_compat_client.complete_structured``).
_PLAN_FALLBACK_RESPONSE_FORMATS: tuple[dict[str, Any] | None, ...] = (
    {"type": "json_object"},
    None,
)


def plan_tool_schema(surface_names: list[str], include_summary: bool) -> dict[str, Any]:
    """Return the OpenAI-style forced ``plan`` tool schema (stand verbatim).

    ``surface_names`` constrains ``next_tool`` to an ``enum`` of the live
    per-turn surface so the model can only plan a tool it can actually call.
    ``include_summary`` prepends an optional short ``reasoning_summary`` field
    (for the ``reasoning_step`` UI trace); default-off keeps the lean shape.
    """
    props: dict[str, Any] = {
        "plan": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "next_tool": {"type": "string", "enum": surface_names},
        "task_complete": {"type": "boolean"},
    }
    required = ["plan", "next_tool", "task_complete"]
    if include_summary:
        props = {
            "reasoning_summary": {"type": "string", "maxLength": PLAN_SUMMARY_MAX_LENGTH},
            **props,
        }
        required = ["reasoning_summary", *required]
    return {
        "type": "function",
        "function": {
            "name": PLAN_TOOL_NAME,
            "description": (
                "Record reasoning before acting: an ordered plan and the "
                "single next tool to call."
            ),
            "parameters": {
                "type": "object",
                "properties": props,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


def _plan_tool_definition(
    surface_names: list[str], include_summary: bool
) -> ToolDefinition:
    """Adapt the plan schema to the core :class:`ToolDefinition` carried on
    :attr:`LLMRequest.tools`. The host adapter renders this to the
    OpenAI tool shape; the forced ``extra['tool_choice']`` is what enforces."""
    schema = plan_tool_schema(surface_names, include_summary)
    fn = schema["function"]
    params = fn["parameters"]
    return ToolDefinition(
        name=fn["name"],
        description=fn["description"],
        parameters=ToolParameterSchema(
            properties=params["properties"],
            required=params["required"],
            # The plan schema freezes ``additionalProperties: false`` —
            # carry it onto the live ``LLMRequest.tools`` entry (not just the
            # standalone helper) so a strict-capable adapter forbids extra
            # model-invented fields. ``None`` elsewhere keeps every other tool
            # permissive/byte-unchanged.
            additional_properties=params["additionalProperties"],
        ),
    )


# ---------------------------------------------------------------------------
# Strategy protocol + implementations
# ---------------------------------------------------------------------------


@runtime_checkable
class AgentLoop(Protocol):
    """Per-turn loop strategy seam.

    :meth:`prepare_turn` runs ONCE per ``query()`` turn, before the shared
    assistant-message loop. It is an async generator that may yield
    pre-action :class:`TurnEvent` envelopes (the Deep plan step emits one
    ``REASONING_STEP``) and returns the :class:`ContextBundle` the shared
    loop should drive with — unchanged for Direct, rebuilt to include the
    planning turn for Deep.
    """

    def prepare_turn(
        self,
        engine: QueryEngine,
        context: ContextBundle,
    ) -> AsyncIterator[TurnEvent]:
        """Yield any pre-action events; the returned context is read via
        ``engine`` state mutated here (history) + a fresh
        :meth:`ContextManager.build_context`. Declared without ``async def``
        per the async-iterator Protocol convention."""
        ...


class DirectStrategy:
    """Direct mode — no pre-action step (today's behaviour, byte-unchanged)."""

    async def prepare_turn(
        self,
        engine: QueryEngine,
        context: ContextBundle,
    ) -> AsyncIterator[TurnEvent]:
        # Direct mode contributes no pre-action step. The native-thinking
        # axis (the Direct-Thinking preset) is threaded into
        # ``LLMRequest.extra`` by the shared stream in ``query.py`` — it is
        # NOT a loop-strategy concern.
        del engine, context
        return
        yield  # pragma: no cover - makes this an async generator


class DeepStrategy:
    """Deep (SGR) mode — one forced ``plan`` step → ``REASONING_STEP`` → action.

    The plan step is the stand's first call: ``tool_choice=plan`` over a
    plan-only surface with native CoT on (bounded by ``reasoning_effort``).
    Its parsed args become exactly one ``REASONING_STEP`` event and a recorded
    planning turn; the shared loop then drives the real action with the full
    surface. The plan never dispatches a side-effecting tool itself — it only
    records intent, mirroring ``case_c_sgr.py``.
    """

    async def prepare_turn(
        self,
        engine: QueryEngine,
        context: ContextBundle,
    ) -> AsyncIterator[TurnEvent]:
        surface_names = [t.name for t in context.tools]
        if not surface_names:
            # No surface to plan over (e.g. an empty-policy tenant). Skip the
            # plan step rather than force an enum-less tool — the shared loop
            # still runs. Universal/no-crash; the surface-pin fix is what
            # keeps the surface non-empty for natural RU prompts.
            _logger.warning(
                "DIAG deep_strategy.plan_skipped_empty_surface run=%s tenant=%s",
                engine.config.run_id,
                engine.config.tenant_id,
            )
            return

        plan_args = await self._fetch_plan(engine, context, surface_names)
        if plan_args is None:
            # The forced plan call yielded no parseable tool args. Degrade to
            # the shared loop (which still acts) rather than wedging the turn.
            _logger.warning(
                "DIAG deep_strategy.plan_unparsed run=%s tenant=%s",
                engine.config.run_id,
                engine.config.tenant_id,
            )
            return

        # Emit exactly one REASONING_STEP carrying the structured plan.
        yield TurnEvent(
            type=EventType.REASONING_STEP,
            run_id=engine.config.run_id,
            payload=self._reasoning_step_payload(engine, plan_args),
        )

        # Record the plan into history so the action turn sees it as context,
        # then refresh the context bundle. The tool_use is paired with a
        # synthetic tool_result so the outbound wire stays pairing-valid.
        self._append_plan_turn(engine, plan_args)

    # -- internals ----------------------------------------------------------

    async def _fetch_plan(
        self,
        engine: QueryEngine,
        context: ContextBundle,
        surface_names: list[str],
    ) -> dict[str, Any] | None:
        """Run the forced single ``plan`` tool call; return its parsed args.

        Reuses the shared provider-delta bridge so the InMemory mock
        (LLMStreamEvent) and the host vLLM adapter (ProviderDelta) are
        consumed identically. Only the FINAL parsed tool args are needed; the
        native CoT continues to flow as ``thinking`` deltas and is not
        re-emitted here (the loop's main stream surfaces CoT to the UI).

        The outbound message list is assembled with the SAME wire path the
        action stream uses (``_prepend_system_sections`` + the unconditional
        ``_repair_outbound_tool_pairing``), so the plan call sees the always-on
        tool-use scaffolding + persona and never ships an orphaned
        tool_use. The stream is wrapped in the shared idle/stall watchdog so a
        provider that wedges before the final plan args cannot hang the run.
        """
        from protocore.runtime.query import (
            _iter_with_idle_watchdog,
            _manifest_request,
            _normalize_outbound_system_messages,
            _observability_context,
            _prepend_system_sections,
            _repair_outbound_tool_pairing,
            build_llm_request,
        )
        from protocore.runtime.request_budget import fit_request_to_context

        rc = engine.config.rc
        include_summary = bool(getattr(rc, "agent_deep_plan_include_summary", False))
        plan_tool = _plan_tool_definition(surface_names, include_summary)

        full_messages = _prepend_system_sections(
            context.system_prompt_sections, context.messages
        )
        full_messages = _repair_outbound_tool_pairing(
            full_messages,
            placeholder=engine.prompt_text("tool_result_pairing_repair"),
        )
        # vLLM-400 backstop (same as the main action stream): any non-leading
        # ``system`` message → ``user``. The genuine system prefix at index 0
        # is untouched.
        full_messages, _converted_system = _normalize_outbound_system_messages(
            full_messages
        )
        if _converted_system and not getattr(
            engine, "_outbound_system_normalized_warned", False
        ):
            engine._outbound_system_normalized_warned = True  # type: ignore[attr-defined]
            _logger.warning(
                "normalized %d non-leading system message(s) to user role at "
                "the plan-call boundary (vLLM-400 guard)",
                _converted_system,
            )

        request = build_llm_request(
            model=engine.effective_model_name,
            messages=full_messages,
            tools=[plan_tool],
            max_tokens=_plan_max_tokens(engine, context),
            # Force the plan tool natively (a provider adapter renders the
            # name into its wire ``tool_choice``; it is what ENFORCES the
            # schema — ``guided_json`` was measured to be a no-op on vLLM).
            forced_tool_choice=PLAN_TOOL_NAME,
            # Deep mode = native CoT ON, bounded by reasoning_effort so the
            # plan call does not get its answer truncated.
            thinking_enabled=True,
            reasoning_effort=engine.effective_reasoning_effort,
            observability=_observability_context(
                engine,
                call_purpose="deep_plan",
                # This is the planning pass, so it says so. It used to say "run",
                # which is not a member of the category set the routing surface
                # publishes at all — the field is typed ``str | None`` here, so
                # nothing rejected it, and the ledger carried a category no grid
                # row could ever match.
                #
                # Labelling it correctly does NOT make it route: the engine holds
                # one provider and builds this request through it, so serving the
                # plan pass from a different model would mean accepting a second
                # provider per purpose — a contract change, not a fix.
                call_category="planning",
            ),
        )
        try:
            request = fit_request_to_context(request, rc)
        except LLMContextWindowExceeded as exc:
            _logger.warning(
                "DIAG deep_strategy.plan_context_overflow run=%s tenant=%s error=%s",
                engine.config.run_id,
                engine.config.tenant_id,
                type(exc).__name__,
            )
            return None

        # Only the forced ``plan`` tool's final args are accepted. The model is
        # pinned to ``tool_choice=plan`` and the surface is plan-only, but a
        # misbehaving provider could still echo a different tool name; never
        # let a non-plan tool's input masquerade as the SGR plan (it would emit
        # a bogus REASONING_STEP). Track the plan call id so a streamed
        # tool_name only on the start delta still matches the final-args delta.
        plan_call_id: str | None = None
        raw_args: dict[str, Any] | None = None
        # Fix #4 — accumulate the plan stream's native CoT (``thinking`` deltas)
        # exactly as the main assistant stream accumulates ``reasoning_buffer``
        # (query.py ``_drive_one_stream``). It is threaded to ``_append_plan_turn``
        # so the recorded planning-turn assistant Message carries
        # ``reasoning_content`` — deepseek thinking-mode 400s on the ACTION turn
        # otherwise ("reasoning_content ... must be passed back to the API"),
        # because the synthetic planning turn previously had it None.
        reasoning_parts: list[str] = []
        await _manifest_request(engine, request, call_purpose="deep_plan")
        plan_stream = _iter_with_idle_watchdog(
            _normalised_deltas(engine.llm.stream_with_tools(request)),
            idle_timeout=rc.llm_stream_idle_timeout_seconds,
            stall_threshold=rc.llm_stream_stall_threshold_seconds,
            reasoning_idle_timeout=rc.llm_stream_reasoning_idle_timeout_seconds,
        )
        try:
            async for delta in plan_stream:
                # Account the plan call's token usage so deep runs do not
                # undercount billing/cache metrics by the pre-action call
                # (mirrors the main stream's usage handling in query.py).
                if delta.kind is ProviderDeltaKind.usage and delta.usage:
                    self._record_plan_usage(engine, delta.usage)
                    continue
                if delta.kind is ProviderDeltaKind.thinking and delta.content:
                    reasoning_parts.append(delta.content)
                    continue
                if delta.tool_name == PLAN_TOOL_NAME and delta.tool_call_id:
                    plan_call_id = delta.tool_call_id
                is_plan = delta.tool_name == PLAN_TOOL_NAME or (
                    plan_call_id is not None and delta.tool_call_id == plan_call_id
                )
                if is_plan and isinstance(delta.tool_input_final, dict):
                    raw_args = dict(delta.tool_input_final)
        except LLMError as exc:
            # A provider stall/error on the plan call must NOT leak an
            # uncaught exception type out of ``prepare_turn``.
            #
            # Provider-robust fallback (Fix #1): a strict forced-tool /
            # ``json_schema`` REJECTION (DeepSeek-class HTTP 400) is surfaced by
            # the adapter as a fallback-worthy ``LLMProviderError`` — its
            # classifier verdict has ``should_fallback=True``. For ONLY those
            # errors, re-elicit the plan WITHOUT the forced tool / strict schema
            # (prompted JSON) so Deep produces a real plan instead of silently
            # degrading. This is gated purely on the classified error (never a
            # hardcoded provider name) so vLLM/qwen/OpenAI/OpenRouter — which
            # accept the forced path and never raise here — are byte-unchanged.
            if _is_fallback_worthy(exc):
                _logger.warning(
                    "DIAG deep_strategy.plan_forced_rejected_fallback run=%s "
                    "tenant=%s error=%s",
                    engine.config.run_id,
                    engine.config.tenant_id,
                    type(exc).__name__,
                )
                return await self._fetch_plan_fallback(
                    engine,
                    full_messages,
                    surface_names,
                    include_summary,
                    max_tokens=_plan_max_tokens(engine, context),
                )
            # Any other provider stall/error: degrade to the shared action loop
            # (which still runs and will surface the SAME provider failure
            # through the mature terminal-error / fallback path) rather than
            # wedging the deep turn before it starts.
            _logger.warning(
                "DIAG deep_strategy.plan_llm_error run=%s tenant=%s error=%s",
                engine.config.run_id,
                engine.config.tenant_id,
                type(exc).__name__,
            )
            return None
        validated = self._validated_plan_args(raw_args, surface_names, include_summary)
        if validated is not None:
            # Stash the captured CoT for ``_append_plan_turn`` (consumed-and-
            # cleared there) ONLY when a valid plan will be recorded — the
            # degrade path (validated is None) records no plan turn, so no
            # stale reasoning is left on the engine. The forced/vLLM path emits
            # native ``thinking`` deltas.
            setattr(engine, _DEEP_PLAN_REASONING_ATTR, "".join(reasoning_parts))
        return validated

    async def _fetch_plan_fallback(
        self,
        engine: QueryEngine,
        full_messages: list[Message],
        surface_names: list[str],
        include_summary: bool,
        *,
        max_tokens: int,
    ) -> dict[str, Any] | None:
        """Re-elicit the SGR plan as PROMPTED JSON when the forced tool is rejected.

        Used only after the forced ``Plan`` tool call raised a fallback-worthy
        provider error (DeepSeek-class strict-schema HTTP 400). The plan object
        (``{plan, next_tool, task_complete[, reasoning_summary]}``) is requested
        in a synthetic instruction — describing the shape + the allowed
        ``next_tool`` enum — with NO ``tools`` and NO forced ``tool_choice``. We
        try ``response_format={"type":"json_object"}`` first (DeepSeek requires
        the literal word "json" in the prompt — the instruction includes it),
        then plain text, stepping down only on a further fallback-worthy
        rejection (mirroring ``complete_structured``'s ladder). The returned
        text is parsed with the tolerant :mod:`protocore.json_utils` parser and
        fed to the UNCHANGED :meth:`_validated_plan_args`. Returns ``None`` (→
        :meth:`prepare_turn` degrades, keeping the ``plan_unparsed`` WARNING) if
        every rung fails to yield a parseable, schema-valid plan.

        Native CoT is NOT requested here: ``enable_thinking``/``reasoning_effort``
        are deliberately omitted so the degrade path ships only the knobs every
        OpenAI-compatible provider accepts; any ``<think>`` the model still emits
        is stripped by ``strip_thinking`` before the parse.
        """
        # Local import — avoid a circular import with ``runtime.query`` (the
        # same pattern ``_fetch_plan`` uses for the shared wire-path helpers).
        from protocore.runtime.query import (
            _iter_with_idle_watchdog,
            _manifest_request,
            _observability_context,
            build_llm_request,
        )
        from protocore.runtime.request_budget import fit_request_to_context

        rc = engine.config.rc
        instruction = _plan_fallback_instruction(surface_names, include_summary)
        fallback_messages: list[Message] = [
            *full_messages,
            Message(
                role=MessageRole.user,
                content_blocks=[TextBlock(text=instruction)],
            ),
        ]
        # ``response_format`` is a plain wire field for OpenAI-compatible
        # endpoints — an adapter forwards unknown ``extra`` keys onto the
        # request body verbatim, and ``None`` means the rung ships none.
        for response_format in _PLAN_FALLBACK_RESPONSE_FORMATS:
            request = build_llm_request(
                model=engine.effective_model_name,
                messages=fallback_messages,
                tools=[],
                max_tokens=max_tokens,
                response_format=response_format,
                observability=_observability_context(
                    engine,
                    call_purpose="deep_plan_fallback",
                    call_category="planning",
                ),
            )
            try:
                request = fit_request_to_context(request, rc)
            except LLMContextWindowExceeded as exc:
                _logger.warning(
                    "DIAG deep_strategy.plan_fallback_context_overflow run=%s "
                    "tenant=%s error=%s",
                    engine.config.run_id,
                    engine.config.tenant_id,
                    type(exc).__name__,
                )
                return None
            text_parts: list[str] = []
            # Fix #4 — also capture any native CoT the fallback rung emits. The
            # ``json_object`` rung keeps CoT off, but the ``plain`` rung may still
            # produce ``thinking`` deltas; threaded to ``_append_plan_turn`` so
            # the planning turn carries ``reasoning_content`` (deepseek echo
            # contract). When empty, ``_append_plan_turn`` falls back to the
            # plan's ``reasoning_summary`` / a placeholder.
            reasoning_parts: list[str] = []
            await _manifest_request(
                engine, request, call_purpose="deep_plan_fallback"
            )
            plan_stream = _iter_with_idle_watchdog(
                _normalised_deltas(engine.llm.stream_with_tools(request)),
                idle_timeout=rc.llm_stream_idle_timeout_seconds,
                stall_threshold=rc.llm_stream_stall_threshold_seconds,
                reasoning_idle_timeout=rc.llm_stream_reasoning_idle_timeout_seconds,
            )
            try:
                async for delta in plan_stream:
                    if delta.kind is ProviderDeltaKind.usage and delta.usage:
                        self._record_plan_usage(engine, delta.usage)
                        continue
                    if delta.kind is ProviderDeltaKind.thinking and delta.content:
                        reasoning_parts.append(delta.content)
                        continue
                    if delta.kind is ProviderDeltaKind.text and delta.content:
                        text_parts.append(delta.content)
            except LLMError as exc:
                if _is_fallback_worthy(exc) and response_format is not None:
                    # This rung's ``response_format`` was itself rejected; step
                    # down to the next (plain) rung. A non-fallback-worthy error
                    # (or the plain rung failing) stops the ladder.
                    _logger.warning(
                        "DIAG deep_strategy.plan_fallback_format_rejected run=%s "
                        "tenant=%s error=%s",
                        engine.config.run_id,
                        engine.config.tenant_id,
                        type(exc).__name__,
                    )
                    continue
                _logger.warning(
                    "DIAG deep_strategy.plan_fallback_llm_error run=%s tenant=%s "
                    "error=%s",
                    engine.config.run_id,
                    engine.config.tenant_id,
                    type(exc).__name__,
                )
                return None
            # The rung's CALL SUCCEEDED (the provider ACCEPTED the request). The
            # step-down ladder is REJECTION-only: a rung the provider accepted is
            # the authoritative plan attempt. If its text fails to parse OR
            # ``_validated_plan_args`` rejects it (malformed JSON, ``next_tool``
            # not in the surface, missing ``task_complete``, …), the plan is
            # INVALID and must be REJECTED — exactly as the forced-tool path
            # returns ``None`` on ``_validated_plan_args`` failure — NOT retried
            # on the looser ``plain`` rung (which could accept a DIFFERENT,
            # hallucinated plan). Stepping down happens ONLY in the
            # ``except LLMError`` branch above, when the provider REJECTED the
            # rung (``_is_fallback_worthy``).
            parsed = _parse_plan_text("".join(text_parts))
            validated = self._validated_plan_args(
                parsed, surface_names, include_summary
            )
            if validated is None:
                _logger.warning(
                    "DIAG deep_strategy.plan_fallback_unparsed run=%s tenant=%s "
                    "response_format=%s",
                    engine.config.run_id,
                    engine.config.tenant_id,
                    response_format.get("type") if response_format else "plain",
                )
                return None
            # Stash the captured CoT for ``_append_plan_turn`` (consumed-and-
            # cleared there); typically empty on the json_object rung, in which
            # case ``_append_plan_turn`` uses the summary/placeholder fallback.
            setattr(engine, _DEEP_PLAN_REASONING_ATTR, "".join(reasoning_parts))
            return validated
        return None

    @staticmethod
    def _record_plan_usage(engine: QueryEngine, usage: dict[str, Any]) -> None:
        """Fold the plan call's usage envelope into ``engine.total_usage``.

        Same arithmetic as the action stream (query.py) so deep runs do not
        silently undercount input/output/cache tokens by the plan call. Also
        feeds the optional cache observer one observation when present.
        """
        cache_read = int(
            usage.get("cache_read_input_tokens", 0)
            or usage.get("cache_read_tokens", 0)
            or usage.get("cached_tokens", 0)
        )
        cache_creation = int(usage.get("cache_creation_input_tokens", 0))
        input_tokens = int(usage.get("input_tokens", 0))
        engine.total_usage.add(
            input_tokens=input_tokens,
            output_tokens=int(usage.get("output_tokens", 0)),
            cache_read_tokens=cache_read,
            cache_creation_tokens=cache_creation,
        )
        # Also against the tree-cumulative ledger, for the same reason this
        # method exists at all: the plan call is a real LLM call, and a deep run
        # that skipped it here would under-count the tree's total work by one
        # call per turn — on the strategy that makes the most of them.
        engine.run_state.ensure_run_work_ledger(engine.config.rc).charge_tokens(
            input_tokens=input_tokens,
            output_tokens=int(usage.get("output_tokens", 0)),
        )
        observer = engine.config.cache_observer
        if observer is not None:
            observer.record_run_cache_hit_rate(
                tenant_id=engine.config.tenant_id,
                cache_read_tokens=cache_read,
                prompt_tokens=input_tokens,
                cache_breakpoint_count=0,
            )

    @staticmethod
    def _validated_plan_args(
        raw_args: dict[str, Any] | None,
        surface_names: list[str],
        include_summary: bool,
    ) -> dict[str, Any] | None:
        """Validate + CANONICALIZE parsed plan args against the plan schema.

        Returns a NEW dict containing only the plan schema keys (``plan``,
        ``next_tool``, ``task_complete``, and ``reasoning_summary`` when the
        summary variant is on) — so even if a provider ignores the strict
        ``additionalProperties: false`` and echoes extra keys, those keys never
        reach ``REASONING_STEP`` history / snapshots / future prompts. Returns
        ``None`` (→ :meth:`prepare_turn` degrades to the shared loop, which
        still acts) when the payload violates the schema: ``plan`` must be a
        non-empty list of strings, ``next_tool`` must be one of the live
        surface, ``task_complete`` must be a real bool, and ``reasoning_summary``
        must be a bounded string — REQUIRED when ``include_summary`` (the
        schema marks it required in that mode), optional+bounded otherwise.
        """
        if not isinstance(raw_args, dict):
            return None
        plan = raw_args.get("plan")
        if not (
            isinstance(plan, list)
            and plan
            and all(isinstance(s, str) for s in plan)
        ):
            return None
        next_tool = raw_args.get("next_tool")
        if next_tool not in surface_names:
            return None
        task_complete = raw_args.get("task_complete")
        if not isinstance(task_complete, bool):
            return None
        canonical: dict[str, Any] = {
            "plan": list(plan),
            "next_tool": next_tool,
            "task_complete": task_complete,
        }
        summary = raw_args.get("reasoning_summary")
        if include_summary:
            # The summary variant marks ``reasoning_summary`` required.
            if not isinstance(summary, str) or len(summary) > PLAN_SUMMARY_MAX_LENGTH:
                return None
            canonical["reasoning_summary"] = summary
        elif summary is not None:
            # Lean variant: tolerate a bounded summary if the model still sent
            # one, but drop an over-long / wrong-typed value rather than reject.
            if isinstance(summary, str) and len(summary) <= PLAN_SUMMARY_MAX_LENGTH:
                canonical["reasoning_summary"] = summary
        return canonical

    def _reasoning_step_payload(
        self, engine: QueryEngine, plan_args: dict[str, Any]
    ) -> dict[str, Any]:
        plan = plan_args.get("plan")
        payload: dict[str, Any] = {
            "turn_id": engine.turn_id(),
            "plan": list(plan) if isinstance(plan, list) else [],
            "next_tool": plan_args.get("next_tool"),
            "task_complete": bool(plan_args.get("task_complete", False)),
        }
        summary = plan_args.get("reasoning_summary")
        if isinstance(summary, str) and summary:
            payload["reasoning_summary"] = summary
        return payload

    def _append_plan_turn(self, engine: QueryEngine, plan_args: dict[str, Any]) -> None:
        """Append the plan as a paired tool_use + tool_result so the recorded
        intent is visible to the action turn AND the wire stays pairing-valid.

        The synthetic tool_result is a non-error acknowledgement; the plan tool
        has NO external side effect (it only records intent), so this never
        touches the workspace / answer plane.

        Fix #4 — the planning-turn assistant Message ALWAYS carries a non-empty
        ``reasoning_content`` (mirroring the main assistant stream, which sets
        ``reasoning_content=reasoning_buffer or None`` in query.py). DeepSeek
        thinking-mode rejects the ACTION turn with HTTP 400
        ("The reasoning_content in the thinking mode must be passed back to the
        API.") when an assistant turn in the echoed context omits it; the prior
        ``reasoning_content=None`` planning turn triggered exactly that. This is
        provider-agnostic — no "deepseek" branch — and the serializer already
        emits ``reasoning_content`` universally, so providers that ignore it are
        unaffected (the Fix #1 regression proved that's safe).
        """
        plan_call_id = engine.new_tool_call_id()
        engine.history.append(
            Message(
                role=MessageRole.assistant,
                content_blocks=[
                    ToolUseBlock(
                        tool_call_id=plan_call_id,
                        name=PLAN_TOOL_NAME,
                        arguments_json=json.dumps(plan_args, ensure_ascii=False),
                    )
                ],
                reasoning_content=self._plan_turn_reasoning(engine, plan_args),
            )
        )
        engine.history.append(
            Message(
                role=MessageRole.tool,
                content_blocks=[
                    ToolResultBlock(
                        tool_call_id=plan_call_id,
                        content=(
                            "Plan recorded. Now perform the planned next_tool. "
                            "If the task needs real sources/citations, delegate "
                            "source-gathering to a configured subagent if one is "
                            "available (see your subagent catalog) and cite ONLY "
                            "tool-returned URLs — never write sources, authors, "
                            "titles, DOIs, or provenance from memory; if you "
                            "cannot reach a requested count from real hits, say so "
                            "and stop. When you report results, reference any "
                            "written/read file by its path plus a short summary; "
                            "do NOT paste a file's full body or a large Read/tool "
                            "output into the final answer (quote at most a few "
                            "lines)."
                        ),
                        is_error=False,
                    )
                ],
            )
        )

    @staticmethod
    def _plan_turn_reasoning(engine: QueryEngine, plan_args: dict[str, Any]) -> str:
        """Resolve the NON-EMPTY ``reasoning_content`` for the planning turn.

        Fallback chain (deepseek's contract needs the field PRESENT + non-empty,
        not verbatim CoT):

        1. the reasoning the plan stream actually produced (native ``thinking``
           deltas captured in ``_fetch_plan``/``_fetch_plan_fallback`` and stashed
           on ``engine`` — consumed and CLEARED here so it never leaks to a later
           turn);
        2. else the plan's ``reasoning_summary`` (present in the summary variant /
           tolerated in the lean variant);
        3. else a minimal bilingual placeholder naming the planned ``next_tool``.

        NEVER returns an empty string — a recorded plan turn must always carry
        ``reasoning_content`` so the deepseek thinking-mode echo contract holds.
        """
        captured = getattr(engine, _DEEP_PLAN_REASONING_ATTR, "")
        # Consume-and-clear the transient so a subsequent turn cannot inherit a
        # stale plan reasoning.
        if hasattr(engine, _DEEP_PLAN_REASONING_ATTR):
            delattr(engine, _DEEP_PLAN_REASONING_ATTR)
        if isinstance(captured, str) and captured.strip():
            return captured
        summary = plan_args.get("reasoning_summary")
        if isinstance(summary, str) and summary.strip():
            return summary
        next_tool = plan_args.get("next_tool")
        next_tool_label = next_tool if isinstance(next_tool, str) and next_tool else "—"
        # Bilingual (EN + RU) per the multilingual-prompt convention.
        return (
            f"Planned next step: call {next_tool_label}. / "
            f"Запланирован следующий шаг: вызвать {next_tool_label}."
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_fallback_worthy(exc: LLMError) -> bool:
    """True iff the adapter classified this error as fallback-worthy.

    An OpenAI-compatible adapter is expected to attach a ``classified``
    verdict object to every raised :class:`~protocore.contracts.llm.LLMError`;
    a DeepSeek-class strict forced-tool / ``json_schema`` HTTP 400 classifies as
    ``format_error`` whose ``should_fallback`` flag is ``True``. Reading
    ``classified`` via ``getattr`` keeps core provider-agnostic and
    import-boundary-clean (the classifier lives in the adapter): we react to
    the verdict, never to a
    provider name, and any adapter that does not attach the attribute (or a
    transient error whose verdict is not fallback-worthy, e.g. a stall/timeout)
    simply does not trigger the prompted-JSON fallback — the deep turn degrades
    to the shared loop exactly as before.
    """
    classified = getattr(exc, "classified", None)
    return bool(getattr(classified, "should_fallback", False))


def _plan_fallback_instruction(
    surface_names: list[str], include_summary: bool
) -> str:
    """Render the prompted-JSON plan instruction for the no-forced-tool fallback.

    Describes the SAME ``{plan, next_tool, task_complete[, reasoning_summary]}``
    shape the forced ``Plan`` tool enforces, plus the allowed ``next_tool`` enum
    (the live surface). Bilingual (EN + RU) per the multilingual-prompt
    convention. The literal word ``json`` is present so DeepSeek's
    ``response_format={"type":"json_object"}`` mode accepts the request (it
    rejects json_object otherwise). The model only RECORDS intent here — the
    shared action loop drives the real tool — so this never asks it to act.
    """
    enum = ", ".join(surface_names)
    summary_field_en = (
        f'  "reasoning_summary": a short string (<={PLAN_SUMMARY_MAX_LENGTH} '
        "chars) summarising your reasoning,\n"
        if include_summary
        else ""
    )
    summary_field_ru = (
        f'  "reasoning_summary": короткая строка (<={PLAN_SUMMARY_MAX_LENGTH} '
        "символов) с кратким обоснованием,\n"
        if include_summary
        else ""
    )
    return (
        "Before acting, record your reasoning as a SINGLE json object and "
        "nothing else (no prose, no markdown fences). The json object must have "
        "exactly these fields:\n"
        '  "plan": a non-empty array of short ordered step strings,\n'
        f"{summary_field_en}"
        f'  "next_tool": the single next tool to call — one of [{enum}],\n'
        '  "task_complete": a boolean (true if no tool call is needed).\n'
        "Do NOT call any tool now; only return the json plan object.\n"
        "\n"
        "Прежде чем действовать, запиши рассуждение как ОДИН json-объект и "
        "ничего больше (без текста и markdown). Поля json-объекта строго:\n"
        '  "plan": непустой массив коротких строк-шагов по порядку,\n'
        f"{summary_field_ru}"
        f'  "next_tool": единственный следующий инструмент — один из [{enum}],\n'
        '  "task_complete": булево (true, если вызов инструмента не нужен).\n'
        "Не вызывай инструменты сейчас; верни только json-объект плана."
    )


def _parse_plan_text(text: str) -> dict[str, Any] | None:
    """Tolerantly parse a prompted-JSON plan object from model text.

    Reuses :mod:`protocore.json_utils` (NO new parser): strip ``<think>`` spans,
    then take the first balanced top-level json object via
    :func:`structured_json_candidates`, falling back to a strict whole-string
    :func:`parse_complete_json`. Returns the raw dict (the canonicalisation +
    schema validation stay in :meth:`DeepStrategy._validated_plan_args`), or
    ``None`` when no json object can be recovered.
    """
    if not text.strip():
        return None
    candidates = structured_json_candidates(text)
    if candidates:
        return candidates[0]
    try:
        return parse_complete_json(strip_thinking(text))
    except OutputParserException:
        return None


def _plan_max_tokens(engine: QueryEngine, context: ContextBundle) -> int:
    """Per-call output cap for the plan step.

    Reuses the loop's existing ``llm_output_max_tokens_ratio`` budget (no new
    magic number) against the context window, with a 1-token floor so the cap
    is always positive.
    """
    rc = engine.config.rc
    window = context.budgets.max_context if context.budgets else rc.model_context_window
    return max(1, int(window * rc.llm_output_max_tokens_ratio))


async def _normalised_deltas(
    upstream: AsyncIterator[ProviderDelta | object] | object,
) -> AsyncIterator[ProviderDelta]:
    """Yield :class:`ProviderDelta` from either a ProviderDelta stream (vLLM
    adapter) or an LLMStreamEvent stream (mock), mirroring
    ``query._as_provider_deltas`` but local to the strategy module to avoid a
    circular import with :mod:`protocore.runtime.query`."""
    import inspect

    from protocore.contracts.llm import LLMStreamEvent, ProviderDelta

    resolved = await upstream if inspect.iscoroutine(upstream) else upstream

    first: Any = None
    async for item in resolved:  # type: ignore[union-attr]
        first = item
        break
    if first is None:
        return

    if isinstance(first, ProviderDelta):
        yield first
        async for item in resolved:  # type: ignore[union-attr]
            if isinstance(item, ProviderDelta):
                yield item
        return

    if isinstance(first, LLMStreamEvent):
        first_evt: LLMStreamEvent = first

        async def _chain() -> AsyncIterator[LLMStreamEvent]:
            yield first_evt
            async for tail in resolved:  # type: ignore[union-attr]
                if isinstance(tail, LLMStreamEvent):
                    yield tail

        async for delta in stream_events_to_provider_deltas(_chain()):
            yield delta
        return

    _logger.warning(
        "deep_strategy: unknown plan-stream item type: %s", type(first).__name__
    )


def select_strategy(run_mode: str) -> AgentLoop:
    """Return the loop strategy for ``run_mode`` — the SINGLE branch point.

    ``run_mode`` is already validated by ``QueryEngineConfig.__post_init__``
    against ``^(direct|deep)$``; ``direct`` is the safe fallback for any
    unexpected value so the loop never wedges on an unknown mode.
    """
    if run_mode == "deep":
        return DeepStrategy()
    return DirectStrategy()


__all__ = [
    "PLAN_SUMMARY_MAX_LENGTH",
    "PLAN_TOOL_NAME",
    "AgentLoop",
    "DeepStrategy",
    "DirectStrategy",
    "plan_tool_schema",
    "select_strategy",
]
