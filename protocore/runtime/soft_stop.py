"""The one wind-down: how a run that must stop early still delivers an answer.

Every bound a run can hit — the tool-call budget, the turn cap, the output-token
budget, a wall-clock deadline, an upstream that stopped answering — used to end
somewhere different, on its own flag, with its own semantics. A tool-call budget
appended a paragraph of English to a tool result and let the agent carry on for
another eighteen calls. A turn cap tried a nudge, then a synthetic answer
forged from the model's last words, then failed. A provider error granted one
best-effort turn, or none. The bounds were reached at five places and answered
five ways, and only one of the five actually stopped anything.

This module is the single answer, and the shape of it is the point:

1. **Say so.** A message goes into the transcript telling the model the budget
   is spent and the run is closing. Localised, because a model told to wrap up
   in a language the conversation is not in tends to switch languages first.
2. **Take the tools away.** Not advise — REMOVE. The surface the model is shown
   on the next turn is the terminal tool and nothing else (plus the artifact
   sealer while an artifact is open). This is why the withdrawal lives in
   :func:`restricted_policy`, applied last in
   :attr:`QueryEngine.effective_tool_policy`: it has to beat the pinned floor,
   the progressive-discovery pins and the retrieval clip, all of which exist
   precisely to keep tools on the surface. An instruction not to use a tool is
   a request; an absent schema is an answer.
3. **Make it write.** The prose gate already refuses a terminal tool call that
   is not preceded by a real answer. Nothing new here — the wind-down just does
   not bypass it.
4. **Let it finish.** The terminal tool runs and the run ends
   ``stop_reason="soft_stop"``, ``completed`` when an answer was produced.

Every step is observable: four ``STATE_CHANGED`` reasons, in order, so "the soft
stop fired" is a thing a log can be asked about rather than inferred from an
agent's behaviour.

The module is pure policy over the engine — it appends to history and sets
latches, and yields the events the loop forwards. Deciding WHEN to enter is the
loop's job (:func:`protocore.runtime.query._soft_stop_cause`); deciding what
entering MEANS is this module's.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Final

from protocore.contracts.tool_registry import ToolVisibilityPolicy
from protocore.contracts.types import (
    SYNTHETIC_RECOVERY_METADATA_KEY,
    Message,
    MessageRole,
    TextBlock,
)
from protocore.logging_utils import get_logger
from protocore.runtime.events import EventType, TurnEvent

if TYPE_CHECKING:  # pragma: no cover - typing only
    from protocore.runtime.query_engine import QueryEngine

_logger = get_logger(__name__)


SYNTHETIC_RECOVERY_SOFT_STOP: Final[str] = "soft_stop"
"""``SYNTHETIC_RECOVERY_METADATA_KEY`` value on the wind-down notification.

Marks the message as the runtime's words rather than the model's, so it cannot
be mistaken for an answer by the predicates that ask whether the run produced
one.
"""


# ---- causes ---------------------------------------------------------------
#
# What ran out. Carried on every event the wind-down emits and stored on the
# engine, so a run that ended in a soft stop says which bound it hit rather than
# only that it hit one.

CAUSE_TOOL_CALL_BUDGET: Final[str] = "tool_call_budget"
CAUSE_MAX_TURNS: Final[str] = "max_turns"
CAUSE_OUTPUT_TOKEN_BUDGET: Final[str] = "output_token_budget"
CAUSE_DEADLINE: Final[str] = "deadline"
CAUSE_PROVIDER_ERROR: Final[str] = "provider_error"

CAUSES: Final[frozenset[str]] = frozenset(
    {
        CAUSE_TOOL_CALL_BUDGET,
        CAUSE_MAX_TURNS,
        CAUSE_OUTPUT_TOKEN_BUDGET,
        CAUSE_DEADLINE,
        CAUSE_PROVIDER_ERROR,
    }
)


# ---- stages ---------------------------------------------------------------

STAGE_NONE: Final[str] = ""
STAGE_NOTIFIED: Final[str] = "notified"
STAGE_WITHDRAWN: Final[str] = "tools_withdrawn"
STAGE_FINALIZED: Final[str] = "finalized"

REASON_NOTIFIED: Final[str] = "soft_stop_notified"
REASON_WITHDRAWN: Final[str] = "soft_stop_tools_withdrawn"
REASON_FINALIZED: Final[str] = "soft_stop_finalized"

STOP_REASON: Final[str] = "soft_stop"
"""``message_stop.stop_reason`` for a run the wind-down closed.

Distinct from ``max_turns`` and from ``error`` because it describes something
neither does: the run hit a bound, was told to stop, and stopped on purpose —
with an answer if the model wrote one. Reporting that as ``max_turns`` scores a
deliberate finish as a resource overrun; reporting it as ``end_turn`` hides that
the run was cut short.
"""


# ---- state ----------------------------------------------------------------


def is_enabled(engine: QueryEngine) -> bool:
    return bool(engine.config.rc.soft_stop_enabled)


def is_armed(engine: QueryEngine) -> bool:
    """True once the wind-down has started, whatever stage it has reached."""
    return bool(getattr(engine, "_soft_stop_cause", None))


def cause(engine: QueryEngine) -> str | None:
    return getattr(engine, "_soft_stop_cause", None) or None


def stage(engine: QueryEngine) -> str:
    return str(getattr(engine, "_soft_stop_stage", STAGE_NONE) or STAGE_NONE)


def tools_withdrawn(engine: QueryEngine) -> bool:
    """True once the surface has been narrowed to the terminal tool.

    Read by :attr:`QueryEngine.effective_tool_policy`, which is the single place
    every surface computation and every dispatch permission check passes
    through — so this one flag governs what the model is SHOWN and what it is
    allowed to RUN, and the two cannot drift apart.
    """
    return stage(engine) in (STAGE_WITHDRAWN, STAGE_FINALIZED)


# ---- the terminal surface -------------------------------------------------


NO_TOOL_SENTINEL: Final[str] = "\x00protocore.no-tool"
"""A name no registered tool can have, used to say "no tools" out loud.

:class:`ToolVisibilityPolicy` reads an EMPTY ``visible`` set as "no whitelist,
everything is visible" — the opposite of what a withdrawal means. A run with no
terminal tool configured ends by writing prose and stopping, so narrowing it to
nothing is exactly right and has to be expressible. This sentinel is what makes
the whitelist non-empty while matching nothing: the surface computation returns
zero tools and the permission gate admits none.
"""


def terminal_surface(engine: QueryEngine) -> frozenset[str]:
    """The tool names that survive the withdrawal.

    The configured terminal tool, plus the artifact sealer while an artifact is
    open. The sealer is in the set for a concrete reason: a run cut short in the
    middle of writing a long file has that file on disk, complete enough to be
    worth keeping and unsealed, and a wind-down that removed the only tool able
    to close it would throw the work away in the name of stopping cleanly.

    Empty when no terminal tool is configured. That is not a degenerate case:
    such a run ends by writing its answer and stopping, so a surface with
    nothing on it is the correct one — the model has no way left to do work and
    the only thing left to do is answer.
    """
    terminal_tool = engine.config.expected_terminal_tool
    if not terminal_tool:
        return frozenset()
    names = {terminal_tool}
    from protocore.runtime import longfile_convergence as _longfile

    # ``terminal_seal_required`` is exactly the "there is an artifact worth
    # sealing" question: truncation-gated, not yet finalized, past the
    # empty-file floor, and inside its forced-finalize budget. Asking it here
    # rather than restating the conditions keeps one definition of when an
    # artifact is open.
    if _longfile.terminal_seal_required(engine):
        sealing_tool = _longfile.sealing_tool_name(engine)
        if sealing_tool is not None:
            names.add(sealing_tool)
    return frozenset(names)


def restricted_policy(
    policy: ToolVisibilityPolicy,
    allowed: frozenset[str],
) -> ToolVisibilityPolicy:
    """Narrow ``policy`` to ``allowed`` — the last word on the tool surface.

    Applied AFTER the pinned floor, the discovery pins and the execution
    profile, because each of those is a mechanism for keeping a tool visible and
    the withdrawal has to outrank all of them. ``visible`` becomes exactly the
    allowed set, ``pinned`` / ``forced_pinned`` are reduced to it (both bypass
    the retrieval clip, so leaving a name in either would put a withdrawn tool
    back on the surface), and every other name is excluded by the whitelist.

    An empty ``allowed`` narrows to NOTHING rather than to everything — see
    :data:`NO_TOOL_SENTINEL` for why that needs saying explicitly.
    """
    whitelist = set(allowed) or {NO_TOOL_SENTINEL}
    return policy.model_copy(
        update={
            "visible": whitelist,
            "pinned": set(policy.pinned) & set(allowed),
            "forced_pinned": policy.forced_pinned & allowed,
        }
    )


# ---- entering -------------------------------------------------------------


def notification_text(engine: QueryEngine, *, cause_name: str) -> str:
    """The notice this cause puts in front of the model.

    The wind-down is one path for several bounds, and the notice used to be one
    text for all of them — which made it say "the run has reached its budget"
    to a run whose budget was untouched: the upstream had refused its requests
    and the retries were spent. A model reads that literally. It infers that it
    was given turns and spent them, and it writes a closing summary of work it
    never did; the operator gets a polite report of nothing and no sign that
    the provider failed. So the cause that is not a budget names itself, and
    the ones that really are budgets keep the general text.

    A cause whose own text is blank falls back to the general one. An empty
    general notice suppresses the message entirely, which is what its constant
    documents.
    """
    rc = engine.config.rc
    if cause_name == CAUSE_PROVIDER_ERROR:
        template = rc.soft_stop_notice_text_provider_error or rc.soft_stop_notice_text
    else:
        template = rc.soft_stop_notice_text
    if not template:
        return ""
    return template.replace("{cause}", cause_name)


def enter(engine: QueryEngine, *, cause_name: str, detail: str = "") -> list[TurnEvent]:
    """Start the wind-down. Idempotent — a second call returns no events.

    Appends the notification to history and narrows the surface, emitting the
    two state changes that make both observable. The caller owns the rest of the
    turn mechanics: granting the wind-down its turns, persisting the snapshot,
    and rebuilding the context so the next stream sees the narrowed surface.

    ``detail`` is what the cause itself said, in the words of whatever produced
    it — the upstream's own error text, typically. It rides on the state events
    and nowhere else: the model is told the shape of the stop by the notice,
    while a host that has to SHOW the operator why a run ended needs the
    original message, and reconstructing it from a log is not showing it.
    """
    if is_armed(engine):
        return []
    if cause_name not in CAUSES:  # pragma: no cover - defensive
        raise ValueError(f"unknown soft-stop cause: {cause_name!r}")

    engine._soft_stop_cause = cause_name
    engine._soft_stop_stage = STAGE_NOTIFIED
    _append_notice(engine, cause_name=cause_name)

    allowed = terminal_surface(engine)
    extra: dict[str, object] = {"soft_stop_detail": detail} if detail else {}
    events = [
        _state_event(engine, REASON_NOTIFIED, cause_name=cause_name, **extra),
    ]
    # The surface is narrowed in the same entry, not a turn later: the model's
    # very next request is the one that must not carry a working tool.
    engine._soft_stop_stage = STAGE_WITHDRAWN
    events.append(
        _state_event(
            engine,
            REASON_WITHDRAWN,
            cause_name=cause_name,
            allowed_tools=sorted(allowed),
            **extra,
        )
    )
    _logger.warning(
        "DIAG query.soft_stop.entered run=%s tenant=%s cause=%s allowed_tools=%s",
        engine.config.run_id,
        engine.config.tenant_id,
        cause_name,
        ",".join(sorted(allowed)) or "-",
    )
    return events


def _has_notice(engine: QueryEngine) -> bool:
    from protocore.runtime.query import _this_run_messages

    key = SYNTHETIC_RECOVERY_METADATA_KEY
    return any(m.metadata.get(key) == SYNTHETIC_RECOVERY_SOFT_STOP for m in _this_run_messages(engine))


def _append_notice(engine: QueryEngine, *, cause_name: str) -> None:
    text = notification_text(engine, cause_name=cause_name)
    if text:
        engine.history.append(
            Message(
                role=MessageRole.user,
                content_blocks=[TextBlock(text=text)],
                metadata={SYNTHETIC_RECOVERY_METADATA_KEY: SYNTHETIC_RECOVERY_SOFT_STOP},
            )
        )


def restore(engine: QueryEngine) -> None:
    """Put the notification back in front of a resumed run that is still wound down.

    The wind-down state travels in the snapshot; the notice travels in history
    only while the run drives (see :func:`leave`). A run resumed from a snapshot
    taken after its end is still bound — its surface is empty — and the model
    must read why, or it meets an empty surface with no explanation.
    """
    if is_armed(engine) and not _has_notice(engine):
        _append_notice(engine, cause_name=cause(engine) or "")


def leave(engine: QueryEngine) -> int:
    """Take the notification out of history once the run is over. Returns how many were removed.

    The notice is an instruction for the run it was written into: "the tools are
    gone, write your answer now". It is true only while that run drives. Left in
    a session's history it outlives the run and becomes the last instruction the
    model sees when the operator writes again — and the model obeys it: it
    reports what it "did not manage to finish" and calls nothing, turn after
    turn, although every tool is back on the surface. That is what happened when
    a wind-down's own final turn died on the provider that caused it: the run
    failed with the notice as its last word, and the operator's "continue" got
    a farewell.

    Called at the end of every drive that reaches a terminal state, success or
    failure alike. A drive that pauses (awaiting an answer) or is interrupted
    keeps it: the run is not over, and a resumed run stays wound down.
    """
    key = SYNTHETIC_RECOVERY_METADATA_KEY
    before = len(engine.history)
    engine.history[:] = [
        m for m in engine.history if m.metadata.get(key) != SYNTHETIC_RECOVERY_SOFT_STOP
    ]
    return before - len(engine.history)


def finalize(engine: QueryEngine) -> TurnEvent | None:
    """Mark the wind-down complete. Returns the event, or ``None`` if already done.

    Emitted when the run reaches its end under a soft stop — either through the
    terminal tool or through the model simply answering and stopping.
    """
    if not is_armed(engine) or stage(engine) == STAGE_FINALIZED:
        return None
    engine._soft_stop_stage = STAGE_FINALIZED
    return _state_event(
        engine,
        REASON_FINALIZED,
        cause_name=cause(engine) or "",
        has_final_answer=engine.has_final_answer,
    )


def _state_event(
    engine: QueryEngine,
    reason: str,
    *,
    cause_name: str,
    **extra: object,
) -> TurnEvent:
    return TurnEvent(
        type=EventType.STATE_CHANGED,
        run_id=engine.config.run_id,
        payload={
            "from": engine.state.value,
            "to": engine.state.value,
            "reason": reason,
            "soft_stop_cause": cause_name,
            **extra,
        },
    )


__all__ = [
    "CAUSES",
    "CAUSE_DEADLINE",
    "CAUSE_MAX_TURNS",
    "CAUSE_OUTPUT_TOKEN_BUDGET",
    "CAUSE_PROVIDER_ERROR",
    "CAUSE_TOOL_CALL_BUDGET",
    "NO_TOOL_SENTINEL",
    "REASON_FINALIZED",
    "REASON_NOTIFIED",
    "REASON_WITHDRAWN",
    "STAGE_FINALIZED",
    "STAGE_NONE",
    "STAGE_NOTIFIED",
    "STAGE_WITHDRAWN",
    "STOP_REASON",
    "SYNTHETIC_RECOVERY_SOFT_STOP",
    "cause",
    "enter",
    "finalize",
    "is_armed",
    "is_enabled",
    "leave",
    "restore",
    "restricted_policy",
    "stage",
    "terminal_surface",
    "tools_withdrawn",
]
