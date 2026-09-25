"""The terminal call a run owes once its answer is delivered.

A run whose contract ends in a terminal tool, and whose model has written its
answer as ordinary prose and stopped, owes exactly one more thing: the call.
Asking for it in words is a request the model may read differently. Measured on
a thinking model, the request was answered with a round of reasoning and no
output; the runtime then asked the model to continue, and the model, reading
"continue" under a transcript whose summary listed earlier questions, started a
second answer to one of them. The reader saw two answers to one question.

So the call is not asked for, it is forced. From the moment the answer is
delivered, every request the run makes is constrained to a tool call and
appends nothing to the transcript, so no request can be read as an invitation
to write. Each forced request has a MODE, chosen before it opens:

* :data:`MODE_TERMINAL` names the terminal tool in the provider's native
  ``tool_choice``. Only the terminal call is admitted from such a turn; any
  other call a provider returns is dropped before it is recorded or run.
* :data:`MODE_ANY_TOOL` requires a tool call and admits any. It follows the
  run's first terminal call the tool refused with
  :data:`~protocore.contracts.types.TERMINAL_REFUSAL_NEEDS_WORK_METADATA_KEY`
  set, so a model told "the declared file is missing" can write it instead of
  being made to repeat the refused call. Every other refusal — an argument or
  validation error — is answered in :data:`MODE_TERMINAL`: the fix is a better
  call, and a free choice of tools after a finished answer is how new research
  (and a second answer) starts.

Requiring *a* tool call is deliberately not how a delivered answer is sealed.
Measured with a thinking-capable model, thinking off, after a finished answer:
a required call was the terminal tool in 15 of 20 samples and new research in
the other 5, and when a file write was admitted as well, the model wrote the
answer into a file it was never asked for in 2 of 10 full runs and then
announced the file in a second message. Prose that only announces work ("now
let me write the report") is therefore sealed like an answer; the terminal
call's declared deliverables are what a host check can refuse, and a refusal
lets the model do the work (see below).

A turn in any mode carries no prose to the reader. A call to anything but the
terminal tool means the model is working again: the forcing is lifted, and the
next answer the model writes arms it again on what is left of the budget. The
same happens when something asks the model a question — a corrective turn from
a gate that refused the call, or a message from the user — because a forced
call cannot answer a question.

The forcing is bounded by ``terminal_tool_forced_max_attempts``: every forced
request spends from it, and so does every lifting for a question. Lifting
because a required call was answered with work does not spend again — the
request that produced the work was already charged. When it is spent the run
completes on the answer it has delivered. That completion is a hard stop: it
does not pass the finish seams a voluntary answer passes.

This module holds only the state — whether the call is being forced, the mode
of the request about to open, and how much of the budget is spent — all on the
engine and all snapshot-persisted. Where the state is read and what the loop
does with it lives in :mod:`protocore.runtime.query` and the terminal-nudge
turn policy.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Final

from protocore.logging_utils import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from protocore.runtime.query_engine import QueryEngine

_logger = get_logger(__name__)

#: The request names the terminal tool; only the terminal call is admitted.
MODE_TERMINAL: Final[str] = "terminal"
#: The request requires a tool call; any tool is admitted.
MODE_ANY_TOOL: Final[str] = "any_tool"

MODES: Final[frozenset[str]] = frozenset(
    {MODE_TERMINAL, MODE_ANY_TOOL}
)

#: ``state_changed`` reason when the forcing starts on a delivered answer.
REASON_FORCED: Final[str] = "terminal_tool_forced"
#: ``state_changed`` reason when a forced request came back without the call.
REASON_RETRY: Final[str] = "terminal_tool_forced_retry"
#: ``state_changed`` reason when the run completes on its delivered answer.
REASON_EXHAUSTED: Final[str] = "terminal_tool_forced_exhausted"
#: ``state_changed`` reason when the run's terminal tool is not a tool it can call.
REASON_UNAVAILABLE: Final[str] = "terminal_tool_unavailable"
#: ``state_changed`` reason when the forcing is lifted so the model can act.
REASON_RELEASED: Final[str] = "terminal_tool_forced_released"


def is_armed(engine: object) -> bool:
    """True while the run's terminal call is being forced.

    Takes any object so the turn policies, which see the run only through its
    narrowed state protocol, can ask without a cast.
    """
    return bool(getattr(engine, "_terminal_call_forced", False))


def request_mode(engine: object) -> str | None:
    """The mode of the forced request about to open, or ``None``.

    ``None`` while the forcing is off, and also while it is on but the next
    request is not forced by it — a run-level precondition owns the slot.
    """
    if not is_armed(engine):
        return None
    mode = getattr(engine, "_terminal_call_forced_mode", None)
    return mode if mode in MODES else None


def set_request_mode(engine: QueryEngine, mode: str | None) -> None:
    if mode is not None and mode not in MODES:  # pragma: no cover - defensive
        raise ValueError(f"unknown forced terminal mode: {mode!r}")
    engine._terminal_call_forced_mode = mode


def arm(engine: QueryEngine) -> bool:
    """Start forcing the terminal call. Returns True if it was not already."""
    if is_armed(engine):
        return False
    engine._terminal_call_forced = True
    engine._terminal_call_forced_mode = None
    return True


def release(engine: QueryEngine, *, reason: str) -> None:
    """Stop forcing. The spent attempts are NOT given back."""
    if not is_armed(engine):
        return
    engine._terminal_call_forced = False
    engine._terminal_call_forced_mode = None
    _logger.warning(
        "DIAG forced_terminal.released run=%s reason=%s attempts=%d",
        engine.config.run_id,
        reason,
        attempts_spent(engine),
    )


def attempts_spent(engine: QueryEngine) -> int:
    return int(getattr(engine, "_terminal_call_forced_attempts", 0))


def exhausted(engine: QueryEngine) -> bool:
    """True once the forced requests the run may spend are all spent."""
    return attempts_spent(engine) >= engine.config.rc.terminal_tool_forced_max_attempts


def charge_attempt(engine: QueryEngine) -> int:
    """Spend one attempt and return which one it was."""
    engine._terminal_call_forced_attempts = attempts_spent(engine) + 1
    return engine._terminal_call_forced_attempts


def spend_all(engine: QueryEngine) -> None:
    """Spend the whole bound at once, for a forcing that cannot be carried out."""
    engine._terminal_call_forced_attempts = (
        engine.config.rc.terminal_tool_forced_max_attempts
    )


__all__ = [
    "MODES",
    "MODE_ANY_TOOL",
    "MODE_TERMINAL",
    "REASON_EXHAUSTED",
    "REASON_FORCED",
    "REASON_RELEASED",
    "REASON_RETRY",
    "REASON_UNAVAILABLE",
    "arm",
    "attempts_spent",
    "charge_attempt",
    "exhausted",
    "is_armed",
    "release",
    "request_mode",
    "set_request_mode",
    "spend_all",
]
