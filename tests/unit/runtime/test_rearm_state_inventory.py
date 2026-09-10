"""Every attribute a ``QueryEngine`` constructs is classified, and stays that way.

A re-arm that resets a hand-written list of fields is wrong the moment someone
adds a field and does not think about the list. The failure is silent and it is
slow: the new field accumulates across turns, and the agent it eventually
strands looks like a model problem hundreds of turns later.

So the classification is enforced from both ends. The engine names what
SURVIVES a re-arm and rebuilds everything else, and the inventory test below
walks ``__init__`` and fails when an attribute appears in neither the engine's
preserved set nor the reset list this module keeps. Adding a field to the
constructor is then a decision about its lifetime, made once, at the time the
field is added.
"""
from __future__ import annotations

import ast
import inspect
import textwrap
from pathlib import Path

import pytest

from protocore.contracts.evidence import VerificationState
from protocore.contracts.run_state import (
    ConsecutiveErrorStreak,
    SignatureStreak,
    ToolStreak,
)
from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.tools import ToolInvocationError
from protocore.contracts.types import (
    Message,
    MessageRole,
    TextBlock,
    ToolCall,
)
from protocore.runtime.loop_state import LoopState
from protocore.runtime.query import _block_identical_tools, _dispatch_tool
from protocore.runtime.query_engine import QueryEngine

from ._tool_fixtures import MockTool

# Every attribute the constructor sets that a re-arm must rebuild. Kept in the
# test rather than in the engine on purpose: the engine states only what
# survives, so a new field defaults to being reset. This list exists to make
# "I thought about it" the thing that fails, not the thing that is assumed.
_RESET_ON_REARM: frozenset[str] = frozenset(
    {
        # The state machine and the run's own totals.
        "state",
        "turn_count",
        "total_usage",
        "_run_started_monotonic",
        "_run_started_epoch",
        "_run_settled_emitted",
        # The knobs a cut retry turned down are put back by the round that
        # follows; a turn opened elsewhere starts from the operator's values.
        "_reasoning_cut_saved",
        "_tool_call_ledger",
        "_tool_call_ledger_seq",
        "_tool_call_ledger_truncated",
        # The wind-down and the finalisation latch it arms.
        "_soft_stop_cause",
        "_soft_stop_stage",
        "_terminal_only_active",
        # Per-turn wire and streaming bookkeeping. ``turn_id()`` is built from
        # the round counter, so a turn opened on a stale one mints ids that
        # collide with the previous turn's.
        "_block_idx",
        "_wire_round_seq",
        "_pending_tool_call_names",
        "_pending_interrupts",
        "_pending_public_message_start",
        "_pending_public_reader_turn_events",
        "_holding_reader_message",
        "_pending_rules_activated",
        "_skill_loaded_bundles",
        # Recovery budgets. Each bounds one failure mode within one question;
        # an exhausted budget carried forward is a turn that gets no attempt.
        "_compaction_attempted_for_current_turn",
        "compaction_backoff_left",
        "_max_output_recovery_count",
        "_terminal_backstop_turn_active",
        "_tool_call_truncated_recovery_count",
        "_mid_chunked_write_paths",
        "_truncation_recovery_prompt_counts",
        "_provider_chain_advances",
        "_consecutive_empty_responses",
        "_post_tool_empty_nudge_count",
        "_transient_stream_retry_count",
        "_empty_completion_redrive_count",
        # Fire-once report that the session's background pool cannot speak for
        # the session. A run re-armed onto a pool that is still detached is
        # entitled to be told again.
        "_background_detach_reported",
        # The death-spiral flag. Left raised, every later turn skips the
        # terminal hooks its caller is waiting on.
        "skip_terminal_hooks",
        # Answer-quality latches, all fire-at-most-once PER QUESTION.
        "_pre_terminal_self_verify_used",
        "_self_verify_extra_turns_used",
        "_pre_dispatch_terminal_verify_used",
        "_terminal_candidate",
        "_terminal_candidate_reveto_used",
        "_finalize_prose_gate_used",
        "_pointer_answer_repair_attempts",
        "_pointer_answer_repair_released",
        "_run_delegated",
        # Verification. A lifecycle only opens for a new execution attempt, so
        # a sealed one carried forward refuses to verify anything again.
        "_verification_lifecycle",
        "_candidate_delivery_gate",
        # Loop guard. The identical-tool counter is a fingerprint census for
        # ONE turn — that is what its limit is documented to bound.
        "_identical_tool_counts",
        "_loop_guard_nudge_count",
        # Repeated-error circuit breaker. Its block list is unioned into the
        # visible surface, so carrying it forward withdraws a tool for good.
        "_circuit_broken_tools",
        "_circuit_breaker_notified_tools",
        "_circuit_breaker_streak",
        # Run-level tool obligations: the preconditions are what the agent owes
        # before it is free to ANSWER, and every turn produces an answer.
        "_tool_precondition_index",
        "_tool_precondition_calls",
        "_tool_precondition_attempts",
        "_tool_precondition_last_error",
        # Declared-file read-back. The satisfied set resets with the rest: a
        # file read many turns ago may well have changed, and a set that only
        # grows is a leak in an agent with no end.
        "_pending_read_paths",
        "_pending_reads_satisfied",
        "_pending_reads_abandoned",
        "_pending_reads_forced_attempts",
        # Large-file convergence — a stall clock and forced-round budgets, all
        # about the artifact the CURRENT turn is writing.
        "_turns_since_last_byte_adding_mutation",
        "_longfile_forced_appends",
        "_longfile_forced_finalizes",
        "_longfile_active_path",
        "_longfile_active_file_bytes",
        "_longfile_active_file_lines",
        "_longfile_mutation_deltas",
        "_longfile_last_mutation_truncated",
        "_longfile_finalized",
        "_longfile_truncated_paths",
        "_longfile_appends_per_path",
        "_longfile_salvage_seq",
        "_longfile_voluntary_seal_used",
        # The cooperative stop. It cancels the turn it was aimed at; a re-arm
        # is the caller asking for a different turn, and a driver that wants
        # the agent to stay stopped stops re-arming it.
        "_stop_requested",
        "_current_turn_task",
    }
)


def _constructor_attributes() -> list[str]:
    """Every ``self.X`` the engine's constructor assigns, in source order."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(QueryEngine.__init__)))
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign | ast.AugAssign):
            targets = [node.target]
        else:
            continue
        for target in targets:
            for sub in ast.walk(target):
                if (
                    isinstance(sub, ast.Attribute)
                    and isinstance(sub.value, ast.Name)
                    and sub.value.id == "self"
                    and sub.attr not in found
                ):
                    found.append(sub.attr)
    return found


class _Poison:
    """A value no engine attribute can legitimately still hold after a re-arm."""


def _attached_attributes() -> set[str]:
    """Attribute names the package puts on an engine outside ``__init__``.

    The constructor is not the only writer. The host attaches its run state and
    the run loop caches things on first use, and none of that is visible to
    ``vars()`` of a freshly built engine — which is exactly why a re-arm that
    copies from a fresh engine cannot reach them. Finding them means reading the
    package, so that is what this does: every ``engine.x = ...`` and
    ``setattr(engine, "x", ...)`` in the source.
    """
    package = Path(inspect.getfile(QueryEngine)).parent.parent
    found: set[str] = set()
    for path in sorted(package.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):  # pragma: no cover - defensive
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AnnAssign | ast.AugAssign):
                targets = [node.target]
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "setattr"
                and len(node.args) >= 2
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id == "engine"
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)
            ):
                found.add(node.args[1].value)
                continue
            else:
                continue
            for target in targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "engine"
                ):
                    found.add(target.attr)
    return found - set(_constructor_attributes())


def _spend_a_turn(engine: QueryEngine) -> None:
    """Put the engine through a turn and leave it terminal."""
    engine.transition_to(LoopState.RUNNING)
    engine.record_tool_call("gather", ok=True)
    engine.transition_to(LoopState.COMPLETED)


# ── the inventory gate ──────────────────────────────────────────────────────


def test_every_constructor_attribute_is_classified() -> None:
    """A new engine attribute belongs to exactly one lifetime, deliberately."""
    constructed = set(_constructor_attributes())
    classified = QueryEngine._REARM_PRESERVED_ATTRS | _RESET_ON_REARM

    unclassified = constructed - classified
    assert not unclassified, (
        "these QueryEngine attributes are in neither list: "
        f"{sorted(unclassified)}. Decide whether each one is continuity the "
        "next turn needs (add it to QueryEngine._REARM_PRESERVED_ATTRS) "
        "or a per-run allowance that must start over (add it to "
        "_RESET_ON_REARM here). Leaving it out is how a living agent quietly "
        "stops working."
    )
    stale = classified - constructed
    assert not stale, f"these names are classified but no longer constructed: {sorted(stale)}"
    overlap = QueryEngine._REARM_PRESERVED_ATTRS & _RESET_ON_REARM
    assert not overlap, f"an attribute cannot both survive and reset: {sorted(overlap)}"


def test_rearm_rebuilds_every_attribute_it_does_not_preserve(engine_factory) -> None:
    """The reset is total, not a list someone maintains by hand."""
    engine = engine_factory()
    _spend_a_turn(engine)
    poisoned = _RESET_ON_REARM - {"state"}  # rearm reads state to refuse a live turn
    for name in poisoned:
        setattr(engine, name, _Poison())

    engine.rearm()

    survivors = sorted(
        name for name in poisoned if isinstance(getattr(engine, name), _Poison)
    )
    assert not survivors, f"rearm left last turn's values behind on: {survivors}"
    assert engine.state is LoopState.PENDING


def test_rearm_keeps_every_attribute_it_preserves(engine_factory) -> None:
    """The continuity is preserved by identity — nothing is rebuilt behind it."""
    engine = engine_factory()
    engine.history.append(
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="кто я?")])
    )
    engine.lanes.append("main")
    engine._steer_queue.append({"text": "look left"})
    engine._live_model_name = "some-other-model"
    engine.last_observed_prompt_tokens = 4_321
    before = {name: getattr(engine, name) for name in QueryEngine._REARM_PRESERVED_ATTRS}
    _spend_a_turn(engine)

    engine.rearm()

    replaced = sorted(
        name for name in QueryEngine._REARM_PRESERVED_ATTRS if getattr(engine, name) is not before[name]
    )
    assert not replaced, f"rearm discarded continuity it must carry forward: {replaced}"


# ── what the gaps did to a living agent ─────────────────────────────────────


def test_the_same_opening_tool_call_is_never_a_loop(engine_factory) -> None:
    """An agent that starts every turn by looking around still may, at turn 400.

    The identical-tool limit counts calls within ONE turn. Kept across turns it
    reclassified a habit as a loop: the tool stopped executing on the turn after
    the limit and every turn after that.
    """
    rc = LoopConstants(model_context_window=4_096, loop_guard_enabled=True)
    engine = engine_factory(rc=rc)
    call = ToolCall(id="c1", name="Observe", arguments={})

    for turn in range(1, rc.loop_guard_identical_tool_limit + 6):
        executable, events = _block_identical_tools(engine, [call])
        assert executable == [call], f"the observing call was refused on turn {turn}"
        assert not events
        _spend_a_turn(engine)
        engine.rearm()


@pytest.mark.asyncio
async def test_a_tool_broken_by_a_passing_cause_comes_back(engine_factory) -> None:
    """The breaker withdraws a tool for the rest of the RUN, not of the agent.

    Unioned into ``effective_tool_policy.blocked`` and never cleared, one bad
    stretch removed the tool from the advertised surface and denied it at
    dispatch for every remaining turn the agent would ever take.
    """
    engine = engine_factory()
    engine._helpers = {}
    tool = MockTool(
        tool_name="Sense", raise_exception=ToolInvocationError("sensor busy")
    )
    engine.tools.register(tool)

    for _ in range(engine.config.rc.max_consecutive_tool_errors):
        async for _evt in _dispatch_tool(engine, ToolCall(id="c", name="Sense", arguments={"v": "x"})):
            pass
    assert "Sense" in engine._circuit_broken_tools
    assert "Sense" in engine.effective_tool_policy.blocked

    _spend_a_turn(engine)
    engine.rearm()

    assert engine._circuit_broken_tools == set()
    assert "Sense" not in engine.effective_tool_policy.blocked


def test_an_interrupted_agent_takes_another_turn(engine_factory) -> None:
    """``stop()`` had no lowering seam: one interruption and the agent was mute.

    ``_query_raw`` returns at its first checkpoint while the flag is raised, so
    a cancelled turn that left it raised meant every later turn ended before it
    called the model.
    """
    engine = engine_factory()
    engine.transition_to(LoopState.RUNNING)
    engine.stop()
    engine.transition_to(LoopState.CANCELLED)
    assert engine.stop_requested

    engine.rearm()

    assert not engine.stop_requested


def test_verification_can_open_again_on_the_next_turn(engine_factory) -> None:
    """Evidence collection opens only for a NEW attempt; a turn is a new one."""
    engine = engine_factory()
    engine.begin_evidence_collection(ledger_id="ledger-1")
    assert engine.verification_lifecycle.state is VerificationState.executing
    _spend_a_turn(engine)

    engine.rearm()

    assert engine.verification_lifecycle.state is VerificationState.not_requested
    engine.begin_evidence_collection(ledger_id="ledger-2")


def test_terminal_hooks_are_not_skipped_forever(engine_factory) -> None:
    """Set once by an LLM-class terminal, the flag silenced every later turn's
    Stop / SessionEnd hooks."""
    engine = engine_factory()
    engine.skip_terminal_hooks = True
    _spend_a_turn(engine)

    engine.rearm()

    assert engine.skip_terminal_hooks is False


@pytest.mark.asyncio
async def test_a_long_life_of_turns_stays_the_agent_it_was(engine_factory) -> None:
    """Continuity and allowances pull in opposite directions; both must hold."""
    rc = LoopConstants(model_context_window=64_000, loop_guard_enabled=True)
    engine = engine_factory(rc=rc)
    engine._helpers = {}
    engine.tools.register(MockTool(tool_name="Observe", response_content="a quiet room"))
    call = ToolCall(id="c", name="Observe", arguments={})

    for turn in range(1, 101):
        engine.history.append(
            Message(role=MessageRole.user, content_blocks=[TextBlock(text=f"turn {turn}")])
        )
        executable, _events = _block_identical_tools(engine, [call])
        assert executable, f"the agent lost its tool on turn {turn}"
        engine.transition_to(LoopState.RUNNING)
        async for _evt in _dispatch_tool(engine, call):
            pass
        engine.transition_to(LoopState.COMPLETED)
        engine.rearm()

        assert engine._soft_stop_cause is None
        assert engine.turn_count == 0
        assert engine.tool_call_ledger == []

    assert len(engine.history) >= 100, "the agent's life is one history"


def test_every_attached_attribute_is_classified() -> None:
    """What the host and the loop attach later is classified too, or it escapes.

    An attribute the constructor never sets is invisible to the re-arm's copy
    from a fresh engine, and invisible to the inventory above. Both blind spots
    are the same one, and this is the test that closes it.
    """
    attached = _attached_attributes()
    classified = QueryEngine._REARM_ATTACHED_DROPPED | QueryEngine._REARM_ATTACHED_KEPT

    unclassified = attached - classified
    assert not unclassified, (
        "these names are attached to an engine outside __init__ and classified "
        f"nowhere: {sorted(unclassified)}. Decide whether a re-arm drops each "
        "one (QueryEngine._REARM_ATTACHED_DROPPED) or the host owns it "
        "(QueryEngine._REARM_ATTACHED_KEPT). A field left out here survives "
        "every re-arm in silence."
    )
    overlap = QueryEngine._REARM_ATTACHED_DROPPED & QueryEngine._REARM_ATTACHED_KEPT
    assert not overlap, f"an attribute cannot be both dropped and kept: {sorted(overlap)}"


def test_rearm_drops_what_the_loop_cached_on_the_engine(engine_factory) -> None:
    """A cache and a fire-once latch do not survive into the next turn."""
    engine = engine_factory()
    _spend_a_turn(engine)
    engine._tool_dispatcher = _Poison()
    engine._outbound_system_normalized_warned = True

    engine.rearm()

    assert not hasattr(engine, "_tool_dispatcher"), (
        "the cached dispatcher survived; it holds a tool-error counter read out "
        "of a run state the host may since have replaced"
    )
    assert not getattr(engine, "_outbound_system_normalized_warned", False), (
        "the fire-once warning latch stayed raised, so the warning it guards "
        "fires once per engine instead of once per run"
    )


def test_rearm_clears_the_per_run_cells_of_the_state_only(engine_factory) -> None:
    """The state object is the run's; the per-turn cells inside it are not."""
    engine = engine_factory()
    _spend_a_turn(engine)
    state = engine.run_state
    state.rc = object()
    state.root_run_id = "r-1"
    state.host["wired"] = object()
    ledger = state.ensure_run_work_ledger(engine.config.rc)
    state.consecutive_error = ConsecutiveErrorStreak(tool_name="Write", count=99)
    state.transport_down = SignatureStreak(signature="Bash:TRANSPORT_DOWN", count=99)
    state.transport_down_injection_pending = True
    state.string_type = ToolStreak(tool_name="Write", count=99)
    state.tool_call_soft_cap_state.counts["Write"] = 99
    host_wiring = (state.rc, state.root_run_id, state.host["wired"])

    engine.rearm()

    assert engine.run_state is state, (
        "the state object is shared with subagents still drawing on its ledger "
        "and must survive a turn boundary"
    )
    left = [
        name
        for name, value in (
            ("consecutive_error", state.consecutive_error),
            ("transport_down", state.transport_down),
            ("transport_down_injection_pending", state.transport_down_injection_pending),
            ("string_type", state.string_type),
            ("soft_cap_counts", state.tool_call_soft_cap_state.counts or None),
        )
        if value
    ]
    assert not left, (
        f"last turn's per-run cells are still on the run state: {left}. An agent "
        "that repeats one failing call each turn crosses a per-run cap no single "
        "turn ever reached."
    )
    assert state.run_work_ledger is ledger, "a re-arm refilled the tree's budget"
    assert (state.rc, state.root_run_id, state.host["wired"]) == host_wiring
