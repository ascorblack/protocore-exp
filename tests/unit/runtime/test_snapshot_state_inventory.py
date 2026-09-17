"""The second axis: what a re-arm keeps, a cold resume must keep as well.

``test_rearm_state_inventory`` settles one question about every engine
attribute — does it survive the next turn, or is it rebuilt? This module asks
the second one, and it only applies to the attributes that answered "survives":
does it also survive the run moving to another process?

The two axes are not the same question, and the gap between them is a whole
class of bug that looks like nothing. An attribute that a re-arm preserves is,
by that decision, continuity the run owns rather than an allowance the turn
spends. If the snapshot then drops it, a run picked up elsewhere comes back as a
subtly different agent — one that has forgotten what it compacted away, which
rule files it had activated, which tools it had pinned into its prompt prefix,
or what it recorded about its own work. Nothing fails. The run simply behaves
as though those things never happened, and the next process to touch it cannot
tell.

So every preserved attribute is classified here too: it is either represented in
``snapshot()`` or it is named below as process-local, with the reason it cannot
travel. Adding a field to ``_REARM_PRESERVED_ATTRS`` makes that a decision taken
once, at the time it is added, instead of an omission discovered later.
"""
from __future__ import annotations

import ast
import inspect
import textwrap

import pytest

from protocore.contracts.snapshot import (
    RUN_SCOPED_STATE_SNAPSHOT_KEY,
    SNAPSHOT_SCHEMA_VERSION,
)
from protocore.contracts.types import Message, MessageRole, TextBlock
from protocore.runtime.compact_checkpoint import CompactCheckpoint
from protocore.runtime.permission_widen import CommandGrant
from protocore.runtime.query_engine import QueryEngine
from protocore.runtime.rules_activation import RuleFile
from protocore.runtime.telemetry import Span

# Attributes a re-arm preserves that a snapshot deliberately does NOT carry, and
# why. Every entry here is a thing the receiving process either already has or
# cannot be given: an object the host injects when it builds the engine, a
# primitive bound to one event loop, or a value that is DERIVED from the
# snapshot on the way in rather than written on the way out.
_PROCESS_LOCAL: dict[str, str] = {
    # Injected collaborators. The host constructs these for the engine it is
    # resuming into; a snapshot of them would be a snapshot of the host.
    "llm": (
        "the provider object belongs to the host. Which rung of the chain this "
        "run sits on is durable and IS carried, as provider_chain_advances plus "
        "provider_chain_model_name, and the engine is re-seated onto it."
    ),
    "compaction_llm": "the summarising provider is injected by the host.",
    "tools": (
        "the registry is built by the host from its own catalog. What the run "
        "did to that surface is carried separately — the broken tools, the "
        "pinned result ids and the pin LRU all travel."
    ),
    "events": "the event stream is the host's transport, not run state.",
    "hooks": "the hook manager is injected by the host.",
    "skills": "the skill store is injected by the host.",
    "blobs": "the blob store is injected by the host.",
    "lifecycle_hooks": "the lifecycle registry is injected by the host.",
    "turn_policies": (
        "a policy set is objects, and objects are constructed, not restored. "
        "What the policies DID to the run — the counters they charged, the "
        "latches they spent — is run state and travels with the run; which "
        "policy objects were installed is a decision the host makes again for "
        "the engine it resumes into."
    ),
    "_tool_shared_state_lock": (
        "an asyncio primitive belongs to one event loop and means nothing in "
        "another process; it guards appends within a single turn."
    ),
    "background_pool": (
        "the pool is the host's and holds nothing durable. What survives is the "
        "ids of the commands that were still running, carried as "
        "background_task_ids, which is the only witness a resumed run has that "
        "a command exists at all."
    ),
    "_persisted_history": (
        "the prefix the session store was handed, held as the message objects "
        "themselves so that identity says whether the sequence grew or was "
        "rewritten. Object identity does not survive a serialisation round "
        "trip, and the process resuming the run has no way to know what the "
        "store took from the process before it — so it starts empty and the "
        "first hand-over after a resume writes the history whole."
    ),
    "_persisted_history_epoch": (
        "the companion of _persisted_history: it counts invalidations of a "
        "marker that itself cannot travel, so carrying it would be carrying "
        "half of a pair."
    ),
    "_resumed_skill_catalog_sha256": (
        "read OUT of the snapshot rather than written into it: it is the digest "
        "the catalog block had in the process that wrote the payload, held only "
        "until this process rebuilds one and can be compared against it."
    ),
    "_resumed_background_task_ids": (
        "read OUT of the snapshot rather than written into it: it is the "
        "background_task_ids of the payload this engine was resumed from, and a "
        "run that was never resumed has none."
    ),
}


def _snapshot_attributes() -> set[str]:
    """Every ``self.X`` that ``snapshot()`` reads, directly or via a derivation."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(QueryEngine.snapshot)))
    return {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    }


# ── the inventory gate ──────────────────────────────────────────────────────


def test_every_preserved_attribute_is_classified() -> None:
    """A field that survives a re-arm either travels or says why it cannot."""
    preserved = set(QueryEngine._REARM_PRESERVED_ATTRS)
    in_snapshot = _snapshot_attributes()

    unclassified = preserved - in_snapshot - set(_PROCESS_LOCAL)
    assert not unclassified, (
        "these attributes survive a re-arm but are in neither the snapshot nor "
        f"the process-local list: {sorted(unclassified)}. A re-arm keeping a "
        "field is the decision that it belongs to the RUN, so a resume onto "
        "another process has to keep it too — read it in snapshot(), or name it "
        "in _PROCESS_LOCAL here with the reason it cannot travel."
    )


def test_no_attribute_is_both_carried_and_declared_process_local() -> None:
    """A reason not to carry a value, next to the code that carries it, is a
    reason that has stopped being true and will be believed anyway."""
    both = set(_PROCESS_LOCAL) & _snapshot_attributes()
    assert not both, f"declared process-local but read by snapshot(): {sorted(both)}"


def test_no_stale_process_local_entries() -> None:
    stale = set(_PROCESS_LOCAL) - set(QueryEngine._REARM_PRESERVED_ATTRS)
    assert not stale, f"named process-local but no longer preserved: {sorted(stale)}"


def test_every_process_local_entry_gives_a_reason() -> None:
    """A one-word entry is a list someone filled in to make a test pass."""
    thin = sorted(name for name, reason in _PROCESS_LOCAL.items() if len(reason.split()) < 6)
    assert not thin, f"these entries do not say why the value cannot travel: {thin}"


# ── what the missing continuity did to a resumed run ────────────────────────


async def test_a_resumed_run_still_knows_what_it_compacted_away(engine_factory) -> None:
    """The checkpoint is the only place the folded-away turns survive at all.

    Dropped on resume, the run does not merely forget them — nothing else holds
    them, so the file operations it performed in those turns stop existing.
    """
    source = engine_factory(run_id="run-1", tenant_id="scope-1", session_id="session-1")
    source.compact_checkpoint = CompactCheckpoint(
        entry_id="ckpt_40_12",
        summary="compacted 12 messages",
        retained_from_index=12,
        file_op_facts=["Write:notes.md", "Edit:notes.md"],
        instructions="keep the outline",
        reason="auto",
    )

    resumed = engine_factory(run_id="run-1", tenant_id="scope-1", session_id="session-1")
    await resumed.resume_from_snapshot(source.snapshot())

    assert resumed.compact_checkpoint == source.compact_checkpoint


async def test_a_resumed_run_still_has_the_rules_it_activated(engine_factory) -> None:
    """Activation is a record of which files THIS run touched. Re-deriving it is
    not possible: the touches are in the past."""
    source = engine_factory(run_id="run-1", tenant_id="scope-1", session_id="session-1")
    source.discovered_rules = [
        RuleFile(path="docs/AGENTS.md", body="be brief", origin="project_mount")
    ]
    source.active_rule_paths = ["docs/AGENTS.md"]

    resumed = engine_factory(run_id="run-1", tenant_id="scope-1", session_id="session-1")
    await resumed.resume_from_snapshot(source.snapshot())

    assert resumed.active_rule_paths == ["docs/AGENTS.md"]
    assert resumed.discovered_rules == source.discovered_rules


async def test_a_resumed_run_keeps_the_tools_it_pinned(engine_factory) -> None:
    """The pin LRU is the agent keeping the tools it went looking for, and its
    ORDER decides which one the next overflow evicts."""
    source = engine_factory(run_id="run-1", tenant_id="scope-1", session_id="session-1")
    for name in ("Grep", "Read", "Write"):
        source.context_manager.pin_tool(name)

    resumed = engine_factory(run_id="run-1", tenant_id="scope-1", session_id="session-1")
    await resumed.resume_from_snapshot(source.snapshot())

    assert resumed.context_manager.pinned_tool_names() == ("Grep", "Read", "Write")


async def test_a_resumed_run_keeps_the_records_it_made(engine_factory) -> None:
    """Grants, the profile audit and the spans are evidence, not allowances.
    Emptying one at a process boundary destroys it rather than freeing a budget."""
    source = engine_factory(run_id="run-1", tenant_id="scope-1", session_id="session-1")
    grant = CommandGrant("program", "git")
    source.session_grants = [grant]
    source.profile_audit = [{"actor": "user", "profile": "careful"}]
    source.spans = [Span(name="tool", attributes={"tool": "Read", "recovery": True})]

    resumed = engine_factory(run_id="run-1", tenant_id="scope-1", session_id="session-1")
    await resumed.resume_from_snapshot(source.snapshot())

    # Read back as the grant it was, not as the row it travelled as: the
    # approval gate asks the grant whether it covers a command.
    assert resumed.session_grants == [grant]
    assert resumed.profile_audit == [{"actor": "user", "profile": "careful"}]
    assert resumed.spans == source.spans


def test_the_skill_catalog_prefix_is_carried_as_a_digest(engine_factory) -> None:
    """The block itself is rebuilt from the store on the new process. Its digest
    travels because those bytes are the head of the cached prompt prefix."""
    source = engine_factory(run_id="run-1", tenant_id="scope-1", session_id="session-1")
    source._skill_catalog_block = "<system-reminder>skills</system-reminder>"

    payload = source.snapshot()

    assert payload["skill_catalog_block_sha256"] is not None
    source._skill_catalog_block = None
    assert source.snapshot()["skill_catalog_block_sha256"] is None


async def test_a_snapshot_missing_the_continuity_block_is_refused(engine_factory) -> None:
    """Not an old payload — the chain has already lifted those. A key still
    absent here is corruption, and inventing a default for it is exactly the
    silent half-restore the whole inventory exists to prevent."""
    source = engine_factory(run_id="run-1", tenant_id="scope-1", session_id="session-1")
    source.history.append(
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="where were we?")])
    )
    payload = source.snapshot()
    del payload["active_rule_paths"]

    resumed = engine_factory(run_id="run-1", tenant_id="scope-1", session_id="session-1")
    with pytest.raises(ValueError, match="active_rule_paths"):
        await resumed.resume_from_snapshot(payload)

    assert resumed.history == []


async def test_a_version_one_payload_is_lifted_and_resumes_empty(engine_factory) -> None:
    """The upcaster earns its place here: a run mid-rollout comes back with no
    continuity recorded, rather than being refused outright."""
    source = engine_factory(run_id="run-1", tenant_id="scope-1", session_id="session-1")
    source.turn_count = 5
    payload = source.snapshot()
    assert payload["schema_version"] == SNAPSHOT_SCHEMA_VERSION == 6
    # A version-1 payload named the run's two allowances at the top level.
    run_state_payload = payload.pop(RUN_SCOPED_STATE_SNAPSHOT_KEY)
    payload["run_work_ledger"] = run_state_payload["run_work_ledger"]
    payload["subagent_tree_budget"] = run_state_payload["subagent_tree_budget"]
    for key in (
        "compact_checkpoint",
        "active_rule_paths",
        "discovered_rules",
        "session_grants",
        "profile_audit",
        "spans",
        "context_manager_pinned_tools",
        "skill_catalog_block_sha256",
    ):
        del payload[key]
    payload["schema_version"] = 1

    resumed = engine_factory(run_id="run-1", tenant_id="scope-1", session_id="session-1")
    await resumed.resume_from_snapshot(payload)

    assert resumed.turn_count == 5
    assert resumed.compact_checkpoint is None
    assert resumed.active_rule_paths == []
    assert resumed.context_manager.pinned_tool_names() == ()


# ── the mirror axis: what travels out has to come back in ───────────────────

# Attributes the snapshot carries whose value is put back by something other
# than a direct assignment inside ``resume_from_snapshot``, and by what. An
# attribute read on the way out and never restored on the way in is
# indistinguishable, to the inventory gate above, from one that round-trips:
# both "appear in snapshot()". This map is where that difference is written
# down, so the next attribute added to the payload and forgotten on the way in
# fails here instead of passing.
_RESTORED_BY_HELPER: dict[str, str] = {
    "provider_chain": (
        "re-seated by _restore_provider_chain_position, which walks the chain "
        "to the rung the payload recorded rather than assigning the chain."
    ),
    "context_manager": (
        "not replaced: the manager the host built is kept, and the run's "
        "effect on it is replayed through it — pin_tool for each carried pin."
    ),
    "run_state": (
        "not replaced: the object is shared by reference with subagents that "
        "may still be drawing on the tree ledger inside it, so the payload is "
        "folded into the live one through apply_snapshot."
    ),
    "_skill_catalog_block": (
        "rebuilt from the host's own skill store on the first turn, then "
        "compared against the digest the payload carried; assigning the "
        "block from the payload would resume onto a catalog that may no "
        "longer exist."
    ),
}


def _restore_targets() -> set[str]:
    """Every ``self.X`` assigned while a snapshot is being taken back in."""
    sources = [QueryEngine.resume_from_snapshot, QueryEngine._restore_provider_chain_position]
    assigned: set[str] = set()
    for function in sources:
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        for node in ast.walk(tree):
            targets: list[ast.expr] = []
            if isinstance(node, ast.Assign):
                targets = list(node.targets)
            elif isinstance(node, ast.AugAssign | ast.AnnAssign):
                targets = [node.target]
            for target in targets:
                for element in ast.walk(target):
                    if (
                        isinstance(element, ast.Attribute)
                        and isinstance(element.value, ast.Name)
                        and element.value.id == "self"
                    ):
                        assigned.add(element.attr)
    return assigned


def test_every_carried_attribute_comes_back() -> None:
    """A value written into the payload is assigned on resume, or says who does."""
    carried = _snapshot_attributes() & set(QueryEngine._REARM_PRESERVED_ATTRS)
    dropped = sorted(carried - _restore_targets() - set(_RESTORED_BY_HELPER))
    assert not dropped, (
        "these attributes are read by snapshot() and never assigned on the way "
        f"back in: {dropped}. Writing a value out without restoring it is the "
        "same silence as never carrying it — assign it in "
        "resume_from_snapshot(), or name it in _RESTORED_BY_HELPER with what "
        "puts it back."
    )


def test_no_stale_restore_helper_entries() -> None:
    """An entry for an attribute the payload stopped carrying is a stale reason."""
    carried = _snapshot_attributes() & set(QueryEngine._REARM_PRESERVED_ATTRS)
    stale = sorted(set(_RESTORED_BY_HELPER) - carried)
    assert not stale, f"named as restored indirectly but no longer carried: {stale}"

    direct = sorted(set(_RESTORED_BY_HELPER) & _restore_targets())
    assert not direct, (
        "these are assigned directly on resume now — drop the indirect "
        f"explanation: {direct}"
    )


def test_every_restore_helper_entry_gives_a_reason() -> None:
    """A one-word entry is a list someone filled in to make a test pass."""
    thin = sorted(
        name for name, reason in _RESTORED_BY_HELPER.items() if len(reason.split()) < 6
    )
    assert not thin, f"these entries do not say what restores the value: {thin}"
