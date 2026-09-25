"""Guard: every reader of the engine's transcript declares the scope it asks about.

``engine.history`` is a SESSION transcript. Cross-run history seeding prepends
the earlier runs of the same session verbatim — their prose, their tool calls,
their tool results — and ``Message`` carries no run id, so the only thing that
separates this run's messages from an earlier run's is the
``SESSION_HISTORY_SEED_METADATA_KEY`` tag. A helper that reaches the raw
sequence and asks a question scoped to ONE run therefore gets an answer about
the whole session, and that answer is plausible: prose from an earlier run
reads as a fluent reply to a question this run was never asked, and an earlier
run's terminal result reads as "this run is already answered".

The rule is that a run-scoped helper draws from
``protocore.runtime.query._this_run_messages`` (or the narrower
``_this_run_model_turns`` built on it) and never from the raw sequence. A
docstring cannot enforce that — nothing stops the next helper from walking
``engine.history`` directly, which is how the readers this guard was written
for came to exist. So the enforcement is here: every function in ``protocore/``
that reaches the transcript is enumerated below with the reason its scope is
what it is, and a function that is not enumerated fails this test by name.

**The check is keyed on the transcript, not on the name of the variable
holding it.** An earlier version of this guard only recognised a receiver
literally spelled ``engine`` or ``self``; renaming a parameter to ``eng``,
binding the engine to a local, or reaching it through ``self.engine`` walked
straight past. Nothing about the *receiver* is inspected now. What is inspected
is the four routes by which the transcript can be reached:

* the attribute, on any receiver expression — ``eng.history``,
  ``self.engine.history``, ``ctx.engine.history_snapshot()``;
* its name as a string, anywhere — ``vars(e)["history"]``,
  ``e.__dict__["history"]``, ``attrgetter("history")``,
  ``e.snapshot()["history"]``;
* a dynamic attribute lookup, including one whose name the checker cannot
  read — ``getattr(e, "his" + "tory")`` is a lookup this file must treat as
  unreadable rather than as absent;
* the transcript handed in as an argument — a helper annotated
  ``history: list[Message]``, ``turns: dict[str, Message]``,
  ``turns: AsyncIterator[Message]``, ``*turns: Message``, one written against an
  import alias of the class (``list[Msg]``), or one with a module-level alias of
  any of those is the same door with the engine one frame away, and its caller
  may be a function this registry already authorises. That last clause is why
  this route exists AND why residual 2 below is a real gap rather than a
  formality: the call site being authorised means nothing else looks at the
  callee.

The registry is deliberately an ALLOWLIST rather than a pattern match. Whole-
transcript questions are legitimate and common — wire-pairing repair, prompt
assembly, token accounting, compaction, identity lookups keyed on a tool call
id — and no lexical rule tells them apart from a run-scoped question. What the
allowlist buys is that answering "which is this?" becomes a required step in
writing a new reader, rather than something the author has to know to ask.

Reading is separated from mutating. Adding, removing or replacing a message
changes the transcript; it does not ask a question about it, so it cannot
mis-attribute one. ``engine.history.append(...)``, ``engine.history.pop()``,
``engine.history[-1:] = [...]`` and ``del engine.history[0]`` are therefore not
findings, and no registry entry is needed to make them legal. The same split
holds one frame away: a parameter whose contents are only written into another
sequence — ``engine.history[0:0] = seed_messages`` — is a change too, and a
parameter a function never names is no question at all, which is what lets a
``Protocol`` method and ``@overload`` stubs through. Only questions are
registered.

**What this guard does not do.** It does not make a class of bug impossible. It
stops a reader that reaches the transcript by one of the routes above from
being written without its scope being declared — which is every reader this
checker can SEE in this package today, and every one of the readers this guard
was written for. What gets past it is listed below rather than left for the
next author to find. Each item is stated at the width it was measured at,
because a stated limit that is wrong is worse than an unstated one: a reader
trusts it and stops looking.

1. An already-registered function edited into a run-scoped reader. That is
   inherent to an allowlist. It is narrowed by
   :func:`test_the_registry_has_no_stale_entries` (a name cannot be recycled),
   by :data:`_Claim.RUN_SCOPED_BY_CONSTRUCTION`, which forces every entry whose
   *reason* asserts a run-scope property to name the test that pins it, and by
   :func:`test_every_pinning_test_still_pins_what_it_claims`, which makes the
   pins a two-sided declaration so an entry cannot be downgraded out from under
   the test written to hold it. What remains open is a NEW entry written
   whole-transcript with a reason that argues about one run, and a downgrade
   whose author edits BOTH sides of the pin into agreement. The phrase check in
   :func:`test_a_claim_that_cannot_be_pinned_does_not_pretend_to_be` is what
   catches the second, and it is a lexical filter over known wordings, so a
   wording it does not list gets past it. Do not read those two as "structural
   check, plus lexical backstop": measured, against a two-sided edit it is the
   lexical one that fires and the structural one that does not. The three
   mutations are recorded at
   :func:`test_every_pinning_test_still_pins_what_it_claims`.
2. A parameter whose annotation this checker cannot resolve to a container of
   ``Message``. The parameter route is annotation-keyed. It resolves quoted
   annotations, module-level type aliases and an import alias of ``Message``
   itself, and it reads ``*args`` and
   ``**kwargs`` as well as ordinary parameters — but ``turns: Any`` and an
   unannotated ``turns`` carry nothing to read, and neither does a container
   whose origin is absent from :data:`_MESSAGE_CONTAINER_ORIGINS`. Such a
   helper is invisible here.

   Its CALLER is a finding only if that caller is not itself registered. An
   ``Any``-annotated helper called from ``_drive_turn`` — which the registry
   authorises ``_whole`` — passes the whole suite with nothing failing;
   measured, not reasoned about. An earlier version of this docstring said the
   call site was a finding "wherever it is written". That was false, and it is
   exactly the mechanism that makes the parameter route worth having: a
   helper's caller is often a function this registry already authorises, so
   the attribute rule at the call site is satisfied and nothing else would look
   at the callee.
3. The transcript reached through a value this checker cannot follow back to a
   name: an engine held in a container, a message list rebuilt element by
   element, an accessor implemented in C.
4. Anything outside ``protocore/``. The seeding splice itself lives in the
   service repository, and so does at least one helper of the same shape as the
   readers fixed here.
"""

from __future__ import annotations

import ast
import enum
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path

import pytest

from protocore.contracts import types as contracts_types
from protocore.contracts.types import SESSION_HISTORY_SEED_METADATA_KEY

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PACKAGE_ROOT = _REPO_ROOT / "protocore"
_TESTS_ROOT = _REPO_ROOT / "tests"

#: The engine attribute that holds the session transcript, and the public
#: accessor that hands out the same thing. Both are doors into unscoped
#: history; a reader that took the second one would dodge a guard that only
#: watched the first.
_TRANSCRIPT_NAMES = frozenset({"history", "history_snapshot"})

#: Calls that CHANGE the sequence. A mutation asks nothing, so it cannot get a
#: run-scoped answer wrong — the full ``list`` mutating surface is here, not
#: just the four that happened to appear in the tree, because a guard that
#: refuses ``pop()`` while allowing ``append()`` sends the author to the
#: registry with a reason that is not true.
_MUTATING_METHODS = frozenset({"append", "extend", "insert", "clear", "pop", "remove", "reverse", "sort"})

#: Annotation heads that make a parameter a CONTAINER of messages rather than
#: one message. ``msg: Message`` is a single turn and reaches no transcript.
#: The mapping heads are here because ``dict[str, Message]`` holds a transcript
#: just as ``list[Message]`` does, and a guard that watched only sequences
#: rewarded the author who reached for a mapping. The async and generator heads
#: are here for the same reason and are the likelier annotation in this runtime:
#: ``AsyncIterator[Message]`` was measured walking past a list that stopped at
#: the synchronous spellings. Adding all of them cost no registry entry: this
#: package has no mapping-of-messages and no generator-of-messages parameter
#: today.
_MESSAGE_CONTAINER_ORIGINS = frozenset(
    {
        "list",
        "tuple",
        "set",
        "frozenset",
        "Sequence",
        "MutableSequence",
        "Iterable",
        "Iterator",
        "Collection",
        "deque",
        "AsyncIterable",
        "AsyncIterator",
        "Generator",
        "AsyncGenerator",
        "dict",
        "Mapping",
        "MutableMapping",
        "defaultdict",
        "OrderedDict",
    }
)

#: Where the run boundary is stated. Everything run-scoped draws from it;
#: ``_this_run_model_turns`` narrows it to the model's own words and is built
#: on it, not alongside it. Four other places re-derive the rule and are
#: declared in :data:`_SEED_KEY_DERIVED_ELSEWHERE` with the reason a filtered
#: sequence cannot serve them, so this is the one SOURCE, not the one mention.
_RUN_BOUNDARY_SOURCE = "protocore/runtime/query.py::_this_run_messages"


class _Claim(enum.Enum):
    """What an entry's reason asserts — and therefore what can verify it."""

    #: The answer is about the whole session by definition: what goes on the
    #: wire, what has to fit the context window, what gets serialised. There is
    #: no run-scope property to pin, and the entry must not pretend otherwise.
    WHOLE_TRANSCRIPT = "whole-transcript"

    #: The function reaches unscoped messages and yet its answer is about ONE
    #: run — because of where it looks, or what it keys on. That is a claim
    #: about behaviour, so it MUST name the tests that make it falsifiable.
    RUN_SCOPED_BY_CONSTRUCTION = "run-scoped-by-construction"

    #: The expression the checker flagged does not reach the transcript at all
    #: (a dynamic lookup whose attribute name it cannot read). Nothing to pin.
    NOT_THE_TRANSCRIPT = "not-the-transcript"


@dataclass(frozen=True)
class _Declaration:
    """One registry entry: the reason, what kind of claim it is, and its pins.

    ``reason`` used to be a bare string, and bare strings are prose that
    nothing checks. Splitting the claim out makes the difference between "this
    is about the whole session" and "this reaches the whole session but answers
    about one run" a thing the test suite can act on: the second kind has to
    name tests, and those tests have to exist.
    """

    reason: str
    claim: _Claim
    pinned_by: tuple[str, ...] = field(default=())


def _whole(reason: str) -> _Declaration:
    return _Declaration(reason=reason, claim=_Claim.WHOLE_TRANSCRIPT)


def _run_scoped(reason: str, *pinned_by: str) -> _Declaration:
    return _Declaration(
        reason=reason,
        claim=_Claim.RUN_SCOPED_BY_CONSTRUCTION,
        pinned_by=pinned_by,
    )


_CLAIMS = "tests/unit/runtime/test_history_registry_claims.py"
_COMPACTION_TESTS = "tests/unit/runtime/test_compaction_intra_run.py"

#: Every function in ``protocore/`` that this checker SEES reaching the
#: transcript, with the reason its question has the scope it has. Keyed
#: ``<path relative to the repository>::<qualified function name>``. The
#: qualifier is load-bearing: the module docstring lists the shapes that reach
#: the transcript without being seen, and they are absent from here because
#: they are invisible, not because they do not exist.
_WHOLE_HISTORY_BY_DESIGN: dict[str, _Declaration] = {
    # --- the transcript that goes on the wire -------------------------------
    "protocore/runtime/query.py::_drive_turn": _whole(
        "wire-pairing repair before a terminal snapshot, plus prompt assembly: "
        "both are about the transcript sent to the provider"
    ),
    "protocore/runtime/query.py::_stream_one_assistant_message": _whole(
        "wire-pairing repair, prompt assembly and the tool-batch protect index — all whole-transcript concerns"
    ),
    "protocore/runtime/query.py::_emit_dispatch_cancel_teardown": _whole(
        "pairs orphan tool_use blocks so a resumed snapshot is wire-valid"
    ),
    "protocore/runtime/query.py::_policy_pair_orphan_tool_calls": _whole(
        "pairs orphan tool_use blocks so a resumed snapshot is wire-valid"
    ),
    "protocore/runtime/turn_policies/compaction.py::PerIterationCompactionPolicy.apply": _whole(
        "the tool-batch protect index is an offset into the whole transcript, "
        "because that is what compaction reads"
    ),
    "protocore/runtime/query.py::_emit_llm_terminal": _whole(
        "pairs orphan tool_use blocks so a resumed snapshot is wire-valid"
    ),
    "protocore/runtime/soft_stop.py::leave": _whole(
        "removes the wind-down notice wherever it sits in the session transcript, however it "
        "got there: the executor's seed may carry one whose wind-down was never driven to its end"
    ),
    "protocore/runtime/query.py::_emit_empty_completion_terminal": _whole(
        "pairs orphan tool_use blocks so a resumed snapshot is wire-valid"
    ),
    "protocore/runtime/query.py::_emit_tool_precondition_terminal": _whole(
        "pairs orphan tool_use blocks so a resumed snapshot is wire-valid"
    ),
    "protocore/runtime/query.py::_settle_interrupted_tool_intents": _whole(
        "closes out calls left in flight by a stopped run: it looks for a "
        "result anywhere in the transcript before deciding an outcome was "
        "never recorded, and a result that landed before the tail is still a "
        "result"
    ),
    "protocore/runtime/query.py::_abandon_pending_approval": _whole(
        "closes a call that was parked at an approval gate: the result has to "
        "go directly after the call it answers, wherever in the transcript "
        "that call sits, which is the same pairing concern as the repairs "
        "below"
    ),
    "protocore/runtime/query.py::_settle_parked_call": _whole(
        "closes a call a person's decision answered — a denial, an answer to "
        "the question the tool asked — and the result has to go directly "
        "after the call it answers, wherever in the transcript that call "
        "sits: the same pairing concern as the abandonment above"
    ),
    "protocore/runtime/query.py::_insert_tool_result_after_use": _whole(
        "places a tool result directly after the call it answers, anywhere in "
        "the sequence it is handed — a pairing concern, like the repairs above"
    ),
    "protocore/runtime/query.py::_llm_history": _whole(
        "prompt assembly: builds the view of the transcript the provider is "
        "sent. Result eviction, the compaction checkpoint and the result split "
        "all rewrite that outbound copy and leave persist untouched, so what "
        "this reads is the whole of what goes on the wire"
    ),
    "protocore/runtime/query_engine.py::QueryEngine.note_history_persisted": _whole(
        "records which messages the session store now holds, over the sequence "
        "it is handed: what is persisted is the whole of what the session has "
        "said, and a marker covering any less of it would let the next "
        "hand-over append onto rows that were never written"
    ),
    "protocore/runtime/history_persist.py::persist_history": _whole(
        "hands the session store what changed in the session transcript: what "
        "is persisted is the whole of what the session has said, because that "
        "is what the next process reads back when it seeds an engine"
    ),
    "protocore/runtime/stale_result_trim.py::trim_stale_results": _whole(
        "prompt assembly over the outbound view it is handed: cuts oversized "
        "tool results down to their head. Every result the request carries "
        "costs what it costs, seeded or not, so the whole view is what has to "
        "be measured"
    ),
    "protocore/runtime/stale_result_trim.py::_calls_of_the_latest_round": _whole(
        "prompt assembly over the outbound view it is handed: finds the last "
        "message in it carrying tool calls, so the results answering that "
        "batch can be left whole. A later message cannot be shadowed by an "
        "earlier one, so the whole sequence is the cheapest place to look"
    ),
    "protocore/runtime/result_eviction.py::evict_history_for_llm": _whole(
        "prompt assembly over the sequence it is handed: replaces unmarked "
        "Read/Grep results in the outbound copy"
    ),
    "protocore/runtime/result_eviction.py::tool_names_by_call_id": _whole(
        "identity lookups keyed on tool call ids, over the outbound view it "
        "is handed. It answers the same question as tool_name_for_result for "
        "every id at once, so its scope is the same: a call id names a single "
        "call, and the view it serves includes seeded turns, whose results "
        "must resolve too"
    ),
    "protocore/runtime/result_eviction.py::pins_invalidated_by_writes": _whole(
        "pairs every result that names a path with the LATER calls that "
        "rewrote it, over the outbound view it is handed. The transcript is "
        "the right scope by construction: a file is rewritten once and stays "
        "rewritten, so a pinned view of it is stale for everything that comes "
        "after the write, and a narrower scan would go on serving the version "
        "before it"
    ),
    "protocore/contracts/snapshot.py::_v4_to_v5": _whole(
        "lifts a stored payload, not a live transcript: it walks the recorded "
        "messages of the snapshot handed to it and states the two fields "
        "version 5 adds to a tool result. It asks nothing of any run"
    ),
    "protocore/runtime/result_eviction.py::tool_name_for_result": _whole(
        "identity lookup keyed on a tool call id, over the outbound view it is "
        "handed. A call id names a single call, so a wider search cannot "
        "return a different tool name — and the view it rewrites includes "
        "seeded turns, whose results must resolve too"
    ),
    "protocore/runtime/compact_checkpoint.py::apply_checkpoint": _whole(
        "prompt assembly: swaps the compacted head for a summary in the "
        "outbound copy it returns"
    ),
    "protocore/runtime/query.py::_prepend_system_sections": _whole(
        "prompt assembly over the outbound message list it is handed"
    ),
    "protocore/runtime/query.py::_repair_outbound_tool_pairing": _whole(
        "wire-pairing repair over the outbound message list it is handed"
    ),
    "protocore/runtime/query.py::_normalize_outbound_system_messages": _whole(
        "prompt assembly over the outbound message list it is handed"
    ),
    "protocore/runtime/query.py::_synthesize_missing_tool_results": _whole(
        "pairs orphan tool_use blocks in the list it is handed so the wire payload is valid"
    ),
    "protocore/runtime/prompt_caching.py::apply_system_and_3": _whole(
        "places cache breakpoints in the outbound message list it is handed"
    ),
    # --- size of the transcript ---------------------------------------------
    "protocore/runtime/query.py::_run_compaction": _whole(
        "token accounting and compaction operate on the whole transcript, "
        "which is what has to fit in the context window"
    ),
    "protocore/runtime/query.py::_handle_context_window_exceeded": _whole(
        "token accounting and forced compaction over the whole transcript"
    ),
    "protocore/runtime/query.py::_drive_one_stream": _whole("message count for a diagnostic log line"),
    "protocore/runtime/compact_checkpoint.py::build_checkpoint": _whole(
        "compaction over the sequence it is handed: what has to fit the "
        "context window is all of it, and the retained tail is chosen by "
        "position rather than by provenance"
    ),
    "protocore/runtime/compact_checkpoint.py::collect_file_op_facts": _whole(
        "compaction: names the file operations in the prefix it is handed so "
        "they survive that prefix being dropped"
    ),
    "protocore/runtime/context/manager.py::ContextManager.build_context": _whole(
        "prompt assembly: the provider is sent the whole transcript"
    ),
    "protocore/runtime/context/manager.py::ContextManager.run_compaction": _whole(
        "compaction shrinks the whole transcript to fit the context window"
    ),
    "protocore/runtime/context/manager.py::ContextManager.force_compaction": _whole(
        "compaction shrinks the whole transcript to fit the context window"
    ),
    "protocore/runtime/context/manager.py::ContextManager.current_prompt_tokens": _whole(
        "token accounting over the sequence it is handed"
    ),
    "protocore/runtime/context/manager.py::ContextManager.needs_compaction": _whole(
        "token accounting over the sequence it is handed"
    ),
    "protocore/runtime/context/manager.py::ContextManager.needs_emergency_compaction": (
        _whole("token accounting over the sequence it is handed")
    ),
    # --- compaction, which is a whole-transcript operation by definition -----
    "protocore/runtime/context/compaction.py::current_tool_batch_protect_index": _whole(
        "the protect index is a position in the transcript that goes on the wire"
    ),
    "protocore/runtime/context/compaction.py::_effective_eligible_upper": _whole(
        "the eligibility window is a range over the whole transcript"
    ),
    "protocore/runtime/context/compaction.py::estimate_history_tokens": _whole(
        "token accounting over the sequence it is handed"
    ),
    "protocore/runtime/context/compaction.py::TokenEstimator.estimate_history": (
        _whole("token accounting over the sequence it is handed")
    ),
    "protocore/runtime/context/compaction.py::estimate_history_tokens_uncalibrated": (
        _whole("token accounting over the sequence it is handed")
    ),
    "protocore/runtime/context/compaction.py::TokenEstimator.estimate_history_uncalibrated": (
        _whole("token accounting over the sequence it is handed")
    ),
    "protocore/runtime/context/compaction.py::_prune_failed_anchor_keys": _whole(
        "the census it prunes is keyed by anchors anywhere in the transcript, "
        "so the question 'is this unit still here' is asked of all of it"
    ),
    "protocore/runtime/context/compaction.py::run_tier1_truncation": _whole(
        "sheds bytes from the whole transcript so it fits the context window"
    ),
    "protocore/runtime/context/compaction.py::_compaction_reference_indices": _whole(
        "locates frozen bootstrap reference blocks anywhere in the transcript"
    ),
    "protocore/runtime/context/compaction.py::_build_summarisation_units": _whole(
        "tool-pairing connected components span the whole transcript; a unit "
        "split anywhere within it would orphan one side of a pair"
    ),
    "protocore/runtime/context/compaction.py::run_tier2_summarisation": _whole(
        "collapses turns anywhere in the transcript to fit the context window"
    ),
    "protocore/runtime/context/compaction.py::_summarise_unit": _whole(
        "summarises one unit of the whole transcript Tier-2 was handed"
    ),
    "protocore/runtime/context/compaction.py::_operator_turn_indices": _whole(
        "the operator's turns are positions in the whole transcript; a "
        "filtered copy has no indices into the original"
    ),
    "protocore/runtime/context/compaction.py::_foldable_indices": _whole(
        "decides which positions anywhere in the transcript the fold may take"
    ),
    "protocore/runtime/context/compaction.py::_fold_spans": _whole(
        "finds runs of old summaries and operator turns anywhere in the transcript"
    ),
    "protocore/runtime/context/compaction.py::_fold_span": _whole(
        "folds one span of the whole transcript Tier-3 was handed"
    ),
    "protocore/runtime/context/compaction.py::_fold_anchor_key": _whole(
        "derives one durable id from the span it is handed; it asks nothing about whose run it is"
    ),
    "protocore/runtime/context/compaction.py::run_tier3_fold": _whole(
        "folds runs of old summaries and operator turns anywhere in the transcript"
    ),
    "protocore/runtime/context/compaction.py::tier1_has_work": _whole(
        "asks, without rewriting, what Tier 1 would shed anywhere in the transcript"
    ),
    "protocore/runtime/context/compaction.py::_plan_tier2": _whole(
        "decides which units anywhere in the transcript Tier 2 would send"
    ),
    "protocore/runtime/context/compaction.py::tier2_has_work": _whole(
        "asks, without calling the summariser, whether Tier 2 has a unit to send"
    ),
    "protocore/runtime/context/compaction.py::tier3_has_work": _whole(
        "asks, without folding, whether the transcript holds a span to fold"
    ),
    "protocore/runtime/context/manager.py::ContextManager.has_proactive_work": _whole(
        "asks every tier whether a proactive pass over the transcript would change it"
    ),
    "protocore/runtime/query.py::_current_prompt_tokens": _whole(
        "measures the whole prompt, which is what the backoff's growth bound compares"
    ),
    "protocore/runtime/query.py::_suspend_proactive_compaction": _whole(
        "records the size of the whole prompt, which is what the growth bound measures"
    ),
    "protocore/runtime/query.py::_proactive_llm_tiers_allowed": _whole(
        "measures the whole prompt's growth since the suspension began"
    ),
    "protocore/runtime/query.py::_proactive_pass_is_idle": _whole(
        "compares the transcript with the one the last idle probe saw, then asks "
        "the tiers about all of it, as the pass itself would"
    ),
    "protocore/runtime/context/manager.py::ContextManager._fold": _whole(
        "hands the whole transcript to the Tier-3 fold after Tier-2"
    ),
    # --- building the seed, from messages the caller supplies ---------------
    "protocore/runtime/context/session_memory.py::_serialize_turns": _whole(
        "renders the turns it is handed; it reaches no engine"
    ),
    "protocore/runtime/context/session_memory.py::build_summary_user_message": _whole(
        "renders the turns it is handed; it reaches no engine"
    ),
    "protocore/runtime/context/session_memory.py::extract_artifacts": _whole(
        "collects artifact references from the turns it is handed"
    ),
    "protocore/runtime/context/session_memory.py::fold_run": _whole(
        "folds the sequence it is handed into durable session memory; it "
        "reaches no engine and does no scoping of its own"
    ),
    "protocore/runtime/context/session_memory.py::bound_catchup_source": _whole(
        "bounds the sequence it is handed to a byte budget"
    ),
    "protocore/runtime/context/session_memory.py::_tail_by_budget": _whole(
        "bounds the sequence it is handed to a byte budget"
    ),
    "protocore/runtime/context/session_memory.py::build_seed": _whole(
        "assembles the seed from the sequence it is handed, treating every "
        "message in it the same way; it reaches no engine"
    ),
    "protocore/runtime/loop_strategies.py::DeepStrategy._fetch_plan_fallback": _whole(
        "re-drives the provider with the outbound message list it is handed"
    ),
    "protocore/runtime/query.py::build_llm_request": _whole(
        "assembles the outbound message list it is handed into one provider "
        "request; it reaches no engine and selects nothing"
    ),
    # --- identity lookups keyed on a tool call id ---------------------------
    "protocore/runtime/query.py::_tool_name_for_call_id": _run_scoped(
        "resolves ONE tool_call_id to its tool name; a call id identifies a "
        "single call, so the search cannot land on another run's",
        f"{_CLAIMS}::test_call_id_lookups_resolve_the_call_they_are_asked_for",
    ),
    "protocore/runtime/query.py::_history_has_tool_result": _run_scoped(
        "presence of the result for ONE tool_call_id",
        f"{_CLAIMS}::test_call_id_lookups_resolve_the_call_they_are_asked_for",
    ),
    "protocore/runtime/query.py::_history_tool_result_is_terminal": _run_scoped(
        "inspects the result for ONE tool_call_id the caller just appended",
        f"{_CLAIMS}::test_call_id_lookups_resolve_the_call_they_are_asked_for",
    ),
    "protocore/runtime/query.py::_tool_call_from_history": _run_scoped(
        "rebuilds ONE parked call from the tool_call_id its interrupt names; "
        "a call id identifies a single call, so the search cannot land on "
        "another run's",
        f"{_CLAIMS}::test_interrupt_resolution_is_keyed_on_the_parked_call",
    ),
    "protocore/runtime/query.py::_apply_updated_input": _run_scoped(
        "rewrites the tool_use block of ONE parked call to the arguments a "
        "person corrected, found by that call's id",
        f"{_CLAIMS}::test_interrupt_resolution_is_keyed_on_the_parked_call",
    ),
    "protocore/runtime/query.py::_assert_history_has_matching_pending_tool_use": (
        _run_scoped(
            "structural check that ONE approved tool call matches its pending tool_use block",
            f"{_CLAIMS}::test_pending_tool_use_assertion_is_keyed_on_the_approved_call",
        )
    ),
    # --- what the round now driving has produced ----------------------------
    "protocore/runtime/query.py::_this_round_messages": _run_scoped(
        "answers about the round now driving: it starts at the message AFTER "
        "the last one a caller put in — the operator's prompt, or the tool "
        "result a parked run was resumed with — so a prior run's prose and "
        "tool calls precede the boundary and cannot answer it. The boundary is "
        "re-derived rather than taken from the seed tag because the tag is set "
        "by the executor and not by every host, and a host that hands over a "
        "session's earlier turns verbatim would otherwise have the predicate "
        "answer for a run that is over",
        f"{_CLAIMS}::test_produced_output_ignores_a_seeded_prior_run",
    ),
    # --- the tail --------------------------------------------------------
    "protocore/runtime/turn_policies/sibling_walk.py::prose_gate_just_injected": _run_scoped(
        "inspects the LAST message only; the seed is prepended, so the tail always belongs to this run",
        f"{_CLAIMS}::test_prose_gate_reads_the_tail_and_not_a_seeded_turn",
    ),
    # --- durable file bytes, which outlive a run ----------------------------
    # The pin holds the GATE — remove it, or make the path comparison
    # permissive, and the test fails. It does not hold the second clause. That
    # clause says only this run's own call binds the path, and verifying it
    # means reading three binders in two modules (longfile_convergence.py and
    # query_engine.py's resume_from_snapshot), which is the reading a pin is
    # supposed to replace. It is true today — the snapshot restore rebinds the
    # SAME run's path — so no defect follows, but the clause is asserted here
    # and checked nowhere.
    "protocore/runtime/longfile_convergence.py::_active_file_tail": _run_scoped(
        "the continuation anchor is the file's durable content, which is "
        "cumulative across the session's runs rather than owned by one of "
        "them; the walk is gated on _longfile_active_path, which only this "
        "run's own byte-adding call can bind",
        f"{_CLAIMS}::test_active_file_tail_needs_this_runs_own_binding",
    ),
    # --- the run boundary, re-derived where compaction needs positions ------
    "protocore/runtime/context/compaction.py::_first_user_turn_index": _run_scoped(
        "returns THIS run's task: it skips seeded prior-run user turns, which precede the new task in history",
        f"{_COMPACTION_TESTS}::test_first_user_turn_skips_seeded_prior_run_turns",
    ),
    "protocore/runtime/context/compaction.py::_session_history_seed_indices": (
        _run_scoped(
            "selects exactly the prior-run turns, so the lossy Tier-2 collapse can be withheld from them",
            f"{_CLAIMS}::test_seed_indices_select_every_seeded_turn_and_nothing_else",
        )
    ),
    # --- the engine's own storage -------------------------------------------
    "protocore/runtime/query_engine.py::QueryEngine.run": _run_scoped(
        "asserts the transcript ends on a user turn, then appends the task — "
        "which is what puts this run's task AFTER the seed and makes the "
        "tail-most user turn this run's",
        f"{_CLAIMS}::test_latest_user_message_is_this_runs_task_not_a_seeded_one",
    ),
    "protocore/runtime/query_engine.py::QueryEngine._history_can_be_continued": _whole(
        "decides whether a continuation has anything to answer: the tail turn, "
        "plus whether a tool call on it is still unpaired — and the result "
        "that would pair it may sit anywhere earlier in the transcript"
    ),
    "protocore/runtime/query_engine.py::QueryEngine.snapshot": _whole(
        "serialises the whole transcript for durable resume"
    ),
    "protocore/runtime/query_engine.py::QueryEngine.resume_from_snapshot": _whole(
        "rehydrates the whole transcript from a durable snapshot"
    ),
    "protocore/runtime/query_engine.py::QueryEngine.history_snapshot": _whole(
        "the accessor that hands out the whole transcript, by contract"
    ),
    "protocore/runtime/query_engine.py::QueryEngine.latest_user_message": _run_scoped(
        "the tail-most user turn; the new task is appended after the seed, so it is always this run's",
        f"{_CLAIMS}::test_latest_user_message_is_this_runs_task_not_a_seeded_one",
    ),
    "protocore/runtime/query_engine.py::QueryEngine.needs_compaction": _whole(
        "token accounting over the whole transcript"
    ),
    "protocore/runtime/query.py::_calibrate_near_compaction_trigger": _whole(
        "sizes the same whole transcript the compaction gate measures, so the "
        "gate reads the provider's count of what it decides on"
    ),
    "protocore/runtime/query_engine.py::QueryEngine.needs_emergency_compaction": _whole(
        "token accounting over the whole transcript"
    ),
    # --- names written down as data, reaching nothing -----------------------
    "protocore/runtime/query_engine.py::QueryEngine": _Declaration(
        reason=(
            "the class body's re-arm preserved-attribute set spells the "
            "transcript's name so the re-arm leaves it alone; the set is "
            "compared against attribute names and reads no message"
        ),
        claim=_Claim.NOT_THE_TRANSCRIPT,
    ),
    # --- dynamic lookups whose attribute name this checker cannot read ------
    "protocore/conformance/suite.py::ContractSuite.test_asynchronous_members_stay_asynchronous": _Declaration(
        reason=(
            "getattr over the member names a Protocol declares, on a host's "
            "adapter; it reaches no engine and names no transcript"
        ),
        claim=_Claim.NOT_THE_TRANSCRIPT,
    ),
    "protocore/conformance/suite.py::ContractSuite.test_asynchronous_iterators_stay_asynchronous_iterators": _Declaration(
        reason=(
            "getattr over the member names a Protocol declares, on a host's "
            "adapter; it reaches no engine and names no transcript"
        ),
        claim=_Claim.NOT_THE_TRANSCRIPT,
    ),
    "protocore/conformance/suite.py::ContractSuite.test_synchronous_members_stay_synchronous": _Declaration(
        reason=(
            "getattr over the member names a Protocol declares, on a host's "
            "adapter; it reaches no engine and names no transcript"
        ),
        claim=_Claim.NOT_THE_TRANSCRIPT,
    ),
    "protocore/conformance/suite.py::ContractSuite.test_every_member_accepts_the_parameters_the_core_passes": _Declaration(
        reason=(
            "getattr over the member names a Protocol declares, on a host's "
            "adapter; it reaches no engine and names no transcript"
        ),
        claim=_Claim.NOT_THE_TRANSCRIPT,
    ),
    "protocore/contracts/observability.py::RequestManifest.with_blob_refs": _Declaration(
        reason=(
            "getattr over the manifest's own value-slot names, to fill in "
            "where a host stored each oversized body; it reaches no engine "
            "and names no transcript"
        ),
        claim=_Claim.NOT_THE_TRANSCRIPT,
    ),
    "protocore/contracts/observability.py::_system_prompt_payload": _whole(
        "reads the leading system messages off the outbound list of ONE "
        "request, which its caller has already assembled; it reaches no "
        "engine and selects nothing"
    ),
    "protocore/tests_support/adapters.py::InMemoryRequestManifestSink.body_of": _Declaration(
        reason=(
            "getattr over the manifest's own value-slot names, to fetch one "
            "recorded body inline or out of the blob store; it reaches no "
            "engine and names no transcript"
        ),
        claim=_Claim.NOT_THE_TRANSCRIPT,
    ),
    "protocore/conformance/test_conformance_package.py::_run_every_check": _Declaration(
        reason=(
            "getattr over the conformance suite's own check names, to run every "
            "one against a subject; it reaches no engine and names no transcript"
        ),
        claim=_Claim.NOT_THE_TRANSCRIPT,
    ),
    "protocore/tests_support/adapters.py::InMemoryRunStore.list": _Declaration(
        reason=(
            "getattr over caller-supplied filter field names on a run record; "
            "it reaches no engine and names no transcript"
        ),
        claim=_Claim.NOT_THE_TRANSCRIPT,
    ),
}


# ---------------------------------------------------------------------------
# The checker
# ---------------------------------------------------------------------------


class _Route(enum.Enum):
    """How an expression reaches the transcript. Decides the remedy offered."""

    ATTRIBUTE = "the attribute itself"
    LITERAL_NAME = "the transcript's name as a string"
    DYNAMIC = "a dynamic attribute lookup"
    UNREADABLE = "a dynamic attribute lookup this check cannot read"
    PARAMETER = "the transcript handed in as an argument"


@dataclass(frozen=True)
class _Touch:
    """One place a function reaches the transcript."""

    lineno: int
    route: _Route


_REMEDIES: dict[_Route, str] = {
    _Route.ATTRIBUTE: (
        "This reaches the session transcript directly. If the question is "
        "scoped to ONE run, draw the messages from "
        "protocore.runtime.query._this_run_messages instead — a prior run of "
        "the same session is seeded into this history and will answer it. If "
        "the question really is about the whole session, register the function "
        "in _WHOLE_HISTORY_BY_DESIGN with _whole(reason)."
    ),
    _Route.LITERAL_NAME: (
        "This names the transcript as a string, which reaches it through "
        "vars(), __dict__, a snapshot dict or an attribute-getter. Same rule "
        "as the attribute: _this_run_messages for a run-scoped question, or a "
        "_whole(reason) entry for a session-wide one."
    ),
    _Route.DYNAMIC: (
        "This looks the transcript up by name at runtime. Same rule as the "
        "attribute: _this_run_messages for a run-scoped question, or a "
        "_whole(reason) entry for a session-wide one."
    ),
    _Route.UNREADABLE: (
        "This looks up an attribute whose name this check cannot read, so it "
        "cannot tell the transcript from anything else. Spell the name as a "
        "literal or as a module-level string constant, or — if the lookup "
        "never names the transcript — register the function with "
        "_Declaration(reason=..., claim=_Claim.NOT_THE_TRANSCRIPT)."
    ),
    _Route.PARAMETER: (
        "This is handed a container of messages, and its caller may pass a "
        "whole session transcript. If the question is scoped to ONE run, take "
        "the run's messages at the CALL SITE (_this_run_messages) and pass "
        "those. If the function treats whatever it is given as one sequence, "
        "register it with _whole(reason). If what it is handed is not a "
        "transcript at all — hand-authored few-shot examples, a fixture, a "
        "batch assembled for one request — register it with "
        "_Declaration(reason=..., claim=_Claim.NOT_THE_TRANSCRIPT) rather than "
        "recording a whole-transcript reason that is not true of it."
    ),
}

#: Not a finding, and stated here because the previous version of this guard
#: refused these and then told the author to declare the function as a
#: whole-transcript READER — a reason that would have been false, and a name
#: permanently pre-authorised for a question it does not ask.
_MUTATIONS_ARE_NOT_FINDINGS = (
    "Changing the transcript is not a finding here: append, extend, insert, "
    "clear, pop, remove, reverse, sort, slice assignment, del and rebinding "
    "the attribute all pass without a registry entry, because a change asks "
    "no question and so cannot mis-attribute one. The same holds one frame "
    "away: a parameter whose every use writes its contents into something else "
    "is a change too, and needs no entry. If your function is listed above and "
    "you believe it only changes the transcript, one of its uses is a question "
    "— a length, an iteration, a subscript read — and that use is the finding."
)


def _enclosing_qualname(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str:
    """Dotted name of the function/class chain containing ``node``."""
    names: list[str] = []
    current: ast.AST | None = node
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(current.name)
        current = parents.get(current)
    return ".".join(reversed(names)) or "<module>"


def _module_string_constants(tree: ast.Module) -> dict[str, str]:
    """Module-level ``NAME = "literal"`` bindings, for resolving ``getattr``.

    ``getattr(engine, FORCE_NEXT_TOOL_ATTR, None)`` is readable; treating it as
    unreadable would put four honest call sites in the registry and teach the
    author that the entry is a formality.
    """
    constants: dict[str, str] = {}
    for stmt in tree.body:
        value = stmt.value if isinstance(stmt, (ast.Assign, ast.AnnAssign)) else None
        if not (isinstance(value, ast.Constant) and isinstance(value.value, str)):
            continue
        targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
        for target in targets:
            if isinstance(target, ast.Name):
                constants[target.id] = value.value
    return constants


def _resolve_annotation(annotation: ast.expr, depth: int = 0) -> ast.expr | None:
    """Parse a STRING annotation into the expression it stands for.

    ``turns: "Sequence[Message]"`` is an ``ast.Constant``, not a subscript, and
    a check that walked the node as written saw no transcript at all. Found by
    attacking this rule rather than by reading it.
    """
    if depth > 4:
        return None
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        try:
            parsed = ast.parse(annotation.value, mode="eval").body
        except SyntaxError:
            return None
        return _resolve_annotation(parsed, depth + 1)
    return annotation


def _import_aliases(tree: ast.AST) -> dict[str, str]:
    """Local name -> the name it was imported as, for every ``import`` in ``tree``.

    An ``import`` binds an :class:`ast.alias`, which is neither an
    :class:`ast.Name` nor an :class:`ast.Attribute`. Nothing in this file used
    to read one, and that single omission was a hole under two separate rules at
    once: ``from … import Message as Msg`` made ``turns: list[Msg]`` invisible
    to the parameter route, and ``from … import SESSION_HISTORY_SEED_METADATA_KEY
    as _SEED_TAG`` made a re-derivation of the run boundary invisible to
    :func:`seed_key_derivations`. Both are the idiomatic spelling, not a
    contrivance. Read once, here, and handed to both.

    Walked rather than read off ``tree.body`` so an import under
    ``if TYPE_CHECKING:`` — where an annotation-only name usually lives — or one
    inside a function counts the same as a top-level one.
    """
    bound: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        for alias in node.names:
            original = alias.name.rsplit(".", 1)[-1]
            bound[alias.asname or original] = original
    return bound


def _message_names(tree: ast.AST) -> frozenset[str]:
    """Every local name that means ``Message`` in this module."""
    aliases = _import_aliases(tree)
    return frozenset({"Message"} | {local for local, original in aliases.items() if original == "Message"})


def _names_message(node: ast.AST, names: frozenset[str] = frozenset({"Message"}), depth: int = 0) -> bool:
    """Whether ``node`` names ``Message``, through aliases and string forms too.

    ``names`` is the module's own vocabulary for the class, from
    :func:`_message_names`; it defaults to the bare class name so this stays
    usable on a fragment with no imports.
    """
    if depth > 4:
        return False
    for inner in ast.walk(node):
        if isinstance(inner, ast.Name) and inner.id in names:
            return True
        if isinstance(inner, ast.Attribute) and inner.attr in names:
            return True
        if isinstance(inner, ast.Constant) and isinstance(inner.value, str):
            resolved = _resolve_annotation(inner, depth)
            if resolved is not None and _names_message(resolved, names, depth + 1):
                return True
    return False


def _module_type_aliases(tree: ast.Module) -> dict[str, ast.expr]:
    """Module-level ``X = list[Message]`` bindings, in all three spellings.

    The registry repeats ``list[Message]`` across thirty entries, so collapsing
    it to an alias is the ordinary tidy-up this code invites rather than an
    evasion — and an alias is a bare ``ast.Name`` at the annotation site, which
    a check that demanded a subscript there could not see.
    """
    aliases: dict[str, ast.expr] = {}
    for stmt in tree.body:
        if isinstance(stmt, ast.TypeAlias):  # type X = list[Message]
            if isinstance(stmt.name, ast.Name):
                aliases[stmt.name.id] = stmt.value
            continue
        value = stmt.value if isinstance(stmt, (ast.Assign, ast.AnnAssign)) else None
        if value is None:
            continue
        targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
        for target in targets:
            if isinstance(target, ast.Name):
                aliases[target.id] = value
    return aliases


def _is_message_container(
    annotation: ast.expr,
    aliases: dict[str, ast.expr] | None = None,
    names: frozenset[str] = frozenset({"Message"}),
    depth: int = 0,
) -> bool:
    """Whether ``annotation`` describes a CONTAINER of ``Message``.

    ``msg: Message`` is one turn and reaches no transcript; ``list[Message]``,
    ``Sequence[Message] | None``, ``tuple[Message, ...]``, ``dict[str, Message]``
    and their quoted forms are transcripts as far as their callers are
    concerned. A bare name is looked up in the module's type aliases before it
    is given up on, so ``_Transcript = list[Message]`` is not a way out, and the
    element type is matched against the module's own ``names`` for the class, so
    ``list[Msg]`` behind ``import Message as Msg`` is not one either.
    """
    if depth > 4:
        return False
    resolved = _resolve_annotation(annotation)
    if resolved is None:
        return False
    aliases = aliases or {}
    for node in ast.walk(resolved):
        if isinstance(node, ast.Name) and node.id in aliases:
            if _is_message_container(aliases[node.id], aliases, names, depth + 1):
                return True
            continue
        if not isinstance(node, ast.Subscript):
            continue
        origin = node.value
        name = origin.attr if isinstance(origin, ast.Attribute) else getattr(origin, "id", None)
        if name not in _MESSAGE_CONTAINER_ORIGINS:
            continue
        if _names_message(node.slice, names):
            return True
    return False


def _asks_nothing_of(node: ast.FunctionDef | ast.AsyncFunctionDef, parameter: str) -> bool:
    """Whether no use of ``parameter`` in ``node`` is a question about it.

    The attribute route already separates changing the transcript from asking
    it a question, because a change cannot mis-attribute an answer. The same is
    true one frame away: ``engine.history[0:0] = seed_messages`` asks the
    handed-in sequence nothing. Refusing that and then telling the author that
    slice assignment needs no registry entry is a message that contradicts
    itself, and there is no true remedy behind it — the function asks no
    question, so scoping at the call site does not apply, and registering a
    pure mutation as a whole-transcript READER states something false.

    Writing the parameter's contents somewhere else is such a use. So is not
    using it at all, which is the case for a ``Protocol`` method and for the
    ``@overload`` stubs above an implementation: a declaration with no body
    reads nothing, and demanding a reason about transcript scope for one puts
    a fiction in the registry.

    "Written somewhere else" means the parameter is the WHOLE value being
    stored. An earlier version of this rule looked only at the assignment's
    TARGETS, so any statement storing into a subscript or an attribute was
    skipped whatever its right-hand side asked — and
    ``engine.answered = any(m.role is MessageRole.assistant for m in turns)``
    is a question about every turn handed in, stored into an attribute. It was
    measured invisible. Requiring the parameter to BE the value keeps the case
    this exemption exists for (``engine.history[0:0] = seed_messages``,
    ``engine.history += turns``) and drops the rest.

    One question anywhere — a ``len()``, an iteration, a subscript read — and
    the exemption is gone for the whole function.
    """
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(node):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    for inner in ast.walk(node):
        if not (isinstance(inner, ast.Name) and inner.id == parameter and isinstance(inner.ctx, ast.Load)):
            continue
        statement: ast.AST | None = inner
        while statement is not None and not isinstance(statement, ast.stmt):
            statement = parents.get(statement)
        if isinstance(statement, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
            if (
                targets
                and all(isinstance(target, (ast.Subscript, ast.Attribute)) for target in targets)
                and statement.value is inner
            ):
                continue
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
            call = statement.value
            if isinstance(call.func, ast.Attribute) and call.func.attr in _MUTATING_METHODS:
                continue
        return False
    return True


def _attribute_route(node: ast.Attribute, parents: dict[ast.AST, ast.AST]) -> _Route | None:
    """Classify ``<anything>.history`` — a read, or a change that asks nothing.

    The receiver is deliberately not looked at. Keying on a receiver spelled
    ``engine`` or ``self`` is what let a reader through under the name ``eng``.
    """
    if isinstance(node.ctx, (ast.Store, ast.Del)):
        return None  # rebinding the attribute
    parent = parents.get(node)
    if isinstance(parent, ast.Attribute) and parent.attr in _MUTATING_METHODS:
        return None  # engine.history.append(...) / .pop() / .remove(...)
    if isinstance(parent, ast.Subscript) and parent.value is node and isinstance(parent.ctx, (ast.Store, ast.Del)):
        return None  # engine.history[-1:] = [...] / del engine.history[0]
    return _Route.ATTRIBUTE


def _call_route(node: ast.Call, constants: dict[str, str]) -> _Route | None:
    """Classify ``getattr`` / ``setattr`` / ``delattr`` on any receiver."""
    if not (isinstance(node.func, ast.Name) and len(node.args) >= 2):
        return None
    if node.func.id not in {"getattr", "setattr", "delattr"}:
        return None
    name_arg = node.args[1]
    if isinstance(name_arg, ast.Constant) and isinstance(name_arg.value, str):
        name: str | None = name_arg.value
    elif isinstance(name_arg, ast.Name):
        name = constants.get(name_arg.id)
    else:
        name = None
    if node.func.id != "getattr":
        return None  # setattr/delattr change the binding; they ask nothing
    if name in _TRANSCRIPT_NAMES:
        return _Route.DYNAMIC
    if name is None:
        return _Route.UNREADABLE
    return None


def _constant_route(node: ast.Constant, parents: dict[ast.AST, ast.AST]) -> _Route | None:
    """Classify a bare ``"history"`` string — the indirect doors, all at once.

    ``vars(e)["history"]``, ``e.__dict__["history"]``, ``attrgetter("history")``
    and ``e.snapshot()["history"]`` all spell the name out. Watching the name
    itself costs two entries in this package and closes those four spellings
    plus any other access that reaches the transcript by writing its name —
    which is not the same as closing every indirect access, only every one that
    says the word.
    """
    if not (isinstance(node.value, str) and node.value in _TRANSCRIPT_NAMES):
        return None
    parent = parents.get(node)
    if isinstance(parent, ast.Subscript) and isinstance(parent.ctx, (ast.Store, ast.Del)):
        return None  # snapshot["history"] = ... writes the key
    if isinstance(parent, ast.Call) and isinstance(parent.func, ast.Name) and parent.func.id in {"setattr", "delattr"}:
        return None
    return _Route.LITERAL_NAME


def direct_history_readers(source: str, module_key: str) -> dict[str, list[_Touch]]:
    """Every function in ``source`` that asks a question of the transcript.

    Returns ``{"<module_key>::<qualified name>": [touches]}``. Importable on
    its own so the guard can be aimed at source that is not on disk — which is
    how it gets tested against a reader nobody has written yet.
    """
    tree = ast.parse(source)
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    constants = _module_string_constants(tree)
    aliases = _module_type_aliases(tree)
    message_names = _message_names(tree)

    found: dict[str, list[_Touch]] = {}

    def record(node: ast.AST, route: _Route, lineno: int | None = None) -> None:
        key = f"{module_key}::{_enclosing_qualname(node, parents)}"
        found.setdefault(key, []).append(_Touch(lineno=lineno if lineno is not None else node.lineno, route=route))

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in _TRANSCRIPT_NAMES:
            route = _attribute_route(node, parents)
            if route is not None:
                record(node, route)
            continue
        if isinstance(node, ast.Call):
            route = _call_route(node, constants)
            if route is not None:
                record(node, route)
            continue
        if isinstance(node, ast.Constant):
            route = _constant_route(node, parents)
            if route is not None:
                record(node, route)
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            arguments = node.args
            named = [
                *arguments.posonlyargs,
                *arguments.args,
                *arguments.kwonlyargs,
            ]
            # ``*turns: Message`` and ``**turns: Message`` annotate ONE element
            # and hand over a container of them. The author who writes them has
            # typed the transcript MORE precisely than ``list[Message]``, so a
            # rule that demanded a container at the annotation site rewarded
            # precision with invisibility.
            variadic = [item for item in (arguments.vararg, arguments.kwarg) if item]
            for arg in (*named, *variadic):
                if arg.annotation is None:
                    continue
                reaches = (
                    _names_message(arg.annotation, message_names)
                    if arg in variadic
                    else _is_message_container(arg.annotation, aliases, message_names)
                )
                if not reaches:
                    continue
                if _asks_nothing_of(node, arg.arg):
                    continue  # written somewhere, or unused; nothing is asked
                record(node, _Route.PARAMETER, lineno=node.lineno)
                break
    return found


#: The smallest sweep that could still be THIS repository.
#:
#: A guard whose scope is computed while it runs can be disarmed by the SHAPE OF
#: THE CHECKOUT rather than by an edit to the rule, and the error runs BOTH
#: WAYS. Too few files makes "no reader violates the boundary" true for the
#: wrong reason: the assertion holds, the test is green, and the signal is
#: indistinguishable from compliance. Too many makes it false for the wrong
#: reason: another branch's code is judged as this branch's, and that failure is
#: the more expensive of the two, because it is repaired by editing this branch
#: to satisfy a complaint about a file that is not in it.
#:
#: Neither is hypothetical, and both were measured on this repository.
#:
#: * The dot-directory skip below used to read the ABSOLUTE path, so a checkout
#:   parked anywhere under a directory whose name begins with a dot swept ZERO
#:   of the package's ninety modules — 0 against 90 for the identical tree.
#:   Working trees are routinely made under such paths.
#: * The main checkout carried two worktrees under
#:   ``protocore/protocore/.worktrees/``, each a complete copy of the repository
#:   INSIDE the root being swept. A walk that skipped only ``__pycache__`` read
#:   516 files where the branch has 90; with build output in those trees the
#:   same walk was measured at 2540. They were removed by an unrelated piece of
#:   work partway through a single afternoon, which is the point: the scope of
#:   this guard changed by hundreds of files with nothing recording it, and
#:   nothing stopping the next one from appearing.
#:
#: So the sweep asks the INDEX what belongs to this branch, and the floor is
#: what remains for the case where there is no index to ask. A ceiling would be
#: the other half of a floor, but enumeration is the honest form of both: the
#: index knows what belongs to the repository and a walk does not.
#:
#: The floors sit well under the counts the repository actually has (90 package
#: modules, 119 test modules) so ordinary deletion does not trip them, and far
#: enough above zero that a collapsed sweep cannot be mistaken for a clean one.
_MIN_PACKAGE_MODULES = 60
_MIN_TEST_MODULES = 80


def _tracked_files(root: Path) -> list[Path] | None:
    """Every tracked ``*.py`` under ``root``, or ``None`` where there is no index.

    ``None`` and ``[]`` are different answers and the caller keeps them apart: a
    non-zero exit means there is no repository to ask — a source archive, a
    synthetic tree in a temporary directory — while a zero exit with no output
    means this branch tracks nothing there, which is a finding.

    The pathspec is ``<root>/*.py``: a git pathspec ``*`` already matches across
    ``/``, so the single star is the recursive form. ``<root>/**/*.py`` is the
    NARROWER pattern despite looking like the more thorough one — it requires an
    intervening directory and drops every file sitting directly in ``root``.
    Narrowing by BASENAME is left to the caller rather than written into the
    pathspec, because ``tests/test_*.py`` would anchor the prefix at ``tests/``
    and silently miss every nested ``tests/unit/.../test_x.py``.
    """
    try:
        spec = f"{root.relative_to(_REPO_ROOT).as_posix()}/*.py"
    except ValueError:
        return None  # not inside this repository; the index has nothing to say
    try:
        listing = subprocess.run(
            [
                "git",
                "-C",
                str(_REPO_ROOT),
                "ls-files",
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
                "--deduplicate",
                "--",
                spec,
            ],
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if listing.returncode != 0:
        return None
    return [_REPO_ROOT / name for name in listing.stdout.split("\0") if name]


def _walked_files(root: Path, pattern: str) -> list[Path]:
    """The fallback for a tree with no index, with the dot-directories dropped.

    The skip exists for ``.git``, ``.venv`` and the nested worktrees this
    repository keeps under ``.worktrees/`` — all of them ``.gitignore``d, none
    of them package source. It is a claim about where a path sits INSIDE
    ``root``, so it is asked of ``relative_to(root)``. The absolute prefix is a
    fact about the developer's disk; reading it made the whole sweep a fact
    about the disk too.

    What the skip cannot reach is a nested checkout parked under a NON-dot
    directory. Only the index closes that, so this path is the fallback and not
    the rule.
    """
    return [
        path
        for path in sorted(root.rglob(pattern))
        if not any(part.startswith(".") or part == "__pycache__" for part in path.relative_to(root).parts)
    ]


def _swept_files(root: Path, pattern: str, floor: int, what: str) -> list[Path]:
    """The files under ``root`` this branch owns, or an error if too few.

    Both halves of the failure above are answered here rather than in any one
    caller, so that rules written later inherit them.

    Membership comes from the index where there is one, which is what stops
    another branch's copy of the source from being read as this branch's. The
    floor is what makes an empty answer LOUD: a rule that asserts the ABSENCE of
    violations cannot itself tell an empty scope from a clean one, so the sweep
    refuses to hand back a scope too small to have been this repository.

    A tracked path deleted in the working tree is still in the index, so what
    cannot be read is dropped rather than raising.
    """
    tracked = _tracked_files(root)
    if tracked is not None:
        paths = sorted(path for path in tracked if fnmatch(path.name, pattern) and path.is_file())
    else:
        paths = _walked_files(root, pattern)
    if len(paths) < floor:
        raise AssertionError(
            f"the sweep for {what} found {len(paths)} file(s) under {root}, "
            f"fewer than the {floor} this repository must have. The scope these "
            "rules check is computed while they run, so a collapsed scope is "
            "otherwise reported as compliance: every 'nothing violates this' "
            "result in this file is void until the sweep is whole again."
        )
    return paths


def _package_sources() -> list[tuple[str, str]]:
    """``(registry module key, source)`` for every module in ``protocore/``.

    The sweep covers the WHOLE package, not just ``protocore/runtime/``: a
    violating reader dropped into ``protocore/tools/`` reaches the same engine
    and would have been invisible to a runtime-only sweep.

    The sweep ENUMERATES WHAT GIT TRACKS rather than walking the disk, so what
    it reads is a fact about this branch and not about the directory the branch
    happens to share. Walking made it the second thing: the main checkout held
    two worktrees under ``protocore/protocore/.worktrees/``, inside the root
    being swept, and a walk read 516 files where the branch has 90. A sibling
    repository lost 99 test results to precisely that shape.

    This is the one place in this file that shells out. The rest of it parses
    rather than imports, deliberately, so that a broken tree can still be
    reported on — and ``git ls-files`` does not weaken that: it is asked what
    this branch CONTAINS, never what it means. Where there is no index to ask,
    :func:`_swept_files` falls back to a filtered walk and the floor beside it
    is what remains.
    """
    return [
        (path.relative_to(_REPO_ROOT).as_posix(), path.read_text())
        for path in _swept_files(_PACKAGE_ROOT, "*.py", _MIN_PACKAGE_MODULES, "package modules")
    ]


def undeclared_history_readers(
    sources: Iterable[tuple[str, str]],
) -> dict[str, list[_Touch]]:
    """Readers in ``sources`` that reach the transcript without declaring a scope.

    THE enforcement. Both the sweep over the real tree and the check against a
    reader nobody has written go through this one function, so weakening it
    fails the tests that prove it works rather than quietly passing them.
    """
    undeclared: dict[str, list[_Touch]] = {}
    for module_key, source in sources:
        for key, touches in direct_history_readers(source, module_key).items():
            if key == _RUN_BOUNDARY_SOURCE or key in _WHOLE_HISTORY_BY_DESIGN:
                continue
            undeclared[key] = touches
    return undeclared


def _describe(undeclared: dict[str, list[_Touch]]) -> str:
    """Failure text that offers the remedy for the route actually taken."""
    lines: list[str] = ["these functions ask a question of the session transcript without declaring their scope:"]
    routes: set[_Route] = set()
    for key, touches in sorted(undeclared.items()):
        linenos = sorted({touch.lineno for touch in touches})
        kinds = sorted({touch.route.value for touch in touches})
        routes.update(touch.route for touch in touches)
        lines.append(f"  {key} (line{'s' if len(linenos) > 1 else ''} {linenos}) via {', '.join(kinds)}")
    for route in sorted(routes, key=lambda item: item.value):
        lines.append(f"[{route.value}] {_REMEDIES[route]}")
    lines.append(_MUTATIONS_ARE_NOT_FINDINGS)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The tests
# ---------------------------------------------------------------------------


def _plant(root: Path, count: int) -> Path:
    """A package-shaped tree of ``count`` modules, plus the noise a sweep skips."""
    package = root / "pkg"
    (package / "sub").mkdir(parents=True)
    for index in range(count):
        (package / f"mod_{index:03d}.py").write_text("x = 1\n")
    (package / ".worktrees" / "copy").mkdir(parents=True)
    (package / ".worktrees" / "copy" / "mod_000.py").write_text("x = 1\n")
    (package / "sub" / "__pycache__").mkdir()
    (package / "sub" / "__pycache__" / "mod_000.py").write_text("x = 1\n")
    return package


def test_the_sweep_does_not_depend_on_where_the_checkout_is_parked(
    tmp_path: Path,
) -> None:
    """The same tree sweeps the same, under a dot-directory or not.

    This is the defect that made every other rule in this file capable of
    passing over nothing. The skip is meant to drop ``.git``, ``.venv`` and the
    nested ``.worktrees/`` copies; asked of the ABSOLUTE path it also drops the
    entire tree whenever some ancestor of the checkout happens to start with a
    dot. Tooling routinely parks worktrees under a dotted home directory, so
    this was the common case, not the exotic one — measured at 0 modules
    swept against 90.
    """
    plain = _plant(tmp_path / "plain", 12)
    dotted = _plant(tmp_path / ".dotted" / "nested", 12)

    swept_plain = _swept_files(plain, "*.py", 12, "planted modules")
    swept_dotted = _swept_files(dotted, "*.py", 12, "planted modules")

    assert [path.relative_to(plain) for path in swept_plain] == [path.relative_to(dotted) for path in swept_dotted]
    # The skip still does its job on both: neither the nested worktree copy nor
    # the bytecode cache is package source.
    assert len(swept_plain) == 12


def test_the_sweep_refuses_to_report_a_scope_that_collapsed(tmp_path: Path) -> None:
    """An empty scope raises instead of being handed on as a clean tree.

    Without this, ``assert not undeclared`` over an empty sweep is a pass, and a
    pass is what the disarmed guard reported. The floor is the difference
    between "this repository holds no violation" and "this run looked at
    nothing".
    """
    empty = tmp_path / "nothing"
    empty.mkdir()
    with pytest.raises(AssertionError, match="found 0 file"):
        _swept_files(empty, "*.py", 1, "planted modules")

    thin = _plant(tmp_path / "thin", 3)
    with pytest.raises(AssertionError, match="found 3 file"):
        _swept_files(thin, "*.py", 12, "planted modules")


def test_the_real_sweeps_clear_their_floors() -> None:
    """The floors are met by the repository as it stands, with room to spare.

    A floor set at or above the live count would fail on the next deletion; one
    set at zero would not fail at all. This states both counts so the margin is
    visible rather than assumed.
    """
    modules = _swept_files(_PACKAGE_ROOT, "*.py", _MIN_PACKAGE_MODULES, "package modules")
    tests = _swept_files(_TESTS_ROOT, "test_*.py", _MIN_TEST_MODULES, "test modules")
    assert len(modules) >= _MIN_PACKAGE_MODULES
    assert len(tests) >= _MIN_TEST_MODULES
    assert _MIN_PACKAGE_MODULES > 0 and _MIN_TEST_MODULES > 0


def test_the_sweep_is_the_index_and_not_the_directory() -> None:
    """What is judged is this branch's code, not what shares its directory.

    The other half of the floor, and the more expensive half. A sweep that
    collapsed reports a clean tree; a sweep that read a nested checkout reports
    a violation in a file this branch does not contain, and the natural repair
    is to edit this branch until the complaint about someone else's file goes
    away.

    Not hypothetical, and not old: the main checkout of this repository held two
    worktrees under ``protocore/protocore/.worktrees/`` — complete copies of the
    repository inside the package root — and a walk that skipped only
    ``__pycache__`` read 516 files against this branch's 90. They were removed
    partway through one afternoon by work that had nothing to do with this
    guard, so the scope moved by hundreds of files with nothing recording it.
    """
    for root, pattern, floor, what in (
        (_PACKAGE_ROOT, "*.py", _MIN_PACKAGE_MODULES, "package modules"),
        (_TESTS_ROOT, "test_*.py", _MIN_TEST_MODULES, "test modules"),
    ):
        tracked = _tracked_files(root)
        assert tracked is not None, "the suite runs in a checkout; git must answer"

        swept = _swept_files(root, pattern, floor, what)
        assert len({path.resolve() for path in swept}) == len(swept), (
            f"a {what} file was swept twice, which is what a walk into a nested checkout looks like"
        )
        on_index = {path.resolve() for path in tracked}
        for path in swept:
            assert path.resolve() in on_index, f"{path} is swept but not tracked"


def test_every_direct_history_reader_is_declared() -> None:
    """No unsanctioned reader asks a question of the raw session transcript.

    A new helper that asks a run-scoped question of ``engine.history`` lands
    here by name. The fix is not to add it to the registry: it is to take its
    messages from ``_this_run_messages``.
    """
    undeclared = undeclared_history_readers(_package_sources())
    assert not undeclared, _describe(undeclared)


def test_the_registry_has_no_stale_entries() -> None:
    """Every declared reader still exists and still reaches the transcript.

    A registry that outlives what it describes silently pre-authorises a name:
    delete a whole-transcript helper, write a run-scoped one under the same
    name, and the guard waves it through.
    """
    live: set[str] = set()
    for module_key, source in _package_sources():
        live.update(direct_history_readers(source, module_key))
    declared = set(_WHOLE_HISTORY_BY_DESIGN) | {_RUN_BOUNDARY_SOURCE}
    assert not (declared - live), (
        f"these registry entries no longer reach the transcript and must be removed: {sorted(declared - live)}"
    )


#: The run boundary is one rule, and it is stated in
#: ``query.py::_this_run_messages``. Two functions restate it, and both are
#: declared here rather than left to be found: compaction needs the boundary as
#: INDICES into the list it is handed, which a filtered copy cannot express, and
#: the session-memory taggers are what put the tag on in the first place.
_SEED_KEY_DERIVED_ELSEWHERE: dict[str, str] = {
    "protocore/contracts/types.py::<module>": (
        "the definition itself; this is where the key's name and its value "
        "come from. NOTE that authorising this scope is what makes residual 2 "
        "of seed_key_derivations reachable: an alias bound HERE from a value "
        "computed off the key does not enter the vocabulary, and the line "
        "computing it is not a finding because this scope is declared"
    ),
    "protocore/runtime/context/compaction.py::_first_user_turn_index": (
        "compaction protects the new task by INDEX into the list it is given; "
        "a filtered copy has no indices into the original"
    ),
    "protocore/runtime/context/compaction.py::_session_history_seed_indices": (
        "compaction withholds the lossy Tier-2 collapse from seeded turns by INDEX into the list it is given"
    ),
    "protocore/runtime/context/compaction.py::_is_plain_operator_turn": (
        "classifies ONE message it is handed: a turn seeded from an earlier "
        "run of the session is not an operator turn of this one, so neither "
        "Tier 2 nor the fold may treat it as one. The callers work by INDEX "
        "into the whole list, which a filtered copy cannot express"
    ),
    "protocore/runtime/context/compaction.py::run_tier2_summarisation": (
        "reactive compaction preserves seed provenance while replacing units "
        "by INDEX into the whole list"
    ),
    "protocore/runtime/context/compaction.py::_plan_tier2": (
        "a unit that mixes seeded and current turns is left intact, decided "
        "by INDEX into the whole list before any call is made"
    ),
    "protocore/runtime/context/compaction.py::_foldable_indices.is_foldable": (
        "reactive folding classifies each indexed message without losing its "
        "seed provenance"
    ),
    "protocore/runtime/context/compaction.py::_fold_spans": (
        "reactive folding splits indexed spans at seed/current boundaries"
    ),
    "protocore/runtime/context/compaction.py::_fold_span": (
        "a seed-only replacement inherits the provenance of the indexed span"
    ),
    "protocore/runtime/context/compaction.py::_fold_item_text": (
        "labels one indexed message for the fold summariser by its provenance"
    ),
    "protocore/runtime/context/compaction.py::run_tier3_fold": (
        "reactive folding transfers seed provenance from each indexed span"
    ),
    "protocore/runtime/context/session_memory.py::_tag_seeded": (
        "writes the tag; this is where the boundary comes from"
    ),
    "protocore/runtime/context/session_memory.py::_tag_seeded_reference": (
        "writes the tag on a dual-tagged reference block"
    ),
}


#: The constant's own name. Written once, read from the import above for its
#: value, so neither the name nor the value can drift from what is watched.
_SEED_KEY_NAME = "SESSION_HISTORY_SEED_METADATA_KEY"


def _imported_module_keys(node: ast.ImportFrom, module_key: str) -> tuple[str, ...]:
    """Registry keys a ``from … import`` could be reading from.

    ``from protocore.runtime.query import _SEED`` inside
    ``protocore/runtime/other.py`` resolves to ``protocore/runtime/query.py``,
    and so does ``from .query import _SEED``. Without this a re-export is a
    second module's problem and neither module looks guilty.

    BOTH candidates are returned, because a dotted path names a module or a
    PACKAGE and the two are keyed differently: ``protocore.runtime`` is
    ``protocore/runtime/__init__.py``, not ``protocore/runtime.py``. Appending
    only ``.py`` missed every package-level re-export — and re-exporting through
    ``__init__.py`` is the house style here, so that was not an exotic gap: all
    11 packages in this tree have one. Measured before the fix, with the whole
    suite green, ``ruff`` and ``mypy`` clean: an alias re-exported from
    ``protocore/safety/__init__.py`` and read in a registry-authorised function
    was invisible, absolute and relative alike, at any nesting depth.

    Returning both rather than probing the key set keeps this a pure function of
    the node: the caller unions whichever candidates the sweep actually holds,
    so a fragment checked on its own behaves the same way as the whole package.
    """
    if node.level:
        parts = module_key.split("/")[:-1]  # the importing module's package
        if node.level > 1:
            parts = parts[: -(node.level - 1)]
        parts = [*parts, *(node.module.split(".") if node.module else [])]
    elif node.module:
        parts = node.module.split(".")
    else:
        return ()
    if not parts:
        return ()
    dotted = "/".join(parts)
    return (f"{dotted}.py", f"{dotted}/__init__.py")


def _namespace_statements(body: list[ast.stmt], in_class: bool = False) -> list[tuple[ast.stmt, bool]]:
    """Statements binding a name in a MODULE or CLASS namespace, with which.

    A binding inside a function needs no resolution: the right-hand side names
    the key in the same function, so the function is already reported. What
    needs resolving is a name that outlives the statement binding it — module
    level, including under ``if TYPE_CHECKING:``, in a ``try``/``except`` import
    fallback or in a ``match`` arm, and a class body, which binds an ATTRIBUTE.

    Missing one of those compound forms does not hide the binding — the line
    still names the key and the MODULE is still reported. It costs the function
    name, which is the part the author actually needs, so the list is kept
    complete rather than left to the module-level fallback.
    """
    found: list[tuple[ast.stmt, bool]] = []
    for stmt in body:
        found.append((stmt, in_class))
        if isinstance(stmt, ast.ClassDef):
            found.extend(_namespace_statements(stmt.body, True))
        elif isinstance(stmt, ast.Match):
            found.extend(_namespace_statements([inner for case in stmt.cases for inner in case.body], in_class))
        elif isinstance(stmt, (ast.If, ast.Try, ast.With)):
            nested = [
                *stmt.body,
                *getattr(stmt, "orelse", []),
                *getattr(stmt, "finalbody", []),
                *[inner for handler in getattr(stmt, "handlers", []) for inner in handler.body],
            ]
            found.extend(_namespace_statements(nested, in_class))
    return found


def _spells_a_key_name(node: ast.AST, attributes: set[str], spelled: set[str]) -> bool:
    """Whether ``node`` evaluates to a STRING that names the key.

    Not the key's value — the identifier it is reachable under, written out as
    text: ``"SESSION_HISTORY_SEED_METADATA_KEY"``, or the name of any alias of
    it, since ``getattr(types, "_SEED_TAG")`` reaches the constant exactly as
    surely as ``getattr(types, "SESSION_…")`` does. ``attributes`` is therefore
    the set to match against, not ``_SEED_KEY_NAME`` alone.

    A name-string does not have to be written where it is used. ``spelled``
    holds the identifiers this package binds to such a string, so a literal
    hoisted to a module constant, parked in a local, or dropped into a container
    and indexed back out is followed to the lookup that consumes it.
    """
    if isinstance(node, ast.Constant):
        return isinstance(node.value, str) and node.value in attributes
    if isinstance(node, ast.Name):
        return node.id in spelled
    if isinstance(node, ast.Attribute):
        return node.attr in spelled
    if isinstance(node, ast.Subscript):
        return _spells_a_key_name(node.value, attributes, spelled)
    return False


def _reads_the_key_by_name(node: ast.AST, attributes: set[str], spelled: set[str]) -> bool:
    """Whether ``node`` is a LOOKUP that reaches the key through its name.

    ``getattr(types, "SESSION_HISTORY_SEED_METADATA_KEY")`` and
    ``vars(types)["SESSION_…"]`` reach the constant without writing a name the
    other rules can see. Restricted to a lookup — a call argument, a keyword
    argument, or a subscript key — on purpose: this package writes the
    constant's name in six docstrings, and a rule that watched the string
    anywhere would report a paragraph of prose as a re-derivation of the run
    boundary.

    What is NOT restricted any more is the vocabulary. This was the one rule
    still matching an INLINE literal against a single hard-coded spelling while
    the other two resolved a name to what it refers to, and the gap was
    landable: hoisting the literal to ``_SEED_TAG_ATTR = "SESSION_…"`` and
    calling ``getattr(_contract_types, _SEED_TAG_ATTR)`` was measured passing —
    one file, no cooperating edit, ``ruff`` / ``mypy`` / the full suite green —
    which is the definitive tidy-up refactor and so exactly the standard the
    rest of this checker is written to. Both halves are widened: the string may
    name any alias (see :func:`_spells_a_key_name`), and it may be bound before
    the lookup rather than written in it.
    """
    if isinstance(node, ast.Call):
        return any(
            _spells_a_key_name(argument, attributes, spelled)
            for argument in (
                *node.args,
                *(keyword.value for keyword in node.keywords),
            )
        )
    if isinstance(node, ast.Subscript):
        return _spells_a_key_name(node.slice, attributes, spelled)
    return False


def _binds_a_key_name(value: ast.expr, attributes: set[str], spelled: set[str]) -> bool:
    """Whether ``value`` — a binding's right-hand side — IS a name of the key.

    The string, or a container literal holding it: ``_ATTRS = ["SESSION_…"]``
    followed by ``getattr(types, _ATTRS[0])`` is the same lookup with one more
    hop, and a container is where a name-string goes when there is more than
    one of them.
    """
    if _spells_a_key_name(value, attributes, spelled):
        return True
    if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
        return any(_spells_a_key_name(element, attributes, spelled) for element in value.elts)
    return False


def _binds_the_key(value: ast.expr, names: set[str], attributes: set[str], spelled: set[str]) -> bool:
    """Whether ``value`` — a binding's right-hand side — IS the key.

    The value itself, not something computed from it: ``_TAG = _SEED`` binds the
    key, ``_FLAG = m.metadata.get(_SEED)`` binds a boolean. Widening this to
    "mentions the key anywhere" would enter every such boolean into the
    vocabulary and report unrelated uses of its name as re-derivations. The
    residual that narrowness leaves is stated in :func:`seed_key_derivations`
    and it is real: a value COMPUTED from the key, even when the computation is
    the identity, is not followed.
    """
    if isinstance(value, ast.Name):
        return value.id in names
    if isinstance(value, ast.Attribute):
        return value.attr in attributes
    if isinstance(value, ast.Constant):
        return value.value == SESSION_HISTORY_SEED_METADATA_KEY
    return _reads_the_key_by_name(value, attributes, spelled)


#: How many passes :func:`_seed_key_vocabulary` may take before it gives up.
#: One pass resolves one link of an alias / re-export chain read in the worst
#: order, and one further pass finds nothing and thereby proves the vocabulary
#: settled — so the deepest chain followed is ``_VOCABULARY_PASSES - 1`` links.
#: Measured against synthetic sources ordered so that every module is read
#: BEFORE the one it imports from: seven hops resolve, eight raise. The real
#: tree settles in ONE pass (no alias of the key exists in it today), so this
#: budget is slack rather than cost.
#:
#: Exhausting it RAISES rather than returning a partial vocabulary — see the
#: docstring for why silence is the wrong failure here.
_VOCABULARY_PASSES = 8


def _seed_key_vocabulary(
    trees: dict[str, ast.Module],
) -> tuple[dict[str, set[str]], set[str], set[str]]:
    """Every name the key is reachable by: as a name, as an attribute, as a string.

    Three sets, because they are not interchangeable.

    * ``names`` is PER MODULE. A bare ``ast.Name`` is resolved in the module
      that binds it — a local called ``_seed`` in an unrelated module means
      nothing here.
    * ``attributes`` is a union. ``types.SESSION_…``, ``query._SEED_TAG`` and
      ``_Keys.SEED`` are all a name this package binds to the key, read off
      whichever object happens to carry it, and this checker does not track
      objects.
    * ``spelled`` is a union too: the identifiers bound to a STRING that names
      the key, so a name-string parked in a constant before it is looked up is
      followed. Union rather than per-module for the same reason ``attributes``
      is — the checker cannot tell which object a name-string will be used
      against. That is deliberately over-broad: a module binding
      ``X = "SESSION_HISTORY_SEED_METADATA_KEY"`` makes ``getattr(o, X)`` a
      finding everywhere, not just in that module. No such binding exists in
      this package (the name appears only inside docstring prose), and the
      alternative — tracking which module's ``X`` a lookup meant — is the object
      tracking this checker deliberately does not attempt.

    Iterated to a fixed point so an alias of an alias, and a re-export of a
    re-export, resolve — and the iteration is bounded, because a bound is the
    only thing that makes "fixed point" safe to say about input this checker
    does not control. ``_VOCABULARY_PASSES`` passes are enough for a chain of
    that many links even when every module is read BEFORE the one it imports
    from, which is the worst order and a real one: ``_package_sources`` sorts by
    path, so a consumer routinely sorts ahead of its source.

    Running out is a HARD FAILURE, not a truncation. The loop used to stop at
    the cap and return whatever it had, so a chain one link too long produced an
    incomplete vocabulary and a clean report — a guard that silently answers "no
    findings" when it has run out of budget is worse than one that has a
    documented limit, because nothing distinguishes its silence from a pass.
    """
    names = {module_key: {_SEED_KEY_NAME} for module_key in trees}
    attributes = {_SEED_KEY_NAME}
    spelled: set[str] = set()

    def measure() -> tuple[int, int, int]:
        return (
            sum(len(bound) for bound in names.values()),
            len(attributes),
            len(spelled),
        )

    for _ in range(_VOCABULARY_PASSES):
        size = measure()
        for module_key, tree in trees.items():
            local = names[module_key]
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    exported: set[str] = set()
                    for candidate in _imported_module_keys(node, module_key):
                        exported |= names.get(candidate, set())
                    for alias in node.names:
                        if alias.name == _SEED_KEY_NAME or alias.name in exported:
                            local.add(alias.asname or alias.name)
                    continue
                # A name-string may be bound anywhere, including inside the
                # very function that looks it up, so this walk is not limited
                # to namespace statements the way the value binding below is.
                if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                    continue
                if node.value is None or not _binds_a_key_name(node.value, attributes, spelled):
                    continue
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                spelled.update(target.id for target in targets if isinstance(target, ast.Name))
            for stmt, in_class in _namespace_statements(tree.body):
                value = stmt.value if isinstance(stmt, (ast.Assign, ast.AnnAssign)) else None
                if value is None or not _binds_the_key(value, local, attributes, spelled):
                    continue
                targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
                bound = {target.id for target in targets if isinstance(target, ast.Name)}
                attributes.update(bound)
                if not in_class:
                    local.update(bound)
            attributes.update(local)
        if size == measure():
            break
    else:
        raise AssertionError(
            "the seed-key vocabulary did not settle in "
            f"{_VOCABULARY_PASSES} passes, so the names below are only the ones "
            "found so far and this check cannot report honestly. An alias or "
            "re-export chain longer than that many links is the cause; raise "
            "_VOCABULARY_PASSES or shorten the chain. Found so far: "
            f"attributes={sorted(attributes)} spelled={sorted(spelled)}"
        )
    return names, attributes, spelled


def seed_key_derivations(
    sources: Iterable[tuple[str, str]],
) -> dict[str, list[int]]:
    """Every function that names the run boundary, however it spells it.

    The key is matched by WHAT IT IS, not by how it is written, and "what it is"
    had to be widened twice before that sentence was true. Every spelling below
    was measured re-deriving the boundary inside a function the registry already
    authorises, computing into an unused local so no behavioural test could mask
    it, with the full suite green and nothing failing:

    * the imported name ``SESSION_HISTORY_SEED_METADATA_KEY``;
    * an IMPORT alias — ``from … import SESSION_…_KEY as _SEED_TAG``. A
      ``from … import`` binds an :class:`ast.alias`, which is not an
      :class:`ast.Name`, and no rule in this file read one. It is the most
      idiomatic of the spellings and it was the last one open;
    * a RE-EXPORT of such an alias out of another module in the package —
      absolute or relative, from a plain module OR from a package
      ``__init__.py``, up to ``_VOCABULARY_PASSES`` links deep. The
      ``__init__.py`` half was open until it was measured: a dotted path names a
      module or a package and the resolver appended only ``.py``, so every
      package-level re-export missed, which is the house style here;
    * MODULE-QUALIFIED access — ``types.SESSION_…_KEY`` behind
      ``from protocore.contracts import types``, and a class-body alias read as
      ``_Keys.SEED``. Attributes were never inspected, though the parameter rule
      in this same file has always read them;
    * any module-level alias, transitively, including an alias OF an import
      alias and one bound under ``if TYPE_CHECKING:``;
    * a NAME OF the key spelled as a string in a dynamic lookup —
      ``getattr(types, "SESSION_…")``, and equally ``getattr(types,
      "_SEED_TAG")`` for any alias, since an alias's name reaches the constant
      just as surely as the constant's own does. The string may be written in
      the lookup or bound before it: hoisted to a module constant, parked in a
      local, passed as a keyword argument, or dropped in a container literal and
      indexed back out. Binding it is reported too, so the function that
      prepares the string is named even when another one consumes it;
    * the key's own string VALUE, taken from
      :data:`~protocore.contracts.types.SESSION_HISTORY_SEED_METADATA_KEY` at
      import time rather than copied here, so this rule cannot drift from the
      constant it watches.

    **What still gets past.** Derived from what the rules above actually do, and
    every entry measured against this checker rather than reasoned about. The
    test of "gets past" used here is that the checker reports NOTHING anywhere —
    a route that names some other function still fails the guard and still
    blocks the change, so it is not a hole:

    1. A value ASSEMBLED piecewise, so that neither a name of the key nor its
       value is ever written whole: ``"protocore.session" + "_history_seed"``,
       ``".".join((…))``, an f-string whose placeholder splits the literal.
       Following those means evaluating arbitrary expressions, which a static
       rule does not do.
    2. **An alias created inside a module whose scope is already declared** —
       which in practice means ``contracts/types.py``, the definition site — by
       any binding form :func:`_binds_the_key` and :func:`_namespace_statements`
       do not model between them. The line creating the alias IS reported, so
       everywhere else this is a finding; there it is authorised, and the alias
       then reads clean as ``types._SEED_TAG`` from anywhere in the package.
       Measured open: a right-hand side COMPUTED from the key even when the
       computation is the identity (``f"{SESSION_…}"``, ``str(…)``, ``…[:]``,
       ``_TAG = ""`` then ``_TAG += SESSION_…``); a walrus (``if (_TAG :=
       SESSION_…)``); a tuple unpack (``(_TAG,) = (SESSION_…,)``); a container
       literal read back by key (``_KEYS = {"seed": SESSION_…}``); a
       ``Literal[…]`` type alias unpacked with ``get_args``. Closing the
       provable subset one form at a time is chasing spellings again — the
       CAUSE is that a whole module scope is authorised when only the
       definition needs to be, and narrowing that authorisation is the fix not
       made here.
    3. A ``from … import *`` of a re-exported alias. The resolver reads
       ``alias.name``, which is ``"*"``, so nothing enters the vocabulary. This
       is NOT claimed as covered — it is not landable, which is a different
       thing: ``ruff`` rejects it with ``F403``/``F405`` and the lint gate runs
       on every change. It stops being blocked the moment those rules are
       silenced.

    Routes that are NOT residuals, checked rather than assumed, because each
    looks like one: a helper function that RETURNS the key, or that carries it
    as a default argument value, names the key in the helper and the helper is
    reported; an instance attribute assigned in a class ``__init__`` reports
    that method, which is not a declared scope even inside ``contracts/types.py``;
    an alias bound in a ``match`` arm resolves like any other namespace binding.

    Items 1 and 2 are also not edits an author reaches for by accident, which is
    the standard the caught list is written to: every spelling above is what
    someone writes while tidying up, not while hiding.

    ``<module>`` is no longer skipped wholesale. The skip was there for "the
    import statement itself", but an ``ast.alias`` was never an ``ast.Name``, so
    it was skipping nothing it needed to — while hiding every module-level
    alias. The definition site in ``contracts/types.py`` is declared instead.

    Importable on its own so the rule can be aimed at source that is not on
    disk, the same way the reader check is. Aliases resolve ACROSS the sources
    handed in, so a re-export is caught when the package is swept whole and a
    single fragment is still read on its own terms.
    """
    trees = {module_key: ast.parse(source) for module_key, source in sources}
    names, attributes, spelled = _seed_key_vocabulary(trees)

    derivations: dict[str, list[int]] = {}
    for module_key, tree in trees.items():
        parents: dict[ast.AST, ast.AST] = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parents[child] = node

        local = names[module_key]
        for node in ast.walk(tree):
            derives = (
                (isinstance(node, ast.Name) and node.id in local)
                or (isinstance(node, ast.Attribute) and node.attr in attributes)
                or (isinstance(node, ast.Constant) and node.value == SESSION_HISTORY_SEED_METADATA_KEY)
                or _reads_the_key_by_name(node, attributes, spelled)
                # Writing the key's NAME down is naming the run boundary even
                # before anything is looked up with it. Reported at the binding
                # too, so the function that prepares the string is named and not
                # only the one that consumes it — they need not be the same
                # function, or the same module. A binding is an ``ast.Assign`` /
                # ``ast.AnnAssign``, never a bare expression statement, so this
                # cannot reach the six docstrings that write the name in prose.
                or (
                    isinstance(node, (ast.Assign, ast.AnnAssign))
                    and node.value is not None
                    and _binds_a_key_name(node.value, attributes, spelled)
                )
            )
            if not derives:
                continue
            qualname = _enclosing_qualname(node, parents)
            derivations.setdefault(f"{module_key}::{qualname}", []).append(node.lineno)
    return derivations


def test_the_run_boundary_is_derived_only_where_declared() -> None:
    """The seed tag is named in ``_this_run_messages`` and the declared places.

    Re-deriving the rule at a call site is how the readers this guard exists
    for came to disagree: five of them skipped seeded turns, four did not, and
    every one of the four had been written by someone reading the module. The
    sweep covers the whole package rather than ``query.py`` alone — a
    re-derivation dropped into ``context/compaction.py`` was invisible while
    this test read one module.
    """
    derivations = seed_key_derivations(_package_sources())
    allowed = {_RUN_BOUNDARY_SOURCE} | set(_SEED_KEY_DERIVED_ELSEWHERE)
    assert set(derivations) <= allowed, (
        "the run boundary must be derived only in _this_run_messages, or in a "
        "place declared in _SEED_KEY_DERIVED_ELSEWHERE with the reason a "
        "filtered sequence cannot serve it; these re-derive it: "
        f"{sorted(set(derivations) - allowed)}"
    )
    assert set(_SEED_KEY_DERIVED_ELSEWHERE) <= set(derivations), (
        "these declared re-derivations no longer name the seed key and must "
        f"be removed: {sorted(set(_SEED_KEY_DERIVED_ELSEWHERE) - set(derivations))}"
    )


#: Spellings of the seed key that all reach the same constant. Every one is a
#: re-derivation of the run boundary and every one has to be seen as such.
#:
#: Every case here was measured PASSING before the rule matched it: dropped into
#: a function the registry already authorises, computing into an unused local so
#: no behavioural test could mask it, the full suite stayed green with nothing
#: failing. A module-level alias and the raw string value were found first and
#: closed AS spellings; an import alias and module-qualified access were sitting
#: behind them, untouched, because closing a spelling is not closing a route.
#: The rule now asks what a name REFERS to, so this list is a record of what was
#: measured rather than the definition of what is caught.
_SEED_KEY_SPELLINGS: list[tuple[str, str]] = [
    (
        "module_level_alias",
        """
_SEED = SESSION_HISTORY_SEED_METADATA_KEY


def _prior_run_turns(history: list[Message]) -> list[Message]:
    return [m for m in history if m.metadata.get(_SEED) is True]
""",
    ),
    (
        "chained_module_level_alias",
        """
_SEED = SESSION_HISTORY_SEED_METADATA_KEY
_TAG = _SEED


def _prior_run_turns(history: list[Message]) -> list[Message]:
    return [m for m in history if m.metadata.get(_TAG) is True]
""",
    ),
    (
        "the_keys_raw_string_value",
        f'''
def _prior_run_turns(history: list[Message]) -> list[Message]:
    return [
        m
        for m in history
        if m.metadata.get("{SESSION_HISTORY_SEED_METADATA_KEY}") is True
    ]
''',
    ),
    (
        "a_module_level_constant_holding_the_value",
        f'''
_SEED = "{SESSION_HISTORY_SEED_METADATA_KEY}"


def _prior_run_turns(history: list[Message]) -> list[Message]:
    return [m for m in history if m.metadata.get(_SEED) is True]
''',
    ),
    (
        "the_imported_name",
        """
def _prior_run_turns(history: list[Message]) -> list[Message]:
    return [
        m
        for m in history
        if m.metadata.get(SESSION_HISTORY_SEED_METADATA_KEY) is True
    ]
""",
    ),
    (
        "an_alias_bound_inside_the_function",
        """
def _prior_run_turns(history: list[Message]) -> list[Message]:
    seed = SESSION_HISTORY_SEED_METADATA_KEY
    return [m for m in history if m.metadata.get(seed) is True]
""",
    ),
    (
        "an_import_alias",
        """
from protocore.contracts.types import SESSION_HISTORY_SEED_METADATA_KEY as _SEED_TAG


def _prior_run_turns(history: list[Message]) -> list[Message]:
    return [m for m in history if m.metadata.get(_SEED_TAG) is True]
""",
    ),
    (
        "an_alias_of_an_import_alias",
        """
from protocore.contracts.types import SESSION_HISTORY_SEED_METADATA_KEY as _SEED_TAG

_TAG = _SEED_TAG


def _prior_run_turns(history: list[Message]) -> list[Message]:
    return [m for m in history if m.metadata.get(_TAG) is True]
""",
    ),
    (
        "an_import_alias_bound_only_for_type_checking",
        """
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from protocore.contracts.types import SESSION_HISTORY_SEED_METADATA_KEY as _SEED_TAG


def _prior_run_turns(history: list[Message]) -> list[Message]:
    return [m for m in history if m.metadata.get(_SEED_TAG) is True]
""",
    ),
    (
        "module_qualified_access",
        """
from protocore.contracts import types


def _prior_run_turns(history: list[Message]) -> list[Message]:
    return [
        m
        for m in history
        if m.metadata.get(types.SESSION_HISTORY_SEED_METADATA_KEY) is True
    ]
""",
    ),
    (
        "a_class_body_alias_read_as_an_attribute",
        """
from protocore.contracts.types import SESSION_HISTORY_SEED_METADATA_KEY


class _Keys:
    SEED = SESSION_HISTORY_SEED_METADATA_KEY


def _prior_run_turns(history: list[Message]) -> list[Message]:
    return [m for m in history if m.metadata.get(_Keys.SEED) is True]
""",
    ),
    (
        "a_name_assembled_by_getattr_at_import_time",
        """
from protocore.contracts import types

_TAG = getattr(types, "SESSION_HISTORY_SEED_METADATA_KEY")


def _prior_run_turns(history: list[Message]) -> list[Message]:
    return [m for m in history if m.metadata.get(_TAG) is True]
""",
    ),
    # The four below all write the constant's name VERBATIM and bind it before
    # the lookup. The rule used to match an inline literal only — it was the one
    # rule never handed the vocabulary the other two got — so extracting that
    # literal to a named constant, the definitive tidy-up edit, reopened the
    # route. Measured landing clean in one file before it was closed.
    (
        "a_name_string_hoisted_to_a_module_constant",
        """
from protocore.contracts import types

_SEED_TAG_ATTR = "SESSION_HISTORY_SEED_METADATA_KEY"


def _prior_run_turns(history: list[Message]) -> list[Message]:
    tag = getattr(types, _SEED_TAG_ATTR)
    return [m for m in history if m.metadata.get(tag) is True]
""",
    ),
    (
        "a_name_string_bound_as_a_local",
        """
from protocore.contracts import types


def _prior_run_turns(history: list[Message]) -> list[Message]:
    attribute = "SESSION_HISTORY_SEED_METADATA_KEY"
    tag = getattr(types, attribute)
    return [m for m in history if m.metadata.get(tag) is True]
""",
    ),
    (
        "a_name_string_passed_as_a_keyword_argument",
        """
from protocore.contracts import types


def _prior_run_turns(history: list[Message]) -> list[Message]:
    tag = getattr(types, name="SESSION_HISTORY_SEED_METADATA_KEY")
    return [m for m in history if m.metadata.get(tag) is True]
""",
    ),
    (
        "a_name_string_in_a_container_literal",
        """
from protocore.contracts import types

_ATTRIBUTES = ["SESSION_HISTORY_SEED_METADATA_KEY"]


def _prior_run_turns(history: list[Message]) -> list[Message]:
    tag = getattr(types, _ATTRIBUTES[0])
    return [m for m in history if m.metadata.get(tag) is True]
""",
    ),
    (
        # ``match`` is a namespace-binding compound statement like ``if`` and
        # ``try``, and it was the one missing from the list. Missing it does not
        # hide the module — the arm names the key — but it costs the FUNCTION
        # name, which is the part the author is sent to.
        "an_alias_bound_in_a_match_arm",
        """
from protocore.contracts.types import SESSION_HISTORY_SEED_METADATA_KEY

match "structured":
    case "structured":
        _SEED = SESSION_HISTORY_SEED_METADATA_KEY
    case _:
        _SEED = SESSION_HISTORY_SEED_METADATA_KEY


def _prior_run_turns(history: list[Message]) -> list[Message]:
    return [m for m in history if m.metadata.get(_SEED) is True]
""",
    ),
]

#: The same key, reached through a name another module in the package binds.
#: A re-export is two behaviour-preserving lines in two different files, and
#: neither file looks guilty read on its own — which is why the vocabulary is
#: resolved across the whole sweep rather than per module.
_SEED_KEY_RE_EXPORTS: list[tuple[str, list[tuple[str, str]]]] = [
    (
        "an_absolute_re_export",
        [
            (
                "protocore/runtime/query.py",
                "from protocore.contracts.types import SESSION_HISTORY_SEED_METADATA_KEY as _SEED_TAG\n",
            ),
            (
                "protocore/runtime/steering.py",
                """
from protocore.runtime.query import _SEED_TAG


def _prior_run_turns(history: list[Message]) -> list[Message]:
    return [m for m in history if m.metadata.get(_SEED_TAG) is True]
""",
            ),
        ],
    ),
    (
        # The consumer is handed over FIRST, which is the order the real sweep
        # produces: it sorts by path, and plenty of module names sort before the
        # one they import from. A single resolution pass would read this module
        # before anything had bound ``_SEED_TAG``, and see nothing.
        "a_relative_re_export_renamed_again_read_before_its_source",
        [
            (
                "protocore/runtime/steering.py",
                """
from .query import _SEED_TAG as _T


def _prior_run_turns(history: list[Message]) -> list[Message]:
    return [m for m in history if m.metadata.get(_T) is True]
""",
            ),
            (
                "protocore/runtime/query.py",
                "from protocore.contracts.types import SESSION_HISTORY_SEED_METADATA_KEY as _SEED_TAG\n",
            ),
        ],
    ),
    (
        # A dotted path names a module or a PACKAGE, and the two are keyed
        # differently — ``protocore.safety`` is ``protocore/safety/__init__.py``.
        # The resolver appended only ``.py``, so this whole shape missed, and it
        # is not an exotic one: every package in this tree re-exports through its
        # ``__init__.py``. Landed clean against the real tree before it was
        # closed — ruff, mypy and the full suite all green.
        "a_re_export_from_a_package_init",
        [
            (
                "protocore/safety/__init__.py",
                "from protocore.contracts.types import "
                "SESSION_HISTORY_SEED_METADATA_KEY as _SEED_TAG\n"
                '\n__all__ = ["_SEED_TAG"]\n',
            ),
            (
                "protocore/runtime/steering.py",
                """
from protocore.safety import _SEED_TAG


def _prior_run_turns(history: list[Message]) -> list[Message]:
    return [m for m in history if m.metadata.get(_SEED_TAG) is True]
""",
            ),
        ],
    ),
    (
        # The same, reached relatively and read before its source, so it needs
        # both the ``__init__.py`` candidate AND a second resolution pass.
        "a_relative_re_export_from_a_package_init_read_before_its_source",
        [
            (
                "protocore/runtime/steering.py",
                """
from . import _SEED_TAG as _T


def _prior_run_turns(history: list[Message]) -> list[Message]:
    return [m for m in history if m.metadata.get(_T) is True]
""",
            ),
            (
                "protocore/runtime/__init__.py",
                "from protocore.contracts.types import SESSION_HISTORY_SEED_METADATA_KEY as _SEED_TAG\n",
            ),
        ],
    ),
    (
        # An ALIAS's name, written as a string. The name-string rule matched the
        # constant's own name and nothing else, so the moment an alias existed
        # anywhere in the package its name was a second, unwatched spelling —
        # and the alias here sits in a module whose scope is authorised, so the
        # line creating it is not itself a finding.
        "an_alias_name_spelled_as_a_string_in_a_lookup",
        [
            (
                "protocore/runtime/query.py",
                "from protocore.contracts.types import SESSION_HISTORY_SEED_METADATA_KEY as _SEED_TAG\n",
            ),
            (
                "protocore/runtime/steering.py",
                """
from protocore.runtime import query


def _prior_run_turns(history: list[Message]) -> list[Message]:
    tag = getattr(query, "_SEED_TAG")
    return [m for m in history if m.metadata.get(tag) is True]
""",
            ),
        ],
    ),
]


@pytest.mark.parametrize(
    "source",
    [pytest.param(source, id=case_id) for case_id, source in _SEED_KEY_SPELLINGS],
)
def test_the_boundary_rule_matches_the_key_not_its_spelling(source: str) -> None:
    """A re-derivation is seen however the key is written.

    A rule that matched one AST spelling was one line deep: an alias bound at
    module level, or the key's own string value inlined, re-derived the
    boundary inside an already-authorised function with the whole suite green.
    Both are behaviour-preserving edits an author reaches for without thinking
    about this guard at all.

    Closing those two left the route open, because they were closed AS
    spellings. ``import … as`` binds an ``ast.alias`` and ``types.SESSION_…`` is
    an ``ast.Attribute``; a rule enumerating node shapes had to be told about
    each, and the two it had not been told about were the two an author is most
    likely to write. The rule now asks what a name REFERS to — see
    :func:`_seed_key_vocabulary` — so a spelling that is a new way of writing an
    existing name needs no new case here.

    The FUNCTION that re-derives must be named, not merely the module. A rule
    that resolved no aliases would still flag the module for the alias's own
    assignment line and report ``…::<module>``, which sends the author looking
    at an import rather than at the four lines that re-derive the boundary. So
    this asserts the name of the deriving function, which is the part only the
    alias resolution can produce.
    """
    derivations = seed_key_derivations([("protocore/runtime/query.py", source)])
    allowed = {_RUN_BOUNDARY_SOURCE} | set(_SEED_KEY_DERIVED_ELSEWHERE)
    assert set(derivations) - allowed, (
        f"this spelling re-derives the run boundary and was not seen: {sorted(derivations)}"
    )
    assert any(key.endswith("::_prior_run_turns") for key in derivations), (
        f"the re-deriving FUNCTION must be named, not just its module: {sorted(derivations)}"
    )


@pytest.mark.parametrize(
    "sources",
    [pytest.param(sources, id=case_id) for case_id, sources in _SEED_KEY_RE_EXPORTS],
)
def test_the_boundary_rule_follows_a_re_export_across_modules(
    sources: list[tuple[str, str]],
) -> None:
    """A name re-exported by another module is still the key.

    Read one file at a time, neither half of a re-export is a re-derivation:
    the first only imports a constant under a shorter name, which every module
    that uses it does, and the second imports a name from a sibling. The
    boundary is re-derived by the pair. So the vocabulary is resolved across the
    sources handed in and iterated to a fixed point, which is also what makes an
    alias of an alias resolve.

    Aimed at the second module, because that is where the question about one run
    is actually asked and where the author has to go.
    """
    derivations = seed_key_derivations(sources)
    allowed = {_RUN_BOUNDARY_SOURCE} | set(_SEED_KEY_DERIVED_ELSEWHERE)
    assert set(derivations) - allowed, (
        f"this re-export re-derives the run boundary and was not seen: {sorted(derivations)}"
    )
    assert "protocore/runtime/steering.py::_prior_run_turns" in derivations, (
        "the module that ASKS the question must be named, not only the one that "
        f"re-exported the name: {sorted(derivations)}"
    )


def test_the_boundary_rule_names_the_module_that_writes_the_keys_name() -> None:
    """Writing the key's NAME down is a finding before anything is looked up.

    The lookup and the binding need not be in the same function, or in the same
    module. A module that prepares ``"SESSION_HISTORY_SEED_METADATA_KEY"`` as a
    string and exports it has named the run boundary; reporting only the
    consumer would leave that module reading innocent, which is exactly the
    two-files-neither-guilty shape the cross-module vocabulary exists for.

    A binding is an ``ast.Assign`` / ``ast.AnnAssign``, never a bare expression
    statement, which is what keeps this clear of the six docstrings in this
    package that write the constant's name in prose — the restriction the
    name-string rule was originally given for.
    """
    sources = [
        (
            "protocore/runtime/query.py",
            '_SEED_ATTRIBUTE = "SESSION_HISTORY_SEED_METADATA_KEY"\n',
        ),
        (
            "protocore/runtime/steering.py",
            "from protocore.contracts import types\n"
            "from protocore.runtime.query import _SEED_ATTRIBUTE\n"
            "\n\ndef _prior_run_turns(history: list[Message]) -> list[Message]:\n"
            "    tag = getattr(types, _SEED_ATTRIBUTE)\n"
            "    return [m for m in history if m.metadata.get(tag) is True]\n",
        ),
    ]
    derivations = seed_key_derivations(sources)
    assert "protocore/runtime/steering.py::_prior_run_turns" in derivations, (
        f"the function that looks the name up must be named: {sorted(derivations)}"
    )
    assert "protocore/runtime/query.py::<module>" in derivations, (
        "the module that WRITES the key's name must be named too, or half of a "
        f"two-file re-derivation reads clean: {sorted(derivations)}"
    )


def test_the_boundary_rule_says_so_when_it_runs_out_of_resolution_budget() -> None:
    """A chain too deep to resolve RAISES; it does not report clean.

    :func:`_seed_key_vocabulary` iterates, and an iteration needs a bound. The
    bound used to be a bare ``range`` with an early break, so a re-export chain
    one link longer than the budget produced a vocabulary that was merely
    incomplete — and an incomplete vocabulary reports NO findings, which is
    indistinguishable from a pass. "Iterated to a fixed point" was then a claim
    the code did not keep.

    A guard is allowed a limit. It is not allowed to hit that limit quietly:
    every route this checker has lost was lost to something that returned clean
    while not looking. So exhaustion is a failure that names the cause and the
    remedy, and the depth at which it happens is measured here rather than
    asserted from the constant, so raising ``_VOCABULARY_PASSES`` cannot make
    this test pass by moving the goalposts with it.
    """

    def chain(links: int) -> list[tuple[str, str]]:
        """``links`` re-exports, every module read BEFORE the one it imports."""
        modules = [
            (
                "protocore/chain/m00.py",
                "from protocore.contracts.types import SESSION_HISTORY_SEED_METADATA_KEY as _T0\n",
            )
        ]
        for index in range(1, links):
            modules.append(
                (
                    f"protocore/chain/m{index:02d}.py",
                    f"from protocore.chain.m{index - 1:02d} import _T{index - 1} as _T{index}\n",
                )
            )
        modules.append(
            (
                "protocore/runtime/steering.py",
                f"from protocore.chain.m{links - 1:02d} import _T{links - 1}\n"
                "\n\ndef _prior_run_turns(history: list[Message]) -> list[Message]:\n"
                f"    return [m for m in history if m.metadata.get(_T{links - 1})]\n",
            )
        )
        return list(reversed(modules))

    resolvable = _VOCABULARY_PASSES - 2
    derivations = seed_key_derivations(chain(resolvable))
    assert "protocore/runtime/steering.py::_prior_run_turns" in derivations, (
        f"a chain of {resolvable} links is inside the budget and must resolve: {sorted(derivations)}"
    )

    with pytest.raises(AssertionError, match="did not settle"):
        seed_key_derivations(chain(_VOCABULARY_PASSES + 1))


def test_the_boundary_rule_reads_the_key_from_the_constant_itself() -> None:
    """The value AND the name watched are the constant's, not copies that drift.

    Change the key in ``contracts/types.py`` and this rule follows it, because
    it imports the constant rather than repeating its text.

    The NAME cannot be imported the same way — a name does not describe itself —
    so :data:`_SEED_KEY_NAME` is a literal, and it is the one thing here that
    could go stale silently: rename the constant and the alias, attribute and
    ``getattr`` rules would all stop matching while the value rule went on
    passing. So the literal is checked against the module it names.
    """
    assert getattr(contracts_types, _SEED_KEY_NAME, None) == SESSION_HISTORY_SEED_METADATA_KEY, (
        f"_SEED_KEY_NAME is {_SEED_KEY_NAME!r}, which is not what the constant "
        "is called in protocore.contracts.types any more; every rule that "
        "matches the key by its NAME is looking for a name nothing binds"
    )

    source = (
        "def _prior(history: list[Message]) -> list[Message]:\n"
        f'    return [m for m in history if m.metadata.get("{SESSION_HISTORY_SEED_METADATA_KEY}")]\n'
    )
    assert seed_key_derivations([("protocore/runtime/query.py", source)])

    unrelated = (
        "def _prior(history: list[Message]) -> list[Message]:\n"
        '    return [m for m in history if m.metadata.get("some.other.key")]\n'
    )
    assert not seed_key_derivations([("protocore/runtime/query.py", unrelated)])


def _test_functions_in_repo() -> set[str]:
    """``<test path>::<test function name>`` for every test in the suite."""
    names: set[str] = set()
    for path in _swept_files(_TESTS_ROOT, "test_*.py", _MIN_TEST_MODULES, "test modules"):
        key = path.relative_to(_REPO_ROOT).as_posix()
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
                names.add(f"{key}::{node.name}")
    return names


def test_every_run_scoped_claim_names_a_test_that_pins_it() -> None:
    """A reason that asserts a run-scope property must be falsifiable.

    ``QueryEngine.latest_user_message`` is why this test exists. Its reason —
    "the tail-most user turn, so it is always this run's" — is a claim about
    behaviour, and turning its reversed walk into a forward one made it false:
    tool retrieval and skill loading then run against a PRIOR run's task, and
    the whole suite passed. The prose read like a guarantee and nothing checked
    it. So an entry may claim a run-scope property only if it names the tests
    that break when the property does, and those tests must exist.
    """
    missing_pins = sorted(
        key
        for key, declaration in _WHOLE_HISTORY_BY_DESIGN.items()
        if declaration.claim is _Claim.RUN_SCOPED_BY_CONSTRUCTION and not declaration.pinned_by
    )
    assert not missing_pins, (
        "these entries claim their answer is about ONE run but name no test "
        f"that pins it: {missing_pins}. Write the test, or restate the reason "
        "as _whole(...) if the answer really is about the whole session."
    )

    available = _test_functions_in_repo()
    dangling = sorted(
        {
            f"{key} -> {pin}"
            for key, declaration in _WHOLE_HISTORY_BY_DESIGN.items()
            for pin in declaration.pinned_by
            if pin not in available
        }
    )
    assert not dangling, (
        "these registry entries name a pinning test that does not exist — a "
        f"renamed or deleted pin leaves the claim unverified again: {dangling}"
    )


def _declared_pins() -> dict[str, tuple[str, ...]]:
    """``<test path>::<test name>`` -> the registry entries it declares it pins.

    Read from a module-level ``PINNED_ENTRIES`` mapping in whichever test file
    the pinning test lives in, by parsing rather than importing — the same
    static discipline the rest of this file keeps.
    """
    declared: dict[str, tuple[str, ...]] = {}
    for path in _swept_files(_TESTS_ROOT, "test_*.py", _MIN_TEST_MODULES, "test modules"):
        key = path.relative_to(_REPO_ROOT).as_posix()
        for stmt in ast.parse(path.read_text()).body:
            targets = (
                stmt.targets
                if isinstance(stmt, ast.Assign)
                else [stmt.target]
                if isinstance(stmt, ast.AnnAssign)
                else []
            )
            if not any(isinstance(target, ast.Name) and target.id == "PINNED_ENTRIES" for target in targets):
                continue
            if stmt.value is None:
                continue
            for test_name, entries in ast.literal_eval(stmt.value).items():
                declared[f"{key}::{test_name}"] = tuple(entries)
    return declared


def test_every_pinning_test_still_pins_what_it_claims() -> None:
    """The pins are a TWO-SIDED declaration, so a downgrade cannot be silent.

    The registry names the tests that hold each run-scope claim, and each of
    those tests names, in its own file, the entries it was written to hold. The
    two lists must agree exactly.

    One-sided pins were one paraphrase deep. ``_tool_name_for_call_id`` could be
    reclassified ``_whole`` with its prose UNTOUCHED and nothing failed —
    measured, exit 0, zero failures — because the reason argues about runs in
    wording no phrase list happened to contain. Delete the pinning test next
    and restore the forward-walk defect in ``latest_user_message``, and the
    suite is still green. Prose cannot carry this: an author can always reword.
    A declaration on the test's own side can, because downgrading an entry
    leaves the test claiming to pin something that no longer claims to be
    pinned, and that is a disagreement rather than a wording.

    What this does NOT catch: a NEW entry, written whole-transcript from the
    start, with a reason that argues about one run. Nothing was ever pinned, so
    there is nothing to disagree with. That residual is the phrase filter's
    job, and the phrase filter is lexical.

    **Nor a downgrade where both sides are edited to agree — and that inverts
    the division of labour this file used to state.** Measured, three
    mutations of ``_tool_name_for_call_id``:

    * reclassified ``_whole``, prose byte-identical, ``PINNED_ENTRIES`` left
      alone — this test fires, and so does the phrase filter;
    * the same, with the entry ALSO removed from ``PINNED_ENTRIES`` so the two
      sides agree again — this test passes. Only the phrase filter fires;
    * the same again, with the phrases the reason happens to use deleted from
      :data:`_RUN_SCOPE_PHRASES` — the whole suite is green, nothing fails.

    So a two-sided pin falls to a two-sided edit, and it is this test's own
    failure text that names both places and invites the author to make them
    agree. Against that edit the LEXICAL filter is the load-bearing check, not
    the backstop — the reverse of what was written here. Neither check is
    structural against an author who is willing to edit what the check reads:
    the pins are data in two files and the phrase list is data in one, and the
    phrase list has nothing watching it at all.
    """
    claimed_by_registry: dict[str, set[str]] = {}
    for key, declaration in _WHOLE_HISTORY_BY_DESIGN.items():
        for pin in declaration.pinned_by:
            claimed_by_registry.setdefault(pin, set()).add(key)

    declared = _declared_pins()
    pins_in_registry = set(claimed_by_registry)
    pins_declared = set(declared)

    assert pins_in_registry == pins_declared, (
        "every test named as a pin must declare, in PINNED_ENTRIES beside "
        "itself, which registry entries it pins — and nothing else may claim "
        "to be a pin. Named by the registry but not declaring themselves: "
        f"{sorted(pins_in_registry - pins_declared)}. Declaring themselves but "
        f"named by no entry: {sorted(pins_declared - pins_in_registry)} — an "
        "entry that was downgraded or deleted leaves its pin stranded here."
    )

    disagreements = sorted(
        f"{pin}: pins {sorted(set(declared[pin]))}, but the registry names it from {sorted(claimed_by_registry[pin])}"
        for pin in pins_in_registry
        if set(declared[pin]) != claimed_by_registry[pin]
    )
    assert not disagreements, (
        "a pinning test and the registry disagree about what is pinned; the "
        "usual cause is an entry downgraded to _whole while the test that "
        f"holds its claim was left in place: {disagreements}"
    )


#: Phrases that make a reason an argument about ONE run rather than about the
#: session. A whole-transcript entry whose reason uses one of them has been
#: downgraded rather than rewritten, which is one way out of the pin
#: requirement.
#:
#: This list is a LEXICAL FILTER and it is one paraphrase deep. It is stated
#: that way because it was measured that way: a run-scope claim reworded to
#: "the search cannot land on another run's" walked past the original four
#: phrases with the whole suite green, and the wording was not invented — it
#: was lifted verbatim from an entry in this very registry.
#:
#: Two things about it are worse than "one paraphrase deep", both measured.
#: Nothing asserts these contents, so deleting a phrase is a silent widening —
#: it is the third step of the mutation described in
#: :func:`test_every_pinning_test_still_pins_what_it_claims`, and it takes the
#: suite from one failure to none. And the list does not cover its own
#: registry: 3 of the 11 entries claiming RUN_SCOPED_BY_CONSTRUCTION word that
#: claim without any phrase below (``_history_has_tool_result``,
#: ``_history_tool_result_is_terminal``,
#: ``_assert_history_has_matching_pending_tool_use`` — all of them arguing from
#: "ONE tool_call_id"). Downgrade one of those and this filter says nothing.
#: Deriving the filter from the live reasons instead of hand-listing it is the
#: fix, and it is not made here.
_RUN_SCOPE_PHRASES = (
    "this run",
    "one run",
    "the current run",
    "the new task",
    "another run",
    "an earlier run",
    "a prior run",
    "prior-run",
    "the same run",
    "its own run",
)


def test_a_claim_that_cannot_be_pinned_does_not_pretend_to_be() -> None:
    """The complement: whole-transcript entries carry no pins, and say so.

    "This is about the whole session" is not a run-scope property, so there is
    nothing for a behavioural test to hold. Letting such an entry carry pins
    would make the two kinds of claim indistinguishable, and the distinction is
    the only thing keeping the first kind honest.
    """
    over_claimed = sorted(
        key
        for key, declaration in _WHOLE_HISTORY_BY_DESIGN.items()
        if declaration.claim is not _Claim.RUN_SCOPED_BY_CONSTRUCTION and declaration.pinned_by
    )
    assert not over_claimed, (
        "these entries do not claim a run-scope property but name pinning "
        f"tests: {over_claimed}. Either the claim is RUN_SCOPED_BY_CONSTRUCTION "
        "or the pins do not belong."
    )
    assert all(declaration.reason.strip() for declaration in _WHOLE_HISTORY_BY_DESIGN.values()), (
        "every registry entry states its reason"
    )

    # One way out of the pin requirement is to leave the reason alone and
    # downgrade the claim beside it. This catches the wordings in
    # _RUN_SCOPE_PHRASES and nothing else — it is a lexical filter, and it was
    # measured being walked past by a reword. It is also, measured, the ONLY
    # check that fires when the downgrade edits both sides of the pin, so it is
    # not the backstop it was once described as; see
    # test_every_pinning_test_still_pins_what_it_claims for the three mutations.
    downgraded = sorted(
        f"{key}: {phrase!r}"
        for key, declaration in _WHOLE_HISTORY_BY_DESIGN.items()
        if declaration.claim is _Claim.WHOLE_TRANSCRIPT
        for phrase in _RUN_SCOPE_PHRASES
        if phrase in declaration.reason.lower()
    )
    assert not downgraded, (
        "these entries claim their question is about the whole session, but "
        f"their reason argues about one run: {downgraded}. Either the reason "
        "is a run-scope argument — in which case the claim is "
        "RUN_SCOPED_BY_CONSTRUCTION and it needs a pin — or it should not say "
        "so."
    )


# ---------------------------------------------------------------------------
# Readers that do not exist in the tree, each written the way a real one would
# be. The guard has to refuse every one on sight — a check that can only
# recognise the violations already catalogued is a list, not a guard.
#
# The first three are the original shapes. Everything after them is a shape
# that got PAST the receiver-name check this guard used to run, found by
# attacking it rather than by reading it.
# ---------------------------------------------------------------------------

_VIOLATING_READERS: list[tuple[str, str, str]] = [
    (
        "raw_attribute",
        "_history_has_delegation_result",
        '''
def _history_has_delegation_result(engine: QueryEngine) -> bool:
    """Return True once this run has delegated to a subagent."""
    for message in reversed(engine.history):
        for block in message.content_blocks:
            if isinstance(block, ToolResultBlock) and not block.is_error:
                return True
    return False
''',
    ),
    (
        "public_accessor",
        "_run_mentioned_the_deadline",
        """
def _run_mentioned_the_deadline(engine: QueryEngine) -> bool:
    return any(
        isinstance(block, TextBlock) and "deadline" in block.text
        for message in engine.history_snapshot()
        for block in message.content_blocks
    )
""",
    ),
    (
        "dynamic_lookup",
        "_run_produced_reasoning",
        """
def _run_produced_reasoning(engine: QueryEngine) -> bool:
    transcript = getattr(engine, "history", ())
    return any(message.role is MessageRole.assistant for message in transcript)
""",
    ),
    (
        "renamed_receiver",
        "_history_has_delegation_result_v2",
        '''
def _history_has_delegation_result_v2(eng: QueryEngine) -> bool:
    """Return True once THIS run has delegated work to a subagent."""
    for message in reversed(eng.history):
        for block in message.content_blocks:
            if isinstance(block, ToolResultBlock) and not block.is_error:
                return True
    return False
''',
    ),
    (
        "engine_rebound_to_a_local",
        "_run_answered_already",
        """
def _run_answered_already(engine: QueryEngine) -> bool:
    transcript = engine
    for message in reversed(transcript.history):
        if message.role is MessageRole.assistant:
            return True
    return False
""",
    ),
    (
        "attribute_chain_receiver",
        "_run_wrote_anything",
        """
class _Convergence:
    def _run_wrote_anything(self) -> bool:
        for message in self.engine.history:
            if message.role is MessageRole.assistant:
                return True
        return False
""",
    ),
    (
        "context_object_receiver",
        "_run_used_a_tool",
        """
def _run_used_a_tool(ctx: RunContext) -> bool:
    return any(
        isinstance(block, ToolUseBlock)
        for message in ctx.engine.history
        for block in message.content_blocks
    )
""",
    ),
    (
        "transcript_as_a_parameter",
        "_run_answered_from",
        '''
def _run_answered_from(history: list[Message]) -> bool:
    """Return True once THIS run has produced an assistant answer."""
    for message in reversed(history):
        if message.role is MessageRole.assistant:
            return True
    return False
''',
    ),
    (
        "transcript_as_a_renamed_parameter",
        "_run_answered_from_turns",
        '''
def _run_answered_from_turns(turns: Sequence[Message]) -> bool:
    """Return True once THIS run has produced an assistant answer."""
    return any(message.role is MessageRole.assistant for message in turns)
''',
    ),
    (
        "transcript_as_a_quoted_parameter",
        "_run_answered_from_quoted",
        '''
def _run_answered_from_quoted(turns: "Sequence[Message]") -> bool:
    """Return True once THIS run has produced an assistant answer."""
    return any(message.role is MessageRole.assistant for message in turns)
''',
    ),
    (
        "transcript_as_a_partly_quoted_parameter",
        "_run_answered_from_inner_quoted",
        '''
def _run_answered_from_inner_quoted(turns: list["Message"]) -> bool:
    """Return True once THIS run has produced an assistant answer."""
    return any(message.role is MessageRole.assistant for message in turns)
''',
    ),
    (
        "vars_lookup",
        "_run_has_any_turn",
        """
def _run_has_any_turn(engine: QueryEngine) -> bool:
    return bool(vars(engine)["history"])
""",
    ),
    (
        "dunder_dict_lookup",
        "_run_turn_count",
        """
def _run_turn_count(engine: QueryEngine) -> int:
    return len(engine.__dict__["history"])
""",
    ),
    (
        "attribute_getter",
        "_run_transcript",
        """
def _run_transcript(engine: QueryEngine) -> list[Message]:
    return operator.attrgetter("history")(engine)
""",
    ),
    (
        "snapshot_dictionary",
        "_run_snapshot_turns",
        """
def _run_snapshot_turns(engine: QueryEngine) -> list[dict[str, object]]:
    return engine.snapshot()["history"]
""",
    ),
    (
        "computed_attribute_name",
        "_run_sneaky_transcript",
        """
def _run_sneaky_transcript(engine: QueryEngine) -> object:
    return getattr(engine, "his" + "tory", ())
""",
    ),
    (
        "getattr_via_a_local_name",
        "_run_indirect_transcript",
        """
def _run_indirect_transcript(engine: QueryEngine) -> object:
    attribute = "history"
    return getattr(engine, attribute, ())
""",
    ),
    (
        "accessor_on_a_chained_receiver",
        "_run_snapshot_via_chain",
        """
def _run_snapshot_via_chain(ctx: RunContext) -> object:
    return ctx.engine.history_snapshot()
""",
    ),
    (
        "walrus_bound_transcript",
        "_run_first_turn",
        """
def _run_first_turn(engine: QueryEngine) -> object:
    if transcript := engine.history:
        return transcript[0]
    return None
""",
    ),
    (
        "async_reader",
        "_run_answered_async",
        """
async def _run_answered_async(eng: QueryEngine) -> bool:
    for message in reversed(eng.history):
        if message.role is MessageRole.assistant:
            return True
    return False
""",
    ),
    (
        "transcript_via_a_module_level_type_alias",
        "_run_answered_from_alias",
        '''
_Transcript = list[Message]


def _run_answered_from_alias(history: _Transcript) -> bool:
    """Return True once THIS run has produced an assistant answer."""
    return any(message.role is MessageRole.assistant for message in history)
''',
    ),
    (
        "transcript_via_a_quoted_type_alias",
        "_run_answered_from_quoted_alias",
        '''
_Transcript = Sequence[Message]


def _run_answered_from_quoted_alias(history: "_Transcript") -> bool:
    """Return True once THIS run has produced an assistant answer."""
    return any(message.role is MessageRole.assistant for message in history)
''',
    ),
    (
        "transcript_via_a_chained_type_alias",
        "_run_answered_from_chained_alias",
        '''
_Turns = list[Message]
_Transcript = _Turns


def _run_answered_from_chained_alias(history: _Transcript) -> bool:
    """Return True once THIS run has produced an assistant answer."""
    return any(message.role is MessageRole.assistant for message in history)
''',
    ),
    (
        "transcript_as_a_vararg",
        "_run_answered_from_varargs",
        '''
def _run_answered_from_varargs(*turns: Message) -> bool:
    """Return True once THIS run has produced an assistant answer."""
    return any(message.role is MessageRole.assistant for message in turns)
''',
    ),
    (
        "transcript_as_a_kwarg",
        "_run_answered_from_kwargs",
        '''
def _run_answered_from_kwargs(**turns: Message) -> bool:
    """Return True once THIS run has produced an assistant answer."""
    return any(m.role is MessageRole.assistant for m in turns.values())
''',
    ),
    (
        "transcript_as_a_mapping",
        "_run_answered_from_mapping",
        '''
def _run_answered_from_mapping(turns: dict[str, Message]) -> bool:
    """Return True once THIS run has produced an assistant answer."""
    return any(m.role is MessageRole.assistant for m in turns.values())
''',
    ),
    (
        "a_splice_helper_that_also_asks_a_question",
        "_seed_prior_runs_and_count",
        '''
def _seed_prior_runs_and_count(engine: QueryEngine, seed: list[Message]) -> int:
    """Splice the seed in — and answer a question about it while there."""
    engine.history[0:0] = seed
    return sum(1 for message in seed if message.role is MessageRole.assistant)
''',
    ),
    (
        "transcript_typed_through_an_import_alias",
        "_run_answered_from_aliased_import",
        '''
from protocore.contracts.types import Message as Msg


def _run_answered_from_aliased_import(turns: list[Msg]) -> bool:
    """Return True once THIS run has produced an assistant answer."""
    return any(message.role is MessageRole.assistant for message in turns)
''',
    ),
    (
        "transcript_as_an_async_iterator",
        "_run_answered_from_a_stream",
        '''
async def _run_answered_from_a_stream(turns: AsyncIterator[Message]) -> bool:
    """Return True once THIS run has produced an assistant answer."""
    async for message in turns:
        if message.role is MessageRole.assistant:
            return True
    return False
''',
    ),
    (
        "transcript_as_a_generator",
        "_run_answered_from_a_generator",
        '''
def _run_answered_from_a_generator(turns: Generator[Message, None, None]) -> bool:
    """Return True once THIS run has produced an assistant answer."""
    return any(message.role is MessageRole.assistant for message in turns)
''',
    ),
]


@pytest.mark.parametrize(
    ("source", "expected_name"),
    [pytest.param(source, name, id=case_id) for case_id, name, source in _VIOLATING_READERS],
)
def test_the_guard_refuses_a_reader_that_does_not_exist_yet(source: str, expected_name: str) -> None:
    """The enforcement names a brand-new violating reader, by every route it has.

    Aimed at the same function the tree sweep uses, so a guard weakened into a
    blanket pass fails HERE — a check that only ever sees code already known to
    be clean can be gutted without anything noticing.

    "Every route it has" is the honest scope: these are the shapes this file
    classifies, not the shapes that exist. Two known shapes are deliberately
    absent because the checker cannot see them — a parameter annotated ``Any``
    and one with no annotation — and they are stated in the module docstring
    rather than quietly missing from this list.
    """
    undeclared = undeclared_history_readers([("protocore/runtime/query.py", source)])
    assert any(key.endswith(f"::{expected_name}") or key.endswith(f".{expected_name}") for key in undeclared), (
        f"{expected_name} was not refused; the guard reported {sorted(undeclared)}"
    )


#: The complement of the attack list: legitimate work a developer does under
#: time pressure, which must not be refused. The mutation cases were refused by
#: the receiver-name version of this guard, which then told the author to
#: register the function as a whole-transcript reader — a reason that is false
#: for a mutation and a name pre-authorised for good. The last four were
#: refused by the version after it, which had separated mutation from question
#: on the attribute route but not on the parameter route: a helper handed the
#: seed to splice, a helper handed turns to extend with, a ``Protocol`` method
#: and ``@overload`` stubs were all findings, and the message each received
#: ended by saying that what it was doing needed no registry entry.
_LEGITIMATE_SOURCES: list[tuple[str, str]] = [
    (
        "run_scoped_reader_drawing_from_the_boundary",
        '''
def _history_has_delegation_result(engine: QueryEngine) -> bool:
    """Return True once this run has delegated to a subagent."""
    for message in reversed(_this_run_messages(engine)):
        for block in message.content_blocks:
            if isinstance(block, ToolResultBlock) and not block.is_error:
                return True
    return False
''',
    ),
    (
        "append",
        """
def _record_turn(engine: QueryEngine, message: Message) -> None:
    engine.history.append(message)
""",
    ),
    (
        "pop_rolls_back_the_turn_just_appended",
        """
def _drop_last_turn(engine: QueryEngine) -> None:
    engine.history.pop()
""",
    ),
    (
        "remove_and_reverse_and_sort",
        """
def _reshuffle(engine: QueryEngine, message: Message) -> None:
    engine.history.remove(message)
    engine.history.reverse()
    engine.history.sort(key=lambda m: m.role.value)
""",
    ),
    (
        "slice_assignment",
        """
def _replace_last_turn(engine: QueryEngine, replacement: Message) -> None:
    engine.history[-1:] = [replacement]
""",
    ),
    (
        "index_assignment",
        """
def _overwrite_turn(engine: QueryEngine, index: int, replacement: Message) -> None:
    engine.history[index] = replacement
""",
    ),
    (
        "del_statement",
        """
def _forget_first_turn(engine: QueryEngine) -> None:
    del engine.history[0]
""",
    ),
    (
        "rebinding_the_attribute",
        """
def _reset_transcript(engine: QueryEngine) -> None:
    engine.history = []
""",
    ),
    (
        "the_seeding_splice",
        """
def _seed_prior_runs(engine: QueryEngine) -> None:
    engine.history[0:0] = list(engine.pending_seed)
""",
    ),
    (
        "a_single_message_parameter",
        """
def _turn_is_user(message: Message | None) -> bool:
    return message is not None and message.role is MessageRole.user
""",
    ),
    (
        "a_non_message_sequence_parameter",
        """
def _load_satisfied(history: list[tuple[str, dict[str, Any]]] | None) -> frozenset[str]:
    return frozenset(name for name, _ in history or ())
""",
    ),
    (
        "getattr_on_a_readable_module_constant",
        """
_FORCE_NEXT_TOOL_ATTR = "_longfile_force_next_tool"


def _forced_tool(engine: QueryEngine) -> object:
    return getattr(engine, _FORCE_NEXT_TOOL_ATTR, None)
""",
    ),
    (
        "a_helper_handed_the_seed_to_splice",
        '''
def _splice_prior_runs(engine: QueryEngine, seed_messages: list[Message]) -> None:
    """Prepend a prior run's turns. Pure mutation - asks nothing."""
    engine.history[0:0] = seed_messages
''',
    ),
    (
        "a_helper_handed_messages_to_extend_with",
        """
def _record_turns(engine: QueryEngine, turns: Sequence[Message]) -> None:
    engine.history.extend(turns)
""",
    ),
    (
        "a_protocol_method_with_no_body",
        """
class MessageSink(Protocol):
    def write(self, messages: list[Message]) -> None: ...
""",
    ),
    (
        "overload_stubs_over_a_registered_implementation",
        """
@overload
def render(messages: list[Message], *, html: Literal[True]) -> str: ...
@overload
def render(messages: list[Message], *, html: Literal[False]) -> bytes: ...
""",
    ),
]


@pytest.mark.parametrize(
    "source",
    [pytest.param(source, id=case_id) for case_id, source in _LEGITIMATE_SOURCES],
)
def test_the_guard_allows_work_that_asks_no_run_scoped_question(source: str) -> None:
    """Reading correctly, and changing the transcript, are both allowed.

    A guard is only as good as the work it lets through, and the cost of
    refusing honest work is not that the author is inconvenienced — it is that
    the only green path on offer records something untrue in the registry,
    which is where the next author reads the precedent from.
    """
    assert undeclared_history_readers([("protocore/runtime/query.py", source)]) == {}


#: Where the mutation exemption stops. Each of these sits next to a mutation
#: and is still a question — asked of the whole session, and answerable
#: wrongly. Stating them as tests keeps the exemption from widening by
#: association: "it was near a pop()" is not a reason.
_STILL_QUESTIONS: list[tuple[str, str]] = [
    (
        "emptiness_guard_before_a_pop",
        """
def _drop_last_turn(engine: QueryEngine) -> None:
    if engine.history:
        engine.history.pop()
""",
    ),
    (
        "length_of_the_transcript",
        """
def _log_transcript_size(engine: QueryEngine) -> None:
    logger.warning("DIAG transcript messages=%d", len(engine.history))
""",
    ),
    (
        "the_tail_message",
        """
def _last_turn_role(engine: QueryEngine) -> str:
    return engine.history[-1].role.value
""",
    ),
    (
        "a_count_taken_while_splicing",
        """
def _seed_prior_runs(engine: QueryEngine, seed_messages: list[Message]) -> int:
    engine.history[0:0] = seed_messages
    return len(seed_messages)
""",
    ),
    (
        "an_answer_stored_into_an_attribute",
        """
def _mark_answered(engine: QueryEngine, turns: list[Message]) -> None:
    engine.answered = any(m.role is MessageRole.assistant for m in turns)
""",
    ),
    (
        "an_answer_stored_into_a_subscript",
        """
def _record_turn_count(state: dict[str, int], turns: list[Message]) -> None:
    state["turns"] = len(turns)
""",
    ),
]


@pytest.mark.parametrize(
    "source",
    [pytest.param(source, id=case_id) for case_id, source in _STILL_QUESTIONS],
)
def test_the_mutation_exemption_stops_at_questions(source: str) -> None:
    """A change is exempt; a question standing next to one is not.

    ``if engine.history:`` is emptiness — and "has this run done anything?" is
    exactly the run-scoped question that must not be asked of the raw list, so
    the exemption cannot extend to it merely because a ``pop()`` follows. The
    last case is the boundary from the other side: a helper handed the seed
    passes while it only splices, and stops passing the moment it also counts.

    An earlier version of this test asserted that the pure splice was a
    finding, and its docstring claimed the remedy offered was true for every
    case here. It was not. The message that refusal produced ended by telling
    the author that slice assignment needs no registry entry, which is what
    that function does and nothing else — a refusal followed by a denial that
    there was anything to refuse. Neither branch of the remedy applied either:
    the function asks nothing, so scoping at the call site is not available,
    and registering a pure mutation as a whole-transcript READER states
    something false about it. The exemption now reaches one frame out, and the
    case moved to :data:`_LEGITIMATE_SOURCES`.
    """
    assert undeclared_history_readers([("protocore/runtime/query.py", source)]) != {}


def test_the_remedy_matches_the_route_the_reader_took() -> None:
    """The failure text tells the author something true for THEIR case.

    One paragraph covering four routes was wrong for three of them. Each route
    carries its own remedy, and the message says outright that changing the
    transcript is not a finding — so nobody registers a ``pop()`` under a
    reason that is not true.
    """
    parameter_case = """
def _run_answered_from(history: list[Message]) -> bool:
    return any(message.role is MessageRole.assistant for message in history)
"""
    attribute_case = """
def _run_answered(eng: QueryEngine) -> bool:
    return any(message.role is MessageRole.assistant for message in eng.history)
"""
    parameter_text = _describe(undeclared_history_readers([("protocore/runtime/query.py", parameter_case)]))
    attribute_text = _describe(undeclared_history_readers([("protocore/runtime/query.py", attribute_case)]))

    assert "CALL SITE" in parameter_text
    assert "reaches the session transcript directly" not in parameter_text
    assert "reaches the session transcript directly" in attribute_text
    assert "CALL SITE" not in attribute_text
    # A sequence handed in may not be a transcript at all — hand-authored
    # few-shot examples typed list[Message] reach no engine. Without this
    # branch the only green path was a whole-transcript reason that is untrue
    # of them, which is a fiction recorded under duress.
    assert "NOT_THE_TRANSCRIPT" in parameter_text
    for text in (parameter_text, attribute_text):
        assert "_this_run_messages" in text
        assert "pop" in text and "not a finding" in text
