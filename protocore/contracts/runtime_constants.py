# ruff: noqa: RUF001 — Bilingual RU+EN default prompt strings intentionally use Cyrillic characters.
"""LoopConstants — frozen Pydantic snapshot + Provider Protocol.

The only configuration surface that flows core ↔ the host. Snapshot is
**always passed by value** into per-turn ``query()`` — no global state, no
module-level cache.

Canonical inputs only — derived values (e.g. compaction trigger tokens)
are computed via :mod:`protocore.runtime.context.budgets`. No prompt
strings live here (anti-pattern from v1 — moved to ``prompts/templates/``).
"""
from __future__ import annotations

from typing import Literal, Protocol, Self, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from protocore.constants import MAX_DATA_NESTING_DEPTH


class LoopConstants(BaseModel):
    """Frozen snapshot of every runtime-tunable threshold.

    Updates from dashboard land in PG ``runtime_constants`` table;
    the host :class:`RuntimeConstantsProvider` reads + caches, watches
    Redis pub/sub for invalidation, and rebuilds a fresh frozen snapshot
    on each invalidation. Snapshot is then injected per-turn.

    ALL fields are canonical inputs — formula-derived values (token budgets +
    the compaction trigger) are computed by
    :func:`protocore.runtime.context.budgets.derive_budgets`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

 # ----- Model context window -----
    model_context_window: int = Field(
        default=49_152,
        gt=0,
        description=(
            "Per-scope context window (tokens). Synced at run setup from the "
            "resolved provider's llm_provider_config.context_window so budgets "
            "track the model actually serving the run; an explicit per-tenant "
            "override takes precedence over the provider window. This default is "
            "the fallback when neither is available."
        ),
    )

 # ----- Compaction thresholds (canonical fractions) -----
    compaction_trigger_ratio: float = Field(
        default=0.8,
        gt=0.0,
        le=1.0,
        description="Fraction of context window above which compaction is triggered.",
    )
    compaction_routine_min_clear_ratio: float = Field(
        default=0.5,
        gt=0.0,
        le=1.0,
        description="Minimum input fraction the routine clear pass must reduce.",
    )
    compaction_emergency_ratio: float = Field(
        default=0.95,
        gt=0.0,
        le=1.0,
        description="Cliff: above this fraction, emergency clear runs unconditionally.",
    )
    compaction_min_gain_ratio: float = Field(
        default=0.03,
        ge=0.0,
        le=0.5,
        description=(
            "A routine compaction that frees less than this fraction of the prompt "
            "achieved nothing the next iteration will not undo; the per-iteration "
            "gate then stands down for compaction_no_gain_backoff_iterations "
            "iterations instead of paying for the same empty pass every time. "
            "The emergency cliff is not subject to the backoff."
        ),
    )
    compaction_no_gain_backoff_iterations: int = Field(
        default=6,
        ge=0,
        description=(
            "Iterations the per-iteration compaction gate skips after a pass that "
            "freed less than compaction_min_gain_ratio. 0 disables the backoff."
        ),
    )
    token_estimate_calibration: float = Field(
        default=1.0,
        ge=1.0,
        le=4.0,
        description=(
            "Multiplier applied to every character-based token estimate. The "
            "heuristic runs short of a real tokenizer on JSON-heavy and non-Latin "
            "text, by a factor that depends on the content; when a provider reports "
            "the true size of a prompt, the loop sets this so that the tiers size "
            "units, budgets and gains in the provider's tokens rather than in an "
            "undercount that leaves them nothing to shrink. 1.0 = uncalibrated."
        ),
    )
    token_estimate_calibration_enabled: bool = Field(
        default=True,
        description=(
            "Whether the loop updates token_estimate_calibration from the prompt "
            "sizes providers report. Off, the estimate stays at its configured factor."
        ),
    )
    compaction_per_iteration_enabled: bool = Field(
        default=True,
        description=(
            "When true (default), the inner tool-iteration loop checks the "
            "compaction trigger BEFORE rebuilding the wire payload for the next "
            "assistant stream, so a long single run with many tool iterations "
            "stays context-bounded instead of inflating monotonically until a "
            "provider 413. The routine trigger (compaction_trigger_ratio) drives "
            "a normal compaction; crossing compaction_emergency_ratio drives a "
            "proactive force_compaction. Set false to revert to the "
            "turn-start-only gate (kill-switch). NOTE: this flag gates ONLY the "
            "per-iteration gate. The turn-start emergency cliff is controlled "
            "SEPARATELY by compaction_emergency_proactive_enabled and is NOT "
            "affected by this flag — a full rollback (no proactive compaction at "
            "all) requires setting BOTH compaction_per_iteration_enabled=false "
            "AND compaction_emergency_proactive_enabled=false."
        ),
    )
    compaction_emergency_proactive_enabled: bool = Field(
        default=True,
        description=(
            "When true (default), the compaction_emergency_ratio is active as "
            "a proactive cliff — when estimated history tokens exceed "
            "model_context_window * compaction_emergency_ratio the runtime runs "
            "force_compaction (both tiers unconditionally) BEFORE streaming, "
            "rather than waiting for the provider to raise a "
            "context-window-exceeded error. Gates the per-iteration emergency "
            "branch and the turn-start emergency branch. Set false to disable "
            "the proactive cliff (reactive-413 recovery still applies)."
        ),
    )

 # ----- Tool result truncation -----
    tool_result_truncation_ratio: float = Field(
        default=0.10,
        gt=0.0,
        le=0.5,
        description=(
            "Tool-result size cap as fraction of context window. Default "
            "0.10 (raised from 0.05, which was too aggressive "
            "for small-window models and truncated multi-thousand-line tool "
            "outputs to a sliver). The "
            "0.5 upper bound is a sanity cap — above 50% a single result "
            "would crowd out history."
        ),
    )

 # ----- System prompt / skill index budgets -----
    system_prompt_max_ratio: float = Field(
        default=0.10,
        gt=0.0,
        le=1.0,
        description="System prompt token budget as fraction of context window.",
    )
    skill_index_budget_ratio: float = Field(
        default=0.01,
        gt=0.0,
        le=1.0,
        description="Skill index token budget as fraction of context window.",
    )

 # ----- Run-level tool preconditions -----
 #
 # Bounds for the ordered "call these tools before you may answer" list a run
 # may carry (``QueryEngineConfig.tool_preconditions``, enforced by
 # ``protocore.runtime.run_tool_preconditions``). A run with none is untouched
 # by every one of these.
 #
 # The ``run_`` prefix is load-bearing in an operator's flat constants list:
 # ``tool_preconditions_enabled`` below is the UNRELATED per-tool dependency
 # DAG ("FinalizeFile may not run before AppendFile"), and
 # ``tool_action_preconditions_*`` is a third, also unrelated, gate.
    run_tool_precondition_max_entries: int = Field(
        default=8,
        ge=1,
        le=32,
        description=(
            "Maximum number of entries in a run's ordered tool-precondition "
            "list. Each entry is satisfied by its own sequence of forced "
            "turns, so the list length multiplies the worst-case number of "
            "provider calls a run spends before the agent is free to answer."
        ),
    )
    run_tool_precondition_max_calls: int = Field(
        default=10,
        ge=1,
        le=100,
        description=(
            "Maximum ``calls`` on a single tool-precondition entry — how many "
            "SUCCESSFUL calls of one tool a caller may demand before the "
            "agent is free to answer. The caller-facing layer rejects an "
            "out-of-range value at run creation; the engine config refuses it "
            "as a last line of defence."
        ),
    )
    run_tool_precondition_max_attempts: int = Field(
        default=3,
        ge=1,
        le=20,
        description=(
            "Consecutive unproductive forced turns tolerated per "
            "tool-precondition entry. A turn is unproductive when the forced "
            "tool errored, was not called at all, or could not be forced "
            "because it was missing from that turn's advertised surface; a "
            "SUCCESSFUL call resets the counter. On exhaustion the run fails "
            "naming the tool and its last error — a precondition the caller "
            "asked for is never quietly skipped, and a tool that can never "
            "succeed can never loop the run."
        ),
    )
    run_tool_precondition_error_max_chars: int = Field(
        default=500,
        ge=1,
        le=4000,
        description=(
            "How much of the failing tool's error text is retained to name "
            "the last error in the run's tool-precondition failure reason. "
            "A tool result can be arbitrarily large; the failure reason "
            "travels on an SSE error frame and into the run row."
        ),
    )

 # ----- Read-back of declared files. A tool whose real output is FILES names
 # the paths in its result metadata; while one of them is unread the loop
 # forces the workspace read tool, so the caller cannot answer from the pointer
 # alone. Engages on the tool's own declaration, releases itself the moment the
 # last path is read, and is inert for a caller that reads what it was given.
    pending_reads_enabled: bool = Field(
        default=False,
        description=(
            "Master kill-switch for the declared-file read-back gate. When "
            "True, a tool result that declares paths the caller must open puts "
            "the workspace read tool in the provider's native tool_choice "
            "until every declared path has been read, so a caller cannot "
            "produce a final answer out of a one-line pointer to a file it "
            "never opened. When False the declarations are ignored entirely "
            "and behaviour is BIT-IDENTICAL to a build without the driver. "
            "Inert either way for a run in which no tool declares anything, "
            "and for a caller that reads its files unprompted. "
            "Tenant-overridable.\n\n"
            "Defaults to False because measurement said so. Forcing the whole "
            "file back into the caller's window undoes the saving that handing "
            "it a pointer was for: measured against the same scenarios, peak "
            "caller context rose from ~32k to ~41k, context compaction "
            "returned where there had been none, turn counts roughly doubled, "
            "and answer quality fell rather than rose. The mechanism works "
            "exactly as designed — its logs show the pending set draining — "
            "so what is wrong is the requirement behind it: a gate proving a "
            "caller used something should demand the smallest sufficient "
            "evidence, not the artifact. An installation that hands over small "
            "declarations, or whose callers cannot be trusted to read at all, "
            "may still want it; it is off by default because the shape it was "
            "built for is not the shape it helps."
        ),
    )
    pending_reads_max_forced_attempts: int = Field(
        default=3,
        ge=1,
        le=20,
        description=(
            "Consecutive forced read turns tolerated while the pending set "
            "clears nothing. A turn is unproductive when the forced read "
            "errored, opened a file nobody asked for, or was not called at "
            "all; any read that clears a pending path resets the counter, so "
            "a caller working through five declared files is never starved by "
            "its own progress. On exhaustion the gate RELEASES — the paths are "
            "abandoned, never forced again, and the agent has its whole tool "
            "surface back — because a file that cannot be read must cost a "
            "bounded number of turns and then nothing at all. A turn where the "
            "read tool was missing from the advertised surface is not charged: "
            "the model was never offered the tool."
        ),
    )
    pending_reads_max_paths: int = Field(
        default=32,
        ge=1,
        le=200,
        description=(
            "Largest number of unread declared paths tracked at once. The set "
            "accumulates across tools — a fan-out of three delegations owes "
            "three sets of reads — and rides on every run snapshot, so a tool "
            "returning a pathological list must not be able to grow it "
            "without limit. Declarations past the cap are dropped with a "
            "warning rather than silently, and the ones already pending are "
            "still enforced."
        ),
    )

 # ----- Token counting -----
    token_count_chars_per_token_latin: float = Field(
        default=4.0,
        gt=0.0,
        description="Latin-prose chars-per-token heuristic.",
    )
    token_count_chars_per_token_cyrillic: float = Field(
        default=2.5,
        gt=0.0,
        description="Cyrillic-prose chars-per-token heuristic.",
    )
    token_count_chars_per_token_cyrillic_json_escape: float = Field(
        default=1.2,
        gt=0.0,
        description="Cyrillic-in-JSON-escape chars-per-token (UTF-8 escaped — doubles).",
    )
    token_count_chars_per_token_cjk: float = Field(
        default=1.5,
        gt=0.0,
        description="CJK chars-per-token heuristic.",
    )
    token_count_chars_per_token_json_struct: float = Field(
        default=3.5,
        gt=0.0,
        description="JSON structural chars-per-token (braces, commas, …).",
    )
    token_count_image_tokens: int = Field(
        default=2_000,
        gt=0,
        description=(
            "Flat per-image token estimate for an image content block in the "
            "pre-flight history estimator. Image blocks carry only a blob ref "
            "(no text/content), so the cheap estimator cannot derive a size; a "
            "conservative fixed value (provider vision blocks are capped at a "
            "few thousand tokens) avoids under-counting and triggering "
            "compaction too late."
        ),
    )

 # ----- Tool retrieval / surface -----
    tool_retrieval_top_k: int = Field(
        default=12,
        gt=0,
        description="BM25 tool retrieval top-K (per-call surface).",
    )

 # ----- Compaction ergonomics -----
    compaction_keep_recent_turns: int = Field(
        default=4,
        gt=0,
        description="Last-N turns kept verbatim across compaction.",
    )
    compaction_tracked_tool_names: tuple[str, ...] = Field(
        default=("Write", "Edit", "Read", "Glob", "Grep"),
        description=(
            "Tool names whose calls a manual ``/compact`` checkpoint keeps as "
            "one-line facts after the messages around them are dropped. The "
            "default names a coding backend's file verbs; a backend in another "
            "domain names the calls whose bare fact must outlive compaction "
            "there (what a unit mined, built, or said in a simulation). "
            "Empty means the checkpoint keeps no per-call facts."
        ),
    )
    compaction_failed_max_retries: int = Field(
        default=2,
        ge=0,
        description="Max compaction retries before run transitions to FAILED.",
    )
    compaction_shed_reasoning_enabled: bool = Field(
        default=True,
        description=(
            "When true (default), Tier-1 compaction strips "
            "Message.reasoning_content (re-emitted chain-of-thought) from "
            "assistant turns older than compaction_keep_recent_turns. Prior "
            "turns' raw thinking is single-turn scaffolding the model never "
            "needs to re-read, but it is heavy uncompactable bloat on a small "
            "window. Set false to keep aged reasoning_content verbatim "
            "(kill-switch)."
        ),
    )
    compaction_bound_reference_blocks_enabled: bool = Field(
        default=True,
        description=(
            "When true (default), Tier-1 compaction may compact an OVER-BUDGET "
            "frozen reference block (a single-text non-tool message tagged "
            "compaction_reference=True in metadata — e.g. the executor's "
            "<environment_context>/<memory-context> bootstrap) older than "
            "compaction_keep_recent_turns to a blobbed placeholder, the same "
            "way Tier-1 sheds large tool results. The original task user turn "
            "and recent window are never eligible. Set false to leave reference "
            "blocks uncompactable (kill-switch)."
        ),
    )
    compaction_protect_first_user_turn: bool = Field(
        default=True,
        description=(
            "When true (default), the FIRST user-role turn in history (the "
            "original task) is never eligible for Tier-2 summarisation, so the "
            "verbatim task statement + constraints survive every compaction. "
            "Required for safety once the per-iteration compaction gate makes "
            "compaction fire often. Set false to allow the original task to be "
            "summarised (kill-switch)."
        ),
    )
    compaction_summariser_parallelism: int = Field(
        default=4,
        ge=1,
        le=16,
        description=(
            "How many summariser calls one compaction pass issues at a time. "
            "A pass over a long history is otherwise a chain of sequential "
            "calls, each of them seconds long, while the run sits in "
            "COMPACTING and produces nothing. The cap is what keeps that "
            "chain from becoming an unbounded fan-out at the provider."
        ),
    )
    compaction_summary_min_unit_tokens: int = Field(
        default=0,
        ge=0,
        description=(
            "A unit estimated below this many tokens is not sent to the "
            "summariser at all. A summariser writes a sentence or three "
            "whatever it is given, so a small unit comes back no smaller and "
            "the call bought nothing; the net-gain guard discards such a "
            "summary, but only after paying for it. 0 leaves the floor at the "
            "empty-wrapper size, which is the smallest it can ever be."
        ),
    )
    compaction_summary_min_words: int = Field(
        default=25,
        ge=1,
        description=(
            "Floor on the word budget handed to the per-turn summariser. The "
            "budget is derived from the size of the unit being replaced, and "
            "a small unit would otherwise be given a budget too small to hold "
            "the identifiers the summary must keep verbatim."
        ),
    )
    compaction_summary_tokens_per_word: int = Field(
        default=6,
        ge=1,
        description=(
            "Tokens of the original a summary may spend one word on: the word "
            "budget in the summariser prompt is the unit's estimated size "
            "divided by this. Room to keep identifiers, not enough to restate "
            "the turn."
        ),
    )
    compaction_fold_enabled: bool = Field(
        default=True,
        description=(
            "When true (default), a third pass folds runs of old compaction "
            "summaries and old operator turns into one consolidated summary. "
            "Tier-2 leaves one summary per tool batch and never re-summarises "
            "one, and it never summarises an operator turn at all, so a long "
            "session accumulates hundreds of small summaries and every "
            "message the operator ever sent until the window is full of them. "
            "The fold keeps the first user turn (the task) and the most recent "
            "operator turns verbatim; older operator instructions survive "
            "inside the fold as exact quotes. Set false to leave those "
            "messages alone (kill-switch)."
        ),
    )
    compaction_fold_min_messages: int = Field(
        default=8,
        ge=2,
        description=(
            "A contiguous run of foldable messages (old summaries and old "
            "operator turns) shorter than this is left alone: folding a "
            "handful of summaries costs a summariser call and frees little."
        ),
    )
    compaction_fold_min_tokens: int = Field(
        default=1_500,
        ge=0,
        description=(
            "A foldable run estimated below this many tokens is left alone, so "
            "an already-folded span is not folded again and again for nothing."
        ),
    )
    compaction_fold_keep_operator_turns: int = Field(
        default=4,
        ge=0,
        description=(
            "How many of the most recent operator turns are never folded, on "
            "top of the protected first user turn: the live instructions the "
            "model is acting on stay verbatim."
        ),
    )
    compaction_fold_max_spans_per_pass: int = Field(
        default=2,
        ge=1,
        description=(
            "How many runs the fold consolidates in one pass. The rest wait "
            "for the next pass, which is what keeps a single COMPACTING pause "
            "short on a history with many foldable runs."
        ),
    )
    compaction_placeholder_preview_chars: int = Field(
        default=240,
        ge=0,
        description=(
            "Max characters of a head/tail preview of the "
            "original content embedded in a compacted tool-result placeholder "
            "so the model knows what was shed and how to re-fetch it (the "
            "originating tool name is also embedded). 0 disables the preview "
            "(blob ref + sha + token count only). The full content remains in "
            "the blob store; v2 has no recall tool yet (deferred), so the "
            "preview + tool name are the recovery breadcrumb."
        ),
    )
    session_memory_running_summary_token_cap: int = Field(
        default=1900,
        ge=0,
        description=(
            "Drift control on the carried running summary. When the accumulated "
            "running summary grows past this ESTIMATED-token cap, the assembly "
            "truncates it (the next fold is still delta-only — the summary is "
            "NEVER recomputed from raw, which would reintroduce O(K²) cost + "
            "non-monotonic fidelity). Keeps the carried artifact compact so a "
            "long session's summary block cannot itself overflow the window. "
            "Measured in a DIFFERENT unit from ``session_memory_summary_max_tokens`` "
            "(the per-fold output budget): this cap is an estimate over stored "
            "characters, that budget is a hard provider-side token bound, and the "
            "writer must be able to re-emit everything this cap admits — on "
            "digest-dense content that costs up to ~3.4x this number in real "
            "tokens. THIS is the side of the pair that gives: the budget is "
            "pinned by how long a fold may take, so the cap follows from "
            "``cap x 3.4 <= 0.80 x budget`` and comes DOWN when the worst "
            "measured ratio rises. Raising it without raising the budget starves "
            "the fold's delta allowance and stalls the summary. See that field "
            "for the measured ratios. Note that a session whose summary "
            "sits AT this cap has stopped absorbing new facts: the fold still "
            "succeeds, but every addition displaces something. 0 disables the cap."
        ),
    )
    session_memory_tail_budget_fraction: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of the seed token budget reserved for the recent RAW tail "
            "(verbatim, tool-pair-safe) in "
            ":func:`protocore.runtime.context.session_memory.build_seed`. The "
            "remaining budget carries the head + running summary + ledger. A "
            "larger tail keeps more recent turns verbatim (higher fidelity, more "
            "tokens); a smaller tail leans harder on the summary."
        ),
    )
    session_memory_head_protect_messages: int = Field(
        default=3,
        ge=0,
        description=(
            "Number of leading messages of the FIRST run kept VERBATIM at the "
            "head of every seed (the original system + first user task + first "
            "assistant turn), never routed through summarisation. Anchors the "
            "original ask/constraints across the whole session (head-protection "
            "— anti lost-in-the-middle). 0 disables head protection."
        ),
    )
    session_memory_fold_min_tokens: int = Field(
        default=0,
        ge=0,
        description=(
            "Lazy-fold gate. The post-run structured-memory "
            "UPDATE skips the LLM running-summary call ENTIRELY (ledger-only, "
            "ZERO LLM cost) while the whole prior session still fits in the SEED's "
            "raw tail (``compaction_trigger_tokens * "
            "session_memory_tail_budget_fraction``) — the running summary would "
            "never be read, so summarising it is wasted load. The fold only "
            "summarises once the cumulative session tokens EXCEED this threshold. "
            "0 (default) DERIVES the threshold from the tail budget so the lazy "
            "gate and :func:`~protocore.runtime.context.session_memory.build_seed` "
            "agree by construction; a positive value overrides it with an explicit "
            "per-tenant token threshold. The deterministic artifact ledger is "
            "ALWAYS updated (it is free, no LLM), regardless of this gate."
        ),
    )

 # ----- Loop budget -----
    max_turns_per_run: int = Field(
        default=200,
        gt=0,
        description="Hard cap on assistant turns within one run.",
    )
    run_max_output_tokens_budget: int = Field(
        default=200_000,
        ge=0,
        description=(
            "Cumulative output-token budget for one run. When "
            "the running total of model output tokens (``engine.total_usage."
            "output_tokens``) exceeds this, the run is terminated FAILED with "
            "``reason='run_output_token_budget_exhausted'`` BEFORE it can keep "
            "spiralling into the provider context-length ceiling. This is a "
            "resource bound orthogonal to ``max_turns_per_run`` (a turn cap): a "
            "spiral that re-emits a large truncated Write burns output tokens "
            "every round, so the token budget trips faster than the turn cap on "
            "exactly the runaway-output failure mode. Default 200k output tokens "
            "is generous for any legitimate single run (≈50 full ~4k-token "
            "turns) yet bounds a runaway loop. Set to ``0`` to disable the "
            "budget entirely (turn cap + recovery budgets still bound the run). "
            "Per-tenant overridable."
        ),
    )
    max_data_nesting_depth: int = Field(
        default=MAX_DATA_NESTING_DEPTH,
        gt=0,
        description=(
            "Nesting-depth ceiling for data structures that arrive from the "
            "model — tool-call argument JSON and message / tool-result "
            "metadata. A walk that would go deeper raises a named, catchable error instead "
            "of running the interpreter out of stack: a RecursionError raised "
            "inside a Pydantic validator or a streaming JSON parser unwinds "
            "through the whole run and, at the point it is caught, carries no "
            "indication of where it came from. Applies wherever a "
            "LoopConstants snapshot is in scope; the pure contract "
            "validators and JSON utilities use the identical structural floor "
            "``protocore.constants.MAX_DATA_NESTING_DEPTH``, which is this "
            "field's default. Raise it only for a workload with genuinely deep "
            "payloads, and keep it well under the interpreter's recursion "
            "limit. Per-tenant overridable."
        ),
    )
    leader_tool_call_soft_cap: int = Field(
        default=80,
        ge=0,
        description=(
            "Advisory SOFT cap on the cumulative number of tool calls the LEADER "
            "agent makes in one run (NOT counting tool calls made inside "
            "subagents — those run in their own engine and are counted against "
            "``subagent_tool_call_soft_cap`` instead). When the leader's running "
            "total reaches this, an advisory wrap-up notice is appended to the "
            "tool result nudging the agent to finalize rather than start new "
            "work, and the notice is repeated on every further call past the "
            "cap. NEVER blocks execution — the hard bounds are "
            "``max_turns_per_run`` and ``run_max_output_tokens_budget``. 0 "
            "disables. Per-tenant overridable."
        ),
    )
    subagent_tool_call_soft_cap: int = Field(
        default=40,
        ge=0,
        description=(
            "Advisory SOFT cap on the cumulative number of tool calls a SUBAGENT "
            "makes within its delegated run. On reaching it the subagent gets an "
            "advisory wrap-up notice nudging it to call SubmitAnswer, and the "
            "notice is repeated on every further call past the cap. "
            "Separate from ``leader_tool_call_soft_cap`` so "
            "subagent budgets are tuned independently. NEVER blocks. 0 disables. "
            "Per-tenant overridable."
        ),
    )
    soft_stop_enabled: bool = Field(
        default=True,
        description=(
            "Master switch for the run wind-down. When a run reaches a bound — "
            "the cumulative tool-call budget, the turn cap, the output-token "
            "budget, the wall-clock deadline, or an upstream that stopped "
            "answering — the runtime notifies the model, REMOVES every tool "
            "from its surface except the terminal one (plus the artifact sealer "
            "while an artifact is open), requires the final answer in prose via "
            "``finalize_prose_gate_enabled``, and ends the run with "
            "``stop_reason='soft_stop'``. All five bounds take the same path, so "
            "'the run was cut short' means one thing and is observable as four "
            "``state_changed`` reasons in order: soft_stop_notified, "
            "soft_stop_tools_withdrawn, soft_stop_finalized, then the terminal "
            "stop. The withdrawal is a change to the tool SURFACE, not advice in "
            "a prompt: the model is not shown a schema it must not call. Set "
            "False and every bound reverts to terminating the run where it is, "
            "with whatever the model had produced by then. Per-tenant "
            "overridable."
        ),
    )
    soft_stop_max_turns: int = Field(
        default=3,
        gt=0,
        description=(
            "Assistant turns granted to the wind-down once it starts, on top of "
            "whatever budget was already spent. It has to be more than one: the "
            "model may need a turn to write the answer, and the prose gate may "
            "spend one refusing a terminal call that arrived without it. Too "
            "large and a run that hit its turn cap keeps going under a different "
            "name; 3 is enough for notify → answer → finalize with one turn of "
            "slack. Only consulted when ``soft_stop_enabled``."
        ),
    )
    soft_stop_notice_text: str = Field(
        default=(
            "[internal control — not part of the reply] The run has reached its "
            "budget ({cause}) and is now closing. Every tool except the "
            "finalizing one has been removed from your surface, so no further "
            "work is possible. Write your final response to the user now, as an "
            "ordinary assistant message in plain prose, in the language of the "
            "conversation: what you did, what you found, and where the results "
            "are. State plainly what is unfinished rather than implying the task "
            "is complete. Then call the terminal tool to end the run. "
            "[внутреннее управление — не часть ответа] Выполнение достигло "
            "предела ({cause}) и сейчас завершается. Все инструменты, кроме "
            "завершающего, убраны из вашей поверхности, продолжать работу "
            "нельзя. Напишите финальный ответ пользователю сейчас — обычным "
            "сообщением ассистента, простым текстом, на языке диалога: что вы "
            "сделали, что выяснили и где лежат результаты. Прямо укажите, что "
            "осталось незавершённым, а не создавайте впечатление выполненной "
            "задачи. Затем вызовите терминальный инструмент, чтобы завершить "
            "выполнение."
        ),
        description=(
            "Bilingual (EN+RU) notice injected as one user turn when the wind-down "
            "starts. ``{cause}`` is substituted with which bound was reached "
            "(tool_call_budget / max_turns / output_token_budget / deadline / "
            "provider_error). Bilingual for the same reason the prose-gate repair "
            "text is: a model told to wrap up in a language the conversation is "
            "not in tends to switch languages before it wraps up. Framed as "
            "internal control so a weak model cannot paraphrase it into the "
            "visible answer. Empty string suppresses the message; the withdrawal "
            "and the state events still happen, and the model is then left to "
            "infer the stop from a surface that no longer carries its tools. "
            "Per-tenant overridable."
        ),
    )
    run_tool_call_ledger_max_entries: int = Field(
        default=500,
        ge=0,
        description=(
            "How many dispatched tool calls one run records in its ledger — "
            "the ordered ``{seq, name, ok}`` list the runtime writes AT the "
            "dispatch. It exists because history cannot answer the question: "
            "compaction replaces a turn with prose about it and keeps none of "
            "the tool names, so a run long enough to be compacted loses the "
            "record of its own work, and the user is shown whatever handful of "
            "calls happened to survive. Past this many entries the tail is "
            "dropped and a truncation flag is set, so the ledger stays bounded "
            "and a reader is never misled about it being complete. 500 covers "
            "any run that is not already pathological. 0 keeps no ledger at "
            "all. Per-tenant overridable."
        ),
    )
    tool_timeout_seconds: int = Field(
        default=90,
        gt=0,
        description="Per-tool dispatch wall-time cap.",
    )
    tool_cancel_drain_seconds: float = Field(
        default=2.0,
        gt=0,
        description=(
            "#6 cancel propagation — bounded wait the core tool dispatcher "
            "(``tool_dispatch.py``) gives a cancelled in-flight tool task to "
            "unwind after a run-level cancel fires. When the per-run cancel "
            "``asyncio.Event`` (``RunScopedState.cancel_event``) is SET while a "
            "tool is mid-flight, the dispatcher cancels the tool task and waits "
            "up to this long for it to settle (so the ``Agent`` tool's subagent "
            "teardown can run) before raising ``CancelledError`` to unblock the "
            "leader. Bounds the worst-case extra delay between cancel and the "
            "leader unblocking. Mirrors the subagent runner's stale-abort drain."
        ),
    )

 # ----- Skill / context ratios -----
    loaded_skills_ratio: float = Field(
        default=0.04,
        gt=0.0,
        le=1.0,
        description="Loaded skill bodies budget as fraction of context window.",
    )
    tool_definitions_ratio: float = Field(
        default=0.05,
        gt=0.0,
        le=1.0,
        description="Tool definitions budget as fraction of context window.",
    )
    user_context_ratio: float = Field(
        default=0.01,
        gt=0.0,
        le=1.0,
        description="User context block (cwd/env/project rules) budget fraction.",
    )
    max_skills_per_run: int = Field(
        default=4,
        gt=0,
        description="Hard cap on loaded skill bodies per run.",
    )

 # ----- Compaction LLM call caps -----
    compaction_summary_max_output_tokens: int = Field(
        default=512,
        gt=0,
        description=(
            "Hard cap on the compaction-LLM's output for the per-turn "
            "summariser call. Small enough that the summary fits in the "
            "system_prompt budget."
        ),
    )
    compaction_summary_temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description=(
            "Sampling temperature for the compaction-LLM summariser. "
            "Low value (0.2) keeps summaries deterministic + consistent."
        ),
    )
    compaction_summary_string_max_chars: int = Field(
        default=1024,
        gt=0,
        description=(
            "JSON-schema ``maxLength`` cap on the summary string field. "
            "Surfaced to XGrammar — enforces output bound at decode time."
        ),
    )
    compaction_fold_max_output_tokens: int = Field(
        default=3_000,
        gt=0,
        description=(
            "Hard cap on the compaction-LLM's output for one fold summary. "
            "Larger than the per-turn cap because a fold summary stands for "
            "many turns at once; a fold that runs out of budget comes back "
            "truncated and is discarded, so the pass paid for nothing."
        ),
    )
    compaction_fold_summary_target_words: int = Field(
        default=500,
        ge=1,
        description=(
            "The word target stated in the fold prompt. It is what makes the "
            "summariser merge repeated checks into one line and keep only the "
            "last known state of each thing, rather than writing until the "
            "output cap stops it mid-sentence."
        ),
    )

 # ----- Skill body capping -----
    skill_body_chars_per_token: int = Field(
        default=4,
        gt=0,
        description=(
            "Chars-per-token heuristic used to soft-cap a loaded skill "
            "body to the per-skill token budget (Latin-prose baseline)."
        ),
    )

 # ----- LLM output cap -----
    llm_output_max_tokens_ratio: float = Field(
        default=0.25,
        gt=0.0,
        le=1.0,
        description=(
            "Fraction of ``max_context`` used as ``LLMRequest.max_tokens`` "
            "for assistant-stream calls (default 0.25 — i.e. quarter of "
            "the window reserved for output)."
        ),
    )
    pinned_tool_max_count: int = Field(
        default=15,
        gt=0,
        description=(
            "Maximum number of pinned tools (always-include) carried into "
            "the tool pool. Caps cache-prefix bloat from "
            "ToolSearch-pinned tools."
        ),
    )
    tool_surface_forced_pins: tuple[str, ...] = Field(
        default=("Agent", "Read", "Write", "Edit", "Bash", "Glob", "Grep"),
        description=(
            "Core tools ALWAYS present in the per-turn surface, bypassing the "
            "BM25 clip (cause-#3 fix). Universal + dashboard-tunable per "
            "tenant. The host builds ``ToolVisibilityPolicy.forced_pinned`` "
            "from this list so delegation plus the six core file tools survive "
            "even a Russian prompt that shares zero tokens with the English tool names "
            "(measured: a Russian query scores zero against every "
            "BM25 score 0.0 → surface collapses to ZERO tools → the model "
            "proses a leaked ``<finalization_contract>`` instead of acting). "
            "Pinning these restores ``[Agent, Bash, Edit, Glob, Grep, Read, Write]`` "
            "for every prompt — keeping direct work and subagent dispatch discoverable "
            "without prompt-level tool-routing instructions. "
            "Set empty to disable the floor (NOT recommended; the catastrophic "
            "no-tools-on-RU failure recurs)."
        ),
    )
    agent_deep_plan_include_summary: bool = Field(
        default=False,
        description=(
            "Deep-mode SGR ``plan`` tool: when True the forced plan schema "
            "gains a short human-readable ``reasoning_summary`` (<=280 chars) "
            "field, surfaced in the ``reasoning_step`` event for a one-line UI "
            "trace. Default False — native CoT already carries the 'why' "
            "(~410-430 chars measured), so the lean schema "
            "{plan, next_tool, task_complete} is the cheapest enforcing shape "
            "(195 plan tokens vs 252 full)."
        ),
    )
    agent_finalize_tool_as_terminal: bool = Field(
        default=True,
        description=(
            "When True, the host executor "
            "sets ``QueryEngineConfig.expected_terminal_tool='Finalize'`` for "
            "both fresh-start and resumed engines, force-pins the ``Finalize`` "
            "tool into the leader surface, AND arms the terminal-tool nudge "
            "(forces ``terminal_tool_nudge_enabled`` True for the run) so a "
            "prose final attempt without a prior ``Finalize`` is repaired and "
            "the gate latches on ``Finalize``. Default True: a model that "
            "recalls perfectly then narrates "
            "'Now let me write this file' and fires 0 tools otherwise slips "
            "through the no-tool end_turn branch and is scored a silent empty. "
            "Arming the nudge by default closes that gap universally — strong "
            "models already finish via the terminal tool, so it is a no-op for "
            "them. Per-tenant overridable (set False to restore the legacy "
            "``leader_config.expected_terminal_tool`` contract). Distinct from "
            "``finalization_gate_enabled`` (which only verifies declared "
            "deliverables); this flag controls the terminal-tool MECHANISM. "
            "The executor arms the nudge so the contract is self-contained — "
            "the standalone core default of ``terminal_tool_nudge_enabled`` "
            "stays False."
        ),
    )

 # ----- Tool-dispatch consecutive same-error cap -----
    tool_dispatch_consecutive_error_cap: int = Field(
        default=4,
        ge=2,
        description=(
            "Per-run cap on consecutive identical tool errors. "
            "Empirical research found the leader can retry an "
            "IDENTICAL failed tool call up to 200 times (e.g. Write storms). Default "
            "4 allows up to 3 retries; the 4th identical (tool_name, "
            "normalised error) tuple is intercepted by the dispatcher and "
            "surfaced as ``DispatchErrorKind.consecutive_error_cap`` with "
            "guidance to try a different tool or argument shape. The streak "
            "resets when the (tool, error_signature) changes or when the "
            "tool succeeds. Floor ``ge=2`` prevents pathological values "
            "(``1`` would reject the very first error)."
        ),
    )
    tool_dispatch_string_type_terminal_cap: int = Field(
        default=3,
        ge=2,
        description=(
            "Separate per-run cap on consecutive Pydantic ``string_type`` "
            "validation errors on the SAME tool. Analysis showed models "
            "can loop for extended periods emitting ``Write {content: [array]}`` "
            "repeatedly. The mainline coercion validator on "
            "``WriteInput.content`` / ``AppendFileInput.content`` / "
            "``BashInput.command`` handles common shapes silently, but if "
            "a model produces an uncoercible value (e.g. ``content=None`` "
            "after stripping required field) we still get ``string_type``. "
            "Once the streak crosses this cap the dispatcher rewrites the "
            "error to ``DispatchErrorKind.consecutive_error_cap`` with a "
            "stronger terminal guidance string instructing the model to "
            "stop retrying the same shape. Independent of "
            "``tool_dispatch_consecutive_error_cap`` so operators can tune "
            "the schema-shape failure mode separately from generic "
            "execution loops. Default 3 ensures the string_type-specific "
            "TERMINAL guidance fires BEFORE the generic "
            "``tool_dispatch_consecutive_error_cap`` (default 4) wraps "
            "the error with vague ``try a different tool or argument "
            "shape`` guidance. Floor ``ge=2`` prevents pathological values."
        ),
    )
    sandbox_down_system_message_threshold: int = Field(
        default=3,
        gt=0,
        description=(
            "How many failures in a row that say the way out to a tool is "
            "down — not that the call was wrong — before the run tells the "
            "model to reach its goal by another route. Which failures those "
            "are is the host's verdict, not a wording this runtime reads: it "
            "asks the bound classifier, and a run of such verdicts on one "
            "tool collapses to a single error signature however differently "
            "each attempt was worded. Without that collapse a run can spend "
            "its whole budget retrying a tool whose transport is gone, "
            "dozens of failed calls deep, because every failure looks new to "
            "a counter that tells errors apart by their text. Independent of "
            "``tool_dispatch_consecutive_error_cap`` so operators can fire "
            "this nudge earlier than the generic cap. The streak resets on a "
            "successful tool call or on any other error, so a brief outage "
            "does not lock the model out of the tool for the rest of the run."
        ),
    )
    max_consecutive_tool_errors: int = Field(
        default=3,
        ge=2,
        description=(
            "Repeated-tool-error circuit breaker — per-run cap on consecutive failures of the "
            "SAME tool with the SAME error CLASS (``DispatchErrorKind``) before "
            "the runtime HARD-STOPS offering AND allowing that tool for the "
            "rest of the run. Distinct from "
            "``tool_dispatch_consecutive_error_cap`` (which only rewrites the "
            "surfaced error to ``consecutive_error_cap`` and tells the model to "
            "'try a different tool/argument' — useless when the tool can NEVER "
            "succeed, e.g. the ``/project`` Read/Grep/Glob/List tools raising "
            "``project knowledge is not attached`` on a non-project session). "
            "Once a tool crosses this cap the core loop adds it to a per-run "
            "circuit-broken set (unioned into ``ToolVisibilityPolicy.blocked`` "
            "so it vanishes from the advertised surface AND is denied at "
            "dispatch) and injects ONE bounded corrective user turn forcing "
            "convergence (answer from the conversation / finalize). Universal — "
            "also dampens any repeated hard-error storm, not just ``/project``. "
            "The streak resets when the (tool, error_class) changes or the tool "
            "succeeds. Default 3 trips on the 3rd identical failure (after 2 "
            "retries); floor ``ge=2`` prevents tripping on the very first "
            "error. NOT catalog/``_FIELD_MAP``-backed (mirrors "
            "``tool_dispatch_consecutive_error_cap``): the Pydantic default "
            "governs at runtime, so changing it here takes effect without a "
            "catalog migration."
        ),
    )

 # ----- LLM recovery loops -----
    max_output_recovery_rounds: int = Field(
        default=3,
        ge=0,
        description=(
            "Max consecutive ``finish_reason='length'`` recovery rounds "
            "before terminal FAILED. Each round synthesises a "
            '\"Resume directly from where you left off, without preamble or '
            'repetition.\" continuation prompt and re-opens the LLM stream. '
            "Set to ``0`` to disable recovery entirely. Shared counter — "
            "text-only truncation AND mid-tool-call truncation each debit the "
            "same per-message budget."
        ),
    )
    tool_call_max_input_chunk_bytes: int = Field(
        default=1024,
        gt=0,
        description=(
            "Soft hint embedded in the truncated-tool-call recovery message. "
            "When a tool call is truncated mid-stream, the agent is instructed "
            "to split outputs into chunks of this size. "
            "Default tightened from 4096 → 1024 after analysis showed Qwen3.5's per-turn "
            "output budget on planning / long_context prompts is "
            "~3000-4000 tokens, so a single Write with ``content`` > 2 KB "
            "reliably truncates. 1024 chars (~20 lines of typical code or "
            "markdown) almost always fits in one tool call and gives the "
            "model a concrete, achievable chunk target."
        ),
    )
    tool_call_max_truncation_recoveries_per_message: int = Field(
        default=4,
        gt=0,
        description=(
            "Per-message budget for the ``args_partial_truncated`` + "
            "``finish_reason='stop'`` recovery branch. Without a cap, a model "
            "stuck in a ``{`` + stop loop would consume one outer iteration per "
            "recovery round until ``max_turns_per_run`` fires. Mirrors the "
            "``max_output_recovery_rounds`` guard pattern. When exhausted, the "
            "loop emits a terminal "
            ":class:`~protocore.contracts.errors.LLMProviderError` so the "
            "agent does not infinite-loop on a misbehaving model. Counter lives "
            "on :class:`~protocore.runtime.query_engine.QueryEngine` as "
            "``_tool_call_truncated_recovery_count`` and resets every new "
            "message via ``reset_recovery_state``. Default 4 allows a fresh "
            "~3-chunk split after the model internalises the directive recovery "
            "message."
        ),
    )
    write_chunk_token_budget: int = Field(
        default=1500,
        gt=0,
        description=(
            "Safe per-call CONTENT token budget the runtime tells the model to "
            "use when it must chunk a large file write. When a mutation tool "
            "call (Write/Edit/AppendFile) is truncated at the output cap, the "
            "chunk-recovery message instructs the model to write the file as "
            "Write(header, <= this many content tokens) -> AppendFile(chunk) "
            "-> FinalizeFile. Empirically a single full-document Write of a "
            "~30 KB article truncates at a 4096 output cap, while chunked "
            "Write+AppendFile completes reliably; this budget is the per-chunk "
            "content target so a chunk + its JSON envelope fits well under any "
            "sane output cap (~1500 tokens ≈ 6 KB of text). Surfaced as "
            "``{chunk_budget_tokens}`` in the recovery message; "
            "tenant-overridable so operators can tune it to their model's "
            "per-call output ceiling."
        ),
    )
    truncation_chunk_recovery_message_en: str = Field(
        default=(
            "Your `{tool_name}` call writing to `{path}` was TRUNCATED — the "
            "model output budget ran out before the file content finished, so "
            "the call is INCOMPLETE and was NOT applied (the file was not "
            "written). The content is too large for one call. Action — write "
            "`{path}` in CHUNKS using this exact protocol, and DO NOT retry the "
            "same oversized `{tool_name}`:\n"
            "1. `Write(path=\"{path}\", content=<first chunk, at most "
            "~{chunk_budget_tokens} tokens>)` — the opening of the file.\n"
            "2. `AppendFile(path=\"{path}\", content=<next chunk>)` — repeat for "
            "each subsequent chunk until the whole file is emitted.\n"
            "3. `FinalizeFile(path=\"{path}\")` — once, when the file is "
            "complete.\n"
            "Re-generate the content from the beginning, split across chunks "
            "now.{mid_chunked_note_en}"
        ),
        description=(
            "English half of the structured chunk-recovery message the runtime "
            "injects when a mutation tool call is truncated at the output cap "
            "(under ANY ``finish_reason``). Names the PATH, the per-call chunk "
            "budget (``{chunk_budget_tokens}`` from ``write_chunk_token_budget``), "
            "and the explicit Write -> AppendFile -> FinalizeFile protocol. "
            "Placeholders: ``{tool_name}``, ``{path}``, ``{chunk_budget_tokens}``, "
            "``{mid_chunked_note_en}`` (a stronger 'you already started chunking "
            "this file, continue with AppendFile' directive filled in on a repeat "
            "truncation of the same path). Production is RU+EN so both halves are "
            "emitted together (EN first). Tenant-overridable."
        ),
    )
    truncation_chunk_recovery_message_ru: str = Field(
        default=(
            "Ваш вызов `{tool_name}`, записывающий в `{path}`, был ОБРЕЗАН — "
            "у модели закончился output budget до того, как контент файла "
            "завершился, поэтому вызов НЕПОЛНЫЙ и НЕ был применён (файл не "
            "записан). Контент слишком большой для одного вызова. Действие — "
            "запишите `{path}` ЧАНКАМИ по этому протоколу и НЕ повторяйте тот же "
            "огромный `{tool_name}`:\n"
            "1. `Write(path=\"{path}\", content=<первый чанк, максимум "
            "~{chunk_budget_tokens} токенов>)` — начало файла.\n"
            "2. `AppendFile(path=\"{path}\", content=<следующий чанк>)` — "
            "повторяйте для каждого следующего чанка, пока весь файл не будет "
            "записан.\n"
            "3. `FinalizeFile(path=\"{path}\")` — один раз, когда файл завершён.\n"
            "Сгенерируйте контент заново с начала, разбив на чанки."
            "{mid_chunked_note_ru}"
        ),
        description=(
            "Russian half of the structured chunk-recovery message. Same "
            "placeholders as ``truncation_chunk_recovery_message_en`` "
            "(``{tool_name}``, ``{path}``, ``{chunk_budget_tokens}``, "
            "``{mid_chunked_note_ru}``). Emitted together with the EN half "
            "(EN first, RU second). Tenant-overridable."
        ),
    )
    truncation_chunk_recovery_mid_chunked_note_en: str = Field(
        default=(
            " NOTE: you have ALREADY started chunking `{path}` — do NOT start "
            "over with `Write`; continue from where you stopped using "
            "`AppendFile`, then `FinalizeFile`."
        ),
        description=(
            "English ``{mid_chunked_note_en}`` directive appended to the "
            "chunk-recovery message when the SAME path is truncated AGAIN "
            "after chunking already began (tracked per-run). Steers the model "
            "to AppendFile instead of re-Writing. Empty on the first truncation "
            "of a path. Tenant-overridable."
        ),
    )
    truncation_chunk_recovery_mid_chunked_note_ru: str = Field(
        default=(
            " ВНИМАНИЕ: вы УЖЕ начали разбивать `{path}` на чанки — НЕ "
            "начинайте заново с `Write`; продолжайте с того места, где "
            "остановились, через `AppendFile`, затем `FinalizeFile`."
        ),
        description=(
            "Russian ``{mid_chunked_note_ru}`` directive "
            "(twin of ``truncation_chunk_recovery_mid_chunked_note_en``)."
        ),
    )
    truncation_chunk_recovery_repeat_budget_divisor: int = Field(
        default=2,
        ge=1,
        description=(
            "when a content-mutation write to a path is "
            "truncated AGAIN before any chunk has SUCCESSFULLY been written (no "
            "Write/AppendFile to that path has landed yet), the recovery message "
            "keeps the FIRST-message protocol (start with `Write(header)`) — it "
            "must NEVER tell the model to `AppendFile` a file that does not exist "
            "yet — but LOWERS the header chunk budget so the next header attempt "
            "is smaller than the one that just truncated. The surfaced "
            "``{chunk_budget_tokens}`` is ``write_chunk_token_budget`` integer-"
            "divided by this divisor once per prior no-success recovery prompt "
            "for that path, floored at "
            "``truncation_chunk_recovery_min_chunk_token_budget``. Default 2 "
            "halves the budget each repeat. 1 disables the reduction (constant "
            "budget). Tenant-overridable."
        ),
    )
    truncation_chunk_recovery_min_chunk_token_budget: int = Field(
        default=256,
        gt=0,
        description=(
            "floor for the lowered header chunk budget when "
            "a write keeps truncating before any successful chunk (see "
            "``truncation_chunk_recovery_repeat_budget_divisor``). The surfaced "
            "``{chunk_budget_tokens}`` never drops below this, so the model is "
            "always asked for a non-trivial first chunk. Must be > 0 and is "
            "clamped to at most ``write_chunk_token_budget``. Tenant-overridable."
        ),
    )
 # ----- Large-file convergence (runtime-driven stall-aware
 # forced convergence). A weak local model at a small output cap writes one
 # header then idle-inspects via non-mutation calls, never appending/finalizing.
 # The runtime detects stalled BYTE PRODUCTION and FORCES the next tool
 # (AppendFile to drive content, FinalizeFile to seal). Empirically, completion
 # improved significantly; forcing (not wording) is the active ingredient.
 # Universal + a strict no-op for a model that produces bytes every turn.
    longfile_convergence_enabled: bool = Field(
        default=True,
        description=(
            "Master kill-switch for the runtime-driven large-file convergence "
            "driver. When False the engine NEVER detects stalls, NEVER forces "
            "AppendFile/FinalizeFile, and the truncation recovery message keeps "
            "its original discard-redo wording — i.e. behaviour is BIT-IDENTICAL "
            "to before this feature. When True the runtime drives a stalled "
            "large-file write to completion (see the ``longfile_*`` knobs below). "
            "Universal: a model that adds bytes every turn never stalls, so the "
            "driver is inert for strong models "
            "even when enabled. Tenant-overridable."
        ),
    )
    longfile_stall_turns: int = Field(
        default=2,
        gt=0,
        description=(
            "stall threshold: the number of consecutive "
            "assistant turns with NO byte-adding mutation (a Write/AppendFile "
            "that actually grew the file) while the active artifact is below "
            "its expected-complete floor (``longfile_expected_floor_bytes``) "
            "before the runtime forces the next tool. The detector keys on BYTE "
            "PRODUCTION, NOT append-count and NOT the prose path (both are "
            "bypassed by the model's 'header-then-idle-inspect' shape). Default "
            "2 allows ONE self-correction turn before forcing (gentler/more "
            "universal than the probe's aggressive K=1, which is also valid). "
            "Tenant-overridable."
        ),
    )
    longfile_max_forced_appends: int = Field(
        default=8,
        gt=0,
        description=(
            "per-run cap on forced ``tool_choice="
            "AppendFile`` rounds. Once this many forced appends have fired the "
            "runtime stops forcing appends (and may force a single FinalizeFile "
            "if the file is at/above floor). Subordinate to "
            "``max_turns_per_run`` so the convergence driver can NEVER spin. "
            "Tenant-overridable."
        ),
    )
    longfile_max_forced_finalizes: int = Field(
        default=2,
        gt=0,
        description=(
            "per-run cap on forced ``tool_choice="
            "FinalizeFile`` rounds (plateau-driven OR done-with-content-driven "
            "OR the terminal seal after enough forced appends). Bounds the seal "
            "so a model that ignores the forced finalize cannot loop. "
            "Tenant-overridable."
        ),
    )
    longfile_plateau_delta_fraction: float = Field(
        default=0.25,
        gt=0.0,
        description=(
            "byte-plateau threshold. Once the file is "
            "at/above its expected floor AND at least "
            "``longfile_plateau_min_mutations`` successful byte-adding "
            "mutations have landed, a forced FinalizeFile fires when the most "
            "recent mutation delta falls below this fraction of the "
            "running-mean delta across those mutations (the body has stopped "
            "growing). Tenant-overridable."
        ),
    )
    longfile_plateau_min_mutations: int = Field(
        default=2,
        gt=0,
        description=(
            "minimum number of successful byte-adding "
            "mutations required before the byte-plateau finalize trigger "
            "(``longfile_plateau_delta_fraction``) is allowed to fire. Prevents "
            "a single early write from being read as a plateau. "
            "Tenant-overridable."
        ),
    )
    longfile_expected_floor_bytes: int = Field(
        default=4096,
        gt=0,
        description=(
            "expected-complete byte floor. Below it the "
            "artifact is treated as INCOMPLETE so the stall detector keeps "
            "driving forced AppendFile; at/above it a forced FinalizeFile is "
            "PERMITTED (subject to the empty-finalize guard "
            "``longfile_min_finalize_fraction``). A conservative universal "
            "default of 4096 bytes (the smallest validated task floor) — a "
            "merely-truncated single header is below this, so a partial write "
            "is never mistaken for a complete file. Tenant-overridable per the "
            "deliverable sizes a tenant expects."
        ),
    )
    longfile_min_finalize_fraction: float = Field(
        default=1.0,
        gt=0.0,
        description=(
            "the HARD empty-finalize guard fraction (the "
            "validated edge). A forced FinalizeFile is NEVER issued unless "
            "``file_bytes >= max(1, longfile_expected_floor_bytes * "
            "longfile_min_finalize_fraction)``. This blocks the failure the "
            "probe exposed (forced-finalize firing on a 0-byte / below-floor "
            "file → an empty deliverable). Default 1.0 requires the file to be "
            "at/above the full floor before any forced seal; lower it (e.g. "
            "0.75) only if a tenant wants to seal slightly-under-floor files. "
            "Tenant-overridable."
        ),
    )
    longfile_tail_anchor_chars: int = Field(
        default=200,
        gt=0,
        description=(
            "number of trailing characters of the on-disk "
            "file read into the INCOMPLETE continue/recovery message as a "
            "'tail anchor' so the model knows EXACTLY where to continue (and "
            "not repeat what is already written). Tenant-overridable."
        ),
    )
    longfile_max_appends_per_path: int = Field(
        default=40,
        gt=0,
        description=(
            "per-path forced-append circuit-breaker. The "
            "per-path counter tallies ALL successful AppendFile calls (forced "
            "AND voluntary); once it reaches this value the convergence DRIVER "
            "stops FORCING appends to that path (it seals the file if it is "
            "at/above floor, else stops driving and lets the run end). It bounds "
            "the FORCED-driver contribution to the self-loop the regression "
            "exposed; it does NOT hard-reject the model's own voluntary appends "
            "at dispatch (the truncation gate removes the small-file "
            "voluntary flood at the root). Default 40 is generous but finite — a "
            "legitimate chunked large-file write needs far fewer. Tenant-overridable."
        ),
    )
    longfile_continue_message_en: str = Field(
        default=(
            "Your write to `{path}` is INCOMPLETE — the file currently holds "
            "{file_bytes} bytes ({file_lines} lines) and it is NOT finished. Do "
            "NOT stop, do NOT declare it complete, and do NOT call FinalizeFile "
            "yet. The current tail of the file is:\n---\n{tail}\n---\n"
            "Continue from EXACTLY where that tail ends by calling "
            "`AppendFile(path=\"{path}\", content=<the NEXT part of the "
            "content>)`, and finish the file. Do NOT repeat any text already "
            "written and do NOT start over with `Write`. Call "
            "`FinalizeFile(path=\"{path}\")` ONLY after the ENTIRE file has "
            "been written."
        ),
        description=(
            "English half of the INCOMPLETE continue "
            "message the convergence driver injects on a stall. States the file "
            "is INCOMPLETE with its current bytes/lines, FORBIDS "
            "stopping/declaring done, and includes the on-disk TAIL ANCHOR "
            "(``{tail}``, last ``longfile_tail_anchor_chars`` chars) so the "
            "model continues from exactly where it stopped. It does NOT "
            "state a byte target (stating a target made a weak model pad a "
            "legitimately small file); it just says 'continue and finish the "
            "file'. It NEVER says 'safe on disk' — that wording is a known "
            "failure (the model reads 'safe' as 'done' and stops producing). "
            "Placeholders: ``{path}``, "
            "``{file_bytes}``, ``{file_lines}``, ``{tail}`` (no "
            "``{expected_floor_bytes}``). Emitted with the RU half (EN first), "
            "per the multilingual rule. ``_FIELD_MAP``-only (no "
            "catalog row, matching the truncation-message precedent). "
            "Tenant-overridable."
        ),
    )
    longfile_continue_message_ru: str = Field(
        default=(
            "Ваша запись в `{path}` НЕПОЛНАЯ — сейчас файл содержит "
            "{file_bytes} байт ({file_lines} строк), поэтому он ещё НЕ "
            "дописан. НЕ останавливайтесь, НЕ объявляйте его завершённым и пока "
            "НЕ вызывайте FinalizeFile. Текущий хвост файла:\n---\n{tail}\n---\n"
            "Продолжайте РОВНО с того места, где заканчивается хвост, вызвав "
            "`AppendFile(path=\"{path}\", content=<следующая часть "
            "содержимого>)`, и допишите файл до конца. НЕ повторяйте уже "
            "записанный текст и НЕ начинайте заново с `Write`. Вызовите "
            "`FinalizeFile(path=\"{path}\")` ТОЛЬКО после того, как ВЕСЬ файл "
            "будет записан."
        ),
        description=(
            "Russian half of the INCOMPLETE continue "
            "message (twin of ``longfile_continue_message_en``; same "
            "placeholders — no ``{expected_floor_bytes}`` byte target). "
            "Emitted together with the EN half (EN first, RU second). "
            "``_FIELD_MAP``-only (no catalog row). Tenant-overridable."
        ),
    )
    llm_provider_chain_max_advances: int = Field(
        default=2,
        ge=0,
        description=(
            "How many times one run may step down its model priority list "
            "after a runtime failure. Each step discards the prefix cache built "
            "on the current provider and starts the next one cold, and every "
            "swap invalidates the reasoning payloads on earlier assistant "
            "turns, so the count is deliberately small. Only failures a "
            "different endpoint could plausibly serve advance the chain — a "
            "rate limit, a timeout, a 5xx, a bad key, an exhausted balance, a "
            "model the endpoint does not carry. A prompt that overflows the "
            "context window, a body that is too large, a malformed request and "
            "a provider's policy refusal all keep their existing recovery and "
            "never advance. 0 disables stepping entirely while still skipping "
            "disabled and plan-restricted models before the run starts."
        ),
    )

 # ----- Transient LLM error retry (429 / timeout) -----
    llm_transient_error_retry_max_attempts: int = Field(
        default=2,
        ge=0,
        description=(
            "Bounded in-place retries for a TRANSIENT upstream LLM failure — a "
            "429 rate-limit (``LLMRateLimitError``) or a request/stream timeout "
            "(``LLMTimeoutError``) — raised on the assistant stream. These "
            "classes are retryable per the error classifier, so the loop first "
            "steps down the run's model priority list (when one is configured "
            "and the advance budget is not spent), and otherwise re-opens the "
            "SAME stream up to this many times with a backoff between attempts "
            "before going terminal. That order is deliberate: a healthy sibling "
            "provider beats sleeping on a sick one. The retry streak resets "
            "after any successful assistant stream, so the bound applies per "
            "consecutive-failure streak, not per run. 0 disables in-place retry "
            "(a transient error then relies solely on the chain, and goes "
            "terminal once that is unavailable/exhausted)."
        ),
    )
    llm_transient_error_retry_backoff_base_seconds: float = Field(
        default=1.0,
        ge=0.0,
        description=(
            "Base backoff (seconds) before the FIRST transient-error retry "
            "(429 / timeout). Attempt N waits "
            "``min(base * 2 ** (N - 1), llm_transient_error_retry_backoff_max_seconds)``. "
            "A server-stated ``Retry-After`` on the classified error, when "
            "present, takes precedence but is still clamped by the max ceiling. "
            "0.0 retries immediately (no pause). Only consulted when "
            "``llm_transient_error_retry_max_attempts`` > 0."
        ),
    )
    llm_transient_error_retry_backoff_max_seconds: float = Field(
        default=8.0,
        ge=0.0,
        description=(
            "Ceiling (seconds) for a single transient-error retry backoff, "
            "bounding both the exponential term and any server-stated "
            "``Retry-After`` so worst-case added latency is "
            "``max * llm_transient_error_retry_max_attempts``. Only consulted "
            "when ``llm_transient_error_retry_max_attempts`` > 0."
        ),
    )

 # ----- LLM stream liveness -----
    llm_stream_idle_timeout_seconds: float = Field(
        default=90.0,
        gt=0.0,
        description=(
            "Hard timeout (seconds) on inactivity in the upstream LLM "
            "stream. When the wall-clock gap between two consecutive "
            "ProviderDelta events exceeds this value the watchdog raises "
            "``LLMStreamIdleError`` and the loop transitions to terminal FAILED."
        ),
    )
    llm_stream_stall_threshold_seconds: float = Field(
        default=5.0,
        gt=0.0,
        description=(
            "WARN-ONLY inter-chunk stall signal (seconds). This threshold now "
            "only feeds core telemetry (``query.py::_iter_with_idle_watchdog`` "
            "warn logging); the hard abort lives at "
            "``llm_stream_idle_timeout_seconds``. Local vLLM streams chunks "
            "every ~6ms, so a 5s inter-chunk gap is almost always executor "
            "event-loop starvation, NOT a dead socket. The adapter's "
            "inter-chunk watchdog tolerates gaps up to "
            "``llm_stream_idle_timeout_seconds`` (and extends, bounded by that "
            "same cap, while the loop-lag gauge reports starvation). This is "
            "NOT a time-to-first-byte budget; initial provider silence is "
            "governed by ``llm_provider_stream_idle_timeout_seconds``. MUST be "
            "strictly less than ``llm_stream_idle_timeout_seconds`` so the "
            "warning-vs-abort distinction holds."
        ),
    )
    llm_stream_reasoning_idle_timeout_seconds: float = Field(
        default=300.0,
        gt=0.0,
        description=(
            "Reasoning-aware idle timeout. When the watchdog has observed at "
            "least one ``ProviderDeltaKind.thinking`` (reasoning/chain-of-thought) "
            "delta within the recent window, the per-iteration ``wait_for`` "
            "budget extends from ``llm_stream_idle_timeout_seconds`` to "
            "this value. Slow MoE/reasoning models can stall ~90 s during "
            "a reasoning gap without surfacing any non-reasoning delta; the "
            "legacy 90 s hard cap cancelled the stream before the model could "
            "emit visible text. Default 300 s gives slow aggregators headroom "
            "while still bounding the worst case. "
            "MUST be ``>= llm_stream_idle_timeout_seconds``."
        ),
    )

 # ----- Continue-prompt fallback -----
    max_consecutive_empty_responses: int = Field(
        default=3,
        ge=0,
        description=(
            "Max consecutive assistant turns with empty content AND "
            "populated ``reasoning_content`` before terminal FAILED. "
            "Each round synthesises a continuation prompt "
            "(``continue_prompt_text``) and re-streams. The 'thinking-tokens "
            "trap' fix: small reasoning models occasionally burn their token "
            "budget on chain-of-thought and emit no visible content. Injecting "
            "a continue-nudge forces them to commit. Set to ``0`` to disable "
            "recovery."
        ),
    )
    continue_prompt_text: str = Field(
        default="Please continue.",
        description=(
            "Synthetic user-role text appended to history when the "
            "assistant turn ends with empty content + populated "
            "reasoning_content. Tenant-configurable so multilingual "
            "deployments can tune the nudge per locale."
        ),
    )
    reasoning_length_cut_retries: int = Field(
        default=2,
        ge=0,
        description=(
            "Retries after a round the output cap cut while the model was "
            "still reasoning (``finish_reason='length'``, reasoning and nothing "
            "else). The cut reasoning is not kept and nothing is appended for "
            "it: each retry sends the same prompt with one knob changed — the "
            "first lowers the reasoning effort to ``low`` and adds the one nudge "
            "in ``reasoning_length_cut_nudge_text``, the second switches "
            "thinking off (``reasoning_length_cut_disable_thinking``). A retry "
            "that would change nothing is skipped, and past the count the run "
            "winds down. The knobs go back to their configured values on the "
            "next round that produces anything. ``0`` disables the retries: the "
            "first cut winds the run down."
        ),
    )
    reasoning_length_cut_nudge_text: str = Field(
        default=(
            "Your previous response reached the output token limit while still "
            "reasoning, before it produced an answer or a tool call, so it was "
            "discarded. Respond more concisely: think briefly, then give the "
            "answer or exactly one tool call."
        ),
        description=(
            "The one synthetic user message sent with the first retry after a "
            "reasoning-only length cut. It names the mechanical cause and asks "
            "for a shorter shape; it never asks the model to resume reasoning "
            "that was not kept."
        ),
    )
    reasoning_length_cut_disable_thinking: bool = Field(
        default=True,
        description=(
            "Whether the last retry after a reasoning-only length cut switches "
            "thinking off for that round. Off, the ladder stops at lowered "
            "effort; a run mode that requires thinking skips the step either way."
        ),
    )

 # ----- Death-spiral guard -----
    skip_terminal_hooks_on_llm_error: bool = Field(
        default=True,
        description=(
            "Death-spiral guard — when ``True`` (the default) the engine "
            "SKIPS Stop / SessionEnd hooks on terminal cause "
            "``llm_provider_error`` (any of LLMProviderError, "
            "LLMStreamIdleError, post-retry LLMContextWindowExceeded, "
            "MaxOutputTokensExhausted). Prevents broken-provider runs from "
            "cascading through error-only hooks. Set to ``False`` for "
            "diagnostic deployments where Stop hooks SHOULD see the LLM error "
            "(e.g. error-classifier hooks)."
        ),
    )

 # ----- Finalization gate -----
    finalization_gate_enabled: bool = Field(
        default=False,
        description=(
            "Kill-switch for the finalization gate. "
            "When True, the executor invokes ``verify_declared_deliverables`` "
            "+ ``decide_finalization`` during ``_finalise_run``, AND passes the "
            "model-facing ``<finalization_contract>`` JSON TEMPLATE block "
            "(``build_finalization_contract_block``) into the leader system "
            "prompt. The gate's verdict (``completed`` / ``partial`` / "
            "``failed`` / ``unknown``) SUPERSEDES the tool-errors-count "
            "downgrade when ``declared_deliverables`` is non-empty. "
            "Default is False: the prose ``<finalization_contract>`` template "
            "made models declare deliverables then ``end_turn`` WITHOUT calling "
            "Write and leaked the raw XML into the chat as plain text. With "
            "the default off the contract block "
            "is no longer injected and the gate short-circuits to ``None`` so "
            "terminal completed/partial classification falls back to the "
            "tool-errors heuristic — which works normally WITHOUT any contract "
            "(``verify_declared_deliverables`` returns ``None``, "
            "``apply_finalization_decision`` passes ``terminal_status`` "
            "through, ``build_finalization_contract_event_for_run`` emits no "
            "event). The typed ``Finalize`` tool "
            "(``agent_finalize_tool_as_terminal``) remains the opt-in "
            "replacement. Set True per-tenant via the Constants page to "
            "restore the gate + the prose template block."
        ),
    )

 # ----- DAG tool-precondition mechanism --------
    tool_preconditions_enabled: bool = Field(
        default=False,
        description=(
            "Enforce :attr:`ToolDefinition.preconditions` at dispatch time. "
            "When True, a tool call whose preconditions are not satisfied "
            "returns a ``[PRECONDITION NOT MET: ...]`` error envelope instead "
            "of dispatching. Ported from the v1 DAG Precondition pattern "
            "(commit ``7dfa1ff``: ``protocore.runtime.tool_preconditions``). "
            "The mechanism is the foundation for "
            "the ``AppendFile`` / ``FinalizeFile`` workflow shipped in the "
            "same batch — ``FinalizeFile`` requires a prior ``AppendFile`` "
            "for the same path before the file can be marked complete. Set "
            "False to disable enforcement (precondition lists are ignored "
            "and every tool call dispatches without the pre-flight check). "
            "Default is False: the only live precondition observed in production "
            "was a false-positive (model called ``FinalizeFile`` after ``Write``, "
            "blocked by ``AppendFile:{path}`` precondition). Mechanism stays wired "
            "but inert until the ``file_path`` vs ``path`` alias resolution lands "
            "in ``resolve_precondition``."
        ),
    )

 # ----- Adaptive safety band -----
    adaptive_safety_band_enabled: bool = Field(
        default=True,
        description=(
            "Kill-switch for the AdaptiveSafetyBand. "
            "When True (default), the LLM output budget is reduced by the "
            "calibrated band so the prompt + max_tokens stay under the "
            "provider context window. Set False to roll back to a static "
            "safety margin (effectively band=0)."
        ),
    )

 # ----- Approval-gate kill-switch -----
    approval_gate_web_enabled: bool = Field(
        default=False,
        description=(
            "If False, require_approval outcomes from PreToolUse hooks are "
            "downgraded to allow for runs originating from chat/web mode. "
            "Sandbox isolation + dangerous_commands.py deny patterns are the "
            "production safety boundary in web mode. Set TRUE only for future "
            "CLI runs where local FS access makes approval meaningful."
        ),
    )

 # ----- Universal terminal-tool nudge -----
 #
 # Any tenant can declare a terminal tool name via
 # ``leader_config.expected_terminal_tool``; when set AND
 # ``terminal_tool_nudge_enabled`` is True the query loop emits a single
 # contract-repair nudge if a run is about to finish without that tool's
 # successful terminal result in history. Defaults to False so tenants
 # that never declared a terminal tool keep their snapshot semantics.
    terminal_tool_nudge_enabled: bool = Field(
        default=False,
        description=(
            "When True AND ``QueryEngineConfig.expected_terminal_tool`` is "
            "non-empty AND no successful terminal tool result is in "
            "history, the query loop injects one additional user message "
            "reminding the model to call the configured terminal tool. "
            "This is a contract-repair guard, not scorer automation: the "
            "model still chooses the tool arguments. The message body comes "
            "from the ``terminal_tool_nudge`` prompt template."
        ),
    )
    preserve_completed_answer_on_stream_error: bool = Field(
        default=True,
        description=(
            "When True, a transient LLM provider / idle-stream error raised on a "
            "harness-forced continuation turn (the terminal-tool nudge or a "
            "continue-prompt injection) does NOT drive the run terminal FAILED "
            "if a substantive user-facing assistant answer was already produced "
            "in a prior turn. Instead the run completes on that already-delivered "
            "answer. Closes the false-negative where a complete reply streamed, "
            "the model never called the terminal tool, the forced finalize-nudge "
            "turn then hit a transient upstream error, and the run was mislabeled "
            "failed even though the user's answer was intact. The substantive "
            "floor reuses ``finalize_prose_gate_min_chars``. Set False to restore "
            "the prior behaviour where any provider error on the continuation "
            "propagates as the run's terminal status."
        ),
    )
    empty_completion_guard_enabled: bool = Field(
        default=True,
        description=(
            "When True, an assistant turn that ends with ``finish_reason='stop'`` "
            "and NO visible text, NO tool calls and NO reasoning_content is NOT "
            "sealed as a silent empty COMPLETED when the run has not yet produced "
            "any visible assistant answer and no terminal tool result is in "
            "history. Instead the loop grants a bounded re-drive "
            "(``empty_completion_guard_max_redrives``) to give the model another "
            "chance to answer, and — once that budget is exhausted — terminates "
            "the run FAILED with a ``no_answer_empty_completion`` reason rather "
            "than reporting an empty turn as a clean answer. Runs a turn that "
            "already delivered a substantive answer, or one whose current turn "
            "carried text/tools/reasoning, are unaffected (they complete "
            "normally). Set False to restore the prior behaviour where a bare "
            "empty end_turn seals COMPLETED with no answer."
        ),
    )
    empty_completion_guard_max_redrives: int = Field(
        default=1,
        ge=0,
        description=(
            "Number of bounded re-drives the empty-completion guard grants when "
            "a turn ends empty with no answer yet in history. Each re-drive "
            "injects an API-valid synthetic assistant+user continue pair and "
            "re-opens the stream once. Exhausting the budget routes to a FAILED "
            "terminal instead of a silent empty COMPLETED. 0 skips the re-drive "
            "and goes straight to the FAILED terminal on the first bare-empty "
            "completion. Only consulted when ``empty_completion_guard_enabled``."
        ),
    )
    terminal_tool_nudge_write_first_enabled: bool = Field(
        default=True,
        description=(
            "Extends the terminal-tool nudge: when the nudge fires AND no "
            "successful file-write tool result "
            "(``terminal_tool_nudge_file_write_tool_names``) is in history, "
            "the nudge text is prefixed with the "
            "``terminal_tool_nudge_write_first`` prompt template so the model is steered "
            "to call the ACTUAL deliverable write tool (Write/AppendFile) "
            "FIRST, not just the terminal tool. This closes the "
            "narrate-then-surrender failure where a model says 'Now let me "
            "write this file' and fires 0 tools. Bounded by the single-shot "
            "nudge latch (never loops). Default True is universal — a strong "
            "model that already wrote the file never sees the prefix (the "
            "history check finds its write result). Set False to restore the "
            "plain terminal nudge."
        ),
    )
    terminal_tool_nudge_file_write_tool_names: tuple[str, ...] = Field(
        default=("Write", "AppendFile"),
        description=(
            "Tool names that count as a file-write deliverable for the "
            "``terminal_tool_nudge_write_first_enabled`` history check. When "
            "the terminal nudge fires and NONE of these tools has a "
            "successful (non-error) result in history, the write-first "
            "prefix is added. Configurable so a tenant with differently "
            "named write tools stays universal — no the host tool name is "
            "hardcoded in core."
        ),
    )

 # ----- Universal prose-gate before a background terminal -----
 # terminal tool. The terminal tool (e.g. ``Finalize``) is being made a
 # pure background gate: its ``answer`` field is removed and its tool_use /
 # tool_result pair is filtered from the stream + durable history, so the
 # user-facing answer MUST be the model's own visible prose. Empirically a
 # small tail of runs call the terminal tool with NO substantive prose after
 # their last real work; for those the runtime injects ONE bounded repair
 # turn instructing the model to write the answer as normal text first, then
 # call the terminal tool, and re-drives once. One-shot per run, snapshot-
 # persisted. Universal — keyed only on ``expected_terminal_tool`` + the
 # visible-prose history predicate; no per-model / per-tool hard-coding.
    finalize_prose_gate_enabled: bool = Field(
        default=True,
        description=(
            "when True AND "
            "``QueryEngineConfig.expected_terminal_tool`` is set AND the run is "
            "about to latch that terminal tool's result while it has produced "
            "NO substantive visible assistant prose after its latest "
            "non-terminal (real work) tool, the runtime VETOES the terminal "
            "dispatch ONCE and injects one bounded repair turn (rendered from the "
            "``finalize_prose_gate_repair`` prompt template) asking the model to emit the "
            "final answer as normal assistant text and THEN call the terminal "
            "tool. The terminal tool is made a background gate (its tool_use / "
            "tool_result pair is filtered from the stream + durable history and "
            "its answer field is dropped), so the user-facing answer must be "
            "the model's prose; this gate guarantees it exists. One-shot per "
            "run (``_finalize_prose_gate_used`` latch, snapshot-persisted) so a "
            "second prose-less terminal after the repair finalises rather than "
            "looping. Universal: keyed only on the per-tenant terminal-tool "
            "contract, never a specific tool name or model. The SAME floor, "
            "latch and repair text also apply at the PLAIN-STOP completion — a "
            "run the model ends with ``finish_reason='stop'`` and no tool call, "
            "which is how a deployment that declares no terminal tool ends "
            "nearly every run and therefore the only path on which the "
            "dispatch-seam veto above can never participate. Because the two "
            "paths share one latch the SHORT-ANSWER test fires at most once per "
            "run in total; the POINTER test rides the same two seams and the "
            "same kill switch but carries its own, larger bound "
            "(``finalize_prose_gate_pointer_max_repair_attempts``), so spending "
            "one no longer silences the other. Set False to restore the prior "
            "behaviour on BOTH paths and BOTH tests (a "
            "payload-only terminal finalises immediately, and a run that stops "
            "with a below-floor answer completes as-is)."
        ),
    )
    finalize_prose_gate_min_chars: int = Field(
        default=1,
        ge=0,
        description=(
            "minimum length (in characters, stripped) a "
            "visible assistant prose block must reach to count as a substantive "
            "user-facing answer for the ``finalize_prose_gate_enabled`` check. "
            "Prose shorter than this floor that appears after the latest real "
            "work tool does NOT satisfy the gate, so the repair turn still "
            "fires. Default 1 = 'any non-empty visible prose after the last work "
            "tool IS the answer' — a terse-but-complete reply (e.g. ``144``, "
            "``Привет``, ``Запомнил: …``) counts and is NOT re-emitted or "
            "veto'd. Deliberately DECOUPLED from "
            "``finalization_empty_contract_min_response_chars`` (the "
            "analytic-contract empty-prose floor, default 100): a 100-char floor "
            "here mis-classified short valid answers as 'no prose', driving the "
            "duplicate re-emit + the bilingual fallback. The genuinely "
            "empty ``Finalize`` tail still has 0 prose, so the gate (and the "
            "the host net, which uses an independent floor of 1) still "
            "protect it. Set to 0 to also accept the empty string (degenerate)."
        ),
    )

 # ----- The narration a delegating leader opens its answer with -----
 #
 # A leader that farmed work out to subagents opens its reply by reporting its
 # own progress — "Now I have all the material. Let me compile the review." —
 # and only then answers. It is not a prompt failure that more prompting fixes:
 # the instruction not to do it has been stated, rendered on every call and
 # measured as ignored. The narration and the answer are one text block by
 # construction (the terminal tool carries no answer field, so the reply is
 # prose), so the only thing left to act on is the block, and the action is to
 # split it and mark the first half collapsed. Nothing is deleted.
    delegated_answer_narration_split_enabled: bool = Field(
        default=True,
        description=(
            "When True, an assistant text block produced by a run that has "
            "DELEGATED at least one subtask is split where it opens with "
            "process narration: the leading narration becomes its own "
            "``collapsed`` content block and the answer continues in a "
            "``public`` one. The mark is carried per block, so the live stream "
            "and the durable transcript render the same thing. Only ever "
            "COLLAPSES — no text is removed from the answer on any path, and a "
            "misfire costs a leading sentence rendered as a chip. Gated on "
            "delegation because that is where the behaviour was measured: "
            "without it a minority of answers open this way, with it almost "
            "every one does, and a run that dispatched no subtask is left "
            "untouched. This is the only place in the runtime where a "
            "reader-facing visibility decision reads the TEXT rather than the "
            "structure of the turn, which is why it has a switch: a lexical "
            "rule fails per language and per phrasing, and to an operator the "
            "failure looks like an answer that lost its first sentence with "
            "nothing in the transcript to connect it to this code. Set False "
            "to leave every text block whole."
        ),
    )
    delegated_answer_narration_scan_chars: int = Field(
        default=600,
        ge=0,
        description=(
            "How far into a text block "
            "``delegated_answer_narration_split_enabled`` may look for leading "
            "narration. The scan also stops at the first paragraph break and "
            "at the first sentence that is not narration, so this is a "
            "ceiling rather than the usual bound — measured openers run to "
            "about 170 characters. It caps the cut point: no split can ever "
            "collapse more than this many characters, whatever the text says. "
            "0 disables the scan (equivalent to the toggle being off)."
        ),
    )
    delegated_answer_narration_min_answer_chars: int = Field(
        default=200,
        ge=0,
        description=(
            "How much answer must survive AFTER the collapsed narration for "
            "``delegated_answer_narration_split_enabled`` to split at all. A "
            "reply that is narration and little else stays whole and visible: "
            "collapsing it would leave the reader an empty bubble, which is "
            "worse than the narration. On the live stream this is also what "
            "the split waits to observe before it commits, so the decision is "
            "never made on a block that turns out to be short. 0 removes the "
            "floor (a block that is entirely narration would then collapse "
            "entirely — degenerate)."
        ),
    )

 # ----- Whether the user can reach the agent's workspace -----
 #
 # A product fact about the surface a run is serving, not a switch for any one
 # mechanism: it says whether the person reading the reply can open the files
 # the agent wrote. A dashboard / IDE surface with a file browser may
 # legitimately be answered with a path; a chat window may not, and there
 # "saved to workspace/report.md" is an empty reply however many characters of
 # filing notice surround it.
    workspace_visible_to_user: bool = Field(
        default=True,
        description=(
            "Whether the user reading a run's reply can browse the agent's "
            "workspace and open the files it wrote. True (the default) is the "
            "surface with a file browser: pointing the user at a written file IS "
            "a complete answer there, so nothing about the answer floor changes. "
            "Set False for a chat-only surface, where the reply is the only "
            "thing the user ever sees: the substantive-answer floor then "
            "additionally treats a reply that is principally a POINTER to a file "
            "this run wrote — it names the file and is a small fraction of what "
            "was written into it — as no answer at all, and spends the gate's "
            "single repair turn asking for the substance in the reply itself. "
            "Tuned by ``finalize_prose_gate_pointer_max_answer_fraction`` and "
            "``finalize_prose_gate_pointer_min_written_chars``. The file is "
            "still written either way: this refuses the ANSWER, never the write."
        ),
    )
    finalize_prose_gate_pointer_max_answer_fraction: float = Field(
        default=0.2,
        ge=0.0,
        description=(
            "How small a reply must be, RELATIVE to the content this run wrote "
            "into the file the reply names, before it counts as principally a "
            "pointer rather than an answer. A reply reaching this fraction of "
            "the written content stands as written. Only consulted when "
            "``workspace_visible_to_user`` is False. Default 0.2 — a fifth of "
            "the document — sits above the filing notices measured in "
            "production (a 13 KB article reported back as 1.2-1.9 KB of 'saved "
            "to <path> … structure: 1. Введение …' is 0.09-0.15 of what was "
            "written) and below a reply that carries the substance back. The "
            "demand scales with the deliverable and is deliberately not capped: "
            "a reply that is a twentieth of a 40 KB report is a pointer to a "
            "document the user cannot open, whatever its absolute length. Lower "
            "it to demand less; 0.0 disables the pointer test outright, leaving "
            "the plain length floor."
        ),
    )
    finalize_prose_gate_pointer_min_written_chars: int = Field(
        default=4_000,
        ge=0,
        description=(
            "How much content this run must have written into a SINGLE file "
            "before a reply that merely points at it can be refused — characters "
            "of the write tool's content argument, accumulated per target path "
            "across the run's successful writes. Only consulted when "
            "``workspace_visible_to_user`` is False. Default 4000, roughly 600 "
            "words, is the line between a deliverable and everything a run "
            "legitimately writes in passing: a scratch file, a config, a patch, "
            "a to-do list, a chunk of code. Below it the pointer test never "
            "fires however terse the reply, which bounds the cost of being wrong "
            "to runs that really did produce a document. With the default "
            "fraction the smallest reply this can ever ask for is 800 "
            "characters. Raise it to narrow the mechanism to large "
            "deliverables; 0 disables the pointer test outright."
        ),
    )
    finalize_prose_gate_pointer_max_repair_attempts: int = Field(
        default=1,
        ge=0,
        description=(
            "How many repair turns the POINTER refusal may spend on one run "
            "before it gives up and lets the run finish with whatever answer it "
            "has. Its own budget, deliberately not the substantive-answer "
            "floor's single shot: the two catch different failures.\n\n"
            "Default 1 because three was measured WORSE than one. The reasoning "
            "for a larger budget was sound and wrong: the pointer test detects "
            "its failure perfectly, a single repair turn was seen to change "
            "nothing, and the neighbouring read-back driver does show second "
            "attempts landing work the first did not — so three looked like the "
            "obvious bound. Measured on the same code and configuration with "
            "only this value changed, the article scenario produced an "
            "acceptable answer in 2 of 3 runs at one attempt and 0 of 3 at "
            "three. The mechanism is plausible: every attempt spends one of the "
            "run's turns and grows the context the answer is written from, so "
            "extra asking can push a run past its ceiling before it writes the "
            "real answer — the repair costs turns and the asking itself buys "
            "nothing, because a request repeated does not become a compulsion. "
            "Raise it only against fresh evidence for a particular deployment; "
            "an unbounded retry turns a run that delivered a filing notice into "
            "a run that delivers nothing at all. Every INJECTED repair turn charges one "
            "attempt, whether or not the reply improved (see "
            "``_charge_pointer_answer_repair``). Snapshot-persisted, so a "
            "cross-pod resume cannot hand a run a fresh budget. Set to 1 for the "
            "single-shot behaviour the substantive-answer floor has; 0 disables "
            "the pointer refusal outright while leaving the length floor and "
            "``workspace_visible_to_user``'s other effects alone."
        ),
    )

 # ----- External trial-deadline early finalize -----
 #
 # When a run executes under an external wall-clock deadline (e.g. an
 # eval harness that reaps the run), submitting the terminal tool after
 # the deadline scores as "no answer provided". When ``agent_max_seconds``
 # is set, the query loop fires the SAME latched terminal-tool nudge that
 # the voluntary-finish/backstop paths use,
 # ``agent_deadline_finalize_slack_seconds`` BEFORE the budget runs out —
 # forcing an early best-effort finalize (emit the terminal tool with
 # whatever durable answer exists) while the run is still live. Universal:
 # keyed off ``expected_terminal_tool`` + these RCs, no per-tenant
 # hardcoding. The latch fires at most once per run (shares the loop's
 # terminal-nudge latch). Default 0.0 disables the budget entirely.
    agent_max_seconds: float = Field(
        default=0.0,
        ge=0.0,
        json_schema_extra={"zero_means_unlimited": True},
        description=(
            "Wall-clock budget (seconds) for a single run. "
            "When > 0 AND ``expected_terminal_tool`` is set AND no terminal "
            "result is in history yet, the query loop forces an early "
            "best-effort finalize (terminal-tool nudge + terminal-only "
            "latch) once the run has been live for "
            "``agent_max_seconds - agent_deadline_finalize_slack_seconds``, "
            "so the model submits its answer BEFORE an external trial / "
            "reaper kills the run. Measured from ``QueryEngine.run`` entry "
            "via a monotonic clock, persisted across resume so a re-driven "
            "run keeps one budget. Default 0.0 disables it (reproduces "
            "today). This is contract-repair finalization, not scorer "
            "automation: the model still chooses the answer."
        ),
    )
    agent_deadline_finalize_slack_seconds: float = Field(
        default=45.0,
        ge=0.0,
        description=(
            "Headroom (seconds) subtracted from "
            "``agent_max_seconds`` to decide when the early-finalize nudge "
            "fires. Must leave enough time for one terminal-tool round-trip "
            "(+ its own ``terminal_tool_answer_timeout_retry_attempts``) "
            "before the external reaper. Inert when ``agent_max_seconds`` "
            "is 0.0."
        ),
    )

 # ----- Pre-terminal self-verify turn -----
 #
 # Before committing a terminal answer, optionally inject ONE bounded,
 # latched corrective turn when an host-supplied trigger detects a
 # problem (e.g. a cited path absent from observed runtime state, or a
 # declared mutation that never landed). The TRIGGER lives in the host;
 # core owns only the latch + the bounded single-turn injection.
 # Default False → no extra turn → bit-identical to prior behaviour.
    pre_terminal_self_verify_enabled: bool = Field(
        default=False,
        description=(
            "Kill-switch for the pre-terminal self-verify turn. When True AND "
            "an host-supplied "
            "``QueryEngineConfig.pre_terminal_self_verify_trigger`` returns "
            "a corrective message at the moment a terminal-tool result would "
            "be committed, the query loop instead injects ONE corrective "
            "user turn and lets the model run one more bounded turn before "
            "finalising. Fires at most once per run (latched). Default "
            "False (no extra turn). Universal: the trigger predicate is "
            "tenant-supplied; core never hardcodes domain-specific logic."
        ),
    )
    pre_terminal_self_verify_max_extra_turns: int = Field(
        default=1,
        ge=0,
        le=3,
        description=(
            "Maximum number of corrective self-verify turns the loop may "
            "inject per run. The latch already bounds this to one fire; "
            "this cap is an additional explicit ceiling so a "
            "future multi-fire variant stays bounded. 0 disables the "
            "injection even when ``pre_terminal_self_verify_enabled`` is "
            "True. Shared budget: the PRE-DISPATCH terminal-tool verify seam "
            "(``pre_dispatch_terminal_verify_enabled``) debits the SAME "
            "per-run counter, so the total number of corrective turns a run "
            "may receive from either self-verify seam is bounded by this one "
            "ceiling."
        ),
    )

 # ----- PRE-DISPATCH terminal-tool verify -----
 #
 # The ``pre_terminal_self_verify_*`` seam above runs AFTER the loop
 # has already dispatched the terminal tool. For a terminal tool whose
 # side effect (an external answer-submission RPC) fires inside its own
 # ``run``, a post-dispatch corrective turn is POST-SUBMIT — it cannot
 # repair a fabricated ref or a declared-but-missing mutation because the
 # scorer already saw the answer. This gate adds a PRE-DISPATCH validation
 # seam: an host-supplied ``QueryEngineConfig.pre_dispatch_terminal_
 # verify_trigger`` is consulted in ``_dispatch_tool`` BEFORE the dispatcher
 # runs the terminal tool. If it returns a corrective message, the loop
 # VETOES the terminal dispatch (no RPC fires), injects ONE bounded
 # corrective user turn, and re-drives so the model can fix the answer
 # before re-submitting. Fires at most once per run (durable latch) and
 # shares the ``pre_terminal_self_verify_max_extra_turns`` ceiling. Default
 # False → no veto → bit-identical to prior behaviour. Universal: the
 # predicate is tenant-supplied (it inspects the un-submitted ``ToolCall``
 # arguments + caller-provided observed state); core never hardcodes
 # domain-specific logic.
    pre_dispatch_terminal_verify_enabled: bool = Field(
        default=False,
        description=(
            "Kill-switch for the PRE-DISPATCH "
            "terminal-tool verify seam. When True AND the tool about to be "
            "dispatched is the configured ``expected_terminal_tool`` AND an "
            "host-supplied "
            "``QueryEngineConfig.pre_dispatch_terminal_verify_trigger`` "
            "returns a corrective message for that un-submitted tool call, "
            "the query loop VETOES the terminal dispatch (the tool's "
            "external side effect never fires), injects ONE corrective user "
            "turn, and re-drives one more bounded turn so the model can "
            "repair the answer BEFORE re-submitting. Fires at most once per "
            "run (durable latch persisted across resume) and debits the "
            "shared ``pre_terminal_self_verify_max_extra_turns`` budget. "
            "Default False reproduces prior behaviour (no veto). Universal: the "
            "predicate is tenant-supplied; core never hardcodes "
            "domain-specific logic. This is contract-repair, not scorer automation — the "
            "model still chooses the corrected answer."
        ),
    )

 # ----- Parallel read-dispatch gate -----
 #
 # The parallel read fast path fans concurrent-safe read tools out under
 # ``asyncio.gather`` (``query.py`` ~1278-1527). These knobs let an
 # operator (a) disable the fan-out entirely (rollback to serial
 # dispatch) and (b) bound the per-batch fan-out so a turn that emits
 # many parallel reads chunks into ≤N-wide gather batches. Bounding the
 # fan-out matters when read handlers append to shared observed state
 # (which they must guard with the engine's shared-state lock — see
 # ``QueryEngine`` — before enabling this for a state-mutating tenant, else a
 # "correct refs zeroed" race can
 # reappear). Defaults reproduce prior behaviour: enabled, generous cap.
    parallel_read_tools_enabled: bool = Field(
        default=True,
        description=(
            "Kill-switch for the parallel read-tool "
            "fast path. When True (default), adjacent "
            "concurrent-safe, non-destructive, non-hook-gated tool calls in "
            "one assistant turn fan out under ``asyncio.gather``. Set False "
            "to dispatch every tool serially (rollback). Behaviour is "
            "otherwise identical; the deterministic transcript-order replay "
            "(snapshot/restore/replay helpers) is preserved either way."
        ),
    )
    parallel_read_tools_max_fanout: int = Field(
        default=0,
        ge=0,
        json_schema_extra={"zero_means_unlimited": True},
        description=(
            "Maximum number of tool calls dispatched in a "
            "single ``asyncio.gather`` batch. ``0`` (default) means "
            "UNLIMITED — every adjacent parallel-eligible run fans out in a "
            "single unbounded gather. A value ``> 0`` chunks a longer "
            "parallel-eligible run into ≤N-wide sub-batches (each going "
            "through the same snapshot→gather→restore→replay sequence, "
            "preserving LLM-requested order) to bound concurrent load on a "
            "backend that degrades under contention; tenants set a finite "
            "cap via a per-tenant override. Inert when "
            "``parallel_read_tools_enabled`` is False. The default ``0`` is "
            "the value-preserving sentinel: it reproduces unbounded fan-out "
            "for every tenant that does not set an explicit cap."
        ),
    )
    parallel_subagents_enabled: bool = Field(
        default=True,
        description=(
            "Master switch for concurrent subagent delegation. When True "
            "(default), if the assistant emits two or more adjacent Agent "
            "(subagent-dispatch) calls in a single turn, those calls run "
            "concurrently under a bounded semaphore instead of strictly one "
            "after another; the leader still blocks until the whole group "
            "finishes and each tool result is recorded in the order the "
            "assistant requested it. Set False to dispatch every delegation "
            "call serially (rollback). Non-delegation tools are unaffected."
        ),
    )
    max_concurrent_subagents: int = Field(
        default=4,
        ge=1,
        le=64,
        description=(
            "Maximum number of delegated subagents that run concurrently "
            "within one leader assistant turn. When the assistant emits more "
            "adjacent Agent calls than this cap, the excess run in waves as "
            "slots free up. 1 makes delegation effectively sequential "
            "(behaviour identical to the serial path). Only takes effect when "
            "parallel_subagents_enabled is True. Raise to widen fan-out at the "
            "cost of more concurrent child runs (each consumes tokens and a "
            "sandbox slot)."
        ),
    )
    max_concurrent_subagents_per_tree: int = Field(
        default=8,
        ge=0,
        description=(
            "Maximum number of parallel-group-dispatched subagent runs that "
            "execute concurrently across the WHOLE run tree (the additive sum "
            "over every nested delegation group, not just one leader turn). "
            "``max_concurrent_subagents`` still bounds the WIDTH of each "
            "individual group; this bounds their SUM so nested delegation across "
            "depth cannot compound multiplicatively (depth x width) and overrun "
            "the shared sandbox / token budget. Enforced deadlock-free by "
            "releasing a run's tree slot while it awaits its own children and "
            "reacquiring it afterwards, so a blocked parent never pins a slot its "
            "descendants need. ``0`` (the value-preserving sentinel, consistent "
            "with parallel_read_tools_max_fanout) means UNLIMITED — the tree cap "
            "is inert and only the per-group width caps apply, reproducing the "
            "pre-cap multiplicative behaviour. Only takes effect when "
            "parallel_subagents_enabled is True. The value is captured when a "
            "tree's budget is first minted (at its first parallel fan-out); an "
            "in-flight tree does not converge to a mid-flight edit — new trees "
            "pick up the change."
        ),
    )
    max_subagent_runs_per_tree: int = Field(
        default=24,
        ge=0,
        description=(
            "CUMULATIVE cap on how many delegated child runs one root run may "
            "START over its whole lifetime, counted across every descendant at "
            "every depth and never reset between waves. The concurrency caps "
            "(max_concurrent_subagents, max_concurrent_subagents_per_tree) and "
            "the depth cap (max_subagent_depth) are all INSTANTANEOUS: a leader "
            "that dispatches a legal-width group, waits for it and dispatches "
            "another passes every one of them on every wave, so before this "
            "constant existed the total number of child runs was bounded only by "
            "wall-clock. When the cap is reached, further delegation calls are "
            "refused with an explicit tool result telling the leader the budget "
            "is spent and to finalize on what it has; children already running "
            "are left alone and the run can still write its answer. ``0`` means "
            "UNLIMITED (the pre-cap behaviour), consistent with the sentinel on "
            "max_concurrent_subagents_per_tree. The default of 24 is a judgement "
            "and not a measurement: it is six full waves at the default fan-out "
            "of 4, comfortably above any delegation pattern seen in normal use "
            "and below the runaway that motivated the bound. Per-tenant "
            "overridable, and narrowable further by an access plan. Captured "
            "when the run's ledger is minted; an in-flight run does not converge "
            "to a mid-flight edit."
        ),
    )
    max_total_tokens_per_tree: int = Field(
        default=20_000_000,
        ge=0,
        description=(
            "CUMULATIVE cap on input+output tokens summed over every LLM call "
            "made by one root run AND all of its descendants. Distinct from "
            "run_max_output_tokens_budget, which counts OUTPUT tokens for a "
            "SINGLE engine: each delegated child is a fresh engine with its own "
            "fresh per-run budget, so per-run bounds do not compose over "
            "delegation and a wide tree can spend an unbounded multiple of what "
            "any one run is allowed. When the cap is reached, further delegation "
            "is refused (the leader is told the budget is spent and finalizes on "
            "what it has); in-flight children are not aborted and the run can "
            "still write its answer, so this bound can never leave a run unable "
            "to finish. ``0`` means UNLIMITED. The default of 20000000 is a "
            "backstop, not the primary bound — max_subagent_runs_per_tree is "
            "expected to bind first in the ordinary runaway; this one catches "
            "the few-children-enormous-contexts shape instead. It is a judgement "
            "and not a measurement. Per-tenant overridable, and narrowable "
            "further by an access plan."
        ),
    )

 # ----- Candidate-answer preservation (keep first non-empty draft) -----
    terminal_candidate_preserve_enabled: bool = Field(
        default=False,
        description=(
            "Kill-switch for terminal-candidate preservation in the core "
            "pre-dispatch terminal-veto path. When True AND a terminal tool call "
            "is vetoed by the pre-dispatch verify seam, the first non-empty "
            "terminal-args draft is persisted as a durable per-run candidate "
            "(via the engine snapshot, so it survives cross-pod resume). If a "
            "later terminal dispatch regresses (required ``message`` empty / "
            "shorter than ``terminal_answer_min_message_chars`` while a "
            "substantive saved candidate exists) the loop flags "
            "``terminal_candidate.regressed`` (and re-vetoes once if repair "
            "budget remains). Core never auto-synthesises the answer body — only "
            "the existing latched corrective nudge is reused. Default False "
            "discards the candidate (bit-identical). Universal (generic over "
            "terminal args + the required-field predicate from terminal-contract "
            "metadata); horizontal-safe (durable snapshot, no module state)."
        ),
    )
    terminal_answer_min_message_chars: int = Field(
        default=0,
        ge=0,
        description=(
            "Regression floor for terminal-candidate preservation. A replacement "
            "terminal answer whose required ``message`` is shorter than this "
            "while a prior non-empty candidate exists is treated as 'regressed'. "
            "0 (default) = off (no floor; only a truly EMPTY replacement counts "
            "as regressed). Only consulted when "
            "``terminal_candidate_preserve_enabled`` is True."
        ),
    )
 # --- Output-token slice reservation for synthesis (budget channel only) ---
    terminal_synthesis_output_reserve_tokens: int = Field(
        default=0,
        ge=0,
        description=(
            "Final-turn-specific output-token floor. On the terminal / "
            "forced-final turn (where the terminal-tool nudge or backstop is "
            "active) the per-message output budget is floored at this many "
            "tokens so the model has room to emit message + refs + outcome. "
            "0 (default) = off (output budget unchanged). This is NOT a raise "
            "of the global ``llm_output_max_tokens_ratio`` (the binding cap is "
            "``context_window * ratio`` — a global raise can worsen context "
            "failures); it applies ONLY on the terminal/backstop turn."
        ),
    )
 # ----- IMemory subsystem — universal, per-tenant, scoped, searchable agent
 # memory. ALL default OFF / conservative so every tenant snapshot is
 # bit-identical until an operator opts in; a tenant that wants no
 # cross-session state uses session/task scope only.
 # See ``protocore.contracts.memory``. Dashboard-ready (Constants page
 # renders the toggles; admin memory API inspects/manages the records).
    memory_enabled: bool = Field(
        default=False,
        description=(
            "Master kill-switch for the IMemory subsystem. When False (default) "
            "the memory tools (remember/recall/forget) are NOT advertised to the "
            "model — they are registered pod-wide but hidden by the visibility "
            "policy, and refused at dispatch if called stale — and the auto-recall "
            "hook never fires, so every tenant is bit-identical to the pre-memory "
            "baseline. When True the tenant gets the scoped, searchable memory "
            "capability (tools + optional auto-recall), with the scope policy "
            "governed by ``memory_default_scope`` / ``memory_allowed_scopes``. "
            "Off-by-default + per-tenant override keeps the product universal."
        ),
    )
    memory_default_scope: Literal[
        "global", "user", "project", "session", "agent", "custom"
    ] = Field(
        default="session",
        description=(
            "Default :class:`~protocore.contracts.memory.MemoryScope` a "
            "``remember`` writes to (and the primary scope a ``recall`` reads) "
            "when the model does not pin an explicit scope. Default 'session' is "
            "the most-isolated scope (no cross-session leak). Product tenants "
            "may set 'user' / 'project' "
            "/ 'global' for durable cross-session memory. The host "
            "dispatcher injects the resolved value into ToolContext.metadata so "
            "pure-core tools stay RC-agnostic at invocation time."
        ),
    )
    memory_allowed_scopes: str = Field(
        default="global,user,project,session,agent,custom",
        description=(
            "Comma-separated allow-list of memory scopes the agent may "
            "request via the tool ``scope`` argument. The host dispatcher "
            "parses this into the per-call allow-list; a model request for a "
            "scope outside the list is rejected with a corrective tool error. "
            "Default permits all six scopes; a tenant that wants isolation "
            "narrows it to 'session' so a run cannot write/read cross-session "
            "memory even if the model asks. Stored as a string (not a list) to "
            "round-trip cleanly through the scalar LoopConstants catalog."
        ),
    )
    memory_write_similarity_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description=(
            "Dedup threshold for the two-stage idempotent "
            ":meth:`IMemory.write`: a candidate whose similarity to an existing "
            "record in the same (scope, scope_key, kind) bucket is >= this value "
            "is MERGED/SKIPPED instead of CREATEing a duplicate. v1 measures "
            "similarity lexically (FTS/trgm); a v2 vector backend uses cosine. "
            "0.85 is the lancedb-pro-derived default — high enough to only "
            "collapse near-identical restatements, low enough to catch obvious "
            "repeats. The store resolves this when ``write`` is called with "
            "``similarity_threshold=None``."
        ),
    )
    memory_max_records_per_scope: int = Field(
        default=0,
        ge=0,
        le=1_000_000,
        json_schema_extra={"zero_means_unlimited": True},
        description=(
            "Soft cap on the number of records the adapter retains per "
            "(tenant, scope, scope_key) bucket. 0 (default) = unbounded (rely on "
            "the future v2 decay/prune lane). A positive value lets the adapter "
            "trim the least-recently-accessed records beyond the cap on write "
            "(bounded growth without a background job). Conservative default "
            "preserves today's behaviour."
        ),
    )

 # ----- Universal resilience layer ----------------------------------------
 #
 # A backend-agnostic classify-then-act resilience layer over BOTH the LLM
 # provider calls and the tool/VM transport, plus active in-loop recovery.
 # Generalises the proven transport-stability
 # primitives — token-bucket failure-rate retry budget, deadline-aware
 # finalization reserve, jittered decorrelated backoff, classify-don't-retry
 # — into universal core knobs. Read by ``protocore.runtime.resilience``
 # (the policy + transport wrapper) and the host wiring (LLM client +
 # transport). EVERY default preserves current behaviour:
 # ``resilience_enabled=False`` makes the policy conservative, the
 # budget/backoff/reserve are all 0/off, and the recovery toggles are all
 # off → bit-identical to prior behaviour.
    resilience_enabled: bool = Field(
        default=False,
        description=(
            "Master kill-switch for the universal resilience layer "
            "(classify-then-act over LLM + tool/VM transport). When False "
            "(default) the policy returns conservative decisions and the "
            "transport wrapper degrades to a plain attempt-count loop with "
            "immediate re-issue — bit-identical to prior behaviour. Flip "
            "True per tenant (with the budget/backoff/reserve knobs) to "
            "engage the failure-rate budget + decorrelated backoff + "
            "deadline reserve."
        ),
    )
    resilience_backoff_base_seconds: float = Field(
        default=0.0,
        ge=0.0,
        le=30.0,
        description=(
            "Base for the universal decorrelated-jitter backoff between "
            "transport/LLM retries. 0.0 (default) → no sleep (immediate "
            "re-issue). A positive value enables AWS decorrelated jitter "
            "capped at "
            "``resilience_backoff_max_seconds`` (de-synchronises concurrent "
            "retriers storming the same host)."
        ),
    )
    resilience_backoff_max_seconds: float = Field(
        default=0.0,
        ge=0.0,
        le=120.0,
        description=(
            "Ceiling for the universal decorrelated-jitter backoff. Only "
            "consulted when ``resilience_backoff_base_seconds`` > 0.0. 0.0 "
            "falls back to the base (no growth)."
        ),
    )
    resilience_deadline_reserve_seconds: float = Field(
        default=0.0,
        ge=0.0,
        le=600.0,
        description=(
            "Finalization reserve (seconds): the universal policy refuses "
            "a transport/LLM retry that would leave less than this slice of "
            "the run's remaining wall-clock for a full final answer "
            "(deadline-aware retry; gRPC A6 'deadline applies across all "
            "attempts'). It ONLY ever causes an earlier give-up — it never "
            "raises the deadline. Inert (0.0 default) unless a remaining "
            "wall-clock budget is tracked (e.g. ``agent_max_seconds`` > 0)."
        ),
    )

 # ----- Active in-loop recovery (nudges) ---------------------------------
 #
 # Three active nudges turn a silent stall into a one-shot recovery
 # (post-tool empty-response, stream-stall break-it-smaller, thinking-only
 # prefill). All default-off. A run that ends without an answer is the run
 # wind-down's concern (``soft_stop_enabled``), which asks the MODEL for its
 # answer rather than assembling one on its behalf.
    resilience_post_tool_empty_nudge_enabled: bool = Field(
        default=False,
        description=(
            "Post-tool empty-response nudge. When True and the model returns "
            "an EMPTY assistant turn (no text, no tool calls, no reasoning) "
            "immediately AFTER executing tools, the loop injects a synthetic "
            "assistant('(empty)') + user('you executed tools but returned empty; "
            "process the results and continue') pair and re-streams ONCE — "
            "keeping the wire sequence API-valid (tool->assistant->user, never "
            "tool->user). Bounded by ``max_consecutive_empty_responses``. "
            "Distinct from the thinking-only-trap recovery (that path handles "
            "empty-WITH-reasoning). Default False = bit-identical."
        ),
    )
    post_tool_empty_nudge_assistant_text: str = Field(
        default="(empty)",
        description=(
            "Synthetic assistant-turn text inserted before the post-tool "
            "empty-response nudge so the wire sequence stays "
            "tool->assistant->user (never tool->user). Only used when "
            "``resilience_post_tool_empty_nudge_enabled`` is True."
        ),
    )
    post_tool_empty_nudge_user_text: str = Field(
        default=(
            "You executed tools but returned an empty response. Process the "
            "tool results above and continue: either call another tool or "
            "produce your answer."
        ),
        description=(
            "Corrective user-turn text for the post-tool empty-response nudge. "
            "Generic — no benchmark/tool/path coaching. Only used when "
            "``resilience_post_tool_empty_nudge_enabled`` is True."
        ),
    )
    request_manifest_inline_value_max_bytes: int = Field(
        default=8192,
        description=(
            "Size ceiling, in bytes of canonical JSON, for one part of a "
            "request manifest to be carried inside the manifest itself. A part "
            "over the ceiling — the ordered messages and the full tool "
            "definitions are the two that grow without bound — travels as its "
            "SHA-256 and its length, and its body goes to the blob store the "
            "host already runs, which is content-addressed so the same body "
            "stored twice costs one object. Raising this makes a manifest "
            "self-contained at the cost of megabytes per provider call on a "
            "long run; lowering it to 0 sends every part to the store. It "
            "changes what a manifest CARRIES, never what was sent: the digests "
            "and the manifest id are identical either way."
        ),
    )
    tool_result_unknown_outcome_placeholder: str = Field(
        default=(
            "The outcome of this tool call was never recorded: the run stopped "
            "after the call was dispatched and before its result was written. "
            "The call may have completed and its effects may already be in "
            "place. Check the current state before issuing it again."
        ),
        description=(
            "Content of the tool_result a resumed run substitutes for a call it "
            "had already dispatched when it stopped. NOT an error result: the "
            "call is not known to have failed, and presenting it as a failure "
            "invites the model to repeat a side effect that already happened. "
            "Used for tools whose repeat is not known to be harmless."
        ),
    )
    tool_result_unknown_outcome_repeatable_placeholder: str = Field(
        default=(
            "This tool call produced no recorded result: the run stopped while "
            "it was in flight. The tool only reads state, so issuing it again "
            "is safe."
        ),
        description=(
            "Content of the tool_result a resumed run substitutes for an "
            "in-flight call to a tool whose repeat is harmless. Same honesty as "
            "the general unknown-outcome text, minus the warning the model does "
            "not need: repeating a read changes nothing."
        ),
    )
    tool_result_approval_denied_placeholder: str = Field(
        default=(
            "This tool call was refused and was never executed. Nothing was "
            "changed by it. Do not issue it again unchanged; find another way "
            "or say why it was needed."
        ),
        description=(
            "Content of the tool_result written for a call an operator refused "
            "outright while resuming the run. Distinct from the abandoned "
            "placeholder: abandoning says no decision will ever come, while "
            "this says a decision was made and it was no. The model is told "
            "both that nothing happened and that repeating the call verbatim "
            "is not the way forward, which is what stops a refusal from "
            "becoming a retry loop."
        ),
    )
    pending_interrupt_ttl_seconds: int = Field(
        default=0,
        ge=0,
        description=(
            "How long a parked interrupt — an approval a person owes a "
            "decision on, a question a tool is waiting on — stays answerable, "
            "in seconds. Zero, the default, means it never expires: an "
            "operator decision may legitimately take a week, and a run that "
            "quietly stopped being resumable overnight is worse than one still "
            "waiting. A host that does have a deadline sets it here: the "
            "expiry is stamped on every interrupt the run parks, and a "
            "resolution that arrives after it is refused — except abandoning "
            "it, which stays available so an expired wait can still be "
            "cleared rather than stranding the run on it."
        ),
    )
    tool_result_approval_abandoned_placeholder: str = Field(
        default=(
            "This tool call was never approved and was never executed. Nothing "
            "was changed by it. Decide whether it is still needed before "
            "issuing it again."
        ),
        description=(
            "Content of the tool_result written for a call that was parked at "
            "an approval gate when the caller resumed the run while abandoning "
            "the approval. Not an error result and not an unknown outcome: the "
            "call demonstrably did not run, and saying so is what stops the "
            "model from treating a decision nobody made as a failure."
        ),
    )

    # ----- Live-run guardrails and interaction (all default off) -----
    loop_guard_enabled: bool = Field(
        default=False,
        description=(
            "When true, a repeating text or thinking tail is cut from the "
            "live stream and identical tool+args calls stop being executed "
            "before max_iterations is burned. Off by default so existing "
            "tenants keep prior loop behaviour."
        ),
    )
    loop_guard_nudge_max: int = Field(
        default=1,
        ge=0,
        description=(
            "How many times the loop may nudge the model to change course "
            "after a repeating stream or identical tool call before the "
            "turn ends with a refused notice."
        ),
    )
    loop_guard_repeat_window_tokens: int = Field(
        default=32,
        gt=0,
        description=(
            "Token-sized window used to detect a repeating passage inside "
            "one streamed answer or thinking channel."
        ),
    )
    loop_guard_repeat_min_chars: int = Field(
        default=24,
        gt=0,
        description=(
            "Minimum character length of a passage before the repeating-"
            "stream detector may fire. Short echoes are ignored."
        ),
    )
    loop_guard_identical_tool_limit: int = Field(
        default=3,
        gt=0,
        description=(
            "How many times the same tool with the same canonical arguments "
            "may execute in one turn before further identical calls are "
            "recorded as results and not run."
        ),
    )
    result_eviction_enabled: bool = Field(
        default=False,
        description=(
            "When true, unmarked Read/Grep tool results are replaced with "
            "placeholders on the next LLM request. Persist keeps the full "
            "result. Off by default."
        ),
    )
    result_eviction_tool_names: tuple[str, ...] = Field(
        default=("Read", "Grep", "read", "grep"),
        description=(
            "Tool names whose results result-eviction may replace with a "
            "placeholder in the next LLM request. The default names the "
            "read-shaped tools of a coding backend, but the set is a TENANT "
            "policy, not a core invariant: a backend whose bulky repeated "
            "results come from other tools (a world-observation tool in a "
            "simulation, a report query in an analytics agent) names them "
            "here instead. An empty tuple does NOT switch eviction off — it "
            "falls back to whatever this deployment declared as its "
            "workspace-inspection tools, which is a wider set than the names "
            "above and not a narrower one. Switching eviction off is "
            "``result_eviction_enabled``."
        ),
    )
    result_eviction_keep_marked: bool = Field(
        default=True,
        description=(
            "When result eviction is on, results the model marked useful "
            "stay verbatim in the next LLM request."
        ),
    )
    run_settled_enabled: bool = Field(
        default=False,
        description=(
            "When true the loop emits a run_settled event only after "
            "compaction, retry and follow-up placement are finished. Off "
            "by default: clients keep using message_stop / run_completed."
        ),
    )
    steer_follow_up_enabled: bool = Field(
        default=False,
        description=(
            "When true, steer (after the current tool batch, before the next "
            "LLM call) and follow_up (after run_settled) queues are active. "
            "Off by default."
        ),
    )
    steer_default_mode: Literal["one-at-a-time", "all"] = Field(
        default="one-at-a-time",
        description="How many pending steer items are placed in one insertion.",
    )
    follow_up_default_mode: Literal["one-at-a-time", "all"] = Field(
        default="one-at-a-time",
        description="How many pending follow-up items are placed in one insertion.",
    )
    max_queued_items: int = Field(
        default=8,
        gt=0,
        description="Maximum pending steer plus follow-up items on one session.",
    )
    max_queued_chars: int = Field(
        default=8000,
        gt=0,
        description="Maximum characters of one queued steer or follow-up item.",
    )

    # ----- Long work, operator control, context (all default off) -----
    background_tasks_enabled: bool = Field(
        default=True,
        description=(
            "Master switch for the session work pool: the place a piece of "
            "work runs when the caller is not going to sit inside it. Two "
            "things draw on it. A shell command may run as a session-scoped "
            "background task whose process group outlives the turn that "
            "started it, and a delegation may be launched into the pool "
            "instead of awaited, so the leader is handed the task's address "
            "at once and woken when it settles.\n\n"
            "On by default. It was off while the pool was only an option for "
            "Bash, and leaving it off once delegation started drawing on the "
            "same switch made the product's own answer to a long child run "
            "unreachable: a background delegation was refused, the caller "
            "read the refusal as advice and re-issued the same batch inline, "
            "and the run spent the child's whole duration blocked in the tool "
            "it had asked not to block in. An installation that wants neither "
            "long-running commands nor detached delegation turns it off, and "
            "both fall back to running in the turn."
        ),
    )
    child_run_retire_grace_seconds: float = Field(
        default=3.0,
        ge=0.0,
        description=(
            "Seconds a finishing child run allows its own background tasks to "
            "end before they are forced. Spent inside the child's own time "
            "budget, not on top of it, so a child that retires slowly is late "
            "rather than over its ceiling."
        ),
    )
    execution_profile_plan_enabled: bool = Field(
        default=False,
        description=(
            "When true, execution_profile=plan is a published tool allowlist "
            "intersected with existing visibility. Off by default."
        ),
    )
    execution_profile_plan_tools: str = Field(
        default="Read,Glob,Grep,WebFetch,AskUser",
        min_length=1,
        description=(
            "Comma-separated tool names advertised under the plan profile. "
            "Write/Edit/Bash-class names must be omitted for a read-only plan."
        ),
    )
    permission_widening_enabled: bool = Field(
        default=False,
        description=(
            "When true, an approval may widen to a program or multiplexer "
            "verb for one plain invocation. Off by default."
        ),
    )
    permission_widening_multiplexer_verbs: str = Field(
        default="git,go,npm,pnpm,yarn,docker,kubectl,helm,make,cargo,uv,pip,terraform",
        min_length=1,
        description="Comma-separated programs whose first argument is a verb.",
    )
    compaction_reserve_tokens: int = Field(
        default=16384,
        gt=0,
        description="Compact when context tokens exceed the window minus this reserve.",
    )
    compaction_manual_enabled: bool = Field(
        default=False,
        description=(
            "When true, an operator /compact or POST compact writes a "
            "checkpoint the next LLM request cannot read through. Off by default."
        ),
    )
    rules_discovery_enabled: bool = Field(
        default=False,
        description=(
            "When true, nested AGENTS.md files are discovered and their "
            "bodies inject only after a filesystem-tool touch. Off by default."
        ),
    )
    rules_max_body_bytes: int = Field(
        default=8192,
        gt=0,
        description="Maximum bytes of one AGENTS.md body injected into the prompt.",
    )
    rules_max_active: int = Field(
        default=16,
        gt=0,
        description="Maximum nested rule bodies active on one session.",
    )
    rules_workspace_trust: Literal["never", "allowlist", "always"] = Field(
        default="never",
        description=(
            "Whether a workspace-written AGENTS.md may activate. never is "
            "the multi-tenant default."
        ),
    )
    rules_skip_dir_names: str = Field(
        default="node_modules,vendor",
        description="Comma-separated directory names skipped during discovery.",
    )
    skills_hot_reload_enabled: bool = Field(
        default=False,
        description=(
            "When true, the next run rebuilds the skill index from the store "
            "instead of a process-lifetime cache. Off by default."
        ),
    )
    tool_result_split_enabled: bool = Field(
        default=False,
        description=(
            "When true, tool results keep a short model content and a UI "
            "details payload. Off by default."
        ),
    )
    tool_result_content_max_chars: int = Field(
        default=8000,
        gt=0,
        description="Maximum characters of tool result content sent to the next LLM request.",
    )

    # ----- Intent, ledger, session tree, lanes (all default off) -----
    intent_settlement_enabled: bool = Field(
        default=False,
        description=(
            "When true, a mutating tool commits an intent with reserved "
            "result ids before dispatch. Off by default."
        ),
    )
    intent_never_replay_tools: str = Field(
        default="Write,Edit,Bash,Finalize,AppendFile",
        min_length=1,
        description="Comma-separated tool names whose crash must not replay.",
    )
    intent_repeat_safe_tools: str = Field(
        default="Read,Grep,Glob,ToolSearch",
        description=(
            "Comma-separated tool names whose repeat is harmless, so a call "
            "interrupted mid-flight may be re-issued without checking what it "
            "left behind. Decides two things: whether a call pays for a "
            "durability write before it is made, and which of the two "
            "unknown-outcome texts a resumed run gives the model. Only tools "
            "that read state belong here; delegation does not, because a "
            "repeated delegation starts a second subtree."
        ),
    )
    usage_ledger_enabled: bool = Field(
        default=False,
        description=(
            "When true, every settled attempt appends a usage row including "
            "fail, retry, and compaction. Off by default."
        ),
    )
    lanes_enabled: bool = Field(
        default=False,
        description=(
            "When true, a session has a main lane plus optional extra lanes "
            "with exclusive per-lane locks. Off by default."
        ),
    )
    lanes_max_per_session: int = Field(
        default=4,
        gt=0,
        description="Cap on lanes in one session including main.",
    )
    typed_hooks_enabled: bool = Field(
        default=False,
        description=(
            "When true, registrations on the lifecycle seam run at the "
            "coordinates they were placed on. Off by default."
        ),
    )
    telemetry_spans_enabled: bool = Field(
        default=False,
        description=(
            "When true, run/turn/step/tool/compact/hook spans are recorded. "
            "Off by default."
        ),
    )

    @model_validator(mode="after")
    def _validate_relationships(self) -> Self:
 # routine trigger must be strictly below emergency cliff
        if self.compaction_trigger_ratio >= self.compaction_emergency_ratio:
            raise ValueError(
                "compaction_trigger_ratio must be < compaction_emergency_ratio"
            )
 # combined overhead budgets must leave room for history
        fixed_overhead = (
            self.system_prompt_max_ratio
            + self.skill_index_budget_ratio
            + self.loaded_skills_ratio
            + self.tool_definitions_ratio
            + self.user_context_ratio
        )
        if fixed_overhead >= 1.0:
            raise ValueError(
                "system + skill + tool + user budgets must sum to < 1.0"
            )
 # stall threshold must be strictly less than idle timeout —
 # otherwise the warning-vs-abort distinction collapses.
        if (
            self.llm_stream_stall_threshold_seconds
            >= self.llm_stream_idle_timeout_seconds
        ):
            raise ValueError(
                "llm_stream_stall_threshold_seconds must be < "
                "llm_stream_idle_timeout_seconds"
            )
 # the reasoning-aware extended idle
 # timeout MUST be at least as large as the baseline idle timeout
 # so it can only widen the watchdog window, never tighten it.
        if (
            self.llm_stream_reasoning_idle_timeout_seconds
            < self.llm_stream_idle_timeout_seconds
        ):
            raise ValueError(
                "llm_stream_reasoning_idle_timeout_seconds must be >= "
                "llm_stream_idle_timeout_seconds"
            )
 # Universal resilience monotonic invariants. The backoff
 # ceiling must not sit below the base when growth is enabled
 # (else the decorrelated-jitter cap would clamp below its own
 # floor). Inert when base is 0.0 (no backoff configured).
        if (
            self.resilience_backoff_base_seconds > 0.0
            and self.resilience_backoff_max_seconds > 0.0
            and self.resilience_backoff_max_seconds
            < self.resilience_backoff_base_seconds
        ):
            raise ValueError(
                "resilience_backoff_max_seconds must be >= "
                "resilience_backoff_base_seconds when backoff is enabled"
            )
        return self

@runtime_checkable
class RuntimeConstantsProvider(Protocol):
    """Provider Protocol — the host reads PG, watches Redis, builds snapshots.

    Per-tenant call. Snapshot is always fresh-as-of-now (provider handles
    cache invalidation under the hood).
    """

    async def get(self, tenant_id: str) -> LoopConstants:
        """Return the latest snapshot for ``tenant_id``."""
        ...


__all__ = ["LoopConstants", "RuntimeConstantsProvider"]
