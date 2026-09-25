# Glossary

Concise definitions of the core's key terms. Each entry maps to a real symbol in
the core and stays consistent with
[`architecture.md`](architecture.md). Terms are grouped by area, not
alphabetised, so related concepts read together.

## Runtime entry points

**`QueryEngine`** (`runtime/query_engine.py`)
: One instance per active run. Owns the **mutable per-conversation state** —
  `history`, the `LoopState` machine, `CompactionState`, `TokenUsage`, plus
  `open_intents`, `usage_rows`, `lanes`, live-control queues, live
  model/thinking overrides, optional `verification`, and recovery latches —
  and persists it via `snapshot()` ↔ `resume_from_snapshot()`, so any pod can
  resume a run another pod started. Construction-time injection lives on the
  immutable `QueryEngineConfig` (including `run_mode`, `tool_preconditions`,
  and optional `provider_chain`). See
  [ReAct loop / orchestrator / query engine / loop state](architecture.md#react-loop--orchestrator--query-engine--loop-state).

**`resume()`** (`runtime/query.py`)
: The one public entry that picks a stored run back up:
  `resume(engine, snapshot, *, approved_tool_call=None, message=None,
  abandon_approval=False, resolutions=None, allow_partial_resolution=False)`,
  an async iterator of `TurnEvent`. It restores the
  snapshot strictly — schema, delivery mode and identity binding are settled
  before the first mutation, so a snapshot from another run is refused and
  nothing is driven — then selects the drive the caller's arguments describe:
  the resolution map, the approved tool call, the message that has arrived, or
  a plain re-drive of the interrupted turn. A run with anything parked refuses
  a plain re-drive, naming the call: it is not a decision on it, and walking
  past one would leave it unanswered and report it to the model as a
  failure. `abandon_approval=True` is how a caller says the decision will
  never come; every parked call is closed with a result saying it was never
  approved and never ran.
  Import it from `protocore.runtime`. See
  [ReAct loop / orchestrator / query engine / loop state](architecture.md#react-loop--orchestrator--query-engine--loop-state).

**`resume_interrupts()`** (`runtime/query.py`)
: Answers everything a run is waiting on, in one drive — the general form, and
  the only one that can express a batch. Three dangerous calls in one assistant
  message park three interrupts and stop the run once; this answers all three
  (approve the first with corrected arguments, deny the second, abandon the
  third) and the results land in history in the order the model asked for them.
  The map is checked in full BEFORE anything executes, because a resume that
  ran two resolutions and then found the third incoherent would already have
  dispatched tools on the strength of a decision set that turned out to be
  nonsense. `allow_partial=True` relaxes what the map must cover and nothing
  else. Reached through `resume(resolutions=...)` in the usual case.

**`PendingInterrupt`** (`contracts/interrupt.py`)
: What a paused run is waiting for, as a value rather than a latch:
  `interrupt_id`, `kind`, the parked call, what the person is being shown, and
  when the wait started and stops being answerable. `InterruptKind` is
  `approval` (a gate parked the call; nothing ran), `question` (the tool ran far
  enough to ask, and the answer is its result) or `external_call` (the result
  arrives by another route), and the kind decides which `InterruptResolution` is
  legal. A run holds however many are open at once; `LoopState.AWAITING` with
  none recorded is refused at the transition, because that is a run that stops
  with nothing that could resume it.

**`resume_approved_tool()`** (`runtime/query.py`)
: Executes one call that was held for approval, verified against the durable
  pending call and idempotent on replay. Reached through `resume()` in the
  usual case; exported from `protocore.runtime` for a caller that has already
  restored its engine. The ReAct turn body itself (`_query_raw`) is private —
  every drive of it binds the driving task and persists a closing snapshot, and
  each inner `yield` is a stop-check checkpoint.

## The three run-state concepts (do not conflate)

These three names are **distinct** and live in different layers; mixing them is a
common error.

**`LoopState`** (`runtime/loop_state.py`, `StrEnum`)
: The **in-turn loop finite-state machine** held on the `QueryEngine` instance —
  the engine's live in-flight state. Seven states:
  `PENDING → RUNNING → {AWAITING | COMPACTING} → {COMPLETED | FAILED | CANCELLED}`.
  `assert_transition()` enforces the legal-edge table; `TERMINAL_STATES` have no
  outgoing edges. See
  [ReAct loop / orchestrator / query engine / loop state](architecture.md#react-loop--orchestrator--query-engine--loop-state).

**`RunStatus`** (`contracts/types.py`, `StrEnum`)
: The **durable run lifecycle** mirrored on the Postgres `runs.status` column:
  `queued | running | completed | partial | error | cancelled | incomplete |
  paused`. `partial` is functionally terminal (loop finished but accumulated tool
  errors), distinct from `completed` and `error`. This is the persisted record,
  not the in-memory loop state.

**`RunState`** (`contracts/types.py`, `BaseModel`)
: The **ephemeral hot working set** held in the Redis hash `run:{id}` — `run_id`,
  `tenant_id`, the current `RunStatus`, `current_turn`, token counters,
  `last_event_id`. A mutable model, not an enum; distinct from the durable `Run`
  record.

## Configuration & constants

**`LoopConstants`** (`contracts/runtime_constants.py`)
: The single mechanism for tunable values — **no inline magic numbers**. A frozen
  Pydantic snapshot (`ConfigDict(frozen=True, extra="forbid")`); every tunable is
  a default-safe field, served per-tenant by a `RuntimeConstantsProvider`.
  `extra="forbid"` means an unknown key is a validation error (rejected), not
  silently dropped, so **core and the host must deploy paired**. See
  [LoopConstants system](architecture.md#loopconstants-system) and
  [`runtime-constants.md`](runtime-constants.md).

## Extension protocols

**`IMemory`** (`contracts/memory.py`, `Protocol`)
: The contract for a **typed, scope-aware, retrieval-ranked memory** of facts the
  agent learns and re-uses (distinct from session transcripts, blobs, the search
  index, and todos). A record is addressed by `(tenant_id, scope, scope_key)`;
  the most-isolated default scope is `session`. Core never imports the
  implementation; the host provides a durable store with lexical recall.
  Default-off
  (`memory_enabled = False`). See
  [IMemory](architecture.md#technology-inventory).

**`IWorkspace`** (`contracts/workspace.py`, `Protocol`)
: The contract for a **session/task-scoped, searchable, atomic scratch
  workspace** — the agent dumps intermediate data once and re-reads/searches it
  many times (a dump-once / re-read-many stability lever). Backs the
  `read`/`write`/`find`/`search` verbs. Host-wired, and the host owns the
  availability flag. See
  [IWorkspace + read-dedup cache](architecture.md#technology-inventory).

## Resilience & finalization

**Adaptive safety band** (host-owned)
: A per-`(provider, model)` band that **learns from token-estimator drift** and
  subtracts a calibrated margin from the per-call output budget, so
  `prompt + max_tokens` stays under the provider window even when the local
  estimator misjudges (e.g. Cyrillic-in-JSON-escape inflation). When no band is
  wired, behaviour is identical to pre-band. See
  [Attempt ledger + adaptive safety band](architecture.md#technology-inventory).

**`AttemptLedger`** (`contracts/attempt_ledger.py`)
: A record of what a (sub)agent **declared** it would produce
  (`DeliverableDeclaration`) and what was **actually verified**
  (`VerificationRecord`), so finalization can decide an honest outcome. Its
  `LedgerOutcome` is a neutral literal (`completed | partial | failed | unknown`),
  not a backend enum; the agent's `SelfReportedStatus` is kept but not trusted
  blindly. See
  [Attempt ledger + adaptive safety band](architecture.md#technology-inventory).

Finalization gate (host-owned)
: The terminal-path guard that closes a finalization gap: a (sub)agent that wrote
  the user-visible artifact but ran out of iterations without calling its
  terminal tool would otherwise be scored "failed". The gate **verifies**
  declared deliverables — it stats each one through the workspace shape the host
  injects — and **decides** the outcome the leader's final turn reports
  (success / partial / failed). All toggles default `False`. See
  [Finalization gate + contract](architecture.md#technology-inventory).

## Grounding & terminal answers

**Grounding / references**
: The deterministic, rubric-blind discipline that the terminal `answer`'s
  citations must be a **subset of what was actually `read`**. A grounding-tracked
  `read` records its path as observed evidence; which tools record is the
  host's binding, not a core constant — a tool carrying the `reads_path` role
  is what the core recognises, never a tool name. Reference normalisation is
  host-owned: a pure, idempotent projection that compares refs on a canonical
  form, so a flat-vs-branded path mismatch is not a false veto; it can only
  remove a false veto, never add one. See
  [Terminal-answer validation + references / grounding + payload normalize](architecture.md#technology-inventory).

## Context, caching & compaction

**Prompt-cache breakpoints** (`runtime/prompt_caching.py`)
: Placement **hints only** for provider prefix-caching. `apply_system_and_3(...)`
  computes the system-and-three strategy: at most `MAX_BREAKPOINTS`
  `CacheBreakpoint`s — system at index 0 plus the last three non-system
  messages. The core always emits the
  hints on `LLMRequest.extra["cache_breakpoints"]`; the host adapter
  translates them into whatever cache markers its provider's wire uses (behind
  a host kill-switch),
  and adapters that don't recognise
  the key ignore it. See
  [Context management and two-tier compaction](architecture.md#technology-inventory).

**Compaction tiers / layers** (`runtime/context/compaction.py`)
: The **two-tier** cascade that keeps the prompt under the provider context
  window across a long run. There is no Tier 3 in this module. **Tier 1**
  (`run_tier1_truncation`) truncates / blobs oversized tool results, replacing
  the body with a placeholder + blob ref. **Tier 2**
  (`run_tier2_summarisation`) replaces whole old non-system turns with a system
  summary, keeping the recent N turns. A pass that tried and failed is
  charged once to a retry budget (`compaction_failed_max_retries`; the
  reactive pass after a provider rejection has its own), and a proactive pass
  with nothing eligible is not opened at all. Past the budget
  `CompactionExhaustedError` on a proactive pass suspends the proactive
  summariser tiers for a bounded stretch (Tier 1 keeps running; a context
  refusal, `rearm()` or a snapshot resume ends it sooner); on a reactive pass it
  hands the turn to the output-cap ladder, and only when no smaller cap is left
  does the loop go to `FAILED`. Operator
  `/compact` is a separate `CompactCheckpoint` path
  (`runtime/compact_checkpoint.py`, `compaction_manual_enabled` default
  `False`). Cross-run fold lives in `runtime/context/session_memory.py`
  (`fold_run`). Triggers and ratios are RC-driven and derived in
  `runtime/context/budgets.py`. See
  [Context management and two-tier compaction](architecture.md#technology-inventory).

## Intent, usage ledger, session tree, lanes, typed hooks, telemetry

These six surfaces are **default-off**. Read the live `Field(...)` default; do
not infer that shipping the code turns them on.

**`IntentRecord`** (`runtime/intent.py`)
: Per-tool-call durability record (`operation_id`, reserved result ids,
  `replay` `never|safe`, and a lifecycle `state`
  `RESERVED|PENDING_APPROVAL|DISPATCHED|PAUSED_ASK_USER|SETTLED`). Written
  before the call is made, unconditionally — not behind a flag, and not only
  for mutating tools. The state is what a resumed run reads: only `DISPATCHED`
  with no result anywhere in history becomes an honest "outcome never
  recorded"; a call parked at a gate, one waiting on a user's answer and one
  merely reserved are none of them unknown outcomes and none of them are
  executed by a resume. `intent_repeat_safe_tools` (default
  `Read,Grep,Glob,ToolSearch`) decides which calls skip the durability write
  and get the milder recovery text; `intent_never_replay_tools` (default
  `Write,Edit,Bash,Finalize,AppendFile`) sets `replay`.
  `intent_settlement_enabled` now gates only the recovery events and the
  ledger row, not the record. Persisted on `QueryEngine.open_intents`. See
  [Intent, usage ledger, session tree, lanes](architecture.md#intent-usage-ledger-session-tree-lanes-typed-hooks-telemetry-live-control-run-work-budget).

**`UsageRow`** (`runtime/usage_ledger.py`)
: One append-only ledger line (`seq`, `kind`, token counts, `success`,
  optional `operation_id`). When `usage_ledger_enabled` is on,
  `correctness_bind.commit_usage` records `inference` / `retry` / `compaction`
  / `abort` / `fail`. A **tool**-kind row is written only on the
  intent-settlement dispatch path. A failed attempt plus its retry is two
  rows. Persisted on `QueryEngine.usage_rows`.

**Session branch** (host-owned)
: A forked or cloned copy of a history path that does **not** mutate the
  source. Gated by a host knob; a clone requires a settled source, and a second
  host knob caps the number of copied messages. **Host-owned** — the loop does
  not branch a session.

**`Lane`** (`runtime/lanes.py`)
: A named cursor over shared history. `main` always exists; extras take
  exclusive locks (`create_lane` / `acquire_lane` / `release_lane`). Gated by
  `lanes_enabled`; `lanes_max_per_session` (default 4) includes main.
  **Host-invoked**; `QueryEngine.lanes` is snapshot-persisted.

**Lifecycle coordinates** (`contracts/types.py::HookEvent`)
: The coordinates the loop dispatches on the lifecycle seam, named by
  `HookEvent`: `run_start`, `turn_start`, `context_transform`,
  `request_prepare`, `response_received`, `request_error`, `turn_end`,
  `pre_tool_use`, `tool_execute`, `post_tool_use`, `pre_compact`,
  `compaction_commit`,
  `compaction_rollback`, `post_compact`, `run_finalize`. Registrations run
  through `HookManager` when `typed_hooks_enabled` is on — that flag is the
  only switch over the seam. A `transform` at `context_transform` is applied to
  the turn's context.

**`Span`** (`runtime/telemetry.py`)
: A low-cardinality telemetry span. Allowed names: `run` / `turn` / `step` /
  `tool` / `compact` / `hook`. Gated by `telemetry_spans_enabled`.
  `is_prometheus_safe_label` refuses `session_id` / `lane_id` / `operation_id`
  / `run_id` as label keys. `mark_recovery` tags a span when an interrupted
  intent is resumed. Lives on `engine.spans` (in-process, **not** snapshotted).

**`IProviderChain`** (`contracts/llm.py`)
: Ordered remaining providers plus a one-way `advance()` cursor. Injected on
  `QueryEngine(..., provider_chain=...)` so a mid-stream provider failure can
  step to the next rung without un-publishing already streamed deltas.

## Skill files

**`SkillFileRef`** (`contracts/skills.py`)
: One file in a multi-file skill bundle: bundle-relative `path`, `size_bytes`,
  `mime_type`, `content_hash` (lowercase hex SHA-256). Every bundle has at
  least the canonical `SKILL_ENTRY_PATH` row (`SKILL.md`, MIME
  `text/markdown`). Bytes come from `ISkillStore.load_file`; the core loop
  does **not** call `list_files` / `load_file` (those are for hosts that
  expose helper files). Catalog rendering uses `store.list` and emits
  `Skill(skill="{name}")` call shapes, not file paths. Store reads key on
  `QueryEngineConfig.account_id`, not `tenant_id`. See
  [Skills routing / surfacing](architecture.md#skills-routing--surfacing).
