# Protocore Core — Architecture

> Audience: an engineer onboarding to the **pure core** (`protocore/`).
> Scope: this document describes the current core library (`protocore/`) only.
> The host adapters, the FastAPI service, frontends, and
> deployment live in sibling repos and are referenced here only at the boundary.

---

## Overview

`protocore` is the **pure core** of the Protocore agent runtime. It is a
Python 3.12+ library of **contracts (protocols + typed models)** and a
**protocol-first ReAct runtime** that drives one agent turn at a time.

It is, by design, a **universal product core**, not a benchmark harness:

- **Zero upward imports.** Core never imports a package that sits above it —
  anything sharing its name with an underscore after it (`protocore_*`). It has
  no database driver, no HTTP endpoint, no orchestration logic. Everything
  outside-facing is a `Protocol` the host implements. Enforced by
  `tests/test_core_import_boundary.py`.
- **Universal / multi-tenant.** No per-task, per-tenant-id, per-prompt, or
  scorer/rubric-shaped logic in any executable path. Every method is
  tenant-scoped; tenant policy is injected (via `LoopConstants` and
  `ToolContext.metadata`), never hard-coded.
- **Everything `LoopConstants`-configurable and default-safe.** Tunable
  values flow through `LoopConstants` (a frozen Pydantic snapshot) or
  `constants.py` (memory-safety caps). New capabilities default **off** or to a
  value that reproduces prior behaviour, so a tenant opts in deliberately.
- **Horizontal-scale-safe.** No module-level dicts, no `asyncio` locks held as
  module state, no per-process authority. Durable and ephemeral cross-process
  state are both provided across the boundary. Correctness-affecting state
  lives per-run on the `QueryEngine` instance.

### Dependency direction

```
protocore (pure core, no upward imports)
  └─> a host distribution (adapters, service layer, HTTP API)
        ├─> frontends (HTTP/SSE only)
        └─> an execution backend (service API contracts only)
```

`protocore` is the root. It must never import upward. The guard test asserts
that importing any `protocore.*` module pulls in **zero** symbols from the
layers above it.

### Public API (`protocore/__init__.py`)

The public surface is **contract-first**: the re-exports are the store /
service interface `Protocol`s the host reaches for most, plus the `Tool` ABC —
`IAgentDispatch`, `IBlobStore`, `IEventStream`, `IHookManager`,
`ILifecycleRegistry`, `ILLMProvider`, `IRunStore`, `ISearchIndex`,
`ISessionStore`, `ISkillStore`, `IToolRegistry`, `ITodoStorage`, and `Tool`.
The rest of the 32 `Protocol`s are imported from their own contract module and
are **not** top-level re-exports: `IMemory` (`contracts/memory.py`),
`IWorkspace` (`contracts/workspace.py`), `IToolTransport` and
`IResilienceClassifier` (`contracts/resilience.py`), `IPromptTemplateProvider`
(`contracts/prompts.py`), `IWorkPool` (`contracts/background.py`),
`IConstantsRegistry` and `ICoreConstantsProvider` (`contracts/config.py`),
`IRequestManifestSink` (`contracts/observability.py`), `IProviderChain`
(`contracts/llm.py`) and `ITurnPolicy` (`contracts/turn_policy.py`).
The surface also re-exports the core type system (`Message`, `ToolCall`,
`ToolResult`, `Event`, `Run`, `Session`, `SubagentDef`, the `ContentBlock`
union, …), `LoopConstants` + `RuntimeConstantsProvider`, the lifecycle
vocabulary (`RegistrationKind`, `LifecycleVerdict`, `LifecycleContext`,
`LifecycleDecision`, `LifecycleOutcome`, `LifecycleScope`,
`LifecycleDisposer`), `EventBus`/`EventName`, the lifecycle `HookManager`,
`DefaultShellSafetyPolicy`, the `@tool` decorator, the envelope/JSON
utilities, and the token-counting helpers (`LanguageProfile`,
`chars_per_token`, `detect_profile`, `estimate_tokens`). It does **not**
re-export `derive_budgets`, `retrieve_tools`, or `bm25_score` — those are
imported directly from their runtime modules (`runtime/context/budgets.py`,
`runtime/tool_retrieval.py`).

The loop machinery (`runtime/query.py` + `runtime/query_engine.py`) is the heart
of the runtime. The loop entry points are imported directly from
`protocore.runtime.query` / `protocore.runtime.query_engine`; they are **not**
re-exported at the top level.

---

## Architecture diagrams

### Layered structure

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ CONTRACTS / PROTOCOLS  (protocore/contracts/ — 30 modules, 32 Protocols)        │
│   types.py  (Message, ToolCall, ToolResult, ContentBlock union, Run, Session,   │
│              ExecutionReport, StopReason, SubagentDef, AgentEnvelope, …)         │
│   the host's adapters: llm.py ILLMProvider + IProviderChain · run.py             │
│     IRunStore + IRunToolErrorCounter · session.py ISessionStore ·                │
│     blob.py IBlobStore · search.py ISearchIndex · todo.py ITodoStorage ·         │
│     tool_registry.py IToolRegistry · skills.py ISkillStore ·                     │
│     agent_dispatch.py IAgentDispatch + IDelegationTool ·                         │
│     background.py IWorkPool + IBackgroundTaskPool + WorkHandle ·                 │
│     events.py IEventStream · hooks.py IHookManager ·                             │
│     memory.py IMemory + IMemoryContentScanner · workspace.py IWorkspace ·        │
│     resilience.py IToolTransport + IResilienceClassifier ·                       │
│     prompts.py IPromptTemplateProvider · middleware.py ILifecycleRegistry ·      │
│     observability.py CacheObserverProtocol + IRequestManifestSink ·              │
│     config.py IConstantsRegistry + ICoreConstantsProvider                        │
│   runtime_constants.py  LoopConstants (frozen, extra="forbid") + Provider        │
│   config.py  ConstantSpec · ConstantGroup · group_from_model                     │
│   turn_policy.py ITurnPolicy · ITurnState · TurnFlags · TurnCoordinate           │
│   interrupt.py PendingInterrupt · InterruptResolution                            │
│   run_state.py RunScopedState · tool_roles.py ToolRole + ToolRoleMap             │
│   snapshot.py schema version + upcaster chain · attempt_ledger.py ·              │
│   evidence.py · tool_chunking.py · tools.py Tool + ToolContext                   │
└──────────────────────────────────────────────────────────────────────────────┘
                                     ▲ implemented by the host / consumed by runtime
┌──────────────────────────────────────────────────────────────────────────────┐
│ RUNTIME — ORCHESTRATION  (protocore/runtime/)                                   │
│                                                                                │
│   QueryEngine (query_engine.py) ── owns mutable per-run state:                  │
│        history · LoopState · CompactionState · TokenUsage ·                     │
│        open_intents · usage_rows · lanes · live_* · steer/follow-up queues ·    │
│        verification · recovery latches ·                                        │
│        snapshot()/resume_from_snapshot()  (any pod can resume)                  │
│   run(message) ── appends, snapshots, drives one turn of TurnEvent              │
│   resume(engine, snapshot, ...) (query.py) ── restore + pick the drive:          │
│        resolution map | approved tool call | arrived message | re-drive        │
│   resume_approved_tool(engine, call) ── run one call held for approval          │
│   resume_interrupts(engine, resolutions) ── answer every wait, in one drive     │
│   build_llm_request(...) ── the one assembler every provider call goes through  │
│   turn_policies/ ── the product decisions of a turn, in TURN_POLICY_ORDER       │
│   loop_strategies.py ── DirectStrategy | DeepStrategy (run_mode)                │
│   intent.py · usage_ledger.py · lanes.py · error_kinds.py ·                      │
│   telemetry.py · correctness_bind.py · child_capabilities.py ·                   │
│   compact_checkpoint.py · live_control.py · run_work_budget.py                  │
│        LoopState (loop_state.py): PENDING→RUNNING→{AWAITING|COMPACTING}→         │
│                                   {COMPLETED|FAILED|CANCELLED}                   │
└──────────────────────────────────────────────────────────────────────────────┘
        │                 │                       │                    │
        ▼                 ▼                       ▼                    ▼
┌───────────────┐ ┌────────────────┐ ┌──────────────────────┐ ┌────────────────┐
│ TOOL SURFACE  │ │ TOOL DISPATCH  │ │ CONTEXT / COMPACTION  │ │ FINALIZATION   │
│ + RETRIEVAL   │ │ + GATING       │ │ context/manager.py    │ │ + GROUNDING    │
│ tool_registry │ │ tool_dispatch  │ │ context/budgets.py    │ │ host-owned:    │
│ tool_retrieval│ │ ToolDispatcher │ │ context/compaction.py │ │  the gate and  │
│ @tool decorat.│ │ tool_permission│ │ context/session_      │ │  the contract  │
│ tool_roles    │ │   Gate (4 stg) │ │   memory.py           │ │ core-owned:    │
│  ToolRoleMap  │ │ tool_precondi- │ │ compact_checkpoint.py │ │  evidence.py · │
│               │ │   tions (DAG)  │ │ token_counting.py     │ │  attempt_      │
│               │ │                │ │                       │ │  ledger.py     │
│               │ │ run_tool_pre-  │ │ prompt_caching.py     │ │                │
│               │ │   conditions   │ │ json_utils strip-     │ │                │
│               │ │   (run forcer) │ │   thinking            │ │                │
└───────────────┘ └────────────────┘ └──────────────────────┘ └────────────────┘
        │                 │                       │                    │
        ▼                 ▼                       ▼                    ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ MEMORY       │ │ WORKSPACE    │ │ RESILIENCE   │ │ SKILLS       │ │ HOOKS/EVENTS │
│ contracts/   │ │ contracts/   │ │ contracts/   │ │ skill_index  │ │ events.py    │
│   memory.py  │ │  workspace.py│ │  resilience  │ │ contracts/   │ │ runtime/     │
│ tools/       │ │ (IWorkspace) │ │  IToolTrans- │ │  skills.py   │ │  events/*    │
│   memory.py  │ │ host-owned:  │ │  port +      │ │  list_files/ │ │ runtime/llm/ │
│ (IMemory)    │ │  the store   │ │  IResilience-│ │  load_file   │ │  delta_bridge│
│              │ │  and the     │ │  Classifier  │ │  (host API)  │ │ hooks/       │
│              │ │  read-dedup  │ │ runtime/     │ │              │ │  manager +   │
│              │ │  cache       │ │  resilience  │ │              │ │ middleware   │
│              │ │              │ │ attempt_     │ │              │ │  contract    │
│              │ │              │ │  ledger ·    │ │              │ │              │
│              │ │              │ │ run_work_    │ │              │ │              │
│              │ │              │ │  budget      │ │              │ │              │
│              │ │              │ │              │ │              │ │              │
└──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
        │                 │                 │                 │                 │
        ▼                 ▼                 ▼                 ▼                 ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ SAFETY  (protocore/safety/)  shell.py DefaultShellSafetyPolicy + deny patterns │
│         + chain_parser.py (segment/substitution grammar)                       │
├──────────────────────────────────────────────────────────────────────────────┤
│ CONFORMANCE  (protocore/conformance/, installed with protocore[testing])       │
│   SUITES — one suite per contract, bound by the host to its own adapter        │
├──────────────────────────────────────────────────────────────────────────────┤
│ HOST-ADAPTER BOUNDARY  (lives in the host distribution — NOT core)            │
│   an OpenAI-compatible ILLMProvider · a durable IMemory · an IWorkspace store  │
│   · run/session persistence · sandbox-backed exec/file tools · a tool          │
│   transport · an IHookManager adapter · a constants registry that declares     │
│   the host's own groups and serves the loop snapshot                           │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Data flow of one agent turn

`QueryEngine.run(message)` appends the user message, stamps the run clock,
persists a turn-start snapshot and drives one turn; `resume(engine, snapshot)`
restores a stored run first and then drives whichever continuation the caller
described. Both bind the driving task so `stop()` can hard-cancel, and both
persist a closing snapshot however they exit. Each inner `yield` from the
private `_query_raw` generator is a stop-check checkpoint; the executor streams
the emitted `TurnEvent`s out over SSE (Redis pub/sub at the host layer).

```
                       ┌─────────────────────────────────────────────┐
 caller: async for evt │  run(message) / resume(engine, snapshot)     │
   in engine.run(msg): │  — an async iterator of TurnEvent, one turn  │
                       └─────────────────────────────────────────────┘
                                          │
   (1) STOP CHECK ──────────────────────►│  stop_requested? → synthesize missing
                                          │   tool_results → CANCELLED
        INTENT RECOVERY ─────────────────►│  settle interrupted tool intents +
                                          │   mark_intent_recovery (correctness_bind)
        LIFECYCLE run_start ────────────►│  fire_lifecycle → deny? → stop 
        MANUAL /compact ────────────────►│  CompactCheckpoint (RC-gated, default off)
   (2) COMPACTION CHECK ─────────────────►│  needs_compaction()?  ── yes ──┐
                                          │                                 ▼
                                          │                 ┌──────────────────────────┐
                                          │                 │ _run_compaction          │
                                          │                 │  Tier 1: truncate/blob   │
                                          │                 │   big tool_results       │
                                          │                 │  Tier 2: summarise old   │
                                          │                 │   turns (never the       │
                                          │                 │   operator's) → snapshot │
                                          │                 │  Tier 3: fold runs of    │
                                          │                 │   old summaries +        │
                                          │                 │   old operator turns     │
                                          │                 │  COMPACTING→RUNNING      │
                                          │                 └──────────────────────────┘
   (3) UserPromptSubmit HOOK ────────────►│  _safe_hook_invoke → deny? → FAILED
                                          │
   (4) BUILD CONTEXT ────────────────────►│  tools = registry.compute_effective_surface
                                          │     (policy → clip → BM25 retrieval)
                                          │  skill catalog (alpha Skill() lines) ◄── SKILLS
                                          │  context_manager.build_context(history,…)
                                          │     ◄── MEMORY auto-recall injected (the host)
                                          │     ◄── context_bootstrap env-docs (turn 1)
   (4b) LOOP STRATEGY ───────────────────►│  select_strategy(run_mode)
                                          │     DirectStrategy: no pre-action step
                                          │     DeepStrategy: forced Plan tool +
                                          │       one REASONING_STEP, then shared loop
                                          ▼
   (5-9) STREAM ONE ASSISTANT MESSAGE  _stream_one_assistant_message(engine, context)
   ┌──────────────────────────────────────────────────────────────────────────────────┐
   │  budget max_tokens  ← AdaptiveSafetyBand (drift margin) ◄── RESILIENCE             │
   │  full_messages = system sections + history                                         │
   │  full_messages = _repair_outbound_tool_pairing(...)  (UNCONDITIONAL)               │
   │  cache_breakpoints = apply_system_and_3(full_messages)  ◄── PROMPT CACHE           │
   │  request = LLMRequest(messages, tools, max_tokens, extra={cache_breakpoints})      │
   │  async for delta in _iter_with_idle_watchdog(engine.llm.stream_with_tools(req)):   │
   │      delta → TurnEvent   (ProviderDelta → content_block_* / tool_use_* / usage)    │
   │                          (delta_bridge.py translates provider deltas) ◄── EVENTS   │
   │      usage → record cache_read/creation + cache observer                           │
   └──────────────────────────────────────────────────────────────────────────────────┘
                                          │
              ┌───────────────────────────┴────────────────────────────┐
              ▼                                                          ▼
   pending_tool_calls?  ── yes ──┐                            no tool_calls (end_turn)
                                 ▼                                       │
   ┌───────────────────────────────────────────────┐    stop_requested re-check
   │ for each call: _dispatch_tool(engine, call)    │            │
   │   ┌─────────────────────────────────────────┐ │            ▼
   │   │ ToolDispatcher.dispatch:                 │ │   terminal-tool nudge / backstop?
   │   │ 1. registry lookup → unknown_tool?       │ │   guaranteed-terminal submit?
   │   │ 2. schema / JSON validation              │ │            │
   │   │ 3. ToolPermissionGate.check (4 stages):  │ │            ▼
   │   │    whitelist→policy→rate→hook (pre_tool) │ │   FINALIZATION GATE + GROUNDING:
   │   │ 4. preconditions DAG→masked? (post-gate) │ │    verify deliverables (stat) ·
   │   │ 5. execute tool.invoke(ctx)              │ │    terminal_answer_validation
   │   │ 6. post_tool_use hook                    │ │      (refs ⊆ reads, canonical)
   │   │ → DispatchOutcome (success/err/approval) │ │
   │   │     ◄── read records grounding ref       │ │            │
   │   │     ◄── WORKSPACE read-dedup cache       │ │            ▼
   │   └─────────────────────────────────────────┘ │      MESSAGE_STOP → COMPLETED
   │  append tool_result Message to history         │ │
   │  snapshot after every tool_result append       │ │
   │  loop back to (5): next assistant message       │ │
   └────────────────────┬───────────────────────────┘ │
                        │  approval_required? → PendingInterrupt(approval) → AWAITING
                        ▼  ask_user? → PendingInterrupt(question) → AWAITING
                           (resume(resolutions={id: InterruptResolution(...)}))
                  (recurse) stream next assistant message
```

Where the subsystems hook in:

- **Memory** injects auto-recalled facts before the LLM call (step 4) and is
  read/written by the `read`/`write`/`recall`-style tools during dispatch.
- **Workspace** backs `read`/`write`/`find`/`search` during dispatch; the
  process-local **read-dedup cache** short-circuits a repeated `read` of the
  same path/content.
- **Grounding** records a citation ref whenever a grounding-tracked `read`
  fires; the **finalization gate / terminal-answer validation** consume that
  ref ledger when the terminal `answer` is produced.
- **Resilience** wraps the outbound LLM call budget (AdaptiveSafetyBand) and is
  available as the universal `IToolTransport` wrapper for tool/VM calls
  (the host binds it).
- **The lifecycle seam and events** fire at every coordinate of a run (the
  provider exchange, pre/post tool, the compaction transaction, run start and
  finalization) and every provider delta becomes a `TurnEvent`.
  When `typed_hooks_enabled` is on, `correctness_bind.fire_lifecycle`
  dispatches every coordinate the loop passes through, and a `transform` at
  `context_transform` is applied to the turn's context. One flag governs the
  whole seam; no coordinate is hidden behind a second, unrelated one.
- **Intent settlement + usage ledger**: **every** dispatched tool commits an
  `IntentRecord` before the call, unconditionally, and a turn opens by closing
  out any record a stopped run left in flight. `intent_settlement_enabled`
  (default-off) gates only the recovery events and the ledger row on top of
  that. Usage rows for
  `inference` / `retry` / `compaction` / `abort` / `fail` go through
  `commit_usage` when `usage_ledger_enabled` is on; a **tool**-kind row is
  written only on the intent-settlement dispatch path.
- **Live control** holds steer / follow-up queues and live model/thinking
  overrides; **CompactCheckpoint** is the operator `/compact` path (not a
  third compaction tier).

---

## Technology inventory

One row per core technology. **Wired into loop?** = referenced by the core
runtime loop (`query.py` / `query_engine.py`); subsystems wired only by 
the host adapter are marked accordingly. **RC toggle(s) + default** records the
governing `LoopConstants` field(s) and their safe/off default.

| Technology | Core files | RC toggle(s) + default | Wired into loop? | Tested? |
|---|---|---|---|---|
| ReAct loop / orchestrator / query engine | `runtime/query.py`, `runtime/query_engine.py`, `runtime/loop_state.py`, `runtime/loop_strategies.py` | n/a (always on); recovery branches RC-gated | Yes | Yes |
| Tool dispatch + gating | `runtime/tool_dispatch.py`, `runtime/tool_permission.py` | gate always on; consecutive-error cap RC | Yes | Yes |
| Tool retrieval / registry | `runtime/tool_registry.py`, `runtime/tool_retrieval.py` | `tool_retrieval_top_k` (clip threshold) | Yes | Yes |
| Tool preconditions | `runtime/tool_preconditions.py`, `runtime/run_tool_preconditions.py` | `tool_preconditions_enabled` = `False`; run-level `QueryEngineConfig.tool_preconditions` empty | DAG + run-level forcer: Yes | Yes |
| Turn policies | `contracts/turn_policy.py`, `runtime/turn_policies/*` | each policy reads its own RC fields; the ORDER is core-owned (`TURN_POLICY_ORDER`) | Yes — the driver consults the registry at 14 coordinates | Yes |
| Tool roles + argument spellings | `contracts/tool_roles.py`, `runtime/child_capabilities.py` | none — the map is `QueryEngineConfig.tool_roles`, declared by the host at registration | Yes | Yes |
| Constants registry | `contracts/config.py` | the system itself (`ConstantSpec` / `ConstantGroup` / `IConstantsRegistry`) | Declaration + resolution: host-side; the loop reads the snapshot | Yes |
| Snapshot schema + upcasters | `contracts/snapshot.py` | none — a schema is not a knob | Yes (every `snapshot()` / `resume_from_snapshot()`) | Yes |
| Interrupts (approval / question / external call) | `contracts/interrupt.py`, `runtime/query.py::resume_interrupts` | none — a parked call is not opt-in | Yes | Yes |
| Session work pool (background commands + child runs) | `contracts/background.py` | none in core; the pool is injected | Yes (`ensure_session_attached`, `drain_wakes`) | Yes |
| Request manifest | `contracts/observability.py`, `runtime/query.py::build_llm_request` | none — a run records when `QueryEngineConfig.request_manifest_sink` is bound | Yes when a sink is bound | Yes |
| Conformance suites | `protocore/conformance/*` | n/a (a test-time package, `protocore[testing]`) | No — the host runs them against its own adapters | Yes |
| Universal resilience layer | `contracts/resilience.py`, `runtime/resilience.py` | `resilience_enabled` = `False`; the transport attempt count is a host knob | Ledger/band: Yes; transport wrapper: host-only | Yes |
| Failure classification | `contracts/resilience.py::IResilienceClassifier`, `runtime/error_kinds.py` | none — the classifier is `QueryEngineConfig.resilience_classifier`; unbound means no wording is recognised | Yes | Yes |
| Run wind-down (soft stop) | `runtime/soft_stop.py` | `soft_stop_enabled` = `True`, `soft_stop_max_turns` = `3` | Yes | Yes |
| Attempt ledger + adaptive safety band | `contracts/attempt_ledger.py`, host-owned band | band wired via per-call output budget | Yes | Yes |
| Finalization gate + contract | host-owned | `terminal_tool_nudge_enabled` (`False`), `terminal_tool_forced_max_attempts`, `terminal_tool_forced_thinking_enabled`, `terminal_tool_nudge_write_first_before_forcing`, `finalize_prose_gate_enabled` | Yes | Yes |
| Terminal-answer validation + references/grounding | host-owned (the core carries the evidence a run collects, `contracts/evidence.py`) | host knobs (validation and reference normalisation are both driven from the host's own model) | Yes | Yes |
| IMemory subsystem | `contracts/memory.py`, `tools/memory.py` | `memory_enabled` = `False`; auto-recall is a host knob | Host-wired (tools held by core contract) | Yes |
| Token counting | `runtime/token_counting.py` (+ the optional `protocore-native` estimator) | `chars_per_token_*` ratios in RC; `PROTOCORE_DISABLE_NATIVE` forces the pure-Python path | Yes | Yes |
| IWorkspace + read-dedup cache | `contracts/workspace.py`, host-owned cache | n/a (no snapshot toggle; the host owns the surface) | **No** (host-wired) | Yes |
| Context management / three-tier compaction / session memory | `runtime/context/manager.py`, `runtime/context/compaction.py`, `runtime/context/budgets.py`, `runtime/context/session_memory.py`, `runtime/compact_checkpoint.py` | ratios in RC; `compaction_manual_enabled` = `False` | Compaction + `/compact`: Yes; session-memory fold: host-wired | Yes |
| Prompt caching | `runtime/prompt_caching.py` | wire translation gated by a host kill-switch | Yes (hints in core; wire translation the host) | Yes |
| Skills routing / surfacing | `runtime/skill_index.py`, `contracts/skills.py` | data-driven (empty store = no block); `skills_hot_reload_enabled` = `False` | Yes (`_ensure_run_skill_catalog`); `list_files`/`load_file` host-only | Yes |
| The lifecycle seam + injection / context_bootstrap | `contracts/middleware.py`, `hooks/manager.py`, `runtime/correctness_bind.py` | `typed_hooks_enabled` = `False`; judge failure mode and context bootstrap are host knobs | One seam, five registration kinds, one coordinate list (`HookEvent`); `decide`/`transform`/`around` fail closed, `observe`/`notify` are isolated | Yes |
| Events / observability / streaming | `events.py`, `runtime/events/*`, `runtime/llm/delta_bridge.py`, `runtime/telemetry.py` | `telemetry_spans_enabled` = `False` | Yes | Yes |
| Intent / usage ledger / session tree / lanes | `runtime/intent.py`, `runtime/usage_ledger.py`, host-owned session tree, `runtime/lanes.py` | `intent_settlement_enabled`, `usage_ledger_enabled`, `lanes_enabled` (all `False`); the session tree is a host knob | Intent + ledger: Yes when on; tree/lanes: host-invoked | Yes |
| Live control + run work budget | `runtime/live_control.py`, `runtime/run_work_budget.py` | `steer_follow_up_enabled` = `False`; tree token/run caps | Yes | Yes |
| Safety (shell policy + chain parser) | `safety/shell.py`, `runtime/chain_parser.py` | policy stack via `register_policy` | Yes | Yes |
| LoopConstants system | `contracts/runtime_constants.py`, `runtime/runtime_constants.py`, `constants.py` | the system itself | Yes | Yes |

A cross-cutting fact: **most new capabilities are default-off** and have no
exercise on the default tenant, so their *enabled* paths are covered by unit
tests rather than live runs. That is intentional — they are opt-in product
capabilities.

---

## Per-technology sections

The tour below is the deep reference for the inventory rows that need more
than a row. Rows it does not expand — retrieval, the precondition DAG, the
resilience wrapper, the attempt ledger, memory, workspace, three-tier
compaction, token counting and prompt caching — behave as the files the table
names, and repeating them here is how a second description comes to disagree
with the first.

### ReAct loop / orchestrator / query engine / loop state

**What & why.** This is the heart of the runtime: a ReAct (reason→act→observe)
loop that runs **one assistant turn at a time** and yields streaming events. It
is split into state and behaviour so any pod can resume a run after another
crashes. The shared assistant loop is **not** a single immutable path:
`QueryEngineConfig.run_mode` selects `DirectStrategy` or `DeepStrategy` in
`runtime/loop_strategies.py` before that shared loop.

**Key classes/files.**

- `runtime/query_engine.py`
  - `QueryEngine` — one instance per active run. Owns the **mutable
    per-conversation state**: `history` (list of `Message`), the `LoopState`
    machine, `CompactionState`, `TokenUsage`, plus `open_intents`, `usage_rows`,
    `lanes`, live-control queues (`_steer_queue` / `_follow_up_queue`),
    live model/thinking overrides (`_live_model_name` /
    `_live_thinking_enabled` / `_live_reasoning_effort`), optional
    `verification` (`VerificationLifecycle`), and the recovery latches
    (terminal-only, guaranteed-terminal, self-verify, circuit-breaker,
    pending-reads, longfile, tool-precondition index, …). Persistence is
    `snapshot()` ↔ `resume_from_snapshot()`; `run()` snapshots at turn start
    and in `finally`. The snapshot also writes `open_intents`, `usage_rows`,
    `lanes`, the live `live_*` fields, the steer/follow-up queues,
    `verification` (when non-default), and those recovery latches.
  - `QueryEngineConfig` — the **immutable injection surface** bound at engine
    construction: `run_id`/`tenant_id`/`session_id`/`model_name`,
    `account_id` (account-wide skill bank key; empty if unresolved),
    `system_prompt_sections`, `tool_visibility_policy`, the `rc` snapshot,
    `run_mode` (`"direct"` | `"deep"`, default `"direct"`),
    `execution_profile`, `thinking_enabled` / `reasoning_effort`,
    `expected_terminal_tool`, `tool_preconditions` (the run-level forcer;
    empty default), the optional `cache_observer`, optional
    `verification_delivery`, and the two **host-supplied trigger
    callables** (`pre_terminal_self_verify_trigger`,
    `pre_dispatch_terminal_verify_trigger`) — both default `None`, so the
    pre-dispatch veto / self-verify machinery is dead unless the host injects
    a callable *and* flips the matching RC.
  - `QueryEngine.__init__` accepts an optional `provider_chain: IProviderChain`
    for mid-stream provider failover. `None` (every caller that configured no
    priority list) leaves existing recovery untouched.
  - `QueryEngine.run(initial_message)` is the **async-generator** driver: it
    appends the user message (or continues against an existing user-final
    history), increments `turn_count`, resets per-turn state, stamps the run
    clock, persists a turn-start snapshot, binds `_current_turn_task`, then
    iterates `_query_raw`. A turn-end snapshot lands in `finally`. Both the
    handle binding and that closing snapshot come from `driving_turn()`, the
    scope every public drive runs inside, so the two obligations that make a
    drive interruptible and resumable are owned in one place.
- `contracts/interrupt.py` — what a paused run is waiting for, as a value
  rather than a latch. `PendingInterrupt(interrupt_id, kind, tool_call_id,
  tool_name, payload, created_at_ms, expires_at_ms)` carries one wait;
  `InterruptKind` says which of three it is (`approval` — a call parked at a
  gate that has NOT run; `question` — a call that ran far enough to ask and
  whose answer is its result; `external_call` — a result arriving by another
  route), and the kind decides which decisions are legal. The engine holds
  however many are open at once, in park order, and writes them to the
  snapshot; `LoopState.AWAITING` with none of them recorded is refused at the
  transition (`loop_state.assert_awaiting_is_witnessed`), because it is a run
  that stops with nothing that could resume it. `InterruptResolution` is one
  decision — `approve` (optionally with `updated_input`, the corrected
  arguments the call is then really run and recorded with), `deny`, `answer`,
  `abandon` — and `plan_resolution` refuses a map that names a wait that is not
  open, answers a kind that does not take that decision, or leaves an open
  interrupt undecided without saying so.
- **Idempotency of the code before an interrupt.** A parked call resumes
  exactly where it stopped: nothing before the interrupt is re-executed, so
  no work between the turn's start and the park is repeated. What a resumed
  run must not do is re-issue the parked call itself, and that is what the
  durable intent record (`runtime/intent.py`) prevents — a record in
  `PENDING_APPROVAL` is a call that never ran, one in `PAUSED_ASK_USER` is a
  call whose answer is still owed, and `intent_never_replay_tools` names the
  tools whose repeat is unacceptable whatever the record says. The two are one
  guarantee read from two sides: the interrupt says what is being waited for,
  the intent says what may be done about it.
- `runtime/query.py` — `resume(engine, snapshot, *, approved_tool_call=None,
  message=None, abandon_approval=False, resolutions=None,
  allow_partial_resolution=False)` is the public resume entry: it
  restores the snapshot strictly (schema, delivery mode and identity binding
  settled before the first mutation), then drives the resolution map, the
  approved call, the arrived message, or a plain re-drive of the interrupted
  turn — whichever the caller's arguments describe. `resolutions` is the
  general form and the only one that can answer a batch: several calls parked
  together are answered in ONE drive, each with its own decision, and their
  results land in the order the model asked for them. A run with anything
  parked refuses a plain re-drive; `abandon_approval=True` closes every parked
  call as never answered.
  `resume_interrupts` is that drive on its own, for a caller that has already
  restored the run. `resume_approved_tool` executes one call held for
  approval, verified against the durable pending call and idempotent on
  replay; both run inside `driving_turn()` too. `_query_raw` implements the
  turn lifecycle: stop check → resume interrupted intents → the
  `run_start` coordinate → optional `/compact` via `CompactCheckpoint` → compaction
  check → UserPromptSubmit hook → build context → `select_strategy(run_mode).prepare_turn`
  → `_stream_one_assistant_message` (recursive on tool_use) → dispatch →
  finalize. Recovery is broader than the 413 / max-output / thinking-trap /
  empty-nudge / idle-watchdog set: the turn also resumes interrupted intents,
  fires the `run_start` coordinate, handles `/compact` via `CompactCheckpoint`, and
  binds usage/hooks through `runtime/correctness_bind.py`
  (`commit_usage`, `fire_lifecycle`, `mark_intent_recovery`,
  `persist_correctness`). Those older recovery branches remain model-agnostic
  and RC-gated.

  Context-window recovery preserves at most one compaction per assistant
  message. An exact provider prompt count can justify one smaller-cap retry
  before compaction. A missing or lower-bound count compacts first; subsequent
  rejections repeatedly reduce the rejected wire cap by
  `context_overflow_retry_output_ratio`, always strictly, and stop after
  `context_overflow_retry_max_attempts` or when no smaller positive cap exists.
- `runtime/loop_strategies.py` — `select_strategy(run_mode)` is the single
  branch point. `DirectStrategy` contributes no pre-action step (the
  auto-tool loop). `DeepStrategy` runs a forced planning tool
  (`extra["forced_tool_choice"]` on the request, CoT bounded by
  `reasoning_effort`), emits exactly one
  `REASONING_STEP` event, then the shared assistant loop drives the real
  action with the full surface.
- `runtime/query.py::build_llm_request` — the one assembler every provider
  call passes through: the action stream, the deep loop's plan call, its
  prompted-JSON fallback and both compaction summarisers. It fixes the
  three things those four used to settle separately — the model in force (the
  live override when one is set), the forced tool (one slot,
  `extra["forced_tool_choice"]`, carrying the tool NAME for an adapter to
  render onto its own wire; or, instead, `extra["tool_choice_required"] =
  True`, meaning "some tool, no prose", rendered as `tool_choice="required"`;
  an adapter without support ignores either — see `LLMRequest.extra`) and the
  temperature (the caller's value, or
  `None` so the host decides — a per-model setting or the server's own
  generation config; the summarisers state theirs).
- `runtime/context/budgets.py` — `derive_budgets` turns one RC snapshot into
  every per-layer token budget, deterministically, with no cache. The
  compaction trigger it returns is the LOWER of two bounds: the configured
  `compaction_trigger_ratio` of the window, and the largest prompt the
  provider would still accept — the window less the output reserve, less
  `request_context_safety_tokens`, less `compaction_trigger_turn_headroom_ratio`
  of the window for the turn that is about to be added. A serving stack that
  counts the requested output against the same window as the prompt rejects
  anything above `window - max output`, so a trigger derived from the ratio
  alone can sit above the cliff and never fire: on a 65 536-token window with
  the stock 0.25 output reserve, 0.8 of the window is 3 276 tokens past the
  point the request stops being accepted. The output-reserve term is gated on
  `provider_reserves_output_in_context_window`, true by default — an endpoint
  that sizes its input window independently of the requested output sets it
  false and gets that share of the window back. Consumers read the effective
  value; the emergency cliff is held strictly above it. Both are whole-prompt
  sizes, so the gate adds to the history's estimate what the last request
  carried besides it — its system messages and tool definitions, in the same
  calibrated tokens; before a run's first request that part is zero.
- `runtime/request_budget.py` — fits every assembled request to the hard
  window by clipping its output cap. The size it fits to is the calibrated
  estimate, except near the edge: once the estimate reaches
  `exact_token_count_margin_ratio` of the prompt size at which the cap starts
  being clipped, a provider that implements the optional
  `IRequestTokenCounter` capability is asked what the request renders to, and
  that count is used instead. The same question is asked of the durable history
  when it is within the margin of the compaction trigger, so the gate decides
  on the provider's tokens — until a fit has counted the turn's full request,
  after which the gate reads the factor that count set instead of paying a
  second round-trip. After the first count the fit asks again only when the
  last count plus the content that count did not see — every message whose
  digest is not among the counted ones, so a message rewritten by compaction
  or eviction is new, and removals are ignored — sized at the worst undercount
  the margin assumes (`1 / (1 - margin)`), could cross the limit, so prose-heavy
  turns count about once near the edge while a large dense tool result is
  counted at once. A count that fails or times out stops counting for
  `exact_token_count_failure_backoff_seconds`. The default margin, 0.75, is `1 - 1/4`: an undercount by
  a factor `f` is only caught when the margin is at least `1 - 1/f`, the
  heuristic was measured running 1.76x short on JSON and 3.53x on hexadecimal
  text, and 4 is the largest factor calibration can express. An exact count
  sets `token_estimate_calibration` outright, and does so before the fit is
  attempted, so a count that proves the request cannot fit raises the factor
  first and the refusal goes to compaction sized in the counted tokens; a usage
  report after the call moves it half-way; a rejection for length raises it to
  the floor the rejection proves (the prompt was at least the window less the
  output cap the loop sent). Counts are kept per request content
  (`exact_token_count_cache_max_entries`); a count that fails or does not
  arrive within `exact_token_count_timeout_seconds` is logged and the estimate
  is used; a provider without the capability sends exactly the requests it
  sent before. Each fresh count logs the estimate beside the measurement,
  which is the drift between the two. The learned factor lives in the run's
  snapshot and survives a resume on the same model; a new run starts from the
  configured `token_estimate_calibration`, so a host that knows its content
  runs dense seeds that value per scope.
- `runtime/context/compaction.py` — three passes over the transcript, in
  order, each one taking what the pass before it could not.
  **Tier 1** replaces an over-budget tool result with a placeholder and puts
  the bytes in the blob store; the content is recoverable and the preview says
  what was shed. It may also shed aged reasoning or an over-budget frozen
  reference while preserving message metadata, including seed provenance.
  Routine **Tier 2** and **Tier 3** leave seeded prior-run turns untouched.
  **Tier 2** summarises old turns through the compaction LLM —
  one atomic unit at a time, an assistant `tool_use` turn and the results
  answering it standing or falling together so no pair is orphaned. It never
  summarises a turn the operator wrote: an instruction is short, so
  paraphrasing it frees almost nothing, and it is specific, so the paraphrase
  is a rewrite — "remove the model-name field from the header" becomes "the
  user asked for changes" and the run acts on that instead. **Tier 3** folds
  what Tier 2 leaves: over a long session, one summary per tool batch plus
  every operator message become the whole window, and neither pass below can
  take a byte off them. Each contiguous run of such messages that is at least
  `compaction_fold_min_messages` long and `compaction_fold_min_tokens` big
  becomes one consolidated summary in which the operator's instructions
  survive as exact quotes; the task turn and the
  `compaction_fold_keep_operator_turns` most recent instructions stay
  verbatim. Reactive provider-overflow recovery may summarise and fold seed-only spans,
  keeping only the `compaction_force_keep_recent_turns` trailing messages (one by
  default); every replacement retains the seed tag and
  spans split at seed/current boundaries, so the host's persistence filter
  keeps the runs separate. Frozen compaction references remain protected. A
  fold is a summary like any other, so a later fold absorbs it once its
  neighbourhood has grown again.

  Both summarisers assemble their request through `build_llm_request`, record
  it to the request manifest, and are told the same two rules the wording of a
  summary lives or dies by: keep every identifier — paths, ids, ports, URLs,
  numbers, error codes — verbatim rather than substituting a plausible value,
  and state an outcome with no tool result or confirmation behind it as
  UNKNOWN rather than as done or not done. The per-turn prompt states its
  budget in characters as well as words, says a longer reply is cut off and
  discarded, asks for the count and the records that matter instead of a copy
  of a long tool result, and names the single key it wants — a model that
  listed every record of a long result wrote a reply the output cap cut, and a
  cut reply is never parsed. Both instructions are templates
  (`compaction_turn_summary`, `compaction_fold_summary`), not literals, so an
  operator serving another language has somewhere to put the translation.

  What a pass is allowed to cost is bounded on every axis: a unit below
  `compaction_summary_min_unit_tokens` is not sent at all (a summariser writes
  a sentence or three whatever it is handed, so below some size the call is
  spent to discover the summary is no smaller — adjacent units that are each
  below it are joined, up to `compaction_summary_group_max_tokens`, and
  summarised as one, so a history made only of short rounds can still shrink),
  the word budget in the prompt
  scales with the unit rather than being a fixed sentence count and is capped
  at what the output cap can hold at
  `compaction_summary_output_tokens_per_word` (four — the English figure of two
  understates JSON escaping and a non-Latin script, and a budget sized that way
  comes back cut off, never parses and is never committed) less
  `compaction_summary_envelope_tokens` for the JSON around the words, and never
  above what the grammar's own `maxLength` will accept, calls go out
  `compaction_summariser_parallelism` at a time instead of one after another
  while the run sits in `COMPACTING`, and the fold takes at most
  `compaction_fold_max_spans_per_pass` runs per pass. A summary that comes
  back no smaller than what it would replace is discarded, never committed.

  A call that fails for a reason belonging to the unit — the request does not
  fit the summariser's own window, or the reply carried no readable summary
  because the output cap cut the envelope — is counted against THAT unit in
  `CompactionState.failed_anchor_keys`, and past
  `compaction_summary_failed_unit_max_attempts` the routine gate stops sending
  it; the fold tier still gets its turn at it. Nothing else is counted: a
  transport failure (a rate limit, a 5xx, a recycled summariser) says nothing
  about the unit, and neither does a summary that merely came back no smaller,
  so neither retires anything. The other units in the batch commit regardless,
  so one unit the summariser cannot handle no longer keeps a pass from shedding
  anything. The forced passes ignore the census and try every unit — they run
  when the alternative is the run ending. The census rides the run snapshot,
  and entries whose unit has left the transcript are pruned at the end of every
  pass, so it does not grow without bound.

  Separately from the census, a pass is charged to a retry budget bounded by
  `compaction_failed_max_retries`. Only a pass that tried and failed is charged
  — a tier raised, or a summariser call was made and nothing came of it — and
  it is charged once, however many tiers failed. Routine and proactive passes
  share `CompactionState.retry_count`; the reactive pass after a provider
  rejection keeps `CompactionState.reactive_retry_count`, because it is the
  only profile that may compact seeded history and proactive failures prove
  nothing about it. Progress by either profile clears both counters, since the
  next pass of either kind faces a different history, and `rearm()` clears
  them too. A transport failure therefore costs the pass one retry and the
  unit nothing, while a unit-shaped failure is counted against the unit as
  well; the forced passes ignore the census either way. Both counters ride the
  run snapshot.

  A proactive pass (routine, turn-start emergency, per-iteration) is decided
  before it opens. `ContextManager.has_proactive_work` asks each tier whether
  it would change anything under the proactive profile, without changing it;
  when none would — a history of seeded turns is the usual case — the gate
  opens no transaction at all: no `COMPACTING`, no events, hooks, usage row or
  snapshot. The engine remembers that probe with the history and the constants
  it saw, and does not ask again until either changes. Past the budget, a
  proactive pass does not end the run: nothing has been rejected yet, so the
  proactive summariser tiers are suspended (a
  `compaction_exhausted_proactive_suspended` state change each time a
  proactive pass exhausts the budget; `retry_count` is not reset, so after the
  suspension one failed pass suspends again) and the request goes out. Tier 1 needs no LLM and keeps running through the suspension. The
  suspension ends after `compaction_proactive_suspension_iterations` gate
  visits, or once the prompt has grown by
  `compaction_proactive_suspension_growth_ratio` of its size when it began,
  whichever comes first. A context refusal lifts it at once — the provider's,
  or the local fit's refusal by estimate or exact count, both of which run the
  reactive pass — and so do `rearm()` and a resume from a snapshot, which does
  not carry it. A reactive pass past its budget hands the turn to the
  output-cap ladder while a smaller cap is left, and fails the run only when
  none is.

  The routine per-iteration gate also stands down for
  `compaction_no_gain_backoff_iterations` iterations after a pass that freed
  less than `compaction_min_gain_ratio`; the count survives the per-message
  recovery reset, which used to clear it before it could skip anything. The
  backoff ends early once the prompt has grown by
  `compaction_no_gain_backoff_growth_ratio` of its size when it was set, and on
  any context refusal.

- `runtime/stale_result_trim.py` — the prompt-shrinking pass that costs no LLM
  call. RC-gated by `tool_result_stale_trim_enabled` (**off by default**), it
  rewrites the REQUEST view only — `engine.history` keeps every byte, so
  persistence, replay and compaction see an untouched transcript. A tool result
  longer than `tool_result_stale_max_chars` is cut to that head once the run has
  moved past it; the newest `tool_result_fresh_count` results and every result
  of the latest round of tool calls are never touched, whatever their size,
  because that is what the model is about to read. Nothing is cut until the
  trimmable excess crosses `tool_result_stale_trim_batch_chars`, so the prompt
  prefix moves for a batch and not for one result, and a trimmed id is sticky
  for the run (`trimmed_tool_result_ids` in the snapshot), so a resumed run
  rebuilds the same prefix. Pins are honoured unless a later write has falsified
  them, and a compacted placeholder is never rewritten. Every line of the cut
  part starting with one of `tool_result_stale_trim_protected_prefixes` (by
  default the `Cite exactly:` / `Cite:` / `cite_as:` / `Source:` family) is
  carried over verbatim, so a result keeps its citation identity when it loses
  its body; the placeholder that replaces the rest names how many characters
  were kept and how many went, so a trimmed result cannot be read as complete
  evidence. It runs after the compaction checkpoint and before the split
  projection, and sizes its head so the split cannot cut its own pointer.
- `runtime/loop_state.py` — `LoopState` is a pure 7-state machine:
  `PENDING → RUNNING → {AWAITING | COMPACTING} → {COMPLETED | FAILED |
  CANCELLED}`. `assert_transition()` enforces the legal-edge table;
  `TERMINAL_STATES` have no outgoing edges. **Distinct** from
  `RunStatus` (the durable PG-row mirror) and `RunState` (the hot Redis-hash
  record) — `LoopState` is the engine instance's in-flight state.

**How invoked/wired.** The host executor constructs a `QueryEngine` on
run admission, then typically `async for evt in engine.run(message)` per turn
(or `async for evt in resume(engine, snapshot)` when it is picking a run back
up). Each `TurnEvent` is forwarded to the SSE bridge. The loop is the
single consumer of every other subsystem.

**RC configurability.** `max_turns_per_run`, `agent_max_seconds` (wall-clock
deadline; `<= 0` = inert), the idle/stall watchdog timeouts, and every recovery
toggle are RC fields. `model_name` is required (no baked-in default). The run's
mode is not a snapshot field: `QueryEngineConfig.run_mode` carries it per run
and defaults to `"direct"`, so a host that wants a tenant-wide default declares
that knob in its own constant group and passes the resolved value in.

**Extension protocol.** Do **not** edit the loop structure. Customise via (a)
lifecycle registrations, (b) a turn policy substituted into
`QueryEngine.turn_policies` by name, (c) `QueryEngineConfig` injected
callables/observers — `run_mode`, `tool_preconditions`, `provider_chain`,
`tool_roles`, `resilience_classifier`, `request_manifest_sink`, (d) RC toggles,
(e) `system_prompt_sections`.

**Terminal-classification notes.** Three terminal-classification behaviours are
worth calling out: (1) the loop re-checks `stop_requested` after streaming and
routes a cancelled run to CANCELLED (not a clean end-turn); (2)
`_synthesize_missing_tool_results` is called at every teardown checkpoint so a
persisted snapshot is always pairing-valid (see
`_repair_outbound_tool_pairing` / `_synthesize_missing_tool_results` in the
inventory table); (3) the `max_turns` exit
is classified as a resource-**exhaustion** terminal — it keeps
`stop_reason=max_turns` on the wire and is treated as an error/non-success class,
not a clean `COMPLETED`.

### Turn policies — where a product decision about a turn lives

**What & why.** The driver of one assistant turn does two jobs. One is
**mechanics**: open a stream, translate deltas into events, dispatch the calls
the model asked for, close the round. The other is **policy**: decide that this
run has spent its budget, that an empty answer earns one more try, that a file
left half-written must be sealed before the run may finish. Mechanics is the
same for every run; policy is a product opinion, and every opinion ever added
to the loop was added by growing a branch inside it. The turn-policy seam is
what stops that: a policy is an object, it declares the coordinates it wants to
be consulted at, and it answers with events to forward plus one directive.

**Key classes/files.**

- `contracts/turn_policy.py` — `ITurnPolicy` (a `name`, the `coordinates` it
  registers at, and one `apply(turn)` that yields the events the loop forwards
  and writes what happens next to `turn.outcome`), `TurnContext`, `ITurnState` (a deliberately small
  structural view of the run a policy may read and change — a policy that needs
  something not named there is reaching into the loop's insides, and the review
  that adds the name is where that gets noticed), `TurnFlags` (the turn-local
  state policies share with the loop, which used to be bare locals of one very
  long function), `TurnCoordinate` (`turn_start`, `turn_budget`,
  `empty_model_turn`, `output_truncated`, `stream_failed`, `turn_end`,
  `stream_settled`, `tool_calls_ready`, `finish_nudge`, `answer_floor`,
  `voluntary_finish`, `terminal_tool_finish`, `iteration_end`,
  `cancel_checkpoint`), `TurnDirective` (`proceed` / `restart_turn` /
  `end_turn`) and `TurnPolicyOutcome`.
- `runtime/turn_policies/` — one module per decision: `longfile.py`,
  `run_ceilings.py`, `empty_model_turn.py`, `truncated_tool_call.py`,
  `output_cap.py`, `terminal_nudge.py`, `answer_floor.py`,
  `empty_completion.py`, `terminal_tool_finish.py`, `compaction.py`,
  `repeat_guard.py`, `sibling_walk.py`, `provider_failure.py`,
  `cancellation.py`.
- `runtime/turn_policies/__init__.py` — `TurnPolicyRegistry` and
  `TURN_POLICY_ORDER`.

**The order is the core's.** `TURN_POLICY_ORDER` declares it once, and a name
absent from that tuple is refused at construction (`UnknownTurnPolicyError`)
rather than silently running last. The order matters where two policies meet —
an unsealed file is sealed *before* the guard that asks whether the turn
produced an answer, because sealing produces one — and an order taken from
whichever list a host happened to build would make that a coincidence. A
registry consults, in order, every policy registered at the coordinate and
stops at the first that answers anything but `proceed`; a policy that has taken
the turn elsewhere is never followed by one assuming it did not. A seam that
cannot obey a directive says so (`UnsupportedTurnDirectiveError`) instead of
dropping it.

**How wired.** `QueryEngine.turn_policies` is `None` for the core's own set.
A host or a test that installs its own set assigns one there, per run, and it
is **merged by name** rather than put in place: a policy replaces the core
policy answering to the same name, and every bound nobody named stays where it
is. `runtime/error_kinds.py::INTERNAL_ERROR_KIND` is read from both sides of
this seam, which is why it is a module of its own — "the loop crashed" must not
have a second spelling on the policy's side.

**Provider failure: what the run says when the endpoint does not answer.**
`provider_failure.py` ranks three recoveries — a sibling on the run's provider
chain, the same endpoint after a bounded backoff, and the answer the run
already has — and two rules keep the last of them honest.

- **Retry is the adapter's verdict, read off the exception.** Every
  `LLMError` carries `retryable`. The class defaults say the honest thing about
  the type (`LLMRateLimitError`, `LLMTimeoutError`, `LLMStreamIdleError` and
  `LLMProviderError` are retryable; `LLMContextWindowExceeded` is not and takes
  no such keyword), and an adapter that classified the response overrides it per
  raise — `LLMProviderError("no such model", retryable=False)` fails on the
  first answer it got. The ladder is bounded by
  `llm_transient_error_retry_max_attempts` (2) with
  `llm_transient_error_retry_backoff_base_seconds` (1.0) doubling up to
  `llm_transient_error_retry_backoff_max_seconds` (8.0), a server-stated
  `Retry-After` taking precedence within that ceiling. The streak resets on any
  clean stream, so the bound is per consecutive-failure streak rather than per
  run. A run that was cancelled, or whose wall-clock budget leaves room only to
  finalise, starts no further attempt; the backoff itself waits on the stop
  event, so a cancel mid-pause is noticed at once. Each attempt is a WARNING
  naming the run and the attempt number, and a `state_changed` event
  (`reason="transient_llm_error_retry"`) the host can surface.
- **A run that produced nothing is not asked to write a report.** The wind-down
  asks the model for the best answer its evidence supports; a run with no prose,
  no tool call and no tool result has none, and asked to close anyway it invents
  the run — the operator reads a polite summary of work that never happened and
  no sign of the failure. So the wind-down is entered only once
  `query.py::_run_produced_output` is true, and otherwise the run goes terminal
  FAILED on the provider's own error. After a tool result exists the partial IS
  an outcome and the wind-down is the right close. Its notice is then per cause
  (`soft_stop_notice_text_provider_error`), because the general one says the run
  reached its budget and a model reads that literally; and its `state_changed`
  events carry `soft_stop_detail` — the upstream's own message — so a host can
  show the operator why the run ended rather than reconstruct it from a log.

### The run snapshot: schema version and upcasters

**What & why.** A snapshot is written by one process and read by another, and
the two are not guaranteed to be the same build. A reader that quietly accepts
a payload it does not understand does not fail — it resumes with fields
missing, latches unset and budgets refilled, and nothing downstream can tell
that apart from a run that legitimately had none of those things. The failure
surfaces much later, as an agent repeating work it already did or spending an
allowance it already spent.

**Key names (`contracts/snapshot.py`).** `SNAPSHOT_SCHEMA_KEY` is where the
version lives in the payload; `SNAPSHOT_SCHEMA_VERSION` is what this build
writes. A payload with no version field at all is read as version 1 — version 1
is exactly the shape the field was added on top of. Anything the build cannot
recognise is refused with `SnapshotSchemaError`, and refusing is the
recoverable outcome: the run stays where it was and an operator sees why.

**Upcasters.** An older payload is not refused where it can be brought forward
instead: one registered `SnapshotUpcaster` per version, each reading the shape
one below it and filling in what that version introduced. A version with no
step is a refusal, because skipping one leaves its fields unset — the same
silent half-restore. Two payload keys are named by the module rather than
spelled at each reader: `RUN_SCOPED_STATE_SNAPSHOT_KEY` (the tree's cumulative
work ledger and its concurrency capacity — the two allowances a resumed run
must not be handed twice) and `PENDING_INTERRUPTS_SNAPSHOT_KEY`.

**RunScopedState.** `contracts/run_state.py` holds the cross-call allowances a
run carries — the streaks the dispatcher counts, the tree work ledger, the
cancel event, the locks, the satisfied preconditions — as one typed object
instead of an untyped dictionary threaded through `ToolContext.metadata` under
an agreed string. A field that moves is a type error at the reader; a host
wiring its own slots keeps them in `RunScopedState.host`, one opaque
compartment, so core never has to know what a host puts there.
`RunScopedState.to_snapshot()` states exactly the two durable allowances and
`apply_snapshot()` puts them back — live objects (an `asyncio.Event`, a
semaphore, a lock) are per-process by nature and are rebuilt by whoever wires
the resumed run. `ToolContext` was narrowed to match: `tenant_id`, `run_id`,
`session_id`, `work_scope`, optional `evidence`, optional `run_state`, and
`metadata` for whatever is left.

### Dispatch: roles, the canonical result, and pairing repair

**What & why.** Dispatch turns one `ToolCall` into one `ToolResult` and puts
it in history. Three things about it are worth stating on their own, because
each replaced a rule the core used to keep in its own head.

**Roles, not names.** The runtime has to know the KIND of a call — did it
produce bytes on disk, does it discharge a read-back obligation, does the
permission gate owe it a shell-safety check. That used to be a comparison
against a tool NAME spelled inside the core, which assumed every installation
names its tools the way the first one did. `contracts/tool_roles.py` replaces
it: `ToolRole` is the capability (`reads_path`, `writes_path`, `appends_path`,
`edits_path`, `finalizes_path`, `searches_workspace`, `runs_shell`,
`fetches_url`, `delegates_work`, `records_plan`, `discovers_tools`,
`asks_user`, `never_delegated`), and `ToolRoleMap` — passed in as
`QueryEngineConfig.tool_roles` — is the host's declaration of which of ITS
names carry which of them, together with the ARGUMENT spellings that go with
them (which key holds the shell command, which holds the body of a write,
which holds a terminal answer). A role the map does not mention is a
capability this installation does not have, and the feature that needs it says
so in a warning rather than going quietly inert.
`runtime/child_capabilities.py::narrow_child_capabilities` reads the same map
to compute what a delegated run may do: narrowing only, never widening, and
applied twice — once when the child's catalogue is resolved and again on each
of the child's calls, so the advertised surface and the gate cannot disagree.

**One value, three audiences.** `ToolResult.content` is the canonical value —
complete, whatever its size — and the projections sit beside it rather than
replacing it. `model_projection` is what the transcript carries in place of the
content when the whole of it does not belong there; `ui_payload` rides the
result event and never enters the transcript at all, so it costs no tokens and
cannot change what the model decides; `canonical_ref` says where the whole
value can be fetched back from once the transcript no longer holds it.
`ToolResult.model_content` is what every path that builds a `ToolResultBlock`
reads, so a tool that names no projection is unaffected. A tool that serves
all three audiences out of one string is why truncating a transcript used to
destroy evidence: there was nothing to truncate but the only copy.

**Durable intent before the call.** Every dispatched call commits an
`IntentRecord` (`runtime/intent.py`) BEFORE the tool is touched, with its
result ids reserved, so a run that dies mid-flight can be told apart from one
that never started — see
[Intent, usage ledger, …](#intent-usage-ledger-session-tree-lanes-typed-hooks-telemetry-live-control-run-work-budget).

**Pairing repair (`runtime/query.py`).** Providers reject a request whose
assistant `tool_use` has no matching `tool_result` — or that carries an
orphaned result, or duplicate ids — with a 400. Pairing is therefore
guaranteed at the wire boundary as defence in depth, not assumed correct from
the mutators upstream of it (compaction, resume from a partial batch,
truncation at `max_tokens`, teardown).

- `_repair_outbound_tool_pairing(messages, placeholder)` — a **pure**,
  unconditional backstop over the outgoing message list, run immediately
  before the request is assembled (and before cache breakpoints are computed,
  so indices address the final list). Four repairs: forward-fill synthetic
  `is_error` results for orphaned `tool_use` blocks, reposition every real
  result directly after its `tool_use`, reverse-strip orphaned results,
  de-duplicate repeated ids.
- `_synthesize_missing_tool_results(history, error_content)` — mutates history
  in place at **every teardown checkpoint** (stop-before-start,
  compaction-failed, lifecycle deny, stop-after-stream, dispatch-cancel, LLM
  terminal error), so the persisted snapshot stays pairing-valid and ordered
  for a resume on another pod. Idempotent.

**Prompt templates.** `tool_result_pairing_repair`, `tool_result_interrupted`.

### Skills routing / surfacing

**What & why.** Surfaces a small **catalog** of available skills into the
system prompt each turn, and loads a full skill body on demand when the user
references it — so domain capability can be added as data, never as per-task
prompt hints. The catalog is a compact, alphabetically ordered
`<system-reminder>` block, built once per run (cached on
`engine._skill_catalog_block`) and placed in the static prompt prefix so it
stays byte-stable across turns (preserving the prompt cache). It is **not**
BM25- or top-K-ranked.

**Key classes/files.** `runtime/skill_index.py` — `render_skills_catalog`
emits `SYSTEM_REMINDER_HEADER` ("Skills are tools, not files… call exactly
`Skill(skill="<name>")`") plus one `Skill(skill="{name}") — {description}`
line per enabled skill, alphabetical by name. Over the token budget the
block degrades to call-shapes only (`Skill(skill="{name}")`).
`derive_skill_index_budget_tokens` is `model_context_window ×
skill_index_budget_ratio` (default 1%). `contracts/skills.py` —
`ISkillStore`, `SkillIndexEntry`, `SkillBundle`, `SkillFileRef`,
`SKILL_ENTRY_PATH` (`SKILL.md`). `list_files` / `load_file` are required
protocol methods for multi-file bundles (at minimum the canonical `SKILL.md`
row; a legacy single-file skill may synthesise that row from `body_md`).
The **core loop never calls** `list_files` / `load_file` — it catalogs via
`list` + `list_enabled_subset` and loads a triggered body via `load` /
`list_subset`. Hosts that expose helper files use the file API themselves.

**How wired.** Step 4 calls `_ensure_run_skill_catalog(engine)`. Skill-store
reads key on `QueryEngineConfig.account_id` (the account-wide bank), **not**
`tenant_id`. When `engine.skills is None` or the store is empty → an
empty-string zero-cost block. Failures are isolated with a WARNING; the run
continues. Per-turn, `<command-name>NAME</command-name>` in the latest user
text loads the matching `SkillBundle.body` as a Layer-3 block, capped by
`max_skills_per_run` (default 4). Project pins (`pinned_skill_names`) are
merged through `list_enabled_subset` so a disabled skill stays off the
catalog.

**RC/extension.** Surfacing is data-driven (empty store = no block).
`skills_hot_reload_enabled` (default `False`) skips the per-run cache and
rebuilds the catalog on every `_ensure_run_skill_catalog` call. Implement
`ISkillStore`; there is no ranker to implement.

### The lifecycle seam + injection / scratchpad + context_bootstrap

**What & why.** One extension seam: observe, decide, transform, or wrap
behaviour at every coordinate of a run, without touching the loop. Plus an
optional turn-1 **context bootstrap** that reads the environment's own
contract/readme docs and prepends a frozen `<environment_context>` orientation
message.

**Key classes/files.** `contracts/middleware.py` — the contract:
`RegistrationKind` (`observe` / `decide` / `transform` / `around` / `notify`),
`LifecycleVerdict`, `LifecycleContext`, `LifecycleDecision`,
`LifecycleOutcome`, `LifecycleScope`, `LifecycleDisposer`, and the
`ILifecycleRegistry` Protocol. `contracts/types.py::HookEvent` — the one list of
coordinates. `hooks/manager.py` — `HookManager`, the core's in-process
implementation: order (priority, then registration), per-registration timeout,
the exception policy, and cancellation that is never read as a verdict.
`contracts/hooks.py` — the out-of-process `IHookManager` contract, `HookResult`,
`HookActionKind`, `HookSpec`. `runtime/correctness_bind.py::fire_lifecycle` is
the loop's single call into the seam.

**How wired.** The registry reaches the engine at construction
(`QueryEngine(..., lifecycle_hooks=...)`) and runs when `typed_hooks_enabled`
is on — one switch over the whole seam, no coordinate gated behind an
unrelated one. `_drive_turn` and the tool dispatch fire `run_start`,
`turn_start`, `context_transform`, `request_prepare`, `response_received`,
`request_error`, `turn_end`, `pre_tool_use`, `tool_execute`, `post_tool_use`,
`pre_compact`,
`compaction_commit`, `compaction_rollback`, `post_compact` and `run_finalize`.
A `transform` at `context_transform` is **applied**: the turn's provider
request is rebuilt from what the chain returned.

The host's `IHookManager` is the same seam reached from another process — an
HTTP endpoint or a model asked to judge. The loop drives it at the permission
gate (`runtime/tool_permission.py`) and around dispatch
(`runtime/tool_dispatch.py`), and it maps `HookActionKind` onto the same
verdicts.

**Exception policy, and why it is split.** `decide`, `transform` and `around`
fail **closed**: a handler that raises, overruns its `timeout_s`, or answers
with something that is not a decision produces a `deny` naming its owner. A
stage that exists to say whether something may happen has not said yes when it
crashes. `observe` and `notify` fail **isolated**: the failure is logged and
recorded in `LifecycleOutcome.failures`, and the verdict, the payload and the
sibling registrations are untouched.

**RC/extension.** `typed_hooks_enabled` (default `False`). The judge hook's
failure mode and deadline, and the context-bootstrap settings, are read by the
surrounding layer and declared there. Extend by registering on the seam with an
owner, a scope, and a disposer, or by implementing `IHookManager` in the host.

### Events / observability / streaming

**What & why.** Streaming is mandatory — every provider delta becomes a typed
`TurnEvent` forwarded immediately. Two distinct event surfaces:

- `events.py` — `EventBus` + `EventName` (~70 names): **in-process** typed
  pub/sub for sibling-handler signalling within a pod (used by HookManager,
  ContextManager, …). Distinct from the cross-pod `IEventStream` (Redis
  Streams) used for SSE reconnect/replay.
- `runtime/events/types.py` — `EventType`: the **per-turn streaming** taxonomy
  (Anthropic-aligned: `message_*`, `content_block_*`, `tool_use_*`,
  `tool_result`, `error`, plus Protocore extensions
  `sandbox_*`/`subagent_*`/`hook_fired`/`tool_call_pending`/`state_changed` and
  loop lifecycle `run_started`/`heartbeat`/`compaction_*`). Later additions
  include `reasoning_step` (Deep-mode plan), `intent_committed`,
  `usage_committed`, `session_forked`, `lane_locked`, `recovery_marked`,
  `compact_checkpoint`, steer/follow-up/queue events (`steer_queued`,
  `follow_up_queued`, `queue_update`), live-control
  `model_changed`/`thinking_changed`, and candidate-verification events
  (`candidate_ready`, `verification_started`, `verification_reported`,
  `repair_requested`, `release_decided`, `candidate_released`), and
  `interrupt_parked` — emitted whenever the run records something it is waiting
  for, carrying that interrupt's id, kind and tool call, so a host learns WHAT
  the run stopped on at the moment it stops rather than by reading the snapshot
  back. Each value is the `event:` line surfaced to SSE clients.
- `runtime/events/envelope.py` — `TurnEvent` (the frozen wire envelope).
- `runtime/llm/delta_bridge.py` — translates a provider's stream into
  `ProviderDelta` → `TurnEvent` (`_normalise_finish_reason`, `is_block_end`,
  …).
  Tool-transport events (`tool_transport_starting`, `tool_transport_ready`,
  `tool_transport_failed`, `tool_transport_teardown`) report the way out to a
  tool coming up, being ready, failing and being torn down; every one of them
  carries the same payload key naming which transport it is about, so a host
  correlates them without parsing text.
- `runtime/events/envelope.py` — `TurnEvent` (the frozen wire envelope).
- `runtime/llm/delta_bridge.py` — translates a provider's stream into
  `ProviderDelta` → `TurnEvent` (`_normalise_finish_reason`, `is_block_end`,
  …).
- `contracts/observability.py` — two optional sinks. `CacheObserverProtocol`
  is the prompt-cache hit-rate sink injected via
  `QueryEngineConfig.cache_observer`. `IRequestManifestSink` answers a
  different question — not "how did this call perform" but "what exactly was
  sent". A provider request is assembled from the history, the compaction
  checkpoint, the pairing repair, the tool surface and the constants in force,
  and until `RequestManifest` existed it lived only in the stack frame that
  made the call: nothing durable could say whether a run that behaved oddly
  was sent a different request or got a different answer to the same one, and
  nothing could re-drive a recorded run without paying for the tokens again.
  The core builds the manifest, computes its id — a SHA-256 over the
  manifest's own canonical serialisation, so it is known before the host has
  written anything and a snapshot can address a manifest by id rather than
  carry it by value — and hands it to the sink. Where it is kept, and for how
  long, is the host's decision: the core has neither a store nor a retention
  policy, and acquiring one mid-run is exactly the obligation the loop must
  not take on.

**How wired.** The loop yields `TurnEvent`s throughout; the usage delta feeds
the cache observer, and `build_llm_request` feeds the manifest sink when
`QueryEngineConfig.request_manifest_sink` is bound. Tracing/observability
sinks are injected across the boundary.

### Safety (shell policy + chain parser + path isolation + approvals)

**What & why.** Validate model-composed shell commands before execution, with
capability-based deny/approval patterns, and isolate workspace paths.

**Key classes/files.**

- `safety/shell.py` — `DefaultShellSafetyPolicy` + `_DENY_PATTERNS`
  (destructive `rm -rf /`, SUID, base64/dd, ANSI-C `$'...'` and locale `$"..."`
  quoting, `$IFS`/`${...IFS}` word-split injection, …). Returns a
  `ShellPolicyDecision` (allow / deny / require-approval).
- `runtime/chain_parser.py` — `parse_chain(...)`: a small shell grammar that
  splits a command on `;`/`|`/`&&` into `CommandSegment`s and surfaces `$()` /
  backtick **substitution bodies** (collected even inside double quotes, not
  single) so per-segment deny patterns re-arm on substitution bodies.
- Path-isolation + approval policies live in `tool_permission.py`
  (`WorkspacePathPolicy`) and the approval flow is the gate's `require_approval`
  stage (loop → AWAITING → resume).

> Note: `DefaultShellSafetyPolicy` **fails open** on a non-match (no
> fail-closed/ambiguous escalation), and `HttpDnsAllowlistPolicy` /
> `WorkspacePathPolicy` are not in the default stack — the host must register
> them via `register_policy`.

### LoopConstants system

**What & why.** The single mechanism for tunable values — **no inline magic
numbers**. Every tunable is a field on a frozen Pydantic snapshot, default-safe,
and dashboard-configurable.

**Key classes/files.**

- `contracts/runtime_constants.py` — `LoopConstants`
  (`model_config = ConfigDict(frozen=True, extra="forbid")`) and the
  `RuntimeConstantsProvider` Protocol (`async get(tenant_id) -> LoopConstants`).
  `extra="forbid"` means an unknown key is a validation error (rejected), not
  silently dropped, so **core and the host must deploy paired**. The snapshot
  includes the default-off surfaces `intent_settlement_enabled`,
  `usage_ledger_enabled`, `lanes_enabled`, `typed_hooks_enabled`,
  `telemetry_spans_enabled` (and `compaction_manual_enabled`,
  `steer_follow_up_enabled`).
- `runtime/runtime_constants.py` — `StaticRuntimeConstantsProvider` +
  `default_runtime_constants(**overrides)` (tests + the in-memory smoke runtime;
  production pods supply a Postgres-backed provider with a Redis cache).
- `constants.py` (~70 lines) — module-level memory-safety caps (`MAX_ARTIFACTS`,
  `MAX_TOOL_CALL_ARGUMENT_BYTES`, `PROTOCOL_VERSION`, `DEFAULT_MODEL`, …).

- `contracts/config.py` — the registry the snapshot is one group of.
  `ConstantSpec` is one knob's descriptor: its wire `kind`, `default`,
  operator-facing `description`, bounds (`minimum` / `maximum`,
  `allowed_values`, `zero_means_unlimited`), and its visibility — `editable`,
  `editable=False` (a row that is shown and refused on write), or
  `not_a_lever="<reason>"` (no row at all: a value the running system derives
  or owns outright, whose appearance in an editor would be an invitation to
  break the deployment). `ConstantGroup` is a set of specs with one `owner` and
  one `key`; `group_from_model` reflects a declaring model into a group, so
  nobody hand-writes a list of hundreds of names that drifts on the first field
  anyone adds, and `build_loop_group` does that for `LoopConstants` itself.
  `IConstantsRegistry` is the declaration side: `declare`, a fail-closed
  `resolve` (a name no group declares raises rather than resolving to a
  default), `defaults`, `coerce` and `repair`. `ICoreConstantsProvider` is how
  the loop asks for the snapshot in force for a scope.

**The tunable surface is a set of groups, not one flat model.** Each group is
declared by the layer that actually reads its values: the loop's own
thresholds are the core's group, and every knob a surrounding layer reads is
declared by that layer, in its own model, through the same `group_from_model`
reflection. A name claimed by two owning groups is refused
(`DuplicateConstantError`); a name claimed by an owning group and by a
`provisional` one — the stand-in a layer keeps while it hands ownership over —
goes to the owner, and the displacement is recorded. That is why a knob
governing authentication, session storage or the transport to a provider is
**not** a field of `LoopConstants` and looking for it there will not find it.

**Adding a tunable.** Add the field to the model whose layer reads it, with a
default-safe value, bounds and a `description`; the group reflects it and the
operator catalogue picks it up from the group. Nothing else in the core has to
be told about it.

### The session work pool and delegated runs

**What & why.** One pool, two kinds of work. A shell command started in the
background and a child run started by delegation are the same thing from the
loop's side: a unit of work with an address, a status, a way to wait for it and
a way to stop it. They used to be two mechanisms — the command was a pool
record with an id, the child run was a function call that blocked its caller
for as long as it took and had no address at all. Nothing could ask a child run
how far along it was, nothing could stop one, and a parent waiting on one held
its turn and its slot in the tree budget for the whole descendant run.

**Key names (`contracts/background.py`).** `IWorkPool` extends
`IBackgroundTaskPool`; `TaskRecord.kind` tells the two apart (`command` /
`agent`), and a subagent handle is simply a `WorkHandle` over a record of kind
`agent`. `AgentRef` says which agent a record of kind `agent` is running and as
which run. `BACKGROUND_TERMINAL_STATUSES` is the set from which a record will
never report again.

**Why it is not run state.** A background task outlives the run that started
it: the command is spawned in one run, the run ends, and whichever run is live
when it finishes is the one that has to be told. So the pool is a collaborator
the host injects, and on a cold start — a fresh process picking up a session
whose tasks were spawned by a process that is gone — the host must put the
session's still-running commands back in the new pool's hands before the loop
asks it anything. `ensure_session_attached` is where the loop asks whether that
happened; a pool answering `False` gets an explicit event on the run rather
than the empty wake list that reads exactly like a session with nothing
running.

**Narrowing a child.** `SubagentDef` is the child's definition and
`runtime/child_capabilities.py::narrow_child_capabilities` computes what it may
do from its parent and nothing else — never a tool, a permission or a hop of
depth more than the parent had, because a run that could widen on the way down
would make every bound above it advisory.

### Intent, usage ledger, session tree, lanes, typed hooks, telemetry, live control, run work budget

Modules that sit beside the shared ReAct loop. Each is default-off unless
the matching RC field says otherwise.

- `runtime/intent.py` — `IntentRecord` / `commit_intent` / `settle_intent` /
  `orphaned_intents` / `unknown_outcome_text` / `replay_policy_for` /
  `repeat_is_safe_for`. **Every** dispatched tool call commits an
  `IntentRecord` with reserved result ids before the tool is touched,
  unconditionally; the record carries a lifecycle `state`
  (`RESERVED|PENDING_APPROVAL|DISPATCHED|PAUSED_ASK_USER|SETTLED`) and the
  turn preamble closes out whatever a stopped run left behind. A crash
  mid-flight is reported as an outcome that was never recorded — never as a
  failure, which would invite a repeat of a side effect that may already have
  happened — and a call parked at a gate, waiting on a user, or merely
  reserved is not reported at all, because none of them ran.
  `intent_repeat_safe_tools` (default `Read,Grep,Glob,ToolSearch`) decides
  which calls skip the durability write and get the milder text;
  `intent_never_replay_tools` (default `Write,Edit,Bash,Finalize,AppendFile`)
  sets `replay`. `intent_settlement_enabled` gates only the recovery events
  and the ledger row. Snapshot field: `open_intents`.
- `runtime/usage_ledger.py` — append-only `UsageRow` list. When
  `usage_ledger_enabled` is on, `correctness_bind.commit_usage` appends a
  row. `_query_raw` records `inference` / `retry` / `compaction` / `abort`
  / `fail` independently of intent. A **tool**-kind row is appended only
  on the intent-settlement dispatch path (the same
  `if intent_settlement_enabled` block that settles the intent). A failed
  attempt plus its retry is two rows. Snapshot field: `usage_rows`.
- Session forking — copying a path of history into a new branch without
  mutating the source. Gated by a host knob; clone requires a settled source,
  and a second host knob caps the number of copied messages.
  **Host-owned** — the loop does not do this.
- `runtime/lanes.py` — named lanes over shared history. `ensure_main`
  makes `main` exist; extras take exclusive locks (`create_lane` /
  `acquire_lane` / `release_lane`). Gated by `lanes_enabled`;
  `lanes_max_per_session` (default 4) includes main. **Host-invoked**;
  `QueryEngine.lanes` is snapshot-persisted so a resume sees the same
  locks.
- `contracts/middleware.py` + `hooks/manager.py` — the lifecycle seam and its
  in-process dispatcher. See
  [the lifecycle seam](#the-lifecycle-seam--injection--scratchpad--context_bootstrap)
  for the coordinates and the exception policy.
- `runtime/telemetry.py` — low-cardinality spans (`run` / `turn` / `step` /
  `tool` / `compact` / `hook`). Gated by `telemetry_spans_enabled`. High-
  cardinality ids stay attributes; `is_prometheus_safe_label` refuses
  `session_id` / `lane_id` / `operation_id` / `run_id` as label keys.
  `mark_recovery` tags a span when an interrupted intent is resumed
  (`correctness_bind.mark_intent_recovery`). Spans live on `engine.spans`
  (in-process); they are **not** in the snapshot.
- `runtime/correctness_bind.py` — glue so intent, ledger, typed hooks, and
  recovery run inside `_drive_turn` (`commit_usage`, `fire_lifecycle`,
  `mark_intent_recovery`, `persist_correctness`).
- `runtime/history_persist.py` — the one call site for every place the loop
  hands the working history to the session store (`HistoryDelta`,
  `HistoryPersister`, `persist_history`). The engine remembers the prefix the
  store already holds, weakly and by object identity, so a round that appended
  two messages hands over two rather than the whole transcript. A store that
  attaches only `persist_session_history` is driven as before; one that also
  attaches `persist_history_delta` is handed the delta and promises, by
  returning, that the write landed — the marker advances only after the call
  returns, and `QueryEngine.forget_persisted_history()` takes the promise back.
  `QueryEngine.note_session_state_changed()` covers a session that changed in a
  way its messages do not show, such as a compaction checkpoint. Not in the
  snapshot: a run picked up on another process cannot know what the store took
  from the one before it, so its first hand-over rewrites.
- `runtime/tool_surface.py` — the advertised tool surface, named rather than
  quoted. `read_tool_surface` serialises and digests the definitions;
  `tool_surface_tokens` costs them once per digest instead of once per LLM
  call; `surface_needs_describing` / `note_surface_described` decide whether
  `tool_surface_advertised` carries each tool's `description`, per reader (the
  session) rather than per process, and `surface_descriptions(digest)` answers
  a reader that kept none. Process-global, bounded by
  `MAX_TOOL_SURFACE_CACHE_ENTRIES` and `MAX_TOOL_SURFACE_AUDIENCES`, and holds
  nothing a run depends on. The request manifest still records the definitions
  in full, so what the provider was sent stays recoverable.
- `runtime/live_control.py` — steer / follow-up queues (`QueuedPrompt`,
  `enqueue`, `place_items`), live model/thinking overrides, and the settled
  helper. Gated by `steer_follow_up_enabled` (default `False`).
- `runtime/run_work_budget.py` — cumulative tree-lifetime budget
  (`max_subagent_runs_per_tree`, `max_total_tokens_per_tree`) for one root
  run and everything it spawns. Sibling of `SubagentTreeBudget` (permits),
  not an extension of it. Exhaustion refuses **delegation**, never the run
  itself.

---

## Extension points (the protocols the host implements)

The core is a set of `Protocol`s; the host provides the concrete adapters.
The full interface surface lives in `contracts/` (each in its own module — there
is **no single `protocols.py`**; the old monolithic file was split per domain).
The principal extension points:

| Protocol | Module | What the host provides |
|---|---|---|
| `ILLMProvider` | `contracts/llm.py` | LLM completions: `stream_with_tools`, `complete_structured`, `complete_text`, and `count_tokens`; a universal LiteLLM/OpenAI-compatible adapter (OpenRouter / vLLM / OpenAI). |
| `IProviderChain` | `contracts/llm.py` | Ordered remaining providers plus a one-way `advance()` cursor. Injected on `QueryEngine(..., provider_chain=...)` for mid-stream failover; `None` leaves existing recovery untouched. |
| `RuntimeConstantsProvider` | `contracts/runtime_constants.py` | Per-tenant `LoopConstants` backed by Postgres + Redis cache. |
| `ISessionStore` | `contracts/session.py` | Session/transcript persistence (Postgres). |
| `IRunStore` | `contracts/run.py` | Run record create/list/read (Postgres + Redis hot record). |
| `IToolRegistry` | `contracts/tool_registry.py` | The concrete `ToolRegistry` is in core; the host registers concrete `Tool`s + visibility policy. |
| `Tool` (ABC) / `@tool` | `contracts/tools.py`, `tools/decorator.py` | Concrete tool implementations (sandbox-backed exec/file tools). |
| `IToolTransport` | `contracts/resilience.py` | The tool/VM transport (e.g. ConnectRPC) the resilience wrapper wraps; optional `rebuild()` hook. |
| `IMemory` | `contracts/memory.py` | A scoped, durable fact store with lexical recall, a two-stage idempotent write and a drift guard, plus an `IMemoryContentScanner`. |
| `IWorkspace` | `contracts/workspace.py` | Durable byte store + Postgres FTS/BM25 manifest, atomic write, per-scope GC. |
| `ISkillStore` | `contracts/skills.py` | Skill bundle storage + lookup **and** multi-file `list_files` / `load_file` (`SkillFileRef`). The core loop catalogs via `list` / `list_enabled_subset` and loads bodies via `load` / `list_subset`; `list_files` / `load_file` are the host file API. The catalog renderer lives in core, `runtime/skill_index.py`. |
| `IHookManager` | `contracts/hooks.py` | The 3-arg dispatcher for hooks whose executor is out of process; driven at the permission gate and around tool dispatch. |
| `ILifecycleRegistry` | `contracts/middleware.py` | The one lifecycle seam — registrations carry an owner, a scope and an idempotent disposer. Default-off behind `typed_hooks_enabled`. |
| `IEventStream` | `contracts/events.py` | Cross-pod durable event stream (Redis Streams) for SSE reconnect/replay. |
| `IBlobStore` | `contracts/blob.py` | Content-addressed blob storage (S3) used by Tier-1 compaction. |
| `ISearchIndex` | `contracts/search.py` | Generic lexical search index. |
| `ITodoStorage` | `contracts/todo.py` | Per-session todo persistence. |
| `IAgentDispatch` | `contracts/agent_dispatch.py` | Subagent dispatch/lookup. |
| `IPromptTemplateProvider` | `contracts/prompts.py` | System-prompt template rendering. |
| `IWorkPool` / `IBackgroundTaskPool` | `contracts/background.py` | The session's work pool: background commands and delegated child runs under one address space, with `ensure_session_attached` on a cold start. |
| `IConstantsRegistry` / `ICoreConstantsProvider` | `contracts/config.py` | Declaration of the host's own constant groups, fail-closed name resolution, and the per-scope `LoopConstants` snapshot the loop reads. |
| `IResilienceClassifier` | `contracts/resilience.py` | The verdict on which neutral failure class a message describes. Core reads failure text it did not write and must not learn to recognise it; an unbound classifier means no wording is recognised, which is the neutral behaviour. |
| `IRequestManifestSink` | `contracts/observability.py` | Where a `RequestManifest` is kept, and for how long. |
| `IDelegationTool` | `contracts/agent_dispatch.py` | The tool shape a delegated run is started through. |
| `IRunToolErrorCounter` | `contracts/run.py` | Durable per-run tool-error counting across processes. |
| `ITurnPolicy` | `contracts/turn_policy.py` | A product decision about a turn, substituted into the core's set by name. |
| `IToolSafetyPolicy` | `runtime/tool_permission.py` | Extra permission policies (`HttpDnsAllowlistPolicy`, `WorkspacePathPolicy`) registered via `register_policy`. |
| `ToolRoleMap` | `contracts/tool_roles.py` | Which of the host's tool names carry which `ToolRole`, and the argument spellings that go with them. Passed as `QueryEngineConfig.tool_roles`. |
| Lifecycle coordinates | `contracts/types.py::HookEvent` | The one list of points a registration may name. |
| `CacheObserverProtocol` | `contracts/observability.py` | Prompt-cache hit-rate sink injected via `QueryEngineConfig.cache_observer`. |
| Self-verify trigger callables | `runtime/query_engine.py` | `pre_terminal_self_verify_trigger` / `pre_dispatch_terminal_verify_trigger` on `QueryEngineConfig`. |

---

## Conventions

- **Import boundary.** Core never imports a package whose name begins
  `protocore_` — the sibling distributions that sit above it. Add
  behaviour via contracts / adapters / RC, not by importing upward. Guard:
  `tests/test_core_import_boundary.py`.
- **No inline magic numbers.** Every tunable the loop reads is a
  `LoopConstants` field (frozen, `extra="forbid"`) or a `constants.py` cap.
  Runtime code reads from the RC snapshot, never a hard-coded literal. A knob
  a surrounding layer reads is declared by that layer, in its own constant
  group (`contracts/config.py`).
- **Horizontal-scale-safe.** No module-level dicts, no module-held locks, no
  per-pod in-memory authority. Correctness-affecting state lives per-run on the
  `QueryEngine` (snapshot/resume); ephemeral cross-pod state is Redis, durable
  state is Postgres — both injected across the boundary. The token-bucket and
  adaptive band take injected locks/stores rather than module state.
- **Streaming is mandatory.** Every drive is an async iterator; every
  provider delta is forwarded immediately as a `TurnEvent`. Do not buffer a
  whole turn before emitting.
- **No backward compatibility.** Dev-version project — break freely, delete
  dead code, no migration shims. (Hence `compaction_thresholds.py` was deleted
  outright once `budgets.py` subsumed it.) The one exception is the run
  snapshot, which crosses builds rather than releases: it states its schema
  version and is either upcast or refused, never half-read
  (`contracts/snapshot.py`).
- **Use `Message` models, never raw dicts.** All messages flow as the Pydantic
  `Message` / `ContentBlock` union from `contracts/types.py`.
- **Do not modify the loop structure.** Customise via a lifecycle
  registration, a turn policy, `QueryEngineConfig` injection, RC toggles, or
  `system_prompt_sections`.
- **A host proves its adapters, and does not wait for a run to do it.**
  `protocore.conformance.SUITES` has one suite per contract; a host binds each
  to its own adapter and finds out in its own test run that the shape is the
  one the core will call. Installed with `pip install "protocore[testing]"`.
- **Production logging = WARNING.** Use `logger.warning(...)` for operationally
  significant events; reserve lower levels for local debugging.

Repo commands: `uv sync --extra dev`, `uv run pytest .`, `uv run ruff check .`,
`uv run mypy --strict` (never with a path — a path replaces the configured
file list and silently drops the test tree).
