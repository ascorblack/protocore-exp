# Protocore Core — архитектура

> Аудитория: инженер, осваивающий **чистое ядро** (`protocore/`).
> Область: этот документ описывает только текущую библиотеку ядра (`protocore/`).
> Адаптеры хоста, сервис FastAPI, фронтенды и
> развёртывание живут в соседних репозиториях и упоминаются здесь только на границе.

---

## Обзор

`protocore` — это **чистое ядро** агентного рантайма Protocore. Это
библиотека на Python 3.12+, состоящая из **контрактов (протоколы + типизированные модели)** и
**protocol-first ReAct-рантайма**, который ведёт по одному ходу агента за раз.

По замыслу это **универсальное ядро продукта**, а не стенд для бенчмарков:

- **Ноль импортов вверх.** Ядро никогда не импортирует пакет, который стоит
  над ним, — ничего с его же именем и подчёркиванием после (`protocore_*`). У него
  нет драйвера базы данных, нет HTTP-эндпоинта, нет логики Kubernetes. Всё, что
  обращено наружу, — это `Protocol`, который реализует хост. Гарантируется
  тестом `tests/test_core_import_boundary.py`.
- **Универсальность / мультитенантность.** Ни в одном исполняемом пути нет логики,
  завязанной на отдельную задачу, tenant-id, промпт или
  скорер/рубрику. Каждый метод scoped по тенанту; политика тенанта инъецируется
  (через `LoopConstants` и `ToolContext.metadata`) и никогда не зашита в код.
- **Всё конфигурируемо через `LoopConstants` и безопасно по умолчанию.** Настраиваемые
  значения проходят через `LoopConstants` (замороженный Pydantic-снимок) или
  `constants.py` (лимиты безопасности по памяти). Новые возможности по умолчанию **выключены** или принимают
  значение, воспроизводящее прежнее поведение, так что тенант подключает их осознанно.
- **Безопасно при горизонтальном масштабировании.** Никаких словарей на уровне модуля, никаких блокировок `asyncio`,
  удерживаемых как состояние модуля, никакой авторитетности на уровне отдельного пода. Долговременное состояние — это Postgres;
  эфемерное межподовое состояние — это Redis (оба предоставляются через границу). Состояние, влияющее на
  корректность, живёт пер-ран на экземпляре `QueryEngine`.

### Направление зависимостей

```
protocore (чистое ядро, ноль импортов вверх)
  └─> хост-дистрибутив (адаптеры, сервисный слой, HTTP API)
        ├─> фронтенды (только HTTP/SSE)
        └─> бэкенд исполнения (только контракты сервисного API)
```

`protocore` — это корень. Он никогда не должен импортировать вверх. Сторожевой
тест утверждает, что импорт любого модуля `protocore.*` подтягивает **ноль**
символов из слоёв над ним.

### Public API (`protocore/__init__.py`)

Публичная поверхность **contract-first**: реэкспортируются те интерфейсные
`Protocol`-ы store / service, за которыми хост тянется чаще всего, плюс ABC
`Tool` — `IAgentDispatch`, `IBlobStore`, `IEventStream`, `IHookManager`,
`ILifecycleRegistry`, `ILLMProvider`, `IRunStore`, `ISearchIndex`,
`ISessionStore`, `ISkillStore`, `IToolRegistry`, `ITodoStorage` и `Tool`.
Остальные из 32 `Protocol`-ов импортируются из собственного контрактного модуля
и на верхнем уровне **не** реэкспортируются: `IMemory` (`contracts/memory.py`),
`IWorkspace` (`contracts/workspace.py`), `IToolTransport` и
`IResilienceClassifier` (`contracts/resilience.py`), `IPromptTemplateProvider`
(`contracts/prompts.py`), `IWorkPool` (`contracts/background.py`),
`IConstantsRegistry` и `ICoreConstantsProvider` (`contracts/config.py`),
`IRequestManifestSink` (`contracts/observability.py`), `IProviderChain`
(`contracts/llm.py`) и `ITurnPolicy` (`contracts/turn_policy.py`).
Поверхность также реэкспортирует основную систему
типов (`Message`, `ToolCall`, `ToolResult`, `Event`, `Run`, `Session`,
`SubagentDef`, объединение
`ContentBlock`, …), `LoopConstants` + `RuntimeConstantsProvider`, словарь
жизненного цикла (`RegistrationKind`, `LifecycleVerdict`, `LifecycleContext`,
`LifecycleDecision`, `LifecycleOutcome`, `LifecycleScope`,
`LifecycleDisposer`),
`EventBus`/`EventName`, lifecycle-`HookManager`, `DefaultShellSafetyPolicy`,
декоратор `@tool`, утилиты envelope/JSON и помощники подсчёта токенов
(`LanguageProfile`, `chars_per_token`, `detect_profile`, `estimate_tokens`). Она
**не** реэкспортирует `derive_budgets`, `retrieve_tools` или `bm25_score` — они
импортируются напрямую из своих рантайм-модулей
(`runtime/context/budgets.py`, `runtime/tool_retrieval.py`).

Машинерия цикла (`runtime/query.py` + `runtime/query_engine.py`) — это сердце
рантайма. Точки входа в цикл импортируются напрямую из
`protocore.runtime.query` / `protocore.runtime.query_engine`; они **не**
реэкспортируются на верхнем уровне.

---

## Архитектурные диаграммы

### Слоевая структура

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ CONTRACTS / PROTOCOLS  (protocore/contracts/)                                   │
│   types.py  (Message, ToolCall, ToolResult, ContentBlock union, Run, Session,   │
│              ExecutionReport, StopReason, AgentEnvelope, …)                      │
│   16 interface Protocols: llm.py ILLMProvider + IProviderChain · run.py          │
│     IRunStore · session.py ISessionStore · blob.py IBlobStore ·                  │
│     search.py ISearchIndex · todo.py ITodoStorage ·                              │
│     tool_registry.py IToolRegistry · skills.py ISkillStore ·                     │
│     agent_dispatch.py IAgentDispatch · events.py IEventStream ·                  │
│     hooks.py IHookManager · memory.py IMemory · workspace.py IWorkspace ·        │
│     resilience.py IToolTransport · prompts.py IPromptTemplateProvider            │
│   runtime_constants.py  LoopConstants (frozen, extra="forbid") + Provider     │
│   terminal_answer_validation.py ·                                             │
│   attempt_ledger.py · tool_action_preconditions.py · observability.py ·          │
│   evidence.py · tool_chunking.py                                                │
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
│   loop_strategies.py ── DirectStrategy | DeepStrategy (run_mode)                │
│   intent.py · usage_ledger.py · lanes.py ·                                      │
│   telemetry.py · correctness_bind.py ·                                          │
│   compact_checkpoint.py · live_control.py · run_work_budget.py                  │
│        LoopState (loop_state.py): PENDING→RUNNING→{AWAITING|COMPACTING}→         │
│                                   {COMPLETED|FAILED|CANCELLED}                   │
└──────────────────────────────────────────────────────────────────────────────┘
        │                 │                       │                    │
        ▼                 ▼                       ▼                    ▼
┌───────────────┐ ┌────────────────┐ ┌──────────────────────┐ ┌────────────────┐
│ TOOL SURFACE  │ │ TOOL DISPATCH  │ │ CONTEXT / COMPACTION  │ │ FINALIZATION   │
│ + RETRIEVAL   │ │ + GATING       │ │ context/manager.py    │ │ + GROUNDING    │
│ tool_registry │ │ tool_dispatch  │ │ context/budgets.py    │ │ finalization_  │
│ tool_retrieval│ │ ToolDispatcher │ │ context/compaction.py │ │   gate.py      │
│ @tool decorat.│ │ tool_permission│ │ context/session_      │ │ finalization_  │
│               │ │   Gate (4 stg) │ │   memory.py           │ │   contract.py  │
│               │ │ tool_precondi- │ │ compact_checkpoint.py │ │                │
│               │ │   tions (DAG)  │ │ token_counting.py     │ │                │
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
│ tools/       │ │ read_dedup_  │ │ runtime/     │ │  skills.py   │ │  events/*    │
│   memory.py  │ │  cache.py    │ │  resilience  │ │  list_files/ │ │ runtime/llm/ │
│ (IMemory)    │ │ (IWorkspace) │ │ attempt_     │ │  load_file   │ │  delta_bridge│
│              │ │              │ │  ledger ·    │ │              │ │ hooks/       │
│              │ │              │ │ adaptive_    │ │              │ │  manager,    │
│              │ │              │ │  safety_band │ │              │ │  manager +   │
│              │ │              │ │ run_work_    │ │              │ │ middleware   │
│              │ │              │ │  budget      │ │              │ │  contract    │
│              │ │              │ │              │ │              │ │              │
└──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
        │                 │                 │                 │                 │
        ▼                 ▼                 ▼                 ▼                 ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ SAFETY  (protocore/safety/)  shell.py DefaultShellSafetyPolicy + deny patterns │
│         + chain_parser.py (segment/substitution grammar)                       │
├──────────────────────────────────────────────────────────────────────────────┤
│ HOST-ADAPTER BOUNDARY  (lives in the host distribution — NOT core)            │
│   LiteLLM/OpenAI-compat ILLMProvider · PgMemoryStore · IWorkspace store ·      │
│   PostgresStateManager · sandbox-backed exec/file tools · ConnectRPC transport │
│   · IHookManager adapter · RuntimeConstantsProvider (Postgres + Redis cache)   │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Поток данных одного хода агента

`QueryEngine.run(message)` добавляет пользовательское сообщение, ставит часы
запуска, сохраняет снимок начала хода и ведёт один ход;
`resume(engine, snapshot)` сначала восстанавливает сохранённый прогон, а затем
ведёт то продолжение, которое описала вызывающая сторона. Оба привязывают
ведущую задачу, чтобы `stop()` мог жёстко отменить, и оба сохраняют снимок на
выходе, каким бы он ни был. Каждый внутренний `yield` из приватного генератора
`_query_raw` — контрольная точка stop-check; исполнитель транслирует выпущенные
`TurnEvent`-ы наружу по SSE (Redis pub/sub на уровне хоста).

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

Где подсистемы встраиваются:

- **Memory** инъецирует автоматически вспомненные факты перед вызовом LLM (шаг 4) и
  читается/пишется инструментами вида `read`/`write`/`recall` во время диспетчеризации.
- **Workspace** обслуживает `read`/`write`/`find`/`search` во время диспетчеризации; процесс-локальный
  **read-dedup cache** замыкает накоротко повторный `read`
  того же пути/содержимого.
- **Grounding** записывает ссылку-цитату всякий раз, когда срабатывает отслеживаемый для grounding `read`;
  **finalization gate / terminal-answer validation** потребляют этот
  реестр ссылок, когда формируется терминальный `answer`.
- **Resilience** оборачивает бюджет исходящего вызова LLM (AdaptiveSafetyBand) и
  доступен как универсальная обёртка `IToolTransport` для вызовов инструментов/VM
  (хост привязывает её).
- **Hooks/events** срабатывают в каждой точке жизненного цикла (UserPromptSubmit, pre/post
  tool, pre/post compact), и каждая дельта провайдера становится `TurnEvent`.
  Когда `typed_hooks_enabled` включён, `correctness_bind.fire_lifecycle`
  испускает каждую координату, через которую проходит цикл, а `transform` на
  `context_transform` применяется к контексту хода. Один тумблер управляет всем
  швом; ни одна координата не спрятана за вторым, посторонним.
- **Intent settlement + usage ledger**: **каждый** диспетчеризуемый инструмент
  коммитит `IntentRecord` до вызова, безусловно, а ход открывается закрытием
  записей, оставленных прогоном, который остановился на лету.
  `intent_settlement_enabled` (выключен по умолчанию) гейтит поверх этого
  только события восстановления и строку журнала. Usage-строки для
  `inference` / `retry` / `compaction` / `abort` / `fail` идут через
  `commit_usage`, когда `usage_ledger_enabled` включён; строка вида **tool**
  пишется только на пути intent-settlement.
- **Live control** держит очереди steer / follow-up и живые переопределения
  model/thinking; **CompactCheckpoint** — операторский путь `/compact` (не
  третий ярус компакции).

---

## Инвентаризация технологий

По одной строке на каждую технологию ядра. **Wired into loop?** = на технологию ссылается основной
рантайм-цикл (`query.py` / `query_engine.py`); подсистемы, подключённые только
адаптером хоста, помечены соответственно. **RC toggle(s) + default** фиксирует
управляющее поле (поля) `LoopConstants` и его безопасный/выключенный default.

| Technology | Core files | RC toggle(s) + default | Wired into loop? | Tested? |
|---|---|---|---|---|
| ReAct loop / orchestrator / query engine | `runtime/query.py`, `runtime/query_engine.py`, `runtime/loop_state.py`, `runtime/loop_strategies.py` | n/a (always on); recovery branches RC-gated | Yes | Yes |
| Tool dispatch + gating | `runtime/tool_dispatch.py`, `runtime/tool_permission.py` | gate always on; consecutive-error cap RC | Yes | Yes |
| Tool retrieval / registry | `runtime/tool_registry.py`, `runtime/tool_retrieval.py` | `tool_retrieval_top_k` (clip threshold) | Yes | Yes |
| Tool preconditions | `runtime/tool_preconditions.py`, `runtime/run_tool_preconditions.py` | `tool_preconditions_enabled` = `False`; run-level `QueryEngineConfig.tool_preconditions` empty | DAG + run-level forcer: Yes | Yes |
| Turn policies | `contracts/turn_policy.py`, `runtime/turn_policies/*` | каждая политика читает свои поля RC; ПОРЯДОК принадлежит ядру (`TURN_POLICY_ORDER`) | Yes — драйвер опрашивает реестр на 14 координатах | Yes |
| Tool roles + argument spellings | `contracts/tool_roles.py`, `runtime/child_capabilities.py` | нет — карта приходит как `QueryEngineConfig.tool_roles`, её объявляет хост при регистрации инструментов | Yes | Yes |
| Constants registry | `contracts/config.py` | сама система (`ConstantSpec` / `ConstantGroup` / `IConstantsRegistry`) | Объявление и разрешение имён — на стороне хоста; цикл читает снимок | Yes |
| Snapshot schema + upcasters | `contracts/snapshot.py` | нет — схема не ручка | Yes (каждый `snapshot()` / `resume_from_snapshot()`) | Yes |
| Interrupts (approval / question / external call) | `contracts/interrupt.py`, `runtime/query.py::resume_interrupts` | нет — припаркованный вызов не opt-in | Yes | Yes |
| Session work pool (фоновые команды + дочерние прогоны) | `contracts/background.py` | в ядре нет; пул инъецируется | Yes (`ensure_session_attached`, `drain_wakes`) | Yes |
| Request manifest | `contracts/observability.py`, `runtime/query.py::build_llm_request` | нет — прогон записывает, когда привязан `QueryEngineConfig.request_manifest_sink` | Yes, когда приёмник привязан | Yes |
| Conformance suites | `protocore/conformance/*` | n/a (пакет времени тестов, `protocore[testing]`) | No — хост гоняет их против собственных адаптеров | Yes |
| Universal resilience layer | `contracts/resilience.py`, `runtime/resilience.py` | `resilience_enabled` = `False`; the transport attempt count is a host knob | Ledger/band: Yes; transport wrapper: host-only | Yes |
| Failure classification | `contracts/resilience.py::IResilienceClassifier`, `runtime/error_kinds.py` | нет — классификатор приходит как `QueryEngineConfig.resilience_classifier`; если он не привязан, ни одна формулировка не распознаётся | Yes | Yes |
| Run wind-down (soft stop) | `runtime/soft_stop.py` | `soft_stop_enabled` = `True`, `soft_stop_max_turns` = `3` | Yes | Yes |
| Attempt ledger + adaptive safety band | `contracts/attempt_ledger.py`, полоса на стороне хоста | band wired via per-call output budget | Yes | Yes |
| Finalization gate + contract | на стороне хоста | `terminal_tool_nudge_enabled` (`False`), `terminal_tool_forced_max_attempts`, `terminal_tool_forced_thinking_enabled`, `terminal_tool_nudge_write_first_before_forcing`, `finalize_prose_gate_enabled` | Yes | Yes |
| Terminal-answer validation + references/grounding | на стороне хоста (ядро несёт собранные прогоном свидетельства, `contracts/evidence.py`) | host knobs (validation and reference normalisation are both driven from the host's own model) | Yes | Yes |
| IMemory subsystem | `contracts/memory.py`, `tools/memory.py` | `memory_enabled` = `False`; auto-recall is a host knob | Host-wired (tools held by core contract) | Yes |
| Token counting | `runtime/token_counting.py` (+ опциональный оценщик `protocore-native`) | `chars_per_token_*` ratios in RC; `PROTOCORE_DISABLE_NATIVE` принудительно оставляет чисто-питоновый путь | Yes | Yes |
| IWorkspace + read-dedup cache | `contracts/workspace.py`, кэш на стороне хоста | n/a (no snapshot toggle; the host owns the surface) | **No** (host-wired) | Yes |
| Context management / three-tier compaction / session memory | `runtime/context/manager.py`, `runtime/context/compaction.py`, `runtime/context/budgets.py`, `runtime/context/session_memory.py`, `runtime/compact_checkpoint.py` | ratios in RC; `compaction_manual_enabled` = `False` | Compaction + `/compact`: Yes; session-memory fold: host-wired | Yes |
| Prompt caching | `runtime/prompt_caching.py` | wire translation gated by a host kill-switch | Yes (hints in core; wire translation the host) | Yes |
| Skills routing / surfacing | `runtime/skill_index.py`, `contracts/skills.py` | data-driven (empty store = no block); `skills_hot_reload_enabled` = `False` | Yes (`_ensure_run_skill_catalog`); `list_files`/`load_file` host-only | Yes |
| Шов жизненного цикла + injection / context_bootstrap | `contracts/middleware.py`, `hooks/manager.py`, `runtime/correctness_bind.py` | `typed_hooks_enabled` = `False`; режим отказа judge и context bootstrap — ручки хоста | Один шов, пять видов регистрации, один список координат (`HookEvent`); `decide`/`transform`/`around` падают закрыто, `observe`/`notify` изолированы | Да |
| Events / observability / streaming | `events.py`, `runtime/events/*`, `runtime/llm/delta_bridge.py`, `runtime/telemetry.py` | `telemetry_spans_enabled` = `False` | Yes | Yes |
| Intent / usage ledger / session tree / lanes | `runtime/intent.py`, `runtime/usage_ledger.py`, ветвление сессии на стороне хоста, `runtime/lanes.py` | `intent_settlement_enabled`, `usage_ledger_enabled`, `lanes_enabled` (all `False`); the session tree is a host knob | Intent + ledger: Yes when on; tree/lanes: host-invoked | Yes |
| Live control + run work budget | `runtime/live_control.py`, `runtime/run_work_budget.py` | `steer_follow_up_enabled` = `False`; tree token/run caps | Yes | Yes |
| Safety (shell policy + chain parser) | `safety/shell.py`, `runtime/chain_parser.py` | policy stack via `register_policy` | Yes | Yes |
| LoopConstants system | `contracts/runtime_constants.py`, `runtime/runtime_constants.py`, `constants.py` | the system itself | Yes | Yes |

Сквозной факт: **большинство новых возможностей выключены по умолчанию** и не
задействованы на тенанте по умолчанию, поэтому их *включённые* пути покрываются модульными
тестами, а не живыми прогонами. Это сделано намеренно — это опциональные продуктовые
возможности.

---

## Секции по технологиям

Ниже — подробный разбор тех строк инвентаризации, которым мало одной строки.
Строки, которых он не разворачивает — retrieval, DAG предусловий, обёртка
устойчивости, attempt ledger, память, workspace, двухъярусная компакция, счёт
токенов и кэширование промпта, — ведут себя ровно так, как файлы, названные в
таблице, и повторять их здесь — верный способ получить второе описание,
расходящееся с первым.

### ReAct loop / orchestrator / query engine / loop state

**Что и зачем.** Это сердце рантайма: ReAct-цикл (reason→act→observe), который
выполняет **по одному ходу ассистента за раз** и выдаёт потоковые события. Он
разделён на состояние и поведение, так что любой под может возобновить ран после
краха другого. Общий цикл ассистента **не** является единственным неизменяемым
путём: `QueryEngineConfig.run_mode` выбирает `DirectStrategy` или `DeepStrategy`
в `runtime/loop_strategies.py` до этого общего цикла.

**Ключевые классы/файлы.**

- `runtime/query_engine.py`
  - `QueryEngine` — один экземпляр на активный ран. Владеет **изменяемым
    состоянием на разговор**: `history` (список `Message`), машиной `LoopState`,
    `CompactionState`, `TokenUsage`, плюс `open_intents`, `usage_rows`,
    `lanes`, очередями live-control (`_steer_queue` / `_follow_up_queue`),
    живыми переопределениями model/thinking (`_live_model_name` /
    `_live_thinking_enabled` / `_live_reasoning_effort`), опциональным
    `verification` (`VerificationLifecycle`) и защёлками восстановления
    (terminal-only, guaranteed-terminal, self-verify, circuit-breaker,
    pending-reads, longfile, индекс tool-precondition, …). Персистентность —
    `snapshot()` ↔ `resume_from_snapshot()`; `run()` снимает snapshot в начале
    хода и в `finally`. Снимок также пишет `open_intents`, `usage_rows`,
    `lanes`, живые поля `live_*`, очереди steer/follow-up, `verification`
    (когда не default) и эти защёлки восстановления.
  - `QueryEngineConfig` — **неизменяемая поверхность инъекции**, привязываемая при
    построении движка: `run_id`/`tenant_id`/`session_id`/`model_name`,
    `system_prompt_sections`, `tool_visibility_policy`, снимок `rc`,
    `run_mode` (`"direct"` | `"deep"`, по умолчанию `"direct"`),
    `execution_profile`, `thinking_enabled` / `reasoning_effort`,
    `expected_terminal_tool`, `tool_preconditions` (run-level forcer;
    пустой default), опциональный `cache_observer`, опциональный
    `verification_delivery` и два **поставляемых хостом триггерных
    колбэка** (`pre_terminal_self_verify_trigger`,
    `pre_dispatch_terminal_verify_trigger`) — оба по умолчанию `None`, так что
    машинерия pre-dispatch veto / self-verify мертва, пока хост не
    инъецирует колбэк *и* не переключит соответствующий RC.
  - `QueryEngine.__init__` принимает опциональный `provider_chain: IProviderChain`
    для mid-stream failover провайдера. `None` (у каждого вызывающего, кто не
    настроил список приоритетов) оставляет существующее восстановление
    нетронутым.
  - `QueryEngine.run(initial_message)` — драйвер-**асинхронный генератор**: он
    добавляет пользовательское сообщение (или продолжает по уже существующей
    истории, заканчивающейся user-сообщением), увеличивает `turn_count`,
    сбрасывает состояние хода, ставит часы запуска, сохраняет снимок начала
    хода, привязывает `_current_turn_task`, затем итерирует `_query_raw`.
    Снимок конца хода попадает в `finally`. И привязка ручки, и этот
    заключительный снимок приходят из `driving_turn()` — области, внутри
    которой идёт любой публичный драйв, так что два обязательства, делающие
    драйв прерываемым и поднимаемым, принадлежат одному месту.
- `contracts/interrupt.py` — то, чего ждёт приостановленный прогон, как
  значение, а не как защёлка. `PendingInterrupt(interrupt_id, kind,
  tool_call_id, tool_name, payload, created_at_ms, expires_at_ms)` несёт одно
  ожидание; `InterruptKind` говорит, какое из трёх это (`approval` — вызов,
  припаркованный на воротах и НЕ исполнявшийся; `question` — вызов, который
  успел спросить, и ответ на вопрос и есть его результат; `external_call` —
  результат придёт другим путём), и вид решает, какие резолюции допустимы.
  Движок держит столько ожиданий, сколько открыто, в порядке парковки, и пишет
  их в снимок; `LoopState.AWAITING` без единого записанного ожидания
  отвергается на переходе (`loop_state.assert_awaiting_is_witnessed`): это
  прогон, который останавливается, не назвав, что могло бы его поднять.
  `InterruptResolution` — одно решение (`approve`, опционально с
  `updated_input` — исправленными аргументами, с которыми вызов и будет
  действительно исполнен и записан; `deny`; `answer`; `abandon`), а
  `plan_resolution` отвергает карту, которая называет незакрытое ожидание,
  отвечает решением, не подходящим виду, или оставляет открытое прерывание без
  решения, не сказав об этом явно.
- **Идемпотентность кода до прерывания.** Припаркованный вызов возобновляется
  ровно там, где остановился: ничто до прерывания не переисполняется, поэтому
  работа между началом хода и парковкой не повторяется. Чего возобновлённый
  прогон делать не должен — переиздавать сам припаркованный вызов, и это
  закрывает durable-запись намерения (`runtime/intent.py`): запись в
  `PENDING_APPROVAL` — вызов, который не исполнялся, запись в
  `PAUSED_ASK_USER` — вызов, ответ на который ещё должен прийти, а
  `intent_never_replay_tools` называет инструменты, повтор которых недопустим
  при любой записи. Это одна гарантия с двух сторон: прерывание говорит, чего
  ждут, намерение — что с этим позволено сделать.
- `runtime/query.py` — `resume(engine, snapshot, *, approved_tool_call=None,
  message=None, abandon_approval=False, resolutions=None,
  allow_partial_resolution=False)` — публичная точка подъёма: она строго
  восстанавливает снимок (схема, режим доставки и привязка идентичности
  проверяются до первой мутации), а затем ведёт карту резолюций, одобренный
  вызов, пришедшее сообщение или простой перезапуск прерванного хода — то, что
  описали аргументы вызывающей стороны. `resolutions` — общая форма и
  единственная, которой выражается батч: несколько припаркованных вместе
  вызовов отвечаются ОДНИМ драйвом, каждый своим решением, и их результаты
  ложатся в том порядке, в котором их просила модель. Прогон с чем-либо
  припаркованным отвергает простой перезапуск; `abandon_approval=True`
  закрывает каждый припаркованный вызов как неотвеченный.
  `resume_interrupts` — тот же драйв отдельно, для вызывающей стороны, которая
  уже восстановила прогон. `resume_approved_tool` исполняет один вызов,
  удержанный на одобрении: сверенный с durable ожидающим вызовом и
  идемпотентный на повторе; оба тоже идут внутри `driving_turn()`. `_query_raw` реализует жизненный цикл хода: stop check → resume прерванных
  намерений → координата `run_start` → опциональный `/compact` через
  `CompactCheckpoint` → compaction check → UserPromptSubmit hook → build
  context → `select_strategy(run_mode).prepare_turn` →
  `_stream_one_assistant_message` (рекурсивно на tool_use) → dispatch →
  finalize. Восстановление шире набора 413 / max-output / thinking-trap /
  empty-nudge / idle-watchdog: ход также возобновляет прерванные намерения,
  стреляет координату `run_start`, обрабатывает `/compact` через
  `CompactCheckpoint` и привязывает usage/hooks через
  `runtime/correctness_bind.py` (`commit_usage`, `fire_lifecycle`,
  `mark_intent_recovery`, `persist_correctness`). Более старые ветви
  восстановления остаются модель-агностичными и закрытыми RC-гейтами.

  Восстановление после переполнения контекстного окна допускает не более одной
  компакции на сообщение ассистента. Точный размер промпта от провайдера может
  разрешить одну попытку с меньшим лимитом до компакции. При отсутствующем
  размере или нижней границе сначала выполняется компакция; дальнейшие отказы
  каждый раз строго уменьшают отклонённый wire-лимит по
  `context_overflow_retry_output_ratio` и прекращаются после
  `context_overflow_retry_max_attempts` либо когда положительный лимит уже
  невозможно уменьшить.
- `runtime/loop_strategies.py` — `select_strategy(run_mode)` — единственная
  точка ветвления. `DirectStrategy` не вносит pre-action шага (auto-tool
  цикл). `DeepStrategy` запускает принудительный инструмент планирования
  (`extra["forced_tool_choice"]` в запросе, CoT ограничен `reasoning_effort`),
  эмитирует ровно одно
  событие `REASONING_STEP`, затем общий цикл ассистента ведёт реальное
  действие с полной поверхностью.
- `runtime/query.py::build_llm_request` — единственный сборщик, через который
  проходит каждый вызов провайдера: поток действий, plan-вызов глубокого
  режима, его fallback на prompted-JSON и оба суммаризатора компакции. Он
  фиксирует три вещи, которые эти четыре пути раньше решали порознь: модель в
  силе (живое переопределение, когда оно задано), принудительный инструмент
  (один слот, `extra["forced_tool_choice"]`, несущий ИМЯ инструмента, чтобы
  адаптер отрисовал его в собственный формат провода; либо вместо него
  `extra["tool_choice_required"] = True` — «какой-нибудь инструмент, без
  прозы», отрисовывается как `tool_choice="required"`; адаптер без поддержки
  игнорирует оба — см. `LLMRequest.extra`) и температуру (значение
  вызывающего или `None`, и тогда решает хост — настройка модели или
  собственный generation config сервера; суммаризаторы задают свою явно).
- `runtime/context/budgets.py` — `derive_budgets` превращает один снимок RC во
  все послойные бюджеты токенов, детерминированно и без кэша. Порог компакции,
  который он возвращает, — МЕНЬШАЯ из двух границ: заданной доли окна
  `compaction_trigger_ratio` и наибольшего промпта, который провайдер ещё
  примет, — окно минус резерв вывода, минус `request_context_safety_tokens`,
  минус доля окна `compaction_trigger_turn_headroom_ratio` на тот ход, который
  вот-вот добавится. Сервер, засчитывающий запрошенный вывод в то же окно, что
  и промпт, отвергает любой запрос больше `окно − максимум вывода`, поэтому
  порог, выведенный из одной лишь доли, может оказаться выше обрыва и не
  сработать никогда: на окне в 65 536 токенов при штатном резерве 0.25 доля 0.8
  лежит на 3 276 токенов дальше точки, после которой запрос перестают
  принимать. Вычет резерва включается флагом
  `provider_reserves_output_in_context_window` (по умолчанию истина): эндпоинт,
  размеряющий входное окно независимо от запрошенного вывода, выставляет его в
  ложь и возвращает себе эту долю окна. Потребители читают эффективное
  значение; аварийный обрыв держится строго выше него. Оба порога — размеры
  всего промпта, поэтому гейт прибавляет к оценке истории то, что последний
  запрос нёс помимо неё, — системные сообщения и определения инструментов, в тех
  же откалиброванных токенах; до первого запроса рана эта часть равна нулю.
- `runtime/request_budget.py` — подгоняет каждый собранный запрос под жёсткое
  окно, урезая лимит вывода. Размер, под который идёт подгонка, — откалиброванная
  оценка, кроме случаев у края: когда оценка доходит до доли
  `exact_token_count_margin_ratio` от размера промпта, с которого лимит начинает
  урезаться, провайдера, реализующего необязательную возможность
  `IRequestTokenCounter`, спрашивают, во что отрисовывается запрос, и берут его
  число. Тот же вопрос задаётся о долговременной истории, когда она в пределах
  этой доли от триггера компакции, так что гейт решает в токенах провайдера, —
  пока подгонка не посчитала полный запрос хода; после этого гейт читает
  коэффициент, выставленный тем подсчётом, и второй запрос не делает. После
  первого подсчёта подгонка спрашивает снова, только если последний подсчёт
  плюс содержимое, которого тот подсчёт не видел (каждое сообщение, чьего
  дайджеста нет среди посчитанных, — так что переписанное компакцией или
  вытеснением сообщение считается новым, а удалённые не учитываются), взятое с
  худшей недооценкой, которую
  предполагает доля (`1 / (1 - доля)`), может перейти предел: ходы с прозой
  считаются примерно один раз у края, а большой плотный результат инструмента
  считается сразу. Подсчёт, который не удался или не уложился во время,
  выключает подсчёты на `exact_token_count_failure_backoff_seconds`. Доля по
  умолчанию, 0.75, равна `1 - 1/4`: недооценка в `f` раз ловится только при доле
  не меньше `1 - 1/f`, эвристика измеренно недосчитывает JSON в 1.76 раза и
  шестнадцатеричный текст в 3.53 раза, а 4 — наибольший множитель, который
  выражает калибровка. Точный подсчёт выставляет `token_estimate_calibration`
  сразу и до попытки подгонки, так что подсчёт, доказывающий, что запрос не
  влезает, сперва поднимает коэффициент, а отказ уходит в компакцию, размеренную
  в посчитанных токенах; отчёт `usage` после вызова сдвигает его наполовину;
  отказ по длине поднимает его до нижней границы, которую отказ доказывает
  (промпт был не меньше окна за вычетом отправленного лимита вывода). Подсчёты
  хранятся по содержимому запроса (`exact_token_count_cache_max_entries`);
  подсчёт, который не удался или не пришёл за
  `exact_token_count_timeout_seconds`, пишется в лог, и используется оценка;
  провайдер без этой возможности отправляет ровно те же запросы, что и раньше.
  Каждый свежий подсчёт пишет в лог оценку рядом с измерением — расхождение
  между ними. Выученный коэффициент живёт в снимке прогона и переживает resume
  на той же модели; новый прогон начинает с настроенного
  `token_estimate_calibration`, так что хост, знающий, что его содержимое
  плотное, задаёт это значение для scope.
- `runtime/context/compaction.py`, `runtime/context/carrier.py`,
  `runtime/context/ledger.py` — каскад компакции, полностью описанный в
  [контракте компакции](compaction.md). Проход открывается, когда весь промпт
  (история плюс системный промпт и инструменты) переходит триггер, и целится в
  `compaction_target_ratio` от него. **Маскирование** заменяет старые или
  слишком большие выводы инструментов заглушкой, которая называет инструмент,
  сохраняет строки, сказанные выводом лишь однажды, и указывает на оригинал в
  blob-хранилище. **Суммаризация** заменяет самые старые отрезки — соседние
  блоки вызовов, объединённые до `compaction_summary_group_max_tokens`, —
  обычным текстом под пятью фиксированными заголовками, который запрашивается
  через `complete_text` с инструкцией в системной роли и читается терпимо;
  **свёртка** сливает серии старых выжимок и реплик оператора. **Пол** убирает
  самые старые отрезки без модели, когда уровни выше не дотянули, так что
  проход всегда заканчивается ниже триггера или когда убирать больше нечего.
  Каждый уровень сначала записывает убираемое в **реестр** — слова оператора,
  файлы, точные значения, упавшие вызовы, последний план — одно сообщение,
  которое код пересобирает на каждом проходе и никогда не показывает
  суммаризатору. Реплика оператора никогда не суммаризуется, пары вызовов
  остаются целыми, а перенесённые из прежнего запуска реплики трогает только
  реактивное восстановление, и каждая замена сохраняет метку переноса.
  `compaction_completed` сообщает о каждом уровне и об `outcome` прохода; обе
  инструкции — шаблоны (`compaction_turn_summary`, `compaction_fold_summary`).
- `runtime/stale_result_trim.py` — сокращение промпта, которое не стоит ни
  одного вызова LLM. Включается константой `tool_result_stale_trim_enabled`
  (**по умолчанию выключено**) и переписывает только вид, уходящий в запрос:
  `engine.history` сохраняет каждый байт, поэтому персист, реплей и компакция
  видят нетронутый транскрипт. Результат инструмента длиннее
  `tool_result_stale_max_chars` обрезается до этой головы, когда прогон уже
  ушёл от него дальше; последние `tool_result_fresh_count` результатов и все
  результаты последнего раунда вызовов не трогаются независимо от размера —
  именно их модель читает прямо сейчас. Пока обрезаемый излишек не перешёл
  `tool_result_stale_trim_batch_chars`, не режется ничего: префикс промпта
  сдвигается ради пачки, а не ради одного результата. Обрезанный id липкий на
  весь прогон (`trimmed_tool_result_ids` в снапшоте), поэтому возобновлённый
  прогон собирает тот же префикс. Пины соблюдаются, пока более поздняя запись
  их не опровергла, а плейсхолдер компакции не переписывается никогда. Каждая
  строка отрезанной части, начинающаяся с одного из префиксов
  `tool_result_stale_trim_protected_prefixes` (по умолчанию семейство
  `Cite exactly:` / `Cite:` / `cite_as:` / `Source:`), переносится
  дословно — результат теряет тело, но не свою цитатную идентичность; в
  заменяющей строке сказано, сколько символов осталось и сколько убрано, чтобы
  обрезанный результат нельзя было принять за полное свидетельство. Проход
  идёт после чекпоинта компакции и перед проекцией разделения, а голову
  подбирает так, чтобы разделение не срезало его собственную строку-указатель.
- `runtime/loop_state.py` — `LoopState` — это чистая машина из 7 состояний:
  `PENDING → RUNNING → {AWAITING | COMPACTING} → {COMPLETED | FAILED |
  CANCELLED}`. `assert_transition()` обеспечивает таблицу легальных рёбер;
  `TERMINAL_STATES` не имеют исходящих рёбер. **Отлична** от
  `RunStatus` (долговременное зеркало PG-строки) и `RunState` (горячая запись в Redis-хеше)
  — `LoopState` — это in-flight-состояние экземпляра движка.

**Как вызывается/подключается.** Исполнитель хоста конструирует `QueryEngine` при
допуске рана, затем обычно `async for evt in engine.run(message)` на каждый ход
(или `async for evt in resume(engine, snapshot)`, когда он поднимает прогон).
Каждый `TurnEvent` пробрасывается в SSE-мост. Цикл — единственный
потребитель любой другой подсистемы.

**Конфигурируемость через RC.** `max_turns_per_run`, `agent_max_seconds` (дедлайн по
настенным часам; `<= 0` = инертен), таймауты idle/stall watchdog и каждый
переключатель восстановления — это поля RC. `model_name` обязателен (нет вшитого default).
Режим прогона не является полем снимка: его несёт `QueryEngineConfig.run_mode`
на каждый прогон, по умолчанию `"direct"`, так что хост, которому нужен
тенантный default, объявляет эту ручку в собственной группе констант и
передаёт разрешённое значение внутрь.

**Протокол расширения.** **Не** редактируйте структуру цикла. Кастомизируйте через (а)
регистрации на шве жизненного цикла, (б) политику хода, подставленную по имени
в `QueryEngine.turn_policies`, (в) инъецированные через `QueryEngineConfig`
колбэки и наблюдатели — `run_mode`, `tool_preconditions`, `provider_chain`,
`tool_roles`, `resilience_classifier`, `request_manifest_sink`,
(г) переключатели RC, (д) `system_prompt_sections`.

**Заметки о терминальной классификации.** Стоит выделить три поведения терминальной
классификации: (1) цикл перепроверяет `stop_requested` после стриминга и
маршрутизирует отменённый ран в CANCELLED (а не в чистый end-turn); (2)
`_synthesize_missing_tool_results` вызывается на каждой контрольной точке teardown, так что
персистированный снимок всегда валиден по парности (см. секцию *Починка парности tool_use /
tool_result*); (3) выход по `max_turns`
классифицируется как терминальное **исчерпание** ресурса — он сохраняет
`stop_reason=max_turns` на проводе и трактуется как класс ошибки/неуспеха,
а не как чистый `COMPLETED`.

### Политики хода — где живёт продуктовое решение о ходе

**Что и зачем.** Драйвер одного хода ассистента делает две разные работы. Одна —
**механика**: открыть поток, перевести дельты в события, диспетчеризовать вызовы,
о которых попросила модель, закрыть раунд. Вторая — **политика**: решить, что
прогон исчерпал бюджет, что пустой ответ заслуживает ещё одной попытки, что
недописанный файл надо запечатать прежде, чем прогону позволят закончиться.
Механика одинакова для любого прогона; политика — продуктовое мнение, и каждое
мнение, когда-либо добавленное в цикл, добавлялось выращиванием ветки внутри
него. Шов политик хода это прекращает: политика — объект, она объявляет
координаты, на которых её надо спрашивать, и отвечает событиями плюс одной
директивой.

**Ключевые классы/файлы.**

- `contracts/turn_policy.py` — `ITurnPolicy` (`name`, `coordinates`, на которых
  она регистрируется, и один `apply(turn)`, который выдаёт события для
  проброса и пишет дальнейшее в `turn.outcome`), `TurnContext`, `ITurnState`
  (намеренно узкий структурный вид прогона, который политике позволено читать и
  менять: политика, которой нужно что-то, не названное там, лезет во внутренности
  цикла, и ревью, добавляющее имя, — то самое место, где это будет замечено),
  `TurnFlags` (ход-локальное состояние, которое политики делят с циклом; раньше
  это были голые локальные переменные одной очень длинной функции),
  `TurnCoordinate` (`turn_start`, `turn_budget`, `empty_model_turn`,
  `output_truncated`, `stream_failed`, `turn_end`, `stream_settled`,
  `tool_calls_ready`, `finish_nudge`, `answer_floor`, `voluntary_finish`,
  `terminal_tool_finish`, `iteration_end`, `cancel_checkpoint`),
  `TurnDirective` (`proceed` / `restart_turn` / `end_turn`) и
  `TurnPolicyOutcome`.
- `runtime/turn_policies/` — по модулю на решение: `longfile.py`,
  `run_ceilings.py`, `empty_model_turn.py`, `truncated_tool_call.py`,
  `output_cap.py`, `terminal_nudge.py`, `answer_floor.py`,
  `empty_completion.py`, `terminal_tool_finish.py`, `compaction.py`,
  `repeat_guard.py`, `sibling_walk.py`, `provider_failure.py`,
  `cancellation.py`.
- `runtime/turn_policies/__init__.py` — `TurnPolicyRegistry` и
  `TURN_POLICY_ORDER`.

**Порядок принадлежит ядру.** `TURN_POLICY_ORDER` объявляет его один раз, и имя,
отсутствующее в этом кортеже, отвергается при построении
(`UnknownTurnPolicyError`), а не молча уезжает в конец. Порядок важен там, где две
политики встречаются: незапечатанный файл запечатывается **до** проверки, дал ли
ход ответ, потому что запечатывание ответ и производит, — а порядок, взятый из
того списка, который случайно собрал хост, сделал бы это совпадением. Реестр
опрашивает по порядку все политики, зарегистрированные на координате, и
останавливается на первой, ответившей чем-либо, кроме `proceed`: за политикой,
уведшей ход в другое место, никогда не идёт та, что считает, будто этого не
было. Шов, который не может исполнить директиву, говорит об этом
(`UnsupportedTurnDirectiveError`), а не роняет её молча.

**Как подключается.** `QueryEngine.turn_policies` равен `None` для собственного
набора ядра. Хост или тест, ставящий свой набор, присваивает его туда, на
прогон, и набор **сливается по имени**, а не подставляется вместо: политика
вытесняет политику ядра с тем же именем, а всякая граница, которую никто не
назвал, остаётся на месте. `runtime/error_kinds.py::INTERNAL_ERROR_KIND`
читается с обеих сторон этого шва — потому он и отдельный модуль: у «цикл
упал» не должно быть второго написания на стороне политики.

**Сбой провайдера: что прогон говорит, когда эндпоинт не отвечает.**
`provider_failure.py` ранжирует три восстановления — соседа по цепочке
провайдеров прогона, тот же эндпоинт после ограниченной паузы и ответ, который у
прогона уже есть, — и два правила держат последнее из них честным.

- **Повтор — это вердикт адаптера, прочитанный с исключения.** У каждого
  `LLMError` есть `retryable`. Умолчания класса говорят честную вещь о типе
  (`LLMRateLimitError`, `LLMTimeoutError`, `LLMStreamIdleError` и
  `LLMProviderError` повторяемы; `LLMContextWindowExceeded` — нет и такого
  ключевого аргумента не принимает), а адаптер, классифицировавший ответ,
  переопределяет его на каждом возбуждении:
  `LLMProviderError("no such model", retryable=False)` падает на первом же
  полученном ответе. Лестница ограничена
  `llm_transient_error_retry_max_attempts` (2), пауза
  `llm_transient_error_retry_backoff_base_seconds` (1.0) удваивается до
  `llm_transient_error_retry_backoff_max_seconds` (8.0), а названный сервером
  `Retry-After` имеет приоритет в пределах того же потолка. Серия сбрасывается
  на любом чистом стриме, так что предел — на серию подряд идущих сбоев, а не на
  прогон. Отменённый прогон, и прогон, которому бюджета по часам хватает лишь на
  финализацию, новой попытки не начинает; сама пауза ждёт событие остановки,
  поэтому отмена посреди неё замечается сразу. Каждая попытка — это WARNING,
  называющий прогон и номер попытки, и событие `state_changed`
  (`reason="transient_llm_error_retry"`), которое хост может показать.
- **Прогон, который ничего не произвёл, не просят писать отчёт.** Сворачивание
  просит у модели лучший ответ, который поддерживают собранные свидетельства; у
  прогона без текста, без вызова инструмента и без результата инструмента их нет,
  и, получив просьбу закрыться, он выдумывает прогон — оператор читает вежливый
  пересказ работы, которой не было, и ни следа сбоя. Поэтому сворачивание
  начинается, только когда `query.py::_run_produced_output` истинно, иначе прогон
  идёт в терминальный FAILED с собственной ошибкой провайдера. После того как
  появился результат инструмента, частичный итог И ЕСТЬ итог, и сворачивание —
  правильное закрытие. Уведомление тогда берётся по причине
  (`soft_stop_notice_text_provider_error`), потому что общее говорит, что прогон
  достиг своего бюджета, а модель читает это буквально; а его события
  `state_changed` несут `soft_stop_detail` — собственное сообщение вышестоящего,
  — чтобы хост мог показать оператору, почему прогон закончился, а не
  восстанавливать это из лога.

### Снимок прогона: версия схемы и upcaster-ы

**Что и зачем.** Снимок пишет один процесс, а читает другой, и они не обязаны
быть одной сборкой. Читатель, который молча принимает непонятный ему payload, не
падает — он поднимает прогон с потерянными полями, сброшенными защёлками и
заново налитыми бюджетами, и ничто ниже по течению не отличит это от прогона, у
которого их честно не было. Отказ всплывает гораздо позже — агентом, который
переделывает уже сделанное или тратит уже потраченное.

**Ключевые имена (`contracts/snapshot.py`).** `SNAPSHOT_SCHEMA_KEY` — где в
payload лежит версия; `SNAPSHOT_SCHEMA_VERSION` — то, что пишет эта сборка.
Payload вообще без поля версии читается как версия 1: версия 1 — ровно та форма,
поверх которой поле и появилось. Всё, чего сборка не узнаёт, отвергается через
`SnapshotSchemaError`, и отказ здесь — восстановимый исход: прогон остаётся там,
где был, и оператор видит почему.

**Upcaster-ы.** Более старый payload не отвергается там, где его можно поднять
вперёд: по одному зарегистрированному `SnapshotUpcaster` на версию, каждый читает
форму на версию ниже и дописывает то, что эта версия принесла. Версия без шага —
отказ, потому что пропуск оставляет её поля незаполненными: то же тихое
полу-восстановление. Два ключа payload названы модулем, а не переписаны у каждого
читателя: `RUN_SCOPED_STATE_SNAPSHOT_KEY` (накопленный журнал работы дерева и
ёмкость его бюджета параллелизма — два разрешения, которые поднятому прогону
нельзя выдать второй раз) и `PENDING_INTERRUPTS_SNAPSHOT_KEY`.

**RunScopedState.** `contracts/run_state.py` держит межвызовные разрешения
прогона — счётчики серий, которые ведёт диспетчер, журнал работы дерева, событие
отмены, локи, удовлетворённые предусловия — одним типизированным объектом вместо
нетипизированного словаря, протянутого через `ToolContext.metadata` под
согласованной строкой. Поле, которое переехало, — это ошибка типов у читателя; хост,
подключающий собственные ячейки, держит их в `RunScopedState.host`, одном
непрозрачном отсеке, так что ядру не нужно знать, что хост туда кладёт.
`RunScopedState.to_snapshot()` называет ровно два долговечных разрешения, а
`apply_snapshot()` возвращает их обратно; живые объекты (`asyncio.Event`,
семафор, лок) по природе процесс-локальны и пересобираются тем, кто подключает
поднятый прогон. `ToolContext` сужен под это: `tenant_id`, `run_id`,
`session_id`, `work_scope`, опциональный `evidence`, опциональный `run_state` и
`metadata` на всё остальное.

### Диспетчеризация: роли, канонический результат и починка парности

**Роли, а не имена.** Рантайму приходится знать ВИД вызова — произвёл ли он
байты на диске, закрывает ли он обязательство перечитать файл, должен ли гейт
разрешений провести для него проверку безопасности шелла. Раньше это было
сравнение с ИМЕНЕМ инструмента, написанным внутри ядра, — то есть допущение,
что всякая установка называет свои инструменты так же, как первая.
`contracts/tool_roles.py` это заменяет: `ToolRole` — способность
(`reads_path`, `writes_path`, `appends_path`, `edits_path`, `finalizes_path`,
`searches_workspace`, `runs_shell`, `fetches_url`, `delegates_work`,
`records_plan`, `discovers_tools`, `asks_user`, `never_delegated`), а
`ToolRoleMap` — приходящая как `QueryEngineConfig.tool_roles` декларация хоста
о том, какие из ЕГО имён какие роли несут, вместе с написаниями аргументов
(`ToolArgumentSlot`): в каком ключе лежит команда шелла, в каком — тело записи,
в каком — ответ терминального инструмента. Роль, которой карта не упоминает, —
способность, которой у этой установки нет, и функция, которой она нужна,
говорит об этом предупреждением, а не тихо становится инертной.
`runtime/child_capabilities.py::narrow_child_capabilities` читает ту же карту,
чтобы вычислить, что позволено делегированному прогону: только сужение,
никогда расширение, и применяется дважды — когда разрешается каталог ребёнка и
на каждом его вызове, — чтобы объявленная поверхность и гейт не могли
разойтись.

**Одно значение, три адресата.** `ToolResult.content` — каноническое значение,
полное, каким бы большим оно ни было, а проекции стоят рядом, а не вместо него.
`model_projection` — то, что транскрипт несёт вместо содержимого, когда целиком
оно там неуместно (`model_content` — свойство, из которого строится каждый
`ToolResultBlock`, так что инструмент, не назвавший проекцию, ничего не
теряет); `ui_payload` едет на событии результата и в транскрипт не попадает
вовсе, поэтому целая отрисованная таблица не стоит токенов и не может изменить
решение модели; `canonical_ref` называет блоб, из которого значение можно
достать целиком, когда транскрипт его больше не держит. Инструмент, который
обслуживает всех троих одной строкой, — причина, по которой усечение
транскрипта раньше уничтожало свидетельства: усекать было нечего, кроме
единственной копии.

**Durable-намерение до вызова.** Каждый диспетчеризуемый вызов коммитит
`IntentRecord` (`runtime/intent.py`) ДО того, как инструмент тронут, с
зарезервированными id результата — см.
[Intent, usage ledger, …](#intent-usage-ledger-session-tree-lanes-typed-hooks-telemetry-live-control-run-work-budget).

**Починка парности (`runtime/query.py`).** Anthropic / OpenAI / vLLM все отклоняют запрос, чей ассистентский
`tool_use` не имеет парного `tool_result` (или осиротевший `tool_result`, или
дублирующиеся id) с HTTP 400. Парность должна гарантироваться на wire-границе
как defense-in-depth — а не предполагаться корректной от вышестоящих мутаторов (компакция,
resume-from-partial-batch, усечение по max_tokens, teardown).

**Ключевые функции (`runtime/query.py`).**

- `_repair_outbound_tool_pairing(messages, placeholder)` — **чистый**,
  безусловный backstop на wire-границе, выполняемый над списком исходящих сообщений прямо
  перед сборкой `LLMRequest` (до вычисления cache-breakpoint, так что
  индексы адресуют финальный список). Четыре починки: forward-fill синтетических
  `is_error`-результатов для осиротевших `tool_use`, репозиционирование каждого реального результата
  прямо после его `tool_use`, reverse-strip осиротевших результатов, дедупликация
  дублирующихся id.
- `_synthesize_missing_tool_results(history, error_content)` — мутирует историю на
  месте на **каждой контрольной точке teardown** (stop-before-start, compaction-failed,
  hook-deny, stop-after-stream, dispatch-cancel, LLM-error terminal), так что
  персистированный снимок остаётся валидным по парности и упорядоченным для resume на другом
  поде. Идемпотентна.

**Как подключается.** `_repair_outbound_tool_pairing` выполняется на каждом исходящем запросе;
`_synthesize_missing_tool_results` выполняется на всех путях аномального выхода.

**Шаблоны промптов.** `tool_result_pairing_repair`, `tool_result_interrupted`.

### Skills routing / surfacing

**Что и зачем.** Выносит маленький **каталог** доступных скиллов в
системный промпт каждый ход и загружает полное тело скилла по требованию, когда пользователь
ссылается на него — так доменная возможность добавляется как данные, а не как по-задачные
подсказки в промпте. Каталог — это компактный, отсортированный по алфавиту блок
`<system-reminder>`, собираемый один раз за ран (кэшируется на
`engine._skill_catalog_block`) и помещаемый в статический префикс
промпта, так что он остаётся байт-стабильным между ходами (сохраняя кэш промпта).
Это **не** BM25- и не top-K-ранжирование.

**Ключевые классы/файлы.** `runtime/skill_index.py` — `render_skills_catalog`
выдаёт `SYSTEM_REMINDER_HEADER` («Skills are tools, not files… call exactly
`Skill(skill="<name>")`») плюс одну строку
`Skill(skill="{name}") — {description}` на каждый включённый скилл, по
алфавиту имён. Сверх токен-бюджета блок деградирует до одних форм вызова
(`Skill(skill="{name}")`). `derive_skill_index_budget_tokens` — это
`model_context_window × skill_index_budget_ratio` (по умолчанию 1%).
`contracts/skills.py` — `ISkillStore`, `SkillIndexEntry`, `SkillBundle`,
`SkillFileRef`, `SKILL_ENTRY_PATH` (`SKILL.md`). `list_files` / `load_file` —
обязательные методы протокола для многофайловых бандлов (как минимум
каноническая строка `SKILL.md`; устаревший однофайловый скилл может
синтезировать её из `body_md`). **Цикл ядра никогда не вызывает**
`list_files` / `load_file` — каталог строится через `list` +
`list_enabled_subset`, тело по триггеру — через `load` / `list_subset`.
Хосты, которые отдают вспомогательные файлы, сами пользуются файловым API.

**Как подключается.** Шаг 4 вызывает `_ensure_run_skill_catalog(engine)`.
Чтения skill-store ключуются по `QueryEngineConfig.account_id` (банк на
аккаунт), **не** по `tenant_id`. Когда `engine.skills is None` или store
пуст → пустой блок нулевой стоимости. Сбои изолируются с WARNING; ран
продолжается. На каждый ход `<command-name>NAME</command-name>` в последнем
пользовательском тексте загружает совпавший `SkillBundle.body` как блок
Layer-3, с потолком `max_skills_per_run` (по умолчанию 4). Пины проекта
(`pinned_skill_names`) подмешиваются через `list_enabled_subset`, так что
выключенный скилл не попадает в каталог.

**RC/расширение.** Surfacing управляется данными (пустой store = нет блока).
`skills_hot_reload_enabled` (по умолчанию `False`) пропускает кэш на ран и
пересобирает каталог на каждый вызов `_ensure_run_skill_catalog`. Реализуйте
`ISkillStore`; ranker реализовывать не нужно.

### Шов жизненного цикла + injection / scratchpad + context_bootstrap

**Что и зачем.** Один шов расширения: наблюдать, решать, преобразовывать или
оборачивать поведение в любой координате прогона, не трогая цикл. Плюс
необязательный **context bootstrap** первого хода, который читает
контракт/readme окружения и добавляет замороженное ориентирующее сообщение
`<environment_context>`.

**Ключевые классы/файлы.** `contracts/middleware.py` — контракт:
`RegistrationKind` (`observe` / `decide` / `transform` / `around` / `notify`),
`LifecycleVerdict`, `LifecycleContext`, `LifecycleDecision`,
`LifecycleOutcome`, `LifecycleScope`, `LifecycleDisposer` и протокол
`ILifecycleRegistry`. `contracts/types.py::HookEvent` — единственный список
координат. `hooks/manager.py` — `HookManager`, внутрипроцессная реализация:
порядок (приоритет, затем порядок регистрации), таймаут на регистрацию,
политика исключений и отмена, которая никогда не читается как вердикт.
`contracts/hooks.py` — внепроцессный контракт `IHookManager`, `HookResult`,
`HookActionKind`, `HookSpec`. `runtime/correctness_bind.py::fire_lifecycle` —
единственный вход цикла в шов.

**Как подключено.** Реестр приходит в движок при конструировании
(`QueryEngine(..., lifecycle_hooks=...)`) и работает при включённом
`typed_hooks_enabled` — один тумблер на весь шов, ни одна координата не спрятана
за чужим. `_drive_turn` и диспетчер инструментов испускают `run_start`,
`turn_start`, `context_transform`, `request_prepare`, `response_received`,
`request_error`, `turn_end`, `pre_tool_use`, `tool_execute`, `post_tool_use`,
`pre_compact`,
`compaction_commit`, `compaction_rollback`, `post_compact` и `run_finalize`.
`transform` на `context_transform` **применяется**: запрос к провайдеру этого
хода пересобирается из того, что вернула цепочка.

`IHookManager` хоста — тот же шов, до которого дотягиваются из другого процесса
(HTTP-эндпойнт или модель-судья). Цикл ведёт его на гейте разрешений
(`runtime/tool_permission.py`) и вокруг диспетча (`runtime/tool_dispatch.py`),
отображая `HookActionKind` на те же вердикты.

**Политика исключений и почему она разная.** `decide`, `transform` и `around`
падают **закрыто**: обработчик, который бросил, вышел за `timeout_s` или
ответил не решением, даёт `deny` с именем владельца. Стадия, существующая
чтобы сказать, можно ли, при аварии не сказала «да». `observe` и `notify`
падают **изолированно**: отказ логируется и попадает в
`LifecycleOutcome.failures`, а вердикт, полезная нагрузка и соседние
регистрации остаются нетронутыми.

**RC/расширение.** `typed_hooks_enabled` (по умолчанию `False`). Режим отказа и
дедлайн judge-хука, настройки context bootstrap читает окружающий слой и
объявляет у себя. Расширяют регистрацией на шве с владельцем, скоупом и
диспозером либо реализацией `IHookManager` в хосте.

### Events / observability / streaming

**Что и зачем.** Стриминг обязателен — каждая дельта провайдера становится типизированным
`TurnEvent`, проброшенным немедленно. Две различные поверхности событий:

- `events.py` — `EventBus` + `EventName` (~70 имён): **in-process** типизированный
  pub/sub для сигналинга между sibling-обработчиками внутри пода (используется HookManager,
  ContextManager, …). Отличен от межподового `IEventStream` (Redis
  Streams), используемого для SSE reconnect/replay.
- `runtime/events/types.py` — `EventType`: **по-ходовая потоковая** таксономия
  (выровнена под Anthropic: `message_*`, `content_block_*`, `tool_use_*`,
  `tool_result`, `error`, плюс расширения Protocore
  `sandbox_*`/`subagent_*`/`hook_fired`/`tool_call_pending`/`state_changed` и
  жизненный цикл цикла `run_started`/`heartbeat`/`compaction_*`). Поздние
  добавления включают `reasoning_step` (план Deep-режима), `intent_committed`,
  `usage_committed`, `session_forked`, `lane_locked`, `recovery_marked`,
  `compact_checkpoint`, события steer/follow-up/очередей (`steer_queued`,
  `follow_up_queued`, `queue_update`), live-control
  `model_changed`/`thinking_changed` и события верификации кандидата
  (`candidate_ready`, `verification_started`, `verification_reported`,
  `repair_requested`, `release_decided`, `candidate_released`) и
  `interrupt_parked` — оно выдаётся всякий раз, когда прогон записывает то, чего
  ждёт, и несёт id прерывания, его вид и вызов инструмента: хост узнаёт, НА ЧЁМ
  прогон остановился, в момент остановки, а не вычитывая снапшот. Каждое
  значение — строка `event:`, показываемая SSE-клиентам.
  События транспорта инструментов (`tool_transport_starting`,
  `tool_transport_ready`, `tool_transport_failed`, `tool_transport_teardown`)
  сообщают, что путь наружу к инструменту поднимается, готов, отказал и
  сворачивается; каждое несёт один и тот же ключ payload, называющий, о каком
  транспорте речь, так что хост соотносит их без разбора текста.
- `runtime/events/envelope.py` — `TurnEvent` (замороженный wire-envelope).
- `runtime/llm/delta_bridge.py` — переводит поток провайдера в
  `ProviderDelta` → `TurnEvent` (`_normalise_finish_reason`, `is_block_end`,
  …).
- `contracts/observability.py` — два опциональных стока.
  `CacheObserverProtocol` — сток hit-rate prompt-cache, инъецируемый через
  `QueryEngineConfig.cache_observer`. `IRequestManifestSink` отвечает на другой
  вопрос — не «как отработал этот вызов», а «что именно было отправлено».
  Запрос к провайдеру собирается из истории, контрольной точки компакции,
  починки парности, поверхности инструментов и действующих констант, и до
  появления `RequestManifest` он существовал только в кадре стека, который его
  и отправил: ничто долговечное не могло сказать, странно ли себя вёл прогон
  из-за другого запроса или из-за другого ответа на тот же, и ничто не могло
  переиграть записанный прогон, не заплатив за токены снова. Ядро строит
  манифест, считает его id — SHA-256 по его же канонической сериализации, так
  что id известен до того, как хост что-либо записал, и снимок может
  адресовать манифест по id, а не носить его целиком, — и отдаёт приёмнику.
  Где он хранится и как долго — решение хоста: у ядра нет ни хранилища, ни
  политики хранения, и заводить их посреди прогона — ровно то обязательство,
  которого цикл на себя брать не должен.

**Как подключается.** Цикл выдаёт `TurnEvent`-ы повсюду; usage-дельта питает
cache observer, а `build_llm_request` питает приёмник манифестов, когда
привязан `QueryEngineConfig.request_manifest_sink`. Стоки
трейсинга/observability инъецируются через границу.

### Safety (shell policy + chain parser + path isolation + approvals)

**Что и зачем.** Валидировать составленные моделью shell-команды перед исполнением, с
паттернами deny/approval на основе capability, и изолировать пути workspace.

**Ключевые классы/файлы.**

- `safety/shell.py` — `DefaultShellSafetyPolicy` + `_DENY_PATTERNS`
  (деструктивный `rm -rf /`, SUID, base64/dd, ANSI-C `$'...'` и locale `$"..."`
  кавычки, `$IFS`/`${...IFS}` инъекция через word-split, …). Возвращает
  `ShellPolicyDecision` (allow / deny / require-approval).
- `runtime/chain_parser.py` — `parse_chain(...)`: маленькая shell-грамматика, которая
  разбивает команду на `;`/`|`/`&&` на `CommandSegment`-ы и показывает `$()` /
  backtick **тела подстановок** (собираются даже внутри двойных кавычек, не
  одинарных), так что по-сегментные deny-паттерны перевзводятся на телах подстановок.
- Path-isolation + approval-политики живут в `tool_permission.py`
  (`WorkspacePathPolicy`), а approval-поток — это стадия `require_approval` гейта
  (цикл → AWAITING → resume).

> Заметка: `DefaultShellSafetyPolicy` **fails open** на несовпадении (нет
> fail-closed/ambiguous-эскалации), а `HttpDnsAllowlistPolicy` /
> `WorkspacePathPolicy` не в стеке по умолчанию — хост должен зарегистрировать
> их через `register_policy`.

### LoopConstants system

**Что и зачем.** Единственный механизм для настраиваемых значений — **никаких inline
magic numbers**. Каждое настраиваемое значение — это поле на замороженном Pydantic-снимке, безопасное
по умолчанию и конфигурируемое из дашборда.

**Ключевые классы/файлы.**

- `contracts/runtime_constants.py` — `LoopConstants`
  (`model_config = ConfigDict(frozen=True, extra="forbid")`) и
  Protocol `RuntimeConstantsProvider` (`async get(tenant_id) -> LoopConstants`).
  `extra="forbid"` означает, что **core и хост должны деплоиться парно** (неизвестное
  поле отклоняется). Снимок включает выключенные по умолчанию поверхности
  `intent_settlement_enabled`, `usage_ledger_enabled`, `lanes_enabled`,
  `typed_hooks_enabled`, `telemetry_spans_enabled` (и
  `compaction_manual_enabled`, `steer_follow_up_enabled`).
- `runtime/runtime_constants.py` — `StaticRuntimeConstantsProvider` +
  `default_runtime_constants(**overrides)` (тесты + in-memory smoke-рантайм;
  продакшен-поды поставляют Postgres-backed провайдер с Redis-кэшем).
- `constants.py` (~70 строк) — лимиты безопасности по памяти на уровне модуля (`MAX_ARTIFACTS`,
  `MAX_TOOL_CALL_ARGUMENT_BYTES`, `PROTOCOL_VERSION`, `DEFAULT_MODEL`, …).

- `contracts/config.py` — реестр, одной из групп которого снимок и является.
  `ConstantSpec` — дескриптор одной ручки: её вид на проводе (`kind`),
  `default`, обращённое к оператору `description`, границы (`minimum` /
  `maximum`, `allowed_values`, `zero_means_unlimited`) и видимость —
  `editable`, `editable=False` (строка показывается, но запись отвергается)
  либо `not_a_lever="<причина>"` (строки нет вовсе: значение, которое система
  выводит или которым владеет сама, и появление которого в редакторе было бы
  приглашением сломать инсталляцию). `ConstantGroup` — набор спецификаций с
  одним `owner` и одним `key`; `group_from_model` отражает объявляющую модель в
  группу, чтобы никто не писал руками список из сотен имён, расходящийся с
  моделью на первом же добавленном поле, а `build_loop_group` делает это для
  самого `LoopConstants`. `IConstantsRegistry` — сторона объявления:
  `declare`, fail-closed `resolve` (имя, которого не объявила ни одна группа,
  бросает исключение, а не разрешается в default), `defaults`, `coerce` и
  `repair`. `ICoreConstantsProvider` — то, как цикл спрашивает снимок,
  действующий для области.

**Настраиваемая поверхность — набор групп, а не одна плоская модель.** Каждую
группу объявляет тот слой, который её значения действительно читает: пороги
самого цикла — группа ядра, а всякая ручка, которую читает окружающий слой,
объявляется этим слоем, в его собственной модели, через то же отражение
`group_from_model`. Имя, заявленное двумя **владеющими** группами, отвергается
(`DuplicateConstantError`); имя, заявленное владеющей группой и
`provisional`-группой — заглушкой, которую слой держит, пока передаёт владение,
— достаётся владельцу, и вытеснение фиксируется. Поэтому ручка,
управляющая аутентификацией, хранением сессий или транспортом к провайдеру,
**не** является полем `LoopConstants`, и искать её там бесполезно.

**Добавление настраиваемого значения.** Добавьте поле в ту модель, чей слой его
читает, с безопасным default-ом, границами и `description`. Больше ядру о нём
знать нечего: группа отражается из модели, и каталог оператора подхватывает
поле оттуда.

### Пул работы сессии и делегированные прогоны

**Что и зачем.** Один пул, два вида работы. Команда, запущенная в фоне, и
дочерний прогон, запущенный делегированием, со стороны цикла — одно и то же:
единица работы с адресом, статусом, способом её дождаться и способом её
остановить. Раньше это были два механизма — команда была записью пула с id, а
дочерний прогон был вызовом функции, который блокировал вызывающего на всё своё
время и адреса не имел вовсе. Никто не мог спросить дочерний прогон, как далеко
он продвинулся, никто не мог его остановить, а родитель, ждавший его, держал
свой ход и своё место в бюджете дерева весь потомковый прогон.

**Ключевые имена (`contracts/background.py`).** `IWorkPool` расширяет
`IBackgroundTaskPool`; `TaskRecord.kind` различает виды (`command` / `agent`), и
ручка субагента — это просто `WorkHandle` над записью вида `agent`. `AgentRef`
говорит, какого агента запись вида `agent` исполняет и каким прогоном.
`BACKGROUND_TERMINAL_STATUSES` — множество статусов, из которых запись больше
никогда ничего не сообщит.

**Почему это не состояние прогона.** Фоновая задача переживает прогон, который её
запустил: команду породил один прогон, прогон закончился, и сказать об окончании
надо тому прогону, который жив в этот момент. Поэтому пул — коллаборатор,
инъецируемый хостом, и на холодном старте (свежий процесс поднимает сессию,
чьи задачи породил уже исчезнувший процесс) хост обязан вернуть ещё работающие
команды сессии в руки нового пула прежде, чем цикл его о чём-нибудь спросит.
`ensure_session_attached` — то место, где цикл спрашивает, случилось ли это;
пул, ответивший `False`, получает явное событие на прогоне, а не пустой список
пробуждений, который читается ровно как сессия, в которой ничего не работает.

**Сужение ребёнка.** `SubagentDef` — определение ребёнка, а
`runtime/child_capabilities.py::narrow_child_capabilities` вычисляет, что ему
позволено, из его родителя и ни из чего больше: ни инструмента, ни разрешения,
ни шага глубины больше, чем было у родителя, — прогон, способный расшириться по
пути вниз, сделал бы всякую границу выше себя рекомендательной.

### Intent, usage ledger, session tree, lanes, typed hooks, telemetry, live control, run work budget

Модули, которые сидят рядом с общим ReAct-циклом. Каждый выключен по умолчанию,
пока соответствующее поле RC не скажет иное.

- `runtime/intent.py` — `IntentRecord` / `commit_intent` / `settle_intent` /
  `orphaned_intents` / `unknown_outcome_text` / `replay_policy_for` /
  `repeat_is_safe_for`. **Каждый** диспетчеризуемый вызов инструмента коммитит
  `IntentRecord` с зарезервированными result id до того, как инструмент
  тронут, безусловно; запись несёт жизненный цикл `state`
  (`RESERVED|PENDING_APPROVAL|DISPATCHED|PAUSED_ASK_USER|SETTLED`), а преамбула
  хода закрывает то, что оставил остановившийся прогон. Краш mid-flight
  сообщается как исход, который не был зафиксирован, — никогда как отказ,
  потому что отказ приглашает повторить побочный эффект, который, возможно, уже
  случился, — а вызов на гейте, вызов в ожидании ответа пользователя и просто
  зарезервированный не сообщаются вовсе: ни один из них не исполнялся.
  `intent_repeat_safe_tools` (по умолчанию `Read,Grep,Glob,ToolSearch`)
  определяет, какие вызовы не платят за запись долговечности и получают мягкий
  текст; `intent_never_replay_tools` (по умолчанию
  `Write,Edit,Bash,Finalize,AppendFile`) задаёт `replay`.
  `intent_settlement_enabled` гейтит только события восстановления и строку
  журнала. Поле снимка: `open_intents`.
- `runtime/usage_ledger.py` — append-only список `UsageRow`. Когда
  `usage_ledger_enabled` включён, `correctness_bind.commit_usage` дописывает
  строку. `_query_raw` пишет `inference` / `retry` / `compaction` /
  `abort` / `fail` независимо от intent. Строка вида **tool** пишется только
  на пути intent-settlement (тот же блок `if intent_settlement_enabled`,
  который settle-ит намерение). Неудачная попытка плюс её retry — две
  строки. Поле снимка: `usage_rows`.
- Ветвление сессии (на стороне хоста) копирует путь истории в новую ветку,
  не мутируя источник. Гейтится настройкой хоста; clone требует
  settled-источник, а вторая настройка хоста ограничивает объём копии.
- `runtime/lanes.py` — именованные lanes над общей историей. Main всегда
  существует; дополнительные берут эксклюзивные блокировки (`acquire_lane` /
  `release_lane`). Гейтится `lanes_enabled`; `lanes_max_per_session` включает
  main.
- `contracts/middleware.py` + `hooks/manager.py` — шов жизненного цикла и его
  внутрипроцессный диспетчер. См.
  [шов жизненного цикла](#шов-жизненного-цикла--injection--scratchpad--context_bootstrap).
- `runtime/telemetry.py` — spans низкой кардинальности (`run` / `turn` / `step` /
  `tool` / `compact` / `hook`). Гейтится `telemetry_spans_enabled`.
  Высококардинальные id остаются атрибутами; `is_prometheus_safe_label`
  отказывается принимать `session_id` / `lane_id` / `operation_id` / `run_id`
  как ключи меток. `mark_recovery` помечает span, когда возобновляется
  прерванное намерение.
- `runtime/correctness_bind.py` — клей, чтобы intent, ledger, типизированные
  хуки и recovery выполнялись внутри `_query_raw` (`commit_usage`,
  `fire_lifecycle`, `mark_intent_recovery`, `persist_correctness`).
- `runtime/live_control.py` — очереди steer / follow-up (`QueuedPrompt`,
  `enqueue`, `place_items`), живые переопределения model/thinking и settled-
  помощник. Гейтится `steer_follow_up_enabled` (по умолчанию `False`).
- `runtime/run_work_budget.py` — кумулятивный бюджет жизни дерева
  (`max_subagent_runs_per_tree`, `max_total_tokens_per_tree`) для одного
  корневого рана и всего, что он порождает. Сиблинг `SubagentTreeBudget`
  (permits), а не его расширение. Исчерпание отказывает в **делегировании**,
  никогда в самом ране.

---

## Точки расширения (протоколы, которые реализует хост)

Ядро — это набор `Protocol`-ов; хост предоставляет конкретные адаптеры.
Полная поверхность интерфейсов живёт в `contracts/` (каждый в своём модуле — единого
`protocols.py` **нет**; старый монолитный файл был разбит по доменам).
Основные точки расширения:

| Protocol | Module | What the host provides |
|---|---|---|
| `ILLMProvider` | `contracts/llm.py` | LLM completions: `stream_with_tools`, `complete_structured`, `complete_text` и `count_tokens`; универсальный LiteLLM/OpenAI-совместимый адаптер (OpenRouter / vLLM / OpenAI). |
| `IProviderChain` | `contracts/llm.py` | Упорядоченные оставшиеся провайдеры плюс односторонний курсор `advance()`. Внедряется на `QueryEngine(..., provider_chain=...)` для mid-stream failover; `None` оставляет существующее восстановление нетронутым. |
| `RuntimeConstantsProvider` | `contracts/runtime_constants.py` | Per-tenant `LoopConstants` backed by Postgres + Redis cache. |
| `ISessionStore` | `contracts/session.py` | Session/transcript persistence (Postgres). |
| `IRunStore` | `contracts/run.py` | Run record create/list/read (Postgres + Redis hot record). |
| `IToolRegistry` | `contracts/tool_registry.py` | The concrete `ToolRegistry` is in core; the host registers concrete `Tool`s + visibility policy. |
| `Tool` (ABC) / `@tool` | `contracts/tools.py`, `tools/decorator.py` | Concrete tool implementations (sandbox-backed exec/file tools). |
| `IToolTransport` | `contracts/resilience.py` | The tool/VM transport (e.g. ConnectRPC) the resilience wrapper wraps; optional `rebuild()` hook. |
| `IMemory` | `contracts/memory.py` | Долговечное хранилище фактов со скоупами и лексическим recall-ом, двухфазной идемпотентной записью и drift-guard, плюс `IMemoryContentScanner`. |
| `IWorkspace` | `contracts/workspace.py` | Durable byte store + Postgres FTS/BM25 manifest, atomic write, per-scope GC. |
| `ISkillStore` | `contracts/skills.py` | Хранение и поиск skill-бандлов **и** многофайловые `list_files` / `load_file` (`SkillFileRef`). Цикл каталогизирует через `list` / `list_enabled_subset` и грузит тела через `load` / `list_subset`; `list_files` / `load_file` — файловый API хоста. Рендерер каталога живёт в ядре, `runtime/skill_index.py`. |
| `IHookManager` | `contracts/hooks.py` | 3-аргументный диспетчер хуков, чей исполнитель вне процесса; цикл ведёт его на гейте разрешений и вокруг диспетча инструмента. |
| `ILifecycleRegistry` | `contracts/middleware.py` | Единственный шов жизненного цикла — у регистрации есть владелец, скоуп и идемпотентный диспозер. По умолчанию выключен тумблером `typed_hooks_enabled`. |
| `IEventStream` | `contracts/events.py` | Cross-pod durable event stream (Redis Streams) for SSE reconnect/replay. |
| `IBlobStore` | `contracts/blob.py` | Content-addressed blob storage (S3) used by Tier-1 compaction. |
| `ISearchIndex` | `contracts/search.py` | Generic lexical search index. |
| `ITodoStorage` | `contracts/todo.py` | Per-session todo persistence. |
| `IAgentDispatch` | `contracts/agent_dispatch.py` | Subagent dispatch/lookup. |
| `IPromptTemplateProvider` | `contracts/prompts.py` | System-prompt template rendering. |
| `IWorkPool` / `IBackgroundTaskPool` | `contracts/background.py` | Пул работы сессии: фоновые команды и делегированные дочерние прогоны в одном адресном пространстве, с `ensure_session_attached` на холодном старте. |
| `IConstantsRegistry` / `ICoreConstantsProvider` | `contracts/config.py` | Объявление собственных групп констант хоста, fail-closed разрешение имён и снимок `LoopConstants` на область, который читает цикл. |
| `IResilienceClassifier` | `contracts/resilience.py` | Вердикт о том, к какому нейтральному классу отказа относится сообщение. Ядро читает текст отказа, который не оно писало, и не должно учиться его узнавать; непривязанный классификатор означает, что ни одна формулировка не распознаётся, — и это нейтральное поведение. |
| `IRequestManifestSink` | `contracts/observability.py` | Где хранится `RequestManifest` и как долго. |
| `IDelegationTool` | `contracts/agent_dispatch.py` | Форма инструмента, через который запускается делегированный прогон. |
| `IRunToolErrorCounter` | `contracts/run.py` | Долговечный счёт ошибок инструментов на прогон, переживающий смену процесса. |
| `ITurnPolicy` | `contracts/turn_policy.py` | Продуктовое решение о ходе, подставляемое в набор ядра по имени. |
| `IToolSafetyPolicy` | `runtime/tool_permission.py` | Extra permission policies (`HttpDnsAllowlistPolicy`, `WorkspacePathPolicy`) registered via `register_policy`. |
| `ToolRoleMap` | `contracts/tool_roles.py` | Какие имена инструментов хоста несут какие `ToolRole`, и написания аргументов, которые к ним прилагаются. Передаётся как `QueryEngineConfig.tool_roles`. |
| Координаты жизненного цикла | `contracts/types.py::HookEvent` | Единственный список точек, которые может назвать регистрация. |
| `CacheObserverProtocol` | `contracts/observability.py` | Prompt-cache hit-rate sink injected via `QueryEngineConfig.cache_observer`. |
| Self-verify trigger callables | `runtime/query_engine.py` | `pre_terminal_self_verify_trigger` / `pre_dispatch_terminal_verify_trigger` on `QueryEngineConfig`. |

---

## Соглашения

- **Граница импортов.** Ядро никогда не импортирует пакет, стоящий над ним
  (`protocore_*`). Добавляйте
  поведение через контракты / адаптеры / RC, а не импортом вверх. Страж:
  `tests/test_core_import_boundary.py`.
- **Никаких inline magic numbers.** Каждое настраиваемое значение, которое читает
  цикл, — это поле `LoopConstants` (frozen, `extra="forbid"`) или лимит из
  `constants.py`. Код рантайма читает из снимка RC, а не из зашитого литерала.
  Ручку, которую читает окружающий слой, объявляет этот слой, в собственной
  группе констант (`contracts/config.py`).
- **Безопасно при горизонтальном масштабировании.** Никаких словарей на уровне модуля, никаких удерживаемых модулем блокировок, никакой
  in-memory-авторитетности на уровне пода. Состояние, влияющее на корректность, живёт пер-ран на
  `QueryEngine` (snapshot/resume); эфемерное межподовое состояние — это Redis, долговременное
  состояние — это Postgres (оба инъецируются через границу). Token-bucket и
  adaptive band принимают инъецированные блокировки/store-ы, а не состояние модуля.
- **Стриминг обязателен.** Любой драйв — асинхронный итератор; каждая дельта
  провайдера пробрасывается немедленно как `TurnEvent`. Не буферизуйте целый ход
  перед выпуском.
- **Никакой обратной совместимости.** Dev-версия проекта — ломайте свободно, удаляйте
  мёртвый код, никаких migration shim. (Поэтому `compaction_thresholds.py` был удалён
  начисто, как только `budgets.py` поглотил его.) Единственное исключение — снимок
  прогона: он пересекает не релизы, а сборки, поэтому объявляет версию схемы и либо
  поднимается upcaster-ом, либо отвергается — но никогда не читается наполовину
  (`contracts/snapshot.py`).
- **Используйте модели `Message`, а не сырые dict-ы.** Все сообщения текут как Pydantic-
  объединение `Message` / `ContentBlock` из `contracts/types.py`.
- **Не модифицируйте структуру цикла.** Кастомизируйте через регистрацию на шве
  жизненного цикла, политику хода, инъекцию `QueryEngineConfig`, переключатели RC
  или `system_prompt_sections`.
- **Хост доказывает свои адаптеры, а не ждёт, пока это сделает прогон.**
  В `protocore.conformance.SUITES` по одному набору на контракт; хост привязывает
  каждый к своему адаптеру и узнаёт в собственном прогоне тестов, что форма — та
  самая, которую вызовет ядро. Ставится как `pip install "protocore[testing]"`.
- **Продакшен-логирование = WARNING.** Используйте `logger.warning(...)` для
  операционно значимых событий; более низкие уровни оставьте для локальной
  отладки.

Команды репозитория: `uv sync --extra dev`, `uv run pytest .`, `uv run ruff check .`,
`uv run mypy --strict` (никогда с путём: путь заменяет настроенный список файлов
и молча выбрасывает дерево тестов).

> Перевод английского оригинала `docs/architecture.md` (коммит `17aedbe`). При изменении оригинала обновите перевод.
