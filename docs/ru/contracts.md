# Контракты — граница ядра

> Аудитория: инженер, подключающий хост-приложение (или любой другой
> хост) к чистому ядру, либо любой, кому нужно точно знать, где заканчивается
> ядро и начинается внешний мир. Область: библиотека ядра `protocore/`.

`protocore` — это набор **контрактов** (Python `Protocol`-ы + типизированные
Pydantic-модели) и protocol-first ReAct-рантайм. Всё, что обращено наружу, —
это `Protocol`, который реализует хост; ядро **не** поставляет ни драйвера базы
данных, ни HTTP-эндпойнта, ни LLM-клиента. Этот документ — каталог этой
границы: интерфейсные протоколы, которые предоставляет хост, система
типов ядра, которая через них проходит, и соглашения, удерживающие поверхность
стабильной.

Для более глубокого взгляда «как это собирается воедино» — цикл, подсистемы,
поток данных одного хода — читайте [`architecture.md`](architecture.md),
структурный источник, который индексирует эта страница.

---

## Пакет контрактов: никакого монолитного `protocols.py`

Поверхность интерфейсов живёт в `protocore/contracts/`, **по одному модулю на
домен**. Здесь сознательно **нет единого `protocols.py`** — старый монолитный
файл был разбит так, чтобы каждая область ответственности (LLM, запуски,
сессии, память, рабочее пространство, …) владела самодостаточным модулем с её
протоколом, её типизированными моделями и её ошибками вместе.

```
protocore/contracts/
  types.py        # the core type system (Message, ContentBlock union, Run, …)
  llm.py          # ILLMProvider + IProviderChain + LLMRequest/LLMResponse + provider deltas
  run.py          # IRunStore
  session.py      # ISessionStore
  blob.py         # IBlobStore
  search.py       # ISearchIndex
  todo.py         # ITodoStorage
  tool_registry.py# IToolRegistry + ToolVisibilityPolicy
  tools.py        # Tool (ABC) + ToolContext + tool errors
  run_state.py    # RunScopedState — the run's live state, incl. the host compartment
  tool_roles.py   # ToolRole / ToolRoleMap / ToolArgumentSlot — what a host's tools DO
  skills.py       # ISkillStore (+ SkillBundle / SkillIndexEntry / SkillFileRef)
  agent_dispatch.py# IAgentDispatch
  events.py       # IEventStream
  hooks.py        # IHookManager + HookResult / HookSpec / HookActionKind
  memory.py       # IMemory (+ IMemoryContentScanner)
  workspace.py    # IWorkspace
  resilience.py   # IToolTransport + IResilienceClassifier (+ the taxonomy)
  prompts.py      # IPromptTemplateProvider
  observability.py# CacheObserverProtocol + IRequestManifestSink + RequestManifest
  background.py   # IWorkPool / IBackgroundTaskPool / WorkHandle / TaskRecord
  middleware.py   # the lifecycle contract — ILifecycleRegistry + the five kinds
  interrupt.py    # PendingInterrupt / InterruptKind / InterruptResolution
  turn_policy.py  # ITurnPolicy / ITurnState / TurnFlags / TurnCoordinate
  config.py       # ConstantSpec / ConstantGroup / IConstantsRegistry
  snapshot.py     # the snapshot's schema version + the upcaster chain
  runtime_constants.py        # LoopConstants + RuntimeConstantsProvider
  attempt_ledger.py
  evidence.py                 # observed evidence + candidate lifecycle (imported by QueryEngine)
  tool_chunking.py            # chunkable-content truncation recovery (imported by the loop)
```

### Принцип реэкспорта contract-first

Реэкспорты управляются через явные списки `__all__`, а граница —
**contract-first**: публичная поверхность начинается с интерфейсных протоколов,
затем идут типизированные модели, которые через них проходят. Значимы два
списка `__all__`:

- **`protocore/contracts/__init__.py`** — поверхность контрактов, реэкспортируемая
  как явный список `__all__` (не каждый символ, который определяют доменные
  модули). Это и есть список, из которого нужно импортировать при реализации
  адаптера. Некоторые символы, определённые в их contract-модулях, сознательно
  **не** входят в этот `__all__` — например, `AttemptLedger`
  (`contracts/attempt_ledger.py`), `ConstantSpec` / `ConstantGroup` /
  `IConstantsRegistry` (`contracts/config.py`), `PendingInterrupt` /
  `InterruptResolution` (`contracts/interrupt.py`), `ITurnPolicy` / `TurnFlags`
  (`contracts/turn_policy.py`), `RequestManifest` / `IRequestManifestSink`
  (`contracts/observability.py`), `SNAPSHOT_SCHEMA_VERSION`
  (`contracts/snapshot.py`), `LLMTimeoutError`
  (`contracts/llm.py`) и `SkillNotFoundError` (`contracts/skills.py`) —
  импортируйте их из их именованного модуля.
- **`protocore/__init__.py`** (пакет верхнего уровня) — **курируемое подмножество**
  той же поверхности для типового случая. Он реэкспортирует те интерфейсы
  хранилищ/сервисов (`I*`), за которыми хост тянется чаще всего, плюс `Tool`,
  систему типов ядра, `LoopConstants`, словарь жизненного цикла и горстку утилит
  рантайма.

> **Обратите внимание:** `IMemory`, `IWorkspace`, `IToolTransport`,
> `IResilienceClassifier`, `IWorkPool`, `IPromptTemplateProvider` и
> `CacheObserverProtocol` экспортируются из
> `protocore.contracts` (и их именованных модулей), но **не** входят в `__all__`
> пакета верхнего уровня `protocore`. Импортируйте их из `protocore.contracts`
> (или из конкретного модуля), а не из пакета верхнего уровня. `IProviderChain`,
> `IConstantsRegistry`, `ICoreConstantsProvider`, `IRequestManifestSink`,
> `IDelegationTool`, `IMemoryContentScanner`, `IRunToolErrorCounter`,
> `ITurnPolicy` и `SkillFileRef` также не входят в
> `protocore.contracts.__all__` — импортируйте их из собственного модуля.

`QueryEngine` **не** реэкспортируется ни на одном из уровней — импортируйте его
напрямую из `protocore.runtime.query_engine`; точки подъёма (`resume`,
`resume_interrupts`, `resume_approved_tool`) — из `protocore.runtime`. См. раздел
[Public API](architecture.md#public-api-protocore__init__py).

---

## Интерфейсные протоколы (что предоставляет хост)

Это швы. Ядро объявляет `Protocol` (или ABC); хост привязывает конкретный
адаптер. Каждая строка — это контракт и то, что поставляет хост.

| Protocol | Module | What the host provides |
|---|---|---|
| `ILLMProvider` | `contracts/llm.py` | LLM-завершения: `stream_with_tools`, `complete_structured` (после цикла, JSON-схема), `complete_text` (после цикла, свободный документ) и `count_tokens`; универсальный LiteLLM/OpenAI-совместимый адаптер (OpenRouter / vLLM / OpenAI). |
| `IProviderChain` | `contracts/llm.py` | Упорядоченные оставшиеся провайдеры плюс односторонний курсор `advance()`. `QueryEngine` внедряет его как `provider_chain` для mid-stream failover; `None` оставляет существующее восстановление нетронутым. Не входит в `protocore.contracts.__all__` — импортируйте из `protocore.contracts.llm`. |
| `IRequestTokenCounter` | `contracts/llm.py` | Необязательная возможность провайдера: `async count_request_tokens(request) -> int \| None` — сколько токенов промпта сервер получит из запроса (сообщения через его чат-шаблон, инструменты, промпт генерации). Вызывается только у предела (`exact_token_count_margin_ratio`); `None` значит «здесь посчитать нельзя», исключение — откат к оценке с предупреждением. Ищется на классе провайдера, так что провайдер без неё не затронут. |
| `RuntimeConstantsProvider` | `contracts/runtime_constants.py` | Потенантные `LoopConstants` (`async get(tenant_id)`), на базе Postgres с Redis-кэшем. |
| `ISessionStore` | `contracts/session.py` | Хранение сессий / транскриптов. |
| `IRunStore` | `contracts/run.py` | Создание / список / чтение записей запусков (долговечная строка + горячая запись). |
| `IToolRegistry` | `contracts/tool_registry.py` | Конкретный `ToolRegistry` поставляется в ядре; хост регистрирует конкретные `Tool`-ы и `ToolVisibilityPolicy`. |
| `Tool` (ABC) / `@tool` | `contracts/tools.py`, `tools/decorator.py` | Конкретные реализации инструментов, привязанные к каноническим именам-глаголам. |
| `IToolTransport` | `contracts/resilience.py` | Транспорт инструмента / VM, который оборачивает обёртка отказоустойчивости; опциональный хук `rebuild()`. |
| `IMemory` | `contracts/memory.py` | Scope-aware FTS/BM25-хранилище памяти + `IMemoryContentScanner` для сканирования инъекций. |
| `IWorkspace` | `contracts/workspace.py` | Долговечное байтовое хранилище + FTS/BM25-манифест, атомарная запись, GC по скоупам. |
| `ISkillStore` | `contracts/skills.py` | Хранение и поиск skill-бандлов **и** многофайловый API: `list_files` / `load_file`, возвращающие строки `SkillFileRef` (как минимум каноническая запись `SKILL.md` / `SKILL_ENTRY_PATH`). **Цикл ядра никогда не вызывает** `list_files` / `load_file` — каталог строится через `list` / `list_enabled_subset`, тело по триггеру — через `load` / `list_subset`. Чтения store ключуются по `QueryEngineConfig.account_id`, не по `tenant_id`. Каталог рендерит `render_skills_catalog` (+ `derive_skill_index_budget_tokens`) в `runtime/skill_index.py` строками `Skill(skill="{name}")`, а не путями файлов. |
| `IHookManager` | `contracts/hooks.py` | Хуки, исполнитель которых живёт вне процесса (`invoke(event, payload, tenant_id)`); цикл ведёт их на гейте разрешений и вокруг диспетча инструмента. |
| `ILifecycleRegistry` | `contracts/middleware.py` | Единственный шов жизненного цикла: observe / decide / transform / around / notify, с владельцем, скоупом и идемпотентным диспозером у каждой регистрации. |
| `IEventStream` | `contracts/events.py` | Кросс-подовый долговечный поток событий для переподключения / повтора SSE. |
| `IBlobStore` | `contracts/blob.py` | Content-addressed хранилище блобов, используемое компакцией Tier-1. |
| `ISearchIndex` | `contracts/search.py` | Универсальный лексический поисковый индекс. |
| `ITodoStorage` | `contracts/todo.py` | Посессионное хранение todo. |
| `IAgentDispatch` | `contracts/agent_dispatch.py` | Диспетчеризация / поиск субагентов. `dispatch` отвечает `SubagentHandle`: дочерний прогон адресуем с момента запуска, а не только когда он закончился. Кому нужна прежняя блокирующая форма — пишет ожидание сам: `await (await dispatch(task)).wait()`. |
| `IBackgroundTaskPool` | `contracts/background.py` | Фоновые команды сессии в том виде, в каком их читает цикл: `list`, `get`, `refresh`, `drain_wakes` и объявление привязки (`mark_session_attached` / `ensure_session_attached`), которое возобновлённый прогон проверяет, прежде чем поверить пустому списку пробуждений. |
| `IWorkPool` | `contracts/background.py` | Всё перечисленное плюс ручки, делающие единицу работы управляемой: `launch(spec, start)` (id выдаётся ДО старта и передаётся в `start`), `handle`, `stop`, `stop_session(scope, grace)` и `subscribe(on_terminal)`. Дочерний прогон — запись `kind="agent"` в том же пуле, поэтому ожидание ребёнка и ожидание фоновой команды — один и тот же вопрос к одному и тому же коллаборатору. |
| `IPromptTemplateProvider` | `contracts/prompts.py` | Рендеринг шаблона системного промпта. |
| `CacheObserverProtocol` | `contracts/observability.py` | Приёмник доли попаданий в кэш промптов, внедряемый через `QueryEngineConfig.cache_observer`. |
| `IRequestManifestSink` | `contracts/observability.py` | Где хранится `RequestManifest` — долговечная запись того, что именно было отправлено провайдеру, — и как долго. Ядро считает манифест и его SHA-256 id и отдаёт их; ни хранилища, ни политики хранения у него нет. |
| `IConstantsRegistry` | `contracts/config.py` | Объявление собственных групп констант хоста (`declare`), fail-closed разрешение имён (`resolve`), `defaults`, `coerce` и `repair`. |
| `ICoreConstantsProvider` | `contracts/config.py` | Снимок `LoopConstants`, действующий для области, свежий на момент запроса. |
| `IResilienceClassifier` | `contracts/resilience.py` | К какому нейтральному `ResilienceErrorClass` относится сообщение об отказе. Формулировки принадлежат тому, что хост поставил за своими инструментами, и шаблон для них в ядре — копия чужой кодовой базы, которая протухает, ни разу не уронив сборку. Хост, не привязавший классификатор, сохраняет нейтральное поведение: каждый отказ различается по собственному тексту, и ни один не считается «транспорт лежит». |
| `IRunToolErrorCounter` | `contracts/run.py` | Долговечный счёт ошибок инструментов на прогон, переживающий смену процесса. |
| `IMemoryContentScanner` | `contracts/memory.py` | Сканирование содержимого памяти на инъекции при записи. |
| `ITurnPolicy` | `contracts/turn_policy.py` | Одно продуктовое решение о ходе. Не адаптер, который обязан поставить хост: ядро везёт собственный набор, а хост подставляет политику в него **по имени**. |

**32 `Protocol`-а** в `contracts/` — это не один плоский список адаптеров: часть
из них — хранилища, которые хост обязан поставить (`ILLMProvider`, `IRunStore`,
`ISessionStore`, `IBlobStore`, `ISearchIndex`, `ITodoStorage`, `IToolRegistry`,
`ISkillStore`, `IAgentDispatch`, `IEventStream`, `IHookManager`, `IMemory`,
`IWorkspace`, `IWorkPool`); часть — опциональные швы, остающиеся инертными,
пока ничего не привязано (`IToolTransport`, `IResilienceClassifier`,
`CacheObserverProtocol`, `IRequestManifestSink`, `IProviderChain`,
`ILifecycleRegistry`); а часть — формы, которые ядро отдаёт *обратно*, а не
просит (`WorkHandle`, `BackgroundTaskView`, `ITurnState`, `ClassifiedLike`).
`IToolSafetyPolicy` (`runtime/tool_permission.py`) — регистрируемая в рантайме
политика разрешений, а не контракт, а `Tool` — это ABC, а не `Protocol`: хост
наследуется от него (или использует `@tool`).

Какого бы вида ни был контракт, хосту не нужно ждать прогона, чтобы узнать, что
его объект имеет форму, которую вызовет ядро: `protocore.conformance` везёт по
одному набору на контракт в `SUITES`, и привязка каждого к адаптеру превращает
это в падающий тест в собственном наборе хоста. См. [`testing.md`](testing.md).

> `IBlobStore` объявлен как ABC; остальные интерфейсы хранилищ/сервисов — это
> `Protocol`-ы. В любом случае правило одно — ядро зависит только от объявленной
> формы и никогда от конкретной реализации.

---

## Система типов ядра

Каждый примитив диалога проходит как одна из этих Pydantic-моделей (соглашение
«используйте модели `Message`, никогда не сырые dict-ы»). Все они живут в
`contracts/types.py`, если не указано иное. Большинство — замороженные
value-объекты.

### Сообщения и контент

- **`Message`** — единственный примитив диалога (с привязкой к роли). Ходы
  ассистента несут `content_blocks` (text + tool_use + thinking вперемешку).
- **`MessageRole`** (`StrEnum`) — `system` · `user` · `assistant` · `tool`.
- **`ContentBlock`** — это **union-тип**, а не класс:
  `TextBlock | ThinkingBlock | ImageRefBlock | ToolUseBlock | ToolResultBlock`.
- **`ContentBlockKind`** (`StrEnum`) — дискриминант: `text` · `thinking` ·
  `image_ref` · `tool_use` · `tool_result`.
- **`TextBlock`** / **`ThinkingBlock`** — простой текст и рассуждения модели
  (последнее обычно вырезается перед сохранением).
- **`ToolUseBlock`** — вызов инструмента, эмитированный ассистентом
  (`tool_call_id`, `name`, `arguments_json`; байты аргументов ограничены).
- **`ToolResultBlock`** — результат вызова инструмента, возвращаемый модели
  (`tool_call_id`, `content`, `is_error`, `metadata`).
- **`ImageRefBlock`** — ссылка на изображение, чьи байты живут в `IBlobStore`.

### Вызовы инструментов и результаты

- **`ToolCall`** — вызов, эмитированный LLM и выставляемый в `Tool.invoke`
  (`id`, `name`, `arguments`). Несёт флаги усечения
  (`truncated_by_output_cap`, `args_partial_truncated`), которые цикл использует
  для обнаружения усечённого посередине потока JSON аргументов.
- **`ToolResult`** — результат одного вызова, названный один раз и
  спроецированный тремя способами. `content` — каноническое значение, полное,
  каким бы большим оно ни было, а `is_error` говорит, удался ли вызов.
  `model_projection` — то, что транскрипт несёт вместо содержимого, когда
  целиком оно там неуместно (`model_content` — свойство, из которого строится
  каждый `ToolResultBlock`, так что инструмент, не назвавший проекцию, ничего
  не теряет); `ui_payload` едет на событии результата и в транскрипт не
  попадает, поэтому не стоит токенов и не может изменить решение модели;
  `canonical_ref` называет место, откуда значение можно достать целиком, когда
  транскрипт его больше не держит; `path` говорит, видом какого пути в
  workspace результат является, — и именно это позволяет более поздней записи
  сказать, что результат больше не верен.
- **`ToolContext`** — контекст одного вызова, передаваемый инструменту: скоуп
  прогона, его `metadata`, `evidence` — когда инструмент производит
  свидетельства, — и `run_state`, живое состояние прогона, которому вызов
  принадлежит.
- **`RunScopedState`** (`contracts/run_state.py`) — всё, что прогон несёт при
  исполнении: событие отмены, разделяемая блокировка, кумулятивный журнал работы
  дерева и его бюджет параллельности, серии, по которым меряются лимиты, и
  `host` — непрозрачный отсек, который ядро не читает и не пишет, чтобы у
  встраивающего было одно место для того, что понимает только он. Хост собирает
  состояние на старте прогона и передаёт движку; `to_snapshot()` называет ту
  часть, которую обязан получить обратно подхват в другом процессе.
- **`ToolRoleMap`** (`contracts/tool_roles.py`) — что делают инструменты хоста,
  сказанное один раз там, где они регистрируются: `ToolRole` на инструмент
  (`reads_path`, `writes_path`, `runs_shell`, `never_delegated`, …) и, на каждый
  `ToolArgumentSlot`, написания аргумента, под которыми приходит значение. Цикл
  спрашивает карту, а не имя инструмента, поэтому установка, назвавшая свой
  пишущий инструмент иначе, сохраняет всё поведение, которое зависит от знания,
  что он пишущий.
- **`ToolDefinition`** — запись реестра (name, description, params schema,
  approval flag, category), которую производит функция `@tool` или подкласс
  `Tool`.
- **`ToolParameterSchema`** — форма параметров инструмента в виде JSON-Schema.
- **`ToolError`** (а также `ToolInvocationError`, `ToolPolicyDenied`,
  в `contracts/tools.py`) — иерархия ошибок инструментов.

### Запуски, сессии, события

Три различные формы запусков — не путайте их (см.
[`architecture.md`](architecture.md)):

- **`RunStatus`** (`StrEnum`) — **долговечный** жизненный цикл запуска,
  отражённый в персистентной колонке `runs.status`: `queued` · `running` ·
  `completed` · `partial` · `error` · `cancelled` · `incomplete` · `paused`.
  `partial` — функционально терминальный статус для запуска, который завершил
  свой цикл, но накопил ошибки диспетчеризации инструментов.
- **`Run`** — долговечная запись запуска (`id`, `tenant_id`, `session_id`,
  `status`, метки времени, опциональная ссылка на detail-блоб).
- **`RunState`** — **эфемерный** горячий рабочий набор (хранится хостом в
  Redis-хэше): `current_turn`, счётчики токенов, `last_event_id`. (Снова
  отличается от `LoopState`, FSM движка в полёте в `runtime/loop_state.py`,
  который *не* является контрактным типом.)
- **`Session`** — корень многоходового диалога (долговечный, никогда не
  удаляется).
- **`Event`** — конверт события в полёте (`run_id`, `name`, `payload`),
  эмитируемый через `IEventStream` / внутрипроцессный `EventBus`.
- **`StopReason`** (`StrEnum`) — почему ход завершился: `end_turn` · `tool_use` ·
  `max_tokens` · `max_turns` · `stop_sequence` · `error` · `cancelled`.
- **`ExecutionReport`** — ограниченная сводка телеметрии по запуску (события,
  записи вызовов инструментов, записи LLM-вызовов, предупреждения, запуски
  субагентов, артефакты и опциональный снимок `AttemptLedger`) со структурными
  ограничениями из `protocore.constants`.

### LLM-запрос / ответ

- **`LLMRequest`** — запрос, который цикл собирает для `ILLMProvider`
  (`messages`, `tools`, `max_tokens`, `extra` — включая подсказки кэша промптов
  `cache_breakpoints`).
- **`LLMResponse`** — форма непотокового ответа, которую возвращают
  `complete_structured` и `complete_text`.
- **`LLMObservabilityContext`** — контекст наблюдаемости на один вызов,
  прикреплённый к LLM-запросу.
- **`IProviderChain`** — не тип запроса/ответа; курсор failover, который
  `QueryEngine` перепривязывает на `self.llm`, когда mid-stream провайдер падает.

### Входящий трафик, блобы, компакция

- **`AgentEnvelope`** — единственный кросс-компонентный контракт входящего
  трафика (`kind`, `payload`, `metadata`; размер payload ограничен).
  Парсится/сериализуется через `parse_envelope` / `serialize_envelope`.
- **`EnvelopeKind`** (`StrEnum`) — `task` · `control` · `result` · `error`.
- **`BlobMetadata`** — запись индекса блоба (`ref`, `content_type`, `size_bytes`,
  `sha256`).
- **`CompactionSourceRef`** — указатель на сжатый блоб результата инструмента,
  сохраняемый как плейсхолдер в формате провода во время компакции Tier-1.

### Верификация и chunking

- **`VerificationLifecycle`** / **`VerificationDelivery`** /
  **`CandidateBundle`** / **`ReleaseDecision`** (`contracts/evidence.py`) —
  жизненный цикл верификации кандидата, который `QueryEngine` снимает как
  `verification` и использует, чтобы гейтить публичную доставку читателю.
  Реэкспортируются из `protocore.contracts`.
- **`is_chunkable_content_mutation`** (`contracts/tool_chunking.py`) — единственный
  предикат восстановления усечения write→append→finalize, заданный в терминах
  ролей (`ToolRole`), которые несёт вызов, а не имён инструментов. Импортируется
  циклом. Не входит в `protocore.contracts.__all__` — импортируйте из
  именованного модуля.

### Прерывания, состояние прогона и схема снимка

- **`PendingInterrupt`** / **`InterruptKind`** / **`InterruptResolution`**
  (`contracts/interrupt.py`) — то, чего ждёт приостановленный прогон, как
  значение, а не как защёлка. Прогон останавливается ради человека тремя
  разными способами, и `InterruptKind` их называет: `approval` (ворота
  припарковали вызов; ничего не исполнялось), `question` (инструмент успел
  спросить, и ответ и есть его результат) и `external_call` (вызов ушёл наружу,
  и результат придёт другим путём). Один булев флаг их не различал, поэтому
  отвеченный вопрос и одобренный вызов приходили в одну дверь, и циклу
  приходилось гадать: ошибка в одну сторону исполняет инструмент, которого
  никто не одобрял, в другую — сообщает модели, что заданный ею вопрос
  вернулся отказом. Прогон держит столько прерываний, сколько открыто, — и
  именно это позволяет ответить на три припаркованных вместе вызова одним
  действием вместо трёх раундов «стоп — вопрос — подъём». Решения:
  `approve` (опционально с `updated_input` — исправленными аргументами, с
  которыми вызов и будет реально исполнен и записан), `deny`, `answer` и
  `abandon`; карта, называющая незакрытое ожидание, отвечающая виду решением,
  которое ему не подходит, или оставляющая прерывание без решения,
  отвергается, если вызывающая сторона прямо не сказала, что остальное
  оставляет припаркованным.
- **`RunScopedState.to_snapshot()`** — два разрешения, которые поднятому
  прогону нельзя выдать второй раз: накопленный журнал работы дерева и ёмкость
  его бюджета параллелизма. Живые объекты (`asyncio.Event`, семафор, лок) по
  природе процесс-локальны и пересобираются тем, кто подключает поднятый
  прогон, а не восстанавливаются из payload.
- **`SNAPSHOT_SCHEMA_VERSION`** / **`SnapshotUpcaster`** /
  **`SnapshotSchemaError`** (`contracts/snapshot.py`) — снимок объявляет
  собственную версию схемы, и читатель, который её не узнаёт, отвергает payload
  целиком, а не поднимает прогон с потерянными полями и заново налитыми
  бюджетами. Более старый payload вместо этого поднимается вперёд там, где его
  накрывает цепочка upcaster-ов: по шагу на версию, каждый читает форму на
  версию ниже и дописывает то, что эта версия принесла. Версия без шага —
  отказ, потому что пропуск оставляет её поля незаполненными: то же тихое
  полу-восстановление. Payload без поля версии — это версия 1.

### Хуки

- **`HookEvent`** (`StrEnum`) — **21** координата жизненного цикла, весь
  словарь шва из `contracts/middleware.py`: `run_start` · `run_finalize` ·
  `session_start` · `session_end` · `turn_start` · `turn_end` ·
  `context_transform` · `request_prepare` · `response_received` ·
  `request_error` · `user_prompt_submit` · `pre_tool_use` · `tool_execute` ·
  `post_tool_use` · `file_changed` · `pre_compact` · `compaction_commit` ·
  `compaction_rollback` · `post_compact` · `subagent_start` · `subagent_stop`.
- **`HookResult`** — вердикт хука (allow / deny / modify, через
  `HookActionKind`).
- **`HookSpec`** — декларативная спецификация зарегистрированного хука.

### Навыки, субагенты, todo

- **`SkillManifest`** / **`SkillIndexEntry`** / **`SkillBundle`** /
  **`SkillFileRef`** — формы каталога навыков. `SkillFileRef` — это строка
  индекса многофайлового бандла (`path`, `size_bytes`, `mime_type`,
  `content_hash`); байты забираются через `ISkillStore.load_file`. У каждого
  бандла есть как минимум `SKILL_ENTRY_PATH` (`SKILL.md`). Устаревший
  однофайловый скилл может синтезировать эту одну строку из `body_md`.
  `SkillFileRef` **не** входит в `protocore.contracts.__all__` — импортируйте
  его из `protocore.contracts.skills`. Каталог цикла — формы вызова
  `Skill(skill="{name}")`, не эти пути. Цикл **не** вызывает `list_files` /
  `load_file`; чтения ключуются по `QueryEngineConfig.account_id`.
- **`SubagentDef`** / **`SubagentTask`** / **`SubagentResult`** — формы
  диспетчеризации субагентов, используемые `IAgentDispatch`. Помимо списков
  инструментов и скиллов определение говорит, КАК ведётся его ребёнок —
  `model`, `max_turns`, `timeout_seconds`, `permission_mode`, `background`, — а
  задача может переопределить три последних на один вызов плюс
  `notify_on_finish` и `expected_seconds`. Всё это — объявления, которые
  исполняет хост: цикл не строит дочерних прогонов, поэтому не может ни выбрать
  им модель, ни запустить их часы.
- **`WorkSpec`** / **`TaskRecord`** / **`WorkHandle`** / **`AgentRef`** — формы
  пула. `WorkSpec` — всё, что нужно, чтобы выдать записи имя до старта работы;
  `TaskRecord` — сама запись (`kind`, `status`, `owner_scope`, код выхода,
  ошибка, длительности, `agent`); `WorkHandle` — `identity()` / `wait()` /
  `stop(grace)` над одной записью, а `SubagentHandle` — тот же handle над
  результатом субагента: второго типа handle нет. `owner_scope` называет
  прогон, обязанный завершить работу, когда это не сессия: делегированный
  прогон делит сессию (а значит, рабочее пространство и пробуждение) и
  сворачивает только то, что запустил сам.
- **`IDelegationTool`** (`contracts/agent_dispatch.py`) — то, как цикл узнаёт
  делегирующий инструмент. Инструмент объявляет контракт вместо атрибута-флага,
  поэтому делегирование — заявленная возможность, а не догадка по утиной
  типизации.
- **`Todo`** / **`TodoStatus`** (`StrEnum`) — форма посессионного хранения todo.

---

## Реализация адаптера

Чтобы привязать ядро к хосту:

1. Реализуйте нужные вам интерфейсные протоколы из `protocore.contracts` (вам не
   нужны все — память по умолчанию выключена (`memory_enabled = False`), а
   конкретные инструменты рабочего пространства живут в хосте).
2. Принимайте и возвращайте модели системы типов ядра — никогда не сырые dict-ы
   на границе.
3. Внедряйте конфигурацию через `LoopConstants` (замороженный снимок) и
   `ToolContext.metadata`; никогда не зашивайте политику тенанта в код.
4. Сконструируйте `QueryEngine` со своими адаптерами и управляйте им через
   `async for evt in engine.run(message)` или поднимите сохранённый прогон
   через `async for evt in resume(engine, snapshot)`.

Механика расширения рантайма — какой шов выбрать (протокол vs хук vs RC-тоггл vs
секция промпта) и жёсткое правило «не модифицировать структуру цикла» — описана
в `extending.md` и [`architecture.md`](architecture.md). Граница импортов (ядро
никогда не импортирует хост) обеспечивается
`tests/test_core_import_boundary.py`.

> Перевод английского оригинала `docs/contracts.md` (коммит `54b6543`). При изменении оригинала обновите перевод.
