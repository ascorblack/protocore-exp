# Protocore

**Protocore is the agent loop, and nothing else.**

It is a Python 3.12+ library holding one thing: the ReAct runtime that drives an
LLM agent turn by turn — the loop, the context budget, the tool surface, the
compaction, the stop conditions. Everything the loop touches from outside is a
`Protocol` you implement: the model client, the stores, the event transport, the
tools themselves. There is no database driver here, no HTTP endpoint, no
deployment logic, and no import that reaches upward out of the package.

That constraint is the point. An agent loop is where the hard, unglamorous
correctness lives — what to do when the model answers with prose instead of the
tool it was told to call, when a tool result is a hundred kilobytes, when the
context window fills mid-turn, when a run must be snapshotted and resumed on
another process. Protocore isolates that from the plumbing so it can be tested
exhaustively and reused across products.

> Русская версия: [`README.md`](README.md) · Docs: [`docs/index.md`](docs/index.md) (EN) · [`docs/ru/index.md`](docs/ru/index.md) (RU)

## What you get

- **20 interface Protocols** — `ILLMProvider`, `IRunStore`, `ISessionStore`,
  `IToolRegistry`, `IMemory`, `IWorkspace`, `ISearchIndex`, `IEventStream`,
  `ISkillStore`, `IHookManager`, and the rest, plus an `IBlobStore` ABC. They
  are the whole outward surface; the core never learns what is behind them.
- **A ReAct runtime** — `QueryEngine` owns the per-run mutable state, `query()`
  drives one turn and yields a stream of typed `TurnEvent`s. Snapshot and resume
  are first-class, so a run survives a process restart.
- **A three-layer tool surface** — tenant policy, then a lean clipped surface,
  then progressive discovery over BM25 retrieval, with a permission gate in
  front of dispatch.
- **Two-tier context compaction** — the loop keeps working when the transcript
  outgrows the window, and the compaction is deterministic enough to test.
- **524 runtime constants** — every tunable value is a field on a frozen
  `RuntimeConstants` snapshot injected per tenant. No magic numbers in the
  executable path, and new behaviour defaults off.
- **In-memory adapters**, shipped inside the package, so you can drive a real
  turn end to end with no external services at all.

## Install

```bash
pip install protocore==2.0.0a22
```

Name the version explicitly. The published release is a pre-release, and pip
skips those unless you ask — but do **not** ask with a bare `--pre`, because
that flag applies to the whole resolution and will pull pre-release builds of
`pydantic` too. The pin comes off when there is a stable release.

Or, to work on it, with [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync --extra dev
```

Python ≥ 3.12. Runtime dependencies are `pydantic`, `jinja2`, and
`typing-extensions` — nothing else.

### Extras

```bash
pip install "protocore[testing]==2.0.0a22"   # run the conformance suites against your adapters
```

`testing` adds a test runner and nothing more. `protocore.conformance` is a
pytest suite that ships inside the wheel, and a host points it at its own
implementations of the contracts (`pytest --pyargs protocore.conformance`). The
core's linter and type checker are deliberately not in it: those belong to
someone changing the core, not to someone using it.

Token estimation also has an optional native implementation, published as the
separate distribution `protocore-native`. Wheels for it are not on the index
yet, so for now it is built from source; the core stays pure Python and selects
the extension only when it can import it, so having it changes speed and nothing
else — same numbers, same contract. When the extension is installed and you want
the Python implementation anyway — to compare the two, to rule it out as the
cause of a discrepancy, or to build a reproducible environment — the environment
variable **`PROTOCORE_DISABLE_NATIVE=1`** keeps it in force. It is read **once,
at import**, and answers "which build of this function am I running" rather than
tuning behaviour: changing it inside a live process does nothing.

## Quickstart

The core is adapter-driven: build a `QueryEngine` with your adapters, then
iterate the events from `engine.run(message)`. The method appends the user's
message and drives one turn to a terminal state.

The example below uses the bundled in-memory adapters, so it runs as-is:

```python
import asyncio

from protocore import (
    Message, MessageRole, StopReason, TextBlock, default_runtime_constants,
)
from protocore.runtime.query_engine import QueryEngine, QueryEngineConfig
from protocore.tests_support.adapters import (
    InMemoryBlobStore, InMemoryEventStream, InMemoryHookManager,
    InMemoryLLMProvider, InMemorySkillStore, InMemoryToolRegistry,
)


async def main() -> None:
    llm = InMemoryLLMProvider()
    llm.queue_response(text="Hello from Protocore.", stop_reason=StopReason.end_turn)

    engine = QueryEngine(
        config=QueryEngineConfig(
            run_id="run-1",
            tenant_id="default",
            session_id="sess-1",
            model_name="smoke-model",
            rc=default_runtime_constants(),
        ),
        llm_provider=llm,
        tool_registry=InMemoryToolRegistry(),
        event_stream=InMemoryEventStream(),
        hook_manager=InMemoryHookManager(),
        skill_store=InMemorySkillStore(),
        blob_store=InMemoryBlobStore(),
    )
    message = Message(
        role=MessageRole.user,
        content_blocks=[TextBlock(text="Say hello.")],
    )
    async for event in engine.run(message):
        print(event.type)
    print("final state:", engine.state)


asyncio.run(main())
```

Swap `InMemoryLLMProvider` for an adapter over a real model client and the same
code answers real prompts. [Getting started](docs/getting-started.md) walks
through what each event means and what to replace next.

> **Import location matters.** `QueryEngine`, `QueryEngineConfig`, and `query`
> come from `protocore.runtime.*`; they are not top-level re-exports. The
> contract types (`Message`, `StopReason`, `RuntimeConstants`, …) *are*.

## Documentation

| Document | What it covers |
|---|---|
| [Documentation hub](docs/index.md) | start here — reading order and a map |
| [Getting started](docs/getting-started.md) | install plus a runnable example |
| [Architecture](docs/architecture.md) | the deep reference |
| [Contracts](docs/contracts.md) | the protocol boundary and the type system |
| [Tools](docs/tools.md) | the lean tool surface and the permission gate |
| [Runtime constants](docs/runtime-constants.md) | the configuration model |
| [Extending the core](docs/extending.md) | adapters, hooks, toggles, prompt sections |
| [Testing](docs/testing.md) | running the suite and the import-boundary guard |
| [Glossary](docs/glossary.md) | key terms |

A full Russian mirror lives under [`docs/ru/`](docs/ru/index.md).

## Development

```bash
uv sync --extra dev
uv run pytest .            # 3215 tests
uv run ruff check .
uv run mypy --strict
uv run bandit -r protocore -q -c pyproject.toml
```

All four gates run on every pull request across Python 3.12, 3.13, and 3.14.
Coverage is enforced at 90%.

The guard worth knowing about is `tests/test_core_import_boundary.py`: it
AST-parses every module in the package and fails if any of them imports a
package that sits above the core — anything sharing the core's name with an
underscore after it. That test is what keeps the rest of this README true.

Contributions are welcome; see [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

[Mozilla Public License 2.0](LICENSE).

The MPL is a **file-level** copyleft. In practice that means: build whatever you
like on top of Protocore and keep it closed — your adapters, your service, your
product are yours. But if you modify a Protocore file itself, that file's source
stays open under the same license, and the notices travel with it. See
[`NOTICE`](NOTICE) for what that asks of you in concrete terms.

Security issues go to [`SECURITY.md`](SECURITY.md), not to the public issue
tracker.
