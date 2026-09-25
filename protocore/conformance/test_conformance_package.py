"""The conformance package checking itself.

Ships inside the package so a host can run it against the core version it
actually has installed::

    pytest --pyargs protocore.conformance

Two things are at stake. The first is coverage: a contract with no suite is a
contract whose implementations are only ever checked by being called during a
run, so the catalogue below is compared against the contracts directory itself
rather than maintained by hand. The second is that the suites detect anything at
all — a structural check that passes for every object is worse than no check,
because it reads as evidence.
"""
from __future__ import annotations

import ast
import asyncio
import inspect
from pathlib import Path
from typing import Any

import pytest

import protocore.conformance as conformance_package
import protocore.contracts as contracts_package
from protocore.conformance import SUITES, ContractSuite, binds_a_subject, bound_suites
from protocore.conformance.suite import _protocol_member, declared_members
from protocore.contracts.events import IEventStream
from protocore.contracts.session import ISessionStore
from protocore.contracts.tool_registry import IToolRegistry
from protocore.tests_support.adapters import (
    InMemoryAgentDispatch,
    InMemoryBlobStore,
    InMemoryEventStream,
    InMemoryHookManager,
    InMemoryLLMProvider,
    InMemoryMemory,
    InMemoryRequestManifestSink,
    InMemoryRunStore,
    InMemorySearchIndex,
    InMemorySessionStore,
    InMemorySkillStore,
    InMemoryTodoStorage,
    InMemoryToolRegistry,
    InMemoryWorkspace,
)

#: Contracts the core implements itself rather than asking a host for. A suite
#: for one of these would bind the core's own class to the core's own
#: declaration and assert that a file agrees with itself. Everything else
#: under ``contracts/`` is something a host supplies, and gets a suite.
_CORE_IMPLEMENTED = frozenset({"ITurnPolicy", "ITurnState", "Tool"})

#: The base classes a contract is spelled with. A dependency written as an
#: abstract base class is supplied by a host exactly as a Protocol is, and
#: ``isinstance`` recognises it without ``@runtime_checkable``, so leaving it
#: out of the walk means the exhaustiveness guarantee quietly does not cover
#: it — which is how the blob store went without a suite.
_CONTRACT_BASES = frozenset({"Protocol", "ABC"})


def _declared_contract_names() -> set[str]:
    """Every contract declared under ``protocore/contracts/``, read from source."""
    directory = Path(contracts_package.__file__).parent
    names: set[str] = set()
    for path in sorted(directory.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            if any(
                isinstance(base, ast.Name) and base.id in _CONTRACT_BASES
                for base in node.bases
            ):
                names.add(node.name)
    return names - _CORE_IMPLEMENTED


# ── the catalogue is exhaustive ─────────────────────────────────────────────


def test_every_declared_contract_has_a_suite() -> None:
    covered = {suite.protocol.__name__ for suite in SUITES}
    declared = _declared_contract_names()

    assert declared - covered == set(), (
        "these contracts are declared with no conformance suite: "
        f"{sorted(declared - covered)}. Add one in "
        "protocore/conformance/suites.py and list it in SUITES."
    )
    assert covered - declared == set(), (
        f"these suites name a contract the core no longer declares: "
        f"{sorted(covered - declared)}"
    )
    assert len(SUITES) == len(declared)


def test_each_suite_names_a_distinct_contract() -> None:
    protocols = [suite.protocol for suite in SUITES]
    assert len(set(protocols)) == len(protocols)


def test_every_suite_is_importable_from_the_package() -> None:
    # A host imports suites from ``protocore.conformance``; one listed in
    # SUITES but not re-exported there is reachable only by a private path.
    exported = set(conformance_package.__all__)
    namespace = vars(conformance_package)
    missing = sorted(
        suite.__name__
        for suite in SUITES
        if suite.__name__ not in exported or namespace.get(suite.__name__) is not suite
    )
    assert missing == [], (
        f"these suites are not re-exported by protocore.conformance: {missing}. "
        "Import them in protocore/conformance/__init__.py and list them in __all__."
    )


def test_every_suite_class_in_the_suites_module_is_listed() -> None:
    from protocore.conformance import suites as suites_module

    defined = {
        obj
        for obj in vars(suites_module).values()
        if inspect.isclass(obj)
        and issubclass(obj, ContractSuite)
        and obj is not ContractSuite
    }
    assert defined - set(SUITES) == set(), (
        "these suite classes are defined but missing from SUITES: "
        f"{sorted(cls.__name__ for cls in defined - set(SUITES))}"
    )


def test_each_suite_is_a_contract_suite_and_says_what_it_is_for() -> None:
    for suite in SUITES:
        assert issubclass(suite, ContractSuite)
        assert (suite.__doc__ or "").strip(), f"{suite.__name__} has no docstring"


def test_a_contract_declares_the_members_the_suite_will_look_for() -> None:
    """Guards the introspection itself: a ``declared_members`` that returned
    nothing would make every suite pass against every object."""
    for suite in SUITES:
        assert declared_members(suite.protocol), (
            f"{suite.protocol.__name__} appears to declare no members; the "
            "suite built on it would assert nothing"
        )


# ── the suites detect a real adapter, and a broken one ──────────────────────


_BOUND: tuple[tuple[str, type], ...] = (
    ("IAgentDispatch", InMemoryAgentDispatch),
    ("IBlobStore", InMemoryBlobStore),
    ("IEventStream", InMemoryEventStream),
    ("IHookManager", InMemoryHookManager),
    ("ILLMProvider", InMemoryLLMProvider),
    ("IMemory", InMemoryMemory),
    ("IRequestManifestSink", InMemoryRequestManifestSink),
    ("IRunStore", InMemoryRunStore),
    ("ISearchIndex", InMemorySearchIndex),
    ("ISessionStore", InMemorySessionStore),
    ("ISkillStore", InMemorySkillStore),
    ("ITodoStorage", InMemoryTodoStorage),
    ("IToolRegistry", InMemoryToolRegistry),
    ("IWorkspace", InMemoryWorkspace),
)


def _suite_for(protocol_name: str) -> type[ContractSuite]:
    for suite in SUITES:
        if suite.protocol.__name__ == protocol_name:
            return suite
    raise AssertionError(f"no suite for {protocol_name}")


def _run_every_check(suite: type[ContractSuite], subject: Any) -> None:
    """Every case in ``suite``, structural and behavioural, against ``subject``.

    A behavioural case is a coroutine, and calling one without awaiting it
    builds an object that does nothing — the same silent no-op the suites
    themselves exist to catch — so they are run rather than merely called.
    """
    instance = suite()
    for name in dir(suite):
        if not name.startswith("test_"):
            continue
        result = getattr(instance, name)(subject)
        if inspect.iscoroutine(result):
            asyncio.run(result)


@pytest.mark.parametrize(("protocol_name", "adapter"), _BOUND, ids=[n for n, _ in _BOUND])
def test_the_cores_own_doubles_pass_their_suite(protocol_name: str, adapter: type) -> None:
    """The doubles are what the core's own tests run against, so they are the
    one set of adapters that must satisfy the contracts by definition."""
    _run_every_check(_suite_for(protocol_name), adapter())


def test_a_missing_member_is_caught() -> None:
    class Hollow:
        pass

    with pytest.raises(AssertionError):
        _run_every_check(_suite_for("ISessionStore"), Hollow())


def test_a_synchronous_implementation_of_an_async_contract_is_caught() -> None:
    """The failure this suite exists for: it type-checks, it has every member,
    and the core awaits a value that is not awaitable."""

    class Synchronous:
        def __getattr__(self, name: str) -> Any:
            if name in declared_members(ISessionStore):
                return lambda *args, **kwargs: None
            raise AttributeError(name)

    with pytest.raises(AssertionError, match="synchronously"):
        _run_every_check(_suite_for("ISessionStore"), Synchronous())


def test_an_asynchronous_implementation_of_a_synchronous_contract_is_caught() -> None:
    """The silent half: the core does not await this one, so an ``async def``
    here returns a coroutine that is never run and the call does nothing."""
    subject = InMemoryToolRegistry()

    async def get_but_awaitable(name: str, tenant_id: str | None = None) -> None:
        return None

    subject.get = get_but_awaitable  # type: ignore[assignment,method-assign]
    assert not inspect.iscoroutinefunction(_protocol_member(IToolRegistry, "get"))

    with pytest.raises(AssertionError, match="asynchronously"):
        _run_every_check(_suite_for("IToolRegistry"), subject)


def test_a_renamed_parameter_is_caught() -> None:
    """Structurally perfect and uncallable: the core passes by keyword."""
    subject = InMemoryEventStream()

    async def emit_under_another_parameter_name(payload: object) -> None:  # pragma: no cover - shape only
        return None

    subject.emit = emit_under_another_parameter_name  # type: ignore[assignment,method-assign]
    declared = declared_members(IEventStream)
    assert "emit" in declared

    with pytest.raises(AssertionError, match="cannot be called"):
        _run_every_check(_suite_for("IEventStream"), subject)


def test_an_unbound_suite_skips_rather_than_fails() -> None:
    """Importing a suite a host has not bound yet must cost that host nothing."""
    suite = _suite_for("ISessionStore")()

    with pytest.raises(BaseException) as excinfo:
        suite.unbound()

    assert "no subject bound" in str(excinfo.value)


# ── a host can tell a bound suite from a skipped one ────────────────────────


def test_an_unbound_suite_is_reported_as_unbound() -> None:
    for suite in SUITES:
        assert not binds_a_subject(suite)


def test_a_suite_a_host_bound_is_reported_as_bound() -> None:
    class TestTheHostsStore(_suite_for("ISessionStore")):  # type: ignore[misc]
        @pytest.fixture
        def subject_factory(self) -> Any:
            return InMemorySessionStore

    assert binds_a_subject(TestTheHostsStore)


def test_a_module_of_unbound_subclasses_reports_nothing_bound() -> None:
    """The shape that exits zero while asserting nothing: every suite imported,
    every subject forgotten."""

    class Namespace:
        class TestTheHostsStore(_suite_for("ISessionStore")):  # type: ignore[misc]
            pass

    assert bound_suites(Namespace) == frozenset()


def test_a_module_reports_the_contracts_it_bound() -> None:
    class Namespace:
        class TestTheHostsStore(_suite_for("ISessionStore")):  # type: ignore[misc]
            @pytest.fixture
            def subject_factory(self) -> Any:
                return InMemorySessionStore

        class TestTheHostsStream(_suite_for("IEventStream")):  # type: ignore[misc]
            @pytest.fixture
            def subject_factory(self) -> Any:
                return InMemoryEventStream

    assert {suite.protocol for suite in bound_suites(Namespace)} == {
        ISessionStore,
        IEventStream,
    }


def test_a_sink_that_does_its_work_in_the_call_is_caught() -> None:
    """The behavioural case: the core awaits this before it opens the stream.

    A store call made inside the method rather than queued behind it is time
    every user of the run waits through, and a synchronous one stalls every
    other task in the process while it runs. Neither is visible in the
    signature, which is why this one case looks at what the adapter does.
    """
    import time as _time

    class _StoresInline(InMemoryRequestManifestSink):
        async def record_request_manifest(self, **kwargs: Any) -> None:
            _time.sleep(0.4)
            await super().record_request_manifest(**kwargs)

    with pytest.raises(AssertionError):
        _run_every_check(_suite_for("IRequestManifestSink"), _StoresInline())
