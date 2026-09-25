"""One conformance suite per contract the core declares.

The list is exhaustive by construction: ``test_conformance_package`` walks
``protocore/contracts/`` and fails if a contract is declared with no suite
here. A contract without a suite is a contract whose implementations are
checked only by being called during a run.

"Contract" is read by what the core asks a host for, not by how it is spelled:
a dependency declared as an abstract base class is supplied by the host in
exactly the way a Protocol is, and ``isinstance`` recognises it without
``@runtime_checkable``, so the same suite body applies unchanged.
"""
from __future__ import annotations

from protocore.conformance.request_manifest import RequestManifestSinkConformance
from protocore.conformance.suite import ContractSuite
from protocore.contracts.agent_dispatch import IAgentDispatch, IDelegationTool
from protocore.contracts.background import (
    BackgroundTaskView,
    IBackgroundTaskPool,
    IWorkPool,
)
from protocore.contracts.blob import IBlobStore
from protocore.contracts.config import IConstantsRegistry, ICoreConstantsProvider
from protocore.contracts.events import IEventStream
from protocore.contracts.hooks import IHookManager
from protocore.contracts.llm import ILLMProvider, IProviderChain, IRequestTokenCounter
from protocore.contracts.memory import IMemory, IMemoryContentScanner
from protocore.contracts.middleware import ILifecycleRegistry
from protocore.contracts.observability import CacheObserverProtocol
from protocore.contracts.prompts import IPromptTemplateProvider
from protocore.contracts.resilience import (
    ClassifiedLike,
    IResilienceClassifier,
    IToolTransport,
)
from protocore.contracts.run import IRunStore, IRunToolErrorCounter
from protocore.contracts.runtime_constants import RuntimeConstantsProvider
from protocore.contracts.search import ISearchIndex
from protocore.contracts.session import ISessionStore
from protocore.contracts.skills import ISkillStore
from protocore.contracts.todo import ITodoStorage
from protocore.contracts.tool_registry import IToolRegistry
from protocore.contracts.workspace import IWorkspace


class AgentDispatchConformance(ContractSuite):
    """The seam a host's subagent launcher must satisfy."""

    protocol = IAgentDispatch


class BackgroundTaskViewConformance(ContractSuite):
    """The read-only view of one background command."""

    protocol = BackgroundTaskView


class BackgroundTaskPoolConformance(ContractSuite):
    """The pool that owns a session's background commands."""

    protocol = IBackgroundTaskPool


class WorkPoolConformance(ContractSuite):
    """The pool that owns both a session's commands and its child runs."""

    protocol = IWorkPool


class DelegationToolConformance(ContractSuite):
    """The tool contract that tells the loop a call starts child runs."""

    protocol = IDelegationTool


class BlobStoreConformance(ContractSuite):
    """The store a value too large to travel in a payload is put in."""

    protocol = IBlobStore


class ConstantsRegistryConformance(ContractSuite):
    """The registry every constant group is declared and resolved through."""

    protocol = IConstantsRegistry


class CoreConstantsProviderConformance(ContractSuite):
    """The source of the constants snapshot one turn runs under."""

    protocol = ICoreConstantsProvider


class EventStreamConformance(ContractSuite):
    """The transport every run event and snapshot leaves the core through."""

    protocol = IEventStream


class HookManagerConformance(ContractSuite):
    """The host's hook dispatch."""

    protocol = IHookManager


class LifecycleRegistryConformance(ContractSuite):
    """The one seam every lifecycle registration is placed on."""

    protocol = ILifecycleRegistry


class LLMProviderConformance(ContractSuite):
    """The provider the loop streams from."""

    protocol = ILLMProvider


class ProviderChainConformance(ContractSuite):
    """The one-way demotion cursor over a run's providers."""

    protocol = IProviderChain


class RequestTokenCounterConformance(ContractSuite):
    """The optional provider capability that sizes a rendered request."""

    protocol = IRequestTokenCounter


class MemoryConformance(ContractSuite):
    """The durable memory store."""

    protocol = IMemory


class MemoryContentScannerConformance(ContractSuite):
    """The screen a memory write passes before it is stored."""

    protocol = IMemoryContentScanner


class CacheObserverConformance(ContractSuite):
    """The prompt-cache observer."""

    protocol = CacheObserverProtocol


class PromptTemplateProviderConformance(ContractSuite):
    """The template source the prompts layer renders through."""

    protocol = IPromptTemplateProvider


class ClassifiedErrorConformance(ContractSuite):
    """The shape the resilience layer reads an error's classification from."""

    protocol = ClassifiedLike


class ResilienceClassifierConformance(ContractSuite):
    """The host's verdict on what kind of failure a message describes."""

    protocol = IResilienceClassifier


class ToolTransportConformance(ContractSuite):
    """The transport a remote tool call travels over."""

    protocol = IToolTransport


class RunStoreConformance(ContractSuite):
    """The durable record of a run."""

    protocol = IRunStore


class RunToolErrorCounterConformance(ContractSuite):
    """The run-scoped tool-error tally the breaker reads."""

    protocol = IRunToolErrorCounter


class RuntimeConstantsProviderConformance(ContractSuite):
    """The source of a scope's runtime constants."""

    protocol = RuntimeConstantsProvider


class SearchIndexConformance(ContractSuite):
    """The index the search tools query."""

    protocol = ISearchIndex


class SessionStoreConformance(ContractSuite):
    """The store a session's history and state live in."""

    protocol = ISessionStore


class SkillStoreConformance(ContractSuite):
    """The store the skill catalog and bundles are read from."""

    protocol = ISkillStore


class TodoStorageConformance(ContractSuite):
    """The todo list's storage."""

    protocol = ITodoStorage


class ToolRegistryConformance(ContractSuite):
    """The catalog the loop resolves tool names against."""

    protocol = IToolRegistry


class WorkspaceConformance(ContractSuite):
    """The agent's file surface."""

    protocol = IWorkspace


#: Every suite this package publishes, one per contract the core declares.
#: One of them lives in its own module — see
#: :mod:`protocore.conformance.request_manifest` for why — and is listed here
#: like the rest, because the catalogue is what the exhaustiveness check reads.
SUITES: tuple[type[ContractSuite], ...] = (
    AgentDispatchConformance,
    BackgroundTaskViewConformance,
    BackgroundTaskPoolConformance,
    WorkPoolConformance,
    DelegationToolConformance,
    BlobStoreConformance,
    ConstantsRegistryConformance,
    CoreConstantsProviderConformance,
    EventStreamConformance,
    HookManagerConformance,
    LifecycleRegistryConformance,
    LLMProviderConformance,
    ProviderChainConformance,
    RequestTokenCounterConformance,
    MemoryConformance,
    MemoryContentScannerConformance,
    CacheObserverConformance,
    PromptTemplateProviderConformance,
    ClassifiedErrorConformance,
    ResilienceClassifierConformance,
    ToolTransportConformance,
    RequestManifestSinkConformance,
    RunStoreConformance,
    RunToolErrorCounterConformance,
    RuntimeConstantsProviderConformance,
    SearchIndexConformance,
    SessionStoreConformance,
    SkillStoreConformance,
    TodoStorageConformance,
    ToolRegistryConformance,
    WorkspaceConformance,
)
