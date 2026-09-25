"""Conformance suites a host runs against its own adapters.

The core declares its dependencies as Protocols and never sees the objects that
satisfy them until a run calls one. These suites move that moment earlier: a
host imports the suite for a contract, binds it to its adapter with a factory
fixture, and finds out in its own test run — rather than in an agent's turn —
that the adapter has the shape the core will call.

Installed with the ``testing`` extra, which brings the ``pytest`` this package
imports:

.. code-block:: console

    pip install "protocore[testing]"

:data:`SUITES` is every suite here, one per contract, and the package's own test
module fails if a contract is declared without one.

A suite a host imports but never binds **skips**, so a host's conformance
directory can pass while asserting nothing. :func:`bound_suites` is how a host
tells the two apart: point it at its own conformance module and compare the
result against the contracts it implements, so that forgetting to bind one is
a failure rather than a silent skip.
"""
from __future__ import annotations

from protocore.conformance.request_manifest import RequestManifestSinkConformance
from protocore.conformance.suite import (
    ContractSuite,
    binds_a_subject,
    bound_contracts,
    bound_suites,
    declared_members,
)
from protocore.conformance.suites import (
    SUITES,
    AgentDispatchConformance,
    BackgroundTaskPoolConformance,
    BackgroundTaskViewConformance,
    BlobStoreConformance,
    CacheObserverConformance,
    ClassifiedErrorConformance,
    ConstantsRegistryConformance,
    CoreConstantsProviderConformance,
    DelegationToolConformance,
    EventStreamConformance,
    HookManagerConformance,
    LifecycleRegistryConformance,
    LLMProviderConformance,
    MemoryConformance,
    MemoryContentScannerConformance,
    PromptTemplateProviderConformance,
    ProviderChainConformance,
    RequestTokenCounterConformance,
    ResilienceClassifierConformance,
    RunStoreConformance,
    RuntimeConstantsProviderConformance,
    RunToolErrorCounterConformance,
    SearchIndexConformance,
    SessionStoreConformance,
    SkillStoreConformance,
    TodoStorageConformance,
    ToolRegistryConformance,
    ToolTransportConformance,
    WorkPoolConformance,
    WorkspaceConformance,
)

__all__ = [
    "SUITES",
    "AgentDispatchConformance",
    "BackgroundTaskPoolConformance",
    "BackgroundTaskViewConformance",
    "BlobStoreConformance",
    "CacheObserverConformance",
    "ClassifiedErrorConformance",
    "ConstantsRegistryConformance",
    "ContractSuite",
    "CoreConstantsProviderConformance",
    "DelegationToolConformance",
    "EventStreamConformance",
    "HookManagerConformance",
    "LLMProviderConformance",
    "LifecycleRegistryConformance",
    "MemoryConformance",
    "MemoryContentScannerConformance",
    "PromptTemplateProviderConformance",
    "ProviderChainConformance",
    "RequestManifestSinkConformance",
    "RequestTokenCounterConformance",
    "ResilienceClassifierConformance",
    "RunStoreConformance",
    "RunToolErrorCounterConformance",
    "RuntimeConstantsProviderConformance",
    "SearchIndexConformance",
    "SessionStoreConformance",
    "SkillStoreConformance",
    "TodoStorageConformance",
    "ToolRegistryConformance",
    "ToolTransportConformance",
    "WorkPoolConformance",
    "WorkspaceConformance",
    "binds_a_subject",
    "bound_contracts",
    "bound_suites",
    "declared_members",
]
