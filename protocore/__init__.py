# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# The same notice covers every file in this package and in ``tests/``.
# See LICENSE for the full text and NOTICE for what it asks of you.
"""Protocore — protocol-first pure core.

Public API:
    - 12 interface contracts in :mod:`protocore.contracts`
    - Core type system (Message, ToolCall, Event, Run, Session, …)
    - :class:`~protocore.contracts.runtime_constants.LoopConstants` frozen snapshot
    - :class:`~protocore.events.EventBus` + :class:`~protocore.events.EventName`
    - :class:`~protocore.hooks.HookManager` — the lifecycle registry (see
      :mod:`protocore.contracts.middleware`)
    - :class:`~protocore.safety.DefaultShellSafetyPolicy`
    - :func:`~protocore.tools.tool` decorator
    - :func:`~protocore.ingress.parse_envelope` + :func:`~protocore.json_utils.parse_complete_json`
    - :mod:`protocore.runtime.token_counting` LanguageProfile + estimator
    - :mod:`protocore.runtime.context.budgets.derive_budgets`
    - :mod:`protocore.runtime.wire_format` compaction placeholders
    - :mod:`protocore.runtime.chain_parser` shell-grammar parser
    - :mod:`protocore.runtime.tool_retrieval` BM25 multilingual retrieval

Loop machinery (QueryEngine + query()) is available in :mod:`protocore.runtime.query_engine`.
"""
from __future__ import annotations

from protocore.constants import (
    DEFAULT_MODEL,
    MAX_ARTIFACTS,
    MAX_DELEGATE_TASK_CHARS,
    MAX_ENVELOPE_PAYLOAD_CHARS,
    MAX_LLM_CALL_DETAILS,
    MAX_REPORT_EVENTS,
    MAX_STRUCTURED_JSON_CHARS,
    MAX_SUBAGENT_RUNS,
    MAX_TOOL_CALL_ARGUMENT_BYTES,
    MAX_TOOL_CALL_DETAILS,
    MAX_WARNINGS,
    PROTOCOL_COMPACTED_TOOL_RESULT_V1,
    PROTOCOL_VERSION,
)
from protocore.contracts import (
    AgentEnvelope,
    BlobMetadata,
    CompactionSourceRef,
    ContentBlock,
    ContentBlockKind,
    EnvelopeKind,
    Event,
    ExecutionReport,
    HookEvent,
    HookResult,
    HookSpec,
    IAgentDispatch,
    IBlobStore,
    IEventStream,
    IHookManager,
    ILifecycleRegistry,
    ILLMProvider,
    IRequestTokenCounter,
    IRunStore,
    ISearchIndex,
    ISessionStore,
    ISkillStore,
    ITodoStorage,
    IToolRegistry,
    LifecycleContext,
    LifecycleDecision,
    LifecycleDisposer,
    LifecycleOutcome,
    LifecycleScope,
    LifecycleVerdict,
    LLMObservabilityContext,
    LLMRequest,
    LLMResponse,
    LoopConstants,
    Message,
    MessageRole,
    Run,
    RunState,
    RunStatus,
    RuntimeConstantsProvider,
    Session,
    SkillBundle,
    SkillIndexEntry,
    SkillManifest,
    StopReason,
    SubagentDef,
    SubagentResult,
    SubagentTask,
    TextBlock,
    ThinkingBlock,
    Todo,
    TodoStatus,
    Tool,
    ToolCall,
    ToolContext,
    ToolDefinition,
    ToolError,
    ToolInvocationError,
    ToolParameterSchema,
    ToolPolicyDenied,
    ToolPrecondition,
    ToolResult,
    ToolVisibilityPolicy,
)
from protocore.contracts.middleware import RegistrationKind
from protocore.events import EventBus, EventName
from protocore.hooks import HookManager
from protocore.ingress import EnvelopeParseError, parse_envelope, serialize_envelope
from protocore.json_utils import (
    JsonOutputParser,
    OutputParserException,
    PartialJSONParser,
    RobustStreamingJSONParser,
    StreamingJSONParser,
    is_strict_json_text,
    parse_complete_json,
    parse_complete_json_any,
    strip_thinking,
    strip_thinking_tokens,
    structured_json_candidates,
    structured_json_strings,
)
from protocore.runtime.runtime_constants import (
    StaticRuntimeConstantsProvider,
    default_runtime_constants,
)
from protocore.runtime.token_counting import (
    LanguageProfile,
    chars_per_token,
    detect_profile,
    estimate_tokens,
)
from protocore.safety import DefaultShellSafetyPolicy, ShellPolicyDecision
from protocore.tools import tool

__version__ = "2.0.0a22"

# Categorized re-export surface — intentionally NOT alphabetically sorted.
__all__ = [  # noqa: RUF022
    # Constants
    "DEFAULT_MODEL",
    "MAX_ARTIFACTS",
    "MAX_DELEGATE_TASK_CHARS",
    "MAX_ENVELOPE_PAYLOAD_CHARS",
    "MAX_LLM_CALL_DETAILS",
    "MAX_REPORT_EVENTS",
    "MAX_STRUCTURED_JSON_CHARS",
    "MAX_SUBAGENT_RUNS",
    "MAX_TOOL_CALL_ARGUMENT_BYTES",
    "MAX_TOOL_CALL_DETAILS",
    "MAX_WARNINGS",
    "PROTOCOL_COMPACTED_TOOL_RESULT_V1",
    "PROTOCOL_VERSION",
    # Contracts — interfaces
    "IAgentDispatch",
    "IBlobStore",
    "IEventStream",
    "IHookManager",
    "ILifecycleRegistry",
    "LifecycleContext",
    "LifecycleDecision",
    "LifecycleDisposer",
    "LifecycleOutcome",
    "LifecycleScope",
    "LifecycleVerdict",
    "ILLMProvider",
    "IRequestTokenCounter",
    "IRunStore",
    "ISearchIndex",
    "ISessionStore",
    "ISkillStore",
    "IToolRegistry",
    "ITodoStorage",
    "Tool",
    # Contracts — types
    "AgentEnvelope",
    "BlobMetadata",
    "CompactionSourceRef",
    "ContentBlock",
    "ContentBlockKind",
    "EnvelopeKind",
    "Event",
    "ExecutionReport",
    "HookEvent",
    "HookResult",
    "HookSpec",
    "LLMRequest",
    "LLMObservabilityContext",
    "LLMResponse",
    "Message",
    "MessageRole",
    "Run",
    "RunState",
    "RunStatus",
    "LoopConstants",
    "RuntimeConstantsProvider",
    "Session",
    "SkillBundle",
    "SkillIndexEntry",
    "SkillManifest",
    "StopReason",
    "SubagentDef",
    "SubagentResult",
    "SubagentTask",
    "TextBlock",
    "ThinkingBlock",
    "Todo",
    "TodoStatus",
    "ToolCall",
    "ToolContext",
    "ToolDefinition",
    "ToolError",
    "ToolInvocationError",
    "ToolParameterSchema",
    "ToolPolicyDenied",
    "ToolPrecondition",
    "ToolResult",
    "ToolVisibilityPolicy",
    # Runtime
    "DefaultShellSafetyPolicy",
    "EnvelopeParseError",
    "EventBus",
    "EventName",
    "HookManager",
    "RegistrationKind",
    "JsonOutputParser",
    "LanguageProfile",
    "OutputParserException",
    "PartialJSONParser",
    "RobustStreamingJSONParser",
    "ShellPolicyDecision",
    "StaticRuntimeConstantsProvider",
    "StreamingJSONParser",
    "chars_per_token",
    "default_runtime_constants",
    "detect_profile",
    "estimate_tokens",
    "is_strict_json_text",
    "parse_complete_json",
    "parse_complete_json_any",
    "parse_envelope",
    "serialize_envelope",
    "strip_thinking",
    "strip_thinking_tokens",
    "structured_json_candidates",
    "structured_json_strings",
    "tool",
    # Version
    "__version__",
]
