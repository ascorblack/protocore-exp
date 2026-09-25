"""end-to-end loop integration for large-file convergence.

Drives the REAL ``QueryEngine.run`` loop with a scripted LLM + byte-reporting
file tools and asserts that on a STALL the runtime FORCES the next tool — i.e.
the forced ``tool_choice`` name actually reaches ``LLMRequest.extra
['forced_tool_choice']`` on the next stream (the seam the host adapter
translates to a native ``tool_choice``). Complements the pure-module unit tests
in ``test_longfile_convergence.py`` by exercising the wiring through ``query.py``.

Scenario (the validated 'header-then-idle' shape): the model writes one
BELOW-FLOOR header that is TRUNCATED at the output cap; the salvage path saves
its partial bytes to disk and the truncation latch fires, then the model
idle-inspects via Read for ``longfile_stall_turns`` turns → the runtime forces
AppendFile; the model appends past the floor and the runtime forces
FinalizeFile. A NON-truncated below-floor write must force NOTHING (the
zero-collateral regression guard).
"""
from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest

from protocore.contracts.llm import LLMRequest, LLMStreamEvent
from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.tools import Tool, ToolContext
from protocore.contracts.types import (
    PARTIAL_ASSISTANT_ATTEMPT_METADATA_KEY,
    SYNTHETIC_RECOVERY_LONGFILE_SALVAGE,
    SYNTHETIC_RECOVERY_METADATA_KEY,
    TERMINAL_TOOL_METADATA_KEY,
    Message,
    MessageRole,
    TextBlock,
    ToolDefinition,
    ToolParameterSchema,
    ToolResult,
    ToolResultBlock,
    ToolUseBlock,
)
from protocore.runtime.loop_state import LoopState
from protocore.runtime.query_engine import QueryEngine, QueryEngineConfig
from protocore.tests_support.adapters import (
    InMemoryBlobStore,
    InMemoryEventStream,
    InMemoryHookManager,
    InMemorySkillStore,
    InMemoryToolRegistry,
)
from tests._fixtures.tool_roles import CONVENTIONAL_TOOL_ROLES

WRITE = "Write"
APPEND = "AppendFile"
FINALIZE = "FinalizeFile"
READ = "Read"
TARGET = "/workspace/big.py"


def _build_engine(*, rc: LoopConstants, llm: object) -> QueryEngine:
    return QueryEngine(
        config=QueryEngineConfig(
            tool_roles=CONVENTIONAL_TOOL_ROLES,
            run_id="run-lfc",
            tenant_id="tenant-test",
            session_id="sess-lfc",
            model_name="qwen3.6-35b-a3b",
            rc=rc,
        ),
        llm_provider=llm,  # type: ignore[arg-type]
        tool_registry=InMemoryToolRegistry(),
        event_stream=InMemoryEventStream(),
        hook_manager=InMemoryHookManager(),
        skill_store=InMemorySkillStore(),
        blob_store=InMemoryBlobStore(),
    )


class _ByteReportingFileTool(Tool):
    """Write/AppendFile/FinalizeFile that returns the PROD-faithful JSON result
    (``bytes_written`` / ``bytes_appended`` + ``bytes_total``) the convergence
    detector keys on. Read returns a minimal ack (no bytes — the idle shape)."""

    def __init__(self, name: str, files: dict[str, str], *, terminal: bool = False) -> None:
        self._name = name
        self._files = files
        self._terminal = terminal
        self.invocations: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def definition(self) -> ToolDefinition:
        props: dict[str, Any] = {"path": {"type": "string"}}
        required = ["path"]
        if self._name in (WRITE, APPEND):
            props["content"] = {"type": "string"}
            required.append("content")
        return ToolDefinition(
            name=self._name,
            description=f"{self._name} a file",
            parameters=ToolParameterSchema(properties=props, required=required),
        )

    async def invoke(self, context: ToolContext, arguments: dict[str, Any]) -> ToolResult:
        self.invocations.append(dict(arguments))
        path = str(arguments.get("path", TARGET))
        meta: dict[str, Any] = {}
        if self._name == WRITE:
            content = str(arguments.get("content", ""))
            self._files[path] = content
            payload = {"path": path, "bytes_written": len(content.encode())}
        elif self._name == APPEND:
            content = str(arguments.get("content", ""))
            self._files[path] = self._files.get(path, "") + content
            total = len(self._files[path].encode())
            payload = {
                "path": path,
                "bytes_appended": len(content.encode()),
                "bytes_total": total,
            }
        elif self._name == FINALIZE:
            payload = {"path": path, "bytes_total": len(self._files.get(path, "").encode())}
        else:  # Read / a run-terminal tool — idle, no bytes
            payload = {"ok": True}
        # A terminal tool (FinalizeFile-as-terminal OR a distinct run-terminal
        # tool like SubmitAnswer) stamps the terminal-metadata flag so the loop's
        # ``_history_tool_result_is_terminal`` predicate latches and the run ends.
        if self._terminal:
            meta[TERMINAL_TOOL_METADATA_KEY] = True
        return ToolResult(
            tool_call_id="tc",
            content=json.dumps(payload),
            is_error=False,
            metadata=meta,
        )


def _ev_tool_call(
    *,
    tool_call_id: str,
    tool_name: str,
    final_input: dict[str, Any],
    truncated_by_output_cap: bool = False,
) -> list[LLMStreamEvent]:
    stop_reason = "length" if truncated_by_output_cap else "tool_use"
    return [
        LLMStreamEvent(name="message_start", payload={}),
        LLMStreamEvent(
            name="tool_use_start",
            payload={"tool_call_id": tool_call_id, "tool_name": tool_name},
        ),
        LLMStreamEvent(
            name="tool_use_stop",
            payload={
                "tool_call_id": tool_call_id,
                "final_input": final_input,
                "truncated_by_output_cap": truncated_by_output_cap,
            },
        ),
        LLMStreamEvent(name="message_stop", payload={"stop_reason": stop_reason}),
    ]


def _ev_prose(text: str) -> list[LLMStreamEvent]:
    return [
        LLMStreamEvent(name="message_start", payload={}),
        LLMStreamEvent(name="content_block_start", payload={"kind": "text"}),
        LLMStreamEvent(name="content_block_delta", payload={"text": text}),
        LLMStreamEvent(name="content_block_stop", payload={}),
        LLMStreamEvent(name="message_stop", payload={"stop_reason": "end_turn"}),
    ]


class _ScriptedLLM:
    def __init__(self, scripts: list[list[LLMStreamEvent]]) -> None:
        self._scripts = scripts
        self._idx = 0
        self.calls: list[LLMRequest] = []

    async def stream_with_tools(self, request: LLMRequest) -> AsyncIterator[LLMStreamEvent]:  # type: ignore[no-untyped-def]
        self.calls.append(request)
        idx = min(self._idx, len(self._scripts) - 1)
        self._idx += 1
        for ev in self._scripts[idx]:
            yield ev

    def count_tokens(self, text, model=None) -> int:  # type: ignore[no-untyped-def]
        return max(1, len(text) // 4)


async def _run(engine: QueryEngine) -> None:
    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="write a big file")])
    async for _evt in engine.run(user_msg):
        pass


def _forced_choice_names(llm: _ScriptedLLM) -> list[str | None]:
    return [req.extra.get("forced_tool_choice") for req in llm.calls]


@pytest.mark.asyncio
async def test_stall_after_header_forces_appendfile_then_finalize() -> None:
    """End-to-end: a TRUNCATED header write is salvaged to disk
    (bytes land + the truncation latch is set) → idle Reads → forced AppendFile
    reaches the wire; appends past floor + idle → forced FinalizeFile.

    The header is truncated (the only path that engages the truncation driver);
    the salvage lands its partial bytes so the truncation-gated driver
    can fire."""
    rc = LoopConstants(
        model_context_window=8_192,
        longfile_convergence_enabled=True,
        longfile_stall_turns=2,
        longfile_expected_floor_bytes=4096,
        longfile_min_finalize_fraction=1.0,
        longfile_max_forced_appends=8,
        longfile_max_forced_finalizes=2,
    )
    files: dict[str, str] = {}
    tools = [
        _ByteReportingFileTool(WRITE, files),
        _ByteReportingFileTool(APPEND, files),
        _ByteReportingFileTool(FINALIZE, files),
        _ByteReportingFileTool(READ, files),
    ]
    header = "x" * 1000  # below the 4096 floor; the partial body the cap cut
    big_chunk = "y" * 4000  # pushes past floor when appended
    scripts = [
        # A TRUNCATED Write whose partial ``content`` the parser recovered.
        # Salvaged to disk (clean synthetic Write) → bytes land + latch set.
        _ev_tool_call(
            tool_call_id="t1", tool_name=WRITE,
            final_input={"path": TARGET, "content": header},
            truncated_by_output_cap=True,
        ),
        # Idle inspect twice → stall (stall_turns=2) → forced AppendFile.
        _ev_tool_call(tool_call_id="t2", tool_name=READ, final_input={"path": TARGET}),
        _ev_tool_call(tool_call_id="t3", tool_name=READ, final_input={"path": TARGET}),
        # The forced turn: model appends a big chunk → file now past floor.
        _ev_tool_call(tool_call_id="t4", tool_name=APPEND, final_input={"path": TARGET, "content": big_chunk}),
        # Idle again → now plausibly complete → forced FinalizeFile.
        _ev_tool_call(tool_call_id="t5", tool_name=READ, final_input={"path": TARGET}),
        _ev_tool_call(tool_call_id="t6", tool_name=READ, final_input={"path": TARGET}),
        # The forced finalize turn: model finalizes.
        _ev_tool_call(tool_call_id="t7", tool_name=FINALIZE, final_input={"path": TARGET}),
        _ev_prose("done"),
    ]
    llm = _ScriptedLLM(scripts)
    engine = _build_engine(rc=rc, llm=llm)
    for tool in tools:
        engine.tools.register(tool)  # type: ignore[attr-defined]

    await _run(engine)

    # The truncated header's partial bytes landed on disk via salvage.
    assert files.get(TARGET, "").startswith(header)
    # The truncation latch engaged the driver on the real target.
    assert engine._longfile_active_path == TARGET
    assert TARGET in engine._longfile_truncated_paths

    forced = _forced_choice_names(llm)
    # An AppendFile was forced on at least one stream, then a FinalizeFile.
    assert APPEND in forced, f"expected a forced AppendFile; got {forced}"
    assert FINALIZE in forced, f"expected a forced FinalizeFile; got {forced}"
    # The append index precedes the finalize index (drive content, then seal).
    assert forced.index(APPEND) < forced.index(FINALIZE)
    # The file actually grew past the floor.
    assert len(files[TARGET].encode()) >= rc.longfile_expected_floor_bytes
    assert FINALIZE in [inv_tool.name for inv_tool in tools if inv_tool.invocations and inv_tool.name == FINALIZE]


@pytest.mark.asyncio
async def test_disabled_rc_never_forces_a_tool() -> None:
    """RC kill-switch: the same idle scenario forces NOTHING when disabled."""
    rc = LoopConstants(
        model_context_window=8_192,
        longfile_convergence_enabled=False,
        longfile_stall_turns=2,
    )
    files: dict[str, str] = {}
    tools = [
        _ByteReportingFileTool(WRITE, files),
        _ByteReportingFileTool(APPEND, files),
        _ByteReportingFileTool(FINALIZE, files),
        _ByteReportingFileTool(READ, files),
    ]
    scripts = [
        _ev_tool_call(tool_call_id="t1", tool_name=WRITE, final_input={"path": TARGET, "content": "x" * 1000}),
        _ev_tool_call(tool_call_id="t2", tool_name=READ, final_input={"path": TARGET}),
        _ev_tool_call(tool_call_id="t3", tool_name=READ, final_input={"path": TARGET}),
        _ev_prose("I think the file is fine."),
    ]
    llm = _ScriptedLLM(scripts)
    engine = _build_engine(rc=rc, llm=llm)
    for tool in tools:
        engine.tools.register(tool)  # type: ignore[attr-defined]

    await _run(engine)

    forced = _forced_choice_names(llm)
    assert all(f is None for f in forced), f"disabled RC must force nothing; got {forced}"
    assert engine.state is LoopState.COMPLETED


@pytest.mark.asyncio
async def test_strong_model_self_completing_is_a_no_op() -> None:
    """A model that adds bytes every turn and self-finalizes is never forced."""
    rc = LoopConstants(
        model_context_window=8_192,
        longfile_convergence_enabled=True,
        longfile_stall_turns=2,
        longfile_expected_floor_bytes=4096,
    )
    files: dict[str, str] = {}
    tools = [
        _ByteReportingFileTool(WRITE, files),
        _ByteReportingFileTool(APPEND, files),
        _ByteReportingFileTool(FINALIZE, files),
    ]
    scripts = [
        _ev_tool_call(tool_call_id="t1", tool_name=WRITE, final_input={"path": TARGET, "content": "a" * 3000}),
        _ev_tool_call(tool_call_id="t2", tool_name=APPEND, final_input={"path": TARGET, "content": "b" * 3000}),
        _ev_tool_call(tool_call_id="t3", tool_name=FINALIZE, final_input={"path": TARGET}),
        _ev_prose("done"),
    ]
    llm = _ScriptedLLM(scripts)
    engine = _build_engine(rc=rc, llm=llm)
    for tool in tools:
        engine.tools.register(tool)  # type: ignore[attr-defined]

    await _run(engine)

    forced = _forced_choice_names(llm)
    assert all(f is None for f in forced), f"strong model must not be forced; got {forced}"
    assert engine._longfile_forced_appends == 0
    assert engine._longfile_forced_finalizes == 0


@pytest.mark.asyncio
async def test_fop_en_006_pattern_no_append_flood() -> None:
    """Zero-collateral regression guard: write several TINY files (NO truncation)
    then idle-inspect via Glob/Read — must force ZERO tools. Without the
    truncation gate this produced a 195-AppendFile self-loop; with it the driver
    never engages a non-truncated file."""
    rc = LoopConstants(
        model_context_window=8_192,
        longfile_convergence_enabled=True,
        longfile_stall_turns=2,
        longfile_expected_floor_bytes=4096,
        longfile_max_forced_appends=8,
    )
    files: dict[str, str] = {}
    tools = [
        _ByteReportingFileTool(WRITE, files),
        _ByteReportingFileTool(APPEND, files),
        _ByteReportingFileTool(FINALIZE, files),
        _ByteReportingFileTool(READ, files),
    ]
    scripts = [
        # Three tiny CLEAN files (no truncation) — alpha/beta/gamma.
        _ev_tool_call(tool_call_id="t1", tool_name=WRITE, final_input={"path": "/workspace/alpha.txt", "content": "alpha content\n"}),
        _ev_tool_call(tool_call_id="t2", tool_name=WRITE, final_input={"path": "/workspace/beta.txt", "content": "beta content\n"}),
        _ev_tool_call(tool_call_id="t3", tool_name=WRITE, final_input={"path": "/workspace/gamma.md", "content": "# gamma\n"}),
        # Then idle inspection toward an inventory (the stall shape).
        _ev_tool_call(tool_call_id="t4", tool_name=READ, final_input={"path": "/workspace/alpha.txt"}),
        _ev_tool_call(tool_call_id="t5", tool_name=READ, final_input={"path": "/workspace/beta.txt"}),
        _ev_tool_call(tool_call_id="t6", tool_name=READ, final_input={"path": "/workspace/gamma.md"}),
        _ev_prose("All three files are written and verified."),
    ]
    llm = _ScriptedLLM(scripts)
    engine = _build_engine(rc=rc, llm=llm)
    for tool in tools:
        engine.tools.register(tool)  # type: ignore[attr-defined]

    await _run(engine)

    forced = _forced_choice_names(llm)
    assert all(f is None for f in forced), f"no truncation → force nothing; got {forced}"
    assert engine._longfile_forced_appends == 0
    assert engine._longfile_forced_finalizes == 0
    # No AppendFile was ever dispatched (the self-loop fingerprint is absent).
    append_tool = next(t for t in tools if t.name == APPEND)
    assert append_tool.invocations == []
    assert engine.state is LoopState.COMPLETED


@pytest.mark.asyncio
async def test_salvage_truncated_partial_lands_bytes_and_engages() -> None:
    """Salvage happy path (end-to-end): a truncated Write WITH a recoverable
    partial ``content`` → the partial is salvaged to disk as a clean Write (bytes
    land + truncation latched) → the driver engages on the next stall."""
    rc = LoopConstants(
        model_context_window=8_192,
        longfile_convergence_enabled=True,
        longfile_stall_turns=1,
        longfile_expected_floor_bytes=4096,
        max_output_recovery_rounds=3,
    )
    files: dict[str, str] = {}
    tools = [
        _ByteReportingFileTool(WRITE, files),
        _ByteReportingFileTool(APPEND, files),
        _ByteReportingFileTool(FINALIZE, files),
        _ByteReportingFileTool(READ, files),
    ]
    partial_body = "def big():\n" + ("    x = 1\n" * 100)  # below floor, valid prefix
    scripts = [
        _ev_tool_call(
            tool_call_id="t1", tool_name=WRITE,
            final_input={"path": TARGET, "content": partial_body},
            truncated_by_output_cap=True,
        ),
        # After salvage the model IDLE-inspects (the header-then-idle shape) →
        # below floor + truncation latched + stall → the runtime FORCES AppendFile.
        _ev_tool_call(tool_call_id="t2", tool_name=READ, final_input={"path": TARGET}),
        # The forced AppendFile turn: model appends past floor.
        _ev_tool_call(tool_call_id="t3", tool_name=APPEND, final_input={"path": TARGET, "content": "z" * 5000}),
        # Idle → plausibly complete → forced FinalizeFile.
        _ev_tool_call(tool_call_id="t4", tool_name=READ, final_input={"path": TARGET}),
        _ev_tool_call(tool_call_id="t5", tool_name=FINALIZE, final_input={"path": TARGET}),
        _ev_prose("done"),
    ]
    llm = _ScriptedLLM(scripts)
    engine = _build_engine(rc=rc, llm=llm)
    for tool in tools:
        engine.tools.register(tool)  # type: ignore[attr-defined]

    await _run(engine)

    # The salvaged partial body landed on disk — bytes did NOT vanish.
    assert files.get(TARGET, "").startswith(partial_body)
    # The truncation latch + active path engaged the driver on the real target.
    assert engine._longfile_active_path == TARGET
    assert TARGET in engine._longfile_truncated_paths
    forced = _forced_choice_names(llm)
    assert APPEND in forced, f"expected a forced AppendFile after salvage; got {forced}"

    original = next(
        message
        for message in engine.history
        if any(
            isinstance(block, ToolUseBlock) and block.tool_call_id == "t1"
            for block in message.content_blocks
        )
    )
    assert original.metadata[PARTIAL_ASSISTANT_ATTEMPT_METADATA_KEY] is True
    assert SYNTHETIC_RECOVERY_METADATA_KEY not in original.metadata

    synthetic = next(
        message
        for message in engine.history
        if message.metadata.get(SYNTHETIC_RECOVERY_METADATA_KEY)
        == SYNTHETIC_RECOVERY_LONGFILE_SALVAGE
    )
    assert PARTIAL_ASSISTANT_ATTEMPT_METADATA_KEY not in synthetic.metadata
    synthetic_ids = [
        block.tool_call_id
        for block in synthetic.content_blocks
        if isinstance(block, ToolUseBlock)
    ]
    assert len(synthetic_ids) == 1
    assert synthetic_ids[0] != "t1"

    original_results = [
        block
        for message in engine.history
        for block in message.content_blocks
        if isinstance(block, ToolResultBlock) and block.tool_call_id == "t1"
    ]
    assert len(original_results) == 1
    assert original_results[0].is_error is True

    recovery_request = llm.calls[1]
    recovery_uses = [
        block.tool_call_id
        for message in recovery_request.messages
        for block in message.content_blocks
        if isinstance(block, ToolUseBlock)
    ]
    recovery_results = [
        block.tool_call_id
        for message in recovery_request.messages
        for block in message.content_blocks
        if isinstance(block, ToolResultBlock)
    ]
    assert recovery_uses.count("t1") == recovery_results.count("t1") == 1
    assert recovery_uses.count(synthetic_ids[0]) == 1
    assert recovery_results.count(synthetic_ids[0]) == 1


@pytest.mark.asyncio
async def test_salvage_ge_floor_forces_append_not_finalize() -> None:
    """A salvage whose recovered partial is ALREADY >=
    ``longfile_expected_floor_bytes`` must still force AppendFile (continue), NOT
    FinalizeFile (seal), on the subsequent stall.

    The salvage recovers only the FIRST chunk of a write the model truncated mid-
    content; the model must APPEND the rest. On the pre-fix code the synthetic
    salvage Write's ``observe_tool_result`` CLEARS the transient
    ``_longfile_last_mutation_truncated`` flag, and because the salvaged bytes are
    >= floor the file reads as ``plausibly_complete`` → the stall forces a
    PREMATURE FinalizeFile, sealing a half-written file. After the fix the salvage
    re-asserts the truncated tail so the file stays incomplete → forced AppendFile.

    FAILS on 7a16d29 (the FIRST forced tool is FinalizeFile); PASSES after the fix
    (the FIRST forced tool is AppendFile).
    """
    rc = LoopConstants(
        model_context_window=8_192,
        longfile_convergence_enabled=True,
        longfile_stall_turns=2,
        longfile_expected_floor_bytes=4096,
        longfile_min_finalize_fraction=1.0,
        longfile_max_forced_appends=8,
        longfile_max_forced_finalizes=2,
        max_output_recovery_rounds=3,
    )
    files: dict[str, str] = {}
    tools = [
        _ByteReportingFileTool(WRITE, files),
        _ByteReportingFileTool(APPEND, files),
        _ByteReportingFileTool(FINALIZE, files),
        _ByteReportingFileTool(READ, files),
    ]
    # The recovered partial is ALREADY past the 4096-byte floor (~12 KB) — the
    # realistic shape (the stand recovered ~10-16 KB before the cap), but the
    # file is still mid-content (the write was truncated).
    partial_body = "x" * (4096 * 3)
    scripts = [
        # A TRUNCATED Write whose partial body the parser recovered (>= floor).
        _ev_tool_call(
            tool_call_id="t1", tool_name=WRITE,
            final_input={"path": TARGET, "content": partial_body},
            truncated_by_output_cap=True,
        ),
        # Idle-inspect twice → stall (stall_turns=2) → the runtime must FORCE.
        _ev_tool_call(tool_call_id="t2", tool_name=READ, final_input={"path": TARGET}),
        _ev_tool_call(tool_call_id="t3", tool_name=READ, final_input={"path": TARGET}),
        # The forced turn: the model appends the rest (drive content to disk).
        _ev_tool_call(tool_call_id="t4", tool_name=APPEND, final_input={"path": TARGET, "content": "y" * 2000}),
        _ev_prose("done"),
    ]
    llm = _ScriptedLLM(scripts)
    engine = _build_engine(rc=rc, llm=llm)
    for tool in tools:
        engine.tools.register(tool)  # type: ignore[attr-defined]

    await _run(engine)

    # The salvaged partial (already >= floor) landed on disk.
    assert files.get(TARGET, "").startswith(partial_body)
    assert len(partial_body.encode()) >= rc.longfile_expected_floor_bytes
    assert engine._longfile_active_path == TARGET
    assert TARGET in engine._longfile_truncated_paths

    forced = [f for f in _forced_choice_names(llm) if f is not None]
    assert forced, "the driver must have forced at least one tool on the stall"
    # The FIRST forced action after a salvage-of-a-truncated-tail MUST be a
    # continue (AppendFile), never a premature seal (FinalizeFile).
    assert forced[0] == APPEND, (
        f"expected the first forced tool to be AppendFile (continue), got {forced}"
    )
    assert APPEND in forced


@pytest.mark.asyncio
async def test_salvage_then_clean_append_reaches_finalize() -> None:
    """Anti-wedge: the fix must NOT wedge finalization.

    After a >= floor salvage forces AppendFile, the model does a GENUINE clean
    (non-truncated) AppendFile; ``observe_tool_result`` clears the transient
    truncated-tail flag, so the file becomes ``plausibly_complete`` and a
    subsequent idle stall reaches the FinalizeFile path. Proves the re-set blocks
    finalize ONLY on the salvage-tail itself, never permanently."""
    # A wider window than the rest of this module: two floors of content plus a
    # clean append is a history the compaction trigger would otherwise sit
    # under, and this test is about the convergence driver, not compaction.
    rc = LoopConstants(
        model_context_window=32_768,
        longfile_convergence_enabled=True,
        longfile_stall_turns=2,
        longfile_expected_floor_bytes=4096,
        longfile_min_finalize_fraction=1.0,
        longfile_max_forced_appends=8,
        longfile_max_forced_finalizes=2,
        max_output_recovery_rounds=3,
    )
    files: dict[str, str] = {}
    tools = [
        _ByteReportingFileTool(WRITE, files),
        _ByteReportingFileTool(APPEND, files),
        _ByteReportingFileTool(FINALIZE, files),
        _ByteReportingFileTool(READ, files),
    ]
    partial_body = "x" * (4096 * 2)  # >= floor recovered partial (mid-content)
    scripts = [
        # Truncated Write (>= floor) → salvage; the truncated tail is re-asserted.
        _ev_tool_call(
            tool_call_id="t1", tool_name=WRITE,
            final_input={"path": TARGET, "content": partial_body},
            truncated_by_output_cap=True,
        ),
        # Idle → stall → forced AppendFile (continue).
        _ev_tool_call(tool_call_id="t2", tool_name=READ, final_input={"path": TARGET}),
        _ev_tool_call(tool_call_id="t3", tool_name=READ, final_input={"path": TARGET}),
        # The forced turn: the model does a CLEAN (non-truncated) AppendFile —
        # this clears the transient truncated-tail flag.
        _ev_tool_call(tool_call_id="t4", tool_name=APPEND, final_input={"path": TARGET, "content": "z" * 1000}),
        # Idle again → now plausibly complete → the FinalizeFile path is reachable.
        _ev_tool_call(tool_call_id="t5", tool_name=READ, final_input={"path": TARGET}),
        _ev_tool_call(tool_call_id="t6", tool_name=READ, final_input={"path": TARGET}),
        # The forced finalize turn: the model finalizes.
        _ev_tool_call(tool_call_id="t7", tool_name=FINALIZE, final_input={"path": TARGET}),
        _ev_prose("done"),
    ]
    llm = _ScriptedLLM(scripts)
    engine = _build_engine(rc=rc, llm=llm)
    for tool in tools:
        engine.tools.register(tool)  # type: ignore[attr-defined]

    await _run(engine)

    forced = [f for f in _forced_choice_names(llm) if f is not None]
    # The continue (AppendFile) precedes the seal (FinalizeFile): the fix blocks
    # finalize only on the salvage-tail, then a clean append re-opens it.
    assert APPEND in forced, f"expected a forced AppendFile after salvage; got {forced}"
    assert FINALIZE in forced, (
        f"finalize must be REACHABLE after a clean append (no wedge); got {forced}"
    )
    assert forced.index(APPEND) < forced.index(FINALIZE)
    # The first forced action is still the continue, not a premature seal.
    assert forced[0] == APPEND
    # The file actually grew past the floor and was finalized.
    assert len(files[TARGET].encode()) >= rc.longfile_expected_floor_bytes
    finalize_tool = next(t for t in tools if t.name == FINALIZE)
    assert finalize_tool.invocations, "the model's FinalizeFile must have run"


@pytest.mark.asyncio
async def test_truncation_with_no_salvageable_content_requests_smaller_chunk() -> None:
    """Truncated Write whose ``content``
    key is ABSENT (cut before any body) is NOT dispatched as an empty file;
    instead the recovery message keeps the chunk protocol and (on a repeat)
    LOWERS the header budget so the retry writes a SMALLER first chunk."""
    rc = LoopConstants(
        model_context_window=8_192,
        longfile_convergence_enabled=True,
        longfile_stall_turns=1,
        longfile_expected_floor_bytes=4096,
        max_output_recovery_rounds=3,
        write_chunk_token_budget=2000,
        truncation_chunk_recovery_repeat_budget_divisor=2,
        truncation_chunk_recovery_min_chunk_token_budget=200,
    )
    files: dict[str, str] = {}
    tools = [
        _ByteReportingFileTool(WRITE, files),
        _ByteReportingFileTool(APPEND, files),
        _ByteReportingFileTool(FINALIZE, files),
        _ByteReportingFileTool(READ, files),
    ]
    scripts = [
        # Truncated Write with NO ``content`` key (the long-en-004 shape).
        _ev_tool_call(
            tool_call_id="t1", tool_name=WRITE,
            final_input={"path": TARGET},
            truncated_by_output_cap=True,
        ),
        # Recovery asks for a chunk protocol; the model writes a smaller chunk.
        _ev_tool_call(tool_call_id="t2", tool_name=WRITE, final_input={"path": TARGET, "content": "small header\n"}),
        _ev_prose("done"),
    ]
    llm = _ScriptedLLM(scripts)
    engine = _build_engine(rc=rc, llm=llm)
    for tool in tools:
        engine.tools.register(tool)  # type: ignore[attr-defined]

    await _run(engine)

    # No empty file was dispatched: the WRITE tool was NOT invoked for an empty
    # body on the first (truncated, content-absent) call — only the model's own
    # later small-chunk Write landed.
    write_tool = next(t for t in tools if t.name == WRITE)
    assert all(inv.get("content") for inv in write_tool.invocations), (
        f"no empty-content Write should have been dispatched; got {write_tool.invocations}"
    )
    # The recovery message was injected (chunk protocol), naming the real path.
    recovery_msgs = [
        block.text
        for msg in engine.history
        if msg.role is MessageRole.user
        for block in msg.content_blocks
        if isinstance(block, TextBlock)
    ]
    assert any(TARGET in t for t in recovery_msgs), "expected a path-named recovery message"
    # The truncated content-absent path was latched (so once a chunk lands the
    # driver can engage), via the state-path (never the placeholder).
    assert TARGET in engine._longfile_truncated_paths


@pytest.mark.asyncio
async def test_repeated_salvage_uses_unique_synthetic_ids() -> None:
    """Two truncated salvageable writes in ONE run must mint DISTINCT synthetic
    tool_call_ids, so the outbound pairing repair does not drop the second
    salvage's assistant/tool blocks (which would hide a real workspace mutation
    from the model)."""
    rc = LoopConstants(
        model_context_window=8_192,
        longfile_convergence_enabled=True,
        longfile_stall_turns=2,
        longfile_expected_floor_bytes=4096,
        max_output_recovery_rounds=5,
    )
    files: dict[str, str] = {}
    tools = [
        _ByteReportingFileTool(WRITE, files),
        _ByteReportingFileTool(APPEND, files),
        _ByteReportingFileTool(FINALIZE, files),
        _ByteReportingFileTool(READ, files),
    ]
    chunk1 = "part one\n" * 50
    chunk2 = "part two\n" * 50
    scripts = [
        # First truncated write (salvageable) → salvage #1.
        _ev_tool_call(tool_call_id="t1", tool_name=WRITE, final_input={"path": TARGET, "content": chunk1}, truncated_by_output_cap=True),
        # A second truncated write (salvageable) on the SAME path → salvage #2.
        _ev_tool_call(tool_call_id="t2", tool_name=APPEND, final_input={"path": TARGET, "content": chunk2}, truncated_by_output_cap=True),
        _ev_prose("done"),
    ]
    llm = _ScriptedLLM(scripts)
    engine = _build_engine(rc=rc, llm=llm)
    for tool in tools:
        engine.tools.register(tool)  # type: ignore[attr-defined]

    await _run(engine)

    # Collect every synthetic salvage assistant tool_use id from durable history.
    from protocore.contracts.types import ToolUseBlock as _TUB
    salvage_ids = [
        block.tool_call_id
        for msg in engine.history
        for block in msg.content_blocks
        if isinstance(block, _TUB) and "longfile-salvage" in (block.tool_call_id or "")
    ]
    assert len(salvage_ids) >= 2, f"expected >=2 salvage dispatches; got {salvage_ids}"
    assert len(salvage_ids) == len(set(salvage_ids)), (
        f"salvage tool_call_ids must be UNIQUE; got duplicates: {salvage_ids}"
    )
    # Both salvaged chunks actually landed on disk (no mutation was hidden).
    assert chunk1 in files.get(TARGET, "")
    assert chunk2 in files.get(TARGET, "")


@pytest.mark.asyncio
async def test_repeat_truncated_full_write_replaces_not_duplicates() -> None:
    """Regression: the salvage must honour the model's OWN tool name, not infer
    Write-vs-AppendFile from the persisted active-file size.

    The dominant Write-spiral shape: the model re-emits the SAME oversized full
    ``Write(P, ...)`` each recovery round (it disobeys "do NOT start over with
    Write"). Round 1 salvages ``prefix1`` to disk (bytes land, path
    byte-tracked); round 2 re-issues a truncated FULL ``Write`` — a REPLACE.

    On the pre-fix code ``is_append = (active_path == P AND bytes > 0)`` is True
    in round 2, so the model's ``Write`` is salvaged as an ``AppendFile`` →
    ``prefix1 + prefix2`` — the document prefix DUPLICATED on disk, which the
    convergence driver can then seal into a corrupted artifact.

    FAILS before the fix (file == prefix1 + prefix2, the salvage op is
    AppendFile); PASSES after (file == prefix2, the salvage honours the Write).
    """
    rc = LoopConstants(
        model_context_window=8_192,
        longfile_convergence_enabled=True,
        longfile_stall_turns=2,
        longfile_expected_floor_bytes=4096,
        max_output_recovery_rounds=5,
    )
    files: dict[str, str] = {}
    tools = [
        _ByteReportingFileTool(WRITE, files),
        _ByteReportingFileTool(APPEND, files),
        _ByteReportingFileTool(FINALIZE, files),
        _ByteReportingFileTool(READ, files),
    ]
    prefix1 = "PREFIX-ONE\n" * 60
    prefix2 = "PREFIX-TWO\n" * 60
    scripts = [
        # Round 1: a truncated full Write → salvages prefix1 (bytes land + path
        # byte-tracked: active_path == TARGET, bytes > 0).
        _ev_tool_call(
            tool_call_id="t1", tool_name=WRITE,
            final_input={"path": TARGET, "content": prefix1},
            truncated_by_output_cap=True,
        ),
        # Round 2: the model RE-EMITS the same oversized full Write (start-over
        # spiral) and it truncates again → salvage. The model issued Write
        # (REPLACE), so the salvaged prefix2 must OVERWRITE the file, never append.
        _ev_tool_call(
            tool_call_id="t2", tool_name=WRITE,
            final_input={"path": TARGET, "content": prefix2},
            truncated_by_output_cap=True,
        ),
        _ev_prose("done"),
    ]
    llm = _ScriptedLLM(scripts)
    engine = _build_engine(rc=rc, llm=llm)
    for tool in tools:
        engine.tools.register(tool)  # type: ignore[attr-defined]

    await _run(engine)

    # The second salvage is a REPLACE (the model's Write) — the file holds ONLY
    # prefix2, never the duplicated prefix1 + prefix2.
    assert files.get(TARGET) == prefix2, (
        "a re-emitted truncated full Write must REPLACE on disk, not be salvaged "
        f"as an AppendFile (duplicated prefix); got {files.get(TARGET)!r}"
    )
    assert prefix1 not in files.get(TARGET, ""), "round-1 prefix must not survive a round-2 full Write"
    # Both salvages were dispatched as Write (the model's own op), not AppendFile.
    append_tool = next(t for t in tools if t.name == APPEND)
    assert append_tool.invocations == [], (
        f"a model Write must never be salvaged via AppendFile; got {append_tool.invocations}"
    )


@pytest.mark.asyncio
async def test_truncated_appendfile_with_zero_tracked_bytes_does_not_replace() -> None:
    """Regression (scenario b): a truncated ``AppendFile`` whose target already
    has content created OUTSIDE the convergence tracker (e.g. via Bash/sandbox,
    or under a different active path so the tracked byte count is 0) must be
    salvaged as an ``AppendFile`` — NEVER rewritten into a whole-file ``Write``
    that REPLACES (destroys) the pre-existing content.

    On the pre-fix code ``is_append = (active_path == P AND bytes > 0)`` is False
    (the tracker never saw the prior content land), so the model's ``AppendFile``
    is salvaged as a ``Write`` → the file is overwritten with just the chunk and
    the prior body is lost.

    FAILS before the fix (file == chunk only, prior content destroyed); PASSES
    after (file == prior content + chunk, the salvage honours the AppendFile).
    """
    rc = LoopConstants(
        model_context_window=8_192,
        longfile_convergence_enabled=True,
        longfile_stall_turns=2,
        longfile_expected_floor_bytes=4096,
        max_output_recovery_rounds=5,
    )
    files: dict[str, str] = {}
    # Pre-existing content the convergence tracker never observed (created via a
    # sandbox/Bash path) → engine._longfile_active_path / active_file_bytes stay
    # unset, so the size-inference would (wrongly) pick Write.
    prior = "EXISTING-CONTENT-FROM-BASH\n" * 40
    files[TARGET] = prior
    tools = [
        _ByteReportingFileTool(WRITE, files),
        _ByteReportingFileTool(APPEND, files),
        _ByteReportingFileTool(FINALIZE, files),
        _ByteReportingFileTool(READ, files),
    ]
    chunk = "APPENDED-CHUNK\n" * 40
    scripts = [
        # The model issues a truncated AppendFile (CONTINUE intent) against a file
        # that already has content the tracker did not see → tracked bytes == 0.
        _ev_tool_call(
            tool_call_id="t1", tool_name=APPEND,
            final_input={"path": TARGET, "content": chunk},
            truncated_by_output_cap=True,
        ),
        _ev_prose("done"),
    ]
    llm = _ScriptedLLM(scripts)
    engine = _build_engine(rc=rc, llm=llm)
    for tool in tools:
        engine.tools.register(tool)  # type: ignore[attr-defined]

    # Precondition: the tracker has NOT seen this file (the size-inference path
    # would have picked Write and destroyed the prior content).
    assert engine._longfile_active_path is None

    await _run(engine)

    # The salvage honoured the model's AppendFile → prior content survives and the
    # chunk is appended; the file was NOT replaced with just the chunk.
    assert files.get(TARGET) == prior + chunk, (
        "a truncated AppendFile with zero tracked bytes must APPEND, not be "
        f"salvaged as a whole-file Write; got {files.get(TARGET)!r}"
    )
    assert prior in files.get(TARGET, ""), "the pre-existing content must not be destroyed"
    # The salvage dispatched the model's own AppendFile, not a Write.
    write_tool = next(t for t in tools if t.name == WRITE)
    assert write_tool.invocations == [], (
        f"a model AppendFile must never be salvaged via Write; got {write_tool.invocations}"
    )


@pytest.mark.asyncio
async def test_pathless_content_present_truncation_is_not_swallowed() -> None:
    """A truncated chunkable write with RECOVERED content but NO resolvable path
    must NOT be classified as salvaged-then-dropped; it stays in the recovery
    path and the model is told to re-issue."""
    rc = LoopConstants(
        model_context_window=8_192,
        longfile_convergence_enabled=True,
        longfile_stall_turns=2,
        longfile_expected_floor_bytes=4096,
        max_output_recovery_rounds=3,
    )
    files: dict[str, str] = {}
    tools = [
        _ByteReportingFileTool(WRITE, files),
        _ByteReportingFileTool(APPEND, files),
        _ByteReportingFileTool(FINALIZE, files),
        _ByteReportingFileTool(READ, files),
    ]
    scripts = [
        # Truncated Write: content present but NO path/file_path key.
        _ev_tool_call(tool_call_id="t1", tool_name=WRITE, final_input={"content": "orphan body\n" * 30}, truncated_by_output_cap=True),
        # Model recovers with a proper Write.
        _ev_tool_call(tool_call_id="t2", tool_name=WRITE, final_input={"path": TARGET, "content": "real\n"}),
        _ev_prose("done"),
    ]
    llm = _ScriptedLLM(scripts)
    engine = _build_engine(rc=rc, llm=llm)
    for tool in tools:
        engine.tools.register(tool)  # type: ignore[attr-defined]

    await _run(engine)

    # No synthetic salvage was dispatched for the pathless call (nothing to
    # write to), and a recovery message WAS injected so the model re-issues.
    write_tool = next(t for t in tools if t.name == WRITE)
    # Only the model's own real Write landed (the pathless orphan was NOT written).
    assert files.get(TARGET) == "real\n"
    assert all("longfile-salvage" not in str(inv) for inv in write_tool.invocations)
    recovery_msgs = [
        block.text
        for msg in engine.history
        if msg.role is MessageRole.user
        for block in msg.content_blocks
        if isinstance(block, TextBlock)
    ]
    assert recovery_msgs, "a recovery message must be injected for the pathless truncation"
    # No fake "the target file" path was latched into convergence state.
    assert "the target file" not in engine._longfile_truncated_paths


# ── TERMINAL SEAL — force FinalizeFile at turn-budget exhaustion ─────────────
def _terminal_seal_tools(files: dict[str, str]) -> list[_ByteReportingFileTool]:
    return [
        _ByteReportingFileTool(WRITE, files),
        _ByteReportingFileTool(APPEND, files),
        _ByteReportingFileTool(FINALIZE, files),
        _ByteReportingFileTool(READ, files),
    ]


class _ForcingAwareScriptedLLM(_ScriptedLLM):
    """A scripted LLM that HONORS a native ``forced_tool_choice`` — when the
    runtime forces ``FinalizeFile`` on a stream (the seam the host adapter
    turns into a native ``tool_choice``), this model emits a FinalizeFile call on
    that turn instead of replaying its script. Models the prod provider behaviour
    so the loop test can prove the forced seal actually SEALS the file. Calls
    where no tool is forced replay the script (the self-continue AppendFile)."""

    def __init__(
        self, scripts: list[list[LLMStreamEvent]], *, honor_finalize: bool = True
    ) -> None:
        super().__init__(scripts)
        self._honor_finalize = honor_finalize
        self._forced_seq = 0

    async def stream_with_tools(self, request: LLMRequest) -> AsyncIterator[LLMStreamEvent]:  # type: ignore[no-untyped-def]
        if self._honor_finalize and request.extra.get("forced_tool_choice") == FINALIZE:
            self.calls.append(request)
            self._forced_seq += 1
            for ev in _ev_tool_call(
                tool_call_id=f"forced-fin-{self._forced_seq}",
                tool_name=FINALIZE,
                final_input={"path": TARGET},
            ):
                yield ev
            return
        async for ev in super().stream_with_tools(request):
            yield ev




@pytest.mark.asyncio
async def test_disabled_rc_truncation_event_has_no_salvaged_paths_key() -> None:
    """With the driver DISABLED, the tool_call_truncation_recovery event payload
    must have NO ``salvaged_paths`` key (it was absent before the feature)."""
    rc = LoopConstants(
        model_context_window=8_192,
        longfile_convergence_enabled=False,
        max_output_recovery_rounds=3,
    )
    files: dict[str, str] = {}
    tools = [
        _ByteReportingFileTool(WRITE, files),
        _ByteReportingFileTool(APPEND, files),
        _ByteReportingFileTool(FINALIZE, files),
        _ByteReportingFileTool(READ, files),
    ]
    scripts = [
        _ev_tool_call(tool_call_id="t1", tool_name=WRITE, final_input={"path": TARGET, "content": "x" * 500}, truncated_by_output_cap=True),
        _ev_tool_call(tool_call_id="t2", tool_name=WRITE, final_input={"path": TARGET, "content": "done\n"}),
        _ev_prose("done"),
    ]
    llm = _ScriptedLLM(scripts)
    engine = _build_engine(rc=rc, llm=llm)
    for tool in tools:
        engine.tools.register(tool)  # type: ignore[attr-defined]

    events: list[Any] = []
    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="write")])
    async for evt in engine.run(user_msg):
        events.append(evt)

    recovery_evts = [
        e for e in events
        if getattr(e, "payload", {}).get("reason") == "tool_call_truncation_recovery"
    ]
    assert recovery_evts, "the truncation-recovery branch must have fired"
    for e in recovery_evts:
        assert "salvaged_paths" not in e.payload, (
            f"disabled driver must NOT add salvaged_paths; got {e.payload}"
        )


# ---------------------------------------------------------------------------
# VOLUNTARY-FINISH terminal seal (run completes by the
# model calling the run-terminal tool, or a prose end_turn — NOT max-turns).
# ---------------------------------------------------------------------------

SUBMIT = "SubmitAnswer"  # the run-terminal tool (distinct from FinalizeFile)


def _voluntary_seal_tools(
    files: dict[str, str], *, with_finalize: bool = True, with_terminal: bool = True
) -> list[_ByteReportingFileTool]:
    """Tool set for the voluntary-finish seal tests. ``with_finalize=False``
    drops FinalizeFile from the surface so the seal must skip silently."""
    tools = [
        _ByteReportingFileTool(WRITE, files),
        _ByteReportingFileTool(APPEND, files),
        _ByteReportingFileTool(READ, files),
    ]
    if with_finalize:
        tools.append(_ByteReportingFileTool(FINALIZE, files))
    if with_terminal:
        tools.append(_ByteReportingFileTool(SUBMIT, files, terminal=True))
    return tools


@pytest.mark.asyncio
async def test_voluntary_seal_via_terminal_tool_seals_truncated_file() -> None:
    """A truncated header is salvaged (≥ floor after a self-continued
    append) and the model VOLUNTARILY ends the run by calling the run-terminal
    tool WITHOUT ever calling FinalizeFile. The runtime must dispatch a SYNTHETIC
    FinalizeFile to seal the file BEFORE the run completes — and still complete
    normally (the model's terminal-tool answer stands).

    FAILS on 1c6fe1fc (no voluntary-finish seal → ``_longfile_finalized`` stays
    False, FinalizeFile never runs). PASSES after the seam."""
    rc = LoopConstants(
        model_context_window=8_192,
        longfile_convergence_enabled=True,
        longfile_stall_turns=2,
        longfile_expected_floor_bytes=4096,
        longfile_min_finalize_fraction=1.0,
        longfile_max_forced_appends=8,
        longfile_max_forced_finalizes=2,
        max_output_recovery_rounds=3,
        max_turns_per_run=20,  # comfortable — the run ends voluntarily, not on budget
    )
    files: dict[str, str] = {}
    tools = _voluntary_seal_tools(files)
    header = "x" * 1000  # below floor; the partial body the cap cut
    chunk = "y" * 4000  # pushes well past the floor when appended
    scripts = [
        # Turn 1: a TRUNCATED Write whose partial body the salvage path writes to disk.
        _ev_tool_call(
            tool_call_id="t1", tool_name=WRITE,
            final_input={"path": TARGET, "content": header},
            truncated_by_output_cap=True,
        ),
        # The model self-continues with ONE big AppendFile (bytes land → no stall).
        _ev_tool_call(
            tool_call_id="t2", tool_name=APPEND,
            final_input={"path": TARGET, "content": chunk},
        ),
        # The model VOLUNTARILY ends via the run-terminal tool — WITHOUT
        # FinalizeFile. The seal must fire here before COMPLETED.
        _ev_tool_call(
            tool_call_id="t3", tool_name=SUBMIT,
            final_input={"path": TARGET},
        ),
    ]
    llm = _ScriptedLLM(scripts)
    engine = _build_engine(rc=rc, llm=llm)
    for tool in tools:
        engine.tools.register(tool)  # type: ignore[attr-defined]

    await _run(engine)

    # The truncation latch engaged the gate; the file grew past floor.
    assert engine._longfile_active_path == TARGET
    assert TARGET in engine._longfile_truncated_paths
    assert len(files[TARGET].encode()) >= rc.longfile_expected_floor_bytes
    # The voluntary-finish seal fired exactly once and SEALED the file.
    assert engine._longfile_voluntary_seal_used is True
    assert engine._longfile_finalized is True
    finalize_tool = next(t for t in tools if t.name == FINALIZE)
    assert finalize_tool.invocations, "the synthetic FinalizeFile must have run"
    assert finalize_tool.invocations[0]["path"] == TARGET
    # The run STILL completed normally (the terminal-tool answer stands).
    assert engine.state is LoopState.COMPLETED


@pytest.mark.asyncio
async def test_voluntary_seal_via_prose_end_turn_seals_truncated_file() -> None:
    """Same shape, but the model VOLUNTARILY ends with a prose
    ``end_turn`` (no run-terminal tool). The runtime must still synthesise a
    FinalizeFile to seal the truncation-gated file before COMPLETED."""
    rc = LoopConstants(
        model_context_window=8_192,
        longfile_convergence_enabled=True,
        longfile_stall_turns=2,
        longfile_expected_floor_bytes=4096,
        longfile_min_finalize_fraction=1.0,
        longfile_max_forced_finalizes=2,
        max_output_recovery_rounds=3,
        max_turns_per_run=20,
    )
    files: dict[str, str] = {}
    tools = _voluntary_seal_tools(files, with_terminal=False)
    header = "x" * 1000
    chunk = "y" * 4000
    scripts = [
        _ev_tool_call(
            tool_call_id="t1", tool_name=WRITE,
            final_input={"path": TARGET, "content": header},
            truncated_by_output_cap=True,
        ),
        _ev_tool_call(
            tool_call_id="t2", tool_name=APPEND,
            final_input={"path": TARGET, "content": chunk},
        ),
        # Voluntary prose finish — NO FinalizeFile.
        _ev_prose("The module is complete."),
    ]
    llm = _ScriptedLLM(scripts)
    engine = _build_engine(rc=rc, llm=llm)
    for tool in tools:
        engine.tools.register(tool)  # type: ignore[attr-defined]

    await _run(engine)

    assert TARGET in engine._longfile_truncated_paths
    assert engine._longfile_voluntary_seal_used is True
    assert engine._longfile_finalized is True
    finalize_tool = next(t for t in tools if t.name == FINALIZE)
    assert finalize_tool.invocations, "the synthetic FinalizeFile must have run"
    assert engine.state is LoopState.COMPLETED


@pytest.mark.asyncio
async def test_voluntary_seal_inert_on_untruncated_file() -> None:
    """Zero-collateral: the SAME voluntary-finish shape WITHOUT a truncation
    dispatches NOTHING (the file was never truncation-gated). Bit-identical."""
    rc = LoopConstants(
        model_context_window=8_192,
        longfile_convergence_enabled=True,
        longfile_stall_turns=2,
        longfile_expected_floor_bytes=4096,
        max_turns_per_run=20,
    )
    files: dict[str, str] = {}
    tools = _voluntary_seal_tools(files, with_terminal=False)
    scripts = [
        # CLEAN (non-truncated) header + clean append → never truncation-gated.
        _ev_tool_call(
            tool_call_id="t1", tool_name=WRITE,
            final_input={"path": TARGET, "content": "x" * 2000},
        ),
        _ev_tool_call(
            tool_call_id="t2", tool_name=APPEND,
            final_input={"path": TARGET, "content": "y" * 4000},
        ),
        _ev_prose("done"),
    ]
    llm = _ScriptedLLM(scripts)
    engine = _build_engine(rc=rc, llm=llm)
    for tool in tools:
        engine.tools.register(tool)  # type: ignore[attr-defined]

    await _run(engine)

    assert TARGET not in engine._longfile_truncated_paths
    assert engine._longfile_voluntary_seal_used is False
    assert engine._longfile_finalized is False
    finalize_tool = next(t for t in tools if t.name == FINALIZE)
    assert finalize_tool.invocations == []
    assert engine.state is LoopState.COMPLETED


@pytest.mark.asyncio
async def test_voluntary_seal_disabled_rc_is_noop() -> None:
    """RC kill-switch: the truncated-then-voluntary-finish shape dispatches NO
    synthetic FinalizeFile when the driver is disabled (bit-identical)."""
    rc = LoopConstants(
        model_context_window=8_192,
        longfile_convergence_enabled=False,
        longfile_stall_turns=2,
        longfile_expected_floor_bytes=4096,
        max_output_recovery_rounds=3,
        max_turns_per_run=20,
    )
    files: dict[str, str] = {}
    tools = _voluntary_seal_tools(files, with_terminal=False)
    header = "x" * 1000
    chunk = "y" * 4000
    scripts = [
        _ev_tool_call(
            tool_call_id="t1", tool_name=WRITE,
            final_input={"path": TARGET, "content": header},
            truncated_by_output_cap=True,
        ),
        _ev_tool_call(
            tool_call_id="t2", tool_name=APPEND,
            final_input={"path": TARGET, "content": chunk},
        ),
        _ev_prose("done"),
    ]
    llm = _ScriptedLLM(scripts)
    engine = _build_engine(rc=rc, llm=llm)
    for tool in tools:
        engine.tools.register(tool)  # type: ignore[attr-defined]

    await _run(engine)

    assert engine._longfile_voluntary_seal_used is False
    assert engine._longfile_finalized is False
    finalize_tool = next(t for t in tools if t.name == FINALIZE)
    assert finalize_tool.invocations == []
    assert engine.state is LoopState.COMPLETED


@pytest.mark.asyncio
async def test_voluntary_seal_skips_when_finalize_not_on_surface() -> None:
    """Skip silently — when FinalizeFile is NOT registered on the run's
    tool surface, the seal is skipped without crashing and the run completes
    (a tenant without the chunk-protocol tools simply cannot be sealed)."""
    rc = LoopConstants(
        model_context_window=8_192,
        longfile_convergence_enabled=True,
        longfile_stall_turns=2,
        longfile_expected_floor_bytes=4096,
        max_output_recovery_rounds=3,
        max_turns_per_run=20,
    )
    files: dict[str, str] = {}
    # FinalizeFile intentionally ABSENT from the surface.
    tools = _voluntary_seal_tools(files, with_finalize=False, with_terminal=False)
    header = "x" * 1000
    chunk = "y" * 4000
    scripts = [
        _ev_tool_call(
            tool_call_id="t1", tool_name=WRITE,
            final_input={"path": TARGET, "content": header},
            truncated_by_output_cap=True,
        ),
        _ev_tool_call(
            tool_call_id="t2", tool_name=APPEND,
            final_input={"path": TARGET, "content": chunk},
        ),
        _ev_prose("done"),
    ]
    llm = _ScriptedLLM(scripts)
    engine = _build_engine(rc=rc, llm=llm)
    for tool in tools:
        engine.tools.register(tool)  # type: ignore[attr-defined]

    # Must not raise even though the seal is eligible but FinalizeFile is absent.
    await _run(engine)

    assert TARGET in engine._longfile_truncated_paths
    # The seal latch never flips (skipped before the latch is set).
    assert engine._longfile_voluntary_seal_used is False
    assert engine._longfile_finalized is False
    assert engine.state is LoopState.COMPLETED


@pytest.mark.asyncio
async def test_soft_cap_annotation_does_not_hide_bytes_from_driver() -> None:
    """Regression (MED): the soft-cap warning must NOT blind the byte-parser.

    ``_dispatch_tool`` appends the ``[Tool call soft-cap warning]`` text to
    ``outcome.content`` for the model BEFORE ``_longfile.observe_tool_result``
    runs. ``observe_tool_result`` → ``_parse_byte_result`` ``json.loads``-es the
    body, so it MUST see the ORIGINAL JSON, not the annotated free text — exactly
    the host invariant that ``AppendFileOutput.next_step`` rides INSIDE the
    JSON. A per-subagent ``subagent_tool_call_soft_caps`` entry for AppendFile
    reaching its limit (the real config surface) on a long-file run is the
    trigger: from the capped call onward every AppendFile result is annotated.

    Pre-fix: the annotated AppendFile content fails ``json.loads`` → the byte
    delta is invisible → ``_longfile_active_file_bytes`` FREEZES at the pre-append
    size (false stall → spurious forced appends mid-production). Post-fix: the
    driver is fed the raw JSON, so the tracked size tracks the real on-disk size.

    FAILS before the fix (tracked size frozen at the Write's 1000 bytes);
    PASSES after (tracked size == the real cumulative 3000 bytes).
    """
    rc = LoopConstants(
        model_context_window=8_192,
        longfile_convergence_enabled=True,
        longfile_stall_turns=2,
        longfile_expected_floor_bytes=4096,
    )
    files: dict[str, str] = {}
    tools = [
        _ByteReportingFileTool(WRITE, files),
        _ByteReportingFileTool(APPEND, files),
        _ByteReportingFileTool(FINALIZE, files),
        _ByteReportingFileTool(READ, files),
    ]
    header = "x" * 1000  # binds the active path; tracked size starts at 1000
    chunk = "y" * 2000  # AppendFile → cumulative on-disk size 3000
    scripts = [
        _ev_tool_call(
            tool_call_id="t1", tool_name=WRITE,
            final_input={"path": TARGET, "content": header},
        ),
        # AppendFile #1 hits the soft cap (limit 1) → its JSON result is annotated
        # with the soft-cap warning for the model.
        _ev_tool_call(
            tool_call_id="t2", tool_name=APPEND,
            final_input={"path": TARGET, "content": chunk},
        ),
        _ev_prose("done"),
    ]
    llm = _ScriptedLLM(scripts)
    engine = _build_engine(rc=rc, llm=llm)
    # The per-subagent soft cap is read off the run's own state, where the host
    # puts it when it composes a child run. A cap of 1 annotates from the first
    # AppendFile.
    engine.run_state.tool_call_soft_caps = {APPEND: 1}
    for tool in tools:
        engine.tools.register(tool)  # type: ignore[attr-defined]

    await _run(engine)

    # Pre-condition: the annotation path actually fired — the persisted AppendFile
    # tool-result the model reads carries the soft-cap warning text. (If this ever
    # stops holding the test no longer exercises the bug.)
    append_results = [
        block
        for msg in engine.history
        for block in msg.content_blocks
        if isinstance(block, ToolResultBlock) and block.tool_call_id == "t2"
    ]
    assert append_results, "the AppendFile tool result must be in history"
    assert "[Tool call soft-cap warning]" in append_results[0].content, (
        "the soft-cap annotation must have fired (precondition for the bug)"
    )

    # The real bug: the annotated AppendFile must STILL update the tracked size.
    # On disk the file is header + chunk == 3000 bytes; the driver must track it.
    assert len(files[TARGET].encode()) == 3000
    assert engine._longfile_active_path == TARGET
    assert engine._longfile_active_file_bytes == 3000, (
        "soft-cap annotation hid the AppendFile's bytes from the convergence "
        f"driver — tracked size froze at {engine._longfile_active_file_bytes} "
        "instead of the real on-disk 3000"
    )
