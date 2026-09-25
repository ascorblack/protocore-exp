"""Tests for :mod:`protocore.runtime.context.manager`."""
from __future__ import annotations

import pytest

from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.types import (
    COMPACTION_SUMMARY_METADATA_KEY,
    ImageRefBlock,
    Message,
    MessageRole,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from protocore.runtime.context.compaction import CompactionState, estimate_history_tokens
from protocore.runtime.context.manager import (
    ContextManager,
    detect_active_language,
)
from protocore.tests_support.adapters import InMemoryBlobStore, InMemoryLLMProvider


def test_detect_active_language_english() -> None:
    msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="hello world")])
    assert detect_active_language(msg) == "en"


def test_detect_active_language_russian() -> None:
    msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="Привет мир")])
    assert detect_active_language(msg) == "ru"


def test_detect_active_language_cyrillic_in_json_escape() -> None:
    msg = Message(
        role=MessageRole.user,
        content_blocks=[TextBlock(text=r'{"q": "Привет"}')],
    )
    assert detect_active_language(msg) == "ru"


def test_detect_active_language_none_message() -> None:
    assert detect_active_language(None) == "en"


def test_detect_active_language_empty_message() -> None:
    msg = Message(role=MessageRole.user, content_blocks=[])
    assert detect_active_language(msg) == "en"


def test_estimate_history_tokens_zero_for_empty() -> None:
    rc = LoopConstants()
    assert estimate_history_tokens([], rc) == 0


def test_estimate_history_tokens_scales_with_content() -> None:
    rc = LoopConstants()
    short = [Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])]
    long_ = [Message(role=MessageRole.user, content_blocks=[TextBlock(text="x" * 1000)])]
    assert estimate_history_tokens(long_, rc) > estimate_history_tokens(short, rc)


# ---------------------------------------------------------------------------
# / / exhaustive per-ContentBlock estimation
# ---------------------------------------------------------------------------


def test_estimate_counts_tool_use_arguments() -> None:
    """a large ToolUseBlock argument payload must NOT count as 0.

    ToolUseBlock has neither ``text`` nor ``content`` — only ``name`` +
    ``arguments_json``. The old attr-probe estimator counted it as 0, so a
    20 KB Write tool-call vanished from the pre-flight compaction gate.
    """
    rc = LoopConstants()
    big_args = '{"path": "x.txt", "content": "' + ("Z" * 20_000) + '"}'
    history = [
        Message(
            role=MessageRole.assistant,
            content_blocks=[
                ToolUseBlock(tool_call_id="t1", name="write_file", arguments_json=big_args),
            ],
        ),
    ]
    estimate = estimate_history_tokens(history, rc)
    # 20 KB of args at ~4 chars/token must be thousands of tokens, never 0.
    assert estimate > 1_000


def test_estimate_counts_reasoning_content() -> None:
    """Message.reasoning_content (re-emitted thinking) counts.

    reasoning_content is a top-level Message field (not a content block) that
    thinking-capable providers re-emit on the wire; it must be included.
    """
    rc = LoopConstants()
    base = Message(
        role=MessageRole.assistant,
        content_blocks=[TextBlock(text="ok")],
    )
    with_reasoning = Message(
        role=MessageRole.assistant,
        content_blocks=[TextBlock(text="ok")],
        reasoning_content="R" * 4_000,
    )
    assert estimate_history_tokens([with_reasoning], rc) > estimate_history_tokens([base], rc)
    # The reasoning text alone must contribute a non-trivial amount.
    assert estimate_history_tokens([with_reasoning], rc) > 500


def test_estimate_counts_thinking_block() -> None:
    """ThinkingBlock.text is counted (it has .text, but assert)."""
    rc = LoopConstants()
    history = [
        Message(
            role=MessageRole.assistant,
            content_blocks=[ThinkingBlock(text="T" * 4_000)],
        ),
    ]
    assert estimate_history_tokens(history, rc) > 500


def test_estimate_counts_image_ref_block_via_constant() -> None:
    """ImageRefBlock has neither text nor content; it must count
    a fixed RC-configurable image-token constant, never 0."""
    rc = LoopConstants()
    history = [
        Message(
            role=MessageRole.assistant,
            content_blocks=[ImageRefBlock(blob_ref="blob://img1")],
        ),
    ]
    estimate = estimate_history_tokens(history, rc)
    assert estimate == rc.token_count_image_tokens
    assert estimate > 0


def test_estimate_counts_tool_result_block() -> None:
    """A ToolResultBlock (has .content) is counted (regression anchor)."""
    rc = LoopConstants()
    history = [
        Message(
            role=MessageRole.tool,
            content_blocks=[ToolResultBlock(tool_call_id="t1", content="R" * 4_000)],
        ),
    ]
    assert estimate_history_tokens(history, rc) > 500


@pytest.mark.asyncio
async def test_context_manager_build_context_returns_bundle() -> None:
    rc = LoopConstants(model_context_window=4_096)
    blobs = InMemoryBlobStore()
    llm = InMemoryLLMProvider()
    mgr = ContextManager(rc=rc, blob_store=blobs, compaction_llm=llm)

    history = [
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="hello")]),
    ]
    bundle = mgr.build_context(history=history, tools=())
    assert bundle.active_language == "en"
    assert bundle.budgets.max_context == 4_096
    assert len(bundle.messages) == 1


@pytest.mark.asyncio
async def test_context_manager_detects_russian() -> None:
    rc = LoopConstants(model_context_window=4_096)
    blobs = InMemoryBlobStore()
    llm = InMemoryLLMProvider()
    mgr = ContextManager(rc=rc, blob_store=blobs, compaction_llm=llm)

    history = [
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="Привет, как дела?")]),
    ]
    bundle = mgr.build_context(history=history, tools=())
    assert bundle.active_language == "ru"


def test_context_manager_needs_compaction_when_history_exceeds_trigger() -> None:
    rc = LoopConstants(model_context_window=512, request_context_safety_tokens=0)
    blobs = InMemoryBlobStore()
    llm = InMemoryLLMProvider()
    mgr = ContextManager(rc=rc, blob_store=blobs, compaction_llm=llm)

    # 8000 chars of text should exceed 0.8 * 512 = ~410 tokens estimate.
    big = Message(
        role=MessageRole.user,
        content_blocks=[TextBlock(text="x" * 8000)],
    )
    assert mgr.needs_compaction([big]) is True


def test_context_manager_no_compaction_for_short_history() -> None:
    rc = LoopConstants(model_context_window=49_152)
    blobs = InMemoryBlobStore()
    llm = InMemoryLLMProvider()
    mgr = ContextManager(rc=rc, blob_store=blobs, compaction_llm=llm)

    small = Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    assert mgr.needs_compaction([small]) is False


def test_calibrated_estimate_governs_compaction() -> None:
    rc = LoopConstants(model_context_window=512, request_context_safety_tokens=0)
    blobs = InMemoryBlobStore()
    llm = InMemoryLLMProvider()
    mgr = ContextManager(rc=rc, blob_store=blobs, compaction_llm=llm)

    big = Message(role=MessageRole.user, content_blocks=[TextBlock(text="x" * 8000)])
    assert mgr.needs_compaction([big]) is True


# ---------------------------------------------------------------------------
# run_compaction Tier-1-failure early return must NOT report
# tokens_after=0 (false "full clear" telemetry).
# ---------------------------------------------------------------------------


class _FailingBlobStore(InMemoryBlobStore):
    """Blob store whose ``put`` always raises — forces Tier 1 to fail."""

    async def put(self, *args: object, **kwargs: object):  # type: ignore[override]
        raise RuntimeError("blob store unavailable")


@pytest.mark.asyncio
async def test_run_compaction_tier1_failure_early_return_sets_tokens_after() -> None:
    """when Tier 1 raises and the retry budget is not yet exhausted,
    run_compaction early-returns. That attempt must carry a real
    ``tokens_after`` (the current estimate), never the default 0 — otherwise
    the caller emits COMPACTION_COMPLETED with tokens_after=0 ≪ tokens_before,
    falsely telling operators the whole context was cleared."""
    from protocore.runtime.context.compaction import CompactionState

    rc = LoopConstants(
        model_context_window=4_096,
        # Generous retry budget so the failure early-returns (not raises).
        compaction_failed_max_retries=5,
        # Keep only the last turn so the big tool result is eligible for Tier 1.
        compaction_keep_recent_turns=1,
    )
    mgr = ContextManager(
        rc=rc,
        blob_store=_FailingBlobStore(),
        compaction_llm=InMemoryLLMProvider(),
    )
    # A big tool result in the eligible (non-kept) region so Tier 1 actually
    # reaches blob_store.put → raises → run_compaction early-returns.
    history = [
        Message(
            role=MessageRole.tool,
            content_blocks=[ToolResultBlock(tool_call_id="t1", content="X" * 8_000)],
        ),
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="recent")]),
    ]
    state = CompactionState()
    attempt = await mgr.run_compaction(
        history=history,
        compaction_state=state,
        tenant_id="t1",
        model_name="mock",
    )

    assert attempt.tokens_before > 0
    # The bug: tokens_after defaulted to 0. The fix sets it to the real estimate.
    assert attempt.tokens_after > 0
    assert attempt.tokens_after == attempt.tokens_before


# ---------------------------------------------------------------------------
# pin LRU + cap enforcement
# ---------------------------------------------------------------------------


def _new_manager(*, cap: int) -> ContextManager:
    rc = LoopConstants(pinned_tool_max_count=cap)
    return ContextManager(
        rc=rc,
        blob_store=InMemoryBlobStore(),
        compaction_llm=InMemoryLLMProvider(),
    )


def test_pin_tool_under_cap_does_not_evict() -> None:
    """Pinning below the cap is a plain append; no eviction returned."""
    mgr = _new_manager(cap=3)
    assert mgr.pin_tool("A") is None
    assert mgr.pin_tool("B") is None
    assert mgr.pinned_tool_names() == ("A", "B")


def test_pin_tool_at_cap_evicts_oldest() -> None:
    """F8 happy path — pinning a new tool when the list is at the cap
    evicts the LRU (oldest) entry and surfaces the evicted name."""
    mgr = _new_manager(cap=3)
    mgr.pin_tool("A")
    mgr.pin_tool("B")
    mgr.pin_tool("C")
    # Cap is 3; pinning a 4th evicts A.
    evicted = mgr.pin_tool("D")
    assert evicted == "A"
    assert mgr.pinned_tool_names() == ("B", "C", "D")


def test_pin_tool_repin_promotes_to_mru() -> None:
    """Re-pinning an already-pinned name moves it to the MRU end so
    the next eviction skips it."""
    mgr = _new_manager(cap=3)
    mgr.pin_tool("A")
    mgr.pin_tool("B")
    mgr.pin_tool("C")
    # Re-pin A → moves to end; A is no longer the LRU.
    assert mgr.pin_tool("A") is None
    assert mgr.pinned_tool_names() == ("B", "C", "A")
    # Now pinning D evicts B (the new LRU).
    assert mgr.pin_tool("D") == "B"
    assert mgr.pinned_tool_names() == ("C", "A", "D")


def test_pin_tool_saturation_evicts_in_strict_order() -> None:
    """F8 saturation: cycle 16 distinct pins through a cap of 15 and
    confirm the eviction order matches strict LRU."""
    mgr = _new_manager(cap=15)
    for i in range(15):
        assert mgr.pin_tool(f"T{i}") is None
    assert len(mgr.pinned_tool_names()) == 15
    # 16th pin evicts T0.
    assert mgr.pin_tool("T15") == "T0"
    assert "T0" not in mgr.pinned_tool_names()
    assert mgr.pinned_tool_names()[-1] == "T15"


def test_pin_tool_empty_name_is_noop() -> None:
    """Defensive guard: empty name is rejected silently (no LRU shift,
    no return)."""
    mgr = _new_manager(cap=3)
    assert mgr.pin_tool("") is None
    assert mgr.pinned_tool_names() == ()


# ---------------------------------------------------------------------------
# The fold, as the manager runs it
# ---------------------------------------------------------------------------


def _foldable_history() -> list[Message]:
    """A window that Tier 1 and Tier 2 can no longer take a byte off.

    Old summaries and operator turns, which is what a long session's window
    becomes: Tier 1 has no tool result to shed and Tier 2 refuses both kinds.
    """
    history: list[Message] = [
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="the task")]),
    ]
    for index in range(10):
        history.append(
            Message(
                role=MessageRole.user,
                content_blocks=[
                    TextBlock(text=f"<compacted-turn id='k{index}'>summary {index} " + "w " * 60 + "</compacted-turn>")
                ],
                metadata={COMPACTION_SUMMARY_METADATA_KEY: True},
            )
        )
    history.append(Message(role=MessageRole.assistant, content_blocks=[TextBlock(text="tail")]))
    history.append(Message(role=MessageRole.user, content_blocks=[TextBlock(text="recent")]))
    return history


def _folding_manager(llm: InMemoryLLMProvider) -> ContextManager:
    return ContextManager(
        rc=LoopConstants(
            model_context_window=1_024,
            request_context_safety_tokens=0,
            compaction_keep_recent_turns=2,
            compaction_fold_min_messages=4,
            compaction_fold_min_tokens=0,
            compaction_fold_keep_operator_turns=0,
        ),
        blob_store=InMemoryBlobStore(),
        compaction_llm=llm,
    )


@pytest.mark.asyncio
async def test_run_compaction_folds_what_the_first_two_tiers_cannot_touch() -> None:
    llm = InMemoryLLMProvider()
    for _ in range(4):
        llm.queue_response(text='{"summary": "ten steps, done"}')
    history = _foldable_history()

    attempt = await _folding_manager(llm).run_compaction(
        history=history,
        compaction_state=CompactionState(),
        tenant_id="t1",
        model_name="mock",
    )

    assert attempt.tier3 is not None
    assert attempt.tier3.spans_folded == 1
    assert attempt.tokens_after < attempt.tokens_before
    assert len(history) < 13


@pytest.mark.asyncio
async def test_force_compaction_folds_too() -> None:
    llm = InMemoryLLMProvider()
    for _ in range(4):
        llm.queue_response(text='{"summary": "ten steps, done"}')
    history = _foldable_history()

    attempt = await _folding_manager(llm).force_compaction(
        history=history,
        compaction_state=CompactionState(),
        tenant_id="t1",
        model_name="mock",
    )

    assert attempt.tier3 is not None
    assert attempt.tier3.spans_folded == 1


@pytest.mark.asyncio
async def test_a_fold_that_raises_does_not_abort_the_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fold shrinks a long window; it is not what makes a request fit.

    Tier 1 and Tier 2 have already freed what they could by the time it runs,
    so a fold that fails outright costs the pass nothing that was not already
    banked — and an exception escaping here would throw that away.
    """

    async def _explode(**_: object) -> None:
        raise MemoryError("the fold could not be built")

    monkeypatch.setattr("protocore.runtime.context.manager.run_tier3_fold", _explode)
    llm = InMemoryLLMProvider()
    for _ in range(4):
        llm.queue_response(text='{"summary": "folded"}')

    state = CompactionState()
    attempt = await _folding_manager(llm).run_compaction(
        history=_foldable_history(),
        compaction_state=state,
        tenant_id="t1",
        model_name="mock",
    )

    # The fold ran and failed: nothing folded, and the pass is charged as a
    # failed attempt rather than passed off as one with nothing to do.
    assert attempt.tier3 is not None
    assert attempt.tier3.spans_folded == 0
    assert attempt.tier3.spans_attempted == 1
    assert state.retry_count == 1


@pytest.mark.asyncio
async def test_a_summariser_failure_inside_the_fold_folds_nothing(
) -> None:
    """A provider that refuses every span leaves the window as it was."""

    class _Exploding(InMemoryLLMProvider):
        async def complete_structured(self, request, response_schema):  # type: ignore[no-untyped-def]
            raise RuntimeError("provider down")

    history = _foldable_history()
    attempt = await _folding_manager(_Exploding()).run_compaction(
        history=history,
        compaction_state=CompactionState(),
        tenant_id="t1",
        model_name="mock",
    )

    assert attempt.tier3 is not None
    assert attempt.tier3.spans_folded == 0
    assert len(history) == 13


@pytest.mark.asyncio
async def test_the_fold_switch_stops_it_before_the_tier_is_entered() -> None:
    llm = InMemoryLLMProvider()
    for _ in range(4):
        llm.queue_response(text='{"summary": "folded"}')
    mgr = ContextManager(
        rc=LoopConstants(
            model_context_window=1_024,
            request_context_safety_tokens=0,
            compaction_keep_recent_turns=2,
            compaction_fold_enabled=False,
            compaction_fold_min_messages=4,
            compaction_fold_min_tokens=0,
        ),
        blob_store=InMemoryBlobStore(),
        compaction_llm=llm,
    )

    attempt = await mgr.run_compaction(
        history=_foldable_history(),
        compaction_state=CompactionState(),
        tenant_id="t1",
        model_name="mock",
    )

    assert attempt.tier3 is None
