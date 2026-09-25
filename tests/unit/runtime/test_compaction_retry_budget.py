"""The compaction retry budget charges failures, not passes.

Three rules, each with the incident that made it necessary:

* A pass that found nothing its profile may touch spends nothing. The
  proactive emergency pass leaves seeded history alone by design, so over a
  history made of earlier runs' turns it has nothing to do — and used to spend
  a retry doing it.
* A reactive pass (after a provider rejection) spends its own budget. The
  reactive profile is the only one that may compact seeded history, so the
  proactive failures must not use up what it is owed.
* A pass is charged at most once, however many of its tiers failed.
"""
from __future__ import annotations

from typing import Any

import pytest

from protocore.contracts.llm import LLMRequest, LLMResponse
from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.types import (
    SESSION_HISTORY_SEED_METADATA_KEY,
    Message,
    MessageRole,
    TextBlock,
)
from protocore.runtime.context.compaction import (
    CompactionExhaustedError,
    CompactionState,
)
from protocore.runtime.context.manager import ContextManager
from protocore.tests_support.adapters import InMemoryBlobStore, InMemoryLLMProvider


class _FlakySummariser(InMemoryLLMProvider):
    """A summariser that fails every call while ``failing`` is set.

    The failure is a transport-shaped one — the kind the per-unit census
    deliberately does not count against the unit.
    """

    def __init__(self) -> None:
        super().__init__()
        self.failing = False
        self.structured_calls = 0

    async def complete_structured(
        self, request: LLMRequest, response_schema: dict[str, Any]
    ) -> LLMResponse:
        self.structured_calls += 1
        if self.failing:
            raise RuntimeError("summariser unavailable")
        self.queue_response(text="condensed")
        return await super().complete_structured(request, response_schema)


def _seeded_history() -> list[Message]:
    """Turns seeded from earlier runs of the session, then the current task."""
    seed = {SESSION_HISTORY_SEED_METADATA_KEY: True}
    return [
        Message(
            role=MessageRole.user,
            content_blocks=[TextBlock(text="prior task " * 1_200)],
            metadata=seed,
        ),
        Message(
            role=MessageRole.assistant,
            content_blocks=[TextBlock(text="prior answer " * 1_200)],
            metadata=seed,
        ),
        Message(role=MessageRole.assistant, content_blocks=[TextBlock(text="ok")], metadata=seed),
        Message(
            role=MessageRole.user,
            content_blocks=[TextBlock(text="another prior task " * 1_200)],
            metadata=seed,
        ),
        Message(
            role=MessageRole.assistant,
            content_blocks=[TextBlock(text="prior answer two " * 1_200)],
            metadata=seed,
        ),
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="CURRENT TASK VERBATIM")]),
    ]


def _current_run_history() -> list[Message]:
    """The same shape with nothing seeded, so the proactive profile may touch it."""
    return [
        message.model_copy(update={"metadata": {}}) for message in _seeded_history()
    ]


def _manager(llm: InMemoryLLMProvider, rc: LoopConstants | None = None) -> ContextManager:
    return ContextManager(
        rc=rc or LoopConstants(model_context_window=65_536),
        blob_store=InMemoryBlobStore(),
        compaction_llm=llm,
    )


async def _force(
    manager: ContextManager,
    history: list[Message],
    state: CompactionState,
    *,
    reactive: bool,
) -> Any:
    return await manager.force_compaction(
        history=history,
        compaction_state=state,
        tenant_id="tenant",
        model_name="model",
        reactive=reactive,
    )


@pytest.mark.asyncio
async def test_seed_only_history_survives_idle_proactive_passes_and_a_flaky_summariser() -> None:
    """The incident: two failed reactive passes no longer end the run.

    Before, the idle proactive pass spent one retry and each failed reactive
    pass another, so under the default bound of two the second reactive
    failure raised before the reactive profile could ever succeed.
    """
    rc = LoopConstants(model_context_window=65_536)
    assert rc.compaction_failed_max_retries == 2
    llm = _FlakySummariser()
    manager = _manager(llm, rc)
    history = _seeded_history()
    state = CompactionState()

    for _ in range(2):
        before = [message.model_dump_json() for message in history]
        calls_before = llm.structured_calls
        await _force(manager, history, state, reactive=False)
        # Nothing the proactive profile may touch: no call, no change, no charge.
        assert llm.structured_calls == calls_before
        assert [message.model_dump_json() for message in history] == before
        assert state.retry_count == 0

        llm.failing = True
        await _force(manager, history, state, reactive=True)
        assert state.retry_count == 0

    assert state.reactive_retry_count == 2
    # Transport failures never enter the per-unit census: the units stay
    # eligible for the pass that can succeed.
    assert state.failed_anchor_keys == {}

    llm.failing = False
    attempt = await _force(manager, history, state, reactive=True)

    assert attempt.tokens_after < attempt.tokens_before
    assert state.reactive_retry_count == 0
    assert state.retry_count == 0
    assert history[-1].text == "CURRENT TASK VERBATIM"


@pytest.mark.asyncio
async def test_a_reactive_pass_that_keeps_failing_still_exhausts_after_the_bound() -> None:
    rc = LoopConstants(model_context_window=65_536, compaction_failed_max_retries=2)
    llm = _FlakySummariser()
    llm.failing = True
    manager = _manager(llm, rc)
    history = _seeded_history()
    state = CompactionState()

    for expected in (1, 2):
        await _force(manager, history, state, reactive=True)
        assert state.reactive_retry_count == expected

    with pytest.raises(CompactionExhaustedError, match="reactive"):
        await _force(manager, history, state, reactive=True)


@pytest.mark.asyncio
async def test_proactive_failures_do_not_spend_the_reactive_budget() -> None:
    rc = LoopConstants(model_context_window=65_536, compaction_failed_max_retries=2)
    llm = _FlakySummariser()
    llm.failing = True
    manager = _manager(llm, rc)
    history = _current_run_history()
    state = CompactionState()

    for expected in (1, 2):
        await _force(manager, history, state, reactive=False)
        assert state.retry_count == expected
    assert state.reactive_retry_count == 0

    # The shared budget is at its bound; the reactive one is untouched, and a
    # reactive pass that fails is charged to it alone.
    await _force(manager, history, state, reactive=True)
    assert state.reactive_retry_count == 1
    assert state.retry_count == 2

    llm.failing = False
    attempt = await _force(manager, history, state, reactive=True)
    assert attempt.tokens_after < attempt.tokens_before
    assert state.retry_count == 0
    assert state.reactive_retry_count == 0


@pytest.mark.asyncio
async def test_a_pass_whose_tiers_all_fail_is_charged_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _tier1_raises(**_: object) -> None:
        raise RuntimeError("blob store unavailable")

    monkeypatch.setattr(
        "protocore.runtime.context.manager.run_tier1_truncation", _tier1_raises
    )
    llm = _FlakySummariser()
    llm.failing = True
    manager = _manager(llm, LoopConstants(model_context_window=65_536))
    state = CompactionState()

    await _force(manager, _current_run_history(), state, reactive=False)

    assert llm.structured_calls > 0
    assert state.retry_count == 1


@pytest.mark.asyncio
async def test_a_routine_pass_whose_summariser_raises_is_charged_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _tier2_raises(**_: object) -> None:
        raise RuntimeError("summariser pool gone")

    monkeypatch.setattr(
        "protocore.runtime.context.manager.run_tier2_summarisation", _tier2_raises
    )
    manager = _manager(_FlakySummariser(), LoopConstants(model_context_window=65_536))
    state = CompactionState()

    await manager.run_compaction(
        history=_current_run_history(),
        compaction_state=state,
        tenant_id="tenant",
        model_name="model",
    )

    assert state.retry_count == 1


@pytest.mark.asyncio
async def test_a_routine_pass_with_nothing_eligible_spends_nothing() -> None:
    llm = _FlakySummariser()
    manager = _manager(llm, LoopConstants(model_context_window=65_536))
    state = CompactionState()

    for _ in range(5):
        await manager.run_compaction(
            history=_seeded_history(),
            compaction_state=state,
            tenant_id="tenant",
            model_name="model",
        )

    assert llm.structured_calls == 0
    assert state.retry_count == 0


def _probe_shapes() -> dict[str, list[Message]]:
    """Histories on either side of the proactive profile's eligibility line."""
    from protocore.contracts.types import (
        COMPACTION_REFERENCE_METADATA_KEY,
        COMPACTION_SUMMARY_METADATA_KEY,
        SYNTHETIC_RECOVERY_METADATA_KEY,
        ToolResultBlock,
        ToolUseBlock,
    )
    from protocore.runtime.context.compaction import _wrap_compaction_summary

    task = Message(role=MessageRole.user, content_blocks=[TextBlock(text="task")])
    tail = [
        Message(role=MessageRole.assistant, content_blocks=[TextBlock(text="latest")]),
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="go on")]),
    ]
    return {
        "empty": [],
        "seeded only": _seeded_history(),
        "current run": _current_run_history(),
        "small turns": [
            task,
            Message(role=MessageRole.assistant, content_blocks=[TextBlock(text="ok")]),
            *tail,
        ],
        "aged reasoning": [
            task,
            Message(
                role=MessageRole.assistant,
                content_blocks=[TextBlock(text="ok")],
                reasoning_content="thinking " * 200,
            ),
            *tail,
        ],
        "large tool result": [
            task,
            Message(
                role=MessageRole.assistant,
                content_blocks=[
                    ToolUseBlock(tool_call_id="c1", name="read", arguments_json="{}")
                ],
            ),
            Message(
                role=MessageRole.tool,
                content_blocks=[ToolResultBlock(tool_call_id="c1", content="r" * 60_000)],
            ),
            *tail,
        ],
        # Only the fold has work: a run of old summaries, nothing Tier 1 or
        # Tier 2 may touch.
        "old summaries only": [
            task,
            *(
                Message(
                    role=MessageRole.user,
                    content_blocks=[
                        TextBlock(
                            text=_wrap_compaction_summary(f"k{i}", f"step {i} " * 150)
                        )
                    ],
                    metadata={COMPACTION_SUMMARY_METADATA_KEY: True},
                )
                for i in range(12)
            ),
            *tail,
        ],
        "over-budget reference block": [
            Message(
                role=MessageRole.user,
                content_blocks=[TextBlock(text="environment " * 20_000)],
                metadata={COMPACTION_REFERENCE_METADATA_KEY: True},
            ),
            task,
            Message(role=MessageRole.assistant, content_blocks=[TextBlock(text="ok")]),
            *tail,
        ],
        # The large result belongs to the batch just produced; with the tail
        # protected there is nothing to do, without it there is.
        "large result in the protected tail": [
            task,
            Message(role=MessageRole.assistant, content_blocks=[TextBlock(text="ok")]),
            Message(role=MessageRole.user, content_blocks=[TextBlock(text="go on")]),
            Message(
                role=MessageRole.assistant,
                content_blocks=[
                    ToolUseBlock(tool_call_id="c9", name="read", arguments_json="{}")
                ],
            ),
            Message(
                role=MessageRole.tool,
                content_blocks=[ToolResultBlock(tool_call_id="c9", content="r" * 60_000)],
            ),
            *tail,
        ],
        "aged recovery nudge": [
            task,
            Message(
                role=MessageRole.user,
                content_blocks=[TextBlock(text="continue")],
                metadata={SYNTHETIC_RECOVERY_METADATA_KEY: "nudge"},
            ),
            Message(role=MessageRole.assistant, content_blocks=[TextBlock(text="ok")]),
            *tail,
        ],
    }


@pytest.mark.parametrize("llm_tiers", [True, False])
@pytest.mark.parametrize("protect_tail", [False, True])
@pytest.mark.parametrize("force", [True, False])
@pytest.mark.parametrize("shape", list(_probe_shapes()))
@pytest.mark.asyncio
async def test_the_probe_agrees_with_the_pass_it_stands_in_for(
    shape: str, force: bool, protect_tail: bool, llm_tiers: bool
) -> None:
    """``has_proactive_work`` is False exactly when the pass would change and call nothing."""
    from protocore.runtime.context.compaction import current_tool_batch_protect_index

    rc = LoopConstants(model_context_window=65_536, compaction_keep_recent_turns=1)
    llm = _FlakySummariser()
    manager = _manager(llm, rc)
    history = _probe_shapes()[shape]
    state = CompactionState()
    protect = current_tool_batch_protect_index(history) if protect_tail else None

    predicted = manager.has_proactive_work(
        history, state, force=force, protect_tail_from_index=protect, llm_tiers=llm_tiers
    )
    before = [message.model_dump_json() for message in history]
    if force:
        await manager.force_compaction(
            history=history,
            compaction_state=state,
            tenant_id="tenant",
            model_name="model",
            protect_tail_from_index=protect,
            llm_tiers=llm_tiers,
        )
    else:
        await manager.run_compaction(
            history=history,
            compaction_state=state,
            tenant_id="tenant",
            model_name="model",
            protect_tail_from_index=protect,
            llm_tiers=llm_tiers,
        )
    changed = [message.model_dump_json() for message in history] != before

    assert predicted == (changed or llm.structured_calls > 0), shape


def test_the_probe_shapes_cover_both_answers_for_every_tier() -> None:
    """Each shape is there to exercise one side of one tier's line."""
    manager = _manager(
        _FlakySummariser(),
        LoopConstants(model_context_window=65_536, compaction_keep_recent_turns=1),
    )
    shapes = _probe_shapes()

    def works(name: str, **kwargs: Any) -> bool:
        return manager.has_proactive_work(shapes[name], CompactionState(), force=True, **kwargs)

    assert not works("seeded only")
    assert works("old summaries only")
    assert not works("old summaries only", llm_tiers=False)
    assert works("over-budget reference block", llm_tiers=False)
    assert works("large result in the protected tail", llm_tiers=False)
    assert not works(
        "large result in the protected tail",
        llm_tiers=False,
        protect_tail_from_index=3,
    )


def test_a_written_off_unit_is_work_only_for_the_forced_pass() -> None:
    """The routine pass honours the census; the forced pass ignores it, and so do their probes."""
    rc = LoopConstants(model_context_window=65_536, compaction_keep_recent_turns=1)
    manager = _manager(_FlakySummariser(), rc)
    history = _current_run_history()
    state = CompactionState()
    # Every unit written off, as unit-shaped failures would leave the census.
    from protocore.runtime.context.compaction import _stable_turn_key

    state.failed_anchor_keys = {
        _stable_turn_key(message): rc.compaction_summary_failed_unit_max_attempts
        for message in history
    }

    assert manager.has_proactive_work(history, state, force=True)
    assert not manager.has_proactive_work(history, state, force=False)


@pytest.mark.asyncio
async def test_progress_of_either_profile_clears_both_budgets() -> None:
    manager = _manager(_FlakySummariser(), LoopConstants(model_context_window=65_536))
    state = CompactionState(retry_count=2, reactive_retry_count=2)

    attempt = await _force(manager, _current_run_history(), state, reactive=False)

    assert attempt.tokens_after < attempt.tokens_before
    assert state.retry_count == 0
    assert state.reactive_retry_count == 0
