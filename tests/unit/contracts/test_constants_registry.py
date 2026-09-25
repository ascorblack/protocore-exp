"""Contracts of the constants registry: descriptors, groups, resolution."""
from __future__ import annotations

import inspect
from typing import get_args

import pytest
from pydantic import BaseModel, Field, ValidationError

from protocore.conformance.suite import (
    _accepted_parameters,
    _protocol_member,
    _required_parameters,
    declared_members,
)
from protocore.contracts.config import (
    CROSS_GROUP_INVARIANTS,
    GROUP_INVARIANTS,
    LOOP_GROUP_KEY,
    ConfigContractError,
    ConstantGroup,
    ConstantSpec,
    ConstantsRegistry,
    ConstantValueError,
    CrossGroupInvariant,
    DuplicateConstantError,
    GroupInvariant,
    IConstantsRegistry,
    ICoreConstantsProvider,
    SpecOverlay,
    UnknownConstantError,
    UnknownInvariantError,
    build_loop_group,
    group_from_model,
)
from protocore.contracts.runtime_constants import LoopConstants


def _spec(name: str, group: str = "sample", **kwargs: object) -> ConstantSpec:
    payload: dict[str, object] = {
        "name": name,
        "kind": "int",
        "default": 1,
        "group": group,
        "owner": "the host",
    }
    payload.update(kwargs)
    return ConstantSpec(**payload)  # type: ignore[arg-type]


class TestConstantSpec:
    def test_an_editable_knob_has_a_catalogue_row(self) -> None:
        spec = _spec("page_size")
        assert spec.has_catalogue_row is True
        assert spec.editable is True

    def test_a_read_only_knob_still_has_a_row(self) -> None:
        spec = _spec("page_size", editable=False)
        assert spec.has_catalogue_row is True

    def test_a_knob_that_is_not_a_lever_has_no_row(self) -> None:
        spec = _spec("page_size", editable=False, not_a_lever="derived at run setup")
        assert spec.has_catalogue_row is False
        assert spec.not_a_lever == "derived at run setup"

    def test_no_row_and_editable_is_a_contradiction(self) -> None:
        with pytest.raises(ValidationError, match="cannot also be editable"):
            _spec("page_size", not_a_lever="derived at run setup")

    def test_a_reason_is_required_when_there_is_no_row(self) -> None:
        with pytest.raises(ValidationError, match="must carry a reason"):
            _spec("page_size", editable=False, not_a_lever="   ")

    def test_bounds_must_not_be_inverted(self) -> None:
        with pytest.raises(ValidationError, match="minimum must be <= maximum"):
            _spec("page_size", minimum=10.0, maximum=1.0)

    def test_bounds_may_coincide(self) -> None:
        assert _spec("page_size", minimum=5.0, maximum=5.0).minimum == 5.0

    def test_an_empty_enumeration_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="non-empty set"):
            _spec("mode", kind="str", default="a", allowed_values=())

    def test_a_repeated_enumeration_value_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="must not repeat"):
            _spec("mode", kind="str", default="a", allowed_values=("a", "a"))

    def test_a_nested_model_list_is_expressible_as_json(self) -> None:
        annotation = LoopConstants.model_fields["compaction_tracked_tool_names"].annotation
        spec = _spec(
            "compaction_tracked_tool_names",
            kind="json",
            default=[],
            annotation=annotation,
        )
        assert spec.kind == "json"
        assert spec.annotation is annotation
        assert spec.minimum is None and spec.maximum is None and spec.allowed_values is None

    def test_the_other_nested_model_list_is_expressible_too(self) -> None:
        annotation = LoopConstants.model_fields["result_eviction_tool_names"].annotation
        spec = _spec(
            "result_eviction_tool_names",
            kind="json",
            default=[],
            annotation=annotation,
        )
        assert spec.annotation is annotation

    def test_a_spec_is_frozen(self) -> None:
        spec = _spec("page_size")
        with pytest.raises(ValidationError):
            spec.name = "other"  # type: ignore[misc]

    def test_an_unknown_descriptor_field_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            _spec("page_size", requires_restart=True)


class TestConstantGroup:
    def test_names_and_defaults_come_from_the_specs(self) -> None:
        group = ConstantGroup(
            key="sample",
            owner="the host",
            specs=(_spec("a"), _spec("b", default=7)),
        )
        assert group.names == frozenset({"a", "b"})
        assert group.defaults() == {"a": 1, "b": 7}

    def test_a_group_resolves_its_own_names(self) -> None:
        group = ConstantGroup(key="sample", owner="the host", specs=(_spec("a"),))
        assert group.spec("a").name == "a"

    def test_a_group_is_fail_closed_on_a_foreign_name(self) -> None:
        group = ConstantGroup(key="sample", owner="the host", specs=(_spec("a"),))
        with pytest.raises(UnknownConstantError):
            group.spec("b")

    def test_a_repeated_name_within_a_group_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="declared twice"):
            ConstantGroup(key="sample", owner="the host", specs=(_spec("a"), _spec("a")))

    def test_a_spec_must_name_its_own_group(self) -> None:
        with pytest.raises(ValidationError, match="declares group"):
            ConstantGroup(key="sample", owner="the host", specs=(_spec("a", group="other"),))

    def test_a_group_is_owning_unless_it_says_otherwise(self) -> None:
        assert ConstantGroup(key="sample", owner="the host").provisional is False

    def test_an_empty_group_is_legal(self) -> None:
        assert ConstantGroup(key="sample", owner="the host").names == frozenset()


class TestCoreConstantsProvider:
    async def test_a_provider_returning_a_snapshot_satisfies_the_protocol(self) -> None:
        class Provider:
            async def get(self, tenant_id: str) -> LoopConstants:
                return LoopConstants()

        provider = Provider()
        assert isinstance(provider, ICoreConstantsProvider)
        assert (await provider.get("scope")).model_context_window > 0

    def test_an_object_without_get_is_not_a_provider(self) -> None:
        class NotAProvider:
            pass

        assert not isinstance(NotAProvider(), ICoreConstantsProvider)


# --- Relationships between constants -------------------------------------


#: A group standing in for one a surrounding layer owns. The core ships no
#: relationship that spans an owner boundary — every one of them has a side
#: outside the loop, so it is declared by the layer that sees both sides — and
#: the mechanism still has to be exercised here, against a declared example.
_OTHER_GROUP_KEY = "elsewhere"

#: The far side of the cross-group example, declared by that other group.
_OTHER_CAP = "elsewhere_outer_timeout_seconds"

_CROSS_EXAMPLE = CrossGroupInvariant(
    key="cross.outer_cap_above_reasoning_idle",
    groups=(_OTHER_GROUP_KEY, LOOP_GROUP_KEY),
    fields=(_OTHER_CAP, "llm_stream_reasoning_idle_timeout_seconds"),
    message=(
        f"{_OTHER_CAP} must be > llm_stream_reasoning_idle_timeout_seconds"
    ),
    predicate=lambda v: (
        v[_OTHER_CAP] > v["llm_stream_reasoning_idle_timeout_seconds"]
    ),
)


def _declared_registry() -> ConstantsRegistry:
    """The loop's own group, plus one standing in for a surrounding layer."""
    registry = ConstantsRegistry(cross_group_invariants=(_CROSS_EXAMPLE,))
    registry.declare(build_loop_group())
    registry.declare(
        ConstantGroup(
            key=_OTHER_GROUP_KEY,
            owner="the host",
            specs=(
                _spec(_OTHER_CAP, group=_OTHER_GROUP_KEY, kind="float", default=900.0),
                _spec("elsewhere_page_size", group=_OTHER_GROUP_KEY, default=20),
                _spec("elsewhere_max_page_size", group=_OTHER_GROUP_KEY, default=100),
            ),
            invariants=(
                GroupInvariant(
                    key="elsewhere.page_size_within_max",
                    fields=("elsewhere_page_size", "elsewhere_max_page_size"),
                    message="elsewhere_page_size must be <= elsewhere_max_page_size",
                    predicate=lambda v: v["elsewhere_page_size"] <= v["elsewhere_max_page_size"],
                ),
            ),
        )
    )
    return registry


def _declared_values(**delta: object) -> dict[str, object]:
    """Default values for every constant :func:`_declared_registry` declares."""
    values: dict[str, object] = {
        **LoopConstants().model_dump(),
        _OTHER_CAP: 900.0,
        "elsewhere_page_size": 20,
        "elsewhere_max_page_size": 100,
    }
    values.update(delta)
    return values


#: One refused value per relationship the loop itself declares, plus the two
#: declared examples: the constant to change, the value that breaks it, and the
#: identifier of the relationship that must object. Every value is legal for
#: the constant on its own — only the relationship refuses it.
REFUSED_VALUES: tuple[tuple[str, dict[str, object]], ...] = (
    ("loop.compaction_trigger_below_emergency", {"compaction_trigger_ratio": 0.95}),
    (
        "loop.request_context_safety_below_window",
        {"request_context_safety_tokens": 49_152},
    ),
    ("loop.overhead_leaves_room_for_history", {"system_prompt_max_ratio": 0.9}),
    ("loop.stall_below_idle", {"llm_stream_stall_threshold_seconds": 90.0}),
    ("loop.reasoning_idle_only_widens", {"llm_stream_reasoning_idle_timeout_seconds": 89.0}),
    (
        "loop.backoff_ceiling_above_base",
        {"resilience_backoff_base_seconds": 5.0, "resilience_backoff_max_seconds": 1.0},
    ),
)


class TestRelationships:
    def test_every_relationship_of_the_snapshot_model_is_expressed(self) -> None:
        expressed = {inv.key for group in GROUP_INVARIANTS.values() for inv in group}
        assert expressed == {key for key, _ in REFUSED_VALUES}

    def test_every_shipped_relationship_belongs_to_the_loop(self) -> None:
        assert set(GROUP_INVARIANTS) == {LOOP_GROUP_KEY}
        assert CROSS_GROUP_INVARIANTS == ()

    def test_a_cross_group_relationship_has_a_side_in_two_groups(self) -> None:
        registry = _declared_registry()
        owners = {registry.resolve(field).group for field in _CROSS_EXAMPLE.fields}
        assert owners == set(_CROSS_EXAMPLE.groups)
        assert len(owners) == 2

    @pytest.mark.parametrize(("key", "delta"), REFUSED_VALUES, ids=[k for k, _ in REFUSED_VALUES])
    def test_the_registry_refuses_the_value_the_snapshot_refuses(
        self, key: str, delta: dict[str, object]
    ) -> None:
        registry = _declared_registry()
        values = _declared_values(**delta)
        assert key in {violation.key for violation in registry.check(values)}
        with pytest.raises(ValidationError):
            LoopConstants(**delta)  # type: ignore[arg-type]

    def test_the_defaults_satisfy_every_relationship(self) -> None:
        registry = _declared_registry()
        assert registry.check(_declared_values()) == ()

    def test_a_relationship_is_skipped_when_a_constant_is_missing(self) -> None:
        registry = _declared_registry()
        values = _declared_values(compaction_trigger_ratio=0.95)
        del values["compaction_emergency_ratio"]
        assert "loop.compaction_trigger_below_emergency" not in {
            violation.key for violation in registry.check(values)
        }

    def test_a_relationship_relates_at_least_two_constants(self) -> None:
        with pytest.raises(ValidationError, match="at least two constants"):
            GroupInvariant(key="k", fields=("a",), message="m", predicate=lambda v: True)

    def test_a_constant_is_not_related_to_itself(self) -> None:
        with pytest.raises(ValidationError, match="related to itself"):
            GroupInvariant(key="k", fields=("a", "a"), message="m", predicate=lambda v: True)

    def test_a_cross_group_relationship_spans_at_least_two_groups(self) -> None:
        with pytest.raises(ValidationError, match="at least two groups"):
            CrossGroupInvariant(
                key="k", groups=("one",), fields=("a", "b"), message="m", predicate=lambda v: True
            )

    def test_a_span_does_not_list_a_group_twice(self) -> None:
        with pytest.raises(ValidationError, match="listed twice"):
            CrossGroupInvariant(
                key="k",
                groups=("one", "one"),
                fields=("a", "b"),
                message="m",
                predicate=lambda v: True,
            )

    def test_a_group_refuses_a_relationship_over_constants_it_does_not_own(self) -> None:
        with pytest.raises(ValidationError, match="does not own"):
            ConstantGroup(
                key="sample",
                owner="the host",
                specs=(_spec("a"),),
                invariants=(
                    GroupInvariant(key="k", fields=("a", "b"), message="m", predicate=lambda v: True),
                ),
            )

    def test_a_group_refuses_two_relationships_with_one_identifier(self) -> None:
        invariant = GroupInvariant(key="k", fields=("a", "b"), message="m", predicate=lambda v: True)
        with pytest.raises(ValidationError, match="declared twice"):
            ConstantGroup(
                key="sample",
                owner="the host",
                specs=(_spec("a"), _spec("b")),
                invariants=(invariant, invariant),
            )


# --- The registry itself --------------------------------------------------


def _group(key: str, *names: str, provisional: bool = False) -> ConstantGroup:
    return ConstantGroup(
        key=key,
        owner="the host",
        provisional=provisional,
        specs=tuple(_spec(name, group=key) for name in names),
    )


class TestConstantsRegistry:
    def test_resolution_is_fail_closed_on_an_unknown_name(self) -> None:
        registry = ConstantsRegistry()
        registry.declare(_group("sample", "a"))
        with pytest.raises(UnknownConstantError, match="no declared group owns"):
            registry.resolve("nowhere")

    def test_an_empty_registry_resolves_nothing(self) -> None:
        with pytest.raises(UnknownConstantError):
            ConstantsRegistry().resolve("a")
        assert ConstantsRegistry().names == frozenset()

    def test_a_declared_name_resolves_to_its_spec(self) -> None:
        registry = ConstantsRegistry()
        registry.declare(_group("sample", "a"))
        assert registry.resolve("a").group == "sample"

    def test_two_owning_groups_claiming_one_name_is_a_refusal(self) -> None:
        registry = ConstantsRegistry()
        registry.declare(_group("one", "a"))
        with pytest.raises(DuplicateConstantError, match="both claim: a"):
            registry.declare(_group("two", "a"))

    def test_a_group_key_is_declared_once(self) -> None:
        registry = ConstantsRegistry()
        registry.declare(_group("one", "a"))
        with pytest.raises(DuplicateConstantError, match="already declared"):
            registry.declare(_group("one", "b"))

    def test_an_owner_takes_a_name_from_a_provisional_group(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        registry = ConstantsRegistry()
        registry.declare(_group("fallback", "a", "b", provisional=True))
        with caplog.at_level("WARNING"):
            registry.declare(_group("owner", "a"))
        assert registry.resolve("a").group == "owner"
        assert registry.resolve("b").group == "fallback"
        assert "fallback" in caplog.text and "owner" in caplog.text

    def test_the_owner_wins_whichever_side_is_declared_first(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        registry = ConstantsRegistry()
        registry.declare(_group("owner", "a"))
        with caplog.at_level("WARNING"):
            registry.declare(_group("fallback", "a", provisional=True))
        assert registry.resolve("a").group == "owner"
        assert "fallback" in caplog.text

    def test_the_provisional_set_empties_as_owners_appear(self) -> None:
        registry = ConstantsRegistry()
        registry.declare(_group("fallback", "a", "b", provisional=True))
        assert registry.provisional_names() == frozenset({"a", "b"})
        registry.declare(_group("one", "a"))
        assert registry.provisional_names() == frozenset({"b"})
        registry.declare(_group("two", "b"))
        assert registry.provisional_names() == frozenset()

    def test_defaults_prefer_the_owning_group(self) -> None:
        registry = ConstantsRegistry()
        registry.declare(
            ConstantGroup(
                key="fallback",
                owner="the host",
                provisional=True,
                specs=(_spec("a", group="fallback", default=99),),
            )
        )
        registry.declare(_group("owner", "a"))
        assert registry.defaults()["a"] == 1

    def test_validate_reports_one_message_per_broken_relationship(self) -> None:
        registry = _declared_registry()
        values = {**LoopConstants().model_dump(), "compaction_trigger_ratio": 0.95}
        assert registry.validate(values) == (
            "compaction_trigger_ratio must be < compaction_emergency_ratio",
        )

    def test_repair_resets_the_offending_constant_and_keeps_the_rest(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        registry = _declared_registry()
        values = _declared_values(
            compaction_trigger_ratio=0.95,
            elsewhere_page_size=7,
        )
        with caplog.at_level("WARNING"):
            repaired, reset = registry.repair(values)
        assert reset == ("compaction_trigger_ratio",)
        assert repaired["compaction_trigger_ratio"] == 0.8
        assert repaired["elsewhere_page_size"] == 7
        assert registry.check(repaired) == ()
        assert "compaction_trigger_ratio" in caplog.text

    def test_repair_leaves_an_acceptable_set_untouched(self) -> None:
        registry = _declared_registry()
        values = _declared_values()
        repaired, reset = registry.repair(values)
        assert reset == ()
        assert repaired == values

    def test_repair_resets_more_than_one_constant_when_it_must(self) -> None:
        registry = _declared_registry()
        values = _declared_values(
            compaction_trigger_ratio=0.95,
            elsewhere_page_size=201,
        )
        _, reset = registry.repair(values)
        assert set(reset) == {"compaction_trigger_ratio", "elsewhere_page_size"}

    def test_repair_refuses_when_the_defaults_are_themselves_unacceptable(self) -> None:
        registry = ConstantsRegistry()
        registry.declare(
            ConstantGroup(
                key="sample",
                owner="the host",
                specs=(_spec("a", default=10), _spec("b", default=1)),
                invariants=(
                    GroupInvariant(
                        key="sample.a_below_b",
                        fields=("a", "b"),
                        message="a must be <= b",
                        predicate=lambda values: values["a"] <= values["b"],
                    ),
                ),
            )
        )
        with pytest.raises(ConfigContractError, match="defaults themselves break"):
            registry.repair({"a": 10, "b": 1})

    def test_a_cross_group_relationship_is_checked_by_the_registry_alone(self) -> None:
        registry = _declared_registry()
        values = _declared_values(**{_OTHER_CAP: 300.0})
        assert "cross.outer_cap_above_reasoning_idle" in {v.key for v in registry.check(values)}
        for group in registry.groups:
            assert group.check(values) == ()

    def test_the_cross_group_level_is_carried_unless_it_is_declined(self) -> None:
        assert ConstantsRegistry(
            cross_group_invariants=(_CROSS_EXAMPLE,)
        ).cross_group_invariants == (_CROSS_EXAMPLE,)
        assert ConstantsRegistry(cross_group_invariants=()).cross_group_invariants == ()

    def test_a_registry_that_declined_the_cross_group_level_misses_them(self) -> None:
        registry = ConstantsRegistry(cross_group_invariants=())
        registry.declare(build_loop_group())
        values = _declared_values(**{_OTHER_CAP: 300.0})
        assert registry.check(values) == ()

    def test_owning_groups_come_first(self) -> None:
        registry = ConstantsRegistry()
        registry.declare(_group("fallback", "a", provisional=True))
        registry.declare(_group("owner", "b"))
        assert [group.key for group in registry.groups] == ["owner", "fallback"]


# --- Reflecting the snapshot model into a group ---------------------------


class TestLoopGroup:
    def test_the_group_is_built_from_the_model_and_nothing_else(self) -> None:
        group = build_loop_group()
        assert group.key == LOOP_GROUP_KEY
        assert group.names == frozenset(LoopConstants.model_fields)
        assert len(group.specs) == len(LoopConstants.model_fields)

    def test_the_group_owns_its_names(self) -> None:
        assert build_loop_group().provisional is False
        assert build_loop_group(provisional=True).provisional is True

    def test_every_default_matches_the_model(self) -> None:
        snapshot = LoopConstants()
        for name, default in build_loop_group().defaults().items():
            assert default == getattr(snapshot, name)

    def test_every_constant_carries_the_model_description(self) -> None:
        group = build_loop_group()
        for name, field in LoopConstants.model_fields.items():
            assert group.spec(name).description == (field.description or "")

    def test_scalar_kinds_follow_the_annotation(self) -> None:
        group = build_loop_group()
        assert group.spec("model_context_window").kind == "int"
        assert group.spec("compaction_trigger_ratio").kind == "float"
        assert group.spec("memory_enabled").kind == "bool"

    def test_a_bound_declared_on_the_field_reaches_the_descriptor(self) -> None:
        spec = build_loop_group().spec("compaction_trigger_ratio")
        assert spec.minimum == 0.0
        assert spec.exclusive_minimum is True
        assert spec.maximum == 1.0

    def test_context_overflow_retry_ratio_has_open_bounds(self) -> None:
        spec = build_loop_group().spec("context_overflow_retry_output_ratio")
        assert spec.default == 0.5
        assert spec.minimum == 0.0
        assert spec.exclusive_minimum is True
        assert spec.maximum == 1.0
        assert spec.exclusive_maximum is True

    def test_forced_compaction_keep_window_is_dashboard_configurable(self) -> None:
        spec = build_loop_group().spec("compaction_force_keep_recent_turns")
        assert spec.default == 1
        assert spec.minimum == 0.0
        assert spec.exclusive_minimum is True
        assert spec.kind == "int"

    def test_stale_result_trimming_is_off_until_a_deployment_asks(self) -> None:
        # It shrinks the request before compaction has to run, which is a
        # choice about how much evidence the model still sees — not a default
        # an existing deployment inherits on an upgrade.
        spec = build_loop_group().spec("tool_result_stale_trim_enabled")
        assert spec.default is False
        assert spec.kind == "bool"

    def test_the_stale_trim_window_and_head_are_dashboard_configurable(self) -> None:
        fresh = build_loop_group().spec("tool_result_fresh_count")
        assert fresh.default == 6
        assert fresh.minimum == 0.0
        assert fresh.exclusive_minimum is False
        assert fresh.kind == "int"
        head = build_loop_group().spec("tool_result_stale_max_chars")
        assert head.default == 2000
        assert head.minimum == 0.0
        assert head.exclusive_minimum is True
        assert head.kind == "int"

    def test_the_stale_trim_batch_threshold_is_dashboard_configurable(self) -> None:
        spec = build_loop_group().spec("tool_result_stale_trim_batch_chars")
        assert spec.default == 40000
        assert spec.minimum == 0.0
        assert spec.exclusive_minimum is False
        assert spec.kind == "int"

    def test_the_lines_a_trim_carries_over_are_dashboard_configurable(self) -> None:
        spec = build_loop_group().spec("tool_result_stale_trim_protected_prefixes")
        assert spec.default == "Cite exactly:,Cite:,cite_as:,Source:"
        assert spec.kind == "str"

    def test_exact_token_counting_is_dashboard_configurable(self) -> None:
        group = build_loop_group()
        enabled = group.spec("exact_token_count_enabled")
        assert enabled.default is True and enabled.kind == "bool"
        margin = group.spec("exact_token_count_margin_ratio")
        assert margin.default == 0.75
        assert margin.minimum == 0.0 and margin.maximum == 1.0
        cache = group.spec("exact_token_count_cache_max_entries")
        assert cache.default == 32 and cache.kind == "int"
        assert cache.exclusive_minimum is True
        timeout = group.spec("exact_token_count_timeout_seconds")
        assert timeout.default == 5.0 and timeout.kind == "float"
        assert timeout.exclusive_minimum is True
        backoff = group.spec("exact_token_count_failure_backoff_seconds")
        assert backoff.default == 300.0 and backoff.kind == "float"
        assert backoff.minimum == 0.0 and backoff.exclusive_minimum is False

    def test_context_overflow_retry_attempt_bound_is_dashboard_configurable(self) -> None:
        spec = build_loop_group().spec("context_overflow_retry_max_attempts")
        assert spec.default == 13
        assert spec.minimum == 1.0
        assert spec.kind == "int"

    def test_compaction_trigger_turn_headroom_is_dashboard_configurable(self) -> None:
        spec = build_loop_group().spec("compaction_trigger_turn_headroom_ratio")
        assert spec.default == 0.15
        assert spec.minimum == 0.0
        assert spec.maximum == 1.0
        assert spec.exclusive_maximum is True
        assert spec.kind == "float"

    def test_summary_budget_ratio_is_dashboard_configurable(self) -> None:
        spec = build_loop_group().spec("compaction_summary_ratio")
        assert spec.default == 0.2
        assert spec.maximum == 1.0
        assert spec.kind == "float"

    def test_failed_unit_attempt_bound_is_dashboard_configurable(self) -> None:
        spec = build_loop_group().spec("compaction_summary_failed_unit_max_attempts")
        assert spec.default == 2
        assert spec.minimum == 1.0
        assert spec.kind == "int"

    def test_backoff_growth_bound_is_dashboard_configurable(self) -> None:
        spec = build_loop_group().spec("compaction_no_gain_backoff_growth_ratio")
        assert spec.default == 0.1
        assert spec.minimum == 0.0 and spec.exclusive_minimum is True
        assert spec.kind == "float"

    def test_proactive_suspension_bounds_are_dashboard_configurable(self) -> None:
        group = build_loop_group()
        visits = group.spec("compaction_proactive_suspension_iterations")
        assert visits.default == 6
        assert visits.minimum == 1.0
        assert visits.kind == "int"
        growth = group.spec("compaction_proactive_suspension_growth_ratio")
        assert growth.default == 0.1
        assert growth.minimum == 0.0 and growth.exclusive_minimum is True
        assert growth.kind == "float"

    def test_compaction_target_is_dashboard_configurable(self) -> None:
        spec = build_loop_group().spec("compaction_target_ratio")
        assert spec.default == 0.6
        assert spec.kind == "float"

    def test_output_reserve_gate_is_dashboard_configurable(self) -> None:
        spec = build_loop_group().spec("provider_reserves_output_in_context_window")
        assert spec.default is True
        assert spec.kind == "bool"

    def test_summariser_deadline_is_dashboard_configurable(self) -> None:
        spec = build_loop_group().spec("compaction_summary_timeout_seconds")
        assert spec.default == 120.0
        assert spec.kind == "float"

    def test_a_constant_without_a_bound_has_none(self) -> None:
        spec = build_loop_group().spec("continue_prompt_text")
        assert spec.minimum is None and spec.maximum is None

    def test_an_enumeration_becomes_a_closed_value_set(self) -> None:
        spec = build_loop_group().spec("memory_default_scope")
        assert spec.kind == "str"
        assert spec.allowed_values is not None
        assert set(spec.allowed_values) == set(
            get_args(LoopConstants.model_fields["memory_default_scope"].annotation)
        )

    def test_a_nested_model_list_is_reflected_as_json(self) -> None:
        group = build_loop_group()
        for name in ("compaction_tracked_tool_names", "result_eviction_tool_names"):
            spec = group.spec(name)
            assert spec.kind == "json"
            assert spec.allowed_values is None
            assert spec.annotation is LoopConstants.model_fields[name].annotation
            assert spec.default

    def test_every_constant_is_classified(self) -> None:
        kinds = {spec.kind for spec in build_loop_group().specs}
        assert kinds <= {"int", "float", "bool", "str", "json"}

    def test_the_group_carries_every_relationship_it_can_check(self) -> None:
        group = build_loop_group()
        expressed = {inv.key for owned in GROUP_INVARIANTS.values() for inv in owned}
        assert {invariant.key for invariant in group.invariants} == expressed

    def test_the_group_refuses_the_values_the_model_refuses(self) -> None:
        group = build_loop_group()
        for key, delta in REFUSED_VALUES:
            values = {**LoopConstants().model_dump(), **delta}
            broken = {violation.key for violation in group.check(values)}
            if key.startswith("cross."):
                assert broken == set()
            else:
                assert key in broken

    def test_the_group_accepts_the_defaults(self) -> None:
        assert build_loop_group().check(LoopConstants().model_dump()) == ()

    def test_the_provisional_group_resolves_every_name(self) -> None:
        registry = ConstantsRegistry(cross_group_invariants=(_CROSS_EXAMPLE,))
        registry.declare(build_loop_group())
        for name in LoopConstants.model_fields:
            assert registry.resolve(name).name == name
        with pytest.raises(UnknownConstantError):
            registry.resolve("no_such_constant")

    def test_an_owning_group_may_claim_a_name_the_provisional_group_holds(self) -> None:
        registry = ConstantsRegistry(cross_group_invariants=(_CROSS_EXAMPLE,))
        registry.declare(build_loop_group(provisional=True))
        registry.declare(_group("memory", "memory_enabled"))
        assert registry.resolve("memory_enabled").group == "memory"
        assert registry.provisional_names() == frozenset(LoopConstants.model_fields) - {
            "memory_enabled"
        }

    def test_two_owning_groups_still_collide_over_the_provisional_one(self) -> None:
        registry = ConstantsRegistry()
        registry.declare(build_loop_group(provisional=True))
        registry.declare(_group("one", "memory_enabled"))
        with pytest.raises(DuplicateConstantError):
            registry.declare(_group("two", "memory_enabled"))

    def test_the_cross_group_relationships_hold_over_the_provisional_group(self) -> None:
        registry = ConstantsRegistry(cross_group_invariants=(_CROSS_EXAMPLE,))
        registry.declare(build_loop_group())
        values = _declared_values(**{_OTHER_CAP: 300.0})
        assert "cross.outer_cap_above_reasoning_idle" in {v.key for v in registry.check(values)}

    def test_a_reflected_registry_repairs_rather_than_discards(self) -> None:
        registry = ConstantsRegistry(cross_group_invariants=(_CROSS_EXAMPLE,))
        registry.declare(build_loop_group())
        values = {
            **LoopConstants().model_dump(),
            "compaction_trigger_ratio": 0.99,
            "model_context_window": 8_192,
        }
        repaired, reset = registry.repair(values)
        assert reset == ("compaction_trigger_ratio",)
        assert repaired["model_context_window"] == 8_192
        assert LoopConstants(**repaired).compaction_trigger_ratio == 0.8

    def test_reflection_follows_a_model_it_has_never_seen(self) -> None:
        class Sample(BaseModel):
            window: int = Field(default=10, ge=1, le=99, description="how wide")

        group = build_loop_group(Sample)
        assert group.names == frozenset({"window"})
        assert group.invariants == ()
        spec = group.spec("window")
        assert (spec.kind, spec.minimum, spec.maximum, spec.description) == ("int", 1.0, 99.0, "how wide")


class TestReflectedBoundsKeepStrictness:
    def test_an_exclusive_floor_is_not_reported_as_an_accepted_value(self) -> None:
        spec = build_loop_group().spec("model_context_window")
        assert (spec.minimum, spec.exclusive_minimum) == (0.0, True)

    def test_an_exclusive_floor_and_an_inclusive_ceiling_are_told_apart(self) -> None:
        spec = build_loop_group().spec("compaction_trigger_ratio")
        assert (spec.minimum, spec.exclusive_minimum) == (0.0, True)
        assert (spec.maximum, spec.exclusive_maximum) == (1.0, False)

    def test_an_inclusive_bound_stays_inclusive(self) -> None:
        class Sample(BaseModel):
            window: int = Field(default=10, ge=1, le=99)

        spec = build_loop_group(Sample).spec("window")
        assert (spec.exclusive_minimum, spec.exclusive_maximum) == (False, False)

    def test_strictness_without_a_bound_is_a_contradiction(self) -> None:
        with pytest.raises(ValidationError, match="no meaning without a minimum"):
            _spec("page_size", exclusive_minimum=True)
        with pytest.raises(ValidationError, match="no meaning without a maximum"):
            _spec("page_size", exclusive_maximum=True)



class TestOneValueAgainstItsOwnDescriptor:
    def test_a_stored_string_is_read_as_the_declared_shape(self) -> None:
        group = build_loop_group()
        assert group.spec("model_context_window").coerce("32000") == 32_000
        assert group.spec("compaction_trigger_ratio").coerce("0.5") == 0.5
        assert group.spec("memory_enabled").coerce("yes") is True
        assert group.spec("memory_enabled").coerce("off") is False

    def test_nothing_stored_is_not_the_empty_value(self) -> None:
        spec = build_loop_group().spec("memory_default_scope")
        assert spec.coerce(None) is None
        assert spec.coerce("") is None

    def test_a_string_that_is_not_the_declared_shape_is_refused(self) -> None:
        group = build_loop_group()
        with pytest.raises(ConstantValueError, match="not an integer"):
            group.spec("model_context_window").coerce("wide")
        with pytest.raises(ConstantValueError, match="not a boolean"):
            group.spec("memory_enabled").coerce("maybe")

    def test_a_value_under_an_exclusive_floor_is_refused(self) -> None:
        with pytest.raises(ConstantValueError, match="must be > 0"):
            build_loop_group().spec("model_context_window").validate_value(0)

    def test_a_value_over_its_ceiling_is_refused(self) -> None:
        with pytest.raises(ConstantValueError, match=r"must be <= 1\.0"):
            build_loop_group().spec("compaction_trigger_ratio").validate_value(2.0)

    @pytest.mark.parametrize("value", [0.0, 1.0])
    def test_context_overflow_retry_ratio_refuses_closed_boundaries(
        self, value: float
    ) -> None:
        with pytest.raises(ValueError):
            LoopConstants(context_overflow_retry_output_ratio=value)
        with pytest.raises(ConstantValueError):
            build_loop_group().spec(
                "context_overflow_retry_output_ratio"
            ).validate_value(value)

    def test_a_value_outside_the_enumeration_is_refused(self) -> None:
        with pytest.raises(ConstantValueError, match="is not one of"):
            build_loop_group().spec("memory_default_scope").validate_value("nonsense")

    def test_a_boolean_is_not_an_integer(self) -> None:
        with pytest.raises(ConstantValueError, match="expected an integer"):
            build_loop_group().spec("model_context_window").validate_value(True)

    def test_the_registry_resolves_before_it_reads(self) -> None:
        registry = ConstantsRegistry()
        registry.declare(build_loop_group())
        assert registry.validate_value("max_turns_per_run", "500") == 500
        with pytest.raises(UnknownConstantError):
            registry.validate_value("no_such_knob", "1")

    def test_the_registry_refuses_the_same_values_a_typed_model_refuses(self) -> None:
        registry = ConstantsRegistry(cross_group_invariants=(_CROSS_EXAMPLE,))
        registry.declare(build_loop_group())
        base = LoopConstants().model_dump()
        for name, value in (
            ("model_context_window", 0),
            ("memory_default_scope", "nonsense"),
            ("compaction_trigger_ratio", 2.0),
        ):
            with pytest.raises(ValidationError):
                LoopConstants(**{**base, name: value})
            assert registry.validate({**base, name: value}), name

    def test_a_value_outside_its_own_bounds_is_repaired_before_relationships(self) -> None:
        registry = ConstantsRegistry(cross_group_invariants=(_CROSS_EXAMPLE,))
        registry.declare(build_loop_group())
        values = {**LoopConstants().model_dump(), "model_context_window": 0}
        repaired, reset = registry.repair(values)
        assert reset == ("model_context_window",)
        assert repaired["model_context_window"] == LoopConstants().model_context_window
        assert registry.check(repaired) == ()

    def test_a_name_the_registry_does_not_declare_is_left_to_its_owner(self) -> None:
        registry = ConstantsRegistry()
        registry.declare(_group("one", "max_turns_per_run"))
        assert registry.check({"max_turns_per_run": 1, "elsewhere": object()}) == ()



class TestTheRegistryContractCoversWhatIsMeasuredThroughIt:
    def test_the_declaration_half_provides_every_member_it_owns(self) -> None:
        """Reading a scope and reacting to invalidation are the host's half;
        everything else the contract declares is answered here."""
        registry = ConstantsRegistry()
        missing = [
            name
            for name in declared_members(IConstantsRegistry)
            if name not in {"snapshot", "invalidate"} and not hasattr(registry, name)
        ]
        assert not missing

    def test_the_contract_declares_every_member_the_handover_is_read_through(self) -> None:
        declared = set(declared_members(IConstantsRegistry))
        assert {
            "coerce",
            "cross_group_invariants",
            "declare",
            "defaults",
            "groups",
            "invalidate",
            "provisional_names",
            "repair",
            "resolve",
            "snapshot",
            "validate",
            "validate_value",
        } <= declared

    def test_a_registry_missing_a_published_member_is_not_the_contract(self) -> None:
        class Partial:
            def declare(self, group: ConstantGroup) -> None:
                raise NotImplementedError

            def resolve(self, name: str) -> ConstantSpec:
                raise NotImplementedError

        assert not isinstance(Partial(), IConstantsRegistry)



#: The names whose floor of zero means "no ceiling at all". An editor that
#: reads this from the descriptor rather than restating it shows the operator
#: "no limit" where the server says so, and cannot drift into showing "a limit
#: of zero" for the same value.
_ZERO_MEANS_UNLIMITED = frozenset(
    {
        "agent_max_seconds",
        "memory_max_records_per_scope",
        "parallel_read_tools_max_fanout",
    }
)


class TestFacetsADeclarationCanStateForItself:
    def test_a_floor_of_zero_that_means_no_ceiling_says_so(self) -> None:
        group = build_loop_group()
        stated = {spec.name for spec in group.specs if spec.zero_means_unlimited}
        assert _ZERO_MEANS_UNLIMITED <= stated

    def test_every_such_name_really_admits_zero(self) -> None:
        group = build_loop_group()
        for name in _ZERO_MEANS_UNLIMITED:
            assert group.spec(name).validate_value(0) == 0

    def test_a_field_states_its_own_category_and_unit(self) -> None:
        class Sample(BaseModel):
            window: int = Field(
                default=10,
                ge=0,
                json_schema_extra={
                    "category": "memory",
                    "unit": "seconds",
                    "zero_means_unlimited": True,
                },
            )

        spec = group_from_model(Sample, key="sample", owner="the host").spec("window")
        assert (spec.category, spec.unit, spec.zero_means_unlimited) == ("memory", "seconds", True)


class TestFacetsSuppliedFromOutsideTheModel:
    def test_a_name_the_deployment_does_not_offer_gets_no_row(self) -> None:
        overlay = SpecOverlay(not_a_lever={"memory_enabled": "memory is mandatory here"})
        group = build_loop_group(overlay=overlay)
        spec = group.spec("memory_enabled")
        assert spec.has_catalogue_row is False
        assert spec.editable is False
        assert group.spec("memory_default_scope").has_catalogue_row is True

    def test_a_read_only_name_keeps_its_row(self) -> None:
        group = build_loop_group(overlay=SpecOverlay(read_only=frozenset({"memory_default_scope"})))
        spec = group.spec("memory_default_scope")
        assert (spec.editable, spec.has_catalogue_row) == (False, True)

    def test_an_overlay_wins_over_what_the_field_declares(self) -> None:
        class Sample(BaseModel):
            window: int = Field(default=10, json_schema_extra={"category": "general"})

        group = group_from_model(
            Sample, key="sample", owner="the host", overlay=SpecOverlay(categories={"window": "memory"})
        )
        assert group.spec("window").category == "memory"

    def test_an_overlay_that_names_nothing_changes_nothing(self) -> None:
        assert build_loop_group(overlay=SpecOverlay()) == build_loop_group()


class TestReflectionServesAnyDeclaringModel:
    def test_a_host_group_is_one_model_and_one_call(self) -> None:
        class Sample(BaseModel):
            page_size: int = Field(default=20, ge=1, le=100, description="rows per page")

        group = group_from_model(
            Sample, key="sample", owner="the host", description="paging"
        )
        assert (group.key, group.owner, group.provisional) == ("sample", "the host", False)
        assert group.spec("page_size").group == "sample"
        assert group.spec("page_size").description == "rows per page"

    def test_a_group_carries_the_relationships_its_own_names_are_in(self) -> None:
        class Sample(BaseModel):
            compaction_trigger_ratio: float = Field(default=0.8)
            compaction_emergency_ratio: float = Field(default=0.95)

        group = group_from_model(Sample, key="sample", owner="the host")
        assert {invariant.key for invariant in group.invariants} == {
            "loop.compaction_trigger_below_emergency"
        }
        assert group.check({"compaction_trigger_ratio": 0.99, "compaction_emergency_ratio": 0.95})

    def test_the_loop_group_is_that_call_with_its_own_key(self) -> None:
        group = build_loop_group()
        assert (group.key, group.owner, group.provisional) == (LOOP_GROUP_KEY, "the loop", False)



class TestADeclarationCrossesAProcessBoundary:
    def test_a_reflected_group_round_trips_through_json(self) -> None:
        original = build_loop_group()
        restored = ConstantGroup.model_validate_json(original.model_dump_json())
        assert restored.key == original.key
        assert restored.names == original.names
        assert {i.key for i in restored.invariants} == {i.key for i in original.invariants}

    def test_a_relationship_still_refuses_after_the_crossing(self) -> None:
        restored = ConstantGroup.model_validate_json(build_loop_group().model_dump_json())
        broken = {**LoopConstants().model_dump(), "compaction_trigger_ratio": 0.99}
        assert {v.key for v in restored.check(broken)} == {
            "loop.compaction_trigger_below_emergency"
        }

    def test_the_live_annotation_stays_behind_and_its_spelling_travels(self) -> None:
        original = build_loop_group()
        restored = ConstantGroup.model_validate_json(original.model_dump_json())
        assert original.spec("memory_default_scope").annotation is not None
        assert restored.spec("memory_default_scope").annotation is None
        assert restored.spec("memory_default_scope").annotation_repr == str(
            LoopConstants.model_fields["memory_default_scope"].annotation
        )
        assert restored.spec("model_context_window").annotation_repr == "int"

    def test_a_relationship_this_build_cannot_answer_is_refused_at_declaration(self) -> None:
        group = ConstantGroup(
            key="sample",
            owner="the host",
            specs=(_spec("a"), _spec("b")),
            invariants=(
                GroupInvariant(
                    key="sample.from_somewhere_else",
                    fields=("a", "b"),
                    message="a must be <= b",
                ),
            ),
        )
        registry = ConstantsRegistry()
        with pytest.raises(UnknownInvariantError, match="knows no predicate"):
            registry.declare(group)

    def test_a_relationship_this_build_knows_is_answered_by_key_alone(self) -> None:
        wire = GroupInvariant(
            key="loop.compaction_trigger_below_emergency",
            fields=("compaction_trigger_ratio", "compaction_emergency_ratio"),
            message="compaction_trigger_ratio must be < compaction_emergency_ratio",
        )
        assert wire.predicate is None
        assert wire.check({"compaction_trigger_ratio": 0.99, "compaction_emergency_ratio": 0.95})
        assert (
            wire.check({"compaction_trigger_ratio": 0.5, "compaction_emergency_ratio": 0.95})
            is None
        )



class TestTheProviderContractIsSpelledTheWayTheCoreSpellsIt:
    def test_the_scope_argument_carries_the_name_every_other_contract_uses(self) -> None:
        """A contract that names the same argument differently from the rest of
        the core is a rename of every implementation, and the conformance
        suites assert on parameter names, so the divergence surfaces as a
        failing adapter rather than as a wording preference."""
        for protocol, member in (
            (ICoreConstantsProvider, "get"),
            (IConstantsRegistry, "snapshot"),
            (IConstantsRegistry, "invalidate"),
        ):
            assert "tenant_id" in _required_parameters(_protocol_member(protocol, member))

    def test_dropping_every_scope_is_expressible_through_the_contract(self) -> None:
        """A scope that ceases to exist is published without naming one, so an
        implementation reached only through the protocol must be able to say
        it. A required scope argument here would make the widest invalidation
        an implementation detail nobody could call."""
        invalidate = _protocol_member(IConstantsRegistry, "invalidate")
        signature = inspect.signature(invalidate)
        assert signature.parameters["tenant_id"].default is None
        assert signature.parameters["group_key"].default is None

        dropped: list[tuple[str | None, str | None]] = []

        class Registry(ConstantsRegistry):
            def invalidate(
                self, tenant_id: str | None = None, group_key: str | None = None
            ) -> None:
                dropped.append((tenant_id, group_key))

            async def snapshot(self, tenant_id: str) -> dict[str, object]:
                return {}

        registry: IConstantsRegistry = Registry()
        registry.invalidate()
        registry.invalidate("scope-1")
        assert dropped == [(None, None), ("scope-1", None)]

    def test_a_provider_written_against_the_core_convention_conforms(self) -> None:
        class Provider:
            async def get(self, tenant_id: str) -> LoopConstants:
                return LoopConstants()

        assert isinstance(Provider(), ICoreConstantsProvider)
        declared = _required_parameters(_protocol_member(ICoreConstantsProvider, "get"))
        accepted, _ = _accepted_parameters(Provider().get)
        assert not declared - accepted
