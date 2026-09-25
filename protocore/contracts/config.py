"""Constants registry contracts — the shape a tunable knob has, and who owns it.

A deployment's tunable surface is not one flat model. It is a set of
**groups**, each declared by the layer that actually reads its values: the
loop's own thresholds are declared by the core, and everything a surrounding
layer reads is declared by that layer. This module owns the vocabulary both
sides speak:

* :class:`ConstantSpec` — one knob's descriptor: what it is, what it means,
  what an operator may set it to, and whether an operator may see it at all.
* :class:`ConstantGroup` — a set of specs with one owner and one key.
* :class:`ICoreConstantsProvider` — how the core asks for the loop snapshot.
* :class:`IConstantsRegistry` — how a group is declared, how a name is
  resolved (fail-closed), and how a scope's values are read back.

Why the descriptor stops where it does. Two facts about a knob deliberately
live elsewhere, because a second declaration of one fact drifts from the
first: whether changing it requires a restart, and which capability toggle it
belongs to. Both are properties of the place that consumes the value, known
only to that code, and they are joined onto the catalogue as an overlay rather
than restated here.

Three visibility states, not two:

* ``editable=True`` — the knob has a catalogue row and an operator may set it;
* ``editable=False`` — it has a row, shown but refused on write;
* ``not_a_lever="<reason>"`` — it has **no row at all**. Not a knob: a value
  the running system derives or owns outright, whose appearance in an editor
  would be an invitation to break the deployment.

Import boundary: pure ``pydantic`` / stdlib. Guard —
``tests/test_core_import_boundary.py``.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping
from typing import Any, Literal, NamedTuple, Protocol, Self, get_args, get_origin, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.fields import FieldInfo

from protocore.contracts.runtime_constants import LoopConstants

logger = logging.getLogger(__name__)

#: The five shapes a constant's value can take on the wire. ``json`` is the
#: escape hatch for the handful of knobs that are neither scalar nor an
#: enumeration — lists of nested models, whose admissible values are described
#: by :attr:`ConstantSpec.annotation` and by nothing narrower.
ConstantKind = Literal["int", "float", "bool", "str", "json"]

#: The words a stored boolean may be written with. A value outside either set
#: is refused rather than read as false: a typo that silently disables a knob
#: is indistinguishable from an operator who meant to disable it.
_TRUE_WORDS = frozenset({"1", "true", "yes", "on"})
_FALSE_WORDS = frozenset({"0", "false", "no", "off"})


def _shape(value: Any) -> str:
    """The name of a value's type, for a refusal message that names it."""
    return type(value).__name__


class ConfigContractError(Exception):
    """Base for every refusal this module raises."""


class UnknownConstantError(ConfigContractError, KeyError):
    """A name was resolved that no declared group owns.

    Resolution is fail-closed by contract. A typed model refuses an unknown
    key loudly; a registry that answered ``None`` instead would turn a
    misspelled knob into a silent default, which is the failure a registry
    exists to prevent.
    """


class DuplicateConstantError(ConfigContractError):
    """Two owning groups claim the same constant name."""


class UnknownInvariantError(ConfigContractError):
    """A relationship arrived by key with no predicate this build knows.

    Evaluating nothing would be the silent outcome: the relationship reads as
    satisfied for every set of values, and the refusal it exists to produce
    never happens.
    """


class ConstantValueError(ConfigContractError, ValueError):
    """A value is not admissible for the constant it was offered for.

    Raised both when a stored string cannot be read as the declared shape and
    when the resulting value falls outside the bounds or the enumeration the
    descriptor carries. It is one error rather than two because a caller
    treats them the same way: refuse the write, or drop the field and fall
    back to the default.
    """


class ConstantSpec(BaseModel):
    """Everything knowable about one tunable value except where it is read.

    ``minimum`` / ``maximum`` / ``allowed_values`` are the operator-facing
    bounds: an editor renders them, and a write is refused against them before
    the value ever reaches a snapshot. ``zero_means_unlimited`` is the one
    piece of semantics a numeric bound cannot carry — a floor of zero that
    means "no ceiling at all" rather than "the smallest ceiling".
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    name: str = Field(min_length=1, description="Constant name, unique across all owning groups.")
    kind: ConstantKind = Field(description="Wire shape of the value.")
    annotation: Any = Field(
        default=None,
        exclude=True,
        description=(
            "The declaring model's own annotation, when there is one. The "
            "escape hatch for a value no scalar bound describes. Held as the "
            "live object, so it is the one part of a descriptor that cannot "
            "cross a process boundary; :attr:`annotation_repr` carries what "
            "can be said about it in text."
        ),
    )
    annotation_repr: str | None = Field(
        default=None,
        description="How the declaring model spells the annotation, for a reader elsewhere.",
    )
    default: Any = Field(default=None, description="Value in force where no override exists.")
    description: str = Field(default="", description="Operator-facing explanation.")
    group: str = Field(min_length=1, description="Key of the group that owns this constant.")
    owner: str = Field(min_length=1, description="Name of the layer that reads it.")
    unit: str | None = Field(default=None, description="Unit of measure, e.g. 'seconds', 'tokens'.")
    minimum: float | None = Field(default=None, description="Lowest accepted value.")
    maximum: float | None = Field(default=None, description="Highest accepted value.")
    exclusive_minimum: bool = Field(
        default=False, description="Whether ``minimum`` is itself refused rather than accepted."
    )
    exclusive_maximum: bool = Field(
        default=False, description="Whether ``maximum`` is itself refused rather than accepted."
    )
    allowed_values: tuple[str, ...] | None = Field(
        default=None, description="Closed set of accepted values, when the knob is an enumeration."
    )
    zero_means_unlimited: bool = Field(
        default=False, description="Whether zero disables the limit rather than setting it to nothing."
    )
    editable: bool = Field(default=True, description="Whether an operator may write this knob.")
    not_a_lever: str | None = Field(
        default=None,
        description=(
            "Reason the knob has no catalogue row at all. Distinct from "
            "``editable=False``, which still shows a row and refuses the write."
        ),
    )
    category: str | None = Field(default=None, description="Grouping hint for an editor's navigation.")

    @property
    def has_catalogue_row(self) -> bool:
        """Whether this constant is offered to an operator at all."""
        return self.not_a_lever is None

    def coerce(self, raw: str | None) -> Any:
        """Read a stored string as this constant's declared shape.

        A store keeps every value as text, and both processes that read one —
        the write path checking an operator's edit and the build path
        assembling a snapshot — need the same reading of it, or a value the
        editor accepted is a value the snapshot silently drops. An empty
        string and ``None`` mean "no value stored", not "the empty value", so
        they come back as ``None``; anything else that cannot be read as the
        declared shape is refused rather than guessed at.
        """
        if raw is None or raw == "":
            return None
        if self.kind == "bool":
            lowered = raw.strip().lower()
            if lowered in _TRUE_WORDS:
                return True
            if lowered in _FALSE_WORDS:
                return False
            raise ConstantValueError(f"{self.name}: {raw!r} is not a boolean")
        if self.kind == "int":
            try:
                return int(raw)
            except ValueError as exc:
                raise ConstantValueError(f"{self.name}: {raw!r} is not an integer") from exc
        if self.kind == "float":
            try:
                return float(raw)
            except ValueError as exc:
                raise ConstantValueError(f"{self.name}: {raw!r} is not a number") from exc
        if self.kind == "json":
            try:
                return json.loads(raw)
            except ValueError as exc:
                raise ConstantValueError(f"{self.name}: {raw!r} is not valid JSON") from exc
        return raw

    def validate_value(self, value: Any) -> Any:
        """Return ``value`` if this constant admits it, else raise.

        The check a relationship cannot make: one value against its own
        declared shape, its own bounds and its own enumeration. Without it a
        stored value that breaks a field bound reaches the snapshot build
        unguarded, and the build is the wrong place to find out — it runs for
        every turn of the scope, and it has no way to answer an operator.
        """
        if self.kind == "bool":
            if not isinstance(value, bool):
                raise ConstantValueError(f"{self.name}: expected a boolean, got {_shape(value)}")
            return value
        if self.kind == "int":
            if isinstance(value, bool) or not isinstance(value, int):
                raise ConstantValueError(f"{self.name}: expected an integer, got {_shape(value)}")
            return self._within_bounds(value)
        if self.kind == "float":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ConstantValueError(f"{self.name}: expected a number, got {_shape(value)}")
            return self._within_bounds(float(value))
        if self.kind == "str":
            if not isinstance(value, str):
                raise ConstantValueError(f"{self.name}: expected a string, got {_shape(value)}")
            if self.allowed_values is not None and value not in self.allowed_values:
                raise ConstantValueError(
                    f"{self.name}: {value!r} is not one of "
                    f"{', '.join(self.allowed_values)}"
                )
            return value
        return value

    def _within_bounds(self, value: float) -> Any:
        if self.minimum is not None:
            if self.exclusive_minimum and value <= self.minimum:
                raise ConstantValueError(f"{self.name}: {value} must be > {self.minimum}")
            if not self.exclusive_minimum and value < self.minimum:
                raise ConstantValueError(f"{self.name}: {value} must be >= {self.minimum}")
        if self.maximum is not None:
            if self.exclusive_maximum and value >= self.maximum:
                raise ConstantValueError(f"{self.name}: {value} must be < {self.maximum}")
            if not self.exclusive_maximum and value > self.maximum:
                raise ConstantValueError(f"{self.name}: {value} must be <= {self.maximum}")
        return value

    @model_validator(mode="after")
    def _validate_descriptor(self) -> Self:
        if self.not_a_lever is not None:
            if not self.not_a_lever.strip():
                raise ValueError(f"{self.name}: not_a_lever must carry a reason, not an empty string")
            if self.editable:
                raise ValueError(
                    f"{self.name}: a constant with no catalogue row cannot also be editable"
                )
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError(f"{self.name}: minimum must be <= maximum")
        if self.exclusive_minimum and self.minimum is None:
            raise ValueError(f"{self.name}: exclusive_minimum has no meaning without a minimum")
        if self.exclusive_maximum and self.maximum is None:
            raise ValueError(f"{self.name}: exclusive_maximum has no meaning without a maximum")
        if self.allowed_values is not None:
            if not self.allowed_values:
                raise ValueError(f"{self.name}: allowed_values must be a non-empty set when present")
            if len(set(self.allowed_values)) != len(self.allowed_values):
                raise ValueError(f"{self.name}: allowed_values must not repeat a value")
        return self


class InvariantViolation(BaseModel):
    """One relationship that the values under examination do not satisfy."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str = Field(min_length=1, description="Stable identifier of the violated relationship.")
    fields: tuple[str, ...] = Field(description="Constants the relationship relates.")
    message: str = Field(min_length=1, description="Why the values are refused.")


class _Invariant(BaseModel):
    """A relationship between two or more constants.

    A single knob's admissible range is expressed by its own spec. What a spec
    cannot say is that one knob must stay below another — a stall threshold
    beneath an idle timeout, a base backoff beneath its ceiling. Those live
    here, and they are part of the registry contract rather than a side effect
    of every constant happening to sit in one model: once a relationship
    spans two owners, no single owner can enforce it.

    ``predicate`` answers "are these values acceptable" — it is never asked
    unless every related constant is present, so a partially built set is
    silently skipped rather than falsely refused.

    A relationship is declared where its constants live, which is not always
    the process that evaluates it: a layer that publishes its group over a
    wire can send the ``key``, the ``fields`` and the ``message``, and no
    encoding carries a live callable. So the predicate is excluded from the
    serialised form and looked up by ``key`` on arrival, out of the table of
    the ones this build knows. A key it does not know is a refusal, not an
    unchecked relationship — see :meth:`ConstantsRegistry.declare`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str = Field(min_length=1, description="Stable identifier, unique across the registry.")
    fields: tuple[str, ...] = Field(description="Constants this relationship relates.")
    message: str = Field(min_length=1, description="Operator-facing reason for the refusal.")
    predicate: Callable[[Mapping[str, Any]], bool] | None = Field(
        default=None,
        exclude=True,
        description="True when the values satisfy the relationship; looked up by key when absent.",
    )

    def resolved_predicate(self) -> Callable[[Mapping[str, Any]], bool]:
        """The callable that answers this relationship, or raise."""
        if self.predicate is not None:
            return self.predicate
        known = KNOWN_PREDICATES.get(self.key)
        if known is None:
            raise UnknownInvariantError(
                f"relationship {self.key!r} arrived without a predicate and this "
                "build knows no predicate by that key; it cannot be evaluated"
            )
        return known

    def check(self, values: Mapping[str, Any]) -> InvariantViolation | None:
        """Return a violation when ``values`` break this relationship."""
        if any(field not in values for field in self.fields):
            return None
        if self.resolved_predicate()(values):
            return None
        return InvariantViolation(key=self.key, fields=self.fields, message=self.message)

    @model_validator(mode="after")
    def _validate_invariant(self) -> Self:
        if len(self.fields) < 2:
            raise ValueError(f"{self.key}: a relationship relates at least two constants")
        if len(set(self.fields)) != len(self.fields):
            raise ValueError(f"{self.key}: a constant is related to itself")
        return self


class GroupInvariant(_Invariant):
    """A relationship whose every constant belongs to one owning group.

    It travels with the group, so the owner enforces it without needing to
    see anyone else's values.
    """


class CrossGroupInvariant(_Invariant):
    """A relationship spanning two or more groups.

    No group can enforce it, because no group sees both sides. It is checked
    at the registry level, over the merged set.
    """

    groups: tuple[str, ...] = Field(description="Keys of the groups this relationship spans.")

    @model_validator(mode="after")
    def _validate_span(self) -> Self:
        if len(self.groups) < 2:
            raise ValueError(f"{self.key}: a cross-group relationship spans at least two groups")
        if len(set(self.groups)) != len(self.groups):
            raise ValueError(f"{self.key}: a group is listed twice")
        return self


class ConstantGroup(BaseModel):
    """A set of constants with one owner, declared as one unit.

    ``provisional`` marks a group that is not the owner of its names but a
    stand-in source of resolution while ownership is still being handed over.
    A provisional group answers ``resolve`` so that no name is ever unknown,
    and it loses every name an owning group claims — see
    :class:`IConstantsRegistry` for the collision rule.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str = Field(min_length=1, description="Stable group key, unique across the registry.")
    owner: str = Field(min_length=1, description="Name of the layer that declares and reads the group.")
    description: str = Field(default="", description="What this group of knobs governs.")
    specs: tuple[ConstantSpec, ...] = Field(default=(), description="The group's constants.")
    invariants: tuple[GroupInvariant, ...] = Field(
        default=(), description="Relationships between this group's own constants."
    )
    provisional: bool = Field(
        default=False,
        description=(
            "Whether this group is a temporary fallback source of resolution "
            "rather than the owner of its names."
        ),
    )

    @property
    def names(self) -> frozenset[str]:
        """Every constant name this group carries."""
        return frozenset(spec.name for spec in self.specs)

    def spec(self, name: str) -> ConstantSpec:
        """Return the spec for ``name``, or raise :class:`UnknownConstantError`."""
        for candidate in self.specs:
            if candidate.name == name:
                return candidate
        raise UnknownConstantError(f"{name} is not declared by group {self.key!r}")

    def defaults(self) -> dict[str, Any]:
        """Every constant's default, as a plain mapping."""
        return {spec.name: spec.default for spec in self.specs}

    def check(self, values: Mapping[str, Any]) -> tuple[InvariantViolation, ...]:
        """Return every relationship of this group that ``values`` break."""
        found = (invariant.check(values) for invariant in self.invariants)
        return tuple(violation for violation in found if violation is not None)

    @model_validator(mode="after")
    def _validate_group(self) -> Self:
        seen: set[str] = set()
        for spec in self.specs:
            if spec.name in seen:
                raise ValueError(f"{self.key}: constant {spec.name!r} is declared twice")
            seen.add(spec.name)
            if spec.group != self.key:
                raise ValueError(
                    f"{self.key}: constant {spec.name!r} declares group {spec.group!r}"
                )
        keys: set[str] = set()
        for invariant in self.invariants:
            if invariant.key in keys:
                raise ValueError(f"{self.key}: relationship {invariant.key!r} is declared twice")
            keys.add(invariant.key)
            foreign = sorted(set(invariant.fields) - seen)
            if foreign:
                raise ValueError(
                    f"{self.key}: relationship {invariant.key!r} relates constants this "
                    f"group does not own: {', '.join(foreign)}"
                )
        return self


@runtime_checkable
class ICoreConstantsProvider(Protocol):
    """How the loop asks for the snapshot in force for one scope.

    The snapshot is always passed by value into a turn: no global state, no
    module-level cache. Freshness — reading the store, watching for
    invalidation, rebuilding — belongs entirely to the implementation.
    """

    async def get(self, tenant_id: str) -> LoopConstants:
        """Return the snapshot in force for ``tenant_id``, fresh as of now."""
        ...


@runtime_checkable
class IConstantsRegistry(Protocol):
    """Declaration, resolution and per-scope reading of every constant group.

    A registry is a **start-time** object: groups are declared while the
    process comes up, and a turn adds no obligation to it. Two rules are part
    of the contract rather than of any one implementation:

    * ``resolve`` is fail-closed. A name no group declares raises
      :class:`UnknownConstantError`; it never resolves to a default.
    * a value is checked against its own descriptor before any relationship
      is evaluated. A number outside its own bounds is wrong on its own
      terms, and reaching the snapshot build with it there is how one bad
      stored row takes down every turn of a scope.
    * a name claimed by two **owning** groups is a refusal
      (:class:`DuplicateConstantError`). A name claimed by an owning group and
      by a provisional one is not: the owning group takes it, and the
      displacement is recorded.
    """

    def declare(self, group: ConstantGroup) -> None:
        """Add ``group`` to the registry, enforcing the collision rule."""
        ...

    @property
    def groups(self) -> tuple[ConstantGroup, ...]:
        """Every declared group, owning groups first."""
        ...

    @property
    def cross_group_invariants(self) -> tuple[CrossGroupInvariant, ...]:
        """Relationships no single group can enforce, checked over the merge."""
        ...

    def resolve(self, name: str) -> ConstantSpec:
        """Return the spec that governs ``name``, fail-closed."""
        ...

    def provisional_names(self) -> frozenset[str]:
        """Names still resolved only by a provisional group.

        Empty once every name has an owner. That is the whole measurement of
        a handover, so it belongs to the contract: a host that composes a
        registry rather than subclassing one must still be able to answer it.
        """
        ...

    def defaults(self) -> Mapping[str, Any]:
        """Every declared constant's default, owning groups winning."""
        ...

    def repair(self, values: Mapping[str, Any]) -> tuple[Mapping[str, Any], tuple[str, ...]]:
        """Return ``values`` made acceptable, plus the names that were reset.

        The alternative a snapshot build must not take is discarding the whole
        set: one bad stored row would take every unrelated setting of the
        scope down with it.
        """
        ...

    def coerce(self, name: str, raw: str | None) -> Any:
        """Read a stored string as ``name``'s declared shape, fail-closed on the name."""
        ...

    def validate_value(self, name: str, raw: str | None) -> Any:
        """Read and check one stored value for ``name``; raise if inadmissible."""
        ...

    def validate(self, values: Mapping[str, Any]) -> tuple[str, ...]:
        """Return one message per rule ``values`` break."""
        ...

    async def snapshot(self, tenant_id: str) -> Mapping[str, Any]:
        """Return every constant's effective value for ``tenant_id``."""
        ...

    def invalidate(self, tenant_id: str | None = None, group_key: str | None = None) -> None:
        """Drop cached values, narrowing by scope and by group.

        Both arguments widen to "everything" when omitted, and they mean
        different things:

        * ``tenant_id=None`` — every scope. Reachable, and not a convenience:
          a scope that ceases to exist takes its stored values with it, and
          the layer publishing that has no one scope to name.
        * ``group_key=None`` — every group of the named scope.

        A caller that can only name one scope cannot express the first, so the
        contract states it rather than leaving each implementation to widen the
        signature on its own.
        """
        ...


#: The five overhead budgets whose fractions must leave room for history.
_OVERHEAD_RATIO_FIELDS = (
    "system_prompt_max_ratio",
    "skill_index_budget_ratio",
    "loaded_skills_ratio",
    "tool_definitions_ratio",
    "user_context_ratio",
)


def _overhead_leaves_room(values: Mapping[str, Any]) -> bool:
    return sum(float(values[field]) for field in _OVERHEAD_RATIO_FIELDS) < 1.0


def _backoff_ceiling_above_base(values: Mapping[str, Any]) -> bool:
    base = float(values["resilience_backoff_base_seconds"])
    ceiling = float(values["resilience_backoff_max_seconds"])
    return not (base > 0.0 and ceiling > 0.0) or ceiling >= base


#: Every relationship whose constants share one owner, keyed by that owner's
#: group. A group is declared with the entry that names it; nothing here is
#: enforced until a group carries it, so a group that has not been declared
#: yet leaves its relationships unchecked rather than half-checked.
GROUP_INVARIANTS: Mapping[str, tuple[GroupInvariant, ...]] = {
    "loop": (
        GroupInvariant(
            key="loop.compaction_trigger_below_emergency",
            fields=("compaction_trigger_ratio", "compaction_emergency_ratio"),
            message="compaction_trigger_ratio must be < compaction_emergency_ratio",
            predicate=lambda v: v["compaction_trigger_ratio"] < v["compaction_emergency_ratio"],
        ),
        GroupInvariant(
            key="loop.request_context_safety_below_window",
            fields=("request_context_safety_tokens", "model_context_window"),
            message="request_context_safety_tokens must be < model_context_window",
            predicate=lambda v: (
                v["request_context_safety_tokens"] < v["model_context_window"]
            ),
        ),
        GroupInvariant(
            key="loop.overhead_leaves_room_for_history",
            fields=_OVERHEAD_RATIO_FIELDS,
            message="system + skill + tool + user budgets must sum to < 1.0",
            predicate=_overhead_leaves_room,
        ),
        GroupInvariant(
            key="loop.stall_below_idle",
            fields=("llm_stream_stall_threshold_seconds", "llm_stream_idle_timeout_seconds"),
            message=(
                "llm_stream_stall_threshold_seconds must be < llm_stream_idle_timeout_seconds"
            ),
            predicate=lambda v: (
                v["llm_stream_stall_threshold_seconds"] < v["llm_stream_idle_timeout_seconds"]
            ),
        ),
        GroupInvariant(
            key="loop.reasoning_idle_only_widens",
            fields=("llm_stream_reasoning_idle_timeout_seconds", "llm_stream_idle_timeout_seconds"),
            message=(
                "llm_stream_reasoning_idle_timeout_seconds must be >= "
                "llm_stream_idle_timeout_seconds"
            ),
            predicate=lambda v: (
                v["llm_stream_reasoning_idle_timeout_seconds"] >= v["llm_stream_idle_timeout_seconds"]
            ),
        ),
        GroupInvariant(
            key="loop.backoff_ceiling_above_base",
            fields=("resilience_backoff_max_seconds", "resilience_backoff_base_seconds"),
            message=(
                "resilience_backoff_max_seconds must be >= resilience_backoff_base_seconds "
                "when backoff is enabled"
            ),
            predicate=_backoff_ceiling_above_base,
        ),
    ),
}

#: Relationships no single group can enforce, shipped by this package: none.
#: Every one of them has a side in a group some surrounding layer owns, so it
#: is declared by the layer that sees both sides and passed to the registry
#: there. The name stays: it is the shape of what such a caller passes.
CROSS_GROUP_INVARIANTS: tuple[CrossGroupInvariant, ...] = ()


#: Every predicate this build can answer a relationship with, by key. A group
#: that arrives from another process carries its relationships' keys, fields
#: and messages but not their callables — no encoding carries one — so the
#: receiving side supplies them from here. A key absent from this table is
#: refused at declaration rather than left unevaluated.
KNOWN_PREDICATES: Mapping[str, Callable[[Mapping[str, Any]], bool]] = {
    invariant.key: predicate
    for invariant in (
        *(one for group in GROUP_INVARIANTS.values() for one in group),
        *CROSS_GROUP_INVARIANTS,
    )
    if (predicate := invariant.predicate) is not None
}


class ConstantsRegistry:
    """The declaration half of a registry: who owns what, and what is legal.

    A **start-time** object. Groups are declared while the process comes up;
    a turn asks it questions and never adds to it. Reading a scope's values
    and reacting to invalidation are the surrounding layer's half — a host
    registry composes this one rather than reimplementing the collision and
    relationship rules.

    Two rules it enforces:

    * a name claimed by two **owning** groups is a refusal. A name claimed by
      an owning group and by a provisional one is the ordinary course of a
      handover: the owning group takes it, and the displacement is recorded.
    * a group is refused at declaration if any of its relationships names a
      predicate this build cannot supply. A group that arrived over a wire
      carries relationships by key, and one whose key means nothing here
      would read as satisfied by every set of values.
    * resolution is fail-closed. A name no group declares — provisional
      groups included — raises rather than resolving to a default.

    The relationships no group can enforce are carried by default, because
    forgetting them is silent: they are exactly the two whose sides sit in
    different groups, so nobody else evaluates them and a registry built
    without them refuses nothing and says nothing. Passing an empty tuple is
    the way to opt out, deliberately.
    """

    def __init__(
        self,
        cross_group_invariants: tuple[CrossGroupInvariant, ...] = CROSS_GROUP_INVARIANTS,
    ) -> None:
        self._owning: list[ConstantGroup] = []
        self._provisional: list[ConstantGroup] = []
        self._cross_group = cross_group_invariants

    @property
    def groups(self) -> tuple[ConstantGroup, ...]:
        """Every declared group, owning groups first."""
        return tuple(self._owning) + tuple(self._provisional)

    @property
    def cross_group_invariants(self) -> tuple[CrossGroupInvariant, ...]:
        """Relationships no single group can enforce."""
        return self._cross_group

    @property
    def names(self) -> frozenset[str]:
        """Every name the registry can resolve."""
        return frozenset().union(*(group.names for group in self.groups)) if self.groups else frozenset()

    def provisional_names(self) -> frozenset[str]:
        """Names still resolved only by a provisional group.

        Empty once every name has an owner — the mechanical proof that the
        handover is finished, in place of comparing two lists by eye.
        """
        owned = frozenset().union(*(g.names for g in self._owning)) if self._owning else frozenset()
        held = frozenset().union(*(g.names for g in self._provisional)) if self._provisional else frozenset()
        return held - owned

    def declare(self, group: ConstantGroup) -> None:
        """Add ``group``, enforcing the collision rule."""
        if any(declared.key == group.key for declared in self.groups):
            raise DuplicateConstantError(f"group {group.key!r} is already declared")
        for invariant in group.invariants:
            invariant.resolved_predicate()
        if group.provisional:
            self._provisional.append(group)
        else:
            self._reject_owning_collisions(group)
            self._owning.append(group)
        self._log_displacements(group)

    def _reject_owning_collisions(self, group: ConstantGroup) -> None:
        for declared in self._owning:
            clash = sorted(declared.names & group.names)
            if clash:
                raise DuplicateConstantError(
                    f"groups {declared.key!r} and {group.key!r} both claim: {', '.join(clash)}"
                )

    def _log_displacements(self, group: ConstantGroup) -> None:
        pairs = (
            [(group, held) for held in self._provisional]
            if not group.provisional
            else [(owning, group) for owning in self._owning]
        )
        for owning, held in pairs:
            clash = sorted(owning.names & held.names)
            if clash:
                logger.warning(
                    "constants: %d name(s) pass from provisional group %r to owner %r: %s",
                    len(clash),
                    held.key,
                    owning.key,
                    ", ".join(clash),
                )

    def resolve(self, name: str) -> ConstantSpec:
        """Return the spec governing ``name``; raise when nothing declares it."""
        for group in self.groups:
            if name in group.names:
                return group.spec(name)
        raise UnknownConstantError(f"no declared group owns constant {name!r}")

    def defaults(self) -> dict[str, Any]:
        """Every declared constant's default, owning groups winning."""
        merged: dict[str, Any] = {}
        for group in reversed(self.groups):
            merged.update(group.defaults())
        return merged

    def coerce(self, name: str, raw: str | None) -> Any:
        """Read a stored string as ``name``'s declared shape, fail-closed on the name."""
        return self.resolve(name).coerce(raw)

    def validate_value(self, name: str, raw: str | None) -> Any:
        """Read and check one stored value for ``name``; raise if it is not admissible."""
        spec = self.resolve(name)
        coerced = spec.coerce(raw)
        if coerced is None:
            return None
        return spec.validate_value(coerced)

    def _value_violations(self, values: Mapping[str, Any]) -> list[InvariantViolation]:
        """Every value that its own descriptor refuses.

        Checked before any relationship, because a value outside its own
        bounds is wrong on its own terms: repairing a relationship around it
        would put an unrelated setting back to its default and leave the
        offending one in place.
        """
        found: list[InvariantViolation] = []
        for name, value in values.items():
            try:
                spec = self.resolve(name)
            except UnknownConstantError:
                continue
            try:
                spec.validate_value(value)
            except ConstantValueError as exc:
                found.append(
                    InvariantViolation(key=f"value.{name}", fields=(name,), message=str(exc))
                )
        return found

    def check(self, values: Mapping[str, Any]) -> tuple[InvariantViolation, ...]:
        """Return every rule ``values`` break: own bounds first, then relationships."""
        violations: list[InvariantViolation] = self._value_violations(values)
        for group in self.groups:
            violations.extend(group.check(values))
        for invariant in self._cross_group:
            violation = invariant.check(values)
            if violation is not None:
                violations.append(violation)
        seen: set[str] = set()
        unique: list[InvariantViolation] = []
        for violation in violations:
            if violation.key not in seen:
                seen.add(violation.key)
                unique.append(violation)
        return tuple(unique)

    def validate(self, values: Mapping[str, Any]) -> tuple[str, ...]:
        """Return one message per rule ``values`` break."""
        return tuple(violation.message for violation in self.check(values))

    def repair(self, values: Mapping[str, Any]) -> tuple[dict[str, Any], tuple[str, ...]]:
        """Return ``values`` made acceptable, plus the names that were reset.

        A set that breaks a rule is not thrown away whole: the offending
        **constant** is put back to its declared default, one at a time, until
        nothing is broken. Discarding the whole set would take every unrelated
        setting down with the one that was wrong.
        """
        working = dict(values)
        reset: list[str] = []
        for _ in range(len(working) + 1):
            violations = self.check(working)
            if not violations:
                break
            offender = self._first_deviating_field(violations[0], working)
            if offender is None:
                raise ConfigContractError(
                    f"defaults themselves break {violations[0].key!r}: "
                    f"{violations[0].message}"
                )
            working[offender] = self.resolve(offender).default
            reset.append(offender)
        else:  # pragma: no cover — one reset per field bounds the loop
            raise ConfigContractError("value repair did not converge")
        if reset:
            logger.warning(
                "constants: %d value(s) reset to their defaults to satisfy relationships: %s",
                len(reset),
                ", ".join(reset),
            )
        return working, tuple(reset)

    def _first_deviating_field(
        self, violation: InvariantViolation, values: Mapping[str, Any]
    ) -> str | None:
        for field in violation.fields:
            if values[field] != self.resolve(field).default:
                return field
        return None


#: Key of the group holding the loop's own thresholds.
LOOP_GROUP_KEY = "loop"


def _kind_of(annotation: Any) -> ConstantKind:
    """The wire shape of one declared annotation.

    Anything that is not a scalar or an enumeration of strings is ``json``:
    a list of nested models has no scalar bound and no closed value set, and
    saying so honestly is better than inventing one.
    """
    if annotation is bool:
        return "bool"
    if annotation is int:
        return "int"
    if annotation is float:
        return "float"
    if annotation is str:
        return "str"
    if get_origin(annotation) is Literal:
        return "str" if all(isinstance(value, str) for value in get_args(annotation)) else "json"
    return "json"


class _Bounds(NamedTuple):
    """The range one declared field admits, strictness kept.

    Collapsing ``gt`` into ``ge`` costs exactly one value at each end, and it
    is the value an operator is most likely to try: a ceiling declared
    ``gt=0`` becomes a floor of zero, an editor offers zero, and the write is
    accepted here and refused where the value is finally used.
    """

    minimum: float | None = None
    maximum: float | None = None
    exclusive_minimum: bool = False
    exclusive_maximum: bool = False


def _bounds_of(field: FieldInfo) -> _Bounds:
    """The lowest and highest accepted value the annotation carries, if any."""
    minimum: float | None = None
    maximum: float | None = None
    exclusive_minimum = False
    exclusive_maximum = False
    for constraint in field.metadata:
        inclusive_lower = getattr(constraint, "ge", None)
        exclusive_lower = getattr(constraint, "gt", None)
        if inclusive_lower is not None:
            minimum, exclusive_minimum = float(inclusive_lower), False
        elif exclusive_lower is not None:
            minimum, exclusive_minimum = float(exclusive_lower), True
        inclusive_upper = getattr(constraint, "le", None)
        exclusive_upper = getattr(constraint, "lt", None)
        if inclusive_upper is not None:
            maximum, exclusive_maximum = float(inclusive_upper), False
        elif exclusive_upper is not None:
            maximum, exclusive_maximum = float(exclusive_upper), True
    return _Bounds(minimum, maximum, exclusive_minimum, exclusive_maximum)


def _annotation_repr(annotation: Any) -> str | None:
    """How the declaration spells an annotation, for a reader in another process."""
    if annotation is None:
        return None
    if isinstance(annotation, type):
        return annotation.__name__
    return str(annotation)


def _allowed_values_of(annotation: Any) -> tuple[str, ...] | None:
    """The closed value set of an enumeration annotation, if it is one."""
    if get_origin(annotation) is not Literal:
        return None
    values = get_args(annotation)
    if not all(isinstance(value, str) for value in values):
        return None
    return tuple(str(value) for value in values)


#: The descriptor facets a declaring model may state for itself, in a field's
#: ``json_schema_extra``. A type and a bound are readable from the
#: declaration; whether a floor of zero means "no ceiling", which drawer of an
#: editor a knob belongs in, and what its number is measured in are not, and
#: they are lost in reflection unless the declaration says them out loud.
DECLARED_FACETS = ("category", "unit", "zero_means_unlimited", "editable", "not_a_lever")


class SpecOverlay(BaseModel):
    """Facets applied to reflected descriptors from outside the model.

    The declaring model says what it can about its own fields. Two kinds of
    fact it cannot: one that belongs to the layer reading the value rather
    than to the value (a knob this deployment does not offer at all), and one
    that is being moved onto the declaration a batch at a time. Both are
    passed here, and both win over what the field declares, because the
    caller assembling the group is the later statement.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    not_a_lever: Mapping[str, str] = Field(
        default_factory=dict,
        description="Names with no catalogue row at all, mapped to the reason.",
    )
    read_only: frozenset[str] = Field(
        default_factory=frozenset, description="Names that have a row but refuse a write."
    )
    categories: Mapping[str, str] = Field(
        default_factory=dict, description="Names mapped to an editor's grouping."
    )
    units: Mapping[str, str] = Field(
        default_factory=dict, description="Names mapped to their unit of measure."
    )
    zero_means_unlimited: frozenset[str] = Field(
        default_factory=frozenset,
        description="Names whose zero disables the limit rather than setting it to nothing.",
    )

    def facets_for(self, name: str) -> dict[str, Any]:
        """The facets this overlay states for ``name``, if any."""
        stated: dict[str, Any] = {}
        if name in self.not_a_lever:
            stated["not_a_lever"] = self.not_a_lever[name]
            stated["editable"] = False
        elif name in self.read_only:
            stated["editable"] = False
        if name in self.categories:
            stated["category"] = self.categories[name]
        if name in self.units:
            stated["unit"] = self.units[name]
        if name in self.zero_means_unlimited:
            stated["zero_means_unlimited"] = True
        return stated


def _declared_facets(field: FieldInfo) -> dict[str, Any]:
    """The facets a field states about itself in ``json_schema_extra``."""
    extra = field.json_schema_extra
    if not isinstance(extra, dict):
        return {}
    return {name: extra[name] for name in DECLARED_FACETS if name in extra}


def spec_from_field(
    name: str,
    field: FieldInfo,
    group: str,
    owner: str,
    overlay: SpecOverlay | None = None,
) -> ConstantSpec:
    """Build a descriptor from one declared field of a snapshot model."""
    annotation = field.annotation
    bounds = _bounds_of(field)
    facets: dict[str, Any] = _declared_facets(field)
    facets.update((overlay or SpecOverlay()).facets_for(name))
    return ConstantSpec(
        name=name,
        kind=_kind_of(annotation),
        annotation=annotation,
        annotation_repr=_annotation_repr(annotation),
        default=field.get_default(call_default_factory=True),
        description=field.description or "",
        group=group,
        owner=owner,
        minimum=bounds.minimum,
        maximum=bounds.maximum,
        exclusive_minimum=bounds.exclusive_minimum,
        exclusive_maximum=bounds.exclusive_maximum,
        allowed_values=_allowed_values_of(annotation),
        **facets,
    )


def group_from_model(
    model: type[BaseModel],
    *,
    key: str,
    owner: str,
    description: str = "",
    provisional: bool = False,
    overlay: SpecOverlay | None = None,
    invariants: tuple[GroupInvariant, ...] | None = None,
) -> ConstantGroup:
    """Reflect a declaring model into a group, with no hand-written list.

    A hand-written list of hundreds of names drifts from the model it
    describes on the first field anyone adds; reflection cannot. Every
    descriptor comes from the declaration itself — its type, its bounds, its
    enumeration, its default and its own words — with :class:`SpecOverlay`
    supplying only what a declaration cannot state about itself.

    ``invariants`` defaults to every relationship in :data:`GROUP_INVARIANTS`
    whose constants this model happens to hold, so nothing goes unchecked
    while ownership of a name moves from one group to another. A group that
    knows its own relationships passes them.
    """
    specs = tuple(
        spec_from_field(name, field, key, owner, overlay)
        for name, field in model.model_fields.items()
    )
    if invariants is None:
        held = {spec.name for spec in specs}
        invariants = tuple(
            invariant
            for group in GROUP_INVARIANTS.values()
            for invariant in group
            if held.issuperset(invariant.fields)
        )
    return ConstantGroup(
        key=key,
        owner=owner,
        description=description,
        specs=specs,
        invariants=invariants,
        provisional=provisional,
    )


def build_loop_group(
    model: type[BaseModel] = LoopConstants,
    *,
    provisional: bool = False,
    overlay: SpecOverlay | None = None,
) -> ConstantGroup:
    """Reflect the loop's own snapshot model into its group.

    An ordinary owning group: the model holds the loop's own values and
    nothing else, so every name in it belongs to the loop and a second group
    claiming one is a collision rather than a handover. The parameter remains
    for a caller mid-handover — a layer moving a name out of its own model
    declares its group provisional so both sides resolve while it moves.
    """
    return group_from_model(
        model,
        key=LOOP_GROUP_KEY,
        owner="the loop",
        description="Thresholds the loop reads, reflected from the snapshot model.",
        provisional=provisional,
        overlay=overlay,
    )


__all__ = [
    "CROSS_GROUP_INVARIANTS",
    "DECLARED_FACETS",
    "GROUP_INVARIANTS",
    "KNOWN_PREDICATES",
    "LOOP_GROUP_KEY",
    "ConfigContractError",
    "ConstantGroup",
    "ConstantKind",
    "ConstantSpec",
    "ConstantValueError",
    "ConstantsRegistry",
    "CrossGroupInvariant",
    "DuplicateConstantError",
    "GroupInvariant",
    "IConstantsRegistry",
    "ICoreConstantsProvider",
    "InvariantViolation",
    "SpecOverlay",
    "UnknownConstantError",
    "UnknownInvariantError",
    "build_loop_group",
    "group_from_model",
    "spec_from_field",
]
