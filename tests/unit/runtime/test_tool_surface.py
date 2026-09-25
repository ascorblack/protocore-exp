"""The advertised tool surface is named once and costed once."""
from __future__ import annotations

from collections.abc import Iterator

import pytest

from protocore.constants import (
    MAX_TOOL_SURFACE_AUDIENCES,
    MAX_TOOL_SURFACE_CACHE_ENTRIES,
)
from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.types import ToolDefinition, ToolParameterSchema
from protocore.runtime import token_counting
from protocore.runtime.tool_surface import (
    forget_tool_surfaces,
    note_surface_described,
    read_tool_surface,
    surface_descriptions,
    surface_needs_describing,
    tool_surface_tokens,
)


@pytest.fixture(autouse=True)
def _clean_surfaces() -> Iterator[None]:
    forget_tool_surfaces()
    yield
    forget_tool_surfaces()


def _tool(name: str, description: str = "does a thing") -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=description,
        parameters=ToolParameterSchema(properties={}, required=[]),
    )


def test_equal_definitions_share_a_digest() -> None:
    one = read_tool_surface((_tool("read"), _tool("write")))
    other = read_tool_surface((_tool("read"), _tool("write")))

    assert other is not one
    assert other.digest == one.digest
    assert other.names == ("read", "write")


def test_a_changed_description_changes_the_digest() -> None:
    one = read_tool_surface((_tool("read", "reads a file"),))
    other = read_tool_surface((_tool("read", "reads a file, or a range of it"),))

    assert other.digest != one.digest


def test_a_schema_edited_in_place_changes_the_digest() -> None:
    """ToolDefinition is frozen; the dict inside its schema is not.

    A surface remembered against object identity would hand back a digest that
    had stopped describing the tool. Nothing is remembered against identity, so
    the edit is seen.
    """
    tool = ToolDefinition(
        name="read",
        description="reads",
        parameters=ToolParameterSchema(properties={"a": {"type": "string"}}, required=[]),
    )
    first = read_tool_surface((tool,))
    tool.parameters.properties["b"] = {"type": "integer"}
    second = read_tool_surface((tool,))

    assert second.digest != first.digest


def test_a_definition_that_is_not_a_model_is_still_costed() -> None:
    class _Bare:
        name = "bare"
        description = "not a pydantic model"

    surface = read_tool_surface((_Bare(),))

    assert surface.names == ("bare",)
    assert tool_surface_tokens(surface, LoopConstants()) > 0


def test_the_definitions_are_costed_once_per_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rc = LoopConstants()
    surface = read_tool_surface((_tool("read"), _tool("write")))
    expected = tool_surface_tokens(surface, rc)

    calls = 0
    original = token_counting.estimate_tokens

    def counted(text: str, constants: LoopConstants) -> int:
        nonlocal calls
        calls += 1
        return original(text, constants)

    monkeypatch.setattr("protocore.runtime.tool_surface.estimate_tokens", counted)

    assert tool_surface_tokens(surface, rc) == expected
    assert calls == 0
    # A surface rebuilt from equal definitions is the same digest, so the
    # estimate is not paid for again either.
    rebuilt = read_tool_surface((_tool("read"), _tool("write")))
    assert tool_surface_tokens(rebuilt, rc) == expected
    assert calls == 0


def test_a_changed_ratio_is_costed_again() -> None:
    surface = read_tool_surface((_tool("read"),))
    lean = tool_surface_tokens(surface, LoopConstants(token_count_chars_per_token_latin=4.0))
    rich = tool_surface_tokens(surface, LoopConstants(token_count_chars_per_token_latin=2.0))

    assert rich > lean


def test_the_calibration_factor_does_not_change_the_definition_cost() -> None:
    surface = read_tool_surface((_tool("read"),))
    plain = tool_surface_tokens(surface, LoopConstants(token_estimate_calibration=1.0))
    scaled = tool_surface_tokens(surface, LoopConstants(token_estimate_calibration=2.0))

    assert scaled == plain


def test_each_reader_is_described_the_surface_once() -> None:
    surface = read_tool_surface((_tool("read"),))

    assert surface_needs_describing(surface.digest, "session-a") is True
    # Asking does not claim: an advertisement that was never delivered leaves
    # the reader still owed its descriptions.
    assert surface_needs_describing(surface.digest, "session-a") is True

    note_surface_described(surface.digest, "session-a")

    assert surface_needs_describing(surface.digest, "session-a") is False
    # A second reader has been told nothing.
    assert surface_needs_describing(surface.digest, "session-b") is True
    # And a changed surface is described again, to everyone.
    other = read_tool_surface((_tool("read", "reads, differently"),))
    assert surface_needs_describing(other.digest, "session-a") is True


def test_the_descriptions_can_be_looked_up_by_digest() -> None:
    surface = read_tool_surface((_tool("read", "reads a file"), _tool("write")))

    assert surface_descriptions(surface.digest) == {
        "read": "reads a file",
        "write": "does a thing",
    }
    assert surface_descriptions("a digest this process never read") is None


def test_the_returned_descriptions_are_a_copy() -> None:
    surface = read_tool_surface((_tool("read", "reads a file"),))
    taken = surface_descriptions(surface.digest)
    assert taken is not None
    taken["read"] = "something else"

    assert surface_descriptions(surface.digest) == {"read": "reads a file"}


def test_only_so_many_surfaces_are_remembered() -> None:
    """A process that somehow meets a new surface per run does not accumulate."""
    digests = [
        read_tool_surface((_tool(f"tool_{i}", f"description {i}"),)).digest
        for i in range(MAX_TOOL_SURFACE_CACHE_ENTRIES + 3)
    ]

    assert surface_descriptions(digests[-1]) is not None
    assert surface_descriptions(digests[0]) is None


def test_evicting_a_surface_also_drops_the_claims_on_it() -> None:
    """A claim outliving the descriptions would be a promise nobody can keep.

    The claim says the reader was told and can be left to its own copy. Once
    this process can no longer answer a lookup for that digest, the reader that
    lost its copy has nowhere to go, so the next advertisement describes the
    surface again rather than naming something unresolvable.
    """
    first = read_tool_surface((_tool("read", "reads a file"),)).digest
    note_surface_described(first, "session-a")
    assert surface_needs_describing(first, "session-a") is False

    for i in range(MAX_TOOL_SURFACE_CACHE_ENTRIES + 1):
        read_tool_surface((_tool(f"other_{i}", f"description {i}"),))

    assert surface_descriptions(first) is None
    assert surface_needs_describing(first, "session-a") is True


def test_a_surface_still_in_use_keeps_its_claims() -> None:
    """Eviction is by least-recent use, and reading a surface is a use."""
    first = read_tool_surface((_tool("read", "reads a file"),)).digest
    note_surface_described(first, "session-a")

    for i in range(MAX_TOOL_SURFACE_CACHE_ENTRIES - 1):
        read_tool_surface((_tool(f"other_{i}", f"description {i}"),))
        read_tool_surface((_tool("read", "reads a file"),))

    assert surface_descriptions(first) is not None
    assert surface_needs_describing(first, "session-a") is False


def test_only_so_many_readers_are_remembered() -> None:
    digest = read_tool_surface((_tool("read"),)).digest
    for i in range(MAX_TOOL_SURFACE_AUDIENCES + 2):
        note_surface_described(digest, f"session-{i}")

    # The oldest reader is told what its tools do a second time; the newest
    # readers are not.
    assert surface_needs_describing(digest, "session-0") is True
    assert surface_needs_describing(digest, f"session-{MAX_TOOL_SURFACE_AUDIENCES + 1}") is False
