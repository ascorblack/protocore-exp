"""The advertised tool surface is named once and costed once."""
from __future__ import annotations

import pytest

from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.types import ToolDefinition, ToolParameterSchema
from protocore.runtime import token_counting
from protocore.runtime.tool_surface import (
    claim_surface_publication,
    forget_tool_surfaces,
    read_tool_surface,
    tool_surface_tokens,
)


@pytest.fixture(autouse=True)
def _clean_surfaces() -> None:
    forget_tool_surfaces()


def _tool(name: str, description: str = "does a thing") -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=description,
        parameters=ToolParameterSchema(properties={}, required=[]),
    )


def test_the_same_definitions_are_serialised_once(monkeypatch: pytest.MonkeyPatch) -> None:
    tools = (_tool("read"), _tool("write"))
    first = read_tool_surface(tools)

    dumps = 0
    original = ToolDefinition.model_dump_json

    def counted(self: ToolDefinition, **kwargs: object) -> str:
        nonlocal dumps
        dumps += 1
        return original(self, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(ToolDefinition, "model_dump_json", counted)
    again = read_tool_surface(tools)

    assert again is first
    assert dumps == 0


def test_equal_definitions_rebuilt_share_a_digest() -> None:
    one = read_tool_surface((_tool("read"), _tool("write")))
    other = read_tool_surface((_tool("read"), _tool("write")))

    assert other is not one
    assert other.digest == one.digest
    assert other.names == ("read", "write")


def test_a_changed_description_changes_the_digest() -> None:
    one = read_tool_surface((_tool("read", "reads a file"),))
    other = read_tool_surface((_tool("read", "reads a file, or a range of it"),))

    assert other.digest != one.digest


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


def test_a_digest_is_published_in_full_once() -> None:
    surface = read_tool_surface((_tool("read"),))

    assert claim_surface_publication(surface.digest) is True
    assert claim_surface_publication(surface.digest) is False
    assert claim_surface_publication("a different surface") is True
