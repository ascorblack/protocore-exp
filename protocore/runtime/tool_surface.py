"""One reading of the advertised tool surface, kept while the surface holds still.

A deployment's tool surface is decided by its registry and changes when that
registry changes — which is to say, almost never, and certainly not once per
run. Two things were nevertheless paid for per run, and both scale with the
number of tools rather than with anything the run did:

* the surface was published in full on every run, so a store keeping the
  advertisement kept the same bytes once per run — 36 KB apiece, 25 MB over a
  few hundred runs, all of it the same sixty descriptions;
* every tool definition was re-estimated for its token weight on every LLM
  call. With the Python token counter at 3-5 MB/s, 32 KB of definitions is
  about 10 ms of arithmetic inside the event loop, per round, for a number
  that could not have changed.

Both are answered by naming the surface. :func:`read_tool_surface` digests the
definitions once and hands back that digest with the names; the advertisement
carries the descriptions only the first time a digest is seen in this process,
and a reader that already has them looks them up by digest.
:func:`tool_surface_tokens` costs the definitions once per digest.

The digest itself is not recomputed per call either. Definitions are frozen, so
a registry that has not changed hands back the same objects, and the most
recent surfaces are remembered against the object identities that produced
them: a hit costs a pointer comparison per tool. A miss serialises, and the
serialisation lands on the digest, so a registry that rebuilt equal definitions
still reuses the token estimate and still does not republish.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass

from protocore.constants import MAX_TOOL_SURFACE_CACHE_ENTRIES
from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.types import ToolDefinition
from protocore.runtime.token_counting import estimate_tokens

__all__ = [
    "ToolSurface",
    "claim_surface_publication",
    "forget_tool_surfaces",
    "read_tool_surface",
    "tool_surface_tokens",
]


@dataclass(frozen=True)
class ToolSurface:
    """The tool surface of one request, named rather than quoted."""

    digest: str
    """Identity of the definitions. Two surfaces with the same digest carry the
    same tools with the same descriptions and the same parameter schemas."""

    names: tuple[str, ...]
    """The tool names, in the order they are advertised."""

    definitions: tuple[str, ...]
    """Each definition as it serialises, parallel to :attr:`names`. Held so the
    token estimate and the digest are paid for together."""


@dataclass
class _Remembered:
    tools: tuple[ToolDefinition, ...]
    surface: ToolSurface


#: The surfaces most recently read, against the definition objects that
#: produced them. Bounded, and the entries hold the definitions, so an identity
#: comparison cannot meet a recycled address.
_recent: list[_Remembered] = []

#: Token estimates by digest and by the chars-per-token ratios they were
#: computed under; the ratios are tunable under a running process.
_estimates: OrderedDict[tuple[str, tuple[float, ...]], int] = OrderedDict()

#: Digests whose definitions have already been published in this process.
_published: OrderedDict[str, None] = OrderedDict()


def read_tool_surface(tools: Sequence[ToolDefinition]) -> ToolSurface:
    """Name the surface ``tools`` describes, serialising it at most once."""
    key = tuple(tools)
    for index, entry in enumerate(_recent):
        if len(entry.tools) == len(key) and all(
            held is now for held, now in zip(entry.tools, key, strict=True)
        ):
            if index:
                _recent.insert(0, _recent.pop(index))
            return entry.surface
    definitions = tuple(tool.model_dump_json() for tool in key)
    digest = hashlib.sha256("\n".join(definitions).encode()).hexdigest()
    surface = ToolSurface(
        digest=digest,
        names=tuple(tool.name for tool in key),
        definitions=definitions,
    )
    _recent.insert(0, _Remembered(tools=key, surface=surface))
    del _recent[MAX_TOOL_SURFACE_CACHE_ENTRIES:]
    return surface


def tool_surface_tokens(surface: ToolSurface, rc: LoopConstants) -> int:
    """The token weight of the definitions, costed once per digest."""
    signature = (
        rc.token_count_chars_per_token_latin,
        rc.token_count_chars_per_token_cyrillic,
        rc.token_count_chars_per_token_cyrillic_json_escape,
        rc.token_count_chars_per_token_cjk,
        rc.token_count_chars_per_token_json_struct,
    )
    key = (surface.digest, signature)
    remembered = _estimates.get(key)
    if remembered is not None:
        _estimates.move_to_end(key)
        return remembered
    total = sum(estimate_tokens(definition, rc) for definition in surface.definitions)
    _estimates[key] = total
    _estimates.move_to_end(key)
    while len(_estimates) > MAX_TOOL_SURFACE_CACHE_ENTRIES:
        _estimates.popitem(last=False)
    return total


def claim_surface_publication(digest: str) -> bool:
    """True the first time this process publishes ``digest`` in full.

    A reader keeps the descriptions against the digest they came under, so
    sending them again says nothing it does not already know. The publisher
    decides rather than asking, because the advertisement is an event and an
    event has no answer to read; a process restart republishes once, which is
    the price of not holding a conversation about it.
    """
    if digest in _published:
        _published.move_to_end(digest)
        return False
    _published[digest] = None
    while len(_published) > MAX_TOOL_SURFACE_CACHE_ENTRIES:
        _published.popitem(last=False)
    return True


def forget_tool_surfaces() -> None:
    """Drop every remembered surface, estimate and publication."""
    _recent.clear()
    _estimates.clear()
    _published.clear()
