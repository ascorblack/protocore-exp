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

Both are answered by naming the surface. :func:`read_tool_surface` serialises
the definitions and digests them; the advertisement then carries the digest
always and the descriptions only when the reader it is going to has not been
sent them under that digest yet, and :func:`tool_surface_tokens` costs the
definitions once per digest.

The serialisation itself is NOT cached, deliberately. Caching it would mean
deciding a surface is unchanged without reading it, and the only cheap way to
decide that is object identity — which :class:`ToolDefinition` does not
support: it is frozen, but its ``parameters`` schema holds a plain ``dict`` and
a plain ``list``, so a registry editing a schema in place would keep its
identity and be handed back a digest that no longer describes it. Serialising
60 definitions and digesting them costs about 0.2 ms, against the 10 ms the
token estimate costs, so the cache that matters is the one keyed on the digest
and the one this module can be sure of.

Everything here is process-global and unsynchronised. One event loop is
assumed, as everywhere else in the runtime: there is no await between any read
and its write below. Under a second loop in a worker thread the worst outcome
is a redundant description payload or a recomputed estimate — the caches are
bounded and hold no state a run depends on.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from protocore.constants import (
    MAX_TOOL_SURFACE_AUDIENCES,
    MAX_TOOL_SURFACE_CACHE_ENTRIES,
)
from protocore.contracts.runtime_constants import LoopConstants
from protocore.runtime.token_counting import estimate_tokens

__all__ = [
    "ToolSurface",
    "forget_tool_surfaces",
    "note_surface_described",
    "read_tool_surface",
    "surface_descriptions",
    "surface_needs_describing",
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


#: Token estimates by digest and by the chars-per-token ratios they were
#: computed under; the ratios are tunable under a running process.
_estimates: OrderedDict[tuple[str, tuple[float, ...]], int] = OrderedDict()

#: What each tool on a digested surface does, so a reader that met a digest it
#: has no descriptions for can be answered without waiting for a run.
_descriptions: OrderedDict[str, dict[str, str]] = OrderedDict()

#: Which readers have been sent the descriptions of which digest.
_described: OrderedDict[tuple[str, str], None] = OrderedDict()


def _serialise(tool: Any) -> str:
    """One definition as text. Tolerates a definition that is not a model.

    Typing says every tool on a request is a ``ToolDefinition``; the code this
    replaced did not believe typing, and a host putting something else on
    ``LLMRequest.tools`` should get a rough estimate rather than an
    ``AttributeError`` raised from inside the token calibrator.
    """
    dump = getattr(tool, "model_dump_json", None)
    return dump() if callable(dump) else str(tool)


def read_tool_surface(tools: Sequence[Any]) -> ToolSurface:
    """Name the surface ``tools`` describes."""
    definitions = tuple(_serialise(tool) for tool in tools)
    digest = hashlib.sha256("\n".join(definitions).encode()).hexdigest()
    names = tuple(str(getattr(tool, "name", "")) for tool in tools)
    if digest not in _descriptions:
        _descriptions[digest] = {
            name: str(getattr(tool, "description", ""))
            for name, tool in zip(names, tools, strict=True)
        }
        while len(_descriptions) > MAX_TOOL_SURFACE_CACHE_ENTRIES:
            evicted, _ = _descriptions.popitem(last=False)
            _forget_claims_for(evicted)
    else:
        _descriptions.move_to_end(digest)
    return ToolSurface(digest=digest, names=names, definitions=definitions)


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


def _forget_claims_for(digest: str) -> None:
    """Drop every reader's claim on a digest this process can no longer explain.

    A claim says a reader was sent the descriptions and can be left to its own
    copy. Once the descriptions are evicted, :func:`surface_descriptions` can
    no longer answer a reader that lost that copy, so the claim is a promise
    the process has stopped being able to keep: the next advertisement of the
    digest describes it again rather than naming something nobody can look up.
    Eviction needs more distinct surfaces than a deployment has, so the scan
    costs nothing on the path that matters.
    """
    for stale in [key for key in _described if key[0] == digest]:
        del _described[stale]


def surface_needs_describing(digest: str, audience: str) -> bool:
    """Whether ``audience`` still has to be told what the tools of ``digest`` do.

    The audience is the reader the advertisement reaches — the session, which
    is the unit a host fans events out over. Per-reader rather than per-process
    on purpose: a process-wide claim assumes every reader is a durable cache
    that was listening from the first run of the pod, and a client that
    connected later, reconnected past the described event, or belongs to
    another tenant is none of those things. Per-reader keeps nearly all of the
    saving, because the runs of one session share a reader.

    Asking does not claim. :func:`note_surface_described` records the claim,
    and is called after the event has been handed to the stream rather than
    while its payload is being built: a run cancelled at that yield would
    otherwise have consumed the only description its reader was ever going to
    get.
    """
    return (digest, audience) not in _described


def note_surface_described(digest: str, audience: str) -> None:
    """Record that ``audience`` has now been sent the descriptions of ``digest``."""
    key = (digest, audience)
    _described[key] = None
    _described.move_to_end(key)
    while len(_described) > MAX_TOOL_SURFACE_AUDIENCES:
        _described.popitem(last=False)


def surface_descriptions(digest: str) -> dict[str, str] | None:
    """What each tool of a digested surface does, or ``None`` if unknown here.

    The way back for a reader that met a digest it has no descriptions for. The
    surface has to have been read in this process — which it has, for any
    digest this process advertised — and an eviction or a restart answers
    ``None``, at which point the next advertisement to that reader describes
    the surface again.
    """
    known = _descriptions.get(digest)
    if known is None:
        return None
    _descriptions.move_to_end(digest)
    return dict(known)


def forget_tool_surfaces() -> None:
    """Drop every remembered estimate, description and claim."""
    _estimates.clear()
    _descriptions.clear()
    _described.clear()
