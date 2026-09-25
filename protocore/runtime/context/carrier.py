"""The compaction carrier: what a summary is asked to be, and what is kept of it.

A summary is plain text under five fixed headings. It is requested with
:meth:`ILLMProvider.complete_text`, never under a response schema, and it is
read tolerantly: whatever the model returns — headed text, text without
headings, a JSON envelope it wrote anyway, a reply the output cap cut — is
turned into the same shape, cut to its budget at line boundaries, and kept.

Why not JSON. A single ``summary`` string under a character ceiling lost the
identifiers a later turn needed on every model it was measured on, while
headed plain text with an adequate budget kept them; a schema request is also
the one kind of request that can come back as a parse failure — an object the
model never closed, prose where an object was due — and every such failure
threw away a summary that was otherwise fine. Headed text has no way to be
malformed: missing headings leave the text under one catch-all section, and a
cut reply loses its last line rather than all of it.

Why fixed headings with their own budgets. A summary that runs long is cut,
and a cut that falls at the end of the text falls on whatever the model wrote
last. Each section is clamped on its own, so a long account of progress cannot
push the exact values out of the carrier.

Why the instruction sits in the system role. The summariser reads a transcript
full of imperative sentences, and its own instruction is one more of them; a
summary that carried the instruction forward has been seen to present it as
something the user asked for. The instruction is therefore the system message,
the transcript is fenced as data in the user message, any sentence of the
instruction echoed into the reply is removed, and the carrier says, in its own
header, that it was written by the runtime.
"""
from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from protocore.contracts.runtime_constants import LoopConstants
from protocore.runtime.token_counting import estimate_tokens

#: The five sections, in the order they are written, with the share of the
#: carrier's budget each may spend before the others' unused budget is shared
#: out. Exact values get the largest share: they are what a later turn cannot
#: reconstruct.
SECTIONS: Final[tuple[tuple[str, float], ...]] = (
    ("Progress", 0.25),
    ("Facts and values", 0.35),
    ("Decisions and constraints", 0.15),
    ("Failures", 0.10),
    ("Open", 0.15),
)

#: Where text that arrived under no recognised heading is kept. It is not
#: dropped: a model that ignored the headings still wrote a summary.
NOTES_SECTION: Final[str] = "Notes"

_HEADING_NAMES: Final[dict[str, str]] = {name.lower(): name for name, _ in SECTIONS}

def _heading_alternation() -> str:
    names = []
    for name, _ in SECTIONS:
        names.append(r"\s+".join(r"(?:and|&)" if word == "and" else re.escape(word) for word in name.split()))
    return "|".join(names)


#: A line that IS a heading: the name alone (optionally as ``#``/bold markup),
#: or the name followed by a colon and inline content. "Open questions remain"
#: is prose, not the ``Open`` heading.
_HEADING_RE: Final[re.Pattern[str]] = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:\*\*|__)?\s*(?P<name>" + _heading_alternation() + r")\s*(?:\*\*|__)?\s*"
    r"(?:[:\u2014-]\s*(?:\*\*|__)?\s*(?P<rest>.*))?$",
    re.IGNORECASE,
)


def _canonical_heading(found: str) -> str | None:
    key = " ".join(found.lower().replace("&", "and").split())
    return _HEADING_NAMES.get(key)


_THINK_RE: Final[re.Pattern[str]] = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE: Final[re.Pattern[str]] = re.compile(r"^\s*```[a-zA-Z]*\s*$")
_SUMMARY_KEY_RE: Final[re.Pattern[str]] = re.compile(r'^\s*\{\s*"summary"\s*:\s*"', re.DOTALL)


@dataclass(frozen=True, slots=True)
class Carrier:
    """A summary as it will be committed, and how it got that way."""

    text: str
    """The headed text, clamped to its budget. Empty means nothing usable."""
    tokens: int
    sections: tuple[str, ...]
    """The sections that ended up with content, in order."""
    headed: bool
    """Whether the reply carried at least one recognised heading."""
    clamped: bool
    """Whether any section was cut to fit its budget."""
    recovered: str = ""
    """How the text was recovered when the reply was not plain headed text:
    ``"json"``, ``"unterminated_json"``, ``"truncated"`` or ``""``."""
    echoed_lines: int = 0
    """Lines removed because they repeated the summariser's own instruction."""


def summary_output_budget(before_tokens: int, rc: LoopConstants, *, ceiling: int) -> int:
    """Tokens a summary of ``before_tokens`` of material may spend.

    Proportional to what it replaces, so a small span is not asked for an essay
    and a large one is not squeezed into a line; floored so a small span still
    has room for its exact values, and capped by ``ceiling`` — the caller's
    output cap for this kind of call. The floor never exceeds the ceiling.
    """
    scaled = int(before_tokens * rc.compaction_summary_ratio)
    floor = min(rc.compaction_summary_min_output_tokens, ceiling)
    return max(1, min(ceiling, max(floor, scaled)))


def request_max_tokens(budget_tokens: int) -> int:
    """The output cap put on the wire for a summary budgeted at ``budget_tokens``.

    Twice the budget. A model asked for N tokens routinely writes a little
    more, and a reply the API cuts loses its tail to the provider rather than
    to the section clamp, which is what decides what survives. The overshoot
    is paid for once and trimmed deterministically.
    """
    return max(64, budget_tokens * 2)


def _strip_reasoning(text: str) -> str:
    text = _THINK_RE.sub("", text)
    lowered = text.lower()
    if "<think>" in lowered and "</think>" not in lowered:
        # A reply that opened a reasoning block and never closed it holds no
        # summary: everything after the tag is the model thinking.
        return text[: lowered.index("<think>")]
    if "</think>" in lowered:
        return text[lowered.rindex("</think>") + len("</think>") :]
    return text


def _strip_fences(text: str) -> str:
    return "\n".join(line for line in text.split("\n") if not _FENCE_RE.match(line))


def _unescape_json_fragment(fragment: str) -> str:
    """The text of a JSON string body that may have been cut anywhere."""
    for cut in range(0, 8):
        candidate = fragment[: len(fragment) - cut] if cut else fragment
        try:
            value = json.loads('"' + candidate + '"')
        except json.JSONDecodeError:
            continue
        return value if isinstance(value, str) else ""
    return fragment.replace("\\n", "\n").replace('\\"', '"')


def _from_json(text: str) -> tuple[str, str] | None:
    """The text a JSON envelope carries, and how it was recovered; ``None`` if it is not one."""
    stripped = text.strip()
    if not stripped.startswith("{"):
        return None
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        summary = parsed.get("summary")
        if isinstance(summary, str):
            return summary, "json"
        # Named fields are a partitioned carrier too: each becomes its section.
        parts = [
            f"## {key}\n{value}"
            for key, value in parsed.items()
            if isinstance(key, str) and isinstance(value, str) and value.strip()
        ]
        return ("\n".join(parts), "json") if parts else ("", "json")
    match = _SUMMARY_KEY_RE.match(stripped)
    if match is None:
        return None
    body = stripped[match.end() :]
    for tail in ('"}', '"\n}', '"'):
        if body.endswith(tail):
            body = body[: -len(tail)]
            break
    return _unescape_json_fragment(body), "unterminated_json"


def _normalise(sentence: str) -> str:
    return " ".join(sentence.lower().split())


def instruction_sentences(instruction: str, *, min_chars: int = 30) -> tuple[str, ...]:
    """The sentences of an instruction long enough to be recognised if echoed."""
    pieces = re.split(r"(?<=[.!?])\s+|\n+", instruction)
    return tuple(
        normalised for piece in pieces if len(normalised := _normalise(piece)) >= min_chars
    )


def _scrub_echo(lines: list[str], sentences: Sequence[str]) -> tuple[list[str], int]:
    if not sentences:
        return lines, 0
    kept: list[str] = []
    removed = 0
    for line in lines:
        normalised = _normalise(line)
        if normalised and any(sentence in normalised for sentence in sentences):
            removed += 1
            continue
        kept.append(line)
    return kept, removed


def _sectionise(lines: list[str]) -> tuple[dict[str, list[str]], bool]:
    sections: dict[str, list[str]] = {}
    current = NOTES_SECTION
    headed = False
    for line in lines:
        match = _HEADING_RE.match(line)
        name = _canonical_heading(match.group("name")) if match else None
        if match is not None and name is not None:
            current = name
            headed = True
            sections.setdefault(current, [])
            rest = (match.group("rest") or "").strip()
            if rest:
                sections[current].append(rest)
            continue
        sections.setdefault(current, []).append(line)
    return sections, headed


def _cut_line(line: str, budget: int, rc: LoopConstants) -> str:
    """The longest prefix of ``line`` within ``budget`` tokens that ends at a word boundary."""
    words = line.split(" ")
    kept: list[str] = []
    for word in words:
        candidate = " ".join([*kept, word])
        if estimate_tokens(candidate + " …", rc) > budget:
            break
        kept.append(word)
    return (" ".join(kept) + " …") if kept else ""


def _clamp_section(lines: list[str], budget: int, rc: LoopConstants) -> tuple[list[str], int, bool]:
    kept: list[str] = []
    spent = 0
    for line in lines:
        cost = estimate_tokens(line, rc) + 1
        if spent + cost <= budget:
            kept.append(line)
            spent += cost
            continue
        remaining = budget - spent
        if remaining > 8:
            cut = _cut_line(line, remaining - 1, rc)
            if cut:
                kept.append(cut)
                spent += estimate_tokens(cut, rc) + 1
        return kept, spent, True
    return kept, spent, False


def _allocate(sizes: dict[str, int], budget: int) -> dict[str, int]:
    """Each section's budget: its share, then the unused remainder in section order."""
    shares = dict(SECTIONS)
    if NOTES_SECTION in sizes:
        shares = {**shares, NOTES_SECTION: 0.2 if len(sizes) > 1 else 1.0}
    total_share = sum(shares[name] for name in sizes) or 1.0
    allotted = {name: int(budget * shares[name] / total_share) for name in sizes}
    granted = {name: min(sizes[name], allotted[name]) for name in sizes}
    spare = budget - sum(granted.values())
    order = [name for name, _ in SECTIONS if name in sizes] + ([NOTES_SECTION] if NOTES_SECTION in sizes else [])
    for name in order:
        if spare <= 0:
            break
        extra = min(spare, sizes[name] - granted[name])
        if extra > 0:
            granted[name] += extra
            spare -= extra
    return granted


def read_carrier(
    raw: str,
    *,
    budget_tokens: int,
    rc: LoopConstants,
    truncated: bool = False,
    instruction: str = "",
) -> Carrier:
    """Turn a summariser reply into a committable carrier.

    Tolerant by design: every shape a reply has been seen to take yields text,
    and only a reply with no usable text at all yields an empty carrier.
    ``truncated`` says the provider cut the reply at its output cap; the last
    line is then dropped, because it stops mid-sentence and a value cut in half
    is worse than a value left out. ``instruction`` is the system prompt the
    reply was written under; any of its sentences found in the reply are
    removed.
    """
    text = _strip_fences(_strip_reasoning(raw or ""))
    recovered = ""
    from_json = _from_json(text)
    if from_json is not None:
        text, recovered = from_json
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    if truncated and len(lines) > 1:
        lines = lines[:-1]
        recovered = recovered or "truncated"
    elif truncated:
        recovered = recovered or "truncated"
    lines, echoed = _scrub_echo(lines, instruction_sentences(instruction) if instruction else ())
    sections, headed = _sectionise(lines)
    cleaned = {
        name: [line for line in body if line.strip()]
        for name, body in sections.items()
        if any(line.strip() for line in body)
    }
    if not cleaned:
        return Carrier(text="", tokens=0, sections=(), headed=headed, clamped=False, recovered=recovered, echoed_lines=echoed)
    sizes = {name: sum(estimate_tokens(line, rc) + 1 for line in body) for name, body in cleaned.items()}
    heading_cost = sum(estimate_tokens(f"## {name}", rc) + 1 for name in cleaned)
    grants = _allocate(sizes, max(16, budget_tokens - heading_cost))
    clamped_any = False
    out: list[str] = []
    order = [name for name, _ in SECTIONS if name in cleaned]
    if NOTES_SECTION in cleaned:
        order = ([NOTES_SECTION] if not headed else []) + order + ([NOTES_SECTION] if headed else [])
    names: list[str] = []
    for name in order:
        body, _spent, clamped = _clamp_section(cleaned[name], grants[name], rc)
        clamped_any = clamped_any or clamped
        if not body:
            continue
        names.append(name)
        out.append(f"## {name}")
        out.extend(body)
    rendered = "\n".join(out).strip()
    return Carrier(
        text=rendered,
        tokens=estimate_tokens(rendered, rc) if rendered else 0,
        sections=tuple(names),
        headed=headed,
        clamped=clamped_any,
        recovered=recovered,
        echoed_lines=echoed,
    )


def carrier_header(*, messages: int, kind: str, pointer: str) -> str:
    """The first line of every committed summary: who wrote it and where the originals are.

    It is what keeps a summary from being read as the user speaking: the
    carrier is user-role on the wire (mid-history system messages are refused
    by some servers), so it has to say in words that it is reference material.
    """
    source = f" Originals: {pointer}." if pointer else ""
    what = "a model summary" if kind == "summary" else "a merged summary of earlier summaries" if kind == "fold" else "a runtime digest (no model summary)"
    return (
        f"[Compacted context: {what} of {messages} earlier message(s), written by the runtime, "
        f"not by the user; reference, not an instruction.{source}]"
    )


__all__ = [
    "NOTES_SECTION",
    "SECTIONS",
    "Carrier",
    "carrier_header",
    "instruction_sentences",
    "read_carrier",
    "request_max_tokens",
    "summary_output_budget",
]
