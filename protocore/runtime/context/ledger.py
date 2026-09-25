"""The compaction ledger: facts that leave the window are carried by code.

A summary is a model's account of what happened, and a model rounds, merges,
paraphrases and — under pressure — substitutes a plausible value for the one
it was shown. The values a later turn needs exactly (a port, a path, an id, a
timestamp, the operator's own wording of a rule) are therefore not left to it.
Whenever compaction takes something out of the window — masks a tool output,
summarises a span, folds summaries, drops a span at the floor — the same pass
reads the outgoing messages and records, by code:

* the operator's instructions, verbatim (clipped at a line boundary);
* the files the agent read, wrote or edited, by the tool's declared role;
* identifiers and exact values of recognisable shape — URLs, paths, UUIDs,
  hashes, timestamps, versions, host:port pairs, handles, id-shaped tokens;
* the tool calls that failed, with the first line of what they said;
* the latest plan the agent recorded and the questions it asked.

The ledger is ONE user-role message, tagged :data:`LEDGER_METADATA_KEY`, that
holds its own structured state in that metadata. Every pass rebuilds it from
that state plus what the pass removed, renders it within its budget and puts
it back at the boundary between the compacted past and the raw present. It is
never shown to the summariser — a record that is rewritten by a model on every
fold compounds its errors — and no tier compacts it.

What the ledger does not try to be. Recognising a value by its shape is not
understanding it: a count ("19 shards") or a name has no shape, and those stay
the summary's job. A shape that also matches noise (every timestamp in a log)
is taken only from the places where noise is rare — the operator's turns, the
agent's own text and its tool arguments — and from tool outputs only the
shapes that are rarely noise (URLs, UUIDs, absolute paths, host:port pairs,
version tags). Past its budget the oldest and lowest-ranked entries go first.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.tool_roles import (
    EMPTY_TOOL_ROLE_MAP,
    ToolArgumentSlot,
    ToolRole,
    ToolRoleMap,
)
from protocore.contracts.types import (
    Message,
    MessageRole,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from protocore.runtime.token_counting import estimate_tokens
from protocore.runtime.tool_arguments import string_argument
from protocore.runtime.wire_format import is_compacted_placeholder

LEDGER_METADATA_KEY: Final[str] = "protocore.compaction_ledger"
"""On the ledger message: its structured state, from which it is rebuilt."""

_OPEN_TAG: Final[str] = "<compaction-ledger>"
_CLOSE_TAG: Final[str] = "</compaction-ledger>"

#: Bounds on the structured state, so the metadata cannot grow without limit on
#: a session that compacts for days. Rendering has its own, smaller, budget.
_MAX_OPERATOR: Final[int] = 40
_MAX_FILES: Final[int] = 120
_MAX_IDENTIFIERS: Final[int] = 600
_MAX_ERRORS: Final[int] = 40
_MAX_OPEN: Final[int] = 30

_OPERATOR_QUOTE_MAX_CHARS: Final[int] = 1_200
_ERROR_LINE_MAX_CHARS: Final[int] = 240
_OPEN_ITEM_MAX_CHARS: Final[int] = 400
_IDENTIFIER_MAX_CHARS: Final[int] = 200
_DISTINCT_LINES_READ: Final[int] = 24
_CONTEXT_MAX_CHARS: Final[int] = 120

#: Rank of a source: lower survives longer when the budget is short. What the
#: operator said first; then what the agent stated and what a tool output said
#: once; last the agent's own call arguments, which are mostly the paths and
#: patterns already listed under the files it touched, and the looser shapes
#: of a tool output.
RANK_OPERATOR: Final[int] = 0
RANK_STATED: Final[int] = 1
RANK_CALL: Final[int] = 2

_SHAPES: Final[tuple[tuple[str, re.Pattern[str], bool], ...]] = (
    # kind, pattern, taken from tool outputs too
    ("url", re.compile(r"\bhttps?://[^\s'\"<>)\]}`,]+"), True),
    ("uuid", re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"), True),
    ("host:port", re.compile(r"\b(?:\d{1,3}(?:\.\d{1,3}){3}|[A-Za-z][A-Za-z0-9.-]*):\d{2,5}\b"), True),
    ("version", re.compile(r"(?<![\w.])v\d+(?:\.\d+){1,3}(?:[-+][0-9A-Za-z.-]+)?\b|(?<![\w.])\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?\b"), True),
    ("path", re.compile(r"(?<![\w/.~-])(?:~|\.{1,2})?/(?:[\w.@+-]+/)*[\w.@+-]+"), True),
    ("timestamp", re.compile(
        r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2}|\s?UTC)?)?"
        r"|\b\d{1,2}:\d{2}\s?UTC\b"
    ), False),
    ("hash", re.compile(r"\b(?=[0-9a-f]*[a-f])(?=[0-9a-f]*\d)[0-9a-f]{7,64}\b"), False),
    ("handle", re.compile(r"(?<![\w@.])@[A-Za-z][\w.-]*\w"), False),
    ("domain", re.compile(r"\b(?:[a-z0-9-]+\.){2,}[a-z]{2,10}\b"), False),
    ("file", re.compile(r"(?<![\w/.-])[\w-]+(?:/[\w.@+-]+)*\.[A-Za-z][A-Za-z0-9]{0,7}\b"), False),
    ("id", re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)+\b"), False),
    ("number", re.compile(r"(?<![\w.:/-])\d{4,6}(?![\w.:/-])"), False),
)

#: What a file name ends in; a "domain" that ends in one of these is a file.
_FILE_SUFFIXES: Final[frozenset[str]] = frozenset(
    "py md json txt toml yaml yml ts tsx js jsx sh log cfg ini lock csv sql html css go rs java kt c h cpp rb php xml".split()
)

_PATH_KEYS: Final[tuple[str, ...]] = ("path", "file_path", "filepath", "file", "filename", "target")

_ROLE_ACTIONS: Final[tuple[tuple[ToolRole, str], ...]] = (
    (ToolRole.writes_path, "wrote"),
    (ToolRole.appends_path, "appended"),
    (ToolRole.edits_path, "edited"),
    (ToolRole.finalizes_path, "finalized"),
    (ToolRole.reads_path, "read"),
    (ToolRole.searches_workspace, "searched"),
)


def _clip(text: str, limit: int) -> str:
    """``text`` cut to ``limit`` characters at a line, else a word, boundary."""
    text = text.strip()
    if len(text) <= limit:
        return text
    head = text[:limit]
    for boundary in ("\n", " "):
        at = head.rfind(boundary)
        if at >= limit // 2:
            return head[:at].rstrip() + " …"
    return head.rstrip() + " …"


def _is_noise(kind: str, value: str) -> bool:
    if kind == "number":
        number = int(value)
        return 1900 <= number <= 2100
    if kind == "domain":
        return value.rsplit(".", 1)[-1] in _FILE_SUFFIXES
    if kind == "id":
        return not any(ch.isdigit() for ch in value) or len(value) < 5
    if kind == "path":
        return value.count("/") < 2 and not value.startswith(("~/", "./", "../"))
    if kind == "file":
        return "." not in value.rsplit("/", 1)[-1] or value.rsplit(".", 1)[-1].isdigit()
    return False


def extract_identifiers(text: str, *, from_tool_output: bool) -> list[tuple[str, str]]:
    """``(kind, value)`` pairs of recognisable shape in ``text``, in order of appearance.

    A value matched by an earlier, more specific shape is not reported again
    by a looser one (a URL's path is not also a path).
    """
    found: list[tuple[int, int, str, str]] = []
    taken: list[tuple[int, int]] = []
    for kind, pattern, in_output in _SHAPES:
        if from_tool_output and not in_output:
            continue
        for match in pattern.finditer(text):
            start, end = match.span()
            if any(start < t_end and end > t_start for t_start, t_end in taken):
                continue
            value = match.group(0).rstrip(".,;:)]}'\"")
            if len(value) < 3 or len(value) > _IDENTIFIER_MAX_CHARS or _is_noise(kind, value):
                continue
            taken.append((start, end))
            found.append((start, end, kind, value))
    found.sort()
    return [(kind, value) for _start, _end, kind, value in found]


_SIGNATURE_RE: Final[re.Pattern[str]] = re.compile(r"\b[0-9a-fA-F]{6,}\b|[0-9]+")


def distinct_lines(text: str, *, limit: int) -> list[str]:
    """The lines of a tool output that are unlike its other lines, at most ``limit``.

    A line's shape is the line with every number and hex run replaced; lines
    whose shape occurs once in the output are its distinct lines. Log lines,
    listings and progress output repeat a handful of shapes, so what is left
    is what the output said only once: a setting, a header, an error, a
    result. Lines carrying a value of recognisable shape come first. This is
    what a masked output keeps in its placeholder, and what the ledger reads
    from a tool output — the rest of the output is where noise lives.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) <= 1:
        return []
    shapes: dict[str, int] = {}
    for line in lines:
        shape = _SIGNATURE_RE.sub("#", line)
        shapes[shape] = shapes.get(shape, 0) + 1
    distinct = [line for line in lines if shapes[_SIGNATURE_RE.sub("#", line)] == 1]
    if len(distinct) == len(lines) and len(lines) > limit:
        # Nothing repeats — prose or a file body — so "distinct" says nothing;
        # prefer the lines that carry values.
        distinct = [line for line in lines if extract_identifiers(line, from_tool_output=False)] or distinct
    valued = [line for line in distinct if extract_identifiers(line, from_tool_output=False)]
    rest = [line for line in distinct if line not in valued]
    return [_clip(line, _ERROR_LINE_MAX_CHARS) for line in (valued + rest)[:limit]]


def _context(line: str, value: str) -> str:
    """The line a value stood on, cut to a window around it at word boundaries."""
    line = " ".join(line.split())
    if len(line) <= _CONTEXT_MAX_CHARS:
        return line
    at = max(0, line.find(value))
    half = max(0, (_CONTEXT_MAX_CHARS - len(value)) // 2)
    start = max(0, at - half)
    end = min(len(line), at + len(value) + half)
    if start > 0:
        space = line.find(" ", start)
        start = space + 1 if 0 <= space < at else start
    if end < len(line):
        space = line.rfind(" ", at + len(value), end)
        end = space if space > 0 else end
    return ("… " if start > 0 else "") + line[start:end] + (" …" if end < len(line) else "")


def _tool_action(name: str, roles: ToolRoleMap) -> str:
    held = roles.roles_of(name)
    for role, action in _ROLE_ACTIONS:
        if role in held:
            return action
    return "touched"


def _arguments(block: ToolUseBlock) -> dict[str, Any]:
    try:
        parsed = json.loads(block.arguments_json or "{}")
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _path_argument(arguments: dict[str, Any], roles: ToolRoleMap) -> str:
    value = string_argument(arguments, ToolArgumentSlot.path, roles=roles)
    if value:
        return value
    for key in _PATH_KEYS:
        candidate = arguments.get(key)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return ""


def _first_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return _clip(line.strip(), _ERROR_LINE_MAX_CHARS)
    return ""


@dataclass(slots=True)
class Ledger:
    """The structured state behind the ledger message."""

    operator: list[str] = field(default_factory=list)
    files: dict[str, str] = field(default_factory=dict)
    """Path → last action; insertion order is recency (re-seen moves to the end)."""
    identifiers: dict[str, list[Any]] = field(default_factory=dict)
    """Value → ``[kind, rank]``; insertion order is recency."""
    errors: list[str] = field(default_factory=list)
    open_items: list[str] = field(default_factory=list)
    compacted_messages: int = 0

    # -- state ---------------------------------------------------------------

    def is_empty(self) -> bool:
        return not (self.operator or self.files or self.identifiers or self.errors or self.open_items)

    def to_dict(self) -> dict[str, Any]:
        return {
            "operator": list(self.operator),
            "files": dict(self.files),
            "identifiers": {key: list(value) for key, value in self.identifiers.items()},
            "errors": list(self.errors),
            "open_items": list(self.open_items),
            "compacted_messages": self.compacted_messages,
        }

    @classmethod
    def from_dict(cls, data: Any) -> Ledger:
        if not isinstance(data, dict):
            return cls()
        identifiers: dict[str, list[Any]] = {}
        for key, value in (data.get("identifiers") or {}).items():
            if isinstance(value, list) and len(value) >= 2:
                context = str(value[2]) if len(value) > 2 else ""
                identifiers[str(key)] = [str(value[0]), int(value[1]), context]
        return cls(
            operator=[str(x) for x in data.get("operator") or []],
            files={str(k): str(v) for k, v in (data.get("files") or {}).items()},
            identifiers=identifiers,
            errors=[str(x) for x in data.get("errors") or []],
            open_items=[str(x) for x in data.get("open_items") or []],
            compacted_messages=int(data.get("compacted_messages") or 0),
        )

    def merge(self, other: Ledger) -> None:
        """Fold another ledger's state in, ``other`` being the newer."""
        for quote in other.operator:
            self._add_operator(quote)
        for path, action in other.files.items():
            self._touch(path, action)
        for value, entry in other.identifiers.items():
            self._identifier(str(entry[0]), value, int(entry[1]), str(entry[2]) if len(entry) > 2 else "")
        for line in other.errors:
            self._error(line)
        if other.open_items:
            self.open_items = list(other.open_items)
        self.compacted_messages += other.compacted_messages

    # -- absorbing -----------------------------------------------------------

    def _add_operator(self, quote: str) -> None:
        if quote and quote not in self.operator:
            self.operator.append(quote)
            del self.operator[: max(0, len(self.operator) - _MAX_OPERATOR)]

    def _touch(self, path: str, action: str) -> None:
        self.files.pop(path, None)
        self.files[path] = action
        while len(self.files) > _MAX_FILES:
            self.files.pop(next(iter(self.files)))

    def _identifier(self, kind: str, value: str, rank: int, context: str = "") -> None:
        previous = self.identifiers.pop(value, None)
        if previous is not None:
            if int(previous[1]) < rank or (int(previous[1]) == rank and not context):
                context = str(previous[2]) if len(previous) > 2 else context
            rank = min(rank, int(previous[1]))
        self.identifiers[value] = [kind, rank, context]
        if len(self.identifiers) > _MAX_IDENTIFIERS:
            # Oldest of the lowest rank present goes first.
            worst = max(int(entry[1]) for entry in self.identifiers.values())
            for key, entry in self.identifiers.items():
                if int(entry[1]) == worst:
                    del self.identifiers[key]
                    break

    def _error(self, line: str) -> None:
        if line and line not in self.errors:
            self.errors.append(line)
            del self.errors[: max(0, len(self.errors) - _MAX_ERRORS)]

    def _identifiers_from(self, text: str, *, rank: int) -> None:
        # A value is kept with the line it stood on: "0.0.0.0:47031" alone does
        # not say it is the admin console's address, and a later turn asked
        # for the admin port has to be able to tell.
        for line in text.splitlines():
            for kind, value in extract_identifiers(line, from_tool_output=False):
                self._identifier(kind, value, rank, _context(line, value))

    def absorb(
        self,
        messages: Iterable[Message],
        *,
        roles: ToolRoleMap = EMPTY_TOOL_ROLE_MAP,
        is_operator: Callable[[Message], bool],
        skip: Callable[[Message], bool],
        tool_names: dict[str, str] | None = None,
    ) -> None:
        """Record what ``messages`` carry before they leave the window.

        ``is_operator`` says which user turns the operator wrote; ``skip`` says
        which messages are compaction artefacts (earlier summaries, the ledger
        itself) and must not be read again — a summary's wording is a model's,
        and taking values from it would launder them into the ledger.
        """
        names = dict(tool_names or {})
        for message in messages:
            if skip(message):
                continue
            self.compacted_messages += 1
            if message.role is MessageRole.user and is_operator(message):
                text = message.text
                self._add_operator(_clip(text, _OPERATOR_QUOTE_MAX_CHARS))
                self._identifiers_from(text, rank=RANK_OPERATOR)
                continue
            for block in message.content_blocks:
                if isinstance(block, TextBlock) and message.role is MessageRole.assistant:
                    self._identifiers_from(block.text, rank=RANK_STATED)
                elif isinstance(block, ToolUseBlock):
                    names[block.tool_call_id] = block.name
                    self._absorb_call(block, roles)
                elif isinstance(block, ToolResultBlock) and not is_compacted_placeholder(block.content):
                    # A masked output was recorded when it was masked; what is
                    # on the block now is the placeholder, whose digests and
                    # sizes are not values anybody used.
                    self.absorb_result(block, tool_name=names.get(block.tool_call_id, ""))

    def _absorb_call(self, block: ToolUseBlock, roles: ToolRoleMap) -> None:
        arguments = _arguments(block)
        held = roles.roles_of(block.name)
        path = _path_argument(arguments, roles)
        if path:
            self._touch(path, _tool_action(block.name, roles))
        if ToolRole.records_plan in held:
            rendered = [
                _clip(f"{key}: {value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)}", _OPEN_ITEM_MAX_CHARS)
                for key, value in arguments.items()
            ]
            self.open_items = [f"plan ({block.name}) — {line}" for line in rendered][:_MAX_OPEN]
        elif ToolRole.asks_user in held:
            question = next((v for v in arguments.values() if isinstance(v, str) and v.strip()), "")
            if question:
                self.open_items.append(_clip(f"asked the operator: {question}", _OPEN_ITEM_MAX_CHARS))
                del self.open_items[: max(0, len(self.open_items) - _MAX_OPEN)]
        for kind, value in extract_identifiers(block.arguments_json or "", from_tool_output=False):
            if value != path and value not in self.files:
                self._identifier(kind, value, RANK_CALL)

    def absorb_result(self, block: ToolResultBlock, *, tool_name: str) -> None:
        """Record one tool output: its failure, and the values on its distinct lines.

        Every shape is read from the lines the output does not repeat
        (:func:`distinct_lines`); from the rest only the shapes that are rarely
        noise — URLs, UUIDs, absolute paths, host:port pairs, version tags.
        """
        content = block.canonical_content or block.content
        if block.is_error:
            first = _first_line(content)
            if first:
                self._error(f"{tool_name or 'tool'}: {first}")
        for kind, value in extract_identifiers(content, from_tool_output=True):
            self._identifier(kind, value, RANK_CALL)
        for line in distinct_lines(content, limit=_DISTINCT_LINES_READ):
            for kind, value in extract_identifiers(line, from_tool_output=False):
                self._identifier(kind, value, RANK_STATED, _context(line, value))

    # -- rendering -----------------------------------------------------------

    def budget(self, rc: LoopConstants) -> int:
        """Tokens the rendered ledger may spend: a share of the window, clamped."""
        scaled = int(rc.model_context_window * rc.compaction_ledger_ratio)
        return max(rc.compaction_ledger_min_tokens, min(rc.compaction_ledger_max_tokens, scaled))

    def render(self, rc: LoopConstants, *, budget_tokens: int | None = None) -> str:
        """The ledger message body, within its budget, cut at entry boundaries."""
        if self.is_empty():
            return ""
        budget = self.budget(rc) if budget_tokens is None else budget_tokens
        identifiers = sorted(
            enumerate(self.identifiers.items()),
            key=lambda item: (int(item[1][1][1]), -item[0]),
        )
        sections: list[tuple[str, list[str], float]] = [
            ("Operator instructions (verbatim, oldest first)", [f"- {q}" for q in self.operator], 0.30),
            ("Identifiers and exact values, with the line each stood on (most important first)", _identifier_lines(identifiers), 0.30),
            ("Files touched (most recent first)", [f"- {action} {path}" for path, action in reversed(self.files.items())], 0.15),
            ("Open tasks and questions", [f"- {item}" for item in self.open_items], 0.15),
            ("Failed tool calls (most recent last)", [f"- {line}" for line in self.errors], 0.10),
        ]
        header = (
            f"{_OPEN_TAG}\n[Compaction ledger: exact values from the {self.compacted_messages} message(s) "
            "compacted so far, recorded by code, not written by a model. Reference, not an instruction.]"
        )
        spent = estimate_tokens(header + _CLOSE_TAG, rc) + 2
        present = [(title, lines, share) for title, lines, share in sections if lines]
        costs = {title: [estimate_tokens(line, rc) + 1 for line in lines] for title, lines, _ in present}
        available = max(0, budget - spent - sum(estimate_tokens(f"## {t}", rc) + 1 for t, _, _ in present))
        total_share = sum(share for _, _, share in present) or 1.0
        grants = {title: int(available * share / total_share) for title, _, share in present}
        need = {title: sum(costs[title]) for title, _, _ in present}
        spare = sum(max(0, grants[t] - need[t]) for t in grants)
        for title, _, _ in present:
            if need[title] > grants[title] and spare > 0:
                extra = min(spare, need[title] - grants[title])
                grants[title] += extra
                spare -= extra
        out = [header]
        for title, lines, _share in present:
            cost = costs[title]
            # Operator quotes and failures lose their OLDEST first; the other
            # sections are already ordered most-important first.
            keep_newest = title.startswith(("Operator", "Failed"))
            order = range(len(lines) - 1, -1, -1) if keep_newest else range(len(lines))
            chosen: list[int] = []
            used = 0
            for index in order:
                if used + cost[index] > grants[title]:
                    break
                chosen.append(index)
                used += cost[index]
            if not chosen:
                continue
            chosen.sort()
            omitted = len(lines) - len(chosen)
            out.append(f"## {title}" + (f" ({omitted} omitted)" if omitted else ""))
            out.extend(lines[index] for index in chosen)
        out.append(_CLOSE_TAG)
        return "\n".join(out)


def _identifier_lines(ordered: Sequence[tuple[int, tuple[str, list[Any]]]]) -> list[str]:
    """One entry per source line, in priority order: several values on one line share it."""
    lines: list[str] = []
    seen: set[str] = set()
    for _index, (value, entry) in ordered:
        context = str(entry[2]) if len(entry) > 2 and entry[2] else ""
        text = context if value in context else value
        if text in seen:
            continue
        seen.add(text)
        lines.append(f"- {text}")
    return lines


def is_ledger(message: Message) -> bool:
    return message.metadata.get(LEDGER_METADATA_KEY) is not None


def ledger_from_history(history: Sequence[Message]) -> Ledger:
    """The ledger state the history carries: every ledger message merged, oldest first."""
    merged = Ledger()
    for message in history:
        if is_ledger(message):
            merged.merge(Ledger.from_dict(message.metadata.get(LEDGER_METADATA_KEY)))
    return merged


def ledger_message(ledger: Ledger, rc: LoopConstants, *, seeded_key: str | None = None) -> Message | None:
    """The ledger as a user-role message carrying its own state, or ``None`` when empty."""
    body = ledger.render(rc)
    if not body:
        return None
    metadata: dict[str, Any] = {LEDGER_METADATA_KEY: ledger.to_dict()}
    if seeded_key:
        metadata[seeded_key] = True
    return Message(role=MessageRole.user, content_blocks=[TextBlock(text=body)], metadata=metadata)


__all__ = [
    "LEDGER_METADATA_KEY",
    "RANK_CALL",
    "RANK_OPERATOR",
    "RANK_STATED",
    "Ledger",
    "distinct_lines",
    "extract_identifiers",
    "is_ledger",
    "ledger_from_history",
    "ledger_message",
]
