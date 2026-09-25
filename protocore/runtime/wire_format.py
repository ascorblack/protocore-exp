"""Compaction wire-format placeholders — byte-deterministic renderer.

``PROTOCOL_COMPACTED_TOOL_RESULT_V1:SNAPSHOT`` / ``:RERUN`` placeholders
preserve exact provenance after compaction.

The placeholder optionally carries the originating tool NAME plus a short
head/tail PREVIEW of the original content, base64-encoded so they can hold
arbitrary bytes (pipes, newlines, Cyrillic) without colliding with the
pipe-delimited frame.  This makes a compacted tool result *recoverable* —
the model can see what was shed and re-fetch it through the normal
file/Bash tools — instead of the prior information dead-end.  (A full
``recall_artifact`` recall tool is deferred; the blob is already stored,
only the read tool is missing.)  Older placeholders with no trailing
tool/preview fields still parse — the extra fields are optional and
default to empty.

Pure renderer = byte-deterministic across pods.
"""
from __future__ import annotations

import base64
import re
from typing import Final, Literal

from protocore.constants import PROTOCOL_COMPACTED_TOOL_RESULT_V1
from protocore.contracts.types import CompactionSourceRef

CompactionVariant = Literal["SNAPSHOT", "RERUN"]

_MARKER: Final[str] = PROTOCOL_COMPACTED_TOOL_RESULT_V1


def _b64(text: str) -> str:
    """URL-safe base64 of ``text`` (no ``|`` or newline in the alphabet)."""
    if not text:
        return ""
    return base64.urlsafe_b64encode(text.encode("utf-8")).decode("ascii")


def _unb64(token: str) -> str:
    """Inverse of :func:`_b64`. Empty token → empty string."""
    if not token:
        return ""
    return base64.urlsafe_b64decode(token.encode("ascii")).decode("utf-8")


def render_compacted_placeholder(
    ref: CompactionSourceRef,
    variant: CompactionVariant = "SNAPSHOT",
) -> str:
    """Render a wire-format placeholder string.

    Format::

        PROTOCOL_COMPACTED_TOOL_RESULT_V1:SNAPSHOT|{blob_ref}|{sha256}|{tokens}|{label}|{tool_name_b64}|{preview_b64}

    Byte-deterministic — same inputs always yield same output. Used as
    the message-content replacement for compacted tool results. The two
    trailing base64 fields hold the originating tool name
    and a short content preview; both default to empty so a ref with no
    enrichment renders the original 5-field frame plus two empty tails.
    """
    return (
        f"{_MARKER}:{variant}|"
        f"{ref.blob_ref}|{ref.sha256}|{ref.original_tokens}|{ref.label}|"
        f"{_b64(ref.tool_name)}|{_b64(ref.preview)}"
    )


# ``label`` is always a pipe-free token (e.g. ``tool_result``) so it is
# matched non-greedily as ``[^|]*``; the two optional trailing base64
# fields (tool_name, preview) are appended after it. Both are optional so
# a legacy 5-field placeholder (no trailing pipes) still parses.
_PARSE_RE = re.compile(
    rf"^{re.escape(_MARKER)}:(?P<variant>SNAPSHOT|RERUN)\|"
    r"(?P<blob_ref>[^|]+)\|"
    r"(?P<sha256>[a-fA-F0-9]*)\|"
    r"(?P<tokens>\d+)\|"
    r"(?P<label>[^|]*)"
    r"(?:\|(?P<tool_name>[^|]*)\|(?P<preview>[^|]*))?$"
)


def parse_compacted_placeholder(text: str) -> tuple[CompactionSourceRef, CompactionVariant] | None:
    """Parse a placeholder string back to a ref + variant.

    Returns ``None`` if ``text`` is not a placeholder. Inverse of
    :func:`render_compacted_placeholder`. Tolerates legacy 5-field
    placeholders (no trailing tool_name/preview fields), and an empty digest —
    which is what a placeholder says when it points at bytes whose hash the
    writer had no honest way to state. Only the first line is the frame:
    compaction follows it with a line written for the model to read.
    """
    match = _PARSE_RE.match(text.strip().split("\n", 1)[0])
    if not match:
        return None
    tool_name_token = match.group("tool_name") or ""
    preview_token = match.group("preview") or ""
    return (
        CompactionSourceRef(
            blob_ref=match["blob_ref"],
            sha256=match["sha256"],
            original_tokens=int(match["tokens"]),
            label=match["label"],
            tool_name=_unb64(tool_name_token),
            preview=_unb64(preview_token),
        ),
        match["variant"],  # type: ignore[return-value]
    )


def is_compacted_placeholder(text: str) -> bool:
    """Return ``True`` if ``text`` begins with the wire-format marker."""
    return text.strip().startswith(f"{_MARKER}:")


__all__ = [
    "CompactionVariant",
    "is_compacted_placeholder",
    "parse_compacted_placeholder",
    "render_compacted_placeholder",
]
