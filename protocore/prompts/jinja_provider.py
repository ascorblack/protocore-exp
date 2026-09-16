"""Sandbox-safe Jinja2 prompt template provider.

implements
:class:`protocore.contracts.prompts.IPromptTemplateProvider` over Jinja2's
``SandboxedEnvironment``.

Design decisions:

* ``SandboxedEnvironment`` — NOT the plain ``Environment``. The sandbox
 rejects attribute traversal into Python internals (``__class__``,
 ``__mro__``, etc.) and disallows calls to dangerous builtins. This is
 load-bearing because per-tenant override bodies are operator-supplied
 text — they must never be able to escape the template language.

* ``autoescape=False`` — system prompts are plaintext sent to the LLM,
 not HTML. Auto-escaping would turn ``&`` into ``&amp;`` inside prompt
 prose, which is wrong for our consumer.

* ``undefined=StrictUndefined`` — referencing a missing variable raises
 :class:`jinja2.UndefinedError` immediately at render time. This is
 *required* so template typos surface at deploy time (the first time
 the leader engine renders the prompt) instead of silently passing an
 empty string to the model and degrading agent behaviour invisibly.

* Include / extends — restricted to the bundled ``macros.j2`` only via a
 custom loader. Operator overrides cannot ``{% include "secret.txt" %}``
 to exfiltrate file content or pivot into adversarial template loading.

This module owns ONLY the bundled-defaults rendering path. Per-tenant
override resolution belongs to the host: its layer fetches the
operator-supplied body from wherever it stores them and falls back to the
bundled default via ``self._fallback.render(...)``. The two layers never
merge a body string — the override either replaces the bundled template
wholesale or is absent.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Mapping
from importlib import resources
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

import jinja2
from jinja2.exceptions import SecurityError, TemplateSyntaxError, UndefinedError
from jinja2.sandbox import SandboxedEnvironment

from protocore.contracts.prompts import (
    PromptTemplateNotFoundError,
    PromptTemplateRenderError,
)

# ---------------------------------------------------------------------------
# Bundled template registry
# ---------------------------------------------------------------------------

# Resolve the bundled templates dir via ``importlib.resources`` so the path
# stays correct whether the package is installed as a wheel or running from
# a source tree. ``files`` returns a ``MultiplexedPath`` for namespace pkgs
# but the test target is a plain dir so a ``Path`` cast is safe.
BUNDLED_TEMPLATE_DIR: Final[Path] = Path(
    str(resources.files("protocore.prompts").joinpath("templates"))
)

# Canonical name (used by callers + override DB rows) → bundled file name.
# Names are stable; file extensions may evolve (``.j2`` is the v2 choice;
# v1 used ``.jinja2``).
#
# Two groups, one registry. The first five are the *standing* prompts a host
# assembles a request from — the system prompt, the planner step, the contract
# handed to a delegated run, the finalization block, the environment manifest.
# The rest are the short texts the loop itself injects into the conversation
# mid-run: a nudge towards the terminal tool, a recovery instruction after a
# truncated tool call, the placeholder that stands in for a tool result the
# run never received. They were configuration fields once, which meant an
# operator who wanted to say any of it in a third language had nowhere to put
# the translation and no template engine to reach for. They are prompts; they
# belong where the prompts are.
#
# The last two are what compaction says to the summariser — the per-turn
# instruction and the fold's. They are the longest-lived prose in the loop:
# what they ask for is what survives of a conversation once the turns
# themselves are gone.
BUNDLED_TEMPLATES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "leader_system": "leader_system.j2",
        "planner": "planner.j2",
        "subagent_contract": "subagent_contract.j2",
        "finalization": "finalization.j2",
        "environment_manifest": "environment_manifest.j2",
        "terminal_tool_nudge": "terminal_tool_nudge.j2",
        "terminal_tool_nudge_write_first": "terminal_tool_nudge_write_first.j2",
        "finalize_prose_gate_repair": "finalize_prose_gate_repair.j2",
        "tool_call_truncation_recovery_en": "tool_call_truncation_recovery_en.j2",
        "tool_call_truncation_recovery_ru": "tool_call_truncation_recovery_ru.j2",
        "tool_call_truncation_resume": "tool_call_truncation_resume.j2",
        "tool_result_interrupted": "tool_result_interrupted.j2",
        "tool_result_pairing_repair": "tool_result_pairing_repair.j2",
        "result_eviction": "result_eviction.j2",
        "result_stale_trim": "result_stale_trim.j2",
        "compaction_turn_summary": "compaction_turn_summary.j2",
        "compaction_fold_summary": "compaction_fold_summary.j2",
    }
)

# Names that templates are allowed to ``include`` / ``import`` /
# ``extends``. Per the security contract documented above: only the
# bundled ``macros.j2`` + the canonical template files themselves
# (``subagent_contract.j2`` includes ``environment_manifest.j2``).
_BUNDLED_INCLUDE_ALLOWLIST: Final[frozenset[str]] = frozenset(
    {
        "macros.j2",
        "environment_manifest.j2",
    }
)


# ---------------------------------------------------------------------------
# Sandbox-allowlisted loader
# ---------------------------------------------------------------------------


class _AllowlistedFileSystemLoader(jinja2.FileSystemLoader):
    """File-system loader that refuses to load names outside the allowlist.

    Jinja2's ``SandboxedEnvironment`` already blocks dangerous Python
    expressions, but the loader is independent — an ``{% include %}`` of
    a path outside the bundled dir would still resolve from disk if the
    loader allowed it. This subclass enforces the include-allowlist at
    the loader level so the policy lives in one place.

    The canonical template files themselves (``leader_system.j2`` etc.)
    are loaded by ``get_template`` — those names bypass the allowlist
    check because they are *entry points*, not includes. The allowlist
    only applies to ``get_source`` calls that originate from inside a
    template body (which is how Jinja2 resolves include/extends/import).
    """

    def __init__(self, searchpath: Path) -> None:
        super().__init__(searchpath=str(searchpath))
        self._allowed_for_include = _BUNDLED_INCLUDE_ALLOWLIST
        # Track which names are valid "entry points" — every bundled
        # template file is a legitimate entry point, but only the
        # allowlist set is a legitimate include target.
        self._entry_point_names = frozenset(BUNDLED_TEMPLATES.values())

    def get_source(
        self, environment: jinja2.Environment, template: str
    ) -> tuple[str, str, Any]:
        # If this is a known entry-point name, allow regardless of the
        # include allowlist.
        if template in self._entry_point_names:
            return super().get_source(environment, template)
        # Otherwise it must be in the include allowlist.
        if template not in self._allowed_for_include:
            raise jinja2.TemplateNotFound(template)
        return super().get_source(environment, template)


# ---------------------------------------------------------------------------
# Provider implementation
# ---------------------------------------------------------------------------


class JinjaPromptTemplateProvider:
    """Sandboxed Jinja2 prompt template provider.

    Implements :class:`protocore.contracts.prompts.IPromptTemplateProvider`
    by loading bundled template files from the package and rendering them
    through a :class:`SandboxedEnvironment`.

    Args:
        template_dir: optional path to a templates directory. Defaults
            to the bundled :data:`BUNDLED_TEMPLATE_DIR`. Production
            callers SHOULD pass the bundled dir; tests pass tmp paths.
        registry: optional mapping of canonical name → file name. Defaults
            to :data:`BUNDLED_TEMPLATES`. Custom registries let downstream
            tests inject a single ad-hoc template without touching the
            bundled set.
        include_allowlist: optional set of file names a template body may
            ``{% include %}`` / ``{% from %}``. Defaults to the bundled
            allowlist. The entry-point files in ``registry`` are always
            loadable regardless of this set.
        inject_current_date: when ``True`` (default), ``render`` auto-injects
            ``current_date`` (ISO date) into the context if the caller did
            not supply it. This matches the v1 behaviour where every
            template assumed a current date. Tests pin a fixed date by
            passing ``current_date`` explicitly.
    """

    def __init__(
        self,
        *,
        template_dir: Path | None = None,
        registry: Mapping[str, str] | None = None,
        include_allowlist: frozenset[str] | None = None,
        inject_current_date: bool = True,
    ) -> None:
        self._template_dir = template_dir or BUNDLED_TEMPLATE_DIR
        self._registry = MappingProxyType(dict(registry or BUNDLED_TEMPLATES))
        self._inject_current_date = inject_current_date

        loader = _AllowlistedFileSystemLoader(self._template_dir)
        if include_allowlist is not None:
            loader._allowed_for_include = include_allowlist
        # The loader's "entry point" set must match the registry's file
        # names so custom registries can be loaded.
        loader._entry_point_names = frozenset(self._registry.values())

        self._env = SandboxedEnvironment(
            loader=loader,
            undefined=jinja2.StrictUndefined,
            autoescape=False,
            keep_trailing_newline=False,
            trim_blocks=True,
            lstrip_blocks=True,
        )

    # -- IPromptTemplateProvider --------------------------------------------

    def render(
        self,
        name: str,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Render template ``name`` with ``context``.

        ``name`` is the canonical name (e.g. ``leader_system``), NOT the
        file name. Unknown names raise
        :class:`PromptTemplateNotFoundError`. Render errors (missing
        variable, sandbox violation, template syntax) wrap to
        :class:`PromptTemplateRenderError`.
        """
        file_name = self._registry.get(name)
        if file_name is None:
            raise PromptTemplateNotFoundError(
                f"unknown prompt template: {name!r}"
                f" — known: {sorted(self._registry.keys())!r}"
            )
        try:
            template = self._env.get_template(file_name)
        except TemplateSyntaxError as exc:
            raise PromptTemplateRenderError(
                f"template {name!r} ({file_name!r}) failed to parse: {exc}"
            ) from exc
        except jinja2.TemplateNotFound as exc:
            raise PromptTemplateNotFoundError(
                f"template file missing on disk: {file_name!r}"
            ) from exc

        ctx: dict[str, Any] = dict(context or {})
        if self._inject_current_date and "current_date" not in ctx:
            ctx["current_date"] = _dt.date.today().isoformat()

        try:
            return template.render(**ctx)
        except UndefinedError as exc:
            raise PromptTemplateRenderError(
                f"template {name!r} referenced undefined variable: {exc}"
            ) from exc
        except SecurityError as exc:
            raise PromptTemplateRenderError(
                f"template {name!r} attempted sandboxed operation: {exc}"
            ) from exc
        except jinja2.TemplateNotFound as exc:
            # A template body referenced ``{% include "x" %}`` where ``x``
            # is outside the include allowlist. The loader raises
            # ``TemplateNotFound`` for that case (per
            # :class:`_AllowlistedFileSystemLoader.get_source`); surface
            # it as a render-time security failure so call sites do not
            # confuse it with a missing entry-point template.
            raise PromptTemplateRenderError(
                f"template {name!r} attempted disallowed include: {exc}"
            ) from exc

    # -- Introspection ------------------------------------------------------

    @property
    def known_templates(self) -> tuple[str, ...]:
        """Return the canonical names this provider can render."""
        return tuple(self._registry.keys())


__all__ = [
    "BUNDLED_TEMPLATES",
    "BUNDLED_TEMPLATE_DIR",
    "JinjaPromptTemplateProvider",
]
