# Security Policy

## Supported versions

Protocore is pre-1.0. Only the latest published version receives security
fixes; there are no maintained back-branches yet. That changes once a 1.0
line exists, and this file will say so when it does.

## Reporting a vulnerability

**Please do not open a public issue for a security problem.**

Report it privately through GitHub's
[private vulnerability reporting](https://github.com/anchor-inference/protocore/security/advisories/new)
on this repository. Enable that feature under *Settings → Code security* if
the link does not resolve; until it is on, please contact a maintainer
directly rather than filing publicly.

Include whatever you have:

- the affected version and how the core was wired (which adapters, which
  `RuntimeConstants` were off the defaults);
- reproduction steps, or a proof of concept;
- what an attacker gets out of it;
- a suggested fix, if you have one in mind.

You do not need a complete analysis to report something. A reproduction and
a description of what looked wrong is enough.

## What to expect

- Acknowledgement within 3 business days.
- A triage verdict — accepted, needs more information, or not a
  vulnerability, with reasoning — within 7 business days.
- A fix timeline that depends on severity. We will tell you what it is
  rather than leaving the report open silently.

We coordinate disclosure with you and credit you in the release notes unless
you would rather we did not.

## Scope

This repository is a **library**. It has no network listener, no database
driver, and no deployment surface of its own, so the interesting boundary is
what it does with untrusted input on behalf of the application embedding it.
In scope, for example:

- prompt-template rendering escaping its sandbox
  (`protocore/prompts/jinja_provider.py`);
- the shell-command safety policy failing to classify a dangerous command
  (`protocore/safety/shell.py`, `protocore/runtime/chain_parser.py`);
- the tool-permission gate granting a tool it should have denied
  (`protocore/runtime/tool_permission.py`);
- unbounded resource consumption from model-supplied data — parser depth,
  token accounting, retrieval (`protocore/json_utils.py`,
  `protocore/constants.py`).

Out of scope: vulnerabilities in an adapter you wrote, in a model provider,
or in a deployment that wires this core up. Those belong to whoever owns
that code — though if a core contract made the mistake easy to make, we do
want to hear about it.
