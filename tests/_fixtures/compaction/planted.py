# ruff: noqa: RUF001 — one planted value is a Cyrillic name on purpose: it has no shape code can recognise.
"""A synthetic long agent session with planted, non-derivable facts.

The method is that of a fold-and-recall study: fifty values are planted,
fourteen are graded by exact match, each appears exactly once, and the graded
ones are spread across the session so that every fold boundary has some in
front of it. None can be derived from another, a range or a pattern; two have
no recognisable shape at all (a count with its unit, a name in Cyrillic), so
only a summary can carry them. The dialogue is an agent's — tool calls, tool
outputs of repetitive log lines, short operator turns — because that is what
the core compacts.

``build(seed)`` returns six chunks of messages, the fourteen graded
``(label, value, where, chunk, sentence)`` and every planted value.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass

from protocore.contracts.types import Message, MessageRole, TextBlock, ToolResultBlock, ToolUseBlock

GRADED: list[tuple[str, str, str, int, str]] = [
    # label, value, where ("result" | "assistant" | "operator"), chunk, sentence
    ("ledger service port", "62114", "result", 0, "listen_port = 62114  # ledger service"),
    ("admin console port", "47031", "result", 1, "admin.bind: 0.0.0.0:47031"),
    ("install lock path", "/opt/kestrel/state/install.lock", "result", 0, "lockfile /opt/kestrel/state/install.lock held by pid 4411"),
    ("ledger shard count", "19 shards", "assistant", 1, "The ledger is split into 19 shards, so the rebalance has to run per shard."),
    ("tenant UUID", "3f1c9a52-7d4e-4b8a-9e21-c5d0b7a6f413", "result", 2, "tenant_id: 3f1c9a52-7d4e-4b8a-9e21-c5d0b7a6f413"),
    ("backup cutoff", "2026-07-14 01:26 UTC", "result", 0, "last consistent backup cutoff: 2026-07-14 01:26 UTC"),
    ("freeze start", "2026-07-18T22:45Z", "operator", 2, "The change freeze starts at 2026-07-18T22:45Z; nothing ships after that."),
    ("certificate expiry", "2026-08-02 06:10 UTC", "result", 3, "notAfter=2026-08-02 06:10 UTC (edge wildcard)"),
    ("cache origin", "https://origin-7.cdn.kestrel-internal.net/assets", "result", 3, "origin = https://origin-7.cdn.kestrel-internal.net/assets"),
    ("DNS zone", "zone-b.kestrel-internal.net", "assistant", 4, "Records for the new pool go into zone-b.kestrel-internal.net, not the legacy zone."),
    ("signing key id", "kid-7F3A9C21", "result", 4, "active signing key: kid-7F3A9C21 (rotated 3 days ago)"),
    ("on-call handle", "@marta.v-oncall", "operator", 1, "If anything pages overnight, the on-call handle is @marta.v-oncall — ping them, not me."),
    ("release tag", "v4.18.2-rc3", "result", 5, "Tagging release v4.18.2-rc3 from commit 9b41e7c"),
    ("owning department", "Отдел расчётов-4", "operator", 4, "Для отчёта: владелец сервиса — Отдел расчётов-4, укажи его в шапке."),
]

DISTRACTOR_KINDS = ("cluster", "tenant code", "licence key", "image digest")

TOOLS = ("Exec", "Read", "Grep", "Exec", "Exec", "Read")

LOG_TEMPLATES = (
    "{ts} INFO  scheduler: job ledger-compact-{n:04d} finished in {ms} ms (rows={rows})",
    "{ts} DEBUG pool: conn {h}/{n} reused, idle={ms}ms",
    "{ts} WARN  replica lag {ms} ms on node-{n:02d}",
    "{ts} INFO  http: GET /v2/accounts/{n}/balance 200 {ms}ms",
    "{ts} INFO  migrate: step {n}/{rows} applied checksum {h}",
    "ok    test_case_{n:03d}  covered -> True   entries {rows} ok {ms}",
)


@dataclass
class Session:
    chunks: list[list[Message]]
    graded: list[tuple[str, str, str, int, str]]
    planted: list[str]


def _filler(rng: random.Random, lines: int) -> str:
    out = []
    for _ in range(lines):
        t = rng.choice(LOG_TEMPLATES)
        out.append(
            t.format(
                ts=f"2026-07-1{rng.randint(0, 9)}T{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d}Z",
                n=rng.randint(1, 9999),
                ms=rng.randint(1, 900),
                rows=rng.randint(10, 5000),
                h=f"{rng.getrandbits(32):08x}",
            )
        )
    return "\n".join(out)


def _distractors(rng: random.Random, count: int) -> list[str]:
    values = []
    for i in range(count):
        kind = DISTRACTOR_KINDS[i % len(DISTRACTOR_KINDS)]
        if kind == "cluster":
            values.append(f"cluster id clu-{rng.getrandbits(24):06x}")
        elif kind == "tenant code":
            values.append(f"tenant code TN-{rng.randint(1000, 9999)}-{rng.choice('ABCDEFGH')}")
        elif kind == "licence key":
            values.append(f"licence key LIC-{rng.getrandbits(40):010X}")
        else:
            values.append(f"image digest sha256:{rng.getrandbits(128):032x}")
    return values


def build(seed: int = 7, *, chunks: int = 6, rounds_per_chunk: int = 21, result_lines: int = 64) -> Session:
    rng = random.Random(seed)
    distractors = _distractors(rng, 36)
    by_chunk: dict[int, list[tuple[str, str, str, int, str]]] = {}
    for fact in GRADED:
        by_chunk.setdefault(fact[3], []).append(fact)
    out: list[list[Message]] = []
    call = 0
    d_index = 0
    for c in range(chunks):
        messages: list[Message] = []
        if c == 0:
            messages.append(Message(role=MessageRole.user, content_blocks=[TextBlock(text=(
                "Migrate the kestrel ledger to the new pool. Check the config, the backups and the certificates first, "
                "keep a record of every port, path, id and time you find, and do not restart anything in production "
                "without asking me."
            ))]))
        facts = list(by_chunk.get(c, []))
        slots = rng.sample(range(2, rounds_per_chunk - 1), k=len(facts) + 6)
        fact_at = {slots[i]: facts[i] for i in range(len(facts))}
        distractor_at = set(slots[len(facts):])
        for r in range(rounds_per_chunk):
            call += 1
            fact = fact_at.get(r)
            if fact is not None and fact[2] == "operator":
                messages.append(Message(role=MessageRole.user, content_blocks=[TextBlock(text=fact[4])]))
                fact = None
            elif r == rounds_per_chunk // 2:
                messages.append(Message(role=MessageRole.user, content_blocks=[TextBlock(text=rng.choice((
                    "Keep going, and summarise what you have found so far at the end.",
                    "Do not touch the legacy zone. Continue with the checks.",
                    "Prefer read-only commands for now.",
                )))]))
            tool = TOOLS[call % len(TOOLS)]
            args = {"command": f"kestrel-ctl inspect --section s{call:03d}"} if tool == "Exec" else ({"path": f"conf/section-{call:03d}.toml"} if tool == "Read" else {"pattern": f"err-{call:03d}", "path": "logs/"})
            thought = f"Checking section {call} of the migration inventory."
            if fact is not None and fact[2] == "assistant":
                thought = fact[4] + " " + thought
                fact = None
            call_id = f"call_{call:04d}"
            messages.append(Message(role=MessageRole.assistant, content_blocks=[
                TextBlock(text=thought),
                ToolUseBlock(tool_call_id=call_id, name=tool, arguments_json=json.dumps(args)),
            ]))
            body = _filler(rng, result_lines)
            lines = body.split("\n")
            if fact is not None and fact[2] == "result":
                lines.insert(rng.randint(3, len(lines) - 3), fact[4])
            if r in distractor_at and d_index < len(distractors):
                lines.insert(rng.randint(3, len(lines) - 3), distractors[d_index])
                d_index += 1
            messages.append(Message(role=MessageRole.tool, content_blocks=[ToolResultBlock(tool_call_id=call_id, content="\n".join(lines))]))
        out.append(messages)
    planted = [f[1] for f in GRADED] + distractors[:d_index]
    return Session(chunks=out, graded=list(GRADED), planted=planted)
