# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **A third compaction pass, for what the first two cannot touch.** Tier 2
  leaves one summary per tool batch and never re-summarises one, and it now
  refuses operator turns outright, so a long session ends up with a window made
  almost entirely of small summaries and instructions that no pass can shrink.
  The fold replaces each contiguous run of them with a single consolidated
  summary in which the operator's instructions survive as exact quotes. The
  task turn and the most recent instructions stay verbatim, a turn seeded from
  an earlier run of the session is never folded, and a fold is a summary like
  any other — a later fold absorbs it once its neighbourhood has grown again.
  `compaction_fold_enabled`, `compaction_fold_min_messages`,
  `compaction_fold_min_tokens`, `compaction_fold_keep_operator_turns`,
  `compaction_fold_max_spans_per_pass`, `compaction_fold_max_output_tokens` and
  `compaction_fold_summary_target_words` govern it; the completion event
  carries `tier3_folded` beside the counts the other tiers report.
- **The summariser instructions are templates.** `compaction_turn_summary` and
  `compaction_fold_summary` join the bundled registry, so the wording is
  reviewable as prose and an operator serving another language has somewhere to
  put the translation.

### Changed

- **Compaction summaries keep exact identifiers.** The summariser is told to
  carry every path, id, port, URL, number and error code through verbatim
  rather than substituting a plausible value, and to state an outcome with no
  tool result or confirmation behind it as UNKNOWN rather than as done or not
  done. A summary that quietly rounds an identifier is worse than no summary,
  because the run reads it back as fact.
- **An operator turn is never summarised.** An instruction is short enough that
  paraphrasing it frees almost nothing and specific enough that the paraphrase
  is a rewrite. Tier 3 is where those turns are condensed, with their wording
  quoted rather than restated.
- **A compaction pass no longer crawls.** Summariser calls go out
  `compaction_summariser_parallelism` at a time instead of one after another
  while the run sits in `COMPACTING`; a unit below
  `compaction_summary_min_unit_tokens` is not sent at all, since a summariser
  writes a sentence or three whatever it is handed and below some size the call
  is spent only to discover the summary is no smaller; and the word budget in
  the prompt scales with the unit being replaced rather than being a fixed
  sentence count. A reply that carries no usable summary is logged with its
  head, so the next one can be diagnosed rather than guessed at.

### Fixed

- **The wind-down notice does not outlive its run.** The "tools are withdrawn,
  write your answer" message the loop writes into history when a run winds
  down stayed there after the run ended. When the wind-down's own final turn
  failed — a provider outage that caused the wind-down also killed the answer —
  the notice was the last thing in the session, and the next turn obeyed it:
  the model reported what it had not finished and called nothing, although
  every tool was back. The notice is removed once the drive reaches a terminal
  state and written back only for a resumed run that is still wound down.

## [2.0.0a4]

This release is the result of a long pass over the core with one question in
front of it: what belongs in a universal agent runtime, and what only ever
belonged to the layer above it. The answer moved a great deal of code out,
tightened what remains into declared contracts, and made several things that
were conventions into checks. The public surface is narrower than it was and
says what it means; that is the point of the release, and it is a breaking one.

### Added

- **Conformance suites, shipped in the package.** `protocore.conformance` is a
  pytest suite a host runs against its **own** implementations of the contracts:
  `pip install "protocore[testing]"`, then `pytest --pyargs protocore.conformance`.
  It replaces "read the Protocol and hope" with a suite that fails when an
  adapter is subtly wrong — a store that loses ordering, a client that reports a
  stream idle without ending it, a sink that drops a field.
- **A constants registry.** Every tunable is declared once, with its bounds, its
  type, its default and the relationships it must hold with its neighbours, in a
  form a program can read. Coercion, validation, a whole-snapshot check and a
  repair that resets an out-of-range field are part of the model rather than
  something each caller reimplements.
- **A request manifest, and a provider that replays it.** Every model request
  now records what it was assembled from, with a digest taken over exactly the
  fields the model can see, so a request is reproducible and a replay that
  diverges is a real difference rather than a timestamp. Observability metadata
  is deliberately outside the digest: the same request stays the same request
  when only its labels differ.
- **A durable record of a tool call before it runs.** The intent is written
  before the effect, so a process that dies between deciding to call a tool and
  calling it resumes with the decision intact, and the charge against a run's
  budget is idempotent per call rather than per attempt.
- **An optional native token estimator**, released separately as
  `protocore-native` and built from source until there are wheels for it. The
  core stays pure Python and selects the extension only when it can import it,
  so having it changes speed and nothing else — the same numbers either way, and
  both arrangements are tested on every supported Python.
  `PROTOCORE_DISABLE_NATIVE=1` keeps the Python implementation in force when the
  extension is installed; it is read once, at import.
- **A `testing` extra** carrying just a test runner, so a host using the
  conformance suites does not inherit the core's linting and typing toolchain.
- **Turn policy as a contract.** The driver of an assistant turn kept the
  mechanics — open a stream, translate deltas, dispatch calls, close the round —
  and every product opinion that had grown into a branch inside it is now an
  object: it declares the named seams of a turn at which it wants to be
  consulted, is consulted in an order the core owns, and answers with events to
  forward and one directive saying what the loop does next. A host's policies
  are **merged with the core's by name rather than replacing them**, so the
  core's own guarantees cannot be switched off by omission, and a directive a
  seam cannot honour — asking to restart a turn at a completion seam — is
  refused with a named error instead of being ignored.
- **One session work pool for both kinds of work.** A background command and a
  delegated child run are the same thing from the loop's side: a unit of work
  with an address, a status, a way to wait for it and a way to stop it. A child
  run therefore has an address, can be asked how far along it is, and can be
  stopped — and a parent waiting on one no longer holds its turn and its slot in
  the tree budget for the whole descendant run. A pool is a collaborator the
  host injects, and a cold start that fails to re-attach a session's still
  running work says so on the run instead of looking like a session with nothing
  running.
- **Interrupts are parked, declared and resumed as a set.** A turn parks every
  held call, announces the whole set in one event, and is resumed with one call
  carrying the resolutions — instead of a turn that could only ever be stopped
  by the first thing that interrupted it.

### Changed

- **The constants snapshot carries the loop's settings and not the host's.**
  What used to be one enormous per-tenant model was a mixture: values the run
  loop reads on every turn, and values only a service layer above the core ever
  looked at. The second group has left the core entirely, and what remains is
  named for what it is. A host that kept its own settings in this model moves
  them into its own; the registry above is how it declares them.
- **One public entry point for resuming a run.** `resume(engine, snapshot, ...)`
  restores from the snapshot first — identity, delivery mode and schema are all
  checked before the first mutation, so a refusal drives nothing — and only then
  chooses how to continue: an approved tool call, a new message, or neither.
  Asking for two at once is an error rather than a guess. The weaker per-turn
  entry point is no longer part of the public surface.
- **One canonical tool result, with projections taken from it.** The typed
  result carries both its success flag and its content blocks, and the shapes a
  model, a user interface and a store each need are derived from it rather than
  maintained beside it.
- **Tool identity comes from a declared role map**, not from tool names spelled
  as literals across the runtime. Which tools delegate, which have side effects,
  which may never be delegated — each is now a property something declares once
  and the loop reads, instead of a name repeated in fifty-five places where a
  rename could silently miss one.

### Fixed

- **The streaming JSON parser no longer costs more than the text it reads.**
  Repairing a partial document used regular expressions that were retried at
  every quote and, on an unterminated string, walked to the end of the buffer
  each time; growth was quadratic. It is now a single string-aware pass, and the
  incremental parser keeps a mirror of the value being built so that the cost of
  a chunk is the depth of the structure rather than the length of the buffer.
  Measured: 28 KiB delivered in 64-byte chunks, 4.84 s to 0.017 s; a single
  repair of a 64 KiB truncated string full of escapes, 18.65 s to 0.016 s.
  A repair that reached a Python-only literal one level down could also return a
  `set` from a JSON parser; every level is now normalised or refused.
- **Token estimates are memoised across a turn**, so a long history is not
  re-measured from scratch on every pass over it.
- **The transient-retry counter resets when the stream settles**, not only on a
  clean round, so a round that ended by tripping the backstop now refreshes the
  retry budget in the same place as every other round.
- `QueryEngine.rearm()` now also restarts the state that is attached to an
  engine *after* it is constructed. The re-arm rebuilds from a fresh engine, and
  `vars()` of a fresh engine cannot see what the host or the run loop attaches
  later, so three things survived every re-arm in silence: the cached tool
  dispatcher, which holds a tool-error counter read out of a helper bag the host
  may since have replaced; the fire-once warning latch for a normalised outbound
  system prompt, which made its warning fire once per engine rather than once
  per run; and the per-run streaks the dispatcher keeps inside the helper bag —
  the consecutive same-tool-same-error cell, the sandbox-down streak and its
  one-shot injection flag, the string-type streak, and the subagent soft-cap
  counts. An agent that repeats one failing call at the start of each turn could
  cross a cap documented as per-run that no single turn ever reached. The bag
  itself belongs to the host and is left alone, as is the run tree's shared work
  ledger. A test now reads the package for every `engine.x = ...` and
  `setattr(engine, "x", ...)` outside the constructor and fails when one is
  classified as neither dropped nor kept.
- **A run resumed from a snapshot is bound to the run it came from.** A snapshot
  whose identity does not match is refused instead of quietly driving another
  run's state.
- **A cold resume restores the whole of a run's accounting**, not the part that
  happened to be constructed with the engine: the position of the provider
  chain, the cumulative budgets of a run tree (a dead process holds no permits,
  so occupied slots come back released), the durable fact that a run was
  cancelled, and the session's background tasks.
- **Context is rebuilt when a request falls back to a generic shape**, and the
  idle-stream branch of provider fallback is now reachable and covered — it
  previously could not be entered at all.

## [2.0.0a3]

Supersedes 2.0.0a2, whose files were removed from the index. That build carried
comments and docstrings that cited internal working documents as the authority
for public behaviour, and that kept the labels a review round leaves behind in
the code it reviewed. A reader outside the project could see that closed
documents govern this library and could read nothing of them. No functional
difference; the prose now states each reason in its own terms.

The publication scanner gained the rules that would have caught it — a document
cited as authority, a review-round trace, a work-package label, a planning or
triage label — so the class cannot come back silently.

### Fixed

- `QueryEngine.rearm()` now restarts every per-run allowance rather than a
  named twelve. An engine that takes an unbounded number of turns on one
  history carried the rest across, and each one ended the agent quietly: the
  identical-tool loop guard counted a fingerprint for the life of the engine,
  so an agent that opened every turn with the same observing call was refused
  it from the fourth turn on; the repeated-error circuit breaker's block list
  is unioned into the visible tool surface, so a tool that failed for a reason
  that had since passed was withdrawn for good; the cooperative stop flag had
  no lowering seam, so an agent interrupted once never spoke again. The reset
  is now expressed the other way round — the engine names what SURVIVES a
  re-arm (history, compaction state, live-control queues, lanes, the injected
  collaborators) and rebuilds everything else — so a field added to the
  constructor resets by default instead of quietly accumulating. A test walks
  the constructor and fails when an attribute is classified as neither.

## [2.0.0a2]

Supersedes 2.0.0a1, whose files were removed from the index. That build
carried comments naming the tooling used to write them and a pointer to a
working document that ships with nothing — no functional difference, but not
what belongs in a published artifact. Nothing else changed.

## [2.0.0a1]

First public release. Withdrawn.

Protocore has existed for some time as the closed core of an agent product.
This is that core, extracted and published under the MPL — the same code, with
the parts that only made sense inside one company's repository rewritten to
describe the boundary rather than the company.

### Added

- The protocol boundary: 20 interface `Protocol`s and an `IBlobStore` ABC in
  `protocore/contracts/`, covering the model client, run and session stores,
  the tool registry, memory, workspace, search, blobs, skills, todos, hooks,
  event transport, and subagent dispatch.
- The ReAct runtime: `QueryEngine` plus `query()`, driving one agent turn at a
  time and emitting typed `TurnEvent`s. Snapshot and resume are first-class.
- A three-layer tool surface — tenant policy, a lean clipped surface, and
  progressive discovery over BM25 retrieval — with a permission gate ahead of
  dispatch and a shell-safety policy behind a real command-chain parser.
- Two-tier context compaction, session memory folding, and a token-budget model
  that keeps a long run inside its window.
- `RuntimeConstants`: 524 per-tenant tunables as a frozen Pydantic snapshot,
  every one of them documented, with new behaviour defaulting off.
- In-memory adapters (`protocore.tests_support.adapters`) that implement the
  same protocols the real ones do, so a turn runs end to end with no external
  services.
- Documentation in English and Russian under `docs/`.
- 2964 tests, a 90% coverage floor, strict typing, lint, and a security scan,
  all gated on Python 3.12, 3.13, and 3.14.

[Unreleased]: https://github.com/ascorblack-labs/protocore-community/compare/v2.0.0a4...HEAD
[2.0.0a4]: https://github.com/ascorblack-labs/protocore-community/releases/tag/v2.0.0a4
[2.0.0a3]: https://github.com/ascorblack-labs/protocore-community/releases/tag/v2.0.0a3
[2.0.0a2]: https://github.com/ascorblack-labs/protocore-community/releases/tag/v2.0.0a2
[2.0.0a1]: https://github.com/ascorblack-labs/protocore-community/releases/tag/v2.0.0a1
