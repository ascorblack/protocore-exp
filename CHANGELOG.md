# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **A failed round never completes on an earlier run's answer.** Completing a
  failed run "on its preserved answer" read every message not tagged as
  seeded, so on a host that hands the engine a session's earlier turns
  untagged, the previous run's reply counted: a run whose every request was
  refused completed with nothing written and no error reported. The answer
  must now have been written in the round that failed, the span after the
  last message a caller put in.
- **A refusal the adapter classified as final is neither retried nor wound
  down.** The retry decision read only the `reason` of an attached
  classification and ignored its `retryable` flag, although `ClassifiedLike`
  carries both; the flag now decides when it is set. And a permanent
  `LLMProviderError` no longer enters the wind-down: its one turn is a request
  to the endpoint that has just refused the run, and it was refused the same
  way. The fallback chain is still tried first.
- **The wind-down notice names a deadline and a stalled model for what they
  are.** A run stopped by its wall clock was told it had reached its budget,
  and a run whose model kept returning reasoning with no answer and no tool
  call was told the model endpoint had failed — which the model then passed on
  to the operator as the reason. `soft_stop_notice_text_deadline` names the
  time limit, and the new `model_no_progress` cause, entered by the
  empty-round policy, carries `soft_stop_notice_text_model_no_progress`. Blank
  texts fall back to the general notice.
- **The wind-down notice does not outlive its run.** The "tools are withdrawn,
  write your answer" message the loop writes into history when a run winds
  down stayed there after the run ended. When the wind-down's own final turn
  failed — a provider outage that caused the wind-down also killed the answer —
  the notice was the last thing in the session, and the next turn obeyed it:
  the model reported what it had not finished and called nothing, although
  every tool was back. The notice is removed once the drive reaches a terminal
  state and written back only for a resumed run that is still wound down.

## [2.0.0a21] - 2026-09-25

### Added

- **`TERMINAL_REFUSAL_NEEDS_WORK_METADATA_KEY`** (`protocore.contracts.types`).
  A terminal tool, or the host check behind it, sets this on an error result
  when the call was refused because work is missing, for example a declared
  file that does not exist. See the forcing below for what it changes.
- **`LLMRequest.extra["tool_choice_required"]`**: `True` asks for some tool
  call and no prose. The `LLMRequest.extra` contract now documents it next to
  `forced_tool_choice`.

### Changed

- **A delivered answer's terminal call is forced, not requested.** This
  applies to a run with `expected_terminal_tool` and
  `terminal_tool_nudge_enabled` that ends a turn with a substantive visible
  answer and no terminal-tool result.
  - Every request after that names the terminal tool in
    `extra["forced_tool_choice"]` and appends nothing to the transcript. The
    free-form `terminal_tool_nudge` text is no longer sent on this path.
  - A terminal-tool error is answered by forcing the tool again by name. The
    one exception is the run's first refusal marked with
    `TERMINAL_REFUSAL_NEEDS_WORK_METADATA_KEY`: the request after it carries
    `extra["tool_choice_required"] = True` instead, so the model can do the
    missing work.
  - A forced turn's text is kept off the live stream and out of history. Calls
    that the forced mode does not admit are withheld from the live stream and
    dropped before they are recorded or run, so a provider that ignores the
    forced choice cannot start new work under the answer.
  - A reasoning-only or finish-less forced round is retried instead of
    receiving `continue_prompt_text`, the reasoning length-cut note or the
    output-cap resume prompt. Before this, a thinking model answered the nudge
    with reasoning only, was told "Please continue.", and started a second
    answer, often to an earlier question in the session.

  **When the forcing steps aside.** The model's text is shown again when it has
  to act rather than seal:
  - a gate refuses the call with a corrective (the prose gate, a host
    pre-dispatch check);
  - a background result, or a user message placed at a tool boundary, arrives;
  - the marked refusal's required call is answered with other work.

  The answer the model ends on is then forced again. A gate's corrective closes
  the answer window, so the refused prose no longer counts as the answer.

  **Budget.** Forced requests and step-asides share
  `terminal_tool_forced_max_attempts` (default 3). The run completes on the
  answer it delivered when the budget is spent (`state_changed` reason
  `terminal_tool_forced_exhausted`), or at once when the terminal tool is not
  registered or the consecutive-error circuit breaker has stopped it
  (`terminal_tool_unavailable`). That completion is a hard stop: it does not
  pass the answer floor, the voluntary artifact seal or follow-up placement.

  **Thinking.** Forced requests go out with thinking off unless
  `terminal_tool_forced_thinking_enabled` is set. Measured on a self-hosted
  thinking model behind vLLM:
  - a named `tool_choice` with thinking on spent a 2,048-token budget reasoning
    in most attempts and returned no call;
  - with thinking off it returned exactly the call every time;
  - `tool_choice="required"` after a refused call returned a tool call in 10 of
    10 samples, the requested write in 8 of them.

  **Other effects.**
  - The terminal tool is pinned to the surface while its call is owed.
  - A continuation run (`run(None)`, including `resume`) keeps a forcing that
    was in progress, with its terminal-only latch.
  - A turn that ends with no answer yet still gets the one-time
    `terminal_tool_nudge` text; the answer it then writes is forced.
  - Prose that only announces work ("now let me write the report") counts as an
    answer and is sealed. A host check on the declared deliverables can refuse
    it with the needs-work marker. A host whose models tend to announce and
    stop can instead set the new `terminal_tool_nudge_write_first_before_forcing`,
    which sends the one-time write-first nudge before sealing while the run has
    written no file.

  **Policy constructors.** A host that assembles its own turn-policy set
  must pass the new arguments. `TerminalNudgePolicy` now requires
  `forced=ForcedTerminalCall(...)`, and `EmptyModelTurnPolicy` requires
  `forcing_terminal_call=`. A host that uses the core's policy set is
  unaffected.

  **Host adapters** must render `forced_tool_choice` as the native single-tool
  choice and `tool_choice_required` as `tool_choice="required"`. When a request
  carries `enable_thinking=False`, they must send thinking off explicitly.

### Fixed

- **A failed terminal call no longer hides the answer before it.** A
  terminal-tool result is recognised through the call it answers, not only by a
  name or terminal flag on the result, so an error result from the terminal
  tool is no longer counted as work. Before, one failed call made the answer
  that preceded it read as progress narration, and the prose gate then refused
  the retried call and asked the model to write its answer again.

## [2.0.0a20]

### Changed

- **`LLMRequest.temperature` is unset unless the caller states one.** The
  contract defaulted it to `0.7`, and `build_llm_request` filled that default
  into every request, so the action stream and the deep loop's plan calls
  always carried a temperature nobody had chosen, and an adapter could not tell
  "no opinion" from "0.7". The field is now `float | None = None`; the
  builder passes `None` through, and `RequestManifest.temperature` records it
  as such. Paths that need a value — the compaction summariser, with
  `compaction_summary_temperature` — still state it. A host adapter should omit
  the wire field when it is `None` (or apply its own per-model setting); an
  adapter that serialises it as-is now sends `null`. The temperature is part of
  `request_digest`, so a request recorded before this change does not
  replay-match the same request rebuilt after it.

- **The wind-down notice asks for the answer, not the work log.** The default
  `soft_stop_notice_text` told the model to report what it did and what it
  found, which invited a narration of the run's steps into the user-facing
  reply — something many hosts' personas forbid. It now asks for the best answer
  from what the run already has and where any results are, says plainly what
  could not be established or finished, and tells the model not to describe its
  steps. Both languages changed; the text is no longer than before. A host that
  seeded the old default into its own constants store keeps the old wording
  until that row is updated.

### Fixed

- **The compaction retry budget charges failed passes, not passes.** Only a
  pass that tried and failed is charged, and only once however many tiers
  failed; a tier that raised is no longer counted again by the no-progress
  rule. The reactive pass after a provider rejection keeps its own count,
  `CompactionState.reactive_retry_count` (same `compaction_failed_max_retries`
  bound, rides the run snapshot), so proactive failures can no longer use up
  the one profile that may compact seeded history. Before, an idle proactive
  pass plus two reactive passes that lost their summariser calls exhausted the
  default budget. Progress by either profile clears both counts, and so does
  `rearm()`. A routine pass that rewrote a Tier 1 block now counts as progress
  even when the estimate did not move.
- **A proactive pass with nothing to do is not opened.** The routine,
  turn-start and per-iteration gates ask `ContextManager.has_proactive_work`
  first. When no tier would change anything under the proactive profile — a
  history of turns seeded from earlier runs is the usual case — there is no
  `COMPACTING` state, no compaction events, hooks, usage row or snapshot. The
  engine then skips the gate without asking again until the history or the
  constants change.
  Before, such a pass ran on every iteration once the estimate crossed the
  gate. `Tier2Result.units_attempted` and `Tier3Result.spans_attempted` report
  the calls a pass made. `tier1_has_work`, `tier2_has_work` and
  `tier3_has_work` answer the same question for each tier. A fold that raises
  now reports a zero `Tier3Result` instead of `None`.
- **Proactive compaction running out of budget no longer fails the run.**
  Nothing has been rejected at that point. The proactive summariser tiers are
  suspended (a `compaction_exhausted_proactive_suspended` state change each
  time, no `ERROR` event) and the request goes out; Tier 1, which needs no LLM, keeps
  running. The suspension ends after `compaction_proactive_suspension_iterations`
  gate visits (default 6) or once the prompt has grown by
  `compaction_proactive_suspension_growth_ratio` (default 0.1) of its size when
  it began. A context refusal — the provider's, or the local fit's — lifts it
  at once and runs the reactive pass, as do `rearm()` and a snapshot resume.
  `ContextManager.run_compaction` and `force_compaction` take `llm_tiers` to run
  Tier 1 alone. `PerIterationCompactionPolicy` no longer takes `pair_orphans`
  or `message_stop`, since it no longer ends the turn.
- **`compaction_no_gain_backoff_iterations` takes effect.** The per-message
  recovery reset zeroed the backoff before the per-iteration gate could read
  it, so a low-gain routine pass ran on every iteration. The count now
  survives the reset. It ends early once the prompt has grown by
  `compaction_no_gain_backoff_growth_ratio` (default 0.1) of its size when it
  was set, so a new large tool result is not left alone, and on any context
  refusal. `PerIterationCompactionPolicy` takes a `prompt_tokens` callable for
  that measurement, and `ITurnState` gains `compaction_backoff_prompt_tokens`.
- **`RequestTokenCounterConformance` and `LifecycleRegistryConformance` are
  importable from `protocore.conformance`.** Both were in `SUITES` but only
  reachable through `protocore.conformance.suites`. The package's own tests now
  fail if a suite in `SUITES` is not re-exported by the package, or if a suite
  class is defined in `suites` without being listed in `SUITES`.

## [2.0.0a19]

### Added

- **A provider can size the rendered request, and is asked near the edge.**
  The optional `IRequestTokenCounter` capability
  (`async count_request_tokens(request) -> int | None`) reports the prompt
  tokens the server renders a request to. When the estimate is within
  `exact_token_count_margin_ratio` (default 0.75) of the point where the output
  cap starts being clipped, or of the compaction trigger, the loop asks for it
  and fits the request — or runs the gate — on that number. The default covers
  every undercount calibration can express (`1 - 1/4`); measured against a vLLM
  tokenizer the heuristic ran 1.76x short on JSON and 3.53x on hexadecimal
  text. Counts are kept per request content
  (`exact_token_count_cache_max_entries`, default 32), bounded by
  `exact_token_count_timeout_seconds` (default 5), and can be switched off with
  `exact_token_count_enabled`. Once the fit has counted a turn's request, the
  compaction gate reuses the factor that count set instead of counting the
  history a second time. After a count, the next one is made only when the
  content that count did not see — messages whose digest is not among the
  counted ones, so compaction or eviction cannot hide dense content arriving
  with it — sized at the worst undercount the margin assumes, could carry the
  prompt over the limit; a count that fails or times out stops
  counting for `exact_token_count_failure_backoff_seconds` (default 300). A provider without the capability sends exactly the
  requests it sent before; a count that fails or times out is logged and the
  estimate is used.

### Fixed

- **The token estimate learns from a count and from a rejection.** An exact
  count sets `token_estimate_calibration` to the measured ratio at once, and
  before the fit is attempted, so a count that proves the request cannot fit
  raises the factor and the refusal goes to compaction sized in the counted
  tokens. A rejection for length, which never carries usage, now raises the
  factor to the floor it proves — the prompt was at least the window less the
  output cap that was sent. On hexadecimal text the estimate had run at 0.28 of
  the server's count. The factor still starts each new run at the configured
  `token_estimate_calibration`; a host whose content is dense sets that per
  scope.

## [2.0.0a18]

### Fixed

- **A proactive compaction pass no longer forfeits the reactive one.** With the
  lower effective trigger, a history made of turns seeded from earlier runs
  crosses the proactive window at turn start; the routine profile leaves seeds
  alone and frees nothing, but the pass was carried into the message as "the
  compaction attempt", so the provider's rejection was answered by the
  output-cap ladder alone and the run died on a prompt that fit the window
  once seeds were summarised. The rejection handler now tracks the reactive
  pass separately and runs it once, after which the ladder applies.
- **An exhausted compaction budget hands the turn back to the output-cap
  ladder** when a strictly smaller cap is still available, instead of ending
  the run on the budget.

## [2.0.0a17]

### Added

- **Stale tool results can be cut down in the request view.** With
  `tool_result_stale_trim_enabled` (off by default), a tool result the run has
  moved past is sent as its first `tool_result_stale_max_chars` characters plus
  a line naming how much was kept and how much was removed, while
  `engine.history` keeps the whole value. The newest `tool_result_fresh_count`
  results and every result of the latest round of tool calls are never cut, and
  nothing is cut until the trimmable excess crosses
  `tool_result_stale_trim_batch_chars`, so the prompt prefix moves for a batch
  rather than for one result. A trimmed result stays trimmed for the run and the
  decision travels in the snapshot as `trimmed_tool_result_ids`. Lines starting
  with one of `tool_result_stale_trim_protected_prefixes` are carried over
  verbatim, so a result whose text names how to cite it keeps that line when it
  loses its body. Pinned results are spared unless a later write falsified the
  pin, and compacted placeholders are never rewritten. It does not delay
  compaction: the compaction gate measures the whole durable transcript, not
  the trimmed request view, so what it buys is a smaller and cheaper prompt at
  the provider, not fewer compactions.

- **A provider failure with nothing behind it no longer ends as an invented
  answer.** A run whose requests the endpoint refused before it wrote a word,
  called a tool or read a result is no longer wound down and asked to deliver
  a final answer — it goes terminal FAILED carrying the provider's own error.
  Once a tool result exists the partial outcome is real, and the wind-down
  still runs; its notice now names a provider failure instead of a budget
  (`soft_stop_notice_text_provider_error`), and its `state_changed` events
  carry the upstream's message as `soft_stop_detail`.

### Changed

- **A summariser failure costs its own unit, not the whole pass.** Tier 2 used
  to discard everything a pass had produced as soon as one call failed, so one
  unit the summariser cannot handle kept a run from shedding a single token
  and the next pass bought the same failure again. Failures are now counted
  per unit in `CompactionState.failed_anchor_keys`; past
  `compaction_summary_failed_unit_max_attempts` (2) the unit is left to the
  fold tier, and the other units in the batch commit. The census is carried in
  the run snapshot. A reply carrying no readable summary now counts as a
  failed call — that is the shape a cut-off reply takes — while a summary that
  is merely no smaller than the original does not — nor does a transport
  failure (a rate limit, a 5xx, a reset socket): only a unit that does not fit
  the summariser's window or whose reply the output cap cut is counted, since
  only those repeat. The forced passes ignore the census entirely and try
  every unit, and entries whose unit has left the history are pruned.
- **The per-turn summariser prompt states its budget in characters too.** It
  now also says that a longer reply is cut off and discarded, asks for a
  count and the records that matter instead of a copy of a long tool result,
  and names the single key it wants. A model that listed every record of a
  long result wrote a reply the output cap cut, and a cut reply is never
  parsed. `compaction_summary_chars_per_word` (6) converts the word budget,
  `compaction_summary_envelope_tokens` (32) reserves room for the JSON around
  the words before the budget is derived, and the budget is also clamped to
  what `compaction_summary_string_max_chars` will accept at decode time.
  The template gains a `max_chars` variable; a per-tenant override that does
  not use it is unaffected.

- **Transient provider failures are retried on the verdict the adapter gives,
  not on the exception type alone.** `LLMError` now carries `retryable`, which
  an adapter sets per raise — `LLMProviderError("no such model",
  retryable=False)` is terminal at once, while a 5xx, a reset connection and a
  silent stream take the bounded ladder that 429s and timeouts already took
  (`llm_transient_error_retry_max_attempts`, backoff doubling from
  `llm_transient_error_retry_backoff_base_seconds`). A context-window overflow
  is never retried this way and takes no such keyword. A cancelled run, or one
  whose wall-clock budget leaves room only to finalise, starts no further
  attempt, and a cancel during a backoff ends the pause immediately. Every
  attempt logs a WARNING naming the run and the attempt number.

### Fixed

- **The compaction trigger now sits below the prompt size the provider will
  still accept.** `derive_budgets` takes the lower of
  `model_context_window * compaction_trigger_ratio` and the window less the
  output reserve, less `request_context_safety_tokens`, less the new
  `compaction_trigger_turn_headroom_ratio` (0.15) for the turn about to be
  added. A server that reserves the output budget inside the context window
  rejects any prompt above `window - max output`; with the stock 0.8 trigger
  and 0.25 output reserve the trigger on a 65 536-token window stood above
  that cliff, so proactive compaction could not fire before the rejection.
  The emergency cliff is held strictly above the effective trigger. The
  output-reserve part of the deduction is gated on
  `provider_reserves_output_in_context_window` (default true): a provider that
  sizes its input window independently of the requested output gives that
  share of the window back by setting it false.
- **A summary's word budget is sized for the script it is written in.** The
  ceiling was half the summariser's output cap, two tokens a word being the
  English figure; JSON escaping and a non-Latin script cost three or four, so
  the summary of every large unit outgrew the cap, came back cut off, never
  parsed and never committed — and the next pass paid for the same units
  again. The rate is now `compaction_summary_output_tokens_per_word` (4), and
  it caps the fold target the same way.

## [2.0.0a16]

### Fixed

- **Reactive compaction can now shrink history seeded from earlier runs.**
  When a provider rejects a request as exceeding its context window, recovery
  keeps only `compaction_force_keep_recent_turns` trailing messages (one by
  default) and may summarise or fold prior-run seed turns, copying the seed tag
  onto every replacement so the host still excludes them from persistence.
  Previously a history made of protected seeds left reactive compaction with
  nothing to compact, and the run failed after the output-cap ladder ran out.
  Proactive emergency compaction keeps the routine window and seed protection.
- Fold summaries label prior-session turns separately from the operator's own
  instructions.

## [2.0.0a15]

### Fixed

- **Bounded context-window recovery when providers report only a prompt-size
  lower bound.** Repeated retries strictly reduce output headroom under
  `context_overflow_retry_max_attempts`, and proactive compaction is not
  repeated reactively for the same assistant message.

## [2.0.0a14]

### Fixed

- A measured context overflow is retried with a smaller output cap before
  history is rewritten; a repeated overflow gets exactly one compaction attempt.

## [2.0.0a13]

### Fixed

- **Context retries now use provider-reported prompt sizes when available.**
  OpenAI-compatible adapters can attach the provider's measured context window
  and input-token count to an overflow. The retry then reserves its configured
  safety margin against that measured size instead of relying only on the local
  estimator, preventing a second request from missing the hard limit by a small
  provider-tokenization difference.

## [2.0.0a12]

### Fixed

- **A context-overflow retry now reduces its output allowance.** Reactive
  compaction still rebuilds the prompt once, and the rebuilt request uses at
  most half the fitted output allowance the provider rejected by default. A
  smaller rebuilt prompt therefore cannot raise the retry back toward the
  normal cap and hit the same context limit twice.

## [2.0.0a11]

### Fixed

- **The compaction summary string limit remains at 1,024 characters.** This
  corrects an unrelated default change that slipped into the preceding
  provider-framing release while retaining its larger request safety margin.
- **Published installation examples now name the current pre-release**, both
  for the runtime package and the optional conformance test extra.

## [2.0.0a10]

### Fixed

- **Near-limit requests reserve more space for provider-rendered framing.**
  The default request margin is 1,024 tokens. Regression coverage now models
  the boundary where an unchanged output cap previously let a provider-side
  prompt count exceed the context window even though the local estimate fit.

## [2.0.0a9]

### Fixed

- **Near-limit requests now keep enough headroom for additive provider
  framing.** The default request margin is 512 tokens, covering chat-template
  and tokenizer overhead that a content-based local estimate cannot see while
  using less than one percent of a 65,536-token context window. Installations
  can still tune the margin, and boundary-focused synthetic tests can disable
  it explicitly.

## [2.0.0a8]

### Fixed

- **The default provider-framing margin now covers real chat-template
  variance.** Live OpenAI-compatible providers may add several tokens that are
  absent from a local message estimate. The default request margin is now 64
  tokens; installations can still tune it, and tiny synthetic test windows can
  explicitly disable it when they are testing an unrelated boundary.

## [2.0.0a7]

### Fixed

- **Provider framing can no longer push an exactly fitted request one token
  beyond the context window.** Complete requests now leave a configurable
  `request_context_safety_tokens` margin after prompt estimation and before
  choosing `max_tokens`. The default one-token margin covers providers that
  count a framing token absent from the local estimate, while the runtime
  constants contract lets an installation reserve more for another tokenizer.
  Invalid margins that consume the whole context window are rejected by both
  snapshot validation and the constants registry.

## [2.0.0a6]

This release makes long-running conversations safer at the two points where a
model can otherwise lose the task: history compaction and the final request
budget sent to a provider.

### Added

- **Bounded recovery for a response that spends its output on reasoning.** A
  length-limited response containing reasoning but no answer or tool call is
  discarded and retried through a small recovery ladder: lower the reasoning
  effort, then disable thinking when the run mode permits it. The ladder and
  its original controls survive snapshot pickup, do not overwrite live
  operator controls, and end with one honest best-effort wind-down request.
- **A hard context-window fit for every provider request.** The complete
  normalized messages and tool schemas are estimated together after the
  ordinary output cap, adaptive safety band and terminal reserve have been
  chosen. `max_tokens` is clipped before the request is manifested or sent;
  when no positive output fits, the action path gets one bounded compaction
  retry and secondary planning or summarisation calls fail locally without
  sending a request already known to be too large.

### Fixed

- **Runtime recovery messages no longer become operator intent during
  compaction.** User-role control messages are identified by provenance rather
  than role alone. Aged recovery nudges are removed on the same atomic working
  copy as the rest of Tier 2, so a failed summariser or recorder cannot leave
  history and compaction accounting disagreeing.
- **Token-estimate calibration follows the model that produced it.** Learned
  calibration survives snapshot pickup for the same model, respects a newer
  configured baseline, and resets on live or provider-chain model changes.
  Late usage from an earlier model is still accounted for but cannot replace
  the active model's calibration. A provider count from one wire request is no
  longer reused as a size floor for a different request or for a history-only
  compaction decision.

## [2.0.0a5]

The repository moved to `https://github.com/anchor-inference/protocore` and is
now the single home of the core; the package on PyPI is still `protocore` and
the import path is unchanged. Everything else in this release is the loop
spending less per round: a session store hears what a round appended instead of
being handed the conversation again, the tool surface travels by digest, and
compaction gains a third pass over what the first two cannot shrink.

### Added

- **A session store can be told what a round added.** The loop used to hand the
  store the whole working history once per round to record the one or two
  messages the round appended, so the cost of writing a turn down grew with the
  length of the conversation rather than with what the conversation just did.
  The engine now remembers the prefix the store already holds and compares the
  current history against it by object identity — `Message` is frozen, so an
  append leaves every earlier object where it was and a compaction, checkpoint
  or eviction builds new ones. `persist_session_history(engine)` is unchanged
  and is still the only method a store must have; a store that can write
  incrementally also attaches `persist_history_delta(engine, delta)` and is
  handed a `HistoryDelta` (`protocore.runtime.history_persist`) naming what was
  appended, or the whole history when the sequence was rewritten.
  `HistoryPersister` carries a default for the second that calls the first.
  Returning is the store's promise that the write landed: the marker advances
  only after the call returns, so a store that raises is offered the same
  messages again, and a store that defers a write it then drops calls
  `QueryEngine.forget_persisted_history()` — which is honoured even from inside
  the write. A hand-over with nothing to say is not made at all, and
  `QueryEngine.note_session_state_changed()` raises the notice for a session
  that changed in a way its messages do not show, such as a checkpoint.
- **The advertised tool surface is named by a digest.**
  `tool_surface_advertised` carries `tool_surface_digest` and
  `tool_surface_described`, and `protocore.runtime.tool_surface` answers
  `surface_descriptions(digest)` for a reader that has nothing kept against a
  digest it met.
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

- **`tool_surface_advertised` no longer repeats every tool's description on
  every run.** This is a wire change. The descriptions are decided by the
  registry and are the same on every run of a deployment, and they were nearly
  the whole event; they now travel with the first advertisement of a digest to
  reach each reader — the session, which is the unit a host fans events out
  over — and `tool_surface_described` says which kind of advertisement this is.
  A reader keeps the descriptions against `tool_surface_digest` and treats a
  missing `description` as "look it up", not "there is none";
  `protocore.runtime.tool_surface.surface_descriptions(digest)` answers a
  reader that has none. What is run-specific — `name`, `sources`, `roles`, and
  which tools are on the surface at all — is on every advertisement. The claim
  that a reader has been described to is recorded only once the event has been
  handed to the stream, so a run cancelled at that point does not spend its
  reader's one description on an event nobody received. The request manifest
  still records the tool definitions in full, so what was sent to the provider
  remains recoverable from the durable record.
- **Tool definitions are costed for tokens once per surface, not once per
  call.** The estimate is cached by digest and by the chars-per-token ratios it
  was computed under. The digest itself is recomputed every call on purpose:
  `ToolDefinition` is frozen but its parameter schema holds a plain `dict`, so
  a surface remembered against object identity would be handed back a digest
  that had stopped describing a schema edited in place.
- **The token estimate cache is no longer split by the calibration factor.**
  `token_estimate_calibration` is a single multiplier over the whole
  per-message partition; it was folded in before the number was cached and then
  keyed on, so the loop's calibrated reading and the calibrator's uncalibrated
  one evicted each other's entries and an alternating pair both walked every
  character of every message. The partition is now cached as the heuristic
  computes it and the factor is applied where the number is handed out, which
  is the same arithmetic for every caller.
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

- **`protocore.__version__` reports the installed version.** It had been left
  at the string the first pre-release was cut with, so a host reading it back
  was told `2.0.0-alpha.1` whatever it had installed.

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

[Unreleased]: https://github.com/anchor-inference/protocore/compare/v2.0.0a6...HEAD
[2.0.0a6]: https://github.com/anchor-inference/protocore/releases/tag/v2.0.0a6
[2.0.0a5]: https://github.com/anchor-inference/protocore/releases/tag/v2.0.0a5
[2.0.0a4]: https://github.com/anchor-inference/protocore/releases/tag/v2.0.0a4
[2.0.0a3]: https://github.com/anchor-inference/protocore/releases/tag/v2.0.0a3
[2.0.0a2]: https://github.com/anchor-inference/protocore/releases/tag/v2.0.0a2
[2.0.0a1]: https://github.com/anchor-inference/protocore/releases/tag/v2.0.0a1
