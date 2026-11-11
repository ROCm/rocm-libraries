# Agent harness for hipBLASLt and TensileLite

One page on purpose: a human must be able to review all of it. It orchestrates commands and docs that
already exist (the component `AGENTS.md` and `TESTING.md` files). It does not duplicate them.

## Scope and precedence

- **Applies to:** `projects/hipblaslt/**`, including `tensilelite/`.
- **Component docs win.** Build, PR, license, and test rules live in `../AGENTS.md`, `../TESTING.md`,
  `../tensilelite/AGENTS.md`, `../tensilelite/TESTING.md`, and the skills in `../skills/`. This file only
  says in what order to verify a change and how to keep the scaffolding honest. On any conflict, the
  component doc wins. Fix this file.
- **Not the kernel corpus harness.** `TESTING.md` describes a test harness for shipped kernels. This
  is a different thing: the workflow an AI agent follows around a change.

## The A/B

The goal is performance: does a change make a kernel faster, slower, or leave it the same? If the
generated asm is identical, performance is identical and no GPU run is needed.

```text
tier 1  CPU  asm A/B, one result per config (TensileLite --cpu-only, baseline vs candidate;
             inputs and rule: "Tier 1 config set" and "What identical means" in CORE.md)
               identical  -> same asm, same performance: that config is fine, skip tier 2
               different  -> pattern check, then tier 2 for that config
tier 2  GPU  perf A/B, only for configs where tier 1 found a difference.
             BEFORE it, read PROFILING.md
             a difference of at least one SD (run-to-run noise) needs a human check
```

Before the A/B (tier 0, local CI): run what CI will run for the paths you changed (the TensileLite tox
lanes, the GTest suite, hipblaslt-bench as CI runs it, and the rest). Commands come from `../TESTING.md`
and the `AGENTS.md` files. If CI would fail it, this step fails. Run in order and stop at the first
failure.

**The pattern check** is a filter, not a score. The list of kernel patterns known to be bad is
know-how, so it lives in the database (`know-how/`, see `DATABASE.md`), not in this repo. Each entry
there has a source and a status. Without the database, skip this check and record it as `skipped`
(see `DATABASE.md`). A hit on a `confirmed` pattern rejects the candidate before it costs GPU time. A
hit on a `candidate` pattern only warns the developer. A clean result means only "no
known problem". The GPU decides the rest. Never report a tier 1 result as a performance win. Pull
requests are not benchmarked in CI today (see `../TESTING.md`), so tier 2 is where a performance claim
is actually checked.

## Rules

1. **Check for new commits first.** At task start, `git fetch` and list new commits on `develop` that
   touch the files in scope. If there are any, tell the developer before starting. A fix may already
   have landed. Rebasing during development is the developer's choice. Before merge, the base skill
   `rocm-pr-quality` (pre-merge gate) does the matching overlap check. Neither the PR bot nor this
   file replaces it.
2. **Record the baseline.** Every comparison run records the baseline commit hash.
3. **Outward actions need the developer.** Push, open, edit, or comment on a PR, or post to a tracker,
   only when the developer says so. The one exception is proposing a database entry (`DATABASE.md`).
   Showing the exact text first is already a root `AGENTS.md` rule. Outside contributors use a GitHub
   issue. AMD staff use JIRA or a GitHub issue.
4. **Respect other users of shared machines.** If a tool refuses or reports that a resource is held,
   read its message and stop. Do not retry in a loop or look for a way around it.
5. **Stopping a process.** Send SIGTERM, wait, and only then SIGKILL.
6. **Confidential details stay out of this repo.** No host names, ssh aliases, internal URLs,
   hardware-specific bug notes, or machine-specific thresholds in tracked files, logs, results, or
   notes. Where they go: `DATABASE.md`.
7. **Snapshot tests (characterization goldens).**
   - A snapshot of generated asm that fails only because the asm changed may be updated once the perf
     A/B shows a gain. Update only the nodes the change touches, and show the developer the diff.
   - A snapshot of an API output guards behavior that callers depend on. If a change alters it, stop
     and ask the developer for a review. Do not update it to get CI green.
8. **The database is private and is loaded by default.** At session start, if `$HARNESS_DB` is set,
   read its `index.md`, then load the entries that matter for the task. Do not copy entries into
   tracked files, PRs, issues, or comments. If it is not set, say so once. If it is `none`, or the
   developer said not to use it, skip it without comment. The contract and entry format are in
   `DATABASE.md`; read it only when the database is set.

## Components

`CORE.md` and `DATABASE.md` are in `shared/stinkytofu/harness/`. Method for adding, retiring, and
re-checking components against a new model: [`CORE.md`](../../../shared/stinkytofu/harness/CORE.md).

| Component | Assumes | Evidence | Status | Last checked (model, date, result) |
| --- | --- | --- | --- | --- |
| tier 0 local CI | model may break what CI would catch | CI runs these on the PR | active | |
| tier 1 asm A/B | model may spend GPU time on a change that did not alter the asm | identical asm needs no GPU run | active | |
| tier 1 pattern check | model may spend GPU time on known-bad kernels | engineer input, experiments | active | |
| tier 2 perf A/B | model overrates its own performance wins | PR benchmarks do not exist in CI | active | |
| progress file | a fresh session has no memory of the last | Anthropic harness post | active | |
| planner / separate evaluator agent | not yet needed | none | not added | |

## State and database

State files and result formats are in [`CORE.md`](../../../shared/stinkytofu/harness/CORE.md). The database contract is in
[`DATABASE.md`](../../../shared/stinkytofu/harness/DATABASE.md).
A tier 2 `results.json` entry needs the fields listed in `PROFILING.md`.
