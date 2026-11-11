# Agent harness for StinkyTofu

One page on purpose: a human must be able to review all of it. It orchestrates scripts and docs that
already exist (invoke, ctest, FileCheck, the project's `AGENTS.md`). It does not duplicate them.

## Scope and precedence

- **Applies to:** `shared/stinkytofu/**`. Other projects, including hipBLASLt and TensileLite, are out
  of scope. Their owners may adopt a copy later. This harness uses TensileLite only to generate the asm
  it compares.
- **Component docs win.** Build, PR, license, and test rules live in `shared/stinkytofu/AGENTS.md` and
  the skills next to it. This file only says in what order to verify a change and how to keep the
  scaffolding honest. On any conflict, the component doc wins. Fix this file.

## Shape

The goal is performance: does a change make a kernel faster, slower, or leave it the same? StinkyTofu
is compiler-like, so the same input gives the same asm. If the asm is identical, performance is
identical and no GPU run is needed. TensileLite loads the rocisa binding that embeds StinkyTofu, so
TensileLite generates the asm that is compared.

## The A/B

```text
tier 1  CPU  asm A/B, one result per config (TensileLite --cpu-only, baseline vs candidate;
             inputs: CORE.md "Tier 1 config set"; rule: CORE.md "What identical means")
               identical  -> same asm, same performance: that config is fine, skip tier 2
               different  -> pattern check, then tier 2 for that config
tier 2  GPU  perf A/B, only for configs where tier 1 found a difference.
             BEFORE it, read projects/hipblaslt/harness/PROFILING.md
             first check that the kernel under test gives correct results (see PROFILING.md),
             then compare perf
             a difference of at least one SD (run-to-run noise) needs a human check
```

Final check (local CI): finish the change using the A/B above. Then, before you submit a PR, run
the StinkyTofu tests locally (build, ctest, FileCheck; see `README.md` and `tests/README.md`). They
are fast, so run them locally and do not leave them to CI.

**Tier 1 needs a fresh binding.** TensileLite loads the rocisa binding, not the `invoke build` output.
After every StinkyTofu edit, run `invoke rocisa` in `projects/hipblaslt/tensilelite` (see its
`AGENTS.md`), in each worktree, or the comparison runs against stale output.

**The pattern check** is a filter, not a score. The patterns to check for come from the
database (`entries/`, see `DATABASE.md`), not from this repo. Each entry there has a source and a
status. Record the result as `skipped` (no database), `no entries` (database read, nothing
matched), or a hit (see `DATABASE.md`). A hit on a `confirmed` pattern rejects the candidate before it costs GPU time. A hit
on a `candidate` pattern only warns the developer. A clean result means only "no known problem". The
GPU decides the rest. Different asm is not a performance win: only tier 2 can show a gain. Identical
asm supports only "no performance change".

## Rules

1. **Check for new commits first.** At task start, run `git fetch origin develop` and list the new
   commits that touch the files in scope (`git log HEAD..FETCH_HEAD -- <path>`). If there are any,
   tell the developer before starting. A fix may already have landed. Rebasing during development is
   the developer's choice. Before merge, the base skill `rocm-pr-quality` (pre-merge gate) does the
   matching overlap check. Neither the PR bot nor this file replaces it.
   - If `git fetch` fails, tell the developer and continue. In a sparse or partial clone it can print
     `bad tree object` or `failed to run repack`. That message comes from background garbage
     collection, so check whether the fetch completed (`git rev-parse FETCH_HEAD`).
   - In such a clone, do not run `git grep` over history or the whole tree. It downloads missing
     objects one at a time and can hang. Search the working tree instead.
2. **Record the baseline.** Every comparison run records the baseline commit hash.
3. **Outward actions need the developer.** Push, open, edit, or comment on a PR, or post to a tracker,
   only when the developer says so. Showing the exact text first is already a root `AGENTS.md` rule.
   Outside contributors use a GitHub issue. AMD staff use JIRA or a GitHub issue.
4. **Respect other users of shared machines.** If a tool refuses or reports that a resource is held,
   read its message and stop. Do not retry in a loop or look for a way around it.
5. **Stopping a process.** Send SIGTERM, wait, and only then SIGKILL.
6. **Confidential details stay out of this repo.** No host names, ssh aliases, internal URLs,
   hardware-specific bug notes, or machine-specific thresholds in tracked files, logs, results, or
   notes. Where they go: `DATABASE.md`.
7. **The database is private and is loaded by default.** At session start, if `$HARNESS_DB` is set,
   read its `index.md`, then load the entries that matter for the task. Do not copy entries into
   tracked files, PRs, issues, or comments. If it is not set, say so once. If it is `none`, or the
   developer said not to use it, skip it without comment. The contract and entry format are in
   `DATABASE.md`; read it only when the database is set. At the end of a task, if you learned
   something that may be worth keeping (a measured result, an approach that failed, a setup quirk),
   tell the developer and ask whether to note it down. Write an entry only if they agree.
8. **Mind the machine's memory.** Builds, Tensile generation, and test runs (Python especially) can
   use a lot of memory, and running out can kill the whole session. Cap the parallelism to fit (for
   example `--jobs 4`, or `CMAKE_BUILD_PARALLEL_LEVEL`), run one heavy job at a time, and check the
   free memory (`free -g`) before a long run. Never build the baseline and the candidate at the same
   time.

## Components

Method for adding, retiring, and re-checking components against a new model: [`CORE.md`](CORE.md).

| Component | Assumes | Evidence | Status | Last checked (model, date, result) |
| --- | --- | --- | --- | --- |
| final check: local CI | model may break what CI would catch | CI runs these on the PR | active | |
| tier 1 asm A/B | model may spend GPU time on a change that did not alter the asm | identical asm needs no GPU run | active | |
| tier 1 pattern check | model may spend GPU time on a candidate the pattern check would reject | engineer input, experiments | active | |
| tier 2 perf A/B | model overrates its own performance wins | | active | |
| progress file | a fresh session has no memory of the last | Anthropic harness post | active | |
| planner / separate evaluator agent | not yet needed | none | not added | |

## State and database

State files and result formats are in [`CORE.md`](CORE.md). The database contract is in
[`DATABASE.md`](DATABASE.md).
A tier 2 `results.json` entry needs the fields listed in
`projects/hipblaslt/harness/PROFILING.md`.
