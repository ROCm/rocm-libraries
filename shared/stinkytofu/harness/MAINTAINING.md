# Maintaining the harness

Read this only when you are asked to change the harness. Agents verifying a change never need it.

## Files and when each is read

| File | Read when | Budget |
| --- | --- | --- |
| `HARNESS.md` (router, one per project) | every task | 4.5 KB |
| `ab.md` | verifying a change that produces asm | 5 KB |
| `compare.md` | running tier 1 | 5 KB |
| `state.md` | `progress.md` exists, or writing results | 5 KB |
| `clone.md` | `git fetch` fails, or the clone is sparse or partial | 5 KB |
| `DATABASE.md` | `$HARNESS_DB` is set, or asked to remember | 7 KB |
| `ENTRY-FORMAT.md` | writing or changing a database entry | 5 KB |
| `PROFILING.md` (hipBLASLt folder) | before a perf A/B | 5 KB |
| `SETUP.md` | setting up a new machine or clone | 5 KB |

A database index line stays under 200 characters, and an entry stays under 3 KB with its review card
first. Past about 100 entries, split the index by scope: a short top-level index that lists the scopes,
and one index file per scope.

## Where a new rule goes

- A rule that applies to **every task** and is about safety may go in the router (`HARNESS.md`). Keep it
  short.
- A rule that applies to one topic goes in that topic's file, and the router's "Read when" table says
  when to read it.
- If a file goes over its budget, split it by when it is needed. Do not raise the budget.
- Run `python3 tools/check-budget.py` after a change. It prints each file's size, an estimate of the
  tokens on the common paths, and fails when a budget is exceeded.

## Components

Each row is one piece of scaffolding. `assumes` is the limitation of the model it covers. A row with no
`evidence` is a candidate for removal.

**Adding a component:** add a row with `assumes` and `evidence`, set it to `on-trial`, run the same
real task with and without it, and keep it only if the outcome is better.

**New model generation:** run a real task on the full harness, then disable one component at a time
and rerun the same task. Keep a component only if removing it makes the outcome worse. Record the model
and the result in the last column of the row. Also try adding capabilities the new model may now
handle. Do not remove several components at once. In the same pass, check stale database entries (old
`last_verified`) and move the ones nobody needs to an unindexed `archive/` folder.

**StinkyTofu** (`ab.md`, `compare.md`):

| Component | Assumes | Evidence | Status | Last checked (model, date, result) |
| --- | --- | --- | --- | --- |
| final check: local CI | model may break what CI would catch | CI runs these on the PR | active | |
| tier 1 asm A/B | model may spend GPU time on a change that did not alter the asm | identical asm needs no GPU run | active | |
| tier 1 pattern check | model may spend GPU time on a candidate the pattern check would reject | engineer input, experiments | active | |
| tier 2 perf A/B | model overrates its own performance wins | | active | |
| progress file | a fresh session has no memory of the last | Anthropic harness post | active | |
| planner / separate evaluator agent | not yet needed | none | not added | |

**hipBLASLt and TensileLite** (`projects/hipblaslt/harness/ab.md`):

| Component | Assumes | Evidence | Status | Last checked (model, date, result) |
| --- | --- | --- | --- | --- |
| final check: local CI | model may break what CI would catch | CI runs these on the PR | active | |
| tier 1 asm A/B | model may spend GPU time on a change that did not alter the asm | identical asm needs no GPU run | active | |
| tier 1 pattern check | model may spend GPU time on a candidate the pattern check would reject | engineer input, experiments | active | |
| tier 2 perf A/B | model overrates its own performance wins | PR benchmarks do not exist in CI | active | |
| progress file | a fresh session has no memory of the last | Anthropic harness post | active | |
| planner / separate evaluator agent | not yet needed | none | not added | |
