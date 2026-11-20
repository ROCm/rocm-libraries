# Lane: performance evidence (`PERF`)

`CONTRIBUTING.md` (*Responsibility of the Author*) makes the author report the impact to performance; the PR template has no field for
it, so it is often missing. This lane judges whether a perf-affecting change carries evidence that matches its risk. It never runs
GPU or cluster work (see *Approval*).

## Trigger

Run when the diff touches any of: solver selection or `IsApplicable` (a solver newly accepted or rejected), `GetDefaultPerformanceConfig`,
`IsValidPerformanceConfig` or `Search` spaces; find mode or immediate-mode defaults (`src/find_controls.cpp`, `src/conv/solver_finders.cpp`,
`src/hip/convolutionocl.cpp`); kernels and kernel build parameters (`src/kernels/`, `ConstructBuildOptions`, tuning params);
perf/find DBs and `*.model` heuristics (`src/kernels/*.db.txt.bz2`, `*.fdb.txt*`, `src/conv/heuristics/`); `GetWorkspaceSize` (memory
and the workspace the API returns); anything executed on every Find or immediate call (db lookup, key building, binary cache,
`src/sqlite/`). Otherwise `N/A: no perf-affecting change`.

## Method

1. Read the PR text, linked tracker item and comments for claims ("faster", "no regression", a table, a plot). Quote each claim.
2. Classify the risk from the diff: **speed** (kernel, config, solver choice), **selection** (a different solver wins on shapes the
   change did not target), **search** (tuning time, find time, first-call latency), **memory** (workspace growth), **hot path** (per-call
   overhead). A default-on change on a shipping path (DB or model entry, default find mode) is the highest risk.
3. Check evidence against risk. The evidence must state: shapes (named network or explicit driver lines, with batch size and layout),
   dtypes, direction, arch and CU count, baseline commit vs head, repetitions and the spread or noise handling, cold vs warm cache
   (first call includes kernel build and db load; `-w` in `MIOpenDriver` separates it), find-db hit vs tuned perf-config vs heuristic
   fallback, and the find mode and `MIOPEN_FIND_ENFORCE` used. A number without shapes and baseline is not evidence.
   Match the evidence to where the cost lands: kernel speed needs per-shape driver or perf-model timings; first-call or per-call
   host latency (Find, `GetSolution`, `GetWorkSpaceSize`, db or cache lookups, key building) needs the latency of that API call plus
   end-to-end startup and step time on a workload that calls the changed entry point, in the default find mode and in immediate
   mode; memory needs reported workspace sizes before and after; selection needs the chosen solver per shape for a shape set. Kernel
   timings cannot show host-side cost. The lane proposes the workload and never prescribes fixed models.
4. Compare the claim with the diff. A "perf-neutral" claim contradicts a changed default or applicability; a speedup measured for one
   dtype or arch while the change widens another is unsupported; a DB or model change evidences only the shapes it touches. Check that
   the baseline had the same DB entries (a stale or missing system DB inflates a speedup).
5. Look for loss on non-target shapes: a newly applicable solver can outrank the old winner in `Normal`, `Hybrid` or `DynamicHybrid`
   without being faster (`docs/reference/env_variables.rst` `MIOPEN_FIND_MODE`; `docs/conceptual/finddb.rst`, `perfdb.rst`).

## Findings

| Tier | When |
|---|---|
| `BLOCKING` | A default-on change to solver selection, a DB or model, a default find mode or the Find/immediate hot path with no evidence at all, or evidence the diff contradicts. Kind `process`, or `defect` if a regression is shown |
| `IMPORTANT` | Evidence exists but covers the wrong shapes, dtype, arch, direction or cache state; no noise handling; a workspace increase with no size figures; the claim is only about the change's target and the risk is wider |
| `SUGGESTION` | Evidence is adequate but unreadable or lacks the exact driver lines needed to reproduce |
| `FUTURE WORK` | A pre-existing perf gap seen on the way; the change does not touch it |

Every finding names the claim or its absence, the risky path in the diff, and the evidence that would settle it. Kind `process` for
missing or mismatched evidence; `defect` only when a measured or derived regression is in hand.

## Approval

Never run a GPU, build or cluster job here. For each item that needs measurement, write an Approval request: the command, the arch
and CU count, the shape list, the baseline and head commits, the iteration count, the expected duration and where it runs. Record the
item as `UNVERIFIED (approval pending)`; do not turn it into a finding at `IMPORTANT` or above without the missing-evidence basis above.
Approved means, depending on host: `MIOpenDriver` (`driver/`; flags `-t 1`, `-i`, `-w`, `-s`, `MIOPEN_PERFORMANCE_LOGS`, see
`driver/README.md`), the network shape lists in `test/perf_models/` driven by `test/test_perf.py` (built with `MIOPEN_TEST_PERF`), the
micro-benchmarks in `speedtests/`, and a perf-evaluation service where the host provides one. State which the author used or which you
propose; compare baseline and head with the same DB, find mode and environment.

## Handoffs

`ARCH:` per-arch speed or gate questions. `CORR:` selection logic bugs. `SIBLING:` a perf fix applied to one direction or dtype only. `TESTB:` a correctness case missing for a new fast path. `DOCS:` a user-visible default or env-var change.
