# ADR 0004: Accept `Common/Parallel.py` below the 95% coverage bar

Status:  Accepted
Defect:  none — accepted coverage ceiling, not a bug
Commit:  a0c7f06 (PR #7989) — https://github.com/ROCm/rocm-libraries/commit/a0c7f061fde1cbbcefcd6fd9c4882de2270b37df — landed on develop via PR #7989's squash merge, 74e4693

## Context
`Parallel.py` wraps several parallel-execution backends: joblib (`n_jobs=1` and
`n_jobs>1`), `multiprocessing.dummy` (threads), `multiprocessing.Pool`, and
`ProcessPoolExecutor`, plus a Windows-only `os.name == "nt"` branch. The pure
helpers, the single-threaded path, and the in-process `n_jobs=1` / thread-based
paths are straightforward to characterize. The remaining lines drive real
fork/spawn OS processes: `ProcessingPool` (`multiprocessing.Pool`),
`ParallelMapReturnAsGenerator` (`ProcessPoolExecutor`), the joblib
generator-return branch, and the Windows-only branch.

Exercising those fork/spawn paths in a unit test is flaky in practice —
pickling constraints, fork-inside-pytest interactions, CI nondeterminism, and
added runtime — and mostly tests the OS process scheduler rather than this
module's own logic.

## Decision
Characterize the pure helpers plus the single-threaded and in-process
(`n_jobs=1`, `multiprocessing.dummy`) paths of `Parallel.py` (→ ~81% line) and
accept the module below the 95% bar rather than drive real
fork/spawn/process-pool execution from the test suite.

## Consequences
`Parallel.py` is an honest <95% module, in the same spirit as the codegen
surface excluded from this characterization effort entirely (see D0 in
`DECISIONS.md`). The fork/spawn paths remain unpinned; a future change to them
carries no characterization safety net until someone accepts the flakiness
cost of exercising real multiprocessing in CI, or replaces this ADR with one
that does.

**Rejected alternatives:**
- Run real `multiprocessing.Pool(2)` / `ProcessPoolExecutor` with
  module-level picklable functions — covers the lines but is flaky and slow;
  rejected for the same reason the codegen surface is out of scope.
- Deep-monkeypatch multiprocessing internals — would assert our own mocks,
  not real behavior; rejected.

**Residual:** ~81% line coverage; the uncovered ~19% is the fork/spawn
process-pool and Windows-only branches described above.
