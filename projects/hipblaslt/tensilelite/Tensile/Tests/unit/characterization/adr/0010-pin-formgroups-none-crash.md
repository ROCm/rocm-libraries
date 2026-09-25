# ADR 0010: Pin the `formGroups("None")` crash on the skipMI / MI-disabled path

Status:  Accepted
Defect:  [AIHPBLAS-4409](https://amd-hub.atlassian.net/browse/AIHPBLAS-4409)
Commit:  ce8c60e (PR #7989) — https://github.com/ROCm/rocm-libraries/commit/ce8c60e66640b60f7fde9122907b593ed6620c7d — landed on develop via PR #7989's squash merge, 74e4693

## Context
`formForkParams(sol, skipMI=True)` — or any solution with
`EnableMatrixInstruction` falsy — sets `temp = "None"` (a *string*, not
Python `None`) and then calls `forkData.append(formGroups(temp))`.
`formGroups` calls `temp.items()`, which raises `AttributeError` on a `str`.
So the entire skipMI / MI-disabled code path is currently broken, and
`TensileLibLogicToYaml(..., skipMI=True)` crashes.

## Decision
Pin the crash (assert `AttributeError`) instead of asserting a
`"None"`-sentinel `Group`, and drive the orchestrator / fork tests through
the MI-enabled (`skipMI=False` + `EnableMatrixInstruction=True`) path, which
works. Characterization records present behavior; add-only forbids fixing
`formGroups`/`formForkParams` here.

## Consequences
This is a real, user-facing bug: the `--skipMI` CLI flag is unusable. The
golden encodes the crash on purpose so the suite stays an honest record of
current behavior. Filed as
[`AIHPBLAS-4409`](https://amd-hub.atlassian.net/browse/AIHPBLAS-4409). The
fix belongs in a separate change (make `formForkParams` build a real
MI-disabled `Group` instead of the string sentinel `"None"`) with its own
regression coverage; when that lands, flip this golden and supersede this
ADR.

**Rejected alternatives:**
- Assert a `"None"` group is produced — would fail (the crash happens
  before that) and misrepresent behavior.
- Skip the path — loses documentation of a real bug on a public CLI flag.

**Residual:** 199 stmts, 4 missed → 98% line. Misses are two yaml-representer
callbacks (`representNone`/`flowSeq`, registered but not invoked by these
tests) and two orchestrator `RuntimeError` guards (empty `solutionIndex` /
missing solution).
