# ADR 0008: Accept `BenchmarkStructs.py` below the coverage bar (`BenchmarkProcess` deferred)

Status:  Accepted
Defect:  none — accepted coverage ceiling, not a bug
Commit:  5dcbbb4 (PR #7989) — https://github.com/ROCm/rocm-libraries/commit/5dcbbb462d5c07b27a7fe303fb5d26ab8d61f55d — landed on develop via PR #7989's squash merge, 74e4693

## Context
`BenchmarkStructs.py`'s pure helpers (`getDefaultsForMissingParameters`,
`separateParameters`, `checkCDBufferAndStrides`), the fork-permutation
cartesian product (`constructForkPermutations` /
`constructLazyForkPermutations`), and `BenchmarkStep` are all
straightforwardly characterizable with hand-built inputs. `BenchmarkProcess`
(`L83-235`) is different: `__init__` / `getConfigParameters` /
`convertParametersToSteps` consume a *complete* benchmark config
(`problemType` + `problemSizeGroup` with `BenchmarkCommonParameters` /
`ForkParameters` / `ProblemSizes` / ...) and build `ProblemType` /
`ProblemSizes` / steps from it — this is an integration path, and hand-built
dicts approximating a full benchmark config would be large and brittle.

## Decision
Cover the pure helpers, fork-permutation cartesian product, and
`BenchmarkStep`. Document `BenchmarkProcess`'s config-to-benchmark-steps
integration builder as needing a full benchmark-config fixture, and defer it.

## Consequences
`BenchmarkStructs.py` is a partial module. `BenchmarkProcess` remains
unpinned until an end-to-end benchmark-config fixture is available to drive
it realistically.

**Rejected alternatives:**
- Hand-author a full benchmark config to drive `BenchmarkProcess` directly —
  rejected as large and brittle, out of proportion to this module's budget.

**Residual:** the integration builder (`BenchmarkProcess`, `L83-235`) is
uncovered; an integration fixture would finish it.
