# ADR 0005: Accept `KernelHelperNaming.py` below the 95% coverage bar

Status:  Accepted
Defect:  none — accepted coverage ceiling, not a bug
Commit:  57f5e3c (PR #7989) — https://github.com/ROCm/rocm-libraries/commit/57f5e3c119431d80d263e98bdb51289d1d345aa9 — landed on develop via PR #7989's squash merge, 74e4693

## Context
`KernelHelperNaming.py` has two halves. The naming/orchestration surface
(`KernelHelperEnum`, `kernelObjectNameCallables`, the five `*Names` functions)
is pure and characterizable. The `init*` functions (`L110-240`) construct
`KernelWriter{BetaOnly,Conversion,ActivationEnumHeader,ActivationFunction,
Reduction}` instances — the GPU code-emit classes explicitly excluded from
this characterization effort's scope (see D0). They are roughly half the
module by line count and are not unit-characterizable without the full
kernel-writer machinery.

## Decision
Characterize only the naming/orchestration surface and document the `init*`
object-construction functions as out-of-scope codegen, accepting the module
at ~34% line coverage.

## Consequences
`KernelHelperNaming.py` is a partial module, in the same spirit as
`Common/Parallel.py` (ADR 0004). The `*Names` functions — which encode the
real kernel-naming contract — are pinned and protected; the `init*`
construction paths remain unpinned until the codegen surface as a whole is
brought into scope.

**Rejected alternatives:**
- Construct the `KernelWriter*` objects to cover `init*` — pulls the
  out-of-scope codegen surface into the unit tests; rejected.
- Drop the module entirely — rejected: the `*Names` functions encode a real,
  worth-pinning contract even though the module can't be fully covered.

**Residual:** ~34% line coverage; the uncovered ~66% is the `init*`
`KernelWriter*`-construction functions (`L110-240`).
