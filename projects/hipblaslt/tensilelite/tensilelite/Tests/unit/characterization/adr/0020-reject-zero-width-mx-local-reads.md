# Reject zero-width MX local reads before code generation

## Status

Accepted

Defect: AIHPBLAS-3868

## Context

The WMMA_V3 in-memory-swizzled MX path computes scale geometry from
`MatrixInstK // MXBlock`. Every layout requires that value to be positive.
Some M-major layouts also produced a local read narrower than one scale block,
so the computed tile count was zero. Solution derivation still marked those
candidates valid, and kernel generation later stopped with an exception.

Three coverage-only configurations used the accepted-invalid state to execute
other emitter branches before the exception. Keeping those tests would make
moving the validation to the correct boundary appear to be a coverage
regression.

## Decision

Validate the scale unit after local-read vector widths are resolved in
`Solution.assignDerivedParameters`. Reject every candidate with a non-positive
`MatrixInstK // MXBlock`; for M-major layouts, also reject a local read narrower
than one scale block. Retain matching defensive exceptions in
`LocalReadMFMA.localReadMX` in case a caller bypasses ordinary solution
derivation.

Keep one configuration-level regression test for the M-major rejection and
direct unit tests for the validation and fallback guards. The dependent PR
#11970 removes the three tests and configurations that depended on reaching
unrelated emitter code, and adds this decision to the `DECISIONS.md` registry.

## Consequences

An unsupported candidate is filtered before kernel objects are created, so it
cannot abort generation of other valid solutions. PR #11970 removes coverage
obtained only from the old failure path and updates the baseline separately.
