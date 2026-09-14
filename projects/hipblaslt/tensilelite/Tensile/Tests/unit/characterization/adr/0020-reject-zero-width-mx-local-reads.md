# Reject zero-width MX local reads before code generation

## Status

Accepted

## Context

The WMMA_V3 in-memory-swizzled MX path computes the number of scale tiles in a
local read from the read width and `MatrixInstK // MXBlock`. Some M-major
layouts produced a read narrower than one scale block, so the computed tile
count was zero. Solution derivation still marked those candidates valid, and
kernel generation later stopped with an exception.

Three coverage-only configurations used the accepted-invalid state to execute
other emitter branches before the exception. Keeping those tests would make
moving the validation to the correct boundary appear to be a coverage
regression.

## Decision

Validate the scale-read width after local-read vector widths are resolved in
`Solution.assignDerivedParameters`. Reject a candidate when the effective
M-major scale-read width is smaller than one scale block. Retain a defensive
exception in `LocalReadMFMA.localReadMX` in case a caller bypasses ordinary
solution derivation.

Remove the three tests and configurations that depended on reaching unrelated
emitter code before this exception. Keep one configuration-level regression
test for the new rejection and direct unit tests for the validation helper.

## Consequences

An unsupported candidate is filtered before kernel objects are created, so it
cannot abort generation of other valid solutions. Coverage obtained only from
the old failure path is no longer included in the floor and will be reflected
in the final baseline update.
