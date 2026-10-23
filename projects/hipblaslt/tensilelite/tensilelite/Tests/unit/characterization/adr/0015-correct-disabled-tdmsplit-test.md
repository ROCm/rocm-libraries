# Correct the disabled TDMSplit characterization

## Status

Accepted

## Context

The `s01_calculateloopnumiter_halfplr_tdm` test expected a configuration with
`TDMSplit: true` to generate assembly. `Solution.assignDerivedParameters`
rejects every such solution with `TDMSplit is currently disabled`, before
`KernelWriterAssembly.calculateLoopNumIter` can run. The test therefore claimed
coverage of emitter lines that it could not reach and failed with an empty
kernel list.

## Decision

Replace the emission and saved-result assertions with direct assertions that
the configuration produces no kernels and reports the documented rejection.
Remove the now-unused saved result. Do not bypass the product validation or
claim coverage of the unreachable TDMSplit emitter path.

## Consequences

The test now detects removal or silent failure of the TDMSplit guard. The
single-wave TDMSplit emitter lines remain uncovered until product support is
enabled and a configuration can reach them through normal solution derivation.
