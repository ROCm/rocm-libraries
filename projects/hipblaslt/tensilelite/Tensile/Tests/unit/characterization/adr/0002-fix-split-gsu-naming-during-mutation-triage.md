# ADR 0002: Pin the split-GSU naming crash

Status:  Accepted
Defect:  AIHPBLAS-XXXX (to file)

## Context
Mutation-validating `SolutionStructs/Naming.py` confirmed a characterized crash in `getKernelNameMin`: with `splitGSU=True`, `GlobalSplitU > 1` or `-1` was first rewritten to the string `"M"` and then evaluated by `"M" > 0`, raising `TypeError`. This prevents canonical names from being built for valid split/automatic GSU solutions.

The same triage identified behavior-neutral sources of equivalent mutants: `getKeyNoInternalArgs` normalized and restored `WorkGroupMappingXCC` immediately before `_getName` performed the same normalization/restoration, and `getParameterValueAbbreviation` had an unreachable final branch after exhaustively handling every admitted composite type.

## Decision
Keep the production implementation unchanged and pin the actual `TypeError` for the `GlobalSplitU` boundary and automatic-sentinel cases. Treat only mutants that preserve this crash or alter unreachable/redundant operations as accepted equivalents, with their exact disposition recorded in `DECISIONS.md`.

## Consequences
The golden suite deliberately preserves a real naming defect rather than silently correcting production behavior during characterization. Mutation certification will retain a small, documented equivalent set caused by the pinned crash and redundant/unreachable source. A future fix must replace the `TypeError` assertion with the intended canonical-name assertion, update the mutation disposition, file or resolve the defect above, and supersede this ADR.
