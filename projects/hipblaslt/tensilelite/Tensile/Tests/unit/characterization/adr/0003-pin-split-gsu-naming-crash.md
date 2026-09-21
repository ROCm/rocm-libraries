# ADR 0003: Pin the split-GSU naming crash

Status:  Accepted
Defect:  AIHPBLAS-4297
Commit:  f5c0ffd (PR #10021) — https://github.com/ROCm/rocm-libraries/commit/f5c0ffda72bd554e4544a3b0a343cea58d98708e (pinned the crash + drafted this ADR) and https://github.com/ROCm/rocm-libraries/commit/cce9e4cefafe9c07bf8ddf702343815616e50a19 (linked the AIHPBLAS-4297 defect) — landed on develop via PR #10021's squash merge, 44b93a0

## Context
Mutation-validating `SolutionStructs/Naming.py` confirmed a characterized crash in `getKernelNameMin`: with `splitGSU=True`, `GlobalSplitU > 1` or `-1` was first rewritten to the string `"M"` and then evaluated by `"M" > 0`, raising `TypeError`.

This is currently a latent debug/configuration path, not a production-reachable codegen path. `DebugConfig.splitGSU` defaults to false, accepts `config["SplitGSU"]` only when supplied, and `SplitGSU` is not registered in `globalParameters`; the normal `TensileCreateLibrary` paths also set `splitGSU = False`. If the flag is registered or otherwise made reachable, split and automatic GSU solutions will fail canonical naming.

## Decision
Keep the production implementation unchanged and pin the actual `TypeError` for the `GlobalSplitU` boundary and automatic-sentinel cases. Treat only mutants that preserve this crash as accepted equivalents, with their exact disposition recorded in `DECISIONS.md`.

## Consequences
The golden suite deliberately preserves a latent naming defect rather than silently correcting production behavior during characterization. A future fix must replace the `TypeError` assertion with the intended canonical-name assertion, update the mutation disposition, resolve the defect above, and supersede this ADR.
