# TensileCreateLibrary/Run.py — characterization target (helper layer)

Run.py is the TensileCreateLibrary driver; the bulk is asm codegen
(processKernelSource / writeAssembly / writeSolutionsAndKernels[TCL] / run() /
generateLogicDataAndSolutions) via KernelWriterAssembly + toolchain, out of
scope. This suite pins the pure / stubbable helpers:

- libraryDir (single / zero / multi arch)
- KernelCodeGenResult / KernelMinResult NamedTuples
- _stinky_asm_verify_wanted (flag/arch matrix) + _stinky_out
- memCompress / memDecompress roundtrip
- _checkInvalidSolutionsAndKernels (ok / err-not-tolerant / err-tolerant)
- _checkInvalidSolutions
- removeInvalidSolutionsAndKernels (ParallelMap2 + Naming stubbed)
- passPostKernelInfoToSolution
- _renameFallbackPlaceholders / renameFallbacksPerArch (leaf / idempotent /
  non-fallback / rows+mapping walk / per-arch deep-copy)

Resistance (codegen, integration-test-covered only): the asm generation, file
writers, passPostKernelInfoToLibrary (large sizeMapping/_state field set), and
run() CLI driver.
