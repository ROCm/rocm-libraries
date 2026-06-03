# Characterization target — `Tensile/BenchmarkStructs.py`

Part of the master-plan remaining-module sweep. Covers getDefaultsForMissing
Parameters, separateParameters (single/multi/None-exit), checkCDBufferAndStrides
(ok/mismatch-exit/not-CEqualD), constructForkPermutations + constructLazyFork
Permutations (cartesian product), and BenchmarkStep (basic + custom-kernel
wildcard). 11 tests.

**Accepted <95% — see DECISIONS D11.** `BenchmarkProcess` (L83-235), the
config->benchmark-steps integration builder, needs a full benchmark config
(problemType + problemSizeGroup) — an end-to-end fixture, deferred.
