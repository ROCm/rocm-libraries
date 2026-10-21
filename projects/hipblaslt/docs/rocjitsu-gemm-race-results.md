# rocJITsu GEMM race-test results

The [race-test runner](rocjitsu-gemm-race-testing.md) runs 100 fixed-seed random
GEMMs plus one required BF16 input per PR, and 1,000 plus that input nightly.
It checks numerical validation and race detection entirely on CPUs using gfx950
emulation.

The current per-PR workload passed locally: all 101 inputs completed with no
races, numerical failures, missing solutions or emulator diagnostics. The
shared CI driver took 42.5 seconds with eight concurrent GEMMs, using the
[matched gfx950 artifacts](https://github.com/ROCm/rocm-libraries/actions/runs/39535282903)
and rocJITsu `29656a8dcc05`. Its 100 random inputs match the nightly prefix
exactly, and both workloads use the same explicit BF16 input. These results
come from local CPU execution; they do not claim completion of hosted CI.

The required BF16 TN case is 256×160×192 with batch1, FP32 computation, alpha1,
beta0, no epilogue and `hpl` initialization. It passes through normal heuristic
selection. It covers the shape without guaranteeing execution of the particular
kernel from [the prefetch race report](https://github.com/ROCm/rocm-libraries/pull/14147).

Failure-handling checks use injected faults in task-owned copies of a selected
MXFP4 kernel. The race control reports a race and the numerical control fails
validation; both produce failed case results and an unsuccessful runner exit.
The runner tests also check deterministic inputs, concurrent execution, replay,
early stopping, missing solutions, subprocess launch failures and timeouts.

## Optional numerical diagnostics

The separate [diagnostic list](../../../.github/scripts/hipblaslt_diagnostic_gemms.json)
contains five numerical reproducers. They are not part of the default per-PR
or nightly workload. They use the same strict checks and fail when run; no
expected-failure exemption is applied.

All use FP32 computation, batch1, packed layout and `rand_int` initialization.
MXFP8 inputs use preswizzled E8M0 block scales on both operands.

| Case | M×N×K | Types; transpose | Alpha; beta | Epilogue | Observed failure |
| --- | --- | --- | --- | --- | --- |
| fp16-gelu-near-zero | 4×1×64 | FP16 → FP16; NN | 1; 1 | GELU | Relative norm 1 for absolute difference 2.98e-7 |
| mxfp8-fp16-overflow-k4096 | 32×32×4096 | MXFP8 → FP16; TN | 2; 1 | FP32 bias+ReLU | Matching infinities produce norm NaN |
| mxfp8-fp16-overflow-k2048 | 32×32×2048 | MXFP8 → FP16; TN | 1; 0.5 | ReLU | Matching infinities produce norm NaN |
| mxfp8-fp16-overflow-seed42-case503 | 32×32×4096 | MXFP8 → FP16; TN | 0.5; 0 | FP32 bias+ReLU | Every output matches the reference, including 32 infinities; norm NaN |
| mxfp8-fp16-overflow-seed42-case683 | 32×32×4096 | MXFP8 → FP16; TN | 0.5; 1 | FP32 bias+ReLU | Every output matches the reference, including 32 infinities; norm NaN |
