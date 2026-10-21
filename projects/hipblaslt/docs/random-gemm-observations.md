# Random GEMM observations

CPU-only gfx950 exploration on 2026-10-21 used rocJITsu `29656a8dcc05` and the
[matched build artifacts](https://github.com/ROCm/rocm-libraries/actions/runs/39535282903).
Every nonpassing input is listed below. The [complete input records](random-gemm-observations.json)
include initialization, leading dimensions, strides and all benchmark controls.

`unsupported` means the loaded library supplied no heuristic solution for that
exact configuration. It does not prove that the public API forbids the input.
The benchmark prints `error: NO solution found!` and exits 0; the runner counts
the case as unsupported and makes the overall run unsuccessful. No replacement
input is sampled. Unsupported cases have no completed GEMM numerical result.

All rows use FP32 computation, FP32 bias type and `rand_int` initialization.
A/B and C/D columns name both types; scales `1001` mean preswizzled E8M0 block
scales on both operands. Layout is packed unless a row specifies
`lda/ldb/ldc/ldd; stride_a/stride_b/stride_c/stride_d`.

| Case | M×N×K | A/B → C/D | Transpose; batch | Alpha; beta | Epilogue; scales | Layout | Observed failure |
| --- | --- | --- | --- | --- | --- | --- | --- |
| explore-broad-600/case-0001 | 4×1×64 | f16_r/f16_r → f16_r/f16_r | NN; 1 | 1; 1 | gelu; 0/0 | packed | GELU: norm 1; absolute difference 2.98e-7 |
| explore-broad-600/case-0028 | 129×2×32 | f8_r/f8_r → bf16_r/bf16_r | NT; 2 | -3; 1 | bias+relu; 0/0 | 132/18/130/145; 4231/577/292/297 | No heuristic solution |
| explore-broad-600/case-0148 | 33×4×4 | f8_r/f8_r → bf16_r/bf16_r | NT; 2 | -3; 0 | bias+none; 0/0 | 49/7/49/34; 228/29/197/168 | No heuristic solution |
| explore-broad-600/case-0521 | 32×32×4096 | f8_r/f8_r → f16_r/f16_r | TN; 1 | 2; 1 | bias+relu; 1001/1001 | packed | Matching infinities; norm NaN |
| explore-broad-600/case-0551 | 32×32×2048 | f8_r/f8_r → f16_r/f16_r | TN; 1 | 1; 0.5 | relu; 1001/1001 | packed | Matching infinities; norm NaN |
| explore-broad-1000-seed42/case-0001 | 4×1×64 | f16_r/f16_r → f16_r/f16_r | NN; 1 | 1; 1 | gelu; 0/0 | packed | GELU: norm 1; absolute difference 2.98e-7 |
| explore-broad-1000-seed42/case-0076 | 129×7×3 | f8_r/f8_r → bf16_r/bf16_r | NT; 3 | 1; 0.75 | gelu; 0/0 | 132/23/130/145; 403/76/917/1022 | No heuristic solution |
| explore-broad-1000-seed42/case-0503 | 32×32×4096 | f8_r/f8_r → f16_r/f16_r | TN; 1 | 0.5; 0 | bias+relu; 1001/1001 | packed | Matching infinities; norm NaN |
| explore-broad-1000-seed42/case-0568 | 7×128×7 | f8_r/f8_r → bf16_r/bf16_r | NT; 2 | 0.75; 0 | bias+gelu; 0/0 | 23/129/10/23; 193/910/1312/2945 | No heuristic solution |
| explore-broad-1000-seed42/case-0683 | 32×32×4096 | f8_r/f8_r → f16_r/f16_r | TN; 1 | 0.5; 1 | bias+relu; 1001/1001 | packed | Matching infinities; norm NaN |
| explore-broad-1000-seed42/case-0754 | 2×129×4 | f8_r/f8_r → bf16_r/bf16_r | NN; 3 | -0.125; -0.125 | bias+relu; 0/0 | 18/7/18/3; 79/904/2323/394 | No heuristic solution |
| explore-broad-1000-seed42/case-0893 | 16×1×1024 | f16_r/f16_r → f32_r/f32_r | NT; 1 | 0.125; 0.75 | relu; 0/0 | packed | No heuristic solution |
| extra-types-probes/case-0009 | 32×64×256 | f6_r/f6_r → f16_r/f16_r | TN; 1 | 1; 0 | gelu; 1001/1001 | packed | No heuristic solution |
| extra-types-probes/case-0010 | 128×192×256 | f6_r/f6_r → f16_r/f16_r | TN; 1 | 1; 0 | gelu; 1001/1001 | packed | No heuristic solution |
| extra-types-probes/case-0011 | 32×64×256 | f6_r/f6_r → bf16_r/bf16_r | TN; 1 | 1; 0 | gelu; 1001/1001 | packed | No heuristic solution |
| extra-types-probes/case-0012 | 128×192×256 | f6_r/f6_r → bf16_r/bf16_r | TN; 1 | 1; 0 | gelu; 1001/1001 | packed | No heuristic solution |
| extra-types-probes/case-0013 | 32×64×256 | f6_r/f6_r → f32_r/f32_r | TN; 1 | 1; 0 | gelu; 1001/1001 | packed | No heuristic solution |
| extra-types-probes/case-0014 | 128×192×256 | f6_r/f6_r → f32_r/f32_r | TN; 1 | 1; 0 | gelu; 1001/1001 | packed | No heuristic solution |
| extra-types-probes/case-0015 | 32×64×256 | bf6_r/bf6_r → f16_r/f16_r | TN; 1 | 1; 0 | gelu; 1001/1001 | packed | No heuristic solution |
| extra-types-probes/case-0016 | 128×192×256 | bf6_r/bf6_r → f16_r/f16_r | TN; 1 | 1; 0 | gelu; 1001/1001 | packed | No heuristic solution |
| extra-types-probes/case-0017 | 32×64×256 | bf6_r/bf6_r → bf16_r/bf16_r | TN; 1 | 1; 0 | gelu; 1001/1001 | packed | No heuristic solution |
| extra-types-probes/case-0018 | 128×192×256 | bf6_r/bf6_r → bf16_r/bf16_r | TN; 1 | 1; 0 | gelu; 1001/1001 | packed | No heuristic solution |
| extra-types-probes/case-0019 | 32×64×256 | bf6_r/bf6_r → f32_r/f32_r | TN; 1 | 1; 0 | gelu; 1001/1001 | packed | No heuristic solution |
| extra-types-probes/case-0020 | 128×192×256 | bf6_r/bf6_r → f32_r/f32_r | TN; 1 | 1; 0 | gelu; 1001/1001 | packed | No heuristic solution |
| extra-types-probes/case-0021 | 32×64×256 | f4_r/f8_r → f16_r/f16_r | TN; 1 | 1; 0 | gelu; 1001/1001 | packed | No heuristic solution |
| extra-types-probes/case-0022 | 128×192×256 | f4_r/f8_r → f16_r/f16_r | TN; 1 | 1; 0 | gelu; 1001/1001 | packed | No heuristic solution |
| extra-types-probes/case-0023 | 32×64×256 | f4_r/f8_r → bf16_r/bf16_r | TN; 1 | 1; 0 | gelu; 1001/1001 | packed | No heuristic solution |
| extra-types-probes/case-0024 | 128×192×256 | f4_r/f8_r → bf16_r/bf16_r | TN; 1 | 1; 0 | gelu; 1001/1001 | packed | No heuristic solution |
| extra-types-probes/case-0025 | 32×64×256 | f4_r/f8_r → f32_r/f32_r | TN; 1 | 1; 0 | gelu; 1001/1001 | packed | No heuristic solution |
| extra-types-probes/case-0026 | 128×192×256 | f4_r/f8_r → f32_r/f32_r | TN; 1 | 1; 0 | gelu; 1001/1001 | packed | No heuristic solution |
| extra-types-probes/case-0027 | 32×64×256 | f8_r/f4_r → f16_r/f16_r | TN; 1 | 1; 0 | gelu; 1001/1001 | packed | No heuristic solution |
| extra-types-probes/case-0028 | 128×192×256 | f8_r/f4_r → f16_r/f16_r | TN; 1 | 1; 0 | gelu; 1001/1001 | packed | No heuristic solution |
| extra-types-probes/case-0029 | 32×64×256 | f8_r/f4_r → bf16_r/bf16_r | TN; 1 | 1; 0 | gelu; 1001/1001 | packed | No heuristic solution |
| extra-types-probes/case-0030 | 128×192×256 | f8_r/f4_r → bf16_r/bf16_r | TN; 1 | 1; 0 | gelu; 1001/1001 | packed | No heuristic solution |
| extra-types-probes/case-0031 | 32×64×256 | f8_r/f4_r → f32_r/f32_r | TN; 1 | 1; 0 | gelu; 1001/1001 | packed | No heuristic solution |
| extra-types-probes/case-0032 | 128×192×256 | f8_r/f4_r → f32_r/f32_r | TN; 1 | 1; 0 | gelu; 1001/1001 | packed | No heuristic solution |
| extra-types-probes/case-0033 | 32×64×256 | f6_r/f8_r → f16_r/f16_r | TN; 1 | 1; 0 | gelu; 1001/1001 | packed | No heuristic solution |
| extra-types-probes/case-0034 | 128×192×256 | f6_r/f8_r → f16_r/f16_r | TN; 1 | 1; 0 | gelu; 1001/1001 | packed | No heuristic solution |
| extra-types-probes/case-0035 | 32×64×256 | f6_r/f8_r → bf16_r/bf16_r | TN; 1 | 1; 0 | gelu; 1001/1001 | packed | No heuristic solution |
| extra-types-probes/case-0036 | 128×192×256 | f6_r/f8_r → bf16_r/bf16_r | TN; 1 | 1; 0 | gelu; 1001/1001 | packed | No heuristic solution |
| extra-types-probes/case-0037 | 32×64×256 | f6_r/f8_r → f32_r/f32_r | TN; 1 | 1; 0 | gelu; 1001/1001 | packed | No heuristic solution |
| extra-types-probes/case-0038 | 128×192×256 | f6_r/f8_r → f32_r/f32_r | TN; 1 | 1; 0 | gelu; 1001/1001 | packed | No heuristic solution |
| extra-types-probes/case-0039 | 32×64×256 | f8_r/bf6_r → f16_r/f16_r | TN; 1 | 1; 0 | gelu; 1001/1001 | packed | No heuristic solution |
| extra-types-probes/case-0040 | 128×192×256 | f8_r/bf6_r → f16_r/f16_r | TN; 1 | 1; 0 | gelu; 1001/1001 | packed | No heuristic solution |
| extra-types-probes/case-0041 | 32×64×256 | f8_r/bf6_r → bf16_r/bf16_r | TN; 1 | 1; 0 | gelu; 1001/1001 | packed | No heuristic solution |
| extra-types-probes/case-0042 | 128×192×256 | f8_r/bf6_r → bf16_r/bf16_r | TN; 1 | 1; 0 | gelu; 1001/1001 | packed | No heuristic solution |
| extra-types-probes/case-0043 | 32×64×256 | f8_r/bf6_r → f32_r/f32_r | TN; 1 | 1; 0 | gelu; 1001/1001 | packed | No heuristic solution |
| extra-types-probes/case-0044 | 128×192×256 | f8_r/bf6_r → f32_r/f32_r | TN; 1 | 1; 0 | gelu; 1001/1001 | packed | No heuristic solution |

The two broad campaigns include the same deliberately retained tiny GELU case.
Five distinct numerical inputs were confirmed by serial replays. No race or
rocJITsu warning/error appeared in the original exploration cases. Neither numerical failures nor
missing solutions are treated as emulator blacklist entries. Eight exploratory
INT8/FP64 inputs initially had incorrect compute-input options; after correcting
those harness settings, all eight passed and are omitted from the failure table.

The [required-case list](../../../.github/scripts/hipblaslt_explicit_gemms.json)
also contains BF16 TN shapes associated with
[the prefetch ordering race](https://github.com/ROCm/rocm-libraries/pull/14147).
Those tests use normal heuristic selection. Passing a shape does not establish
that the specific custom-schedule kernel from that report executed; the selected
kernel is recorded in each result. The reported one-solution regression remains
the evidence for that exact schedule.

A later run with the explicit list produced one additional nonpassing case:

| Case | M×N×K | Types; transpose; batch | Alpha; beta; epilogue | Observation |
| --- | --- | --- | --- | --- |
| explicit-and-random-100/case-0072 | 32×416×224 | FP32 throughout; TT; 1 | 0.5; -1; FP32 bias+ReLU | `SET_EVENT_MISS: event_id=6 events_.size()=19` after numerical completion; bench exits 0, runner fails. One serial replay passes without the warning. |

This warning remains unclassified and has no blacklist entry. The full input
is included in the machine-readable records above.

## Required cases against the shipping payload

The new required-case list ran alongside 100 random samples. All 114 inputs
were attempted: 108 passed, five failed numerical validation, and one random
case emitted the shutdown warning documented above. No race was reported.
The original BF16 batched case also passed an independent replay in 102 seconds.

| Required case | M×N×K; batch | Result |
| --- | --- | --- |
| bf16-tn-prefetch-k128 | 256×160×128; 1 | passed: numerics and race detection passed |
| bf16-tn-prefetch-k129 | 256×160×129; 1 | passed: numerics and race detection passed |
| bf16-tn-prefetch-k192 | 256×160×192; 1 | passed: numerics and race detection passed |
| bf16-tn-prefetch-k193 | 256×160×193; 1 | passed: numerics and race detection passed |
| bf16-tn-prefetch-k256 | 256×160×256; 1 | passed: numerics and race detection passed |
| bf16-tn-prefetch-k512 | 256×160×512; 1 | passed: numerics and race detection passed |
| bf16-tn-prefetch-k64 | 256×160×64; 1 | passed: numerics and race detection passed |
| bf16-tn-prefetch-k65 | 256×160×65; 1 | passed: numerics and race detection passed |
| bf16-tn-prefetch-reported | 1024×4096×4096; 2 | passed: numerics and race detection passed |
| fp16-gelu-near-zero | 4×1×64; 1 | failed: bench exited 1 |
| mxfp8-fp16-overflow-k2048 | 32×32×2048; 1 | failed: bench exited 1 |
| mxfp8-fp16-overflow-k4096 | 32×32×4096; 1 | failed: bench exited 1 |
| mxfp8-fp16-overflow-seed42-case503 | 32×32×4096; 1 | failed: bench exited 1 |
| mxfp8-fp16-overflow-seed42-case683 | 32×32×4096; 1 | failed: bench exited 1 |

All BF16 TN cases use BF16 operands/output, FP32 computation, alpha1, beta0,
no epilogue or block scales, and `hpl` initialization. The reduced K192 case
selects a 32×16 kernel and the original shape selects a 256×128 kernel; neither
is the 256×160 custom schedule in #14147. These passes therefore establish
shape execution, not a before/after verification of that particular race.
