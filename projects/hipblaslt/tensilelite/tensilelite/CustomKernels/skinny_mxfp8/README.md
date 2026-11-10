<!--
Copyright Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# Skinny MXFP8 custom kernels

Skinny GEMM kernels for MXFP8 (FP8 E4M3 A and B, E8M0 scales per 32 elements, BF16 D), hipBLASLt TN, with one
small ("skinny") extent and the other wide. They serve LLM decode and wider skinny batches up to the kernel `bm`.

gfx950 only. Five kernels, one `.s` file each. Each file carries one `custom.config` and one Tensile test YAML under
`Tests/custom/custom_skinny_mxfp8_tn_sel<N>.yaml`.

Each kernel computes `D = alpha * (A^T B) + beta * C` with f32 alpha and beta; the matmul gtests cover alpha 1,
beta 0 and alpha 1.5, beta 0.5.

## Kernels

| Idx | Test YAML (`Tests/custom/`) | `.s` file prefix | Path | Threads | Macrotile |
| --- | --------------------------- | ---------------- | ---- | ------- | --------- |
| 0 | `custom_skinny_mxfp8_tn_sel0.yaml` | `gemm_BM16_TN1_..._WAVES2_..._PF4_GROUP1_NUM_XCDS8_` | MFMA persistent | 128 | [16, 16, 4096] |
| 1 | `custom_skinny_mxfp8_tn_sel1.yaml` | `gemm_BM16_TN2_..._WAVES1_..._PF3_GROUP8_NUM_XCDS1_` | MFMA persistent | 64 | [32, 32, 4096] |
| 2 | `custom_skinny_mxfp8_tn_sel2.yaml` | `gemm_BM1_TN1_LANES_K16_..._PF2_` | VALU, one workgroup per tile | 64 | [4, 1, 4096] |
| 3 | `custom_skinny_mxfp8_tn_sel3.yaml` | `gemm_BM16_TN2_..._WAVES1_..._PF4_GROUP1_NUM_XCDS8_` | MFMA persistent | 64 | [32, 32, 4096] |
| 4 | `custom_skinny_mxfp8_tn_sel4.yaml` | `gemm_BM16_TN2_..._WAVES1_..._PF4_GROUP8_NUM_XCDS1_` | MFMA persistent | 64 | [32, 32, 4096] |

The kernels differ in the compute path, the number of waves, the wide tile width, the prefetch depth, and the
workgroup-to-tile mapping (`GROUP`, `NUM_XCDS`), and no single variant is fastest everywhere. The `custom.config` at the end
of each `.s` is the source of truth for macrotile, threads, grid and predicates.

## Numerics

The f32 result is converted to bf16 to nearest even (`v_cvt_pk_bf16_f32`, which every one of the five `.s` files
uses): D is the bf16 of `fma(beta, C, alpha * acc)` with alpha applied in f32 before the rounding. With beta zero
(the bit test, so -0.0 counts as zero) C is never read. The f32 accumulation order is that of the kernel (K is
reduced across lanes, waves and MFMA steps), so D is not claimed to be bit-identical to other kernels. No ulp
measurement is recorded here.

The VALU kernel (sel2) converts each operand to bf16 with its block scale applied before the dot product: a
`v_cvt_scalef32_pk_bf16_fp8` takes the E4M3 values and the E8M0 byte moved into the exponent field of an f32 (the
scale 2^(e-127)) for A and for B alike, and `v_dot2c_f32_bf16` accumulates the bf16 pairs in f32. bf16 holds every
E4M3 mantissa exactly, but the scaled values are limited to the bf16 range, and a scale byte of 0 becomes an f32
zero scale (the product is 0, not a * 2^-127) and 255 an infinite one. The MFMA kernels apply the scales inside the
scaled MFMA and do not have these limits.

## Metadata contract

The host does not parse the kernel symbol; everything it needs is in `custom.config`:

- `macrotile: [MT0, MT1, depth]` over `(m, n)` and `threads`.
- `grid: [TilesXYBatch, One, One]`: a 1-D launch of `ceil(m / MT0) * ceil(n / MT1) * batch` workgroups, because the
  kernels read only `workgroup_id_x`. Every kernel and every shard solution requires batch 1
  (`AssertSizeEqual: {2: 1}`), so the launch is `ceil(m / MT0) * ceil(n / MT1)` workgroups. The `numWorkGroupsX`
  kernarg is the `NumWorkGroups` semantic, the number of launched workgroups.
- Each kernel picks the skinny operand itself: `m <= n` makes m skinny, otherwise n is. `wide = max(m, n)` and
  `skinny = min(m, n)`.
- The persistent MFMA kernels (sel0, sel1, sel3, sel4) loop over tiles with stride `numWorkGroupsX` behind an entry
  guard, so any grid of at least one workgroup is correct. Their square macrotiles make `TilesXY` equal their tile
  count, `ceil(wide / 16) * ceil(skinny / 16)` (`ceil(wide / 32) * ceil(skinny / 16)` for the TN2 kernels sel1, sel3
  and sel4), whenever the skinny extent is at most 16, which always holds for the 16x16 kernel (sel0); the TN2
  kernels launch fewer workgroups above that and each loops. sel1 and sel4 differ from sel3 only in the
  workgroup-to-tile mapping (`GROUP8`, `NUM_XCDS1`) and, for sel1, the prefetch depth.
- The VALU kernel (sel2) does not read `numWorkGroupsX` and has no guard: it needs at least its tile count
  `ceil(wide / 4) * skinny` workgroups, and extra ones are masked. `TilesXY` of the `[4, 1]` macrotile is
  `ceil(m / 4) * n`, which equals that count only when m is the wide extent, hence the predicates `m > 4` and `n < 5`.
- No split-K: `GlobalSplitU` is 1, so the host infers no split-K workspace, and the workspace pointer is passed as
  `ConstantZero` and never read. The four MFMA kernels (sel0, sel1, sel3, sel4) reduce all of K for every output
  inside one workgroup in a launch-independent order. The VALU kernel (sel2) also reduces each output inside one
  workgroup, and every row reduces K in the same order. So all five declare `uniformSummationOrder: true`, the
  uniform summation order check (on by default) accepts them, and their test YAMLs run under that default.
- `MatrixInstruction: [16, 16, 128, 1]` is the scaled MFMA of the MFMA kernels and gives the client its MX-scale K
  granule; the VALU kernel declares it too for that gate.
- The f8 scaled MFMA (`v_mfma_scale_f32_16x16x128_f8f6f4`) needs 12 wait states on gfx950 before its result is read,
  as the LLVM hazard recognizer of amdclang inserts. The generator emits `s_nop 11` (12) after the last tail MFMA, and
  the four MFMA `.s` files were edited to `s_nop 13` for a margin of 2.
- K predicates: `AssertSummationElementMultiple: 32` on every kernel, the MX scale block (also what the generic
  MXFP8 kernels require). The kernels handle any such K; a K that is not a multiple of 32 runs the generic kernels.
  `AssertSizeGreaterThan: {3: 0}` rejects K = 0, which the kernels have no path for.
- Bias, activation and the scale-alpha vector are off in `custom.config` `Features` (`SupportsBias`,
  `SupportsActivation`, `SupportsScaleAlpha`), so each solution gets its own `ProblemType` with those off and rejects
  such problems even though a logic file's shared `ProblemType` has them. They fall through to the generic kernels.
- `MXScaleFormat: HostPreSwizzle` tells the client which scale layout to generate.
- The size arguments are named `m_size`, `n_size` and `k_size` in the `.s` HSA metadata because some LLVM versions
  (reported with ROCm 7.2.3) parse a bare `n` as a YAML boolean and reject the metadata; `m` and `k` are renamed for
  consistency. The names are descriptive only.

### Kernarg layout

96 bytes (`kernarg_segment_size` in the `.s` metadata). C and D are separate pointers.

| Offset | Name | Semantic | Meaning |
| ------ | ---- | -------- | ------- |
| 0 | D | `AddressD` | Output pointer. |
| 8 | A | `AddressA` | A pointer. |
| 16 | B | `AddressB` | B pointer. |
| 24 | scalesA | `AddressMXScaleA` | A scales pointer. |
| 32 | scalesB | `AddressMXScaleB` | B scales pointer. |
| 40 | ws | `ConstantZero` | `uint64`, passed as 0 and never read. |
| 48 | C | `AddressC` | C pointer. |
| 56 | ldD | `StrideD0Bytes` | D leading stride in bytes. |
| 60 | ldA | `StrideA0Bytes` | A leading stride in bytes. |
| 64 | ldB | `StrideB0Bytes` | B leading stride in bytes. |
| 68 | m_size | `SizeFree0` | m. |
| 72 | n_size | `SizeFree1` | n. |
| 76 | k_size | `SizeSum` | K. |
| 80 | ldC | `StrideC0Bytes` | C leading stride in bytes. |
| 84 | alpha | `Alpha` | f32. |
| 88 | beta | `Beta` | f32. |
| 92 | numWorkGroupsX | `NumWorkGroups` | Launched workgroups; read by the persistent MFMA kernels only. |

Leading strides in kernargs are **byte** counts. `generateCustomCall` computes them exactly in integers
(`multiplyElementSize(ld, elementBytes())`), because a floating-point `ld * elementBytes()` rounds to 24 bits and
corrupted leading dimensions above 2^24 (for example a bf16 ldD of 16777219). Solution predicates (`AssertStride*`)
use **element** leading dimensions.

### Size bounds

- `AssertTensorBytesLessThan: 2147483648` (2^31, the C++ `TensorBytesLessThan` predicate) requires every non-empty
  A, B, C and D to have `totalAllocatedBytes()` below the value, on all five kernels. The kernels compute each
  tensor's 32-bit buffer `num_records` as `(rows - 1) * ld + extent`, and the MFMA kernels use the byte offset
  0x80000000 as the out-of-range sentinel, so every buffer must stay below 2^31. The scale extents are bounded
  by those of A and B, so they need no separate guard. The predicate belongs to the solution, so it also applies to
  forced solution indices and to tuning.
- Batch is 1 (`AssertSizeEqual: {2: 1}`) on every kernel.
- The four MFMA kernels declare `AssertAnySizeLessThan: {0: 5, 1: 5}` (min(m, n) <= 4, an `Or` of two C++
  `SizeLessThan` predicates), so a forced solution index or a tuning run cannot put a larger problem on them. The
  VALU kernel keeps `n < 5` and `m > 4`.
- m, n and K are at most 2^20 (`AssertSizeLessThan: {0: 1048577, 1: 1048577, 3: 1048577}`), the range the kernels
  were generated for. The VALU kernel further requires `n < 5` (`1: 5`) and `m > 4`
  (`AssertSizeGreaterThan: {0: 4}`), see the metadata contract.
- The leading dimensions need no bound of their own: every ld product of an in-range row stays inside the buffer
  below 2^31, and rows outside the tensor (including rows of the skinny block past the skinny extent) are replaced
  by an out-of-range offset before any memory access.
- `AssertSizeGreaterThan` rejects empty problems.

Scales use the `HostPreSwizzle` layout with padding: K blocks padded to a multiple of 8 and the free extent to a
multiple of 32.

## hipBLASLt device library

hipBLASLt routes to these kernels only through an Equality logic file generated by TensileLite tuning of the five
test YAML forks over the decode shapes. Problems that are not listed, and those the predicates reject (bias,
activation, scale-alpha-vec, K that is not a multiple of 32, sizes out of bounds), run the generic kernels.

### Shard

The shard is `projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/gfx950/gfx950/Equality/gfx950_Cijk_Alik_Bljk_F8BS_MXAE8B32_MXBE8B32_BH_Bias_HA_S_SAB_SAV_UserArgs_SkinnyGEMM.yaml`.
It is the unmodified output of one TensileLite LibraryLogic run (`LibraryType: Equality`); the file was only
renamed.

- **Kernels:** the five `CustomKernel` blocks of the test YAMLs, each forked together with its own size predicates
  (a fork-parameter `Groups` entry), so that the VALU kernel is not launched for skinny m. The ProblemType is the
  one of the generic MXFP8 library, so hipBLASLt loads the shard next to it. That ProblemType has bias, activation
  and scale-alpha-vec, so the client always attaches them; the tuning feeds identity values (zero bias, scale-alpha
  vector and scales of 1, activation none), which makes the benchmarked problem equal to the plain GEMM the kernels
  compute. The generated ProblemType carries nine keys at their default values that the older generic file does not
  write; every other key is equal.
- **Benchmark settings:** alpha 1 and beta 0 (the decode case, the kernels skip C), kernel (GPU) time, 20 warmups, 20
  enqueues per sync, a 1024 MB rotating buffer, and `NumElementsToValidate: -1`, so every solution is fully validated
  on every size. `UseEffLike: False` makes the efficiencies GFLOP/s, like the other Equality files: duplicate keys
  across logic files resolve by the highest efficiency, and `test_skinny_mxfp8_logic.py` checks that the shard wins
  every key it shares. The tuning runs under the default uniform summation order, which all five kernels declare.
- **Sizes:** the tuning listed the sizes explicitly: 69 shapes in both orientations (m wide with n skinny 1 to 4,
  and the mirror with n wide and m skinny), 138 sizes in all. They are the 74 decode shapes minus five that were
  pruned because the generic kernels are faster on them
  (5120x4x25600, 8192x2x28672, 128256x1x8192, 128256x2x8192, 128256x4x8192).
- **Pruning:** a shape is dropped in both orientations when the generic heuristic kernel (a hipBLASLt build without
  the skinny kernels) is strictly faster in either, as timed by `hipblaslt-bench`. The five shapes above were dropped
  this way. One run of `hipblaslt-bench` timed all 138 remaining sizes in both orientations and dropped none: the
  shard kernel was faster than the generic one in every size (smallest margin 1.03 times, 84.45 vs 86.80 us for
  2x32768x14336).
- **Final comparison:** measured on gfx950 with `hipblaslt-bench` (50 iterations, 15 cold iterations)
  over the 74 decode shapes (n skinny); speedup is the base time over the new time. Against a build without the
  skinny kernels (the generic kernels), over all 74 shapes: geomean 1.62x, median 1.45x, min 0.99x. 69 shapes run a skinny
  kernel; the other 5 (5120x4x25600, 8192x2x28672, 128256x1x8192, 128256x2x8192, 128256x4x8192) run the generic
  kernel in both builds. Over the 69 skinny shapes: geomean 1.68x, median 1.48x, min 1.03x (8192x1x28672); the
  largest speedups are 3.35x for 1792x2x5120 and 3.33x for 1792x1x5120.
- **Result:** 138 keys, 69 shapes in each orientation. Keys per solution (0 to 4): 73, 3, 5, 56 and 1; the VALU
  kernel (solution 2) wins only skinny n. Solution 0 has 32 skinny-n and 41 skinny-m keys, solution 1 has 2 and 1,
  solution 3 has 30 and 26, and solution 4 has only the skinny-m key 3x16384x25600.

### Building

Build the skinny solutions **and** the Origami MXFP8 fallbacks (required for `matmul_skinny_mxfp8_tn_fallback*`
gtests and out-of-box shapes). Do **not** filter to the skinny solutions alone: that produces a tiny device library
with only the five custom solutions and no fallback kernels.

```bash
cd projects/hipblaslt
invoke build -ca gfx950 -f '*F8BS_MXAE8B32_MXBE8B32*' -c
export HIPBLASLT_TENSILE_LIBPATH=$PWD/build/release/Tensile/library/gfx950
```

## Tensile metadata

```bash
# Activate the TensileLite virtualenv first (see tensilelite/AGENTS.md).
cd projects/hipblaslt/tensilelite
python -m tensilelite.AddCustomConfig \
  tensilelite/CustomKernels/skinny_mxfp8/<kernel>.s \
  --yaml tensilelite/Tests/custom/custom_skinny_mxfp8_tn_sel<N>.yaml \
  --origin skinny_mxfp8 \
  --repository "Generated out of tree" \
  --version 1.0.0
```
