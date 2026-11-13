<!--
Copyright Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# Skinny MXFP8 custom kernels

Skinny GEMM kernels for MXFP8 (FP8 E4M3 A and B, E8M0 scales per 32 elements, BF16 D), hipBLASLt TN, with one
small ("skinny") extent and the other wide. They serve LLM decode and wider skinny batches up to the kernel `bm`.
gfx950 only. Each kernel computes `D = alpha * (A^T B) + beta * C` with f32 alpha and beta.

Each of the five kernels is one `.s` file with its own `custom.config` and one Tensile test YAML under
`Tests/custom/custom_skinny_mxfp8_tn_sel<N>.yaml`. The `custom.config` at the end of each `.s` is the source of truth
for macrotile, threads, grid and predicates.

## Kernels

| Idx | `.s` file prefix | Path | Threads | Macrotile |
| --- | ---------------- | ---- | ------- | --------- |
| 0 | `gemm_BM16_TN1_..._WAVES2_..._PF4_GROUP1_NUM_XCDS8_` | MFMA persistent | 128 | [16, 16, 4096] |
| 1 | `gemm_BM16_TN2_..._WAVES1_..._PF3_GROUP8_NUM_XCDS1_` | MFMA persistent | 64 | [32, 32, 4096] |
| 2 | `gemm_BM1_TN1_LANES_K16_..._PF2_` | VALU, one workgroup per tile | 64 | [4, 1, 4096] |
| 3 | `gemm_BM16_TN2_..._WAVES1_..._PF4_GROUP1_NUM_XCDS8_` | MFMA persistent | 64 | [32, 32, 4096] |
| 4 | `gemm_BM16_TN2_..._WAVES1_..._PF4_GROUP8_NUM_XCDS1_` | MFMA persistent | 64 | [32, 32, 4096] |

The kernels differ in compute path, wave count, wide tile width, prefetch depth and workgroup-to-tile mapping
(`GROUP`, `NUM_XCDS`); no single variant is fastest everywhere.

## Numerics

D is the bf16 (round to nearest even, `v_cvt_pk_bf16_f32`) of `fma(beta, C, alpha * acc)`, with alpha applied in f32
before the rounding. With beta zero (the bit test, so -0.0 counts as zero) C is never read. The f32 accumulation order
is that of the kernel, so D is not claimed to be bit-identical to other kernels.

The VALU kernel (sel2) converts each operand to bf16 with its block scale applied (`v_cvt_scalef32_pk_bf16_fp8`) and
accumulates with `v_dot2c_f32_bf16`. The scaled values are limited to the bf16 range, a scale byte of 0 becomes a zero
scale (the product is 0, not a * 2^-127) and 255 an infinite one. The MFMA kernels apply the scales inside the scaled
MFMA and do not have these limits.

## Metadata contract

The host does not parse the kernel symbol; everything it needs is in `custom.config`.

- `macrotile: [MT0, MT1, depth]` over `(m, n)` and `threads` size the launch.
- `grid: [TilesXYBatch, One, One]` is a 1-D launch of `ceil(m / MT0) * ceil(n / MT1)` workgroups (batch is 1); the
  `numWorkGroupsX` kernarg is that count.
- Each kernel picks the skinny operand itself: `m <= n` makes m skinny, otherwise n is.
- The persistent MFMA kernels (sel0, sel1, sel3, sel4) loop over tiles with stride `numWorkGroupsX` behind an entry
  guard, so any grid of at least one workgroup is correct.
- The VALU kernel (sel2) has no guard and needs at least `ceil(wide / 4) * skinny` workgroups, which `TilesXY` gives
  only when m is the wide extent; hence its predicates `m > 4` and `n < 5`.
- `GlobalSplitU` is 1 and the workspace pointer (`ConstantZero`) is never read. Every output is reduced inside one
  workgroup in a launch-independent order, so all five declare `uniformSummationOrder: true`.
- `MatrixInstruction: [16, 16, 128, 1]` gives the client its MX-scale K granule (the VALU kernel declares it too).
- The scaled MFMA needs 12 wait states before its result is read; the generator emits `s_nop 11` (12) after the last
  tail MFMA and the four MFMA `.s` files keep it.
- `AssertSummationElementMultiple: 32` (the MX scale block, as in the generic MXFP8 kernels, so other K has no MXFP8
  kernel at all) and `AssertSizeGreaterThan: {3: 0}` (K = 0 has no path).
- `SupportsBias`, `SupportsActivation` and `SupportsScaleAlpha` are false, so each solution gets its own
  `ProblemType` that rejects such problems; they fall through to the generic kernels.
- `MXScaleFormat: HostPreSwizzle` tells the client which scale layout to generate.
- `AssertMXScalesPreSwizzled: true` (C++ `MXScalesPreSwizzled`) passes only when neither operand's scales are flagged
  packed. hipBLASLt flags every format except `Block_32_UE8M0_32_8_EXT` as packed. Through hipBLASLt on gfx950 the
  host scale-layout checks already stop plain 32-block scales (`--scaleA 3 --scaleB 3`) earlier, since every generic
  MXFP8 solution is also `HostPreSwizzle`; the predicate covers direct TensileLite users and library selection. The
  TensileLite client leaves the flags false, so tuning and the Tensile tests still select these kernels.
- The size arguments are `m_size`, `n_size`, `k_size` in the `.s` metadata because some LLVM versions (seen with ROCm
  7.2.3) parse a bare `n` as a YAML boolean; the names are descriptive only.

### Kernarg layout

96 bytes (`kernarg_segment_size`). C and D are separate pointers. Leading strides are **byte** counts, computed in
integers by `generateCustomCall`; the solution predicates (`AssertStride*`) use **element** leading dimensions.

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

### Size bounds

- Batch is 1 (`AssertSizeEqual: {2: 1}`); m, n and K are at most 2^20 (`AssertSizeLessThan`), the range the kernels
  were generated for; `AssertSizeGreaterThan` rejects empty problems.
- The MFMA kernels declare `AssertAnySizeLessThan: {0: 5, 1: 5}` (min(m, n) <= 4), so a forced solution index or a
  tuning run cannot put a larger problem on them. The VALU kernel keeps `n < 5` and `m > 4`.
- `AssertTensorBytesLessThan: 2147483648` requires every non-empty A, B, C and D to be below 2^31 bytes. `num_records`
  and byte offsets are 32-bit, and the MFMA kernels send the A and B row offset of a row outside the tensor to
  0x80000000 (`v_mov_b32 v, -2147483648` before the `v_cndmask_b32`), which must stay beyond those buffers. The VALU
  kernel has no sentinel. Leading dimensions need no bound of their own.

**`num_records` invariant.** Each kernel builds `num_records` for every SRD from the tensor extent; it must never come
from the host or be enlarged (for example to `0x7fffffff`). The prologue and tail load past the end of K when K is
below the prefetch depth or not a multiple of the kernel's K step, and that is safe only because the hardware
range-checks `soffset` against `num_records`. The K predicates are deliberately not tightened. With `ld` in bytes and
`word3` = `0x20000` for every buffer:

- A: `(m - 1) * ldA + roundup4(K)`; B: `(n - 1) * ldB + roundup4(K)`; up to 3 bytes past the last row.
- MX scales of A: `ceil(m / 32) * roundup256(K)`; of B: `ceil(n / 32) * roundup256(K)`; whole padded blocks of the
  host pre-swizzled layout, backed by the host padding.
- D: `(n - 1) * ldD + 2 * m`; C: `(n - 1) * ldC + 2 * m`.

## hipBLASLt device library

hipBLASLt routes to these kernels only through the Equality logic shard below, generated by TensileLite tuning of the
five test YAML forks over the decode shapes. Problems that are not shard keys, or that the predicates reject, run the
generic kernels.

The shard is `projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/gfx950/gfx950/Equality/gfx950_Cijk_Alik_Bljk_F8BS_MXAE8B32_MXBE8B32_BH_Bias_HA_S_SAB_SAV_UserArgs_SkinnyGEMM.yaml`.

- **Provenance:** it started as a LibraryLogic output and was hand-edited (predicates and wave geometry), so it cannot
  be regenerated without a new tuning run; `test_skinny_mxfp8_logic.py` checks it against each kernel's embedded
  `custom.config`. The geometry matters because `sizeMapping.workGroupSize` feeds the launch-limit and auto-GSU checks.
- **Chip ids:** the shard lists `DeviceNames: [Device 75a0]` and serves the ids in `SUPPORTED_CHIP_ID_FALLBACKS`
  (`Architectures.py`): 75b0, 75a2, 75b2, 75a3, 75b3, 75a8 and 75b8. Validated on 0x75a3 (MI355X) only.
- **Shadowing:** an exact-chip row that returns a solution wins over the fallback row, so a `gfx950_idXXXX/` MXFP8 TN
  logic file would stop the shard from applying on that chip; `test_no_chip_specific_logic_shadows_the_skinny_shard`
  fails if one appears.

### Verification

Run on one MI355X (gfx950, chip 0x75a3) from `projects/hipblaslt` after the build below. `TENSILE_DB=0x8000` prints the
`Running kernel:` line, which ends in `[MatchingTag::Equal]` for a shard key. Scale `1001` is the pre-swizzled
32-block format the kernels read.

```bash
export HIP_VISIBLE_DEVICES=0 TENSILE_DB=0x8000
export HIPBLASLT_TENSILE_LIBPATH=$PWD/build/release/Tensile/library/gfx950
COMMON="--transA T --transB N --a_type f8_r --b_type f8_r --c_type bf16_r --d_type bf16_r --compute_type f32_r"
build/release/clients/hipblaslt-bench -m 51200 -n 2 -k 5120 $COMMON --scaleA 1001 --scaleB 1001 --verify --verify_strict
```

Each row replaces `-m -n -k` of that command; the kernel is `Equal` and verification passes.

| Shard key (m x n x K) | Expected `Running kernel:` |
| --------------------- | -------------------------- |
| 51200x2x5120 | `gemm_BM16_TN1_..._WAVES2_..._PF4_GROUP1_NUM_XCDS8_...` (sel0) |
| 2x10240x5120 (mirror) | sel0 |
| 32768x2x14336 | `gemm_BM16_TN2_..._PF3_GROUP8_NUM_XCDS1_...` (sel1) |
| 28672x1x4096 | `gemm_BM1_TN1_LANES_K16_..._PF2_...` (sel2, VALU) |
| 5120x1x25600 | `gemm_BM16_TN2_..._PF4_GROUP1_NUM_XCDS8_...` (sel3) |
| 3x16384x25600 | `gemm_BM16_TN2_..._PF4_GROUP8_NUM_XCDS1_...` (sel4) |

Negative controls, each the 51200x2x5120 command without `--verify`, with the change below. A generic kernel is a
`Cijk_` kernel with `[MatchingTag::Others]`.

| Control | Change | Expected result |
| ------- | ------ | --------------- |
| Batch 2 | `--batch_count 2` | generic kernel |
| Bias | `--bias_vector --bias_type bf16_r` | generic kernel |
| K not a multiple of 32 | `-k 5124` | `NO solution found`, no MXFP8 kernel supports it |
| Plain 32-block scales | `--scaleA 3 --scaleB 3` | `NO solution found`, see the `AssertMXScalesPreSwizzled` bullet |

### Building

Build the skinny solutions **and** the Origami MXFP8 fallbacks (needed by the `matmul_skinny_mxfp8_tn_fallback*`
gtests and out-of-box shapes). Do **not** filter to the skinny solutions alone: that yields a device library with only
the five custom solutions and no fallback kernels.

```bash
cd projects/hipblaslt
invoke build -ca gfx950 -f '*F8BS_MXAE8B32_MXBE8B32*' -c
```

`custom.config` was added with `python -m tensilelite.AddCustomConfig <kernel>.s --yaml
tensilelite/Tests/custom/custom_skinny_mxfp8_tn_sel<N>.yaml --origin skinny_mxfp8 --repository "Generated out of
tree" --version 1.0.0`; see `tensilelite/CustomKernels/README.md` for the flags and the validation rules.
