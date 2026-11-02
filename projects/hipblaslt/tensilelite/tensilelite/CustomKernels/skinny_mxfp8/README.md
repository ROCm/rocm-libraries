<!--
Copyright Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# Skinny MXFP8 custom kernels

Skinny GEMM kernels for MXFP8 (FP8 E4M3 A and B, E8M0 scales per 32 elements, BF16 D), hipBLASLt TN, with one
small ("skinny") extent and the other wide. They serve LLM decode and wider skinny batches up to the kernel `bm`.

gfx950 only. Four kernels, one `.s` file each. Each file carries one `custom.config` and one Tensile test YAML under
`Tests/custom/custom_skinny_mxfp8_tn_sel<N>.yaml`.

Each kernel computes `D = alpha * (A^T B) + beta * C` for any alpha and beta in f32.

## Kernels

Four kernels, one `.s` file each, selected by the sweep to win on different shape classes: they differ in the
compute path (MFMA or VALU), the number of waves, the wide tile width (`tn`), the prefetch depth and the grid
mapping, and no single variant is fastest everywhere. The shard lists, per shape, which of the four is used.
Macrotile, threads, `kStep`, grid and predicates are in the `custom.config` at the end of each `.s`, which is the
source of truth. The table gives the K bound and the tensor-bytes class of each kernel (see "Size bounds");
`kStep` is 256 where not stated.

| Idx | Test YAML (`Tests/custom/`) | `.s` file prefix | Notes |
| --- | --------------------------- | ---------------- | ----- |
| 0 | `custom_skinny_mxfp8_tn_sel0.yaml` | `gemm_BM16_TN1_..._WAVES1_..._PF6_` | MFMA, 1 wave, 1 wide tile, 3-D grid; K <= 2^31 - 256; tensor bytes < 2^31 |
| 1 | `custom_skinny_mxfp8_tn_sel1.yaml` | `gemm_BM16_TN1_..._WAVES2_..._PF3_` | MFMA, 2 waves, 1 wide tile, flat persistent grid; K <= 2^30; tensor bytes < 2^31 |
| 2 | `custom_skinny_mxfp8_tn_sel2.yaml` | `gemm_BM1_TN1_LANES_K16_..._PF2_` | VALU, 3-D grid, `kStep` 1024; K <= 2^31 - 1024; tensor bytes < 2^32 |
| 3 | `custom_skinny_mxfp8_tn_sel3.yaml` | `gemm_BM16_TN2_..._WAVES1_..._PF4_` | MFMA, 1 wave, 2 wide tiles, flat persistent grid; K <= 2^30; tensor bytes < 2^31 |

## Numerics

The f32 result is converted to bf16 to nearest even (`v_cvt_pk_bf16_f32`, which every one of the four `.s` files
uses): D is the bf16 of `fma(beta, C, alpha * acc)` with alpha applied in f32 before the rounding. With beta zero
(the bit test, so -0.0 counts as zero) C is never read. The f32 accumulation order is that of the kernel (K is
reduced across lanes, waves and MFMA steps), so D is not claimed to be bit-identical to other kernels. No ulp
measurement of the current kernels is recorded here.

## Metadata contract

The host does not parse the kernel symbol; everything it needs is in `custom.config`:

- `macrotile: [wide, skinny, depth]`, oriented along `max(m, n)` and `min(m, n)`, and `threads`: tile counts are
  `ceil(wide extent / macrotile[0])` and `ceil(skinny extent / macrotile[1])`.
- `grid: [TilesWide, TilesSkinny, Batch]`, or the flat `[TilesWideSkinnyBatch, One, One]` for the `GRID_3D0`
  kernels (sel1 and sel3), whose grid is `tilesWide * tilesSkinny * batch` along x. `TilesWide`, `TilesSkinny` and
  `TilesWideSkinnyBatch` may only be combined with `One` and `Batch`. `NumWorkGroupsX/Y/Z` kernargs report the
  grid as launched.
- `kStep`: K elements consumed by one k-loop iteration (256 for the MFMA kernels, 1024 for sel2).
  It must be a multiple of the MX block and the host rounds `Spk` up to it. It only sets `Spk` and the client
  scale gate; it does not constrain K, because the kernels run a K tail behind a run-time branch.
- `Spk` kernarg: K rounded up to `kStep` (at least one step). This is not split-K: the kernel reduces all of K for
  an output in one chunk.
- `noSplitK: true`: no split-K workspace and no uniform-summation-order launch checks. Without it, handwritten
  custom kernels are refused when uniform summation order is on (the default).
- K predicates: `AssertSummationElementMultiple: 32` on every kernel, the MX scale block (also what the generic
  MXFP8 kernels require). The kernels handle any such K, whole `kStep` steps or not, so K is not tied to `kStep`;
  a K that is not a multiple of 32 runs the generic kernels. `AssertSizeGreaterThan: {3: 0}` rejects K = 0, which
  the kernels have no path for.
- Bias, activation and the scale-alpha vector are off in `custom.config` `Features` (`SupportsBias`,
  `SupportsActivation`, `SupportsScaleAlpha`), so each solution gets its own `ProblemType` with those off and rejects such problems even
  though the shard's shared `ProblemType` has them. They fall through to the generic kernels.
  `MXScaleFormat: HostPreSwizzle` tells the client which scale layout to generate.

### Kernarg layout

120 bytes (`kernarg_segment_size` in the `.s` metadata). C and D are separate pointers.

| Offset | Name | Semantic | Meaning |
| ------ | ---- | -------- | ------- |
| 0 | D | `AddressD` | Output pointer. |
| 8 | A | `AddressA` | A pointer. |
| 16 | B | `AddressB` | B pointer. |
| 24 | scalesA | `AddressMXScaleA` | A scales pointer. |
| 32 | scalesB | `AddressMXScaleB` | B scales pointer. |
| 40 | ws | `ConstantZero` | Passed as 0 (`uint64`). The kernels store to a split-K workspace only when `k > spk`, checked at run time; the host always passes `spk >= k`, so that path is never taken and no workspace is needed. |
| 48 | C | `AddressC` | C pointer. |
| 56 | ldD | `StrideD0Bytes` | D leading stride in bytes. |
| 60 | ldA | `StrideA0Bytes` | A leading stride in bytes. |
| 64 | ldB | `StrideB0Bytes` | B leading stride in bytes. |
| 68 | m | `SizeFree0` | m. |
| 72 | n | `SizeFree1` | n. |
| 76 | k | `SizeSum` | K. |
| 80 | spk | `Spk` | K rounded up to `kStep`. |
| 84 | ldC | `StrideC0Bytes` | C leading stride in bytes. |
| 88 | alpha | `Alpha` | f32. |
| 92 | beta | `Beta` | f32. |
| 96 | numWorkGroupsX | `NumWorkGroupsX` | Launched grid. |
| 100 | numWorkGroupsY | `NumWorkGroupsY` | Launched grid. |
| 104 | numWorkGroupsZ | `NumWorkGroupsZ` | Launched grid. |
| 108 | wholeTiles | `ConstantZero` | `int32`, passed as 0. Part of the remainder narrowing of the 3-D MFMA kernels with a power-of-two `tn`; ignored when `pieces` is below 2. Only sel0 loads offsets 108 and 112 (checked in the `.s` files); the other three kernels ignore them. |
| 112 | pieces | `ConstantZero` | `int32`, passed as 0, which disables the remainder narrowing. Only sel0 loads it; the other three kernels ignore it. |

Leading strides in kernargs are **byte** counts. `generateCustomCall` computes them exactly in integers
(`multiplyElementSize(ld, elementBytes())`), because a floating-point `ld * elementBytes()` rounds to 24 bits and
corrupted leading dimensions above 2^24 (for example a bf16 ldD of 16777219). Solution predicates (`AssertStride*`)
use **element** leading dimensions.

### Size bounds

- `AssertTensorBytesLessThan` (the C++ `TensorBytesLessThan` predicate) requires every non-empty A, B, C and D
  to have `totalAllocatedBytes()` below the value: 2^31 (2147483648) for the MFMA kernels (sel0, sel1, sel3) and
  2^32 (4294967296) for the VALU kernel (sel2). The kernels compute each tensor's 32-bit buffer
  `num_records` with 32-bit scalar arithmetic (A `(m-1)*ldABytes + roundup4(K)`, B `(n-1)*ldBBytes + roundup4(K)`,
  D `(n-1)*ldDBytes + 2*m`, C `(n-1)*ldCBytes + 2*m`), which with K a multiple of 32 equals the descriptor
  extent and wraps at 4 GiB (for example 1048576x1x4096 has A of exactly 4 GiB and is rejected). The MFMA kernels
  also use the byte offset 0x80000000 as the out-of-range sentinel, so the emitter requires every buffer to be
  below 2 GiB; the VALU sentinel is 0xFFFFFFFF, which allows 4 GiB. The scale extents are bounded by those of A
  and B, so they need no separate guard. The split-K workspace is not used (`noSplitK`, `ws` is `ConstantZero`),
  so its size, which the emitter also bounds for MFMA, does not enter. The predicate belongs to the solution, so
  it also applies to forced solution indices and to tuning.
- `AssertSizeLessThan: {0: 1048577, 1: 1048577}` caps m and n at 1048576 on every kernel. It is a conservative
  range cap, not the overflow guard.
- `AssertSizeLessThan` also bounds K on every kernel, because they do signed 32-bit arithmetic on it. sel1 and sel3
  (`GRID_3D0`) compute the K-split count as `(K + spk - 1) / spk` in signed 32 bits, so
  they need K <= 2^30 (`3: 1073741825`) to keep `K + Spk - 1` below 2^31. The VALU kernel reads K and Spk as
  signed 32-bit values (arithmetic shift, `s_min_i32`) and the host rounds Spk up to `kStep`, so Spk must stay
  below 2^31: sel2 (kStep 1024) needs K <= 2^31 - 1024 (`3: 2147482625`). sel0 (kStep 256) needs K <= 2^31 - 256 (`3: 2147483393`) for the same
  reason: with K above it the host rounds Spk to 2^31, which the kernel reads as a negative number and skips the
  tail masking. sel1 and sel3 (kStep 256, K <= 2^30) and sel2 (kStep 1024, K <= 2^31 - 1024) keep Spk below 2^31.
- `AssertSizeGreaterThan: {0: 0, 1: 0, 3: 0}` rejects empty problems.

Scales use the `HostPreSwizzle` layout with padding: K blocks padded to a multiple of 8 and the free extent to a
multiple of 32.

## hipBLASLt device library

The shard
`gfx950/gfx950/Equality/gfx950_Cijk_Alik_Bljk_F8BS_MXAE8B32_MXBE8B32_BH_Bias_HA_S_SAB_SAV_UserArgs_SkinnyGEMM.yaml`
has `LibraryType: Equality` and lists the shapes routed to the four kernels in `ExactLogic`, with `SolutionIndex`
equal to the index in the table above (sel0 to sel3).

Routing: the Equality library is keyed by `(batch, K, m, n)` only, so a problem routes to a listed solution when
its `(m, n, batch, k)` is in `ExactLogic` and the solution's predicates pass. The leading-dimension entries of the
`ExactLogic` keys are ignored for routing; padded ldc/ldd, and lda/ldb that satisfy
`AssertStrideA/BMultiple: {1: 4}`, still route.

Each kernel serves a skinny extent on n (m wide) and the mirrored skinny extent on m (n wide), with
`wide = max(m, n)` and `skinny = min(m, n)`. The shard has 142 keys: 71 shapes in both orientations (skinny extents
1 to 4), with per-orientation kernel choices; the keys per kernel (sel0 to sel3) are 30, 63, 14 and 35. The sweep
selection had a fifth kernel (BM2, TN4, VALU), which won no shape in the benchmark run and was removed.
The 71 shapes are the 74 sweep shapes minus 128256x{1,2,4}x8192, and the keys come from one TensileLite benchmark
run of all 148 sizes (the 74 shapes and their mirrors). Pruning rule: a shape is dropped in
both orientations when, in either orientation, the first non-skinny heuristic solution (what hipBLASLt runs
without the shard) is strictly faster than the skinny kernel (measured with `hipblaslt-bench`; see the shard
header); keep such shapes out when regenerating `ExactLogic`, which comes from TensileLite benchmarking (the
regeneration scripts are offline developer helpers, not part of the tree). On the kept shapes the generic kernel
takes 1.02 to 2.79 times (median 1.59) the time of the skinny kernel; the shard header has the measurements. The
shape llama3-8b-qkv_proj 6144x2x4096 is kept and served by sel3.

Everything else keeps the Origami kernels, in particular bias, activation, the scale-alpha vector, K that the
predicates reject (not a multiple of 32), sizes above the bounds above, and
any shape not listed in `ExactLogic`.

### Building

Build the skinny shard **and** the Origami MXFP8 fallbacks (required for `matmul_skinny_mxfp8_tn_fallback*`
gtests and out-of-box shapes). Do **not** use `*SkinnyGEMM*` alone: that produces a tiny device library with only
the four custom solutions and no fallback kernels. `-f 'gfx950/Equality/*'` also picks the shard up.

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
  --repository "TensileLite C++ skinny GEMM emitter (sweep selection)" \
  --version 1.0.0
```
