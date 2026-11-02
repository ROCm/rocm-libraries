<!--
Copyright Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# Skinny MXFP8 custom kernels

Skinny GEMM kernels for MXFP8 (FP8 E4M3 A and B, E8M0 scales per 32 elements, BF16 D), hipBLASLt TN, with one
small ("skinny") extent and the other wide. They serve LLM decode and wider skinny batches up to the kernel `bm`.

gfx950 only. Six kernels, one `.s` file each. Each file carries one `custom.config` and one Tensile test YAML under
`Tests/custom/custom_skinny_mxfp8_tn_<variant>.yaml`.

Each kernel computes `D = alpha * (A^T B) + beta * C` for any alpha and beta in f32.

## Kernels

Six kernels, one `.s` file each, selected by the sweep to win on different shape classes: they differ in the
number of waves, the wide tile width (`tn`), the prefetch depth and the grid mapping, and no single variant is
fastest everywhere. The shard lists, per shape, which of the six is used. Macrotile, threads, `kStep`, grid
and predicates are in the `custom.config` at the end of each `.s`, which is the source of truth.

| Idx | Test YAML (`Tests/custom/`) | `.s` file prefix | Notes |
| --- | --------------------------- | ---------------- | ----- |
| 0 | `custom_skinny_mxfp8_tn_sel0.yaml` | `gemm_BM16_TN1_..._WAVES4_..._PF2_` | MFMA, 4 waves |
| 1 | `custom_skinny_mxfp8_tn_sel1.yaml` | `gemm_BM16_TN1_..._WAVES2_..._PF4_` | MFMA, 2 waves |
| 2 | `custom_skinny_mxfp8_tn_sel2.yaml` | `gemm_BM16_TN3_..._WAVES1_..._PF6_` | MFMA, 1 wave, 3 wide tiles |
| 3 | `custom_skinny_mxfp8_tn_sel3.yaml` | `gemm_BM16_TN2_..._WAVES1_..._PF4_` | MFMA, 1 wave, 2 wide tiles |
| 4 | `custom_skinny_mxfp8_tn_sel4.yaml` | `gemm_BM2_TN2_LANES_K16_..._PF1_` | VALU, `kStep` 1024 |
| 5 | `custom_skinny_mxfp8_tn_sel5.yaml` | `gemm_BM16_TN2_..._WAVES2_..._PF3_..._GRID_3D0_` | MFMA, flat 1-D grid |

## Numerics

The f32 result is converted to bf16 by truncation toward zero, not round-to-nearest-even (the epilogue shifts the
f32 bits right by 16). D can therefore differ from the generic kernels by up to 1 bf16 ulp (measured with
`hipblaslt-bench --ulp`: max 0.5 ulp and average 0.25 ulp, versus 0 for the generic kernels). A quiet NaN stays NaN,
but an f32 NaN whose payload lies only in the low 16 mantissa bits becomes +/-Inf after truncation.

## Metadata contract

The host does not parse the kernel symbol; everything it needs is in `custom.config`:

- `macrotile: [wide, skinny, depth]`, oriented along `max(m, n)` and `min(m, n)`, and `threads`: tile counts are
  `ceil(wide extent / macrotile[0])` and `ceil(skinny extent / macrotile[1])`.
- `grid: [TilesWide, TilesSkinny, Batch]`, or the flat `[TilesWideSkinnyBatch, One, One]` for the `GRID_3D0`
  kernel, whose grid is `tilesWide * tilesSkinny * batch` along x. `TilesWide`, `TilesSkinny` and
  `TilesWideSkinnyBatch` may only be combined with `One` and `Batch`. `NumWorkGroupsX/Y/Z` kernargs report the
  grid as launched.
- `kStep`: K elements consumed by one k-loop iteration (256, and 1024 for the K16 kernel). It must be a multiple
  of the MX block and divide the admitted K multiple, because the k loops have no tail path.
- `Spk` kernarg: K rounded up to `kStep` (at least one step). This is not split-K: the kernel reduces all of K for
  an output in one chunk.
- `noSplitK: true`: no split-K workspace and no uniform-summation-order launch checks. Without it, handwritten
  custom kernels are refused when uniform summation order is on (the default).
- K predicates: `AssertSummationElementMultiple: 256` on every kernel, plus `AssertSizeMultiple: {3: 1024}` on the
  K16 kernel, so the admitted K multiple equals `kStep`; `AssertSizeGreaterThan: {3: 0}` rejects K = 0, which the
  kernels have no path for.
- Bias, activation and the scale-alpha vector are off in `custom.config` `Features` (`SupportsBias`,
  `SupportsActivation`, `SupportsScaleAlpha`), so each solution gets its own `ProblemType` with those off and rejects such problems even
  though the shard's shared `ProblemType` has them. They fall through to the generic kernels.
  `MXScaleFormat: HostPreSwizzle` tells the client which scale layout to generate.

### Kernarg layout

112 bytes (`kernarg_segment_size` in the `.s` metadata). C and D are separate pointers.

| Offset | Name | Semantic | Meaning |
| ------ | ---- | -------- | ------- |
| 0 | D | `AddressD` | Output pointer. |
| 8 | A | `AddressA` | A pointer. |
| 16 | B | `AddressB` | B pointer. |
| 24 | scalesA | `AddressMXScaleA` | A scales pointer. |
| 32 | scalesB | `AddressMXScaleB` | B scales pointer. |
| 40 | ws | `ConstantZero` | Passed as 0. The kernels have a split-K workspace store path guarded by `k > spk`; the host always passes `spk >= k`, so it is never taken and no workspace is needed. |
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

Leading strides in kernargs are **byte** counts. `generateCustomCall` computes them exactly in integers
(`multiplyElementSize(ld, elementBytes())`), because a floating-point `ld * elementBytes()` rounds to 24 bits and
corrupted leading dimensions above 2^24 (for example a bf16 ldD of 16777219). Solution predicates (`AssertStride*`)
use **element** leading dimensions.

### Size bounds

- `AssertTensorBytesLessThan: 4294967296` (the C++ `TensorBytesLessThan` predicate) requires every non-empty A, B, C
  and D to have `totalAllocatedBytes() < 2^32`. The kernels compute each tensor's 32-bit buffer `num_records` with
  32-bit scalar arithmetic (A `(m-1)*ldABytes + roundup4(K)`, B `(n-1)*ldBBytes + roundup4(K)`, D
  `(n-1)*ldDBytes + 2*m`, C `(n-1)*ldCBytes + 2*m`), which with K a multiple of 256 equals the descriptor extent
  and wraps at 4 GiB (for example 1048576x1x4096 has A of exactly 4 GiB and gave wrong results). The scale extents
  are bounded by those of A and B, so they need no separate guard. It is a solution predicate, so it also applies
  to forced solution indices and to tuning.
- `AssertSizeLessThan: {0: 1048577, 1: 1048577}` caps m and n at 1048576, the largest wide extent that was
  validated. It is not the overflow guard.
- `AssertSizeLessThan` also bounds K on the two kernels that do signed 32-bit arithmetic on it. The K16 kernel
  (`LANES_K16`, kStep 1024) reads K and Spk as signed 32-bit (arithmetic shift by 10, `s_min_i32`): K = 2^31 - 256
  already gives Spk = 2^31, which is negative, so K must not exceed 2^31 - 1024 (`3: 2147482625`). The `GRID_3D0`
  kernel computes the K-split count as `(K + spk - 1) / spk` in signed 32 bits, so for K a multiple of 256 it needs
  K <= 2^30 (`3: 1073741825`) to keep `K + Spk - 1` below 2^31. The other four kernels have no K upper bound beyond
  the tensor-bytes guard.
- `AssertSizeGreaterThan: {0: 0, 1: 0, 3: 0}` rejects empty problems.

Scales use the `HostPreSwizzle` layout with padding: K blocks padded to a multiple of 8 and the free extent to a
multiple of 32.

## hipBLASLt device library

The shard
`gfx950/gfx950/Equality/gfx950_Cijk_Alik_Bljk_F8BS_MXAE8B32_MXBE8B32_BH_Bias_HA_S_SAB_SAV_UserArgs_SkinnyGEMM.yaml`
has `LibraryType: Equality` and lists the shapes routed to the six kernels in `ExactLogic`.

Routing: the Equality library is keyed by `(batch, K, m, n)` only, so a problem routes to a listed solution when
its `(m, n, batch, k)` is in `ExactLogic` and the solution's predicates pass. The leading-dimension entries of the
`ExactLogic` keys are ignored for routing; padded ldc/ldd, and lda/ldb that satisfy
`AssertStrideA/BMultiple: {1: 4}`, still route.

Each kernel serves a skinny extent on n (m wide) and the mirrored skinny extent on m (n wide), with
`wide = max(m, n)` and `skinny = min(m, n)`. The shard lists both orientations (skinny extents 1 to 4), with
per-orientation kernel choices. Shapes where the generic kernels are faster are left out in both orientations
(see the shard header); keep them out when regenerating `ExactLogic`, which comes from TensileLite benchmarking
(the regeneration scripts are offline developer helpers, not part of the tree).

Everything else keeps the Origami kernels, in particular bias, activation, the scale-alpha vector, K that is not a
multiple of the kernel `kStep`, sizes above the bounds above, and any shape not listed in `ExactLogic`.

### Building

Build the skinny shard **and** the Origami MXFP8 fallbacks (required for `matmul_skinny_mxfp8_tn_fallback*`
gtests and out-of-box shapes). Do **not** use `*SkinnyGEMM*` alone: that produces a tiny device library with only
the six custom solutions and no fallback kernels. `-f 'gfx950/Equality/*'` also picks the shard up.

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
  --yaml tensilelite/Tests/custom/<matching>.yaml \
  --origin skinny_mxfp8 \
  --repository "TensileLite C++ skinny GEMM emitter (sweep selection)" \
  --version 1.0.0
```
