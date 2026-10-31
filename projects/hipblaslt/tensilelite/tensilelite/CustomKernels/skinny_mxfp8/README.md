<!--
Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# Skinny MXFP8 custom kernels

Skinny GEMM kernels for MXFP8 (FP8 E4M3 A and B, E8M0 scales per 32 elements, BF16 D), hipBLASLt TN, with one
small ("skinny") extent and the other wide. They serve LLM decode and wider skinny batches up to the kernel `bm`.

gfx950 only. Six kernels, one `.s` file each. Each file carries one `custom.config` and one Tensile test YAML under
`Tests/custom/custom_skinny_mxfp8_tn_<variant>.yaml`.

Each kernel computes `D = bf16(alpha * (A^T B) + beta * C)` for any alpha and beta.

## Kernels (SolutionIndex order in Range logic)

| Idx | YAML | Emitter symbol | bm | tn | lanesK | waves | macrotile | threads |
| --- | --- | --- | -- | -- | ------ | ----- | --------- | ------- |
| 0 | `custom_skinny_mxfp8_tn_bm16_grid3d.yaml` | `gemm_BM16_TN1_..._PF6_..._GRID_3D1_...` | 16 | 1 | 64 | 1 | [16, 1, 4096] | 64 |
| 1 | `custom_skinny_mxfp8_tn_bm16_persistent.yaml` | `gemm_BM16_TN1_...` (persistent) | 16 | 1 | 64 | 2 | [16, 1, 4096] | 128 |
| 2 | `custom_skinny_mxfp8_tn_bm32_persistent.yaml` | `gemm_BM32_TN2_..._PF3_...` (persistent) | 32 | 2 | 64 | 2 | [32, 2, 4096] | 128 |
| 3 | `custom_skinny_mxfp8_tn_bm32_grid3d.yaml` | `gemm_BM32_TN4_..._PF3_...` (grid3d) | 32 | 4 | 64 | 1 | [32, 4, 4096] | 64 |
| 4 | `custom_skinny_mxfp8_tn_bm16_tn2_grid3d0.yaml` | `gemm_BM16_TN2_..._GRID_3D0_...` | 16 | 2 | 64 | 1 | [16, 2, 4096] | 64 |
| 5 | `custom_skinny_mxfp8_tn_bm2_tn4.yaml` | `gemm_BM2_TN4_..._WAVES4_...` | 2 | 4 | 64 | 4 | [4, 4, 4096] | 256 |

Macrotile and threads match each kernel's `custom.config` and its Tensile test YAML under `Tests/custom/`.

ExactLogic in the shipped Range shard comes from **TensileLite benchmarking** on the
selection shape grid (`LibraryType: Range`). To regenerate logic locally, use the offline
helpers under `~/mxfp8-helpers` (see `~/mxfp8-helpers/README.md`); those scripts and
YAMLs are not part of the hipBLASLt tree.

## hipBLASLt device library

MX scale + bias routing uses the Range shard
`gfx950/gfx950/Range/gfx950_Cijk_Alik_Bljk_F8BS_MXAE8B32_MXBE8B32_BH_Bias_HA_S_SAB_SAV_UserArgs_SkinnyGEMM.yaml`.

Build the skinny Range shard **and** Origami MXFP8 fallbacks (required for
`matmul_skinny_mxfp8_tn_fallback*` gtests and out-of-box shapes). Do **not**
use `*SkinnyGEMM*` alone — that produces a tiny device library with only the six
custom solutions and no fallback kernels.

```bash
cd projects/hipblaslt
invoke build -ca gfx950 -f '*F8BS_MXAE8B32_MXBE8B32*' -c
export HIPBLASLT_TENSILE_LIBPATH=$PWD/build/release/Tensile/library/gfx950
```

## Tensile metadata

```bash
source ~/.tensile/bin/activate
cd projects/hipblaslt/tensilelite
python -m tensilelite.AddCustomConfig \
  tensilelite/CustomKernels/skinny_mxfp8/<kernel>.s \
  --yaml tensilelite/Tests/custom/<matching>.yaml \
  --origin skinny_mxfp8 \
  --repository "TensileLite C++ skinny GEMM emitter (local emission)" \
  --version 1.0.0
```

Leading matrix strides in kernargs are **byte** counts (emitter contract). Logic and
`custom.config` use `StrideA0Bytes` / `StrideB0Bytes` / `StrideC0Bytes` /
`StrideD0Bytes`; `generateCustomCall` fills them as `ld * elementBytes()`.
Solution predicates (`AssertStride*`) still use **element** leading dimensions.
