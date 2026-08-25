---
id: family-moe
title: "MoE family — common tech × architecture"
type: family
operator_families: [moe]
architecture_families: [cdna, rdna, gfx12]
tags: [moe, routing]
related:
  - family-overview
  - kernel-fused-moe
  - technique-fusion
  - technique-tiling
  - family-gemm
  - process-escape-hatch
sources:
  - project-rocke
  - project-hipblaslt
  - project-composablekernel
---

# MoE

Rocke composition (CK Tile 15-style): `topk_softmax` → `moe_sorting` →
`moe_smoothquant` → grouped / block-scale GEMM ×2 with SiLU-mul → reduce.
Launchers: `fused_moe.py`, `moe_fused_mega.py`, `moe_fused_mega_fp8.py`,
`fused_moe_e2e.py`. gfx1250: `instances/gfx1250/fused_moe_mega_wmma.py`.

Grouped GEMM is the compute heart — reuse `family-gemm` levers on the expert
tiles, then fuse the surrounding gather/activation/reduce.

## Common levers

| Lever | Where | Notes |
|---|---|---|
| Grouped vs ragged dispatch | `grouped_gemm` / fused mega | token packing vs padding |
| Expert GEMM tile | same as GEMM `TileSpec` | LDS/VGPR vs expert N,K |
| Block-scale / fp8 | `block_scale_gemm`, `moe_fused_mega_fp8` | CDNA fp8 MFMA; gfx1250 fp8 WMMA |
| Sorting pipeline | histogram / scan / scatter / persistent | three-kernel vs fused |
| Activation fusion | SiLU-mul in epilogue | cuts a round trip |
| Prefill vs decode token counts | dispatcher | occupancy vs skinny-M GEMM (`flatmm`) |

## Architecture columns

| | gfx942 / gfx950 | gfx1151 | gfx1250 |
|---|---|---|---|
| Sorting / topk / smoothquant | yes (no MMA) | yes | yes |
| Block-scale GEMM | MFMA | rejected in support matrix | WMMA fused-moe mega |
| hipBLASLt grouped GEMM | solution library | — | TensileLite + stinkytofu schedule |

If grouped-GEMM levers and fusion cannot move the limiter: `process-escape-hatch`
(new dispatch/fusion mapping, not another expert tile).
