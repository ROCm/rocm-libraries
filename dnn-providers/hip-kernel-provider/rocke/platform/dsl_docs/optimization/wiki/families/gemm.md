---
id: family-gemm
title: "GEMM family — common tech × architecture"
type: family
operator_families: [gemm]
architecture_families: [cdna, rdna, gfx12]
tags: [gemm, routing]
related:
  - family-overview
  - kernel-gemm-universal
  - technique-tiling
  - technique-software-pipeline
  - technique-mfma-atom
  - technique-wmma-atom
  - technique-epilogue
  - process-escape-hatch
  - kernel-wmma-gemm-gfx1250
  - migration-gfx950-to-gfx1250
sources:
  - project-rocke
  - project-hipblaslt
  - project-tensile
  - project-tensilelite
  - project-origami
  - project-stinkytofu
  - project-rocroller
---

# GEMM

Rocke specs: `UniversalGemmSpec` / `TileSpec` / `TraitSpec` in
`platform/python/rocke/instances/common/gemm_universal.py`. Related:
`batched_gemm`, `grouped_gemm`, `flatmm`, `streamk_gemm`, `block_scale_gemm`,
`mx_gemm`, `gemm_multi_d`, `gemm_multi_abd`.

Upstream generators that implement the same lever space: hipBLASLt +
TensileLite (`projects/hipblaslt/tensilelite`), Tensile (`shared/tensile`),
Origami tile selection (`shared/origami`), rocRoller (`shared/rocroller`),
stinkytofu scheduling (`shared/stinkytofu`, gfx1250 asm).

## Common levers (all gfx that compile the instance)

| Lever | Rocke field | When to change | Probe |
|---|---|---|---|
| Macro tile | `tile_m/n/k` | more reuse vs LDS/VGPR | `probe_occupancy.py` |
| Warp grid | `warp_m/n` | occupancy vs ILP | occupancy + ISA histogram |
| Pipeline | `TraitSpec.pipeline`: `mem` / `compv3` / `compv4` | hide DRAM; `compv4` needs async LDS | ISA `s_waitcnt` / async DMA count |
| Scheduler | `intrawave` / `interwave` | producer/consumer split | mainloop ISA |
| Epilogue | `default` / `cshuffle` | contiguous acc → direct; scattered → LDS shuffle | store opcode width |
| Persistent / Stream-K | `persistent`, `StreamKGemmSpec` | tail / skinny K | grid occupancy |
| Chiplet swizzle | `chiplet_swizzle` | multi-XCD CDNA | — |
| Quant | `block_scale_gemm` / `mx_gemm` | fp8/mx paths | intrinsic counts |

## Architecture columns

| Lever | gfx942 | gfx950 | gfx1151 / gfx1201 | gfx1250 |
|---|---|---|---|---|
| f16 atom | MFMA 16×16×16, 32×32×8 | + K-pack 16×16×32, 32×32×16 | WMMA 16×16×16 wave32 | WMMA 16×16×32 |
| `compv4` + cshuffle | yes | yes | **no** — `mem`+`default` only | WMMA builders, not gfx9 pipeline |
| fp8 MFMA/WMMA | native fp8 MFMA | fp8 + fp4/fp6 + MX | iu8/iu4 WMMA (gfx1151) | fp8 WMMA K=64 |
| Async LDS | yes | yes | no | GFX12 async-to-LDS |
| Support matrix | `instances/SUPPORT_MATRIX.md` | same | `gemm_multi_d` rejected (needs cshuffle) | `instances/gfx1250/wmma_gemm.py` |

## Sources to read with a GEMM gap

- TensileLite kernel generation and prefetch/LDS options — `project-tensilelite`
- hipBLASLt heuristic / solution map — `project-hipblaslt`
- Origami analytical tile pick — `project-origami`
- stinkytofu waitcnt / DAG schedule (gfx1250 asm) — `project-stinkytofu`

If those sources and this table cannot move the probe signature:
`process-escape-hatch`.
