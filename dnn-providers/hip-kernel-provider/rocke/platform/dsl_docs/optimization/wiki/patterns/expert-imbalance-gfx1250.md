---
id: pattern-expert-imbalance-gfx1250
title: "MoE expert tile imbalance on gfx1250"
type: pattern
tags: [moe, gfx1250]
architectures: [gfx1250]
architecture_families: [gfx12]
operator_families: [moe]
symptoms: [expert-imbalance, low-cu-util, tail-grid]
candidate_techniques:
  - technique-gfx1250-tile-schedule
  - technique-fusion
  - technique-gfx1250-block-scale
related:
  - family-moe
  - kernel-fused-moe
  - kernel-block-scale-gemm-gfx1250
sources: [project-rocke]
---

# Expert imbalance

Analog of KernelWiki MoE load-imbalance. Routed row counts differ; static
grouped GEMM leaves some CTAs idle. gfx1250 path is
`fused_moe_mega_wmma.py` / fp8 block-scale, not hipBLASLt CLC.

Pack valid tokens, fuse SiLU-mul, then a persistent walk over remaining
expert tiles. Do not pad every expert to the max row count and call it
fixed.
