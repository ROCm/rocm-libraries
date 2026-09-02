---
id: hw-fp8-wmma-gfx1250
title: "gfx1250 fp8/bf8 WMMA (NVFP4 analog)"
type: hardware
architectures: [gfx1250]
architecture_families: [gfx12]
tags: [fp8, wmma, block-scale, gfx1250]
hardware_features: [fp8, block-scale, wmma]
confidence: verified
related:
  - hw-wmma-gfx1250
  - technique-gfx1250-block-scale
  - kernel-block-scale-gemm-gfx1250
sources: [project-rocke]
aliases: [fp8-wmma]
---

# fp8 / bf8 WMMA K=64

Analog of KernelWiki NVFP4 / block-scaled MMA: narrow types with an explicit
scale path. gfx1250 catalog ships **unscaled** fp8/bf8 WMMA 16×16×64
(`wmma_gfx1250_f32_16x16x64_{fp8,bf8}_*`). A/B fragments are `<8 x i32>`
(32 packed bytes/lane). Format/reuse immediates pinned to 0 in
`Gfx1250Backend._emit_wmma_fp8`.

**Not catalogued** (do not emit): block-scaled F4, block16 converts, IU4,
SWMMAC sparse. `has_tdm=false`.

Block-scaled expert GEMM: `instances/gfx1250/block_scaled_gemm.py`
(`BlockScaledGemmSpec`, default `block_k=128`). Scales are fp16/fp32 tensors
in that ABI — not MX UE8M0 and not NVFP4 per-16 E2M1. Preserve the scale
dtype and granularity; do not rename encodings.
