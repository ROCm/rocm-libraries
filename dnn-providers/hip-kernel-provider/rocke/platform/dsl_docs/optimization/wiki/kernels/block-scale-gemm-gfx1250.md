---
id: kernel-block-scale-gemm-gfx1250
title: "gfx1250 block-scaled fp8 GEMM"
type: kernel
architectures: [gfx1250]
architecture_families: [gfx12]
operator_families: [gemm, moe]
tags: [gemm, fp8, block-scale, gfx1250]
confidence: source-reported
reproducibility: snippet
kernel_types: [block-scale-gemm, gemm]
languages: [rocke]
rocke_primitive: "instances/gfx1250/block_scaled_gemm.py"
related:
  - technique-gfx1250-block-scale
  - hw-fp8-wmma-gfx1250
  - family-moe
sources: [project-rocke]
---

# Block-scaled fp8 GEMM (gfx1250)

`BlockScaledGemmSpec` — RCR, WMMA K=64, `block_k=128`, fp8/bf8 A/B, fp16/bf16
C, fp16/fp32 scales. Example: `examples/gfx1250/gemm/block_scaled_gemm_verify.py`.
Fused MoE fp8 sibling: `instances/gfx1250/fused_moe_fp8.py` /
`fused_moe_mega_wmma.py`.
