---
id: kernel-fused-moe
title: "Fused MoE (rocke)"
type: kernel
architectures: [gfx942, gfx950, gfx1250]
architecture_families: [cdna, gfx12]
operator_families: [moe]
tags: [moe]
confidence: source-reported
reproducibility: snippet
kernel_types: [moe, grouped-gemm]
languages: [rocke]
rocke_primitive: "instances/common/fused_moe.py"
related: [family-moe, family-gemm, technique-fusion]
sources: [project-rocke, project-hipblaslt]
---

# Fused MoE

Launcher stitches topk → sort → smoothquant → expert GEMMs → SiLU-mul → reduce.
Optimize the grouped GEMM with `family-gemm`, then fuse the edges. gfx1250 uses
`fused_moe_mega_wmma.py`. hipBLASLt grouped GEMM is the library counterpart for
the expert tiles.
