---
id: kernel-attention-2d
title: "Tiled attention 2D (rocke)"
type: kernel
architectures: [gfx942, gfx950, gfx1151]
architecture_families: [cdna, rdna]
operator_families: [attention]
tags: [attention]
confidence: verified
reproducibility: snippet
kernel_types: [attention, fmha]
languages: [rocke]
rocke_primitive: "library/kernels/*/attention_tiled_2d.py"
related: [family-attention, technique-mfma-atom, technique-software-pipeline]
sources: [project-rocke, project-composablekernel]
---

# Tiled attention 2D

Prefill / window path. Geometry knobs `num_warps`, `tile_size`,
`block_m_per_warp`; CDNA atom flags `use_mfma_32x32`,
`use_transposed_qk_32x32`; default-off interleave and buffer flags belong in
Step 0. Scalar `UnifiedAttention2DSpec` is the oracle, not the hero.

3D split-KV is the decode sibling. FMHA CK-Tile 01 ports sit in
`instances/common/fmha_*.py`.
