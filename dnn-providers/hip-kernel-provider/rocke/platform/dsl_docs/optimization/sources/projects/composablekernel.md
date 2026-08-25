---
id: project-composablekernel
title: "Composable Kernel"
type: project
repo: ROCm/rocm-libraries
tree: projects/composablekernel
tags: [composablekernel]
operator_families: [gemm, attention, convolution, moe, small-ops]
architecture_families: [cdna, rdna]
related: [project-rocke, family-overview]
kernel_types: [gemm, attention, convolution, moe]
---

# Composable Kernel

CK Tile C++ templates and examples that rocke instances mirror (universal GEMM
dispatcher schema, FMHA 01, fused MoE 15, grouped conv). Rocke’s
`UniversalGemmSpec` is explicitly the DSL counterpart of
`dispatcher/codegen/unified_gemm_codegen.py`.

Read `projects/composablekernel` on `develop`. Prefer rocke helpers when the
lever already exists in the DSL; use CK when inventing a body that CK already
ships.
