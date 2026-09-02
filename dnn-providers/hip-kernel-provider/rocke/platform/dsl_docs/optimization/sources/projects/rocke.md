---
id: project-rocke
title: "rocke (this tree)"
type: project
repo: ROCm/rocm-libraries
tree: dnn-providers/hip-kernel-provider/rocke
tags: [rocke]
operator_families: [gemm, attention, convolution, moe, small-ops]
architecture_families: [cdna, rdna, gfx12]
related: [family-overview]
kernel_types: [gemm, attention, convolution, moe]
---

# rocke

Python+C++ CK-Tile-style authoring for AMDGPU. Instances under
`platform/python/rocke/instances/`, attention product under `library/`,
hardware SSOT `platform/python/rocke/core/arch/data/arch_specs.json`.
This wiki lives in `platform/dsl_docs/optimization/`.
