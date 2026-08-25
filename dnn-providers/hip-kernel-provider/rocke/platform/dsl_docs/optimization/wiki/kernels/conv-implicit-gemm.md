---
id: kernel-conv-implicit-gemm
title: "Implicit-GEMM convolution (rocke)"
type: kernel
architectures: [gfx942, gfx950, gfx1151]
architecture_families: [cdna, rdna]
operator_families: [convolution]
tags: [convolution]
confidence: verified
reproducibility: snippet
kernel_types: [convolution, implicit-gemm-conv]
languages: [rocke]
rocke_primitive: "instances/common/conv_implicit_gemm.py"
related: [family-convolution, technique-async-copy, technique-epilogue]
sources: [project-rocke, project-composablekernel, project-miopen, project-hipdnn]
---

# Implicit GEMM conv

NHWC×KYXC as a GEMM via the transform DAG (`pad`/`embed`/`unmerge`). Levers:
async DMA, `lds_k_pad`, MFMA atom, cshuffle vs direct. Direct grouped 16c/4c
are separate instances with different atom needs (`family-convolution`).
