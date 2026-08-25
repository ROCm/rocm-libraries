---
id: project-miopen
title: "MIOpen"
type: project
repo: ROCm/rocm-libraries
tree: projects/miopen
tags: [miopen, convolution]
operator_families: [convolution]
architecture_families: [cdna, rdna]
related: [family-convolution, project-hipdnn]
kernel_types: [convolution]
---

# MIOpen

High-performance ML primitives, including convolution algorithm selection
(implicit GEMM, Winograd, FFT, …). Host-side source for “which conv algorithm
maps to this shape.” Kernel-body levers for rocke implicit/direct conv stay on
`family-convolution`.

`projects/miopen/README.md` on `develop`.
