---
id: project-tensile
title: "Tensile"
type: project
repo: ROCm/rocm-libraries
tree: shared/tensile
tags: [tensile, gemm]
operator_families: [gemm]
architecture_families: [cdna]
related: [family-gemm, project-tensilelite]
kernel_types: [gemm]
---

# Tensile

Benchmark-driven GEMM / batched GEMM / tensor-contraction generator. Primarily
the rocBLAS backend (`shared/tensile`). Historical source of tile, prefetch,
and LDS-buffer vocabulary that TensileLite and rocke still share.

Read `shared/tensile/README.md` on `develop`. For hipBLASLt-era kernels prefer
`project-tensilelite`.
