---
id: project-rocroller
title: "rocRoller"
type: project
repo: ROCm/rocm-libraries
tree: shared/rocroller
tags: [gemm]
operator_families: [gemm]
architecture_families: [cdna]
related: [family-gemm]
kernel_types: [gemm]
---

# rocRoller

AMDGPU assembly kernel generator (`shared/rocroller`) with a GEMM client and
architecture YAML. Another source of tiling / scheduling vocabulary for CDNA
GEMM. Read `shared/rocroller/README.md` on `develop`.
