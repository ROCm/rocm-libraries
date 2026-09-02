---
id: project-tensilelite
title: "TensileLite"
type: project
repo: ROCm/rocm-libraries
tree: projects/hipblaslt/tensilelite
tags: [tensile, gemm]
operator_families: [gemm, moe]
architecture_families: [cdna, gfx12]
related: [project-hipblaslt, project-stinkytofu, family-gemm]
kernel_types: [gemm]
---

# TensileLite

hipBLASLt’s in-tree kernel generator (`projects/hipblaslt/tensilelite`):
assembly GEMM, LayerNorm/Softmax/AMax extras, gfx1250 path scheduled by
stinkytofu. Rocke GEMM pipelines (`mem`/`compv*`) are the DSL analog of
TensileLite prefetch/LDS buffering — map concepts, do not paste asm.

See `tensilelite/README.md` and `tensilelite/AGENTS.md` on `develop`.
