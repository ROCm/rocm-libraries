---
id: project-hipblaslt
title: "hipBLASLt"
type: project
repo: ROCm/rocm-libraries
tree: projects/hipblaslt
tags: [hipblaslt, gemm]
operator_families: [gemm, moe]
architecture_families: [cdna, gfx12]
related: [family-gemm, project-tensilelite, project-stinkytofu]
kernel_types: [gemm, grouped-gemm]
---

# hipBLASLt

Flexible GEMM library (`hipblasLtMatmul`) with epilogue fusion (GELU/ReLU/SiLU
bias). Backend kernel provider is the optimized generator (TensileLite under
`projects/hipblaslt/tensilelite`). Use as the **source of GEMM techniques and
solution heuristics**, not as a dump of measured TFLOPS.

Sparse checkout of this worktree may omit `projects/`; read from the monorepo
`develop` tree. Docs: `projects/hipblaslt/README.md`, `AGENTS.md`.
