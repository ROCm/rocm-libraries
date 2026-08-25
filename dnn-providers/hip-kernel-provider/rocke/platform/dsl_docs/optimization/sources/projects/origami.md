---
id: project-origami
title: "Origami"
type: project
repo: ROCm/rocm-libraries
tree: shared/origami
tags: [gemm, tiling]
operator_families: [gemm]
architecture_families: [cdna, rdna]
related: [technique-tiling, family-gemm]
kernel_types: [gemm]
---

# Origami

Analytical GEMM tile/mapping selection from compute and memory **latency
models** (`shared/origami`). Use it as a prior for `TileSpec` sweeps — then
confirm with `probe_occupancy` / a real harness. Not a substitute for Step 0.
