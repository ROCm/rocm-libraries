---
id: technique-persistent-streamk
title: "Persistent kernels and Stream-K"
type: technique
tags: [persistent-kernel, stream-k, common]
confidence: source-reported
reproducibility: snippet
arch_specific: false
architecture_families: [cdna]
operator_families: [gemm]
rocke_primitive: "TraitSpec.persistent / StreamKGemmSpec"
related: [technique-tiling, family-gemm]
sources: [project-rocke, project-hipblaslt]
---

# Persistent / Stream-K

Keep CTAs resident and pull tiles (persistent), or split K across CTAs with a
fixup reduction (`StreamKGemmSpec`, `global_atomic_add_f32`). Use when the grid
is too small or K-tail imbalance dominates — not as a default for large square
GEMM.

```python
from rocke.instances.common.streamk_gemm import StreamKGemmSpec
```

Support matrix: Stream-K is MFMA-only (rejected on gfx1151). hipBLASLt/Tensile
expose analogous splitting in the solution library.
