---
id: technique-chiplet-swizzle
title: "Chiplet / XCD grid swizzle (CDNA)"
type: technique
tags: [chiplet-swizzle, chiplet-xcd]
confidence: source-reported
reproducibility: snippet
arch_specific: true
architecture_families: [cdna]
architectures: [gfx942, gfx950]
operator_families: [gemm]
rocke_primitive: "TraitSpec.chiplet_swizzle"
related: [family-gemm, technique-tiling]
sources: [project-rocke]
---

# Chiplet swizzle

Multi-die CDNA packages group CUs into XCDs. Round-robin workgroups can thrash
L2. `TraitSpec.chiplet_swizzle` remaps flattened blockIdx onto XCD-local
stripes (`chiplet_wgm`, `chiplet_num_xcds`, `chiplet_chunk_size`). Launch grid
stays `(N_tiles, M_tiles)`.

```python
trait = TraitSpec(pipeline="compv4", chiplet_swizzle=True, chiplet_num_xcds=8)
```

Irrelevant on single-die RDNA. Confirm locality with profiler L2 metrics, not
with a software TFLOP number.
