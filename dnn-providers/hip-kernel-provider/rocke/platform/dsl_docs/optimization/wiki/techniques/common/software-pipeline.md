---
id: technique-software-pipeline
title: "Software pipeline and double buffering"
type: technique
tags: [software-pipeline, double-buffering, common]
confidence: source-reported
reproducibility: snippet
arch_specific: false
architecture_families: [cdna, rdna, gfx12]
operator_families: [gemm, attention, convolution]
rocke_primitive: "TraitSpec.pipeline / SoftwarePipeline"
related: [technique-async-copy, pattern-pipeline-stalls]
sources: [project-rocke, project-tensilelite]
prerequisites: [technique-async-copy]
---

# Software pipeline

Overlap DRAM→LDS (or LDS→VGPR) of tile i+1 with MMA of tile i.
Rocke GEMM: `pipeline="mem"` (single buffer), `"compv3"` (hints), `"compv4"`
(double-buffer + sched groups). Attention: K/V ring depth and early-V.
RDNA instances in the support matrix only ship `mem`.

```python
trait = TraitSpec(pipeline="compv4", scheduler="intrawave", epilogue="cshuffle")
```

Extra stages cost LDS and can drop occupancy. Confirm with an ISA waitcnt
diff, not by counting `sched_barrier` (those are often compile-time fences).
TensileLite implements the same idea as prefetch/LDS buffers in generated asm.
