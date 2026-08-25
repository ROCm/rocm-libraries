---
id: technique-vectorized-io
title: "Wide global loads and stores"
type: technique
tags: [vectorized-io, common]
confidence: source-reported
reproducibility: snippet
arch_specific: false
architecture_families: [cdna, rdna, gfx12]
operator_families: [gemm, attention, convolution, moe, small-ops]
rocke_primitive: "buffer_load_vN_f16 / buffer_store_vN_f16"
related: [technique-epilogue, pattern-tail-oob, pattern-memory-bound]
sources: [project-rocke]
---

# Vectorized I/O

Use dwordx2/x4 buffer or global ops when addresses are contiguous and aligned.
Scalar fp16 stores in the epilogue are a common false “compute-bound” kernel.

```python
# IR builder: buffer_store_vN_f16 / global_store_vN_f16 with N in {1,2,4}
# DirectEpilogue emits these when lane-owned C is contiguous.
```

Tail shapes need OOB-safe descriptors (`pattern-tail-oob`). `probe_isa_inspect`
should show `buffer_store_dwordx*` not a flood of shorts.
