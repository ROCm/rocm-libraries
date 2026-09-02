---
id: technique-epilogue
title: "Direct vs cshuffle epilogue"
type: technique
tags: [epilogue, common]
confidence: source-reported
reproducibility: snippet
arch_specific: false
architecture_families: [cdna, rdna]
operator_families: [gemm, convolution]
rocke_primitive: "DirectEpilogue / CShuffleEpilogue"
related: [technique-vectorized-io, pattern-valu-plumbing]
sources: [project-rocke]
---

# Epilogue

If per-lane accumulators map to contiguous outputs, store directly (vector).
If the MFMA layout is scattered, stage through LDS (`cshuffle`) then wide
store. gfx1151 support matrix: `default` only — instances that require
cshuffle are rejected.

```python
trait = TraitSpec(pipeline="compv4", epilogue="cshuffle")  # CDNA GEMM default
# epilogue="default" → DirectEpilogue
```

Confirm with ISA: `buffer_store_dwordx4` vs short stores. Attention’s “epilogue”
is online softmax + PV store — see `pattern-valu-plumbing`.
