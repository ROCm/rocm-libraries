---
id: technique-gfx1250-asynccnt-pipeline
title: "Multi-stage async DMA with s_wait_asynccnt"
type: technique
tags: [asynccnt-pipeline, software-pipeline, arch-specific]
symptoms: [pipeline-stalls, async-without-asynccnt, memory-bound]
confidence: verified
reproducibility: snippet
arch_specific: true
architecture_families: [gfx12]
architectures: [gfx1250]
operator_families: [gemm, attention]
rocke_primitive: "global_load_async_to_lds + s_wait_asynccnt(n)"
related:
  - hw-asynccnt
  - hw-async-global-lds
  - technique-software-pipeline
  - technique-gfx12-async-lds
  - pattern-async-without-asynccnt
sources: [project-rocke]
prerequisites: [hw-async-global-lds, hw-asynccnt]
---

# ASYNCcnt software pipeline

Analog of KernelWiki pipeline-stages / ping-pong: keep tile i+1’s DMA in
flight while WMMA consumes tile i. Stage count is LDS bytes vs occupancy,
not a universal 2 or 3.

```python
# issue next tile copies, then drain only the current tile
b.global_load_async_to_lds(..., width_bytes=16)
b.s_wait_asynccnt(n_next)  # n_next = copies already issued for the next tile
# WMMA on current LDS
```

`n=0` is a full drain (prologue/epilogue). Waiting on loadcnt/dscnt does
not complete ASYNCcnt. Extra stages that do not change the ISA async-copy
count are still catalog tiling — not this technique.
