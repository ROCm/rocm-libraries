---
id: pattern-memory-bound
title: "Memory bandwidth bound"
type: pattern
tags: [async-copy, vectorized-io]
symptoms: [memory-bound]
candidate_techniques: [technique-async-copy, technique-vectorized-io, technique-software-pipeline, technique-tiling]
related: [pattern-compute-bound, family-overview]
sources: [project-rocke]
operator_families: [gemm, attention, convolution, moe, small-ops]
---

# Memory bound

Roofline or rocprof attributes runtime to data movement. Confirm with bytes
moved vs published HBM peak (hardware spec), not with a software TFLOP claim.

Likely: low arithmetic intensity, no reuse, narrow/uncoalesced loads, missing
async DRAM→LDS, cache-unfriendly streaming.

First actions: wider vector loads (`technique-vectorized-io`), `compv4` /
prefetch (`technique-async-copy`, `technique-software-pipeline`), larger K-tile
if LDS allows (`technique-tiling`).
