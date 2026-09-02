---
id: pattern-valu-plumbing
title: "Low matrix util, high VALU, low memory stall"
type: pattern
tags: [epilogue]
symptoms: [valu-plumbing]
candidate_techniques: [technique-epilogue, technique-ds-read-tr, technique-occupancy]
related: [pattern-compute-bound, kernel-attention-2d]
sources: [project-rocke]
---

# VALU plumbing, not HBM

The matrix engine is starved by cross-lane permutes, softmax VALU, or a scalar
epilogue. Per-iter ISA histogram: permute / `ds_bpermute` / short stores. Fix
layout (32×32 chained MFMA, `ds_read_tr`, vector epilogue), not tile_m.
