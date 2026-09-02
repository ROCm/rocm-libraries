---
id: pattern-compute-bound
title: "Not reaching matrix-core throughput"
type: pattern
tags: [mfma, wmma]
symptoms: [compute-bound, low-mfma-count]
candidate_techniques: [technique-mfma-atom, technique-wmma-atom, technique-gfx1250-wmma-k32, technique-software-pipeline, technique-epilogue]
related: [pattern-valu-plumbing, family-overview, pattern-low-wmma-count]
sources: [project-rocke]
---

# Compute bound

Matrix engine should dominate; ISA shows too few MFMA/WMMA, the wrong atom, or
the epilogue/VALU plumbing eating the hot loop.

First actions: `probe_intrinsic_counts` then `technique-mfma-atom` /
`technique-wmma-atom` for this gfx column; K-pack on gfx950; software pipeline
so the engine is fed; vector epilogue if stores serialize the loop.

If those catalog levers already match the ISA and the count still cannot move:
`pattern-catalog-exhausted` / `process-escape-hatch` (new mapping, not another
atom in the same family).
