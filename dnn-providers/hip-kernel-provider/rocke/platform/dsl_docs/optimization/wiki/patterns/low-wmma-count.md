---
id: pattern-low-wmma-count
title: "Correct but slow, low WMMA count (gfx1250)"
type: pattern
tags: [wmma, gfx1250]
architectures: [gfx1250]
architecture_families: [gfx12]
symptoms: [low-wmma-count, compute-bound]
candidate_techniques:
  - technique-gfx1250-wmma-k32
  - technique-gfx1250-asynccnt-pipeline
  - technique-gfx1250-block-scale
  - technique-epilogue
related:
  - pattern-compute-bound
  - pattern-low-mfma-count
  - hw-wmma-gfx1250
sources: [project-rocke]
---

# Low WMMA count

Same idea as `pattern-low-mfma-count` for this gfx. `probe_intrinsic_counts`
shows too few WMMA, the 16×16×16 atom (wrong), or VALU plumbing. First
actions: K=32 f16 / K=64 fp8, feed the engine with ASYNCcnt pipeline, then
epilogue. If the atom already matches and the histogram still cannot move:
`process-escape-hatch`.
