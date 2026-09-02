---
id: pattern-low-mfma-count
title: "Correct but slow, low MFMA/WMMA count"
type: pattern
tags: [mfma, wmma]
symptoms: [low-mfma-count]
candidate_techniques: [technique-mfma-atom, technique-wmma-atom, technique-vectorized-io, technique-isa-inspect]
related: [pattern-compute-bound]
sources: [project-rocke]
---

# Low matrix-instruction count

Wrong atom, scalar FMA path, or a dispatcher that never selected the tiled
kernel. `probe_intrinsic_counts` first. Then family-table atom column for this
gfx, and confirm the inner loop in ISA — not the launch knobs.
