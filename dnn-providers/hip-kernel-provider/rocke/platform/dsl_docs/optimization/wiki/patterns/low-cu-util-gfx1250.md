---
id: pattern-low-cu-util-gfx1250
title: "Low CU utilization on gfx1250"
type: pattern
tags: [tile-schedule, gfx1250]
architectures: [gfx1250]
architecture_families: [gfx12]
symptoms: [low-cu-util, tail-grid, launch-bound]
candidate_techniques:
  - technique-gfx1250-tile-schedule
  - technique-occupancy
  - technique-tiling
related:
  - pattern-tail-grid-gfx1250
  - pattern-occupancy-loss
sources: [project-rocke]
---

# Low CU utilization

Occupancy probe is not the limiter, but waves/CUs sit idle. Causes: grid
too small; tail wave; expert imbalance; launch-bound tiny kernels.

Candidates: larger grid / persistent tile walk
(`technique-gfx1250-tile-schedule`); more waves if VGPR/LDS allow
(`technique-occupancy`). Not a reason to re-enable MFMA.
