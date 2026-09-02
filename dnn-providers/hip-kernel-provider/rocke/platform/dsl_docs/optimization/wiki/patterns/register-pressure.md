---
id: pattern-register-pressure
title: "Register pressure / low occupancy"
type: pattern
tags: [occupancy, agpr]
symptoms: [register-pressure, occupancy-loss]
candidate_techniques: [technique-occupancy, technique-agpr-acc, technique-tiling, technique-software-pipeline]
related: [pattern-occupancy-loss, hw-lds]
sources: [project-rocke]
---

# Register pressure

`probe_occupancy` reports VGPR/AGPR limiter or spills. Larger tiles, extra
pipeline stages, and 32×32 atoms all raise VGPR. On CDNA, accumulators can sit
in AGPR (`technique-agpr-acc`); RDNA/gfx1250 have no AGPR file.

Cut unroll, shrink tile_k, drop a pipeline stage, or lower `waves_per_eu` only
after measuring occupancy — occupancy wins are not automatic.
