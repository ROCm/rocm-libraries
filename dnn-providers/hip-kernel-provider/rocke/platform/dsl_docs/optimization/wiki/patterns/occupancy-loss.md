---
id: pattern-occupancy-loss
title: "ISA improved, runtime got worse"
type: pattern
tags: [occupancy]
symptoms: [occupancy-loss]
candidate_techniques: [technique-occupancy, technique-epilogue, technique-tiling]
related: [pattern-register-pressure]
sources: [project-rocke]
---

# Occupancy loss after an atom/pipeline change

The new ISA looks “better” (more MFMA, fewer waits) but occupancy dropped
below the wave count that hid latency. Run `probe_occupancy` on both sides
before keeping the change. Often the fix is cshuffle vs direct epilogue, not
reverting the atom.
