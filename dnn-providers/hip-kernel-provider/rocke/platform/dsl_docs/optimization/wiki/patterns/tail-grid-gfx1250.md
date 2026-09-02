---
id: pattern-tail-grid-gfx1250
title: "Last wave of the grid leaves CUs idle"
type: pattern
tags: [tile-schedule, gfx1250]
architectures: [gfx1250]
architecture_families: [gfx12]
symptoms: [tail-grid, low-cu-util]
candidate_techniques:
  - technique-gfx1250-tile-schedule
  - technique-tiling
related:
  - pattern-low-cu-util-gfx1250
  - pattern-occupancy-loss
sources: [project-rocke]
---

# Tail grid

Analog of KernelWiki tail-effect. Do **not** invent a CU count. Measure
resident CTAs vs launched tiles; a static grid whose last wave cannot fill
the occupancy limit leaves CUs idle.

Persistent software scheduling can redistribute remaining tiles; it cannot
create parallelism when fewer independent tiles remain than resident slots.
No CLC in the catalog.
