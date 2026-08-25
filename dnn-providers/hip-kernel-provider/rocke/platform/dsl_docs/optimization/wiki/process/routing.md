---
id: process-routing
title: "How to route a problem"
type: process
tags: [routing]
related: [process-optimization-loop, process-probe-sequence, process-escape-hatch, family-overview, hw-gfx1250]
sources: [project-rocke]
---

# How to route a problem

1. **Operator family** — gemm / attention / convolution / moe / small-ops.
2. **gfx id** — from the device or `arch_specs.json`. Map to a family: CDNA MFMA, RDNA WMMA, or gfx1250.
3. **Step 0** — `process-optimization-loop`. Sweep knobs that already exist.
4. **Probe** — `process-probe-sequence` (occupancy → intrinsics → ISA → bench → rocprof).
5. **Symptom** — `query.py --symptom … --operator … --architecture …`.
6. **Family table** — `family-overview` then `family-<op>`. Common tech first; arch-specific only when the column requires it.
7. **One lever** — open the `technique-*` page, change the named spec field, verify, re-measure.
8. **If the catalog stalls** — same limiter after Step 0 + ≥3 one-lever ISA diffs, and the next idea is a retry: `process-escape-hatch` (required). Invent a new mapping; do not retile.
9. **gfx1250 / port** — `hw-gfx1250` analog map, then `query.py --type migration --architecture gfx1250`. Do not copy gfx950 MFMA or gfx1201 16×16×16 packs.
10. **Record** — qualitative keep/revert next to the builder. No software-achieved numbers in git.
