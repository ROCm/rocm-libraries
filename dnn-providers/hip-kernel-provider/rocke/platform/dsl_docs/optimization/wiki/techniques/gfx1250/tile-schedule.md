---
id: technique-gfx1250-tile-schedule
title: "Tile scheduling and persistent CTAs (no CLC)"
type: technique
tags: [tile-schedule, persistent-kernel, arch-specific]
symptoms: [tail-grid, low-cu-util, expert-imbalance]
confidence: inferred
reproducibility: snippet
arch_specific: true
architecture_families: [gfx12]
architectures: [gfx1250]
operator_families: [gemm, moe, attention]
rocke_primitive: "software tile id loop (no clusterlaunchcontrol)"
related:
  - pattern-tail-grid-gfx1250
  - pattern-low-cu-util-gfx1250
  - technique-persistent-streamk
  - technique-chiplet-swizzle
sources: [project-rocke, project-stinkytofu]
---

# Tile scheduling on gfx1250

Analog of KernelWiki tile-scheduling / CLC **without** Cluster Launch
Control. The catalog has no `clusterlaunchcontrol`. Persistent software
loops and raster order are the tools.

```python
# linear id → (m_tile, n_tile); persistent CTA walks a work range
m = tile_id // tiles_n
n = tile_id % tiles_n
```

Do not claim Stream-K: `technique-persistent-streamk` is MFMA-path in the
support matrix. gfx1250 grouped MoE imbalance is packing + persistent
expert walk (`pattern-expert-imbalance-gfx1250`), not a hardware work
queue. Chiplet/XCD raster is device-specific — do not copy gfx950
`chiplet_swizzle` constants.
