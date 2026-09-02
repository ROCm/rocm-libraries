---
id: technique-tiling
title: "Tile and CTA geometry"
type: technique
tags: [tiling, common]
confidence: source-reported
reproducibility: snippet
arch_specific: false
architecture_families: [cdna, rdna, gfx12]
operator_families: [gemm, attention, convolution, moe]
rocke_primitive: "TileSpec / WarpGrid / num_warps"
related: [technique-occupancy, family-gemm]
sources: [project-rocke, project-origami]
---

# Tiling

One CTA owns a `tile_m × tile_n` output (GEMM) or `num_warps × block_m_per_warp`
rows (attention). Larger tiles raise reuse and LDS/VGPR. Origami
(`shared/origami`) picks GEMM tiles analytically; rocke still requires a
measured occupancy check.

```python
from rocke.instances.common.gemm_universal import TileSpec, TraitSpec, DataSpec, UniversalGemmSpec

tile = TileSpec(tile_m=128, tile_n=128, tile_k=64, warp_m=2, warp_n=2,
                warp_tile_m=32, warp_tile_n=32, warp_tile_k=16)
```

Illegal geometries fail `is_valid_spec` / `__post_init__`. Sweep ±1 warp and
tile_k around the occupant-legal set (`technique-occupancy`) before inventing a
new body.
