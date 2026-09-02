---
id: hw-wmma
title: "WMMA"
type: hardware
architectures: [gfx1151, gfx1201, gfx1250]
architecture_families: [rdna, gfx12]
tags: [wmma]
hardware_features: [wmma]
confidence: verified
related: [technique-wmma-atom, technique-gfx1250-wmma-k32, hw-rdna, hw-wmma-gfx1250]
sources: [project-rocke]
aliases: [wmma]
---

# WMMA

Wave matrix multiply-accumulate on the SIMD VALU (not a separate MFMA unit).
ABI and K dimension differ by gfx: 16×16×16 (gfx1151/gfx1201) vs 16×16×32
(gfx1250). Treat each gfx as its own packing, then read the matching technique
page.
