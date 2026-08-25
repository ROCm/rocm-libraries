---
id: hw-rdna
title: "RDNA family (WMMA, wave32)"
type: hardware
architectures: [gfx1151, gfx1201]
architecture_families: [rdna]
tags: [wmma, wave32, lds]
hardware_features: [wmma, wave32]
confidence: verified
related: [hw-gfx1151, hw-gfx1201, hw-wmma, technique-wave32]
sources: [project-rocke]
---

# RDNA

Wave32, WMMA, no AGPR, 64 KiB LDS, no gfx9 async-LDS in the catalog. gfx1151
and gfx1201 **do not share** the WMMA operand ABI (duplicated halves vs
compact gfx12 packing). Always pick the matching `wmma_gemm` builder.
