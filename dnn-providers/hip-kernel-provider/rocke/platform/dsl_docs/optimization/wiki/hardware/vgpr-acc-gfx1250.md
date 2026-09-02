---
id: hw-vgpr-acc-gfx1250
title: "VGPR accumulators on gfx1250 (no TMEM / no AGPR)"
type: hardware
architectures: [gfx1250]
architecture_families: [cdna, gfx12]
tags: [wmma, vgpr-acc, gfx1250]
hardware_features: [vgpr-acc, wmma]
confidence: verified
related:
  - hw-wmma-gfx1250
  - technique-agpr-acc
  - migration-agpr-to-vgpr
  - pattern-register-pressure
sources: [project-rocke]
aliases: ["no tmem", "no agpr"]
---

# VGPR accumulators (TMEM analog — inverted)

KernelWiki TMEM is dedicated on-chip storage for `tcgen05` D. gfx1250 has
**neither** TMEM **nor** AGPR (`limits.agprs: 0`). WMMA D is a VGPR fragment
(`<8 x float>` per lane for 16×16). Occupancy fights the same VGPR file that
holds A/B plumbing and the epilogue.

Porting from gfx950 is the **inverse** of Hopper-register → Blackwell-TMEM:
you **lose** AGPR headroom, you do not gain a side buffer. See
`migration-agpr-to-vgpr`.

Do not emit `accvgpr` moves. Do not size tiles as if 160 KiB LDS made
registers free — LDS and VGPR still trade occupancy independently.
