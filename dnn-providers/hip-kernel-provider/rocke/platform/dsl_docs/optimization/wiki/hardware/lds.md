---
id: hw-lds
title: "LDS capacity and banks"
type: hardware
architectures: [gfx90a, gfx942, gfx950, gfx1151, gfx1201, gfx1250]
architecture_families: [cdna, rdna, gfx12]
tags: [lds]
hardware_features: [lds]
confidence: verified
related: [technique-lds-swizzle, pattern-lds-conflict]
sources: [project-rocke]
---

# LDS

| gfx | LDS bytes | Typical banks (docs) |
|---|---|---|
| gfx90a / gfx942 / gfx1151 / gfx1201 | 65536 | 32 on gfx942; RDNA see ISA |
| gfx950 / gfx1250 | 163840 | 64 on gfx950 |

Bank-conflict periods are opcode-specific (`arch/gfx950.md`). Rocke knobs:
`LdsLayout`, `lds_k_pad`.
