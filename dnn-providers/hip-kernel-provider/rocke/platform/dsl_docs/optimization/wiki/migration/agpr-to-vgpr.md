---
id: migration-agpr-to-vgpr
title: "AGPR accumulators to VGPR (TMEM migration analog)"
type: migration
from_arch: gfx950
to_arch: gfx1250
architectures: [gfx950, gfx1250]
architecture_families: [cdna, gfx12]
tags: [migration, vgpr-acc, gfx1250]
confidence: verified
reproducibility: pseudocode
related:
  - hw-vgpr-acc-gfx1250
  - technique-agpr-acc
  - pattern-register-pressure
  - technique-occupancy
sources: [project-rocke]
---

# AGPR → VGPR

Analog of KernelWiki `migration-register-to-tmem`, **inverted**. Hopper WGMMA
keeps D in registers and Blackwell moves it to TMEM (relieving VGPR). gfx950
MFMA can park D in AGPR; gfx1250 WMMA **must** keep D in VGPR (`agprs: 0`).

| Concern | gfx950 MFMA | gfx1250 WMMA |
|---|---|---|
| Accumulator location | AGPR (and/or VGPR) | VGPR fragment only |
| MMA issue | wave64 MFMA | wave32 WMMA |
| Occupancy knob | AGPR vs VGPR split | tile + epilogue VGPR only |
| Accvgpr moves | legal | illegal |

## Porting sequence

1. Delete accvgpr copy / `a_mfma` live ranges.
2. Size the 16×16 `<8 x float>` fragment plus epilogue temps against 256 VGPR.
3. Re-run `probe_occupancy`. A gfx950 tile that was AGPR-safe can spill here.
4. Prefer a smaller CTA or fewer live stages over inventing a TMEM-like
   side buffer — none exists (`has_tdm=false`).
