---
id: hw-cdna
title: "CDNA family (MFMA, wave64)"
type: hardware
architectures: [gfx90a, gfx942, gfx950]
architecture_families: [cdna]
tags: [mfma, wave64, agpr, lds]
hardware_features: [mfma, agpr, wave64]
confidence: verified
related: [hw-gfx942, hw-gfx950, hw-mfma, technique-mfma-atom]
sources: [project-rocke]
aliases: [cdna]
---

# CDNA

Wave64, MFMA, AGPR present. Async LDS from gfx942. Transpose LDS read from
gfx950. LDS 64 KiB (gfx90a/gfx942) or 160 KiB (gfx950). Exact atoms and flags:
`platform/python/rocke/core/arch/data/arch_specs.json`.

gfx1250 is catalogued `family: cdna` but runs **WMMA/wave32** — use `hw-gfx1250`,
not this page.
