---
id: hw-mfma
title: "MFMA"
type: hardware
architectures: [gfx90a, gfx942, gfx950]
architecture_families: [cdna]
tags: [mfma]
hardware_features: [mfma]
confidence: verified
related: [technique-mfma-atom, hw-cdna]
sources: [project-rocke]
aliases: [mfma]
---

# MFMA

CDNA matrix-FMA. One matrix core per SIMD; designed overlap is MFMA on one wave
with VALU on another. Atom catalog is per-gfx in `arch_specs.json` and
`helpers/atoms.py`. Not present on gfx1151/gfx1201/gfx1250 (those use WMMA).
