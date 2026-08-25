---
id: technique-agpr-acc
title: "AGPR accumulators (CDNA)"
type: technique
tags: [agpr]
confidence: source-reported
reproducibility: snippet
arch_specific: true
architecture_families: [cdna]
architectures: [gfx90a, gfx942, gfx950]
operator_families: [gemm, attention, convolution]
rocke_primitive: "AGPR file / MFMA acc"
related: [technique-occupancy, hw-cdna]
sources: [project-rocke]
---

# AGPR

CDNA exposes an accumulator VGPR file (256 AGPRs in `arch_specs.json`). MFMA
can target AGPR and free architectural VGPRs for addressing/pipeline. RDNA and
gfx1250 list `agprs: 0`.

Spills into AGPR vs VGPR are backend-sensitive (llvm flavor). Interpret
occupancy with the production `llvm22` comgr; llvm20 can report false
AGPR-spill walls. Do not “fix” a spill that vanishes on llvm22.
