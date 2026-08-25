---
id: pattern-lds-conflict
title: "LDS bank conflicts / lgkmcnt stalls"
type: pattern
tags: [lds, lds-swizzle]
symptoms: [lds-stall]
candidate_techniques: [technique-lds-swizzle, technique-ds-read-tr, technique-tiling]
related: [hw-lds, pattern-pipeline-stalls]
sources: [project-rocke]
---

# LDS stall

ATT `s_waitcnt lgkmcnt` or rocprof LDS stall. gfx942: 32 banks / 64 KiB.
gfx950 and gfx1250: 64 banks / 160 KiB — stride math changes.

Pad (`lds_k_pad`), XOR vs padding swizzle (`technique-lds-swizzle`), and on
gfx950 prefer `ds_read_tr` (`technique-ds-read-tr`) instead of a software
transpose.
