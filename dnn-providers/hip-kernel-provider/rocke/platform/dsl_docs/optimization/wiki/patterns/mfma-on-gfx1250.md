---
id: pattern-mfma-on-gfx1250
title: "gfx950 MFMA tiled builder selected on gfx1250"
type: pattern
tags: [wmma, gfx1250]
architectures: [gfx1250]
architecture_families: [gfx12]
symptoms: [mfma-on-gfx1250, low-wmma-count]
candidate_techniques:
  - technique-gfx1250-wmma-k32
  - technique-wave32
related:
  - migration-gfx950-to-gfx1250
  - hw-wmma-gfx1250
  - kernel-wmma-attention-gfx1250
sources: [project-rocke]
---

# MFMA pipeline on gfx1250

ISA shows MFMA, gfx9 `buffer_load_lds`, or `ds_read_*_tr_*`, or the kernel
failed to build because those flags are false. gfx1250 must use
`library/kernels/gfx1250/` and `instances/gfx1250/`. Catalog
`has_mfma` / `has_async_lds` / `has_ds_read_tr` stay false so the gfx950
attention builder cannot sneak in.

Open `migration-gfx950-to-gfx1250` and start from the WMMA instance, not a
retargeted gfx950 file.
