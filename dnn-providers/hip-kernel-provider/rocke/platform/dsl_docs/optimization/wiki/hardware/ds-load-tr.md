---
id: hw-ds-load-tr
title: "ds_load_tr16_b128 (layout-constrained transpose)"
type: hardware
architectures: [gfx1250]
architecture_families: [gfx12]
tags: [ds-load-tr, gfx1250]
hardware_features: [ds-load-tr]
confidence: verified
related:
  - technique-gfx1250-ds-load-tr
  - technique-ds-read-tr
  - pattern-silent-ds-load-tr
  - hw-wmma-gfx1250
sources: [project-rocke]
aliases: [ds-load-tr, ds_load_tr16_b128]
---

# `ds_load_tr16_b128`

Hardware transpose while reading LDS, analog of pairing a TMA/swizzle
descriptor with the physical shared layout. Opcode is **not** gfx950
`ds_read_*_tr_*` (`has_ds_read_tr=false` on purpose). Wave32 per-lane
distribution differs; f16 vs bf16 use **different** llvm intrinsics
(`Gfx1250Backend.ds_tr16_b128_spec`).

Authoring still calls `IRBuilder.ds_read_tr16_b128`; the gfx1250 backend
rewrites the intrinsic.

The load is only correct if LDS was stored in the layout the opcode expects
(column-major for the WMMA B operand). Row-major coalesced DRAM→LDS plus a
transpose-read is a silent correctness bug — and LLVM may **insert** this
opcode when it sees an LDS load feeding WMMA (`pattern-silent-ds-load-tr`).
Rocke marks those loads volatile so the pass keeps `ds_read_b128`.
