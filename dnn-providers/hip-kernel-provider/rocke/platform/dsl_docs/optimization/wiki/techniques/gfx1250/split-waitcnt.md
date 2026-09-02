---
id: technique-gfx1250-split-waitcnt
title: "Emit split loadcnt/dscnt instead of s_waitcnt"
type: technique
tags: [split-waitcnt, waitcnt, arch-specific]
symptoms: [wrong-waitcnt-abi, missing-waitcnt]
confidence: verified
reproducibility: snippet
arch_specific: true
architecture_families: [gfx12]
architectures: [gfx1250]
operator_families: [gemm, attention, convolution, moe, small-ops]
rocke_primitive: "Gfx1250Backend.emit_lds_barrier_drain"
related:
  - hw-split-waitcnt
  - hw-asynccnt
  - pattern-wrong-waitcnt-abi
  - technique-isa-inspect
sources: [project-rocke]
prerequisites: [hw-split-waitcnt]
---

# Split waitcnt

Let the backend drain LDS barriers. Do not emit `llvm.amdgcn.s.waitcnt` —
it is not selectable. ISA should show `s_wait_loadcnt` / `s_wait_dscnt`
(or fused `s_wait_loadcnt_dscnt`) before `s_barrier`, and `s_wait_asynccnt`
on the DMA path.

`s_barrier_bare` is for producer/consumer loops that already waited; a
bare barrier with no prior dscnt/asynccnt is `pattern-missing-waitcnt` on
this gfx.

Confirm with `probe_isa_inspect` on a gfx1250 object, not a gfx950 listing.
