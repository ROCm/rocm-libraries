---
id: hw-split-waitcnt
title: "gfx1250 split waitcnt"
type: hardware
architectures: [gfx1250]
architecture_families: [gfx12]
tags: [split-waitcnt, waitcnt, gfx1250]
hardware_features: [split-waitcnt]
confidence: verified
related:
  - hw-asynccnt
  - technique-gfx1250-split-waitcnt
  - pattern-wrong-waitcnt-abi
sources: [project-rocke]
aliases: [split-waitcnt]
---

# Split waitcnt

gfx1250 **removed** the monolithic `s_waitcnt` intrinsic
(`Gfx1250Backend.emits_legacy_s_waitcnt = False`). LDS-barrier drain emits:

```text
s_wait_loadcnt 0   # if draining VMEM→LDS
s_wait_dscnt 0     # LDS ds_read/ds_write
s_barrier
```

clang may fuse the two waits into `s_wait_loadcnt_dscnt 0`. Raw `s_barrier`
does **not** auto-insert `s_wait_dscnt`; same-wave LDS write→read without the
drain is a NaN / stale-LDS race.

Generic `tile.s_waitcnt` encodings and 57-bit SRD packing are still
**deferred** in the backend (see the `Gfx1250Backend` note). Copying a gfx950
`s_waitcnt` mask onto gfx1250 is a miscompile (`pattern-wrong-waitcnt-abi`).
Async DMA is a fourth counter: `hw-asynccnt`.
