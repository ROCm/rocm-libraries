---
id: migration-gfx950-to-gfx1250
title: "Migrating gfx950 MFMA kernels to gfx1250 WMMA"
type: migration
from_arch: gfx950
to_arch: gfx1250
architectures: [gfx950, gfx1250]
architecture_families: [cdna, gfx12]
tags: [migration, gfx1250, wmma]
confidence: verified
reproducibility: pseudocode
related:
  - hw-wmma-gfx1250
  - hw-vgpr-acc-gfx1250
  - migration-agpr-to-vgpr
  - migration-gfx9-async-to-gfx12-async
  - migration-ds-read-tr-to-ds-load-tr
  - technique-gfx1250-wmma-k32
  - pattern-mfma-on-gfx1250
sources: [project-rocke]
---

# gfx950 → gfx1250

Analog of KernelWiki `migration-wgmma-to-tcgen05`: a **redesign**, not an
opcode rename. gfx950 is wave64 MFMA + AGPR + `buffer_load_lds` +
`ds_read_*_tr_*`. gfx1250 is wave32 WMMA + VGPR-only +
`global_load_async_to_lds_*` + `ds_load_tr16_b128`.

Do not admit `attention_arch.py` / gfx950 tiled MFMA builders.
`has_mfma=false`, `has_async_lds=false`, `has_ds_read_tr=false` exist so that
path cannot silently select.

## Required changes

1. **Wave size 32.** XOR-butterfly / `ds_bpermute` trees use `wave_size=32`
   (`technique-wave32`). Block dim = `num_warps * 32`.
2. **Replace MFMA atoms with WMMA 16×16×32** (fp8: 16×16×64). Lane maps are
   a hypothesis — run `wmma_probe.py`.
3. **Move accumulators out of AGPR** — `migration-agpr-to-vgpr`.
4. **Replace async ABI** — `migration-gfx9-async-to-gfx12-async`.
5. **Replace transpose-read ABI** — `migration-ds-read-tr-to-ds-load-tr`.
6. **Replace waitcnt** — `hw-split-waitcnt`. Never copy a gfx950 mask.
7. **Rebuild the epilogue** for the 16×16 column-distributed VGPR fragment.
8. **Retile.** 160 KiB LDS is the same *number* as gfx950, but wave32 + no
   AGPR change occupancy. Do not keep `compv4`+cshuffle as a gfx9 pipeline
   token; use gfx1250 WMMA builders (`instances/gfx1250/`).

```text
gfx950:  wave64  MFMA  AGPR  buffer_load_lds   ds_read_*_tr_*   s_waitcnt
gfx1250: wave32  WMMA  VGPR  global_load_async ds_load_tr16     split + asynccnt
```
