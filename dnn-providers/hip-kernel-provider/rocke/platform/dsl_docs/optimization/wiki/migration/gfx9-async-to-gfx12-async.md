---
id: migration-gfx9-async-to-gfx12-async
title: "buffer_load_lds to global_load_async_to_lds"
type: migration
from_arch: gfx942
to_arch: gfx1250
architectures: [gfx942, gfx950, gfx1250]
architecture_families: [cdna, gfx12]
tags: [migration, async-lds, gfx1250]
confidence: verified
reproducibility: snippet
related:
  - hw-async-global-lds
  - hw-asynccnt
  - technique-async-copy
  - technique-gfx12-async-lds
  - pattern-async-without-asynccnt
sources: [project-rocke]
---

# gfx9 async-LDS → GFX12 async-LDS

| | gfx942+ (`has_async_lds`) | gfx1250 (`has_async_global_lds`) |
|---|---|---|
| Opcode | `buffer_load_lds` / `global_load_lds` | `global_load_async_to_lds_b*` |
| LDS address | M0 lane-contiguous base | explicit per-lane LDS index |
| Wait | `s_waitcnt vmcnt` | `s_wait_asynccnt` |
| Catalog bit that admits gfx950 MFMA pipeline | `has_async_lds=true` | **false** on purpose |

```python
# gfx1250 — not AsyncTileLoader's gfx9 buffer_load_lds path
b.global_load_async_to_lds(src, src_i, lds, [lds_i], width_bytes=16)
b.s_wait_asynccnt(0)
```

`AsyncTileLoader` / `compv4` on gfx9 is not a drop-in. Re-issue copies with
the GFX12 helper, then overlap with `s_wait_asynccnt(n)`
(`technique-gfx1250-asynccnt-pipeline`).
