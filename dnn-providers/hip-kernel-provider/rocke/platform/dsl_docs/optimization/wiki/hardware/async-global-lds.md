---
id: hw-async-global-lds
title: "GFX12 async global→LDS (TMA analog)"
type: hardware
architectures: [gfx1250]
architecture_families: [gfx12]
tags: [async-lds, gfx1250]
hardware_features: [asynccnt]
confidence: verified
related:
  - technique-gfx12-async-lds
  - hw-asynccnt
  - technique-gfx1250-asynccnt-pipeline
  - technique-async-copy
  - migration-gfx9-async-to-gfx12-async
sources: [project-rocke]
aliases: ["global_load_async_to_lds"]
---

# Async global→LDS

Analog of KernelWiki TMA: hardware moves DRAM bytes into LDS without a
VGPR round-trip. It is **not** TMA — no tensor-map descriptor, no cluster
multicast, no `mbarrier::complete_tx`. Each lane names its own LDS address
(`global_load_async_to_lds_b{32,64,128}`). gfx9 `buffer_load_lds` does **not**
select (`has_async_lds=false` on purpose).

Completion is `hw-asynccnt` (`s_wait_asynccnt`), not `s_waitcnt vmcnt`.

```python
b.global_load_async_to_lds(src, src_index, lds, [lds_i], width_bytes=16)
b.s_wait_asynccnt(0)  # or n>0 for ping-pong overlap
```

`width_bytes` in {1, 4, 8, 16}. `coherency` is the gfx12 cachepolicy immediate
(`technique-gfx1250-cache-policy`). On non-gfx1250 backends `s_wait_asynccnt`
lowers to nothing — only emit this path when `has_async_global_lds`.
