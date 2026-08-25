---
id: hw-asynccnt
title: "ASYNCcnt (mbarrier analog)"
type: hardware
architectures: [gfx1250]
architecture_families: [gfx12]
tags: [asynccnt, waitcnt, gfx1250]
hardware_features: [asynccnt]
confidence: verified
related:
  - hw-async-global-lds
  - hw-split-waitcnt
  - technique-gfx1250-asynccnt-pipeline
  - pattern-async-without-asynccnt
sources: [project-rocke]
aliases: [asynccnt, s_wait_asynccnt]
---

# ASYNCcnt

Analog of KernelWiki mbarrier **transaction count**: a dedicated counter for
in-flight `global_load_async_to_lds` / `global_store_async_from_lds`. It is
**not** an mbarrier object — no arrival count, no phase/parity, no shared
memory descriptor.

Separate from:

| Counter | Traffic |
|---|---|
| ASYNCcnt | async global↔LDS DMA |
| loadcnt | VMEM register loads |
| dscnt | LDS `ds_read` / `ds_write` |

`s_wait_asynccnt n` waits until **at most n** async copies remain. `n=0`
drains. `n = copies_in_next_tile` is the ping-pong overlap
(`technique-gfx1250-asynccnt-pipeline`). Waiting on `vmcnt`/`lgkmcnt` does
not complete GFX12 async DMA — `pattern-async-without-asynccnt`.
