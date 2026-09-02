---
id: technique-gfx1250-cache-policy
title: "GFX12 async load cachepolicy (th/scope)"
type: technique
tags: [cache-policy, async-lds, arch-specific]
symptoms: [memory-bound]
confidence: inferred
reproducibility: snippet
arch_specific: true
architecture_families: [gfx12]
architectures: [gfx1250]
operator_families: [gemm, attention]
rocke_primitive: "global_load_async_to_lds(..., coherency=)"
related:
  - hw-async-global-lds
  - technique-gfx12-async-lds
  - technique-vectorized-io
sources: [project-rocke]
prerequisites: [hw-async-global-lds]
---

# Cachepolicy on async DMA

Analog of KernelWiki cache-policy: the gfx12 `cpol` immediate on
`global_load_async_to_lds` (`coherency` 0..3; bits[0:2]=th, bits[3:4]=scope).
`0` is the default. `2` (`CACHE_STREAM` / SLC) is the documented one-shot
streaming choice in `IRBuilder.global_load_async_to_lds`.

This is one lever. It does not replace vector width, tile K, or ASYNCcnt
depth. Change it only with an ISA/cache-counter diff on the same harness.
