---
id: technique-gfx12-async-lds
title: "GFX12 async global→LDS (gfx1250)"
type: technique
tags: [async-lds, async-copy]
confidence: verified
reproducibility: snippet
arch_specific: true
architecture_families: [gfx12]
architectures: [gfx1250]
operator_families: [gemm, attention]
rocke_primitive: "global_load_async_to_lds_b128 / s_wait_asynccnt"
related: [technique-async-copy, hw-gfx1250, technique-gfx1250-wmma-k32, hw-async-global-lds, hw-asynccnt, technique-gfx1250-asynccnt-pipeline, technique-gfx1250-cache-policy]
sources: [project-rocke]
---

# GFX12 async-to-LDS

Not `buffer_load_lds`. gfx1250 uses `global_load_async_to_lds_b128` plus
`s_wait_asynccnt`. Catalog `has_async_global_lds=true` while `has_async_lds`
(the gfx9 ABI) is false so the gfx950 MFMA pipeline does not silently select.

Transpose LDS read is `ds_load_tr16_b128`, not `ds_read_*_tr_*`.

Confirm with ISA on a gfx1250 object; copying gfx950 waitcnt encodings is a
miscompile (`waitcnt_model: split_gfx1250`).
