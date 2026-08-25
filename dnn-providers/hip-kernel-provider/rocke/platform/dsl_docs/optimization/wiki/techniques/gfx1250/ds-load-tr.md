---
id: technique-gfx1250-ds-load-tr
title: "Use ds_load_tr16_b128 with a matching LDS layout"
type: technique
tags: [ds-load-tr, arch-specific]
symptoms: [silent-ds-load-tr, valu-plumbing]
confidence: verified
reproducibility: snippet
arch_specific: true
architecture_families: [gfx12]
architectures: [gfx1250]
operator_families: [gemm, attention]
rocke_primitive: "IRBuilder.ds_read_tr16_b128 (lowers to ds_load_tr16_b128)"
related:
  - hw-ds-load-tr
  - pattern-silent-ds-load-tr
  - technique-ds-read-tr
  - technique-lds-swizzle
sources: [project-rocke]
prerequisites: [hw-ds-load-tr]
---

# Transpose-read on gfx1250

Analog of KernelWiki shared-memory swizzle: the physical LDS layout must
match the opcode. Call the same helper as gfx950; the backend picks
`ds_load_tr16_b128`.

```python
b_frag = b.ds_read_tr16_b128(lds_b, idx, dtype=F16)  # → <8 x half> on gfx1250
```

Keep **plain** `ds_read` when LDS is row-major for coalesced DRAM stores
(conv/GEMM wavelet). Rocke already volatiles those loads
(`blocks_ds_load_tr16`). If you *want* the transpose, store the WMMA B
layout explicitly and confirm ISA shows `ds_load_tr16_b128` without a
VALU permute tail.
