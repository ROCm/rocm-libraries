---
id: technique-ds-read-tr
title: "Transpose LDS read (gfx950)"
type: technique
tags: [ds-read-tr]
confidence: verified
reproducibility: snippet
arch_specific: true
architecture_families: [cdna]
architectures: [gfx950]
operator_families: [gemm, attention]
rocke_primitive: "ds_read_tr16 / has_ds_read_tr"
related: [hw-gfx950, technique-lds-swizzle, pattern-valu-plumbing, technique-gfx1250-ds-load-tr, migration-ds-read-tr-to-ds-load-tr]
sources: [project-rocke]
---

# `ds_read_*_tr_*` (CDNA4)

Hardware transpose while reading LDS. Catalog: `has_ds_read_tr=true` only on
gfx950 among MFMA parts. gfx1250 has a **different** opcode
(`ds_load_tr16_b128`) — do not reuse this path (`technique-gfx1250-ds-load-tr`).

```python
# Lowering emits ds.read.tr16 when the LDS layout and arch flag allow it.
# Confirm: probe_isa_inspect should show ds_read_b64_tr_b16 / ds_read_b128_tr_b16
# not a VALU permute tail.
```

If the ISA still has `v_permlane` / `ds_bpermute` after enabling tr-read, the
layout is not paired with the reader (`technique-lds-swizzle`).
