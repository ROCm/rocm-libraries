---
id: technique-lds-swizzle
title: "LDS padding and bank swizzle"
type: technique
tags: [lds-swizzle, lds, common]
confidence: source-reported
reproducibility: snippet
arch_specific: false
architecture_families: [cdna, rdna, gfx12]
operator_families: [gemm, attention, convolution]
rocke_primitive: "LdsLayout / lds_k_pad"
related: [hw-lds, technique-ds-read-tr, pattern-lds-conflict]
sources: [project-rocke]
symptoms: [lds-stall]
---

# LDS swizzle

Bank index is `(byte_address / 4) % n_banks`. gfx942: 32 banks; gfx950/gfx1250:
64 banks — a stride that fully conflicts on 32 banks may only 2-way conflict
on 64.

```python
# Implicit GEMM: pad K in LDS. Async packed layouts often need pad 0 plus a
# consumer-side read swizzle instead.
spec.lds_k_pad = 8
```

Prefer padding on gfx950; XOR swizzle is the usual gfx942 choice. Pair the
layout with the reader (`ds_read_tr` on gfx950). Measure with
`analyze_lds_conflicts.py` / ATT lgkmcnt, not with a guess.
