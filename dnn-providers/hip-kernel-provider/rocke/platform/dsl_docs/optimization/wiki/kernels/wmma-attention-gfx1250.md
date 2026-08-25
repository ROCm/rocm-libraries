---
id: kernel-wmma-attention-gfx1250
title: "gfx1250 WMMA attention 2D/3D"
type: kernel
architectures: [gfx1250]
architecture_families: [gfx12]
operator_families: [attention]
tags: [attention, wmma, gfx1250]
confidence: source-reported
reproducibility: snippet
kernel_types: [attention, fmha]
languages: [rocke]
rocke_primitive: "library/kernels/gfx1250/attention_tiled_2d.py"
related:
  - family-attention
  - hw-wmma-gfx1250
  - technique-wave32
  - technique-gfx12-async-lds
  - migration-gfx950-to-gfx1250
sources: [project-rocke]
---

# WMMA attention (gfx1250)

Wave32 WMMA 16×16×32 over a paged KV cache. Shared body:
`library/kernels/gfx1250/_wmma_attention_common.py`.

```text
Q*K^T (WMMA) → masked online softmax → P in LDS
             → V in LDS → P*V (WMMA)
```

2D = prefill (`attention_tiled_2d.py`); 3D = split-KV decode
(`attention_tiled_3d.py`). Not the gfx950 MFMA tiled builder. Verify:
`library/builders/gfx1250/attention/`.
