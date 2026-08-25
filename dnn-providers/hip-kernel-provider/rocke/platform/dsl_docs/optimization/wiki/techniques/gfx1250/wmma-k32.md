---
id: technique-gfx1250-wmma-k32
title: "gfx1250 WMMA 16×16×32"
type: technique
tags: [wmma-atom, wmma]
confidence: verified
reproducibility: snippet
arch_specific: true
architecture_families: [gfx12, cdna]
architectures: [gfx1250]
operator_families: [gemm, attention, moe]
rocke_primitive: "wmma_gfx1250_f32_16x16x32_f16"
related: [hw-gfx1250, technique-wmma-atom, technique-gfx12-async-lds, hw-wmma-gfx1250, hw-asynccnt, hw-ds-load-tr, kernel-wmma-gemm-gfx1250]
sources: [project-rocke, project-stinkytofu]
---

# gfx1250 WMMA K=32

gfx1250 is a CDNA *product* on the GFX12 *programming model*: wave32, WMMA, no
MFMA. Primary fp16/bf16 atom is **16×16×32**, not gfx1201’s 16×16×16. fp8 WMMA
is K=64. Do not admit the gfx950 MFMA tiled builder — `has_mfma=false` and the
gfx9 `has_async_lds` / `has_ds_read_tr` flags stay false on purpose.

```python
# instances/gfx1250/wmma_gemm.py  and fused_moe_mega_wmma.py
# op_id: wmma_gfx1250_f32_16x16x32_f16
```

Asm scheduling for generated kernels in this family is stinkytofu’s job
(`shared/stinkytofu`). Rocke must still verify the WMMA intrinsic and waitcnt
model (`waitcnt_model: split_gfx1250`).
