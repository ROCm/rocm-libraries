---
id: technique-wmma-atom
title: "WMMA atom selection (RDNA)"
type: technique
tags: [wmma-atom, wmma]
confidence: verified
reproducibility: snippet
arch_specific: true
architecture_families: [rdna]
architectures: [gfx1151, gfx1201]
operator_families: [gemm, attention]
rocke_primitive: "WmmaAtom / instances/gfx1151/wmma_gemm.py"
related: [hw-wmma, technique-wave32, technique-gfx1250-wmma-k32]
sources: [project-rocke]
---

# WMMA (RDNA)

Wave32 matrix ops, not MFMA. gfx1151: `wmma.f32.16x16x16.f16` with duplicated
halves in the gfx11 ABI. gfx1201: gfx12 ABI, **no** cross-half duplication
(A/B `v8f16` per lane). Do not copy gfx1151 packing onto gfx1201.

```python
# Native builders (not the CDNA UniversalGemmSpec pipeline):
# instances/gfx1151/wmma_gemm.py
# instances/gfx1201/wmma_gemm.py
```

Portable GEMM on gfx1151 is `mem` + `default` epilogue only. Lane maps: see
`arch_specs.json` comments; gfx1201 maps are marked hypothesis pending
`examples/gfx1201/wmma_probe.py`.
