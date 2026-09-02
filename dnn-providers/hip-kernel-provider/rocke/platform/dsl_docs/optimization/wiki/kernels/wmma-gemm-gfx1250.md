---
id: kernel-wmma-gemm-gfx1250
title: "gfx1250 WMMA GEMM"
type: kernel
architectures: [gfx1250]
architecture_families: [gfx12]
operator_families: [gemm]
tags: [gemm, wmma, gfx1250]
confidence: experimental
reproducibility: snippet
kernel_types: [gemm]
languages: [rocke]
rocke_primitive: "instances/gfx1250/wmma_gemm.py"
related:
  - hw-wmma-gfx1250
  - family-gemm
  - technique-gfx1250-wmma-k32
  - pattern-wmma-lane-map
sources: [project-rocke, project-stinkytofu]
---

# WMMA GEMM (gfx1250)

RCR fp16, one wave per 16×16 tile, K=32 WMMA, no LDS in the bring-up
instance. Lane maps are exercised by `examples/gfx1250/wmma_probe.py` —
treat them as a hypothesis until that probe matches the numpy reference.

```python
from rocke.instances.gfx1250.wmma_gemm import WmmaGemmSpec, build_wmma_gemm
kd = build_wmma_gemm(WmmaGemmSpec(), arch="gfx1250")
```

TensileLite + stinkytofu are the library-side generators for production
asm GEMM on this gfx (`project-stinkytofu`).
