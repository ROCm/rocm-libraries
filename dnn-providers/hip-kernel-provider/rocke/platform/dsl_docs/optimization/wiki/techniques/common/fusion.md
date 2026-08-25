---
id: technique-fusion
title: "Kernel and epilogue fusion"
type: technique
tags: [fusion, common]
confidence: source-reported
reproducibility: snippet
arch_specific: false
architecture_families: [cdna, rdna, gfx12]
operator_families: [gemm, attention, convolution, moe, small-ops]
rocke_primitive: "helpers/fuse.py / FusedEpilogue"
related: [family-moe, technique-epilogue]
sources: [project-rocke, project-composablekernel]
---

# Fusion

Fuse elementwise / bias / activation / quant into the producer kernel when the
epilogue still fits the resource budget. Rocke: `helpers/fuse.py`
(`FusedEpilogue`, `BiasAdd`, `ReLU`). MoE: SiLU-mul in the GEMM epilogue.
Conv: `deep_fused_conv_pool`.

```python
from rocke.helpers.fuse import FusedEpilogue, BiasAdd, ReLU
```

Fusion that blows VGPR and drops occupancy is a net loss — `technique-occupancy`.
