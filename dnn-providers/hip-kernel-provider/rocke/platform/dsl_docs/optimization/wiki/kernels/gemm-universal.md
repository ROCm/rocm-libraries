---
id: kernel-gemm-universal
title: "Universal GEMM (rocke)"
type: kernel
architectures: [gfx942, gfx950, gfx1151]
architecture_families: [cdna, rdna]
operator_families: [gemm]
tags: [gemm]
confidence: verified
reproducibility: snippet
kernel_types: [gemm]
languages: [rocke]
rocke_primitive: "instances/common/gemm_universal.py"
related: [family-gemm, technique-tiling, technique-software-pipeline, technique-epilogue]
sources: [project-rocke, project-hipblaslt, project-tensilelite]
---

# Universal GEMM

`UniversalGemmSpec` mirrors the CK dispatcher tile/trait/data schema: RCR
fp16 family, pipelines `mem`/`compv3`/`compv4`, epilogues `default`/`cshuffle`.
gfx1151 compiles `mem`+`default` only.

```python
from rocke.instances.common.gemm_universal import (
    TileSpec, TraitSpec, DataSpec, UniversalGemmSpec,
)
```

Siblings: batched/grouped/flatmm/streamk/block-scale/mx. Compare levers with
hipBLASLt solutions (`project-hipblaslt`) rather than copying their tiles
blindly — occupancy differs.
