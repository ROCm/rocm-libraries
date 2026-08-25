---
id: technique-gfx1250-block-scale
title: "fp8 block-scaled GEMM on gfx1250"
type: technique
tags: [block-scale, fp8, arch-specific]
symptoms: [low-wmma-count]
confidence: source-reported
reproducibility: snippet
arch_specific: true
architecture_families: [gfx12]
architectures: [gfx1250]
operator_families: [gemm, moe]
rocke_primitive: "instances/gfx1250/block_scaled_gemm.py"
related:
  - hw-fp8-wmma-gfx1250
  - kernel-block-scale-gemm-gfx1250
  - family-gemm
  - family-moe
sources: [project-rocke]
prerequisites: [hw-fp8-wmma-gfx1250]
---

# Block-scale fp8

Analog of KernelWiki fine-grained quantization. Use the gfx1250 spec, not
an MX/NVFP4 layout from another ISA.

```python
from rocke.instances.gfx1250.block_scaled_gemm import BlockScaledGemmSpec

spec = BlockScaledGemmSpec(
    name="expert", M=..., N=..., K=...,
    dtype_a="fp8", dtype_b="fp8", dtype_c="bf16",
    scale_dtype="fp32", block_k=128, matrix_path="wmma",
)
```

`matrix_path="auto"` resolves to WMMA K=64. Do not emit omitted F4/block16
atoms. Keep scale dtype/granularity identical through quantize–GEMM–store.
Verify with `examples/gfx1250/gemm/block_scaled_gemm_verify.py`.
