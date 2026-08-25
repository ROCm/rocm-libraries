---
id: hw-wmma-gfx1250
title: "gfx1250 WMMA (tcgen05 analog)"
type: hardware
architectures: [gfx1250]
architecture_families: [cdna, gfx12]
tags: [wmma, gfx1250]
hardware_features: [wmma, fp8]
confidence: verified
related:
  - technique-gfx1250-wmma-k32
  - hw-vgpr-acc-gfx1250
  - hw-fp8-wmma-gfx1250
  - migration-gfx950-to-gfx1250
  - kernel-wmma-gemm-gfx1250
sources: [project-rocke]
aliases: [wmma-gfx1250, "16x16x32"]
---

# gfx1250 WMMA

Analog of KernelWiki `hw-tcgen05-mma`: the matrix engine you schedule around.
It is **not** a fifth-generation Tensor Core. Issue is **per wave** (32 lanes),
D lives in a `<8 x float>` VGPR fragment, and there is no TMEM descriptor.

Catalog atoms (`arch_specs.json`):

| A/B | C | M×N×K | `op_id` |
|---|---|---|---|
| fp16 / bf16 | fp32 | 16×16×32 | `wmma_gfx1250_f32_16x16x32_{f16,bf16}` |
| fp8 / bf8 combos | fp32 | 16×16×64 | `wmma_gfx1250_f32_16x16x64_*` |
| fp32 | fp32 | 16×16×4 | `wmma_gfx1250_f32_16x16x4_f32` |

Not gfx1201’s 16×16×16. Not gfx950 MFMA. Block-scaled F4 / block16 forms are
intentionally omitted.

## Fragment ABI (hypothesis until the probe matches)

`instances/gfx1250/wmma_gemm.py`: A/B are `<16 x half>` per lane. K=32 splits
across lane-halves (lanes 0–15 hold K 0..15, 16–31 hold K 16..31). Accumulator
slot `i` of lane `l` maps to `(row = m0 + (l//16)*8 + i, col = n0 + l%16)`.
Confirm with `examples/gfx1250/wmma_probe.py` before trusting a new layout.

Lowering (`Gfx1250Backend.emit_wmma`): 8-operand llvm intrinsic
`(negA, A, negB, B, fmt, C, i1, i1)`; bf16 is `<16 x bfloat>` directly (no
gfx11 i16 bitcast).
