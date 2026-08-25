---
id: family-small-ops
title: "Small-ops family — common tech × architecture"
type: family
operator_families: [small-ops]
architecture_families: [cdna, rdna, gfx12]
tags: [small-ops, routing]
related:
  - family-overview
  - technique-vectorized-io
  - technique-wave32
  - technique-occupancy
  - process-escape-hatch
sources:
  - project-rocke
  - project-composablekernel
---

# Small-ops

Rocke: `elementwise`, `reduce`, `layernorm2d`, `rmsnorm2d`,
`add_rmsnorm2d_*`, `smoothquant`, `transpose` / `permute_nd`, `pooling`,
`img2col`. These emit generic AMDGPU IR; the gfx id mainly selects the
comgr triple and **wave size**.

## Common levers

| Lever | Field | Notes |
|---|---|---|
| Vector width | `vec` in {2,4,8} | align to dtype; tail-safe remainder |
| Block size | 64…1024 | occupancy vs reduction tree depth |
| LDS reduction | `block_lds_reduce` | when the row does not fit in registers |
| Fused add+norm | `add_rmsnorm2d_*` | avoid extra bandwidth |
| Output dtype | i8 / fp8 on rdquant | fp8 cvt is CDNA `v_cvt_pk_*` |

## Architecture columns

| | CDNA wave64 | RDNA / gfx1250 wave32 |
|---|---|---|
| Default `wave_size` | 64 | **must be 32** for XOR-butterfly reductions — a wave64 build issues an invalid shuffle |
| fp8 packed convert | yes | i8 out only on `add_rmsnorm2d_rdquant` (gfx1151) |
| Matrix core | unused | unused |

Symptom “wrong answers only on gfx1151 reduce/norm” is almost always
`technique-wave32`, not an algorithm bug. Bandwidth-bound small-ops whose
`vec` / fusion table is exhausted: `process-escape-hatch` (usually fuse into
the producer, not a fancier reduce tree).
