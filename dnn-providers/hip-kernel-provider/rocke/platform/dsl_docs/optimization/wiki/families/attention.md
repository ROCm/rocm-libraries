---
id: family-attention
title: "Attention / FMHA family — common tech × architecture"
type: family
operator_families: [attention]
architecture_families: [cdna, rdna, gfx12]
tags: [attention, routing]
related:
  - family-overview
  - kernel-attention-2d
  - technique-mfma-atom
  - technique-software-pipeline
  - technique-ds-read-tr
  - technique-wave32
  - process-escape-hatch
  - kernel-wmma-attention-gfx1250
  - migration-gfx950-to-gfx1250
sources:
  - project-rocke
  - project-composablekernel
---

# Attention

Rocke: scalar `UnifiedAttention*Spec` plus tiled
`library/kernels/gfx950/attention_tiled_2d.py` (and gfx942 / gfx1151 / gfx1250
variants). FMHA family lives under `instances/common/fmha_*.py` and CK Tile 01
ports. Path select: `select_2d_config` / `select_3d_config` (2D prefill, 3D
split-KV decode).

## Common levers

| Lever | Where | Notes |
|---|---|---|
| 2D vs 3D vs reduce | dispatcher | decode / long KV → 3D; chunked prefill / window → 2D |
| `num_warps`, `tile_size`, `block_m_per_warp` | tiled spec | geometry; illegal combos fail `__post_init__` |
| QK/PV atom | `use_mfma_32x32` (CDNA) | 32×32 matches C→A layout; 16×16 pays permute |
| Transposed softmax | `use_transposed_qk_32x32` | drops a cross-lane reduction |
| K/V buffering | single vs double buffer flags | default-off flags are Step 0 sweep items |
| Softmax–MFMA interleave | `use_softmax_mfma_interleave` | default-off; not in the preset-bench grammar |
| Early V | `use_early_v_schedule` | overlap V copy with QK+softmax |
| Split-KV | 3D + reduce kernel | long decode |

## Architecture columns

| | gfx942 | gfx950 | gfx1151 | gfx1250 |
|---|---|---|---|---|
| Tiled MFMA 2D | yes (no f16 K-pack 32) | yes + `ds_read_tr` + 32×32×16 | WMMA FMHA (`wmma_fmha_fwd`) | WMMA attention builders |
| Scalar unified 2D/3D | yes | yes | yes (wave32) | yes |
| `fmha_varlen` / sage / sparse | MFMA atom lookup | yes | often rejected | per-instance |
| Occupancy picture | 64 KB LDS | 160 KB LDS, 4 waves/CU common on fat tiles | VGPR cap 256 | wave32 + 160 KB LDS |

Sweep **raw flags**, not only named presets — Step 0 in `process-optimization-loop`.
If the flags and this table cannot move the limiter: `process-escape-hatch`.
