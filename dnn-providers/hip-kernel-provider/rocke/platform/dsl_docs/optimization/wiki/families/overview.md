---
id: family-overview
title: "Operator × architecture routing tables"
type: family
operator_families: [gemm, attention, convolution, moe, small-ops]
architecture_families: [cdna, rdna, gfx12]
tags: [routing, common]
related:
  - family-gemm
  - family-attention
  - family-convolution
  - family-moe
  - family-small-ops
  - process-routing
  - hw-cdna
  - hw-rdna
  - process-escape-hatch
  - hw-gfx1250
  - migration-gfx950-to-gfx1250
sources:
  - project-rocke
  - project-hipblaslt
  - project-tensilelite
  - project-stinkytofu
  - project-composablekernel
---

# Route a problem to a technique

Pick the **operator family** (row) and the **architecture family** (column).
Common techniques apply across a row. Arch-specific techniques apply in one
column. Exact gfx facts live on `hw-gfx*` pages; this table is the dispatcher.

Hardware capacities below are from `platform/python/rocke/core/arch/data/arch_specs.json`.

## Architecture families (hardware)

| | CDNA MFMA (`gfx90a`/`gfx942`/`gfx950`) | RDNA WMMA (`gfx1151`/`gfx1201`) | gfx1250 (CDNA product, GFX12 ISA) |
|---|---|---|---|
| Wave | 64 | 32 | 32 |
| Matrix engine | MFMA (`hw-mfma`) | WMMA (`hw-wmma`) | WMMA 16×16×32 (`technique-gfx1250-wmma-k32`) |
| AGPR | yes (`technique-agpr-acc`) | none | none |
| LDS | 64 KiB (gfx942) / 160 KiB (gfx950), 32/64 banks | 64 KiB | 160 KiB |
| Async DRAM→LDS | `buffer_load_lds` on gfx942+ (`technique-async-copy`) | no | `global_load_async_to_lds_*` (`technique-gfx12-async-lds`) |
| Transpose LDS read | gfx950 `ds_read_*_tr_*` only | no | `ds_load_tr16_b128` (distinct opcode) |
| Default rocke pipeline | `mem` / `compv3` / `compv4` | `mem` + `default` epilogue only | WMMA builders, not gfx950 MFMA pipeline bits |

## Common technologies (every operator family)

These are the levers to try **before** an arch-specific rewrite. Each cell is a
`technique-*` id.

| Technology | GEMM | Attention | Convolution | MoE | Small-ops |
|---|---|---|---|---|---|
| Tile / CTA geometry | `technique-tiling` | same (`num_warps`, `tile_size`) | `block_q` / `block_groups` | expert tile + grouped GEMM | vector width / block size |
| Software pipeline / double buffer | `compv4` (`technique-software-pipeline`) | K/V ring, early-V | H-row + ping-pong LDS | GEMM stages | rarely worth it |
| LDS layout / bank pad | `technique-lds-swizzle` | QK/V tiles | K-pad, row layout | scale/weight tiles | block reduce LDS |
| Wide global I/O | `technique-vectorized-io` | KV cache loads | NHWC vector loads | expert gather | `vec` 2/4/8 |
| Occupancy vs VGPR/LDS | `technique-occupancy` | waves/CU on prefill | same | same | usually not the limiter |
| Epilogue | direct vs cshuffle (`technique-epilogue`) | online softmax + store | wide store | silu-mul fusion | vector store |
| Persistent / Stream-K | `technique-persistent-streamk` | split-KV decode | — | grouped persistent | — |
| Fusion | epilogue fusion (`technique-fusion`) | QK+softmax+PV | conv+bias+act+pool | gate-up + SwiGLU | add+rmsnorm |
| ISA / resource inspect | `technique-isa-inspect` | same | same | same | same |

## Architecture-specific techniques

| Technique | CDNA | RDNA | gfx1250 |
|---|---|---|---|
| Matrix atom pick | `technique-mfma-atom` | `technique-wmma-atom` | `technique-gfx1250-wmma-k32` |
| K-pack f16 16×16×32 / 32×32×16 | gfx950 only | n/a | WMMA K=32 is the default atom |
| Accumulators in AGPR | `technique-agpr-acc` | n/a (VGPR only) | n/a — VGPR only (`hw-vgpr-acc-gfx1250`) |
| Transpose LDS read | `technique-ds-read-tr` (gfx950) | n/a | `technique-gfx1250-ds-load-tr` |
| Chiplet / XCD grid swizzle | `technique-chiplet-swizzle` | n/a | device-specific |
| Wave size in reductions | wave64 XOR tree | `technique-wave32` (must set `wave_size=32`) | wave32 |
| Async copy opcode | gfx9 `buffer_load_lds` | none in catalog | `technique-gfx12-async-lds` + `technique-gfx1250-asynccnt-pipeline` |
| Waitcnt | monolithic `s_waitcnt` | gfx11 waitcnt | `technique-gfx1250-split-waitcnt` |
| Porting | — | — | `migration-gfx950-to-gfx1250` / `migration-gfx1201-to-gfx1250` |

## Next hop

1. `get_page.py family-<operator>` for the operator table (knobs + rocke specs).
2. `query.py --symptom <from probe> --architecture <gfx>`.
3. `get_page.py hw-<gfx>` for the atom / LDS / occupancy caps.
4. If the table cannot move the limiter: `process-escape-hatch`.
5. gfx1250 port or Blackwell-shaped question: `get_page.py hw-gfx1250` (analog map), then `query.py --type migration --architecture gfx1250`.
