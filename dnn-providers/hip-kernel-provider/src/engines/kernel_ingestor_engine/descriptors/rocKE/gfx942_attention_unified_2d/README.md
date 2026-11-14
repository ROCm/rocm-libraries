# gfx942_attention_unified_2d

Paged unified-attention prefill (vLLM's `unified_attention`) for gfx942, built by rocKE
(`kernels/gfx942/attention_tiled_2d.py`, builder `build_unified_attention_2d_tiled`).
Engine: `hipkernel:Gfx942AttentionUnified2D`.

A hipDNN SDPA graph is dense, so the engine serves the graph's contiguous BSHD K/V as a paged
cache with an identity block table and stages that table, `seq_lens` and `query_start_len` in the
plan's workspace (`packs/Gfx942AttentionUnified2DGeometry.hpp`).

## Coverage

- Architecture: gfx942 only.
- Variants: 50 (all `kernel_source.kind = rocke`, compiled at pack time). The selection is
  `variant_reduce`'s cut of a 120-variant full space (20 cells, each at five geometries plus
  rocKE's production stack for its dtype and head size): every cell keeps its default
  geometry plus the alternatives a correctness-checked gfx942 sweep measured as needed. The
  production stack (mask limit, hoist, conflict-free V stores, K ring; direct-global Q at D64)
  is what rocKE's own dispatcher builds for a long prefill; it ships in every fp16 cell and
  every bf16 D64 cell, and only the default geometry ships at bf16 D128, where it lost on
  every sweep graph.
- Cells: head size x dtype x GQA ratio. The kernels read the head counts as kernel arguments
  and bake only the ratio (KMD `num_queries_per_kv`), so one variant serves every head pair of
  its ratio.
- dtype: bf16, fp16. head_size: 64, 128. GQA ratio: 1, 2, 4, 8, 16.
- mask: bottom-right causal (KMD `mask_type` 2), the only band the gfx942 tiled 2D body
  computes.
- Geometries (llvm backend, `block_size` 16), one registry candidate each, five per head size
  in the full space; the first of each is its cells' default:
  - D64: transposed 32x32x8 MFMA 4 warps x 32 rows KV tile 64; transposed 8 warps x 32 rows
    KV tile 64 (also built with direct Q loads and the conflict-free V store); narrow 16x16
    MFMA 4 warps x 16 rows KV tile 64; narrow 4 warps x 32 rows KV tile 64; narrow 2 warps x
    16 rows KV tile 64.
  - D128: transposed 32x32x8 MFMA 2 warps x 32 rows KV tile 32; transposed 4 warps x 32 rows
    KV tile 32; narrow 4 warps x 16 rows KV tile 32; narrow 2 warps x 32 rows KV tile 32;
    narrow 2 warps x 16 rows KV tile 16.

  A correctness-checked gfx942 sweep of every LDS-legal geometry over 80 prefill shapes, run
  through rocKE's own launcher, chose the full space's five per head size.
- Batch: any. The kernel takes the batch as the `num_seqs` kernarg and derives its `cu_q`
  binary-search trip count from it at run time.
- Fixed in every variant: no sliding window, no sinks, no ALiBi, no softcap, no fp8, 32-bit K/V
  addressing (K/V larger than 0x7FFF0000 bytes is declined).

## Files

| File | In | Notes |
|---|---|---|
| `artifacts/gfx942_attention_unified_2d.kdp.json` | DVC | 0.07 MB, the 50 variants. Pointer: `artifacts.dvc` (one artifact for the folder) |
| `gfx942_attention_unified_2d.kmd.json` | git | variant fields |
| `gfx942_attention_unified_2d.udd.json` | git | dispatch symbol |
| `gfx942_attention_unified_2d.ued.json` | git | engine, knobs `tile_size`, `num_warps`, `block_m_per_warp`, `use_mfma_32x32x8`, `use_transposed_qk_32x32`, `use_k_sliced_ring`, `use_q_direct_global`, `use_conflict_free_v_store`, `use_transposed_mask_limit`, `use_transposed_invariant_hoist` |
| `gfx942_attention_unified_2d.uhd.json` | git | native selector |
| `kernel_dtype_matches_graph.umd.json` | git | dtype matcher |

## Notes

- Fetch `artifacts/` with `dvc pull -r ingestor artifacts.dvc`. See `../../README.md`.
- The native symbols (`hipkernel.gfx942_attention_unified_2d.*`) are registered in
  `packs/Gfx942AttentionUnified2DNative.cpp`.
- Changing the KDP changes its md5. Update the `.dvc` pointer in the same commit.
- Generated from `projects/hipdnn/tools/IngestorGenerator/configs/gfx942_attention_unified_2d.yaml`.
  A regeneration keeps descriptor ids stable: `generate.py` reuses the ids of this installed
  bundle by default (generator README, "Descriptor ids"). Generate into scratch space and
  splice additions with `tools/splice_additions.py` rather than copying a regenerated
  directory over this one, because the C++ files are regenerated from scratch and the KDP is
  DVC-backed.
