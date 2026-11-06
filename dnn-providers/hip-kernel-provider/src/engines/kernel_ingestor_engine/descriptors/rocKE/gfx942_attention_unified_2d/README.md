# gfx942_attention_unified_2d

Paged unified-attention prefill (vLLM's `unified_attention`) for gfx942, built by rocKE
(`kernels/gfx942/attention_tiled_2d.py`, builder `build_unified_attention_2d_tiled`).
Engine: `hipkernel:Gfx942AttentionUnified2D`.

A hipDNN SDPA graph is dense, so the engine serves the graph's contiguous BSHD K/V as a paged
cache with an identity block table and stages that table, `seq_lens` and `query_start_len` in the
plan's workspace (`packs/Gfx942AttentionUnified2DGeometry.hpp`).

## Coverage

- Architecture: gfx942 only.
- Variants: 100 (all `kernel_source.kind = rocke`, compiled at pack time): 20 cells, each at
  five geometries.
- Cells: head size x dtype x GQA ratio. The kernels read the head counts as kernel arguments
  and bake only the ratio (KMD `num_queries_per_kv`), so one variant serves every head pair of
  its ratio.
- dtype: bf16, fp16. head_size: 64, 128. GQA ratio: 1, 2, 4, 8, 16.
- mask: bottom-right causal (KMD `mask_type` 2), the only band the gfx942 tiled 2D body
  computes.
- Geometries (llvm backend, `block_size` 16), one registry candidate each, five per head size;
  the first of each is its cells' default:
  - D64: transposed 32x32x8 MFMA 4 warps x 32 rows KV tile 64; transposed 8 warps x 32 rows
    KV tile 64; narrow 16x16 MFMA 4 warps x 16 rows KV tile 64; narrow 4 warps x 32 rows KV
    tile 64; narrow 2 warps x 16 rows KV tile 64.
  - D128: transposed 32x32x8 MFMA 2 warps x 32 rows KV tile 32; transposed 4 warps x 32 rows
    KV tile 32; narrow 4 warps x 16 rows KV tile 32; narrow 2 warps x 32 rows KV tile 32;
    narrow 2 warps x 16 rows KV tile 16.

  An MI300X sweep of every LDS-legal geometry over 80 prefill shapes, with correctness checked,
  chose them: each head size's five are within 0.6% of the best geometry on every shape.
- Batch: any. The kernel takes the batch as the `num_seqs` kernarg and derives its `cu_q`
  binary-search trip count from it at run time.
- Fixed in every variant: no sliding window, no sinks, no ALiBi, no softcap, no fp8, 32-bit K/V
  addressing (K/V larger than 0x7FFF0000 bytes is declined).

## Files

| File | In | Notes |
|---|---|---|
| `artifacts/gfx942_attention_unified_2d.kdp.json` | DVC | 0.13 MB, the 100 variants. Pointer: `artifacts.dvc` (one artifact for the folder) |
| `gfx942_attention_unified_2d.kmd.json` | git | variant fields |
| `gfx942_attention_unified_2d.udd.json` | git | dispatch symbol |
| `gfx942_attention_unified_2d.ued.json` | git | engine, knobs `tile_size`, `num_warps`, `block_m_per_warp`, `use_mfma_32x32x8`, `use_transposed_qk_32x32` |
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
