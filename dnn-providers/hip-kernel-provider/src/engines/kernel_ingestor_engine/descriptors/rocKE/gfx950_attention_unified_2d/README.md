# gfx950_attention_unified_2d

Paged unified-attention prefill (vLLM's `unified_attention`) for gfx950, built by rocKE
(`kernels/gfx950/attention_tiled_2d.py`, builder `build_unified_attention_2d_tiled`).
Engine: `hipkernel:Gfx950AttentionUnified2D`.

A hipDNN SDPA graph is dense, so the engine serves the graph's contiguous BSHD K/V as a paged
cache with an identity block table and stages that table, `seq_lens` and `query_start_len` in the
plan's workspace (`packs/Gfx950AttentionUnified2DGeometry.hpp`).

## Coverage

- Architecture: gfx950 only.
- Variants: 792 (all `kernel_source.kind = rocke`, compiled at pack time): 198 cells, each at
  four geometries. Variant reduction on MI355X may drop geometries per cell; the count here is
  the shipped selection's.
- dtype: bf16, fp16. head_size: 64, 128 (D256 prefill is deferred).
- mask: no mask, top-left causal, bottom-right causal (KMD `mask_type` 0, 1, 2).
- Query heads 4 to 128 and KV heads 1 to 128, varying per variant (group of 1 to 16).
- Geometries (llvm backend, `block_size` 16), one registry candidate each; the first is every
  cell's default:
  - narrow 16x16 MFMA, 4 warps x 16 rows, KV tile 64;
  - narrow 16x16 MFMA, 2 warps x 16 rows, KV tile 32;
  - wide 32x32 MFMA, 4 warps x 32 rows, KV tile 64;
  - transposed 32x32 MFMA, 4 warps x 32 rows, KV tile 64.
- Batch capacity 511: `num_seqs` 511 bakes 9 binary-search iterations, which resolve at most
  2^9 - 1 sequences. The engine declines a graph with a larger batch.
- Decode (seqlen_q 1) whose GQA group divides 16 is left to `hipkernel:Gfx950AttentionUnified3D`.
- Fixed in every variant: no sliding window, no sinks, no ALiBi, no softcap, no fp8, 32-bit K/V
  addressing (K/V larger than 0x7FFF0000 bytes is declined).

## Files

| File | In | Notes |
|---|---|---|
| `artifacts/gfx950_attention_unified_2d.kdp.json` | DVC | the 792 variants. Pointer: `artifacts.dvc` (one artifact for the folder) |
| `gfx950_attention_unified_2d.kmd.json` | git | variant fields |
| `gfx950_attention_unified_2d.udd.json` | git | dispatch symbol |
| `gfx950_attention_unified_2d.ued.json` | git | engine, knobs `tile_size`, `num_warps`, `block_m_per_warp`, `use_mfma_32x32`, `use_transposed_qk_32x32` |
| `gfx950_attention_unified_2d.uhd.json` | git | native selector |
| `kernel_dtype_matches_graph.umd.json` | git | dtype matcher |

## Notes

- Fetch `artifacts/` with `dvc pull -r ingestor artifacts.dvc`. See `../../README.md`.
- The native symbols (`hipkernel.gfx950_attention_unified_2d.*`) are registered in
  `packs/Gfx950AttentionUnified2DNative.cpp`.
- Changing the KDP changes its md5. Update the `.dvc` pointer in the same commit.
- Generated from `projects/hipdnn/tools/IngestorGenerator/configs/gfx950_attention_unified_2d.yaml`;
  after the first generation, splice additions rather than copying a regenerated directory over
  this one (every run mints fresh UUIDs).
