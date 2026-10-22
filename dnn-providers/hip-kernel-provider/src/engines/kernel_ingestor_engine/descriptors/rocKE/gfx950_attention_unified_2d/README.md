# gfx950_attention_unified_2d

Paged unified-attention prefill (vLLM's `unified_attention`) for gfx950, built by rocKE
(`kernels/gfx950/attention_tiled_2d.py`, builder `build_unified_attention_2d_tiled`).
Engine: `hipkernel:Gfx950AttentionUnified2D`.

A hipDNN SDPA graph is dense, so the engine serves the graph's contiguous BSHD K/V as a paged
cache with an identity block table and stages that table, `seq_lens` and `query_start_len` in the
plan's workspace (`packs/Gfx950AttentionUnified2DGeometry.hpp`).

## Coverage

- Architecture: gfx950 only.
- Variants: 216 (all `kernel_source.kind = rocke`, compiled at pack time): 198 at head_size 64 and
  128, and 18 at head_size 256.
- dtype: bf16, fp16 at head_size 64 and 128; bf16 only at head_size 256 (fp16 D256 is not served).
- head_size: 64, 128, 256.
- mask: no mask, top-left causal, bottom-right causal (KMD `mask_type` 0, 1, 2) at head_size 64 and
  128; the two causal masks (1, 2) at head_size 256.
- Query heads 4 to 128 and KV heads 1 to 128, varying per variant (group of 1 to 16).
- One geometry per head size: narrow 16x16 MFMA, 4 warps x 16 rows (`block_m` 64) at head_size 64
  and 128; the D256 prefill geometry (transposed 32x32, 2 warps x 32 rows, `block_m` 64, with the
  production route's codegen constellation) at head_size 256. KV tile 64 tokens, `block_size` 16,
  llvm backend in both.
- Batch capacity 511: `num_seqs` 511 bakes 9 binary-search iterations, which resolve at most
  2^9 - 1 sequences. The engine declines a graph with a larger batch.
- Fixed in every variant: no sliding window, no sinks, no ALiBi, no softcap, no fp8.

## Files

| File | In | Notes |
|---|---|---|
| `artifacts/gfx950_attention_unified_2d.kdp.json` | DVC | the 216 variants. Pointer: `artifacts.dvc` (one artifact for the folder) |
| `gfx950_attention_unified_2d.kmd.json` | git | variant fields |
| `gfx950_attention_unified_2d.udd.json` | git | dispatch symbol |
| `gfx950_attention_unified_2d.ued.json` | git | engine, knobs `tile_size`, `num_warps`, `block_m_per_warp` |
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
