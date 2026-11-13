# gfx950_attention_dense

Dense attention forward for gfx950, built by rocKE (`kernels/gfx950/attention_dense.py`,
builder `build_attention_dense`). Engine: `hipkernel:Gfx950AttentionDense`.

## Coverage

- Architecture: gfx950 only.
- Variants: 52 (all `kernel_source.kind = rocke`, compiled at pack time).
- dtype: bf16, fp16.
- head_size: 64, 128.
- causal: both.
- block_m: 128, 256. block_n: 32, 64, 128, 256.
- Query heads and KV heads: any. They are not variant fields -- the kernel reads
  `num_query_heads` / `num_kv_heads` as kernargs from the graph, so one variant per
  (dtype, head_size, causal, tile) serves every head configuration. Each variant is
  built at the canonical 8 / 8 (a spec field, not metadata).
- Fixed in every variant: batch 1, seqlen_q 512, seqlen_kv 512, no sliding window, not ragged,
  not varlen, not paged, not persistent.

## Files

| File | In | Notes |
|---|---|---|
| `artifacts/gfx950_attention_dense.kdp.json` | DVC | 76 KB, the 52 variants. Pointer: `artifacts.dvc` (one artifact for the folder) |
| `gfx950_attention_dense.kmd.json` | git | variant fields |
| `gfx950_attention_dense.udd.json` | git | dispatch symbol |
| `gfx950_attention_dense.ued.json` | git | engine, knobs `block_m`, `block_n` |
| `gfx950_attention_dense.uhd.json` | git | native selector |
| `kernel_dtype_matches_graph.umd.json` | git | dtype matcher |

## Notes

- Fetch `artifacts/` with `dvc pull -r ingestor artifacts.dvc`. See
  `../../README.md`.
- The native symbols (`hipkernel.gfx950_attention_dense.*`) are registered in
  `packs/Gfx950AttentionDenseNative.cpp`.
- Changing the KDP changes its md5. Update the `.dvc` pointer in the same commit.
- **Two rankers, and which one decides depends on the arch.** The UED binds
  `sort_kernel_catalog` as `default` (the native UHD) plus a `gfx950` list holding the
  trained `tflops` and `time` models, installed under
  `heuristics/4b3a0123-578f-4e9c-a965-b010a18ff107/sort_kernel_catalog/gfx950/<metric>/`.
  This pack ships for gfx950 alone, so the trained models are what rank it in a complete
  build. The native `score` symbol is the fallback, reached when no model loads, which in
  practice means a build whose `model.bin` was never pulled. The two do not always agree,
  and the knob suite's cold cases pin where they differ.
- **The native symbol reads the problem.** `preferredTile()` picks a tile from head size,
  mask and `seqlen_q`, and candidates are ranked by closeness to it, with the historical
  ascending `(block_m, block_n)` order as the tie-break. It is no longer a fixed
  256/64-first order. See
  `rocke/library/builders/gfx950/attention/prefill/TILE_SELECTION_CASE_STUDY.md`.
- **Regenerating the KDP now has a UHD consequence.** `engine.trained_catalog_sha256`
  fingerprints the catalog the trained models were measured against, so any kernel added,
  removed or renamed invalidates it and the generator's gate fails until the models are
  retrained and the fingerprint updated. Make the retain/retrain/drop choice explicitly
  (see the `hipdnn-ingestor-engine` RUNBOOK, "Trained heuristics survive regeneration
  only by intent") rather than regenerating past it.
