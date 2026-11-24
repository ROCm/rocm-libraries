# gfx950_attention_dense

Dense attention forward for gfx950, built by rocKE (`kernels/gfx950/attention_dense.py`,
builder `build_attention_dense`). Engine: `hipkernel:Gfx950AttentionDense`.

## Coverage

- Architecture: gfx950 only.
- Variants: 60 (all `kernel_source.kind = rocke`, compiled at pack time): 52 without
  additive bias, 8 with it.
- dtype: bf16, fp16.
- head_size: 64, 128.
- causal: both.
- block_m: 128, 256. block_n: 32, 64, 128, 256.
- Additive bias (`use_additive_bias`): every head size, dtype and mask, at the 128/64 tile
  only. So a bias graph needs `Sq % 128 == 0` and `Skv % 64 == 0`, and a forced
  `block_m`/`block_n` knob other than 128/64 matches no bias kernel.
- `attn_mask`: the engine serves an fp32 tensor `[B|1, Hq|1, Sq|1, Skv]` (extent 1 or
  full on each leading axis, indexed by the query head), real, not ragged and not
  pass-by-value, with a unit-stride key axis and its largest element index below 2^31.
  It declines any other `attn_mask`, with the cause in the INFO log.
  `-inf` masks a key; a row with no unmasked key outputs 0, as the hipDNN reference
  does for a row of `-inf` scores. A finite value, `finfo(float32).min` included, is a
  score like any other (a row of them gets the uniform softmax), and a NaN on a visible
  key makes its row NaN, as in the reference.
- ALiBi (`use_alibi`, hipDNN's `alibi_mask` with the slopes computed in the kernel): the 26 kernels
  (the causal plain variants of this bundle, built with ALiBi) live in the satellite bundle
  `../gfx950_attention_dense_alibi/` (same engine and KMD, KDP of its own).
- ALiBi together with an `attn_mask`: no kernel adds both, so the engine declines such a
  graph under the cause `alibi_mask`.
- Query heads and KV heads: any. They are not variant fields -- the kernel reads
  `num_query_heads` / `num_kv_heads` as kernargs from the graph, so one variant per
  (dtype, head_size, causal, tile) serves every head configuration, the bias variants
  included. Each variant is built at the canonical 8 / 8 (a spec field, not metadata).
- Fixed in every variant: batch 1, seqlen_q 512, seqlen_kv 512, no sliding window, not ragged,
  not varlen, not paged, not persistent.

## Files

| File | In | Notes |
|---|---|---|
| `artifacts/gfx950_attention_dense.kdp.json` | DVC | 92 KB, the 60 variants. Pointer: `artifacts.dvc` (one artifact for the folder) |
| `gfx950_attention_dense.kmd.json` | git | variant fields |
| `gfx950_attention_dense.udd.json` | git | dispatch symbol |
| `gfx950_attention_dense.ued.json` | git | engine, knobs `block_m`, `block_n` |
| `gfx950_attention_dense.uhd.json` | git | native selector |
| `kernel_dtype_matches_graph.umd.json` | git | dtype matcher |

## Notes

- Fetch `artifacts/` with `dvc pull -r ingestor artifacts.dvc`. See
  `../../README.md`.
- The native symbols (`hipkernel.gfx950_attention_dense.*`) are registered in
  `packs/Gfx950AttentionDenseNative.cpp`, which binds this engine's arch descriptor
  (tile rules, launch contracts, the `use_additive_bias` field). Matching, dispatch and the
  saved launch values, the additive bias included, are shared with the gfx942 engine in
  `packs/AttentionDenseNative.cpp`.
- Changing the KDP changes its md5. Update the `.dvc` pointer in the same commit.
- Saved plans store their launch values under `hipkernel.gfx950_attention_dense.dispatch.v2`
  (the kernel reads the head counts at runtime), or `.v3` for a plan that serves a bias
  kernel (v2 plus the bias uid and its batch, query-head and query-row strides). Plans
  saved under `.v1` (head counts baked into the kernel) still load. All three stay
  registered so saved plans keep loading.
- **Two rankers, and which one decides depends on the arch.** The UED binds
  `sort_kernel_catalog` as `default` (the native UHD) plus a `gfx950` list holding the
  trained `tflops` and `time` models, installed under
  `heuristics/4b3a0123-578f-4e9c-a965-b010a18ff107/sort_kernel_catalog/gfx950/<metric>/`.
  This pack ships for gfx950 alone, so the trained models are what rank it in a complete
  build. The native `score` symbol is the fallback, reached when no model loads, which in
  practice means a build whose `model.bin` was never pulled. The two do not always agree,
  and the knob suite's cold cases pin where they differ.
- **The trained models cover the 60-kernel catalog, bias kernels included, but were not
  shown a bias graph.** They were retrained on this catalog (52 collapsed kernels plus 8
  additive-bias kernels), and `engine.trained_catalog_sha256` pins it. The corpus tool
  cannot yet emit a graph that carries an `attn_mask`, so no training graph has a bias
  candidate (AICK-3322). That does not change a choice today: the 8 bias kernels are one
  per head size, dtype and mask, all at 128/64, so a graph with an `attn_mask` has one
  bias kernel to pick. It starts to matter when a second bias tile is added, which needs
  that corpus option and another retrain.
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
