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
- **No trained heuristic.** The UED binds `sort_kernel_catalog: {default: <native UHD>}`
  only -- no per-arch lists, no `predict_engine`, and no installed `heuristics/<ued-id>/`
  subtree -- so ranking is the native `score` symbol (baseline 256/64 first, then
  ascending tile). The UHD retain/retrain/drop choice a regeneration forces (see the
  `hipdnn-ingestor-engine` RUNBOOK, "Trained heuristics survive regeneration only by
  intent") is therefore a no-op here: there is no model to invalidate. Record that again
  if a model is ever promoted for this engine.
