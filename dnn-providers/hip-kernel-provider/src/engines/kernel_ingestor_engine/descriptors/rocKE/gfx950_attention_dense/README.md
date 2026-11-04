# gfx950_attention_dense

Dense attention forward for gfx950, built by rocKE (`kernels/gfx950/attention_dense.py`,
builder `build_attention_dense`). Engine: `hipkernel:Gfx950AttentionDense`.

## Coverage

- Architecture: gfx950 only.
- Variants: 972 (all `kernel_source.kind = rocke`, compiled at pack time): 840 without
  additive bias, 132 with it.
- dtype: bf16, fp16.
- head_size: 64, 128.
- causal: both.
- block_m: 128, 256. block_n: 32, 64, 128, 256.
- Additive bias (`use_additive_bias`): every head configuration, dtype and mask, at the
  128/64 tile only. So a bias graph needs `Sq % 128 == 0` and `Skv % 64 == 0`, and a
  forced `block_m`/`block_n` knob other than 128/64 matches no bias kernel.
- `attn_mask`: the engine serves an fp32 tensor `[B|1, Hq|1, Sq|1, Skv]` (extent 1 or
  full on each leading axis, indexed by the query head), real, not ragged and not
  pass-by-value, with a unit-stride key axis and its largest element index below 2^31.
  It declines any other `attn_mask`, with the cause in the INFO log.
  `-inf` (and NaN, or any value below about -1.1e29) masks a key; a row with no
  unmasked key outputs 0, as the hipDNN reference does for a row of `-inf` scores.
- Query heads 4 to 128 and KV heads 1 to 128, varying per variant.
- Fixed in every variant: batch 1, seqlen_q 512, seqlen_kv 512, no sliding window, not ragged,
  not varlen, not paged, not persistent.

## Files

| File | In | Notes |
|---|---|---|
| `artifacts/gfx950_attention_dense.kdp.json` | DVC | 1.5 MB, the 972 variants. Pointer: `artifacts.dvc` (one artifact for the folder) |
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
- Saved plans store their launch values under `hipkernel.gfx950_attention_dense.dispatch.v2`
  (the kernel reads the head counts at runtime), or `.v3` for a plan that serves a bias
  kernel (v2 plus the bias uid and its batch, query-head and query-row strides). Plans
  saved under `.v1` (head counts baked into the kernel) still load. All three stay
  registered so saved plans keep loading.
