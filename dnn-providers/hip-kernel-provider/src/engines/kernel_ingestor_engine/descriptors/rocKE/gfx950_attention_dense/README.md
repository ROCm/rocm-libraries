# gfx950_attention_dense

Dense attention forward for gfx950, built by rocKE (`kernels/gfx950/attention_dense.py`,
builder `build_attention_dense`). Engine: `hipkernel:Gfx950AttentionDense`.

## Coverage

- Architecture: gfx950 only.
- Variants: 840 (all `kernel_source.kind = rocke`, compiled at pack time).
- dtype: bf16, fp16.
- head_size: 64, 128.
- causal: both.
- block_m: 128, 256. block_n: 32, 64, 128, 256.
- Query heads 4 to 128 and KV heads 1 to 128, varying per variant.
- Fixed in every variant: batch 1, seqlen_q 512, seqlen_kv 512, no sliding window, not ragged,
  not varlen, not paged, not persistent.

## Files

| File | In | Notes |
|---|---|---|
| `artifacts/gfx950_attention_dense.kdp.json` | DVC | 1.2 MB, the 840 variants. Pointer: `artifacts.dvc` (one artifact for the folder) |
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
