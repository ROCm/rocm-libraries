# gfx942_attention_dense

Dense attention forward for gfx942, built by rocKE (`kernels/gfx942/attention_dense.py`,
builder `build_attention_dense`). Engine: `hipkernel:Gfx942AttentionDense`.

## Coverage

- Architecture: gfx942 only.
- Variants: all `kernel_source.kind = rocke`, compiled at pack time. The selection is
  `projects/hipdnn/tools/IngestorGenerator/configs/gfx942_attention_dense.yaml`.
- dtype: bf16, fp16.
- head_size: 64, 128.
- Mask: causal (top-left), unmasked, and causal sliding windows of 128, 512, 1024 and 2048
  keys (graph `left_bound` 127, 511, 1023 and 2047).
- Knobs: `block_m`, `block_n`, `lds_row_pad`, `pv_priority`.
- Batch, sequence lengths and both head counts are runtime kernel arguments, so every
  variant serves any shape its tile divides. Each variant is built at batch 1,
  seqlen_q = seqlen_kv = 512 and 8 query/KV heads.
- Fixed in every variant: not ragged, not varlen, not paged, not persistent, no sinks, no
  LSE output.

## Files

| File | In | Notes |
|---|---|---|
| `artifacts/gfx942_attention_dense.kdp.json` | DVC | the variants. Pointer: `artifacts.dvc` (one artifact for the folder) |
| `gfx942_attention_dense.kmd.json` | git | variant fields |
| `gfx942_attention_dense.udd.json` | git | dispatch symbol |
| `gfx942_attention_dense.ued.json` | git | engine, knobs `block_m`, `block_n` |
| `gfx942_attention_dense.uhd.json` | git | native selector |
| `kernel_dtype_matches_graph.umd.json` | git | dtype matcher |

## Notes

- Fetch `artifacts/` with `dvc pull -r ingestor artifacts.dvc`. See
  `../../README.md`.
- The native symbols (`hipkernel.gfx942_attention_dense.*`) are registered in
  `packs/Gfx942AttentionDenseNative.cpp`.
- Changing the KDP changes its md5. Update the `.dvc` pointer in the same commit.
