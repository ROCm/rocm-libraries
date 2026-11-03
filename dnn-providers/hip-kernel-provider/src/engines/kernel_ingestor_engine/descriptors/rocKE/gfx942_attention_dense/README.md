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
  keys (graph `left_bound` 128, 512, 1024 and 2048: `left_bound` L keeps L keys, the
  diagonal included, SdpaSemanticsRegister SEM-001). A window with
  seqlen_q - seqlen_kv >= W is declined: its last query rows see no key.
- Stats (LSE): every variant has an `emit_lse` twin (KMD field `emit_lse` 1, name slot
  `_lse1`) that also writes the natural-log softmax normalizer through a trailing
  `lse_ptr`. A graph whose stats output is the packed FP32 [B, Hq, Sq, 1] tensor is served
  only by the twins, and every other graph only by the `emit_lse` 0 variants. Max,
  Sum_exp, the dropout RNG dump and any other stats layout or dtype are declined.
- Knobs: `block_m`, `block_n`, `lds_row_pad` and `pv_priority` (the UED's), every field
  that varies within a cell; the native score ranks all four.
- Batch, sequence lengths and both head counts are runtime kernel arguments, so every
  variant serves any shape its tile divides. Each variant is built at batch 1,
  seqlen_q = seqlen_kv = 512 and 8 query/KV heads.
- Fixed in every variant: not ragged, not varlen, not paged, no sinks, and the grid the
  native dispatch launches (non-persistent, `qb_minor` block order, one KV
  segment), which the registry candidate `gfx942_dense_qbminor` holds fixed.

## Files

| File | In | Notes |
|---|---|---|
| `artifacts/gfx942_attention_dense.kdp.json` | DVC | the variants. Pointer: `artifacts.dvc` (one artifact for the folder) |
| `gfx942_attention_dense.kmd.json` | git | variant fields |
| `gfx942_attention_dense.udd.json` | git | dispatch symbol |
| `gfx942_attention_dense.ued.json` | git | engine, knobs `block_m`, `block_n`, `lds_row_pad`, `pv_priority` |
| `gfx942_attention_dense.uhd.json` | git | native selector |
| `kernel_dtype_matches_graph.umd.json` | git | dtype matcher |

## Notes

- Fetch `artifacts/` with `dvc pull -r ingestor artifacts.dvc`. See
  `../../README.md`.
- The native symbols (`hipkernel.gfx942_attention_dense.*`) are registered in
  `packs/Gfx942AttentionDenseNative.cpp`.
- Changing the KDP changes its md5. Update the `.dvc` pointer in the same commit.
