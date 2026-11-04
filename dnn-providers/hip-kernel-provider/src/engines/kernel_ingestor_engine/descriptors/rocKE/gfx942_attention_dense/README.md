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
- Launch grids: the one-CTA-per-block grid at the `qb_minor` block order (every cell), at
  `bt_hkv_minor` (the causal cells) and at `hq_minor_swz` (bf16 D128 causal), and at D128
  the persistent grid: 304 CTAs (`num_persistent`) striding over the work items in a
  pinned `persist_decode` order (`bt_hkv_minor` where the decodes balance causal work at
  the build shape, `qb_major` for no mask and the 128-key window; plus `hq_minor_swz` in
  bf16 D128 causal), with a fast-division magic/shift pair per decode divisor as kernel
  arguments. KMD fields `persistent`, `num_persistent`, `persist_decode`,
  `nonpersist_decode`. `hq_minor_swz` serves only graphs with Hq a multiple of 8 (and on
  the persistent grid an even query-block count under causal balancing); a persistent
  decode only where its fast-division dividends stay below 2^30. The cold score prefers
  the persistent grid at D128 without a narrow window when 304 is a multiple of the
  device's CU count, and the `qb_minor` grid otherwise.
- Body levers (KMD fields, no launch change): `lds_row_pad`, `pv_priority`,
  `lds_num_buffers`, `v_reg_prefetch`, `causal_diag_split`, `o_store_width`,
  `pv_loop_order`, `use_cfvst`, `use_exp2_fast` and `lds_k_group_pad`. `use_cfvst` and
  `use_exp2_fast` carry the value the kernel resolves (its policy depends on dtype, head
  size and grid), so they have no KMD default. The shipped selection holds every body
  lever at its policy except `lds_row_pad`, `pv_priority`, `lds_num_buffers` and
  `v_reg_prefetch`; the manifest's full enumeration (the sweep catalog) moves all of
  them. `window_wave_skip` and the narrow-window tile stay at their defaults: window cells
  are floor-only.
- Knobs: `block_m`, `block_n`, `lds_row_pad`, `pv_priority`, `persistent`,
  `persist_decode`, `nonpersist_decode`, `lds_num_buffers`, `v_reg_prefetch`,
  `causal_diag_split`, `o_store_width`, `pv_loop_order`, `use_cfvst`, `use_exp2_fast` and
  `lds_k_group_pad` (the UED's; the string ones are pinned by ordinal), every field that
  varies within a cell of the full enumeration; the native score ranks all of them, the
  policy arm of each first.
- Batch, sequence lengths and both head counts are runtime kernel arguments, so every
  variant serves any shape its tile divides. Each variant is built at batch 1,
  seqlen_q = seqlen_kv = 512 and 8 query/KV heads.
- Fixed in every variant: not ragged, not varlen, not paged, no sinks, one KV segment.
  Each grid is held fixed by its registry candidate: `gfx942_dense_qbminor`,
  `gfx942_dense_btminor_aot`, `gfx942_dense_hqswz_aot` and `gfx942_dense_persist_aot`.

## Files

| File | In | Notes |
|---|---|---|
| `artifacts/gfx942_attention_dense.kdp.json` | DVC | the variants. Pointer: `artifacts.dvc` (one artifact for the folder) |
| `gfx942_attention_dense.kmd.json` | git | variant fields |
| `gfx942_attention_dense.udd.json` | git | dispatch symbol |
| `gfx942_attention_dense.ued.json` | git | engine, knobs `block_m`, `block_n`, `lds_row_pad`, `pv_priority`, `persistent`, `persist_decode`, `nonpersist_decode`, `lds_num_buffers`, `v_reg_prefetch`, `causal_diag_split`, `o_store_width`, `pv_loop_order`, `use_cfvst`, `use_exp2_fast`, `lds_k_group_pad` |
| `gfx942_attention_dense.uhd.json` | git | native selector |
| `kernel_dtype_matches_graph.umd.json` | git | dtype matcher |

## Notes

- Fetch `artifacts/` with `dvc pull -r ingestor artifacts.dvc`. See
  `../../README.md`.
- The native symbols (`hipkernel.gfx942_attention_dense.*`) are registered in
  `packs/Gfx942AttentionDenseNative.cpp`.
- Changing the KDP changes its md5. Update the `.dvc` pointer in the same commit.
