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
  seqlen_q - seqlen_kv >= W, whose last query rows see no key, is declined as this pack's
  own policy; `windowLeavesRowsWithoutKeys` in `packs/AttentionDenseNative.cpp` says why.
  An installed pack built before the kernel change still holds the old code.
- Stats (LSE): the `emit_lse` variants (KMD field `emit_lse` 1, name slot `_lse1`) also
  write the natural-log softmax normalizer through a trailing `lse_ptr`. A graph whose
  stats output is the packed FP32 [B, Hq, Sq, 1] tensor is served only by them, and every
  other graph only by the `emit_lse` 0 variants. The LSE cells are reduced on their own
  timings, so they need not mirror the non-LSE cells. Max, Sum_exp, the dropout RNG dump
  and any other stats layout or dtype are declined.
- Launch grids: the one-CTA-per-block grid at the `qb_minor` block order (every cell),
  `bt_hkv_minor` and `hq_minor_swz` where the reduction kept them, and at D128 without a
  window the persistent grid: 304 CTAs (`num_persistent`) striding over the work items in
  a pinned `persist_decode` order (`qb_major`, `bt_hkv_minor`, `hkv_major` or
  `hq_minor_swz`, as kept), with a fast-division magic/shift pair per decode divisor as
  kernel arguments. KMD fields `persistent`, `num_persistent`, `persist_decode`,
  `nonpersist_decode` and `kv_segments` (always 1: split KV is not served).
  `hq_minor_swz` serves only graphs with Hq a multiple of 8 (and on the persistent grid an
  even query-block count under causal balancing); a persistent decode only where its
  fast-division dividends stay below 2^30. The cold score prefers
  the persistent grid at D128 without a narrow window when 304 is a multiple of the
  device's CU count, and the `qb_minor` grid otherwise.
- Body levers (KMD fields, no launch change): `lds_row_pad`, `pv_priority`,
  `lds_num_buffers`, `v_reg_prefetch`, `causal_diag_split`, `o_store_width`,
  `pv_loop_order`, `use_cfvst`, `use_exp2_fast` and `lds_k_group_pad`. `use_cfvst` and
  `use_exp2_fast` carry the value the kernel resolves (its policy depends on dtype, head
  size and grid), so they have no KMD default. The shipped selection moves `lds_row_pad`,
  `pv_priority`, `v_reg_prefetch`, `causal_diag_split`, `lds_k_group_pad` and
  `use_exp2_fast` off their policy where the reduction kept such a variant, and holds the
  rest at it; the manifest's full enumeration (the sweep catalog) moves all of them.
  `window_wave_skip` and the narrow-window tile stay at their defaults: window cells are
  floor-only.
- Knobs: `block_m`, `block_n`, `lds_row_pad`, `pv_priority`, `persistent`,
  `persist_decode`, `nonpersist_decode`, `lds_num_buffers`, `v_reg_prefetch`,
  `causal_diag_split`, `o_store_width`, `pv_loop_order`, `use_cfvst`, `use_exp2_fast` and
  `lds_k_group_pad` (the UED's; the string ones are pinned by ordinal), every field that
  varies within a cell of the full enumeration; the native score ranks all of them, the
  policy arm of each first.
- Batch, sequence lengths and both head counts are runtime kernel arguments, so every
  variant serves any shape its tile divides. Each variant is built at batch 1,
  seqlen_q = seqlen_kv = 512 and 8 query/KV heads.
- Each grid is held fixed by its registry candidate: `gfx942_dense_qbminor`,
  `gfx942_dense_btminor_aot`, `gfx942_dense_hqswz_aot` and `gfx942_dense_persist_aot`.
- Not covered by this engine (follow-ups). A graph-level gap is declined with an INFO line
  naming the cause. A window size the catalog lacks, or a sequence length no shipped tile
  divides, passes the graph match and then matches no kernel.
  - Not yet supported by this pack; the gfx942 kernel builds it, no variant ships it:
    - Attention sinks (`use_sinks`; gpt-oss). When this work started, hipDNN's SDPA
      references could not validate sinks; #14358 added that on 2026-11-02. gpt-oss's W=128
      layers need sink variants of the shipped W=128 cells.
    - Bottom-right causal with seqlen_q != seqlen_kv (`causal_bottom_right`).
    - Split KV (`kv_segments` > 1; needs a workspace and a combine launch).
    - Window rows that see no key (seqlen_q - seqlen_kv >= W), with or without a stats
      output: see the Mask entry above.
  - Not in the catalog: sliding windows other than 128, 512, 1024 and 2048 keys (W is a
    compile-time parameter). W=4096 (Mistral-7B-v0.1, StarCoder2, Command R7B, OLMo 3,
    EXAONE 4.0; all D128) is the first to add.
  - Not implemented in the gfx942 kernel: sequence lengths no shipped tile divides (no
    partial-tile path), varlen, ragged (packed) tensors and paged K/V (other rocKE kernels
    serve them), bottom-right windows, non-causal bands (`right_bound` > 0), head sizes
    other than 64 and 128, FP32 and FP8 data types, dropout, additive / ALiBi / block
    masks, scale tensors, and the Max and Sum_exp outputs. The shipped tiles are block_m
    64, 128, 256 and 512 with block_n 32, 64 and 128, per cell as listed in the selection.
  - Addressing limit: K/V under 2^31 bytes and Q/O under 2^31 elements (32-bit offsets).
  - Design limit: FLOAT accumulation, so a DOUBLE compute type is declined.

## Files

| File | In | Notes |
|---|---|---|
| `artifacts/gfx942_attention_dense.kdp.json` | DVC | the variants. Pointer: `artifacts.dvc` (one artifact for the folder) |
| `gfx942_attention_dense.kmd.json` | git | variant fields |
| `gfx942_attention_dense.udd.json` | git | dispatch symbol |
| `gfx942_attention_dense.ued.json` | git | engine, knobs `block_m`, `block_n`, `lds_row_pad`, `pv_priority`, `persistent`, `persist_decode`, `nonpersist_decode`, `lds_num_buffers`, `v_reg_prefetch`, `causal_diag_split`, `o_store_width`, `pv_loop_order`, `use_cfvst`, `use_exp2_fast`, `lds_k_group_pad` |
| `gfx942_attention_dense.uhd.json` | git | native selector |
| `kernel_dtype_matches_graph.umd.json` | git | dtype matcher |
| `heuristics/a74a5afb-30c0-41c2-8006-501d2961f808/sort_kernel_catalog/gfx942/tflops/` | git + DVC | the L2 kernel ranker: `heuristic.uhd.json`, `model.bin.dvc`, `.gitignore` in git; `model.bin` from DVC |
| `heuristics/a74a5afb-30c0-41c2-8006-501d2961f808/predict_engine/gfx942/tflops/` | git + DVC | the L1 engine predictor, same layout |

## Models

Two trained models (UHDs) are bound in the UED for gfx942; other architectures keep the native
selector. Each is tracked at `heuristics/<UED id>/<role>/gfx942/tflops/`
(`heuristics/a74a5afb-30c0-41c2-8006-501d2961f808/...`): `heuristic.uhd.json`, `model.bin.dvc`
and `.gitignore` in git, `model.bin` from DVC.

| Role | UHD | Does |
|---|---|---|
| L2 `sort_kernel_catalog` | `18bb432e-d871-4290-9ae8-f5111aaf2ffe` | ranks this catalog's kernels and picks the one to run |
| L1 `predict_engine` | `3f368687-5222-489f-b354-ee0dca01bded` | predicts the engine's TFLOPS for the L2's pick |

Neither model reads the attention scale (every training graph used 1/sqrt(d), so it would stand in
for the head size), a tensor uid, or a feature that depends on how many tensors the graph holds
(`graph.tensors[N]`, `graph.tensor_count`, `graph.logical_bytes`: K and V may be one tensor). They
read each operand through `graph.nodes[0].{q,k,v,o}`, and both read an unset `generate_stats` as
false. The L2 reads
no FLOPs-derived feature: hipDNN publishes none for a sliding-window graph, and a ranker that
cannot evaluate falls back to declared kernel order, so the L2 ranks every graph the engine
serves. No training graph had a window (`hipdnn_corpus_gen`'s `sdpa_fwd` has no window
parameter), so its window rankings, and the TFLOPS scores it reports for them, are extrapolated.
The L1 predicts TFLOPS, which is undefined for a window graph, and reports no prediction there.

Retrain the L2 when the catalog's kernels change. The L1 is measured under the L2, and its
selector revision digests the L2 and the code objects packed at build time, so retrain it
whenever the L2, the catalog (the KDP) or the compiler that packs it changes;
`TestGfx942AttentionDenseModelBinding` fails the build when the revision no longer matches.

A build without `dvc pull -r ingestor` on a `model.bin.dvc` packs the bundle without that
model: the pack warns, the model's UHD does not ship, and the loader logs the dangling id
and runs the engine with that role off (declared kernel order for the L2). An L2 left out this
way also turns the L1 off: the build's selector revision no longer matches the L1's. Release
builds set `HIPKERNELPROVIDER_REQUIRE_DVC_DESCRIPTORS=ON`, which makes it an error.

## Notes

- Fetch `artifacts/` with `dvc pull -r ingestor artifacts.dvc`, and the models with
  `dvc pull -r ingestor --recursive heuristics/`. See `../../README.md`.
- The native symbols (`hipkernel.gfx942_attention_dense.*`) are registered in
  `packs/Gfx942AttentionDenseNative.cpp`.
- Changing the KDP changes its md5. Update the `.dvc` pointer in the same commit, retrain the
  L1, and retrain the L2 too if the KDP's kernels changed (see Models).
