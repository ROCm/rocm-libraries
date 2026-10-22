# gfx950_attention_unified_3d

Split-KV paged unified attention for single-token decode on gfx950, built by rocKE
(`kernels/gfx950/attention_tiled_3d.py`) as one module, `build_unified_attention_3d_tiled_module`:
the segment kernel and the reduce kernel that combines its partial results. Engine:
`hipkernel:Gfx950AttentionUnified3D`.

Each catalog entry is a multi-launch kernel (UKD 1.1 `sources`): the `segment` launch (the
module's primary kernel) and the `reduce` launch (its `reduce` companion), both resolved out of
the one packed code object. The engine serves the graph's contiguous BSHD K/V as a paged
cache with an identity block table, and stages that table, `seq_lens` and `query_start_len` in
the plan's workspace ahead of the segment scratch (`packs/Gfx950AttentionUnified3DGeometry.hpp`).

## Coverage

- Architecture: gfx950 only.
- Decode only: one query token per sequence (every kernel is built with the decode grid).
- Variants: 348 (all `kind = rocke` multi-launch kernels, compiled at pack time): 116 cells at
  three segment counts. Variant reduction across `num_segments` on MI355X may drop counts per
  cell; the count here is the shipped selection's.
- dtype: bf16, fp16. head_size: 64, 128 (the bf16 D256 decode cohort is deferred).
- mask: top-left causal and bottom-right causal (KMD `mask_type` 1, 2). A no-mask decode graph is
  served by the bottom-right causal kernel: with one query row the two admit the same keys, and
  rocKE builds no-mask decode as that kernel.
- Query heads 4 to 128 and KV heads 1 to 128, varying per variant, with a group of 1, 2, 4, 8 or
  16 (the segment kernel fixes BLOCK_M at 16).
- Segments: 16 (every cell's default), 32 and 64. The count only splits the KV loop; any value
  computes the same output.
- No batch capacity: the decode grid launches one CTA per sequence and runs no binary search.
- Fixed in every variant: `block_size` 16 and a one-page KV tile, no sliding window, no sinks, no
  ALiBi, no softcap, no fp8.

## Files

| File | In | Notes |
|---|---|---|
| `artifacts/gfx950_attention_unified_3d.kdp.json` | DVC | the 348 variants. Pointer: `artifacts.dvc` (one artifact for the folder) |
| `gfx950_attention_unified_3d.kmd.json` | git | variant fields |
| `gfx950_attention_unified_3d.udd.json` | git | dispatch symbol |
| `gfx950_attention_unified_3d.ued.json` | git | engine, knob `num_segments` |
| `gfx950_attention_unified_3d.uhd.json` | git | native selector |
| `kernel_dtype_matches_graph.umd.json` | git | dtype matcher |

## Notes

- Fetch `artifacts/` with `dvc pull -r ingestor artifacts.dvc`. See `../../README.md`.
- The native symbols (`hipkernel.gfx950_attention_unified_3d.*`) are registered in
  `packs/Gfx950AttentionUnified3DNative.cpp`.
- Changing the KDP changes its md5. Update the `.dvc` pointer in the same commit.
- Generated from `projects/hipdnn/tools/IngestorGenerator/configs/gfx950_attention_unified_3d.yaml`;
  after the first generation, splice additions rather than copying a regenerated directory over
  this one (every run mints fresh UUIDs).
