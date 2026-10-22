# gfx950_attention_unified_3d

Split-KV paged unified attention for single-token decode on gfx950, built by rocKE
(`kernels/gfx950/attention_tiled_3d.py`): the segment kernel `build_unified_attention_3d_tiled`
and the reduce kernel `build_unified_attention_reduce_tiled`. Engine:
`hipkernel:Gfx950AttentionUnified3D`.

Each catalog entry is two code objects: the segment kernel as its `kernel_source` and the reduce
kernel as its one `companions` entry. The engine serves the graph's contiguous BSHD K/V as a paged
cache with an identity block table, and stages that table, `seq_lens` and `query_start_len` in
the plan's workspace ahead of the segment scratch (`packs/Gfx950AttentionUnified3DGeometry.hpp`).

## Coverage

- Architecture: gfx950 only.
- Decode only: one query token per sequence (every kernel is built with the decode grid).
- Variants: 134 (all `kernel_source.kind = rocke` with one rocke companion, compiled at pack
  time): 116 at head_size 64 and 128, and 18 at head_size 256.
- dtype: bf16, fp16 at head_size 64 and 128; bf16 only at head_size 256 (rocKE's d256_decode
  cohort).
- mask: top-left causal and bottom-right causal (KMD `mask_type` 1, 2). A no-mask decode graph is
  served by the bottom-right causal kernel: with one query row the two admit the same keys, and
  rocKE builds no-mask decode as that kernel.
- Query heads 4 to 128 and KV heads 1 to 128, varying per variant, with a group of 1, 2, 4, 8 or
  16 (the segment kernel fixes BLOCK_M at 16).
- Segments: 16 at head_size 64 and 128; at head_size 256 the split rule's count for batch 1 at
  4096 keys (32, 64 or 128). The count only splits the KV loop; any value computes the same output.
- Fixed in every variant: `block_size` 16 and a one-page KV tile, no sliding window, no sinks, no
  ALiBi, no softcap, no fp8.

## Files

| File | In | Notes |
|---|---|---|
| `artifacts/gfx950_attention_unified_3d.kdp.json` | DVC | the 134 variants. Pointer: `artifacts.dvc` (one artifact for the folder) |
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
