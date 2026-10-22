# gfx942_attention_unified_3d

Single-token (seqlen_q 1) split-KV attention decode for gfx942, built by rocKE
(`kernels/gfx942/attention_tiled_3d.py`, builder `build_unified_attention_3d_module`).
Engine: `hipkernel:Gfx942AttentionUnified3d`.

Every variant is one rocKE module with two launches: `segment` writes per-segment
partials into the workspace and `reduce` combines them. Both launches share one spec.

## Coverage

- Architecture: gfx942 only.
- Variants: 2,092 in 132 cells (all `kind = rocke`, two launches each, compiled at pack
  time). A cell is (dtype, head_size, query heads, KV heads).
- dtype: bf16, fp16.
- head_size: 64, 128, 256.
- Query heads 1 to 128 and KV heads 1 to 64, varying per cell.
- block_size: 16, 32, 64. tile_size_override follows from (head_size, block_size): 16, 32, 64.
- num_segments: 2 to 128 (15 values), varying per cell.
- Fixed in every variant: strided (dense) KV, decode grid, no sliding window, no sinks,
  softcap, ALiBi or QQ bias, `num_seqs` 32, `kv_cache_policy` stream.

## Files

| File | In | Notes |
|---|---|---|
| `artifacts/gfx942_attention_unified3d.kdp.json` | DVC | 5.2 MB, the 2,092 variants. Pointer: `artifacts.dvc` (one artifact for the folder) |
| `gfx942_attention_unified3d.kmd.json` | git | variant fields |
| `gfx942_attention_unified3d.udd.json` | git | dispatch symbol |
| `gfx942_attention_unified3d.ued.json` | git | engine, knobs `block_size`, `num_segments` |
| `gfx942_attention_unified3d.uhd.json` | git | native selector |
| `kernel_dtype_matches_graph.umd.json` | git | dtype matcher |

The file stem is the generator's engine slug (`gfx942_attention_unified3d`); the folder is
the selection's `authored_subpath`.

## Provenance

- Selection: `projects/hipdnn/tools/IngestorGenerator/configs/gfx942_attention_unified_3d.yaml`,
  expanded from the manifest `gfx942_attention_unified_3d.engine.yaml` (rocKE dispatcher
  registry, `attention_strided_decode` candidate). AICK-2683.
- Generated at rocm-libraries `4b59b88b9ea` with:

  ```bash
  python projects/hipdnn/tools/IngestorGenerator/generate.py \
    --config projects/hipdnn/tools/IngestorGenerator/configs/gfx942_attention_unified_3d.yaml \
    --output-dir <scratch>
  ```

  then the KDP moved into `artifacts/` and the other descriptors copied as emitted.
- `tests/test_registry_regeneration.py` (g1 to g4) gates this bundle against the selection
  and the manifest.

## Notes

- Fetch `artifacts/` with `dvc pull -r ingestor artifacts.dvc`. See
  `../../README.md`.
- The native symbols (`hipkernel.gfx942_attention_unified3d.*`) are registered in
  `packs/Gfx942AttentionUnified3dNative.cpp`. Until that native pack lands, the loader
  refuses this engine at provider load.
- Changing the KDP changes its md5. Update the `.dvc` pointer in the same commit.
