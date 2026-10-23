# gfx942_attention_unified_3d

Single-token (seqlen_q 1) split-KV attention decode for gfx942, built by rocKE
(`kernels/gfx942/attention_tiled_3d.py`, builder `build_unified_attention_3d_module`).
Engine: `hipkernel:Gfx942AttentionUnified3d`.

Every variant is one rocKE module with two launches: `segment` writes per-segment
partials into the workspace and `reduce` combines them. Both launches share one spec.

## Coverage

- Architecture: gfx942 only.
- Variants: 2,192 in 138 cells (all `kind = rocke`, two launches each, compiled at pack
  time). A cell is (dtype, head_size, query heads, KV heads): 23 head pairs per head size.
  2,192 exceeds the selection's 1000-kernel AOT budget (`aot_budget_per_arch`), so
  `generate.py` and `registry_enumerator.py` print a warning (an error only under `--strict`).
- dtype: bf16, fp16.
- head_size: 64, 128, 256.
- Query heads 1 to 128 and KV heads 1 to 64, varying per cell.
- block_size: 16, 32, 64. tile_size_override follows from (head_size, block_size): 16, 32, 64.
- num_segments: 2 to 128 (15 values), varying per cell.
- Fixed in every variant: strided (dense) KV, decode grid, no sliding window, no sinks,
  softcap, ALiBi or QQ bias, `kv_cache_policy` stream.
- `num_seqs` is recorded as 32 in every spec but is IR-neutral: the decode grid's code
  does not depend on it. The engine sets it at launch from the graph's batch, so 32 is
  not a batch limit.

## Files

| File | In | Notes |
|---|---|---|
| `artifacts/gfx942_attention_unified3d.kdp.json` | DVC | 5.4 MB, the 2,192 variants. Pointer: `artifacts.dvc` (one artifact for the folder) |
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
  registry, `attention_strided_decode` candidate).
- Generated with:

  ```bash
  python projects/hipdnn/tools/IngestorGenerator/generate.py \
    --config projects/hipdnn/tools/IngestorGenerator/configs/gfx942_attention_unified_3d.yaml \
    --output-dir <scratch>
  ```

  then the KDP moved into `artifacts/` and the other descriptors copied as emitted.
- The 16/16 head pair (100 variants) was added later with
  `tools/splice_additions.py`, which keeps the UUID of every retained kernel and appends
  the new entries at the end of the KDP.
- `projects/hipdnn/tools/IngestorGenerator/tests/test_registry_regeneration.py` gates this
  bundle against the selection and the manifest:
  - g1 `test_g1_every_selection_kernel_is_a_registry_kernel_at_defaults`;
  - g2 `test_g2_the_shipped_catalog_is_generated_from_the_selection`;
  - g3 `test_g3_the_enumerated_registry_reproduces_the_shipped_catalog`;
  - g4 `test_g4_the_selection_header_equals_the_manifest`.

## Notes

- Fetch `artifacts/` with `dvc pull -r ingestor artifacts.dvc`. See
  `../../README.md`.
- The native symbols (`hipkernel.gfx942_attention_unified3d.*`) are registered in
  `packs/Gfx942AttentionUnified3dNative.cpp`, which launches `segment` then `reduce` on the
  handle's stream. Its census is `TestGfx942AttentionUnified3dPacks`.
- Changing the KDP changes its md5. Update the `.dvc` pointer in the same commit.
