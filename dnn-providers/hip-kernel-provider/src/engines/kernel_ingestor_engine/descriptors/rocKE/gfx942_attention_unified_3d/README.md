# gfx942_attention_unified_3d

Single-token (seqlen_q 1) split-KV attention decode for gfx942, built by rocKE
(`kernels/gfx942/attention_tiled_3d.py`, builder `build_unified_attention_3d_tiled_module`).
Engine: `hipkernel:Gfx942AttentionUnified3d`.

Every variant is one rocKE module with two launches: `segment` writes per-segment
partials into the workspace and `reduce` combines them. Both launches share one spec.

## Coverage

- Architecture: gfx942 only.
- Variants: 560 in 30 cells (all `kind = rocke`, two launches each, compiled at pack
  time). A cell is (dtype, head_size, GQA ratio): the kernels read the query and KV head
  counts as kernel arguments and bake only their ratio, so one kernel serves every head
  pair of its ratio. 560 is within the selection's 1000-kernel AOT budget.
- dtype: bf16, fp16.
- head_size: 64, 128, 256.
- GQA ratio (`num_queries_per_kv`): 1, 2, 4, 8, 16; any head counts with that ratio.
- block_size: 16, 32, 64. tile_size_override follows from (head_size, block_size): 16, 32, 64.
- num_segments: 2 to 128 (16 values), varying per cell.
- K/V cache policy: `stream` in every variant, pinned by the manifest's
  `enumerate.spec_overrides`; the reason is in the header comment of
  `projects/hipdnn/tools/IngestorGenerator/configs/gfx942_attention_unified_3d.engine.yaml`.
- Fixed in every variant: strided (dense) KV, decode grid, no sliding window, no sinks,
  softcap, ALiBi or QQ bias.
- `num_seqs` is recorded as 32 in every spec but is IR-neutral: the decode grid's code
  does not depend on it. The engine sets it at launch from the graph's batch, so 32 is
  not a batch limit.

## Files

| File | In | Notes |
|---|---|---|
| `artifacts/gfx942_attention_unified3d.kdp.json` | DVC | 1.4 MB, the 560 variants. Pointer: `artifacts.dvc` (one artifact for the folder) |
| `gfx942_attention_unified3d.kmd.json` | git | variant fields |
| `gfx942_attention_unified3d.udd.json` | git | dispatch symbol |
| `gfx942_attention_unified3d.ued.json` | git | engine, knobs `block_size`, `num_segments` |
| `gfx942_attention_unified3d.uhd.json` | git | native selector |
| `kernel_dtype_matches_graph.umd.json` | git | dtype matcher |

The file stem is the generator's engine slug (`gfx942_attention_unified3d`); the folder is
the selection's `authored_subpath`.

## Models

One trained model (UHD), the L2 `sort_kernel_catalog` kernel ranker `ced4e2d1-e66a-4ab5-bfb2-fa3626f881d8`
(ranks this catalog's kernels), is bound in the UED for gfx942. It is tracked at
`heuristics/4208dcf6-3128-4146-ab2a-affa5b6ab4eb/sort_kernel_catalog/gfx942/tflops/` (the UED id,
role, arch and metric): `heuristic.uhd.json`, `model.bin.dvc` and `.gitignore` in git, `model.bin`
from DVC.

The L1 `predict_engine` (predicts the engine's TFLOPS) is held: its selector revision was recorded on
an older build, so the runtime refuses it on a build of the current source. It follows in a separate
change, once re-measured on a build of current source. It is measured under the L2 and its selector
revision digests the code objects packed at build time, so retrain it whenever the L2 or the catalog
(the KDP) changes.

A build without `dvc pull -r ingestor` on a `model.bin.dvc` packs the bundle without that
model: the pack warns, the model's UHD does not ship, and the loader logs the dangling id
and runs the engine with that role off (declared kernel order for the L2). Release builds set
`HIPKERNELPROVIDER_REQUIRE_DVC_DESCRIPTORS=ON`, which makes it an error.

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

  then the KDP moved into `artifacts/` and the other descriptors copied as emitted. The
  run reuses the shipped bundle's ids (`--ids-from` this folder): a kernel whose metadata
  tuple is unchanged keeps its id.
- `projects/hipdnn/tools/IngestorGenerator/tests/test_registry_regeneration.py` gates this
  bundle against the selection and the manifest:
  - g1 `test_g1_every_selection_kernel_is_a_registry_kernel_at_defaults`;
  - g2 `test_g2_the_shipped_catalog_is_generated_from_the_selection`;
  - g3 `test_g3_the_enumerated_registry_reproduces_the_shipped_catalog`;
  - g4 `test_g4_the_selection_header_equals_the_manifest`.

## Notes

- Fetch `artifacts/` with `dvc pull -r ingestor artifacts.dvc` and each model with
  `dvc pull -r ingestor <model.bin.dvc>` (see Models), or everything at once with
  `dvc pull -r ingestor --recursive <this folder>`. See `../../README.md`.
- The native symbols (`hipkernel.gfx942_attention_unified3d.*`) are registered in
  `packs/Gfx942AttentionUnified3dNative.cpp`, which launches `segment` then `reduce` on the
  handle's stream. Its census is `TestGfx942AttentionUnified3dPacks`.
- Changing the KDP changes its md5. Update the `.dvc` pointer in the same commit.
