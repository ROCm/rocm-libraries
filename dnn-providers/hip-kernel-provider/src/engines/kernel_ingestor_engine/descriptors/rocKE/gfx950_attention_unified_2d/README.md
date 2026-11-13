# gfx950_attention_unified_2d

Paged unified-attention prefill (vLLM's `unified_attention`) for gfx950, built by rocKE
(`kernels/gfx950/attention_tiled_2d.py`, builder `build_unified_attention_2d_tiled`).
Engine: `hipkernel:Gfx950AttentionUnified2D`.

A hipDNN SDPA graph is dense, so the engine serves the graph's contiguous BSHD K/V as a paged
cache with an identity block table and stages that table, `seq_lens` and `query_start_len` in the
plan's workspace (`packs/Gfx950AttentionUnified2DGeometry.hpp`).

## Coverage

- Architecture: gfx950 only.
- Variants: 420 (all `kernel_source.kind = rocke`, compiled at pack time):
  - 222 without LSE: every one of the 198 cells at the default geometry, plus 24 variants of
    two other geometries in the D128 (and one D64) cells where a gfx950 sweep measured them
    winning. This is `variant_reduce`'s cut of the 792-variant full space (198 cells at four
    geometries).
  - 198 LSE-on twins (`*_lsehm`, KMD `emit_lse` 1): each cell's default-geometry kernel, which
    also writes the SDPA stats output head-major ([B, Hq, S, 1], `lse_layout` head_major). A
    graph that asks for stats matches only these; any other graph matches only the rest.
- dtype: bf16, fp16. head_size: 64, 128 (D256 prefill is deferred).
- mask: no mask, top-left causal, bottom-right causal (KMD `mask_type` 0, 1, 2).
- Query heads 4 to 128 and KV heads 1 to 128, varying per variant (group of 1 to 16).
- Geometries (llvm backend, `block_size` 16), one registry candidate each; the first is every
  cell's default:
  - narrow 16x16 MFMA, 4 warps x 16 rows, KV tile 64;
  - narrow 16x16 MFMA, 2 warps x 16 rows, KV tile 32 (3 bf16 causal variants);
  - transposed 32x32 MFMA, 4 warps x 32 rows, KV tile 64 (21 variants).

  The full space's wide 32x32 geometry (4 warps x 32 rows, KV tile 64) won no graph and is not
  shipped.
- Batch: any. The kernel takes the batch as the `num_seqs` kernarg and derives its `cu_q`
  binary-search trip count from it at runtime, so no batch is baked. The KMD still declares
  `num_seqs` (511); the engine does not read it.
- Decode (seqlen_q 1) whose GQA group divides 16 is left to `hipkernel:Gfx950AttentionUnified3D`.
- Fixed in every variant: no sliding window, no sinks, no ALiBi, no softcap, no fp8, 32-bit K/V
  addressing (K/V larger than 0x7FFF0000 bytes is declined).

## Trained ranker

On gfx950 the catalog is ranked by trained `sort_kernel_catalog` models, one per metric
(`tflops`, `time`), bound in the UED's role map as `"gfx950": [<tflops id>, <time id>]` beside
the native scorer (`default`, the baseline geometry first (4 warps x 16 rows, T64)), which still ranks every other arch. The models
are under `heuristics/800f4c70-d2e0-4a34-ab93-08fb527e2d82/sort_kernel_catalog/gfx950/<metric>/`: `heuristic.uhd.json` in
git, `model.bin` in DVC (`model.bin.dvc`). Only the cells that ship more than one variant give
a model a choice; elsewhere a graph matches one kernel and the model and the native scorer
agree.

- Trained on 810 SDPA graphs over the 21 multi-variant cells (D64/D128 cells shipping two or three tile geometries), each kernel timed on gfx950
  (`hipdnn_corpus_gen`, `hipdnn_bench`, `uhd_gen`; the `hipdnn-uhd` skill).
- Validated on held-out graphs (never trained on): both models pick the fastest kernel more
  often than the native scorer and lower its geomean regret, though a few graphs run slower than
  with the native pick; a retrain on a larger corpus scored
  the same, so these models stay. The PR that shipped them gives the measured ratios. Stage V on gfx950 (`validate_l2.py`,
  `decide.py l2 --baseline native`): the runtime reads each model, no winner record decides,
  and the regret gate passes.
- The engine config binds the ids (`engine.heuristics`) and records the fingerprint of the
  catalog they were trained on (`engine.trained_catalog_sha256`). `tests/test_trained_rankers.py` in the IngestorGenerator fails
  when the shipped KDP no longer matches: any kernel added, removed, renamed or changed needs a
  retrain (the skill's `generate.md`), then the new ids and
  `tools/catalog_fingerprint.py <this bundle>/artifacts/<pack>.kdp.json` in both configs.

## Files

| File | In | Notes |
|---|---|---|
| `artifacts/gfx950_attention_unified_2d.kdp.json` | DVC | 0.58 MB, the 420 variants. Pointer: `artifacts.dvc` (one artifact for the folder) |
| `gfx950_attention_unified_2d.kmd.json` | git | variant fields |
| `gfx950_attention_unified_2d.udd.json` | git | dispatch symbol |
| `gfx950_attention_unified_2d.ued.json` | git | engine, knobs `tile_size`, `num_warps`, `block_m_per_warp`, `use_mfma_32x32`, `use_transposed_qk_32x32` |
| `gfx950_attention_unified_2d.uhd.json` | git | native selector |
| `kernel_dtype_matches_graph.umd.json` | git | dtype matcher |

## Notes

- Fetch `artifacts/` with `dvc pull -r ingestor artifacts.dvc`. See `../../README.md`.
- The native symbols (`hipkernel.gfx950_attention_unified_2d.*`) are registered in
  `packs/Gfx950AttentionUnified2DNative.cpp`.
- Changing the KDP changes its md5. Update the `.dvc` pointer in the same commit.
- Generated from `projects/hipdnn/tools/IngestorGenerator/configs/gfx950_attention_unified_2d.yaml`.
  A regeneration keeps descriptor ids stable: `generate.py` reuses the ids of this installed
  bundle by default (generator README, "Descriptor ids"). Separately, generate into scratch
  space and splice additions with `tools/splice_additions.py` rather than copying a regenerated
  directory over this one, because the C++ files are regenerated from scratch and the KDP is
  DVC-backed.
