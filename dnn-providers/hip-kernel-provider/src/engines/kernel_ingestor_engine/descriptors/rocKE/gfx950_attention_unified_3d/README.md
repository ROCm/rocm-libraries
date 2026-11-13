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
- Variants: 239 (all `kind = rocke` multi-launch kernels, compiled at pack time):
  - 123 without LSE: every one of the 116 cells at 16 segments, plus 7 variants at 32 or 64
    segments in the 5 cells where a gfx950 sweep measured them winning. This is
    `variant_reduce`'s cut of the 348-variant full space (116 cells at three segment counts).
  - 116 LSE-on twins (`*_lse`, KMD `emit_lse` 1): each cell's 16-segment kernel, whose reduce
    also writes the SDPA stats output. At decode its token-major LSE is the head-major
    [B, Hq, 1, 1] stats layout. A graph that asks for stats matches only these; any other
    graph matches only the rest.
- dtype: bf16, fp16. head_size: 64, 128 (the bf16 D256 decode cohort is deferred).
- mask: top-left causal and bottom-right causal (KMD `mask_type` 1, 2). A no-mask decode graph is
  served by the bottom-right causal kernel: with one query row the two admit the same keys, and
  rocKE builds no-mask decode as that kernel.
- Query heads 4 to 128 and KV heads 1 to 128, varying per variant, with a group of 1, 2, 4, 8 or
  16 (the segment kernel fixes BLOCK_M at 16).
- Segments: 16 (every cell's default); 32 and 64 only in bf16 bottom-right causal D128 cells.
  The count only splits the KV loop; any value computes the same output.
- No batch capacity: the decode grid launches one CTA per sequence and runs no binary search.
- Fixed in every variant: `block_size` 16 and a one-page KV tile, no sliding window, no sinks, no
  ALiBi, no softcap, no fp8.

## Trained ranker

On gfx950 the catalog is ranked by trained `sort_kernel_catalog` models, one per metric
(`tflops`, `time`), bound in the UED's role map as `"gfx950": [<tflops id>, <time id>]` beside
the native scorer (`default`, the fewest segments first), which still ranks every other arch. The models
are under `heuristics/14daee8f-98ca-491f-9497-c8ad4e7eefac/sort_kernel_catalog/gfx950/<metric>/`: `heuristic.uhd.json` in
git, `model.bin` in DVC (`model.bin.dvc`). Only the cells that ship more than one variant give
a model a choice; elsewhere a graph matches one kernel and the model and the native scorer
agree.

- Trained on 994 SDPA graphs over the 5 multi-variant cells (bf16 bottom-right causal D128 cells shipping 16, 32 and/or 64 segments), each kernel timed on gfx950
  (`hipdnn_corpus_gen`, `hipdnn_bench`, `uhd_gen`; the `hipdnn-uhd` skill).
- Validated on held-out graphs (never trained on): both models pick the fastest kernel more
  often than the native scorer, lower its geomean regret and shrink its worst case, though a few
  graphs run slower than with the native pick. The PR that
  shipped them gives the measured ratios. Stage V on gfx950 (`validate_l2.py`, `decide.py
  l2 --baseline native`): the runtime reads each model, no winner record decides, and the
  regret gate passes.
- The engine config binds the ids (`engine.heuristics`) and records the fingerprint of the
  catalog they were trained on (`engine.trained_catalog_sha256`). `tests/test_trained_rankers.py` in the IngestorGenerator fails
  when the shipped KDP no longer matches: any kernel added, removed, renamed or changed needs a
  retrain (the skill's `generate.md`), then the new ids and
  `tools/catalog_fingerprint.py <this bundle>/artifacts/<pack>.kdp.json` in both configs.

## Files

| File | In | Notes |
|---|---|---|
| `artifacts/gfx950_attention_unified_3d.kdp.json` | DVC | 0.52 MB, the 239 variants. Pointer: `artifacts.dvc` (one artifact for the folder) |
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
- Generated from `projects/hipdnn/tools/IngestorGenerator/configs/gfx950_attention_unified_3d.yaml`.
  A regeneration keeps descriptor ids stable: `generate.py` reuses the ids of this installed
  bundle by default (generator README, "Descriptor ids"). Separately, generate into scratch
  space and splice additions with `tools/splice_additions.py` rather than copying a regenerated
  directory over this one, because the C++ files are regenerated from scratch and the KDP is
  DVC-backed.
