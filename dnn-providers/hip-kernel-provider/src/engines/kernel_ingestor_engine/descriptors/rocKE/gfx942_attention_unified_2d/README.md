# gfx942_attention_unified_2d

Paged unified-attention prefill (vLLM's `unified_attention`) for gfx942, built by rocKE
(`kernels/gfx942/attention_tiled_2d.py`, builder `build_unified_attention_2d_tiled`).
Engine: `hipkernel:Gfx942AttentionUnified2D`.

A hipDNN SDPA graph is dense, so the engine serves the graph's contiguous BSHD K/V as a paged
cache with an identity block table and stages that table, `seq_lens` and `query_start_len` in the
plan's workspace (`packs/Gfx942AttentionUnified2DGeometry.hpp`).

## Coverage

- Architecture: gfx942 only.
- Variants: 95 (all `kernel_source.kind = rocke`, compiled at pack time), 48 at D64 and 47 at
  D128. The selection is `variant_reduce`'s cut (eps 0.05) of a 190-variant full space: 20
  cells, each at five geometries plus rocKE's production stack for its dtype and head size,
  plus the 1-warp narrow geometry at every KV tile that fits the LDS (four at D64, three at
  D128). Every cell keeps its default geometry plus the alternatives a correctness-checked
  gfx942 autotune sweep measured as needed.
  - Default geometry (every cell): 20.
  - Production stack (what rocKE's own dispatcher builds for a long prefill): every D64 cell
    (mask limit, hoist, conflict-free V stores, sliced-K ring, direct-global Q), every fp16
    D128 cell (the same without direct-global Q, depth-2 ring) and every bf16 D128 cell (mask
    limit and hoist with a single K buffer): 20.
  - 1-warp narrow (16x16 MFMA, 16 rows): 44, the fastest kernel for most short-prefill graphs
    (D64: 8 at KV tile 16, 7 at 32, 8 at 64, 4 at 128; D128: 2 at 16, 9 at 32, 6 at 64).
  - Narrow 2 warps x 16 rows: KV tile 16 in every D128 cell (10); KV tile 64 at bf16 D64
    ratio 16 (1).
- Cells: head size x dtype x GQA ratio. The kernels read the head counts as kernel arguments
  and bake only the ratio (KMD `num_queries_per_kv`), so one variant serves every head pair of
  its ratio.
- dtype: bf16, fp16. head_size: 64, 128. GQA ratio: 1, 2, 4, 8, 16.
- mask: bottom-right causal (KMD `mask_type` 2), the only band the gfx942 tiled 2D body
  computes.
- Geometries (llvm backend, `block_size` 16), one registry candidate each, five per head size
  in the full space; the first of each is its cells' default:
  - D64: transposed 32x32x8 MFMA 4 warps x 32 rows KV tile 64; transposed 8 warps x 32 rows
    KV tile 64 (also built with direct Q loads and the conflict-free V store); narrow 16x16
    MFMA 4 warps x 16 rows KV tile 64; narrow 4 warps x 32 rows KV tile 64; narrow 2 warps x
    16 rows KV tile 64.
  - D128: transposed 32x32x8 MFMA 2 warps x 32 rows KV tile 32; transposed 4 warps x 32 rows
    KV tile 32; narrow 4 warps x 16 rows KV tile 32; narrow 2 warps x 32 rows KV tile 32;
    narrow 2 warps x 16 rows KV tile 16.
  - Both head sizes: narrow 16x16 MFMA 1 warp x 16 rows, KV tiles 16, 32, 64 and (D64 only)
    128; the 128-token tile exceeds the 64 KiB LDS at D128.

  A correctness-checked gfx942 sweep of every LDS-legal geometry over 80 prefill shapes, run
  through rocKE's own launcher, chose the full space's five per head size. A later sweep of
  every LDS-legal registry geometry over short (seqlen_q <= 64), mid and long prefill added
  the 1-warp narrow geometry, which no other geometry matched on short prefill.
- Reduction evidence: one autotuned hipDNN ranking per graph over 401 bottom-right graphs
  (real, recast, synthetic and short-prefill corpora), cross-checked against an independent
  rocKE-launcher sweep (the kept set is within 0.9% geomean of the best of the full space);
  the final sweep of this selection (3 rounds, correctness on) measured a 1.7% noise floor,
  at which `variant_reduce` keeps 94 of these 95 and proposes no rule.
- Batch: any. The kernel takes the batch as the `num_seqs` kernarg and derives its `cu_q`
  binary-search trip count from it at run time.
- Fixed in every variant: no sliding window, no sinks, no ALiBi, no softcap, no fp8, 32-bit K/V
  addressing (K/V larger than 0x7FFF0000 bytes is declined).

## Trained ranker

On gfx942 the catalog is ranked by a trained `sort_kernel_catalog` model for the `tflops`
metric, bound in the UED's role map as `"gfx942": [<tflops id>]` beside the native scorer
(`default`), which stays the fallback when the model is not pulled. The model is under
`heuristics/7bd1634d-b1dc-4486-a2e6-7e576b327258/sort_kernel_catalog/gfx942/tflops/`:
`heuristic.uhd.json` in git, `model.bin` in DVC (`model.bin.dvc`). A `time` model was trained
too and is not shipped: it lost to the native scorer on short-context MHA prefill.

- Trained on 3200 bottom-right SDPA graphs (2880 trained, 320 held out), every kernel of the
  graph's cell timed on gfx942 (`hipdnn_corpus_gen`, `hipdnn_bench`, `uhd_gen`; the
  `hipdnn-uhd` skill). The searched part of the shape space is all append (Sq < Skv):
  `sdpa_fwd` allows bottom-right alignment only for Sq < Skv. Equal-length prefill came in
  through `--model-shapes`: the stage 7 sweep's 401 shapes plus seeded synthetic Sq = Skv
  shapes, weighted toward short-context MHA and MQA, where a first round lost to native.
- Validated on the held-out graphs (never trained on): the model lowers the native scorer's
  mean regret in every regime. End to end the gain is small: on graphs that run in tens of
  microseconds hipDNN's per-call cost hides the kernel differences, and single timings vary
  widely there. Stage V on gfx942 (`validate_l2.py`, `decide.py l2 --baseline native`): the
  runtime reads the model, no winner record decides, and the regret gate passes once those
  short graphs are timed five times each. The PR that shipped it gives the measured ratios.
- The engine config binds the ids (`engine.heuristics`) and records the fingerprint of the
  catalog they were trained on (`engine.trained_catalog_sha256`).
  `tests/test_trained_rankers.py` in the IngestorGenerator fails when the shipped KDP no
  longer matches: any kernel added, removed, renamed or changed needs a retrain (the skill's
  `generate.md`), then the new ids and
  `tools/catalog_fingerprint.py <this bundle>/artifacts/<pack>.kdp.json` in both configs.
  `TestGfx942AttentionUnified2DModelBinding` (census) fails the build when the runtime would
  refuse a bound model.
- A build without `dvc pull -r ingestor` on a `model.bin.dvc` packs the bundle without that
  model: the pack warns, the model's UHD does not ship, and the engine ranks with the native
  scorer. Release builds set `HIPKERNELPROVIDER_REQUIRE_DVC_DESCRIPTORS=ON`, which makes it an
  error.

## Files

| File | In | Notes |
|---|---|---|
| `artifacts/gfx942_attention_unified_2d.kdp.json` | DVC | 0.16 MB, the 95 variants. Pointer: `artifacts.dvc` (one artifact for the folder) |
| `gfx942_attention_unified_2d.kmd.json` | git | variant fields |
| `gfx942_attention_unified_2d.udd.json` | git | dispatch symbol |
| `gfx942_attention_unified_2d.ued.json` | git | engine, knobs `tile_size`, `num_warps`, `block_m_per_warp`, `use_mfma_32x32x8`, `use_transposed_qk_32x32`, `use_k_sliced_ring`, `use_q_direct_global`, `use_conflict_free_v_store`, `use_transposed_mask_limit`, `use_transposed_invariant_hoist` |
| `gfx942_attention_unified_2d.uhd.json` | git | native selector |
| `heuristics/7bd1634d-b1dc-4486-a2e6-7e576b327258/sort_kernel_catalog/gfx942/tflops/` | git + DVC | trained ranker: `heuristic.uhd.json` in git, `model.bin` in DVC (`model.bin.dvc`) |
| `kernel_dtype_matches_graph.umd.json` | git | dtype matcher |

## Notes

- Fetch `artifacts/` and the trained ranker with `dvc pull -r ingestor artifacts.dvc heuristics/7bd1634d-b1dc-4486-a2e6-7e576b327258/sort_kernel_catalog/gfx942/tflops/model.bin.dvc`. See `../../README.md`.
- The native symbols (`hipkernel.gfx942_attention_unified_2d.*`) are registered in
  `packs/Gfx942AttentionUnified2DNative.cpp`.
- Changing the KDP changes its md5. Update the `.dvc` pointer in the same commit.
- Generated from `projects/hipdnn/tools/IngestorGenerator/configs/gfx942_attention_unified_2d.yaml`.
  A regeneration keeps descriptor ids stable: `generate.py` reuses the ids of this installed
  bundle by default (generator README, "Descriptor ids"). Generate into scratch space and
  splice additions with `tools/splice_additions.py` rather than copying a regenerated
  directory over this one, because the C++ files are regenerated from scratch and the KDP is
  DVC-backed.
