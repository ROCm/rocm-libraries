# attention_general_bwd

SDPA backward for gfx942 and gfx950, built by rocKE (`kernels/common/attention_bwd*.py`, catalog
`dispatch/attention_bwd/hipdnn_catalog.py`). Engine: `hipkernel:AttentionGeneralBwd`, a slotted
engine (RFC 0021) with a shared workspace (`ws.bytes`):

- `prep` (one launch): selected by the graph alone; one prep kernel matches a graph.
- `main` (launches `main`, `dq`, `convert_dk`, `convert_dv`; `dq` is the split-dQ kernel in
  `main_splitdq_gfx950`): every main kernel of the class the graph pins (dtype, head size,
  sequence mode, stage vec, dK/dV mode, mask class, dQ mode) whose own plan can stand in for the
  engine's pick: same prep launch and workspace layout, legal for the graph and the device
  (`packs/AttentionGeneralBwdNative.cpp`, `resolveCandidate` in
  `packs/AttentionGeneralBwdGeometry.hpp`). A THD (ragged) graph admits only the engine's pick:
  a temporary limit, see "Main-slot ranking".

## Coverage

- Architectures: gfx942, gfx950 (one pack per slot and arch, plus `main_splitdq_gfx950`).
- Kernels: prep 24 (gfx942) / 25 (gfx950); main 202 (gfx942) / 213 + 3 split-dQ (gfx950). One
  main kernel per (class, tuned-table key, head packing).
- dtype: bf16, fp16. head_size: 32, 64, 128. Layouts: BSHD, BHSD, THD. Masks: none, causal
  (top-left, bottom-right), sliding window. dQ: atomic, split (gfx950).

## Main-slot ranking

The main slot's `sort_kernel_catalog` binds the native scorer as `default`
(`attention_general_bwd_main.uhd.json`, symbol `hipkernel.attention_general_bwd.main_score`):
the engine's own pick (rocKE's policy, the most specific tuned entry else the class default)
first, then priority and id. On gfx942 and gfx950 a trained model ranks the main candidates
instead, bound as `"gfx942": [...]`, `"gfx950": [...]` beside `default`. The models are under
`heuristics/0bd5a4d5-70c4-4e48-9da8-6edc0f174b06/main/sort_kernel_catalog/<arch>/`:
`heuristic.uhd.json` in git, `model.bin` in DVC (`model.bin.dvc`, remote `ingestor`). The prep
slot has one candidate per graph and no ranker.

- Metric-less tree rankers (`tree_data`). Each model reads a subset of the main slot's knobs
  (`main_block_n`, `tuning_key_id`, `head_pack_on`, `lb_mirror_on`, `lb_interleave_on`,
  `worklist_on`, `static_scale_id`: int restatements of the KMD's choice fields) and of the
  graph features (shapes, data type, mask class, and for gfx942 the engine pick's tile and
  tuned key); the exact list is the model's `features_signature` in its `heuristic.uhd.json`
  and differs per arch. The label is `robustMeanMs` of `hipdnn_bench --sweep`, warm cache.
- Trained on dense graphs only (BSHD and BHSD, causal and window masks), each main candidate
  timed on the device with `hipdnn_corpus_gen` (`sdpa_bwd`), `hipdnn_bench` and `uhd_gen` (the
  `hipdnn-uhd` skill), plus dense graphs around the rocKE backward census. The graph sets and
  the judging scripts are not in the repository. THD graphs are not timed (`hipdnn_bench` does
  not fill the ragged offset tables of a graph, so a THD run reads an invalid table) and keep
  the engine's pick. This is a temporary limit: lift the THD rule in `mainCandidatePlanOf` once
  the bench times THD and a model is trained on it.
- Validated on held-out graphs (never trained on) offline and at runtime on each arch, native
  tree against model tree on the same graphs. On gfx950 a few held-out graphs are not ranked at
  least as well as by the native pick; the measured results are in the PR description, not here.
- Correctness: `IntegrationGpu<arch>AttentionGeneralBwd.EveryAdmittedMainKernelMatchesTheReference`
  runs every main kernel the engine admits for each catalog-row graph against hipDNN's GPU
  reference, so on those graphs a ranked-first kernel is one checked on the device;
  `TestAttentionGeneralBwdReplay` checks every admitted candidate's launches against rocKE's
  planner. `hipdnn_bench`'s own cross-check cannot decide fp16 rows (its input fill overflows
  dQ), so those training rows carry no verdict.
- If the loader refuses a model (a failed provenance or feature check, a missing `model.bin`),
  the slot ranks in declared order, not with the engine's pick first: the SDK logs an error
  naming the model and arch (`descriptor loader: ... disabled: ...`). A shipped-ranker gate
  that fails with the KDPs and models pulled is how that surfaces in the tests.

## Retraining

The config records the models (`descriptor-packaging/tools/general_bwd_trained_rankers.yaml`:
the main slot's `heuristics` and `trained_catalog_sha256`), and `TestShippedSlotRankers` in
`projects/hipdnn/tools/IngestorGenerator/tests/test_trained_rankers.py` fails (with the KDPs
pulled) when a shipped main-slot catalog no longer matches the fingerprint the model was trained
on: any main kernel added, removed or changed needs a retrain.

1. Make the graph sets on each arch: `hipdnn_corpus_gen --operation sdpa_bwd --engine-name
   hipkernel:AttentionGeneralBwd ...` and dense graphs around the rocKE backward census (BSHD
   and BHSD, several sequence lengths and batches; build the graphs with rocKE's
   `dispatch/attention_bwd/hipdnn.py` and keep those the engine accepts). Those scripts are not
   in the repository.
2. Time every admitted candidate: `hipdnn_bench --graph <g> --engine-name
   hipkernel:AttentionGeneralBwd --plugin-dir <dir> --sweep` writes one CSV per graph.
3. Merge the CSVs into one file before adding them: `python -m uhd_gen.dataset add --csv` reads
   each file's `numerically_valid` column on its own, and files that mix `True` and `Unknown`
   verdicts then fail to store as one table. Then `python -m uhd_gen.dataset add --into <store>
   --csv <merged.csv> --engine hipkernel:AttentionGeneralBwd --engine-id <id> --arch <arch>
   --selector-revision <rev> --collected-at <timestamp>`.
4. Train per arch: `python -m uhd_gen generate --dataset <store> --slot main --no-promote
   --target robustMeanMs --objective min --features <shape and knob features>
   --descriptor-tree <this directory's parent tree> --engine hipkernel:AttentionGeneralBwd
   --arch <arch> --feature-evaluator <build>/bin/hipdnn_uhd_features --output-dir <out>`.
5. Judge the model against the native pick on its held-out graphs (a manual step: the slot
   has no validate script), then install it: `python -m uhd_gen promote --model-dir
   <out>/model --descriptor-tree <this directory> --engine hipkernel:AttentionGeneralBwd
   --role sort_kernel_catalog --arch <arch> --corpus <out>/corpus.json --feature-evaluator
   <build>/bin/hipdnn_uhd_features`.
6. Record the new ids and `projects/hipdnn/tools/IngestorGenerator/tools/catalog_fingerprint.py
   <every main-slot KDP of the arch under artifacts/>` in `general_bwd_trained_rankers.yaml`,
   regenerate with `general_bwd_catalog.py --generate --verify`, `dvc add` the new `model.bin`,
   and push it.

## Files

| File | In | Notes |
|---|---|---|
| `artifacts/*.kdp.json` | DVC | the five packs; pointer `artifacts.dvc` (remote `ingestor`) |
| `attention_general_bwd.ued.json` | git | engine; slots, main-slot knobs and ranker bindings |
| `attention_general_bwd_{prep,main}.kmd.json` | git | slot variant fields |
| `attention_general_bwd_{prep,main}.udd.json` | git | slot dispatch symbols |
| `attention_general_bwd_main.uhd.json` | git | native main-slot scorer |
| `heuristics/.../main/sort_kernel_catalog/<arch>/` | git + DVC | trained main-slot rankers |
| `*.umd.json` | git | dtype and per-pack operation matchers |

## Notes

- Fetch the DVC content with `dvc pull -r ingestor` on `artifacts.dvc` and each `model.bin.dvc`.
  See `../../README.md`.
- The packer copies `heuristics/` into every arch shard. A shard installed on its own (without
  the other arch's packs) logs that the other arch's model is disabled, because that model's
  provenance names UMDs only the other shard carries; the device's own model still binds.
- Generated from `projects/hipdnn/tools/IngestorGenerator/configs/attention_general_bwd.yaml`,
  written by `descriptor-packaging/tools/general_bwd_catalog.py`; regeneration keeps descriptor
  ids stable.
