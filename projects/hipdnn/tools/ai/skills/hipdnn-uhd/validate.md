# Validate and install

Installing a UHD where the provider ships it **changes which kernel (or engine) is chosen**.
Stage V proves the engine reads the candidate and chooses at least as well, before anything
in the checkout changes. Everything runs on the local build and GPU, with the skill's
scripts (`$UHD/scripts/`, [SKILL.md](SKILL.md)): `candidate_tree.py`, `validate_l1.py`,
`validate_l2.py` and `decide.py`, which exits 0 when every gate passes. Graphs come from
`eval_slice.py` ([data.md](data.md)). Upstream installation semantics: `projects/hipdnn/tools/uhd_gen/promote.py`.

## The graphs

```bash
python3 $UHD/scripts/eval_slice.py sample <each training set's eval_slice.json> \
  --count 200 --trained <the model's run>/generation_manifest.json --out sample.json
```
draws up to 200 graphs spread evenly over the slices' regimes, and refuses if any is among
the model's `training_problem_keys`. A slice is a tenth of its set, so a smoke set (200
graphs per operation) gives every slice graph, about 20 per operation. A closed shape set has
no slice: its whole set is the graph list, and the verdict is recall.

`<model dir>` below is the model's `models[i].model_dir` (the directory holding
`heuristic.uhd.json`).

## Stage V: engine predictor (L1)

```bash
python3 $UHD/scripts/validate_l1.py --bench <build>/bin/hipdnn_bench \
  --plugin-dir <abs plugin dir> --engine-id <id> \
  --descriptor-root <tree> --candidate-dir <model dir> --declared-id <uhd id> \
  --sample sample.json --metric <metric> --work-dir <new dir> --out rows.json \
  [--promote <engine> sort_kernel_catalog <arch key> <L2 model dir> \
   --promote <engine> predict_engine <arch key> <model dir>]
python3 $UHD/scripts/decide.py l1 --rows rows.json --candidate-dir <model dir> --out report.json
```

- `<tree>`: the descriptor root (`<plugin dir>/arch_content/hip-kernel-provider`) for an
  opaque engine; the per-arch root for a descriptor engine.
- **Opaque engine**: `--declared-id` is the id its provider declares for the arch and metric.
  The copy holds the candidate as the only document carrying it. `--descriptor-root` is the
  tree the engine ships in, or an empty directory when its provider ships none (MIOpen).
- **Descriptor engine**: `--declared-id` is the model's own `id`, and `--promote` installs, in
  order, the L2 its rows were measured under and then the candidate. The candidate's truth is
  measured under that tree, the shipped model's under the build's own.
- Gates: `identity` (the engine could only have read the candidate), `available` (it predicts
  for every graph it executes; `--min-available` relaxes it), `metric` (every answer names its
  declared metric), `selector_revision` (the engine reports the revision the model was trained
  against), `accuracy` (candidate median |error| ≤ shipped × 1.05, `--max-regression`
  changes it). Both models' accuracy is always reported. If the branch ships no model of the
  role for the arch, pass `--no-shipped-baseline`: the accuracy gate then asks only that the
  candidate was scored, and the report says `baseline: none`.

## Stage V: catalog ranker (L2)

1. Offline regret on the held-out graphs' stored catalog sweeps:
   ```bash
   python3 -m uhd_gen.dataset export --from <L2 dataset dir> --out exported.json \
     --contribution <the model's contribution ids> --metric <metric>
   python3 $UHD/scripts/eval_slice.py rows exported.json <slices> --out slice_rows.json
   python3 -m uhd_gen evaluate --input slice_rows.json --model-dir <model dir> \
     --eval-fraction 1.0 --feature-evaluator <build>/bin/hipdnn_uhd_features --output candidate.json
   python3 -m uhd_gen evaluate --input slice_rows.json --model-dir <shipped ranker dir> \
     --target <the candidate's target> --eval-fraction 1.0 ... --output shipped.json
   ```
   The shipped ranker is the UHD the installed UED's `sort_kernel_catalog` entry names for the
   arch key. A `native` adapter is the engine's own scoring code: there is no model to score
   offline, so the comparison is the paired GPU timing below (`--baseline native`).
2. Whether the runtime reads it, on the GPU:
   ```bash
   python3 $UHD/scripts/validate_l2.py --bench <build>/bin/hipdnn_bench --plugin-dir <abs plugin dir> \
     --engine-id <id> --engine <engine> --root <per-arch root> --role sort_kernel_catalog \
     --arch-key <arch key> --candidate-dir <model dir> --sample sample.json \
     --ranking-metric <metric> --work-dir <new dir> --out rows.json
   python3 $UHD/scripts/decide.py l2 --rows rows.json --candidate-report candidate.json \
     [--shipped-report shipped.json] [--baseline model|native|none] --out report.json
   ```
   Every run gets an empty `HIPDNN_CACHE_DIR` and INFO logging. Gates: `available`, `read` (the
   runtime never said "the model is not used", "ranking degrades to static_order" or "cannot
   be brought up"), `winner_records` (no benchmark winner record ordered the catalog), `regret`
   (candidate mean top-1 regret no worse than shipped's plus noise slack; `--baseline none`
   when the build binds no L2 for the arch).

`passed` is true only when every gate holds, and `decide.py` exits 1 otherwise; the report
names each failing gate's problems. For a smoke round, read the gates, not the exit code: a
smoke round passes with only `accuracy` failing ([generate.md](generate.md), "Tier").
Every metric's model is validated separately.

## Install (only with the user's go-ahead)

Install into the user's checkout, on the branch they name. Inspect `git status` first: never
reset, stash or overwrite their work.

- **Descriptor engine**: `python3 -m uhd_gen promote --model-dir <model dir>
  --descriptor-tree <the UED's directory in the checkout> --engine <engine> --role <role>
  --arch <arch key> --feature-evaluator <build>/bin/hipdnn_uhd_features`. It installs the UHD
  and its payload under the UED's `heuristics/` and rebinds only the role map entry of the
  model's own metric; it refuses a UED it cannot bind safely.
- **Opaque engine**: `--descriptor-tree` is the provider directory holding the shipped
  `.uhd.json` and its payload (for AITER,
  `dnn-providers/hip-kernel-provider/src/engines/asm_sdpa_engine/descriptors/predict_engine/<arch>/`),
  with `--uhd-id <declared id>`. promote replaces the shipped model in place when it serves
  exactly this metric and arch, and refuses any other model under the id. An engine whose
  provider ships no descriptor tree (MIOpen) reads its models only from the descriptor roots
  the environment names (`HIPDNN_DESCRIPTOR_PATH`, `HIPDNN_DESCRIPTOR_RUNTIME_DIR`): there is
  no place in the checkout to install into. Deploy the model directory under such a root, and
  report that it is deployed, not installed.
- `--dry-run` first to see the plan. Then `git diff --stat` must list only the model's files
  (UHD, payload, and for a descriptor engine its UED); a descriptor engine's L2 and L1 go in
  together. Commit only when the user asks; the message names the model, device, revision,
  the sets it trained on and the Stage V verdict. Never push without permission.

## Metrics at runtime

With more than one metric bound under one arch key, the engine picks its kernel at plan build
by the effective ranking metric: `HIPDNN_HEUR_RANKING_METRIC`, else the heuristic
descriptor's ranking-metric attribute, else `tflops`. Validate each metric's model under its
own metric (`--metric` / `--ranking-metric`).

## Rollback

`git revert` the install commit. Before a commit, `git checkout -- <the model's files>` (and
remove the new `heuristics/` files) restores the shipped model. Keep backups of a previous UHD
pair outside the descriptor tree: every `*.uhd.json` under it is scanned, and a copy inside is
a duplicate installed identity that makes the next promote refuse.

## Report

Model directory, engine, device, role, metric, the graphs (slices, count), every gate with its
problems (L1: executed/available graphs, both models' median |error|; L2: candidate vs shipped
regret, `read`, winner records), the verdict, and -- when installed -- the checkout, branch,
files changed and commit, or that nothing was installed. A model measured on one product
serves every product of its arch once installed; say which products it was validated on.

## Common failures

| Symptom | Fix |
|---|---|
| `identity` fails | another document carries the declared id in the tree, or the installed copy differs from the model |
| `selector_revision` fails | the engine changed since training; collect at the current revision and retrain |
| `available` fails | the engine did not predict for some executed graphs; read `unavailable_sample` for its reasons |
| L1 opaque model never read | trained without the declared `--uhd-id` |
| L2 `read` fails | the runtime refused the ranker (a knob axis missing from the UED's knobs or the KMD fields, a features hash from another `hipdnn_uhd_features`); see [triage.md](triage.md) |
| L2 `winner_records` fails | the build carries a benchmark winner record that outranks any model |
| every config `config_not_applicable` | the tree handed in is one engine's folder, not the per-arch root that owns `kpack/` |
| promote refuses an invalid marker | training rows carry `numerically_valid: false`: a kernel or UMD fix, not a retrain |
| promote refuses the arch key | the UED's role map uses `default` or another key than the model was trained for |
