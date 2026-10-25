# Generate

One engine, one device, all on the local machine:

shape set (`hipdnn_corpus_gen`, asks the engine on the GPU) → evaluation slice
(`eval_slice.py choose`) → per role: collect (`uhd_gen generate --collect-only`, timed on the
GPU) → store (`python3 -m uhd_gen.dataset add`) → train (`uhd_gen generate --dataset`, scored
on the slice).

The store layout, the device key, the tools and the slice are in [data.md](data.md);
`$UHD/scripts/` is the skill's scripts directory ([SKILL.md](SKILL.md)). Use
[evaluate.md](evaluate.md) to judge a model, [validate.md](validate.md) to check it installed
and put it in the checkout, [triage.md](triage.md) when a step fails, and
[onboarding.md](onboarding.md) when the engine has never been trained.

## Inputs

Take from the user; never guess silently:

- **engine**: the name hipDNN registers for it (e.g. `hipkernel:Gfx950AttentionDense`). When
  the user names an operation or a kind of engine instead ("SDPA", "conv"), find the
  candidates from the checkout, before any build:
  - every name registered with `HIPDNN_REGISTER_ENGINE` in
    `projects/hipdnn/data_sdk/include/hipdnn_data_sdk/utilities/EngineNames.hpp`, and the code
    that implements it (`grep -rn <NAME> dnn-providers`), whose plan builders name the
    operations it serves. A name with no implementation is reserved, not an engine;
  - the `name` of every UED built into a plugin (below), whose `graph_match` names its
    operation.

  Keep those an engine model can be trained for (below: a model binding) and that serve the
  operation on this device's arch. None left: stop as in step 0. More than one: ask which.
  `hipdnn_list_engines` on a build says which of them that build registers.
- **device**: the local GPU ([data.md](data.md), "The device"); with several products present,
  ask which one.
- **tier**: `smoke` (200 graphs), `standard` (2000) or `exhaustive` (5000) **per operation**:
  an engine serving two operations gets twice the graphs. Say so in the report. `smoke` first
  on any new engine or branch.
- **build** and **store** directories (store default `~/uhd-work`).

## What the engine is

Read it from the checkout. An engine model is only ever read through a **binding**, and the
binding decides everything below:

- **Descriptor engine**: a UED (`*.ued.json`) whose `name` is the engine, that the provider's
  build stages into the plugin directory: after a build, a `*.ued.json` under
  `<plugin dir>/arch_content/` (`grep -rl '"name": "<engine>"' <plugin dir>/arch_content`); in
  source, one under the root the provider's `CMakeLists.txt` stages descriptors from (e.g.
  the hip-kernel-provider's `src/engines/*/descriptors/`), never one under `tests`,
  `test_descriptors`, `fixtures`, `integration_tests` or `examples`. Its role maps are its
  bindings: `sort_kernel_catalog` (L2, ranks its kernel catalog) and/or `predict_engine` (L1,
  predicts its own selection), each keyed by gfx arch or `default`; that key is the `--arch`
  generate and promote take. Its **per-arch root** is the arch's shard of the staged tree: the
  directory named for the arch on the path from the provider's staged directory down to the
  installed UED (for a provider that packs kernels, the directory that also holds `kpack/`).
  Hand that root, never the UED's own folder, to every tool.
- **Opaque engine** (no UED): L1 only, bound in its provider's code. The engine overrides
  `getPrediction` and passes the UHD ids it reads to the plugin SDK's binding
  (`EngineModelBinding` / `resolveDeclaredEnginePredictions`, in
  `projects/hipdnn/plugin_sdk/include/hipdnn_plugin_sdk/`); grep its provider for those to find
  the ids. Each id is declared per arch, or under `default` for every arch (the model's own
  `training_arches` then decide which archs it answers for), and per metric. The model must be
  trained under exactly the id declared for its arch and metric (`--uhd-id`): under any other
  id it installs and is never read. An engine with no binding cannot be trained: its
  `getPrediction` answers `Engine does not provide predictions`.
- **Archs**: where a model of the engine can be trained and read, per operation: the archs its
  binding names (role-map gfx keys; with only `default`, the `arch` lists of its KDPs; for an opaque
  engine, its declared ids' archs, every arch under `default`), **and** that the operation's
  plan builder accepts (its `isApplicable` check). Both must hold for each operation
  separately.
- **Operation**: the corpus_gen operation (`projects/hipdnn/tools/corpus_gen/operations/`)
  whose graph the engine serves -- not its name. For a descriptor engine, the one its UED's
  graph matcher accepts (the native symbol in `graph_match.native`, plus any `scope: graph`
  UMD); for an opaque engine, those of the plan builders it registers. When unsure, a census
  run names them ([corpus-facets.md](corpus-facets.md)).
- **Closed shape set**: listed `coverage: pack` in
  `projects/hipdnn/tools/corpus_gen/operations/engines.json`.

## Workflow

0. **Serves this device**: the device's arch ([data.md](data.md), "The device") is among the
   engine's archs (above). If it is not, stop before building anything: the engine cannot be
   trained on this GPU. Report the archs it serves, and the engines that do serve this arch
   (the same reading over every candidate; with a passing build, a census of each registered
   engine, below, confirms it on the GPU).
1. **Build** ([data.md](data.md), "The tools"): an existing build only if it passes every
   check there.
2. **Registers**: `<build>/bin/hipdnn_list_engines --plugin-dir <abs plugin dir>` prints the
   engine and its id. Not listed: [triage.md](triage.md).
3. **Shape set**, one directory per operation `op` the engine serves (corpus_gen offers each
   proposed problem to the engine on the GPU); `--keep q.p=v` narrows to a facet
   ([onboarding.md](onboarding.md)):
   ```bash
   <build>/bin/hipdnn_corpus_gen \
     --operation <op> --engine-name <engine> --plugin-dir <abs plugin dir> \
     --output <store>/corpora/<engine>/<device>/<set>/<op> --count <tier count> \
     --seed <unused seed> --bench-path <build>/bin/hipdnn_bench \
     [--exclude-corpus <each active set>/<op>/manifest.json ...] [--keep q.<p>=<v> ...] \
     [--kdp-root <pack or dir>]
   ```
   Exclude every active set of this engine and device; take a seed none of them used. A
   descriptor engine's packs are a source (`--kdp-root` with the engine's installed
   directory); a closed shape set requires them and is not searched. Exit 0 is a set (a
   warning when the engine serves fewer than `--count`); 2 is an empty set; 3 is a short set
   the engine was not shown to exhaust. In its summary, `<op>: N problems` is every problem
   the search found the engine serves, and `Generated M problems` the set after the cut to
   `--count`; `K of T declared regime combinations not tried (maxSkeleton bound)` means the
   first pass tried `--max-skeleton` (default 512) of the operation's declared combinations,
   which is expected; so are `no declared archetype was admitted for ...; that combination is
   exploration only` lines ([triage.md](triage.md), "Harmless noise"). Move a set that did
   not exit 0 to `archive/` ([data.md](data.md)). On exit 2, rerun one operation with
   `HIPDNN_LOG_LEVEL=info --count 1` into a scratch directory, its log to a file, and look for
   the engine's own lines: its plan builders log why they decline a graph (e.g. `isApplicable`
   lines), and that refusal says why it serves nothing. No refusal but device-query errors
   means the GPU was not visible ([triage.md](triage.md)).
4. **Evaluation slice** (not for a closed shape set):
   `python3 $UHD/scripts/eval_slice.py choose <set>/*/manifest.json`, every
   operation of the set in one slice.
5. **Collect, per role, L2 first** (one GPU run each; `--output-dir` must not exist). One
   collection covers every operation of the set (`--graphs` takes each `<set>/<op>`):
   ```bash
   python3 -m uhd_gen generate --graphs <set>/<op> ... --collect-only \
     --engine <engine> --engine-id <signed decimal id> --role <role> \
     --bench <build>/bin/hipdnn_bench --plugin-dir <abs plugin dir> \
     --descriptor-tree <tree> --output-dir <stage dir> [--shard K/N]
   ```
   `<stage dir>` is any new directory (scratch will do); the collection moves into the store
   below.
   `--descriptor-tree` is the ONE tree the engine loads while it is measured
   (`HIPDNN_DESCRIPTOR_DIR`), so it is the tree the build installs for it: a descriptor
   engine's per-arch root; for an opaque engine, the directory its provider's
   `CMakeLists.txt` stages descriptors into under `<plugin dir>/arch_content/`, or an empty
   directory when the build stages none for that provider. A descriptor engine's L1 is measured under the L2 it will be
   installed with: build a candidate tree first with `python3
   $UHD/scripts/candidate_tree.py
   --root <per-arch root> --farm <new dir> --promote <engine> sort_kernel_catalog <arch key>
   <L2 model dir> --feature-evaluator <build>/bin/hipdnn_uhd_features`, pass that `--farm` as `--descriptor-tree`, and run the collection with
   `HIPDNN_CACHE_DIR` set to a new empty directory, so no benchmark winner record decides the
   kernel in the L2's place. Several GPUs of the device: one `--shard K/N` each, one
   collection each.
   Then move each collection to `collections/<engine>/<device>/<revision>/<date>/<set>-<time>/`
   ([data.md](data.md)), the revision and `collected_at` read from its
   `collection_manifest.json`. The engine reports one selector revision for all its
   operations, even when the revision's name mentions only one.
   The directory's name becomes the contribution's label in the dataset, which
   `--contribution` selects by.
6. **Store**: `python3 -m uhd_gen.dataset add --into <store>/datasets/<engine>/<device>/<revision>/<role>[/under-<L2 id>]
   --collection <collections> --corpus-id <set>`.
7. **Train** (CPU):
   ```bash
   python3 -m uhd_gen generate --dataset <dataset dir> \
     --descriptor-tree <tree> --engine <engine> --engine-id <id> --role <role> --arch <arch key> \
     --eval-benchmarks <slice or holdout.json> --metric <metric> \
     --feature-evaluator <build>/bin/hipdnn_uhd_features --no-promote \
     --output-dir <store>/models/<model> [--uhd-id <declared id>] [--recall]
   ```
   `<tree>`: an empty directory for an opaque engine (promote's check refuses a declared id
   already installed in the tree), the per-arch root for a descriptor engine. Several sets:
   `eval_slice.py holdout` writes their union. A closed shape set: `--recall` instead of
   `--eval-benchmarks`. `--metric` takes `tflops` or `time` (repeatable, one model each). Train
   one per metric the role binds: the `score.metric` of each UHD the binding names (a bound
   UHD with no `score.metric`, e.g. a `native` adapter, binds none). With none bound, train
   `tflops` when the engine publishes `graph.flops`; without it, `uhd_gen` falls back to a
   time-ranked model with no metric and says so. Train the L2, then collect and train the L1
   under it (step 5).
8. **Report** and hand off: each model directory is `models[i].model_dir` in
   `<output dir>/generation_manifest.json` (the directory holding `heuristic.uhd.json`, the
   payload and `eval_report.json`), beside that model's `training_problem_keys` and
   `eval_problem_keys`. [evaluate.md](evaluate.md) reads it, then [validate.md](validate.md).

## Decision rules

- **One device.** Every set, collection, dataset and model is one product of one arch.
- **Roles.** L2 before L1, and the L1 is collected under the L2 just trained: an L1 trained
  before its L2 models a selector that will not be installed.
- **Accumulate.** A second set for the same engine and device excludes the first; it does not
  replace it. Archive a key's sets ([data.md](data.md)) to start afresh.
- **Revision.** Collections of different selector revisions go to different datasets; a model
  trains from one.
- **Tier.** A smoke round proves the pipeline, not the model: it passes when every graph is
  collected, the model trains and is scored held out, and Stage V's identity, available,
  metric and selector-revision gates pass. Its accuracy gate is reported, and a smoke model is
  expected to lose to an incumbent model trained on far more graphs. Move to `standard` after a
  clean smoke round; only a model that also passes the accuracy gate is installed.
- Nothing here changes the checkout: models are written under the store, and installing one
  is done in [validate.md](validate.md), with the user's go-ahead.

## Report

Engine (kind, roles, metrics), device, the set(s) and graph counts per operation, the slice
size, collections and revision, the dataset, each model directory with its held-out score
(`eval_report.json`: problems scored, the headline metric, every warning verbatim) and how it
was scored (`eval_benchmarks`, `holdout` or `recall`).

## Common failures

| Symptom | Fix |
|---|---|
| engine not listed by `hipdnn_list_engines` | relative `--plugin-dir`, the descriptors were not staged, or the engine is not built for this arch ([triage.md](triage.md)) |
| corpus_gen exit 2, "Generated 0 problems" | the engine serves nothing proposed: it does not run on this arch (step 0; the INFO rerun shows the plan builder's refusal), or the wrong operation or `--keep` facets ([onboarding.md](onboarding.md)) |
| corpus_gen `no declaration for operation`, `metadata error` | the build is older than the checkout's `corpus_gen/operations/`: use a build of this checkout ([data.md](data.md)) |
| a tool fails to start, `lib...so: cannot open shared object file` | the build was configured at another path: use a build configured at this checkout's path ([data.md](data.md)) |
| corpus_gen refuses a pack-coverage engine | pass `--kdp-root` |
| `--output-dir must not exist` | each run writes a new directory |
| every graph fails collection although the shape set succeeded, the bench log naming a kernel file or archive it cannot open | the build staged no kernels for this arch ([data.md](data.md), "Kernel files") |
| `N of M graph(s) failed collection` | read `failed_graphs` in `generation_manifest.json`: one kernel, or the device |
| generate: no training data | the dataset holds no `training` rows at that revision/role |
| an opaque model installs but the engine never reads it | trained without the id its binding declares (`--uhd-id`) |
