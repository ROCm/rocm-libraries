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

- **engine**: the name hipDNN registers (`ASM_SDPA_ENGINE`, `hipkernel:Gfx950ConvFwd`). When
  the user names an operation or a kind of engine instead ("SDPA", "conv"), find the
  candidates from the checkout, before any build:
  - the `HIPDNN_REGISTER_ENGINE` names in
    `projects/hipdnn/data_sdk/include/hipdnn_data_sdk/utilities/EngineNames.hpp`, each
    implemented under its provider's `src/engines/<name lowercased>/` (MIOpen's under
    `dnn-providers/miopen-provider/engines/`), whose plan builders
    name the operations it serves (a name listed there need not be built by this checkout:
    `hipdnn_list_engines` on a build says what registers);
  - the `name` of every shipped `*.ued.json` (below), whose `graph_match` names its operation.

  Drop a name with no engine directory (it is reserved, not implemented) and an opaque
  engine that declares no UHD id (a model of it is never read). Keep those that serve the
  operation on this device's arch (below). None left: stop as in step 0. More than one: ask
  which.
- **device**: the local GPU ([data.md](data.md), "The device"); with several products present,
  ask which one.
- **tier**: `smoke` (200 graphs), `standard` (2000) or `exhaustive` (5000) **per operation**:
  "200 shapes" for an engine serving `sdpa_fwd` and `sdpa_bwd` is 400 graphs. Say so in the
  report. `smoke` first on any new engine or branch.
- **build** and **store** directories (store default `~/uhd-work`).

## What the engine is

Read it from the checkout:

- **Descriptor engine**: a shipped `*.ued.json` whose `name` is the engine. Shipped UEDs are
  under a provider's `src/engines/*/descriptors/`; those under `tests`, `test_descriptors`,
  `fixtures` and `descriptor-packaging/examples/` are not built into the plugin. Its role
  maps name the roles it binds: `sort_kernel_catalog` (L2, ranks its kernel catalog) and/or
  `predict_engine` (L1, predicts its own selection). Its
  per-arch root is the directory under `<plugin dir>/arch_content/hip-kernel-provider/` that
  holds both its installed UED and `kpack/`
  (`find <plugin dir>/arch_content -name kpack -type d`). The role map's key is the gfx arch or
  `default`; that key is the `--arch` generate and promote take.
- **Opaque engine** (no descriptor set): L1 only, bound by UHD ids its provider declares in
  code. Either one per arch -- AITER's `ASM_SDPA_ENGINE`, `L1_MODEL_IDS` in
  `dnn-providers/hip-kernel-provider/src/engines/asm_sdpa_engine/AsmSdpaEngine.hpp` -- or one
  per metric for every arch, bound under `default` -- `MIOPEN_ENGINE` and
  `MIOPEN_ENGINE_DETERMINISTIC`, `MIOPEN_ENGINE_L1_MODELS` and
  `MIOPEN_ENGINE_DETERMINISTIC_L1_MODELS` in `dnn-providers/miopen-provider/MiopenContainer.cpp`,
  where the model's own `training_arches` decide which archs it answers for. The model must be
  trained under exactly the declared id for its arch and metric (`--uhd-id`): under any other
  id it installs and is never read.
- **Archs**: where a model of the engine can be trained and read, per operation:
  - a descriptor engine: the gfx keys of its UED role maps; a `default` key names no arch,
    so with only `default` the archs are its KDP `targets`;
  - an opaque engine: the archs it declares a UHD id for (every arch, when its ids are bound
    under `default`; none, when it declares no id -- such an engine cannot be trained),
    **and** that the operation's plan builder accepts (its `isApplicable`
    check). Both must hold: AITER declares gfx942 and gfx950, but its backward builder
    accepts gfx942 only, so `sdpa_bwd` has no model on gfx950.
- **Operation**: the corpus_gen operation (`projects/hipdnn/tools/corpus_gen/operations/`)
  whose graph the engine serves -- not its name. For a descriptor engine it is the one its
  UMD's graph match accepts; for an opaque engine, the operations of the plan builders it
  registers (an SDPA engine's forward and backward builders serve `sdpa_fwd` and
  `sdpa_bwd`). When unsure, a census run names them ([corpus-facets.md](corpus-facets.md)).
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
   <build>/bin/hipdnn_corpus_gen --operations <checkout>/projects/hipdnn/tools/corpus_gen/operations \
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
   exploration only` lines ([triage.md](triage.md), "Harmless noise"). Move a set that did not exit 0 to `archive/`
   ([data.md](data.md)). On exit 2, rerun one operation with `HIPDNN_LOG_LEVEL=info --count 1`
   into a scratch directory, its log to a file, and look for the engine's own lines
   (`grep isApplicable`): a refusal such as `Device string does not match` says why it serves
   nothing. No such line, but `Could not query device string`, means the GPU was not visible
   ([triage.md](triage.md)).
4. **Evaluation slice** (not for a closed shape set):
   `python3 $UHD/scripts/eval_slice.py choose <set>/*/manifest.json`, every
   operation of the set in one slice.
5. **Collect, per role, L2 first** (one GPU run each; `--output-dir` must not exist), with
   the engine's kernel-file variables set ([data.md](data.md), "Kernel files"). One collection
   covers every operation of the set (`--graphs` takes each `<set>/<op>`):
   ```bash
   python3 -m uhd_gen generate --graphs <set>/<op> ... --collect-only \
     --engine <engine> --engine-id <signed decimal id> --role <role> \
     --bench <build>/bin/hipdnn_bench --plugin-dir <abs plugin dir> \
     --descriptor-tree <tree> --output-dir <stage dir> [--shard K/N]
   ```
   `<stage dir>` is any new directory (scratch will do); the collection moves into the store
   below.
   `--descriptor-tree` is the ONE tree the engine loads while it is measured
   (`HIPDNN_DESCRIPTOR_DIR`), so it is the tree the engine ships in: for an opaque engine of
   the hip-kernel-provider, its descriptor root (`<plugin dir>/arch_content/hip-kernel-provider`);
   for an engine whose provider ships no descriptor tree (MIOpen), an empty directory; for a
   descriptor engine, the per-arch root. A descriptor engine's L1 is measured under the L2 it will ship
   with: build a candidate tree first with `python3
   $UHD/scripts/candidate_tree.py
   --root <per-arch root> --farm <new dir> --promote <engine> sort_kernel_catalog <arch key>
   <L2 model dir>`, pass that `--farm` as `--descriptor-tree`, and run the collection with
   `HIPDNN_CACHE_DIR` set to a new empty directory, so no benchmark winner record decides the
   kernel in the L2's place. Several GPUs of the device: one `--shard K/N` each, one
   collection each.
   Then move each collection to `collections/<engine>/<device>/<revision>/<date>/<set>-<time>/`
   ([data.md](data.md)), the revision and `collected_at` read from its
   `collection_manifest.json`. The engine reports one selector revision for all its operations, even when
   its name mentions one of them (`asm-sdpa-fwd` covers `sdpa_bwd` too).
   The directory's name becomes the contribution id in the dataset, which `--contribution`
   selects by.
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
   `--eval-benchmarks`. One model per metric the role binds (`--metric` repeatable). Train the
   L2, then collect and train the L1 under it (step 5).
8. **Report** and hand off: each model directory is `models[i].model_dir` in
   `<output dir>/generation_manifest.json` (the directory holding `heuristic.uhd.json`, the
   payload and `eval_report.json`), beside that model's `training_problem_keys` and
   `eval_problem_keys`; the install line `uhd_gen` prints names a temporary path, so use the
   manifest's;
   [evaluate.md](evaluate.md) reads it, then [validate.md](validate.md).

## Decision rules

- **One device.** Every set, collection, dataset and model is one product of one arch.
- **Roles.** L2 before L1, and the L1 is collected under the L2 just trained: an L1 trained
  before its L2 models a selector that will not ship.
- **Accumulate.** A second set for the same engine and device excludes the first; it does not
  replace it. Archive a key's sets ([data.md](data.md)) to start afresh.
- **Revision.** Collections of different selector revisions go to different datasets; a model
  trains from one.
- **Tier.** A smoke round proves the pipeline, not the model: it passes when every graph is
  collected, the model trains and is scored held out, and Stage V's identity, available,
  metric and selector-revision gates pass. Its accuracy gate is reported, and a smoke model is
  expected to lose to a shipped model trained on far more graphs. Move to `standard` after a
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
| corpus_gen exit 2, "Generated 0 problems" | the engine serves nothing proposed: it does not run on this arch (step 0; the INFO rerun shows e.g. `Device string does not match ...`), or the wrong operation or `--keep` facets ([onboarding.md](onboarding.md)) |
| corpus_gen `no declaration for operation`, `metadata error` | the build is older than the checkout's `corpus_gen/operations/`: use a build of this checkout ([data.md](data.md)) |
| a tool fails to start, `lib...so: cannot open shared object file` | the build was configured at another path: use a build configured at this checkout's path ([data.md](data.md)) |
| corpus_gen refuses a pack-coverage engine | pass `--kdp-root` |
| `--output-dir must not exist` | each run writes a new directory |
| every graph fails collection with `failed to load kernel module` (after a shape set that succeeded) | the engine's kernel files are not where it looks: set its kernel-file variables ([data.md](data.md)) |
| `N of M graph(s) failed collection` | read `failed_graphs` in `generation_manifest.json`: one kernel, or the device |
| generate: no training data | the dataset holds no `training` rows at that revision/role |
| an opaque model installs but the engine never reads it | trained without the declared `--uhd-id` |
