# Data

Everything here runs on the local machine, against the local build and GPU. Measurements are
**converted** once into a stored dataset, and every step that trains or scores **reads** the
dataset. The tools are hipDNN's own: `hipdnn_corpus_gen`, `hipdnn_bench`, `uhd_gen`
(`projects/hipdnn/tools/uhd_gen`) and `python3 -m uhd_gen.dataset`.

## The tools

- **A build for this GPU and this checkout.** An existing build directory serves only if all
  of these hold; otherwise make a new one (never reconfigure someone else's):
  - `GPU_TARGETS` in its `CMakeCache.txt` includes the device's arch;
  - `CMAKE_HOME_DIRECTORY` there is the checkout's path as you see it (a build configured at
    another mount point loads its libraries from that path and fails to start);
  - `<build>/lib/hipdnn_plugins/engines/` holds the plugin of the engine's provider, and a
    non-empty `arch_content/` when that provider stages descriptors or kernels there;
  - its tools are as new as the sources they are built from: no commit touching them is
    newer than the binaries
    (`git log -1 --format=%ci -- projects/hipdnn dnn-providers ':!projects/hipdnn/tools/ai'
    ':!projects/hipdnn/docs' ':!*.md'` against the binaries' times; skills and docs build
    nothing).
    A `hipdnn_corpus_gen` older than the checkout reads the current `corpus_gen/operations/`
    with `no declaration for operation` or `metadata error` lines.
- **Building.** From the checkout root (relative paths in this skill are from there), with
  the superbuild preset that builds hipDNN together with the engine's provider (one per
  provider in `CMakePresets.json`, e.g. `hip-kernel-provider`), for the device's arch, with
  the kernel ingestor on (the UHD tools and runtime build only with it):
  ```bash
  cmake --preset <provider preset> -B <build> -G Ninja -DGPU_TARGETS=<arch> \
    -DHIPDNN_ENABLE_KERNEL_INGESTOR=ON -DENABLE_CLANG_TIDY=OFF -DENABLE_CLANG_FORMAT=OFF
  cmake --build <build> -- -k 0
  ```
  Add what the provider's own configure step asks for; it stops with the option to set. For
  the hip-kernel-provider that is `-DHIPKERNELPROVIDER_ENABLE_ROCKE=ON` (its ingestor packs
  rocKE kernels) and, when configure reports `comgr loaded '<library>' instead of the
  requested '<stub>'`, `-DHIPKERNELPROVIDER_ROCKE_COMGR_LIB=<library>`. The checkout must be
  writable: the build generates untracked files into it (e.g. FlatBuffers headers), and on a
  read-only checkout it fails early; build from a writable clone. Build everything (`ALL`
  stages descriptors and kernels). Test binaries that fail to link do not matter here, so the
  build may exit non-zero; `-k 0` builds everything else. The build is usable when it has
  `bin/hipdnn_corpus_gen`, `bin/hipdnn_bench`, `bin/hipdnn_uhd_features`,
  `bin/hipdnn_list_engines`, `bin/hipdnn_validate_descriptors` and the provider's plugin under
  `<build>/lib/hipdnn_plugins/engines/`. Configure's `hipDNN: SDPA support disabled` is about
  the frontend's SDPA API; the tools and the engines do not need it.
- **Kernel files.** An engine that loads prebuilt kernels reads them from where its provider's
  build stages them beside the plugin (under `<plugin dir>/arch_content/`), for the archs that
  build lists; the provider's `CMakeLists.txt` names both. Without them every graph fails at
  collection with the engine's load error naming the missing file, although the shape set
  (which only asks whether the engine applies) succeeded.
- **uhd_gen** needs LightGBM, pandas and pyarrow. Run it from the checkout with
  `PYTHONPATH=<checkout>/projects/hipdnn/tools`, or install it into your own environment with
  `pip install -e projects/hipdnn/tools/uhd_gen`. Pass
  `--feature-evaluator <build>/bin/hipdnn_uhd_features` wherever `uhd_gen` takes it.
- **Containers.** When the tools run in a container, run it as yourself, so the store stays
  yours to reorganise, with the GPU's devices and the host's numeric ids of both groups that
  own them (a group name resolves inside the image, where its id can differ):
  ```bash
  docker run --rm --user $(id -u):$(id -g) --device=/dev/kfd --device=/dev/dri \
    --group-add $(getent group video | cut -d: -f3) \
    --group-add $(getent group render | cut -d: -f3) \
    -v <checkout>:<checkout> -v <build>:<build> -v <store>:<store> -v $UHD:$UHD \
    -e PYTHONPATH=<checkout>/projects/hipdnn/tools \
    -w <checkout> <image> ...
  ```
  Mount every directory a step reads or writes at its own path (the checkout, the build, the
  store, scratch, and `$UHD` for the scripts), and pass `PYTHONPATH`.
  On a machine whose GPUs are shared out (a scheduler grants you some), the container sees
  every GPU whose device node it is given, and an index such as `ROCR_VISIBLE_DEVICES` counts
  within your grant, not within the container's view. Give it only your GPU's nodes: read
  its PCI address where the grant applies (`amd-smi list`, outside the container), and pass
  `/dev/kfd` with `$(readlink -f /dev/dri/by-path/pci-<address>-render)` and
  `$(readlink -f /dev/dri/by-path/pci-<address>-card)` instead of all of `/dev/dri`; pass
  your granted CPUs as `--cpuset-cpus $(taskset -cp $$ | awk '{print $NF}')`. Check `rocminfo`
  inside the container lists the gfx agent before running any tool: without the render
  group the GPU is invisible, and every engine appears to serve nothing.
- After any CMake reconfigure of a provider that stages descriptors, rebuild its staging
  targets (or build `ALL`) before trusting an engine result: a reconfigure can empty
  `lib/hipdnn_plugins/engines/arch_content/`, and every engine then declines every graph.
- Plugin directories are passed as absolute paths; `hipdnn_list_engines --plugin-dir <abs>`
  prints each registered engine as `NAME (0xHEX)`.
- `uhd_gen` takes `--engine-id` as a signed decimal:
  `python3 -c "v=int('0xAD075A5EA86DD563',16); print(v-(1<<64) if v>=1<<63 else v)"`.
  `hipdnn_bench` and `hipdnn_corpus_gen` take either form.

## The device

A dataset holds one GPU product of one architecture. Read both off the device:

```bash
rocminfo | grep -m1 -E '^\s+Name:\s+gfx'          # the arch, e.g. gfx950
amd-smi static --asic | grep -m1 MARKET_NAME       # the product, e.g. AMD Instinct MI355X
```

Without `amd-smi`, the product is the `Marketing Name` of the agent whose `Name` is the gfx
arch; `rocminfo` lists CPU agents first, so the first `Marketing Name` is usually the CPU's.

The device key is `<arch>-<product>`: the marketing name with `AMD`, `Instinct` and `Radeon`
dropped, lowercased, spaces as `-` (`gfx950-mi355x`, `gfx1100-rx-7900-xtx`). Two products of
one arch never train together. A model installs per gfx arch, so it serves every product of
that arch; say so when the arch has other products. On a machine with several GPUs, measure
on one product at a time (`HIP_VISIBLE_DEVICES`, or `uhd_gen generate --device`).

## The local store

One directory the user names (default `~/uhd-work`), laid out so a key is its path:

| What | Where |
|---|---|
| shape sets (corpora) | `corpora/<engine>/<device>/<set>/<operation>/{graphs/, manifest.json, manifest.csv}` and `<set>/eval_slice.json` |
| raw collections | `collections/<engine>/<device>/<revision>/<date>/<set>-<time>/` (one `uhd_gen generate --collect-only` output each; `<date>` and `<time>` are the UTC `collected_at` in its `collection_manifest.json`, e.g. `2026-10-08/set-0001-20261008T002600Z`) |
| datasets | `datasets/<engine>/<device>/<revision>/<role>[/under-<L2 id>]/` (`dataset.parquet`, `dataset_manifest.json`) |
| models | `models/<model>/` (one `uhd_gen generate` output each) |
| archived sets | `archive/` (a set moved here counts for nothing) |

Write any `:` in an engine name (UED names are scoped, e.g. `hipkernel:Gfx950AttentionDense`)
as `-` in paths. The revision
segment is the selector revision with `/` as `-`.

- **Shape sets accumulate.** Sets are numbered in order (`set-0001`, `set-0002`, ...). A new
  set of an engine on a device passes `--exclude-corpus <set>/<operation>/manifest.json` for
  every active set of that engine and device, and a `--seed` none of them used (each
  `manifest.json` records its `seed`). corpus_gen walks through the excluded configurations
  and returns new ones; together the sets are one growing set. To start a key afresh, move
  its sets to `archive/`.
- **Only a set that succeeded is active.** corpus_gen writes its output directory even when
  it exits 2 (an empty set) or 1; move such a directory to `archive/` at once, so no later
  set excludes it or trains from it.
- **Evaluation slice.** Every set reserves a tenth of each regime once, when it is made, with
  the skill's `scripts/eval_slice.py`: `eval_slice.py choose <set>/*/manifest.json` (every
  operation of the set) writes `<set>/eval_slice.json`.
  Every model trained from the set holds those graphs out (`uhd_gen generate
  --eval-benchmarks`; several sets: `eval_slice.py holdout <slices> --out
  eval_benchmarks.json`), and is scored and validated on them.
- **Closed shape set.** An engine `projects/hipdnn/tools/corpus_gen/operations/engines.json`
  lists with `coverage: pack` serves exactly its pack: corpus_gen takes the pack
  (`--kdp-root`), there is no slice, and the model trains on every shape and is scored by
  recall (`uhd_gen generate --recall`).
- **Selector revision.** Every collection records the engine's `selector_revision` in its
  `collection_manifest.json`. Rows of different revisions never train together; a revision
  change (the engine's code changed) needs new measurements.
- A **contribution** is one source's rows (a collection, an old generate folder, a CSV set).
  Each row records its contribution, when it was measured, its `purpose` (`training`,
  `holdout` or `eval`) and its corpus. `dataset.parquet` keeps every row, failures and
  repeats included. `training.parquet`, rewritten after every change, is what training reads:
  the `training` rows that carry a usable measurement, one per configuration per shape, the
  one with the lowest `avgTimeMs`.
- A contribution's **id** is derived from its binding and rows, so the same measurements are
  stored once whatever they are called. Its **label** is the name it was added under
  (`--label`; default the collection's directory, the generate folder, or the first CSV's
  name). `--contribution` takes either.
- **Corrections**: `add` only appends. `python3 -m uhd_gen.dataset remove --from <dataset dir>
  --contribution <ids or labels>` drops contributions; `add ... --replace <ids or labels>`
  swaps them for the source being added, in one write; `python3 -m uhd_gen.dataset retag --in
  <dataset dir> --contribution <ids or labels> --purpose <purpose>` (or `--corpus-id`) changes
  a tag and keeps the measurements. Each is recorded in the manifest's `history`, with the
  removed entry.

## Workflow

1. **What is stored**: list `datasets/`; each `dataset_manifest.json` lists every
   contribution with its date, devices, purpose and origin. A model's
   `models/<model>/generation_manifest.json` names the dataset and contributions it
   trained on, and under `models[i]` its `training_problem_keys` / `eval_problem_keys`.
2. **Measure more**: [generate.md](generate.md) makes a set, collects it and adds each collection.
3. **Add a collection**:
   `python3 -m uhd_gen.dataset add --into <dataset dir> --collection <collection dir>...
   --corpus-id <set>`. The binding (engine, role, revision) comes from the collection; one
   that contradicts the dataset is refused whole. Adding the same measurements again adds
   nothing, under any label.
4. **Convert measurements made elsewhere** with `python3 -m uhd_gen.dataset add --into ...`:

   | Source | Flags | Notes |
   |---|---|---|
   | hipdnn_bench / §8.3 CSVs | `--csv <files>` with `--engine`, `--engine-id`, `--arch`, `--selector-revision`, `--collected-at`, and `--role` when it is not `sort_kernel_catalog`; `--label` names it | CSV rows carry no binding, so all of it is stated; the dataset directory names the device. Where a column holds a configuration string, `--expand-descriptor <that column> --scope-by <the column naming its schema>` splits it into positional columns so a catalog ranker can tell configurations apart (e.g. `kernel.descriptor` scoped by `kernel.solver_id`); `--resolve-duplicates` / `--best-column` decide repeated rows |
   | an old generate folder | `--generation <dir>` | its measurements, converted rather than re-measured |
   | raw collections | `--collection <dirs>` | as step 3 |

   `--purpose` (default `training`) and `--corpus-id` apply to every source. Measurements made
   for evaluation are `--purpose eval`, or they are trained on and every later evaluation on
   them is recall.
5. **Train from stored data**: `uhd_gen generate --dataset <dataset dir>` (every training
   contribution) or with `--contribution <ids or labels>`; the rest of the command is as in
   [generate.md](generate.md). A descriptor engine's L1 trains from the dataset measured under its
   L2 (`under-<L2 id>`).
6. **Size from stored data**: `uhd_gen size --dataset <dataset dir> --test-set
   <dataset>/sizing/test_set.json --create-test-set --target 0.7 --tolerance 0.10
   --corpus-manifest <set>/<operation>/manifest.csv ... --descriptor-tree <tree> --engine ...
   --engine-id ... --arch ... --feature-evaluator ... --output-dir <dir>` fits a learning
   curve on a pinned test set (created once, reused every round) and writes
   `regime_quotas.json`; `--previous <last sizing_report.json>` checks the last round's
   prediction. Grow the set with those quotas: `hipdnn_corpus_gen ... --regime-quotas
   <regime_quotas.json> --count 0` (the quotas alone), excluding every active set.

## Published engines

An engine whose measurements come from another source only converts and trains: no shape
set, collection or validation. Its rows name no label or features, so they are stated at
generate: `--target`, `--objective`, `--features` and, for a two-layer ranker,
`--group-by-feature` (the column naming a candidate's group, e.g. `kernel.solver_id`). Use the
source's numeric problem columns, its group column and the positional columns
`--expand-descriptor` produced as features.

## Decision rules

- Re-measure only what is not stored. A training change (features, boosting, a new
  `uhd_gen`) retrains from the dataset; a revision change needs new measurements.
- A refused conversion was measured under another binding; it belongs in another dataset.
  Do not force it.
- A CSV with no FLOP count cannot derive `tflops`: train with `--target minTimeMs
  --objective min` (it ranks identically within a problem).
- Never choose a device or merge two devices' rows for the user.

## Report

The dataset path, device, revision and role; training rows/graphs and contributions (dates,
devices, purposes, origins); for a conversion, the contribution added (or that it was there
already) and its counts; for training, the contributions read and the model directory.

## Common failures

| Symptom | Fix |
|---|---|
| `dataset disagrees on selector_revision` / binding differs | the source was measured under another binding: another dataset |
| `knob addressing contradicts the dataset` | the source encodes a knob differently from the rows stored; find which is stale |
| `--csv needs ...: the rows carry no binding` | pass the flags it names (`--engine`, `--engine-id`, `--arch`, `--selector-revision`, `--collected-at`) |
| generate: no training data | nothing collected or converted for that engine/device/role at that revision |
| generate: contribution not in the dataset | it is in another revision's or role's dataset, or is holdout/eval |
| every engine declines every graph | the descriptor staging was emptied by a reconfigure: rebuild the staging targets |
