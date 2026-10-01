# Reproducing the shipped heuristics

Everything needed to regenerate, re-measure and independently check the L1/L2 models this
branch ships. No measured performance numbers are recorded here: the procedure produces
them on your own hardware, which is the only place they mean anything.

## What this branch ships

| engine | arch | roles (metrics) | binding |
|---|---|---|---|
| `hipkernel:Gfx950AttentionDense` | gfx950 | `sort_kernel_catalog`, `predict_engine` (`tflops`, `time`) | UED role map; the pack and its `heuristics/` tree are in the shipped descriptor root, `dnn-providers/hip-kernel-provider/src/engines/kernel_ingestor_engine/descriptors/rocKE/gfx950_attention_dense/` |
| `ASM_SDPA_ENGINE` (AITER) | gfx942, gfx950 | `predict_engine` (`tflops`) | UUID declared in `AsmSdpaEngine.hpp` (`L1_MODEL_IDS`); documents staged from `src/engines/asm_sdpa_engine/descriptors/predict_engine/<arch>/` |
| `MIOPEN_ENGINE`, `MIOPEN_ENGINE_DETERMINISTIC` | `default` | `predict_engine` (`tflops`, `time`) | UUIDs declared in `MiopenContainer.cpp`; the ids ship, the models are trained with `generate.sbatch` below |

There is no gfx942 `attention_dense` pack: on gfx942 the rocKE provider registers no dense
attention engine, and AITER is the SDPA engine that carries a model there. The
`descriptor-packaging/examples/descriptors` tree is a packaging fixture, not the shipped
root; `UHD_PRODUCTION_ROOT=<path relative to the checkout>` packs it (or any other tree)
instead when a script needs it.

An engine with no descriptor set (AITER, MIOpen) has its model bound by the UUID the
provider declares (RFC 0019 §4.1, Open Question 7) rather than by a role map. The
document's `id` IS the binding: change it and the engine silently reports no model.

## 0. What every job needs

- **`UHD_BRANCH` is required.** Every script clones `https://github.com/ROCm/rocm-libraries.git`
  at that branch, so the branch must be pushed; `UHD_BUNDLE` (last section) carries commits
  the clone cannot see. `corpus5000.sbatch` with `UHD_BUILD` reuses a build and clones nothing.
- **Site settings are passed at submission.** The `#SBATCH` lines in the scripts are
  defaults only; `--partition`, `--account` and `--constraint` on the `sbatch` command line
  override them, and are whatever your site calls them.
- **`/exchange` is the container mount point.** The scripts read corpora and models from,
  and keep their results under, `/exchange`; the submission maps a node-visible directory
  there with `--container-mounts=<dir>:/exchange` (`$HOME` below). Without the container
  flags the script runs on the bare node, where `/exchange` does not exist and apt refuses
  to install.
- **The submit directory and `--output` must exist on the compute node.** Submit with
  `--chdir=/tmp` and point `--output` at a node-visible directory; `generate.sbatch` and
  `corpus5000.sbatch` also tee their whole log into the kept directory, because the
  container's view of the submit directory is discarded with it.
- **The sbatch files need LF line endings.** `sbatch` refuses a script with DOS line
  breaks, which is what a Windows checkout with `core.autocrlf=true` writes; convert with
  `tr -d '\r'` (or `dos2unix`) before submitting from such a tree.

A working gfx950 collection, L2 then L1, both metrics:

```bash
sbatch --partition=<p> --account=<a> --constraint=GFX950 --gres=gpu:1 \
  --chdir=/tmp --output=<node-visible dir>/%x-%j.log \
  --container-image=docker://rocm/dev-ubuntu-24.04:7.14.0-full \
  --container-writable --container-remap-root --container-mounts=$HOME:/exchange \
  --export=ALL,UHD_BRANCH=<branch>,UHD_ENGINE=hipkernel:Gfx950AttentionDense,UHD_ARCH=gfx950,UHD_ROLES=l2+l1,UHD_L2_METRICS=tflops+time,UHD_L1_METRICS=tflops+time,UHD_GRAPHS=/exchange/<corpus>,UHD_KEEP=/exchange/<out> \
  generate.sbatch
```

The examples below abbreviate the common flags as:

```bash
SUBMIT="sbatch --partition=<p> --account=<a> --gres=gpu:1 \
  --chdir=/tmp --output=<node-visible dir>/%x-%j.log \
  --container-image=docker://rocm/dev-ubuntu-24.04:7.14.0-full \
  --container-writable --container-remap-root --container-mounts=$HOME:/exchange"
```

## 1. Build the corpus (per engine, on a GPU)

A corpus is generated **for an engine**: every candidate problem is offered to it, so what
comes out is what that engine serves. `--engine-name` is required and needs the provider
built and staged, so this runs on a GPU node -- through `corpus5000.sbatch` (both SDPA
engines of the node's arch; `UHD_ARCH`, `UHD_COUNT`, `UHD_SEED`, `UHD_OUT`), or by hand in
a job as below. Deterministic from the seed and the in-tree inputs, and `benchmark` ids are
content-derived so results join across runs.

```bash
$SUBMIT --constraint=GFX950 \
    --export=ALL,UHD_BRANCH=<branch>,UHD_ARCH=gfx950,UHD_OUT=/exchange/corpus-950 \
    corpus5000.sbatch
```

By hand:

```bash
cd projects/hipdnn/tools
GEN=<build>/bin/hipdnn_corpus_gen
PLUGINS=<build>/lib/hipdnn_plugins/engines
PACKS=../../../dnn-providers/hip-kernel-provider/src/engines/kernel_ingestor_engine/descriptors/rocKE

# rocKE, gfx950: its pack proposes, the engine admits. This is also the comparison corpus
# of step 3.
$GEN --operations corpus_gen/operations --operation sdpa_fwd --plugin-dir $PLUGINS \
    --engine-name hipkernel:Gfx950AttentionDense --kdp-root $PACKS/gfx950_attention_dense \
    --output /tmp/corpus-950 --count 1000 --seed 0

# AITER, gfx950: what AITER serves, with the comparison graphs held out by construction.
$GEN --operations corpus_gen/operations --operation sdpa_fwd --plugin-dir $PLUGINS \
    --engine-name ASM_SDPA_ENGINE --exclude-corpus /tmp/corpus-950/manifest.json \
    --output /tmp/corpus-950-aiter --count 2500 --seed 11
```

Graphs land in `<output>/graphs/`, beside `manifest.json`. `--count` is met unless the
engine physically serves fewer: an engine whose kernels match exact geometries is capped
at its pack's shapes, and the tool says so and exits 0. Any other shortfall exits 3.

No `--keep` is needed to narrow a corpus to an engine's facets: the engine decides. For
reference, AITER's gfx942 forward table is four kernels (bf16, hd128/hd192->128, no mask
and bottom-right causal) and its gfx950 table is two, with no causal kernel at all:

```bash
python3 -c "import csv,sys; rows=list(csv.DictReader(open(sys.argv[1])));
print(len(rows), 'kernels'); [print(r) for r in rows]" \
    dnn-providers/hip-kernel-provider/src/engines/asm_sdpa_engine/asm/asm_kernels/gfx950/fmha_v3_fwd/fmha_fwd.csv
```

## 2. Collect and train (one GPU, per engine)

`generate.sbatch` builds the branch, counts what the engine admits, then runs L2 followed
by L1 — in that order, because an immediate run executes whatever the installed catalog
ranker picked, so L1's labels describe the selector that ships. Section 0 has the rocKE
gfx950 submission; AITER takes L1 alone:

```bash
$SUBMIT --constraint=GFX950 --time=06:00:00 \
    --export=ALL,UHD_BRANCH=<branch>,UHD_GRAPHS=/exchange/corpus-950-aiter,UHD_ENGINE=ASM_SDPA_ENGINE,UHD_ROLES=l1,UHD_ARCH=gfx950,UHD_KEEP=/exchange/out-950-aiter \
    generate.sbatch
```

`UHD_ROLES` is `+`-separated: sbatch's own `--export` parser splits its value on commas.
`UHD_METRICS` (e.g. `tflops+time`, same separator) trains one UHD per ranking metric per
role from the same run; unset, each role trains its default `tflops` model.
`UHD_L2_METRICS` / `UHD_L1_METRICS` override it for one role. `UHD_PROVIDERS` (same
separator, default `hip-kernel-provider`) picks the providers built; `miopen-provider` adds
`MIOPEN_ENGINE` and `MIOPEN_ENGINE_DETERMINISTIC` (convolutions):

```bash
$SUBMIT --constraint=GFX950 --time=06:00:00 \
    --export=ALL,UHD_BRANCH=<branch>,UHD_PROVIDERS=miopen-provider,UHD_GRAPHS=/exchange/conv-corpus-950,UHD_ENGINE=MIOPEN_ENGINE,UHD_ROLES=l1,UHD_METRICS=tflops+time,UHD_ARCH=gfx950,UHD_KEEP=/exchange/out-950-miopen \
    generate.sbatch
```

An engine with no UED (AITER, MIOpen) reads only the UHD ids its provider declares, one
per metric (`declared_ids.sh` mirrors the provider tables); `generate.sbatch` and
`bakeoff.sbatch` pass them as `--uhd-id METRIC=UUID` (`UHD_IDS` overrides for generate)
and `generate` refuses an id that contradicts what the engine reports. MIOpen declares its
ids under `default`, so its models are promoted there (`UHD_L1_ARCH` overrides). A corpus
root is read through its `manifest.json`; a graph whose collection fails is skipped and
recorded (`failed_graphs` in `generation_manifest.json`) unless more than 5% fail.
`UHD_SHARDS` / `UHD_SHARD` split a large corpus across jobs by interleaved slices.

Each run keeps `l1/corpus.csv` (one measured row per graph), `l1/model/` (the artifact and
`eval_report.json`) and `declined.txt`. With several metrics these become
`l1/corpus_<metric>.csv` and `l1/model_<metric>/` (L1 measures each metric separately,
because the engine's kernel choice follows the metric), and `l2/model_<metric>/` beside
one shared `l2/corpus.csv`. An engine with no catalog to rank takes `l1` alone.

## 3. Compare engines on what was measured

```bash
python3 compare_engines.py --manifest /tmp/corpus-950/manifest.json \
    --engine rocKE=/exchange/out-950-dense/l1/corpus.csv \
    --engine AITER=/exchange/out-950-aiter/l1/corpus.csv
```

Reports coverage, per-regime winners and the margin distribution over the graphs both
engines serve. Coverage and contest size matter as much as the winner: these two engines
overlap on a minority of any corpus, so an aggregate "who is faster" hides that they are
mostly solving different problems.

## 4. Check the models the way the runtime will

`bakeoff.sbatch` installs several trained models into one runtime and asks every engine to
predict every graph — the cross-engine question L1 exists for — once per metric in
`UHD_METRICS` (default `tflops`). `UHD_PROVIDERS` works as for `generate.sbatch`.

```bash
$SUBMIT --constraint=GFX950 \
    --export=ALL,UHD_BRANCH=<branch>,UHD_CORPUS=/exchange/corpus-950,UHD_ARCH=gfx950,"UHD_MODELS=rocKE=/exchange/out-950-dense/l1/model:hipkernel:Gfx950AttentionDense;AITER=/exchange/out-950-aiter/l1/model:ASM_SDPA_ENGINE",UHD_KEEP=/exchange/bakeoff-950 \
    bakeoff.sbatch
```

A MIOpen convolution bake-off for both metrics names each metric's model directory:
`UHD_PROVIDERS=miopen-provider+hip-kernel-provider`, `UHD_METRICS=tflops+time`,
`UHD_MODELS=miopen-tflops=<out>/l1/model_tflops:MIOPEN_ENGINE;miopen-time=<out>/l1/model_time:MIOPEN_ENGINE;...`.

Then score it — predicted winner against measured winner, per metric and per regime, with
the regret of the predicted pick when they disagree:

```bash
python3 score_predictions.py --manifest /tmp/corpus-950/manifest.json \
    --predictions /exchange/bakeoff-950/predictions.json \
    --measured /exchange/out-950-dense/l1/corpus.csv \
    --measured /exchange/out-950-aiter/l1/corpus.csv
```

Engines are named as the runtime names them, so any engine scores. Each metric is scored
in its own direction: the winner of `tflops` is the highest figure, of `time` the lowest,
and a measured row counts only toward the metric its `binding` was collected under. The
agreement ratio is the number to judge L1 by — not its absolute error.

**Every model in one bake-off must come from builds that report the same selector
revision.** The loader refuses a model whose recorded `trained_against.selector_revision`
is not the one the provider reports, because L1 decides which engine runs.

For `ASM_SDPA_ENGINE` that revision is **derived**, not written by hand:
`hip-kernel-provider/asm-sdpa-fwd/<16 hex>`, computed at configure time by
`asm_sdpa_engine/AsmSdpaSelectorRevision.cmake` over every input that decides forward
selection: codegen, the forward CSVs and `.co` kernels of every arch directory, and the
forward dispatch sources (plan builder, plan, shared mask classification, argument
layout). Text inputs are hashed with line endings normalized to LF, so a Windows and a
Linux checkout of one commit report the same revision; each input is a configure
dependency, so editing one reconfigures. Read the current value off the descriptors or
the build rather than copying one out of this document — it changes exactly when the
forward surface changes, which is the whole point.

It is derived because both hand-written alternatives fail, in opposite directions:

- **too wide** — naming the build hash expires every model on every commit, and naming
  the provider *release* is the same mistake one step smaller: `0.2.0 -> 0.2.1` for a fix
  that cannot touch this engine expires both shipped models, and the only symptom is
  UNAVAILABLE on every engine-selection query;
- **too narrow** — a fixed version expires *nothing* when a vendored kernel is swapped
  under it. That is the silent direction and the worse one: L1 is the score compared
  across engines, so a stale estimate changes which engine runs rather than merely
  misreporting a number.

A backward-only kernel drop leaves the revision alone by construction: the digest does not
read backward kernels, and a backward kernel cannot move a forward throughput number.
`UHD_COMMIT=<sha>` pins `bakeoff.sbatch` to an older build when you do need to reproduce
against one.

An `ASM_SDPA_ENGINE` retrain keeps the declared id (`L1_MODEL_IDS`), and the build already
stages the shipped model under that id. `promote` owns that model by its id, metric and
training architectures, not by where it lies, so it replaces the shipped document and
artifact in place (under their existing file names) rather than installing a second copy.
To ship the retrain, promote the kept model into the source tree the same way:

```bash
cd projects/hipdnn/tools
python -m uhd_gen promote --model-dir /exchange/out-950-aiter/l1/model \
    --descriptor-tree ../../../dnn-providers/hip-kernel-provider/src/engines/asm_sdpa_engine/descriptors \
    --engine ASM_SDPA_ENGINE --role predict_engine --arch gfx950 \
    --feature-evaluator <build>/bin/hipdnn_uhd_features --dry-run   # then without --dry-run
```

The id comes from the collection's recorded `binding.uhd_id` (or `--uhd-id
tflops=<L1_MODEL_IDS entry>`). It is refused, not overwritten, if the installed model
under that id estimates another metric or was trained for another arch.

## 5. Which engine answers at all

`engine_matrix.sbatch` is the cheap first check: it lists the engines a build registers and
asks each of them for a prediction on every graph of `UHD_CORPUS` (built for
`UHD_GPU_TARGETS`, default gfx942), printing per-engine coverage and, for any engine that
answers nothing, its own words at `HIPDNN_LOG_LEVEL=info`.

Use it before spending hours on a sweep. It catches in minutes what otherwise costs a full
collection: a target list that skipped the `ALL`-only descriptor staging, a corpus pinning
a field an engine refuses outright, a corpus spelling a causal alignment an engine's
kernels do not serve, and an AITER `.co` catalog resolved from the install path rather
than the build tree (`HIPDNN_AITER_ASM_DIR`).

## When the node builds the wrong commit

Compute sites do not all resolve `github.com` to the same mirror: a node can clone a branch
tip hours behind the one `git ls-remote` reports from the login node. Carry the commits
yourself rather than trusting the clone:

```bash
git bundle create delta.bundle <a commit the mirror has>..HEAD --branches=<branch>
scp delta.bundle <cluster>:~/
$SUBMIT --export=ALL,UHD_BRANCH=<branch>,UHD_BUNDLE=/exchange/delta.bundle,... <script>.sbatch
```

Every script here fetches the bundle over the clone and checks out its tip, so the job
builds the tree you meant. The clone is 200 commits deep, so the bundle's base must be
within that depth of the branch tip; `UHD_PREREQ_REFS` (generate, corpus5000) fetches
further branches a merged bundle depends on.

## Where a shipped artifact came from

Each shipped model records its own provenance: the `name` of its `heuristic.uhd.json`
(for AITER, `asm_sdpa_engine_tflops.uhd.json`) carries the engine, arch and graph count it
was trained on, and `trained_against` carries the selector revision -- for a UED engine
also the descriptors and feature-semantics revision it was measured against. A retrain is
checked against those fields, not against a figure copied into a document.

## Known gaps

- **Causal cross attention** (`seqlen_q > seqlen_kv` with a causal mask) is refused by
  `hipdnn_corpus_gen`: the declared FLOP count goes non-positive, so no label can be derived.
