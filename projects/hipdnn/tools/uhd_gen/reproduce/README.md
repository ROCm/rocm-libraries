# Reproducing the shipped SDPA heuristics

Everything needed to regenerate, re-measure and independently check the L1/L2 models for
`hipkernel:Gfx942AttentionDense`, `hipkernel:Gfx950AttentionDense` and `ASM_SDPA_ENGINE`
(AITER). No measured performance numbers are recorded here — the procedure produces them
on your own hardware, which is the only place they mean anything.

**Two branches, on purpose.** This procedure spans both, and the split is deliberate
rather than an omission:

| on `users/jscampb/uhd-heuristics-e2e` (this branch) | on `users/jascampb/uhd-integration-test-branch` |
|---|---|
| the productized path: `rocKE/gfx942_attention_dense`, the `ASM_SDPA_ENGINE` descriptors, `hipdnn_corpus_gen`, and every script in this directory except the two named opposite | the experiment path: the `rocKE/gfx950_attention_dense` descriptor pack (`gfx950_attention_dense.{ued,kmd,umd,kdp,udd,uhd}.json` and its `heuristics/` tree) and the flyDSL POC — `flydsl_catalog.sbatch`, `flydsl_enable.sbatch`, the Flydsl* packs, `flydsl_poc_scratch/` |

A step below that names a gfx950 rocKE descriptor or a `flydsl_*.sbatch` needs the
integration branch checked out; everything else runs here. The rocKE *kernels* for gfx950
are on both branches (`rocke/library/kernels/gfx950/`); it is the packaged descriptor set
that is not, so on this branch that engine has kernels and no installed heuristic.

## What is installed, and where it came from

| engine | arch | roles | binding |
|---|---|---|---|
| `hipkernel:Gfx942AttentionDense` | gfx942 | `sort_kernel_catalog`, `predict_engine` | UED role map, `rocKE/gfx942_attention_dense/heuristics/` |
| `hipkernel:Gfx950AttentionDense` | gfx950 | `sort_kernel_catalog`, `predict_engine` | UED role map, `rocKE/gfx950_attention_dense/heuristics/` — **integration branch only** |
| `ASM_SDPA_ENGINE` | gfx942, gfx950 | `predict_engine` | UUID declared in `AsmSdpaEngine.hpp`; document staged from `src/engines/asm_sdpa_engine/descriptors/` |

AITER owns no descriptor set, so its model is bound by the UUID the provider declares
(RFC 0019 §4.1, Open Question 7) rather than by a role map. The document's `id` IS the
binding: change it and the engine silently reports no model.

## 0. What every job needs

The sbatch files here take their inputs through `/exchange`, which is the container's view
of your home directory. Every submission therefore carries the same container flags; leaving
them off runs the script on the bare node, where `/exchange` does not exist and apt refuses
to install (measured: run 67932435).

```bash
SUBMIT="sbatch --cpus-per-task=16 --mem=96G --gres=gpu:1 \
  --container-image=docker://rocm/dev-ubuntu-24.04:7.14.0-full \
  --container-writable --container-remap-root \
  --container-mounts=$HOME:/exchange"
```

Stage corpora and artifacts in `$HOME` on the login node; they appear under `/exchange`
inside the job. `UHD_BUNDLE` (see the last section) is how a job builds the tree you pushed
rather than whatever its site mirror happens to serve.

## 1. Build the corpus (per engine, on a GPU)

A corpus is generated **for an engine**: every candidate problem is offered to it, so what
comes out is what that engine serves. `--engine-name` is required and needs the provider
built and staged, so this runs on a GPU node -- through `corpus5000.sbatch`, or by hand in
a job as below. Deterministic from the seed and the in-tree inputs, and `benchmark` ids are
content-derived so results join across runs.

```bash
cd projects/hipdnn/tools
GEN=<build>/bin/hipdnn_corpus_gen
PLUGINS=<build>/lib/hipdnn_plugins/engines
PACKS=../../../dnn-providers/hip-kernel-provider/descriptor-packaging/examples/descriptors/rocKE

# rocKE, gfx950: its pack proposes, the engine admits. This is also the comparison corpus
# of step 3. --kdp-root names the integration branch's pack; from this branch it does not exist.
$GEN --operations corpus_gen/operations --operation sdpa_fwd --plugin-dir $PLUGINS \
    --engine-name hipkernel:Gfx950AttentionDense --kdp-root $PACKS/gfx950_attention_dense \
    --output /tmp/corpus-950 --count 1000 --seed 0

# AITER, gfx950: what AITER serves, with the comparison graphs held out by construction.
$GEN --operations corpus_gen/operations --operation sdpa_fwd --plugin-dir $PLUGINS \
    --engine-name ASM_SDPA_ENGINE --exclude-corpus /tmp/corpus-950/manifest.json \
    --output /tmp/corpus-950-aiter --count 2500 --seed 11
```

Graphs land in `<output>/graphs/`, beside `manifest.json`. `--count` is met unless the
engine physically serves fewer: rocKE's kernels match exact geometries, so its corpus is
capped at the pack's shapes, and the tool says so and exits 0. Any other shortfall exits 3.

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
ranker picked, so L1's labels describe the selector that ships.

```bash
$SUBMIT --constraint=GFX950 --time=08:00:00 \
    --export=ALL,UHD_GRAPHS=/exchange/corpus-950,UHD_ENGINE=hipkernel:Gfx950AttentionDense,UHD_ROLES=l2+l1,UHD_ARCH=gfx950,UHD_KEEP=/exchange/out-950-dense \
    generate.sbatch
$SUBMIT --constraint=GFX950 --time=06:00:00 \
    --export=ALL,UHD_GRAPHS=/exchange/corpus-950-aiter,UHD_ENGINE=ASM_SDPA_ENGINE,UHD_ROLES=l1,UHD_ARCH=gfx950,UHD_KEEP=/exchange/out-950-aiter \
    generate.sbatch
```

The first submission needs the integration branch: `hipkernel:Gfx950AttentionDense` has no
descriptor pack here, so on this branch it registers nothing to collect against. The AITER
submission runs on either.

`UHD_ROLES` is `+`-separated: sbatch's own `--export` parser splits its value on commas.
`UHD_METRICS` (e.g. `tflops+time`, same separator) trains one UHD per ranking metric per
role from the same run; unset, each role trains its default `tflops` model as before.

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
predict every graph — the cross-engine question L1 exists for.

```bash
$SUBMIT --constraint=GFX950 \
    --export=ALL,UHD_CORPUS=/exchange/corpus-950,UHD_ARCH=gfx950,"UHD_MODELS=rocKE=/exchange/out-950-dense/l1/model:hipkernel:Gfx950AttentionDense;AITER=/exchange/out-950-aiter/l1/model:ASM_SDPA_ENGINE",UHD_KEEP=/exchange/bakeoff-950 \
    bakeoff.sbatch
```

Then score it — predicted winner against measured winner, per regime, with the throughput
given up when they disagree:

```bash
python3 score_predictions.py --manifest /tmp/corpus-950/manifest.json \
    --predictions /exchange/bakeoff-950/predictions.json \
    --measured rocKE=/exchange/out-950-dense/l1/corpus.csv \
    --measured AITER=/exchange/out-950-aiter/l1/corpus.csv
```

Then join its `predictions.json` against the measured `corpus.csv` files: for every graph
both engines serve, does the model that predicts the higher figure belong to the engine
that measured faster? That ratio is the number to judge L1 by — not its absolute error.

**Every model in one bake-off must come from builds that report the same selector
revision.** The loader refuses a model whose recorded `trained_against.selector_revision`
is not the one the provider reports, because L1 decides which engine runs.

For `ASM_SDPA_ENGINE` that revision is **derived**, not written by hand:
`hip-kernel-provider/asm-sdpa-fwd/<16 hex>`, where the digest is computed at configure
time over the vendored forward kernels, the CSVs codegen reads to describe them, and the
forward dispatch sources. Read the current value off the descriptors or the build rather
than copying one out of this document — it changes exactly when the forward surface
changes, which is the whole point.

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

## 4b. Including flyDSL — integration branch only

`flydsl_catalog.sbatch` and `flydsl_enable.sbatch` are on
`users/jascampb/uhd-integration-test-branch`, with the rest of the flyDSL POC; they are
not on this branch and this section does not run here. It is kept because the procedure is
the same one, and because the environment variables below are what a composed rocKE+flyDSL
tree needs on either branch.

flyDSL's kernels are not committed anywhere; `flydsl_catalog.sbatch` clones
`https://github.com/ROCm/FlyDSL.git`, builds all 240 variants with the `flydsl==0.3.2` wheel,
checks every one still carries the 608-byte kernarg and the pack's entry point, and stages
them. `flydsl_enable.sbatch` then proves the engine registers and executes one graph.

```bash
$SUBMIT --constraint=GFX950 --time=04:00:00 \
    --export=ALL,UHD_KEEP=/exchange/flydsl-catalog flydsl_catalog.sbatch
```

Collection and bake-off then take `UHD_COMPOSE_FLYDSL=1`,
`UHD_HSACO_DIR=/exchange/flydsl-catalog` and `UHD_FLYDSL_CATALOG=/exchange/flydsl-catalog`:
the pack lives outside `arch_content`, and `HIPDNN_DESCRIPTOR_DIR` replaces the search roots
rather than adding to them, so rocKE and flyDSL are only both visible from one composed tree.

## 5. Which engine answers at all

`engine_matrix.sbatch` is the cheap first check: it lists the engines a build registers and
asks each of them for a prediction on every graph, printing per-engine coverage and, for
any engine that answers nothing, its own words at `HIPDNN_LOG_LEVEL=info`.

Use it before spending hours on a sweep. Four separate defects were found by it alone: a
target list that skipped the `ALL`-only descriptor staging (2 engines registered instead of
10), a corpus pinning `mma_core_mode` that AITER refuses outright, a corpus spelling
causality top-left when AITER's gfx942 kernels are bottom-right, and an AITER `.co` catalog
resolved from the install path rather than the build tree
(`HIPDNN_AITER_ASM_DIR`).

## When the node builds the wrong commit

Compute sites do not all resolve `github.com` to the same mirror: a node can clone a branch
tip hours behind the one `git ls-remote` reports from the login node. Carry the commits
yourself rather than trusting the clone:

```bash
git bundle create delta.bundle <a commit the mirror has>..HEAD --branches=<branch>
scp delta.bundle <cluster>:~/
sbatch --export=ALL,UHD_BUNDLE=/exchange/delta.bundle,... <script>.sbatch
```

Every script here fetches the bundle over the clone and checks out its tip, so the job
builds the tree you meant.

## What the shipped artifacts were actually built from

Exact provenance for everything the two branches produced, so a check can reproduce the
same inputs rather than similar ones. The `branch` column is the tree you must have
checked out for that row to run: `e2e` is this branch, `integration` is
`users/jascampb/uhd-integration-test-branch`, which carries the gfx950 rocKE pack and the
flyDSL POC.

| artifact | branch | built by |
|---|---|---|
| comparison corpus (1000 graphs, gfx950) | integration | `--count 1000 --seed 0 --kdp-root <rocKE/gfx950_attention_dense> --min-candidates 2 --head-dim 64 --head-dim 128` |
| gfx942 corpus (5000 graphs) | e2e | `--count 5000 --seed 0` |
| flyDSL 240-kernel catalog | integration | `flydsl_catalog.sbatch` (waves 1/2/4 x stagger on/off x lazy on/off, setprio on) |
| rocKE gfx950 L1+L2, flyDSL L1+L2 | integration | `generate.sbatch`, `UHD_ROLES=l2+l1`, on the comparison corpus |
| AITER gfx950 L1 | e2e | `--count 2500 --seed 11 --dtype bf16 --head-dim 128 --causal 0 --exclude-corpus <comparison manifest>` then `generate.sbatch UHD_ROLES=l1` |
| the 94.2% number | integration | `bakeoff.sbatch` over the comparison corpus, then `score_predictions.py` |

The corpus rows are recorded as they were run, under the retired Python `corpus_build`. To
rebuild them with `hipdnn_corpus_gen`: `--out` is `--output`, and the engine the row was
built for is named with `--engine-name`, which replaces the facet flags -- the engine now
decides what it serves, where `--dtype/--head-dim/--causal` guessed it. `--min-candidates`
is gone: pack density is an upper bound on what the matcher offers at runtime, never a count
of it. A rebuilt corpus is not byte-identical to the recorded one: the sampler now holds
declared shares per categorical combination rather than in expectation per draw, and causal
graphs are expressed as bounds (`right_bound = 0`) rather than the deprecated `causal_mask`
flag, which providers read as top-left whatever the alignment said.

flyDSL additionally needs a FlyDSL checkout; `flydsl_catalog.sbatch` clones
`https://github.com/ROCm/FlyDSL.git` itself, and the standalone builders take `FLYDSL_REPO`.

## Known gaps

- **flyDSL (`hipkernel:FlydslAttention`)** is not part of this: its 21 HSACOs are not
  committed and its builder imports `kernels.attention.flash_attn_gfx950`, which is not in
  this repository — the rocKE wheels are built from `rocke/library`, whose `kernels/gfx950/`
  carries the productized `attention_dense` and no `experiments/` tree. It needs either the
  prebuilt HSACOs or that kernel snapshot.
- **`hipkernel:Gfx950AttentionTiled`** requires page tables and correctly declines every
  dense SDPA graph, so a dense corpus cannot exercise it.
- **Causal cross attention** (`seqlen_q > seqlen_kv` with a causal mask) is refused by
  `hipdnn_corpus_gen`: the declared FLOP count goes non-positive, so no label can be derived.
