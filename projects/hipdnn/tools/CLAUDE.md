# Generating SDPA heuristics (L1 and L2) — operator runbook

You are driving `corpus_build` and `uhd_gen` to produce the two models an engine needs, on
real hardware, and to prove they load. Read this whole file before submitting anything: the
expensive failures below all *succeed* for an hour first and then throw the work away.

Run everything from `projects/hipdnn/tools`. That is the working directory both tools
assume, and `corpus_build.REPO` resolves its in-tree inputs from this file's location, not
from where you stood.

## The two models, and why the order is fixed

| role | question it answers | who has one |
|---|---|---|
| `sort_kernel_catalog` (L2) | which of MY candidate kernels is fastest for this graph | an engine that enumerates a catalog |
| `predict_engine_tflops` (L1) | how fast will I run this graph, in TFLOPS | every engine, including ones with no catalog |

**Always L2 first, then L1, in that order and ideally in one job.** An immediate run
executes whatever the installed catalog ranker picked, so L1's labels describe the selector
that ships. Training L1 against an uninstalled L2 measures a selector nobody will run.

L1 is the only score compared *across* engines. That is why it is calibrated TFLOPS, why a
stale one is refused rather than de-rated, and why its errors matter more than L2's.

## Step 0 — find out what the engines can actually serve

Do this before building a corpus. An engine that cannot serve a facet contributes nothing
but declines, and a corpus of declines looks exactly like a broken pipeline.

```bash
# AITER ships a tiny table. gfx942: 4 kernels (bf16, hd128/hd192->128, mask 0 and 2).
# gfx950: 2 kernels, both mask 0 -- NO causal kernel at all.
cat ../../../dnn-providers/hip-kernel-provider/src/engines/asm_sdpa_engine/asm/asm_kernels/gfx950/fmha_v3_fwd/fmha_fwd.csv
# rocKE dense packs declare their geometries; count the facets you care about:
python3 -c "import json,collections,sys; kdp=json.load(open(sys.argv[1]));
md=[k['metadata'] for k in kdp['kernelDescriptors']];
[print(f, collections.Counter(str(m.get(f)) for m in md).most_common(6)) for f in ('dtype','head_size','causal')]" \
  ../../../dnn-providers/hip-kernel-provider/descriptor-packaging/examples/descriptors/rocKE/gfx950_attention_dense/gfx950_attention_dense.kdp.json
```

Then confirm on hardware with `uhd_gen/reproduce/engine_matrix.sbatch`, which lists the
engines a build registers and asks each for a prediction on every graph. It costs ~25
minutes and has caught four separate defects that would each have wasted a full sweep.

## Step 1 — build the corpus (offline, no GPU, deterministic)

```bash
python3 -m corpus_build --out /tmp/corpus --count 1000 --seed 0 \
    [--kdp-root <a pack directory>] [--min-candidates 2] \
    [--dtype bf16] [--head-dim 128]
```

- Same seed and same in-tree inputs reproduce the same graphs byte for byte. `benchmark`
  ids are content-derived, so rows join across runs and machines.
- `--kdp-root` narrows the packed-geometry source to one arch's pack. `--min-candidates`
  defaults to 3; the gfx950 dense pack carries two block_m variants per geometry, so pass 2
  or every packed geometry is dropped as "nothing to rank".
- `--dtype` / `--head-dim` gate all three sources through one predicate. Use them to build
  a corpus the engines you are comparing can both serve; leave them off for a broad corpus
  where each engine trains on its own admissible subset.
- The manifest records the filter, what each source lost, and the regime of every graph.

## Step 2 — collect and train

```bash
sbatch --constraint=GFX950 --gres=gpu:1 --time=08:00:00 \
    --container-image=docker://rocm/dev-ubuntu-24.04:7.14.0-full --container-writable \
    --container-remap-root --container-mounts=$HOME:/exchange \
    --export=ALL,UHD_GRAPHS=/exchange/corpus,UHD_ENGINE=hipkernel:Gfx950AttentionDense,UHD_ROLES=l2+l1,UHD_ARCH=gfx950,UHD_KEEP=/exchange/out-dense \
    uhd_gen/reproduce/generate.sbatch
```

An engine with no catalog (AITER, MIOpen) takes `UHD_ROLES=l1` alone.

What the job prints, in order, and what each number means:

| line | read it as |
|---|---|
| `engine: <name> 0x… -> <id>` | the engine registered; a missing one means descriptors were not staged |
| `admitted N of M` | **this** sets the wall clock, not the corpus size |
| `L2 generate rc=0 in Ns` | collection + training + promotion all succeeded |
| `L1 generate rc=0 in Ns` | same, against the L2 that was just installed |

Artifacts land in `UHD_KEEP`: `l1/corpus.csv` (one measured row per graph),
`l1/model/heuristic.uhd.json` + `model.bin`, `l1/model/eval_report.json`, `declined.txt`.

## Step 3 — verify the model the way the runtime will

```bash
sbatch … --export=ALL,UHD_CORPUS=/exchange/corpus,UHD_ARCH=gfx950,\
"UHD_MODELS=rocKE=/exchange/out-dense/l1/model:hipkernel:Gfx950AttentionDense;AITER=/exchange/out-aiter/l1/model:ASM_SDPA_ENGINE",\
UHD_KEEP=/exchange/bakeoff uhd_gen/reproduce/bakeoff.sbatch
```

Installs every model into one runtime and asks each engine to predict every graph. Join
`predictions.json` against the measured `corpus.csv` files: for each graph both engines
serve, does the higher prediction belong to the engine that measured faster? That agreement
rate is what L1 is for. Absolute error is secondary — a model biased low everywhere still
picks correctly.

`compare_engines.py` does the measured half (coverage, per-regime winners, margins).

## Failure modes — exact strings, causes, fixes

Every one of these was hit on a real run and cost between 20 minutes and two hours.

| symptom | cause | fix |
|---|---|---|
| 2 engines register instead of 10 | descriptors are staged by `add_custom_target(... ALL)` rules; a `--target <list>` build skips them | build everything: `cmake --build "$BUILD" -j` |
| every engine declines every graph, same message | the graph pins something an engine refuses, e.g. `mma_core_mode` | run one query with `HIPDNN_LOG_LEVEL=info` and read the builder's own line (`[SdpaFwdPlanBuilder::isApplicable] …`) |
| one engine serves 0, others fine | facet mismatch (dtype, head dim, causal anchor) | compare the corpus facets against that engine's kernel table (Step 0) |
| `failed to load kernel module from /opt/rocm/...co` | AITER resolves its catalog from the install path when unset | `export HIPDNN_AITER_ASM_DIR=<build>/…/asm_kernels` (the dir holding the arch subdirs) |
| `full-graph graph.flops must be a positive finite number` | causal cross attention: the declared FLOP count goes non-positive | rebuild the corpus with current `corpus_build`, which refuses those shapes |
| `L1 generate rc=1` seconds after "LAYER 1" begins | `UHD_ROLES=l2,l1` — sbatch's `--export` splits on commas | use `UHD_ROLES=l2+l1` |
| `expected one UED for --engine 'X', found 0` | promoting an opaque engine's model as if it had a role map | current `uhd_gen` installs it by declared UUID; check the model's `id` equals the UUID in the provider header |
| engine answers but `scored 0` predictions | the model's `trained_against.selector_revision` is not what the provider reports | all models in one bake-off must come from builds reporting the same revision |
| job builds an unexpected commit | compute sites resolve `github.com` to mirrors that lag the login node | `git bundle create delta.bundle <base>..HEAD`, stage it, pass `UHD_BUNDLE=/exchange/delta.bundle` |

## Invariants — do not break these

1. **An opaque engine's model `id` IS its binding.** AITER and MIOpen own no descriptor
   set; they look up a UUID declared in provider code (`AsmSdpaEngine.hpp`). A model with a
   minted UUID installs cleanly and is never read. Set the declared one.
2. **`selector_revision` names a provider release, not a build.** It is compared for
   equality and a mismatch is refused. If you find yourself regenerating models after an
   unrelated commit, the revision string has regained a git hash — fix the string, not the
   models.
3. **Never publish a model that scored nothing.** Non-positive predicted TFLOPS are
   reported as declines (`metrics.unscored_rows`) because the runtime treats them as
   INVALID and falls back to static ordering; a model where *every* row declines must fail.
4. **No measured performance numbers in the repository.** Models and methodology, yes;
   CSVs, eval reports and TFLOPS figures in committed docs, no.
5. **A corpus must not decide the contest.** Anything pinned in the graph document that an
   engine refuses (`mma_core_mode`, a single causal anchor) silently removes that engine
   before any measurement happens.

## When you are done

Promote into the source tree and commit the artifacts only:

```bash
python3 -m uhd_gen promote --model-dir <out>/l2/model \
    --descriptor-tree ../../../dnn-providers/hip-kernel-provider/descriptor-packaging/examples/descriptors \
    --engine hipkernel:Gfx950AttentionDense --role sort_kernel_catalog --arch gfx950 --dry-run
```

Drop `--dry-run` once the plan reads correctly. For an engine with no UED, the same command
writes the document under `heuristics/<engine>/<role>/<arch>/`; that path must also be
staged by CMake or the loader never sees it (see `HIPDNN_ASM_SDPA_DESCRIPTOR_FILES`).

Then re-run `engine_matrix.sbatch` against the committed branch: the models are proven when
the engines report `available` with TFLOPS instead of `unavailable`.
