# Reproducing the shipped SDPA heuristics

Everything needed to regenerate, re-measure and independently check the L1/L2 models this
branch installs for `hipkernel:Gfx942AttentionDense`, `hipkernel:Gfx950AttentionDense` and
`ASM_SDPA_ENGINE` (AITER). No measured performance numbers are recorded here — the
procedure produces them on your own hardware, which is the only place they mean anything.

## What is installed, and where it came from

| engine | arch | roles | binding |
|---|---|---|---|
| `hipkernel:Gfx942AttentionDense` | gfx942 | `sort_kernel_catalog`, `predict_engine_tflops` | UED role map, `rocKE/gfx942_attention_dense/heuristics/` |
| `hipkernel:Gfx950AttentionDense` | gfx950 | `sort_kernel_catalog`, `predict_engine_tflops` | UED role map, `rocKE/gfx950_attention_dense/heuristics/` |
| `ASM_SDPA_ENGINE` | gfx942, gfx950 | `predict_engine_tflops` | UUID declared in `AsmSdpaEngine.hpp`; document staged from `src/engines/asm_sdpa_engine/descriptors/` |

AITER owns no descriptor set, so its model is bound by the UUID the provider declares
(RFC 0019 §4.1, Open Question 7) rather than by a role map. The document's `id` IS the
binding: change it and the engine silently reports no model.

## 1. Build the corpus (offline, no GPU)

Deterministic from the seed and the in-tree inputs — the same command reproduces the same
graphs byte for byte, and `benchmark` ids are content-derived so results join across runs.

```bash
cd projects/hipdnn/tools
# gfx942: the whole declared space, 5000 graphs
python3 -m corpus_build --out /tmp/corpus-5000 --count 5000 --seed 0
# gfx950: drawn from that arch's own packed geometries, both head dims its engines serve
python3 -m corpus_build --out /tmp/corpus-950 --count 1000 --seed 0 \
    --kdp-root ../../../dnn-providers/hip-kernel-provider/descriptor-packaging/examples/descriptors/rocKE/gfx950_attention_dense \
    --min-candidates 2 --head-dim 64 --head-dim 128
```

`--dtype` and `--head-dim` exist because an engine that cannot serve a facet contributes
only declines: AITER's gfx942 forward table is four kernels (bf16, hd128/hd192→128), its
gfx950 table is two and carries no causal kernel at all. Check what any build actually
ships before widening a corpus:

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
sbatch --constraint=GFX950 --gres=gpu:1 --time=08:00:00 \
    --export=ALL,UHD_GRAPHS=/exchange/corpus-950,UHD_ENGINE=hipkernel:Gfx950AttentionDense,UHD_ROLES=l2+l1,UHD_ARCH=gfx950,UHD_KEEP=/exchange/out-950-dense \
    generate.sbatch
sbatch --constraint=GFX950 --gres=gpu:1 --time=06:00:00 \
    --export=ALL,UHD_GRAPHS=/exchange/corpus-950,UHD_ENGINE=ASM_SDPA_ENGINE,UHD_ROLES=l1,UHD_ARCH=gfx950,UHD_KEEP=/exchange/out-950-aiter \
    generate.sbatch
```

`UHD_ROLES` is `+`-separated: sbatch's own `--export` parser splits its value on commas.

Each run keeps `l1/corpus.csv` (one measured row per graph), `l1/model/` (the artifact and
`eval_report.json`) and `declined.txt`. An engine with no catalog to rank takes `l1` alone.

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
sbatch --constraint=GFX950 --gres=gpu:1 \
    --export=ALL,UHD_CORPUS=/exchange/corpus-950,UHD_ARCH=gfx950,"UHD_MODELS=rocKE=/exchange/out-950-dense/l1/model:hipkernel:Gfx950AttentionDense;AITER=/exchange/out-950-aiter/l1/model:ASM_SDPA_ENGINE",UHD_KEEP=/exchange/bakeoff-950 \
    bakeoff.sbatch
```

Then join its `predictions.json` against the measured `corpus.csv` files: for every graph
both engines serve, does the model that predicts the higher figure belong to the engine
that measured faster? That ratio is the number to judge L1 by — not its absolute error.

**Every model in one bake-off must come from builds that report the same selector
revision.** The loader refuses a model whose recorded `trained_against.selector_revision`
is not the one the provider reports, because L1 decides which engine runs. That revision
names the provider *release* and the engine's dispatch generation
(`hip-kernel-provider/<version>/asm-sdpa-untuned-v1`), not the build hash — a hash would
expire every model on every commit. `UHD_COMMIT=<sha>` pins `bakeoff.sbatch` to an older
build when you do need to reproduce against one.

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

## Known gaps

- **flyDSL (`hipkernel:FlydslAttention`)** is not part of this: its 21 HSACOs are not
  committed and its builder imports `kernels.attention.flash_attn_gfx950`, which is not in
  this repository — the rocKE wheels are built from `rocke/library`, whose `kernels/gfx950/`
  carries the productized `attention_dense` and no `experiments/` tree. It needs either the
  prebuilt HSACOs or that kernel snapshot.
- **`hipkernel:Gfx950AttentionTiled`** requires page tables and correctly declines every
  dense SDPA graph, so a dense corpus cannot exercise it.
- **Causal cross attention** (`seqlen_q > seqlen_kv` with a causal mask) is refused by
  `corpus_build`: the declared FLOP count goes non-positive, so no label can be derived.
