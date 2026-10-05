# Depthwise forward group-merge degree model

The corpus, the fitting driver and the scoring harness behind
`dispatch.grouped_convolution.fwd_group_merge_for_geometry()` — the analytic
selector that picks the merged-group degree `Gm` for depthwise forward
convolution on gfx950.

Narrative, methodology and measured results are in
[`../fwd_merged_groups_case_study.md`](../fwd_merged_groups_case_study.md).
**This README is the runbook**: how to regenerate everything from scratch, and
what each file is for.

Start with [`model.py`](model.py). Its module docstring is the explanation of
the heuristic — what merged groups does, which four mechanisms move with the
degree, why occupancy is *not* the one that limits it, and what each of the
eight fitted constants is weighing. `dispatch` ships the same arithmetic inlined
(it imports nothing by design); `score_shipped.py` proves the two copies agree.

## Why a model and not a table

`Gm` folds consecutive conv groups into one GEMM so that depthwise convolution
— one channel per group, so `N_gemm = 1` and `K_gemm = Y*X` — stops issuing a
degenerate GEMM. Larger `Gm` wins back vector width, cache-line utilisation and
K-padding, and halves the CTA count on every doubling; but it grows each CTA's
working set linearly in `Y*X*Gm`, and past some point that dominates. The
turnover is shape-dependent and not monotone in anything simple.

The benchmark can sweep the degree. Dispatch cannot: it has to answer in
microseconds, from request geometry alone, for shapes nobody measured. Hence a
closed form — seven candidate degrees, one `argmin`, no table lookup, no
measurement at runtime.

## Files

| file | role |
|---|---|
| `model.py` | **the model.** Readable reference + `pick()`; `python3 model.py` cross-checks it against `fitfast`'s vectorised path |
| `corpus_fwd.py` | loads sweep CSVs into normalised per-shape degree curves |
| `gen_shapes.py` | generates the shape corpus (model stages + stratified grid) as MIOpenDriver lines |
| `shapes_fwd.txt` | the generated corpus actually used, checked in so results are reproducible |
| `measure_degrees.py` | runs the sweep with the tile pinned and the degree axis fully open |
| `fitfast.py` | **the fitter.** Vectorised random search + coordinate refinement, nested-model and cross-validation support |
| `score_shipped.py` | scores the real dispatch function, and checks it has not drifted from `model.py` |
| `tiles.py` | the eight tiles the model is scored at, with provenance |
| `run_tiles.py` | sweeps all eight tiles, one CSV each |
| `score_tiles.py` | does the degree model transfer across tiles? is 64x64x64 the right tile? |
| `fit_tiles.py` | pooled and per-tile refit, to separate the three places `tile_m` enters the cost |

Measured CSVs (`degrees.csv`, `degrees_tuning.csv`, `tiles_out/*.csv`) are **not
checked in** — they are several megabytes of raw per-kernel timings. Regenerate
them with the commands below. Every script degrades to a clean "no measured
corpus present" rather than failing when they are absent.

## Regenerating from scratch

All commands run from this directory, on a gfx950 host.

### 1. Shapes

```bash
python3 gen_shapes.py --count 400 --seed 20251001 --model-cap 200 -o shapes_fwd.txt
```

Half the corpus is real depthwise stages lifted from ~36 published architectures
(mobilenet v1–v4, efficientnet, convnext, replknet, yolo, …); half is a
stratified random grid, cell-capped so no single `(G, Y*X, N, stride)` cell can
dominate the fit. Shapes are rejected above a tensor-bytes and a MAC budget so
the sweep stays tractable.

### 2. Measure

```bash
python3 measure_degrees.py shapes_fwd.txt -o degrees.csv --jobs 32
```

Two things this driver does that matter:

* **It pins the tile** to the one dispatch ships. The sweep benchmark has no
  tile-pinning switch, and a full tile × warp × pipeline × epilogue sweep over
  400 shapes is untenable; pinning collapses the per-shape work to exactly the
  degree axis, which is the one thing under test.
* **It forces `--group-merge-window -1 --unmerged-frac 1.0`.** The benchmark
  normally prunes the degree axis to a window *centred on this very policy*.
  Measuring the policy inside that window would score it against itself. Any
  measurement that evaluates the degree policy must disable the window.

It also forces `--verify` and `--csv-top 999`: an unverified degree must never
be allowed to set an oracle, and the default CSV cap silently drops the slowest
rows — which on depthwise means `Gm=1`, the baseline every curve is normalised
against.

### 3. Fit

```bash
python3 fitfast.py --cv              # select the form by CV on TRAIN, then fit
python3 fitfast.py --model wave      # or fit any nested model, for ablation
```

The objective is the geometric mean over shapes of the *realised fraction* — the
share of a shape's own measured best that the single modelled pick keeps. It is
piecewise constant in the parameters (the model's output is an `argmin`), so
there is no gradient and the search is deliberately brute force: ~10^6 vectorised
draws, multiplicative coordinate refinement, several restarts.

`--model` selects a nested model from `fitfast.MODELS`, with every other constant
pinned to its neutral value — the mechanism for asking "is this term earning its
place?" by cross-validation instead of by argument.

To adopt a refit, copy the printed constants into `model.py`'s `Constants` **and**
into the `_FWD_MERGE_*` block in `library/dispatch/grouped_convolution.py`, then
re-run step 4.

### 4. Score what ships

```bash
python3 score_shipped.py --verbose
```

Puts every measured shape through the real dispatch entry point, confirms it
still agrees with `model.py` shape for shape, and reports the realised fraction
against the three-cap rule the model replaced. Exits non-zero on drift.

Note this report is **in-sample** — it scores the whole corpus, including the
rows the constants were fitted on. The held-out comparison is the one quoted in
the case study.

### 5. The eight-tile study (optional)

```bash
python3 run_tiles.py                 # one sweep per tile in tiles.py
python3 score_tiles.py               # transfer, and whether 64x64x64 is right
python3 fit_tiles.py                 # pooled + per-tile refit
```

Six of the eight tiles are the configurations composable_kernel instantiates for
its own merged-groups convolution; the other two are the shipped tile and the
winner of an in-house tile sweep.

This exists because the model was fitted at one tile and `tile_m` enters its cost
in three separate places — the CTA count, the memory term's bytes per tile, and
the brake's footprint — so with `tile_m` held constant only two combinations of
the three were ever identified. Varying the tile is what separates them.
`fit_tiles.py` splits on whole *shapes*, not shape-tile pairs: a shape present in
train at one tile and in test at another would leak its degree curve across the
split, and the degree curve is exactly what is under test.

See Finding 5 of the case study for what this measured.

## Scope and when to refit

Fitted on **gfx950**, **16-bit** (fp16/bf16), **forward**, at the **single tile
dispatch ships**. It is a degree model, not a tile model. Refit if the arch, the
element size, or dispatch's tile selection changes. Backward (dgrad/wgrad)
depthwise is a different arrangement and is not covered.
