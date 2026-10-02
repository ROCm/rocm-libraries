<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# Depthwise wgrad group-merge degree model

The corpus, the measurement driver and the scoring harness behind
`dispatch.grouped_convolution._wgrad_merge_degree()` — the rule that picks the
merged-group degree `Gm` for depthwise backward-weight convolution on gfx950.

Narrative, methodology and **relative** measured results are in
[`../wgrad_merged_groups_case_study.md`](../wgrad_merged_groups_case_study.md).
**This README is the runbook**: how to regenerate everything from scratch, and
what each file is for.

## Why there is no model here

The forward direction's merge selector is a fitted closed form with eight
constants, with its own corpus and fitter under `examples/gfx950/conv_fwd/`.
Backward-weight ended up with none: the shipped rule is *merge as hard as the
kernel's own admissibility gate allows, at the tile that already ships*, and the
fitting work in this directory is the record of three richer policies being
measured and rejected. `score_policy.py` re-derives that rejection from the raw
sweeps.

That asymmetry is not an oversight, and it is the main thing a reviewer should
take from this directory. The two directions place `Gm` on different GEMM axes
and face different limits:

* **Forward** puts `Gm` on GemmN and GemmK, so the degree trades vector width
  against a working set that grows linearly in `Y*X*Gm` — a genuine interior
  optimum that moves with the shape, which is what a model is for.
* **Wgrad** puts `Gm` on both GemmM and GemmN, and the kernel caps it at
  `Y*X*cpg*Gm <= tile_n`. With the tile fixed that bound is tight enough that
  the admissible set is usually two or three degrees wide, and the top of it is
  very nearly always the right answer. There is no interior optimum left to
  model.

So the useful question for wgrad was never "which degree" but "is it worth
widening the tile to admit a larger degree". Measurement said no. See Finding 3
of the case study.

## Files

| file | role |
|---|---|
| `gen_shapes.py` | generates the shape corpus (model stages + stratified grid) as MIOpenDriver lines |
| `shapes_wgrad.txt` | the generated corpus actually used, checked in so results are reproducible |
| `_smoke.txt` | a handful of those shapes, for a two-minute end-to-end check |
| `measure_degrees.py` | runs the sweep with the tile pinned and the degree axis fully open, one CSV per `tile_n` rung |
| `corpus_wgrad.py` | loads sweep CSVs into the per-shape `(tile_n, Gm)` surface |
| `score_policy.py` | **the decision.** Scores the shipped rule and the three rejected policies side by side |

Measured CSVs (`deg_tn*.csv`) are **not** checked in — they are megabytes of raw
per-kernel timings, and they carry absolute numbers. Regenerate them with the
commands below; `score_policy.py` degrades to a clean "no measured corpus
present" when they are absent.

## Regenerating from scratch

All commands run from this directory, on a gfx950 host.

### 1. Shapes

```bash
python3 gen_shapes.py --count 200 --model-cap 100 -o shapes_wgrad.txt
```

Half the corpus is real depthwise stages lifted from published architectures;
half is a stratified random grid. The generator deliberately keeps a minority
of **unmergeable** shapes — filters too large for even two groups to share a
tile, and group counts no ladder rung divides. Those are the population a
selector is most likely to damage, since they get none of the upside, so a
corpus without them cannot detect the damage. `-F 4` selects the wgrad
direction.

### 2. Measure

```bash
python3 measure_degrees.py shapes_wgrad.txt -o deg_tn64.csv           # shipped tile
python3 measure_degrees.py shapes_wgrad.txt -o deg_tn128.csv --tile 64x128x64/2x2/32
python3 measure_degrees.py shapes_wgrad.txt -o deg_tn256.csv --tile 64x256x64/2x2/32
```

`--tile` is `MxNxK/warpMxwarpN/warpTileMN` and defaults to the tile dispatch
ships, so the first run needs no flag. Only the `N` extent moves across the
three: depthwise wgrad has `wg_M = kpg = 1`, so widening `tile_m` buys nothing
and `tile_m` stays 64 throughout.

Four things this driver does that matter:

* **It pins everything except the degree.** A full tile × warp × pipeline ×
  epilogue sweep over 200 shapes is untenable, and the degree axis is the one
  thing under test. The `tile_n` rung is the one deliberate exception, swept as
  three separate runs so the tile-vs-degree question can be asked at all.
* **It forces `--split-k 0 --two-stage always`.** A merged wgrad tile computes a
  `Gm x Gm` block of group pairs and wants only the diagonal; the packed-atomic
  split-K epilogue cannot drop the off-diagonal pairs, so merging and
  `split_k > 1` only coexist through the two-stage path. Letting the sweep
  choose freely would compare a merged two-stage kernel against an unmerged
  atomic one and attribute the epilogue's effect to the degree.
* **It forces `--verify`.** An unverified kernel must never be allowed to set an
  oracle.
* **It forces `--csv-top 999`.** The default cap silently drops the slowest
  rows, which on depthwise means `Gm = 1` — the baseline every ratio is taken
  against.

If you only want to confirm the shipped rule rather than re-derive the
rejections, the `tile_n = 64` sweep alone is sufficient.

### 3. Score

```bash
python3 score_policy.py deg_tn64.csv deg_tn128.csv deg_tn256.csv -v
python3 score_policy.py --shipped deg_tn64.csv        # one rung is enough
```

Reports every policy against two reference points: the per-shape **oracle** (how
much is left on the table) and the **shipped unmerged point** (what the change
is worth). Read the geometric mean over shapes, not the aggregate — the module
docstring explains why the aggregate reversed one of these verdicts.

### 4. Confirm what dispatch does

The selector itself is covered by
[`library/tests/dispatch/test_grouped_conv_wgrad_dispatch.py`](../../../../../../library/tests/dispatch/test_grouped_conv_wgrad_dispatch.py)
(class `TestWgradMergeDegree`), which is the load-bearing gate here — from the
`rocke/` root rather than from this directory:

```bash
python3 -m pytest library/tests/dispatch/test_grouped_conv_wgrad_dispatch.py -q
```

`test_every_choice_is_admissible` is the one that matters. At `split_k == -1`
the kernel gate's split-K clause is vacuous, so `support()` validates little
more than the tile bound — a degree the gate refuses does not merely merge
badly, it makes `support()` reject and depthwise wgrad stops dispatching at all.
That test runs every selected degree back through the real gate, so a clause
that moves in the kernel and not in the selector fails there.

## Scope and when to revisit

Measured on **gfx950**, **16-bit** (fp16/bf16), **backward-weight**, at the
**single tile dispatch ships**. The shipped rule has no fitted constant, so it
cannot go stale the way a fitted model can — but the *decision not to spend a
wider tile* is a measured one, and is worth re-checking if the wgrad tile, the
LDS budget, or the split-K resolution order changes. Forward and dgrad are
different arrangements and are not covered; dgrad has no merge support at all.
