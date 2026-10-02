<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# Depthwise wgrad merged groups — gfx950 case study

How `dispatch.grouped_convolution._wgrad_merge_degree()` was derived, and why it
ended up with no fitted constant in it at all.

Per [`platform/AGENTS.md`](../../../../../AGENTS.md) §Compliance this document
records **methodology, levers, and relative ratios only**. Every number below is
a ratio between two rocke configurations measured in the same process on the
same silicon; no absolute latency, throughput, or achieved-FLOP figure appears
here or in any commit message. Measured scope: gfx950, 16-bit (fp16/bf16),
`--direction wgrad`, depthwise (`cpg == kpg == 1`), 197 shapes, tile pinned to
what dispatch ships except where a tile sweep is explicitly named. Ratios are
taken against **the configuration that shipped before this change** — the same
tile, unmerged — because that is the only baseline that answers "what is this
worth".

## Table of contents

- [What was investigated](#what-was-investigated)
- [The instrument](#the-instrument)
- [Finding 1: wgrad's merge bound is a different animal from forward's](#finding-1-wgrads-merge-bound-is-a-different-animal-from-forwards)
- [Finding 2: climbing the ladder changes only the load vector width](#finding-2-climbing-the-ladder-changes-only-the-load-vector-width)
- [Finding 3: spending a wider tile to admit a larger degree is a bad trade](#finding-3-spending-a-wider-tile-to-admit-a-larger-degree-is-a-bad-trade)
- [Finding 4: the residual a model would chase is bounded at about 5%](#finding-4-the-residual-a-model-would-chase-is-bounded-at-about-5)
- [Rejected: a vectorisation target](#rejected-a-vectorisation-target)
- [Rejected: a grid-occupancy floor](#rejected-a-grid-occupancy-floor)
- [The metric that lied](#the-metric-that-lied)
- [Correctness gate](#correctness-gate)
- [Results](#results)
- [Caveats](#caveats)
- [Replay](#replay)
- [Keep / revert / defer decisions](#keep--revert--defer-decisions)

## What was investigated

Forward depthwise convolution had just gained an analytic merged-group selector:
a fitted closed form with eight constants that picks `group_merge` per shape.
The question here was the obvious follow-up — do the same for the backward
direction.

First the scope had to be established, and it is narrower than "backward".
Walking the kernel gates, the only backward direction with `group_merge` support
is **backward-weight**; dgrad has none. So this is a wgrad study.

Wgrad's GEMM is `wg_M = kpg`, `wg_N = spatial * cpg` where `spatial = Z*Y*X`,
and `wg_K = N*Ho*Wo`. Depthwise collapses that to a per-group GEMM of
`1 x (Y*X) x (N*Ho*Wo)`: the free axis is **one element wide per group**, so
every dY and X load is scalar no matter what the tile is. That is the pathology.
Merging `Gm` groups into one tile makes the free axis `Gm` elements long, and
that is the entire mechanism — for depthwise wgrad, **`Gm` *is* the load vector
width**. Nothing else about the kernel changes.

## The instrument

[`merge_degree_model/`](merge_degree_model/) — see its README for the runbook.
Three pieces:

- `gen_shapes.py` builds a 200-line corpus: half real depthwise stages from
  published architectures, half a stratified random grid, with a deliberate
  minority of **unmergeable** shapes (filters too large for two groups to share
  a tile, group counts no ladder rung divides). Those are the population a
  selector can only damage, so a corpus without them cannot detect the damage.
- `measure_degrees.py` pins tile, warp, pipeline and epilogue and sweeps the
  degree and split-K, one CSV per `tile_n` rung. It forces verification on, and
  forces `--split-k 0 --two-stage always` — see [Correctness gate](#correctness-gate)
  for why the epilogue has to be held fixed.
- `score_policy.py` is the file that decided what ships. It scores the shipped
  rule and the three rejected policies against the same surface.

## Finding 1: wgrad's merge bound is a different animal from forward's

This is the finding the rest follows from, and it is why the forward model does
not transfer.

Forward puts `Gm` on GemmN and GemmK. The tile bound (`Gm <= tile_n`) is slack
for every degree the gate offers, so the tile is simply not part of the
decision; what the degree trades is vector width against a working set growing
linearly in `Y*X*Gm`. That is a genuine interior optimum, it moves with the
shape, and a model is the right tool for it.

Wgrad puts `Gm` on **both** GemmM and GemmN, and the gate caps it at
`Z*Y*X*cpg*Gm <= tile_n` (plus `kpg*Gm <= tile_m`, which depthwise satisfies
trivially since `kpg == 1`). With the tile fixed that bound bites hard. At the
shipped width:

| filter | spatial | admissible degrees |
|---|---|---|
| 1x1 | 1 | up to 64 |
| 1x3 / 3x1 | 3 | up to 16 |
| 3x3 | 9 | up to 4 |
| 3x5 | 15 | up to 4 |
| 5x5 | 25 | up to 2 |
| 7x7 | 49 | **none** — `49 * 2` already exceeds the tile |

The admissible set is two or three degrees wide on most real shapes, and empty
on large filters. There is no interior to search. So the useful question for
wgrad was never "which degree" but "is it worth widening the tile to admit a
larger one" — which is Finding 3.

## Finding 2: climbing the ladder changes only the load vector width

Before measuring, it is worth knowing what the degree can and cannot affect,
because three plausible costs turn out to be invariant.

Walking the "largest admissible degree at each tile rung" ladder:

- **Total MAC work is invariant.** Merging reshapes the GEMM; it does not add
  or remove multiply-accumulates.
- **Tile fill is invariant.** The bound is `spatial * Gm <= tile_n`, and the
  largest admissible degree is the one that fills the tile; every rung fills it
  to the same fraction.
- **Launched CTA count is invariant.** This one was a surprise and it refuted a
  working hypothesis. Merging divides the group axis of the grid by `Gm`, which
  looks like it ought to starve the machine. But `select_split_k_wgrad` runs
  *after* merging and sees the merged grid, so it simply raises `split_k` to
  compensate. Across every candidate measured, total CTAs pin in a narrow band
  near full occupancy regardless of degree.

What does change is the load vector width (up, with the degree) and the LDS
footprint (up, with the tile). So climbing a rung at a **fixed** tile only ever
buys width and costs nothing — which is why the top of the admissible set is
almost always right. Climbing a rung by **widening the tile** buys width and
costs LDS, which is a real trade, and Finding 3 measures it.

## Finding 3: spending a wider tile to admit a larger degree is a bad trade

The tempting policy is a ladder: climb `tile_n` until the degree reaches some
target. It is the only way a 7x7 stage merges at all. It was measured at three
rungs across the full corpus, in three variants.

All three lose. Against the shipped unmerged point, by geometric mean over
shapes:

| policy | tile mix | geomean | shapes regressed (<0.98x) | worst |
|---|---|---|---|---|
| ladder, no occupancy floor | 47 / 80 / 70 | 2.26x | 26 / 197 | 0.46x |
| ladder, floor applied | 55 / 75 / 67 | 2.30x | 20 / 197 | 0.49x |
| ladder, no target cap | 33 / 16 / 148 | 1.94x | 45 / 197 | 0.46x |
| **shipped: one tile, max degree** | **197 / 0 / 0** | **1.98x** | **1 / 197** | **0.97x** |

The ladder's extra ~16% of geometric mean is bought with 20–26 shapes in 197
regressing by up to roughly half. Two things about that tail decided it:

- **It is concentrated on the widest rung, and the rung is bad on its own
  terms.** Counting every measured configuration by how far it lands below its
  own shape's best: **88%** of widest-rung configurations are more than 20% off,
  against **49%** at the shipped tile and 58% in between. The widest rung is not
  a slightly worse bet, it is mostly a bad one — past a point the LDS cost of
  the wider tile exceeds what the extra vector width returns.
- **The damage scales with how far the policy climbs.** The "no target cap"
  variant, which keeps climbing while the degree keeps rising, pushes three
  quarters of the corpus onto the widest rung — and it is the worst of the
  three by a wide margin, the only one that falls below the single-tile rule
  outright. More rungs spent, more cliff.

All three ladder variants carry a fallback that keeps the shipped tile for a
shape that cannot merge at any rung, and that fallback is load-bearing: the
26 corpus shapes in that category, forced to the widest tile instead, run at
**0.64x** — they pay the full LDS cost for width they get nothing from. A
reviewer reading the table above should know the ladders are already scored
*with* their best-case guard in place.

For a dispatch **default** — which has to serve shapes nobody measured — trading
a bounded upside for an unbounded tail is the wrong side of the exchange. The
upside is capped by the oracle; the downside is not capped by anything in the
corpus.

A secondary observation, recorded because it is the honest reason the ladder is
not simply "wrong": the ladder's wins are real, and a *tuning* path (as opposed
to a dispatch default) that measures a specific shape would be right to take
them. This study rejects the ladder as a default, not as a fact.

## Finding 4: the residual a model would chase is bounded at about 5%

This is what closed the question.

Across the 135 corpus shapes with more than one admissible degree at the shipped
tile, picking the maximum admissible degree is the **exact** best degree on 97
of them. On the 38 where it is not, the gap to the best degree at that tile is:

- worst case **0.954x**,
- median **0.990x**,
- and **zero** shapes below 0.95x.

So the entire quantity a fitted model could recover is under 5% on a quarter of
the mergeable shapes, and under 1% on half of those. Taken over the whole
corpus, the trivial rule captures **0.998 of the per-shape oracle** at the
shipped tile.

A model with eight constants, a corpus to maintain, and a drift risk against the
kernel gate, in exchange for at most a few percent on a minority of shapes, is
not a good trade. The forward direction has an interior optimum worth fitting;
this one does not. That asymmetry is the finding, and it is the thing to
re-check if the wgrad tile or the gate's bound ever changes.

## Rejected: a vectorisation target

Hypothesis: stop merging once the merged run reaches the widest buffer load, on
the theory that degrees past that point buy no additional width and only cost
LDS.

The theory is sound and the knob is inert. At a fixed tile the
`spatial * Gm <= tile_n` bound binds *before* the vectorisation target on nearly
every shape in the corpus — the only shapes that reach a wide enough run to
trigger the cap are 1x1 and 1x3 stages, which the gate was already going to cap.
The knob only ever changes an outcome in combination with the tile ladder, and
the tile ladder is rejected. Dropped as a free parameter that does not vary.

## Rejected: a grid-occupancy floor

Hypothesis: refuse to merge past the point where too few merged groups survive
to fill the machine.

Refuted by Finding 2. Split-K is resolved after merging and refills the grid, so
the floor declines speedup it has no reason to decline. Scored over the corpus
it is approximately neutral in aggregate and slightly negative per shape — it
wins on a handful of genuinely small-group shapes (every miss it fixes has a
group count in the single or low double digits) and loses on more shapes than
that elsewhere.

The small-group effect it was built for **is** real; it is just already handled
by the admissibility bound, which caps the degree at the group count anyway.

## The metric that lied

Worth recording because it reversed a verdict mid-study.

The first scoring harness reported an aggregate ratio, `sum(baseline) /
sum(picked)` over the corpus. That is **time-weighted**: a single expensive
shape can conceal a hundred cheap regressions. Under it, the grid-occupancy
floor looked worth having — by a margin smaller than run-to-run noise — while
the per-shape geometric mean showed it losing on more shapes than it won.

`score_policy.py` reports both, and the module docstring says plainly that only
the geometric mean, taken against the point that ships today rather than against
the unreachable oracle, decided anything. If this study is ever extended, read
that one.

## Correctness gate

Per the optimization runbook, never report speed without correctness.

Verification is forced on inside `measure_degrees.py` rather than left to the
caller, so an unverified kernel cannot set an oracle. Across the full campaign —
three tile rungs, 197 shapes, the entire degree ladder, and the split-K ladder
at each point — **13244 ranked configurations verified, 0 failures.**

The epilogue has to be held fixed for this to mean anything, which is the reason
for `--split-k 0 --two-stage always`. A merged wgrad tile computes a `Gm x Gm`
block of group *pairs* and wants only the diagonal; the packed-atomic split-K
epilogue cannot drop the off-diagonal pairs, so merging and `split_k > 1` only
coexist through the two-stage path. Letting the sweep choose freely would
compare a merged two-stage kernel against an unmerged atomic one and attribute
the epilogue's effect to the degree.

Dispatch-level and on-silicon gates for the shipped selector, all green:

| gate | result |
|---|---|
| `library/tests/dispatch/` + wgrad merge-gate + layering | 511 passed, 2279 subtests |
| `library/tests/test_conv_wgrad_correctness.py` (on gfx950) | 66 passed, 346 subtests |
| every selected degree replayed through the real kernel gate | 816 requests admitted, 0 refused |

That last one is the load-bearing test. At `split_k == -1` (auto) the gate's
split-K clause is vacuous, so `support()` validates little more than the tile
bound — a degree the gate refuses does not merely merge badly, it makes
`support()` reject and **depthwise wgrad stops dispatching at all**. The
selector re-derives the bound rather than probing the gate (building a
`WgradConvSpec` per degree to answer a closed-form question is a lot of work to
learn `spatial * Gm <= tile_n`), so the two can drift;
`TestWgradMergeDegree.test_every_choice_is_admissible` runs every choice back
through the real gate across a grid of shapes, so a clause that moves in the
kernel and not in the selector fails there.

## Results

The shipped rule, over 197 depthwise wgrad shapes, against the same tile
unmerged:

| statistic | ratio |
|---|---|
| geometric mean | **1.98x** |
| median | 1.25x |
| p75 | 3.69x |
| p90 | 7.12x |
| best shape | 36.7x |
| worst shape | 0.97x |
| shapes regressed below 0.98x | 1 / 197 |
| shapes unchanged within ±2% | 90 / 197 |
| shapes at or above 2x | 80 / 197 |
| fraction of the per-shape oracle captured at the shipped tile | 0.998 |

The 90 unchanged shapes are the unmergeable ones — large filters, or group
counts no ladder rung divides — which the rule correctly leaves alone. The
single regression is a 5x5 stage that merges to degree 2 and loses about 3%.

By degree, geometric mean against unmerged:

| degree | shapes | geomean |
|---|---|---|
| 1 (no merge) | 62 | 1.00x |
| 2 | 34 | 1.70x |
| 4 | 81 | 3.05x |
| 8 | 8 | 3.78x |
| 16 | 8 | 2.74x |
| 64 | 4 | 6.80x |

## Caveats

- **The degree-64 row is measured but not currently reachable.** All four of
  those shapes are 1x1 depthwise, and `dispatch_conv_grouped` refuses grouped
  pointwise wgrad outright ("grouped pointwise (1x1) wgrad is not yet
  supported") before the selector is consulted. They are admissible at the
  *kernel* gate, so the benchmark measures them, but no dispatch today reaches
  them. Excluding all 1x1 shapes moves the headline geometric mean from 1.98x
  to 1.94x, so the result does not depend on them.
- **The scalar-B wgrad variant is excluded by construction.** It pins its own
  narrower tile, which the degree bound was not derived against; that path keeps
  the unmerged behaviour it was tuned with. The two do not overlap in practice —
  the scalar-B predicate excludes depthwise and merging is depthwise-only — so
  the guard is belt-and-braces rather than a policy choice.
- **The corpus has 200 generated lines but 197 distinct descriptors.** Three
  pairs differ only in a field the driver's shape string does not carry, so the
  scorer merges them. The effect is three shapes' worth of weight, not a
  correctness issue.
- **gfx950 and 16-bit only.** Forward and dgrad are different arrangements and
  are not covered by any of this; dgrad has no merge support at all.
- **The decision not to spend a wider tile is a measured one**, unlike the rule
  itself. It is worth re-checking if the wgrad tile, the LDS budget, or the
  order in which split-K is resolved relative to merging changes.

## Replay

Everything is in [`merge_degree_model/`](merge_degree_model/); its README is the
runbook. The short version, from that directory on a gfx950 host:

```bash
python3 gen_shapes.py --count 200 --model-cap 100 -o shapes_wgrad.txt
python3 measure_degrees.py shapes_wgrad.txt -o deg_tn64.csv            # shipped tile
python3 measure_degrees.py shapes_wgrad.txt -o deg_tn128.csv --tile 64x128x64/2x2/32
python3 measure_degrees.py shapes_wgrad.txt -o deg_tn256.csv --tile 64x256x64/2x2/32
python3 score_policy.py deg_tn64.csv deg_tn128.csv deg_tn256.csv -v
```

To confirm the shipped rule only, the first sweep alone is enough:
`python3 score_policy.py --shipped deg_tn64.csv`. The rejected ladders need the
other two rungs.

The measured CSVs are deliberately not checked in: they are raw per-kernel
timings and carry absolute numbers. `score_policy.py` degrades to a clean "no
measured corpus present" when they are absent.

## Keep / revert / defer decisions

**Keep.**

- `_wgrad_merge_degree()` as the depthwise wgrad merge policy: the kernel gate's
  own admissibility rule, evaluated at the shipped tile, and nothing more. It
  has **no tuned constant**, which is the point — it cannot drift away from the
  gate, because it *is* the gate.
- `TestWgradMergeDegree.test_every_choice_is_admissible` as the drift guard.
- The scalar-B exclusion in the `_group_merge` closure.
- `merge_degree_model/` in examples, so the rejections can be re-derived rather
  than taken on trust.

**Reverted during the study.**

- The `tile_n` ladder, and with it the `_tile()` change that would have let the
  selector move tile geometry. Dispatch's wgrad tile is untouched by this work;
  the blast radius of the change is one spec field.
- The vectorisation target and the grid-occupancy floor, as free parameters that
  never varied an outcome on their own.

**Deferred.**

- Grouped pointwise (1x1) wgrad. Merging is at its most valuable exactly there —
  `spatial == 1` admits the full degree ladder — but the dispatch candidate
  refuses the shape before the selector runs. Lifting that refusal is a separate
  change with its own correctness surface.
- Re-checking the tile decision against a changed LDS budget or split-K
  resolution order, per the caveat above.
- Porting any of this to dgrad, which would first require merge support in the
  dgrad kernel.
