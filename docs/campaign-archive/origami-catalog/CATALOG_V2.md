# SK3 Catalog v2 / v3 — cascade-ranked distillation over a retuned union pool

gfx1100 (Navi31, 96 CU), HHS-TN, `Cijk_Alik_Bljk_HHS_BH_Bias_HA_S_SAV`.
1,500 frozen evaluation shapes × 3 reps per arm, runtime selection only, zero failed rows in any arm.
Baseline **G0** = the shipped GridBased selector (`reports/final_confirmed_details.csv`).

## Headline

| arm | catalog | selector | geomean vs G0 | >10% worse | >25% worse | P10 | better |
|---|---|---|---|---|---|---|---|
| stock_sk3 | SK3 v1 (192) | stock Origami | 96.92% | 412 | 139 | 76.34% | 610 |
| tuned_sk3 | SK3 v1 (192) | tuned Origami | 96.85% | 412 | 137 | 76.40% | 638 |
| stock_grid | GridBased (298) | stock Origami | 92.96% | 438 | 178 | 71.96% | 570 |
| tuned_grid | GridBased (298) | tuned Origami | 95.51% | 359 | 135 | 76.20% | 666 |
| tuned_v2 | v2 union (104) | tuned Origami | 94.89% | 435 | 164 | 73.58% | 686 |
| stock_v2 | v2 union (104) | stock Origami | 94.85% | 423 | 172 | 73.26% | 679 |
| **tuned_v3** | **v3 guarded (76)** | **tuned Origami** | **98.46%** | 407 | 176 | 72.08% | **796** |
| stock_v3 | v3 guarded (76) | stock Origami | 97.70% | 390 | 175 | 72.59% | 736 |
| tuned_v4 | v4 3-bucket (82) | tuned Origami | 96.46% | 432 | 151 | 74.82% | 709 |
| **tuned_v5** | **v5 trap-removed (61)** | **tuned Origami** | **99.92%** | 386 | 165 | 73.06% | **810** |
| **tuned_v6** | **v6 global-objective (58)** | **tuned Origami** | **100.05%** | 394 | 177 | 72.13% | 799 |

`tuned_v3` is the best Origami-selected arm measured in this campaign, by a margin that
survives bootstrapping:

| contrast | geomean | 95% CI | verdict |
|---|---|---|---|
| tuned_v3 / tuned_sk3 | 101.66% | [100.57%, 102.76%] | first faster |
| tuned_v3 / tuned_grid | 103.08% | [101.94%, 104.26%] | first faster |
| tuned_v3 / tuned_v2 | 103.76% | [102.65%, 104.88%] | first faster |
| tuned_v2 / stock_v2 | 100.04% | [99.26%, 100.80%] | no difference |
| tuned_v3 / stock_v3 | 100.78% | [99.81%, 101.72%] | no difference |
| stock_v3 / tuned_sk3 | 100.87% | [99.82%, 101.98%] | no difference |

## The two registered predictions, judged plainly

**Prediction 1 — v2 beats both single-catalog arms: FAILED.** v2 measured 94.89%, below
`tuned_sk3` (96.85%) and `tuned_grid` (95.51%). The union catalog was worse than either
catalog it was built from. **v3, the guarded revision, succeeded** at 98.46%.

**Prediction 2 — the tail improves: FAILED, for v2 and v3 alike.** Against v1's 412/137
shapes >10%/>25% below G0, v3 gives 407/176 and its P10 *fell* from 76.40% to 72.08%.
The plan pre-registered that "if geomean rises but the tail does not, the catalog did not
fix what blocks deployment — report as failure." On the full 1,500-shape set, that is the
correct verdict for the tail.

The qualification, which is real but does not overturn the above: the tail damage is
confined to shapes with a small dimension. Restricted to shapes with **every** dimension
≥1000 (205 shapes), v3 has **zero** shapes >25% below G0 and 8 >10% — identical to the
best previous arm — while the geomean rises to **107.33%**.

## Why v2 failed, and what v3 changed

v2 is the union of the GridBased and SK3 pools, cascade-ranked. It posted the best large
and medium numbers of any arm to that point (98.58% / 97.84%) and collapsed on tiny
(92.13% vs SK3's 110.20%).

Mechanism, measured directly: **on tiny shapes v2's runtime picked an SK0 kernel on 329
of 337 shapes (98%)** — despite SK3 winning that regime 110.20% vs 94.94%. The union did
not give the selector an escape hatch in the low-parallelism regime; it gave it a worse
option that Origami actively prefers.

So the correction is not a better ranker but a **smaller menu**: v3 applies a *regime
guard* that removes SK0 members whose Origami top-3 appearances are predominantly in the
low-parallelism (tiny/gemv) regime. Members that also serve large/medium are kept. At the
shipped threshold 0.10 this drops 28 of v2's 78 SK0 members, leaving 76 (50 SK0 + 26 SK3),
and raises the SK3 share of low-regime selections from 2% to 71%.

**The general lesson: catalog restriction is a feature when the selector is unreliable.**
The right response to a selector that chooses badly in a known regime is to delete the
options it chooses, not to add better ones.

Guard threshold sweep (offline, share of low-regime selections served by SK3):

| threshold | SK3 share, low regime | members |
|---|---|---|
| 0.50 | 42% | 96 |
| 0.35 | 47% | 90 |
| 0.20 | 58% | 83 |
| **0.10** | **71%** | **76** |

## Per-regime

By size (geomean vs G0):

| group | n | tuned_sk3 | tuned_grid | tuned_v2 | tuned_v3 | stock_v3 |
|---|---|---|---|---|---|---|
| large | 260 | 94.03% | 98.16% | 98.58% | **103.07%** | 100.16% |
| medium | 504 | 95.84% | 97.32% | 97.84% | **102.49%** | 99.15% |
| small | 399 | 89.71% | **92.08%** | 91.29% | 89.42% | 88.82% |
| tiny | 337 | **110.20%** | 94.94% | 92.13% | 100.31% | 104.95% |

By geometry:

| group | n | tuned_sk3 | tuned_grid | tuned_v2 | tuned_v3 | stock_v3 |
|---|---|---|---|---|---|---|
| gemv | 87 | **108.42%** | 93.92% | 91.99% | 100.63% | 104.87% |
| rect | 496 | 97.93% | 96.18% | 95.19% | **99.35%** | 97.64% |
| skinny | 452 | 94.10% | 94.34% | 93.85% | 94.66% | **95.60%** |
| square | 465 | 96.37% | 96.25% | 96.14% | **100.89%** | 98.54% |

v3 is the first Origami-selected arm in this campaign to **beat the shipped GridBased
selector** — on large (103.07%), medium (102.49%) and square (100.89%). Its cost is the
`small` regime, where it is the worst arm (89.42%), and it recovers only part of SK3's
tiny/gemv lead (100.31% / 100.63% against 110.20% / 108.42%).

### All dimensions ≥1000 (205 shapes)

The four-arm campaign's conclusion reversed under this restriction, so it is always
reported. v3 does not reverse — it widens.

| arm | geomean | >10% worse | >25% worse | P10 | large (92) | medium (113) |
|---|---|---|---|---|---|---|
| tuned_sk3 | 97.29% | 25 | 1 | 89.57% | 95.62% | 98.67% |
| tuned_grid | 99.21% | 8 | 0 | 92.91% | 97.84% | 100.33% |
| **tuned_v3** | **107.33%** | 8 | 0 | 92.39% | **103.51%** | **110.55%** |

Contrasts: v3/tuned_sk3 = 110.32% [108.54, 112.18]; v3/tuned_grid = 108.19% [106.63, 109.82].
Absolute geomean throughput 70,183 GFLOP/s vs 63,616 / 64,868.

## Selector tuning is still worth nothing on a distilled catalog

`tuned_v2 / stock_v2` = 100.04%, CI [99.26%, 100.80%] and `tuned_v3 / stock_v3` = 100.78%,
CI [99.81%, 101.72%] — both indistinguishable, reproducing the v1 result
(`tuned_sk3 / stock_sk3` = 100.07%). Origami tuning paid only on the undistilled
GridBased catalog (+2.74%). Once a catalog has been distilled *using* tuned Origami, the
tuning has already been spent: it is baked into membership, and applying it again at
runtime adds nothing. This is now observed on two independent catalogs and should be
treated as the expected behaviour, not a null result. On three catalogs now, so the
practical consequence is worth stating: **a distilled catalog can ship against a stock
runtime**, removing a code dependency from the deployment. `stock_v3` alone reaches 97.70%
of G0 — better than every arm in the original four — with no selector change at all.

The stock/tuned split on v3 is not uniform, though: stock is better on tiny (104.95% vs
100.31%) and gemv (104.87% vs 100.63%), tuned is better on large (103.07% vs 100.16%) and
medium (102.49% vs 99.15%). The two cancel in the aggregate. If the deployment target is
large/medium GEMM, the tuned runtime is worth keeping despite the flat headline.

## v4 — a corrected guard that made things worse

`small` was v3's one clear loss (89.42%, worst of any arm, 399 shapes), so v4 tested the
obvious fix. The v3 guard computes the low share as `low/(low+high)`, which leaves `small`
out of the denominator entirely. Offline that looked like a plain defect: the top dropped
member serves **12 small shapes against 2 low and 2 high** and was cut on a 2-vs-2
tiebreak, and in total the guard deleted **29.6% of the catalog's small-regime top-3
service**. v4 changes the denominator to `low/(low+small+high)` at t=0.20, which restores
small service from 70.4% to 83.5% of v2's and high service from 91.2% to 97.5%, at the
cost of low-regime SK3 purity (100% → 90.6%). 82 members.

**It failed.** v4 measured **96.46%**, and `tuned_v4 / tuned_v3` = **97.98%**, CI
[97.17%, 98.79%] — v3 is genuinely better. Under `--min-dim 1000` the gap is larger still
(103.81% vs 107.33%).

| group | n | tuned_v3 | tuned_v4 | Δ |
|---|---|---|---|---|
| large | 260 | 103.07% | 100.80% | −2.27 |
| medium | 504 | 102.49% | 99.42% | −3.07 |
| small | 399 | 89.42% | **89.99%** | **+0.57** |
| tiny | 337 | 100.31% | 96.77% | −3.54 |

**The diagnosis was wrong.** Restoring 13 points of small-regime service bought **0.57%**
on `small` — inside noise — while the readmitted members cost 2–3.5 points in every other
regime. The deleted service was real; it was not the cause of the `small` deficit. Whatever
limits `small` is not catalog coverage, so no membership rule will fix it: the candidates
are that the pool contains no good kernel for those shapes, or that the selector misranks
them regardless of what is available. Distinguishing those needs an oracle pass over
`small`, not another catalog.

**The transferable lesson is about the offline metric.** "Share of top-3 service retained"
looked like a faithful proxy for what the guard costs, and it is not: a readmitted member
does not only serve the bucket it was readmitted for — it also gets selected in regimes
where it loses. The metric counts the intended benefit and none of the collateral cost.
Every offline catalog-composition metric used here has this asymmetry, so **treat them as
generators of candidate builds, never as predictors of measured outcome.** The guard
threshold sweep in the previous section deserves the same scepticism.

This also strengthens the v2 finding rather than weakening it: three catalogs now
(v2 = 104, v4 = 82, v3 = 76) rank in inverse order of size. Restriction is doing the work.

## The oracle pass: it is a selection failure, everywhere below `large`

v4 showed `small` is not a membership problem but could not say what it *is*. An oracle
pass settles it: every deduped pool member (440 kernels) measured on 64 construction-panel
shapes, stratified across regimes, split into three ratios. Run on the **panel, never the
frozen evaluation set** — the output is an oracle, and anything built from an eval-set
oracle could no longer be evaluated on it.

| bucket | n | coverage (best_v3 / best_pool) | **selection (pick / best_v3)** | realised (pick / best_pool) |
|---|---|---|---|---|
| large | 6 | 0.9594 | **0.9826** | 0.9427 |
| medium | 14 | 0.9424 | **0.8702** | 0.8201 |
| small | 30 | 0.9247 | **0.7838** | 0.7248 |
| tiny | 14 | 0.9243 | **0.7355** | 0.6798 |

**Coverage is flat at ~0.92–0.96 and selection collapses monotonically from 0.98 to 0.74.**
The v3 catalog *contains* a kernel within 7.5% of the pool oracle for `small`; Origami then
picks one 22% off it. Worst cases are severe — `8448x64x1024` at 0.544 of the best member
present, `2048x448x16` at 0.562.

This closes the question v4 opened. `small` is not short of kernels, so no membership rule
can fix it — which is exactly why v4 failed, and the oracle explains *why* rather than just
recording that it did.

**The framing correction matters more than the `small` answer.** Ranking quality is not
weak in one regime; it degrades monotonically with available parallelism, and it is *worst*
on `tiny` (0.7355) — the regime where v3 measures a comfortable 100.31% of G0. Performance
relative to G0 conflates selector quality with how good G0 itself is: on tiny, G0's own
choice is far enough from the oracle that even a badly-selected SK3 kernel beats it; on
small, G0 is close to the oracle and the same selector quality shows as a 10-point loss.
**Ratios against a baseline cannot diagnose a selector — only an oracle can.**

Headroom, if selection on `small` were perfect against the catalog v3 already ships:
89.42 / 0.7838 ≈ **114% of G0**. The kernels are there.

The actionable reading is that the remaining gap is not a catalog problem at all: either
the analytical model must be made to rank in the low-parallelism regime, or those regimes
must be served by a different selection mechanism. See the correction below for what that
mechanism actually is. A prediction/lookup hybrid split on parallelism is the experiment this points
to, not another catalog.

## v5 — removing the traps the oracle identified

The oracle said the gap is selection, and restriction is the only lever that has ever moved
selection here. v5 applies it with **measurement** instead of a model or an appearance
count: for each of the 30 `small` panel shapes the oracle knows what every kernel achieved,
so it knows exactly which member Origami ranks first and how far off the catalog's own best
it is. A member repeatedly ranked first and repeatedly far off is a trap; dropping it hands
those shapes to whatever Origami ranks next.

The search is greedy (a drop changes what gets picked everywhere) and guarded (a drop is
rejected if any other bucket loses more than 0.005 geomean selection efficiency). Origami's
score for a (shape, config) pair does not depend on the rest of the set, so one rank order
per shape is computed once and each candidate subset is pure bookkeeping over measured
numbers. 15 drops, **61 members** (39 SK0 + 22 SK3).

Offline it moved `small` selection efficiency 0.7986 → 0.8904 with medium and tiny also
improving and large unchanged.

**Measured: 99.92% of G0 — parity with the shipped selector**, and `tuned_v5 / tuned_v3` =
101.49%, CI [100.65%, 102.36%].

| group | n | tuned_v3 | tuned_v5 | Δ |
|---|---|---|---|---|
| large | 260 | 103.07% | **104.45%** | +1.38 |
| medium | 504 | 102.49% | **105.41%** | +2.92 |
| small | 399 | 89.42% | 90.48% | +1.06 |
| tiny | 337 | 100.31% | 100.28% | −0.03 |

Restricted to all dimensions ≥1000 it is stronger still, and this is the **first arm whose
tail also improves**:

| arm | geomean | >10% worse | >25% worse | worst | P10 |
|---|---|---|---|---|---|
| tuned_grid | 99.21% | 8 | 0 | 80.28% | 92.91% |
| tuned_v3 | 107.33% | 8 | 0 | 81.01% | 92.39% |
| **tuned_v5** | **112.13%** | **2** | 0 | **89.41%** | **94.25%** |

73,316 GFLOP/s geomean against G0's implied 65,383 and v3's 70,183.

**The offline metric was again wrong about where the gain would come from.** It was built to
fix `small` and predicted +9.2 points of selection efficiency there; measured, `small` moved
+1.06% while the win came from medium (+2.92), large (+1.38) and skinny (+2.52). Net
positive this time, but the v4 lesson stands unchanged and is now demonstrated in both
directions: these metrics choose which build to make, they do not tell you what it will do.

`small` remains unsolved at 90.48%.

## v6 — a different objective, the same ceiling

v5 optimised `small` alone and the gains arrived in medium, so v6 optimises the objective
that actually matches the measurement: geomean selection efficiency across all buckets,
**weighted by the evaluation set's bucket counts** rather than by the panel's (the panel
over-samples `small` for statistical power, so its own mix is a distribution nobody
measures). The per-bucket guard is retained — a weighted mean will otherwise trade a small
bucket away for a large one, and a catalog that collapses a regime is not shippable
whatever it scores in aggregate.

18 drops, **58 members**. Offline it beats v5 on the shared objective (0.8980 vs 0.8892),
mostly on medium and tiny, at some cost on `small`.

**Measured: 100.05% of G0 — and indistinguishable from v5.** `tuned_v6 / tuned_v5` =
100.13%, CI [99.47%, 100.78%]. Under `--min-dim 1000`, 112.20% against 112.13%.

| group | n | tuned_v5 | tuned_v6 |
|---|---|---|---|
| large | 260 | 104.45% | 104.43% |
| medium | 504 | 105.41% | **106.08%** |
| small | 399 | **90.48%** | 89.95% |
| tiny | 337 | 100.28% | **100.59%** |

The 0.9-point offline edge did not convert, which is the §v4/§v5 rule holding for a third
time. The useful reading is not that v6 failed but that **two independently-motivated
objectives, optimised to convergence over the same measured data, land on the same
performance.** That is evidence the ~100%-of-G0 plateau is a real ceiling for this
pool-plus-selector, not the local optimum of one objective. v6 reaches it with 3 fewer
kernels, which is the only concrete reason to prefer it.

## Where the campaign ends

| catalog | members | vs G0 | all dims ≥1000 |
|---|---|---|---|
| SK3 v1 | 192 | 96.85% | 97.29% |
| GridBased | 298 | 95.51% | 99.21% |
| v2 union | 104 | 94.89% | — |
| v3 regime guard | 76 | 98.46% | 107.33% |
| v4 corrected guard | 82 | 96.46% | 103.81% |
| **v5 trap removal** | **61** | **99.92%** | **112.13%** |
| **v6 global objective** | **58** | **100.05%** | **112.20%** |

Six catalogs. **Every catalog that added options lost; every catalog that removed them
won**, and the two best are the two smallest. Starting from a 298-kernel production catalog
and a 192-kernel StreamK one, the best configuration found is 58 kernels — a 5× reduction
that reaches parity with the shipped per-shape lookup table overall and beats it by 12% on
shapes where every dimension is ≥1000.

Two things remain unfixed and neither is a catalog problem:

- **`small`** sits at ~90% across v3, v4, v5 and v6 alike. The oracle shows coverage is
  fine and selection is not; nothing that can be expressed as membership has moved it.
- **The full-set tail.** 394 shapes >10% below G0, essentially unchanged from v1's 412.
  It improves only under the dimension restriction (8 → 2 shapes >10%).

Both point at the same next step, and it is not another catalog — see the section below.

## Construction

Pool = GridBased 298 ∪ SK3 v1 192 ∪ 11 accepted retuned variants, deduped → **294**.

**Cascade ranker** — Origami proposes top-3, linear22 re-ranks those 3, winner and
runner-up enter. Neither model is ever used alone: linear22 is +2.23% as an *arbiter* over
a shortlist but −18.8% as a *ranker* over a whole pool. It changed Origami's top-1 on
**212 of 432** panel shapes; **35** of v3's 76 members exist only because it did. Origami's
top-3 spanned both families on **174** shapes.

**Pool-first collision collapse** — Origami is blind to StaggerU/WGM/PGR/PLR, so a retuned
variant ties *exactly* with its parent and `rank_configs` breaks the tie by enumeration
order. Collapsing after selection is too late; the loss has already happened. Collapsing
the pool first found **255 of 294 members colliding**, and **20** groups were resolved by
Stage-C measurement overriding linear22 (whose WGM weight is +6.0, i.e. it prefers WGM=8
while the retune measured WGM=1 winning in 9 of 11 accepted variants).

**Retune (Stage A/B/C)** used ~1 h of its 12 h cap: 11 of the swept variants passed both
holdout and guard. The sweep terminated on the legitimate condition — the ranked list of
recipes with `geo_conv < 1.0` was exhausted — not on budget.

## Data quality and caveats

All six arms: no non-ok rows, median CV 1.55–1.82%, within-arm drift 98.15–101.34%.

**Drift is the caveat on the v3/tuned_sk3 contrast.** v3's own within-arm drift is 98.15%,
i.e. ±1.85%, and the contrast is +1.66% — the same order. The +3.08% vs `tuned_grid` and
the +8 to +10% figures under `--min-dim 1000` are comfortably outside drift; the narrow
full-set win over `tuned_sk3` should not be leaned on alone.

## Reproduce

```bash
ab_analyze.py \
  --arm tuned_sk3=measurements/branch_tuned_sk3.csv \
  --arm tuned_grid=measurements/branch_tuned_grid.csv \
  --arm tuned_v2=measurements/branch_tuned_v2.csv \
  --arm stock_v2=measurements/branch_stock_v2.csv \
  --arm tuned_v3=measurements/branch_tuned_v3.csv \
  --contrast tuned_v3/tuned_v2 --contrast tuned_v3/tuned_sk3 --contrast tuned_v3/tuned_grid \
  --baseline reports/final_confirmed_details.csv --baseline-label G0 \
  --manifest state/build_manifest.json --stratify 1,2
```


## Correction: G0 is not a lookup table, and that changes the next step

Earlier notes in this report described G0 as a "per-shape lookup table" and proposed serving
low-parallelism shapes by exact match. **Both were wrong, and the check is one line:**

    G0 exact table: 9,680 distinct sizes
    of the 1,500 evaluation shapes, 0 appear in it — 0.0%, in every bucket

G0 wins without a single exact hit. Element `[11]` of a logic file is not merely a type tag:
for anything other than `FreeSize`/`Prediction`, `LibraryIO.parseLibraryLogicList` assigns it
as the **distance metric** of a Matching library (`LibraryIO.py:665-670`), with element `[7]`
as the reference table. So `GridBased` is nearest-neighbour in problem space over 9,680 tuned
reference points — an interpolator, not a dictionary. An exact-match hybrid would have fired
on 0% of the evaluation set.

### What that buys: the mechanism isolated, from data already collected

The `*_grid` arms are the **same 298 GridBased solutions** as G0 with `[11]` rewritten to
`Prediction` and `[7]` nulled. Catalog held constant, mechanism swapped:

| bucket | n | G0 (matching) | tuned Origami | stock Origami | mechanism cost |
|---|---|---|---|---|---|
| large | 260 | 100.00% | 98.16% | 96.93% | **−1.84** |
| medium | 504 | 100.00% | 97.32% | 94.20% | **−2.68** |
| small | 399 | 100.00% | 92.08% | 89.26% | **−7.92** |
| tiny | 337 | 100.00% | 94.94% | 92.59% | **−5.06** |

**Swapping nearest-neighbour matching for analytical prediction costs 8 points on `small`
and under 2 on `large`, with the catalog identical.** This is the cleanest isolation in the
campaign and it needed no new measurement — it independently reproduces the oracle's
selection-efficiency collapse (0.98 large → 0.78 small) from a completely different angle.

It also explains the whole scoreboard. v5/v6 beat G0 on large and medium because a distilled
catalog more than covers a ~2-point mechanism deficit there; nothing they do on `small`
covers an 8-point one, which is why four successive catalogs all land at ~90%.

### Why the hybrid cannot be built as a catalog change

A Matching library with a distance metric **never misses** — it always returns its nearest
reference point. Shipping a Matching logic and a Prediction logic together therefore does
not produce "exact first, predict on miss": the matching row answers everything and the
prediction library is never consulted. That is the same mechanism behind the campaign's
standing rule that element `[7]` must be nulled when converting a catalog to `Prediction`.

The C++ runtime *does* support the composite — `ExactLogicLibrary::findBestSolution`
(`ExactLogicLibrary.hpp:102-156`) iterates predicate rows and falls through — but the
existing predicates discriminate on **hardware chip ID**, not problem size, and the YAML
offers no way to express "this row only for shapes below N". A true prediction/matching
hybrid needs a problem-size predicate on the row, which is a source change, not a catalog
one. That is the concrete next step, and it is now specified rather than guessed at.
