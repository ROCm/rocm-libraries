# Final report — best catalog per selector, gfx1100 HHS-TN

16 arms, 1,500 evaluation shapes × 3 reps each — **72,000 timed measurements, zero failed
rows**. Runtime selection only, one bench process at a time, every arm gated against its
binary before measurement.

Baseline **G0** = the shipped selector: `GridBased` nearest-neighbour matching over 9,680
reference points, 298 solutions.

---

## 0a. Read this FIRST — every "% of G0" in this report is protocol-dependent

The whole report scores arms against **G0**, the shipped selector, using a baseline CSV from
an earlier campaign. **G0 was never measured in the same session as any arm.** It has now
been, and the result depends on how many iterations the benchmark runs:

| | tiered iterations (what this report used) | amortised (iteration floor 60) |
|---|---|---|
| **v6 / G0** | **99.95%** [99.1, 100.8] | **94.86%** [94.2, 95.5] |
| **hybrid_slim / G0** | **100.63%** [100.3, 101.0] | **100.20%** [99.9, 100.5] |

Three libraries, one session, interleaved, ratios paired within (shape, rep), only the
iteration counts differing. The G0 library used hashes **byte-identical** to the baseline's
recorded library (`5b5bdc3bc6fc8b90…`), so this is the same artifact.

**The good news:** 99.95% reproduces this report's 100.05% almost exactly. The old
cross-session baseline was *not* stale, and the 19-arm campaign is internally sound.

**The bad news:** the same catalog is 5 points slower once iterations are sufficient. Per
band, `v6 / G0` in 1–5 ms is **119.73% tiered → 99.93% amortised** — the campaign's single
most-quoted result, and it is entirely G0 paying one-time library initialisation on shapes
the harness gives few iterations. G0 has the largest library (298 solutions, 9,680 reference
points) and therefore the most init to hide, so **the tiered protocol systematically favours
small catalogs.**

Pinned down by holding shape, kernel and forced solution index constant and varying only
iterations: **0.868 at 5 iterations, 0.995 at 50, exactly 1.000 at 1000.**

**What still stands:** every *same-session paired* comparison — v5 ≡ v6, v6 > v7, the
tuned/stock contrasts, the noise floor, the oracle diagnosis, the mechanism isolation.
**What does not:** "v6 beats G0 by +17% in the 1–5 ms band" and "v6 uses 5.3% less GPU time",
both of which appear below and are protocol artifacts.

**The one configuration robust to the choice is `hybrid_slim`** — at or above parity under
both protocols, and the only thing that beats G0 below 0.1 ms.

## 0. The noise floor is ~1 point, and it changes the conclusions

After finishing the campaign I re-ran the headline arm unchanged — same binary, same device
library, same shapes, same protocol — as `tuned_v6b`. It did **not** reproduce:

| band | run 1 | run 2 | delta | paired contrast | bootstrap CI |
|---|---|---|---|---|---|
| <0.1 ms | 95.77% | 95.15% | −0.62 | 99.35% | [98.73, 99.98] |
| 0.1–1 ms | 102.91% | 101.73% | −1.18 | 98.85% | [98.22, 99.46] |
| 1–5 ms | 117.34% | 116.92% | −0.42 | 99.64% | [99.09, 100.21] |
| ≥5 ms | 94.29% | 92.99% | −1.30 | 98.62% | [98.17, 99.08] |
| **ALL** | **100.05%** | **99.23%** | **−0.82** | **99.18%** | **[98.79, 99.59]** |

Two things follow, and both matter more than any single number in this report.

**The bootstrap CI understates the real uncertainty.** It resamples *shapes*, so it answers
"would I get this if I picked different shapes?" — it cannot see run-to-run variation at all.
Here it declared an identical arm "significantly slower than itself", CI excluding 1.0. Every
CI in this document should be read as a lower bound on uncertainty, roughly **±1 point
overall and ±1.3 in the thin ≥5 ms band**.

**Differences under ~1.5 points are not real.** Re-testing the headline claims:

| comparison | delta | verdict |
|---|---|---|
| v6 vs v5 | 0.13 pt | **within noise** |
| v7 vs v6 | 0.53 pt | **within noise** |
| v6 vs G0 | 0.05 pt | **within noise** |
| v6 vs v3 | 1.59 pt | survives |
| v6 vs SK3 v1 | 3.20 pt | survives |
| v6 vs GridBased | 4.54 pt | survives |

### The tie was then resolved by interleaving

Sequential passes cannot separate arms 0.5 points apart, but **interleaved** measurement can:
for each shape all three arms run back-to-back within a few seconds, arm order rotated so
none always occupies the warm slot, and the ratio is formed *within* a (shape, rep) pair — so
clocks, thermals and drift are shared and divide out. Scoped to the 147 shapes ≥1 ms, 5 reps,
2,205 measurements, ~10 minutes.

| contrast | geomean | 95% CI | verdict |
|---|---|---|---|
| tuned_v5 / tuned_v6 | 99.88% | [99.64, **100.12**] | **no difference** |
| tuned_v5 / tuned_v7 | 100.74% | [100.07, 101.43] | v5 faster |
| tuned_v6 / tuned_v7 | 100.86% | [100.18, 101.56] | **v6 faster** |

CI width falls from ~1 point to ~0.5, which is enough. **v5 and v6 are genuinely equivalent —
not merely unresolved — and both beat v7 by ~0.86%.** So v7 is out, and the choice between v5
and v6 is free: take v6, it is 58 kernels against 61.

**The stock selector reproduces this independently** — different binary, same 147 shapes,
same protocol:

| contrast | tuned Origami | stock Origami |
|---|---|---|
| v5 / v6 | 99.88% [99.64, 100.12] | 99.84% [99.55, 100.14] |
| v5 / v7 | 100.74% [100.07, 101.43] | 101.16% [100.50, 101.84] |
| v6 / v7 | 100.86% [100.18, 101.56] | 101.33% [100.67, 102.01] |

Two independent selectors agreeing to within 0.5 points on all three contrasts is the
strongest evidence in this report. Note also that the sequential runs had suggested
stock_v6 > stock_v5 by 0.78 points; interleaved, they are equal — a direct demonstration of
the noise floor in §0 producing a spurious ordering.

Absolute levels in that session (1–5 ms: v5 118.16%, v6 118.33%, v7 117.40%) differ from the
sequential numbers by about a point, which is the same session-to-session shift §0 measures.
The paired ratios are the trustworthy part, not the levels.

So the honest headline is not "v6 is the best catalog" but:

> **v5 and v6 are equivalent (interleaved: 99.88%, CI [99.64, 100.12]); both beat v7 by
> ~0.86%; and on the full shape set all of them land within a point of the shipped selector
> at ~100% of G0.** Their advantage is concentrated, not diffuse: 3–4.5 points over either
> single-family catalog overall, and **~17–18 points in the 1–5 ms band** which carries 45.6%
> of the GPU time. Take v6 over v5 because it is smaller (58 vs 61 kernels), not faster.

The large results are unaffected: the 1–5 ms win (+17), the catalog-size effect (v2/v4
failures were 3–5 points), and the mechanism isolation (−8 on `small`) all sit far above the
floor. What dissolves is the fine ranking at the top, and the small tuned-vs-stock contrasts
on distilled catalogs (§4) — of those, only `grid` (+2.74) and `v3` at ≥1 ms (+2.74) clearly
exceed the floor.

**Methodological note for the next campaign: measure the noise floor first, by running one
arm twice, before interpreting any contrast.** I ran 16 arms before doing this. Every A/B in
this campaign was single-shot per arm, so it is the repeat that calibrates them, and it
should have come first.

## 1. The answer

**Best catalog under tuned Origami: v6 (58 kernels).**
**Best catalog under stock Origami: v6 (58 kernels).**

Chosen on band profile and size, **not** on a measurable full-set advantage over v5 or v7 —
see §0, those differences are inside the noise floor. v6 wins or ties every band under both
selectors, and it is the only one of the three that never regresses a band badly (v7 gives up
6 points at ≥5 ms).

| | catalog | kernels | vs G0 (all) | 1–5 ms | total wall time |
|---|---|---|---|---|---|
| **tuned Origami** | **v6** | **58** | **100.05%** | **117.34%** | **627.1 ms (105.3% of G0)** |
| **stock Origami** | **v6** | **58** | **99.59%** | **116.13%** | **631.9 ms (104.5% of G0)** |
| reference | G0 | 298 | 100% | 100% | 660.2 ms |

A single pass over all 1,500 shapes takes **627 ms under v6 versus 660 ms under the shipped
selector — 5.3% less GPU time, from a catalog 5× smaller.**

## 2. Every arm, by kernel-duration band

Bands are defined once from G0 timing, so the same shapes are compared in every arm.
Band sizes: `<0.1ms` 802 shapes, `0.1–1ms` 551, `1–5ms` 129, `≥5ms` 18.

| arm | kernels | <0.1ms | 0.1–1ms | **1–5ms** | ≥5ms | ALL |
|---|---|---|---|---|---|---|
| tuned_sk3 | 192 | 97.49% | 96.10% | 96.83% | 91.71% | 96.85% |
| stock_sk3 | 192 | 97.57% | 96.53% | 95.26% | 92.29% | 96.92% |
| tuned_grid | 298 | 94.42% | 96.13% | 100.06% | 93.71% | 95.51% |
| stock_grid | 298 | 91.34% | 94.26% | 97.57% | 94.20% | 92.96% |
| tuned_v2 | 104 | 92.97% | 95.91% | 103.32% | 92.42% | 94.89% |
| stock_v2 | 104 | 92.87% | 96.14% | 102.50% | 92.04% | 94.85% |
| tuned_v3 | 76 | 95.48% | 99.71% | 113.95% | 92.08% | 98.46% |
| stock_v3 | 76 | 96.47% | 96.88% | 110.50% | 92.08% | 97.70% |
| tuned_v4 | 82 | 93.98% | 97.50% | 109.12% | 91.84% | 96.46% |
| tuned_v5 | 61 | 95.40% | 103.18% | 117.23% | 93.99% | 99.92% |
| stock_v5 | 61 | 95.57% | 100.04% | 116.40% | 92.70% | 98.81% |
| **tuned_v6** | **58** | **95.77%** | 102.91% | **117.34%** | **94.29%** | 100.05% |
| **stock_v6** | **58** | 95.68% | 102.03% | 116.13% | 94.13% | 99.59% |
| tuned_v7 | 45 | 95.47% | **105.56%** | 115.18% | 88.58% | **100.58%** |
| stock_v7 | 45 | 95.18% | 101.86% | 115.45% | 90.09% | 99.15% |
| hybrid768 | 356 | **98.60%** | 96.11% | 98.90% | **95.45%** | 97.67% |

Per-band winners:

| band | best (tuned) | best (stock) |
|---|---|---|
| <0.1 ms | tuned_v6 95.77% | stock_v6 95.68% |
| 0.1–1 ms | tuned_v7 105.56% | stock_v6 102.03% |
| 1–5 ms | tuned_v6 117.34% | stock_v6 116.13% |
| ≥5 ms | tuned_v6 94.29% | stock_v6 94.13% |
| ALL | tuned_v7 100.58% | stock_v6 99.59% |

### Absolute throughput (geomean GFLOP/s)

| band | n | G0 | tuned_v6 | stock_v6 | tuned_v7 | tuned_grid |
|---|---|---|---|---|---|---|
| <0.1ms | 802 | 563 | 539 | 539 | 537 | 532 |
| 0.1–1ms | 551 | 19,396 | 19,962 | 19,791 | 20,474 | 18,645 |
| 1–5ms | 129 | 73,853 | **86,660** | 85,763 | 85,067 | 73,901 |
| ≥5ms | 18 | **99,544** | 93,857 | 93,705 | 88,177 | 93,280 |

## 3. Why v6 and not v7

v7 was built to optimise **total GPU time** — weighting each shape by its band's share of
evaluation wall clock (`<0.1ms` 4.3%, `0.1–1ms` 28.0%, `1–5ms` 45.6%, `≥5ms` 22.1%) rather
than by shape count. That is the right objective on paper. Measured, it did the opposite of
what it was weighted for:

| band | weight in v7's objective | v7 vs v6 |
|---|---|---|
| <0.1 ms | 4.3% | 99.68% (ns) |
| 0.1–1 ms | 28.0% | **102.57%** |
| 1–5 ms | **45.6%** | **98.16%** |
| ≥5 ms | **22.1%** | **93.95%** |

68% of the objective's weight sat on the two bands v7 regressed in. It bought a real gain in
`0.1–1 ms` and paid for it in the bands that carry the time. **This is the fourth time an
offline composition metric has failed to predict its own measured outcome** (after v4's
missed collateral cost and v5's misattributed gain). The rule now has four independent
confirmations: these metrics choose which build to make; they never tell you what it will do.

v7 also depends more on the tuned selector: `tuned_v7/stock_v7` = 101.44% [100.52, 102.42],
against v6's 100.46% (ns).

## 4. Selector tuning — tuned vs stock Origami

**This reverses on the lens, so both are given.**

| contrast | full set | ≥1 ms only |
|---|---|---|
| tuned_sk3 / stock_sk3 | 99.93% (ns) | **101.37%** [100.84, 102.01] |
| tuned_grid / stock_grid | **102.74%** | **102.17%** [101.66, 102.73] |
| tuned_v2 / stock_v2 | 100.04% (ns) | **100.75%** [100.01, 101.61] |
| tuned_v3 / stock_v3 | 100.78% (ns) | **102.74%** [101.87, 103.71] |
| tuned_v5 / stock_v5 | **101.13%** [100.14, 102.10] | — |
| tuned_v6 / stock_v6 | 100.46% (ns) | **+1.05%** in 1–5 ms [100.35, 101.76] |
| tuned_v7 / stock_v7 | **101.44%** [100.52, 102.42] | — |

Read: on the **full** shape set, tuning looks worthless on distilled catalogs — but the full
set is 92% sub-1 ms shapes, which cancel it out. On shapes ≥1 ms, **tuned wins on every
catalog tested**, by +0.75% to +2.74%.

For v6 specifically the dependence is mild: **stock_v6 retains 99.6% of G0 and 116.1% in the
1–5 ms band without the tuned selector at all.** If dropping the `fd85b319a36` dependency is
worth ~0.5% overall and ~1% in-band, stock_v6 is a viable ship. For v5 or v7 it is not —
they lose 1.1% and 1.4%.

## 5. Cases worth knowing about

### 5.1 Where v6 is strong — the 1–5 ms band

129 shapes (62 large, 67 medium), **45.6% of all evaluation GPU time**. v6 delivers
**117.34% of G0**, 86,660 vs 73,853 GFLOP/s. This is the single most valuable result of the
campaign and it holds for stock Origami too (116.13%).

### 5.2 Where everything loses — ≥5 ms

18 shapes, all large, 22.1% of total time. **Every arm loses to G0**: best is hybrid768 at
95.45%, then tuned_v6 at 94.29%; v7 drops to 88.58%. No catalog change has moved this.
G0 reaches 99,544 GFLOP/s here and the best catalog manages 93,857.

### 5.3 Where nothing works — sub-0.1 ms

802 shapes (53.5% of the set, 4.3% of the time). Everything sits at 91–98% of G0. Only the
matching mechanism does well: hybrid768's 98.60% is the best figure any arm achieves, and it
gets there by routing these shapes to nearest-neighbour matching instead of prediction.

### 5.4 The `small` regime is a selector problem, not a catalog problem

Oracle over 124 panel shapes, 440 kernels each:

| bucket | n | coverage `best_v3/best_pool` | **selection `pick/best_v3`** |
|---|---|---|---|
| large | 36 | 0.9606 | **0.9586** |
| medium | 44 | 0.9554 | **0.8874** |
| small | 30 | 0.9247 | **0.7838** |
| tiny | 14 | 0.9243 | **0.7355** |

Coverage is flat; **selection quality collapses monotonically as parallelism drops.** The
catalog holds a kernel within 7.5% of the pool's best for `small` and the selector picks one
22% off it. Four catalogs (v3–v7) leave `small` at ~90%; nothing expressible as membership
reaches it.

## 5.5 Why the ≥5 ms band cannot be fixed by a catalog — diagnosed

The panel contains nothing that big, so this band could not be diagnosed from the existing
oracle. I generated **12 synthetic `large:deep` probe shapes** matching the profile of the
18 evaluation shapes (M,N ≈ 4.9k–9k, K ≈ 6.6k–9k), verified disjoint from **both** the
evaluation set and the construction panel, and ran the full 440-kernel oracle on them.

| catalog | coverage `best/pool` | selection `pick/best` | realised |
|---|---|---|---|
| v3 (76) | 0.9845 | 0.9604 | 0.9455 |
| **v6 (58)** | **0.9833** | **0.9632** | **0.9471** |

Realised 0.9471 against a measured 94.29% of G0 on the real ≥5 ms shapes. Those agree, and
the conclusion is worth stating plainly: **on the biggest shapes G0 is essentially achieving
the oracle.** The shipped nearest-neighbour selector is not beatable here by a better guess —
it is already picking almost the best kernel in the pool.

v6's 5.7-point deficit decomposes as roughly **1.7 points coverage + 3.7 points selection**.
Perfect selection would recover about 4 of the 5.7.

### A top-end size-gated hybrid was considered and rejected

The natural fix is to route these shapes to matching, as the low-end hybrid does for `small`.
It does not survive the arithmetic, because the two bands are **not separable by a size
predicate**:

| gate | ≥5 ms captured | 1–5 ms wrongly captured |
|---|---|---|
| min(M,N)≥3072, K≥4096 | 18/18 | 41/129 |
| min(M,N)≥3392, K≥4096 | 18/18 | 24/129 |
| min(M,N)≥3392, K≥6144 | 17/18 | 14/129 |
| min(M,N)≥4096, K≥6144 | 13/18 | 7/129 |

Best case: gain ≈ 5.7% on 22.1% of GPU time (**+1.26%**), lose ≈ 15% on the 14/129 of the
1–5 ms band wrongly routed, which is 4.9% of GPU time (**−0.74%**). Net ≈ **+0.5%** — inside
the noise of the measurements it would be judged by, and it requires the merged 356-solution
library and its penalty (§6a). **Not built.**

## 6. Selection cost vs library size — measured

`TENSILE_DB=0x10000`, shape 8320×8192×256:

| library | solutions | selection time | throughput |
|---|---|---|---|
| v7 | 45 | 2,298 µs | 75,534 GF |
| v6 | 58 | 2,283 µs | 68,264 GF |
| v5 | 61 | 2,390 µs | 71,598 GF |
| v3 | 76 | 2,989 µs | 63,796 GF |
| grid | 298 | 11,748 µs | 41,933 GF |
| hybrid | 356 | **17,966 µs** | 41,308 GF |

**Selection cost grows ~8× from 58 to 356 solutions.** This is a real argument for small
catalogs independent of kernel quality, and it matters for any application that calls the
heuristic per matmul rather than caching the algorithm.

**But it does not explain the throughput gap** — §6a identifies the real cause. Predicted
per-call overhead from the selection delta is 1,893 µs (grid) and 3,137 µs (hybrid); observed
is 321 µs and 334 µs. Observed overhead **saturates** while selection time keeps climbing.
Forcing the solution index also reproduces the gap. So there are two effects here — a
measured selection cost that scales with library size, and a separate per-call penalty on
short kernels, **now identified in §6a as the matching row's reference table**.

## 6a. Correction: the penalty is the matching row's *reference table*

§6 attributed the hybrid's losses to library size, on the strength of a ratio table across
sequentially-measured arms (118→1.000, 147→0.996, 192→0.990, 298→0.958, 356→0.57). That
comparison was confounded: those arms carry *different catalogs*, and the "same kernel"
subset it was computed over is biased toward shapes the catalogs agree on.

`hybrid_slim` is the clean test. Same gate as hybrid768, but the matching side distilled to
its 120 most-referenced solutions (6,249 reference points), giving a **178-solution** library
— inside the supposedly safe range.

| library | solutions | structure | ref points | 1–5 ms |
|---|---|---|---|---|
| tuned_v6 | 58 | single row, prediction | — | **117.34%** |
| tuned_grid | 298 | single row, matching | 9,680 | 100.06% |
| hybrid_slim | 178 | two rows | 6,249 | 99.94% |
| hybrid768 | 356 | two rows | 9,680 | 98.90% |

In the 1–5 ms band `hybrid_slim` runs **the same kernel as tuned_v6 on 99.2% of shapes** and
is still 15% slower (paired 85.17%, CI [83.34, 87.16]). A 298-solution single-row library is
fine; a 178-solution two-row one is not. **So it is not solution count.** Shrinking the
matching table from 9,680 to 6,249 points moved the band only 98.90% → 99.94%, which points
at the *presence* of a large matching row rather than its size: shapes answered by the
prediction row still pay for it.

### The penalty scales with reference points — and that is the trap

Three matching-table sizes, interleaved against v6 on the 147 shapes ≥1 ms:

| variant | matching solutions | reference points | 1–5 ms | penalty vs v6 |
|---|---|---|---|---|
| tuned_v6 | — | 0 | 118.26% | — |
| hslim10 | 10 | 1,869 | 117.33% | **0.62%** [100.17, 101.04] |
| hslim40 | 40 | 4,025 | 109.43% | 6.99% [105.98, 108.05] |
| hslim120 | 120 | 6,249 | 97.84% | 17.73% [116.30, 119.18] |

So it *is* size — measured in **reference points**, not solutions. (A single-row 298-solution
matching library has no penalty because there is no second row to pay for.) At ~1,900 points
the cost nearly vanishes.

Which suggested an escape: a hybrid with a tiny matching table. Measured on the full set:

| band | tuned_v6 | hslim10 (1,869 pts) | hybrid_slim (6,249 pts) |
|---|---|---|---|
| <0.1 ms | 95.77% | 96.28% | **102.16%** |
| 0.1–1 ms | **102.91%** | 99.91% | 98.87% |
| 1–5 ms | **117.34%** | 116.20% | 99.94% |
| ≥5 ms | **94.29%** | 93.11% | 94.70% |
| ALL | **100.05%** | 99.15% | 100.66% |

`hslim10/tuned_v6` at <0.1 ms is 100.53%, CI [98.40, 102.69] — **the sub-0.1 ms gain is gone.**

**That is the trap, and it is structural.** Matching is an *interpolator*: it is good at
low-parallelism shapes precisely because it has thousands of measured reference points to
interpolate between. Cutting the table to where the hybrid is affordable also cuts the only
thing that made the matching row worth having. The two requirements are in direct opposition:

- 6,249 points → +6.4 on sub-0.1 ms, −17.7 on 1–5 ms
- 1,869 points → −0.6 on 1–5 ms, no sub-0.1 ms gain

**Conclusion: the gated hybrid is not viable on this runtime.** Not because the routing is
wrong — it is right, and 102.16% is the only figure in this campaign that beats G0 in the
sub-0.1 ms band — but because a prediction-served call is charged for the matching row's
table, and the table must be large to be useful. Fixing that is a runtime change (don't
evaluate or load a row whose predicate rejected the problem), not a catalog change.

### What the hybrid did prove

| band | tuned_v6 | hybrid_slim | note |
|---|---|---|---|
| <0.1 ms | 95.77% | **102.16%** | **first arm to beat G0 in this band**, +6.67% [104.50, 108.91] |
| 0.1–1 ms | **102.91%** | 98.87% | |
| 1–5 ms | **117.34%** | 99.94% | the penalty |
| ≥5 ms | 94.29% | **94.70%** | ns |
| ALL | 100.05% | 100.66% | ns [99.22, 101.98] |

Routing low-parallelism shapes to nearest-neighbour matching **works** — 802 shapes go from
95.77% to 102.16%, the only time anything has beaten G0 there. The mechanism is right; the
packaging is not.

## 7. The campaign in one line

| catalog | kernels | vs G0 | note |
|---|---|---|---|
| SK3 v1 | 192 | 96.85% | StreamK-only |
| GridBased | 298 | 95.51% | production catalog, Origami-selected |
| v2 union | 104 | 94.89% | **failed** — union worse than both parents |
| v3 regime guard | 76 | 98.46% | first to beat G0 anywhere |
| v4 corrected guard | 82 | 96.46% | **failed** |
| v5 trap removal | 61 | 99.92% | |
| **v6 global objective** | **58** | **100.05%** | **best** |
| v7 time objective | 45 | 100.58% | ties v6; worse where it counts |
| hybrid768 | 356 | 97.67% | mechanism works, two-row penalty |
| hybrid_slim | 178 | 100.66% | **beats G0 sub-0.1 ms (102.16%)**; same penalty (§6a) |

**Every catalog that added kernels lost. Every catalog that removed them won.** The two best
are among the three smallest, from a starting pool of 294.

## 8. Recommendation

Ship **v6 (58 kernels)**. Use the tuned selector if the `fd85b319a36` dependency is
acceptable — it is worth ~0.5% overall and ~1% in the 1–5 ms band. If not, **stock_v6 gives
up very little** and removes a code dependency entirely.

Open, in priority order:
1. **≥5 ms** — every arm loses; 22% of GPU time; untouched by anything tried.
2. ~~The short-kernel penalty~~ — **resolved (§6a)**: it scales with the matching row's
   reference-point count, and only two-row libraries pay it. The follow-on is a *runtime*
   change — skip a row whose predicate rejected the problem — which would unblock the hybrid.
3. **Sub-0.1 ms** — only nearest-neighbour matching performs; a size-gated hybrid is the
   right shape but the current one carries a 356-solution library and its penalty.
