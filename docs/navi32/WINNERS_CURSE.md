# Why the coverage extension was reverted: single-shot argmax selects noise

The catalog from commit `5b1cf1bc21a` was benchmarked and **regresses**. It has been reverted.
This is the measurement and the mechanism, because the mechanism generalises well beyond this
catalog.

## The measurement

4 arms x 600 shapes, cold, 60 CU, time-derived iterations. Numbers below are run 1 at 429/600
shapes; treated/control split is by the kernel each arm **actually dispatched**, so the control
ran a byte-identical catalog.

| group | n | A/A floor | extship (the pushed catalog) | nogate | geomean |
|---|---|---|---|---|---|
| ALL | 397 | 100.48% | 99.90% | 99.22% | 98.59% |
| **treated** | 85 | 100.46% | **98.70%** | 98.53% | **94.49%** |
| control | 312 | 100.49% | 100.21% | 99.39% | 99.73% |

The control does not move, so the regression is attributable to the re-mapped rows. `nogate`
is no better, so the tiny/gemv gate is not the cause.

## Confirmed across two independent runs

The aggregate is not the evidence; per-shape reproducibility is. Treated shapes present in both
runs (n=55 at the time of writing, run 2 still filling):

| | |
|---|---|
| per-shape delta correlation run 1 vs run 2 | **r = +0.983** |
| same sign in both runs | **91%** |
| same statistic on the A/A arm (noise reference) | **r = +0.351** |
| slower in **both** runs | 30/55 (55%) |
| median | run 1 −1.5%, run 2 −2.0% |

Worst repeat offenders: `skinny_N −43%`, `skinny_M −35%`, `skinny_M −31%`, `gemv −29%`.

This is the same test that certified the original re-map as a genuine **+2.1%** (r=0.961, 90% sign
agreement, against an A/A reference of 0.551). Applied here it certifies the opposite with a
tighter correlation: **the same shapes lose every time, so the loss is structural.**

## The mechanism: the winner's curse

For each treated query, compare what the matrix **predicted** at the grid row against what the
benchmark **measured** at the query:

```
matrix-predicted gain at the ROW : median  +22.8%
benchmark gain at the QUERY      : median   -2.5%
correlation                      : r = -0.441      <- NEGATIVE
predictions with the right sign  : 40%             <- worse than a coin flip
```

Worst cases: predicted **+546.6% -> actual -31.0%**, predicted +352.8% -> actual -35.4%.
A predicted 5x speedup is not a kernel, it is a near-zero denominator: the incumbent happened to
measure absurdly slow **once**.

`--algo_method all` measures 298 kernels in **one shot** per shape. Taking the argmax over 298
noisy samples selects whichever kernel drew the most favourable noise. The expected value of a
maximum over noisy draws is biased upward, and the bias grows with the number of candidates. So
the apparent gain is largely the noise itself, and re-pointing away from a sane incumbent loses.

**No magnitude band escapes it:**

| predicted gain | n | median actual | helped |
|---|---|---|---|
| 2-5% | 8 | -0.7% | 38% |
| 5-10% | 17 | -0.5% | 47% |
| 10-25% | 22 | -3.8% | 45% |
| 25-60% | 27 | -1.9% | 41% |
| 60-150% | 11 | -3.3% | 36% |
| >150% | 7 | -28.4% | 0% |

Capping implausible predictions does not rescue it either (cap at +5%: still -0.7% median).
**The `--min-gain 0.02` gate is useless against this**, because the noise is far larger than 2%
on the affected rows.

## No stratum rescues it either — including `med`

Complete run 1, 597 shapes / 126 treated. **63% of treated shapes got slower, median -2.6%,
worst -43%.** Per stratum the damage is uneven, and `med` looks superficially fine:

| stratum | n | predicted | actual | corr r | helped |
|---|---|---|---|---|---|
| ALL | 126 | +25.7% | -2.6% | -0.406 | 37% |
| **med** | 49 | +21.2% | **+1.2%** | **-0.142** | **53%** |
| skinny_M | 43 | +33.3% | -4.3% | -0.419 | 28% |
| skinny_N | 31 | +29.7% | -4.0% | -0.411 | 29% |

**Do not read `med` as a working case.** 53% helped is a coin flip, and its correlation is still
*negative*. `med` is noise centred slightly positive, not signal. A "re-map `med` only" variant
would be selecting on the same broken predictor over a subset where it happens to do less harm —
which is how you get a result that survives one benchmark and fails the next.

Skinny strata are hit hardest, which fits: their throughput is more variable shape-to-shape, so a
single-shot measurement has a wider noise distribution, so the maximum over 298 draws is biased
further upward.

## Why the offline analyses all looked so good

Every encouraging number in `COVERAGE_REPORT.md` was computed **from the same single-shot matrix**
that contains the bias — so they inherited it rather than testing it:

* "15.2% median reachable headroom, 89% clear the gate" — that is the winner's-curse bias,
  measured against itself.
* "0.0% headroom on rows shipped already used" — near-zero by construction, since `shipped` IS
  the argmax of those rows. Circular, and I read it as a clean calibration.
* The inverted-U in problem size, the 98% transfer tax, the catchment-selection test: all derived
  from the same matrix, all uncheckable against it.

**An offline oracle built from single-shot measurements cannot detect its own selection bias.
Only a fresh, independent measurement can** — which is exactly what the benchmark was, and it
took ~1 h to run against ~14 h of sweeping.

## What would actually fix it

1. **Repeat the measurement.** Sweep each row 2-3x and select on the median, or require the
   winner to win in a majority of repeats. This is the direct fix and it is the reason the
   original re-map (which validated at +2.1%) is not affected in the same way.
2. **Shrink toward the incumbent.** Only re-point when the challenger beats the incumbent by more
   than the *measured per-row spread*, not a fixed 2%.
3. **Select on a neighbourhood, not a point** — though note this was tested offline and was a
   wash (98.34% vs 98.28%); it may fare better once the underlying measurements are de-noised.
4. **Validate on a fresh sample before shipping, always.** The pushed commit was correctly marked
   `[UNVALIDATED]`; that label is what kept this recoverable.

## Status

* Catalog reverted to the previously validated shipped re-map.
* Tooling, shape lists, eval set and `cold_matrix_summary.json` retained — they are all still
  useful, provided the matrix is rebuilt with repeats.
* `PREDICTIONS.md` predicted **+1 to +3%** and listed "any stratum regressing beyond the A/A floor"
  as a falsification condition. The prediction was wrong in sign and the condition fired.
