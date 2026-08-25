# Why the coverage extension was reverted: `--algo_method all` ranks kernels differently than they run

The catalog from commit `5b1cf1bc21a` was benchmarked and **regresses**. It has been reverted.
The mechanism generalises well beyond this catalog, so it is written up in full — including the
diagnosis I published first, which was wrong.

*(Filename kept because earlier commits reference it. The winner's curse is what I initially
blamed; tests 1-3 below refute it.)*

## The measurement

600 shapes x 4 arms x 2 complete runs, cold, 60 CU, time-derived iterations, **0 timeouts**. The
treated/control split is by the kernel each arm **actually dispatched**, so the control ran a
byte-identical catalog.

| group | n | A/A floor | extship (the pushed catalog) | nogate | geomean |
|---|---|---|---|---|---|
| ALL | 597 | 100.37% | 100.00% | 99.04% | 98.38% |
| **treated** | 126 | 100.31% | **99.18%** | 99.01% | **93.75%** |
| control | 471 | 100.39% | 100.25% | 99.06% | 99.66% |

The control does not move, so the loss is attributable to the re-mapped rows. `nogate` is no
better, so the tiny/gemv gate is not the cause. **63% of treated shapes got slower, median -2.6%,
worst -43%.**

## Confirmed across two complete independent runs

| treated | run 1 | run 2 | spread |
|---|---|---|---|
| wall-clock | 99.18% | 99.20% | **0.02** |
| geomean | 93.75% | 93.94% | 0.19 |
| A/A floor | 100.31% | 100.85% | 0.54 |

| | |
|---|---|
| per-shape delta correlation run 1 vs run 2 | **r = +0.986** |
| same sign in both runs | **93%** |
| A/A arm, same statistic (noise reference) | **r = +0.414** |
| slower in **both** runs | **75/126** treated shapes |

Per stratum both runs agree to within ~0.5 pt: `med` 100.94/100.89, `skinny_M` 96.60/96.98,
`skinny_N` 93.93/94.41, `gemv` 73.03/74.04 (n=3, not claimable).

The regression reproduces to two decimal places. This is the same test that certified the original
re-map as a genuine **+2.1%** (r=0.961, A/A reference 0.551); applied here it certifies the
opposite, with a tighter correlation.

## The mechanism, established by three tests

### Test 1 — is it the winner's curse? NO.

If argmax over 298 single-shot samples were selecting noise, an independent repeat would collapse
the apparent gain. 127 rows were re-measured from scratch:

| predictor | median predicted | corr vs actual |
|---|---|---|
| single shot (what shipped) | +25.7% | -0.406 |
| **an independent 2nd shot** | **+22.7%** | **-0.415** |
| mean of the two | +24.1% | -0.411 |
| *(actual measured)* | **-2.6%** | |

The repeat reproduces the same +22.7%, and the run-1 winner is still the winner **68%** of the
time. **The gain is reproducible; it is not noise.** "Repeat and take the median" — the fix this
document originally recommended — would have changed nothing.

### Test 2 — is it row->query transfer? NO.

Measure the same two kernels at the QUERY shape instead of the row key, removing transfer entirely:

| same two kernels, measured at | median | corr vs benchmark |
|---|---|---|
| the ROW key, `--algo_method all` | +25.7% | -0.406 |
| the QUERY shape, `--algo_method all` | +17.6% | -0.368 |
| the QUERY shape, the benchmark | **-2.6%** | 1.000 |

Identical shape, identical kernels, and the two methods still disagree — 41% sign agreement.

### Test 3 — it is INSTRUMENT BIAS, and it is kernel-dependent.

Absolute throughput for the same kernel on the same shape, matrix vs benchmark:

```
shipped's kernel : matrix reports 1.33x the benchmark's GFlop/s   (p10 0.75, p90 2.13)
extship's kernel : matrix reports 2.11x                            (p10 1.11, p90 2.51)
ratio of biases  : 1.16x
```

**`--algo_method all` systematically overstates throughput, and overstates KERNEL-DEPENDENTLY.**
A uniform bias would cancel in a ratio and the ranking would survive; at 1.16x it does not.
Running 298 kernels back-to-back in one process does not reproduce the conditions of a single
dispatch — despite `--flush --rotating 512`. So **argmax over that enumeration selects whichever
kernel best exploits the measurement artifact**: the chosen kernel is inflated 2.11x while the
incumbent is inflated only 1.33x, which is exactly how a reproducible "+23% gain" becomes a
reproducible -2.6% loss.

## No stratum rescues it — including `med`

| stratum | n | predicted | actual | corr r | helped |
|---|---|---|---|---|---|
| ALL | 126 | +25.7% | -2.6% | -0.406 | 37% |
| **med** | 49 | +21.2% | **+1.2%** | **-0.142** | **53%** |
| skinny_M | 43 | +33.3% | -4.3% | -0.419 | 28% |
| skinny_N | 31 | +29.7% | -4.0% | -0.411 | 29% |

**Do not read `med` as a working case.** 53% helped is a coin flip and its correlation is still
negative. A "re-map `med` only" variant would select on the same broken predictor over a subset
where it does less harm — which is how you get a result that survives one benchmark and fails the
next.

## Why every offline analysis looked so good

Every encouraging number in `COVERAGE_REPORT.md` was computed **from the same biased matrix**, so
it inherited the bias instead of testing it:

* "15.2% median reachable headroom, 89% clear the gate" — the instrument bias measured against
  itself.
* "0.0% headroom on rows shipped already used" — near-zero by construction, since `shipped` IS the
  argmax of those rows. Circular, and I presented it as a clean calibration.
* The inverted-U in problem size, the 98% transfer tax, the catchment-selection test — all from the
  same matrix, none able to detect the problem.

**An offline oracle cannot detect a bias in the instrument that built it. Only measurement the way
the kernel will actually run can** — that benchmark took ~1 h against ~14 h of sweeping.

## What would actually fix it

1. **Rank kernels the way they will run — MEASURED, not asserted.** `--solution_index` (one kernel
   per process) was checked against the catalog benchmark on 30 treated queries:

   | instrument | median ratio | corr vs benchmark | sign agreement |
   |---|---|---|---|
   | `--algo_method all` at the query shape | +17.6% claimed | **-0.368** | 41% |
   | **`--solution_index`, one kernel per process** | **0.988** vs benchmark **0.988** | **+0.989** | **93%** |

   Single-dispatch reproduces the benchmark almost exactly, and independently confirms the
   regression (0.988x = the chosen kernel is slower). **This is the valid ranking instrument.**

   Cost: one process per (kernel, shape). A full 9 680 x 298 sweep this way is infeasible, so a
   two-stage scheme — shortlist with `--algo_method all`, then rank the shortlist single-dispatch —
   was the obvious candidate. **It is now TESTED and REFUTED.**

## The two-stage rescue does not work, and here is the whole picture

All 298 kernels were measured **one process at a time** on four shapes, one per stratum, and
compared against the enumeration's ordering:

| shape | stratum | true winner's rank in the enumeration | enum's pick delivers |
|---|---|---|---|
| 204x713x606 | med | **38**/298 | 90.0% of achievable |
| 23x344x893 | skinny_M | **57**/298 | 94.7% |
| 542x90x414 | skinny_N | **153**/298 | 97.1% |
| 627x5x2074 | gemv | **5**/298 | 97.0% |

Ranks of 5, 38, 57 and **153** mean a shortlist would need K≈153 — over half the pool — to be
safe, which saves nothing. **Two-stage is dead.**

The more useful number is what each choice actually delivers:

| shape | incumbent (navi31 pick) | enum's pick | true best | |
|---|---|---|---|---|
| 204x713x606 med | 81.9% | 90.0% | 100% | switching helps |
| 23x344x893 skinny_M | 95.0% | 94.7% | 100% | **switching loses** |
| 542x90x414 skinny_N | 93.7% | 97.1% | 100% | switching helps |
| 627x5x2074 gemv | 97.1% | 97.0% | 100% | **switching loses** |

**This is the complete explanation.** The incumbent is already at 82-97% of achievable. The
enumeration's argmax lands at 90-97% — statistically indistinguishable from it. So re-pointing a
row onto the enumeration's pick is a **coin flip**, which is exactly the 37-53% "helped" rate the
benchmark measured. Meanwhile the true best sits 3-18% above both, so **the headroom is real and
the enumeration simply cannot locate it.**

That also explains why every offline number looked so good: the matrix reported the enum pick as
beating the incumbent by +23%, when in single-dispatch reality the two are within a couple of
points of each other.
2. **Calibrate before trusting any enumeration.** For a sample of (kernel, shape) pairs, measure
   both ways and check the bias ratio is ~1.0. Here it is 1.16x, and that alone predicts failure.
3. **Repeating the sweep does NOT help** — refuted in test 1. Neither does capping implausible
   gains, nor `--min-gain`, nor stratum gating.
4. **Validate on a fresh independent measurement before shipping, always.** The pushed commit was
   marked `[UNVALIDATED]`; that label is what kept this recoverable.

## Scope

This affects **ranking** decisions built on `--algo_method all`. It does not automatically
invalidate the lean-Grid *reduction*, which was validated by an independent A/B (99.54%
wall-clock) and passed. But any conclusion resting on the matrix's ordering should be treated as
unverified until re-measured single-dispatch.

## Status

* Catalog reverted to the previously validated shipped re-map.
* Tooling, shape lists, eval set and `cold_matrix_summary.json` retained — still useful, but the
  matrix must not be used for ranking without the calibration in fix 2.
* `PREDICTIONS.md` predicted **+1 to +3%** and listed "any stratum regressing beyond the A/A floor"
  as falsification. The prediction was wrong in sign and the condition fired.
