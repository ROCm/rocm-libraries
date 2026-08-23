# StreamK on gfx1100 TN HHS is a per-shape win and a wall-clock loss

Offline re-analysis of the 2026-08-19 grid campaign. **No new GPU time** — this reuses
`p1_modes.csv` (1500 shapes × 2 reps × 6 grid modes) and asks a question the campaign
collected the data for but did not ask.

Scripts: `gated_policy.py`, `gated_robust.py`. Campaign report:
`~/hhs_tn_grid_vs_resource_origami_9k/reports/STREAMK_CAMPAIGN.md`.

---

## Headline

| | per-shape geomean | flops-weighted wall-clock |
|---|---|---|
| StreamK **off** (mode 4) vs shipped predictor (mode 6) | **96.95%** | **102.17%** |

**The two metrics disagree in sign.** The campaign concluded "the predictor is vindicated"
from the 96.95% geomean. On wall-clock, the shipped predictor is a **~2% net loss** — you
would finish the same 1495-shape suite faster with StreamK disabled entirely.

Both numbers are correct. They answer different questions, and only one of them is the
question a library ships against.

## Why the sign flips — the whole mechanism, in one table

| band | n | share of shapes | **share of kernel time** | geomean | tput-wtd |
|---|---|---|---|---|---|
| `<0.1ms` | 797 | 53.3% | **3.5%** | 93.92% | 93.79% |
| `0.1-1ms` | 551 | 36.9% | 19.5% | 100.18% | 102.04% |
| `1-5ms` | 129 | 8.6% | **48.1%** | 101.91% | 102.06% |
| `>=5ms` | 18 | 1.2% | **28.9%** | 101.66% | 103.56% |

Geomean and throughput agree *within* every band. The reversal is **purely a weighting
effect**: geomean gives 53% of its vote to shapes holding 3.5% of the wall-clock, where
StreamK genuinely wins 6%, and 1.2% of its vote to shapes holding 29% of the wall-clock,
where it loses.

StreamK pays exactly where the campaign said it does — small shapes. That band is just
almost irrelevant to total runtime on this suite.

### Validation of the pipeline

The per-band geomeans reproduce the campaign's independently-published table to the last
digit (`93.92 / 100.18 / 101.91 / 101.66` vs published `93.92 / 100.17 / 101.91 / 101.66`).
The re-analysis is reading the data the same way the original did.

### It is not a handful of giant shapes

| dropped (largest time consumers) | n | geomean | tput-wtd |
|---|---|---|---|
| 0 | 1495 | 96.95% | **102.17%** |
| 10 (20.6% of time) | 1485 | 96.92% | 101.50% |
| 100 (**68.9% of time**) | 1395 | 96.63% | **101.07%** |
| 200 | 1295 | 96.07% | 99.29% |

Remove the top 100 shapes — over two thirds of all kernel time — and the reversal still
holds. It only closes once 200 shapes are gone, by which point most of the large-shape
population that the effect is *about* has been deleted.

## The gated policy

If StreamK helps below 0.1 ms and hurts above 1 ms, gate it. The catch: those bands are
**measured duration**, which a selector cannot observe. A shippable predicate must key on
problem *size*. Threshold fitted on a train half, scored on the held-out half:

| policy | geomean | tput-wtd | scope |
|---|---|---|---|
| always off (mode 4) | 96.95% | 102.17% | all |
| **ORACLE per-shape** | 102.57% | 103.41% | all (unachievable) |
| duration-gated `>=0.5ms` | 100.35% | 102.25% | all (oracle — uses measured time) |
| **size-gated `M*N >= 2.867e6`** | **101.30%** | **102.08%** | **TEST half** |
| size-gated `M*N*K >= 1.51e8` | 100.34% | 101.96% | TEST half |
| size-gated `K >= 1` | 97.96% | 101.80% | TEST half |

**`M*N` is the right predicate, and it is better than either extreme.** It keeps ~all of
the wall-clock win (102.08% vs always-off's 102.17%) while *also* being +1.30% on geomean
instead of −3.05% — i.e. it does not pay for the large shapes by sacrificing the small
ones. It holds out-of-sample.

`M*N` beating `M*N*K` is mechanically sensible: StreamK's benefit is about having too few
tiles to fill the machine, and tile count is `M*N`-driven. `K` alone is the wrong axis and
scores worst, as expected.

## What this does and does not establish

**Established.** On this 1495-shape TN HHS suite, on this card, the shipped grid predictor
loses ~2% of wall-clock relative to disabling StreamK, and a size-gated policy recovers
that without giving up the small-shape win. Robust to dropping 69% of the kernel time.

**Not established.** The gated policy has **never been run as a real arm**. Every number in
that table is a projection formed by recombining two measured arms per shape. It assumes
the selector's *kernel choice* is unchanged by the grid mode — which the campaign did
verify (selection agreement `m4` vs `m6` = 100.0%, same library), so the assumption is
sound, but projection is not measurement.

**Noise floor.** Rep-vs-rep on the same arm: median 0.40%, p95 5.51% per shape. Aggregate
figures over ~750–1495 shapes are far tighter than the per-shape p95, and the campaign's
own per-band bootstrap CIs already exclude 100% for `1-5ms` ([101.25, 102.57]). The
`>=5ms` row is the weakest: n=18, and OFF wins on exactly 50.0% of them, so its 103.56%
comes from magnitude rather than frequency. Do not quote that row alone.

**Suite dependence.** "Share of kernel time" is a property of *this* shape distribution. A
workload made only of small GEMMs would reverse the conclusion back. The finding is that
the metric choice is load-bearing, not that StreamK is universally bad.

## Next step, if this is worth pursuing

Patch `k_split_aware` in `shared/origami/src/origami/streamk.cpp` to return
`data_parallel` above an `M*N` threshold, build, and measure it as a **real arm** against
mode 6 on the same 1500 shapes — paired, same session, with an A/A repeat for the floor.
That converts the projection into a measurement and is ~1 h of GPU time.

## The transferable lesson

This is the same trap recorded for the SK3-vs-SK4 work (`REPORT.md` §6) and for the
catalog campaign: **a ratio aggregated one way can invert when aggregated another.** There
the trap was ratio-within-config vs absolute-across-config; here it is per-shape vote vs
per-second weight. In both cases the honest fix is the same — report both, and say which
one the decision is being made on.
