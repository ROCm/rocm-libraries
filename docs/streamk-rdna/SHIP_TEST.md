# Would shipping StreamK for navi31 TN HHS actually win?

The question forced by the scope finding in `GATE_RESULT.md`: a default gfx1100 build ships
**no** StreamK kernels (2560 `StreamK: 0` against 22 `StreamK: 3`, and those 22 sit in
`Experimental/`, which `tasks.py` excludes by default). So before asking whether the gate
helps, ask whether StreamK belongs in the library at all.

**FINAL. 1500 shapes, 12000 measurements, zero failures.** Data `results/ship_test.csv`,
analysis `ship_analyze.py`.

---

## Arms, and why they are in this order

| pos | arm | what it is |
|---|---|---|
| 1 | `grid_sk0` | `devlib_stock_grid`, 298 `StreamK: 0` solutions — **shipped-representative baseline** |
| 2 | `sk3` | `exp/stock` SK3 catalog, no gate |
| 3 | `sk3_gate` | same catalog + `ORIGAMI_MN_GATE=2867000` |
| 4 | `grid_sk0_aa` | identical to arm 1 — **A/A, deliberately last** |

The A/A twin is placed at the far end so the pair brackets the whole interleave and measures
the **maximum** position drift. Measured at full n: **99.89% tput — 0.11 pt**, a third of
what the plateau run carried. Drift is removed by a linear-in-position model (there is no
definitionally-identical partition across *different libraries*), and at 0.11 pt that model
does almost no work.

`--fixed-iters 20` was mandatory: the libraries differ in size (298 vs 192 solutions) and
tiered iteration counts charge one-time library init unevenly — an artifact worth 5 points
in an earlier campaign.

## Result

| corrected tput vs shipped | full n | drop top 10 (17.7% of time) |
|---|---|---|
| `sk3` (no gate) | **97.91%** | 99.98% |
| `sk3_gate` | **100.25%** | 101.72% |
| gate effect (the gap) | **+2.34 pt** | +1.74 pt |

**Answer: with the gate, shipping StreamK is roughly parity — not a win. Without the gate it
is a loss.** The gate is what closes a ~2.3 pt hole, and it is the only part of this run
worth acting on.

Per band, corrected tput — this is the whole mechanism:

| band | n | % of kernel time | `sk3` | `sk3_gate` |
|---|---|---|---|---|
| `<0.1ms` | 794 | 5.4% | **115.47%** | **116.33%** |
| `0.1-1ms` | 551 | 25.9% | 102.64% | 104.94% |
| `1-5ms` | 129 | 43.6% | 97.43% | 99.23% |
| `>=5ms` | 18 | 25.1% | **91.39%** | **94.79%** |

StreamK is worth **+15%** on sub-0.1 ms shapes and **−9%** on the largest ones. Sub-0.1 ms is
53% of shapes but 5.4% of the wall-clock; `>=1ms` is 10% of shapes but 69% of it. That is
mechanically what StreamK is for — filling an underutilised machine is a small-shape problem
— and it is why the per-shape and per-second views disagree so violently.

## The cross-library number needs its jackknife printed beside it

| dropped | `sk3` | `sk3_gate` |
|---|---|---|
| 0 | 97.91% | 100.25% |
| 1 | 97.86% | 100.26% |
| 5 | 99.79% | 101.45% |
| 10 | 99.98% | 101.72% |
| 25 | 100.56% | 102.51% |
| 50 | 101.61% | 103.42% |

Concentration: top 5 shapes = 11.0% of kernel time, top 10 = 17.7%, top 25 = 30.5%.

`sk3_gate` stays at or above 100% at **every** depth, so its verdict is stable. `sk3` does
not — it crosses from loss to parity once the largest handful are removed, so "StreamK
without the gate loses" is partly a statement about those shapes.

**Mid-run this was much worse.** At n=1024 the verdict flipped sign after dropping five
shapes, and at n=338 vs n=681 the headline moved 3.4 pt because the `>=5ms` band grew from
4 shapes / 15% of time to 13 / 36%. Two headlines were stated and retracted before the
population filled in. On this suite, **never quote a cross-library wall-clock figure without
the concentration and jackknife tables** — `ship_analyze.py` now prints both unconditionally.

## The clean contrast, and a protocol-dependence finding

Everything above confounds StreamK with catalog: 192 tuned SK3 solutions against 298 SK0
ones. Prior work puts the SK3 catalog alone at +4.26% and tuning at +2.74% on the Grid
catalog, so the cross-library numbers are **not** attributable to StreamK.

`sk3_gate` vs `sk3` has no such confound — same library, the gate is the only difference:

| read | n=338 | n=681 | n=1024 | **final** |
|---|---|---|---|---|
| corrected tput | 102.69% | 102.74% | 102.35% | **102.39%** |

Rock steady, and an independent reproduction of `gate_full`'s result on a different library
build and a different protocol.

**But the magnitudes differ, and the reason is protocol:** 102.39% here against `gate_full`'s
101.34% suite figure. This run uses `--fixed-iters 20`; `gate_full` used tiered counts giving
large shapes only 5 iterations. The gate's entire benefit lives on large shapes, so resolving
them better makes the gate look better. Same direction, magnitude set by iteration counts —
a concrete instance of the protocol dependence that has bitten this workspace before.

## What to take from this

1. **The gate is real and reproducible** — +1.4% to +2.4% depending on protocol, across two
   independent runs, two library builds, stable under jackknife, no catalog confound.
2. **Shipping StreamK is not a win on this suite**, gate or no gate — parity at best.
3. **Neither is realizable on a default build**, which contains no StreamK kernels at all.
4. **Metric choice decided the answer three separate times.** Per-shape geomean says StreamK
   is +7%; wall-clock says parity. Always report both, and say which one the decision rests on.
