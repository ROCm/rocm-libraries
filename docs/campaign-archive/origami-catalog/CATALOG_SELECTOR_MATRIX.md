# Catalog × selector matrix — gfx1100 HHS-TN

Two independent keys: **which kernels are in the library** (catalog) and **how one is
chosen** (selector). Everything is a ratio to the shipped default — the GridBased pool
selected by GridBased nearest-neighbour matching — which is therefore 100% by
definition.

1,500 evaluation shapes × 3 reps per arm, runtime selection, tiered iteration counts.
Aggregation: max over reps for GEMV, median otherwise; then geomean of per-shape ratios.
**worst 10% / best 10%** are geomeans over the bottom and top decile of per-shape
ratios — the tails that decide whether a catalog is deployable, since a healthy mean
can hide a tenth of shapes running far below the baseline.

| catalog | kernels | selector | ALL | worst 10% | best 10% | tiny | small | medium | large | gemv | skinny | rect | square |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Grid pool** | 298 | **GridBased** | **100.00** | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| Grid pool | 298 | tuned Origami | 95.51 | 62.84 | 124.97 | 94.94 | 92.08 | 97.32 | 98.16 | 93.92 | 94.34 | 96.18 | 96.25 |
| Grid pool | 298 | stock Origami | 92.96 | 57.43 | 123.76 | 92.59 | 89.26 | 94.20 | 96.93 | 93.44 | 92.24 | 92.88 | 93.66 |
| SK3 v1 | 192 | tuned Origami | 96.85 | 62.49 | 148.16 | 110.20 | 89.71 | 95.84 | 94.03 | 108.42 | 94.10 | 97.93 | 96.37 |
| SK3 v1 | 192 | stock Origami | 96.92 | 63.54 | 149.83 | 110.84 | 91.62 | 93.74 | 94.71 | 107.75 | 93.33 | 98.40 | 96.99 |
| v2 union | 104 | tuned Origami | 94.89 | 60.74 | 127.55 | 92.13 | 91.29 | 97.84 | 98.58 | 91.99 | 93.85 | 95.19 | 96.14 |
| v2 union | 104 | stock Origami | 94.85 | 60.02 | 128.09 | 93.29 | 90.41 | 97.14 | 99.58 | 93.93 | 94.53 | 95.20 | 94.96 |
| v3 guard | 76 | tuned Origami | 98.46 | 59.16 | 148.31 | 100.31 | 89.42 | 102.49 | 103.07 | 100.63 | 94.66 | 99.35 | 100.89 |
| v3 guard | 76 | stock Origami | 97.70 | 57.68 | 152.30 | 104.95 | 88.82 | 99.15 | 100.16 | 104.87 | 95.60 | 97.64 | 98.54 |
| v4 3-bucket | 82 | tuned Origami | 96.46 | 59.50 | 140.25 | 96.77 | 89.99 | 99.42 | 100.80 | 95.88 | 93.27 | 97.36 | 98.81 |
| v5 traps | 61 | tuned Origami | 99.92 | 59.20 | 157.01 | 100.28 | 90.48 | 105.41 | 104.45 | 99.17 | 97.18 | 100.54 | 102.14 |
| v5 traps | 61 | stock Origami | 98.81 | 55.13 | 163.26 | 105.52 | 87.24 | 102.13 | 103.05 | 105.05 | 96.82 | 98.50 | 99.97 |
| v6 global | 58 | tuned Origami | 100.05 | 58.48 | 162.65 | 100.59 | 89.95 | 106.08 | 104.43 | 98.38 | 97.19 | 100.77 | 102.44 |
| v6 global | 58 | stock Origami | 99.59 | 55.15 | 162.30 | 104.80 | 87.27 | 103.99 | 105.00 | 106.36 | 97.84 | 99.10 | 100.63 |
| v7 time | 45 | tuned Origami | 100.58 | 58.44 | 169.26 | 101.07 | 89.32 | 107.39 | 105.60 | 98.82 | 97.67 | 101.58 | 102.74 |
| v7 time | 45 | stock Origami | 99.15 | 54.19 | 168.63 | 104.95 | 86.61 | 103.73 | 103.85 | 105.98 | 97.64 | 98.70 | 99.87 |
| hybrid_slim | 58+120 | both, size-gated | 100.66 | 81.35 | 118.72 | 103.85 | 102.66 | 99.73 | 95.49 | 103.53 | 102.67 | 100.11 | 98.79 |

Shape counts: tiny 337, small 399, medium 504, large 260 | gemv 87, skinny 452, rect 496, square 465.

## Why the GridBased column is empty below the first row

The GridBased selector is nearest-neighbour interpolation over 9,680 **measured**
reference points (logic element `[7]`). Those points are measurements of specific
kernels, so the table is welded to the pool it was built from — pointing it at v6 would
mean re-benchmarking 9,680 shapes against v6's 58 kernels. Origami predicts
analytically and needs no table, which is why it is the only portable selector here.

## Reading the rows

- **`small` (399 shapes) is where everything fails.** Every Origami row sits at 87–92%,
  worse than the Grid pool under Origami itself. No catalog moved it. Only
  `hybrid_slim` clears it (102.66%), by handing those shapes back to GridBased —
  evidence that `small` is a *selector* failure, not a catalog one.
- **SK3 v1 and v6 are near-opposites.** SK3 owns tiny (110.20) and gemv (108.42) and
  loses large (94.03); v6 owns medium (106.08) and large (104.43) and gives up gemv.
  The v2 union was built to get both and got neither (92.13 tiny, 98.58 large).
- **Stock Origami is consistently better on tiny/gemv.** v6: tiny 104.80 stock vs
  100.59 tuned, gemv 106.36 vs 98.38; same on v3/v5/v7. The fitted parameters help
  medium/large and hurt low-parallelism.
- **`hybrid_slim` is flat, not spiky** — 95.5–103.9 across every column where v6 spans
  89.9–106.1. It wins the aggregate by having no hole rather than by excelling; the
  worst-decile column is where that shows up numerically.

## Caveat: this table is protocol-dependent

These are **tiered** iteration counts, which favour small catalogs: G0 carries the
largest library (298 solutions, 9,680 points) and so pays the most one-time
initialisation, which a tiered harness charges to it. Measured in-session with an
iteration floor:

| | tiered | amortised |
|---|---|---|
| v6 / G0 | 99.95% [99.1, 100.8] | **94.86%** [94.2, 95.5] |
| hybrid_slim / G0 | 100.63% [100.3, 101.0] | **100.20%** [99.9, 100.5] |

So compare the v2–v7 rows **to each other** freely; their standing against the G0 row
depends on the protocol. `hybrid_slim` is the only configuration at or above parity
under both. See `FINAL_CATALOG_REPORT.md` §0a.
