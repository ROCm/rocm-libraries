# Grid pool vs v6 — three selectors

Scoped comparison of the **baseline catalog** (the shipped 298-kernel GridBased pool) and
the **best distilled catalog** (v6, 58 kernels), across the three selectors that can drive
them. Everything is a ratio to the shipped default — Grid pool selected by GridBased
nearest-neighbour matching — which is 100% by definition.

1,500 evaluation shapes × 3 reps per row, runtime selection, tiered iteration counts.
Aggregation: max over reps for GEMV, median otherwise; then geomean of per-shape ratios.

## What the three numbers mean

| column | definition | how to read it |
|---|---|---|
| **geomean** | geometric mean of the per-shape ratios | typical performance. Geometric, not arithmetic: these are ratios, and a 2× gain must cancel a 2× loss rather than outweigh it |
| **P10** | 10th percentile | **90% of shapes are at or above this** — the practical floor |
| **P90** | 90th percentile | **10% of shapes are at or above this** — the practical ceiling |

P10/P90 are the *boundaries* of the worst and best tenth, not averages within them. They
state a limit — "90% of shapes are at least this fast" — while deliberately excluding the
extreme tail, so a handful of pathological shapes cannot set the number. (An average over
the bottom decile answers a different question: how bad the bad cases are on average. Those
figures are in `CATEGORY_DISTRIBUTIONS.md`.)

## Category definitions

From `harness/generate_evaluation_shapes.py`. **Size is by output elements M×N, not FLOPs**
— K does not enter it.

| size | rule | n |
|---|---|---|
| **tiny** | M×N &lt; 256² (65,536) | 337 |
| **small** | 256² ≤ M×N &lt; 1024² (1.05 M) | 399 |
| **medium** | 1024² ≤ M×N &lt; 4096² (16.8 M) | 504 |
| **large** | M×N ≥ 4096² | 260 |

| geometry | rule (lo = min(M,N), hi = max(M,N)) | n |
|---|---|---|
| **gemv** | lo = 1 — a vector, no 2-D tiling possible | 87 |
| **skinny** | hi ≥ 8 × lo — one dimension dominates | 452 |
| **rect** | 2 × lo ≤ hi &lt; 8 × lo — moderately elongated | 496 |
| **square** | hi &lt; 2 × lo — near-equal dimensions | 465 |

The two axes are independent: a shape is placed in one size class and one geometry class,
so every row of the size table and the geometry table covers all 1,500 shapes.

> **v6 cannot be driven by the GridBased selector.** That selector is nearest-neighbour
> interpolation over 9,680 *measured* reference points, so its table is welded to the pool
> it was built from. Using it with v6 would mean re-benchmarking 9,680 shapes against v6's
> 58 kernels. Origami predicts analytically and needs no table — it is the only selector
> here that ports between catalogs.

## Overall — all 1,500 shapes

| catalog | kernels | selector | geomean | P10 | P90 |
|---|---|---|---|---|---|
| **Grid pool** | 298 | **GridBased** | **100.00** | 100.00 | 100.00 |
| Grid pool | 298 | tuned Origami | 95.51 | 76.20 | 113.15 |
| Grid pool | 298 | stock Origami | 92.96 | 71.96 | 111.21 |
| v6 global | 58 | tuned Origami | 100.05 | 72.13 | 135.27 |
| v6 global | 58 | stock Origami | 99.59 | 71.41 | 134.71 |

The shapes below P10 are not spread evenly: for v6 + tuned Origami they are 37% tiny, 49% small, 10% medium, 4% large.
So the global floor is set almost entirely by the small end. The per-size table below is
the honest place to read limits.

## By size — `geomean [P10 / P90]`, % of G0

| catalog | kernels | selector | tiny | small | medium | large |
|---|---|---|---|---|---|---|
| **Grid pool** | 298 | **GridBased** | 100.0  [100 / 100] | 100.0  [100 / 100] | 100.0  [100 / 100] | 100.0  [100 / 100] |
| Grid pool | 298 | tuned Origami | 94.9  [69 / 110] | 92.1  [68 / 116] | 97.3  [80 / 116] | 98.2  [91 / 106] |
| Grid pool | 298 | stock Origami | 92.6  [61 / 111] | 89.3  [57 / 115] | 94.2  [77 / 112] | 96.9  [89 / 106] |
| v6 global | 58 | tuned Origami | 100.6  [66 / 158] | 90.0  [60 / 120] | 106.1  [84 / 140] | 104.4  [86 / 131] |
| v6 global | 58 | stock Origami | 104.8  [68 / 173] | 87.3  [54 / 123] | 104.0  [80 / 139] | 105.0  [88 / 130] |

n: tiny 337, small 399, medium 504, large 260.

## By geometry — `geomean [P10 / P90]`, % of G0

| catalog | kernels | selector | gemv | skinny | rect | square |
|---|---|---|---|---|---|---|
| **Grid pool** | 298 | **GridBased** | 100.0  [100 / 100] | 100.0  [100 / 100] | 100.0  [100 / 100] | 100.0  [100 / 100] |
| Grid pool | 298 | tuned Origami | 93.9  [69 / 117] | 94.3  [68 / 121] | 96.2  [78 / 111] | 96.2  [85 / 109] |
| Grid pool | 298 | stock Origami | 93.4  [63 / 115] | 92.2  [67 / 117] | 92.9  [73 / 110] | 93.7  [81 / 108] |
| v6 global | 58 | tuned Origami | 98.4  [68 / 153] | 97.2  [67 / 142] | 100.8  [79 / 129] | 102.4  [78 / 133] |
| v6 global | 58 | stock Origami | 106.4  [77 / 158] | 97.8  [64 / 144] | 99.1  [77 / 130] | 100.6  [74 / 133] |

n: gemv 87, skinny 452, rect 496, square 465.

## Reading it

- **Neither Origami row beats the baseline overall.** On the same 298-kernel pool, swapping
  GridBased matching for Origami costs 4.5 points tuned, 7.0 stock.
- **Distilling to v6 recovers the average** — 100.05 tuned, 99.59 stock — but unevenly: it
  gains on medium and large and stays ~10 points down on `small`.
- **The floor does not recover.** Every Origami row has a P10 far below the baseline's 100,
  and distilling does not lift it. A smaller catalog leaves the selector nothing
  near-optimal to fall back on when it misjudges a shape.
- **Tuned vs stock flips by regime.** Tuned wins medium and large; stock wins tiny and gemv
  on both catalogs. The fitted parameters help where parallelism is ample.
- **Kernel count.** 298 → 58 is a 5× smaller library and ~70 ms less process cold start
  (240 ms → 171 ms). Per-call selection cost was not measurably different between
  selectors, but see the caveat below — that measurement has a known weakness.

## Caveats

**Protocol.** Tiered iteration counts favour small catalogs: the baseline carries the
largest library and pays the most one-time initialisation, which a short benchmark charges
to it. Measured in-session with an iteration floor, v6/G0 falls from 99.95% to **94.86%**.
Compare rows *to each other* freely; their standing against the baseline row moves ~5
points with the protocol.

**Noise floor ~1 point.** A repeat of an identical arm moved 0.82 points, and a bootstrap
CI called it significantly slower than itself — it resamples shapes and cannot see
run-to-run variation. Treat differences under ~1.5 points as not real.

**Per-call selection cost is not reliably measured.** The benchmark queries the heuristic
once and reuses the algorithm for every timed iteration, so the near-zero per-call figure
describes reuse, not the cost of a fresh query. Cold-start numbers are unaffected.
