# HSS-TN StreamK catalog — results (gfx1100, stock Origami)

**Framing.** HSS-TN is the pipeline **shakedown**, not a headline. Production ships a
**3-solution** library here with an 11-point exact table, so beating it is close to
unfalsifiable and its numbers must not lead any summary. What this target establishes is that
the method works end to end: cross-dtype transplant, correctness, offline/runtime parity,
oracle, distillation, and measurement.

## Setup

| | |
|---|---|
| Selector | **stock** Origami (no fitted weights; tuned API absent from the interpreter) |
| Baseline `g0` | production `navi31_Cijk_Alik_Bljk_HSS_BH_Bias_HAS_SAV_UserArgs.yaml`, 3 solutions, 11 exact points |
| Pool | 775 kernels, transplanted from HHS-TN recipes under an HSS ProblemType |
| Evaluation | 1,511 frozen shapes: **11 on-table**, **1,500 off-table**, 115 strata |
| Fitting panels | 150 + 110 shapes, disjoint from the evaluation set (asserted in code) |
| Bench | build-tree `hipblaslt-bench`, `env -u LD_LIBRARY_PATH`, interleaved arms, 3 reps |

**Noise floor, measured in-session** by registering the *same* library as two arms:
geomean **1.0031**, P10 0.986, P90 1.023, median rep spread 1.7%. No aggregate contrast below
~0.3% is meaningful; no single-shape claim below ~2% is.

## Gates

| gate | criterion | result |
|---|---|---|
| K1 | <30% transplant survival → stop | **PASS** — 775 legal kernels from 3 native |
| K2 | any member fails `--verify`; >5% → halt | **PASS** — 775/775 correct on 12 non-square shapes, worst `norm_error` 9.89e-6 vs 1e-4 |
| K3 | >2% offline/runtime top-1 disagreement → halt | **PASS** — **100%** agreement, 190/190 shapes |
| K5 | HSS must beat production by ≥10% | **PASS** — +17.7% geomean |
| K6 | catalog P10 below production's P10 → do not ship | **PASS** (literal) — P10 53.2 vs 40.6 GFLOP/s; only `small` is 1.0% lower, inside the noise floor |

K3 initially failed at 92.6%. The cause was not the model: the offline `config_t` carried
tensile params, GRVW and vector widths that the runtime's deserializer never sets, and had both
workspace sizes at 0 instead of `SIZE_MAX`. Details in the runbook.

## Results, evaluation set (1,511 shapes), ratio to production

| catalog | kernels | geomean | P10 | P90 | wins | losses | medium | large | small | tiny |
|---|---|---|---|---|---|---|---|---|---|---|
| v1 identity collapse | 105 | 1.1161 | 0.781 | 1.696 | 950 | 501 | 0.9605 | 1.0356 | 1.1582 | 1.3773 |
| v2 subset search | 85 | 1.1525 | 0.812 | 1.754 | 1027 | 417 | 0.9866 | 1.0673 | 1.1951 | 1.4358 |
| **v3 tier-balanced** | **72** | **1.1767** | **0.862** | **1.784** | **1079** | **363** | **1.0297** | **1.1092** | **1.2017** | **1.4302** |

P10/P90 are **percentiles of the per-shape ratio**, not means of a decile. Category-weighted
tails for v3: P10 **0.898**, P90 **1.651**.

Quality improved monotonically while the catalog **shrank** 105 → 85 → 72. Nothing was added at
any step; every gain came from removing kernels.

### On-table vs off-table

| stratum | n | geomean |
|---|---|---|
| off_table | 1,500 | **1.1784** |
| on_table | 11 | **0.9622** |

Production wins its own exact table, as expected — a Prediction catalog sets `[7]` to null and
discards those measured points. Worth stating plainly: **all 11 on-table points are cubes
(M = N = K)**, so production's shipped exact answers cover only square shapes.

### Where the win comes from

`tiny` 1.430, `gemv` ~1.39, `deep` 1.16, rising to **5.4x** on shapes like 4x31x9000 and
769x1x8192 — the split-K regime. StreamK kernels are the outright coverage winner on 39% of
oracle shapes and are concentrated there.

## The mechanism (the transferable part)

The runtime's Prediction library builds its `config_t` from **ten fields only** and never calls
`set_tensile_params`. It therefore **cannot see StreamK, GSU, or any scheduling parameter**.
Consequences:

1. 775 pool kernels collapse to **105 distinct model-identities**. Members of an identity are
   indistinguishable to the selector; which one wins is decided by **position in the solution
   table**.
2. "Make the catalog StreamK-heavy" cannot be done by selection. **You make StreamK the sole
   occupant of its identity.** v3 keeps 48 StreamK kernels of 72 (67%).
3. Therefore the design lever is **removal**, not addition. Deleting a kernel the selector
   wrongly prefers promotes its next choice, and because ranking depends only on identity
   fields, deletion cannot change *which identity* is picked — only what is found there.

Oracle (150 shapes): selection/coverage **0.814**; identity collapse alone is
**+6.45% geomean for free** and moves P10 from 0.641 to 0.753 — it buys the *tail*, unlike
ordinary distillation which sells it.

## Honest limitations

- **24% of shapes (363/1,511) still regress**, worst 0.386 (31x6145x31). The mean win is real
  and so is this tail.
- Fitting-panel ratios are **not** comparable to evaluation ratios: the catalog side comes from
  the oracle matrix and the baseline from the A/B harness, at different iteration counts. The
  panel objective read 1.77 / 1.58 where the honest evaluation contrast is 1.18. The panel is a
  valid *ranking* device only, because the baseline is a per-shape constant across subsets.
- The panel is also **not distribution-matched** to the evaluation set, even within a tier: it
  reported `medium` at 1.34 where the evaluation set gave 1.03.
- 56 of 105 identities were never selected on the first panel, so their member choice was a
  panel-wide fallback rather than a measured one.
- A 3-kernel baseline is a weak opponent. **These numbers must not be presented as a selector
  victory.**
