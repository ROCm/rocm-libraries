# HHS-NN StreamK catalog — results (gfx1100, stock Origami)

**This is the honest headline.** Unlike HSS-TN (3-solution baseline), production here ships
**70 solutions with a 471-point exact table**. It is a strong opponent, and the result is
correspondingly modest.

## Verdict

**Do not ship.** K4 passes; **K6 does not**. A +3.1% mean on the off-table stratum is real, but
it is bought partly out of the top end and the large/medium tails.

| gate | criterion | result |
|---|---|---|
| K2 | any member fails `--verify`; >5% → halt | **TRIPPED at 10.7%**, then resolved — see below |
| K3 | >2% offline/runtime top-1 disagreement | **PASS** — 99.47% raw, 100% effective (the one disagreement is an exact latency tie) |
| K4 | beat production by >2x noise floor **off-table** | **PASS** — +3.10% vs a 0.5% bar |
| K6 | catalog P10 below production's → do not ship on a mean win | **FAIL** — P90 60,836 → 54,304 GFLOP/s; per-tier P10 lower in large, medium and tiny |

Noise floor (same library as two arms, in-session): geomean **1.0025**, P10 0.975, P90 1.036.

## Results — 1,971 evaluation shapes (471 on-table, 1,500 off-table)

| catalog | kernels | geomean | P10 | P90 | wins | losses | off_table | on_table | tiny | small | medium | large |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v1 identity collapse | 100 | **0.9929** | 0.718 | 1.254 | 929 | 855 | 0.9903 | 1.0009 | 1.090 | 1.008 | 0.940 | 0.960 |
| v3 tier-balanced search | 72 | **1.0287** | 0.853 | 1.243 | 994 | 746 | **1.0310** | 1.0217 | 1.123 | 1.038 | 0.978 | 1.005 |

Category-weighted tails (v3): P10 **0.877**, P90 **1.294**. Aggregate throughput **+1.0%**.
Regressions: **746/1,914 shapes (39%)**, worst 0.320 (11x11x8192).

**Identity collapse alone is not enough against a strong baseline.** On HSS-TN it was worth
+6.45% and produced a winning catalog on its own. Here the same step lands at **0.9929 — a net
loss** — and only the *restricted* catalog (72 kernels) gets above parity. The lever is removal,
and the stronger the baseline, the more of it is required.

## K2: 80 kernels that ran fast and computed the wrong answer

Cross-*layout* transplant (TN → NN) produced 80 of 749 members (10.7%) that fail validation on
**every** shape. They separate perfectly on two parameters:

| | StoreVectorWidth | VectorWidthA |
|---|---|---|
| 80 failing | 2 or 4 | 2 or 4 |
| 669 passing | 1 | 1 |

They assemble, launch and report plausible GFLOP/s. Dropped as a named family; the remaining
669 verify clean (worst `norm_error` 9.67e-5) and K3 is unchanged.

This was itself hidden by a bug in the checker: `hipblaslt-bench` prints the string **`failed`**
in the `atol`/`rtol` columns for a failing solution, and comparing `norm_error > NaN` is always
false — so the first run reported **0 failures alongside a worst `norm_error` of 15.28**. Both
fixes are in the runbook.

## Why the gain is small

The oracle explains it. Selection reaches only **78.4%** of pool coverage (P10 0.489), and
identity collapse lifts that to 88.6% — so a perfect selector has ~27% of headroom available.
Almost none of it is convertible against this baseline, because production's 471-point exact
table already answers the shapes where the pool's advantage is largest. The catalog sets
`[7]` to null and gives that up; it wins on-table by only 1.02x.

Where it does win, it is the same regime as every other target: `tiny` 1.123 and deep-K shapes
where StreamK is the coverage winner (SK0 107 / SK3 38) and the selector, blind to StreamK,
picks an SK0 twin unless the SK3 kernel is the sole occupant of its identity.

## What would be needed to ship

1. **Keep production's exact table.** A two-row library (`Prediction` gated, `Matching`
   catch-all) preserves the 471 measured points instead of discarding them. The gate must be
   expressible with existing predicates (`LeadingFree{0,1}SizesGreaterOrEqual`).
2. **Fix the large/medium tail**, which is where P10 and P90 are lost, most likely by excluding
   those regimes from the Prediction row entirely rather than by further pruning.
3. Only then re-measure. The current v3 is not a shippable artifact.
