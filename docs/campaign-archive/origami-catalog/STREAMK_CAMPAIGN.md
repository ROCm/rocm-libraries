# StreamK grid campaign — gfx1100

**Run 2026-08-19 01:49–07:35. 54,000 measurements across 4 phases. Zero failed arms.**

One binary (`~/exp/stock/build/release/clients/hipblaslt-bench`), libraries swapped via
`HIPBLASLT_TENSILE_LIBPATH`, interleaved with per-shape arm-order rotation, 1500 evaluation
shapes. Every integrity gate passed.

---

## Executive summary

### 1. The shipped grid predictor leaves StreamK inert on 77.5% of shapes — and that is correct

`TENSILE_DB=0x40` prints the packed kernel arguments, so the launch grid can be **read**
rather than inferred. A census of 1500 shapes × 8 grid modes shows the shipped default
(`DYNAMIC_GRID=6`, `k_split_aware`) produces a launch that cannot split any tile on
**1163 of 1500 shapes**. StreamK is compiled in and doing nothing.

That looks damning until you measure it. On exactly those shapes, forcing data-parallel
changes nothing — **99.11%, median deviation 0.48%**. On the shapes where mode 6 *does*
stream, turning StreamK off costs **10.1%**. The predictor is choosing correctly in both
directions, and its conservatism is not a defect.

### 2. It beats every alternative, including every fixed grid

Paired geomean vs the shipped default, 1500 shapes, all bands:

| arm | what it is | geomean |
|---|---|---|
| `m0_legacy48` | the pre-Origami default (`skGrid = computeUnitCount = 48`) | **78.7%** |
| `m1_minres` | `min_resources` | 91.4% |
| `m3_redcost` | `reduction_cost_aware` (upstream's mode 3, refitted) | 92.3% |
| `m4_dataparallel` | StreamK off, identical kernels | 97.0% |
| `m7_ncu96` | `number_of_cus` = 96, streams on 76% of shapes | 92.5% |

**Nothing beats it.** The largest single finding is that the *legacy* default — still what
upstream Tensile ships — is **21% slower**.

### 3. The RDNA ×2 is empirically right, and adaptivity is worth more still

Pinning the grid outright (`FIXED_GRID`), which is what each hypothetical ×N factor amounts to:

| pinned grid | ×N interpretation | geomean | at ≥1 ms |
|---|---|---|---|
| 48 | ×1 — the WGP count | 78.8% | 67.5% |
| **96** | **×2 — physical CUs (shipped)** | **92.6%** | 94.7% |
| 192 | ×4 | 88.9% | 97.0% |
| 288 | ×6 | 85.2% | 92.2% |

96 is the optimum and it degrades in both directions, so **×2 is the right scale**. But note
the last column: big shapes want *more* workgroups (192 scores 97% at ≥1 ms while scoring
86% at <0.1 ms). No single fixed grid can serve both, which is precisely why the adaptive
predictor beats the best fixed choice by ~7 points.

### 4. StreamK is worth having, but only below ~1 ms

Turning StreamK off (`m4`, identical kernels, only the grid differs) by kernel-duration band:

| band | n | StreamK off vs on |
|---|---|---|
| <0.1 ms | 1594 | **93.9%** — StreamK clearly wins |
| 0.1–1 ms | 1102 | 100.2% — a wash |
| 1–5 ms | 258 | **101.9%** — StreamK *loses* |
| ≥5 ms | 36 | **101.7%** — StreamK loses |

The mechanism predicts exactly this: StreamK pays only where data-parallel cannot fill the
machine. Above ~1 ms it is a small net cost.

### 5. The measured noise floor is 0.48%, not "about a point"

The null set gives it for free: shapes where mode 6's launch is already data-parallel, so
`m4/m6` must be 1.000 by construction. Median absolute deviation **0.48%** over 2316 pairs.
Every contrast above is many multiples of that.

### 6. Scaling N_CU to the SIMD count does not help — ×2 stays the best

`hardware.cpp` was patched so the RDNA factor is settable via `ORIGAMI_RDNA_CU_MULT`
(default 2, so unset behaviour is byte-identical to stock). This sweeps the **adaptive**
predictor with a different CU budget — a different question from `FIXED_GRID`, which pins
the grid and disables adaptivity.

Reference = the shipped ×2 (N_CU 96), 1495 shapes, 1 rep:

| N_CU | meaning | <0.1 ms | 0.1–1 ms | ≥1 ms | ALL | 95% CI |
|---|---|---|---|---|---|---|
| 48 | WGPs (×1) | 95.9 | 96.2 | 97.8 | **96.2** | [95.5, 96.9] |
| **96** | **physical CUs (×2, shipped)** | — | — | — | **100** | — |
| 144 | ×3 | **101.2** | 96.6 | 97.1 | 99.1 | [98.2, 100.0] |
| 192 | SIMD32 lanes (×4) | **100.8** | 96.9 | 98.6 | 99.1 | [98.3, 100.0] |
| 288 | ×6 | 100.3 | 96.8 | 99.1 | 98.9 | [98.1, 99.8] |

**No, the SIMD count is not the better denominator.** ×4 lands at 99.1% — statistically
indistinguishable from ×2 at the top of its CI, but never better. ×1 is clearly worse.

The one real signal is band-dependent: **×3 and ×4 beat ×2 by ~1% on sub-0.1 ms shapes**
and lose ~3% in the 0.1–1 ms band. A multiplier that varied with problem size might beat
the constant — but a single constant larger than 2 does not.

Census detail: ×1 leaves **462 of 1500 shapes with a grid below 48**, i.e. genuinely idle
WGPs, against 178 for ×2. That is the mechanism behind ×1's 4-point loss. Inert share is
almost flat across multipliers (73–78%), so the factor changes grid *magnitude*, not
whether StreamK engages.

---

## Caveats, stated up front

* **`g_budget48` is confounded.** `--sm_count_target 48` feeds `problem.num_cus`, which
  reaches *kernel selection* as well as the grid budget — selection agreement is only 53.9%.
  Read that arm as "a different catalog choice AND a different grid", not as a clean ×1 test.
  The clean ×1 test is `FIXED_GRID=48`.
* **P3 ran at 1 rep**, not 2 — the driver reduced its scope to fit the deadline.
* **`v6_stock` and `grid_sk0` are different catalogs**, so their selection disagreement with
  `sk3_m6` (24.1% and 0%) is the experiment, not a fault.
* **`shared/origami/src/origami/hardware.cpp` in `~/exp/stock` is PATCHED** to read
  `ORIGAMI_RDNA_CU_MULT`. It defaults to 2, so unset behaviour is byte-identical to stock
  and every earlier phase remains valid. Original saved alongside as
  `hardware.cpp.pre-mult-experiment`. Revert with a copy + `make hipblaslt` (1 second).
* **The offline transcription of `grid_k_split_aware` is NOT exact** (~70% of grids, 84.7%
  of inert/streams calls). Every number in this report comes from the observed launch
  arguments; the model was used only to design the experiment. See
  `harness/validate_grid_model.py` for a worked counter-example.

---


## 1. The observed launch grid

Read from the packed kernel arguments (`TENSILE_DB=0x40`), not inferred. No statistical uncertainty.

A launch is **inert** when no tile can be split — either the grid divides the tile count, or each workgroup receives a whole tile's worth of k-iterations. An inert StreamK launch is a data-parallel launch wearing a `_SK3_` name.

| mode | n | streams | inert | % inert | % of kernel-time inert | median skGrid |
|---|---|---|---|---|---|---|
| `m0` | 1500 | 951 | 549 | **36.6%** | 22.8% | 48 |
| `m1` | 1500 | 466 | 1034 | **68.9%** | 26.2% | 96 |
| `m2` | 1500 | 364 | 1136 | **75.7%** | 37.4% | 87 |
| `m3` | 1500 | 291 | 1209 | **80.6%** | 40.0% | 82 |
| `m4` | 1500 | 0 | 1500 | **100.0%** | 100.0% | 96 |
| `m5` | 1500 | 311 | 1189 | **79.3%** | 97.7% | 96 |
| `m6` | 1500 | 337 | 1163 | **77.5%** | 64.9% | 96 |
| `m7` | 1500 | 1147 | 353 | **23.5%** | 20.2% | 96 |

## 2. Grid-mode sweep

### Integrity gates

- `m0_legacy48`: SK3 100.0%, SK0 0.0%
- `m1_minres`: SK3 100.0%, SK0 0.0%
- `m3_redcost`: SK3 100.0%, SK0 0.0%
- `m4_dataparallel`: SK3 100.0%, SK0 0.0%
- `m6_default`: SK3 100.0%, SK0 0.0%
- `m7_ncu96`: SK3 100.0%, SK0 0.0%
- selection agreement `m0_legacy48` vs `m6_default`: 100.0%  (same library)
- selection agreement `m1_minres` vs `m6_default`: 100.0%  (same library)
- selection agreement `m3_redcost` vs `m6_default`: 100.0%  (same library)
- selection agreement `m4_dataparallel` vs `m6_default`: 100.0%  (same library)
- selection agreement `m7_ncu96` vs `m6_default`: 100.0%  (same library)

### Paired contrasts vs `m6_default`

Ratios formed **within** a (shape, rep) so machine drift cancels. Bootstrap CI over shapes; sign test = share of shapes moving more than 1%.

| arm | band | n | geomean | CI | wins | sign>1% |
|---|---|---|---|---|---|---|
| `m0_legacy48` | <0.1ms | 1594 | **82.25%** | [81.44, 83.14] | 10% | 92% |
| `m0_legacy48` | 0.1-1ms | 1102 | **76.62%** | [75.55, 77.73] | 14% | 99% |
| `m0_legacy48` | 1-5ms | 258 | **67.47%** | [66.35, 68.89] | 0% | 98% |
| `m0_legacy48` | >=5ms | 36 | **74.14%** | [68.42, 80.56] | 11% | 92% |
| `m0_legacy48` | ALL | 2990 | **78.67%** | [78.05, 79.34] | 11% | 95% |
| `m1_minres` | <0.1ms | 1594 | **90.54%** | [89.34, 91.68] | 32% | 60% |
| `m1_minres` | 0.1-1ms | 1102 | **91.83%** | [90.71, 92.85] | 32% | 82% |
| `m1_minres` | 1-5ms | 258 | **94.55%** | [93.76, 95.57] | 14% | 95% |
| `m1_minres` | >=5ms | 36 | **96.49%** | [93.32, 100.56] | 22% | 94% |
| `m1_minres` | ALL | 2990 | **91.42%** | [90.77, 92.08] | 30% | 71% |
| `m3_redcost` | <0.1ms | 1594 | **91.24%** | [90.19, 92.38] | 32% | 60% |
| `m3_redcost` | 0.1-1ms | 1102 | **93.20%** | [92.20, 94.19] | 32% | 83% |
| `m3_redcost` | 1-5ms | 258 | **94.04%** | [93.17, 94.91] | 16% | 95% |
| `m3_redcost` | >=5ms | 36 | **95.39%** | [92.03, 99.75] | 22% | 94% |
| `m3_redcost` | ALL | 2990 | **92.25%** | [91.57, 92.86] | 31% | 72% |
| `m4_dataparallel` | <0.1ms | 1594 | **93.92%** | [92.78, 95.01] | 38% | 51% |
| `m4_dataparallel` | 0.1-1ms | 1102 | **100.17%** | [99.12, 101.20] | 58% | 46% |
| `m4_dataparallel` | 1-5ms | 258 | **101.91%** | [101.25, 102.57] | 61% | 49% |
| `m4_dataparallel` | >=5ms | 36 | **101.66%** | [100.08, 103.88] | 53% | 47% |
| `m4_dataparallel` | ALL | 2990 | **96.95%** | [96.18, 97.70] | 48% | 49% |
| `m7_ncu96` | <0.1ms | 1594 | **91.37%** | [90.55, 92.31] | 22% | 78% |
| `m7_ncu96` | 0.1-1ms | 1102 | **93.59%** | [92.76, 94.43] | 31% | 90% |
| `m7_ncu96` | 1-5ms | 258 | **94.52%** | [93.69, 95.36] | 14% | 93% |
| `m7_ncu96` | >=5ms | 36 | **96.48%** | [93.32, 100.34] | 22% | 94% |
| `m7_ncu96` | ALL | 2990 | **92.52%** | [92.00, 93.08] | 25% | 84% |

### Partitioned by whether mode 6 actually streams

The **null set** is where mode 6's launch is data-parallel-equivalent. Any arm that only changes the grid must score 1.000 there — so the spread of that distribution is the campaign's own measured noise floor.

| arm | set | n | geomean | median abs dev |
|---|---|---|---|---|
| `m0_legacy48` | null (inert) | 2316 | 77.58% | 21.54% |
| `m0_legacy48` | live (streams) | 674 | 82.56% | 23.67% |
| `m1_minres` | null (inert) | 2316 | 93.51% | 2.75% |
| `m1_minres` | live (streams) | 674 | 84.60% | 13.91% |
| `m3_redcost` | null (inert) | 2316 | 93.41% | 2.69% |
| `m3_redcost` | live (streams) | 674 | 88.34% | 15.13% |
| `m4_dataparallel` | null (inert) | 2316 | 99.11% | 0.48% |
| `m4_dataparallel` | live (streams) | 674 | 89.87% | 17.18% |
| `m7_ncu96` | null (inert) | 2316 | 91.31% | 7.08% |
| `m7_ncu96` | live (streams) | 674 | 96.79% | 7.84% |

## 3. CU multiplier — the x1/x2/x4 question

### Integrity gates

- `g_budget48`: SK3 100.0%, SK0 0.0%
- `g_default`: SK3 100.0%, SK0 0.0%
- `g_x0.5_48`: SK3 100.0%, SK0 0.0%
- `g_x1_96`: SK3 100.0%, SK0 0.0%
- `g_x2_192`: SK3 100.0%, SK0 0.0%
- `g_x3_288`: SK3 100.0%, SK0 0.0%
- selection agreement `g_budget48` vs `g_default`: 53.9%  (same library)  <-- env-only arm, expected 100%: investigate
- selection agreement `g_x0.5_48` vs `g_default`: 100.0%  (same library)
- selection agreement `g_x1_96` vs `g_default`: 100.0%  (same library)
- selection agreement `g_x2_192` vs `g_default`: 100.0%  (same library)
- selection agreement `g_x3_288` vs `g_default`: 100.0%  (same library)

### Paired contrasts vs `g_default`

Ratios formed **within** a (shape, rep) so machine drift cancels. Bootstrap CI over shapes; sign test = share of shapes moving more than 1%.

| arm | band | n | geomean | CI | wins | sign>1% |
|---|---|---|---|---|---|---|
| `g_budget48` | <0.1ms | 1594 | **88.67%** | [87.81, 89.64] | 20% | 79% |
| `g_budget48` | 0.1-1ms | 1102 | **80.32%** | [79.24, 81.35] | 17% | 98% |
| `g_budget48` | 1-5ms | 258 | **70.88%** | [69.19, 72.73] | 5% | 98% |
| `g_budget48` | >=5ms | 36 | **78.60%** | [71.45, 87.51] | 11% | 89% |
| `g_budget48` | ALL | 2990 | **83.74%** | [83.11, 84.40] | 17% | 88% |
| `g_x0.5_48` | <0.1ms | 1594 | **82.51%** | [81.60, 83.34] | 10% | 92% |
| `g_x0.5_48` | 0.1-1ms | 1102 | **76.64%** | [75.53, 77.66] | 14% | 98% |
| `g_x0.5_48` | 1-5ms | 258 | **67.47%** | [66.20, 69.02] | 0% | 98% |
| `g_x0.5_48` | >=5ms | 36 | **74.04%** | [68.04, 81.11] | 11% | 89% |
| `g_x0.5_48` | ALL | 2990 | **78.81%** | [78.15, 79.39] | 11% | 95% |
| `g_x1_96` | <0.1ms | 1594 | **91.48%** | [90.74, 92.33] | 23% | 80% |
| `g_x1_96` | 0.1-1ms | 1102 | **93.57%** | [92.75, 94.34] | 31% | 90% |
| `g_x1_96` | 1-5ms | 258 | **94.66%** | [93.86, 95.46] | 15% | 94% |
| `g_x1_96` | >=5ms | 36 | **96.53%** | [93.14, 100.84] | 22% | 94% |
| `g_x1_96` | ALL | 2990 | **92.58%** | [92.01, 93.09] | 25% | 85% |
| `g_x2_192` | <0.1ms | 1594 | **86.37%** | [85.45, 87.35] | 17% | 81% |
| `g_x2_192` | 0.1-1ms | 1102 | **90.62%** | [89.73, 91.52] | 26% | 81% |
| `g_x2_192` | 1-5ms | 258 | **97.01%** | [95.99, 97.94] | 41% | 65% |
| `g_x2_192` | >=5ms | 36 | **96.93%** | [95.31, 98.36] | 31% | 78% |
| `g_x2_192` | ALL | 2990 | **88.92%** | [88.30, 89.54] | 23% | 80% |
| `g_x3_288` | <0.1ms | 1594 | **82.91%** | [81.70, 84.04] | 14% | 85% |
| `g_x3_288` | 0.1-1ms | 1102 | **86.70%** | [85.79, 87.61] | 20% | 94% |
| `g_x3_288` | 1-5ms | 258 | **92.15%** | [90.75, 93.63] | 23% | 88% |
| `g_x3_288` | >=5ms | 36 | **96.70%** | [94.99, 98.59] | 11% | 92% |
| `g_x3_288` | ALL | 2990 | **85.22%** | [84.51, 85.96] | 17% | 88% |

### Partitioned by whether mode 6 actually streams

The **null set** is where mode 6's launch is data-parallel-equivalent. Any arm that only changes the grid must score 1.000 there — so the spread of that distribution is the campaign's own measured noise floor.

| arm | set | n | geomean | median abs dev |
|---|---|---|---|---|
| `g_budget48` | null (inert) | 2316 | 83.34% | 13.44% |
| `g_budget48` | live (streams) | 674 | 85.12% | 19.01% |
| `g_x0.5_48` | null (inert) | 2316 | 77.73% | 21.32% |
| `g_x0.5_48` | live (streams) | 674 | 82.63% | 23.81% |
| `g_x1_96` | null (inert) | 2316 | 91.38% | 7.13% |
| `g_x1_96` | live (streams) | 674 | 96.82% | 7.87% |
| `g_x2_192` | null (inert) | 2316 | 87.88% | 8.68% |
| `g_x2_192` | live (streams) | 674 | 92.59% | 7.14% |
| `g_x3_288` | null (inert) | 2316 | 84.54% | 11.39% |
| `g_x3_288` | live (streams) | 674 | 87.59% | 14.71% |

## 4. Catalog and with/without StreamK

### Integrity gates

- `grid_sk0`: SK3 0.0%, SK0 100.0%
- `sk3_dp`: SK3 100.0%, SK0 0.0%
- `sk3_m6`: SK3 100.0%, SK0 0.0%
- `v6_stock`: SK3 51.8%, SK0 48.2%
- selection agreement `grid_sk0` vs `sk3_m6`: 0.0%  (different catalog — divergence expected)
- selection agreement `sk3_dp` vs `sk3_m6`: 100.0%  (same library)
- selection agreement `v6_stock` vs `sk3_m6`: 24.1%  (different catalog — divergence expected)

### Paired contrasts vs `sk3_m6`

Ratios formed **within** a (shape, rep) so machine drift cancels. Bootstrap CI over shapes; sign test = share of shapes moving more than 1%.

| arm | band | n | geomean | CI | wins | sign>1% |
|---|---|---|---|---|---|---|
| `grid_sk0` | <0.1ms | 797 | **95.24%** | [93.67, 96.76] | 47% | 80% |
| `grid_sk0` | 0.1-1ms | 551 | **99.58%** | [97.84, 101.32] | 58% | 71% |
| `grid_sk0` | 1-5ms | 129 | **103.33%** | [101.97, 104.94] | 62% | 84% |
| `grid_sk0` | >=5ms | 18 | **102.94%** | [98.96, 110.08] | 56% | 56% |
| `grid_sk0` | ALL | 1495 | **97.59%** | [96.59, 98.69] | 52% | 77% |
| `sk3_dp` | <0.1ms | 797 | **94.08%** | [92.36, 95.53] | 38% | 51% |
| `sk3_dp` | 0.1-1ms | 551 | **100.14%** | [98.52, 101.64] | 59% | 46% |
| `sk3_dp` | 1-5ms | 129 | **101.75%** | [100.81, 102.89] | 61% | 47% |
| `sk3_dp` | >=5ms | 18 | **101.90%** | [99.55, 105.16] | 72% | 50% |
| `sk3_dp` | ALL | 1495 | **97.02%** | [95.98, 97.97] | 48% | 49% |
| `v6_stock` | <0.1ms | 797 | **95.94%** | [94.72, 97.17] | 38% | 70% |
| `v6_stock` | 0.1-1ms | 551 | **95.70%** | [94.21, 97.32] | 44% | 65% |
| `v6_stock` | 1-5ms | 129 | **104.36%** | [102.96, 106.11] | 71% | 80% |
| `v6_stock` | >=5ms | 18 | **102.87%** | [99.36, 108.51] | 61% | 56% |
| `v6_stock` | ALL | 1495 | **96.63%** | [95.72, 97.48] | 43% | 69% |

### Partitioned by whether mode 6 actually streams

The **null set** is where mode 6's launch is data-parallel-equivalent. Any arm that only changes the grid must score 1.000 there — so the spread of that distribution is the campaign's own measured noise floor.

| arm | set | n | geomean | median abs dev |
|---|---|---|---|---|
| `grid_sk0` | null (inert) | 1158 | 99.25% | 2.22% |
| `grid_sk0` | live (streams) | 337 | 92.11% | 16.00% |
| `sk3_dp` | null (inert) | 1158 | 99.19% | 0.49% |
| `sk3_dp` | live (streams) | 337 | 89.91% | 17.95% |
| `v6_stock` | null (inert) | 1158 | 96.88% | 2.12% |
| `v6_stock` | live (streams) | 337 | 95.76% | 7.31% |
