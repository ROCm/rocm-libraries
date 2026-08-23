# gfx1100 fp16 HHS-TN: v6 Prediction catalog vs current develop GridBased

Radeon RX 7900 XTX (gfx1100, 25.75 GB). Run 2026-08-19.
Harness: AIHPBLAS-4345 200-problem TN sweep, converted bf16 → fp16.

| arm | branch | HHS-TN logic | LibraryType | solutions |
|---|---|---|---|---|
| **develop** | `origin/develop` @ `a9b7332a925` | `..._Bias_HAS_SAV_UserArgs.yaml` | **GridBased** | **298** (all StreamK:0) |
| **v6** | `users/vmijovic/gfx1100-hhs-tn-v6-stock-ship` @ `4701dacf824` | `..._Bias_HA_S_SAV_UserArgs.yaml` | **Prediction** | **58** (36 SK0 + 22 SK3) |

Both arms are worktrees of the same repo, built with a **byte-identical**
`--logic-filter 'navi31/*/navi31_Cijk_Alik_Bljk_HHS_*'`, so the HHS-TN logic file is the
only *intended* difference. Provenance in `MANIFEST.json`.

---

## 1. Headline

**Speedup = v6 ÷ develop. >100% means v6 is faster.** 173 problems, ABBA order
(dev₁ v6₁ v6₂ dev₂), adaptive timing.

| metric | value |
|---|---|
| **geomean speedup** | **96.98%** |
| **throughput-weighted** (Σdev_us ÷ Σv6_us) | **106.51%** |
| median | 99.64% |
| p10 / p90 | 68.65% / 111.02% |
| kernel changed | 172 / 173 problems |
| v6 used StreamK (SK3) | 61 problems — develop: **0** |

**The two headline numbers disagree, and that is the result.** v6 is ~3% slower on the
*unweighted* geomean but ~6.5% faster once problems are weighted by the time they actually
take. v6 loses on small/fast problems, which dominate the *count*, and wins on large ones,
which dominate the *clock*. Quoting either number alone would be misleading.

| bucket | n | share |
|---|---|---|
| severe_regression (<75%) | 20 | 11.6% |
| regression (<90%) | 22 | 12.7% |
| loss | 6 | 3.5% |
| **tie (±5.72%)** | **91** | **52.6%** |
| win | 15 | 8.7% |
| strong_win (>110%) | 19 | 11.0% |

## 2. Measured noise floor

Both arms were run twice, so a same-arm ratio must be 1.0 by construction:

| arm | geomean of its two passes | median abs dev | p95 abs dev |
|---|---|---|---|
| develop | 100.20% | 0.24% | 2.48% |
| v6 | 100.11% | 0.16% | 2.86% |

**Tie deadband = ±5.72%** (2× the worse p95), derived rather than assumed. Note this is
much looser than the ~0.5% floor measured under an interleaved protocol on this same
machine — adaptive timing over 173 heterogeneous problems is inherently noisier, so
anything inside ±5.7% here is not a result.

## 3. Where v6 wins and loses

### By kernel duration

| band | n | geomean |
|---|---|---|
| <0.1 ms | 16 | **86.81%** |
| 0.1–1 ms | 29 | **88.44%** |
| 1–5 ms | 23 | **107.61%** |
| ≥5 ms | 105 | 98.90% |

### By category — the actionable view

| category | n | geomean | |
|---|---|---|---|
| 9. Large K | 10 | **113.09%** | ← best |
| 8. Large K | 10 | 105.00% | |
| 13. Very Large GEMMs | 10 | 104.95% | |
| 4. Large M | 8 | 104.44% | |
| 18. Medium batch | 5 | 104.39% | |
| 2. Medium GEMMs | 10 | 102.43% | |
| 10. Large M and N | 10 | 102.15% | |
| 14. M = 1 | 10 | 100.80% | |
| 6. Large N | 10 | 100.62% | |
| 15. N = 1 | 10 | 99.07% | |
| 3. Large GEMMs | 10 | 99.04% | |
| 11. Large N and K | 10 | 98.01% | |
| 12. Large M and K | 10 | 97.71% | |
| 17. Small batch | 10 | 96.64% | |
| 5. Large M | 10 | 93.91% | |
| 7. Large N | 10 | 86.03% | |
| **1. Small GEMMs** | 10 | **82.18%** | |
| **16. K = 1** | 10 | **69.76%** | ← worst |

**The two clear defects are `K = 1` (69.8%) and `Small GEMMs` (82.2%).** Both are
consistent with a 58-kernel distilled catalog having dropped the degenerate/small-shape
kernels that develop's 298 still carries — exactly the trade a distillation makes. They are
also the cheapest thing to fix: adding a handful of small-shape kernels back would not
undo the large-K wins.

## 4. Verification performed

| check | result |
|---|---|
| Both arms built with identical `--logic-filter` | ✅ confirmed in both cmake lines |
| Library `.dat` sha256 differs between arms | ✅ |
| `HIPBLASLT_TENSILE_LIBPATH` per arm, derived not typed | ✅ (`armlib.py`) |
| Observed macro-tiles ⊆ own catalog's MT set | ✅ dev 12/62, v6 13/36 |
| **v6 emits SK3 kernels; develop's catalog contains none** | ✅ **the decisive routing proof** |
| kernel_change_rate | ✅ 172/173 |
| OOM/failed problems identical on both arms | ✅ symmetric |

**Routing proof, in detail.** Full-kernel-name matching against the logic files'
`SolutionNameMin` **does not work here**: the built kernels carry an extra parameter token
(`SKWS`) that the logic YAMLs predate, so every name diverges by a one-token shift. Two
naming-independent checks were used instead — macro-tile-set containment, and the fact that
develop's catalog is 298 solutions with **zero** StreamK:3 while v6 ran SK3 on 61 problems.
develop cannot produce an SK3 kernel, so that is direct evidence each arm loaded its own
catalog.

`TENSILE_PREDICTION_LIB=0` produced *identical* kernels on v6 — which on gfx1201 would void
the experiment, but here is expected and benign: each arm has only its own file for this
ProblemType, with no competing Equality library to fall back to.

## 5. Bug found in the test kit (please pass on)

`scripts/grid_sweep_adaptive.sh` and `scripts/wgm_sweep_adaptive.sh` hardcode
`--compute_type c_f32_r`. That value is valid in the bench **YAML** but is **rejected on the
command line**:

```
Invalid value for --compute_type c_f32_r
```

Both scripts drive the CLI, so on this build **every row of both sweeps comes out `nan`**.
The correct CLI value is `f32_r`. Our copies are patched (via a `COMPUTE_TYPE` default, so
the bf16 path is unchanged); the shipped kit is presumably affected the same way wherever
the bench validates that flag.

## 6. Caveats

- **This is a branch comparison, not a catalog A/B.** The arms differ in catalog *and*
  binary. That is the right question for "should v6 ship", but it means the delta is not
  attributable to the catalog alone.
- **`selection_efficiency` (§6) is within-arm only.** It normalises against each arm's own
  oracle, so a 58-kernel catalog that always picks its own best scores 1.00 while being
  absolutely slower than a 298-kernel catalog at 80%. It must never be compared across arms.
- **27 of 200 problems were dropped**: 5 for device OOM (>25 GB on a 25.75 GB card) and 22
  for host allocation failure (6–15 GB). All failed **identically on both arms**, so the
  measured set stays symmetric.
- **Megagrid (kit step 7) not run** — 262k bf16 problems, multi-day, and it characterises
  grid-table cliffs rather than this comparison.
- The kit's step-5 "force DP" uses `TENSILE_STREAMK_DATA_PARALLEL=1`, which is a **dead
  store** on this tree (overwritten by the `skDynamicGrid > 0` branch). Patched locally to
  `TENSILE_STREAMK_DYNAMIC_GRID=4`; noted so results are comparable to other sites.

## 7. Selection efficiency and grid/WGM triage

Run on a **60-problem stratified subset** (3 per category at the p10/p40/p70 cost percentiles), identical on both arms. The full 195-problem oracle projected to ~36 h (v6 ~6 h, develop ~30 h at 298 candidate kernels per problem vs v6's 58) — beyond the overnight budget. The rank-1 headline in §1 is unaffected: it covers all 173 problems.

### v6 (Prediction, 58 solutions) — WITHIN-ARM ONLY

| metric | value |
|---|---|
| problems compared | 53 / 60 |
| mean efficiency | **84.8%** |
| median efficiency | **94.4%** |
| optimal picks | 3 / 53 |
| below 70% | 9 / 53 |

Worst selections:

| m | n | k | efficiency | category |
|---|---|---|---|---|
| 80 | 80 | 49872 | **5.9%** | 9. Large K, very small M and N |
| 48 | 80 | 620224 | **7.8%** | 9. Large K, very small M and N |
| 128 | 32 | 222368 | **18.5%** | 9. Large K, very small M and N |
| 1392 | 1680 | 319312 | **44.1%** | 8. Large K, smaller M and N |
| 640 | 1020544 | 1 | **51.6%** | 16. K = 1 |
| 2080 | 1033808 | 1 | **51.7%** | 16. K = 1 |

**Read this against §1, not instead of it.** v6's ranker is far from its own oracle on *Large K, very small M and N* (5.9-18.5%), yet §1 shows category 9 is where v6 **beats** develop (113.1%). Both are true: v6 picks a poor kernel from its own 58 and still beats develop's pick. Efficiency measures the ranker; §1 measures the product.

> **Caveat on the generated `heuristic_efficiency_report.md`:** the kit's parser emits a templated report whose header (date, host, branch, `gfx1250`, `bf16tn.yaml`) is hardcoded from the original bf16 run. Only the computed numbers are from this run; the provenance lines in that file are NOT ours. Use `heuristic_efficiency.csv`.

> **Do not compare efficiency across arms.** It normalises against each arm's own oracle, so v6's 58-kernel pool is flattered relative to develop's 298.

### develop (GridBased, 298 solutions) — WITHIN-ARM ONLY

| metric | value |
|---|---|
| problems compared | 53 / 60 |
| mean efficiency | **90.4%** |
| median efficiency | **93.7%** |
| below 70% | 3 / 53 |

Side by side — **with the caveat that these are not directly comparable**, since each normalises against its own oracle and v6's 58-kernel pool is flattered by having fewer ways to be wrong:

| arm | pool | mean | median | below 70% |
|---|---|---|---|---|
| v6 | 58 | 84.8% | 94.4% | 9/53 |
| develop | 298 | 90.4% | 93.7% | 3/53 |

That develop scores **higher** despite the handicap is the informative part: v6's ranker is the weaker of the two at picking from what it has. Its wins in §1 come from the catalog containing better kernels (notably StreamK), not from ranking them better.

### Grid and WGM sweeps (15 worst-efficiency problems per arm)

**Grid: no effect.** Pinning `TENSILE_STREAMK_FIXED_GRID` across {34…792} improved **0 of 15** problems by >5% on either arm. Combined with §1's finding that develop never emits an SK3 kernel, the grid predictor is not a lever on this suite.

> The kit's "force DP" mode uses `TENSILE_STREAMK_DATA_PARALLEL=1`, a dead store on this tree; patched locally to `TENSILE_STREAMK_DYNAMIC_GRID=4`.

**WGM: a real lever, and it is mispredicted.**

| arm | problems improved >5% by a non-default WGM | best cases |
|---|---|---|
| v6 | **3 / 15** | 640×1020544×1 → **143%** at WGM=6; 51904×848×31376 → **128%** at WGM=8 |
| develop | **5 / 15** | 33264×7152×10736 → **186%** at WGM=8; 755776×48×48 → **116%** at WGM=3 |

Up to **1.86×** from changing one launch parameter, on both arms, concentrated in the shapes that already select badly. This is the most actionable finding in the triage and it is **not** specific to v6 — develop is affected more.

---

Artifacts: `results/ab_rank1_compare.{csv,md}`, `results/<arm>/`, `logs/`, `MANIFEST.json`,
`solution_allowlists.json`, `memory_preflight.json`.
