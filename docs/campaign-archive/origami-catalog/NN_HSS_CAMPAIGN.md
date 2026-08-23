# NN/HSS catalog campaign — stock Origami, StreamK catalogs (gfx1100)

Companion campaign to the HHS-TN v6 work. Three new problem types were requested; **two are
complete**, one remains. The selector is **stock Origami throughout** — no fitted weights, and
the tuned entry points are physically absent from the interpreter used for ranking.

| document | contents |
|---|---|
| [CATALOG_SELECTOR_MATRIX_NN_HSS.md](CATALOG_SELECTOR_MATRIX_NN_HSS.md) | catalog x selector matrix: geomean overall, by geometry, by size |
| [HSS_TN_RESULTS.md](HSS_TN_RESULTS.md) | HSS-TN, full detail and gates |
| [HHS_NN_RESULTS.md](HHS_NN_RESULTS.md) | HHS-NN, full detail and gates |
| [INTEGRATION.md](INTEGRATION.md) | what was merged into the ship branch, and what was not |

## Headline

| target | production baseline | best catalog | geomean | P10 | off-table | verdict |
|---|---|---|---:|---:|---:|---|
| **HSS-TN** | 3 solutions, 11 exact pts | 72 kernels | **1.177** | 0.862 | 1.178 | **integrate** |
| **HHS-NN** | 70 solutions, 471 exact pts | 72 kernels | **1.029** | 0.853 | 1.031 | **do not ship** (K6) |
| HSS-NN | 6 solutions, 11 exact pts | — | — | — | — | not started |

Noise floor, measured in-session by registering the same library as two arms: **1.0031** (HSS-TN)
and **1.0025** (HHS-NN), P10 ~0.975–0.986. Nothing below ~0.3% is a result.

> **These ratios are per-ProblemType, not product-level.** Every arm was measured against a
> library holding only the target's ProblemType. In a full product library the HSS-TN catalog is
> reached **only when ScaleAlphaVec is requested** — a plain GEMM, even with `--bias_vector`, is
> served by a separate 4-solution non-SAV library that this work does not touch. The baseline it
> replaces served exactly the same path, so the comparison is sound; the scope is narrower than
> the number suggests. See [INTEGRATION.md](INTEGRATION.md).

## The finding that governs everything else

**The runtime selector cannot see StreamK.** `Tensile/Serialization/PredictionLibrary.hpp`
builds its `config_t` from exactly ten fields — macro tile, matrix instruction, occupancy,
workgroup mapping, cache hints, subtile, hand-optimised-main-loop, and two workspace sizes — and
**never calls `set_tensile_params`**. There is no `stream_k`, no GSU, no scheduling state.

Three consequences:

1. A 775-kernel pool presents only **105 distinguishable choices**. Members of one identity are
   invisible to each other; which one wins is decided by **position in the solution table**.
2. "Make the catalog StreamK-heavy" is not achievable by selection. **You make StreamK the sole
   occupant of its identity.** The integrated HSS-TN catalog is 67% StreamK by construction.
3. The design lever is **removal, not addition**. Deleting a kernel the selector wrongly prefers
   promotes its next choice, and cannot change *which identity* is picked.

Every catalog improvement in this campaign came from removing kernels. Quality rose
monotonically as the catalog shrank — 105 → 85 → 72 (HSS-TN) and 100 → 72 (HHS-NN) — improving
the geomean **and** the P10 tail at every step. Nothing was ever added.

## Gain is set by the baseline, not by the method

The same pipeline, selector and distillation gave +17.7% and +2.9%. The difference is the
opponent, not the technique:

- **HSS-TN** production ships **3 solutions and 11 exact points** (all cubes, M=N=K). Beating it
  is close to unfalsifiable; its number must never be quoted as a selector victory.
- **HHS-NN** production ships **70 solutions and a 471-point exact table**. A `Prediction`
  catalog sets `[7]` to null and discards all of it, giving away exactly the shapes where a
  specialised pool would otherwise win.

Correspondingly, **identity collapse alone wins on HSS-TN (+11.6%) and loses on HHS-NN
(0.9929)**. Only restriction crosses parity against a strong baseline.

## Gates

| gate | HSS-TN | HHS-NN |
|---|---|---|
| K1 transplant survival | PASS (775 from 3 native) | PASS (669 usable) |
| K2 every member `--verify` | PASS 775/775 | **TRIPPED 10.7%**, resolved by dropping a named family |
| K3 offline/runtime parity | PASS **100%** (190/190) | PASS 99.47% raw, 100% effective |
| K4 / K5 beat production | PASS +17.7% | PASS +3.10% off-table (bar 0.5%) |
| K6 tail not sold for the mean | PASS | **FAIL** — P90 60,836 → 54,304 GFLOP/s |

## Two silent failures worth carrying forward

**K3's first run (92.6%) was not a modelling disagreement.** The offline `config_t` carried
tensile params, GRVW and vector widths the runtime never sets, and left both workspace sizes at
0 instead of `SIZE_MAX` (which gates StreamK reduction strategies). Separately, adding
`/opt/rocm/lib` to `LD_LIBRARY_PATH` to satisfy a missing `libomp.so` makes the bench load the
**system** `libhipblaslt` — it runs, prints plausible GFLOP/s, and measures a different library.

**The correctness gate was inverted.** `hipblaslt-bench` prints the literal string `failed` in
the `atol`/`rtol` columns for a failing solution. Parsed to NaN, `norm_error > NaN` is always
false, so every failure passed: the first HHS-NN run reported **0 failures alongside a worst
`norm_error` of 15.28**. Fixed, it found **80 kernels (10.7%) that compute wrong results**, all
with `StoreVectorWidth>1` and `VectorWidthA>1` — valid in TN, wrong in NN, and they assemble,
run and report plausible throughput. Cross-*dtype* transplant was clean (775/775); cross-*layout*
was not.

## Method, reusable

1. `--algo_method all` times the **whole pool in one bench process** per shape, so an oracle over
   775 kernels costs one run per shape (~1.4 s small, ~36 s at 4096³) and a full-pool correctness
   sweep costs 12 runs rather than 9,300.
2. Dump the shape x kernel matrix once; then **catalog subsets can be searched entirely offline**,
   because `rank_configs` is deterministic over ten fields. No device-library build is needed to
   compare candidates.
3. Always carry a **tail guard** — reject any step that raises the mean while lowering P10 — and
   let the frozen evaluation set, never the fitting panel, arbitrate.

Full method, with every script embedded, is in the tensile-tuning skill runbook
(`references/wiki/05_workflow/catalog_campaign_runbook.md`).
