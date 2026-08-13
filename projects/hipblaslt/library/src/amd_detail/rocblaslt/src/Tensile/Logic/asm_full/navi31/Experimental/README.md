# Experimental: gfx1100 HHS-TN Origami catalog (v6)

`navi31_Cijk_Alik_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs.yaml` — a 58-entry, Origami-selected
kernel catalog for `Cijk_Alik_Bljk_HHS_BH_Bias_HA_S_SAV` (fp16 in, fp32 accumulate, TN,
batch 1) on gfx1100 / Navi31 / RX 7900 XTX.

**This directory is not built by default.** `TensileCreateLibrary` skips any path component
named `experimental` unless `--experimental` is passed, so adding this file does not change
the behaviour of a normal build.

## What it is

A distillation of the union of the production GridBased pool and a StreamK=3 pool, down to
58 entries (36 SK0 + 22 SK3). Selection at runtime is by **Origami**, via logic element
`[11] = 'Prediction'` with element `[7]` (ExactLogic) **null**.

> **If `[7]` is not null, matching answers every query and Origami is never consulted.**
> This is the single most common way to convert a catalog to Origami selection and get no
> Origami at all.

## Measured

19 benchmark arms, 1,500 evaluation shapes × 3 reps each, 122,248 timed measurements, zero
failed rows. Baseline **G0** is the shipped `GridBased` selector (298 solutions, 9,680
reference points).

| | vs G0 (all) | <0.1 ms | 0.1–1 ms | 1–5 ms | ≥5 ms |
|---|---|---|---|---|---|
| v6, tuned Origami | 100.05% | 95.77% | 102.91% | **117.34%** | 94.29% |
| v6, stock Origami | 99.59% | 95.68% | 102.03% | 116.13% | 94.13% |

A full pass over all 1,500 shapes takes **627 ms under v6 against 660 ms under G0** — 5.3%
less GPU time from a catalog 5× smaller. The advantage is concentrated, not diffuse: it is
almost entirely the 1–5 ms band, which carries 45.6% of total GPU time.

`tuned` here means the Origami heuristics in `fd85b319a36` on this branch; `stock` is
`origin/develop`. **v6 works well under either** — the tuned selector is worth ~0.5% overall
and ~1% in the 1–5 ms band, so shipping against a stock runtime is viable.

Report numbers in **kernel-duration bands**, not as a single geomean: 92% of this evaluation
set runs a kernel shorter than 1 ms, and an aggregate is therefore a report about short
kernels. On the full set, selector tuning looks worthless on distilled catalogs; restricted
to shapes ≥1 ms it wins on every catalog tested (+0.75% to +2.74%).

## Caveats

- **Noise floor is ~1 point.** A repeat of an identical arm moved 0.82 points, and the
  bootstrap CI declared it "significantly slower than itself" — it resamples shapes and
  cannot see run-to-run variation. Treat differences under ~1.5 points as not real. An
  interleaved comparison (all arms per shape, back-to-back) gets this to ~0.5.
- **v5 and v6 are equivalent**, interleaved: 99.88%, CI [99.64, 100.12]. v6 is shipped
  because it is smaller (58 entries vs 61), not because it is faster.
- **58 entries, 56 unique.** Two pairs are parameter-identical duplicates (indices 3/4 and
  37/38). Harmless — Origami ties on them and the device library deduplicates downstream —
  and this file is exactly the artifact the numbers above were measured on. Deduplicating to
  56 should be behaviour-neutral but has not been measured.
- **Regimes this does not fix.** `small` sits at ~90% across four successive catalogs, and
  every arm loses to G0 at ≥5 ms. Both are selector-quality limits, not catalog coverage: an
  oracle over the pool shows coverage flat at 0.92–0.96 while selection collapses from 0.96
  (large) to 0.74 (tiny).
- **One open question.** A gated prediction/matching hybrid measured badly and was written
  off, then the loss was traced to *one-time library initialisation* amortised over the
  benchmark's iteration count — at 1000 iterations the penalty is exactly 1.000. Re-measured
  with amortising iterations the hybrid is slightly *faster* than v6, and it is the only
  configuration that beat G0 below 0.1 ms (102.16%). **Not confirmed by a full-set run**, so
  v6 remains the recommendation, but do not treat "hybrid not viable" as settled.

## Companion artifact

`shared/origami/models/linear22_frozen.json` — the frozen 22-feature linear ranker whose
weights `FixedLinearArbiterLibrary` / `FixedLinearCatalogLibrary` (added in `81ca85e44bb`)
deserialize. Those classes declare `std::vector<double> weights` and do not hardcode them,
so without that file the code on this branch has no model to run.

It was used **offline** here, never at runtime: it re-ranks Origami's top 3 during catalog
construction. That restriction is deliberate and measured — the same model is **+2.23% as an
arbiter** over a shortlist and **−18.8% as a ranker** over a whole pool.

## Method

Full runbook, including the pitfalls and the corrections, lives in the `tensile-tuning`
skill: `references/wiki/05_workflow/catalog_campaign_runbook.md`. If you are porting this to
another SKU, the highest-value advice is to **run the oracle pass before building any
catalog** — it distinguishes a coverage problem from a selection problem, and those need
opposite responses. Here it would have saved two failed catalogs.
