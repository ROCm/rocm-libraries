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

## Measured — and the number depends on how you benchmark

**Read this before quoting a figure.** v6's standing against the shipped selector (G0)
changes by 5 points depending on the benchmark's iteration count, because G0 carries a much
larger library (298 solutions, 9,680 reference points) and therefore a larger **one-time
initialisation cost**. Benchmarks that run few iterations per shape leave that cost
unamortised and charge it to G0.

Three libraries, one session, interleaved, ratios paired within (shape, rep). Only the
iteration counts differ:

| | tiered iterations | amortised (floor 60) |
|---|---|---|
| **v6 / G0** | **99.95%** [99.1, 100.8] | **94.86%** [94.2, 95.5] |
| **hybrid / G0** | **100.63%** [100.3, 101.0] | **100.20%** [99.9, 100.5] |

The mechanism is clearest per band. `v6 / G0` in the 1–5 ms band is **119.73%** under tiered
and **99.93%** under amortised — that entire 20-point advantage is G0 paying init on shapes
the harness gives very few iterations, and it disappears when the cost is amortised.

| band | v6/G0 tiered | v6/G0 amortised |
|---|---|---|
| <0.1 ms | 95.08% | 94.30% |
| 0.1–1 ms | 103.09% | 94.37% |
| 1–5 ms | **119.73%** | **99.93%** |
| ≥5 ms | 96.55% | 99.77% |

**What to conclude.** A long-running application initialises the library once, so the
amortised column is the deployment-relevant one — and there **v6 is ~5% slower than the
shipped selector.** Under the tiered protocol, which is what the original 19-arm campaign
used throughout, v6 is at parity. Neither number is wrong; they answer different questions.

**Ship v6 only if you have measured your own workload's iteration behaviour.** If your calls
are short and frequent with a warm library, expect the amortised column.

**A hybrid configuration is robust to the choice** — see below.

`tuned` above means the Origami heuristics in `fd85b319a36` on this branch; `stock` is
`origin/develop`. v6 works under either: the tuned selector is worth ~0.5% overall.

Report in **kernel-duration bands**, not a single geomean. 92% of this evaluation set runs a
kernel shorter than 1 ms, so an aggregate is really a report about short kernels — and on the
full set selector tuning looks worthless on distilled catalogs while on shapes ≥1 ms it wins
on every catalog tested (+0.75% to +2.74%).

## The configuration that holds up under both protocols

Gating by problem size — nearest-neighbour matching for low-parallelism shapes, Origami
prediction for the rest — measures **100.63% / 100.20%** of G0 (tiered / amortised), the only
configuration tested that is at or above parity under both. It also beats v6 below 0.1 ms by
7–8 points, the only thing in the campaign to beat G0 in that band across 802 shapes.

It is not shipped here because it is not expressible as a single logic file: it requires
merging a Prediction logic and a Matching logic and then setting the row predicates, which
the current tooling does by patching the serialized library post-build. The recipe, the
scripts and the measurements are in the `tensile-tuning` skill runbook
(`references/wiki/05_workflow/catalog_campaign_runbook.md`, §10).

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
- **How the init cost was pinned down.** Same shape, same kernel, forced solution index,
  varying only iterations: at 5 the ratio is 0.868, at 50 it is 0.995, at 1000 it is exactly
  1.000. A one-time cost divided by a small iteration count is indistinguishable from a
  per-call cost, and that is what a tiered harness produces on large shapes. This was
  initially mis-diagnosed three times — as library size, as selection overhead, and as a
  two-row structural cost — before the iteration sweep settled it.
- **Provenance of the G0 comparison.** The G0 device library built for these numbers hashes
  byte-identical to the baseline recorded in the original campaign
  (`5b5bdc3bc6fc8b90…`), so the two protocols are compared against the same artifact, not a
  rebuild.

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
