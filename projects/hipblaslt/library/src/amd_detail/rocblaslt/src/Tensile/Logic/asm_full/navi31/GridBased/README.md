# gfx1100 HHS-TN Origami catalog (v6)

`navi31_Cijk_Alik_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs.yaml` — a 58-entry, Origami-selected
kernel catalog for `Cijk_Alik_Bljk_HHS_BH_Bias_HA_S_SAV` (fp16 in, fp32 accumulate, TN,
batch 1) on gfx1100 / Navi31 / RX 7900 XTX.

**This catalog IS built by default on this branch.** It previously lived under
`navi31/Experimental/`, which `TensileCreateLibrary` skips unless `--experimental` is passed
(`tensilelite/Tensile/TensileCreateLibrary/Run.py:1107`). It now sits in `GridBased/` and the
298-solution `navi31_Cijk_Alik_Bljk_HHS_BH_Bias_HAS_SAV_UserArgs.yaml` has been deleted,
because the two cannot coexist — see "Why the old catalog had to go" below.

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

---

# Why the old catalog had to go

`navi31_Cijk_Alik_Bljk_HHS_BH_Bias_HAS_SAV_UserArgs.yaml` (298 solutions, `[11] = GridBased`,
`[7]` = 9,680 reference points) declares a **byte-identical ProblemType** to this catalog.
Two logic files with the same ProblemType do not coexist:

- Under lazy loading both emit the *same* placeholder device-library file,
  `TensileLibrary_HH_HH_HA_Bias_SAV_UA_Type_HH_HPA_Contraction_l_Alik_Bljk_Cijk_Dijk_gfx1100.dat.zlib`.
- Without lazy loading, `PredicateLibrary.merge`
  (`tensilelite/Tensile/SolutionLibrary.py:360-377`) appends them as two rows sorted by
  `_MATCHING_ORDER` (`Tensile/Properties.py:68-75`), where `PredictionMatching`=2 sorts ahead
  of `GridBasedMatching`=3, and `ExactLogicLibrary::findBestSolution`
  (`tensilelite/include/Tensile/ExactLogicLibrary.hpp:102-156`) returns on the first row that
  yields a non-null solution. Whichever row is first shadows the other completely.

`TensileCreateLibrary` neither errors nor warns on the duplicate. Reverting the commit that
deletes the old file restores the shipped GridBased selector.

# Verification on this branch (2026-08-14)

Clean standalone build of `projects/hipblaslt` from this branch: Release, `GPU_TARGETS=gfx1100`,
`HIPBLASLT_ENABLE_LAZY_LOAD=ON`, client on, **no logic filter** — the source tree alone decides
what is built. Then 1,500 frozen evaluation shapes x 3 reps, runtime selection only, one bench
process at a time. **4,500 rows, `status=ok` on 100%, zero failures.**

## The right catalog is selected — proven

| check | result |
|---|---|
| v6 logic YAML | 58 solution entries (56 unique names; two duplicate pairs, kept as-measured) |
| built HHS-TN device library | **111** kernel objects (58 solutions x GSU variants) |
| same-tree library built from the deleted G0 catalog | **476** kernel objects, 408 of them G0-only |
| selected kernels in the v6 library | **4,500 / 4,500 = 100%** |
| selections from a G0-only kernel | **0** |

Kernel names in the library are compared, not `SolutionNameMin` from the YAML: the runtime
name has auto-valued parameters resolved (`ICIWn1` -> `ICIW0`) and Tensile emits GSU variants,
so a YAML-name comparison reports false failures.

## Reproduction against the previously measured arm

The same configuration (stock Origami + v6) was measured on 2026-08-13 on `eae132fefcf`.
Joining the two ledgers on `shape_id`:

- **Kernel agreement: 1,500 / 1,500 = 100%.** Origami picks the identical kernel on every
  single shape. The wiring reproduces exactly.
- Throughput, paired geomean: **95.84%** [95.43, 96.27]. This missed the pre-registered
  [98%, 102%] band, so it was decomposed rather than accepted:

| contrast | ALL | <0.1ms | 0.1-1ms | 1-5ms | >=5ms |
|---|---|---|---|---|---|
| old build re-run today / old build 2026-08-13 (**session drift**) | 98.44% | 98.00% | 98.55% | 99.88% | 99.97% |
| new build / old build, same session (**build difference**) | 97.12% | 99.52% | 95.36% | 94.49% | 101.91% |
| new / recorded (confounded) | 95.84% | 98.03% | 93.44% | 92.88% | 101.61% |

Neither term is the catalog: kernel choice is identical in all of them.

## The build difference is library size, not the catalog

Three libraries, one binary, one session, interleaved, ratios paired within (shape, rep),
147 shapes >=1 ms x 5 reps. `v6full` is this branch's product library (all navi31 problem
types); `v6only` and `g0` are single-problem-type libraries built from the same tree.

| contrast | tiered iterations | amortised (floor 60) |
|---|---|---|
| `v6full / v6only` | **88.62%** [87.59, 89.59] | **100.50%** [100.44, 100.56] |
| `v6full / g0` | 102.50% [101.99, 103.02] | **100.43%** [99.96, 100.95] — no difference |
| `v6only / g0` | 115.67% [114.28, 117.10] | 99.94% [99.44, 100.42] — no difference |

The 11.4-point `v6full` penalty is **entirely** one-time library initialisation that the
tiered protocol charges to whichever library is larger; it vanishes at an iteration floor of
60. This is the same artifact documented above, now observed from the other side — previously
it flattered v6 because G0 was the big library; here it punishes the full product build
because the single-problem-type comparison libraries are the small ones.

**Under the deployment-relevant protocol, v6 ties G0 on shapes >=1 ms** (100.43%, CI spans
1.0), which reproduces the corrected campaign figure for the 1-5 ms band (99.93%). It does
not contradict the campaign's full-set amortised figure of 94.86%, which is dominated by the
sub-1 ms shapes not measured here.

## What this branch does and does not claim

It claims: the catalog is wired in, a default build uses it, and Origami selects from it and
only it. That is proven above.

It does not claim a performance win. At >=1 ms v6 and G0 are indistinguishable once
iterations are amortised, and the campaign measured v6 **behind** G0 on the full evaluation
set under the same protocol. The reasons to prefer it are catalog size (58 vs 298 solutions,
620K -> 284K of code object, ~70 ms less cold start) rather than steady-state throughput.
