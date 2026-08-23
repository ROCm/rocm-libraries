# Reviewer claims — code paths, status, and the experiments that would settle them

Prepared 2026-08-18 for the StreamK / gfx1100 review.

Updated 2026-08-19 with campaign results.

All source citations are against **TensileLite** and **Origami** in **`~/exp/stock`** — the
tree the benchmark binaries were actually built from, confirmed via
`build/release/CMakeCache.txt`. (An earlier draft cited `~/rocm-libraries`, a checkout that
was never built here; every line number and the CU-count claim were wrong as a result.)
Upstream `shared/tensile` is a different implementation that rejects StreamK on WMMA
outright and is not what runs here.

Line numbers drift. `streamk_presentation/verify.py` re-checks every citation used in the
deck against the live source on each run; if this file and the deck disagree, trust
`verify.py`.

---

## Summary

| # | Claim | Status after the 2026-08-19 campaign |
|---|---|---|
| 1 | Grid predictor not validated on this architecture | **ANSWERED — it is now validated, and it wins.** Nothing beats the shipped default; the legacy alternative is 21% slower. Residual: its provenance is still unjustified in-tree. |
| 2 | StreamK+grid inherits DP+grid's drawbacks at similar density | **CONFIRMED by our own data.** Catalogs agreeing on 0–24% of kernel choices land within 3 points on aggregate throughput. |
| 3 | WGM is predicted by StreamK and may need tweaking | **RETIRED by a grep** — no shipped solution leaves WGM at 0, so the predictor never fires. |
| 4 | Origami tile selection may not be doing well here | **Partially addressed.** Top-1 accuracy is a weak proxy for throughput; the clean oracle test is still un-run. |

**Headline number:** the shipped grid predictor leaves StreamK **inert on 77.5% of 1500
shapes** — measured by reading the launch arguments, not inferred — and that turns out to be
the *correct* decision on 1163 of them.

---

## Claim 1 — the grid predictor

> "The grid predictor hasn't been validated for performance on this architecture, that
> selects the number of workgroups to launch for a streamk kernel."

### Status: open, and the true statement is stronger than the claim

**It is not an opt-in path.** `getSKGrid` (`src/ContractionSolution.cpp:3220-3315`) is an
if/else-if chain over four environment-controlled knobs. The second branch tests
`skDynamicGrid > 0`, and `TENSILE_STREAMK_DYNAMIC_GRID` **defaults to 6**
(`include/Tensile/AMDGPU.hpp:271-273`). So Origami's predictor is the path every StreamK
launch we have ever measured already took. The `skGrid = cuCount` fallback at the end of
the chain is unreachable unless you explicitly export the variable as 0.

**6 is an enum, not a multiplier.** It is cast straight to `origami::grid_selection_t`
(`shared/origami/include/origami/types.hpp:157-167`):

| value | algorithm | uses the per-arch constants table? |
|---|---|---|
| 0 | `number_of_cus` | no — returns `N_CU` |
| 1 | `min_resources` | no |
| 2 | `energy_aware` | no |
| 3 | `reduction_cost_aware` | four fitted coefficients, **no architecture branching, no attribution comment** |
| 4 | `data_parallel` | no |
| 5 | `analytical` | **yes** |
| **6** | **`k_split_aware` ← the default** | **no — reads only `N_CU`** |

**So there is nothing on the default path that could have been tuned for RDNA.** Origami
*does* have a per-architecture constants table and gfx1100 *does* have a real entry in it
(`shared/origami/include/origami/hardware.hpp`, mem perf ratios, `parallel_mi_cu = 2`, and
four WMMA instruction latencies). Mode 5 consults it. Mode 6 does not. `grid_k_split_aware`
(`shared/origami/src/origami/streamk.cpp:328-408`) branches on hardcoded literals —
`MinItersPerCU = 8`, tile fractions `{0, ½, ⅛, ⅕, ¼, ⅓}`, split multipliers
`{16,12,8,6,4,3,2,1}`, a 128 MB workspace cap — with no architecture guard anywhere.

### The CU-count question (not raised by the reviewer)

**Correction.** An earlier draft of this document asserted that our build was on the
un-fixed side of an RDNA CU-count bug. That was wrong, and it was wrong because it cited
`~/rocm-libraries` — a checkout that was **never built here**. The benchmark binaries come
from `~/exp/stock` (confirmed via `build/release/CMakeCache.txt`), and that tree **has the
fix**. `streamk_presentation/config.py` now points at the built tree and `verify.py`'s
citation gate pins every line number to it.

What is actually true is more interesting. The build carries **two different CU counts at
once**, by design:

| | value on a 7900 XTX | used for |
|---|---|---|
| `AMDGPU::computeUnitCount` = `multiProcessorCount` | **48** | `skMaxCUs`, `GRID_MULTIPLIER`, the `DYNAMIC_GRID=0` fallback, `calculateAutoGSU` |
| Origami `hardware.N_CU` = `multiProcessorCount × 2` (`shared/origami/src/origami/hardware.cpp:132`) | **96** | **the shipped StreamK grid budget**, and every `num_active_cus / N_CU` term in the latency model |

The ×2 is deliberate and commented: *"On RDNA, HIP runs in WGP mode by default…
`multiProcessorCount` reports the number of WGPs (2 CUs each) rather than physical CUs.
Origami reasons in physical CUs, so scale the reported count back up."*

**The open question is not "is ×2 a bug". It is whether ×2 is the right budget for a
co-resident launch grid, and nobody has measured it.** A workgroup occupies a whole WGP and
there are 48 of them; the fixup spin-wait requires the grid to be resident. ×1 is arguable
on those grounds, ×2 on physical-CU grounds, and ×4 has no physical meaning but could still
win by oversubscription. All are one environment variable apart.

Measured (2026-08-19, `--iters 1`, 4096×8192×4096, MT 256×128, 1024 tiles):

| `DYNAMIC_GRID` | observed `skGrid` | streams? |
|---|---|---|
| 0 (legacy fallback) | **48** | yes |
| 4 (`data_parallel`) | 1024 | no |
| **6 (shipped default)** | **1024** | **no — identical to data-parallel** |
| 7 (`number_of_cus`) | **96** | yes |

### Experiments

| preset | arms | what it settles |
|---|---|---|
| `claim1-predictor` | `DYNAMIC_GRID` ∈ {6, 0} | the headline: predictor vs a plain `N_CU` grid |
| `claim1-allmodes` | `DYNAMIC_GRID` ∈ {0…6} | which of the seven algorithms is actually best here |
| `claim1-cucount` | `FIXED_GRID` ∈ {24, 48, 96, 192} vs predictor | the 48-vs-96 question, directly |
| `claim1-oversub` | `GRID_MULTIPLIER` ∈ {1,2}, `MAX_CUS` ∈ {0,48} | whether oversubscription helps at all |

### RESULT — measured 2026-08-19, 54,000 runs. **The claim is answered, and the predictor is vindicated.**

Full report: `~/hhs_tn_grid_vs_resource_origami_9k/reports/STREAMK_CAMPAIGN.md`.

**The predictor is now validated on this architecture, and it wins.** Paired geomean vs the
shipped default over 1500 shapes:

| arm | geomean vs default |
|---|---|
| `m0` legacy `skGrid = computeUnitCount = 48` (still upstream's default) | **78.7%** |
| `m1` min_resources | 91.4% |
| `m3` reduction_cost_aware | 92.3% |
| `m4` StreamK off (identical kernels) | 97.0% |
| `m7` number_of_cus = 96 | 92.5% |

Nothing beats mode 6. The legacy default that preceded it is **21% slower**.

**The 77.5%-inert finding is real but is not a defect.** A census (`TENSILE_DB=0x40`, which
prints the packed `skGrid`/`skTiles`/`SKItersPerWG`) shows mode 6 produces a launch that
cannot split any tile on 1163 of 1500 shapes. But on exactly those shapes, forcing
data-parallel changes nothing — **99.11%, median deviation 0.48%** — and on the shapes where
it does stream, turning StreamK off costs **10.1%**. The predictor is right both ways.

**On the CU-multiplier question:** pinning the grid gives 48 → 78.8%, **96 → 92.6%**,
192 → 88.9%, 288 → 85.2%. The ×2 is empirically the right scale. But every fixed grid loses
to the adaptive predictor, because the optimum is shape-dependent — 192 scores 97% on ≥1 ms
shapes and 86% on <0.1 ms ones.

**Residual concern that survives:** the default's *provenance* is still unjustified in-tree.
Commit `947ed1ef87` moved it `3 → 6` in one line with an empty body. It happens to be right
on gfx1100; nothing in the repository records why it was expected to be.

---

## Claim 2 — StreamK + grid

> "I noticed you're using a grid library to validate streamk. Streamk+grid has most of the
> drawbacks of DP+grid, especially if you're using a grid of similar density."

### Status: accepted as a methodology criticism. There is no code defect to fix.

**"Grid" is overloaded and the two meanings are unrelated:**

- **(a) the launch grid** — `skGrid`, how many workgroups the kernel launches with. Decided
  at `solve()` time by `getSKGrid`, *after* a kernel has been chosen. That is claim 1.
- **(b) "GridBased"** — a *distance metric* for picking which kernel to use
  (`include/Tensile/Distance.hpp:370-406`). That is claim 2.

**The metric is stranger than the name suggests:**

```cpp
// Distance.hpp:370  GridBasedDistance
distance = abs(K - gridK);      // K ONLY. M and N are not in the distance at all.
```

M, N and batch are resolved *first*, by three nested `std::lower_bound` calls that snap
**down** to the largest table entry ≤ the problem
(`include/Tensile/PropertyMatching.hpp:1159-1225`). A problem larger than everything in the
table matches the largest row. Only then is |ΔK| minimised within that cell. If the
predicate rejects the winner, the fallback is a reverse **linear scan** of the table
(`:1363-1401`) — not a nearest neighbour in any geometric sense.

**Coupling check: none.** Selection is StreamK-agnostic. The only StreamK-specific
behaviour anywhere in the selection path is the `ExperimentalStreamK` row gate in
`include/Tensile/ExactLogicLibrary.hpp`, which is an on/off toggle
(`TENSILE_SOLUTION_SELECTION_METHOD=2`), not a size predicate. There are no predicates
gating a StreamK solution on minimum K or on a tiles-vs-CU condition. `getSKGrid` runs at
`solve()` time and is blind to which library chose the kernel.

**So the reviewer's point is about validation design, not about a bug.** A dense table
always finds *something* nearby, so the library never returns nothing and the validation
never looks broken — but "close in table coordinates" is not "close in tiles-versus-48",
which is what StreamK efficiency actually turns on. And our table was populated by racing
StreamK kernels against each other, not against the DP alternative.

### What we should concede unprompted

Our headline SK3-vs-G0 comparisons vary **catalog** and **kernel type** at the same time.
They are system comparisons and cannot be read as "StreamK is worth X%". The one clean
measurement we have is the fixed-catalog SK3-vs-SK0 arm
(`reports/linear_sk0_vs_linear_sk3.json`, 1500 shapes):

| tiles @128×128 vs the 48 WGP slots | n | geomean SK3/SK0 |
|---|---|---|
| < ½×48 | 411 | **1.130** |
| ½–1× | 160 | 0.936 |
| 1–2× | 218 | 0.960 |
| 2–4× | 138 | 0.999 |
| 4–16× | 247 | 0.962 |
| > 16× | 326 | 0.972 |
| **all** | **1500** | **1.008** |

That is the number to lead with. StreamK is a wash overall; the gain is entirely in shapes
too small to fill the machine, exactly as the mechanism predicts.

### Experiments

`claim2-density` — same shapes, SK3 catalog vs GridBased catalog. **Read the selection
agreement table, not the geomean.** The harness reports per-shape kernel identity precisely
because two arms can tie on throughput while disagreeing on a third of the shapes.

### RESULT — measured 2026-08-19 (1 rep, deadline-reduced)

vs the SK3 catalog, paired over ~1500 shapes:

| catalog | <0.1 ms | 0.1–1 ms | 1–5 ms | ALL | kernel agreement |
|---|---|---|---|---|---|
| `grid_sk0` (298 sol, all SK0) | 95.2% | 99.6% | **103.3%** | 97.6% | 0% |
| `v6_stock` (58 sol, 22 SK3) | 95.9% | 95.7% | **104.4%** | 96.6% | 24.1% |

**The reviewer's methodology point is confirmed by our own data.** The two catalogs agree
with the SK3 catalog on 0% and 24% of shapes respectively, yet land within 3 points of it on
aggregate throughput. A dense table hides enormous selection divergence behind a similar
mean — which is exactly the failure mode the claim describes.

Note also the band reversal: both alternative catalogs **beat** SK3 above 1 ms and lose below
it. Any single headline number conceals that.

---

## Claim 3 — WGM prediction

> "WGM is predicted by streamk and may need to be tweaked for this architecture."

### Status: RETIRED. Cost: one grep, zero GPU time.

The prediction gate (`src/ContractionSolution.cpp:975-978`) requires **all five** of:

```cpp
sizeMapping.streamK != 0 && skgrid != 0 && sizeMapping.workGroupMapping == 0
  && sizeMapping.workGroupMappingXCC == -1
  && sizeMapping.nonTemporalA < 4 && sizeMapping.nonTemporalB < 4
```

The decisive one is `workGroupMapping == 0`. In the shipped gfx1100 StreamK logic
(`~/exp/logic_src/sk3_prediction.yaml`, 192 solutions, all StreamK mode 3):

| `WorkGroupMapping` | solutions |
|---|---|
| 8 | 168 |
| 4 | 13 |
| 1 | 11 |
| **0** | **0** |

**Not one shipped solution leaves WGM at 0**, so the gate never opens and the logic-file
value wins at `:1039` every time. The predictor the claim is about does not run on anything
we ship. The same holds for the shipv6 HHS and HSS navi31 files.

Two footnotes worth having ready:

- **`TENSILE_STREAMK_DYNAMIC_WGM` is a dead environment variable.** It is parsed into
  `AMDGPU::skDynamicWGM` (`AMDGPU.hpp:278-280`, default 0) and then never read by the gate,
  which tests `workGroupMapping == 0` directly.
- **`TENSILE_FIXED_WGM`** (default `INT_MAX`) *does* override unconditionally at `:1056`.
  That is the knob if we ever want to test WGM sensitivity as a question in its own right —
  which is reasonable, just not the question that was asked.

For completeness, had the predictor fired it would have used
`origami::select_workgroup_mapping`, which has no gfx11/RDNA guard and would start from
`ceil(sqrt(N_CU / NUM_XCD))` with `NUM_XCD = 1`.

### RESULT

**Settled without machine time. The claim does not bite on any shipped kernel.**

---

## Claim 4 — Origami tile selection

> "Finally, the tile selection (the main thing origami is responsible for) may not be doing
> well on this architecture."

### Status: open, supported by our own measurements — but it should be sharpened

**First, scope it. There are two selection paths and the claim only applies to one:**

| path | how tiles are chosen | Origami at runtime? |
|---|---|---|
| **G0 / GridBased** — production | offline, by benchmarking; runtime is a table lookup | **no** |
| **O3 / Prediction** — experimental | `origami::rank_configs` runs **per GEMM invocation** (`include/Tensile/PredictionLibrary.hpp:183`) | **yes** |

**Second, concede it with our own numbers.** From the 56-shape validation panel
(`state/resource_v1_cpp_defaults_validation_metrics.json`):

| metric | value |
|---|---|
| top-1 geomean | **76.0%** |
| median | 87.8% |
| P10 | 66.6% |
| P5 | 36.8% |
| worst shape | 7.3% |
| below 50% | 7.1% of shapes |

**Third, sharpen it — this is the useful part.** The same model given more than one guess:

| guided top-K | geomean |
|---|---|
| top-2 | 90.1% |
| top-4 | 93.2% |
| top-8 | 95.4% |
| top-16 | 96.8% |

The correct tile is nearly always *in* the shortlist. It is the **ranking within the
shortlist** that is unreliable. That is a far more tractable statement than "tile selection
is bad", and it points at a specific remedy — a cheap re-rank over the top-K — rather than
a new model.

**Caveat to state before anyone quotes it:** the headline 97.55% O3-vs-G0 figure is a
*system* comparison (different catalogs, SK3 vs SK0) and is not a tile-selector score.

### Experiments

`claim4-tiles` — same grid policy on both arms, so any difference is the tile choice. The
harness reports the macro-tile distribution per arm. The stronger experiment is a top-1 vs
top-K oracle over a fixed catalog, which isolates the ranker from the catalog entirely.

### RESULT — partially addressed

Not the primary target of this campaign, but two relevant observations fell out of it:

* The macro-tile mix is **much wider than assumed** — 62 distinct tiles across the eval set,
  led by `256x128x16` (229 shapes), `128x160x64` (109), `128x128x64` (77). Any analysis that
  assumed a nominal `128x128` tile (including an earlier draft of ours) was mis-specified.
* Catalog choice moves throughput by only ~3 points on aggregate while changing the selected
  kernel on 76–100% of shapes. Selection quality and throughput are **weakly coupled** at
  this granularity, which means top-1 accuracy is a poor proxy for end-to-end value.

The clean top-1-vs-top-K oracle over a fixed catalog remains un-run.

---

## Running any of this

```bash
cd ~/hhs_tn_grid_vs_resource_origami_9k

python harness/streamk_env_ab.py --list-presets
python harness/streamk_env_ab.py --preset claim1-predictor --dry-run     # always first
python harness/streamk_env_ab.py --preset claim1-predictor --reps 5 --min-ms 1.0
python harness/streamk_env_ab.py --preset claim1-allmodes --report-only  # re-analyse
```

### Protocol — non-negotiable, learned the hard way on this rig

1. **One process per arm.** `TENSILE_STREAMK_*` are read into function-local statics
   (`AMDGPU.hpp`), so they latch on first use. You cannot sweep them inside a single bench
   process; a harness that tried would silently measure the first arm N times. This one
   spawns per arm by construction.
2. **Interleave, and rotate the arm order.** The run-to-run floor on this machine is about
   **1 point**, measured by re-running a single arm. Sequential A/B drifts past that.
   Bootstrap CIs resample *shapes* and are blind to run-to-run variation, so they cannot
   substitute.
3. **Confirm the kernel actually ran as StreamK.** `getSKGrid` silently sets
   `skGrid = tiles` — plain data-parallel — when the workspace is short or the iteration
   counts would overflow. An arm you believe is StreamK can run DP end to end and report a
   perfectly plausible number. The harness decodes `_SK<n>_` out of every kernel name and
   prints the mix per arm; **check that table first, before reading any throughput**.
4. **Report in kernel-duration bands.** 92% of the evaluation set is sub-1 ms and several
   conclusions reverse on that lens. `--min-ms 1.0` is the default for resolution; run
   `--min-ms 0.0` too before concluding anything.
5. **Watch library size.** Arms whose libraries differ in size pay different one-time init,
   and the tiered iteration counts charge that to whichever arm is measured shortest. Use
   `--fixed-iters` when arms are not the same library.

### Cheap checks worth doing before booking machine time

Claim 3 was retired by a grep. Before any GPU run, it is worth asking of each remaining
claim: *is there a static fact that would make this moot?* Two that already paid off here
were the `WorkGroupMapping` histogram and the two-tree `N_CU` diff.
