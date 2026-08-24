# lean-Grid on navi32 and navi33 — ~100 kernels, full grid

Applying the [lean-Grid methodology](https://github.com/../lengrid_plan.md) to the navi31 TN
catalogs this branch ports to navi32 (gfx1101) and navi33 (gfx1102).

**Result: each catalog goes from ~300 kernels to ~100 while keeping navi31's full ~9 700-row
grid, at parity on wall-clock and ~1% on per-shape geomean, on both targets.**

Against what navi32 ships today (73 kernels, 471 grid rows) that is a **20x denser grid** for
**a third of the ported catalog's kernels**.

---

## What changed

| | element | navi32 ships today | after |
|---|---|---|---|
| **GRID** (lookup points) | `[7]` | 471 rows | **9 680–9 870 — navi31's full grid, every row kept** |
| **KERNELS** (solution pool) | `[5]` | 73 | **100–103** (from 298–316) |

**No grid row is ever deleted.** Rows whose kernel is dropped are *rerouted* to their tile's
representative. Asserted in `lean_grid.py`, not checked afterwards: emitted `len(element[7])`
must equal the source row count, every row index must land in the kept set, and the grid keys
must be unchanged. A bug that silently dropped rows would otherwise look exactly like a
successful reduction.

## Measured result

Read **above the ~25 µs dispatch floor** (implied `2MNK/gflops` > 40 µs). Two thirds of the grid
sits at that floor, where a smaller catalog is faster for reasons unrelated to kernel quality —
an all-shapes average reports parity for almost any catalog.

**navi32 @ 60 CU** (n=146 above-floor of 206 shapes, 2 071 measurements):

| arm | geomean | wall-clock |
|---|---|---|
| `full_aa` — A/A noise floor | 99.95% | 99.39% |
| `identity` — null surgery | 99.91% | 99.65% |
| **`lean100`** — 97 kernels | **98.99%** | **99.54%** |
| `rand100` — decoy, same kernel count | 96.14% | 99.89% |

**navi33 @ 32 CU** (n=150 above-floor of 198 shapes, 1 657 measurements):

| arm | geomean | wall-clock |
|---|---|---|
| `full` — navi31 port | 135.87% | **112.34%** |
| **`lean100`** | 134.19% | **111.97%** |
| A/A floor | 99.96% | 99.80% |

Isolating lean against the port: **98.81% geomean / 99.67% wall-clock**.

**AuxH @ 60 CU** (n=115 above-floor of 172 shapes, 1 243 measurements) — the adaptation test,
since the aux epilogue changes the store path:

| arm | geomean | wall-clock |
|---|---|---|
| `full_aa` — A/A floor | 99.75% | 99.70% |
| **`lean100`** — 99 kernels vs 250 | **101.83%** | **100.34%** |

**Verdict across all three measured arm-sets: parity to slightly better, at ~1/3 the kernels.**

| arm-set | lean geomean | lean wall-clock | A/A floor (wall) |
|---|---|---|---|
| navi32 HHS @ 60 CU | 98.99% | 99.54% | 99.39% |
| navi33 HHS @ 32 CU | 98.81% | 99.67% | 99.80% |
| navi32 AuxH @ 60 CU | **101.83%** | **100.34%** | 99.70% |

On HHS: parity on wall-clock (inside the A/A floor), ~1% cost on per-shape geomean (just above
it). On AuxH: measurably *better* than the full catalog on both metrics. Per-stratum, nothing
regresses beyond ~0.7% — skinny_M 99.30/99.78%, skinny_N 101.17/99.63%, gemv 99.70–99.82%,
large 99.24/99.60/99.80%.

The AuxH gain is not a surprise: representatives were chosen from **measurement at 60 CU**,
while every ProblemType's grid inherits the same copied 96-CU picks. Re-basing the choice is
worth more where the original picks fit the target least.

## The controls are what make this credible

| control | requirement | measured |
|---|---|---|
| `identity` vs `full` | ≈ 100, else the tooling is not neutral | 99.65% wall, and **the identical kernel on 207/207 shapes** |
| `rand100` vs `lean100` | random must be worse, else the metric has no power | 2.9 pt worse on geomean; **94.46% vs 97.25% under jackknife** |
| A/A arm | in-session noise floor | 99.39% (navi32), 99.80% (navi33) |

**The decoy nearly fooled the wall-clock metric.** At zero jackknife depth `rand100` reads
99.89% — apparently fine. The top 5 shapes hold 42.8% of all kernel time, and random happened
to do well on those. Drop them and it collapses to 94.46% while lean holds 97.25%. Without the
jackknife I would have reported "the metric has no power".

## Ship gates — all four catalogs, both architectures

| catalog | solutions | kernels | gfx1101 | gfx1102 |
|---|---|---|---|---|
| HHS | 100 | 97 | ✅ `0x46` | ✅ `0x47` |
| BBS | 101 | 98 | ✅ `0x46` | ✅ `0x47` |
| AuxH | 102 | 99 | ✅ `0x46` | ✅ `0x47` |
| AuxB | 103 | 100 | ✅ `0x46` | ✅ `0x47` |

0 assembler errors, 0 `overflowedResources`, solution counts matching exactly what was emitted.

## Post-ship verification

The per-catalog gates above prove each logic file *compiles* in isolation. Three further checks
prove the shipped tree is actually sound:

| check | gfx1101 | gfx1102 |
|---|---|---|
| full-tree device-library build (all 38 logic files + `Equality/`) | 1 364 kernels | 1 183 kernels |
| assembler errors / `overflowedResources` | 0 / 0 | 0 / 0 |
| code objects produced | 54 | 46 |
| lean catalogs present in the built libraries | ✅ | ✅ |

Solution counts reconcile: gfx1101 shows HHS=109 / BBS=113 because `navi32/Equality/` contributes
9 and 12 solutions to the same ProblemType libraries; gfx1102's `Equality/` has no TN files, so it
shows the lean counts exactly (100/101/102/103).

**Numerical correctness: 40/40 PASS, 0 FAIL** (`hipblaslt-bench -v`, 40 shapes spanning every
stratum, lean catalog built for gfx1100). Lean only reroutes to kernels that already existed, so
correctness ought to be inherited — but a reroute can send a shape to a kernel it was never
exercised on, which is worth checking rather than assuming. The benchmark arm-sets measured
speed, not answers.

**Not verified, and not claimable here: gfx1101 and gfx1102 binaries were never executed.** This
machine has only a gfx1100 card. Their correctness rests on running the *identical catalog
content* on gfx1100 plus a clean build for the real targets.

## Two deviations from the published method, both forced by measurement

**1. Representatives must be chosen by measurement on the target, not the source SKU's row
counts.** Done the published way, reroutes measured a **median 0.72×** on this hardware — 88% of
probed rows losing >20%. The mechanism is specific: rows tuned at `PGR=0/PLR=0` get rerouted
onto `PGR=2/PLR=1` kernels. The navi31 pool is **two stacked tuning campaigns** (`SolutionIndex`
0–131 carry unresolved defaults, 132–297 resolved), and **87% of grid rows sit in tiles that mix
them** — so tile identity alone does not imply interchangeability. Choosing representatives from
measurement removes the loss entirely.

**2. Fill the kernel budget tail-first.** Optimising total weighted time left **gemv 8.9%
slower** — invisible in the mean, because gemv has 13 above-floor rows against med's 1 712.
Repairing the worst stratum before optimising the mean took gemv to 1.0001 and cost 0.02 pt of
mean. *Select on the tail, not the mean.*

## Method notes worth reusing

- **`--algo_method all` measures every solution on a shape in one ~1 s call.** That turns "price
  one reroute policy" into "price every possible catalog offline from one sweep" — 451 shapes
  × 298 kernels in 22 minutes, versus a projected 4.4 h for pairwise probing.
- **`--solution_index N` maps to the YAML `SolutionIndex`**, and the `[N]:` prefix in
  `--algo_method all` output is that same index (0..297 once each, plus a trailing duplicate
  which is the winner replay — not a 299th kernel).
- **Instrument noise**: `iters=20` gives median 0.50% / p90 3.03% over 3 repeats. `iters=5` is
  p90 **11.7%** — too noisy to choose between kernels. `iters=60` buys 0.7 pt for 3× the cost.
- **Masked-stream hang rate scales with kernel brevity and mask tightness**: 2.7% at 60 CU,
  6.9% at 32 CU, 17.4% on the aux path (short kernels, many launches). Common-mode across arms,
  so ratios hold; budget for it.

## Traps hit, each of which returned a passing signal while doing nothing

- **A build gate reported "298 kernels, 0 errors" — a clean pass — while never testing the lean
  catalog.** `TensileCreateLibrary` recurses, and the gate directory still held two files from a
  previous campaign; it merged all three (298+73+100 = **471 solutions**, exactly the count in
  the artifact). Every gate now runs in a fresh tree and asserts one YAML before building, and
  reports `solutions=` from the built library rather than an exit code.
- **The adaptation silently rerouted 9 tiles to a *different macro-tile*** — MT256×16, MT224×32,
  MT64×160, MT32×224, all extreme skinny. Few rows (0.04–0.48%) but that is coverage loss, the
  documented catastrophic mode. Fixed for +2/3/4 kernels; every ProblemType now has 100% tile
  coverage.
- **An A/A control reported against the wrong baseline read as a 81.9% "noise floor"** — it was
  the port's effect mislabelled. The report now only pairs an `_aa` arm with the library it
  mirrors.

## Not done, and not claimed

- **BBS and AuxB were not benchmarked.** Their timing is inferred, not measured. The inference
  is better founded than "same family", though: comparing the *reroute maps* — for every grid
  key, which recipe the source uses and which recipe lean routes it to — the adapted catalogs
  agree with the measured HHS map on **9 077/9 680 shared keys (93.8%)**, and at exactly that
  rate for all three:

  | adapted vs measured HHS map | agreement |
  |---|---|
  | BBS | 93.8% |
  | AuxH | 93.8% |
  | AuxB | 93.8% |

  **AuxH is measured and agrees at the same 93.8%**, so the benchmarked arm-set is a direct
  proxy for the adaptation BBS and AuxB rely on. Each catalog reroutes ~40% of its rows to a
  different recipe (HHS 39.9%, BBS 40.0%, AuxH 40.8%, AuxB 40.6%) — near-identical, as expected
  when one reduction is carried across.
- The four ProblemTypes share **one measured grid** (9 680 keys with byte-identical GFlops
  across fp16/bf16 and different epilogues — not four independent measurements), so they are
  treated as one experiment rather than n=4.

## The bigger prize is the re-tune, not the reduction

Incidental to the Phase 0 sweep: **the grid's own kernel choice is a median 30% slower than the
best kernel available for that shape at 60 CU.** That is not a lean effect — navi31's tiles
encode 96-CU thresholds, and tail-wave efficiency at small/skinny shapes is strongly CU-count
dependent (gemv ~0.25, tiny ~0.04 at 96 CU). Lean can neither cause nor cure it.

Lean costs ~1% on geomean. The CU-threshold mismatch is leaving ~30% on the table. That is a
separate measured campaign, deliberately kept out of this one so attribution stays clean.
