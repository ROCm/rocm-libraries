# Measured: a size gate on StreamK is worth ~1.3% of wall-clock on gfx1100 TN HHS

Companion to `GATED_POLICY.md`, which projected this result offline. This is the
measurement: a patched library, a fresh paired run, an A/A control arm.

**Status: FINAL. 1500 shapes, 9000 measurements, zero failures.** The headline was stable
throughout the run — above-gate wall-clock read 101.25 / 101.25 / 101.30 / **101.36** at
n = 400 / 850 / 1300 / 1500.

Patch: `ORIGAMI_MN_GATE` in `shared/origami/src/origami/streamk.cpp`.
Data: `results/gate_full.csv` (9000 rows). Analysis: `gate_analyze.py`. Log: `logs/gate_interim.md`.

---

## Result

Three arms, interleaved, one binary, one library, 2 reps:

| arm | env |
|---|---|
| `gate_off` | `TENSILE_STREAMK_DYNAMIC_GRID=6` (shipped predictor) |
| `gate_on` | `... + ORIGAMI_MN_GATE=2867000` |
| `gate_off_aa` | identical to `gate_off` — **the A/A control** |

Each partition is read against **its own** A/A zero point:

| set | n | A/A geo | A/A tput | ON geo | ON tput | **ON ÷ A/A geo** | **ON ÷ A/A tput** | % of kernel time |
|---|---|---|---|---|---|---|---|---|
| **above gate** | 563 | 100.14% | 100.13% | 102.59% | 101.49% | **102.44%** | **101.36%** | 87.0% |
| below gate (control) | 931 | 100.07% | 100.16% | 99.99% | 100.39% | 99.92% | 100.23% | 13.0% |

Whole suite: **geomean 100.96%, flops-weighted wall-clock 101.34%.**

**The gate is worth +1.36% of wall-clock on the 87% of kernel time it can act on, and
provably nothing on the rest.** Selection agreement is 100.0% and the macro-tile mix is
byte-identical across all three arms — the gate moves the launch grid and never the kernel
choice, which is exactly what the patch is supposed to do.

Zero failures in 9000 measurements.

By duration band (a single geomean hides that this reverses):

| band | n | geomean | tput-wtd | ON wins | % of kernel time |
|---|---|---|---|---|---|
| `<0.1ms` | 796 | 100.13% | 100.76% | 46.0% | 4.3% |
| `0.1-1ms` | 551 | 102.25% | 102.05% | 59.7% | 27.0% |
| `1-5ms` | 129 | 100.63% | 100.96% | 58.1% | 45.0% |
| `>=5ms` | 18 | 100.88% | 101.37% | 55.6% | 23.6% |

The gain is broad rather than concentrated: every band is at or above 100%, and the
largest single contribution comes from `0.1-1ms`, not from the giant shapes.

## Why the A/A arm is the whole reason to believe this

Without it, two errors were live and I made both before it caught them.

**1. The below-gate shapes read ~1% slow early on, and that is not the gate.** At n=400 the
control partition sat at 98.95% — alarming, since the gate *cannot* touch those shapes
(the code returns early only when `M*N >= threshold`). The A/A arm showed the identical
deficit on the identical shapes (99.12%), so it was a property of the small-shape set, not
a side effect. By n=1500 it had decayed to 100.07% — it was small-sample noise all along.
Without an A/A arm, the honest reading would have been "the gate has an unexplained side
effect", and the wrong reading would have been "the gate helps below its own threshold".

**2. A single global A/A correction would have been wrong.** The systematic is ~0.2% above
the gate and was ~0.8% below it. One global number over-corrects the half that needs no
correction. The analyzer reports A/A **per partition** for that reason.

The below-gate row is the load-bearing control: at 99.92% / 100.23% it says the gate does
nothing where the code says it must do nothing. If that row moved, nothing else here
would be interpretable.

## Measurement revises the projection down

| | projected (`GATED_POLICY.md`) | measured |
|---|---|---|
| whole-suite wall-clock | 102.08% | **101.34%** |
| whole-suite geomean | 101.30% | **100.96%** |
| above-gate wall-clock | — | **101.36%** |

The projection recombined two previously-measured arms per shape; this is a fresh run
against a patched library. They agree in direction and order but **the projection was
optimistic**, and the measurement is what should be quoted. Recorded because the projection
is the kind of number that otherwise survives into a summary unchallenged.

## The threshold is a plateau — now MEASURED, not projected

The projection (Aug-19 census) said any cut in `[5e5, 3e6]` lands within 0.03 pt. That was
the most load-bearing unmeasured claim in this report, so it was measured directly: five
arms (`gate_off`, `gate_off_aa`, gates at 1e6 / 2.867e6 / 1e7), 1500 shapes, 2 reps,
**15000 measurements, zero failures**. Data `results/gate_plateau.csv`, analysis
`plateau_analyze.py`.

**Verdict: the plateau holds.** Calibrated whole-suite wall-clock:

| cut | 1e6 | **2.867e6** | 1e7 |
|---|---|---|---|
| calibrated tput | 99.44% | **99.49%** | 99.48% |

Spread **0.05 pt** — the three thresholds are indistinguishable. The threshold is not a
knife-edge fitted to one dataset.

**But the uncalibrated numbers say something else, and they are wrong.** See below.

### Retraction

The earlier claim that 2.867e6 is *conservative*, with more available below it, is
**withdrawn**. Gating the band `1e6 <= M*N < 2.867e6` **costs 1.62 pt on those shapes**
(n=216, 6.6% of kernel time — about 0.11 pt of suite time). StreamK is genuinely earning
its keep there. Lowering the cut does not gain; it loses slightly.

### The measurement nearly produced a manufactured conclusion

| | uncalibrated (÷ A/A) | calibrated |
|---|---|---|
| `g_1e6` | 100.99% | 99.44% |
| `g_2867k` | 101.24% | 99.49% |
| `g_1e7` | 101.34% | 99.48% |
| **spread** | **0.35 pt** | **0.05 pt** |

Uncalibrated, the ranking is monotone — "higher threshold is better". It is an artefact.
The **common partition** (`M*N >= 1e7`, n=352), where all three arms are *definitionally
identical* and must therefore agree, shows a spread of **0.31 pt that is likewise monotone
in the arms' interleave order** (`g_1e6` = arm 3, `g_2867k` = 4, `g_1e7` = 5). Later-measured
arms run warmer and score higher.

0.35 pt of apparent signal against 0.31 pt of measured drift: **essentially the whole
cross-threshold difference was position, not threshold.** Dividing each arm by its own
common-partition value removes it.

An A/A arm cannot catch this — it occupies one fixed slot while the variants occupy others.
Only a partition where the arms are identical by construction can. `plateau_analyze.py`
flags `MONOTONE IN ARM ORDER` automatically, because by eye a monotone sequence reads as
signal.

Note the threshold was fitted on the **Aug-19 census** and evaluated on a **fresh run
against a patched library** — out-of-sample across datasets, not merely across a
train/test split of one.

## Which gate number to quote

The gate has been measured several times and the figures differ. They are not in conflict —
they differ by **protocol and by scope**, and the spread is itself a finding. Quote from
this table, not from a number lifted out of a section.

| # | source | protocol | scope | geomean | wall-clock |
|---|---|---|---|---|---|
| 0 | `GATED_POLICY.md` — **projection, not a measurement** | recombined two prior arms | whole suite | 101.30% | 102.08% |
| 1 | `gate_full` (9000 meas) | tiered iters | whole suite | 100.96% | **101.34%** |
| 2 | `gate_full` (9000 meas) | tiered iters | above-gate shapes only (87.0% of time) | 102.59% | **101.49%** |
| 3 | `ship_test` (12 000 meas) | `--fixed-iters 20` | whole suite, within-catalog | 100.92% | **102.35%** |

Every row is **raw** — no A/A normalisation, no drift correction — so the column is
internally comparable. Normalisation moves them very little, and the corrected variants are
recorded here so neither version can be mistaken for the other:

| row | raw wall-clock | after normalisation | shift |
|---|---|---|---|
| 2 (÷ its A/A arm) | 101.49% | 101.36% | −0.13 pt |
| 3 (linear-in-position drift model) | 102.35% | 102.39% | +0.04 pt |

Row 2's A/A arm measured 100.13% and row 3's 99.89% across the full interleave, so in both
cases the correction is smaller than the effect by an order of magnitude. Earlier drafts of
this table mixed raw and normalised figures between rows; that is fixed, and it is the exact
error most likely to recur when someone adds a row.

**Note the body of this report quotes the normalised figures** — "+1.36% on 87% of kernel
time" is row 2 after A/A division, which is the right number for a claim about the gate
because it removes the measurement's own zero-point error. The table above is raw so the
rows stay comparable across runs with different control designs. If the body and the table
appear to disagree, this is why; both are in the shift table.

**If you need one number: +1.3% to +2.4% of wall-clock, and the range is protocol, not
uncertainty.** Rows 1 and 3 measure the same intervention on the same shapes and differ by
1.0 pt purely because tiered iteration counts give large shapes 5 iterations while
`--fixed-iters 20` gives them 20. The gate's entire benefit lives on large shapes, so
resolving them better makes it look better. Neither is wrong; they answer "what would this
be worth under *this* measurement regime".

Row 0 is a projection and was **optimistic against every subsequent measurement** — the
second time in this campaign that recombining measured arms flattered a configuration that
was never actually run. Do not quote it.

Row 2 is the honest per-shape framing (the gate cannot act below its threshold, so diluting
it across untouched shapes understates the local effect), but row 1 or 3 is what a library
owner cares about.

Not in this table: `gate_plateau`'s calibrated figures (99.44 / 99.49 / 99.48). Those are
ratios to a common-partition baseline used to rank thresholds against each other; they are
not absolute gate effects and must not be read as such.

## SCOPE — the measurement library is not the shipping library

Established 2026-08-22, and it governs how every number above should be read.

`~/exp/stock` is **not** a stock build. Its gfx1100 logic was deliberately replaced:

```
82580dfc726  exp: prune gfx1100 logic to the SK3 Prediction catalog only
```

It carries 192 `StreamK: 3` solutions, which is why 100% of measured kernels are `_SK3_`.
That was the right call for the campaign — StreamK has to be *present* to be studied — but
it is not what a user gets.

What actually ships for navi31, counted across the whole logic tree in
`.../Logic/asm_full/navi31/`:

| | count |
|---|---|
| `StreamK: 0` | **2560** |
| `StreamK: 3` | **22 — all in `Experimental/`, one TN HHS file** |

and `projects/hipblaslt/tasks.py` defaults `experimental=False`, so `Experimental/` is
excluded from a default build. The shipped `GridBased/` TN HHS logic is **596 solutions,
every one `StreamK: 0`**. The NN HHS logic does not mention StreamK at all.

**Consequence: on a default gfx1100 build there are no StreamK kernels for HHS, so neither
`TENSILE_STREAMK_DYNAMIC_GRID` nor `ORIGAMI_MN_GATE` can fire, and the +1.34% is not
realizable as-is.** Shipping this gain requires first shipping StreamK solutions for navi31
— a much larger decision than the gate itself. The gate is a correct answer to "given
StreamK kernels, when should the launch stay data-parallel"; it is not a drop-in win.

This also bounds the earlier grid campaign (`reports/STREAMK_CAMPAIGN.md`), which used the
same library: its "the predictor is vindicated" verdict is a statement about the SK3
catalog, not about default-build behaviour.

Unaffected by this scope limit, because none of it depends on what ships: the `glc`
assembler fix (`UPSTREAM_PR_1_glc.md`), the `TENSILE_STREAMK_TILES` clamp crash
(`UPSTREAM_PR_2_clamp.md`), the geomean-vs-wall-clock metric reversal, and the
inert-partition / position-drift method in `RUNBOOK.md`.

## What this does not establish

- **One shape distribution.** "% of kernel time" is a property of this 1500-shape TN HHS
  suite. A workload of only small GEMMs reverses the conclusion — StreamK genuinely wins
  below 0.1 ms. The finding is that the *metric choice* is load-bearing, not that StreamK
  is bad.
- **One dtype, one orientation, one card.** gfx1100 TN HHS.
- **Not a shipping recommendation yet.** The patch is env-gated and defaults to stock. Making
  this the default is a behaviour change to the shipped predictor and needs review, wider
  dtype/orientation coverage, and a CDNA check — `cu_count` and the tile/CU balance differ
  there, and nothing in this campaign touched it.
- **`>=5ms` is thin** (n=18). Do not quote that band alone.

## Reproduce

```bash
# default is byte-identical to stock: the gate is skipped unless the env var is set
cd ~/exp/stock/build/release && make hipblaslt -j$(nproc)

# confirm the gate moves the grid and only above threshold (reads kernel args directly)
TENSILE_DB=0x40 ./clients/hipblaslt-bench --api_method c -m 4096 -n 4096 -k 1024 \
  --transA T --transB N --a_type f16_r --b_type f16_r --c_type f16_r --d_type f16_r \
  --compute_type f32_r --algo_method heuristic --requested_solution 1 --cold_iters 1 --iters 1 \
  | grep numWorkGroups                       # 183 unset -> 1376 (= tiles) with the gate

cd ~/hhs_tn_grid_vs_resource_origami_9k/harness
python3 streamk_env_ab.py --reps 2 --min-ms 0.0 \
  --bench ~/exp/stock/build/release/clients/hipblaslt-bench \
  --arms "gate_off=$LIB:TENSILE_STREAMK_DYNAMIC_GRID=6" \
         "gate_on=$LIB:TENSILE_STREAMK_DYNAMIC_GRID=6,ORIGAMI_MN_GATE=2867000" \
         "gate_off_aa=$LIB:TENSILE_STREAMK_DYNAMIC_GRID=6" \
  --out ~/sk_modes/results/gate_full.csv

python3 ~/sk_modes/gate_analyze.py ~/sk_modes/results/gate_full.csv
```

`sk_grid`/`sk_tiles` are `-1` in a perf pass — `TENSILE_DB=0x40` is deliberately off there
because dumping kernel args pollutes the timing. Grid movement cannot be read from that
CSV; partition on the gate predicate instead, as `gate_analyze.py` does.
