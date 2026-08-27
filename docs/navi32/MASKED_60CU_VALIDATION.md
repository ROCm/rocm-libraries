# Does the catalog win survive genuine 60-CU execution?

The campaign headline (**+23.9% wall-clock** for the widened TN HHS catalog) was measured with
60-CU **selection** (`--sm_count_target 60`) but 96-CU **execution**, because the CU-masked
stream was recorded as hanging ~37% of runs — too often for a 5 000-run sweep. That left the
central premise of the whole campaign ("benchmarking this card at 60 CUs approximates navi32")
*assumed rather than tested*.

It is now tested. **The win holds.**

## Result

207 shapes (stratified subsample of the 998-shape eval set, all 109 strata represented),
3 arms x 2 reps, **genuine CU-masked execution** — `hipExtStreamCreateWithCUMask`, 30 of 48
WGPs = 60 of 96 CUs, verified live by throughput (3 237 vs 4 958 GFlop/s = 65.3% against an
ideal 62.5%).

| arm | geomean | wall-clock |
|---|---|---|
| `ship_aa` (A/A control) | 100.12% | **99.89%** |
| `wide` (298-solution catalog) | **125.22%** | **122.71%** |

**A/A floor 0.11 pt** — the tightest of any arm comparison in this campaign, so the +22.7% is
roughly 200x the noise.

The jackknife is *favourable*: dropping the largest time consumers **raises** the win
(122.7% -> 124.6% -> 127.6% -> 136.2% after dropping 5/10/25/50), so it is not carried by a
handful of big shapes. The quoted figure is the conservative end.

| by size | wall-clock | | by geometry | wall-clock |
|---|---|---|---|---|
| large (37) | 121.4% | | **gemv (11)** | **246.9%** |
| medium (67) | 124.8% | | skinny (63) | 129.2% |
| small (53) | 126.8% | | rect (67) | 123.5% |
| tiny (49) | 131.1% | | square (65) | 121.1% |

Same shape of result as the 96-CU sweep: gains concentrate on GEMV and small shapes, exactly
where a 471-row nearest-neighbour table serves a distant neighbour.

## Was 96-CU execution a fair proxy? Yes — with a caveat

Restricting both regimes to the **same 206 shapes**:

| regime | geomean | wall-clock | its own A/A |
|---|---|---|---|
| 96-CU execution (the shipped measurement) | 125.6% | 125.7% | 102.1% |
| **60-CU execution (this run)** | **125.7%** | **122.7%** | **99.9%** |
| difference | **+0.1 pt** | **-2.9 pt** | |

**Geomean is identical.** Wall-clock is 2.9 pt lower under real 60-CU execution. The 96-CU
sweep's *own* A/A control sat at **102.1%** — ~2 pt of arm-position drift against this run's
0.11 pt — so for HHS alone, drift is a plausible explanation.

> **It is not the explanation in general.** Once the other three ProblemTypes were measured
> (below), BBS shows a **-3.2 pt** shift against a 96-CU A/A of just **100.18%**. Drift of
> 0.18 pt cannot produce a 3.2 pt shift, so a real regime effect exists independently. See the
> all-four table.

Mechanically the residual makes sense too: **large shapes gain least (121.4%) and dominate
wall-clock**, so any regime change that slightly favours large shapes moves the time-weighted
metric while leaving the per-shape geomean untouched. That is precisely the pattern seen.

**Conclusion: the emulation shortcut was sound, and the shipped +23.9% is if anything
marginally optimistic.** Under genuine 60-CU execution the same comparison gives **+22.7%** on
a stratified subsample. The claim does not need retracting; it needs a ~1 pt haircut and this
citation.

## Correction to the runbook: the hang rate is 2%, not 37%

The runbook says the masked stream is "unusable for long sweeps" on the strength of **3
timeouts in 8 runs**. Over **1 242 masked runs** the real rate is **2.0% (25 timeouts)**, all
recovered by the harness's timeout + `pkill`.

An 8-run sample cannot distinguish 37% from 2%; the 95% CI on 3/8 runs down to about 8%. The
first masked run after an idle GPU *does* reliably hang, which is what an 8-run probe
oversamples.

**So execution fidelity is affordable after all** — a full 998-shape masked sweep costs roughly
3 hours, not the impossible budget assumed. Future campaigns on this card should use it rather
than fall back to selection fidelity.

## Second use: re-testing the WGM null on hardware that can express it

The campaign rejected re-forking `WorkGroupMapping` as a null. But the hypothesis is about
raggedness at **30 WGPs** (30/8 = 3.75), and that sweep ran at **48 WGPs**, where 48/8 = 6.00
divides perfectly. WGM is a host-side runtime argument affecting workgroup scheduling at
*execution*; `--sm_count_target` does not reach it. **The test could not have expressed its own
hypothesis.**

Redone at real 60 CUs — 205 shapes, 4 arms, 2 reps, 1 656 runs, 1.3% hang:

| arm | geomean | wall-clock |
|---|---|---|
| WGM10 | 100.18% | 99.78% |
| WGM6 | 100.24% | 100.35% |
| A/A control | 100.13% | **99.93%** |

**The null holds** — 0.57 pt spread against a 0.07 pt floor, flat at every jackknife depth
(0/5/10/25/50 dropped). The original conclusion was right; the original test was not capable of
establishing it.

The real finding is that **the divisibility intuition is wrong for this parameter**. WGM
reorders workgroup *indices* into column-major supergroups for L2 reuse; raggedness affects
only the final supergroup, not how the machine is tiled. A clean factor of the CU count buys
nothing.

**Control run first, because a broken variant generator produces exactly this null.** The three
libraries have *identical* compiled symbol lists (238 kernels, same names) — WGM is not baked
into the kernel. Parsing the library msgpack confirms they genuinely carry **WGM 6 / 8 / 10
across all 298 solutions**. Two regex probes of the same binary had first suggested the
libraries were identical, matching `WorkGroupMapping` as a prefix of `WorkGroupMappingRR` and
then a capability *flag* named `wgm` rather than its value. **Parse the format; do not grep it.**

One sub-threshold lead, recorded not claimed: on **skinny** shapes both alternatives beat WGM8
by ~1.5 pt (101.90 / 101.91 against a 100.44 local A/A, n=62). Two independent arms agreeing
is more than a single stratum usually offers. Not enough to act on.

## Third use: the Origami `Prediction` rejection was half artefact

The campaign rejected an Origami `Prediction` library as "13 pt worse than GridBased". That
measurement carried a **systematic asymmetry**:

* Origami was asked to predict for **60 CUs** (`--sm_count_target 60`, verified reporting
  `N_CU 60`);
* GridBased's table came from **navi31 tuning at 96 CUs**;
* both then executed on **96 CUs**.

That penalises precisely the arm whose choices were made for a machine it was not run on. With
the CU mask, Origami's prediction target matches execution reality for the first time.

Both libraries hold the **identical 298-solution pool** (verified: `pred298` is
`PredictionMatching`, `wgm8` is `GridBasedMatching`), so this isolates the selector.

`pred298` vs `gridcat`, same 206 shapes, same aggregation:

| regime | geomean | wall-clock |
|---|---|---|
| 96-CU execution (original) | 91.31% | 97.11% |
| **60-CU execution (matched)** | **96.72%** | **98.73%** |
| A/A floor, 96-CU | 99.78% | 102.09% |
| A/A floor, 60-CU | 99.61% | **99.46%** |

**About half the gap was the regime, not the selector.** Geomean deficit 8.7 pt -> 3.3 pt;
wall-clock 2.9 pt -> 1.3 pt. Measured standalone by `analyze.py` at 60 CUs, `pred298` scores
**97.32% geomean / 99.12% wall-clock** against a 100.09 / 99.68 A/A — i.e. **within 0.7 pt of
the control on the time-weighted metric**, essentially wall-clock parity.

Where it still loses is unchanged in character, only in size — by wall-clock:
**GEMV 87.4%, tiny 96.1%, small 94.6%**, while **large 99.1% and medium 100.2% are at parity**.
Small shapes are many by count and few by time, which is exactly why the per-shape geomean
penalty (3.3 pt) exceeds the wall-clock penalty (1.3 pt).

**The rejection stands, but far less emphatically, and for a narrower reason.** GridBased's
dense table beats analytical selection *on small shapes*; on everything else the two are
equivalent. A selector that used Origami above a size threshold and the table below it would
plausibly capture both — the same problem-size predicate that an earlier campaign concluded
needs a C++ row predicate rather than a catalog.


## All four catalogs at real 60 CUs — the shipped wins hold

HHS was validated first; the other three shipped catalogs still rested on the 96-CU regime.
Measured on one shape per stratum (109 shapes, full stratum coverage), 3 arms x 2 reps each.
Comparison is on the **identical shape subset** in both regimes.

| PT | n | 96-CU wall | **60-CU wall** | shift | 96-CU A/A | 60-CU A/A |
|---|---|---|---|---|---|---|
| HHS | 206 | 125.7% | **122.7%** | -2.9 | 102.09% | 99.92% |
| BBS | 109 | 124.1% | **120.8%** | -3.2 | 100.18% | 99.52% |
| AuxH | 109 | 119.9% | **118.7%** | -1.2 | 100.35% | 99.56% |
| AuxB | 109 | 117.4% | **117.4%** | +0.0 | 100.56% | 100.24% |

**Every shipped catalog win survives genuine 60-CU execution** — +17.4% to +22.7% on these
subsets, against the shipped +18.8% to +23.9% measured on the full 998 at 96 CUs.

> **Re-measured at 10x resolution (DPM pinned), 2026-08-28.** Everything in this document was
> measured on default power management, where a library-vs-library ratio only resolves to **~15%**
> — the card alternates between two power states 6.3% apart in throughput, correlated with package
> power at r = +0.955 while temperature stays flat. Pinning the DPM level
> (`rocm-smi --setperflevel high`, see `NAVI32_RUNBOOK.md` §7) gives **~1.4%**. BBS `wide`(246) vs
> `ship`(55), 32-CU emulation, 2 reps:
>
> | shape | thin | wide | wide/thin |
> |---|---|---|---|
> | 4096x3584x4096 | 20 447 | 25 862 | 1.2648 |
> | 3584x4096x2048 | 19 232 | 23 166 | 1.2046 |
> | 2560x4096x512 | 16 447 | 20 783 | 1.2636 |
> | 1792x1280x512 | 14 668 | 17 220 | 1.1740 |
> | 960x320x256 | 5 682 | 7 248 | 1.2755 |
> | **704x192x512** | 6 686 | 6 626 | **0.9910** |
>
> **geomean +19.1%**, against **+19.6%** on `auto` — the headline reproduces within half a point.
>
> **The per-shape detail is what is new.** At 15% resolution a 0.991 and a 1.27 cannot be told
> apart; at 1.4% they can. Widening pays **+17% to +27.5% on five of six shapes and is neutral on
> 704x192x512** — so it is not the uniform gain the averages imply. Pinned figures are
> absolute-throughput-shifted and are deliberately kept out of the `auto` tables above.

**The 96-CU regime overstates the win by 0 to 3.2 pt, never understates it.** That is not a
uniform constant to subtract: three of four move down, AuxB is flat. But no ProblemType gains
from the more faithful regime, so the shipped headline numbers should be read as a **ceiling**
for a real 60-CU part, not a point estimate.

Standalone 60-CU numbers from `analyze.py` (vs each catalog's own A/A floor):

| PT | geomean | wall-clock | A/A wall |
|---|---|---|---|
| HHS | 125.22% | 122.71% | 99.89% |
| BBS | 112.49% | 121.36% | 100.04% |
| AuxH | 114.55% | 118.66% | 99.78% |
| AuxB | 113.33% | 117.91% | 100.25% |

**Caveat stated rather than buried:** the 96-CU sweeps ran `--reps 1` and these `--reps 2`. The
contrast takes best-of-reps per shape, so more reps can only make the 60-CU arm look *better* —
which means the measured shift is, if anything, conservative.

## Fourth use: the catalog-extension rejection, tightened — and where the headroom actually is

The third rejected hypothesis ("extending the catalog past ~300 solutions is worth only +2.8%
oracle") was the one never re-tested in the matched regime. It costs no GPU time: five distinct
libraries were measured at genuine 60 CUs over the same 207 shapes, so the oracle can be
recomputed directly.

**The control that makes this readable.** An oracle taken *across separate sweeps* picks the
luckiest measurement per shape, so it is inflated by run-to-run noise **even when every arm is
the same library** — and the inflation grows with how many arms you minimise over. Measured, by
taking oracles over k copies of `wgm8`:

| k arms | same-library oracle = pure noise |
|---|---|
| 2 | 100.02% |
| 3 | 100.43% |
| 4 | 100.71% |
| 5 | **100.94%** |

Any k-arm oracle must be read against the floor **of matching cardinality**. Comparing a 2-arm
oracle to a 5-arm floor, or quoting a raw oracle at all, overstates the opportunity.

| oracle set | raw | floor | **real** |
|---|---|---|---|
| `wgm8` + `pred298` | 101.49% | 100.02% | **+1.47** |
| `wgm8` + `wgm6` + `wgm10` | 100.47% | 100.43% | **+0.05** |
| `wgm8` + `pred298` + `navi32ship` | 101.63% | 100.43% | +1.21 |
| everything built (9 arms) | 102.65% | 100.94% | **+1.71** |

**Three things fall out.**

1. **The catalog-extension rejection holds, and tightens.** Perfect per-shape selection over
   *everything* built is worth **+1.71 pt**, below the +2.8% originally quoted — which was a raw
   oracle at 96 CUs with no noise floor subtracted.

2. **The WGM variants add +0.05 pt.** A third independent confirmation of the WGM null, from a
   different direction than either previous test: even an oracle that could pick the best WGM
   per shape gains nothing.

3. **Prediction supplies +1.47 of the +1.71.** Essentially *all* the remaining headroom comes
   from combining GridBased with Origami's analytical selection — **not from more kernels.**
   That reframes the improvement path: the opportunity is in the **selector**, not the catalog.

This is consistent with the head-to-head result above — Prediction ties GridBased on large and
medium shapes and loses only on small ones — and with a finding from an earlier gfx1100
campaign, that a real hybrid needs **a problem-size row predicate in C++, not a bigger catalog**.

**An oracle is not an achievable number.** It assumes perfect per-shape foreknowledge; a real
size-gated selector would capture some fraction of +1.47. What it bounds is where the
opportunity *is*, and it says the catalog is done.


## Closing the hybrid question: a realizable gate captures ~a quarter of the oracle

The oracle above says pairing GridBased with Prediction is worth **+1.47 pt**. An oracle
assumes perfect per-shape foreknowledge; what would ship is a predicate on problem size. So:
does a realizable gate capture it?

Fitted and scored **out of sample** (threshold chosen on one half of the shapes, scored on the
other; fitting and scoring on the same shapes manufactures a result):

| gate shape | feature | in-sample | **out-of-sample** |
|---|---|---|---|
| threshold (`pred` above) | flops | 99.41% | **99.08%** |
| threshold | output `M*N` | 99.63% | 99.14% |
| threshold | `min(M,N)` | 99.54% | 99.21% |
| **band** (`pred` inside) | **flops** | 100.48% | **100.47%** |
| band | `M*N` | 100.28% | 99.43% |
| band | `min(M,N)` | 100.15% | 99.24% |

100.00% = GridBased alone; `always-pred298` = 98.73%.

**No monotone threshold works at all** — every one lands *below* GridBased alone. That is
because Prediction's edge is not at the top end: by size it scores large 99.1%, **medium
100.2%**, small 94.6%, tiny 96.1%. The advantage sits in a *middle band*, which a
`>= threshold` predicate cannot express, so the best threshold degenerates to "never use
Prediction".

**A flops band does work, and is worth +0.47 pt out of sample.** In-sample and out-of-sample
agree to 0.01 pt (100.48 / 100.47), which is the signature of a stable rule rather than an
overfit one — the two features that *do* overfit (`M*N`, `min(M,N)`) give themselves away by
going negative out of sample.

**Verdict: real but small.** A shippable gate captures roughly **a quarter** of the +1.47–2.0 pt
oracle; the rest is not separable by problem geometry at all. Against the C++ work of adding a
problem-size row predicate to the selector, +0.47 pt is unlikely to be worth it — and the point
of measuring was to find that out before writing it.

This closes all three rejected hypotheses *and* the one lever the oracle analysis left open.


## Fifth use: is the win the solution POOL or the shape TABLE?

Every measurement in this campaign changed **both** at once — 73 -> 298 solutions *and*
471 -> ~9 700 table rows — so the phrase used throughout ("a 471-row table serves most shapes
from a distant neighbour") was never actually tested against the alternative that the *pool*
simply lacks the right kernel.

**gfx1153 separates them for free.** It ships the **identical 64-solution BBS pool** as
navi33/navi32 with a **different 472-row table**. Retargeted to gfx1100, the two logic files
differ in exactly **472 lines of 12 927** — the table rows, nothing else; solution names are
identical.

| arm | geomean | wall-clock |
|---|---|---|
| A/A control | 99.85% | **99.70%** |
| `tab1153` — different table, **same pool** | 100.04% | **99.79%** |
| `wide` — navi31's 306-solution pool | 116.50% | **120.46%** |

**Over this 64-solution pool the table contributes nothing detectable: 0.09 pt above the A/A
control.** The pool contributes the entire +20.5%. Flat at every jackknife depth and in every size
band (large 99.9%, medium 99.4%, small 100.1%, tiny 100.3%).

> **"Against a 0.30 pt floor" overstates the resolution — see the 2026-08-27 correction below.**
> The A/A floor measures noise between two *identical* arms and is blind to the variance that
> limits a library-vs-library ratio, which is **2-5%** here. Read this null as **"no table effect
> above a few percent"**, which is what makes it consistent with the 3.4 pt effect found later over
> a richer pool — that one sits below what this arm could have seen.

> **Scope this to the thin pool.** Read as a general claim it is wrong — over the shipped
> 306-solution pool the table is worth 3.4 pt. See the correction below.

**This corrects the mental model, not the conclusion.** "The win is the catalog" stands — but
the mechanism is *coverage*, not *mapping quality*. The thin catalog is slow because the pool
does not contain a well-sized tile for the shape, and no re-mapping can conjure one — which is
also why gfx1153's mapping pass (a real one: ~940 of its 472 rows differ) bought nothing
measurable.

> ### Correction: "do not re-fit the table" was too strong
>
> I originally wrote that the table contributes nothing and that one should **"add kernels, not
> re-run the shape-table fit"**. That over-generalised from a thin pool to every pool, and the
> follow-up experiment shows it is wrong for the pool that actually ships.
>
> Holding the *pool* at 306 solutions and sparsifying only the *table* — 9 692 rows down to 471,
> stratified so all 306 solutions stay reachable (a naive random subsample reaches only 160 of
> 306 and would confound the two variables):
>
> | arm | pool | table | wall-clock |
> |---|---|---|---|
> | `wide` | 306 | 9 692 | 100% |
> | A/A control | — | — | 100.38% |
> | **`wsparse`** | **306** | **471** | **96.62%** |
> | `thin` | 64 | 471 | 82.84% |
>
> So the 17.2 pt gap decomposes roughly **80/20**:
> **pool 13.8 pt, table 3.4 pt** — the table effect is ~9x the A/A floor, so it is real.
>
> **Both results hold; they are conditional on each other.** Over a *thin* pool the table is
> worthless (99.79% vs a 99.70% control) — there is nothing worth pointing at, so a denser
> lookup cannot help. Over a *rich* pool it is worth ~3.4 pt. The table's value is a function of
> the pool's, which is precisely what a single experiment at one pool size could not show.
>
> Corrected advice: **widen the pool first — it is ~4x the lever — then the table fit becomes
> worth doing.** By size, the table matters most on tiny shapes (91.4%) and not at all on small
> (100.0%).
>
> ### Correction, 2026-08-27 — "~9x the A/A floor, so it is real" is not a sound test
>
> **The A/A floor is the wrong yardstick for this comparison.** An A/A control runs a library
> against *itself*, so both arms warm up identically: it bounds **noise** and is structurally blind
> to the variance that actually limits a library-vs-library ratio. Measured on this machine,
> repeats of the *same* ratio at *fixed* iterations scatter **2-5%**, and five attempts to reduce
> that (clocks, more reps, more iterations, interleaved pairing, cold-card warm-up, a robust
> median) **all failed**. So a 3.4 pt effect is **at** the resolution limit, not 9x above it.
>
> **The effect does reproduce, but not as significant.** An independent draw of **70 random shapes
> from `eval_shapes_1000.json`** (seed 20260827, 0 skipped, dense vs sparse over an identical
> 246-kernel pool at 32 CU) gives geomean **1.029, 95% CI [0.989, 1.074]** — consistent with the
> 3.4 pt figure, but **the interval includes 1.0**. Per-shape it changes sign: dense wins 32,
> sparse wins 21, 17 tie, worst **0.540**, best **2.145**.
>
> **What to carry forward:** the 80/20 decomposition and "widen the pool first" are unaffected —
> the 13.8 pt pool effect is an order of magnitude above the floor. The **table** half should be
> quoted as *"~3 pt, at the limit of what this setup resolves"*, and any future table work should
> be judged on a many-shape aggregate with a CI rather than against an A/A floor.
>
> Full method, the warmup mechanism, and the five failed floor-reduction attempts:
> `skills/tensile-tuning/references/wiki/05_workflow/lean_catalog_port.md` §1c in
> `ror-claude-skills` (branch `vmijovic/skills`), plus
> `tools/lean_catalog/check_convergence.py` which enforces the doubling check.

**Caveat, stated because it cuts both ways.** gfx1153's table was fitted for gfx1153, not for
this configuration, so this is not a test of an *optimally* fitted table. But note the result is
symmetric: a table fitted for a *different* architecture does not **hurt** either (99.79 vs a
99.70 control). At 471 rows the mapping is insensitive in **both** directions, which is a
stronger statement than either arm alone.

Consistency check: the `wide` arm reads 120.46% here against 121.36% in the earlier BBS 60-CU
sweep — 0.9 pt apart on independently built libraries and separate runs.


## Reproduce

```bash
python3 bench_arms.py \
  --arms ship=$HOME/navi32/libs/navi32ship/library/gfx1100 \
         wide=$HOME/navi32/libs/wgm8/library/gfx1100 \
         ship_aa=$HOME/navi32/libs/navi32ship/library/gfx1100 \
  --shapes state/eval_shapes_masked.json --out results/P12_masked60.csv \
  --reps 2 --cus 60 --fixed-iters 20 --timeout 45      # masked is the DEFAULT
python3 analyze.py results/P12_masked60.csv ship
python3 compare_masked.py
```

**`HIPBLASLT_TENSILE_LIBPATH` must point at the arch subdirectory** (`.../library/gfx1100`),
not the arm root and not `.../library`. Pointing one level up produces `status=error` on every
single row at full speed — rows appear at the normal rate and every one is empty. Check
`status` counts before reading any number.
