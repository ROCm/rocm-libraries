# Re-mapping the grid for 60 CU — what it bought, and what it broke

Follow-on to the lean-Grid campaign. The grid shipped on navi32/navi33 was tuned on navi31 at
**96 CU**; these parts run at 60 and 32. This asked whether re-pointing `element[7]` at
measured-better kernels — no new kernels, no pool change — recovers the gap.

**Answer: partly, and it must be gated.** Ungated it regresses tiny shapes by up to 35%. Gated,
it clears the bar — see the correction below, which reverses an earlier conclusion in this file.

---

## Headline

| | |
|---|---|
| in-sample ceiling, measured **at grid keys** | **+19.5%** median |
| realized **held-out**, off-lattice queries | **+1.8%** wall-clock |
| control (queries served by untouched rows) | +0.1–0.3% — nothing |
| A/A floor | 100.2–100.4% |

**The ~10x gap between ceiling and realized is the finding.** Optimising a grid row for its own
shape transfers only weakly to the queries that actually snap to it — even though the median
row/query size inflation is 1.00x. Every eval query is off-lattice (0/207 land on a grid key),
so this is the honest number and the +19.5% is not.

## What it broke — the aggregate hid a tail regression

Treated queries, wall-clock vs the shipped lean catalog (>100 = better):

| stratum | n | A/A | argmax | robust |
|---|---|---|---|---|
| **tiny** | 5 | 99.66% | **64.80%** | **71.37%** |
| **gemv** | 10 | 99.02% | **94.88%** | **84.06%** |
| skinny_M | 26 | 100.25% | 102.59% | 101.73% |
| skinny_N | 24 | 100.35% | 101.50% | 101.34% |
| med | 22 | 99.98% | **103.62%** | 103.35% |
| large | 18 | 100.22% | 101.83% | 102.22% |

By duration: **<50 us regresses to 95.5%**; 50 us–5 ms gains 2–3.4%.

A **35% regression on tiny shapes** is a product regression. It moved the time-weighted mean by
almost nothing, because those shapes are too short to matter to it — which is exactly why
wall-clock alone is not a sufficient bar. *Select on the tail, not the mean.*

**Mechanism.** For tiny/gemv, kernel time sits at or below the ~25 us dispatch floor, so the
"best kernel" at a grid key is chosen from near-noise and does not transfer to the smaller
queries that snap to it. Gating those strata out costs ~0.2 pt of aggregate and removes the tail.

## What the measurement chain established

1. **Warm rankings do not survive cold measurement** — 12% top-1 agreement, size-dependent.
   All prior work in this repo was warm-cache (`hipblaslt-bench` defaults `--rotating 0`, no
   `--flush`; the *tuning* environment defaults to `rotating=512, flush=true`). Same shape, same
   kernel, 512³: warm 24 971 GF/s, cold 8 010 — **3.1x**. `--flush` is a bare flag; `--flush 1`
   is rejected.
2. **The shipped lean catalog is safe under cold measurement** — 100.0% vs the original pick on
   every stratum. That was the live risk when the ranking gate fired, and it is cleared.
3. **Iterations must be time-derived, not fixed.** At a flat 30 iterations the cold small-shape
   noise floor is **8–10% median / 14% p90** — enough to swamp the signal. Scaled to a ~10 ms
   accumulated target (≈400 iterations for tiny, 6 for the largest) it is 0.68% / 1.96%.
4. **The top of the ranking is a plateau** — top-5 within ~2%, and on 73% of rows the winner is
   not clear of the noise floor. So the argmax is largely a coin-flip *among near-ties*, which
   explains the 12% agreement without implying the measurement is unreliable. The re-map's job
   is not to find the optimum but to move picks that are 19.5% *outside* the plateau into it.
5. **Plateau-aware selection buys nothing** — 101.83% vs 101.84% against the raw argmax. Cleanly
   refuted; the simpler rule is as good.

## Controls

| control | requirement | measured |
|---|---|---|
| untouched-row queries | must not move | +0.1–0.3%, at/below their own A/A floor |
| A/A arm | in-session floor | 100.2–100.4% |
| held-out split | 89 train / 86 test **serving rows** | 106 treated vs 101 control queries |

The control is what makes the +1.8% attributable: the effect tracks exactly the rows that were
re-mapped, in the same run.

## Two process bugs, both self-inflicted

- **A script deadlocked on its own launcher.** `run_remap_chain.sh` was created with a heredoc
  *and* started in the same shell command, so the launcher's command line contained the script
  body — including the `matrix_sweep` string the script then waited on via `pgrep -f`. It blocked
  on its creator indefinitely. The rule is broader than "pgrep matches itself": **`pgrep -f` on
  any string that appears in your tooling is unsafe.** Wait on something structural.
- **The validation nearly measured nothing.** It was set to test a re-map touching only 25 of 207
  eval queries, because rows were sampled by grid population while queries concentrate on a
  different 175 rows. Found by checking which rows the eval queries actually snap to.

## The gated variant, measured

Skipping tiny/gemv rows, 105 treated queries, cold @ 60 CU:

| arm | geomean | wall-clock |
|---|---|---|
| A/A floor | 100.14% | 100.13% |
| **gated re-map** | **101.50%** | **101.96%** |

| stratum | n | A/A | gated | (ungated was) |
|---|---|---|---|---|
| tiny | 5 | 99.64% | **99.93%** | 64.80% |
| gemv | 10 | 99.20% | **98.11%** | 94.88% |
| skinny_M | 26 | 100.05% | **102.55%** | 102.59% |
| skinny_N | 24 | 100.14% | 101.16% | 101.50% |
| med | 22 | 100.69% | **102.57%** | 103.62% |
| large | 18 | 100.07% | 101.98% | 101.83% |

The gate removes the tiny regression entirely (64.80% -> 99.93%, i.e. parity) at no cost to the
aggregate — +1.96% wall-clock against a 100.13% floor.

### CORRECTION — the gemv reading was noise, and my explanation for it was wrong

An earlier revision of this report claimed gemv stayed ~1.1 pt soft and that a row-side gate was
*structurally* incapable of fixing it, because strata are a property of the QUERY while the gate
applies to the ROW. **Both halves of that were wrong**, and checking rather than asserting is
what showed it:

* **The mechanism is not real at this scale.** Only **3 of 23** gemv queries (13%) are served by
  a different-stratum row. 87% are served by gemv rows — which the gate excluded.
* **The observation is not real either.** Of the 10 gemv queries in the gated benchmark, only
  **2** were served by a re-mapped row. The other **8 ran an identical catalog** and still showed
  a median `lean/gated` ratio of **0.975**, with their own A/A column spanning 0.939–1.048.
  That is the per-stratum noise floor at n=10, not a regression.
* The 2 genuinely re-mapped gemv queries went 0.942 and **1.242** — net positive.

**So the gated re-map shows no demonstrated stratum regression.** It meets the pre-registered
bar. The honest residual caveat is different and more mundane: **per-stratum resolution at n≈10
is roughly ±2.5%**, so every small-stratum number in this report — in either direction — is
weaker evidence than the aggregate.

## Status and recommendation

**SHIPPED for HHS-TN on both architectures**, after an independent confirming run settled the
open question.

| | run 1 | run 2 | spread |
|---|---|---|---|
| gated wall-clock | 101.96% | **102.29%** | 0.33 |
| A/A floor | 100.13% | 100.08% | 0.06 |
| gated geomean | 101.50% | 102.16% | 0.67 |

**The decisive test was not the aggregate but per-shape reproducibility.** Gains correlate at
**r = 0.961** across independent runs with **90% sign agreement**, against an A/A noise reference
of **r = 0.551** computed the same way on the same shapes. The same shapes win both times, so the
effect is structural rather than noise averaging positive. A single run cannot make this
distinction at all, which is why one good-looking number was never enough.

Per stratum, run 2 confirms no regression anywhere — and **gemv reads 100.73%**, closing the
question the correction above opened:

| stratum | run 1 | run 2 |
|---|---|---|
| gemv | 98.11% | **100.73%** |
| tiny | 99.93% | 100.00% |
| skinny_M | 102.55% | **102.89%** |
| skinny_N | 101.16% | 101.51% |
| med | 102.57% | **104.65%** |
| large | 101.98% | 101.97% |

**Scope of the ship — deliberately partial.** Only **HHS-TN** is re-mapped. The cold matrix was
measured against the HHS kernel library; BBS/AuxH/AuxB have different kernels (other data types,
other epilogues), so the measured best-per-row does not transfer to them. They keep their lean
mapping until measured in their own right. Both architectures pass the build gate: 97 kernels,
0 assembler errors, 0 `overflowedResources`, ELF `0x46`/`0x47`, grid unchanged at 9 680 rows.

Even gated, the realized gain is ~+2% wall-clock on treated queries — real, attributable, above
the floor, but modest against the ~1% that lean cost. **The larger lever is coverage:** only
2 139 of 9 680 grid rows were measured, so 78% of the grid still carries its 96-CU mapping.
Extending the sweep would scale the gain at ~4x the measurement cost.

## Post-ship verification

The gate above proves the re-mapped catalog compiles in isolation. The shipped tree was then
built as a whole, with the re-map coexisting with all 38 logic files and the `Equality/` path:

| | gfx1101 | gfx1102 |
|---|---|---|
| full-tree device-library build | 1 364 kernels | 1 183 kernels |
| assembler errors / `overflowedResources` | 0 / 0 | 0 / 0 |
| code objects | 54 | 46 |

**Numerical correctness: 47 PASS, 0 FAIL** (`hipblaslt-bench -v`, shapes spanning every stratum,
re-mapped catalog built for gfx1100). A re-map only changes *which* pre-existing kernel a row
names, so correctness ought to be inherited — but it sends shapes to kernels they were never
exercised on, which is worth checking rather than assuming.

*Harness note: 48 shapes were generated and 47 tested. `while read` drops a final line with no
trailing newline. So this is 47/47 passed with one never tested, not 47/48.*

**Not verified, and not claimable here:** gfx1101/gfx1102 binaries were never executed — this
machine has only a gfx1100 card. Their correctness rests on running the identical catalog content
on gfx1100 plus a clean build for the real targets.
