# Re-mapping the grid for 60 CU — what it bought, and what it broke

Follow-on to the lean-Grid campaign. The grid shipped on navi32/navi33 was tuned on navi31 at
**96 CU**; these parts run at 60 and 32. This asked whether re-pointing `element[7]` at
measured-better kernels — no new kernels, no pool change — recovers the gap.

**Answer: partly, and it must be gated.** Ungated it regresses tiny shapes by up to 35%.

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

**gemv remains ~1.1 pt below its own A/A, and the gate cannot fix it as built.** Strata are a
property of the QUERY, but the gate is applied to the ROW. A gemv query can snap to a row
classified `med` or `skinny_N`, which is re-mapped. At n=10, with the A/A itself 0.8 pt off
100, this is marginal rather than clearly real — but it is unresolved, and a row-side gate is
structurally incapable of resolving it. A query-side guard would have to live in the selector,
not the catalog.

## Status and recommendation

**Not shipped.** The gated variant clears the aggregate bar (+1.96% vs a 100.13% floor, tiny
restored to parity) but gemv is still soft and the cause is a structural limit of row-side
gating, not a tuning knob. Shipping a known-soft stratum on n=10 is not worth +2%.

Even gated, the realized gain is ~+2% wall-clock on treated queries — real, attributable, above
the floor, but modest against the ~1% that lean cost. **The larger lever is coverage:** only
2 139 of 9 680 grid rows were measured, so 78% of the grid still carries its 96-CU mapping.
Extending the sweep would scale the gain at ~4x the measurement cost.
