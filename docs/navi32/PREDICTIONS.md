# Pre-registered predictions — coverage extension, TN HHS navi32

Written 2026-08-25 14:25, with the matrix **36% complete (3 460/9 680 rows)** and **before any
arm has been benchmarked**. The point is to state what should happen while it is still possible
to be wrong about it. Anything I revise after seeing results gets marked as such.

## What the catalogs do to the eval set (measured offline, not predicted)

At 36% coverage, of 600 equal-weight-per-row eval queries:

| catalog | queries changed | treated | control | median gain **at the row** |
|---|---|---|---|---|
| `extended` (gated + holdout) | 20 | 20 | **0** | 19.3% |
| `extended_ship` (gated, no holdout) | 51 | 20 | 31 | 19.6% |
| `nogate` (ungated, no holdout) | 106 | 41 | 65 | 25.4% |

`extended` changing **zero** control queries is the check that the control-group bug is actually
fixed — held-out rows are byte-identical to shipped, so those queries must be pure noise.

Scaling by newly-measured rows, at full coverage expect roughly **~110 of 300 treated queries**
to change kernel under `extended`, and **~230 of 600** under `nogate`.

## Predictions (falsifiable)

1. **Control group reads 100.0% ± the A/A floor.** If treated and control both move, the effect
   is not the re-map and nothing else in this report is trustworthy. This is the load-bearing one.
2. **Treated group gains low single digits, not ~19%.** The row-level headroom is ~19%, but the
   measured transfer tax is ~2% median (p10 ~78–82%) and only ~38% of treated queries change at
   all, so the aggregate should land around **+1 to +3%**. A result near +19% would mean I am
   measuring at grid keys by mistake, not that it worked spectacularly.
3. **`nogate` >= `extended_ship`.** The gate was justified against the `argmax` catalog (tiny
   64.80%), but `full` measures tiny at 99.79%, and tiny/gemv have the *best* kernel-choice
   transfer of any stratum (99.5%/99.4% vs med's 95.2%). If `nogate` instead regresses tiny, the
   gate was right for a reason I have not identified and the transfer analysis is missing
   something.
4. **`shipped` and `full` roughly reproduce their P4 values** (101.00% / 103.98% vs `lean`) on
   this brand-new eval set. If they do not, the finding is about the harness, not the catalogs.
5. **`large` stays unclaimable** (n=16 by serving row). It will not be averaged into a headline.

## Added 15:50 — after the user caught an inflated ceiling

I had been reporting headroom as `max` over all **298** measured kernels, but the shipping
catalog is the **lean 100** and only those can be installed. That is a ceiling that cannot be
reached. Corrected (non-gated strata): newly-measured rows **15.2%** median reachable headroom
(not 16.6%), and rows `shipped` already used **0.0%** (not 0.8%) — i.e. shipped already picks the
best *available* kernel on everything it measured, so all remaining gain comes from unmeasured
rows. **Rule: an oracle over a superset of what you can ship flatters everything derived from it.**

Note the offline *methodology* experiments (transfer tax, catchment-aware selection, per-stratum
kernel spread) were computed over the 298 pool. Their conclusions are comparative and should not
flip, but any number quoted from them is a full-pool number, not a shippable one.

6. **The oracle sweep will show the lean-100 catalog costs little vs the full 298 pool** on the
   eval queries. The lean campaign claimed parity at 1/3 the kernels, but that was a benchmark
   result, never checked against a ceiling. Prediction: median gap **under ~2%**, with a tail
   where a dropped kernel was genuinely the right one (the M=192 N=128 K=4096 example loses
   14368 -> 11586 GFlop/s, ~19%, to the reduction). If the median gap is large, the lean
   reduction is quietly costing performance that no benchmark on this eval set has exposed.

## Added 16:55 — eval-set representativeness, verified not assumed

Headroom turns out to be an **inverted U in problem size** (median 9.8% at log10 M·N·K=4, peaking
**36.1% at 7**, back to 7.1% at 10). That raised two ways the eval set could mislead, both now
measured:

* **Jitter pushing queries into a lower-headroom regime** — it does not. Median downward shift is
  **0.01 decades** (p90 0.21), because the grid is dense enough that catchments are narrow.
  Query and serving-row decade mixes agree to 0.4 pt.
* **Equal-weight-per-row sampling misrepresenting the grid** — it does not. Largest decade
  deviation **2.0 pt**; every stratum within **1 pt** (med 45.3 vs 45.6, large 2.7 vs 2.7).

Headroom-weighted expectation over the eval set: **21.0% at the row**. Prediction 2 stands — the
realised figure should be far lower (transfer tax plus only ~38% of treated queries changing at
all), and a result anywhere near 21% would indicate I am measuring at grid keys by mistake.

## What would make me abandon the ship

- Any stratum regressing beyond the A/A floor in **both** runs.
- Per-shape gains failing to correlate across the two runs (r near the A/A reference ~0.55),
  which would mean noise averaging positive rather than a structural effect.
