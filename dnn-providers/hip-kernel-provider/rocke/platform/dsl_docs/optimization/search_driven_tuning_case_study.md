# Search-driven tuning of rocKE kernels: a case study

This document records the method and the relative results of applying the
rocKE search engine (`rocke.dispatch.tuning.search`) to three kernel families:
a WMMA attention forward kernel on `gfx1201`, and the universal fp16 GEMM and
the dense attention candidates on `gfx950`. It covers measured relative
speedups over each family's baseline at a small number of problem shapes, the
method that produced them, and the caveats. It is a worked example of the
procedure in [`search-optimize-rocke.md`](utilities/skills/search-optimize-rocke.md),
not a performance claim for any product. Absolute timings are not recorded here.

## Method

- **Baseline.** The dispatcher's own pick for the problem, or for a dense
  candidate its default spec (see "Baselines" below).
- **Space.** The spec's structural fields (tile geometry, warp layout, MFMA/WMMA
  atom, pipeline, epilogue) and its default-off levers (scheduling hints,
  occupancy hints, LDS layout hints, swizzles, padding). Spaces ranged from tens
  of configurations to millions.
- **Strategy.** Exhaustive where the legal space was small enough to enumerate;
  otherwise random, hill climbing with lookback and a bandit portfolio, warm
  started from the baseline.
- **Numeric gate.** Every kernel arm was checked on silicon against an
  independent reference before its first timing (fp64 or fp32 reference on
  sampled output rows, input dtype fp16, max-abs tolerance 2e-2). A config that
  failed was never timed.
- **Timing and confirmation.** One stream, interleaved measurements, then the
  winner and the baseline re-timed together in five interleaved rounds. The
  reported speedup is the paired confirmed speedup, with its p-value; screening
  values are not reported.
- **Noise floor.** An A/A control (the baseline spec compiled twice under
  different names) differed from the baseline by roughly 0.5% to 2%.

## Results (relative to each baseline; fp16; several shapes per family)

| Family | Target | Evaluations per search | Confirmed speedup range |
|---|---|---|---|
| WMMA attention forward, exhaustive sweep of the legal space | `gfx1201` | tens to low hundreds | about 1.2x to 1.9x |
| Universal GEMM (RCR) | `gfx950` | 120 | about 1.4x to 2.4x |
| Dense attention, grid and persistent candidates | `gfx950` | 40 | 1.0x to about 1.2x |

Every reported speedup is a paired confirmed value with p < 1e-3. Within a
family, speedups depended on the problem: some searches found no configuration
that beat the baseline beyond noise, and those are included as 1.0x.

Observations that held across families:

- **Scheduling and occupancy hints were among the largest single levers** where
  the baseline left them off, and they are shape-independent: one extra binary
  serves every shape that wants it.
- **A few structural choices recurred in winners** (a larger K tile and a
  wider-K MFMA atom for GEMM; a larger KV block for the dense attention grid
  candidate). These are candidates for a changed default and need a full
  production cohort before they become one.
- **Warm start mattered more than strategy.** Cold random search varied widely
  between seeds; hill climbing and the portfolio, started from the baseline,
  were consistent. Where the legal space was small enough to enumerate,
  exhaustive search was the right choice.
- **Where the baseline was already tuned, the gain was small**, as for the
  persistent dense attention candidate.

## Baselines

- `gfx1201`: that architecture has no dispatcher selection of its own for this
  kernel. The baseline is the `gfx1151` dispatcher heuristic with the levers that
  only exist on `gfx1151` removed and causal tile skipping kept for causal masks.
- `gfx950` GEMM: the dispatcher's pick for each shape.
- `gfx950` dense attention: each candidate's default spec. Production `auto` for
  these problems selects a different (unified) path and was not measured.

## Caveats

- Results are bounded to the measured shapes, fp16, and one device per target.
  They are not general statements about a kernel family or an architecture.
- The `gfx1201` occupancy hint was set through a kernel attribute outside the
  spec in the harness. It is a prototype result until the hint is a spec field.
- Dense attention used one seed per strategy at a small budget.
- A searched configuration is a measured result, not a shipped kernel. See
  "Shipping" in the skill. These results do not estimate how many variants a
  consumer would need to carry; reviewers who see per-shape winners multiply
  instances should say so.

## Provenance

Search engine as added in the commit that introduces
`dispatch/tuning/search/`; harnesses were scratch drivers around
`CallableBackend` (and `tune_candidate` for the dispatch candidates) following the
recipe in the skill. Timing used the runtime's `time_launches` on HIP events.
