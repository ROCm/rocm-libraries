# Autotuner Overview

`helpers/autotune.py` is the in-process autotuner. It mirrors Triton's `@triton.autotune` design but specialized for the CK DSL's spec-dataclass + IR-builder + launcher pipeline.

## Concepts

- **Config**: one `AutotuneConfig(spec=..., name=..., extra={})` per point in the search space. `spec` is a kernel spec dataclass (e.g. `UniversalGemmSpec`, `ImplicitGemmConvSpec`). Compiler-tuning experiments can carry a typed `CodegenPolicy` in `extra["codegen_policy"]`; the caller applies it to the built kernel.
- **Key**: a tuple of runtime shape / dtype / layout values, produced by the user-supplied `key_fn`. The autotuner caches the per-key winner.
- **Bench callable**: user-supplied `bench_fn(config, **runtime_args) -> ms`. The caller builds the kernel from `config.spec`, runs warmups, and returns a HIP-event-timed average (typically via `time_launches`).
- **Launch callable**: user-supplied `launch_fn(config, **runtime_args) -> None`, invoked once per real launch with the winning config.
- **Cache**: in-memory dict, plus an optional JSON file on disk keyed by the `key_fn` tuple.

When sweeping scheduler policies, give every policy variant a distinct config
name and compiled-artifact location. See
[`../optimization/scheduler-policy.md`](../optimization/scheduler-policy.md).

## Top-Level API

```text
AutotuneConfig(spec, name, extra={})
AutotuneResult(config_name, ms_per_iter, error=None)   # .is_ok property
AutotuneKey(graph_hash, shape, dtype, layout="RCR", arch="gfx950",
            compiler="comgr", lowerer="unknown", spec_hash="any")
make_autotune_key(*, graph_hash, shape, dtype, layout="RCR", arch="gfx950",
                  compiler="comgr", lowerer="unknown", spec_hash="any")
Autotuner(configs, *, key_fn, bench_fn, launch_fn, cache_path=None,
          warmup_iters=10, bench_iters=50, verbose=False)
autotune_sweep(configs, *, bench_fn, on_progress=None)
    -> (winner_config, List[AutotuneResult])
spec_replace(spec, **kwargs)  # dataclasses.replace alias
```

## End-to-End Recipe

```python
from rocke.helpers import Autotuner, AutotuneConfig, autotune_sweep
from rocke.runtime.launcher import time_launches
from rocke.instances import (
    UniversalGemmSpec, TileSpec, TraitSpec,
    build_universal_gemm,
)

configs = [
    AutotuneConfig(name="t128_a32x32x16",
                   spec=UniversalGemmSpec(
                       name="hero",
                       tile=TileSpec(128,128,32, warp_m=2, warp_n=2,
                                     warp_tile_m=32, warp_tile_n=32, warp_tile_k=16),
                       trait=TraitSpec(pipeline="compv4", epilogue="cshuffle"),
                   )),
    AutotuneConfig(name="t256_a32x32x16",
                   spec=UniversalGemmSpec(
                       name="hero",
                       tile=TileSpec(256,128,32, warp_m=2, warp_n=2,
                                     warp_tile_m=32, warp_tile_n=32, warp_tile_k=16),
                       trait=TraitSpec(pipeline="compv4", epilogue="cshuffle"),
                   )),
]

# bench_fn owns build + compile + launch + timing for one config.
def bench_gemm(config, *, M, N, K, dtype, A, B, C, **_):
    kdef = build_universal_gemm(config.spec)
    # ... compile_kernel(kdef), construct a KernelLauncher, then:
    return time_launches(launcher_call, warmup=10, iters=100)  # ms/iter

# launch_fn dispatches a single real launch with the winning config.
def launch_gemm(config, *, M, N, K, dtype, A, B, C, **_):
    ...  # build/compile (cached) + launch once

tuner = Autotuner(
    configs,
    key_fn=lambda *, M, N, K, dtype, **_: (int(M), int(N), int(K), str(dtype)),
    bench_fn=bench_gemm,
    launch_fn=launch_gemm,
    cache_path="~/.cache/rocke_autotune.json",
)

# First call for a key sweeps + caches; later calls dispatch directly.
tuner(M=4096, N=4096, K=4096, dtype="fp16", A=A, B=B, C=C)
```

First call for a given key:

1. `key_fn(**runtime_args)` produces the cache key; on a miss the sweep runs.
2. The autotuner calls `bench_fn(config, **runtime_args)` for each config — the
   caller builds/compiles the kernel from `config.spec` and returns a
   HIP-event-timed ms/iter (typically via `time_launches`).
3. Picks the config with the lowest ms (`AutotuneResult.is_ok` filters errors).
4. Caches `(key, winner_name)` to disk.
5. Subsequent calls look up the cached winner and call `launch_fn` directly.

Subsequent calls for the same key:

1. Look up the cached winner.
2. Launch directly with no re-tuning.

## Why It's Fast On CK DSL

- Each config builds in ~15-30 ms warm (vs minutes per CK Tile template instantiation).
- A 10-20 config sweep finishes in 20-40 s on a current ROCm box.
- The HIP-event timer in `time_launches` is the same one production uses, so the chosen config reflects real device-side wall time.
- The disk cache makes a subsequent Python startup pay zero retuning cost.

## Cache Layout

The JSON cache is a flat dict:

```json
{
  "(4096, 4096, 4096, 'fp16')": "t128_a32x32x16",
  "(8192, 8192, 8192, 'fp16')": "t256_a32x32x16"
}
```

`AutotuneKey` extends the basic tuple with `arch`, `compiler`, and `lowerer` fields to avoid cache poisoning across ROCm versions or backend changes.

## Manual Sweep (No Caching)

`autotune_sweep(configs, *, bench_fn, on_progress=None)` runs the sweep without
any cache, returning a `(winner_config, results)` pair where `results` is a list
of `AutotuneResult` rows. Use this for one-shot exploration and CSV export.

```python
from rocke.helpers import autotune_sweep

# bench_fn takes only the config here (close over the fixed shape/args).
winner, results = autotune_sweep(
    configs,
    bench_fn=lambda cfg: bench_gemm(
        cfg, M=4096, N=4096, K=4096, dtype="fp16", A=A, B=B, C=C
    ),
)

for r in sorted(results, key=lambda r: r.ms_per_iter):
    print(f"{r.config_name:30s}  {r.ms_per_iter * 1000:8.2f} us  ok={r.is_ok}")
```

## Differences vs Triton autotune

- Configs are typed `Spec` dataclasses, not kwargs. The search space is checked at construction.
- Timing uses HIP events through `time_launches`, not host timing.
- The cache is in-memory by default; pass `cache_path=` to also persist it as JSON across processes.
- Build + launch is < 30 ms per config warm vs minutes per CK Tile template.
- Errors raised by `bench_fn` (validation failure, unsupported config) are recorded as `AutotuneResult(error=...)` rather than crashing the sweep.

## Failure Modes

- `key_fn` accidentally captures non-hashable values (lists, tensors). Use `int`, `str`, `tuple` only.
- `bench_fn` builds a kernel whose launch signature doesn't match the `runtime_args` it was handed. Validate the signature against the spec before adding the config.
- Cache file on a slow filesystem; the autotuner does a write per winner. Move the cache to a fast disk if many keys are seen.
- `bench_fn` uses too few iters to discount cold-cache effects on the first config. Run a throw-away warmup loop inside `bench_fn` before timing.

## When To Use vs The Sweep CLI

- `helpers/autotune.py::Autotuner` is for in-process selection from a known config set, with caching.
- `python -m rocke.sweep` + `python -m rocke.sweep_bench` is for offline cartesian sweeps with median + spread + CSV output. Use that when the goal is to *find* a good set of configs.
- The two compose: run the offline sweep, harvest the best 5-10 configs, register them as `AutotuneConfig`s in the dispatcher.

## Search engine (`rocke.dispatch.tuning.search`)

The Autotuner above measures every config it is given, and a dispatcher sweep
walks a candidate's whole knob space (`production` or `full`). Real spaces are
too large for that: one attention variant's axes span about 10^10 knob
combinations and a GEMM family's tile, pipeline and lever space has thousands of
legal configs per problem. Each measurement costs a compile, a numeric check and
a timed run. The search engine replaces enumeration with strategies that decide
what to measure next from what has been measured. It works for every family,
because it only sees a space of parameters, a validity rule and a measurement
backend.

### Pieces

| Module | What it owns |
|---|---|
| `params` | `Param` (ordinal / categorical / bool, optionally gated) and `SearchSpace`: genomes, decode/encode, canonical keys, distance, single moves, validity repair, stratified sampling, `from_configs`, `from_axes` |
| `knobspace` | `KnobSearchSpace`: a dispatcher `KnobSpace` for one request. Its validity comes from `canonicalize`, its keys are `config_key`, and its structural sampler is the pruned axis walk |
| `trial` | `Trial`, `Observation`, `Status`, `Fidelity`, `Budget`, `BudgetMeter`. All of them round-trip through JSON |
| `strategy` | the `Strategy` ask/tell base class, plus `RandomSearch` and `Exhaustive` |
| `de` | `DifferentialEvolution`: rand/1/bin, current-to-best/1/bin and current-to-pbest/1/bin, with no adaptation, jDE or SHADE adaptation |
| `halving` | `SuccessiveHalving`, `Hyperband` over a fidelity ladder, with racing |
| `stats` | paired and Welch t tests, the exact Wilcoxon signed-rank test, Holm, `Race` (on measurement calls), `NoiseModel` (per-rep and per-call noise) |
| `surrogate` | priors (`CallablePrior`, `TablePrior`, `LightGBMPrior`), `TPE`, `GPEI` |
| `local` | `HillClimb`: first-improvement climbing, knob-combination re-sweeps, ordinal two-step escapes, lookback over previous centers, restarts |
| `portfolio` | `Portfolio` (a discounted Thompson-sampling bandit over member strategies, optionally cost-aware) and `WarmStart` |
| `measure` | the `Backend` protocol, the `Evaluator` that enforces the measurement rules, and the `CallableBackend`, `SyntheticBackend` and `TableBackend` backends |
| `db` | `TunedRecord`, `TuningDB` (the store dispatch serves, plus transfer queries), `TraceWriter` |
| `driver` | `run_search`: the ask/tell loop, budgets, checkpoint/resume, final confirmation |
| `api` | `make_strategy`, `tune`, `tune_candidate`, `search_configs`, `SpecBackend`, `record_result` |
| `replay` | offline scoring of strategies against a recorded sweep table |
| `synthetic` | GEMM-shaped and sphere problems with known optima, used by tests and the CLI demo |

The package needs only NumPy (for the GP) and the standard library. It never
imports torch, a family or the IR layer, and it emits no code, so it has no
byte-identity impact.

### The space

A genome is one level index per parameter. A level is a knob payload, and `()`
means "leave the base value". Decoding a genome merges the payloads of the
active parameters. A gated parameter (for example `prefetch_depth`, gated by
`prefetch`) is active only while its gate is off level 0. Its gene stays in the
genome while the gate is off, so DE keeps it as latent material, but it does not
change the decoded config or the key. Two genomes with the same key are one
config and are measured once.

`params_from_axes` maps dispatcher axes to parameters:

- A numeric single-knob axis becomes ordinal when the knob's base value is
  known. A `choices` axis lists the base value itself; its level is merged into
  `()`, so no two levels of a parameter name the same config (a duplicate next
  to `()` used to stop a climber from ever stepping up from the base). A
  `values` axis gets `()` inserted at the base value's position. With the base
  unknown, the axis is categorical with `()` first: an ordinal axis must know
  where `()` lies.
- Any other single-knob axis is categorical, its base-equal choice merged into
  `()`; an axis whose only choice restates the base is dropped.
- `KnobSearchSpace` learns base values the kernel does not report (fields a
  policy or the request resolves): a choice whose canonical config equals the
  default's restates the base.
- `flag` becomes bool.
- A `gated` axis whose choices are the full product of its sub-knobs becomes a
  bool gate plus one gated parameter per sub-knob.
- Any other axis becomes one categorical parameter over its choices.

The outer knob becomes one more parameter.

Validity is the space's own predicate (for `KnobSearchSpace`, `canonicalize`
returning a spec), and it is cached. `repair(genome)` maps an invalid genome to
a valid one. It tries these, in order:

1. Every single move, nearest first. Moves at equal distance are tried in an
   order drawn from the strategy's RNG: a fixed order by parameter index and
   level skewed every repaired population towards the first parameters and
   their low levels (on a space symmetric in two tile knobs, repair changed
   the first several times as often as the second).
2. A greedy prefix repair along the parameter order, used when the space
   declares a `prefix_valid` predicate. This matches the dispatcher's rule that
   constraints only point to earlier axes. Each parameter keeps its target level
   while the prefix stays valid.
3. Pairs of the 24 nearest single moves (equal distances again in RNG order).
4. The nearest valid genome a strategy already holds (ties at random).
5. A random valid genome.

Strategies therefore never spend a measurement on a config the space refuses.
Spaces with long dependency chains, such as attention's unified-kernel flags,
are almost never hit by uniform random genomes. For those, `random_valid` uses
the space's structural sampler (the dispatcher's pruned random walk).

`SearchSpace.from_configs(rows)` turns a flat list (Autotuner configs, a
recorded sweep) into a space. Each field that varies becomes a parameter. Only
the listed configs are valid, and repair lands on the nearest listed config.
Because the list is known, the space indexes its members:

- random valid configs are drawn from the list itself;
- `listed()` returns every member, so candidate pools no larger than the list
  use it directly;
- repair falls back to the exact nearest member (NumPy scan over per-parameter
  level-distance tables, ties broken by the strategy's RNG) instead of pairs,
  the archive or random draws.

Recorded grids are often sparse in the product of their fields (a conv-wgrad
tile grid covers a small fraction of it). Without the index, uniform sampling
and move-based repair mostly miss there: a strategy could not even draw a
random valid config on some recorded lists. In a sparse list (members under a
quarter of the product) the nearest member is tried first; in a dense list
single moves come first because they are cheaper and almost always succeed.
`member_index=False` restores the generic behaviour for ablations.

### Strategies

All strategies share one protocol. `ask(n)` returns up to `n` trials, and `[]`
means the strategy has nothing new to propose. `tell(observations)` takes both
the strategy's own trials and other strategies' trials, and a strategy
recognizes its own by `uid`. `state_dict()` returns JSON data and
`load_state_dict` restores it. Every random draw comes from the strategy's
seeded `random.Random`, so a run is a pure function of the seed and the
observations.

- **Differential evolution** (`de`, `jde`, `de_rand`).
  - Ordinal parameters are positions on the level axis. The arithmetic
    operators apply, followed by stochastic rounding, so small differences still
    move on axes with few levels.
  - Categorical and bool parameters use the discrete analogue: the mutant takes
    a donor's level with probability `min(F, 0.5)` where its difference vectors
    disagree, the first term that fires winning (in current-to-pbest the pull
    towards the elite is not overwritten by the random term, and a large
    adapted F does not turn categorical genes into uniform recombination), plus
    a small uniform reset rate.
  - Binomial crossover treats a gate and its sub-knobs as one block when the
    gate state differs between target and mutant.
  - Trial vectors are repaired to valid configs, and the repaired levels are
    written back into the trial (Lamarckian repair). A trial whose config is
    already known is resolved without measuring.
  - `de` is current-to-pbest/1/bin with SHADE success-history adaptation and an
    archive of replaced parents. `jde` is rand/1/bin with jDE self-adaptation.
    `de_rand` is classic rand/1/bin with fixed F and CR.
  - Foreign finds (from other strategies) that beat the worst member replace it
    at the next generation.
  - Selection is noise-aware once the evaluator has seen noise above its floor:
    a trial within noise of its parent replaces it (so a lucky parent does not
    block every comparable trial), and only an improvement beyond noise counts
    for the F/CR adaptation. Parents are not re-measured; the confirmation
    round settles the final pick.
  - A population that collapses onto one config restarts around its elite; DE
    reports `done` after many generations of only known configs. The default
    population size is bounded by the number of valid configs when the space
    knows it.
- **Successive halving / Hyperband** (`sh`, `hyperband`).
  - The evaluator owns a fidelity ladder, cheapest first. Rungs can differ in
    reps, a proxy problem, or cold versus hot timing. Every observation carries
    its fidelity pool (proxy and cache state); a config's calls are pooled only
    within one pool.
  - Each rung first races the survivors (`stats.Race`). The unit of
    replication is a measurement *call*, never a rep: most timing noise is per
    call (clock and thermal state, placement), so reps of one call are not
    independent samples, and treating them as such dropped equal configs most
    of the time. Calls in the same interleaved batch pair up (a paired t test
    on log ratios); otherwise Welch's t on the logs; a config with fewer calls
    is compared with a z test against the per-call noise the evaluator learned
    from configs measured more than once, or not at all. Every ordered pair is
    tested with Holm's correction (so choosing the leader from the same data
    does not inflate errors), at `alpha / rungs` because elimination repeats at
    every rung.
  - Each rung then keeps the best `1/eta`.
  - Hyperband cycles brackets from aggressive (many configs, cheapest rung) to
    conservative (few configs, full fidelity). Each bracket is seeded with the
    incumbent.
- **TPE** (`tpe`).
  - Observations are split at the `gamma` quantile into good and bad. Each
    parameter gets a smoothed density per side, counted only where the
    parameter is active.
  - Candidates are drawn from the good density, plus single moves of the best
    configs, and scored by log l/g plus a decaying prior term.
  - Batched asks treat pending configs as bad ("constant liar").
- **GP + expected improvement** (`gp`).
  - The model is a Matern-5/2 GP on log time, over ordinal positions and
    one-hot categoricals. The lengthscale is chosen by marginal likelihood from
    a small grid and refit every few asks.
  - Training uses at most 160 points: the best half plus the most recent.
  - With a prior, the mean function is `a + b * log prior`, fitted to the
    measurements, so the GP models only the selector's residual.
  - A prior's predictions are put on one scale for the whole search
    (`PriorScale`: log for times, linear otherwise), fixed once from the
    prior's `scale` attribute or its first full pool and kept in the strategy
    state; two sets of predictions of one prior are never on different scales.
  - Each proposal scores a stratified candidate pool. Its points are repaired
    with a small validity-check cap (`POOL_REPAIR_CHECKS`), past which a random
    valid genome replaces them: on predicate spaces where few genomes are valid
    (the universal-GEMM space) a full repair per pool point otherwise
    dominates a whole search's wall time.
- **Hill climbing** (`hill`). Each center runs three stages:
  1. Every single move.
  2. A combination re-sweep: every subset of two or more of the best
     near-neutral single moves (within `combo_tol` of the center, at most
     `combo_knobs` knobs), because knob effects interact.
  3. Ordinal two-step moves.

  The climber uses first improvement beyond `min_gain`, or beyond the learned
  noise when that is larger. After a center's stages it looks back over the
  last `lookback` centers, then restarts from the best *foreign* config that
  was never a center (a warm start or another strategy's find), then from a
  random unseen config. Its own rejected neighbors are never restart points:
  they are worse than an optimum it already climbed, and offline replay showed
  a climber that restarted from them staying in a poor basin for its whole
  budget. It moves to any better find from another strategy.

  Without a start, a solo climber first measures `n_init` stratified configs
  and climbs from the best of them; one random start in a space whose times
  span orders of magnitude can spend the budget walking out of a slow region.
  Inside a portfolio the climber skips this probe (`n_init=0`) and starts from
  the other members' finds. Warm starts beyond the first are measured right
  after the start and become restart points.
- **Portfolio** (`portfolio`, the default).
  - Warm starts are evaluated first, repaired when the space refuses them: the
    dispatcher's pick, the selector's top-k, nearest-problem and other-arch
    winners from the DB.
  - Then a discounted Thompson-sampling bandit gives each ask to the member most
    likely to improve the incumbent (the reward is an improvement beyond noise).
    Every arm's evidence decays on each update, idle arms included, so an arm
    productive long ago does not keep a stale, confident posterior. With
    `cost_aware` (off by default; tested on synthetic compile costs, not yet
    evaluated on recorded data) it ranks members by improvements per second,
    compile time included.
  - Every observation goes to every member.
  - The default members are `hill`, `gp` and `de`. Per-family defaults come
    from offline replay against recorded sweeps (the `replay` command); see
    "Choosing a strategy" below.
- `random` and `exhaustive` are the baselines and the oracle for small spaces.

### Choosing a strategy (`"auto"`, `FAMILY_DEFAULTS`)

`tune(..., strategy="auto")` (also `Autotuner(strategy="auto")`,
`search_configs`, the CLI) resolves the strategy with `recommend(family,
space_size, budget)`:

- a space no larger than the evaluation budget is enumerated (`exhaustive`),
  the only strategy that guarantees the optimum;
- otherwise `strategy="auto:<family>"` takes the family's entry of
  `FAMILY_DEFAULTS`, and a plain `"auto"` the default portfolio.

`FAMILY_DEFAULTS` also lists, per family, the warm-start sources worth
collecting, most useful first. The table was chosen by replaying every strategy
against recorded rocKE sweeps of each family (tile grids of the universal GEMM
and of the conv-wgrad implicit GEMM, permute / elementwise / reduction variant
spaces, CK contraction instance lists), several seeds per problem, with
recorded failures charged as failed evaluations and timing noise simulated
from repeated measurements in the same sweeps. The qualitative findings:

- Warm starts matter more than the choice of strategy. The same problem's
  winners in another dtype on the same arch, the winners of the nearest shapes,
  and the selector's top-k each put the first few evaluations close to the
  best config; combined with a search they need several times fewer
  evaluations than a cold start for the same result. Winners from another arch
  transfer much less well. The selector as a prior (GP mean or TPE term) helps
  about as much as its top-k as warm starts; a purely analytic latency model
  as the prior did not help.
- Cold, the default portfolio is never far behind the best single strategy at
  any budget replayed, so it stays the default. GP leads on spaces of a few
  hundred configs; TPE leads very early on the largest grids but plateaus; DE
  needs larger budgets. On the sparse conv-wgrad grids a solo climber started
  from its stratified probe led both cold and under noise, so that family
  defaults to `hill`; on small variant spaces (permute) the climber was the
  weakest, so it is not a general default.
- Successive halving / Hyperband over a rep ladder did not pay off: with
  realistic timing noise the cheapest rung ranks configs too poorly, and a
  measurement's fixed cost (load, cache flush, check) makes low rungs not much
  cheaper. Multi-fidelity is worth using when a rung is much cheaper, e.g. a
  proxy problem.
- Under simulated noise the strategies' own picks trail the best config they
  measured only slightly, and the final confirmation of the finalists closes
  part of that gap.

Priors: a `Prior` predicts a config's time ("lower is better", `nan` when it
has no prediction). `LightGBMPrior` loads a LightGBM text model, such as a
trained config-selector model, and `TablePrior` serves exported predictions.
`prior_ranked` scores a pool (the whole space when it is small) and then polishes
the top picks by greedy descent on the prior itself, which costs no measurements.
TPE and GP seed half their startup batch with the prior's picks and the other
half with a stratified sample, so a wrong prior cannot capture the search.

### Measurement rules (`Evaluator`)

A `Backend` implements three methods:

- `compile(config, key) -> CompileOutcome(ok, artifact, seconds, resources)`
- `check(config, key, artifact) -> (ok, why)`
- `measure(config, key, artifact, fidelity) -> per-rep times`

Optional members: `measure_batch(items, fidelity)`, which interleaves reps
across configs; `environment()`, JSON naming the toolchain and device the
backend compiles and times on; `recover()`, which drains or resets the device
after a timed-out call. A backend whose ladder has proxy rungs takes the
fidelity as an extra argument of `compile` and `check` (a proxy is another
problem with its own specialization, launch grid and numeric check); the
evaluator detects this and refuses a proxy ladder for a backend that does not.
The evaluator guarantees the following:

- **Mandatory numeric check, per artifact.** A backend without `check` is
  refused. Every artifact compiled in the current process is checked before its
  first timing, whatever an earlier process (or cache) concluded about the same
  config, so a config that a new compiler or rocKE build miscompiles can never
  be timed. A config that fails is `numeric_fail` with value `inf`, so it is
  never fast and never cached as good.
- **Failures are outcomes.** `compile_error`, `spill` (any of `scratch_bytes`,
  `spill_vgpr`, `spill_sgpr`, `private_segment_size` above 0 when
  `reject_spill`), `timeout` and `run_error` all carry value `inf`. Structural
  failures (compile error, spill, numeric failure) hold at every fidelity of the
  same problem; a cached one answers an ask at any rung, relabelled as that
  rung (a strategy waiting for a full-fidelity answer gets one). Transient
  failures (timeout, run error) are remembered for one fidelity and one process
  only.
- **One failure does not fail a batch.** When an interleaved batch raises, its
  configs are re-measured one at a time before any is blamed. A batch of `n`
  configs gets `n` times the per-config measure limit.
- **Persistent cache, scoped to the environment.** `cache_path` persists
  timings and structural failures. Its fingerprint covers the space, the
  fidelity ladder and the environment: a hash of the rocKE emitter sources, the
  comgr library (path, size, ROCm version), the codegen environment variables,
  the device (target id, name, CU count, host) when HIP is loaded, and
  `backend.environment()` plus the `environment` argument. A cache from another
  toolchain is dropped; one from another machine keeps its structural failures
  and drops its timings. Check verdicts and transient failures are never
  persisted. Compile seconds are recorded on every observation and counted
  against a seconds budget; `Evaluator.needs_compile` tells a cost-aware caller
  which configs would pay a compile.
- **Paired, noise-aware timing.** One round's fresh measurements at one fidelity
  go through `measure_batch` when the backend has it, so their samples are
  paired by round (`Observation.batch`). The `NoiseModel` learns two
  components: the rep-to-rep spread within a call, and the call-to-call spread
  of a config's medians (from configs measured more than once: rungs of one
  pool, confirmation rounds). Strategies use the noise of one call's median as
  their improvement threshold; until the call component is learned it is
  assumed as large as the rep component.
- **Timeouts are a last line of defence.** With `compile_timeout` /
  `measure_timeout` set, backend calls run on one dedicated daemon worker
  thread (the same thread for every call, so thread-local GPU state such as a
  current stream persists; a backend sets its own stream there). A call over
  its limit cannot be killed in process and is abandoned. After a check or
  measure timeout the evaluator is *dirty*: kernels may still be in flight and
  would contaminate later timings, so it calls `backend.recover()` and measures
  nothing more until the abandoned call has ended; `run_search` stops a dirty
  search (`stopped="evaluator_dirty"`). Backends that can hang (compilers,
  launches) should run them in a subprocess with their own hard limits.

`run_search(confirm=k, baseline=genome)` re-measures the `k` best distinct
configs and the baseline (the dispatcher's default pick) together, for
`confirm_rounds` interleaved rounds; the round is the unit of replication. The
winner is the best geometric mean of the per-round medians. This removes the
winner's curse, the low bias of the best of many noisy measurements; it also
reports which finalists are statistically indistinguishable from the winner,
and the winner's paired speedup over the baseline with a one-sided p-value (a
paired t test on per-round log ratios). Configs without a live artifact
(measured before a resume, served from a cache, or never measured, like a
baseline) are compiled and checked again before they are timed.

### Budgets, determinism and resume

`Budget(evaluations=, seconds=, cost=)` limits are combined: the first limit
reached stops the search.

- **evaluations** counts fresh measurements. Cache hits and configs the space
  refuses are free.
- **cost** sums the `Fidelity.cost` of fresh measurements.
- **seconds** is wall time, compiles included, so a time-limited run is not
  reproducible. The deadline is also passed into each round, so a large batch
  stops starting new configs at the limit.

The confirmation is paid from the budget: `(k + baseline) * rounds`
evaluations are reserved under an evaluation budget, their cost under a cost
budget and their estimated time under a seconds budget. A budget too small to
leave `k` search evaluations skips the confirmation and says so
(`confirmed["skipped"]`). Without a ladder, `tune` gives the multi-fidelity
strategies (by name, or an instance with `fidelities`) the default rep ladder.

With an evaluation budget and a deterministic backend, a run is reproducible
from its seed. A run split by `checkpoint=`/`resume=True` (strategy state,
evaluator cache, spent budget, and the trials the budget cut, which are
evaluated first on resume) makes the same decisions as one run of the total
budget. The tests check this for every strategy. A checkpoint is refused for
another space or strategy.

### Results: identity, tuning DB, traces

`tune_candidate(candidate, request, backend, db=...)` searches a dispatch
candidate's own space:

- `candidate.search_space(request)` comes from `make_tuned_candidate`, so every
  measured config is a spec dispatch can serve.
- Warm starts come from the dispatcher's pick and from the DB: the slot's
  current record, the nearest problems by log2 feature distance, and other
  archs' winners for the same problem.
- The dispatcher's default pick is the confirmation baseline (see above).
- The winner is recorded as a `TunedRecord`: family, candidate, arch, problem
  key, `tuning_id` (`stem@config_key`, with the defaults fingerprint inside),
  knobs, the confirmed value with its sample count and spread, the paired
  speedup over the default with its p-value and round count, the toolchain and
  machine fingerprints, strategy, evaluations, seed, and trace path.

`TuningDB` never compares raw times from different sessions, nodes or clock
states. `put` stores a record when the slot is empty, when it re-measures the
same config, when the slot's record was made with another toolchain, when the
caller found the slot's record stale (its knobs no longer canonicalize to its
key; `tune_candidate` checks this), with `force`, or when its own paired
speedup is significant and larger than the old record's. `serve(request,
...)` pins a request only to a *significant* record (speedup above
`min_speedup` with p below `alpha`): a pick within noise of the default is
recorded but never served. Every `put` re-reads, merges and writes the file
under a lock file, so concurrent tuners sharing a DB lose nothing.

`TuningDB.pin(request, record)` returns the request pinned to the record's
algorithm, spec_id, tuning_id and knobs. The candidate replays the knobs, and
refuses the pin when they no longer reproduce the key. A stale record is
reported, never substituted.

`TraceWriter` writes one JSON line per observation: config, status, value,
samples, compile and measure seconds, cached flag, and the incumbent after it.
It writes a header line first and an end line last.

### Integration points

- **Autotuner.** `Autotuner(configs, ..., strategy="portfolio", budget=24,
  check_fn=...)` searches the config list (specs flattened to fields) instead of
  timing all of it. `check_fn` is required in this mode. The winner is cached as
  before.
- **Sweeps.** `rocke.sweep.search_instances(specs, measure=..., check=...,
  strategy=..., budget=...)` is the searched counterpart of building and timing
  every instance. `search_configs` does the same for any config list.
- **Dispatch.** `tune_candidate` (above). `KernelCandidate.search_space` is the
  hook a family gets for free from `make_tuned_candidate`.
- **CLI.**
  - `python -m rocke.dispatch.tuning.search strategies` lists the strategies.
  - `demo` runs a strategy on a synthetic problem with a known optimum.
  - `replay` scores strategies on a recorded sweep CSV: anytime curves, the
    fraction of the oracle, and the evaluations needed to reach 95/98/99%.
    `--status-col` keeps recorded failures (they cost an evaluation) and marks
    early-stopped rows as lower bounds that are never the oracle;
    `--noise` / `--call-noise` simulate per-rep and per-measurement noise.
  - `tune --factory module:function` runs a family-provided space and backend
    with a trace, checkpoint, cache and DB.
  - `show` prints a DB or summarizes a trace.
- **hipTensor autotune mode.** hipTensor reaches rocKE only through C++. All the
  data the loop exchanges is language-neutral JSON: knob dicts, `tuning_id`,
  `TunedRecord`, traces and checkpoints. Two integration shapes work without
  changing the engine:
  1. hipTensor runs an offline tune through the CLI or `tune_candidate` and
     ships the resulting DB, and its plan code looks up `(family, arch,
     problem)` and serves significant records only.
  2. hipTensor drives the ask/tell loop in-process from C++ (next section),
     first in config-list mode.

### C++ port of the ask/tell loop (design note)

The loop is host-side and portable; the hard part is not the loop but
*validity and identity*. In Python they come from each family's
`KnobSpace.canonicalize` and build, plus family gates such as the universal
GEMM's `universal_spec_supported` and `launch_shape_error`. A C++-only host has
three ways to get them, in order of cost:

1. **Config-list mode (the first hipTensor integration).** hipTensor already
   enumerates its candidate instances. Each instance becomes one row of
   scalar fields plus its canonical key, and the space is
   `SearchSpace.from_configs` semantics: only listed rows are valid, repair
   snaps to the nearest listed row, keys are the rows' own. No family validity
   logic is ported; the instance list *is* the validity rule. This needs only
   the generic space code below.
2. **A precomputed config list from rocKE.** For a rocKE family, Python
   enumerates (or samples) the canonical configs offline and ships them with
   their `config_key`s; the C++ host runs config-list mode over them.
3. **A full port** of a family's canonicalization and gates, or an embedded
   Python. Only worth it for families tuned online in C++ over spaces too large
   to list.

Canonical serialization (what makes keys agree across languages). Keys are
`stable_json_hash(payload)`: SHA-256 of `json.dumps(payload, sort_keys=True,
separators=(",", ":"))`, hex, first 16 characters (config keys of list spaces
are the compact JSON itself). A C++ writer must reproduce Python's JSON
exactly:

- objects with keys sorted by code point, no whitespace, `","` and `":"`
  separators; arrays in order; strings with Python's escaping (`\"`, `\\`,
  `\n`, `\t`, other controls as `\u00XX`, non-ASCII as `\uXXXX` with surrogate
  pairs, since `ensure_ascii` is on);
- `true` / `false` / `null` for booleans and None; a boolean knob serializes
  as `true` / `false`, never as `1` / `0`;
- integers in plain decimal; floats in Python's shortest round-trip `repr`
  (`1.0`, `0.1`, `1e-05`, `1e+16`), never in C's `%g`; NaN and infinities are
  not allowed in a key payload;
- knob payloads only hold JSON scalars (`knob_items` enforces it).

A conformance file of payloads and expected hashes, generated by Python, is
the test both sides run.

Proposed C ABI (opaque handles, plain structs, JSON at the edges):

```c
typedef struct rocke_search_space rocke_search_space;  /* params + validity */
typedef struct rocke_search_strategy rocke_search_strategy;

typedef struct {               /* one asked config */
  const int32_t *genome; int32_t dims;
  const char *uid;             /* owned by the strategy until tell */
  int32_t fidelity;            /* ladder index, -1 = full */
} rocke_search_trial;

typedef struct {               /* one observation */
  const char *uid; const char *key;   /* key NULL when refused */
  int32_t status;              /* ROCKE_SEARCH_OK, _COMPILE_ERROR, _NUMERIC_FAIL, ... */
  double value;                /* median time; +inf unless OK */
  const double *samples; int32_t n_samples;
  int32_t fidelity; int32_t full; int32_t batch; int32_t cached;
} rocke_search_observation;

typedef struct {               /* backend callbacks (the host's code) */
  void *user;
  int (*compile)(void *user, const char *config_json, const char *key,
                 int32_t fidelity, void **artifact, double *seconds);
  int (*check)(void *user, void *artifact, int32_t fidelity, char *why, size_t n);
  int (*measure_batch)(void *user, void *const *artifacts, int32_t n,
                       int32_t fidelity, double *samples /* n x reps */);
  int (*canonical)(void *user, const char *config_json, char *key, size_t n);
} rocke_search_backend;

int rocke_search_space_from_json(const char *params_json, rocke_search_space **out);
int rocke_search_space_from_configs(const char *rows_json, rocke_search_space **out);
int rocke_search_strategy_create(const char *name, rocke_search_space *space,
                                 uint64_t seed, const char *options_json,
                                 rocke_search_strategy **out);
int rocke_search_ask(rocke_search_strategy *s, int32_t n,
                     rocke_search_trial *trials, int32_t *n_out);
int rocke_search_tell(rocke_search_strategy *s,
                      const rocke_search_observation *obs, int32_t n);
int rocke_search_state_json(rocke_search_strategy *s, char **json);  /* free with rocke_search_free */
int rocke_search_load_state_json(rocke_search_strategy *s, const char *json);
void rocke_search_free(void *p);
```

The evaluator rules (check every artifact before timing, structural vs
transient failures, fidelity relabelling, batch isolation, the environment
fingerprint) live in the host's driver, which owns the backend; the strategy
side never calls the backend. State JSON carries a schema version
(`CHECKPOINT_VERSION`, `CACHE_VERSION`, `DB_VERSION`); a C++ checkpoint loads
into Python and back, so a Python replay can audit a C++ run.

Strategies to port first: `HillClimb` and DE need only an RNG, integer genomes
and the history map; TPE needs counting; the GP needs a small Cholesky.
Python's `random.Random` is MT19937 with a SHA-512-seeded state while
`std::mt19937` with `seed_seq` seeds differently, so a port either ports the
Python seeding (bit-identical decisions) or uses its own RNG and is validated
statistically.

Keep the ask/tell boundary as it is: strategies never call the backend. The
same strategy object then runs live, against a recorded table, or inside a
foreign host.

### Guidance for live backends

- **Warm-start GEMM searches from the dispatcher.** The universal-GEMM space
  (classic tile grid plus the dispatcher's body levers: direct-to-LDS rings,
  interleave and AGPR bodies, chiplet groups, `iglp_opt`, persistent CTAs) is
  large and its best bodies sit in a narrow corner. Start from the dispatcher
  pick and its sweep neighbourhood (what `tune_candidate` does, or
  `WarmStart`s); a cold search finds the right bodies but needs a much larger
  budget to find the lever combinations on top of them. The final interleaved
  confirmation keeps the dispatcher pick unless a searched config beats it
  beyond noise.
- **The numeric check is load-bearing.** A fast wrong kernel is the likeliest
  winner of an unchecked search, so every evaluated config is checked against
  a full-output reference. The check is how configs that the universal-GEMM
  gates admitted but that computed wrong results (a `direct_to_lds` body
  launched with a partial K tile, `pad_k` set) were identified; the launch
  contract (`launch_shape_error`) now refuses them.
- **Backends need structural pre-filters and hard compile limits.** Configs
  whose accumulator tile cannot fit the register file spill, never win, and own
  the compile long tail (long compiles with large compiler memory). A live
  backend should refuse them in the space, and compile in a worker with a
  wall-clock and memory limit.
- **With a disk compile cache, measurement and strategy time dominate a
  search**, hence the GP pool repair cap above.
- **Family-neutral.** The same `tune` / `run_search` path tunes any family
  whose space it can see, e.g. permute variant x knob lists with
  `auto:permute` (GP).
- **The dispatch-serving path, end to end.** On a tuned candidate,
  `KnobSearchSpace` -> `tune_candidate` (warm-started from and confirmed
  against the candidate's default) -> `TuningDB.put` -> `TuningDB.serve` ->
  `dispatch_*` -> `DispatchResult.bind_torch` -> launch selects exactly the
  recorded `tuning_id`.
- **Search-time measurements.** Racing and halving need several calls per
  config; a backend that times single, unbatched calls only exercises the
  final interleaved confirmation's statistics.

### Testing

The tests live in `platform/tests/dispatch/dispatch_tests/tuning/test_search_*.py`
and `library/tests/dispatch/attention/test_tuning_search.py`. They cover:

- space semantics (gating, keys, distance, repair optimality against brute
  force, the structural sampler);
- the statistics against reference values, including the false-elimination rate
  of racing under the null;
- every strategy on synthetic problems with known optima, compared with random
  search;
- determinism under a seed, budget accounting, resume equality for every
  strategy, and JSON round-trips;
- numeric failures never winning, spills, timeouts, the persistent cache and
  interleaved pairing;
- the DE adaptation and gated crossover, the hill climber's combination
  re-sweep and lookback, prior seeding, halving and racing, portfolio allocation
  (plain and cost-aware) and warm starts;
- the dispatch round trip on a toy family and on attention's real spaces (keys
  are `config_key`s, every production spec is reachable, the searched winner
  pins back);
- the Autotuner and sweep hooks, the CLI, and a torch-free import;
- config-list spaces (exact nearest-member repair against brute force, random
  tie-breaking, member sampling, the dense/sparse repair order), the replay
  scorer (recorded failures, lower bounds, ideal versus incumbent curves, cost
  budgets, the status column) and `recommend` / `strategy="auto"`;
- failure handling: stale caches (another toolchain, another machine,
  re-checks per process, transient failures), mixed-fidelity portfolios with
  failing configs (no deadlock), repair symmetry and random tie-breaking,
  `choices`-axis geometry (also on attention's real spaces), batch-timeout
  isolation, dirty-evaluator behaviour and the single worker thread, proxy
  fidelities, prior-scale consistency, the DB's serving policy, staleness and
  concurrent writers, confirmation budgets under every budget kind, and Race's
  false-elimination rate under per-call noise, a data-chosen leader and
  repeated looks.

`test_search_edge_cases.py` pins exact boundaries and formulas (comparison
directions, off-by-ones, constants, accumulations) so that a mutation of
the search package's operators fails a test.
