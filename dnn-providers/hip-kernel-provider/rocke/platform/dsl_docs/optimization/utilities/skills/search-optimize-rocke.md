---
name: search-optimize-rocke
description: >
  Optimize a rocKE kernel by searching its knob space with memory: default-off
  levers as knobs, a staged search with lookback and combination re-sweeps, a
  measurement gate (numeric check, interleaved timing, paired confirmation, A/A
  control), warm starts from the dispatcher pick, and a per-shape-class view of
  the result. Use when single-lever moves stall, when the space is too large to
  enumerate, or when asked to tune or sweep a kernel family.
  Usage: /search-optimize-rocke
allowed-tools: Read Edit Bash Grep Glob Agent
---

# Search-driven optimization (rocKE)

This skill adds the search method to the loop in
[`optimization_runbook.md`](../../optimization_runbook.md) (Step 0, The Loop,
"Search with lookback"). Read those first; this page does not replace them. The
search engine is `rocke.dispatch.tuning.search`; its API and the choice of
strategy are in [`autotune/overview.md`](../../../autotune/overview.md#search-engine-rockedispatchtuningsearch).

## Principles

1. **Levers are knobs.** Each idea is a default-off spec field. A lever that loses
   alone stays registered: it can win in combination or on a new base.
2. **Exhaust the existing space first (Step 0).** Include the default-off levers.
   A curated preset benchmark is not the full space.
3. **Search with memory.** Single moves, then every subset of the near-winners,
   then two-step moves, with a lookback stack. Re-sweep everything after a
   structural change.
4. **A number has to earn its place.** Numeric check before any timing, paired
   confirmation against the baseline, an A/A control for the noise floor.
5. **The LLM proposes levers; the engine searches them.** Use the ISA and
   profile to invent knobs; use the engine to find where they pay.
6. **A searched winner is not a shipped kernel.** See [Shipping](#shipping).

## Procedure

### 1. Orient
- Read the family's builder, spec dataclass, dispatcher candidates and their
  knob axes (`library/dispatch/<family>/`). Note what the dispatcher's sweep can
  and cannot express (a single-knob sweep never sees interactions).
- Check which levers are legal on the target arch: `is_valid_spec(spec, arch)`
  gates levers per arch.
- Smoke-check the engine without a GPU: `python -m rocke.dispatch.tuning.search
  demo --strategy portfolio`.

### 2. Lever ledger
Keep one table per kernel and shape class: lever, values, default, legality,
status (untested, lost, won, combination) and the base it was measured on. Update
it every round. A "lost" row carries the base it lost on.

### 3. Harness
Build a `Backend` around the real kernel; the engine enforces the rules around
three calls:

```python
from rocke.dispatch.tuning.search import (
    Budget, CallableBackend, CompileOutcome, Fidelity, Param, SearchSpace, WarmStart, tune,
)

space = SearchSpace(
    [Param.ordinal("tile_m", (64, 128, 256)), Param.boolean("lds_swizzle"),
     Param.categorical("pipeline", ("mem", "compv3", "compv4"))],
    is_valid=lambda c: spec_is_legal(c),        # geometry + is_valid_spec
)

def compile_(config, key):                      # -> CompileOutcome
    try:
        return CompileOutcome(True, artifact=build_and_bind(config))
    except Exception as e:                      # a compile error is a result
        return CompileOutcome(False, detail=f"{type(e).__name__}: {e}")

def check_(config, key, artifact):              # independent oracle -> (ok, why)
    out = run_once(artifact)
    return max_abs_error(out, oracle_rows) <= tol, "max_abs=..."

def measure_(config, key, artifact, fid):       # per-rep times (one unit)
    return [time_launches(artifact.launch, warmup=0, iters=20) for _ in range(fid.reps)]

result = tune(
    space, CallableBackend(compile_, check_, measure_),
    strategy="hill", budget=Budget(evaluations=120), seed=0,
    warm=[WarmStart(dispatcher_pick_config, "dispatcher")],
    baseline=space.encode(dispatcher_pick_config),
    confirm=3, confirm_rounds=5,
    ladder=(Fidelity(reps=8, warmup=3),),
)
print(result.confirmed["speedup"], result.confirmed["p_value"])
```

- The oracle is independent (fp32 or fp64 on sampled rows) at a declared dtype
  and tolerance. Byte-identity, compilation or a successful launch is not a
  numeric check.
- Build the baseline from what the dispatcher ships for that arch. A baseline
  that fails to build, or silently omits an arch-conditional lever, inflates
  every win.
- Add an A/A row (the same spec built twice under different names) to any
  exhaustive run. It shows the noise floor.
- For a dispatch candidate, `tune_candidate(candidate, request, backend_factory)`
  uses the candidate's own knob space, the dispatcher default as warm start and
  baseline, and records the winner in a `TuningDB`.
- Report `Observation.detail` for every non-ok status; an opaque `compile_error`
  hides an unsupported arch or a validity gap.

### 4. Choose the strategy by space size
- Space no larger than the evaluation budget: `exhaustive`; `strategy="auto"`
  picks it.
- Larger: the default `portfolio`, or `hill` on sparse or rugged grids. `gp` leads
  on a few hundred configs; differential evolution needs larger budgets;
  successive halving pays only when a rung is a much cheaper proxy problem, not
  merely fewer reps.
- Always pass `baseline=` so the final confirmation reports a paired speedup.

### 5. Lookback discipline
1. Single moves from the base.
2. Every subset (two or more) of the moves within about 1% of the base or better.
3. Two-step moves on ordinal knobs.

Accept the first improvement beyond noise, and push the old base on the stack.
When a base is exhausted, revisit the stack, then a foreign winner, then a random
unseen config. After a structural lever lands, re-sweep the lost levers.

### 6. Propose new levers
Inspect the ISA of the current best (see
[`isa-inspection-rocke.md`](isa-inspection-rocke.md)): spills, `s_waitcnt`
placement, LDS bank conflicts, occupancy cliffs, instruction mix per iteration.
Turn each hypothesis into a default-off spec field with `__post_init__`
validation, mirrored in the C++ engine (the byte-identity rule), then return to
step 3. A lever set only through a kernel attribute outside the spec is a
prototype: the packager and the dispatcher build from the spec and never see it.

### 7. Per shape class
Search per class (decode, short and long prefill, causal, windowed, GQA ratio),
not per shape. Keep the best per class and run the full production cohort before
claiming anything; a smoke set under-samples the variance.

### 8. Record
- A replayable case study next to the kernel (`*case_study.md`), losses included.
- Repo and PR text carry relative ratios only, with the numeric-gate statement
  (see `platform/AGENTS.md`, "Public software-performance evidence"). Absolute
  latency and throughput stay in the approved internal record.

## Shipping

A searched winner is a measured result. To reach a consumer it becomes a changed
default, or a variant or selection rule where the consumer picks kernels. In
hipDNN, kernels are packaged from descriptors at build time and the
Python/C++ dispatcher is not in the serving path; a tuned configuration ships
only as a changed rocKE default, an added descriptor variant, or a matcher rule.
Every shipped variant is a code object, so:

- Prefer a winner that can become the default over one that needs its own variant.
- A knob that does not change the shape-independent binary (a scheduling hint, an
  occupancy hint) multiplies binaries only by the values you ship, not by the
  number of shapes. Tile geometry and anything that bakes shape into the code
  object multiplies instances.
- Search wide, then prune to the smallest set of variants that keeps the gain.
  Tell the owners of the consumer if per-shape winners start to multiply
  instances.

## Failure modes seen
- A preset-based sweep hid default-off levers that a raw-lever sweep found.
- A fixed arm order with best-of-N selection produced a phantom win; shuffle arms
  every round and re-time finalists.
- A compile cache keyed only by the spec served stale binaries after an emitter
  change; key it by the emitter sources too.
- Per-stage timing stamps biased the timing; confirm with them off.
- A one-knob "rejected" verdict on an old base is not a verdict on the new base.
- A config can pass `is_valid_spec` and still fail the numeric gate; the gate is
  what keeps it out of the timings.
