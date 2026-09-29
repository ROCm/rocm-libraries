# Attention dispatcher — agent guide

This folder owns the **path-level dispatch** for unified attention: which kernel
family (2D tiled prefill vs 3D split-KV decode) and which specialized candidate
handles a given `AttentionRequest`.

## What the dispatcher decides (and what it does NOT)

**Decides:** kernel path (`"2d"` or `"3d"`), candidate name, algorithm tag,
and spec identity `(path, head_size, block_size)`.

**Does NOT decide, on the unified path:** CTA geometry (`num_warps`, `tile_size`),
`num_segments`, `waves_per_eu`, or any other performance knob. Those live in
`builders/common/attention_spec_builder.py` and
`kernels/common/attention_unified.py`. The parity identity with C++ is
`(path, head_size, block_size)` only — C++ reads `num_segments` as a parameter
passed from Python, it does not recompute it.

**Standalone candidates are a bounded exception.** A candidate that owns its own
kernel module builds that kernel's own spec here, tuning included:
`gfx950_dense.py::_base_spec` resolves tile geometry from the frozen variant plus persist /
wide-DMA, and `gfx942_dense.py::_base_spec` resolves those plus `waves_per_eu`. Those specs
are consumed only by their own builder and never enter the C++ parity identity. One
rule governs the exception: **any value the kernel bakes into its `kernel_name` must
be resolved into the concrete spec before build**. The default must come from the
kernel's policy function; a swept knob value may replace it only when the
body, symbol, and cache all read that same spec field. `gfx942_dense.py` calls
`kernels.gfx942.attention_dense._tuned_waves_per_eu` for the default, and the
sweep's WPE loop varies it from there. The gfx942 symbol always carries WPE;
gfx950 appends it when non-default. This keeps the emitted attribute and identity
in lockstep for runtime caches and AOT packaging.

The launcher cache itself is keyed by `attention_dense_cache_key`, not by the symbol
name, so the name is not a backstop for this — on the gfx950 runtime-shape path the
symbol carries no `sq`/`sk` tokens at all (batch and the seqlens are runtime kernel
params). Correctness rests entirely on the key.

## Candidate registry — priority table

| priority | candidate | declared arches | module | scope |
|---|---|---|---|---|
| 3 | `attention_gfx942_dense` | gfx942 | `gfx942_dense.py` | bf16/fp16 D64/D128 dense prefill; grid and persistence are knobs (`spec_id=gfx942_dense`, opt-in) |
| 3 | `attention_gfx950_dense_grid_default` | gfx950 | `gfx950_dense.py` | dense grid, 256×64 tile (`spec_id=gfx950_dense_grid_default`, opt-in) |
| 3 | `attention_gfx950_dense_persist_default` | gfx950 | `gfx950_dense.py` | dense persist, 256×64 tile (opt-in) |
| 3 | `attention_gfx950_dense_persist_widedma_default` | gfx950 | `gfx950_dense.py` | persist + wide-DMA, 256×64 tile (opt-in) |
| 3 | `attention_gfx950_dense_grid_bm128` | gfx950 | `gfx950_dense.py` | dense grid, 128×64 tile (opt-in) |
| 3 | `attention_gfx950_dense_persist_bm128` | gfx950 | `gfx950_dense.py` | dense persist, 128×64 tile (opt-in) |
| 3 | `attention_gfx950_dense_persist_widedma_bm128` | gfx950 | `gfx950_dense.py` | persist + wide-DMA, 128×64 tile (opt-in) |
| 5 | `attention_gfx942_dense_pipe` | gfx942 | `gfx942_unified.py` | fp16 2D prefill flash |
| 5 | `attention_gfx950_d256` | gfx950 | `gfx950_unified.py` | bf16 D256 2D prefill |
| 5 | `attention_gfx1250_wmma` | gfx1250 | `gfx1250.py` | fp16 WMMA FMHA forward (opt-in only) |
| 5 | `attention_d256_decode` | gfx942, gfx950 | `generic.py` | bf16 D256 3D decode |
| 10 | `attention_unified_2d` | all | `generic.py` | generic 2D prefill fallback |
| 10 | `attention_unified_3d` | all | `generic.py` | generic 3D decode fallback |
| 30 | `attention_gfx{942,950}_u{2d,3d}_*` | one arch each | `gfx942_unified.py`, `gfx950_unified.py` | explicit geometry/codepath candidates; sweep/opt-in only |

Lower priority number = higher precedence. Generic candidates (10) remain the
fallback for everything a specialized candidate does not claim.

The priority-30 tuning candidates are generated from
`GFX942_TUNING_VARIANTS` and `GFX950_TUNING_VARIANTS` (geometry: codepath,
tile, warps, rows per warp, segments, compile backend). Every other tuning
field on the tiled kernel specs is a declared `KnobAxis` in `axes.py`
(`_GFX950_2D_AXES`, `_GFX942_2D_AXES`, `_3D_AXES`); `test_tuning_space.py`
fails if a kernel field is on no axis. Axes are ordered prerequisites-first,
and the space is walked depth-first with the kernel's own spec validator
pruning illegal prefixes. There is no size cap: the full space is millions of
specs per shape on the transposed paths, so `sweep_space` is a lazy stream and
`sample_space(req, n, seed)` draws `n` random legal specs per candidate. Held
out on purpose: `KNOWN_WRONG_KNOBS` (gfx942 `use_k_hbm_direct`). They reject
any request that does not pin both `algorithm="unified_tuning"` and their own
`spec_id` before constructing a spec, so normal dispatch does not pay the
enumeration cost and the historical winner is unchanged. Sweeps probe them
through `opt_in_probe` and execute the returned `AttentionTuningSpec`, which
also owns its build, cache key, and launch grid.

Production `dispatch_attention` uses `ATTENTION_ROUTE_REGISTRY` (path labels
plus pin-able specialized candidates). `registered_attention_combos` /
`dispatch_attention_all` use `ATTENTION_EXECUTION_REGISTRY`, which requires
`build` and `bind_torch` on every candidate.

Policy-free spec construction lives next to its consumers in
`dispatch/attention/tuning_specs.py`. It is shared by the gfx942/gfx950 tuning
candidates and never calls `_select_*`,
`_enable_*`, `_num_segments`, or `_resolve_lds_budget`; problem semantics are
derived from `UnifiedAttentionProblem`, while explicit geometry/codegen points
are accepted or rejected by the concrete spec and `supports_tiled_*` validators.
This is deliberately separate from the heuristic production builders.

Four candidate families are **opt-in only** and never win under `algorithm="auto"`:
`attention_gfx942_dense`, every `attention_gfx950_dense_*` variant, and
`attention_gfx1250_wmma`, plus every priority-30 unified tuning candidate.
Registering a kernel makes it reachable; making it an
arch's default is a separate decision that wants benchmark evidence, so none
of them silently displaces the unified path its arch routes to today.

### Dense and unified share one selection model

A dense or unified tuning spec is an `AttentionTuningSpec` (path `dense`, `2d`
or `3d`): a registered variant plus a canonical knob dict, named by
`tuning_id = {variant_id}_wpe{N}@{config_key}`. `config_key` hashes an explicit,
versioned payload (arch, path, variant, knobs) and is the same on every problem;
the platform `ARCHITECTURE.md` section 11.1 has the full contract. Selection
pins a spec on `AttentionRequest`:

1. `algorithm` + `spec_id` name the candidate; an opt-in candidate admits a
   request only when **both** match. The algorithm values are per family:
   `attention_dense` for every dense variant, `unified_tuning` for every
   unified tuning geometry, `wmma_attention_fwd` for the gfx1250 WMMA kernel.
2. `tuning_knobs` (the knob dict recorded next to the id) rebuilds
   the configuration directly; it must reproduce `tuning_id` unless
   that is `auto`.
3. Otherwise `tuning_id`: `auto` is the variant's default spec; any
   other id is looked up in the production set only (a full-space id needs
   its knobs). Ids are matched on their `config_key`, never the display stem.

A pin that no longer resolves raises `rocke.dispatch.core.PinRefused` with the
candidate's reason and never falls back to another kernel.

There is no auto policy for dense: an unpinned request routes to the unified
path, and nothing picks a tile, persistence or wide DMA on the caller's
behalf. `attention_tuning_spec(req, spec_id, tuning_id, knobs)` and
`tuning_spec_with_knobs(req, spec_id, knobs)` in `dispatch.attention` are the
two helpers callers use; both go through dispatch, so they return exactly what
a pinned request selects.

Every spec a tuning candidate hands out comes from one canonicalize step,
the shared `rocke.dispatch.tuning.KnobSpace.canonicalize` (attention's
subclasses are `DenseSpace` in `dense_rules.py` and `UnifiedSpace` in
`unified_rules.py`), whether it was swept,
sampled, pinned by knobs, or found by id. Knobs that compile to the default
are dropped (default values, restated policies, knobs the body does not read);
illegal ones are refused with the reason (`KNOWN_WRONG_KNOBS`, fields on no
axis, codepath-fixed knobs, kernel-validator rejections). So one kernel has
one id, and a knob pin cannot reach a configuration a sweep would not.

gfx950 dense is six frozen `(tile × persist × wide-DMA)` variants. The gate is
the kernel's own `supports_attention_dense`: dispatch adds no eligibility rule
of its own, so wide DMA is offered on every shape the kernel accepts.

Each dense variant's `sweep_space` / `sample_space` walks the gfx950 dense knob
space declared as `_GFX950_DENSE_AXES` in `axes.py` (registered as
`("gfx950", "dense")`): K/V LDS pads, lazy rescale and its threshold, the PV
scheduling knobs (`iglp_mode`, the fence and its mask, the sched_group
template and its DS-read count, each on its own axis), exp2 and PV-loop
codegen, the O store width, `num_persistent` (symbolic policies such as
`gqa_pair` and CU multiples, resolved per problem), `persist_decode`, and
`interleave`. `production` sets every applicable knob to each legal value one
at a time from the default spec, pairing a value with its prerequisite when it
is illegal alone; `full` walks or samples the pruned product. The kernel's
spec validator is the only legality gate. An explicit value equal to its
policy, or a knob the body does not read for that spec, re-emits the same IR
under a new symbol, so `_dense_redundant_knob` prunes it. The field-coverage
test in `test_tuning_space.py` classifies every `Gfx950AttentionDenseSpec`
field as problem, variant, WPE loop, untunable (`lds_num_buffers`), or swept.

gfx942 dense uses the same walk over `_GFX942_DENSE_AXES`
(`("gfx942", "dense")`). It registers one candidate, so geometry
(`persistent`, `block_m`, `block_n`) is swept rather than fixed by a variant,
and the persistent-only knobs are pruned per spec. Its knobs are the K row and
V row pads, the D64 K group pad, the conflict-free-V store and its swizzle,
exp2, `iglp` / `iglp_mode`, the PV fence mask, `pv_priority`,
`pv_loop_order`, the bf16 O store width, and `causal_diag_split`. The
LDS-saving knobs lead as enablers so an over-budget tile can still reach the
setting that fits. Untunable here: `lds_num_buffers` (only 1 is implemented)
and `lazy_rescale` (the body never reads it).
`registered_attention_combos(req)` is the multi-engine bench
entry: it probes `ATTENTION_EXECUTION_REGISTRY` for `req.arch` and flattens each
candidate's `sweep_space` (dense, WMMA, and unified tuning). Routing-only
unified path labels are omitted.

**Tier 3 is reserved for opt-in candidates.** Because they outrank every other
tier, that opt-in check is the only thing keeping them off the default path — a
tier-3 candidate whose `support()` forgets it would silently claim all traffic
for its arch. `Capability` cannot express the check (it constrains the request's
*selector*, not its shape), so it stays in the predicate and every tier-3
candidate has to carry it.

## Layout

One module per architecture, each owning its candidates and exporting
`register(route, execution)`: routing labels go on the route registry, anything
with `build` and `bind_torch` on the execution registry too. `__init__.py`
holds the registry assembly (one loop over the modules) and the entry points. gfx942 and gfx950 are split by kernel family:
`gfx{942,950}_dense.py` own the standalone dense kernel's candidates, and
`gfx{942,950}_unified.py` own every candidate that runs the unified tiled
kernels (gfx942 `dense_pipe`, the gfx950 D256 fast path, and each arch's
priority-30 tuning catalog). Both dense modules build their candidates with
`candidate.make_dense_candidate`, and both unified catalogs with
`candidate.make_tuning_candidate`; an arch module supplies only its base spec,
validator, variants and features. The family-neutral machinery (axis
helpers, the walk and sampler, `config_key` / `tuning_id`, `KnobSpace`,
`make_tuned_candidate`, the conformance kit) is `rocke.dispatch.tuning` in the
platform; attention keeps only its data and rules: `axes.py` (knob axes,
production stacks, held-out knobs), `dense_rules.py` / `unified_rules.py` (the
`KnobSpace` subclasses), and `candidate.py` (the two factories).
`tests/dispatch/attention/test_tuning_contract.py` runs the conformance kit.

**Adding a tuned family** (any operator, not only attention): follow the
template in the platform `ARCHITECTURE.md` section 14 -- request and tuned
spec in `common.py`, axes as data, a `KnobSpace` subclass with the kernel's
rules, one `make_tuned_candidate` per variant, grid and block owned by the
spec, and a test that calls `assert_tuning_contract`. `common.py` holds what every candidate
shares (request, spec, gates) and imports none of the arch modules, so assembly
order does not matter. `generic.py` is for candidates declaring more than one arch --
the two unified paths and `d256_decode` -- which is not the same as portable:
each still lists its arches explicitly, because there is no family wildcard.

The two generic candidates declare every known arch on purpose: they select a
*path*, and `attention_unified` picks the concrete backend downstream from the
running device (wave64 MFMA on gfx942/gfx950, wave32 WMMA on gfx1250, the
arch-neutral scalar kernel elsewhere). They are an explicit, tested exception to
the wave-size consistency invariant.

## How to add a new specialized candidate

Follow the `_make_d256_decode_candidate()` pattern in `generic.py`:

1. **Add a cohort predicate** in `kernels/common/attention_unified.py` —
   a pure function of `UnifiedAttentionProblem` that returns `True` for the
   target shape family. This is the single source of truth for membership;
   import it lazily inside the factory to keep `dispatch/` arch-neutral.

2. **Declare a `Capability`** on the candidate: the explicit `arches` it serves,
   its `dtypes`, any head-size or block-size bounds as `ShapeRange`, and the
   `supports_features` it can handle (`causal`, `sliding_window`, `sinks`).
   This is required — `register()` rejects a candidate without one. Anything the
   capability declares must not be re-checked in `support()`.

3. **Add a factory function** `_make_<name>_candidate()` in the module for the
   arch it serves (`generic.py` if it declares more than one).
   The `support()` closure carries only what is left, in order: request errors →
   `_selector_matches` → `supports_native_unified_attention` → cohort predicate →
   path check (`select_path() == "2d"` or `"3d"`).

4. **Register** it from that module's `register(route, execution)`. A new arch module
   also needs one line in `__init__.py` to join the assembly loop.

   Point `build` at the real builder if the candidate has one. The unified
   paths do not: they return an `AttentionSpec` naming a *path*, and no builder
   consumes that. A standalone kernel like `attention_gfx1250_wmma` returns its
   builder's own spec instead, which is why it can declare `build`, a real
   grid, and a real signature where the unified candidates declare none.

5. **Add CPU-only dispatch tests** in
   `tests/dispatch/attention/test_<name>_wiring.py`. Cover: registration,
   spec_id, algorithm, priority ordering, rejection gates (wrong arch/dtype/
   cohort/path), and routing for each target arch. Use `_PinnedArch` context
   manager to avoid GPU dependency. Call `candidate.admits(req)`, not
   `candidate._supports(req)` — the latter skips the capability prefilter and is
   no longer a complete verdict, which is what the underscore is there to say.
   The coverage invariants in
   `test_declared_coverage.py` apply to the new candidate automatically.

6. **No C++ changes needed.** The dispatcher is Python-only.

## Engine-level selection: the ranker seam

`dispatch_attention(req, *, ranker=None)` chooses among the candidates that
support `req` in two stages:

1. `CandidateRegistry.supported(req)` filters to eligible candidates, already
   sorted ascending by `(priority, name)`.
2. A **ranker** — `Callable[(request, supported) -> reordered]` — reorders them
   best-first; `dispatch_attention` takes `ranked[0]`.

When no ranker is supplied, `priority_ranker` (the identity pass) keeps
registered `(priority, name)` order. A heuristic ranker is a **drop-in replacement** that
scores candidates against problem metadata (or offline benchmark data) and sorts
by score — no change to the registry or candidates. Safety invariant enforced by
the registry: a ranker may reorder or drop candidates but **cannot introduce one
the request does not support** (raises `ValueError`).

This is the **engine-level** half of the intended hierarchical design. The
**per-engine** half is each candidate's `select_spec` (today thin: it records
`(path, head_size, block_size, …)` and defers geometry). A future phase can move
the `_select_*` / `_enable_*` heuristics from
`builders/common/attention_spec_builder.py` into per-engine `select_spec`s so an
engine owns both "am I eligible?" and "how do I tune myself."

Coverage: `tests/dispatch/attention/test_ranker.py`.

## Additive registration (open/closed)

Adding a candidate must not change any existing candidate's `supports()` verdict
or `select_spec()` output. This is an executable invariant in
`tests/dispatch/attention/test_additive_registration.py`: it seeds a **fresh**
`CandidateRegistry` from `attention_candidates()`, registers a throwaway example
engine into that copy (never the shipped singleton), and asserts every
pre-existing candidate's behavior is byte-identical with and without it.

## Per-engine spec_fn (geometry ownership)

The long-run goal is the GEMM shape: each engine builds its own kernel spec
(`platform/python/rocke/dispatch/gemm/bf16_rcr.py` — one `_spec_*` per candidate),
instead of one shared `_tiled_spec_from_problem` cascade of `if` branches.

Migration is incremental — one cohort at a time.

1. **Extract** the cohort's branch from `_tiled_spec_from_problem`
   (`builders/common/attention_spec_builder.py`) into a named
   `_spec_<cohort>(problem)` — a self-contained builder (resolves its own arch /
   spec class). Pure move: byte-identical, no value change.
2. The cascade branch **delegates** to it (`return _spec_<cohort>(problem)`), so
   the shared function shrinks by one branch.
3. If the cohort has a matching dispatch candidate, that candidate **documents
   ownership** of the `spec_fn` (a docstring linkage). Some cohorts have no
   candidate yet (they ride the generic `unified_2d`) — those spec_fns are
   ORPHANS awaiting a future engine; note that in the spec_fn docstring. Either
   way, geometry stays in the **builder layer** — do NOT move it into a candidate.
   The dispatcher still decides only `(path, head_size, block_size)`, and the C++
   parity identity is unchanged (see the top of this doc).
4. **Test** byte-identity + non-interference (see the
   `library/tests/test_per_engine_spec_fns.py` -- table-driven, one entry per
   cohort), then
   GPU-verify the cohort's arch (kernel name / built spec unchanged vs pre-change).

Migrated so far (all builder-layer spec_fns in
`builders/common/attention_spec_builder.py`; `_tiled_spec_from_problem` and
`_tiled_3d_spec_from_problem` are now clean arch dispatchers):
- `_spec_gfx942_fp16_flash` — owned by `attention_gfx942_dense_pipe`.
- `_spec_gfx942_bf16_flash` — ORPHAN (no dispatch candidate yet; routed via the
  generic `unified_2d`). Needs a future `gfx942_bf16` engine.
- `_spec_generic_2d_non_gfx950` — the non-flash 2D residual for EVERY non-gfx950
  arch (gfx942, gfx1201, gfx1151, ...), built from the shared
  `_base_2d_generic_fields` only. ORPHAN.
- `_spec_gfx950_generic` — the shared `_base_2d_generic_fields` plus the
  gfx950-only schedule tail + the D256 gfx950 fast-route override folded in (kept
  behind the `_kau.` module handle for test-steering). The 2D `_spec_field_names`
  guards are gone -- the per-arch split replaced them.
- `_spec_generic_3d` — the shared gfx942/gfx950 3D split-KV fallthrough (one
  function; the `_gfx942_3d_*` helpers self-gate, so no arch split).

Remaining: gfx1250 (2D + 3D) -- still inline early-returns in both cascades,
DEFERRED (no gfx1250 hardware to GPU-verify this pass). The `_kau.` D256
indirection must be preserved by any code that touches the gfx950 override.

## Multi-engine benchmarking: `attention_sweep_space`

The probe that walks opt-in candidates and expands `sweep_space` now lives on
`CandidateRegistry` (`combos` / `sweep_space` / `dispatch_all`). Every operator
family wraps those methods (`registered_*_combos`, `*_sweep_space`,
`dispatch_*_all`). Attention keeps a thin wrapper so gfx950 dense still returns
its standalone dense spec rather than the unified path label.

`attention_sweep_space(req)` is the unified 2D/3D slice of that primitive: the
deduped spec of every candidate that supports `req` and carries a `path`. The
prefill benches
(`benchmarks/gfx{942,950}/attention/prefill/benchmark_prefill2d_live.py`) consume
it via the opt-in `--variants sweep` lane — a shared helper
(`benchmarks/common/attention_sweep.py:run_sweep`) that times each launched path
the registry offers and records which engine names mapped to it. Contract tests:
`tests/dispatch/attention/test_sweep_space.py`. The same enumeration for GEMM,
KDA, grouped conv, MoE, and norm is the family `*_sweep_space` /
`dispatch_*_all` wrappers.

**Two sweep levels.** `production` (the default) walks the curated named stacks
exhaustively. Those stacks leave off the knobs the kernels label as dead ends
(`use_q_reread` on gfx950, `use_conflict_free_v` on gfx942). Dead ends are not
`KNOWN_WRONG_KNOBS`; that set is only gfx942 `use_k_hbm_direct`, which stays
out of both levels. `full` is the sampled non-production space: every other
kernel knob, dead ends included. A single gfx942 transposed-x8 geometry has
roughly 16M legal knob settings, so `full` is consumed by sampling:
`tuning_sample` / `seed` (default 256, 0 walks the full stream). Sampling is a
random walk uniform at each knob decision, not uniform over the whole legal
set. `production` ignores `tuning_sample`.

Dense candidates share the level context but own a small WPE axis: production
walks the shipped policy plus WPE 2 and 4; full walks WPE 1 through 4. WPE is
part of the `tuning_id`, so an id replays one value.

Every consumer takes `sweep_level` plus `candidate_prefix` / `tuning_id_prefix`:
`run_sweep` (exposed as `--sweep-level` / `--sweep-tuning-sample` /
`--sweep-seed` / `--sweep-candidate-prefix` / `--sweep-tuning-id-prefix` /
`--sweep-limit` on both prefill benches) and the table sweeps under
`benchmarks/gfx950/attention/{decode,prefill}/`.

`benchmarks/common/attention_combo_sweep.py` is the arch-parameterized HW lane:
it walks `registered_attention_combos` for an arbitrary shape grid on gfx942 or
gfx950, launches dense and unified specs through their respective runners,
checks each against an SDPA reference, and streams one JSONL row per config so a
fault loses only the config that caused it. It names no candidate — the set
comes from the registry, so registering a candidate is enough to have it swept.
Host validation (build + verify + lower) runs on `--jobs` worker processes;
isolated GPU runs are spread across `--gpus`; a config whose lowered code
matches one already validated for the shape is a `duplicate` and is not run.
`--list-only` runs on a CPU host; `--limit`/`--offset` make a full run
resumable.

Framework-phase caveat: because geometry is deferred (see below), engines that
route to the same launched path collapse to one timed entry. The decode benches
(`benchmark_decode_live.py`) do **not** yet have a sweep lane — they lack a
variant-loop; adding one is a mechanical follow-up reusing the same shared
`run_sweep` helper.

## DEFERRED — production wiring + heuristic selection

The registry is currently **not load-bearing for GPU execution**. Production runs
through `run_unified_attention_torch` (`kernels/common/attention_unified.py`),
which calls `problem.select_path()` + `_tiled_spec_from_problem()` **directly**,
bypassing `dispatch_attention` / `ATTENTION_REGISTRY`. The dispatcher is exercised
only by these CPU tests and the benches' `prod` / `sweep` lanes.

Two later increments make it load-bearing:

- **Thin routing (low risk):** route `run_unified_attention_torch`'s path choice
  through the registry. The 2d-vs-3d decision is already the same pure
  `select_path()` both sides use, so this is byte-identical by construction;
  geometry still comes from `_tiled_spec_from_problem`.
- **Full engine-owned geometry (larger):** each engine's `select_spec` produces the
  tuned spec, absorbing the relevant `_select_*` branches. If C++ selection parity
  is required, geometry parity is the "separate, larger effort" noted in the
  `attention.py` module docstring ("DEFERRED — arch-tuned block geometry").

Heuristic-driven selection (a real scoring ranker) is independent of the above and
is a drop-in via the ranker seam once a scoring signal exists.

## How to tune `num_segments` for a new cohort

See the worked example:
`benchmarks/gfx942/attention/decode/TUNING_D256.md`

## Testing (CPU-only, no GPU)

```bash
python -m unittest discover \
    -s library/tests/dispatch/attention -p "test_*.py" -v
```

Dispatch tests are CPU-only. Cardinality checks walk the reduced tuning space
and take longer than the wiring tests.
