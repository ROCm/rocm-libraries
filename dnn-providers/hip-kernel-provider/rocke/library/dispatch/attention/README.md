# Attention dispatch

Registration procedure and shared registry mechanics live in
[`../AGENTS.md`](../AGENTS.md). This page only covers what is attention-specific:
route versus execution, how the tuning knob space is built, capability versus
support, and the production boundary.

This package selects attention implementations and exposes a uniform execution
contract for benchmarks and graph integrations. It deliberately separates:

1. **Routing** — choose the production attention path or specialized family.
2. **Execution** — enumerate concrete kernels that can be built and launched.

Keeping those concerns separate lets production retain its established 2D/3D
path policy while tuning and benchmarking use fully specified kernels.

## Request flow

All entry points consume `AttentionRequest` from `common.py`.

```text
AttentionRequest
    │
    ├─ candidate.admits(request)
    │    ├─ Capability.check(request)
    │    │    declarative arch, dtype, shape and feature coverage
    │    └─ residual support callback
    │         cohort, selector and cross-field checks
    │
    ├─ candidate.select_spec(request)
    │    routing label or concrete executable spec
    │
    └─ DispatchResult
         build()
         bind_torch(tensors)
         grid / block / signature / kernel_id
```

Call `candidate.admits(request)`, not the private `_supports` callback.
`admits` always applies both the declared `Capability` and the residual support
predicate.

## Two registries

### `ATTENTION_ROUTE_REGISTRY`

Used by `dispatch_attention`. It contains:

- generic `attention_unified_2d` and `attention_unified_3d` path labels;
- specialized production routing records;
- explicit opt-in candidates so exact `algorithm` / `spec_id` pins remain
  replayable.

The generic unified candidates are intentionally **routing-only**. Their specs
identify a path, head size, and block size, but do not own concrete CTA
geometry. They therefore do not provide build or Torch-binding callbacks.

`ATTENTION_REGISTRY` remains a compatibility alias for this registry.

### `ATTENTION_EXECUTION_REGISTRY`

Used by `registered_attention_combos` and `dispatch_attention_all`. Registration
requires every candidate to provide:

- concrete `select_spec`;
- `build`;
- `signature`;
- `grid`;
- `block`;
- `bind_torch`.

Routing-only path labels are absent. Dense gfx942/gfx950, gfx1250 WMMA, and
explicit unified-tuning candidates are executable.

```python
from dispatch.attention import AttentionRequest, dispatch_attention_all

request = AttentionRequest(
    batch=1,
    nhead_q=32,
    nhead_k=8,
    seqlen_q=1024,
    seqlen_k=1024,
    hdim_q=128,
    hdim_v=128,
    arch="gfx950",
    dtype="bf16",
    mask_type=1,
)

for result in dispatch_attention_all(request):
    kernel_ir = result.build()
    binding = result.bind_torch(
        {"q": q, "k": k, "v": v, "out": out, **paged_metadata}
    )
    binding.launch(stream=stream)
```

Each executable candidate expands into many concrete specs through its
`sweep_space` / `sample_space`; how that space is built is the next section.
Production `algorithm="auto"` selection never sees the opt-in candidates.

## Tuning knob space

A swept configuration is built in four layers. A registered candidate fixes a
coarse geometry, `select_spec` fills in the problem, a walk over the arch's
knob axes varies the tuning fields, and a `waves_per_eu` loop runs inside each
knob set. That loop is the shared `KnobSpace` outer-knob hook: both attention
spaces derive from `WavesPerEuSpace` (`waves.py`), which names `waves_per_eu`
as the outer knob and puts it in the id stem, and each supplies its own WPE
values:

```text
candidate            one geometry variant, fixed at registration
 └─ base spec        select_spec(request): problem semantics + variant geometry
     └─ knob set     one walk step over the arch's KnobAxis list
         └─ WPE      waves_per_eu loop
             = one concrete spec: validated, never a duplicate, stable identity
```

The axes and production stacks are data in `axes.py`; `dense_rules.py` and
`unified_rules.py` state attention's legality rules as `KnobSpace`
subclasses (through `waves.py`'s `WavesPerEuSpace`); `candidate.py` turns a
variant into a registered candidate.
Walking, sampling, canonicalizing and naming are the shared
`rocke.dispatch.tuning` (platform `ARCHITECTURE.md` section 14). Geometry
catalogs live in the arch modules.

### Variants: what a candidate fixes

| Family | Registered candidates | Module | Fixed per candidate |
|---|---|---|---|
| Unified tiled tuning | `AttentionGeometryVariant` catalog: 109 on gfx950, 75 on gfx942 | `gfx{942,950}_unified.py` | path, codepath, builder, tile policy, warps, rows per warp, segments, compile backend |
| gfx950 dense | three `Gfx950DenseVariant`s under two algorithms: `attention_dense_grid` (grid) and `attention_dense_persist` (persist, persist + wide DMA) | `gfx950_dense.py` | `persistent`, `wide_lds_dma`; the tile is a knob |
| gfx942 dense | one candidate (`attention_dense`) | `gfx942_dense.py` | nothing; persistence and the tile are knobs |

The gfx950 grid and persistent bodies are separate algorithms because they
serve different requests (only the grid body runs a moving bottom-right causal
diagonal), and wide DMA exists only on the persistent body. Wide DMA has no
ragged path, so where the 256×64 tile would be ragged its default starts at
128×64; it records `block_m` in every id so one id names one tile on every
problem.

Unified variant names are `attention_{arch}_u{path}_{variant_id}`, for example
`attention_gfx950_u2d_narrow_nw1_mw16_t1xb_llvm` (the `spec_id` drops the
`attention_` prefix). A codepath also fixes base knobs through
`_CODEPATH_KNOBS`; `transposed32`, for instance, always sets `use_mfma_32x32`
and `use_transposed_qk_32x32`. The gfx942 `gfx942_4warp_gqa` builder reads no
tuning knobs, so its variant has no axes.

The base spec is what the candidate would launch unswept, and what
`tuning_id="auto"` selects. Unified tuning candidates build it with
`tuning_specs.py` from the request and the variant. Dense candidates build it
in the arch module's `_base_spec` from the request and the variant's geometry;
`candidate.make_dense_candidate` wraps it in an `AttentionTuningSpec` with
`path="dense"`, so dense and unified specs carry the same `tuning_id`, build,
cache key and launch methods.

### Axes: the knob space as data

`KnobAxis(name, choices, enabler)` is one tuning decision. Each choice is a
tuple of `(field, value)` pairs, and `choices[0]` is always `()`, meaning
"leave the base value". Axes are built with a few helpers:

- `_flag(name)`: off or on.
- `_values(name, default, values)`: every value except the default.
- `_gated(gate, {sub: values})`: gate off, or on with every sub-knob
  combination. Sub-knobs are read only while the gate is on, so they never
  vary while it is off.
- `_choices_axis(name, values)`: every value, the base's included, for axes
  whose base comes from the request or a policy (dense geometry).

Every knob is its own axis, on dense as on unified. Knobs the kernel refuses
to combine (for example gfx950 IGLP with the manual PV fence) are separate
axes, and the kernel's validator prunes the illegal combinations.

`_AXES[(arch, path)]` holds the list for `2d`, `3d` and `dense` on each arch.
The unified test `test_every_kernel_tuning_field_is_swept` and the dense
field-classification tests fail when a kernel spec gains a field that no axis
or exemption covers.

Order matters. Axes run prerequisites-first, so any "X requires Y" relation
points at an earlier axis. **Enabler** axes lead the list: they are knobs that
can turn an otherwise illegal setting legal. Examples are the gfx950 2D
LDS-saving knobs, gfx942 direct-Q plus conflict-free V store (needed by
`num_warps=8, block_m_per_warp=32`), and the gfx942 dense conflict-free V and
K pads (which let an over-budget tile fit).

Some knobs are held out:

- `KNOWN_WRONG_KNOBS` produced wrong output in a reference sweep and are never
  offered.
- `DEAD_END_KNOBS` are correct but documented as slower. The unified
  production stacks leave them off; the full level still samples them.
- `DENSE_UNTUNABLE_KNOBS` accept a single value (`lds_num_buffers`) or are
  never read by that arch's body (gfx942 `lazy_rescale`).

### Validity and pruning

A walk asks `is_valid(knobs)` for each prefix, with undecided axes at their
base values. The checks run in this order:

- **Unified:** `_policy_conflict` rejects knobs that only exist on another
  codepath (softmax interleave off the transposed body, `sched_barrier` off
  the 16x16 loop), because they would re-emit the same IR under a new name.
  Then `tuning_specs.py` builds the kernel spec, so the spec's `__post_init__`
  and the kernel's `supports_tiled_*` run. On gfx950 2D, `UnifiedSpace.validate`
  then checks what that validator does not model: the LDS footprint
  (`_gfx950_2d_lds_bytes`, the pool the LLVM lowering packs) against the
  arch's LDS capacity, and a padded K LDS that Q aliases, which computes wrong
  output. Both were confirmed on MI355X: the over-budget specs fail codegen,
  and padded-K / aliased-Q specs mismatched the reference (most of them, some
  nondeterministically).
- **Dense:** `DenseSpace` (through `KnobSpace.canonicalize`) applies the
  knobs to the base spec, so the dataclass validators run. It drops any
  setting that would compile to the same IR as another under a new symbol:
  - a choice equal to the base value;
  - a changed `num_persistent` on a non-persistent spec;
  - an explicit value equal to what its `resolved_*` policy picks;
  - the persistent-only knobs on a non-persistent spec, and `interleave` off
    the causal `qb_major` decode;
  - the arch's own rules in `_DENSE_ARCH[arch].inert`, such as a gfx950 fence
    mask while the fence is off or a gfx942 V pad without the conflict-free V
    store.

  Finally the kernel's `supports_attention_dense` runs.

Because every relation points at an earlier axis, a failing prefix cannot be
repaired later and its whole subtree is skipped. The one exception: while
enabler axes are undecided, an invalid prefix is kept.

### Canonicalization: one kernel, one id

The walk prunes; canonicalization normalizes. Every spec a candidate hands out
-- swept, sampled, pinned by knobs, or found by id -- is made by one function
`KnobSpace.canonicalize` (through `DenseSpace` or `UnifiedSpace`), from the
knob dict (`waves_per_eu` included):

- **converted** first to the type the knob's axis declares (the WPE to the
  type of its values): `True` and `1`, or `2` and `2.0`, hash alike, so they
  must name one configuration; a value with no lossless conversion is refused;
- **dropped**, because the result compiles to the kernel without them: values
  equal to the default spec (or to the codepath's fixed value); on dense,
  restated policies and the inert rules above; on unified, knobs only another
  codepath emits, gated sub-knobs whose gate is off, and on gfx950 2D a KQ LDS
  pad the kernel lays out as no pad (native-FP8 K, misaligned slabs, or more
  than one K buffer);
- **refused**, with the reason: `KNOWN_WRONG_KNOBS`, fields no axis of that
  variant sets (problem fields, variant geometry, untunable knobs, knobs out
  of scope for the problem), changing a knob the codepath fixes, whatever the
  kernel spec or validator rejects, and on gfx950 2D a spec over the LDS
  budget or a padded K that Q aliases.

What remains is the canonical knob dict the spec carries as `knobs`, and the
input to its `tuning_id`.

One more filter applies to dense before walking. `_DENSE_ARCH[arch].scope`
drops axes the base spec's problem or variant never reads, such as K/V row pads
at head size 64.

### Walks: production, full, sample

| Level | Unified tuning | Dense |
|---|---|---|
| `production` | Hand-curated `_PRODUCTION_STACKS` per (arch, codepath), on top of the codepath base knobs, plus two gfx950 micro-axes (padded K on the `ksb_qdreg` stack, softmax interleave on three transposed stacks). WPE {policy, 2, 4} for 2D, {policy, 1–4} for 3D. | The base spec, then each applicable axis choice on its own (`_one_knob_at_a_time`). WPE {policy, 2, 4}. |
| `full` | `_iter_knob_sets`: depth-first product of every axis. WPE {policy, 1–4}. | The same depth-first walk. WPE {policy, 1–4}. |
| `full` + `tuning_sample=n` | `_random_knob_set`: one random walk down the pruned tree per draw, uniform at each axis, seeded by `"{seed}:{candidate}"`; `n` distinct specs, stopping after `20n` draws. | The same sampler. |

The full product runs to millions of specs per shape on the transposed unified
paths and on dense, which is why `full` is normally sampled. The streams are
lazy, so a walk is never materialized unless the caller asks for it.

The level reaches the candidates through `configure_sweep(level,
tuning_sample)`. It sets a context variable that `sweep_space` reads, and
returns the sample count, which is always 0 at `production`; a negative
`tuning_sample` is refused.
`CandidateRegistry.iter_combos` calls `sample_space(req, n, seed)` when that
count is positive and `sweep_space(req)` otherwise.

Selection pins a spec the same way for dense and unified:

- `algorithm` (`attention_dense` on gfx942, `attention_dense_grid` or
  `attention_dense_persist` on gfx950, `unified_tuning`) plus `spec_id` pin a
  candidate. An opt-in candidate admits a request only when both name it (its
  refusal names both), so an unpinned request always routes to the unified
  path.
- `tuning_knobs`, when set, is canonicalized directly into the spec:
  constant time, no walk. With a non-`auto` `tuning_id` it must
  reproduce that id, or the candidate refuses the request.
- A bare `tuning_id` is looked up in the production set only; a full-space
  (sampled) id must be pinned with its knobs. Ids match on `config_key`, not
  the display stem. A pin that no longer resolves raises `PinRefused` with the
  reason; nothing falls back. An id from
  another variant is rejected before the search.
- `tuning_id="auto"` selects the candidate's base spec.
- `emit_lse=True` asks for the optional FP32 LSE output (natural-log softmax
  normalizer). It is the `lse` request feature; only the dense candidates declare
  it, so an unpinned LSE request is refused instead of reaching a kernel without
  the output. It is a dense output field (`DENSE_OUTPUT_FIELDS`), so sweeps
  never vary it. The caller passes the buffer to `bind_torch` as `tensors["lse"]`;
  the dense runner validates its shape, dtype, device, and overlap.

A candidate resolves a request once: `support` and `select_spec` share a
per-candidate cache, so a pin is not resolved twice.

The request carries no individual tuning fields: tile, persistence, wide DMA,
`interleave`, `num_persistent` and `waves_per_eu` are all spec fields, reached
through the knob dict. Callers use `attention_tuning_spec(req, spec_id,
tuning_id, knobs)` to replay a recorded configuration, or
`tuning_spec_with_knobs(req, spec_id, knobs)` to set knob values on the base
spec. Both go through dispatch.

### Identity and deduplication

Every tuning spec carries `variant_id`, its canonical `knobs`, and

```text
tuning_id  = "{variant_id}_wpe{N}@{config_key}"
config_key = hash(TUNING_ID_VERSION, ABI, arch, path, variant_id, knobs,
                  fingerprint(the variant's defaults))
```

`config_key` hashes that explicit list rather than the spec's dataclass
fields, so it is the same on every problem the variant admits. The defaults
fingerprint covers every declared default of the kernel spec, so adding a
defaulted field or changing a default changes every id of that variant: a
stored pin is then refused instead of silently building a different kernel,
and has to be re-swept. This broad invalidation is intentional, including for
a newly added behavior-neutral default: validate the replacement ids before
publishing them and remove the old ids from the external tuning store. The
refusal reports the stored id and the newly canonicalized id; it never adopts
the replacement silently. A change to the payload or the canonicalization
rules bumps `TUNING_ID_VERSION` in `rocke.dispatch.tuning.identity`. gfx942 dense
resolves `persistent` and `waves_per_eu` per problem, so those two are always
recorded with their effective value. Pins match on `config_key`; the
`{variant_id}_wpe{N}` stem (`waves.py`: `waves_per_eu` is the attention
spaces' outer knob) is only read to skip variants a bare id cannot name
(`KnobSpace.stem_prefix`). Knob values take the type their axis declares, so
`True` and `1` name the same configuration. The compiled binary has a separate
identity: `AttentionTuningSpec.identity()` (the kernel cache key), which
`KernelId.spec_hash` hashes.

Dense kernel specs are also named by `kernel_name()`, which tags every knob
away from its default: gfx950 folds the codegen knobs into one `cg<hash>`
token, and gfx942 appends a tag per knob. Every stream -- production, full
walk, and sampler -- yields each `tuning_id` once, since two walk points can
canonicalize to the same knobs. The benchmark lanes add a backstop: the combo sweep and dense
table sweep hash the lowered IR with the kernel name blanked, and record a
match as `duplicate` instead of running it. Their rows carry `knobs` next to
`tuning_id`, which is what replays the row (`--run-knobs` for the isolated
child, `tuning_knobs` for any caller).

Dispatch results carry the configuration too: `KernelId.tuning_id`, and a
stored request pinned to the spec's id and knobs, so `request_hash` differs per
configuration and `dispatch_attention(result.request)` reselects what ran.

### From the knob space to the benchmarks

```text
candidate.sweep_space / sample_space
 └─ CandidateRegistry.iter_combos            (probes opt-in candidates)
     ├─ iter_registered_attention_combos / dispatch_attention_all
     │    └─ attention_combo_sweep.py, dense_prefill_table_sweep.py,
     │       decode_table_sweep.py           (dense + unified tuning)
     └─ attention_sweep_space                (unified 2D/3D specs only)
          └─ attention_sweep.run_sweep       (--variants sweep in the live
                                              prefill benchmarks)
```

### Extending the space

- **New unified knob:** add a defaulted field to the tiled kernel spec. Add
  its `KnobAxis` to the arch's 2D or 3D list after its prerequisites; make it
  an enabler only if it can make a geometry legal. Add it to
  `_TRANSPOSED_ONLY_KNOBS` / `_NARROW_ONLY_KNOBS` if it is inert on some
  codepath, and add it to a production stack if it should be timed by
  default.
- **New dense knob:** add a defaulted, name-tagged field to the dense spec.
  Add its axis to `_GFX9xx_DENSE_AXES`. If it only applies to some problems,
  add a scope rule in `_DENSE_ARCH`; if it depends on another knob, add an
  inert rule there (it names the knob, so canonicalization can drop it). List
  it in `DENSE_UNTUNABLE_KNOBS` instead if it has nothing to sweep.
- **New geometry:** add an `AttentionGeometryVariant` to the arch's unified
  catalog. Dense tile values are knobs: `_GFX950_BLOCK_M` / `_GFX950_BLOCK_N`
  and `_GFX942_BLOCK_M` / `_GFX942_BLOCK_N`. A new gfx950 dense body is a
  `Gfx950DenseVariant` with its own algorithm.

In every case the coverage tests in
`tests/dispatch/attention/test_tuning_space.py` fail until the new field sits
on an axis or in an exemption.

## Capability versus support

`Capability` is declarative and queryable without constructing a kernel spec.
It owns independent request constraints:

- supported architectures;
- dtypes;
- allowed dimension values/ranges;
- supported features such as causal masking, sliding-window attention, sinks,
  and FP8.

The residual support callback owns constraints that require relationships
between fields or candidate-specific policy:

- opt-in selector checks;
- request structural validation;
- shape cohorts and 2D/3D path compatibility;
- whether a geometry candidate produces any valid concrete tuning spec.

Per tuning spec, legality is the kernel's own validators plus, on the gfx950
2D path, `UnifiedSpace.validate`: the tiled 2D validator does not model LDS,
so a static LDS budget rejects specs codegen would refuse, and a padded K
LDS is refused when Q aliases K (it computes wrong output). A pad the kernel
lays out as no pad (double-buffered K, native-FP8 K, misaligned slabs) is
inert and dropped, so it does not mint a second id. Points they reject are
omitted before they enter `sweep_space`; kernel builders are not responsible
for repairing dispatcher tuning points.

## Concrete tuning specs

`tuning_specs.py` constructs policy-free kernel specs shared by gfx942 and
gfx950 candidates. It accepts explicit geometry and codegen knobs and derives
only problem semantics such as dtype, masks, heads, and cache addressing.

It does **not** call production selection heuristics or silently resize an
invalid point. Concrete kernel validators remain the structural gate.

The resulting `AttentionTuningSpec` is also the runtime's launch contract:
`run_unified_attention_torch(tuning_spec=...)` compiles `spec.build()` under
`spec.cache_key()` and launches with `spec.launch_grid(problem)` /
`spec.launch_block()`. The runtime never decodes `builder_kind`. The spec is
the one owner of grid and block: the dispatcher's `grid` / `block` and the
Torch bindings all call these methods.

## Torch bindings

`bindings.py` is the shared dispatcher adapter for:

- gfx942/gfx950 dense attention;
- explicit unified 2D/3D tuning kernels;
- gfx1250 WMMA attention.

Bindings close over caller-owned tensors and return `TorchBinding(launch, grid,
block)`. Torch is never imported at dispatch module load time; it remains a
runtime dependency supplied by the ROCm environment.

The binding layer translates tensors and optional metadata to each runner's
actual ABI. Callers do not branch on spec classes.

For explicit paged attention, binding validates tensor shapes, dtypes, layouts,
sequence metadata, and every used physical block ID before creating the launch
closure. Metadata is a one-time snapshot; callers must rebind after mutation.
The narrowly scoped `unsafe_skip_paged_value_validation=True` option is only for
trusted callers that enforce immutable, bounds-checked metadata externally.

Physical K/V cache size is not known during request-only dispatch. Once binding
sees `k.shape[0]`, it refreshes the explicit kernel spec's i32/i64 addressing
mode before compilation/cache lookup. That is a runtime specialization: it
changes the kernel identity and cache key, never `tuning_id`, so an id recorded
after binding still replays from the request. FP8 OCP versus
FNUZ is a property of the architecture: the tuning wrapper records the request
encoding so a mismatch is rejected, and the kernel name derives the FNUZ suffix
from the gfx942 spec rather than from a free field.

## Package layout

- `__init__.py` — registry assembly and public entry points.
- `common.py` — arch-neutral request/spec types and shared gates.
- `generic.py` — candidates that cover more than one architecture.
- `gfx1250.py` — architecture-owned candidates.
- `gfx942_dense.py` — the gfx942 dense-kernel candidate (geometry swept by its
  knob space).
- `gfx942_unified.py` — gfx942 unified-kernel candidates: the fp16 `dense_pipe`
  flash path and the finite tuning geometry catalog.
- `gfx950_dense.py` — gfx950 dense-kernel candidates (the grid body, the
  persistent body, and the persistent body with wide DMA; the tile is a knob).
- `gfx950_unified.py` — gfx950 unified-kernel candidates: the D256 prefill fast
  path and the finite tuning geometry catalog.
- `axes.py` — the tuning space as data: knob axes, codepath knobs,
  production stacks, and the held-out knob sets.
- `dense_rules.py` — dense scope, policy and inert-knob rules, stated as
  the `DenseSpace` hooks.
- `unified_rules.py` — unified geometry variants, codepath rules, and the
  gfx950 2D LDS / KQ-pad legality, stated as the `UnifiedSpace` hooks.
- `waves.py` — `waves_per_eu` as the attention spaces' outer knob: the WPE
  sweep values and `WavesPerEuSpace`, which sets the outer knob and the
  `{variant_id}_wpe{N}` id stem for both spaces.
- `candidate.py` — `make_tuning_candidate` and `make_dense_candidate`, both
  over `rocke.dispatch.tuning.make_tuned_candidate`.
- `tuning_specs.py` — shared explicit kernel-spec/build construction.
- `bindings.py` — shared Torch execution adapters.

Benchmark-only concerns remain outside dispatch. For example, theoretical FLOP
accounting lives in `benchmarks/common/attention_flops.py`.

## Adding a candidate

Follow [`../AGENTS.md`](../AGENTS.md). Each module exports
`register(route, execution)`: routing labels go on `ATTENTION_ROUTE_REGISTRY`,
and anything with `build` and `bind_torch` also goes on
`ATTENTION_EXECUTION_REGISTRY`. Set `opt_in=True` on
sweep-only candidates so `algorithm="auto"` cannot select them.

## Current production boundary

The generic unified runtime still resolves final architecture-tuned geometry in
`kernels/common/attention_unified.py`. The route registry mirrors the 2D/3D
decision and is exercised by dispatch and benchmark paths, but production
unified launch has not yet moved all geometry ownership into candidates.

Dense, WMMA, and explicit tuning candidates already use the complete executable
contract described above.
