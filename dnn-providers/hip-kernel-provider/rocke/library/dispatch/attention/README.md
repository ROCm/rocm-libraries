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
knob set:

```text
candidate            one geometry variant, fixed at registration
 └─ base spec        select_spec(request): problem semantics + variant geometry
     └─ knob set     one walk step over the arch's KnobAxis list
         └─ WPE      waves_per_eu loop
             = one concrete spec: validated, never a duplicate, stable identity
```

Everything below lives in `tuning_common.py` except the geometry catalogs,
which live in the arch modules.

### Variants: what a candidate fixes

| Family | Registered candidates | Module | Fixed per candidate |
|---|---|---|---|
| Unified tiled tuning | `AttentionGeometryVariant` catalog: 109 on gfx950, 75 on gfx942 | `gfx{942,950}_unified.py` | path, codepath, builder, tile policy, warps, rows per warp, segments, compile backend |
| gfx950 dense | six `Gfx950DenseVariant`s (tile x persist x wide DMA) | `gfx950_dense.py` | `block_m` / `block_n`, `persistent`, `wide_lds_dma` |
| gfx942 dense | one candidate | `gfx942_dense.py` | nothing; its geometry is swept |

Unified variant names are `attention_{arch}_u{path}_{variant_id}`, for example
`attention_gfx950_u2d_narrow_nw1_mw16_t1xb_llvm` (the `spec_id` drops the
`attention_` prefix). A codepath also fixes base knobs through
`_CODEPATH_KNOBS`; `transposed32`, for instance, always sets `use_mfma_32x32`
and `use_transposed_qk_32x32`. The gfx942 `gfx942_4warp_gqa` builder reads no
tuning knobs, so its variant has no axes.

The base spec is what the candidate would launch unswept. Unified tuning
candidates build it with `tuning_specs.py` from the request and the variant.
Dense candidates use their production factory (`_dense_spec`), so the first
swept spec is always the shipped one.

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
- Combined axes, when knobs are mutually exclusive: dense `pv_schedule` puts
  IGLP and the manual PV fence / sched_group template in one decision.

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

- **Unified:** `_policy_conflict` rejects exclusive knobs and knobs that only
  exist on another codepath (softmax interleave off the transposed body,
  `sched_barrier` off the 16x16 loop). Then `tuning_specs.py` builds the kernel
  spec, so the spec's `__post_init__` and the kernel's `supports_tiled_*` run.
  Last, on gfx950 2D, `_supports_tuning_spec` checks KQ-pad eligibility and a
  static LDS model against the arch's LDS capacity.
- **Dense:** `_dense_knob_spec` applies the knobs to the base spec, so the
  dataclass validators run. It then drops any setting that would compile to
  the same IR as another under a new symbol:
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

Two more filters apply to dense before walking. `_DENSE_ARCH[arch].scope`
drops axes the base spec's problem or variant never reads, such as K/V row pads
at head size 64. `dense_pinned_axes(request)` drops axes the request pins:
`dense_persistent` or `dense_persist_decode` set to anything but `auto`.

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
returns the sample count, which is always 0 at `production`.
`CandidateRegistry.iter_combos` calls `sample_space(req, n, seed)` when that
count is positive and `sweep_space(req)` otherwise.

Pins narrow a walk to one point on an axis:

- `AttentionRequest.dense_waves_per_eu=1..8` pins dense WPE.
- `dense_persistent` / `dense_persist_decode` pin those dense axes.
- `algorithm` + `spec_id` pin a candidate.
- `attention_tuning_id` makes a unified tuning candidate's `select_spec`
  return exactly that spec. It searches the production specs first, then the
  full space, so a sampled id still replays.

With `attention_tuning_id="auto"`, unified tuning candidates select their first
production spec. Dense candidates always select the base spec.

### Identity and deduplication

Unified tuning specs carry `tuning_id`, a readable geometry/WPE prefix plus a
hash of the complete spec (see "Concrete tuning specs" below). Dense specs are
identified by `kernel_name()`, which tags every knob away from its default:
gfx950 folds the codegen knobs into one `cg<hash>` token, and gfx942 appends a
tag per knob. Samplers deduplicate by these identities. The benchmark lanes
add a backstop: the combo sweep and dense table sweep hash the lowered IR with
the kernel name blanked, and record a match as `duplicate` instead of running
it. Their rows also carry `knobs`, the non-default dense fields in readable
form.

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
  an enabler only if it can make a geometry legal. Add a `_policy_conflict`
  entry if it is inert on some codepath, and add it to a production stack if
  it should be timed by default.
- **New dense knob:** add a defaulted, name-tagged field to the dense spec.
  Add its axis to `_GFX9xx_DENSE_AXES`. If it only applies to some problems,
  add a scope rule in `_DENSE_ARCH`; if it depends on another knob, add an
  inert rule there. List it in `DENSE_UNTUNABLE_KNOBS` instead if it has
  nothing to sweep.
- **New geometry:** add an `AttentionGeometryVariant` to the arch's unified
  catalog, or a `Gfx950DenseVariant` to the gfx950 dense tuple. gfx942 dense
  geometry values live in `_GFX942_BLOCK_M` / `_GFX942_BLOCK_N`.

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

Explicit tuning also applies a per-spec support verdict in `tuning_common.py`.
This is where dispatcher-owned constraints such as gfx950 LDS capacity,
buffering combinations, active K-LDS padding, and padded-K/Q-alias exclusions
are enforced. Unsupported points are omitted before they enter `sweep_space`.
Kernel builders are not responsible for repairing dispatcher tuning points.

## Concrete tuning specs

`tuning_specs.py` constructs policy-free kernel specs shared by gfx942 and
gfx950 candidates. It accepts explicit geometry and codegen knobs and derives
only problem semantics such as dtype, masks, heads, and cache addressing.

It does **not** call production selection heuristics or silently resize an
invalid point. Concrete kernel validators remain the final structural gate;
dispatcher support in `tuning_common.py` applies tuning-policy exclusions.

The resulting `AttentionTuningSpec` is also the runtime's launch contract:
`run_unified_attention_torch(tuning_spec=...)` compiles `spec.build()` under
`spec.cache_key()` and launches with `spec.launch_grid(problem)` /
`spec.launch_block()`. The runtime never decodes `builder_kind`, and the
dispatcher's `grid` / `block` call the same methods.

Each `AttentionTuningSpec.tuning_id` contains a readable geometry/WPE prefix
plus a stable hash over:

- ABI and architecture;
- path and builder kind;
- compile backend;
- the complete kernel spec;
- the reduce spec, when present.

An exact `attention_tuning_id` therefore replays one configuration.

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
mode and stable tuning identity before compilation/cache lookup. FP8 OCP versus
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
- `gfx950_dense.py` — gfx950 dense-kernel candidates (frozen tile × persist ×
  wide-DMA variants and the dense ranker).
- `gfx950_unified.py` — gfx950 unified-kernel candidates: the D256 prefill fast
  path and the finite tuning geometry catalog.
- `tuning_common.py` — candidate construction, the per-arch knob axes, the
  pruned depth-first enumeration and random sampler, support filtering,
  stable IDs, and sweep expansion.
- `tuning_specs.py` — shared explicit kernel-spec/build construction.
- `bindings.py` — shared Torch execution adapters.

Benchmark-only concerns remain outside dispatch. For example, theoretical FLOP
accounting lives in `benchmarks/common/attention_flops.py`.

## Adding a candidate

Follow [`../AGENTS.md`](../AGENTS.md). Attention adds two registration lines:
routing labels go on `ATTENTION_ROUTE_REGISTRY`, and anything with `build` and
`bind_torch` also goes on `ATTENTION_EXECUTION_REGISTRY`. Set `opt_in=True` on
sweep-only candidates so `algorithm="auto"` cannot select them.

## Current production boundary

The generic unified runtime still resolves final architecture-tuned geometry in
`kernels/common/attention_unified.py`. The route registry mirrors the 2D/3D
decision and is exercised by dispatch and benchmark paths, but production
unified launch has not yet moved all geometry ownership into candidates.

Dense, WMMA, and explicit tuning candidates already use the complete executable
contract described above.
