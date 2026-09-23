# Attention dispatch

Registration procedure and shared registry mechanics live in
[`../AGENTS.md`](../AGENTS.md). This page only covers what is attention-specific:
route versus execution, capability versus support, and the production boundary.

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

Use `candidate_prefix` and `tuning_id_prefix` when enumerating large spaces.

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
- `gfx942.py`, `gfx950.py`, `gfx1250.py` — architecture-owned candidates.
- `gfx942_tuning.py`, `gfx950_tuning.py` — finite geometry catalogs.
- `tuning_common.py` — candidate construction, named stacks, support filtering,
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
