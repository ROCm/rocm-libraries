# rocKE Dispatch

`rocke.dispatch` owns operator-to-kernel selection: routing one request to one
kernel, enumerating every eligible kernel and configuration for a sweep, and
naming each selection with a stable identity. It does not benchmark or collect
performance evidence; benchmark harnesses live under `rocke.benchmark` and
`library/benchmarks`. The design is in [ARCHITECTURE.md](ARCHITECTURE.md).

## Layout

```text
rocke/dispatch/
  core.py          # request / candidate / registry / result / identity contracts
  __init__.py      # public dispatch exports
  tuning/          # shared machinery for tuned (sweepable) candidates
    axes.py        #   KnobAxis and the helpers that declare a knob space
    walk.py        #   sweep levels, pruned depth-first walk, sampler
    identity.py    #   config_key / tuning_id
    space.py       #   KnobSpace: canonicalize, default, stream, sample, find
    candidate.py   #   make_tuned_candidate and pin resolution
    spec.py        #   TunedSpec / TunableRequest protocols
    testing.py     #   assert_tuning_contract, the conformance kit
  gemm/
    common.py      # GEMM request / selector helpers
    support.py     # GEMM config and shape support predicates
    binding.py     # GEMM launch bindings
    fp16_rcr.py    # UniversalGemm FP16 RCR
    bf16_rcr.py    # UniversalGemm BF16 RCR
  families/
    moe.py  norm.py
```

Tests live under `platform/tests/dispatch/dispatch_tests/` (`core/`, `gemm/`,
`moe/`, `norm/`, `tuning/`).

Families whose predicates import arch-specific kernel modules live in the
library tree and use the same `core.py` contracts and coverage invariants:

- `library/dispatch/attention/` — SDPA. Per-arch modules (`generic.py`,
  `gfx942_dense.py`, `gfx942_unified.py`, `gfx950_dense.py`,
  `gfx950_unified.py`, `gfx1250.py`), the attention knob space (`axes.py`,
  `dense_rules.py`, `unified_rules.py`, `waves.py`, `candidate.py`), `bindings.py`, and
  `__init__.py` with the registries and entry points. It keeps two registries:
  a route registry that production selects from (it includes path labels with
  no builder) and an execution registry that sweeps enumerate. See
  `library/dispatch/AGENTS.md` and `library/dispatch/attention/README.md`.
- `library/dispatch/kda/` and `library/dispatch/gdn/` — chunkwise KDA and GDN,
  separate families (gated delta-rule recurrences, no code shared with SDPA).
  See `dsl_docs/instances/kda.md`.
- `library/dispatch/grouped_convolution.py` — grouped convolution.

## Current Scope

The two UniversalGemm cases (FP16 and BF16 RCR) and the moe / norm families
are implemented over the portable instance builders; each defines its
normalized request, registry, and declared per-candidate coverage.

```python
from rocke.dispatch import GemmRequest, dispatch_gemm_fp16, dispatch_gemm_bf16

result = dispatch_gemm_fp16(GemmRequest(M=4096, N=4096, K=4096, arch="gfx950"))
result_bf16 = dispatch_gemm_bf16(
    GemmRequest(M=4096, N=4096, K=4096, arch="gfx950", dtype="bf16")
)

print(result.kernel_id.selection_key)
print(result.candidate.name)
print(result.grid, result.block)
```

Most candidates offer one configuration per request. Attention's dense and
unified tuning candidates are the first **tuned** candidates: each is one
geometry variant whose configuration is a point in a knob space
(see [Tuned candidates](#tuned-candidates)).

### Declared coverage and the arch gate

Every candidate declares a `Capability`: the exact `gfx` targets it was built
for, its dtype and layout, and any shape bounds expressible as data. This is
mandatory — `register()` rejects a candidate whose capability is `None`, since
an undeclared candidate is invisible to `for_arch()` and `coverage()` and would
make both answer by omission.

That declaration is the arch gate. Without one, an RDNA/WMMA candidate would report
support on a CDNA arch (its spec rebuilds wave64 and a 16x16x16 MFMA atom that
also exists on CDNA), wrongly out-ranking the intended CDNA candidate. The
regression is pinned by `dispatch_tests/gemm/test_arch_family_gate.py`, the GEMM
family-wide invariants by `dispatch_tests/gemm/test_capability.py`, and the same
invariants for moe / norm by `dispatch_tests/core/test_declared_coverage.py`.

The gate is an explicit arch list rather than a `cdna`/`rdna` label, for two
reasons. Family does not imply wave size — gfx1250 is CDNA at wave32 — so a
family gate would admit a wave32 target into wave64 MFMA kernels. And the right
list is per candidate, not per family: bf16's cshuffle path runs on gfx90a where
fp16's does not, and the bf16 decode candidate needs a deep-K atom that exists
only on gfx950.

A candidate's declared arches must agree on wave size, with two named
exceptions: the 30 `norm2d` candidates and attention's two `unified_*` path
candidates, neither of which bakes a wave size into its geometry. See
ARCHITECTURE.md section 10.

Capability is a prefilter, not the whole answer. Ask `candidate.admits(req)` for
the complete verdict:

```python
ok, why = candidate.admits(req)   # capability, then the residual predicate
```

`_supports()` is the residual predicate alone. Since it no longer re-checks arch
or dtype, it is not a complete gate; the underscore marks it as private so that
calling it directly reads as the violation it is.

### Identity

`KernelId` is the stable identity used by compile caches, manifests, logs, and
benchmark records. It includes the operation family, candidate, algorithm,
`spec_id`, target arch, ABI version, request hash, spec hash, and — for a tuned
spec — its `tuning_id`. It exposes two keys because they answer different
questions:

- `compile_key` — `arch:abi_version:spec_hash`. Identifies the compiled binary.
  Problem-independent, so every request that selects the same spec shares one
  compile. This is what an HSACO cache should key on.
- `selection_key` — every field, including the request hash. Identifies the
  routing decision, which is what tuning records and dispatch logs index by.

`spec_hash` hashes the spec's explicit `identity()` payload when it defines one
(so wrapper metadata such as a debug switch stays out), and every dataclass
field otherwise. `cache_key` is a deprecated alias for `selection_key`.

### Lookup without dispatching

`CandidateRegistry` answers three questions that need no request:

```python
registry.get("universal_gemm_fp16_cdna_mem")   # by name
registry.resolve(kernel_id)                    # by persisted id, ABI-checked
registry.coverage()                            # JSON manifest of what is registered
```

`resolve` is the replay path for the candidate: it rejects a `KernelId` whose
`abi_version` no longer matches the registry, so a tuning record from an older
build fails loudly instead of binding to a kernel whose kernarg layout has
since changed. For a tuned spec, the configuration replays from its
`tuning_id` and knobs (below).

### Every kernel and configuration: sweeps

```python
from rocke.dispatch.tuning import configure_sweep

configure_sweep("production")            # curated set; "full" = whole knob space
for result in registry.iter_dispatch_all(req, kernel_id=kernel_id_fn, sample=0):
    kernel = result.build()
    ...
```

`iter_combos` / `sweep_space` / `iter_dispatch_all` walk the whole registry,
probe opt-in candidates by pinning their own `algorithm` / `spec_id`, and expand
each candidate's `sweep_space` — or, with `sample > 0`, draw that many specs per
candidate from its full space (a negative `sample` is refused). Production
`dispatch_*` never sees an opt-in candidate; one refuses an unpinned request
with a reason that names both selectors it needs. Each result's stored request
is pinned to its spec (`core.pin_to_spec`), so `request_hash` differs per
configuration and the stored request reselects exactly what ran; a spec with
no `tuning_id` resets the request's `tuning_id` / `tuning_knobs` to
`"auto"` / `()`, so a stale pin is never carried along. Attention wraps these
as `iter_dispatch_attention_all(req, sweep_level=..., tuning_sample=...)`.

## Tuned candidates

A tuned candidate is one registered **variant** (a geometry) plus a **knob
space** its configurations are drawn from. Its specs carry:

```text
tuning_id  = "{stem}@{config_key}"   # stem: the family's display name
config_key = hash(TUNING_ID_VERSION, ABI, arch, path, variant_id, knobs,
                  fingerprint(the variant's defaults))
knobs      = the canonical knob dict: fields set away from the default spec
```

The stem defaults to the variant id; attention shows the occupancy hint too
(`{variant_id}_wpe{N}`). Pins match on `config_key` only.

`config_key` covers that explicit list, not the spec's fields, so it is the
same on every problem the variant admits; `identity()` / `spec_hash` name the
problem-bound binary. Requests pin a configuration with two fields:

```python
from dataclasses import replace

pinned = replace(req, algorithm=cand.algorithm, spec_id=cand.spec_id,
                 tuning_id=row["tuning_id"], tuning_knobs=row["knobs"])
spec = cand.select_spec(pinned)      # rebuilt from the knobs, checked against the id
```

`tuning_knobs` is validated when the request is built. Pins are matched on the
`config_key` suffix, not the display stem. A bare `tuning_id` resolves only
within the production set (dispatch never searches the full space); `"auto"` is
the variant's default spec. A pin that no longer resolves -- removed variant,
unknown id, knobs that no longer reproduce the key -- raises
`rocke.dispatch.PinRefused` with the reason and never falls back to another
kernel. Because the key includes a fingerprint of the defaults the knobs are
relative to, a changed default refuses old pins instead of silently building a
different kernel; a long-lived cache should also store `spec_hash` and treat a
mismatch on replay as a miss (ARCHITECTURE.md section 11.1). Knobs that compile to the default are dropped and
illegal ones are refused with a reason, so one kernel has one id however it was
reached (sweep, sample, knob pin, or id). Knob values are converted to the type
their axis declares: `True` and `1`, or `2` and `2.0`, compare and hash equal,
so they name one configuration, and a value with no lossless conversion is
refused.

Everything above comes from `rocke.dispatch.tuning`, which names no kernel
field. What a family needs beyond its axes is a `KnobSpace` hook:

- `validate(base, kernel)` is the legality gate for anything the kernel's own
  validator does not model; it also prunes the walk. Attention's gfx950 2D
  space uses it for the LDS budget and the padded-K / aliased-Q rule.
- `outer_knob` with `outer_values(base, level)` and `outer_default(base)`
  names one field that is cheap to sweep on top of every knob set: the walk
  checks a knob set once and only rebuilds it per outer value, so the drop
  rules must not read it. Attention uses `waves_per_eu`
  (`library/dispatch/attention/waves.py`); a space without one sweeps nothing
  extra.
- `stem(kernel)` / `stem_prefix()` give the display part of the id
  (`variant_id` by default) and the prefix a bare id must carry to be looked
  up.

A family writes its axes as data, a `KnobSpace` subclass with its kernel's
rules, and one
`make_tuned_candidate(...)` per variant, then checks it with
`assert_tuning_contract`. ARCHITECTURE.md section 14 is the step-by-step
template; `platform/tests/dispatch/dispatch_tests/tuning/test_tuning_template.py`
is a complete toy family in one file.

## Run Tests

No-GPU dispatch tests (platform, then library):

```bash
cd dnn-providers/hip-kernel-provider/rocke
PYTHONPATH=platform/python python -m pytest platform/tests/dispatch -q
cd library && PYTHONPATH=../platform/python:. python -m pytest tests/dispatch -q
```

Runtime tests in `dispatch_tests/gemm` are GPU-gated and skip when no ROCm GPU
is visible.

Broader no-GPU regression checks:

```bash
cd dnn-providers/hip-kernel-provider/rocke/platform
PYTHONPATH=python python tests/test_rocke.py
PYTHONPATH=python python -m pytest tests/instances/test_rocke_multiarch.py -k TestGfx950ByteIdentical
PYTHONPATH=python:tests/instances python -m rocke_ir_parity_harness \
  --compare tests/golden/rocke_representative_ir_sha256.json
```

## Onboard A New Operator Family

1. Add an operator package, for example `rocke/dispatch/<family>/` (or
   `library/dispatch/<family>/` when its kernels live in the library).
2. Keep shared request and selector helpers in `<family>/common.py`, and
   support predicates in `<family>/support.py`, split into config support
   (arch, dtype, tile, wave shape, MMA/WMMA availability, LDS, pipeline) and
   request support (runtime shape, layout, fusion).
3. Add one case module per stable dispatch surface (`gemm/bf16_rcr.py`) or per
   architecture lane (`gfx950.py`) once the family has earned one
   (ARCHITECTURE.md section 8).
4. Register candidates with `name`, `family`, `algorithm`, `spec_id`,
   `abi_version`, `priority`, a `Capability`, and the
   support / select / build / signature / grid / block / sweep hooks.
5. If the kernel has knobs worth sweeping, build its candidates on
   `rocke.dispatch.tuning` instead of writing those hooks
   (ARCHITECTURE.md section 14).
6. Return a `DispatchResult` with a `KernelId` from `core.make_kernel_id`.
7. Add operator-local tests under `platform/tests/dispatch/dispatch_tests/<family>/`
   (or `library/tests/dispatch/<family>/`), including `assert_tuning_contract`
   for tuned candidates.

## Onboard A New GEMM Case

For a new GEMM case, such as BF16 RCR, add `gemm/bf16_rcr.py` and
`dispatch_tests/gemm/test_bf16_rcr.py`, and reuse:

- `GemmRequest` from `gemm/common.py` if the request shape is compatible;
- `selector_matches` from `dispatch/core.py` for `algorithm` / `spec_id` filtering;
- `GemmSupportQuery`, `gemm_config_supported`, and `request_shape_supported`
  from `gemm/support.py` when the support model matches UniversalGemm.

Add a case-local ABI version, for example:

```python
GEMM_BF16_RCR_ABI_VERSION = "hipkg-gemm-bf16-rcr/v1"
```

Do not put case-specific ABI constants or request fields in `dispatch/core.py`.
