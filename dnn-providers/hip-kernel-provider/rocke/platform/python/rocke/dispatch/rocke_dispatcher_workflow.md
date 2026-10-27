# How a request travels through dispatch

This page follows the common jobs the dispatcher does, one at a time, and names
the function that does each step. It uses attention as the running example
because it exercises every feature; GEMM, MoE and norm take the same route with
fewer steps. [README.md](README.md) is the reference and
[ARCHITECTURE.md](ARCHITECTURE.md) the full design; this page is the tour.

Paths below are relative to `dnn-providers/hip-kernel-provider/rocke/`.

## Words used on this page

| Term | Plain meaning |
|---|---|
| request | What the caller wants computed: the problem sizes, data type, target GPU (`arch`), and optionally which kernel to use. For attention, `AttentionRequest`. |
| candidate | One registered kernel option the dispatcher can choose. It knows which problems it accepts, how to build its kernel, and how to launch it (`KernelCandidate`). |
| registry | The list of candidates for one operator, kept in priority order (`CandidateRegistry`). Lower priority number wins. |
| capability | A candidate's declared limits written as data: which GPUs, data types and sizes it handles. Checked first because it is cheap. |
| spec | The full description of one kernel to build: every size, flag and setting. Building a spec produces the kernel code. |
| `algorithm`, `spec_id` | Two fields on the request that name a candidate. Both default to `auto`, meaning "you choose". |
| opt-in candidate | A candidate the dispatcher never chooses on its own. It is used only when the request names it with both `algorithm` and `spec_id`. Sweeps and benchmarks use these. |
| tuned candidate | A candidate with many possible settings (a **knob space**) instead of one fixed spec. Each attention tuning geometry and each dense candidate is one. |
| knob | One setting of a tuned kernel that changes speed but not the answer, such as an unroll choice or a buffer layout. |
| knob axis | One knob and the values it may take, declared as data (`KnobAxis`). |
| configuration | One tuned candidate with specific knob values set. |
| outer knob | A knob swept on top of every other knob setting because it is cheap to vary. Attention uses `waves_per_eu`, the GPU occupancy hint. |
| `tuning_id` | The name of one configuration, for example `narrow_nw1_mw16_t4xb_llvm_wpe2@4910058571931d95`. The part before `@` is for people; the part after is the `config_key`. |
| `config_key` | A short hash of the candidate's variant and its knob values. Two configurations are the same exactly when their keys match. |
| `tuning_knobs` | The knob values recorded next to a `tuning_id`, so the configuration can be rebuilt exactly. |
| sweep level | How much of a tuned candidate's space a sweep covers: `production` (a curated list) or `full` (everything legal, usually sampled). |
| `KernelId` | The identity of one selection, used by caches, logs and benchmark rows. |
| `PinRefused` | The error raised when a request names a kernel or configuration that no longer exists. The dispatcher never quietly picks something else. |
| LDS | The GPU's small on-chip shared memory. A kernel that needs more than the chip has cannot be compiled. |

## Where the code lives

| Area | Path | What it holds |
|---|---|---|
| Shared contracts | `platform/python/rocke/dispatch/core.py` | Requests, candidates, registries, results, identities, pinning helpers. |
| Shared tuning machinery | `platform/python/rocke/dispatch/tuning/` | Knob axes (`axes.py`), walking and sampling (`walk.py`), ids (`identity.py`), the knob space (`space.py`), the tuned-candidate factory (`candidate.py`), and the test kit (`testing.py`). It names no kernel setting. |
| GEMM, MoE, norm | `platform/python/rocke/dispatch/gemm/`, `families/` | Families whose kernels live in the platform. |
| Attention | `library/dispatch/attention/` | Per-GPU candidate modules, the attention knob space and its rules, Torch launch adapters, and the entry points. |
| KDA, GDN, convolution | `library/dispatch/kda/`, `gdn/`, `grouped_convolution.py` | Other library families on the same contracts. |
| Benchmarks | `library/benchmarks/` | Harnesses that run what dispatch offers and measure it. They never decide what is legal. |

Attention keeps two registries in `library/dispatch/attention/__init__.py`.
The **route registry** is what normal selection picks from; it includes two
labels, the unified 2D and 3D paths, that stand for "let the generic kernel
choose its own shape". The **execution registry** holds only candidates that
can be built and launched, and is what sweeps walk.

---

## 1. "Give me a kernel for this problem"

The everyday path: the caller leaves `algorithm` and `spec_id` at `auto`.

```text
dispatch_attention(req)                      library/dispatch/attention/__init__.py
 └─ ATTENTION_REGISTRY.select(req)           platform/.../dispatch/core.py
     ├─ supported(req): for each candidate, in priority order
     │    ├─ skip opt-in candidates (the request did not name them)
     │    └─ candidate.admits(req)
     │         ├─ capability.check(req)      cheap: GPU, data type, sizes
     │         └─ candidate._supports(req)   the remaining rules, in code
     └─ first admitted candidate wins
 └─ candidate.select_spec(req)               the spec for this problem
 └─ make_dispatch_result(...)                adds KernelId, grid, block, explanation
```

For attention, a few specialized kernels come first for the shapes they cover:
gfx942 fp16 2D prefill (`attention_gfx942_dense_pipe`), gfx950 bf16 head-size
256 prefill (`attention_gfx950_d256`), and bf16 head-size 256 decode
(`attention_d256_decode`). Everything else falls to one of the two unified
path labels, `attention_unified_2d` or `attention_unified_3d`, and the generic
kernel behind it picks its own tile shape for the problem. The result's
`explanation` lists the choice, and `result.build()` / `result.bind_torch()`
compile and launch it.

## 2. "Give me that specific kernel"

The caller names a candidate with both fields, for example
`algorithm="attention_dense_persist", spec_id="gfx950_dense_persist"`. On
gfx950 the `algorithm` names the dense kernel body (grid or persistent) and
the `spec_id` names the candidate under it (the persistent body has a second
one with wide DMA); the tile is a knob. gfx942 has one dense candidate,
`algorithm="attention_dense", spec_id="gfx942_dense"`, with persistence as a
knob.

```text
dispatch_attention(req)
 └─ select(req)
     ├─ opt-in candidates whose algorithm matches are now visible
     ├─ candidate.admits(req): checks the pin, then the problem
     └─ nothing admitted and spec_id is not "auto"
          └─ raise PinRefused(...)            with each named candidate's reason
```

An opt-in candidate admits a request only when **both** fields name it; its
refusal message says which fields to set. If the named candidate has been
removed, or refuses the problem, the caller gets `PinRefused` with the reason.
The dispatcher does not fall back to another kernel, because a caller who
pinned a kernel wants that kernel or an error.

## 3. "Give me one exact configuration of it"

For a tuned candidate the request can also carry `tuning_id` and
`tuning_knobs`, usually copied from a benchmark row. The candidate is built by
`make_tuned_candidate` (`dispatch/tuning/candidate.py`), and its
`select_spec` does this:

```text
resolve(req)                                 cached per request
 ├─ base(req)                                the problem the space builds from
 └─ resolve_pinned(req)
      ├─ no id and no knobs ──────────► space.default(base)   the candidate's default
      ├─ knobs given ─────────────────► space.canonicalize(base, knobs)
      │                                   then the result's config_key must
      │                                   equal the one in tuning_id
      └─ only an id ──────────────────► space.find(base, id)
                                          looks in the production list only
```

If the knobs no longer rebuild the same `config_key`, because a default or a
knob's meaning changed, the pin is refused rather than building a different
kernel. A bare id is only looked up in the curated production list, because
the full space can hold millions of configurations; a configuration from the
full space has to travel with its knobs.

Two helpers wrap this for callers: `attention_tuning_spec(req, spec_id,
tuning_id, knobs)` replays a recorded configuration, and
`tuning_spec_with_knobs(req, spec_id, knobs)` sets knob values on the default.

## 4. "Show me everything I could run" (sweeps)

Benchmarks want every candidate and configuration for a problem, opt-in ones
included.

```text
iter_dispatch_attention_all(req, sweep_level=..., tuning_sample=n)
 └─ iter_at_level(level, ...)                the level applies only while iterating
     └─ ATTENTION_EXECUTION_REGISTRY.iter_dispatch_all(req)
         └─ iter_combos(req)
             for each candidate (skipping other GPUs by capability):
              ├─ opt_in_probe(req, candidate)  copy of req naming this candidate
              ├─ candidate.admits(probe)
              ├─ n > 0 ? sample_space(probe, n, seed)
              │        : sweep_space(probe)    the curated list, or the whole space
              └─ yield (candidate, spec) for each spec
         └─ pin_to_spec(req, candidate, spec)  the stored request names this spec
         └─ make_dispatch_result(...)
```

`pin_to_spec` writes the spec's `tuning_id` and knobs into the result's
request, so the stored request selects exactly what ran, and two configurations
of one candidate get different request hashes. A spec that is not tuned clears
those fields instead, so an old pin is never carried along. A negative sample
count is refused; zero means "walk everything", which can be millions of specs
on the larger attention spaces.

`attention_sweep_space(req)` is the same walk returning only unified 2D/3D
specs, for the paged-attention benchmark.

## 5. How a tuned candidate makes its configurations

Every configuration a tuned candidate hands out, whether swept, sampled,
pinned by knobs or found by id, comes from one place: the candidate's
`KnobSpace` (`dispatch/tuning/space.py`). A family supplies a subclass with its
rules; attention has `UnifiedSpace` (`unified_rules.py`) and `DenseSpace`
(`dense_rules.py`), both built on `WavesPerEuSpace` (`waves.py`).

**Walking.** The knob axes are listed in `library/dispatch/attention/axes.py`,
with prerequisites first. The walk (`walk.py`) tries axis values one by one and
asks `is_valid` about each partial setting; a setting that fails is not
explored further, since no later knob can repair it. The exception is the
leading "enabler" axes, such as single-buffered K, which can make a too-large
setting fit, so the walk keeps going past a failure until those are decided.

| Level | Unified attention | Dense attention |
|---|---|---|
| `production` | Hand-picked knob stacks per codepath | The default, then each knob changed on its own |
| `full` | Every legal combination (usually sampled) | Every legal combination (usually sampled) |

**Checking one setting** (`KnobSpace.canonicalize`), in order:

| Step | What happens | Example |
|---|---|---|
| Fix types | Each value is converted to the type its axis declares; a value that cannot convert cleanly is refused. | `1` and `True` become the same `True`. |
| Refuse | Settings that must never be built are rejected with a reason. | A knob known to give wrong answers; a field that is not a knob; changing a value the variant fixes. |
| Build | The family builds the kernel spec; the kernel's own checks run. | An unsupported tile size is refused here. |
| Drop no-ops | Knobs that compile to the same kernel as leaving them out are removed. | A value equal to the default; a padded-K layout the kernel turns off. |
| Validate | The family's final legality check. | On gfx950 2D attention: the kernel would need more LDS than the chip has, or a padded K buffer that Q shares (this gave wrong output on MI355X). |
| Name it | The remaining knobs are hashed into `config_key`, and the family's `stem` adds the readable part of `tuning_id`. | `transposed32_nw2_mw32_t2xb_hipcc_wpenone@dc4ec33bb0fe3416` |

Dropping no-ops is what keeps "one kernel, one id": two different-looking knob
settings that produce the same kernel get the same `tuning_id`.

**The outer knob.** After a setting passes, the space rebuilds it once per
outer-knob value, without re-running the checks above. For attention the outer
knob is `waves_per_eu`, and the `_wpe{N}` part of the id shows its value.
The shared machinery knows nothing about `waves_per_eu`; attention's
`WavesPerEuSpace` names it.

## 6. "Run it and tell me how fast it is" (benchmarks)

The combo sweep (`library/benchmarks/common/attention_combo_sweep.py`) turns
sweep results into measurements. It never decides what is legal; dispatch
already did.

```text
iter_dispatch_attention_all(req, ...)        workstream 4
 └─ validate_config(result)                  on CPU worker processes
     ├─ candidate.admits(...)
     ├─ result.build()                       the kernel code
     └─ lower to LLVM or HIP and hash it      same hash as an earlier one → "duplicate"
 └─ run each new config in its own process   a crash stays a result, not the end
     ├─ child rebuilds the spec from (candidate, tuning_id, knobs)   workstream 3
     ├─ result.bind_torch(tensors)           library/dispatch/attention/bindings.py
     ├─ launch, then compare with a PyTorch reference
     └─ time the launches
 └─ one row per config: status, time, error, tuning_id, knobs, spec_hash
```

Running each configuration in a child process means a kernel that faults the
GPU is recorded as a crash and the sweep continues. The row carries everything
needed to replay it later.

## 7. "Use the winner from that benchmark later" (replay)

Store four fields from the benchmark row: `spec_id`, `tuning_id`, `knobs`, and
`spec_hash`. To use it:

1. Build the request with `algorithm` and `spec_id` set to the candidate's, and
   `tuning_id` / `tuning_knobs` from the row.
2. Call `dispatch_attention(req)`. That takes workstream 2, then 3.
3. If it raises `PinRefused`, the kernel, defaults, or identity schema changed
   since the benchmark. The error reports the stored id and the id those knobs
   produce now. Re-sweep and validate that replacement before updating the
   tuning store, or explicitly go back to `auto`; dispatch never substitutes it.
4. Compare `result.kernel_id.spec_hash` with the stored one. A mismatch means
   the compiled kernel would differ from the one measured; treat it as a miss.

`config_key` catches changed defaults inside dispatch, and `spec_hash` catches
anything else that changed the compiled kernel.

## 8. "What is registered for this GPU?"

These need no request:

| Question | Call |
|---|---|
| Which candidates declare gfx950? | `registry.for_arch("gfx950")` |
| Everything registered, as JSON | `registry.coverage()` |
| The candidate a stored `KernelId` names | `registry.resolve(kernel_id)`, which also refuses an id from an older kernel interface version |
| A candidate by name | `registry.get(name)` |

## Identity at a glance

| Name | Changes when | Used for |
|---|---|---|
| `spec_hash` | Anything in the spec changes, including problem sizes. | Naming the compiled kernel. |
| `KernelId.compile_key` | GPU, interface version, or spec changes. | Compile caches: every problem that selects the same spec shares one compile. |
| `KernelId.selection_key` | Anything above, or the request. | Dispatch logs and benchmark rows. |
| `config_key` | The variant, its knob values, or the defaults those knobs are relative to. Not problem sizes. | Naming a configuration across problems; checking a replayed pin. |
| `tuning_id` | Same as `config_key`, plus the readable stem. | What people and benchmark rows quote. |
