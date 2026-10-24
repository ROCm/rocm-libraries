# RFC: cuDNN v8 shim for hipDNN

- **Status**: Accepted; amended by §9 (builders over hipDNN, no emulated backend C API)
- **Authors**: Mitch Ousdahl
- **Follows**: [RFC 0012: cuDNN shim for hipDNN](./0012_CuDNN_Shim.md) §9
- **Amends**: RFC 0012 §1 ("Not"), §4.2 (v9-only translation units), §4.3 (aliasing of
  `DescriptorType_t`, `NormMode_t`, `RngDistribution_t`), §4.7 (v9-only umbrella), §4.8 (surface
  removal, §9 R4)
- **Supporting reference**: [RFC 0021 Supporting Reference](./0021_CuDNN_V8_Shim_Reference.md) —
  identifier mapping, class inventory, per-op rules, consumer surfaces, legacy samples, version gates

## Table of Contents

1. [Summary](#1-summary)
2. [Requirements](#2-requirements)
3. [Design](#3-design)
4. [Key Decisions](#4-key-decisions)
5. [Risks](#5-risks)
6. [Execution Plan](#6-execution-plan)
7. [Testing Plan](#7-testing-plan)
8. [Future Considerations](#8-future-considerations)
9. [Amendment: builders over hipDNN](#9-amendment-builders-over-hipdnn)

## 1. Summary

### Why

RFC 0012 covers the cuDNN frontend v9 graph API. Much cuDNN-consuming code still uses the older
frontend **v0.x builder API** (`TensorBuilder`, `OperationBuilder`, `ExecutionPlanBuilder`, …):

- PyTorch runs all cuDNN convolution through `Conv_v8.cpp` (on by default) and its int8 quantized ops
  through `native/quantized/cudnn/`.
- XLA/TensorFlow keep convolution, fusions and normalization on v0.x. They also read plans and engine
  configs through `cudnnBackendGetAttribute` directly, which the shim does not provide (§9 R1).
- PaddlePaddle's convolution path is v0.x.

None of these compile against the v9 shim. See
[Reference §4](./0021_CuDNN_V8_Shim_Reference.md#4-consumer-surfaces).

### What

Add to the existing shim:

1. The **v0.x builders**, ported from cudnn-frontend v1.24.0 and exposed through the existing
   `cudnn_frontend.h`, as upstream does. They build hipDNN graphs, engine configs, plans and variant
   packs directly, through an internal core (`detail/v8`).
2. The C surface the builders and their consumers name: C types and enums, `cudnnBackendExecute` and
   `cudnnGetLastErrorString`, both passing straight through to hipDNN. `cudnnBackendDescriptor_t` is
   `hipdnnBackendDescriptor_t`.

Graphs a hipDNN engine supports run. Everything else compiles and fails at runtime with
`CUDNN_STATUS_NOT_SUPPORTED`.

### Not in scope

- The cuDNN backend descriptor C API (`cudnnBackendCreateDescriptor`, `…SetAttribute`,
  `…GetAttribute`, `…Finalize`, `…DestroyDescriptor`) and its attribute-name and attribute-type enums.
  Code that calls it directly does not compile (§9 R1).
- The legacy v7 C API (`cudnnConvolutionForward`, `cudnnSetTensorNdDescriptor`, …).
- ABI compatibility; consumers recompile.
- Python bindings.
- Provider gaps: a correctly translated graph that no engine supports (for example int8 convolution)
  fails as it would on unsupported NVIDIA hardware.

## 2. Requirements

- Header-only, like RFC 0012.
- The full v0.x builder surface compiles name-for-name against cudnn-frontend v1.24.0, with the C
  identifiers of §3.7.
- Convolution (fwd / dgrad / wgrad), pointwise, matmul, reduction, resample and batch / layer / RMS
  norm translate to hipDNN and run on any engine that claims them.
- Everything else returns `CUDNN_STATUS_NOT_SUPPORTED` with a message naming the builder and the
  rule. Builders throw `cudnnException` unless `NV_CUDNN_DISABLE_EXCEPTION` is defined; no other
  exception type leaves a builder, getter, filter or C entry point. Nothing is silently dropped except
  the store-and-ignore items listed in §3.4.
- Per-call overhead over the hipDNN equivalent on the same plan, measured by
  `samples/cudnn_shim/ConvFpropV8.cpp --overhead`: `cudnnBackendExecute` < 1 µs;
  `VariantPackBuilder::build()` + `cudnnBackendExecute` < 5 µs. After the builders moved onto hipDNN
  the latter measures p50 0.76–0.97 µs on an idle gfx1151 host (p90 about 1.4 µs); the gate stays at
  5 µs because 1 µs leaves no margin on a shared CI runner.
- MIT licensed, with NVIDIA attribution on every derived file. Nothing copied from NVIDIA's proprietary
  cuDNN headers.
- No behavior change for v9 users.

## 3. Design

### 3.1 Layout

Everything lives in `projects/hipdnn/frontend/include/hipdnn_compatibility/cudnn/`, behind the existing
`HIPDNN_ENABLE_CUDNN_COMPATIBILITY` option. Public filenames match upstream.

```
cudnn.h                       extended: includes cudnn_graph.h
cudnn_graph.h                 new: cudnnBackendExecute, cudnnGetLastErrorString
cudnn_backend_base.h          new (port)
cudnn_frontend_shim.h         new (port, subset)
cudnn_frontend_utils.h        extended: cudnnException, to_string, enum converters
cudnn_frontend_Logging.h      new (port, subset)
cudnn_frontend_<X>.h          new (port): Tensor, ConvDesc, PointWiseDesc, MatMulDesc,
                              ReductionDesc, Resample, Rng, Operation, OperationGraph, VariantPack,
                              Heuristics, Engine, EngineConfig, EngineFallbackList, Filters,
                              ExecutionPlan, Errata, ExecutionPlanCache, Reorder_Tensor,
                              EngineConfigGenerator, find_plan, get_plan
cudnn_frontend.h              extended: v0.x headers and the _v8 aliases
detail/c_types.h              new: C types and enums
detail/type_mapping.h         new: cuDNN <-> hipDNN enum translation, by name
detail/v8/                    new, internal core:
    error.h  boundary.h  records.h  validate.h  lowering.h  engine_space.h
    engine_views.h  notes.h  plan.h  variant_pack.h
```

- `c_types.h` holds the C types and enums, including `cudnnHandle_t` (moved from `cudnn.h`).
  `cudnn_graph.h` compiles on its own.
- `detail/v8` is not public API. Builders call it; it calls hipDNN only through
  `hipdnn_frontend::detail::hipdnnBackend()`, so direct and runtime-loaded hipDNN both work and the mock
  backend works in tests.
- As upstream, `EngineConfigGenerator`, `find_plan` and `get_plan` are separate includes.
- A translation unit may mix v0.x builders, the v9 graph API and `cudnnBackendExecute`.
- Consumers that define their own cuDNN C symbols (for example dlopen stubs) must drop them; the shim
  defines those functions inline. `NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING` is ignored; runtime loading of
  hipDNN is chosen by linking `hipdnn_frontend_dynamic`.

### 3.2 Why translate instead of alias

The hipDNN backend has the same entry points as cuDNN's and accepts a graph built from individually
finalized operation descriptors, so the control flow maps one to one. The data does not:

- enum values differ almost everywhere;
- cuDNN's parameter descriptors (convolution, pointwise, matmul, reduction, resample, RNG) don't exist;
  hipDNN sets those attributes on the operation;
- pointwise slots differ (X/B/T/Y/DX/DY vs IN_0/IN_1/IN_2/OUT_0);
- hipDNN heuristics accept only FALLBACK mode;
- hipDNN has no numerical notes, JSON plans, alpha/beta or unified norm operation; knobs are strings and
  engine IDs are 64-bit hashes;
- hipDNN allocates new descriptors on get, where cuDNN fills the caller's.

So the shim never casts between cuDNN and hipDNN values; it translates by name. Only the handle and
descriptor types are aliased, so plans and variant packs pass to hipDNN unchanged.

Each builder's `build()` hands its values to the core, which applies the checks upstream's backend
would apply at finalize, then lowers to hipDNN. Builders keep upstream's public API and every member
that did not talk to the backend; the members that did are rewritten
([Reference §2.6](./0021_CuDNN_V8_Shim_Reference.md#26-ported-headers-rewritten-and-verbatim-members)).

### 3.3 Objects and ownership

- Every built object holds an immutable record in its `ManagedOpaqueDescriptor`: tensor, parameter
  descriptor, operation, graph, engine, engine config, plan or variant pack. `OpaqueBackendPointer`
  pairs that record with a non-owning view of the hipDNN descriptor the record owns;
  `get_raw_desc()` returns the view.
- The view is null for tensors, parameter descriptors, operations, heuristics and fallback lists. A
  graph shows its finalized hipDNN `OPERATIONGRAPH`, an engine and a knob-free config the graph's
  default `ENGINE` and `ENGINECFG`, a config with knob choices its own `ENGINECFG`, a plan its
  `EXECUTION_PLAN`, a variant pack its `VARIANT_PACK`
  ([Reference §2.5](./0021_CuDNN_V8_Shim_Reference.md#25-class--record--raw-descriptor)).
- Records are shared through `shared_ptr`, and hipDNN's own reference counts (plan → config → engine →
  graph) keep a plan valid after its graph, heuristics and config are destroyed. Copies of an
  `ExecutionPlan` share one record.
- Lazily computed state (engine enumeration, knob configs, plan JSON) is guarded by `std::call_once`.
- The core throws `ShimError`. Every builder, lazy getter, filter and C entry point calls it through
  `detail::v8::guarded`, which maps `ShimError` to its status, `std::bad_alloc` to `ALLOC_FAILED` and
  any other exception to `INTERNAL_ERROR`; the builder then reports through upstream's
  `set_error_and_throw_exception`. Filters return `false` on failure, as upstream.

### 3.4 Lowering to hipDNN

Descriptor builders and `OperationBuilder` validate at `build()`, so builders fail where upstream's do.
hipDNN descriptors are created only when `OperationGraphBuilder::build()` lowers the graph:

1. Tensors are merged by unique ID. Two tensors with the same ID but different shape, strides, type or
   flags are `BAD_PARAM`.
2. Operations are topologically sorted, keeping the user's order when it is already valid. Providers
   match fusions by node position, so order matters.
3. Each operation is lowered by filling `hipdnn_frontend`'s own node attributes and calling its
   validation and descriptor packers. Shape inference and plugin-version checks therefore match
   hipDNN's frontend exactly.

The graph must have a handle (`BAD_PARAM` otherwise; cuDNN ≥ 9.8 makes it optional). Graph build
runs no heuristics and needs no GPU.

**Tensors**

| cuDNN | hipDNN |
|---|---|
| alignment ≥ 16 | not set (16 is hipDNN's default; cuDNN's value is a lower bound) |
| alignment < 16 | forwarded; needs plugin API ≥ 1.3.0, which no provider reports yet |
| by-value | runtime pass-by-value tensor ([RFC 0016](./0016_RuntimePassByValueTensors.md)) |
| constant value | `HIPDNN_ATTR_TENSOR_VALUE_EXT`; bytes copied at `TensorBuilder::build()` |
| ragged offset | forwarded; needs plugin API ≥ 1.3.0 |
| vector count ≠ 1, reordering | `NOT_SUPPORTED` |

**Operations**

| cuDNN | hipDNN | Notes |
|---|---|---|
| Conv fwd / dgrad / wgrad | same | convolution parameters folded onto the op |
| Pointwise | pointwise | X→IN_0, B→IN_1, T→IN_2, Y→OUT_0; backward: DY→IN_0, X→IN_1, DX→OUT_0 |
| Matmul | matmul | padding value, M/N/K overrides → `NOT_SUPPORTED` |
| Reduction | reduction | |
| Resample fwd / bwd | resample | fractional values, NEAREST, BILINEAR, backward X/Y → `NOT_SUPPORTED` |
| Norm fwd / bwd | batch / layer / RMS norm | by `NORM_MODE`; compute type FLOAT; exp. avg. factor → momentum; instance, group, ada-layer, peer stats → `NOT_SUPPORTED` |
| Gen-stats, BN finalize, BN bwd weights, RNG, reshape, paged cache load | — | upstream's slot checks, then `NOT_SUPPORTED` |

**Other rules**

- **Pointwise defaults are not forwarded, except the ReLU lower clip.** Upstream sets every parameter
  of the mode, including defaults; MIOpen rejects a fusion if a lower-clip slope is present at all. The
  lower clip is always sent unless a slope is set, because hipBLASLt's ReLU epilogue requires it; every
  engine reads 0 as standard ReLU. Beside a slope a lower clip of 0 is not sent, so hipBLASLt cannot
  mistake leaky ReLU for plain ReLU. Non-default values are narrowed to float.
- **Alpha / beta.** Alpha ≠ 1 or beta ≠ 0 on any op → `NOT_SUPPORTED`. No hipDNN engine accepts the
  alternative (an extra scalar multiply).
- **Store and ignore:** NaN propagation (hipDNN has no equivalent).
- **`NOT_SUPPORTED` when set:** dynamic shapes, override shapes.

### 3.5 Engine selection

- **Modes.** All heuristic modes map to FALLBACK, with one WARN per mode per process (as RFC 0012 §6.3
  requires). SM-count and device-property targets → `NOT_SUPPORTED`.
- **Enumeration.** On the first engine query for a graph (`getEngineCount`, heuristics, `EngineBuilder`),
  the shim runs FALLBACK heuristics once, finalizes each default engine config and caches the results.
  Engines whose config fails are dropped. Requires a GPU. Heuristic results are this snapshot in rank
  order; building heuristics makes no further hipDNN call.
- **Engine index.** cuDNN engines are small integers; the shim numbers hipDNN engines `0..N-1` by sorted
  engine ID. Ranking is conveyed by result order only. Indices survive ranking changes but not changes
  to the set of engines (new provider, different GPU).
- **Notes.** Behavior notes map by name. `NONDETERMINISTIC` is reported for every engine not known to be
  deterministic, so deterministic filtering works; hipDNN's numerical-note attribute is never queried.
  No other numerical notes are reported.
- **Knobs.** hipDNN knobs that map onto a cuDNN knob type are shown; others are hidden, and choosing a
  hidden knob is `NOT_SUPPORTED`. A choice of `-1` is upstream's "unset" and is ignored unless the knob
  is shown and accepts `-1`. `EngineConfigBuilder::build()` checks the choices and builds the config's
  hipDNN `ENGINECFG`, so a knob failure surfaces there.

### 3.6 Plans and execution

- `ExecutionPlanBuilder::build()` creates the hipDNN plan on the graph's handle. Upstream's builder never
  sends the plan handle, and the shim ignores it too. `loadFromJson` requires `setHandle`.
- `cudnnBackendExecute` is `toCudnnStatus(hipdnnBackendExecute(handle, plan, pack))`: one hipDNN call
  and no checks of its own. A plan runs on whichever handle it is given. The hipDNN variant pack is built in
  `VariantPackBuilder::build()`.
- JSON plans: `getJsonRepresentation` always returns a valid JSON object with engine index, ID (as a
  string), name, notes, knobs and, when hipDNN can serialize the plan, a base64 payload.
  `loadFromJson` accepts only this format and requires the payload. Neither direction needs nlohmann.
- Kernel cache → `NOT_SUPPORTED`.
- `cudnnGetLastErrorString` is hipDNN's (thread-local, cleared on read). The core reads hipDNN's error
  on every hipDNN failure it handles or tolerates, so a builder message carries hipDNN text only when
  hipDNN failed.

### 3.7 Types, enum values and versions

- Every C identifier used by the v0.x headers, NVIDIA's legacy samples and the PyTorch consumers is
  declared, from a checked-in list extracted from those sources and pinned by a test, except the
  backend descriptor C API's five functions and its attribute-name and attribute-type enums
  (`cudnnBackendAttributeName_t`, `cudnnBackendAttributeType_t`), and `cudnnMoeGroupedMatmulMode_t`,
  which only enum converters named (§9 R4). The extractor lists each
  exclusion with its reason. Enums are removed or kept whole; a kept enum keeps every enumerator.
- Names are declared for source compatibility only and carry no MIT claim (as in RFC 0012 §6.4).
- Enum values are the shim's own: numbered from 0 in alphabetical order of the enumerator name, so they
  can't reproduce NVIDIA's numbering, and `*_COUNT` = last + 1. Exception: `cudnnBackendTensorReordering_t`
  follows the shim's `TensorReordering_t`.
- Values are append-only within a `hipdnn_frontend` major version, pinned by a test. Knob types appear
  in plan tags and serialized autotune results, so renumbering would break persisted data. Removing a
  whole enum retires its values.
- `DescriptorType_t`, `NormMode_t` and `RngDistribution_t` are shim-owned rather than aliased (amends RFC
  0012 §4.3): hipDNN has no counterpart. They become aliases if hipDNN adds one.
- `CUDNN_VERSION` stays 92200 and the frontend pin 1.24.0. Ported headers `static_assert` the pin so a
  bump forces a re-diff.
- `cudnnBackendInitialize` is not provided (deprecated; nothing calls it).

### 3.8 Porting rules

- Keep upstream's public API, and keep every member that does not touch the backend verbatim. One
  public signature changes: the errata helper `check_shape` takes the operation's
  `ManagedOpaqueDescriptor` and a shim slot enum instead of a raw descriptor and an attribute name
  (`check_rule` and `check_errata` are unchanged). Members
  that set or got backend attributes call the core instead, with upstream's check order and statuses;
  [Reference §2.6](./0021_CuDNN_V8_Shim_Reference.md#26-ported-headers-rewritten-and-verbatim-members)
  lists them per header for upstream re-diffs. Other edits:
  - namespace and include paths;
  - error messages name the builder and the rule, not attribute enums;
  - private members that only held backend descriptors (knob-info and knob-choice arrays) removed;
  - CUDA event timing → hipDNN's profiling-control descriptor on the handle's stream;
  - dynamic loading, CUDA-graph and v7 helpers removed;
  - friend declarations of v9 node classes removed;
  - edits needed to build under hipDNN's `-Werror` warning set, with no warning suppression.
- Fixed upstream bugs: `Knob::getMaxValue` and `RngDesc_v8::getUniformDistMin` return the wrong field;
  `VariantPackBuilder_v8::build` passes the data-pointer count as the `UNIQUE_IDS` count, which over-reads
  the caller's UIDs when fewer UIDs than pointers are given. It now passes the UID count, and the shim
  reports the mismatch as `BAD_PARAM`. `OperationGraphBuilder_v8::addOperation` on a builder with no
  operations yet stores the descriptor at `ops[-1]` (the count starts at -1); it now starts from 0.
  `OperationBuilder_v8::setcDesc` copies `CUDNN_DIM_MAX + 1` padding, dilation and stride entries into
  arrays of `CUDNN_DIM_MAX`; the arrays now hold `CUDNN_DIM_MAX + 1`. The errata `check_shape` reads
  the fifth element of its dimension array, which a 4-D tensor never writes; it is now zero.
  `time_sorted_plan` collects timed plans in a `std::set` ordered by time, which keeps
  only one of several plans with equal times; it now uses a `std::multiset`. `cudnnFindPlanAndCache`
  reads the front of the sorted plans even when none ran; it now fails with `NOT_SUPPORTED` (a thrown
  `cudnnException`, or under `NV_CUDNN_DISABLE_EXCEPTION` an unbuilt plan whose status is `BAD_PARAM`).
  A successful descriptor-builder `build()` after a failed one kept the earlier status; it now resets
  it. `EngineFallbackList_v8::get_raw_desc()` dereferenced a null pointer; it is now null. The
  misspelled alias `EngineHeuristicsBuilde` is kept.
- FE-enum JSON rendering goes through a shim name table. `describe()` output matches upstream, but the
  shim defines no nlohmann serializer for hipDNN's enums, so serializing a frontend enum directly yields
  an integer.
- Operation tags match upstream byte for byte (errata rules and plan caches match on them). Plan tags
  use the shim's engine index and knob values.
- JSON-dependent code (errata) compiles unless `CUDNN_FRONTEND_SKIP_JSON_LIB` or
  `HIPDNN_FRONTEND_SKIP_JSON_LIB` is defined.
- `cudnnReorderFilterAndBiasInt8x32` → `NOT_SUPPORTED`.

### 3.9 Alternatives considered

| Alternative | Why not |
|---|---|
| Alias `cudnnBackend*` to `hipdnnBackend*` and use upstream headers as-is | Compiles, then misbehaves (§3.2) |
| Emulate the backend descriptor C API and port the builders unchanged onto it (phases 1–11) | Built and then replaced (§9): a second descriptor model of about 9k lines in every TU, kept for one known raw caller (XLA's read-only autotune persistence) |
| C API in a compiled library | Breaks the header-only requirement |
| Separate `cudnn_frontend_v8.h` | Forces include changes in hipified code; kept as the fallback if compile time suffers |
| Compile-only stubs | Unblocks compilation but runs nothing |

## 4. Key Decisions

| Decision | Benefit | Cost |
|---|---|---|
| Builders over hipDNN through an internal core; no emulated C API | One descriptor model; hipDNN descriptors pass through unchanged | Raw C callers (XLA) don't compile; builders' backend-facing members differ from upstream |
| `cudnnBackendExecute` passes straight through | One hipDNN call; plans run on any handle | No check that the handle matches the plan's |
| Validate at each `build()`, lower at `OperationGraphBuilder::build()` | Errors where upstream raises them; tensors merged across ops | Shim manages record lifetimes |
| Shim-owned, alphabetical, append-only enum values | No proprietary numbering; stable persisted data | No ABI compatibility |
| Skip pointwise defaults and alignment ≥ 16 | Target graphs find engines | Must track cuDNN defaults |
| Alpha / beta → `NOT_SUPPORTED` | Clear error | Scaled graphs don't run |
| Engine index by engine ID | Stable across re-ranking | Differs from the v9 shim; renumbers when engines change |
| Lower through `hipdnn_frontend` nodes | Same behavior as hipDNN's frontend | Depends on `hipdnn_frontend` internals |
| Extend the existing umbrella | Unmodified includes and mixed v8/v9 files work | ~15k more lines per umbrella include |

## 5. Risks

| Risk | Mitigation |
|---|---|
| Compile time | The builders and the core are instantiated in every TU that includes `cudnn_frontend.h`; `cudnn.h` adds only the C types and the two passthroughs. Measured on single TUs and the v9 sample corpus. The split umbrella remains the fallback for the v0.x builder headers only |
| Licensing of declared names | Fewer names since the attribute enums went (§9 R4); names only, no MIT claim; shim-assigned values; legal review still required before any non-POC merge (deferred for the proof of concept) |
| FALLBACK-only heuristics and missing notes change engine choice | One WARN per mode; documented in `KNOWN_DIVERGENCES.md` |
| Unsupported by providers: int8, four-op conv fusions, alignment < 16, ragged tensors | Out of scope; expected failures in the corpus harness |
| Lifetime bugs | Records own hipDNN descriptors and objects hold views; tests destroy everything but the plan and execute it; ASan run |
| Core exceptions escaping builders (PyTorch catches only `cudnnException`) | `guarded` at every public entry; tested in the throwing and the `NV_CUDNN_DISABLE_EXCEPTION` binaries |
| A plan executed on another thread's handle (PyTorch's quantized ops cache plans process-wide) | Runs on that handle, pinned by unit and integration tests; documented in `KNOWN_DIVERGENCES.md` |
| Upstream drift | Pin `static_assert`; re-diff in RFC 0012 §6.1's quarterly upstream review, guided by the Reference §2.6 member list |
| `hipdnn_frontend` internals change | Same tree and version; GPU-free lowering tests catch breakage |
| Maintenance of ~7.7k ported lines and ~7k of core and type mapping | Same owner as the RFC 0012 shim |
| PyTorch compile check passes on empty files (PyTorch's ROCm build compiles the cuDNN code out) | The probe forces the code on in the PyTorch tree's own hipified files; a canary fails an empty file |
| Corpus harness not run by TheRock yet (TheRock's `HIPDNN_ENABLE_CUDNN_COMPATIBILITY` is OFF) | The superbuild CI Linux job compiles the v9 and legacy corpus on every run and runs it when a GPU is present (`/dev/kfd`) |
| PyTorch compile check needs a PyTorch ROCm build | The check is local; a C++20 twin of the PyTorch surface test runs in CI |

## 6. Execution Plan

Each phase builds and is tested on its own. Each adds its own `KNOWN_DIVERGENCES.md` entries.

| Phase | Content |
|---|---|
| 0 | This RFC |
| 1 | C types, descriptor model, entry points, tensor and parameter descriptors |
| 2 | Operation descriptors |
| 3 | Graph lowering, engine enumeration |
| 4 | Heuristics, engines, plans, variant packs, execute, JSON |
| 5 | Builders: base, tensor, parameter descriptors, variant pack |
| 6 | Builders: operations, operation graph |
| 7 | Builders: heuristics, plans, filters, errata, cache, find / get plan |
| 8 | GPU samples and overhead measurement |
| 9 | Legacy-samples corpus harness |
| 10 | PyTorch compile check |
| 11 | Docs and final validation |
| 12 | Core extracted from the emulated C API into `detail/v8`, no behavior change |
| 13 | Builder-level tests for every rule, passing on the emulated C API |
| 14 | Cutover: builders over the core; descriptor C API removed except `cudnnBackendExecute` |
| 15 | Surface shrink (§9 R4) |
| 16 | Docs and overhead measurement |
| 17 | Final validation |

Phases 1–4 built the emulated C API that phases 12–15 replaced.

## 7. Testing Plan

| Tier | Proves |
|---|---|
| Unit tests on the mock backend (`hipdnn_frontend_tests`) | Core rules without builders (`TestCudnnShimV8Core*`); builders, lowering, tags, filters, errata, plan search, thread safety; error boundary, raw-descriptor table, plan lifetime, `cudnnBackendExecute` passthrough (`TestCudnnShimV8BuilderContracts`, `TestCudnnShimV8CApiStandalone`) |
| Separate binary with `NV_CUDNN_DISABLE_EXCEPTION` | Status-return path, including the error boundary (can't share a binary with the throwing build) |
| Binary linked against `hipdnn_frontend_dynamic` only | Builders, filters and `cudnnBackendExecute` link without the hipDNN backend library |
| Compile-only targets | JSON-off builds, public surface pins, mixed v8/v9 file, `cudnn_graph.h` standalone |
| Integration, real backend, no GPU | Every lowered operation passes hipDNN's descriptor validation |
| Integration, test plugins, GPU | Engine enumeration, plans, execute (including on another handle), knobs, multiple engines, plan search |
| GPU samples (MIOpen, hipBLASLt) | Numerical results, overhead |
| v9 sample corpus | Existing v9 behavior unchanged |
| NVIDIA legacy samples | Source compatibility with per-case expectations ([Reference §5](./0021_CuDNN_V8_Shim_Reference.md#5-legacy-sample-catalogue)) |
| PyTorch compile check | `Conv_v8.cpp` and the quantized cuDNN files compile with `torch_hip`'s flags |

## 8. Future Considerations

- Alpha / beta, once a provider accepts scalar operands in pointwise fusions.
- Numerical notes, if providers start declaring them.
- More operations (gen-stats, BN finalize, RNG, reshape, instance / group norm) as hipDNN gains them.
- A split umbrella if compile time becomes a problem.
- A read-only descriptor query for raw C callers such as XLA, only on a concrete request.
- The legacy v7 C API only on a concrete request; separate RFC.

## 9. Amendment: builders over hipDNN

Phases 1–11 shipped the original design: an emulated backend descriptor C API (about 9.1k lines of shim
descriptors, an attribute table and reference-counted shim handles) with the builders ported unchanged
on top. Phases 12–17 replaced it with the design above. Sections 1–8 describe the result; this section
records the decisions.

| # | Decision |
|---|---|
| R1 | Drop the backend descriptor C API except `cudnnBackendExecute`. This reverses the earlier rejection of "C++ builders straight over hipDNN, no public C API", which rested on raw C callers. The only one found is XLA's autotune persistence, which reads plans and engine configs through `cudnnBackendCreateDescriptor` / `GetAttribute` / `DestroyDescriptor` (`cuda_dnn.cc` L5081-5199 at the [Reference §4](./0021_CuDNN_V8_Shim_Reference.md#4-consumer-surfaces) commit); it no longer compiles. PyTorch's, Paddle's and XLA's builder code stays source compatible |
| R2 | `cudnnBackendExecute` is a passthrough. The emulation rejected a plan executed on a handle other than its graph's with `BAD_PARAM`; now hipDNN runs it on the given handle. Pinned by unit and integration tests |
| R3 | NVIDIA's legacy "Tensor creation comparison" calls the raw API. A harness-only `tests/cudnn_samples/legacy/overlay/cudnn.h` declares the calls it makes as `NOT_SUPPORTED` stubs, plus the five tensor attribute names and two attribute types it passes; the case is an expected failure. The stub is never installed under `include/` |
| R4 | Unused identifiers are removed by whole function or whole enum only: the five descriptor functions, `cudnnBackendAttributeName_t`, `cudnnBackendAttributeType_t`, and `cudnnMoeGroupedMatmulMode_t` (named only by enum converters, removed with them). Kept enums keep every enumerator and value. This shrinks the surface, which RFC 0012 §4.8 forbids within a major version; it is waived for the proof of concept, and a non-POC release would need a `hipdnn_frontend` major version bump |
| R5 | Error messages may change. Builder messages name the builder and the rule, not attribute enums; the legacy-corpus anchors and test message checks are re-pinned |
| R6 | `cudnnBackendDescriptorType_t` stays whole: PyTorch's plan-cache key, Paddle's template argument and `OperationBuilder`'s constructor argument |
| R7 | `cudnnGetLastErrorString` is hipDNN's; see §3.6 |
| R8 | Unchanged: plan JSON format, tags (including `_k<type>=<value>`), dense engine index, knob projection, `-1` unset knob, ReLU-clip rule, default elision, UID merge, stable sort, disconnected graphs, every status code except R2's |
| R9 | Per-call overhead re-measured on the new implementation: `VariantPackBuilder::build()` + `cudnnBackendExecute` p50 0.76–0.97 µs on an idle host, p90 about 1.4 µs. The gate stays at 5 µs (not 1 µs) so a shared CI runner does not flake; see §2 |

Also amended by this section: §3.7's "every identifier is declared" now excludes R4's removals, and
§3.8's "port upstream headers unchanged in behavior" is now "unchanged public API and verbatim
non-backend members".
