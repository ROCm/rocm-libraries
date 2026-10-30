# RFC 0022: cuDNN v8 shim for hipDNN

- **Authors**: Mitch Ousdahl
- **Status**: In progress
- **Status notes**: The v0.x builders, their `detail/v8` core and the C types, `cudnnBackendExecute`
  and `cudnnGetLastErrorString` ship in `frontend/include/hipdnn_compatibility/cudnn/` behind
  `HIPDNN_ENABLE_CUDNN_COMPATIBILITY` (default `OFF`, #13970), with v8 GPU samples, the legacy-sample
  corpus and a local PyTorch compile probe. The backend descriptor C API is not provided, no PyTorch
  build runs on the shim end to end, and CI does not yet run the matmul, layer-norm and resample
  samples numerically (see the Roadmap).
- **Last verified**: 2026-10-30 against `develop` (d730d16216b)
- **Follows**: [RFC 0012: cuDNN shim for hipDNN](./0012_CuDNN_Shim.md) §9
- **Amends**: RFC 0012 §1 ("Not"), §4.2 (v9-only translation units), §4.3 (aliasing of
  `DescriptorType_t`, `NormMode_t`, `RngDistribution_t`), §4.7 (v9-only umbrella)
- **Supporting reference**: [RFC 0022 Supporting Reference](./0022_CuDNN_V8_Shim_Reference.md) —
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
9. [Design decisions](#9-design-decisions)

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
[Reference §4](./0022_CuDNN_V8_Shim_Reference.md#4-consumer-surfaces).

### What

Add to the existing shim:

1. The **v0.x builders**, ported from cudnn-frontend v1.24.0 and exposed through the existing
   `cudnn_frontend.h`, as upstream does. They build hipDNN graphs, engine configs, plans and variant
   packs directly, through an internal core (`detail/v8`).
2. The C surface the builders and their consumers name: C types and enums, `cudnnBackendExecute` and
   `cudnnGetLastErrorString`, both passing straight through to hipDNN. `cudnnBackendDescriptor_t` is
   `hipdnnBackendDescriptor_t`.

Graphs a hipDNN engine supports run. Everything else compiles; at run time it fails to build with
`CUDNN_STATUS_NOT_SUPPORTED` or `CUDNN_STATUS_BAD_PARAM`, or builds and finds no engine (an empty
heuristics list).

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
  `VariantPackBuilder::build()` + `cudnnBackendExecute` < 5 µs. The latter measures p50 0.76–0.97 µs
  on an idle gfx1151 host (p90 about 1.4 µs). Wall-clock noise on a busy host exceeds 1 µs, so the
  test (ctest label `perf`) reports a missed budget and fails on it only with `--enforce-budget`.
- MIT licensed, with NVIDIA attribution on every derived file. Nothing copied from NVIDIA's proprietary
  cuDNN headers.
- No change to v9 graph results. The umbrella also declares the cuDNN C types and enums,
  `CUDNN_DIM_MAX`, `cudnnBackendExecute` and `cudnnGetLastErrorString` (§3.1), and the `OPENSOURCE`
  heuristic-mode warning is logged once per process, as RFC 0012 §4.5 specifies.

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
detail/c_types.h              new: C types and enums (generated, §3.7)
detail/frontend_types.h       new: FE enums, knob types and helpers, re-exported by
                              cudnn_frontend_utils.h
detail/type_mapping.h         new: cuDNN <-> hipDNN enum translation, by name
detail/v8/                    new, internal core:
    error.h  boundary.h  records.h  validate.h  lowering.h  build_descriptor.h
    engine_space.h  engine_views.h  notes.h  plan.h  variant_pack.h
```

- `c_types.h` holds the C types and enums, including `cudnnHandle_t` (moved from `cudnn.h`), and ends
  with a `detail::LAST_ENUMERATOR<E>` bound per enum that the core's range checks read.
  `cudnn_graph.h` compiles on its own.
- `detail/v8` is not public API. Builders call it; it calls hipDNN only through
  `hipdnn_frontend::detail::hipdnnBackend()`, so direct and runtime-loaded hipDNN both work and the mock
  backend works in tests.
- As upstream, `EngineConfigGenerator`, `find_plan` and `get_plan` are separate includes.
- A translation unit may mix v0.x builders, the v9 graph API and `cudnnBackendExecute`.
- Consumers that declare their own cuDNN C types, enums or functions (for example dlopen stubs) must
  drop them; the shim declares them and defines the functions inline.
  `NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING` is ignored; runtime loading of hipDNN is chosen by linking
  `hipdnn_frontend_dynamic`.

### 3.2 Why translate instead of alias

The hipDNN backend has the same entry points as cuDNN's and accepts a graph built from individually
finalized operation descriptors, so the control flow maps one to one. The data does not:

- enum values differ almost everywhere;
- cuDNN's parameter descriptors (convolution, pointwise, matmul, reduction, resample, RNG) don't exist;
  hipDNN sets those attributes on the operation;
- pointwise slots differ (X/B/T/Y/DX/DY vs IN_0/IN_1/IN_2/OUT_0);
- hipDNN's heuristic descriptor has one mode, FALLBACK; cuDNN's modes A and B correspond to hipDNN
  prediction policies, set as a policy order;
- hipDNN has no numerical notes, JSON plans, alpha/beta or unified norm operation; knobs are strings and
  engine IDs are 64-bit hashes;
- hipDNN allocates new descriptors on get, where cuDNN fills the caller's.

So the shim never casts between cuDNN and hipDNN values; it translates by name. Only the handle and
descriptor types are aliased, so plans and variant packs pass to hipDNN unchanged.

Each builder's `build()` hands its values to the core, which applies the checks upstream's backend
would apply at finalize, then lowers to hipDNN. Builders keep upstream's public API and every member
that did not talk to the backend; the members that did are rewritten
([Reference §2.6](./0022_CuDNN_V8_Shim_Reference.md#26-ported-headers-rewritten-and-verbatim-members)).

### 3.3 Objects and ownership

- Every built object holds an immutable record in its `ManagedOpaqueDescriptor`: tensor, parameter
  descriptor, operation, graph, engine, engine config, plan or variant pack. `OpaqueBackendPointer`
  pairs that record with a non-owning view of the hipDNN descriptor the record owns;
  `get_raw_desc()` returns the view. `OpaqueBackendPointer` adds one public member to upstream's: a
  constructor that takes the record together with a key only `detail::v8::RecordAccess` creates, so
  code outside the core cannot call it. The record and the descriptor type are private, and the core
  reaches them through `detail::v8::RecordAccess`.
- The view is null for tensors, parameter descriptors, operations, heuristics and fallback lists, and
  for any object whose `build()` failed. A graph shows its finalized hipDNN `OPERATIONGRAPH`, an
  engine and a knob-free config the graph's default `ENGINE` and `ENGINECFG`, a config with knob
  choices its own `ENGINECFG`, a plan its `EXECUTION_PLAN`, a variant pack its `VARIANT_PACK`
  ([Reference §2.5](./0022_CuDNN_V8_Shim_Reference.md#25-class--record--raw-descriptor)).
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
| by-value | runtime pass-by-value tensor ([RFC 0016](./0016_RuntimePassByValueTensors.md)); `BAD_PARAM` on a virtual tensor |
| constant value | `HIPDNN_ATTR_TENSOR_VALUE_EXT`; bytes copied at `TensorBuilder::build()`. The value's size must equal the data type's element size, except that a wider integer whose value fits an `INT8`, `UINT8`, `INT32` or `BOOLEAN` tensor is stored as one element; `BAD_PARAM` otherwise and on a virtual tensor |
| ragged offset | forwarded; needs plugin API ≥ 1.3.0 |
| vector count > 1, reordering | `NOT_SUPPORTED` (a count ≤ 1 is not sent, as upstream) |

**Operations**

| cuDNN | hipDNN | Notes |
|---|---|---|
| Conv fwd / dgrad / wgrad | same | convolution parameters folded onto the op |
| Pointwise | pointwise | X→IN_0, B→IN_1, T→IN_2, Y→OUT_0; backward: DY→IN_0, X→IN_1, DX→OUT_0 |
| Matmul | matmul | padding value, M/N/K overrides → `NOT_SUPPORTED` |
| Reduction | reduction | |
| Resample fwd / bwd | resample | fractional values, NEAREST, BILINEAR, `EDGE_VAL_PAD` padding, backward X/Y, an index tensor outside MAXPOOL → `NOT_SUPPORTED` |
| Norm fwd / bwd | batch / layer / RMS norm | by `NORM_MODE`; compute type FLOAT; exp. avg. factor → momentum; instance, group, ada-layer, peer stats → `NOT_SUPPORTED` |
| Gen-stats, BN finalize, BN bwd weights, RNG, reshape, paged cache load | — | upstream's slot checks, then `NOT_SUPPORTED` |

**Other rules**

- **Pointwise defaults are not forwarded.** Upstream sets every parameter of the mode, including
  defaults; MIOpen rejects a fusion if a lower-clip slope is present at all. Plain ReLU therefore
  reaches hipDNN with no parameters, as from the v9 API, and relies on the providers reading an unset
  lower clip as 0 (§9 R9). Non-default values are narrowed to float, so an infinite lower clip
  saturates; an upper clip at or above `FLT_MAX` clips nothing and is not sent. A NaN clip is
  `BAD_PARAM`.
- **Alpha / beta.** Alpha ≠ 1 or beta ≠ 0 on any op → `NOT_SUPPORTED`. No hipDNN engine accepts the
  alternative (an extra scalar multiply). A pointwise alpha2 is checked only on modes with a B input
  and on activation-backward modes; cuDNN ignores it on one-input forward modes.
- **Store and ignore:** NaN propagation (hipDNN has no equivalent); the epsilon of batch-norm
  inference, batch-norm backward and RMS-norm backward (hipDNN's operations take inverse variance or
  saved statistics instead; see `KNOWN_DIVERGENCES.md`).
- **`NOT_SUPPORTED` when set:** dynamic shapes, override shapes.

### 3.5 Engine selection

- **Modes.** The v0.x builders run hipDNN's fallback selection for every heuristic mode, with one WARN
  per mode other than FALLBACK per process (as RFC 0012 §6.3 requires); `get_heuristics_list` accepts
  `HeurMode_t::OPENSOURCE` the same way. The v9 graph API instead maps A and B to hipDNN's ModeA and
  ModeB prediction policies and falls back only for OPENSOURCE. A positive SM-count target and a
  device-property target → `NOT_SUPPORTED`; an SM count of 0 is the all-SM default.
- **Enumeration.** On the first engine query for a graph (`getEngineCount`, heuristics, `EngineBuilder`),
  the shim runs FALLBACK heuristics once, finalizes each default engine config and caches the results.
  Engines whose config fails are dropped. Requires a GPU. Heuristic results are this snapshot in rank
  order; building heuristics makes no further hipDNN call.
- **Engine index.** cuDNN engines are small integers; the shim numbers hipDNN engines `0..N-1` by sorted
  engine ID. Ranking is conveyed by result order only. Indices survive ranking changes but not changes
  to the set of engines (new provider, different GPU). An index past the engine count is
  `NOT_SUPPORTED`, as in cuDNN; a negative one is `BAD_PARAM`. Only plan tags and `setGlobalEngineIdx`
  use the index; plan JSON identifies the engine by ID, so a saved plan still loads after the set
  changes.
- **Notes.** Behavior notes map by name. `NONDETERMINISTIC` is reported for every engine not known to be
  deterministic, so deterministic filtering works; hipDNN's numerical-note attribute is never queried.
  No other numerical notes are reported. The allowlist is the v9 shim's (`isDeterminismClaimed`), so both
  shims keep the same engines. v0.x filtering runs in the caller, so an empty result reaches it as an
  empty list where the v9 shim returns `GRAPH_NOT_SUPPORTED`.
- **Knobs.** hipDNN knobs that map onto a cuDNN knob type are shown; others are hidden, and choosing a
  hidden knob is `NOT_SUPPORTED`. A choice of `-1` is upstream's "unset" and is ignored unless the knob
  is shown and accepts `-1`. `EngineConfigBuilder::build()` checks the choices and builds the config's
  hipDNN `ENGINECFG`, so a knob failure surfaces there.

### 3.6 Plans and execution

- `ExecutionPlanBuilder::build()` creates the hipDNN plan on the graph's handle. Upstream's builder never
  sends the plan handle, and the shim ignores it too. `loadFromJson` requires `setHandle`.
- `cudnnBackendExecute` is `toCudnnStatus(hipdnnBackendExecute(handle, plan, pack))`: one hipDNN call
  and no checks of its own. A plan runs on whichever handle it is given; hipDNN rejects a handle on
  another device than the plan's with `BAD_PARAM`. The hipDNN variant pack is built in
  `VariantPackBuilder::build()`.
- JSON plans: `getJsonRepresentation` always returns a valid JSON object with engine index, ID (as a
  string), name, notes, knobs, the requested knob choices when they differ from the applied ones (so a
  loaded plan's tag keeps its `-1` entries) and, when hipDNN can serialize the plan, a base64 payload.
  `loadFromJson` accepts only this format and requires the payload. Neither direction needs nlohmann.
- Kernel cache → `NOT_SUPPORTED`.
- `cudnnGetLastErrorString` is hipDNN's (thread-local, cleared on read). The core reads hipDNN's error
  on every hipDNN failure it handles or tolerates, so a builder message carries hipDNN text only when
  hipDNN failed. Profiling failures during plan search log hipDNN's message at INFO instead.

### 3.7 Types, enum values and versions

- Every C identifier that the v0.x headers, NVIDIA's legacy samples and the PyTorch consumers name is
  declared, from a checked-in list extracted from those sources and pinned by a test, except the
  exclusions the extractor lists with a reason each, among them the backend descriptor C API's five
  functions and its attribute-name and attribute-type enums (`cudnnBackendAttributeName_t`,
  `cudnnBackendAttributeType_t`), `cudnnMoeGroupedMatmulMode_t`, which only upstream's enum
  converters name (§9 R4), the v7 filter and reorder API, the CUDA-graph and causal-conv1d API, and
  `cudnnGetExecutionPlanWorkspaceSize`. The extractor takes the cudnn-frontend repository, tag and
  commit from the cudnn-samples harness manifest (`tests/cudnn_samples/manifest.json`, `"corpus"`), so
  one pin serves both; `pins.json` holds the sha256 digests of the scanned files and the PyTorch pin,
  and the extractor refuses inputs that do not match. `c_types.h` is generated from the same table
  (`--emit-c-types`).
- Enums are declared or left out whole relative to the extracted list: a declared enum holds every
  enumerator the sources name. An enumerator none of them names, such as cuDNN's
  `CUDNN_NOT_PROPAGATE_NAN`, is not declared.
- Names are declared for source compatibility; `detail/c_types.h` carries the NVIDIA notice for the
  upstream headers that name them (RFC 0012 §6.4).
- Enum values are the shim's own: numbered from 0 in ASCII order of the enumerator name, so they
  can't reproduce NVIDIA's numbering, and each count sentinel is last + 1 (an `ENUM_COUNT` row in the
  pinned value table). Exceptions: `cudnnStatus_t` keeps the
  v9 shim's 0–14 numbering, and `cudnnBackendTensorReordering_t` follows the shim's
  `TensorReordering_t`. An alias has no value of its own: `CUDNN_RESAMPLE_AVGPOOL` equals
  `CUDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING`, as in cuDNN, and the pinned value table lists it as an
  `ENUM_ALIAS` row.
- Values are append-only within a `hipdnn_frontend` major version, pinned by a test. Knob types appear
  in plan tags and serialized autotune results, so renumbering would break persisted data. Removing a
  whole enum retires its values.
- `DescriptorType_t`, `NormMode_t` and `RngDistribution_t` are shim-owned rather than aliased (RFC
  0012 §4.3): hipDNN has no counterpart. They can become aliases only at a `hipdnn_frontend` major
  version, and only of a hipDNN enum that has every enumerator the shim declares, because the change
  alters the parameter types of public builder setters and helpers (RFC 0012 §4.8).
- `CUDNN_VERSION` stays 92200 and the frontend pin 1.24.0. Ported headers `static_assert` the pin so a
  bump forces a re-diff.
- `cudnnBackendInitialize` is not provided (deprecated; nothing calls it).

### 3.8 Porting rules

- Keep upstream's public API, and keep every member that does not touch the backend verbatim. One
  public signature differs from upstream: the errata helper `check_shape` takes the operation's
  `ManagedOpaqueDescriptor` and a shim slot enum instead of a raw descriptor and an attribute name
  (`check_rule` and `check_errata` match upstream). Members
  that set or got backend attributes call the core instead, with upstream's check order and statuses;
  [Reference §2.6](./0022_CuDNN_V8_Shim_Reference.md#26-ported-headers-rewritten-and-verbatim-members)
  lists them per header for upstream re-diffs. Other edits:
  - namespace and include paths;
  - upstream's global `json` alias is declared in the shim namespace as `cudnn_frontend::json`, so it
    does not collide with a consumer's own global `json` unless `using namespace cudnn_frontend;` is
    in scope and the consumer's alias names a different type; such code names its alias as `::json`;
  - error messages name the builder and the rule, not attribute enums;
  - private members that only held backend descriptors (knob-info and knob-choice arrays) removed;
  - CUDA event timing → hipDNN's profiling-control descriptor on the handle's stream;
  - dynamic loading, CUDA-graph and v7 helpers removed;
  - friend declarations of v9 node classes removed;
  - edits needed to build under hipDNN's `-Werror` warning set, with no warning suppression.
- Fixed upstream bugs, upstream's behavior first:
  - `Knob::getMaxValue` and `RngDesc_v8::getUniformDistMin` return the wrong field; the shim returns
    the maximum and the uniform minimum.
  - `VariantPackBuilder_v8::build` passes the data-pointer count as the `UNIQUE_IDS` count, which
    over-reads the caller's UIDs when fewer UIDs than pointers are given; the shim passes the UID count
    and reports the mismatch as `BAD_PARAM`.
  - `VariantPackBuilder_v8::setDataPointers` and `setUids` and
    `OperationGraphBuilder_v8::setOperationGraph` pass a negative count to `reserve` or `resize`, which
    throws `std::length_error` even under `NV_CUDNN_DISABLE_EXCEPTION`; the shim reports `BAD_PARAM`.
  - `OperationGraphBuilder_v8::addOperation` on a builder with no operations yet stores the descriptor
    at `ops[-1]` (the count starts at -1); the shim starts from 0.
  - `OperationBuilder_v8::setcDesc` copies `CUDNN_DIM_MAX + 1` padding, dilation and stride entries
    into arrays of `CUDNN_DIM_MAX`; the shim's arrays hold `CUDNN_DIM_MAX + 1`.
  - The errata `check_shape` reads the fifth element of its dimension array, which a 4-D tensor never
    writes; the shim's is zero.
  - `time_sorted_plan` collects timed plans in a `std::set` ordered by time, which keeps only one of
    several plans with equal times; the shim uses a `std::multiset`. Upstream also ignores the status
    of each timed execution and its CUDA event errors, and keeps the span or the previous or initial
    time of 0, which can rank an unmeasured plan first; the shim gives a plan whose timed execution or
    timing fails on any run the time `FLT_MAX`, so it sorts after every measured plan.
  - `cudnnFindPlanAndCache` reads the front of the sorted plans even when none ran; the shim fails with
    `NOT_SUPPORTED` (a thrown `cudnnException`, or under `NV_CUDNN_DISABLE_EXCEPTION` an unbuilt plan
    whose status is `BAD_PARAM`).
  - A successful descriptor-builder `build()` after a failed one keeps the earlier status, and
    `TensorBuilder_v8::build` refuses to run again after a failure past its prechecks ("already
    pointing to something"). The shim's descriptor builders share one `build()` tail,
    `detail::v8::buildDescriptor`, which resets the status; `TensorBuilder` keeps its ragged offset
    and constant in the builder, so a fixed builder builds again, and only a builder that already
    built a tensor is `BAD_PARAM`.
  - `get_raw_desc()` dereferences a null pointer after a `build()` that failed before creating it
    (reachable under `NV_CUDNN_DISABLE_EXCEPTION`); the shim returns null.
  - `get_heuristics_list(std::vector<HeurMode_t>, ...)` throws `std::out_of_range` for
    `HeurMode_t::OPENSOURCE`; the shim runs fallback selection (§3.5) and reports a value outside the
    enum as `BAD_PARAM`.
- The misspelled alias `EngineHeuristicsBuilde` is kept.
- FE-enum JSON rendering goes through a shim name table. `describe()` output matches upstream, but the
  shim defines no nlohmann serializer for hipDNN's enums, so serializing a frontend enum directly yields
  an integer.
- Operation tags match upstream byte for byte (errata rules and plan caches match on them). Plan tags
  use the shim's engine index and knob values.
- JSON-dependent code (errata, the `json` alias) compiles unless `CUDNN_FRONTEND_SKIP_JSON_LIB` or
  `HIPDNN_FRONTEND_SKIP_JSON_LIB` is defined.
- `cudnnReorderFilterAndBiasInt8x32` → `NOT_SUPPORTED`.

### 3.9 Alternatives considered

| Alternative | Why not |
|---|---|
| Alias `cudnnBackend*` to `hipdnnBackend*` and use upstream headers as-is | Compiles, then misbehaves (§3.2) |
| Emulate the backend descriptor C API and port the builders unchanged onto it | A second descriptor model of about 9k lines in every TU, kept for one known raw caller (XLA's read-only autotune persistence; §9 R1) |
| C API in a compiled library | Breaks the header-only requirement |
| Separate `cudnn_frontend_v8.h` | Forces include changes in hipified code; kept as the fallback if compile time suffers |
| Compile-only stubs | Unblocks compilation but runs nothing |

## 4. Key Decisions

| Decision | Benefit | Cost |
|---|---|---|
| Builders over hipDNN through an internal core; no emulated C API | One descriptor model; hipDNN descriptors pass through unchanged | Raw C callers (XLA) don't compile; builders' backend-facing members differ from upstream |
| `cudnnBackendExecute` passes straight through | One hipDNN call; plans run on any handle on the plan's device | No check of its own; hipDNN checks only the handle's device (§5) |
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
| Compile time | The builders and the core are instantiated in every TU that includes `cudnn_frontend.h`; `cudnn.h` adds the C types, the two passthroughs and the core's error boundary (`detail/v8/boundary.h`, `error.h`). Measured on single TUs and the v9 sample corpus. The split umbrella remains the fallback for the v0.x builder headers only |
| Licensing of declared names | Names only, with the NVIDIA notice in `detail/c_types.h`; shim-assigned values; the legal review is complete |
| v0.x heuristics run fallback selection only, and missing notes change engine choice | One WARN per mode; documented in `KNOWN_DIVERGENCES.md` |
| Unsupported by providers: int8, four-op conv fusions, alignment < 16, ragged tensors | Out of scope; expected failures in the corpus harness |
| Lifetime bugs | Records own hipDNN descriptors and objects hold views; tests destroy everything but the plan and execute it; ASan run |
| Core exceptions escaping builders (PyTorch catches only `cudnnException`) | `guarded` at every public entry; tested in the throwing and the `NV_CUDNN_DISABLE_EXCEPTION` binaries |
| A plan executed on another thread's handle (PyTorch's quantized ops cache plans process-wide) | Runs on that handle, pinned by unit and integration tests; documented in `KNOWN_DIVERGENCES.md` |
| Plain ReLU and ReLU6 after a matmul need the hipBLASLt provider to read an unset lower clip as 0 (§9 R9) | The hipBLASLt provider does so since rocm-libraries#14363; with an older provider those graphs have no hipBLASLt engine |
| A plan executed on a handle for another device | The shim does not check it; hipDNN's `hipdnnBackendExecute` rejects it with `BAD_PARAM` before any provider call (rocm-libraries#14364). PyTorch keys every v8 plan cache on the device, so it never does this |
| Upstream drift | Pin `static_assert`; re-diff in RFC 0012 §6.1's quarterly upstream review, guided by the Reference §2.6 member list |
| `hipdnn_frontend` internals change | Same tree and version; GPU-free lowering tests catch breakage |
| Maintenance of ~7.7k ported lines and ~7k of core and type mapping | Same owner as the RFC 0012 shim |
| PyTorch compile check passes on empty files (PyTorch's ROCm build compiles the cuDNN code out) | The probe forces `AT_CUDNN_ENABLED()` on in the PyTorch tree's own hipified files, and every compiled file must define a canary, so a body preprocessed out fails |
| Corpus failures surface only in TheRock CI | TheRock builds hipDNN with the shim on and runs the v9 and legacy corpus on its test families; the superbuild CI does not run the corpus |
| PyTorch compile check needs a PyTorch ROCm build | The check is local; the PyTorch-surface unit test also compiles as C++20 in CI (`hipdnn_frontend_cudnn_v8_tu_cpp20`) |

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
| 15 | Unused C identifiers left out (§9 R4) |
| 16 | Docs and overhead measurement |
| 17 | Final validation |

## 7. Testing Plan

| Tier | Proves |
|---|---|
| Unit tests on the mock backend (`hipdnn_frontend_tests`) | Core rules without builders (`TestCudnnShimV8Core*`); builders, lowering, tags, filters, errata, plan search, thread safety; error boundary, raw-descriptor table, plan lifetime, `cudnnBackendExecute` passthrough (`TestCudnnShimV8BuilderContracts`); `cudnn_graph.h` standalone (`TestCudnnShimV8CApiStandalone`); public surface pins (`TestCudnnShimV8Tu`); a mixed v8/v9 file (`TestCudnnShimV8MixedTu`) |
| Separate binary with `NV_CUDNN_DISABLE_EXCEPTION` | Status-return path, including the error boundary (can't share a binary with the throwing build) |
| Binary linked against `hipdnn_frontend_dynamic` only | Builders, filters and `cudnnBackendExecute` link without the hipDNN backend library |
| Compile-only targets | JSON-off builds; `find_plan.h`, `get_plan.h` and `EngineConfigGenerator.h` alone, under each JSON opt-out; logging and SDPA off; the public surface pins as C++20 |
| Integration, real backend, no GPU | Every lowered operation passes hipDNN's descriptor validation |
| Integration, test plugins, GPU | Engine enumeration, plans, execute (including on another handle), knobs, multiple engines, plan search |
| GPU samples (`samples/cudnn_shim/*V8.cpp`; MIOpen, hipBLASLt, hip-kernel) | Numerical results against a CPU reference for conv fprop / dgrad / wgrad, conv + bias + ReLU, pointwise ReLU, matmul + bias + activation, batch norm forward training and backward, layer norm forward and backward, and resample forward and backward. Each test fails when fewer variants run than its `--require-ran` floor or when a variant named by `--require-variant` (the fp32 NCHW variants of every sample with a non-zero floor) does not run. Variants with no engine are skipped: matmul (hipBLASLt only; none on gfx115x under Windows), layer norm and resample (hip-kernel-provider only), `CUDNN_CONVOLUTION` mode and asymmetric padding (MIOpen rejects them). The reduction sample is built but no provider runs reduction, so it always reports Skipped. TheRock's hipdnn-samples job loads only the MIOpen provider, so CI skips matmul, layer norm and resample (their floors are 0) until a TheRock change that loads the hipBLASLt and hip-kernel providers reaches this repository's pin; until then manual gfx942 runs cover them. Overhead (§2) is reported under ctest label `perf`. The v8 samples reject `--engine-id` and `--engine-name` |
| v9 sample corpus | Existing v9 behavior unchanged |
| NVIDIA legacy samples | Source compatibility; each case is judged against its manifest tier: `RUN_CLEAN`, `XFAIL_RUN`, or `VACUOUS`, which must run without crashing and is reported as skipped ([Reference §5](./0022_CuDNN_V8_Shim_Reference.md#5-legacy-sample-catalogue)). The corpus is pinned by commit: configure fails with E22 when a git checkout is at any other commit; a supplied corpus that is not a readable git checkout only warns, and the E20 version check still applies |
| PyTorch compile check | `Conv_v8.cpp` and the quantized cuDNN files compile with the PyTorch tree's flags, `AT_CUDNN_ENABLED()` forced on and two header substitutions: `ATen/cudnn/Descriptors.h` truncated before its v7 descriptor wrappers (they need the v7 C API the shim lacks) and `ATen/cudnn/Handle.h` replaced by the hipDNN integration header |

## 8. Future Considerations

- Alpha / beta, once a provider accepts scalar operands in pointwise fusions.
- Numerical notes, if providers start declaring them.
- More operations (gen-stats, BN finalize, RNG, reshape, instance / group norm) as hipDNN gains them.
- A split umbrella if compile time becomes a problem.
- A read-only descriptor query for raw C callers such as XLA, only on a concrete request.
- The legacy v7 C API only on a concrete request; separate RFC.

## 9. Design decisions

| # | Decision |
|---|---|
| R1 | No backend descriptor C API except `cudnnBackendExecute`: the builders reach hipDNN through the core. The only raw C caller found is XLA's autotune persistence, which reads plans and engine configs through `cudnnBackendCreateDescriptor` / `GetAttribute` / `DestroyDescriptor` (`cuda_dnn.cc` L5081-5199 at the [Reference §4](./0022_CuDNN_V8_Shim_Reference.md#4-consumer-surfaces) commit); it does not compile. PyTorch's, Paddle's and XLA's builder code is source compatible |
| R2 | `cudnnBackendExecute` is a passthrough: hipDNN runs a plan on whichever handle it is given, including one other than its graph's, provided the handle is on the plan's device. Pinned by unit and integration tests |
| R3 | NVIDIA's legacy "Tensor creation comparison" calls the raw API. A harness-only `tests/cudnn_samples/legacy/overlay/cudnn.h` declares the calls it makes as `NOT_SUPPORTED` stubs, plus the five tensor attribute names and two attribute types it passes; the case is an expected failure. The stub is never installed under `include/` |
| R4 | Identifiers are left out relative to the extracted list (§3.7). An excluded function or enum is left out whole: among them the five descriptor functions, `cudnnBackendAttributeName_t`, `cudnnBackendAttributeType_t`, and `cudnnMoeGroupedMatmulMode_t` (named only by upstream's enum converters, which the shim omits for it). An enumerator none of the sources names, such as `CUDNN_NOT_PROPAGATE_NAN`, is not declared. None of them was declared on `develop`, so relative to `develop` the change only adds API, as RFC 0012 §4.8 requires; `hipdnn_frontend` takes a minor version bump to 0.5.0 |
| R5 | Builder error messages name the builder and the rule, not attribute enums, so they differ from cuDNN's; the legacy-corpus anchors and the tests' message checks pin the shim's messages |
| R6 | `cudnnBackendDescriptorType_t` stays whole: PyTorch's plan-cache key, Paddle's template argument and `OperationBuilder`'s constructor argument |
| R7 | `cudnnGetLastErrorString` is hipDNN's; see §3.6 |
| R8 | Per-call overhead: `VariantPackBuilder::build()` + `cudnnBackendExecute` p50 0.76–0.97 µs on an idle host, p90 about 1.4 µs. Its budget is 5 µs, not 1 µs, and a missed budget is reported rather than failed unless `--enforce-budget` is passed, because wall-clock noise on a busy host exceeds 1 µs; see §2 |
| R9 | Pointwise defaults are elided, a ReLU lower clip of 0 included. This relies on the providers accepting every equivalent ReLU encoding (rocm-libraries#14363: hipBLASLt reads an unset lower clip as 0 and rejects a non-zero slope, MIOpen reads a slope of 0 as none), which also covers the same graphs for v9 and native users |
