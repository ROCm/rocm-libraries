# RFC: cuDNN v8 shim for hipDNN

- **Status**: Accepted
- **Authors**: Mitch Ousdahl
- **Follows**: [RFC 0012: cuDNN shim for hipDNN](./0012_CuDNN_Shim.md) §9
- **Amends**: RFC 0012 §1 ("Not"), §4.2 (v9-only translation units), §4.3 (aliasing of
  `DescriptorType_t`, `NormMode_t`, `RngDistribution_t`), §4.7 (v9-only umbrella)
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

## 1. Summary

### Why

RFC 0012 covers the cuDNN frontend v9 graph API. Much cuDNN-consuming code still uses the older
frontend **v0.x builder API** (`TensorBuilder`, `OperationBuilder`, `ExecutionPlanBuilder`, …) and the
**cuDNN backend descriptor C API** beneath it:

- PyTorch runs all cuDNN convolution through `Conv_v8.cpp` (on by default) and its int8 quantized ops
  through `native/quantized/cudnn/`.
- XLA/TensorFlow keep convolution, fusions and normalization on v0.x and call `cudnnBackendGetAttribute`
  directly.
- PaddlePaddle's convolution path is v0.x.

None of these compile against the v9 shim. See
[Reference §4](./0021_CuDNN_V8_Shim_Reference.md#4-consumer-surfaces).

### What

Add to the existing shim:

1. An emulated **cuDNN backend descriptor C API** (`cudnnBackendCreateDescriptor`, `…SetAttribute`,
   `…GetAttribute`, `…Finalize`, `…Execute`, `…DestroyDescriptor`, `cudnnGetLastErrorString`) plus the
   C types and enums the v0.x surface uses. Shim-owned descriptors hold cuDNN-shaped state and translate
   to hipDNN backend descriptors.
2. The **v0.x builders**, ported from cudnn-frontend v1.24.0 onto (1) and exposed through the existing
   `cudnn_frontend.h`, as upstream does.

Graphs a hipDNN engine supports run. Everything else compiles and fails at runtime with
`CUDNN_STATUS_NOT_SUPPORTED`.

### Not in scope

- The legacy v7 C API (`cudnnConvolutionForward`, `cudnnSetTensorNdDescriptor`, …).
- ABI compatibility; consumers recompile.
- Python bindings.
- Provider gaps: a correctly translated graph that no engine supports (for example int8 convolution)
  fails as it would on unsupported NVIDIA hardware.

## 2. Requirements

- Header-only, like RFC 0012.
- The full v0.x surface and its backend C API compile name-for-name against cudnn-frontend v1.24.0.
- Convolution (fwd / dgrad / wgrad), pointwise, matmul, reduction, resample and batch / layer / RMS
  norm translate to hipDNN and run on any engine that claims them.
- Everything else returns `CUDNN_STATUS_NOT_SUPPORTED` with a message naming the operation, attribute or
  value. Builders throw `cudnnException` unless `NV_CUDNN_DISABLE_EXCEPTION` is defined. Nothing is
  silently dropped except the store-and-ignore items listed in §3.4.
- Per-call overhead of `VariantPackBuilder::build()` + `cudnnBackendExecute`: < 1 µs over the hipDNN
  equivalent on the same plan.
- MIT licensed, with NVIDIA attribution on every derived file. Nothing copied from NVIDIA's proprietary
  cuDNN headers.
- No behavior change for v9 users.

## 3. Design

### 3.1 Layout

Everything lives in `projects/hipdnn/frontend/include/hipdnn_compatibility/cudnn/`, behind the existing
`HIPDNN_ENABLE_CUDNN_COMPATIBILITY` option. Public filenames match upstream.

```
cudnn.h                       extended: includes cudnn_graph.h
cudnn_graph.h                 new: backend C entry points
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
detail/backend/               new, included strictly in this order:
    c_types.h  attribute_table.h  descriptor.h  type_mapping.h  tensor_descriptor.h
    parameter_descriptors.h  operation_descriptors.h  operation_graph_descriptor.h
    engine_descriptors.h  execution_descriptors.h  backend_api.h
```

- `c_types.h` holds all C types and enums, including `cudnnHandle_t` (moved from `cudnn.h`). No
  `detail/backend` header includes `cudnn.h` or `cudnn_graph.h`, so `cudnn_graph.h` compiles on its own.
- As upstream, `EngineConfigGenerator`, `find_plan` and `get_plan` are separate includes.
- A translation unit may now mix v0.x builders, the v9 graph API and raw `cudnnBackend*` calls.
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

So the shim never casts between cuDNN and hipDNN values; it translates by name.

The ported builders call the emulated C API exactly as upstream calls cuDNN's, which keeps them close
to upstream and puts all translation in one place. All hipDNN calls go through
`hipdnn_frontend::detail::hipdnnBackend()`, so direct and runtime-loaded hipDNN both work and the mock
backend works in tests.

### 3.3 Descriptor model

- `cudnnBackendDescriptor_t` is `void*` pointing at a shim handle (magic word plus a reference-counted
  descriptor). Destroying a handle drops only the caller's reference, so a graph keeps its tensors and
  operations alive.
- One table defines, per attribute, the owning descriptor type, accepted attribute types and count
  rule. Count queries follow cuDNN: a null array or a requested count of 0 returns the count.
- `GetAttribute` on a descriptor-valued attribute fills the caller's descriptor in place, as cuDNN does.
- Every entry point catches all exceptions, maps them to `cudnnStatus_t`, and records a per-thread
  message for `cudnnGetLastErrorString`.
- Finalized descriptors are immutable; lazily computed state is guarded by `std::call_once`. Copies of an
  `ExecutionPlan` can execute concurrently.

### 3.4 Lowering to hipDNN

Operation and parameter descriptors validate at `cudnnBackendFinalize`, so builders fail where
upstream's do. hipDNN descriptors are created only when the operation graph is finalized:

1. Tensors are merged by unique ID. Two tensors with the same ID but different shape, strides, type or
   flags are `BAD_PARAM`.
2. Operations are topologically sorted, keeping the user's order when it is already valid. Providers
   match fusions by node position, so order matters.
3. Each operation is lowered by filling `hipdnn_frontend`'s own node attributes and calling its
   validation and descriptor packers. Shape inference and plugin-version checks therefore match
   hipDNN's frontend exactly.

The graph must have a handle (`BAD_PARAM` otherwise; cuDNN ≥ 9.8 makes it optional). Graph finalize
runs no heuristics and needs no GPU.

**Tensors**

| cuDNN | hipDNN |
|---|---|
| alignment ≥ 16 | not set (16 is hipDNN's default; cuDNN's value is a lower bound) |
| alignment < 16 | forwarded; needs plugin API ≥ 1.3.0, which no provider reports yet |
| by-value | runtime pass-by-value tensor ([RFC 0016](./0016_RuntimePassByValueTensors.md)) |
| constant value | `HIPDNN_ATTR_TENSOR_VALUE_EXT`; bytes copied at finalize |
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
| Gen-stats, BN finalize, BN bwd weights, RNG, reshape, paged cache load | — | `NOT_SUPPORTED` |

**Other rules**

- **Pointwise defaults are not forwarded.** Upstream sets every parameter of the mode, including
  defaults; MIOpen rejects a fusion if a lower-clip slope is present at all. Non-default values are
  narrowed to float.
- **Alpha / beta.** Alpha ≠ 1 or beta ≠ 0 on any op → `NOT_SUPPORTED`. No hipDNN engine accepts the
  alternative (an extra scalar multiply).
- **Store and ignore:** NaN propagation (hipDNN has no equivalent).
- **`NOT_SUPPORTED` when set:** dynamic shapes, override shapes.

### 3.5 Engine selection

- **Modes.** All heuristic modes map to FALLBACK, with one WARN per mode per process (as RFC 0012 §6.3
  requires). SM-count and device-property targets → `NOT_SUPPORTED`.
- **Enumeration.** On the first engine query for a graph, the shim runs FALLBACK heuristics once,
  finalizes each default engine config and caches the results. Engines whose config fails are dropped.
  Requires a GPU.
- **Engine index.** cuDNN engines are small integers; the shim numbers hipDNN engines `0..N-1` by sorted
  engine ID. Ranking is conveyed by result order only. Indices survive ranking changes but not changes
  to the set of engines (new provider, different GPU).
- **Notes.** Behavior notes map by name. `NONDETERMINISTIC` is reported for every engine not known to be
  deterministic, so deterministic filtering works. No other numerical notes are reported.
- **Knobs.** hipDNN knobs that map onto a cuDNN knob type are shown; others are hidden, and choosing a
  hidden knob is `NOT_SUPPORTED`.

### 3.6 Plans and execution

- A plan's handle must match its graph's handle (`BAD_PARAM` otherwise). hipDNN takes the stream from
  the execute handle and the plan context from the graph handle, so a mismatch can't be served
  correctly.
- `cudnnBackendExecute` checks the handle and makes exactly one `hipdnnBackendExecute` call. The hipDNN
  variant pack is built when the shim's variant pack is finalized.
- JSON plans: `getJsonRepresentation` always returns a valid JSON object with engine index, ID (as a
  string), name, notes, knobs and, when hipDNN can serialize the plan, a base64 payload.
  `loadFromJson` accepts only this format and requires the payload. Neither direction needs nlohmann.
- Kernel cache and variant-pack intermediates → `NOT_SUPPORTED`.

### 3.7 Types, enum values and versions

- Every C identifier used by the v0.x headers, NVIDIA's legacy samples and the PyTorch consumers is
  declared, from a checked-in list extracted from those sources and pinned by a test.
- Names are declared for source compatibility only and carry no MIT claim (as in RFC 0012 §6.4).
- Enum values are the shim's own: numbered from 0 in alphabetical order of the enumerator name, so they
  can't reproduce NVIDIA's numbering, and `*_COUNT` = last + 1. Exception: `cudnnBackendTensorReordering_t`
  follows the shim's `TensorReordering_t` because upstream writes that enum's bytes into the attribute.
- Values are append-only within a `hipdnn_frontend` major version, pinned by a test. Knob types appear
  in plan tags and serialized autotune results, so renumbering would break persisted data.
- `DescriptorType_t`, `NormMode_t` and `RngDistribution_t` are shim-owned rather than aliased (amends RFC
  0012 §4.3): hipDNN has no counterpart. They become aliases if hipDNN adds one.
- `CUDNN_VERSION` stays 92200 and the frontend pin 1.24.0. Ported headers `static_assert` the pin so a
  bump forces a re-diff.
- `cudnnBackendInitialize` is not provided (deprecated; nothing calls it).

### 3.8 Porting rules

- Port upstream headers unchanged in behavior. Edits:
  - namespace and include paths;
  - CUDA timing calls → HIP;
  - dynamic loading, CUDA-graph and v7 helpers removed;
  - friend declarations of v9 node classes removed;
  - edits needed to build under hipDNN's `-Werror` warning set, with no warning suppression.
- Fixed upstream bugs: `Knob::getMaxValue` and `RngDesc_v8::getUniformDistMin` return the wrong field;
  `VariantPackBuilder_v8::build` passes the data-pointer count as the `UNIQUE_IDS` count, which over-reads
  the caller's UIDs when fewer UIDs than pointers are given. It now passes the UID count, and the shim
  reports the mismatch as `BAD_PARAM`. `OperationGraphBuilder_v8::addOperation` on a builder with no
  operations yet stores the descriptor at `ops[-1]` (the count starts at -1); it now starts from 0.
  `OperationBuilder_v8::setcDesc` copies `CUDNN_DIM_MAX + 1` padding, dilation and stride entries into
  arrays of `CUDNN_DIM_MAX`; the arrays now hold `CUDNN_DIM_MAX + 1`. The misspelled alias
  `EngineHeuristicsBuilde` is kept.
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
| C++ builders straight over hipDNN, no public C API | Breaks raw C callers (XLA) and moves the builders away from upstream |
| C API in a compiled library | Breaks the header-only requirement |
| Separate `cudnn_frontend_v8.h` | Forces include changes in hipified code; kept as the fallback if compile time suffers |
| Compile-only stubs | Unblocks compilation but runs nothing |

## 4. Key Decisions

| Decision | Benefit | Cost |
|---|---|---|
| Emulate the C API, port builders on top | Builders stay close to upstream; raw C callers work | More code than C++-only builders |
| Validate early, lower at graph finalize | Errors where upstream raises them; tensors merged across ops | Shim manages descriptor lifetimes |
| Shim-owned, alphabetical, append-only enum values | No proprietary numbering; stable persisted data | No ABI compatibility |
| Skip pointwise defaults and alignment ≥ 16 | Target graphs find engines | Must track cuDNN defaults |
| Alpha / beta → `NOT_SUPPORTED` | Clear error | Scaled graphs don't run |
| Engine index by engine ID | Stable across re-ranking | Differs from the v9 shim; renumbers when engines change |
| Lower through `hipdnn_frontend` nodes | Same behavior as hipDNN's frontend | Depends on `hipdnn_frontend` internals |
| Extend the existing umbrella | Unmodified includes and mixed v8/v9 files work | ~10k more lines per umbrella include |

## 5. Risks

| Risk | Mitigation |
|---|---|
| Compile time | The header-only C API emulation is instantiated in every TU that includes `cudnn.h` (after Phase 3: about +1.5 s per TU). Accepted to keep the shim header-only; measured each phase on single TUs and the v9 sample corpus. The split umbrella remains the fallback for the v0.x builder headers only |
| Licensing of declared names | Names only, no MIT claim; shim-assigned values; legal review before Phase 1 merges |
| FALLBACK-only heuristics and missing notes change engine choice | One WARN per mode; documented in `KNOWN_DIVERGENCES.md` |
| Unsupported by providers: int8, four-op conv fusions, alignment < 16, ragged tensors | Out of scope; expected failures in the corpus harness |
| Descriptor lifetime bugs | Reference counting, scoped hipDNN descriptors, ASan run |
| Upstream drift | Pin `static_assert`; re-diff in RFC 0012 §6.1's quarterly upstream review |
| `hipdnn_frontend` internals change | Same tree and version; GPU-free lowering tests catch breakage |
| Maintenance of ~10k ported lines | Same owner as the RFC 0012 shim |
| PyTorch compile check passes on empty files (ROCm wheels compile cuDNN code out) | Shadow headers force the code on; a canary fails an empty file |
| Corpus and PyTorch harnesses not run by TheRock yet | Run in the superbuild CI Linux job |

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

## 7. Testing Plan

| Tier | Proves |
|---|---|
| Unit tests on the mock backend (`hipdnn_frontend_tests`) | C API semantics, lowering, builders, tags, filters, errata, thread safety |
| Separate binary with `NV_CUDNN_DISABLE_EXCEPTION` | Status-return path (can't share a binary with the throwing build) |
| Compile-only targets | JSON-off builds, public surface pins, mixed v8/v9 file, `cudnn_graph.h` standalone |
| Integration, real backend, no GPU | Every lowered operation passes hipDNN's descriptor validation |
| Integration, test plugins, GPU | Engine enumeration, plans, execute, knobs, multiple engines (adds one test plugin) |
| GPU samples (MIOpen, hipBLASLt) | Numerical results, raw C API, overhead |
| v9 sample corpus | Existing v9 behavior unchanged |
| NVIDIA legacy samples | Source compatibility with per-case expectations ([Reference §5](./0021_CuDNN_V8_Shim_Reference.md#5-legacy-sample-catalogue)) |
| PyTorch compile check | `Conv_v8.cpp` and the quantized cuDNN files compile |

## 8. Future Considerations

- Alpha / beta, once a provider accepts scalar operands in pointwise fusions.
- Numerical notes, if providers start declaring them.
- More operations (gen-stats, BN finalize, RNG, reshape, instance / group norm) as hipDNN gains them.
- A split umbrella if compile time becomes a problem.
- The legacy v7 C API only on a concrete request; separate RFC.
