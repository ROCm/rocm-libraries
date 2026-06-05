# Ragged Tensors — Per-stage analysis for the `Tensors.hpp` design

Companion to `RaggedTensorsTentativePlan.md` (RFC 0011). This document walks every stage in the codebase where an `ITensor` / `TensorBase`-derived object is created, owned, viewed, or consumed, and works out concretely how **Option A** (`RaggedTensor<T, IndexT>` as a self-owning `TensorBase<T>` subclass) and **Option C** (`RaggedView<T, IndexT>` over `IrregularTensor<T>`, constructed in the plan layer) drop into each one.

The point of doing this stage-by-stage rather than at the type-system level is that the codebase has **two distinct flow paths** in which ragged-aware tensors will need to participate:

1. **The user-facing path** (samples like `BnInference.cpp`, `SdpaFprop.cpp`): the user owns concrete `utilities::Tensor<T>` objects and threads `void*` device pointers into the graph via `variantPack`. No `TensorAttributes → ITensor` helper is exposed at this layer.
2. **The integration test path** (`IntegrationGraphVerificationHarness` → `GraphTensorBundle` → `CpuReferenceGraphExecutor` → per-op plan → `CpuFpReferenceSdpa`): the harness *does* have a `createTensorFromAttribute(...)` helper that walks a `TensorAttributes` and produces an `ITensor`, but everything past the graph executor's `execute(...)` boundary collapses back to `unordered_map<int64_t, void*>`. The plan rebuilds typed `ShallowTensor<T>` wrappers around those `void*`s on the fly and passes them as `TensorBase<T>&` into `CpuFpReferenceSdpa::forward`.

Both paths cross the same narrow waist: `unordered_map<int64_t, void*>` (the `variantPack`). Any ragged design has to either (a) accept that the ragged metadata lives elsewhere and is recombined with the raw pointer downstream of the waist, or (b) widen the waist to carry ragged metadata, which is a much larger change than this RFC contemplates. Both Option A and Option C in their current shape implicitly take path (a) — the wiring is recovered at the plan layer, not pushed through `variantPack`. The differences live entirely in **what the plan layer reconstructs** and **how the user-facing path connects a runtime tensor to a graph tensor**.

A reading of the user's two concerns up front:

- **Concern 1 (the sample / user flow).** The user wants a clean path from a `TensorAttributes` (e.g. the `shared_ptr` that `graph->batchnorm_inference(...)` returns for `y`) to a runtime `utilities::Tensor<T>`. *In the ragged case*, that path is necessarily multi-input: there is no single `TensorAttributes` that names both the primary and its aux tensors at the user level today (the primary's `TensorAttributes` carries a *reference* to the aux's `TensorAttributes`, but the user must still allocate a runtime buffer for each one). Today's user-facing `utilities::Tensor<T> t(attr->get_dim(), layout)` pattern *does not* have a corresponding ragged convenience constructor. This is true regardless of A vs C — the difference between them is what type the user ends up holding, and how convenient that type is to feed back into the runtime.
- **Concern 2 (the integration test flow).** Every stage between "graph executor receives `variantPack`" and "`CpuFpReferenceSdpa::forward` reads scalar values" needs to be examined to decide where the ragged view lives. Both options can be made to fit, but they fit at different points and pay different costs at each one.

What follows is the stage-by-stage walk for both flows, with code-anchored discussion of A vs C at each one. A summary recommendation is at the end.

---

## Cast of code (the relevant types and helpers, briefly)

For grounding, here is the cast as it actually exists today:

- **`hipdnn_frontend::graph::TensorAttributes`** (frontend): describes a tensor in the graph (`dims`, `strides`, `data_type`, `uid`, `is_virtual`, `name`, …). The RFC's `set_ragged_offset(...)` and `set_seq_len(...)` are additions on this class. There is **no** `ITensor` reference on a `TensorAttributes`.
- **`hipdnn_data_sdk::utilities::ITensor`** (data_sdk): the runtime-polymorphic, type-erased interface (`dims()`, `strides()`, `rawHostData()`, `rawDeviceData()`, `elementCount()`, `elementSpace()`, …).
- **`TensorBase<T>`**: typed virtual base (`getHostValue(indices)`, `setHostValue(value, indices)`, `fillWithValue(v)`, `fillWithSentinelValue()`, `memory()`, `markHostModified()` / `markDeviceModified()`, …).
- **`Tensor<T, HostAlloc, DeviceAlloc>`**: owning, host+device, `MigratableMemory<T>`-backed concrete tensor. Invariant: `_packed = (elementCount == elementSpace)`.
- **`PinnedTensor<T>`**: alias for `Tensor<T, PinnedHostAllocator<T>, …>`.
- **`ShallowTensor<T>`**: a non-owning `TensorBase<T>` view over a `void*` plus `dims` and `strides`. This is what the plan layer constructs on every call to `execute(...)` to give `CpuFpReferenceSdpa` something typed to walk.
- **`createTensorFromAttribute(const hipdnn_frontend::graph::TensorAttributes&)`** in `hipdnn_test_sdk/utilities/SdkFrontendTypeConversions.hpp`: dispatches on `attr.get_data_type()` and returns `std::unique_ptr<ITensor>` (concretely `std::make_unique<Tensor<T>>(attr.get_dim(), attr.get_stride())`). **Test-SDK only.** Not exposed to user-facing samples.
- **`createTensorFromAttribute(const flatbuffers_sdk::data_objects::TensorAttributes&)`** in `hipdnn_test_sdk/utilities/detail/FlatbufferTensorAttributesUtils.hpp`: the same idea but consuming the flatbuffer version, used by `CpuReferenceGraphExecutor` and `GraphTensorBundle`.
- **`createShallowTensor<T>(TensorAttributesT, void*)`** in the same header: builds the per-call `ShallowTensor<T>` from `(dims, strides, ptr)`.
- **`GraphTensorBundle`**: `unordered_map<int64_t, unique_ptr<ITensor>>` keyed by UID. `toHostVariantPack()` / `toDeviceVariantPack()` flatten this to `unordered_map<int64_t, void*>`.
- **`CpuReferenceGraphExecutor::execute(graphBuffer, size, variantPack)`**: deserializes the flatbuffer graph, builds a plan per node, allocates intermediate virtual tensors as additional `ITensor`s and patches their `rawHostData()` into the variant pack, then runs each plan.

The key shape-of-the-pipe observation:

```
TensorAttributes (graph)  ─┐                              ┌── ShallowTensor<T>  ─→ CpuFpReferenceSdpa
                           │── createTensorFromAttribute  │       (built per call)        (TensorBase<T>&)
                           ▼                              │
                       ITensor                            │
                  (Tensor<T>, owned by                    │
                   GraphTensorBundle or                   │
                   virtualTensors vector)                 │
                           │                              │
                           ▼                              │
                  rawHostData() / rawDeviceData()         │
                           │                              │
                           ▼                              │
                 unordered_map<int64_t, void*>  ──────────┘
                            (variantPack — the narrow waist)
```

Every stage discussed below is one of those boxes.

---

## Flow 1 — User-facing samples (`BnInference.cpp`, `SdpaFprop.cpp`)

### Stage 1.1 — Graph construction (`createTensor` → `Tensor_attributes`)

```cpp
auto x      = createTensor({n, c, h, w}, inputType, layout);                 // sample helper
auto scale  = createTensor({1, c, 1, 1}, intermediateType, layout);
// …
auto y      = graph->batchnorm_inference(x, mean, invVariance, scale, bias, bnAttributes);
y->set_output(true);
```

`createTensor` here is `samples/utils/Helpers.hpp`'s thin wrapper around `graph::Graph::tensor(...)`. The returned values are `std::shared_ptr<TensorAttributes>`. Nothing on the runtime tensor side exists yet.

**Option A.** No change at this stage. The RFC additions (`set_ragged_offset(...)`, `set_seq_len(...)`) modify `TensorAttributes` so the user can declare the wiring; that is a separate concern from A vs C.

**Option C.** Same — no change at this stage. The wiring is declared on `TensorAttributes` exactly as Option A would do it.

This stage is invariant to the choice of A vs C.

### Stage 1.2 — Runtime tensor allocation (`utilities::Tensor<T> xTensor(x->get_dim(), layout)`)

This is the stage the user is asking about. Today's sample:

```cpp
utilities::Tensor<InputType>        xTensor(x->get_dim(), layout);
utilities::Tensor<IntermediateType> scaleTensor(scale->get_dim());
// …
xTensor.fillWithRandomValues(static_cast<InputType>(0.0f), static_cast<InputType>(1.0f));
```

There is **no** `Tensor::Tensor(const TensorAttributes&)` constructor. The user threads `attr->get_dim()` and (sometimes) layout in manually. The test-SDK has `createTensorFromAttribute(const TensorAttributes&) -> std::unique_ptr<ITensor>`, but it is not in the user-facing data_sdk and it returns a polymorphic pointer rather than a concrete typed value the user wants.

For a ragged tensor, the user needs **three** things, not one:

1. A primary buffer whose size is the *physical* packed length (sum of per-batch sequence lengths, with alignment padding), **not** `prod(dims)`.
2. A `ragged_offset` aux buffer (size `B+1`).
3. Optionally a `seq_lens` aux buffer (size `B`).

`attr->get_dim()` alone is not enough information for (1) — the padded dims and the physical element count are independent in the ragged case. The graph's `TensorAttributes` carries `dims = [B, S_max, H, D]` for the primary, but the runtime allocation needs `physicalElementCount` as a separate input (computed from the aux tensors and the alignment policy, or pre-determined by the user).

#### Option A at this stage

The user writes (illustratively):

```cpp
// Allocate aux buffers first.
utilities::Tensor<int32_t> qRaggedOffset(raggedOffAttr->get_dim());       // [B+1]
utilities::Tensor<int32_t> qSeqLens     (seqLenAttr  ->get_dim());        // [B]
qRaggedOffset.fillFromHost(/* user-supplied offsets */);
qSeqLens     .fillFromHost(/* user-supplied seq lens */);

// Allocate the primary as a RaggedTensor.
auto qShared = std::make_shared<RaggedTensor<InputType>>(
    /*paddedDims=*/q->get_dim(),
    /*strides=*/q->get_stride(),
    /*physicalElementCount=*/computedPhysicalCount,
    /*raggedOffset=*/std::shared_ptr<TensorBase<int32_t>>(/* aliasing ctor over qRaggedOffset */),
    /*seqLens=*/    std::shared_ptr<TensorBase<int32_t>>(/* aliasing ctor over qSeqLens     */));

// Wire into variantPack — exactly the existing pattern.
variantPack[q          ->get_uid()] = qShared       ->rawDeviceData();
variantPack[raggedOffAttr->get_uid()] = qRaggedOffset .rawDeviceData();
variantPack[seqLenAttr  ->get_uid()] = qSeqLens      .rawDeviceData();
```

Properties:

- The user holds a single owning value (`qShared`) that conceptually is the ragged tensor and has the right `dims`/`elementSpace` semantics out of the box.
- The aux tensors are still separate `Tensor<int32_t>` objects the user has to allocate and feed into the `RaggedTensor` ctor. There is no escape from this — the aux data has to exist on host and device and has to be addressable by a UID in `variantPack`.
- A `Tensor::Tensor(const TensorAttributes&)` convenience constructor *could* be added to make `xTensor(x)` work for non-ragged tensors. That is independent of A vs C, but in A the same convenience does **not** work for `RaggedTensor` because the ctor needs the aux tensors and `physicalElementCount`, none of which are on a single `TensorAttributes`. **A `RaggedTensor::RaggedTensor(const TensorAttributes&)` ctor is not constructible** without a side channel that resolves the aux UIDs to existing runtime tensors. That side channel does not exist at the user layer today and would have to be invented.
- The "is the runtime aux the same one I declared in the graph?" question is left to the user's eyeballs at this layer (the graph holds the *attributes* of `qRaggedOffset`; the user has to remember to construct a runtime `qRaggedOffset` whose UID matches in `variantPack`). This is the same situation as for any other tensor today and is not made worse by A.

The user's first concern — *"it would be useful for a user to be able to easily go from the `TensorAttributes` used in and returned by the graph→batchnorm_inference call to the required tensor layout for ragged tensors"* — has a partial answer in A: the user can hold a single `RaggedTensor` and that single object has the right `dims`, the right `elementSpace`, and (with the iterator hook) the right iteration semantics. But the user still has to construct the aux tensors by hand and pass them in. There is no `TensorAttributes → RaggedTensor` one-call helper, and there structurally cannot be one without resolving aux UIDs to runtime tensors *somewhere*.

#### Option C at this stage

The user writes (illustratively):

```cpp
// Aux buffers — exactly the same as in A.
utilities::Tensor<int32_t> qRaggedOffset(raggedOffAttr->get_dim());
utilities::Tensor<int32_t> qSeqLens     (seqLenAttr  ->get_dim());
qRaggedOffset.fillFromHost(/* … */);
qSeqLens     .fillFromHost(/* … */);

// Primary buffer — an IrregularTensor that owns the packed storage.
auto qStorage = std::make_shared<IrregularTensor<InputType>>(
    /*paddedDims=*/q->get_dim(),
    /*strides=*/q->get_stride(),
    /*physicalElementCount=*/computedPhysicalCount);

// If the user wants to iterate it themselves (CPU-side init, debug print, etc.):
auto qView = std::make_shared<RaggedView<InputType>>(
    qStorage,
    std::shared_ptr<TensorBase<int32_t>>(/* aliasing ctor over qRaggedOffset */),
    std::shared_ptr<TensorBase<int32_t>>(/* aliasing ctor over qSeqLens     */));

// Wire into variantPack — note we point at qStorage's device data, not qView.
variantPack[q             ->get_uid()] = qStorage    ->rawDeviceData();
variantPack[raggedOffAttr ->get_uid()] = qRaggedOffset.rawDeviceData();
variantPack[seqLenAttr    ->get_uid()] = qSeqLens     .rawDeviceData();
```

Properties:

- The user holds an `IrregularTensor` (the storage) and *optionally* a `RaggedView` over it (if they want iteration). The view is non-owning and re-buildable from the storage + aux tensors at any later point.
- The user has to know that `IrregularTensor` is the "I just want the buffer" type and `RaggedView` is the "I want to walk it" type. This is a meaningful conceptual addition at the user layer.
- The same observation as for A applies: a `IrregularTensor::IrregularTensor(const TensorAttributes&)` convenience constructor *could* exist, but it would need `physicalElementCount` from somewhere outside `TensorAttributes`. (More on this in *Possible mitigation* below.)
- The "single owning object" pattern from A is split into two values (`qStorage` + optionally `qView`) for the user to track. The wiring-into-variantPack step uses `qStorage`, so the user must remember which of the two is the one to feed into `variantPack`. This is a small but real ergonomic cost at the user surface.

#### Side-by-side at Stage 1.2 (user allocation)

| Aspect | Option A | Option C |
|---|---|---|
| Number of value-typed primary objects the user holds | 1 (`RaggedTensor`) | 1 (`IrregularTensor`) — `RaggedView` only as needed |
| Aux tensors the user must allocate by hand | 1–2 (`ragged_offset`, optionally `seq_lens`) | Same |
| Can a `TensorAttributes → primary` one-call helper exist? | No (ctor needs aux runtime tensors and `physicalElementCount`) | No (ctor needs `physicalElementCount`); a `(TensorAttributes, physicalCount) → IrregularTensor` helper is straightforward |
| The single object the user passes to `variantPack[uid] = …` is | `ragged.rawDeviceData()` | `irregular.rawDeviceData()` |
| The user's iteration affordance | `ragged.begin()/end()` walks valid elements directly | Have to build a `RaggedView` first; then `view.begin()/end()` walks valid elements |
| Conceptual surface area added at the user layer | 1 new class | 2 new classes |

There is **no design** that lets a sample do `utilities::Tensor<T> qTensor(q)` and silently make the ragged case work. The information required is not on a single `TensorAttributes`. The most we can offer the user is:

- A factory that takes (primary `TensorAttributes`, aux runtime tensors, `physicalElementCount`) and returns the appropriate runtime object (`RaggedTensor` for A, or `(IrregularTensor, RaggedView)` pair for C). Both options can supply this. Neither dodges the multi-input nature of the request.

> **Possible mitigation (applies to both A and C, but is more natural in C).** If we add a *runtime-side* `TensorBundle`-like helper that mirrors the `GraphTensorBundle` in the test harness — `UID → owning_runtime_tensor` — and a factory `bundle.add(attr, …)` that for non-ragged attrs builds a `Tensor<T>` and for ragged attrs walks the graph's declared `ragged_offset` / `seq_len` UIDs in the bundle to assemble the aux refs automatically, then **the user-facing call becomes one-call per `TensorAttributes` regardless of ragged-ness**. This factory is structurally identical in A and C; it just hands back a different concrete type. C makes it marginally more natural because the factory can return the bare `IrregularTensor` (the thing whose `rawDeviceData()` belongs in `variantPack`) and defer view construction until iteration is requested. A returns a `RaggedTensor` that carries the aux refs around forever even when the user only needed the buffer. *This helper is out of scope for the immediate RFC but is the right answer to "easy `TensorAttributes` → runtime mapping" for the user.*

### Stage 1.3 — `variantPack[uid] = tensor.rawDeviceData()`

This is the narrow waist. It is `unordered_map<int64_t, void*>`. The same in A and C: the user supplies one `void*` per UID, and the ragged metadata is carried separately (via the additional aux UIDs in the same `variantPack`).

This stage is invariant to A vs C.

### Stage 1.4 — `graph->execute(handle, variantPack, workspace)`

The frontend / backend execution flow consumes `variantPack` and dispatches to the kernel provider. From the *backend's* perspective the ragged metadata is recovered from the graph's serialized form (which carries `ragged_offset_tensor_uid` and `seq_lens_tensor_uid`), and the kernel reads the raw pointer for the primary plus the raw pointers for the aux tensors out of `variantPack`. This is the same in A and C — the runtime tensor's wrapper type is invisible past `.rawDeviceData()`.

This stage is invariant to A vs C.

### Stage 1.5 — User-side CPU validation (optional, sample-specific)

When samples invoke `CpuFpReferenceBatchnorm::fwdInference(...)` directly, they pass the same `utilities::Tensor<T>` they allocated in Stage 1.2 as a `TensorBase<T>&`. For ragged inputs the reference needs to walk only valid elements, which means **the reference function receives a `TensorBase<T>&` and the iteration strategy needs to be ragged-aware automatically** (otherwise the reference iterates over padding).

- **Option A.** The user passes the `RaggedTensor` directly to the reference. The reference does `q.begin() / q.end()` (or per-batch loops bounded by `q.validSeqLen(b)`) and gets ragged iteration through the polymorphic index hook. **Single object, single call site, no view construction.**
- **Option C.** The user has to build a `RaggedView` first and pass it. The view's `begin()/end()/validSeqLen(b)` produce the same behaviour. **Two-step: storage + view, then call.**

A has a small but real ergonomic edge at this stage in the user flow: the user holds one object that "is" the ragged tensor and feeds it straight into both `variantPack` (via `rawDeviceData()`) and the reference (as a `TensorBase<T>&`). C requires the user to remember which value goes where: the `IrregularTensor` for the variant pack, the `RaggedView` for the reference. This is the inverse of the cost C pays in the test harness (covered below).

---

## Flow 2 — Integration test harness (`IntegrationGraphVerificationHarness` → `…` → `CpuFpReferenceSdpa`)

The harness flow has many more stages than the user flow because it owns both the GPU and CPU executions, and the executor cleanly splits "owning storage" from "per-call typed view." We walk every stage where an `ITensor` or `TensorBase` derived object exists.

### Stage 2.1 — `generateBundles(graph, cpuBundle, gpuBundle, outputTensorIds)`

Code:

```cpp
graph.visit([&](const hipdnn_frontend::graph::INode& node) {
    for(const auto& tensorAttr : node.getNodeOutputTensorAttributes()) {
        if(tryAddTensorToBundles(tensorAttr, cpuBundle, gpuBundle))
            outputTensorIds.push_back(tensorAttr->get_uid());
    }
    for(const auto& tensorAttr : node.getNodeInputTensorAttributes()) {
        tryAddTensorToBundles(tensorAttr, cpuBundle, gpuBundle);
    }
});

// inside tryAddTensorToBundles:
cpuBundle.tensors.insert({tensorId, createTensorFromAttribute(*tensorAttr)});
gpuBundle.tensors.insert({tensorId, createTensorFromAttribute(*tensorAttr)});
```

`createTensorFromAttribute(const hipdnn_frontend::graph::TensorAttributes&)` returns `std::unique_ptr<ITensor>` — concretely a `Tensor<T>(attr.get_dim(), attr.get_stride())`. For each `TensorAttributes` in the graph (excluding virtuals), the harness allocates a fresh owning concrete tensor and stores it in both `cpuBundle.tensors` and `gpuBundle.tensors`.

This is where ragged-aware allocation must enter. `Tensor<T>(dims, strides)` is wrong for ragged primaries because:

- It allocates `prod(dims)` elements, but the ragged buffer needs `physicalElementCount` elements (which may be smaller or larger after alignment padding).
- It sets `_packed = (elementCount == elementSpace)`, but that invariant is meaningless for ragged buffers.

#### Option A at Stage 2.1

The `createTensorFromAttribute` switch needs a new branch:

```cpp
if (attr.has_ragged_offset()) {
    // dispatch on attr.get_data_type() →
    //   make_unique<RaggedTensor<T, int32_t>>(
    //     attr.get_dim(),
    //     attr.get_stride(),
    //     physicalElementCountFor(attr, …),    // *** see below ***
    //     /*raggedOffset=*/ nullptr,           // *** see below ***
    //     /*seqLens=*/     nullptr);
}
```

Two structural problems show up immediately:

1. **Where does `physicalElementCount` come from?** It is not on `TensorAttributes`. The RFC's flatbuffer addition is `ragged_offset_tensor_uid` (a pointer to *another* tensor) — it doesn't carry the packed length. The packed length is implicit in the values stored in the `ragged_offset` tensor at runtime, but at *allocation* time those values haven't been generated yet. So either (a) `TensorAttributes` carries an additional field `physical_element_count` (an RFC scope extension), or (b) the harness has to allocate the `ragged_offset` tensor *first*, populate it (with `initializeBundle`), and then walk it to compute the packed length before allocating the primary. Option (b) requires re-ordering Stage 2.1 around Stage 2.2 (init), which `tryAddTensorToBundles` is not currently structured to do. Option (a) is a cross-cutting addition.
2. **`RaggedTensor`'s ctor requires non-null `raggedOffset`.** At Stage 2.1 we are walking attributes one at a time without a deterministic visit order between primaries and their aux tensors. The aux's `unique_ptr<ITensor>` may not exist in the bundle yet when we get to the primary. We must therefore either (a) do two passes (allocate auxes first, primaries second, reading the graph's wiring on the second pass), or (b) construct `RaggedTensor` with `raggedOffset = nullptr` and patch it in later (which violates A's "constructor-only / immutable" rule from the plan, and also undermines A's ctor-time structural validation).

The natural resolution is **two-pass allocation**: pass 1 allocates everything that is *not* a ragged primary; pass 2 walks the graph and for each ragged primary, looks up the already-allocated aux tensors in the bundle and constructs the `RaggedTensor` with the aux `shared_ptr`s. This requires `GraphTensorBundle` to use `shared_ptr` instead of `unique_ptr` (or expose a `share()` that produces aliasing `shared_ptr`s into the bundle's storage). It also tightly couples `tryAddTensorToBundles` to the existence of the RFC's `TensorAttributes::get_ragged_offset()` API.

#### Option C at Stage 2.1

The `createTensorFromAttribute` switch also gets a new branch, but the new branch is **only** for the primary storage:

```cpp
if (attr.has_ragged_offset()) {
    return make_unique<IrregularTensor<T>>(
        attr.get_dim(),
        attr.get_stride(),
        physicalElementCountFor(attr, …));    // *** see below ***
}
```

The aux tensors are allocated as ordinary `Tensor<int32_t>` by the *same* function on their own `TensorAttributes` entry — no special case, no two-pass requirement, no aux-resolution at allocation time. **`RaggedView` is not allocated here**; it is constructed at Stage 2.5 (plan execute) from data already in `variantPack`.

The same `physicalElementCount` sourcing problem from A applies — Option C does not solve it. C does, however, decouple the *allocation* phase from the *wiring* phase: the bundle ends up holding owning `IrregularTensor` and owning `Tensor<int32_t>` aux tensors, with no constructor-time relationship between them. Whether the wiring matches what the graph says is checked later, when the view is built. There is no ordering constraint between primary and aux allocation.

#### Side-by-side at Stage 2.1

| Aspect | Option A | Option C |
|---|---|---|
| Branch added in `createTensorFromAttribute` | One branch returning `RaggedTensor<T>` | One branch returning `IrregularTensor<T>` |
| Source of `physicalElementCount` | Must come from outside `TensorAttributes` (RFC extension or harness pre-pass) | Same — neither option solves this; both have the same problem |
| Ordering constraint between primary and aux allocation | Must allocate aux before primary (so `RaggedTensor` ctor can take the aux ref) | None — independent allocations |
| Storage of aux refs on the primary | Required at ctor time | Not stored anywhere yet |
| Owning-pointer type in the bundle | Must shift `unique_ptr` → `shared_ptr` so `RaggedTensor` can take the aux ref | Can stay `unique_ptr` |
| Risk of "primary allocated before its aux" | Real (requires two-pass) | None |

C is meaningfully simpler at this stage. A's "constructor-only auxes" rule, which the plan presents as a strength, becomes an ordering constraint on the harness's bundle-building loop. C avoids the issue by deferring view construction to a stage where everything is already allocated and reachable.

### Stage 2.2 — `initializeBundle(graph, bundle, seed)`

```cpp
for(auto& tensorPair : bundle.tensors) {
    bundle.randomizeTensor(tensorPair.first, DEFAULT_MIN, DEFAULT_MAX, seed);
}
```

The bundle's `randomizeTensor` calls `it->second->fillTensorWithRandomValues(min, max, seed)` via `ITensor`. For ragged primaries, "randomize" should mean "randomize the valid elements and either zero or leave-uninitialized the padding." Whether the iterator walks valid elements only or the entire physical buffer is decided by the *polymorphic index hook* on the concrete tensor type.

- **Option A.** The primary in the bundle is `RaggedTensor<T>`. Its overridden `makeIndex(...)` returns a `RaggedCompositeIndex` that walks only valid elements (using `raggedOffset()` and `seqLens()`). Randomization Just Works without harness changes.
- **Option C.** The primary in the bundle is `IrregularTensor<T>`. Its `begin()/end()` *throws* (per the plan's recommendation #3). The bundle's `randomizeTensor` will fail at this stage unless either (i) we make `IrregularTensor` randomizable by some other means (a direct linear walk over the physical buffer, which is wrong for ragged because it touches padding), or (ii) the harness has to wrap the `IrregularTensor` in a `RaggedView` for the duration of initialization and ask the view to iterate. (ii) requires that the harness has access to the aux tensors at this stage, which it does — they're in the same bundle.

C's randomization path is therefore:

```cpp
if (isRaggedPrimary(bundle, uid, graph)) {
    auto view = makeRaggedViewFromBundle<T>(bundle, uid, graph);
    view->fillTensorWithRandomValues(min, max, seed);   // ragged-aware iteration
} else {
    bundle.tensors[uid]->fillTensorWithRandomValues(min, max, seed);
}
```

This is a few lines of code in the harness's `initializeBundle`, but it is real new code and it requires `makeRaggedViewFromBundle` to exist (it is essentially the same factory the plan layer uses in Stage 2.5).

#### Side-by-side at Stage 2.2

| Aspect | Option A | Option C |
|---|---|---|
| `bundle.randomizeTensor(uid)` does the right thing for a ragged primary | Yes, via `RaggedTensor`'s polymorphic index hook | Not directly — `IrregularTensor` throws on iteration |
| Harness code to make initialization ragged-aware | None | Construct a temporary `RaggedView` per ragged primary at init time |
| Aux tensors need to be initialized before primary | No (their iteration is dense and independent) | Yes, because the temporary `RaggedView` reads `ragged_offset` to know which elements to walk. This is a real ordering constraint on `initializeBundle`. |

A is meaningfully simpler at this stage. C requires the harness to know which UIDs are ragged primaries (cheap — the graph's `TensorAttributes` says) and to construct a throwaway view per ragged primary at init. C also introduces an init-time ordering constraint (`ragged_offset` populated before primary randomized) that A does not have — but A had the *allocation*-time ordering constraint that C did not. The constraints are at different stages but they exist in both options. C's init-time constraint is more sensitive because the aux values are user-supplied data (the harness has to know how to generate them), whereas A's allocation-time constraint is purely structural.

### Stage 2.3 — `executeGpuGraph(handle, graph, gpuBundle)`

```cpp
auto variantPack = bundle.toDeviceVariantPack();    // unordered_map<int64_t, void*>
result = graph.execute(handle, variantPack, workspace.get());
```

`toDeviceVariantPack()` walks the bundle and produces `{uid → tensor->rawDeviceData()}`. For both A and C, the primary's `rawDeviceData()` returns the device pointer of its underlying storage (`RaggedTensor`'s own `MigratableMemory` in A; `IrregularTensor`'s `MigratableMemory` in C). The aux tensors are ordinary `Tensor<int32_t>` in both options and produce their own device pointers in the variant pack.

The backend dispatch is identical in A and C — both produce the same `variantPack` content. This stage is invariant to A vs C.

### Stage 2.4 — `executeCpuGraph(graph, cpuBundle)`

```cpp
auto [serializedGraph, serErr] = graph.to_binary();
CpuReferenceGraphExecutor().execute(
    serializedGraph.data(), serializedGraph.size(), bundle.toHostVariantPack());
```

`toHostVariantPack()` is the same flatten as `toDeviceVariantPack()` but reading host pointers. The executor then sees a `void*`-only variantPack and the flatbuffer-deserialized graph; it has lost all knowledge of the runtime tensor *type*. From this point forward, the path to a usable ragged abstraction goes through `flatbuffer_utilities::TensorAttributes → RaggedView` reconstruction.

This stage is invariant to A vs C in terms of *what* gets passed (raw pointers + serialized graph), but the *next* stage (executor) has to reconstruct different things depending on A vs C.

### Stage 2.5 — `CpuReferenceGraphExecutor::execute(...)`: virtual tensor allocation

```cpp
for(const auto& [id, attr] : tensorMap) {
    if(attr->virtual_() && updatedVariantPack.find(id) == updatedVariantPack.end()) {
        auto tensor = detail::createTensorFromAttribute(*attr);     // ITensor
        tensor->fillWithSentinelValue();
        virtualTensors.push_back(std::move(tensor));
        updatedVariantPack[id] = virtualTensors.back()->rawHostData();
    }
}
```

This is the "intermediate tensors" allocation. If a ragged primary is virtual (i.e. an intermediate produced by one op and consumed by another within the same graph), the executor needs to allocate it with `physicalElementCount` semantics — same allocation problem as Stage 2.1.

- **Option A.** Branch on `attr->ragged_offset_tensor_uid().has_value()` and dispatch to `make_unique<RaggedTensor<T>>`. Same two-pass ordering problem as Stage 2.1: must allocate aux first.
- **Option C.** Branch on the same condition and dispatch to `make_unique<IrregularTensor<T>>`. No ordering problem because the view is not built here.

The advantage of C at this stage is the same as at Stage 2.1: virtual intermediates whose iteration is never required (the executor just allocates them, threads their pointer through, and the next op's plan addresses them directly) are naturally `IrregularTensor` and incur no cost for view machinery they wouldn't use. The follow-up #1 in the plan ("verify the `IrregularTensor`-based allocation pattern fits cleanly into `CpuReferenceGraphExecutor`") is exactly this — and the answer, having now read the executor, is that **it fits cleanly**: the executor stores intermediates in a `std::vector<unique_ptr<ITensor>>` and only ever calls `rawHostData()` on them. Neither `begin()/end()` nor `getHostValue(...)` is invoked at the executor level, which is precisely the contract `IrregularTensor` satisfies.

### Stage 2.6 — `executor->execute(variantPackWithVirtualTensorsAdded)` per node

This dispatches into `SdpaFwdPlan::execute(...)` (or the analogous plan for each op):

```cpp
void execute(const std::unordered_map<int64_t, void*>& variantPack) override
{
    auto shallowQTensor = createShallowTensor<QDataType>(
        _params.qTensor, variantPack.at(_params.qTensor.uid));
    // … same for k, v, o, attnMask …

    utilities::CpuFpReferenceSdpa::forward<QDataType, KDataType, VDataType, ODataType, float>(
        *shallowQTensor, *shallowKTensor, *shallowVTensor, *shallowOTensor,
        _params.attnScaleValue, shallowAttnMaskTensor.get(),
        _params.leftBound, _params.rightBound, _params.topLeftAlignment);
}
```

`createShallowTensor<T>` builds a `ShallowTensor<T>(ptr, dims, strides)` — a non-owning `TensorBase<T>` view over the raw pointer. This is constructed **every time `execute` is called** and discarded after the call. It is the stage at which the type system regains visibility into the data.

This is **the** stage where the A-vs-C difference matters most.

#### Option A at Stage 2.6

There is no `RaggedTensor` to receive from the variant pack — the variant pack only carries `void*`. The plan layer has two options:

1. **Construct an aliasing `RaggedTensor`** from `(ptr, padded_dims, strides, physical_count, aux_ptrs, aux_dims, aux_strides)`. This requires `RaggedTensor` to have a second constructor that takes a `void*` rather than allocating its own `MigratableMemory`, or a separate `ShallowRaggedTensor` companion type. The plan as written has `RaggedTensor` *always* owning its memory. So in Option A, one of two things has to happen at this stage:
   - **(a)** Introduce a `ShallowRaggedTensor<T, IndexT>` companion that is the non-owning, plan-layer-built equivalent of `ShallowTensor<T>`. This brings the new-types-count for A from 1 to 2.
   - **(b)** Skip the `RaggedTensor` abstraction entirely at the plan layer and have the plan layer build a `ShallowTensor<T>` over the primary's pointer (as today), then *separately* hand the aux pointers and offsets into `CpuFpReferenceSdpa::forward` as extra arguments. This is a wider signature change on the CPU reference and is precisely the "no `shared_ptr` migration through CPU-ref signatures" line item that A's row in the comparison table claims as a feature.
2. **Plus**, A's plan-layer `resolveTensor` reconciliation helper is supposed to run here. Its job is to verify that the runtime's `RaggedTensor::raggedOffset()` pointer matches `tensorMap.at(graph.qRaggedOffsetUid)`. But at this stage the runtime *is* a `void*` — there is no `RaggedTensor::raggedOffset()` to compare against. The reconciler only makes sense in *Flow 1* (user-facing), where the user constructs a `RaggedTensor` with explicit aux refs and the test framework wants to check those against the graph's declaration. In Flow 2 (this stage), the variantPack-as-`void*` waist means there is no second declaration to reconcile against, so the reconciler is degenerate. **A's defining cost — the reconciler — does not apply in the test harness flow.** It applies only in the user flow, and only if the user opts into a code path that uses the reconciler. (Today there is no such code path; the user wires `variantPack` by hand.)

This is an important re-evaluation: the plan's case against A leans heavily on the `resolveTensor` reconciler as dead-weight. In the test harness flow, that dead weight is **not present** — A and C both reconstruct ragged metadata from the graph at the plan layer.

#### Option C at Stage 2.6

The plan layer constructs a `RaggedView` directly from the variant pack:

```cpp
auto qShallow = createShallowIrregularTensor<QDataType>(
    _params.qTensor, variantPack.at(_params.qTensor.uid));        // ShallowIrregularTensor<T>
auto qRagged  = createShallowAux<int32_t>(
    _params.qRaggedOffsetTensor, variantPack.at(_params.qRaggedOffsetTensor.uid));
auto qSeqLens = _params.qSeqLensTensor.has_value()
    ? createShallowAux<int32_t>(*_params.qSeqLensTensor, variantPack.at(_params.qSeqLensTensor->uid))
    : nullptr;
auto qView = std::make_shared<RaggedView<QDataType>>(qShallow, qRagged, qSeqLens);

utilities::CpuFpReferenceSdpa::forward<…>(*qView, …);
```

This works cleanly **only if `RaggedView`'s underlying is a `shared_ptr<TensorBase<T>>` rather than a `shared_ptr<IrregularTensor<T>>`** — because at this stage the underlying is not an `IrregularTensor` (which would own memory), it is a `ShallowTensor`-like non-owning wrapper over the variant-pack pointer.

This is a real structural finding about Option C as currently specified in `RaggedTensorsTentativePlan.md`:

> ```cpp
> RaggedView(std::shared_ptr<IrregularTensor<T>>  underlying, …);
> ```

This signature **does not fit the plan-layer construction site**. The plan layer does not have an `IrregularTensor<T>` at that point — it has a `void*` from `variantPack` and the dims/strides/physicalElementCount it cached in its params. To make the plan layer's site work, either:

- **(c1)** `RaggedView`'s underlying must be widened to `shared_ptr<TensorBase<T>>` (so it can accept either an owning `IrregularTensor` or a non-owning shallow wrapper). This is the cleanest fix and is what the plan's *follow-up #6* ("Future extension point for padded mode") already foreshadows. The cost is that the ctor's structural validation can no longer assume `_underlying->elementSpace() == _underlying->physicalElementCount()` — it has to be `elementSpace()`, full stop, which the shallow wrapper would have to compute and report correctly.
- **(c2)** Introduce a *third* type, `ShallowIrregularTensor<T>`, the non-owning analog of `IrregularTensor<T>`. This widens C's type count from 2 to 3 (`IrregularTensor`, `ShallowIrregularTensor`, `RaggedView`).
- **(c3)** Drop the `IrregularTensor` from `RaggedView`'s ctor and have the plan layer construct a `RaggedView` directly from `(ptr, dims, strides, physical_count, aux_ptrs, …)`. This collapses `RaggedView` into a single-type ragged-aware shallow tensor — and at that point it is essentially equivalent to `ShallowRaggedTensor` from A's option (1a). The architectural difference between A and C dissolves at the plan layer.

The cleanest fix is **(c1)** — widen `RaggedView`'s underlying to `shared_ptr<TensorBase<T>>`. The plan's own follow-up #6 anticipates this for the padded-mode extension; the test-harness flow turns it from "future extension" to "required for the in-scope use case." This is a small but non-cosmetic correction to the plan as written.

Note that **A and C converge** at this stage: both end up needing some flavour of non-owning ragged-aware `TensorBase<T>` subclass for the plan layer to build. A calls it `ShallowRaggedTensor`; C calls it "`RaggedView` with a `TensorBase<T>` underlying" or `ShallowIrregularTensor` + `RaggedView`. The difference is whether that non-owning ragged-aware type is *the only* ragged type (collapsed C) or is a *second* type alongside an owning one (A as written, or C with the widening).

#### `physicalElementCount` propagation into the plan

Both A and C need the plan layer to *know* `physicalElementCount` for each ragged tensor at execute time so it can be passed to whatever non-owning constructor is used. Today, `SdpaFwdParams` stores `TensorAttributesT qTensor` (the flatbuffer's unpacked form) which has `dims` and `strides` but no `physical_element_count`. The flatbuffer addition the RFC names — `ragged_offset_tensor_uid: long = null` — does not carry it either.

There are two options, applicable to both A and C:

- **(p1)** Add `physical_element_count: long = null` to `tensor_attributes.fbs`. Authored by whoever serializes the graph (the frontend, after the user calls `set_physical_element_count(...)` on the `TensorAttributes`). The plan caches it in `params` along with everything else.
- **(p2)** Compute `physical_element_count` at plan-execute time by reading the `ragged_offset` aux's host values. This requires the plan to read the aux tensor first, find its last element, and use that as the packed length. This works for offset-defined packed mode (the packed length is exactly `ragged_offset[B]`), but it adds an extra host-side read per execute. It also doesn't handle alignment padding cleanly.

`(p1)` is the structurally simpler fix and is *also* the fix the user-facing Stage 1.2 needs. Recommend adding it to the RFC.

#### Side-by-side at Stage 2.6

| Aspect | Option A | Option C (as written) | Option C (widened to `TensorBase<T>` underlying) |
|---|---|---|---|
| Type built in plan layer | Either `ShallowRaggedTensor<T>` (new type) or the existing `ShallowTensor<T>` + extra args to `forward` | Does not fit — `RaggedView` ctor demands `IrregularTensor` underlying | `RaggedView<T>` over a `ShallowTensor<T>` |
| New non-owning ragged type required | Yes (`ShallowRaggedTensor`) — 2 types total in A | N/A | None additional — `ShallowTensor<T>` already exists |
| `resolveTensor` reconciliation has anything to reconcile against | No (no runtime `RaggedTensor` to compare) | Same | Same |
| `CpuFpReferenceSdpa::forward` signature change required | Only if we go the "extra args to forward" route in A | N/A | None (view passed as `TensorBase<T>&`, signature unchanged) |
| Number of types touched at this stage | 2 (`RaggedTensor` + `ShallowRaggedTensor`) | N/A | 2 (`IrregularTensor` for bundle/intermediate, `RaggedView` for plan layer) |
| `physicalElementCount` propagation needed | Yes | Yes | Yes |

**The architectural difference between A and C is largely invisible at the plan layer.** Both need a non-owning ragged-aware wrapper to construct here. C's advantage over A is at *other* stages (1.2 user flow, 2.1 bundle alloc, 2.5 intermediate alloc), not this one.

### Stage 2.7 — `CpuFpReferenceSdpa::forward(...)`: the reference itself

Today:

```cpp
template <class QDataType, class KDataType, class VDataType, class ODataType, class CT = float>
static void forward(
    const hipdnn_data_sdk::utilities::TensorBase<QDataType>& q,
    const hipdnn_data_sdk::utilities::TensorBase<KDataType>& k,
    const hipdnn_data_sdk::utilities::TensorBase<VDataType>& v,
          hipdnn_data_sdk::utilities::TensorBase<ODataType>& o,
    std::optional<float> attnScaleValue = std::nullopt,
    const hipdnn_data_sdk::utilities::TensorBase<CT>* attnMask = nullptr,
    int64_t leftBound = -1, int64_t rightBound = -1, bool topLeftAlignment = true,
    hipdnn_data_sdk::utilities::TensorBase<float>* lse = nullptr)
{
    // … rank checks …
    // … per-batch, per-head, per-sq parallel loop, with q.getHostValue / k.getHostValue / etc.
    auto sdpaFwdFunc = [&](const std::vector<int64_t>& indices) {
        const auto b = indices[0]; const auto h = indices[1]; const auto sq = indices[2];
        // … inner loop over skv, d … uses q.getHostValue({b, h, sq, d}), etc.
    };
    const std::vector<int64_t> parallelDims = {batch, numHeads, seqQ};
    auto parallelFunc = …makeParallelTensorFunctor(sdpaFwdFunc, parallelDims);
    parallelFunc(std::thread::hardware_concurrency());
    o.memory().markHostModified();
}
```

The reference uses `getHostValue(indices)` and `setHostValue(value, indices)` — *random-access* into the tensor, not iterator-based. The ragged behaviour needed by the reference is **not** "iterate only valid elements" but "**bound the inner `sq` and `skv` loops by `validSeqLen(b)`**" — i.e. don't read past the valid sequence length for the current batch. Iteration over padding inside `getHostValue` is fine *if no one calls `getHostValue` with an out-of-valid-range index*.

This changes the substance of what ragged support looks like inside the reference. It is **not** primarily about the iterator strategy (that machinery matters for `randomizeTensor`, `fillWithSentinelValue`, comparison, but not for the loop bodies of the reference). It is primarily about the reference *knowing the valid sequence length per batch* so its loops can be bounded:

```cpp
const int64_t validSeqQ_b  = q.validSeqLen(b);              // (shared element #6)
const int64_t validSeqKv_b = k.validSeqLen(b);
for (int64_t sq = 0; sq < validSeqQ_b; ++sq) {
    for (int64_t skv = 0; skv < validSeqKv_b; ++skv) {
        // … as before …
    }
}
```

And:

```cpp
const std::vector<int64_t> parallelDims = {batch, numHeads, seqQMax};   // padded for scheduling
auto sdpaFwdFunc = [&](const std::vector<int64_t>& indices) {
    const auto b = indices[0]; const auto h = indices[1]; const auto sq = indices[2];
    if (sq >= q.validSeqLen(b)) return;                                  // early out
    // … as before …
};
```

This is the same change in A and C, because `validSeqLen` is on the shared base (`TensorBase`) per the plan's *shared element #6*. The reference receives `TensorBase<T>&` in both options. The only difference is that in C the object pointed at is a `RaggedView` whose `validSeqLen` reads its `_seqLens` aux; in A the object is a `RaggedTensor` whose `validSeqLen` reads its `_seqLens` aux. The reference code is identical.

A subtle point: the reference also uses `parallelDims = {batch, numHeads, seqQ}` to chunk parallel work. For the ragged case, using `seqQ_max` as `parallelDims[2]` is correct (the work scheduler doesn't care that some `(b, sq)` pairs are no-ops); the early-out inside the lambda makes them no-ops cheaply. Alternatively, using a per-batch `seqQ` would require a different parallel decomposition (no longer a Cartesian product), which is a bigger change. The early-out approach is sufficient.

This stage is invariant to A vs C.

### Stage 2.8 — `o.memory().markHostModified()`

The reference flags the output's host memory as modified so subsequent device reads will trigger a host-to-device copy. The output `o` is the same `TensorBase<T>&` from Stage 2.6, and `memory()` is the underlying `MigratableMemory`.

- **Option A.** The output's `memory()` returns `RaggedTensor`'s own `MigratableMemory` (or, in the plan-layer flow, the shallow ragged wrapper's `memory()` which forwards to the underlying variant-pack-pointed memory — which the shallow wrapper does not own and which arguably has no `MigratableMemory` to mark, because `ShallowTensor` is a `void*` view). This is an existing problem with `ShallowTensor` today, not new to ragged.
- **Option C.** Same — the `RaggedView`'s `memory()` forwards to `_underlying->memory()`, which for the plan-layer case is the shallow wrapper's memory, with the same existing problem.

This stage is invariant to A vs C and inherits whatever the existing `ShallowTensor` story is for `memory()`. (Reading the existing tests, `ShallowTensor` *does* have a `memory()` that returns something usable — likely a `MigratableMemory` over the existing `void*` without owning it — but verifying that is out of scope for this analysis.)

### Stage 2.9 — Output comparison (`verifyGraph` validation loop)

```cpp
for(const auto& tensorId : outputTensorIds) {
    auto& cpuTensor = cpuBundle.tensors.at(tensorId);
    auto& gpuTensor = gpuBundle.tensors.at(tensorId);
    gpuTensor->markDeviceModified();
    bool valid = _tensorIdToValidatorMap.at(tensorId)->allClose(*cpuTensor, *gpuTensor);
}
```

The validator iterates both tensors element-by-element to compare. For ragged outputs, the comparison should walk only valid elements (comparing padding values is meaningless and may produce false negatives).

- **Option A.** `cpuTensor` and `gpuTensor` are `RaggedTensor<T>` (since they were allocated as such at Stage 2.1). The validator iterates via the polymorphic index hook and walks valid elements. Works without harness changes.
- **Option C.** `cpuTensor` and `gpuTensor` are `IrregularTensor<T>`. Their `begin()/end()` throws. The validator needs the harness to wrap them in `RaggedView`s for the comparison, same as it did for randomization at Stage 2.2. Same `makeRaggedViewFromBundle` helper, applied at the validation site.

Same trade-off as Stage 2.2: A gets it for free; C needs ~5 extra lines in the validation loop to construct views first.

### Stage 2.10 — Summary of harness changes per option

| Stage | Option A change | Option C change |
|---|---|---|
| 2.1 — `generateBundles` / `tryAddTensorToBundles` | New branch in `createTensorFromAttribute` for `RaggedTensor`; **two-pass allocation** (aux before primary); bundle storage shifts `unique_ptr → shared_ptr` | New branch in `createTensorFromAttribute` for `IrregularTensor`; bundle unchanged |
| 2.2 — `initializeBundle` | None — `RaggedTensor.fillTensorWithRandomValues` walks valid elements | Wrap ragged primaries in temporary `RaggedView` for init; **ordering constraint**: aux populated before primary randomized |
| 2.5 — `CpuReferenceGraphExecutor` virtual tensor alloc | Same as 2.1 (two-pass + shared_ptr) | New branch returning `IrregularTensor`; no ordering |
| 2.6 — `SdpaFwdPlan::execute` | Build `ShallowRaggedTensor` (new type) or modify `forward` signature; reconciler is degenerate here | Build `RaggedView` over a `ShallowTensor`-like underlying — *requires widening `RaggedView`'s underlying to `shared_ptr<TensorBase<T>>` as plan currently writes it* |
| 2.7 — `forward` body | `validSeqLen`-bounded inner loops | Identical |
| 2.9 — Validation loop | None | Wrap ragged outputs in temporary `RaggedView` for `allClose` |

A is simpler at 2.2 and 2.9 (free via iterator hook on the owning type); C is simpler at 2.1 and 2.5 (no ordering constraint, no `shared_ptr` shift in the bundle); and at 2.6 the two options converge (both need a non-owning ragged-aware wrapper, and C's plan-as-written must be amended to make this construction site work).

---

## Where the two options actually diverge, distilled

The plan's comparison table is accurate at the *type-system* level, but it overweights the "single source of truth" / "no reconciler" benefits because those benefits apply in a flow (the user-facing flow with a reconciler) that **does not currently exist**. The plan layer in the test harness reconstructs ragged metadata from the graph in both A and C — there is no second source of truth in the harness path for either option to make consistent.

Stripped to what actually differs by stage:

| Question | A (`RaggedTensor` owning) | C (`IrregularTensor` + `RaggedView`) |
|---|---|---|
| At allocation time, can the harness use a one-pass loop? | No — needs aux before primary | Yes |
| Can the bundle stay `unique_ptr<ITensor>`? | No — must share aux refs | Yes |
| Does `bundle.randomizeTensor(uid)` automatically iterate valid elements only on ragged primaries? | Yes | No — needs a wrap step |
| Does `validator.allClose(*cpu, *gpu)` automatically iterate valid elements only? | Yes | No — needs a wrap step |
| In the plan layer, is a non-owning ragged-aware wrapper required? | Yes (`ShallowRaggedTensor`, or `forward` signature change) | Yes (`RaggedView` over `ShallowTensor`, with widened underlying) |
| Total new types introduced (for the test-harness-driven scope) | 2 (`RaggedTensor`, `ShallowRaggedTensor`) | 2 (`IrregularTensor`, `RaggedView`) — same count |
| User-facing single-object ergonomic | Yes — one `RaggedTensor` value | No — `IrregularTensor` for `variantPack`, `RaggedView` for iteration |
| Risk surface for "primary disagrees with its aux" | A reconciler exists (and is needed only in the user-facing flow) | None |
| Iteration in graph intermediates that are never iterated | Costs `RaggedTensor` ctor with aux refs that no one uses | Costs nothing — `IrregularTensor` exists for this |

**Once Option C is corrected to support a non-owning underlying in `RaggedView`, the cost asymmetries become symmetric**: A pays at the harness allocation/init/validation stages; C pays at the user-facing-single-object stage and at the harness `initializeBundle`/`verifyGraph` wrap-up stages. The total new type counts are equal (2 each). The "single source of truth on the graph" claim that anchors C's recommendation evaporates in the test-harness flow.

---

## Revised recommendation

The plan currently recommends C primarily on the strength of the "single source of truth + no reconciler" argument. That argument is sound **in the user-facing flow with a runtime-side bundle helper** (the *Possible mitigation* sketched in Stage 1.2), but that helper does not exist yet and is out of scope for the RFC's stated scope. In the **test-harness flow as it exists today**, the recommendation is less clear-cut:

- A is simpler at the bundle layer (allocation, randomization, validation: all free via the polymorphic index hook on the owning type), at the price of a two-pass allocation ordering constraint and a `unique_ptr → shared_ptr` shift in `GraphTensorBundle`.
- C is simpler at the bundle-allocation/intermediate-allocation layer (no ordering constraint, no `shared_ptr` shift), at the price of wrap-step code in `initializeBundle` and `verifyGraph`, and *requires* widening `RaggedView`'s underlying to `shared_ptr<TensorBase<T>>` for the plan-layer construction site to even compile.

For the user-facing flow, neither option provides a true `TensorAttributes → runtime` one-call helper, because the information required (`physicalElementCount` and the aux runtime tensors) is not on a single `TensorAttributes`. **A multi-input factory (or a runtime-side bundle helper) is needed in both cases.** C makes that factory marginally more natural by separating "the buffer" (`IrregularTensor`, what `variantPack` needs) from "the iterable" (`RaggedView`, what CPU references need); A makes it marginally more natural by giving the user one value that handles both.

**Concrete suggestion.**

1. **Add `physical_element_count` to the flatbuffer and to `TensorAttributes`.** Both options need it; sourcing it from the graph is cleaner than recomputing at execute time, and it makes Stage 1.2 user-side ragged allocation tractable for both options.
2. **Pick the option whose harness-side ergonomics matter more for the near-term work.** If the immediate work is "make `IntegrationGpuSdpaForward` and `IntegrationGpuSdpaBackward` pass with ragged inputs," A's allocate-init-validate "just works" loop is meaningfully easier to land. If the immediate work is "lay the groundwork for an executor-allocated intermediates story that won't paint itself into a corner with the future padded-mode extension," C's two-types story is the more durable foundation, but it needs the plan revisions in this document to actually fit the plan-layer construction site.
3. **Either way, write the runtime-side `TensorBundle` helper.** Most of the ergonomic distance between A and C *at the user layer* collapses once that helper exists — both options become a one-call factory dispatch from the user's perspective, and the user's two-vs-one-object question is hidden behind the factory.
4. **Drop the `resolveTensor` reconciler from the plan as currently scoped.** It does not run in the test-harness flow (the only flow this RFC needs to land), and the user-facing flow doesn't have a code path that would construct the redundant declaration the reconciler exists to check. If A is chosen, the reconciler can be added later if and when a user-side path is built that justifies it.

Net: **the choice between A and C is closer than the plan suggests**, the type counts are equal, the user-facing one-call ergonomic does not exist in either option without additional helper work, and the in-scope test-harness flow rewards different things in each option at different stages. A is marginally easier to land in the immediate term; C is marginally more durable past the current scope. Both work.
