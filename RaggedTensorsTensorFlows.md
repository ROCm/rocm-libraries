# Tensor flows in the hipDNN samples and integration tests

This document describes the transformations that tensor-shaped values undergo as they move through the codebase, in two flows:

- **Flow 1** — User-facing samples (e.g. `BnInference.cpp`, `SdpaFprop.cpp`).
- **Flow 2** — Integration-test harness (`IntegrationGraphVerificationHarness` → `GraphTensorBundle` → `CpuReferenceGraphExecutor` → per-op plan → `CpuFpReferenceSdpa`).

It deliberately does not discuss any specific ragged-tensor design. The aim is to enumerate, stage-by-stage, what tensor objects exist, what their static type is, who owns them, and what happens to them. This is the reference description the design alternatives in `RaggedTensorsTentativePlan.md` and `RaggedTensorsStageByStageAnalysis.md` are written against.

---

## Cast of types

A few baseline types matter at every stage:

- **`hipdnn_frontend::graph::TensorAttributes`** (frontend, in-memory): graph-side description of a tensor — `dims`, `strides`, `data_type`, `uid`, `is_virtual`, `name`, plus the existing handful of getter/setter pairs. There is **no** reference to runtime data on a `TensorAttributes`.
- **`hipdnn_flatbuffers_sdk::data_objects::TensorAttributes`** (flatbuffer-encoded): the serialized analog of the frontend's `TensorAttributes`, the form the graph executor consumes after deserialization. It mirrors the same fields.
- **`hipdnn_data_sdk::utilities::ITensor`** (runtime, polymorphic): the type-erased base of the runtime tensor hierarchy. Carries `dims()`, `strides()`, `rawHostData()`, `rawDeviceData()`, `elementCount()`, `elementSpace()`, and a `memory()` to access host/device migration state.
- **`TensorBase<T>`**: the typed virtual base. Adds `getHostValue(indices)`, `setHostValue(value, indices)`, `fillWithValue(v)`, `fillWithSentinelValue()`, `markHostModified()`, `markDeviceModified()`, iteration entry points.
- **`Tensor<T, HostAlloc, DeviceAlloc>`**: the concrete owning runtime tensor. Backs `host` and `device` storage with `MigratableMemory<T>`. Constructor invariant: `_packed = (elementCount == elementSpace)`. The default sample-level type.
- **`PinnedTensor<T>`**: alias for `Tensor<T, PinnedHostAllocator<T>, …>`.
- **`ShallowTensor<T>`**: a non-owning `TensorBase<T>` view over a `void*` plus `dims` and `strides`. Used inside per-op plans to turn `variantPack`'s raw pointers into something the CPU references can call.
- **`unordered_map<int64_t, void*>`** (a.k.a. the `variantPack`): the narrow waist. The data structure the user supplies to `graph.execute(...)` and the data structure every per-op plan's `execute(...)` consumes. Type-erased: only `uid → void*`.

A few helpers also recur:

- **`hipdnn_test_sdk::utilities::createTensorFromAttribute(const hipdnn_frontend::graph::TensorAttributes& attr)`**: dispatches on `attr.get_data_type()` and returns `unique_ptr<ITensor>` (concretely `make_unique<Tensor<T>>(attr.get_dim(), attr.get_stride())`). Test-SDK only; not exposed to user-facing samples.
- **`hipdnn_test_sdk::detail::createTensorFromAttribute(const flatbuffers_sdk::data_objects::TensorAttributes& attr)`**: same idea but consuming the flatbuffer-deserialized form. Used by `CpuReferenceGraphExecutor` when it needs to allocate intermediate tensors.
- **`hipdnn_test_sdk::detail::createShallowTensor<T>(TensorAttributesT, void*)`**: builds a `ShallowTensor<T>` from `(dims, strides, ptr)`. Called inside per-op plans on every `execute(...)`.
- **`hipdnn_test_sdk::utilities::GraphTensorBundle`**: `unordered_map<int64_t, unique_ptr<ITensor>>` keyed by tensor UID, with helpers `randomizeTensor(uid, …)`, `toHostVariantPack()`, `toDeviceVariantPack()`.

The shape of the pipe through both flows:

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

The user-facing flow and the test-harness flow both walk this pipe — what changes is who constructs each value, who owns it, and where the typed view at the right end comes from.

---

## Flow 1 — User-facing samples

Reference file: `projects/hipdnn/samples/batchnorm/BnInference.cpp`. The pattern is consistent across samples: graph build → runtime allocate → wire → execute → (optional) validate.

### Stage 1.1 — Graph construction

```cpp
auto graph = std::make_shared<graph::Graph>();
graph->set_io_data_type(inputType)
     .set_intermediate_data_type(intermediateType)
     .set_compute_data_type(hipdnn_frontend::DataType::FLOAT);

auto x           = createTensor({n, c, h, w}, inputType, layout);          // sample helper
auto scale       = createTensor({1, c, 1, 1}, intermediateType, layout);
auto bias        = createTensor({1, c, 1, 1}, intermediateType, layout);
auto mean        = createTensor({1, c, 1, 1}, intermediateType, layout);
auto invVariance = createTensor({1, c, 1, 1}, intermediateType, layout);

auto bnAttributes = graph::BatchnormInferenceAttributes();
bnAttributes.set_name("bn_inference_node");

auto y = graph->batchnorm_inference(x, mean, invVariance, scale, bias, bnAttributes);
y->set_output(true);

HIPDNN_FE_CHECK_SKIPPABLE(graph->build(handle));
```

**What exists after this stage.**
- A `graph::Graph` instance.
- One `std::shared_ptr<TensorAttributes>` per input (`x`, `scale`, `bias`, `mean`, `invVariance`) and one for the output (`y`).
- Each `TensorAttributes` has a `uid` assigned (the value returned by `attr->get_uid()`).

**What does not exist after this stage.**
- Any `ITensor` / `TensorBase` runtime object. No host or device memory is allocated yet.

### Stage 1.2 — Runtime tensor allocation

```cpp
utilities::Tensor<InputType>        xTensor(x->get_dim(), layout);
utilities::Tensor<IntermediateType> scaleTensor(scale->get_dim());
utilities::Tensor<IntermediateType> biasTensor(bias->get_dim());
utilities::Tensor<IntermediateType> meanTensor(mean->get_dim());
utilities::Tensor<IntermediateType> invVarianceTensor(invVariance->get_dim());
utilities::Tensor<InputType>        yTensor(y->get_dim(), layout);
```

**What exists after this stage.**
- Six `Tensor<T>` objects, value-owned in the user's stack frame.
- Each has its `MigratableMemory<T>` allocated (host buffer + device buffer).
- The link from a `Tensor<T>` to the corresponding `TensorAttributes` is by *the user's eyeballs* — there is no runtime cross-reference between them. The user passes `x->get_dim()` (and optionally `layout`) into the `Tensor<T>` constructor; there is no `Tensor::Tensor(const TensorAttributes&)` constructor in the user-facing data_sdk.

**Note on the test-SDK helper.** `hipdnn_test_sdk::utilities::createTensorFromAttribute(const TensorAttributes&)` does exist and would let the user write `auto xPtr = createTensorFromAttribute(*x);`. It is, however, in the test_sdk (not data_sdk), it returns `unique_ptr<ITensor>` rather than a concrete typed value, and samples do not use it. The user-facing pattern is the explicit `Tensor<T>(attr->get_dim(), layout)` form shown above.

### Stage 1.3 — Host-side initialization

```cpp
xTensor   .fillWithRandomValues(static_cast<InputType>(0.0f),  static_cast<InputType>(1.0f));
scaleTensor.fillWithRandomValues(static_cast<IntermediateType>(0.0f),
                                 static_cast<IntermediateType>(1.0f));
biasTensor .fillWithRandomValues(static_cast<IntermediateType>(0.0f),
                                 static_cast<IntermediateType>(1.0f));
meanTensor .fillWithRandomValues(static_cast<IntermediateType>(0.0f),
                                 static_cast<IntermediateType>(1.0f));
invVarianceTensor.fillWithRandomValues(static_cast<IntermediateType>(0.1f),
                                       static_cast<IntermediateType>(1.0f));
```

**What happens at this stage.**
- `fillWithRandomValues` walks the tensor's host buffer (via the iterator hook) and assigns values. The host side of `MigratableMemory` is marked as the source of truth; the device side will be lazily synced on the next access.

### Stage 1.4 — Variant-pack assembly

```cpp
std::unordered_map<int64_t, void*> variantPack;
variantPack[x          ->get_uid()] = xTensor          .memory().deviceData();
variantPack[scale      ->get_uid()] = scaleTensor      .memory().deviceData();
variantPack[bias       ->get_uid()] = biasTensor       .memory().deviceData();
variantPack[mean       ->get_uid()] = meanTensor       .memory().deviceData();
variantPack[invVariance->get_uid()] = invVarianceTensor.memory().deviceData();
variantPack[y          ->get_uid()] = yTensor          .memory().deviceData();
```

**What happens at this stage.**
- Each `tensor.memory().deviceData()` call ensures the device side of the tensor's `MigratableMemory` is up to date (triggering a host-to-device copy for inputs that were just filled on host), and returns the device pointer.
- The `variantPack` is the *only* runtime structure handed into the graph executor. Once assembled, all type information about the runtime tensors is lost: from the executor's perspective, each tensor is just a `void*` keyed by `uid`.

### Stage 1.5 — Graph execution

```cpp
HIPDNN_FE_CHECK(graph->execute(handle, variantPack, nullptr));
```

**What happens at this stage.**
- The frontend dispatches to the configured backend (e.g. `hip-kernel-provider`).
- The backend looks up each tensor's `void*` in `variantPack` by the UID it cached when the graph was built, and passes those pointers to the kernel.
- Kernel execution is asynchronous on the stream attached to the handle. The runtime `Tensor<T>` objects on the user's side are not touched by the backend except via `.deviceData()`.

### Stage 1.6 — Device-to-host sync (output marking)

```cpp
yTensor.memory().markDeviceModified();
auto yHostPtr = yTensor.memory().hostData();    // triggers D→H copy
```

**What happens at this stage.**
- `markDeviceModified()` tells `MigratableMemory` that the device side is now the source of truth and the host side is stale.
- The subsequent `hostData()` call observes the staleness and copies device → host. `yHostPtr` is a typed pointer into `yTensor`'s host buffer.

### Stage 1.7 — Optional CPU validation

```cpp
utilities::Tensor<InputType> yRefTensor(y->get_dim(), layout);

hipdnn_test_sdk::utilities::CpuFpReferenceBatchnorm::fwdInference(
    xTensor, scaleTensor, biasTensor, meanTensor, invVarianceTensor, yRefTensor);

auto validator
    = hipdnn_test_sdk::utilities::CpuFpReferenceValidation<InputType>(tolerance, tolerance);
const bool yValid = hipdnn_test_sdk::utilities::validateAndReport<InputType>(
    std::cout, "y", validator, yRefTensor, yTensor, tolerance, tolerance);
```

**What happens at this stage.**
- A fresh `Tensor<T> yRefTensor` is allocated to hold the reference output.
- The CPU reference (here `CpuFpReferenceBatchnorm::fwdInference`) takes the *same* `Tensor<T>` objects the user already allocated and reads/writes scalar values through `getHostValue` / `setHostValue` / iteration.
- The reference signature accepts `TensorBase<T>&` (the templated typed base), not `ITensor&` or `shared_ptr<…>`. The user's `Tensor<T>` values bind to this directly because `Tensor<T>` derives from `TensorBase<T>`.
- The validator compares the GPU result (`yTensor`, marked device-modified above) against the CPU reference result (`yRefTensor`, host-only) element-wise.

### Summary of object lifetimes in Flow 1

| Stage | New tensor-shaped objects this stage produces | Owner | Disposed when |
|---|---|---|---|
| 1.1 | `shared_ptr<TensorAttributes>` × 6 (graph nodes) | The `Graph` (owns the underlying `TensorAttributes`); the `shared_ptr` is held by the user for `get_uid()` lookups | Graph teardown |
| 1.2 | `Tensor<T>` × 6 (runtime owning) | User stack frame | Function return |
| 1.3 | — (fills host buffers in existing tensors) | — | — |
| 1.4 | `unordered_map<int64_t, void*>` (variant pack) | User stack frame | Function return |
| 1.5 | — (kernel reads/writes through device pointers) | — | — |
| 1.6 | — (marks device side modified, lazy D→H copy) | — | — |
| 1.7 | `Tensor<T>` (reference output) | User stack frame | Function return |

There is **no helper at the user layer that goes directly from a `TensorAttributes` to a runtime `Tensor<T>`**. Each runtime tensor is constructed by hand using `attr->get_dim()` (and sometimes `layout`); the linkage between a `TensorAttributes` and a runtime tensor is established only by the user's `variantPack[attr->get_uid()] = tensor.memory().deviceData()` step.

---

## Flow 2 — Integration-test harness

Reference files:
- `dnn-providers/hip-kernel-provider/src/integration_tests/IntegrationGraphVerificationHarness.hpp` (the harness template).
- `dnn-providers/hip-kernel-provider/src/integration_tests/asm_sdpa_engine/IntegrationGpuSdpaForward.cpp` (uses the harness).
- `projects/hipdnn/test_sdk/include/hipdnn_test_sdk/utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp` (the CPU executor).
- `projects/hipdnn/test_sdk/include/hipdnn_test_sdk/utilities/cpu_graph_executor/detail/SdpaFwdPlan.hpp` (the per-op plan for SDPA forward).
- `projects/hipdnn/test_sdk/include/hipdnn_test_sdk/utilities/CpuFpReferenceSdpa.hpp` (the CPU reference body).

The harness builds two parallel allocations of every (non-virtual) tensor in the graph — one to run on the GPU and one to run through the CPU reference — initialized identically and compared afterwards.

### Stage 2.1 — Graph build (test setup)

The test fixture constructs a graph the same way a sample would: a `Graph` is populated with nodes and `TensorAttributes`. Then:

```cpp
auto result = graph.build(_handle);
ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;
```

**What exists after this stage.**
- The graph and its `TensorAttributes` objects, identical to Stage 1.1.

### Stage 2.2 — Bundle generation (`generateBundles`)

```cpp
hipdnn_test_sdk::utilities::GraphTensorBundle gpuBundle;
hipdnn_test_sdk::utilities::GraphTensorBundle cpuBundle;
std::vector<int64_t> outputTensorIds;

generateBundles(graph, cpuBundle, gpuBundle, outputTensorIds);

// inside generateBundles → tryAddTensorToBundles:
cpuBundle.tensors.insert(
    {tensorId, hipdnn_test_sdk::utilities::createTensorFromAttribute(*tensorAttr)});
gpuBundle.tensors.insert(
    {tensorId, hipdnn_test_sdk::utilities::createTensorFromAttribute(*tensorAttr)});
```

`generateBundles` walks the graph (`graph.visit(...)`), and for every input and output `TensorAttributes` on every node:

1. Skips if `tensorAttr->get_is_virtual()` (virtual / intermediate tensors are handled later by the CPU executor; the GPU side handles them implicitly via workspace).
2. Skips if the bundle already has a tensor for this UID (avoids double-allocating shared inputs).
3. Otherwise, calls `createTensorFromAttribute(*tensorAttr)`, which returns `unique_ptr<ITensor>` — concretely `make_unique<Tensor<T>>(attr.get_dim(), attr.get_stride())` for the right `T` based on the attr's `data_type`.
4. Inserts the owning pointer into both `cpuBundle.tensors` and `gpuBundle.tensors` (separate allocations — same shape/dtype, but distinct memory).

**What exists after this stage.**
- Two `GraphTensorBundle` objects: each contains `unordered_map<int64_t, unique_ptr<ITensor>>` keyed by tensor UID, with one entry per non-virtual `TensorAttributes` in the graph.
- For every non-virtual `TensorAttributes`, an `ITensor` (concretely `Tensor<T>`) with its own `MigratableMemory`.
- `outputTensorIds`: a `vector<int64_t>` of UIDs the harness will later compare GPU-vs-CPU on.

**Note on visit order.** `graph.visit` walks in topological order; outputs of each node are added before the node's inputs. There is no relationship in the visit order between a "primary" tensor (e.g. `q`) and any auxiliary tensor it may reference (e.g. `ragged_offset`). At the `tryAddTensorToBundles` call site, the bundle has only the per-`TensorAttributes` information `createTensorFromAttribute` can see — `dims`, `strides`, `data_type`. Anything that would require *other* `TensorAttributes` (such as the `TensorAttributes` of a `seq_lens` tensor referenced by an `SdpaAttributes`, which is attached to the *node*, not to the primary's `TensorAttributes`) is unavailable at this stage.

### Stage 2.3 — Bundle initialization (`initializeBundle`)

```cpp
for(auto& tensorPair : bundle.tensors)
{
    bundle.randomizeTensor(tensorPair.first, DEFAULT_MIN, DEFAULT_MAX, seed);
}
```

This is invoked twice (once for `cpuBundle`, once for `gpuBundle`) with the *same* seed, so both bundles end up byte-identical on the host side.

`bundle.randomizeTensor(uid, min, max, seed)` looks up the `ITensor` for `uid` and calls `it->second->fillTensorWithRandomValues(min, max, seed)`. The `ITensor`'s implementation walks the host buffer through the iterator hook and assigns values.

**What happens at this stage.**
- Both bundles' tensors have host buffers populated with the same random values.
- The host side of each tensor's `MigratableMemory` is marked as the source of truth; device side will lazily sync on first access.

### Stage 2.4 — GPU execution (`executeGpuGraph`)

```cpp
int64_t workspaceSize;
auto result = graph.get_workspace_size(workspaceSize);
hipdnn_data_sdk::utilities::Workspace workspace(static_cast<size_t>(workspaceSize));

auto variantPack = bundle.toDeviceVariantPack();    // unordered_map<int64_t, void*>
result = graph.execute(handle, variantPack, workspace.get());
```

`toDeviceVariantPack()` walks the `gpuBundle.tensors` map and produces `{uid → tensor->rawDeviceData()}`. Each `rawDeviceData()` call ensures the device buffer is in sync with the host buffer (triggers H→D copy on first call after random init).

**What happens at this stage.**
- The GPU side of every tensor in `gpuBundle` is populated.
- `graph.execute(...)` dispatches to the backend (same as Stage 1.5).
- Outputs are produced into the device buffers of the output tensors in `gpuBundle`. The host side of those tensors is now stale.

### Stage 2.5 — CPU execution (`executeCpuGraph`)

```cpp
auto [serializedGraph, serErr] = graph.to_binary();

hipdnn_test_sdk::utilities::CpuReferenceGraphExecutor().execute(
    serializedGraph.data(), serializedGraph.size(), bundle.toHostVariantPack());
```

`toHostVariantPack()` is the same flatten as `toDeviceVariantPack()` but reading host pointers: `{uid → tensor->rawHostData()}` over the `cpuBundle.tensors` map.

`graph.to_binary()` serializes the in-memory graph (including all `TensorAttributes`) into a flatbuffer-encoded byte buffer.

**What happens at this stage.**
- The CPU executor receives a serialized graph and a `void*`-only variant pack. It has no direct access to the runtime `ITensor` objects in the bundle — only to their host pointers.

### Stage 2.6 — CPU executor: deserialize and plan

Inside `CpuReferenceGraphExecutor::execute(graphBuffer, size, variantPack)`:

```cpp
auto graphWrap
    = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(graphBuffer, size);

std::vector<std::unique_ptr<detail::IGraphNodePlanExecutor>> planExecutors;
for(uint32_t i = 0; i < graphWrap.nodeCount(); i++)
{
    auto& node = graphWrap.getNode(i);
    planExecutors.push_back(buildPlanForNode(graphWrap, node));
}
```

For each node in the (topologically sorted) graph, `buildPlanForNode` selects a per-op plan builder (via a signature key keyed on `node.attributes_type()` and the data types of its tensors), validates that the builder is applicable, and produces a `unique_ptr<IGraphNodePlanExecutor>`.

For SDPA forward, this produces a `SdpaFwdPlan<QType, KType, VType, OType>` whose `_params` is a `SdpaFwdParams` carrying *unpacked copies* of the relevant `TensorAttributesT` (the flatbuffer's mutable C++ form), the scale value, mask bounds, and an optional attention-mask `TensorAttributesT`. **No `ITensor` lookups happen at planning time.** The plan caches only the graph-side descriptions.

### Stage 2.7 — CPU executor: virtual-tensor allocation

```cpp
std::vector<std::unique_ptr<hipdnn_data_sdk::utilities::ITensor>> virtualTensors;
const std::unordered_map<int64_t, void*> variantPackWithVirtualTensorsAdded
    = populateVariantPackWithMissingVirtualTensors(
        variantPack, graphWrap.getTensorMap(), virtualTensors);

// inside populateVariantPackWithMissingVirtualTensors:
for(const auto& [id, attr] : tensorMap)
{
    if(attr->virtual_() && updatedVariantPack.find(id) == updatedVariantPack.end())
    {
        auto tensor = detail::createTensorFromAttribute(*attr);     // ITensor
        tensor->fillWithSentinelValue();
        virtualTensors.push_back(std::move(tensor));
        updatedVariantPack[id] = virtualTensors.back()->rawHostData();
    }
}
```

For every `TensorAttributes` in the deserialized graph that is marked virtual and has no entry in the incoming variant pack, the executor allocates a fresh `Tensor<T>` (via the flatbuffer-side `createTensorFromAttribute`), fills it with sentinel values, stores ownership in `virtualTensors`, and patches `rawHostData()` into the variant pack.

**What happens at this stage.**
- Additional `Tensor<T>` instances exist on the executor's stack for intermediates.
- The `variantPackWithVirtualTensorsAdded` is now complete: every UID referenced by any node in the graph has a corresponding `void*`.

### Stage 2.8 — Per-op plan dispatch

```cpp
for(auto& executor : planExecutors)
{
    executor->execute(variantPackWithVirtualTensorsAdded);
}
```

Each per-op plan's `execute(variantPack)` is invoked in topological order. For `SdpaFwdPlan`:

```cpp
void execute(const std::unordered_map<int64_t, void*>& variantPack) override
{
    auto shallowQTensor = createShallowTensor<QDataType>(
        _params.qTensor, variantPack.at(_params.qTensor.uid));
    auto shallowKTensor = createShallowTensor<KDataType>(
        _params.kTensor, variantPack.at(_params.kTensor.uid));
    auto shallowVTensor = createShallowTensor<VDataType>(
        _params.vTensor, variantPack.at(_params.vTensor.uid));
    auto shallowOTensor = createShallowTensor<ODataType>(
        _params.oTensor, variantPack.at(_params.oTensor.uid));

    std::unique_ptr<hipdnn_data_sdk::utilities::TensorBase<float>> shallowAttnMaskTensor;
    if(_params.attnMaskTensor.has_value())
    {
        shallowAttnMaskTensor = createShallowTensor<float>(
            *_params.attnMaskTensor, variantPack.at(_params.attnMaskTensor->uid));
    }

    utilities::CpuFpReferenceSdpa::forward<QDataType, KDataType, VDataType, ODataType, float>(
        *shallowQTensor, *shallowKTensor, *shallowVTensor, *shallowOTensor,
        _params.attnScaleValue, shallowAttnMaskTensor.get(),
        _params.leftBound, _params.rightBound, _params.topLeftAlignment);
}
```

`createShallowTensor<T>(tensorAttrT, void*)` (in `FlatbufferTensorAttributesUtils.hpp`) returns `unique_ptr<ShallowTensor<T>>` — a non-owning typed view over `(ptr, dims, strides)`.

**What happens at this stage.**
- For each input/output the plan needs, a per-call `ShallowTensor<T>` is built. These are non-owning views: they do not own memory and do not duplicate the data. They live only for the duration of this `execute` call.
- The plan passes these as `TensorBase<T>&` (the base class of `ShallowTensor<T>`) into the CPU reference function.

### Stage 2.9 — CPU reference (`CpuFpReferenceSdpa::forward`)

Reduced signature:

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
    // … per-batch, per-head, per-sq parallel loop …
    auto sdpaFwdFunc = [&](const std::vector<int64_t>& indices) {
        const auto b = indices[0]; const auto h = indices[1]; const auto sq = indices[2];

        for (int64_t skv = 0; skv < seqKv; ++skv) {
            auto dot = CT(0);
            for (int64_t d = 0; d < headDim; ++d) {
                dot += static_cast<CT>(q.getHostValue({b, h, sq, d}))
                     * static_cast<CT>(k.getHostValue({b, kvHeadK, skv, d}));
            }
            scores[skv] = dot * scale;
        }
        // … softmax, weighted sum over V …

        for (int64_t dv = 0; dv < headDimV; ++dv) {
            o.setHostValue(/* result */, {b, h, sq, dv});
        }
    };
    auto parallelFunc = detail::makeParallelTensorFunctor(sdpaFwdFunc, {batch, numHeads, seqQ});
    parallelFunc(std::thread::hardware_concurrency());

    o.memory().markHostModified();
    if (lse) lse->memory().markHostModified();
}
```

**What happens at this stage.**
- The reference reads scalars from inputs via `q.getHostValue(indices)` / `k.getHostValue(...)` / `v.getHostValue(...)` (and `attnMask` if present).
- The reference writes scalars to the output via `o.setHostValue(value, indices)`.
- All reads and writes go through the typed `TensorBase<T>` interface; the underlying buffer is the `void*` from the variant pack (which itself is the host buffer of the corresponding `Tensor<T>` in `cpuBundle`).
- After the parallel loop, `o.memory().markHostModified()` flags the output's host memory as the source of truth so subsequent device reads would trigger a copy.

### Stage 2.10 — Plan and shallow-tensor teardown

After the plan's `execute(...)` returns:
- All `ShallowTensor<T>` instances built in Stage 2.8 go out of scope and are destroyed. They never owned memory, so destruction is trivial.
- The next plan in `planExecutors` is invoked, repeating Stages 2.8–2.10 with a different op.

After the executor's loop finishes:
- `virtualTensors` (the vector of owning intermediate `Tensor<T>`s) goes out of scope and is destroyed. The intermediate buffers are freed.

### Stage 2.11 — Output comparison (`verifyGraph` validation loop)

Back in the harness, after both `executeGpuGraph` and `executeCpuGraph` have returned:

```cpp
for(const auto& tensorId : outputTensorIds)
{
    auto& cpuTensor = cpuBundle.tensors.at(tensorId);
    auto& gpuTensor = gpuBundle.tensors.at(tensorId);

    gpuTensor->markDeviceModified();    // device buffer is the source of truth post-execute

    if(_tensorIdToValidatorMap.find(tensorId) == _tensorIdToValidatorMap.end())
    {
        FAIL() << …;
    }

    bool valid = _tensorIdToValidatorMap.at(tensorId)->allClose(*cpuTensor, *gpuTensor);
    ASSERT_TRUE(valid) << …;
}
```

**What happens at this stage.**
- `gpuTensor->markDeviceModified()` flags the GPU result's device buffer as the source of truth, so any subsequent `hostData()` call inside `allClose` will trigger a D→H copy.
- The validator (registered earlier by the test via `registerValidator(...)` with a tolerance per output) is looked up by UID and invoked on the two `ITensor` references.
- `allClose(*cpuTensor, *gpuTensor)` iterates both tensors element-by-element and compares with the registered tolerance.

### Summary of object lifetimes in Flow 2

| Stage | New tensor-shaped objects this stage produces | Owner | Disposed when |
|---|---|---|---|
| 2.1 | `shared_ptr<TensorAttributes>` × N (graph nodes) | Graph | Graph teardown |
| 2.2 | `unique_ptr<ITensor>` × 2N (one per non-virtual attr × 2 bundles) | `GraphTensorBundle` (in `cpuBundle.tensors` / `gpuBundle.tensors`) | Bundle goes out of scope at end of `verifyGraph` |
| 2.3 | — (fills host buffers in existing tensors) | — | — |
| 2.4 | `unordered_map<int64_t, void*>` (GPU variant pack) | Stack frame inside `executeGpuGraph` | `executeGpuGraph` returns |
| 2.5 | Serialized graph buffer (`vector<uint8_t>`); `unordered_map<int64_t, void*>` (CPU variant pack) | Stack frame inside `executeCpuGraph` | `executeCpuGraph` returns |
| 2.6 | `GraphWrapper` (over the serialized graph); per-node `unique_ptr<IGraphNodePlanExecutor>` × N | `CpuReferenceGraphExecutor::execute` stack frame | `execute` returns |
| 2.7 | `unique_ptr<ITensor>` for every virtual intermediate; augmented variant pack | `virtualTensors` vector and the executor's stack frame | `execute` returns |
| 2.8 | `unique_ptr<ShallowTensor<T>>` × number of plan inputs/outputs (per plan invocation) | Plan's `execute` stack frame | Plan's `execute` returns |
| 2.9 | — (reads/writes scalar values into existing host buffers) | — | — |
| 2.10 | — (shallow tensors and plan executors destroyed in reverse order) | — | — |
| 2.11 | — (reads existing host buffers in both bundles) | — | — |

---

## The five places where an `ITensor` / `TensorBase`-derived object exists

Pulling the above out of the per-stage narrative, an `ITensor` / `TensorBase` object exists at exactly five conceptual locations:

1. **The user's stack frame in a sample** (Flow 1, Stages 1.2 onward). Concrete `Tensor<T>` values per `TensorAttributes`. One per input/output the user cares about. The user constructs these from `attr->get_dim()` (and possibly `layout`); there is no helper that does it from a `TensorAttributes` directly at the user-facing level.

2. **The integration harness's `GraphTensorBundle`** (Flow 2, Stage 2.2). One `unique_ptr<ITensor>` per non-virtual `TensorAttributes`, allocated by `createTensorFromAttribute` (test-SDK helper that dispatches on `attr.get_data_type()` and returns `make_unique<Tensor<T>>(attr.get_dim(), attr.get_stride())`). Two parallel bundles exist (CPU and GPU), with identical contents on the host side.

3. **The CPU executor's `virtualTensors` vector** (Flow 2, Stage 2.7). One `unique_ptr<ITensor>` per virtual `TensorAttributes` in the deserialized graph, allocated by the flatbuffer-side `createTensorFromAttribute` and filled with sentinel values. Lives for the duration of `CpuReferenceGraphExecutor::execute`.

4. **Per-op plan `execute(...)` invocations** (Flow 2, Stage 2.8). One `unique_ptr<ShallowTensor<T>>` per input/output the plan needs, built by `createShallowTensor<T>(tensorAttrT, void*)` from the variant pack on every `execute` call. Non-owning views; live for the duration of the call.

5. **CPU reference function arguments** (Flow 1 Stage 1.7, Flow 2 Stage 2.9). The reference receives `TensorBase<T>&` (sometimes `const&`, sometimes `*`). In Flow 1 this binds to the user's `Tensor<T>` directly; in Flow 2 this binds to a `ShallowTensor<T>` from category 4 above. The reference reads/writes scalar values via `getHostValue(indices)` / `setHostValue(value, indices)`.

The `variantPack` (`unordered_map<int64_t, void*>`) is the type-erased waist between categories 1 + 2 (which hold typed owning tensors) and categories 3 + 4 + 5 (which need typed views, reconstructed from `(attr, void*)` pairs on the executor side).
