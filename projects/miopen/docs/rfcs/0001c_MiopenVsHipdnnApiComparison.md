# MIOpen vs. hipDNN API — comparison for an XLA migration

- Status: informational / analysis (not a normative RFC)
- Audience: XLA/StreamExecutor ROCm backend maintainers, MIOpen & hipDNN teams
- Related: [`0001_HipdnnForwardingWrapper.md`](0001_HipdnnForwardingWrapper.md), [`0001a_InvestigationReference.md`](0001a_InvestigationReference.md)

## 0. Purpose

XLA's ROCm backend currently drives deep-learning primitives (convolution,
batch norm, pooling, softmax, activation, fused attention, etc.) through
**MIOpen**. We want to understand what it takes to move XLA onto **hipDNN**.

This document compares the two APIs at the level that matters for a
StreamExecutor integration: the object model, how a computation is described
and executed, algorithm selection, workspace management, data types/layouts,
error handling, and the concrete per-operation mapping. It then lays out the
migration options (including the *transitional* MIOpen→hipDNN forwarding wrapper
already scoped in `0001_HipdnnForwardingWrapper.md`) and the feature gaps that
XLA must design around.

The short version: **MIOpen is a flat, imperative C API (handle + typed
descriptors + per-op find/execute). hipDNN is a graph-based API modeled very
closely on NVIDIA cuDNN v8+ — a `hipdnnBackend*` descriptor C API plus a
header-only C++ frontend (`hipdnn_frontend`) that mirrors `cudnn_frontend`.**
For XLA, the strategically important consequence is that hipDNN's frontend is
API-shaped like cuDNN's graph API, so XLA code paths that already target the
cuDNN graph/`cudnn_frontend` API are structurally much closer to hipDNN than to
MIOpen.

---

## 1. TL;DR comparison

| Dimension | MIOpen | hipDNN |
|---|---|---|
| Paradigm | Flat imperative C API; one call per op | Graph-based; describe an op-graph, compile, execute |
| Surface XLA links | `libMIOpen.so`, `<miopen/miopen.h>` (C) | `libhipdnn_backend.so` (C) + header-only `hipdnn_frontend` (C++) |
| Closest NVIDIA analog | cuDNN v7 "legacy" per-op API | cuDNN v8+ backend descriptor API + `cudnn_frontend` |
| Core objects | `miopenHandle_t`, `miopenTensorDescriptor_t`, `miopenConvolutionDescriptor_t`, per-op descriptors | `hipdnnHandle_t`, `hipdnnBackendDescriptor_t` (one opaque type, typed by enum) / frontend `Graph`, `Tensor_attributes` |
| Describe a computation | Set typed descriptors, call op fn | Build op-graph of nodes+tensors (`conv_fprop`, `batchnorm`, `matmul`, `pointwise`, `sdpa_*`, ...) |
| Algorithm selection | `miopenFind*Algorithm` (benchmark) or Immediate mode (`GetSolution`/`RunSolution`); Find-DB/perf-DB on disk | Engine heuristics → engine configs → execution plans; optional plan autotune; JSON/binary plan (de)serialization |
| Fusion | Explicit `miopenFusionPlan` API (limited op set) + a few fixed fused entry points | Native: any supported subgraph fuses (conv+bias+act, norm+pointwise, attention, ...) |
| Workspace | Caller queries size per op, allocates, passes in | Query `get_workspace_size()`; caller allocates; passed in variant pack/execute |
| Tensor layout | Encoded as **strides**; some layout enums; vectorized layouts (NCHWc) supported | Encoded as **dims + strides** (no layout enum); vectorized/packed layouts not expressible as plain strides |
| Scaling (α/β) | `alpha`/`beta` params on most ops | **No α/β on the conv op** in the base graph API; expressed as explicit `pointwise` scale/add nodes instead |
| Grouped/depthwise conv | `miopenSetConvolutionGroupCount` | No group-count attribute on the base conv-forward op (gap today) |
| Status type | `miopenStatus_t` | `hipdnnStatus_t` (backend) / `Error`/`error_t` (frontend) |
| Maturity | Production, broad framework adoption | Beta ("not recommended for production" per docs); plugin-based, actively growing |
| Backends behind it | MIOpen solvers/kernels | Plugin engines (incl. a **MIOpen provider** that calls back into MIOpen), hipBLASLt, hip-kernel, etc. |

---

## 2. Programming model

### 2.1 MIOpen — imperative, per-op

MIOpen exposes a flat C API in `projects/miopen/include/miopen/miopen.h`. The
usage pattern for a convolution is representative of the whole library:

1. `miopenCreate(&handle)` (or `miopenCreateWithStream`).
2. Create + populate typed descriptors: `miopenCreateTensorDescriptor` /
   `miopenSetTensorDescriptor` for X/W/Y, `miopenCreateConvolutionDescriptor` /
   `miopenInitConvolutionNdDescriptor` (+ `miopenSetConvolutionGroupCount`).
3. Query workspace: `miopenConvolutionForwardGetWorkSpaceSize(...)`.
4. Pick an algorithm — either
   - **Find mode**: `miopenFindConvolutionForwardAlgorithm(...)` benchmarks
     candidates and returns a sorted `miopenConvAlgoPerf_t[]`, or
   - **Immediate mode**: `miopenConvolutionForwardGetSolutionCount` /
     `...GetSolution` / `...GetSolutionWorkspaceSize` /
     `...CompileSolution` / `miopenConvolutionForwardImmediate`.
5. Execute: `miopenConvolutionForward(handle, alpha, xDesc, x, wDesc, w,
   convDesc, algo, beta, yDesc, y, workSpace, workSpaceSize)`.

Each operation family has its own parallel set of functions:
`miopenBatchNormalizationForwardTraining/Inference(+_V2/_V3)`,
`miopenPoolingForward`, `miopenLRNForward`, `miopenSoftmaxForward(_V2)`,
`miopenActivationForward`, `miopenRNNForward*`, plus an explicit fusion API
(`miopenCreateFusionPlan`, ...). Tuning state (Find-DB, perf-DB, `.kdb` kernel
DBs) lives on disk and is read/written implicitly.

Key properties for an integrator:
- No graph object; every op is described and launched independently.
- The caller owns algorithm selection and workspace explicitly.
- Descriptors are strongly typed C handles; the ABI is stable and broadly used.

### 2.2 hipDNN — graph-based, two layers

hipDNN (see `projects/hipdnn/docs/user-guides/what-is-hipdnn.rst`) has two
public layers:

- **Backend** — a C API (`libhipdnn_backend.so`, `<hipdnn_backend.h>`) built
  around a single opaque descriptor type whose behavior is selected by an enum,
  exactly like cuDNN's backend API:
  - `hipdnnCreate` / `hipdnnDestroy` / `hipdnnSetStream`
  - `hipdnnBackendCreateDescriptor(type, &desc)` /
    `hipdnnBackendSetAttribute(desc, name, type, count, ptr)` /
    `hipdnnBackendGetAttribute(...)` / `hipdnnBackendFinalizeDescriptor(desc)`
  - `hipdnnBackendExecute(handle, executionPlan, variantPack)`
  - Descriptor kinds include `HIPDNN_BACKEND_TENSOR_DESCRIPTOR`,
    `HIPDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR`,
    `HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR`, `ENGINEHEUR` → `ENGINECFG` →
    `EXECUTION_PLAN`, and `HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR`.

- **Frontend** — a header-only C++ library (`<hipdnn_frontend.hpp>`) that wraps
  the backend C API in an ergonomic graph builder, mirroring `cudnn_frontend`.
  The workflow (`projects/hipdnn/docs/user-guides/how-to/build-execute-hipdnn.rst`):

```cpp
using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

Graph graph;
graph.set_io_data_type(DataType::HALF)
     .set_compute_data_type(DataType::FLOAT);

auto x = graph.tensor(TensorAttributes().set_dim({N,C,H,W}).set_stride({...}));
auto w = graph.tensor(TensorAttributes().set_dim({K,C,R,S}).set_stride({...}));
auto y = graph.conv_fprop(x, w, ConvFpropAttributes()
                              .set_padding({ph,pw})
                              .set_stride({sh,sw})
                              .set_dilation({dh,dw}));
y->set_output(true).set_uid(Y_UID);

hipdnnHandle_t handle; hipdnnCreate(&handle);
graph.build(handle);                    // build_operation_graph + create_execution_plans + check_support + build_plans
int64_t ws = 0; graph.get_workspace_size(ws);

std::unordered_map<int64_t, void*> variantPack = {
    {x->get_uid(), xptr}, {w->get_uid(), wptr}, {y->get_uid(), yptr}};
graph.execute(handle, variantPack, workspace);
```

Frontend graph node builders available today include (from
`projects/hipdnn/frontend/include/hipdnn_frontend/Graph.hpp`):
`conv_fprop`, `conv_dgrad`, `conv_wgrad`, `batchnorm`, `batchnorm_inference`,
`layernorm`, `rmsnorm`, `matmul`, `pointwise` (1/2/3-input), `reduction`,
`resample` (pooling), and `sdpa_backward` (scaled-dot-product attention).
Graph lifecycle methods: `validate`, `build_operation_graph`,
`create_execution_plans`, `check_support`, `build_plans`, `build`,
`get_workspace_size`, `execute`, plus plan selection
(`deselect_engines`, `deselect_workspace_greater_than`,
`set_preferred_engine_id_ext`) and (de)serialization (`serialize`/`deserialize`,
`to_json`/`to_binary`).

Key properties for an integrator:
- You describe an op (or a fused subgraph), then compile once, then execute
  many times; tensors are bound at execute time by **UID → device pointer** in
  a *variant pack*, not baked into the descriptor.
- Engine/algorithm selection is a first-class, queryable object (heuristics
  produce ranked engine configs; plans can be serialized/cached).
- Fusion is the native mode of operation, not a bolt-on.

### 2.3 Mental-model mapping

| MIOpen concept | hipDNN backend | hipDNN frontend |
|---|---|---|
| `miopenHandle_t` | `hipdnnHandle_t` | `hipdnnHandle_t` (passed to `build`/`execute`) |
| `miopenTensorDescriptor_t` | `HIPDNN_BACKEND_TENSOR_DESCRIPTOR` | `Tensor_attributes` (`graph.tensor(...)`) |
| `miopenConvolutionDescriptor_t` + op call | `..._OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR` | `graph.conv_fprop(x, w, ConvFpropAttributes{...})` |
| (implicit, one op) | `HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR` | the `Graph` object |
| `miopenFind*` / Immediate `GetSolution` | `ENGINEHEUR`→`ENGINECFG`→`EXECUTION_PLAN` | `create_execution_plans` / `build_plans` |
| `miopenConvAlgoPerf_t` / solution id | engine config + execution plan | execution plan (indexable, serializable) |
| workspace size query per op | `HIPDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE` | `get_workspace_size()` |
| pointers passed to op call | `HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR` | `variantPack` map (`uid → ptr`) |
| `miopenStatus_t` | `hipdnnStatus_t` | `Error` / `error_t` |

---

## 3. Data types

Every MIOpen data type has a hipDNN equivalent (from the prototype's worked
translation in `0001a_InvestigationReference.md`):

| MIOpen | hipDNN |
|---|---|
| `miopenHalf` | `HIPDNN_DATA_HALF` |
| `miopenFloat` | `HIPDNN_DATA_FLOAT` |
| `miopenBFloat16` | `HIPDNN_DATA_BFLOAT16` |
| `miopenDouble` | `HIPDNN_DATA_DOUBLE` |
| `miopenInt8` | `HIPDNN_DATA_INT8` |
| `miopenInt32` | `HIPDNN_DATA_INT32` |
| `miopenInt64` | `HIPDNN_DATA_INT64` |
| `miopenFloat8_fnuz` | `HIPDNN_DATA_FP8_E4M3_FNUZ` |
| `miopenBFloat8_fnuz` | `HIPDNN_DATA_FP8_E5M2_FNUZ` |

Note: dtype *expressibility* is 1:1, but whether the **selected engine**
supports a given dtype/shape is a separate question resolved by `check_support`
/ engine heuristics. In hipDNN you additionally separate **IO data type**
(`set_io_data_type`) from **compute/accumulation type**
(`set_compute_data_type`) and **intermediate type**
(`set_intermediate_data_type`) at graph granularity, whereas MIOpen derives the
accumulation type implicitly and takes IO types per descriptor.

---

## 4. Tensors and layouts

- **MIOpen** encodes layout as strides on the tensor descriptor, and also has
  some layout enums; it supports **vectorized/packed layouts** (e.g.
  `NCHWc4`/`NCHWc8`, `CHWNc*`) used by some high-performance kernels.
- **hipDNN** takes **dims + strides** on each tensor (`HIPDNN_ATTR_TENSOR_DIMENSIONS`,
  `HIPDNN_ATTR_TENSOR_STRIDES`) with **no separate layout enum** — NCHW vs NHWC
  is simply a different stride vector. Each tensor also carries a **unique id**
  (`HIPDNN_ATTR_TENSOR_UNIQUE_ID`) that links the graph to the variant pack.

**Migration implication:** ordinary NCHW/NHWC/NCDHW/NDHWC layouts map cleanly by
copying strides. **Vectorized layouts have no plain dims+strides form** and are
a gap — XLA paths that rely on MIOpen vectorized layouts cannot be expressed in
hipDNN's base graph API today and would have to stay on MIOpen (or fall through
to the MIOpen provider engine).

---

## 5. Convolution — the deep dive

Convolution is the first op family targeted by the forwarding wrapper and the
best-worked example, so it is the clearest lens on the API gap. The following
field-by-field mapping is taken from the prototype in
`0001a_InvestigationReference.md` (which builds the hipDNN backend graph by hand
for `miopenConvolutionForward`):

**Maps 1:1**

| MIOpen input | hipDNN backend target |
|---|---|
| `handle` | paired `hipdnnHandle_t` → `HIPDNN_ATTR_OPERATIONGRAPH_HANDLE` / `..._EXECUTION_PLAN_HANDLE` |
| `xDesc`/`wDesc`/`yDesc` dims | `HIPDNN_ATTR_TENSOR_DIMENSIONS` (×3) |
| `xDesc`/`wDesc`/`yDesc` strides (layout) | `HIPDNN_ATTR_TENSOR_STRIDES` (×3) |
| tensor `dataType` | `HIPDNN_ATTR_TENSOR_DATA_TYPE` (×3) |
| tensor identity | `HIPDNN_ATTR_TENSOR_UNIQUE_ID` (×3) |
| `convDesc.padA` | `HIPDNN_ATTR_CONVOLUTION_PRE_PADDINGS` **and** `..._POST_PADDINGS` (MIOpen padding is symmetric → set pre = post) |
| `convDesc.strideA` | `HIPDNN_ATTR_CONVOLUTION_FILTER_STRIDES` |
| `convDesc.dilationA` | `HIPDNN_ATTR_CONVOLUTION_DILATIONS` |
| `convDesc.c_mode` (`miopenConvolution`) | `HIPDNN_ATTR_CONVOLUTION_CONV_MODE = HIPDNN_CROSS_CORRELATION` |
| `convDesc` spatialDim | length of the padding/stride/dilation arrays |
| (derived) accumulation type | `HIPDNN_ATTR_CONVOLUTION_COMP_TYPE` |
| `x`/`w`/`y` device pointers | variant pack: `HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS` keyed by `..._UNIQUE_IDS` |

**Maps, but not faithfully (observable differences)**

- `algo` (`miopenConvFwdAlgorithm_t`) is **ignored** — hipDNN selects its own
  engine via heuristics. The caller's explicit algorithm choice is not honored.
- `workSpace`/`workSpaceSize` are **not reused** — hipDNN computes and owns its
  own workspace requirement.

**Does not map (gaps today)**

| Case | Reason |
|---|---|
| non-identity `alpha`/`beta` | hipDNN conv-forward op has **no** α/β scaling attribute; only α=1, β=0 is expressible. (Scaling must be expressed as separate `pointwise` nodes.) |
| `groupCount != 1` (grouped/depthwise) | no group-count attribute on the base conv-forward op |
| `c_mode = miopenTranspose` (deconv) | not a forward-conv op (maps to `conv_dgrad`) |
| spatial dim > 5 | prototype boundary / engine limit |
| vectorized layouts (NCHWc*) | no plain dims+strides form |

For **backward** convolution, MIOpen's `miopenConvolutionBackwardData` →
hipDNN `conv_dgrad` and `miopenConvolutionBackwardWeights` → `conv_wgrad`
(frontend), with the same style of translation.

---

## 6. Algorithm selection & autotuning

| Aspect | MIOpen | hipDNN |
|---|---|---|
| Discovery | `miopenFind*Algorithm` (benchmarks, returns sorted `miopenConvAlgoPerf_t[]`) or Immediate mode (`GetSolutionCount`/`GetSolution`/`GetSolutionWorkspaceSize`/`CompileSolution`) | `create_execution_plans(modes)` from engine heuristics; ranked engine configs → execution plans |
| Caller control | caller passes chosen `algo`/solution id to the execute call | caller can filter/pick plans (`deselect_engines`, `deselect_workspace_greater_than`, `set_preferred_engine_id_ext`), otherwise heuristics choice |
| Autotune | Find mode benchmarks online; results cached in Find-DB/perf-DB | `build_plans(BuildPlanPolicy::ALL)` + explicit plan timing; plan can be pinned |
| Persistence | on-disk Find-DB, perf-DB, `.kdb` kernel DB (implicit) | plan **serialize/deserialize** (`to_json`/`to_binary`) — caller owns the cache; no implicit global on-disk tuning DB |
| Compilation | hipRTC/offline kernels cached in `.kdb` | engine plugins compile; hipDNN's MIOpen provider shares MIOpen's kernel cache |

**Migration implication:** XLA today typically drives MIOpen in a
find-then-cache pattern via StreamExecutor's autotuning, persisting a chosen
algorithm per convolution config. Under hipDNN, XLA would instead build/compile
a plan and **serialize the plan** into its own autotune cache (XLA already does
exactly this for cuDNN via `cudnn_frontend` plan serialization, so the pattern
is familiar). The tuning-DB semantics differ: MIOpen's is implicit and global;
hipDNN's is explicit and caller-owned.

---

## 7. Fusion, normalization, attention

- **MIOpen** offers an explicit `miopenFusionPlan` API for a limited set of
  fusions, plus a few fixed fused entry points (e.g. conv+bias+activation) and
  standalone `BatchNormalization*`, `Softmax*`, `Pooling`, `LRN`, `Activation`,
  and `RNN*` calls.
- **hipDNN** fuses natively: any supported subgraph of nodes can be compiled to
  a single plan. Directly relevant to XLA's fused-op paths:
  - `batchnorm` / `batchnorm_inference` / `layernorm` / `rmsnorm`
  - `matmul` (+ `pointwise` epilogues for GEMM fusion via hipBLASLt engine)
  - `pointwise` (1/2/3-input elementwise: scale, add, activation, etc.)
  - `reduction`, `resample` (pooling)
  - `sdpa_backward` (scaled-dot-product / flash attention)

**Migration implication:** XLA's cuDNN backend uses the cuDNN graph API for
norm+pointwise fusion and flash attention. Because hipDNN's frontend mirrors
`cudnn_frontend`, those graph-shaped code paths in XLA port far more directly to
hipDNN than they ever could to MIOpen's flat API. Conversely, **RNN/LSTM/GRU**
(`miopenRNN*`), **LRN**, and **CTC** have no frontend node today and would
remain on MIOpen or need new hipDNN engines.

---

## 8. Handles, streams, error handling, lifecycle

- **Streams:** MIOpen binds a HIP stream to the handle (`miopenCreateWithStream`
  / `miopenSetStream`); hipDNN likewise (`hipdnnSetStream`). Both are per-handle.
  For XLA, one hipDNN handle per StreamExecutor stream mirrors current MIOpen
  usage.
- **Error handling:** MIOpen returns `miopenStatus_t`; hipDNN backend returns
  `hipdnnStatus_t`, and the frontend returns a richer `Error`/`error_t` object
  (error code + message). StreamExecutor's `dnn` layer would translate hipDNN
  errors to `absl::Status` the same way it translates MIOpen today.
- **Resource lifetime (frontend):** the hipDNN frontend is RAII/`shared_ptr`
  based (tensors are `std::shared_ptr<Tensor_attributes>`; the `Graph` owns
  backend descriptors). This is more C++-idiomatic than MIOpen's create/destroy
  descriptor pairs and reduces manual cleanup — but note hipDNN's own coding
  rules stress that backend raw pointers must be RAII-wrapped, so integrators
  using the **backend C API directly** manage descriptor lifetimes explicitly
  (like MIOpen).
- **Thread-safety / statefulness:** hipDNN graphs are build-once/execute-many;
  a compiled+serialized plan is the reusable unit. This fits XLA's
  compile-time-plan / run-time-execute split well.

---

## 9. How XLA consumes MIOpen today, and what changes

XLA's ROCm DNN support lives in StreamExecutor's ROCm backend
(`stream_executor/rocm/rocm_dnn.*`, implementing the `dnn::DnnSupport`
interface). It uses MIOpen for convolution (find/immediate + execute),
batch norm, pooling, LRN, softmax, activation, and — increasingly — graph-style
fused ops (norm, attention) via the generic `dnn` graph abstraction that the
cuDNN backend implements with `cudnn_frontend`.

What a hipDNN migration changes, per subsystem:

| XLA/SE subsystem | Today (MIOpen) | Under hipDNN |
|---|---|---|
| Conv fwd/bwd | `miopenFind*` + `miopenConvolution*` | build `conv_fprop`/`conv_dgrad`/`conv_wgrad` graph, compile plan, execute; α/β/groups need node-level handling |
| Conv autotuning | Find-DB + `miopenConvAlgoPerf_t` | build_plans + **serialize plan** into XLA's autotune cache |
| Batchnorm | `miopenBatchNormalization*` | `graph.batchnorm` / `batchnorm_inference` |
| Fused attention | cuDNN-frontend-shaped path on NV; MIOpen-side varies | `graph.sdpa_*` (close to the cuDNN graph path) |
| Norm/pointwise fusion | limited (`miopenFusionPlan`) | native graph fusion (`layernorm`/`rmsnorm` + `pointwise`) |
| RNN, LRN, CTC | `miopenRNN*`, `miopenLRNForward`, CTC | **no hipDNN frontend node today** — stay on MIOpen |
| Pooling | `miopenPoolingForward` | `graph.resample` |
| Softmax | `miopenSoftmax*` | express via graph (reduction + pointwise) or keep MIOpen |

---

## 10. Migration options

There are three broad strategies, not mutually exclusive.

### Option A — Transitional: ride the MIOpen→hipDNN forwarding wrapper (zero XLA changes)

`0001_HipdnnForwardingWrapper.md` proposes shipping `libMIOpen.so` as a thin
wrapper that can forward selected op families (convolution first) to hipDNN
behind the *unchanged* MIOpen C API, gated by `MIOPEN_HIPDNN_FORWARDING`.

- **Pro:** XLA gets hipDNN-backed execution for forwarded ops with **no code
  change and no rebuild** — it keeps calling `miopenConvolutionForward`. This is
  the lowest-risk way to get early hipDNN coverage and to A/B performance.
- **Con:** It is explicitly temporary and *narrow*: only ops in the compile-time
  forwarding set are affected; α/β≠identity, `groupCount≠1`, vectorized layouts,
  and unsupported shapes fall back to MIOpen; tuning-DB is bypassed for
  forwarded ops; it does not expose hipDNN's fusion/graph capabilities to XLA.

**Recommendation:** use the wrapper as the *bridge* to de-risk correctness and
perf on convolution while XLA's native hipDNN integration is built — not as the
end state.

### Option B — Native integration via the hipDNN **frontend** (recommended end state)

Add a hipDNN-backed implementation of StreamExecutor's `dnn` interface (a
sibling to the MIOpen one) that builds `hipdnn_frontend::graph::Graph` objects,
compiles plans, serializes them into XLA's existing autotune cache, and executes
with a UID→pointer variant pack.

- **Pro:** ergonomic C++ RAII API; native fusion + attention; maps closely to
  XLA's existing `cudnn_frontend` code paths, maximizing code/design reuse;
  plan serialization fits XLA autotuning.
- **Con:** header-only frontend hard-links the backend today (a `dlopen`
  runtime-load mode is a *prerequisite* being added per the forwarding RFC's
  §7 — relevant if XLA wants a soft/optional hipDNN dependency); frontend is
  C++ (fine for XLA).

### Option C — Native integration via the hipDNN **backend** C API

Drive `hipdnnBackend*` descriptors directly (as the wrapper prototype does).

- **Pro:** pure C ABI, no C++ frontend dependency, finest control; a stable
  base-graph API the forwarding wrapper already pins to.
- **Con:** verbose (manual descriptor + attribute plumbing, manual lifetime
  management); you reimplement much of what the frontend already provides.

**Recommendation for XLA:** **Option A now (bridge), Option B as the target.**
Model the hipDNN `dnn` backend on XLA's `cudnn_frontend` integration.

---

## 11. Feature-gap / risk register for XLA

| Item | Status in hipDNN today | Impact on XLA | Mitigation |
|---|---|---|---|
| α/β scaling on conv | Not on base conv op | Conv with non-trivial scaling | Express scale/bias as `pointwise` nodes; else keep on MIOpen |
| Grouped / depthwise conv | No group-count on base conv op | Grouped/depthwise convs | Stay on MIOpen (or wrapper falls back); needs hipDNN engine support |
| Vectorized layouts (NCHWc*) | Not expressible as dims+strides | High-perf int8/packed paths | Keep on MIOpen |
| RNN / LSTM / GRU | No frontend node | XLA RNN ops | Keep on MIOpen |
| LRN, CTC | No frontend node | Legacy models | Keep on MIOpen |
| Explicit algorithm pinning | Heuristics choose; caller filters, not forces `algo` | Reproducibility/determinism | Use plan serialization + engine deselect/pin; validate determinism |
| Maturity | Beta, "not for production" per docs | Stability/coverage risk | Gate rollout per-op; keep MIOpen fallback; lean on wrapper A/B |
| Dependency posture | frontend hard-links backend (dlopen mode WIP) | Optional-dependency builds | Track hipDNN `RUNTIME_LOAD_BACKEND` work |
| Tuning DB semantics | caller-owned plan cache (not implicit global) | Autotune cache format/warm-up | Reuse XLA's cuDNN-frontend-style plan cache |

---

## 12. Recommendations

1. **Bridge with the forwarding wrapper.** Once `0001_HipdnnForwardingWrapper.md`
   Phase 2 lands, run XLA's ROCm conv workloads with
   `MIOPEN_HIPDNN_FORWARDING=enabled` to get hipDNN-backed convolution with zero
   XLA changes and to establish correctness/perf baselines. Keep MIOpen as the
   instant rollback (`MIOPEN_HIPDNN_FORWARDING=disabled`).
2. **Build a native hipDNN `dnn` backend in StreamExecutor**, modeled on the
   cuDNN-frontend integration, starting with conv + the fused ops XLA already
   expresses as graphs (attention, norm+pointwise). Serialize plans into XLA's
   autotune cache.
3. **Keep MIOpen for the gap set** (grouped/depthwise conv, vectorized layouts,
   RNN, LRN, CTC, α/β scaling that can't be re-expressed) until hipDNN engine
   coverage lands. A hybrid backend (hipDNN where supported, MIOpen otherwise)
   is the realistic steady state during the transition.
4. **Drive hipDNN feature requests** from XLA's needs: group-count on conv,
   α/β (or a documented pointwise-scaling recipe), and any missing op nodes.
5. **Track the two hipDNN-side prerequisites** the forwarding RFC identifies —
   the frontend `dlopen`/`RUNTIME_LOAD_BACKEND` mode and the stable base-graph
   API pin — since they equally affect any direct XLA→hipDNN linkage decision.

---

## 13. References

- MIOpen public API: `projects/miopen/include/miopen/miopen.h`
- hipDNN frontend: `projects/hipdnn/frontend/include/hipdnn_frontend.hpp`,
  `.../hipdnn_frontend/Graph.hpp`
- hipDNN backend C API: `projects/hipdnn/backend/include/hipdnn_backend.h`
- hipDNN overview & workflow:
  `projects/hipdnn/docs/user-guides/what-is-hipdnn.rst`,
  `projects/hipdnn/docs/user-guides/how-to/build-execute-hipdnn.rst`
- Existing MIOpen provider (hipDNN→MIOpen translation reference):
  `dnn-providers/miopen-provider/`
- Forwarding wrapper design + worked conv translation:
  `0001_HipdnnForwardingWrapper.md`, `0001a_InvestigationReference.md`
