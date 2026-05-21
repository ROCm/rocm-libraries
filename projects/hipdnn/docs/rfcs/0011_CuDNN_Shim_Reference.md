# RFC 0011 — Supporting Reference: Verified cuDNN FE v1.24 Surface

This document is the long-form companion to
[RFC 0011: cuDNN shim for hipDNN](./0011_CuDNN_Shim.md). It collects the
verbose, source-verified material that the main RFC summarises: the full
`Graph` C++ signature, the complete `*_attributes` coverage table, the
PyTorch consumer surface, the heuristics/plan API enumeration, the error and
logging API surface, and the umbrella-header `<cudnn.h>` / v0.x problem.

All citations are against the local copy of
[NVIDIA/cudnn-frontend](https://github.com/NVIDIA/cudnn-frontend) at tag
**v1.24.0** (`CUDNN_FRONTEND_VERSION = 12400`, verified in
`include/cudnn_frontend_version.h`).

## Table of Contents

1. [`Graph` class — verified signature](#1-graph-class--verified-signature)
2. [Full `*_attributes` coverage table](#2-full-_attributes-coverage-table)
3. [Verified PyTorch consumer surface](#3-verified-pytorch-consumer-surface)
4. [Heuristics and plan selection — verified API](#4-heuristics-and-plan-selection--verified-api)
5. [Error handling and logging — verified API](#5-error-handling-and-logging--verified-api)
6. [The `<cudnn.h>` / v0.x umbrella-header problem](#6-the-cudnnh--v0x-umbrella-header-problem)

---

## 1. `Graph` class — verified signature

The principal class is `<shim_ns>::graph::Graph`. It contains a
`hipdnn_frontend::graph::Graph` by value and forwards calls. The shape
below is the upstream `cudnn_frontend::graph::Graph` in v1.24, with line
numbers from `include/cudnn_frontend/graph_interface.h`.

```cpp
namespace hipdnn_frontend::compatibility::cudnn_frontend::graph {

// Signatures verified against include/cudnn_frontend/graph_interface.h.
class Graph {
public:
    // Construction / graph-level attributes
    Graph();
    Graph& set_io_data_type(DataType_t);                  // :1714
    Graph& set_intermediate_data_type(DataType_t);        // :1712
    Graph& set_compute_data_type(DataType_t);             // :1716
    Graph& set_dynamic_shape_enabled(bool);               // :1718
    Graph& set_override_shape_enabled(bool);              // :1720
    Graph& set_sm_count(int32_t);                         // :1722
    Graph& set_sm_version(int32_t);                       // :1724
    Graph& set_kernel_cache(std::shared_ptr<KernelCache>);// :1726
    Graph& set_device_properties(std::shared_ptr<const DeviceProperties>); // :1728
    Graph& set_name(std::string const&);                  // :1731

    // Tensor creation
    std::shared_ptr<Tensor_attributes> tensor(Tensor_attributes const&);
    std::shared_ptr<Tensor_attributes> tensor_like(
        std::shared_ptr<Tensor_attributes> const&, std::string const& name = "");
    error_t query_tensor_attributes_of_uid(
        int64_t uid, Tensor_attributes&) const;           // :1737

    // Node-adding methods — one per cuDNN FE node-attribute type.
    // Verified upstream: every node method takes its *_attributes
    // parameter BY VALUE (not by reference) — e.g.:
    //   conv_fprop(shared_ptr<Tensor_attributes>,
    //              shared_ptr<Tensor_attributes>, Conv_fprop_attributes)
    //     -> shared_ptr<Tensor_attributes>                  (:1799)
    //   matmul(...,Matmul_attributes) -> shared_ptr<Tensor_attributes>
    //   rmsnorm(...,Rmsnorm_attributes) -> array<...,2>     (:1839)
    //   batchnorm(...,Batchnorm_attributes) -> array<...,5> (:1779)
    //   layernorm(...,Layernorm_attributes) -> array<...,3> (:1764)
    //   sdpa(Q,K,V, SDPA_attributes) -> array<...,2>{O,Stats}   (:1857)
    //   sdpa_backward(Q,K,V,O,dO,Stats, SDPA_backward_attributes)
    //     -> array<...,3>{dQ,dK,dV}                         (:1923)
    //   slice(t, Slice_attributes) -> shared_ptr<Tensor_attributes>
    // The shim must match by-value (not by-ref) signatures exactly.

    // Plan lifecycle (all return error_t = error_object)
    error_t validate();
    error_t build_operation_graph(cudnnHandle_t);
    error_t create_execution_plans(std::vector<HeurMode_t> const&);
    error_t check_support(cudnnHandle_t);                 // overload taking handle
    error_t check_support();                              // no-arg overload also exists
    error_t build_plans(cudnnHandle_t,
                        BuildPlanPolicy_t = BuildPlanPolicy_t::HEURISTICS_CHOICE,
                        bool do_multithreaded_builds = false);  // :2218
    error_t build_plans(BuildPlanPolicy_t = BuildPlanPolicy_t::HEURISTICS_CHOICE,
                        bool do_multithreaded_builds = false);  // :2229
    error_t build_plan_at_index(int64_t index);           // :2242
    int64_t get_execution_plan_count() const;             // :1982

    // Plan filtering (chained — return *this)
    Graph& select_numeric_notes(std::vector<NumericalNote_t> const&);   // :2272
    Graph& deselect_numeric_notes(std::vector<NumericalNote_t> const&); // :2290
    Graph& select_behavior_notes(std::vector<BehaviorNote_t> const&);   // :2263
    Graph& deselect_behavior_notes(std::vector<BehaviorNote_t> const&); // :2281
    Graph& deselect_workspace_greater_than(int64_t workspace);          // :2245
    Graph& deselect_shared_mem_greater_than(int64_t shared_mem);        // :2251

    // Per-plan introspection
    error_t get_behavior_notes(std::vector<BehaviorNote_t>&) const;     // :2302
    error_t get_behavior_notes_for_plan_at_index(
        int64_t index, std::vector<BehaviorNote_t>&) const;             // :2299

    // Execution — FOUR overloads, all take the variant pack by NON-const
    // reference (upstream signatures, verified at :1278-1314):
    error_t execute(cudnnHandle_t,
                    std::unordered_map<std::shared_ptr<Tensor_attributes>, void*>& vp,
                    void* workspace) const;
    error_t execute(cudnnHandle_t,
                    std::unordered_map<int64_t /*uid*/, void*>& vp,
                    void* workspace) const;
    error_t execute(cudnnHandle_t,
                    std::unordered_map<int64_t, void*>& vp,
                    void* workspace,
                    std::vector<int64_t> const& override_uids,
                    std::vector<std::vector<int64_t>> const& override_shapes,
                    std::vector<std::vector<int64_t>> const& override_strides) const;
    error_t execute(cudnnHandle_t,
                    void** sorted_user_ptrs, int n_user, void* workspace) const;
    // (plus execute_plan_at_index overloads with the same key-type variants)

    // Introspection
    int64_t get_workspace_size() const;
    error_t get_workspace_size(int64_t&) const;
    error_t serialize(std::vector<uint8_t>&) const;
    error_t deserialize(cudnnHandle_t, std::vector<uint8_t> const&);

private:
    hipdnn_frontend::graph::Graph _inner;
    // … any auxiliary maps needed to preserve cuDNN-FE-style identity for
    //   shared_ptr<Tensor_attributes> across calls (see RFC §4.4.1).
};

} // namespace
```

Each `*_attributes` class is a thin wrapper holding the corresponding hipDNN
attribute object plus any cuDNN-only fields that hipDNN doesn't model
natively. The wrapper's setters perform the enum/type translation on the way
in; on calls into hipDNN, the wrapper materializes the inner hipDNN attribute
object.

## 2. Full `*_attributes` coverage table

**Full set of v9 `*_attributes` classes** in upstream
`include/cudnn_frontend/graph_properties.h` (v1.24.0, verified):

| Class | Line | hipDNN equivalent in `frontend/include/hipdnn_frontend/attributes/` |
|-------|------|----|
| `BN_finalize_attributes` | 656 | (none — bn-finalize) |
| `Genstats_attributes` | 690 | (none) |
| `Conv_fprop_attributes` | 704 | `ConvolutionFpropAttributes` |
| `Batchnorm_backward_attributes` | 795 | `BatchnormBackwardAttributes` |
| `DBN_weight_attributes` | 824 | (none) |
| `Conv_dgrad_attributes` | 837 | `ConvolutionDgradAttributes` |
| `Matmul_fp8_attributes` | 929 | (none — FP8 matmul) |
| `Matmul_attributes` | 973 | `MatmulAttributes` |
| `Pointwise_attributes` | 1032 | `PointwiseAttributes` |
| `Instancenorm_backward_attributes` | 1133 | (none) |
| `Layernorm_backward_attributes` | 1154 | (none — bwd combined with fwd in hipDNN?) |
| `Layernorm_attributes` | 1175 | `LayernormAttributes` |
| `AdaLayernorm_attributes` | 1208 | (none — adaptive layernorm) |
| `AdaLayernorm_backward_attributes` | 1235 | (none) |
| `Instancenorm_attributes` | 1256 | (none) |
| `Batchnorm_attributes` | 1283 | `BatchnormAttributes` |
| `Batchnorm_inference_attributes` | 1320 | `BatchnormInferenceAttributes` / `BatchnormInferenceAttributesVarianceExt` |
| `Reduction_attributes` | 1333 | `ReductionAttributes` |
| `Rng_attributes` | 1377 | (none — RNG as a graph op) |
| `Resample_attributes` | 1459 | `ResampleFwdAttributes` (fwd only) |
| `Reshape_attributes` | 1579 | (none) |
| `Transpose_attributes` | 1636 | (none) |
| `Rmsnorm_attributes` | 1673 | `RMSNormAttributes` |
| `RoPE_attributes` | 1706 | (none) |
| `RoPE_backward_attributes` | 1742 | (none) |
| `Rmsnorm_backward_attributes` | 1780 | `RMSNormBackwardAttributes` |
| `SDPA_attributes` | 1922 | `SdpaAttributes` (gated by `HIPDNN_ENABLE_SDPA`) |
| `SDPA_backward_attributes` | 2253 | `SdpaBackwardAttributes` |
| `SDPA_fp8_backward_attributes` | 2493 | (none — FP8 SDPA) |
| `Softmax_attributes` | 2701 | (none — Softmax as a top-level op) |
| `DiagonalBandMask_attributes` | 2730 | (none) |
| `Conv_wgrad_attributes` | 2761 | `ConvolutionWgradAttributes` |
| `Slice_attributes` | 2853 | (none) |
| `PagedCacheLoad_attributes` | 2898 | (none) |
| `Block_scale_quantize_attributes` | 2911 | `BlockScaleQuantizeAttributes` |
| `Block_scale_dequantize_attributes` | 2952 | `BlockScaleDequantizeAttributes` |
| `Concatenate_attributes` | 3026 | (none) |
| `Moe_grouped_matmul_attributes` | 3053 | (none) |
| `Moe_grouped_matmul_bwd_attributes` | 3082 | (none) |

(Plus `CustomOpAttributes` on the hipDNN side, which has no cuDNN
equivalent.)

For each `*_attributes` class with a hipDNN equivalent, the shim provides a
wrapper that forwards setters and materializes the inner hipDNN attribute
object. For each class **without** a hipDNN equivalent, the shim will:

1. Declare the attribute class and `Graph::*` method with the cuDNN FE
   signature, so source compiles.
2. Return `error_code_t::GRAPH_NOT_SUPPORTED` at runtime when the user
   attempts to add the node to a graph.
3. Document the omission in the shim header's Doxygen.

## 3. Verified PyTorch consumer surface

Verified by direct inspection of `D:\develop\src\pytorch` against the v9
graph API namespace (`fe::graph::*`, `cudnn_frontend::graph::*`):

- **`aten/src/ATen/native/cudnn/MHA.cpp` is the only file in PyTorch that
  uses the v9 graph API.** All other cuDNN consumers in PyTorch
  (`Conv_v8.cpp`, `aten/src/ATen/native/quantized/cudnn/*`) use the v0.x /
  v8 backend descriptor API (`TensorBuilder`, `OperationBuilder`,
  `ExecutionPlanBuilder`, `VariantPackBuilder`, `EngineConfigGenerator`,
  `hasNumericalNote`, etc.) which is explicitly out of scope per the main
  RFC §1 "Not".

The exact v9 surface used by `MHA.cpp` (this is the entire ask for a
PyTorch-driven Phase 1):

| Category | Symbols used |
|----------|--------------|
| Node ops on `Graph` | `sdpa(Q,K,V,opts) → [O,Stats]`, `sdpa_backward(Q,K,V,O,dO,Stats,opts) → [dQ,dK,dV]` |
| `Graph` lifecycle | ctor, `set_io_data_type`, `set_intermediate_data_type`, `set_compute_data_type`, `tensor()`, `validate()`, `build_operation_graph()`, `create_execution_plans({HeurMode_t::A})`, `check_support()`, `build_plans()`, `get_workspace_size()`, `execute(handle, map<int64_t,void*>, workspace)` |
| `Tensor_attributes` setters | `set_uid`, `set_name`, `set_dim`, `set_stride`, `set_data_type`, `set_is_pass_by_value`, `set_output`, `set_ragged_offset` |
| `SDPA_attributes` setters | `set_name`, `set_is_inference` (FE ≤ 11200; **marked `[[deprecated]]` in v1.24** at `graph_properties.h:2046`, forwards to `set_generate_stats(!value)`), `set_generate_stats` (FE > 11200; `:2029`), `set_causal_mask` (`:2116`), `set_attn_scale(float)` (`:2058`; PyTorch uses the float overload, not the `shared_ptr<Tensor_attributes>` one at `:2052`), `set_seq_len_q` (`:2095`), `set_seq_len_kv`, `set_padding_mask` (`:2082`), `set_dropout(float, shared_ptr<Tensor_attributes>, shared_ptr<Tensor_attributes>)` (`:2162`; the other 2-arg `set_dropout(mask, scale)` overload at `:2172` is also exposed but PyTorch doesn't use it), `set_bias` (`:2064`) |
| `SDPA_backward_attributes` setters | `set_name`, `set_causal_mask`, `set_attn_scale`, `set_seq_len_q`, `set_seq_len_kv`, `set_padding_mask`, `set_dropout(p, seed, offset)`, `set_bias` |
| Enum values | `DataType_t::{HALF, BFLOAT16, FLOAT, INT32, INT64}`, `HeurMode_t::A` |
| Macro | `CUDNN_FRONTEND_VERSION` (compared against `11200`) |

Notably PyTorch's MHA path does **not** use:
- Any of the plan-filter methods (`select_*` / `deselect_*` notes,
  `deselect_workspace_greater_than`, `deselect_shared_mem_greater_than`)
- `NumericalNote_t` / `BehaviorNote_t` or `get_behavior_notes*`
- Serialize / deserialize on `Graph`
- The `shared_ptr<Tensor_attributes>`-keyed `execute()` overload — PyTorch
  assigns UIDs explicitly and uses the `unordered_map<int64_t, void*>`
  variant pack throughout. This sidesteps the tensor-identity problem
  (RFC §4.4.1) entirely for PyTorch.
- Any of `cudnn_frontend::graph::{Conv_fprop,Conv_dgrad,Conv_wgrad,Matmul,
  Pointwise,Batchnorm,Layernorm,RMSNorm,Reduction,Resample,Slice,Reshape,
  Rng,Softmax}_attributes` or their `Graph::*` methods

## 4. Heuristics and plan selection — verified API

What exists on `Graph` in v9 (`include/cudnn_frontend/graph_interface.h`):

- `error_t create_execution_plans(std::vector<HeurMode_t> const&)`
- `int64_t get_execution_plan_count() const`
- `error_t check_support(cudnnHandle_t)` (and a no-arg overload)
- `error_t build_plans(cudnnHandle_t, BuildPlanPolicy_t, ...)`
- Filter methods returning `Graph&` for chaining:
  - `select_numeric_notes(std::vector<NumericalNote_t> const&)` (`graph_interface.h:2272`)
  - `deselect_numeric_notes(std::vector<NumericalNote_t> const&)` (`:2290`)
  - `select_behavior_notes(std::vector<BehaviorNote_t> const&)` (`:2263`)
  - `deselect_behavior_notes(std::vector<BehaviorNote_t> const&)` (`:2281`)
  - `deselect_workspace_greater_than(int64_t workspace)` (`:2245`) — internally `plans.set_max_workspace_allowed(workspace)`
  - `deselect_shared_mem_greater_than(int64_t shared_mem)` (`:2251`) — internally `plans.set_max_shared_mem_allowed(shared_mem)`
- Per-plan metadata:
  - `error_t get_behavior_notes(std::vector<BehaviorNote_t>&) const`
  - `error_t get_behavior_notes_for_plan_at_index(int64_t, std::vector<BehaviorNote_t>&) const`

Internally these forward to a `plans` member of type `Execution_plan_list`
(defined in `include/cudnn_frontend/plans.h`), but that type is not part of
the user-facing API.

hipDNN's current selection model (see RFC 0007) is engine-knob driven and
does not present heuristic modes to the user. The shim will:

- Accept all `HeurMode_t` values without error in
  `create_execution_plans()`. Map `A`, `B`, `FALLBACK`, `OPENSOURCE` to
  hipDNN's default engine-selection behavior.
- Log (at `Logging::DEBUG`) the requested mode and the actual hipDNN
  behavior, so users debugging a perf regression can see that their
  heuristic choice was effectively ignored.

For `deselect_workspace_greater_than` and `deselect_shared_mem_greater_than`
(both confirmed present in v1.24 upstream), the shim must wrap them in a
form that's enforceable against hipDNN's plan list:

- The workspace-size cap can be enforced post-hoc by filtering out plans
  whose `get_workspace_size()` exceeds the limit before `build_plans()`.
- The shared-memory cap requires per-plan shared-memory-usage metadata from
  hipDNN, which is **not** currently exposed.

⚠️ **OPEN QUESTION**: Does
`hipdnn_frontend::graph::Graph::create_execution_plans()` expose per-plan
shared-memory-usage metadata? If not, the
`deselect_shared_mem_greater_than` filter has the same hipDNN-side
extension dependency as the note filters above; pending that, it must
either no-op (with a debug log) or unconditionally reject if `shared_mem`
is non-zero.

## 5. Error handling and logging — verified API

cuDNN FE's `error_object` (verified at `graph_helpers.h:55`) is a `struct`
with public fields `code: error_code_t` and `err_msg: std::string`, plus
methods `get_code()`, `get_message()`, `is_good()`, `is_bad()`,
`operator==(error_code_t)`, `operator!=(error_code_t)`. `error_t` is a
typedef for `error_object` in the same header. hipDNN's `Error` (in
`hipdnn_frontend/Error.hpp`) is structurally similar. Mapping:

- The shim's `error_object` is a transparent wrapper around
  `hipdnn_frontend::Error`, preserving the full public surface name-for-name
  (both fields and methods). `error_t` is exposed as a typedef.
- The shim's `error_code_t` is an `enum class` mirroring cuDNN FE's set of
  16 codes (full list in main RFC §4.3). `detail/status_translation.hpp`
  provides `to_hipdnn(error_code_t) -> hipdnn_frontend::ErrorCode` and the
  reverse.
- Codes with no exact hipDNN counterpart (e.g., `NVRTC_COMPILATION_FAILED`,
  `INVALID_CUDA_DEVICE`) collapse to the nearest hipDNN value on the way
  down, and synthesize the cuDNN code on the way up if we can distinguish
  from context.
- The shim must also provide the cuDNN FE error-handling macros:
  `CHECK_CUDNN_FRONTEND_ERROR(x)`, `RETURN_CUDNN_FRONTEND_ERROR_IF(...)`,
  `CUDNN_FE_LOG`, `CUDNN_FE_LOG_LABEL`, `CUDNN_FE_LOG_LABEL_ENDL`,
  `CUDNN_FE_LOG_BANNER` (all defined in upstream `graph_helpers.h:100+`).
  PyTorch's `MHA.cpp` uses `AT_CUDNN_FRONTEND_CHECK` which expands to
  `CHECK_CUDNN_FRONTEND_ERROR`-equivalent logic.

Logging (verified in `include/cudnn_frontend_Logging.h`):

- **Env vars**: `CUDNN_FRONTEND_LOG_INFO` is an integer log level (default
  `0` = disabled; non-zero = enabled). `CUDNN_FRONTEND_LOG_FILE` is a file
  path, or the literal `"stdout"` / `"stderr"`.
- **Defaults**: logging is **OFF** by default (`getLogLevel()` returns 0 if
  the env var is unset).
- **Compile-time gate**: `NV_CUDNN_FRONTEND_DISABLE_LOGGING` forces logging
  to always-off.
- **Public API**: `cudnn_frontend::getLogLevel()`, `isLoggingEnabled()`,
  `isLoggingTensorDumpEnabled()`, `getStream()`, plus the macros listed
  above.

The shim will provide identically-named env vars, compile gate, free
functions, and macros, all delegating to `hipdnn_frontend::Logging`. The
defaults match upstream (logging off) — no decision needed.

## 6. The `<cudnn.h>` / v0.x umbrella-header problem

The shim provides **source-level** compatibility for cuDNN frontend v9 only.
Explicitly out of scope:

- **ABI compatibility**: a `.so`/`.dll` built against real cuDNN cannot be
  swapped for the shim; user code must recompile.

However, the upstream `cudnn_frontend.h` (verified at lines 106–152) creates
a structural problem the original RFC understated:

```cpp
// Verbatim from upstream include/cudnn_frontend.h:
#include <cudnn.h>                                            // line 106
#include "cudnn_frontend_ConvDesc.h"                          // v0.x/v8 desc
#include "cudnn_frontend_Heuristics.h"                        // v0.x/v8 desc
#include "cudnn_frontend_Engine.h"                            // v0.x/v8 desc
// ... 14 more v0.x/v8 headers ...
#include "cudnn_frontend/graph_interface.h"                   // v9 graph API
// ...
namespace cudnn_frontend {
using ConvDesc                  = ConvDesc_v8;                // line 138
using ConvDescBuilder           = ConvDescBuilder_v8;
using EngineHeuristicsBuilder   = EngineHeuristicsBuilder_v8;
// ... 11 more v0.x type aliases ...
}
```

Consequences:

1. **`<cudnn.h>` is unconditionally included** by the upstream umbrella
   header, which is where the C-API types `cudnnHandle_t`,
   `cudnnStatus_t`, `cudnnDataType_t`, `cudnnTensorFormat_t`,
   `cudnnBackend*_t`, etc. actually come from. Because the v9 graph API
   uses these types in its own signatures (`Graph::execute(cudnnHandle_t,
   ...)`, etc.), the shim *must* provide a stub `cudnn.h` to support v9
   TUs at all.
2. **The v0.x/v8 type names (`ConvDesc`, `ConvDescBuilder`,
   `EngineHeuristics`, `EngineConfig`, `Operation`, `Tensor`, ...) are
   exposed in `namespace cudnn_frontend` whether the consumer uses them
   or not** by the upstream umbrella. Any consumer who `#include`s the
   *upstream* umbrella gets these names. PyTorch's `Conv_v8.cpp` and the
   quantized cuDNN ops actively use them.

### Decision in this RFC

Ship a **v9-only umbrella header** that does not mirror upstream's
include-everything pattern. The shim's `cudnn_frontend.h` is hand-curated:
it pulls in only the v9 graph API surface, the FE-namespace enums, the
error / logging machinery, and the stub `cudnn.h` for C-API types.

The stub `cudnn.h` covers only what v9 method signatures reference:

- `cudnnHandle_t` (typedef'd to a hipdnn-derived opaque pointer)
- `cudnnStatus_t` enum (mapped to `hipdnnStatus_t`)
- `cudnnDataType_t`, `cudnnTensorFormat_t`, `cudnnConvolutionMode_t`,
  `cudnnReduceTensorOp_t`, `cudnnNormFwdPhase_t`, `cudnnBackendHeurMode_t`,
  `cudnnBackendNumericalNote_t`, `cudnnBackendBehaviorNote_t`,
  `cudnnBackendDescriptorType_t`

If implementation finds that any v9 surface item refers to a v0.x C++
type by name (e.g., `cudnn_frontend::Tensor` appearing in some
nominally-public-but-internal `INode` signatures), the shim provides a
minimal stub for that specific type only. The full v0.x C++ surface
is **out of scope** for this RFC (main RFC §1, §9).

### Considered-but-deferred options for v0.x source compatibility

These were considered while authoring this RFC and were all deferred to
the follow-up "v0.x compile-only stub layer" RFC enumerated in main
RFC §9:

- **(A) Compile-only stubs for v0.x types** in this RFC. Declare every
  v0.x class/builder in the shim's umbrella with the same signatures,
  with method bodies that return `error_code_t::GRAPH_NOT_SUPPORTED` at
  runtime. Lets `Conv_v8.cpp` *compile* against the shim; convolution
  path is effectively disabled at runtime until consumers are rewritten
  to v9 or a fuller v0.x shim is built. **Deferred** because the v0.x
  surface PyTorch alone uses is significant (13+ builder classes plus
  free functions); the cost is comparable to the v9 wrapper itself and
  this RFC needs to ship.
- **(B) Require the consumer to gate v0.x usage**. Ship a v9-only
  umbrella (the chosen approach above); require consumers to keep v0.x
  TUs out of the build via their existing platform-gating mechanism
  (e.g., excluding `Conv_v8.cpp` from the ROCm build of PyTorch).
  Increases the friction of the "textual hipify" workflow for projects
  that mix v9 and v0.x. **This is what this RFC effectively does**, but
  framed as a non-goal rather than a workflow we recommend — consumers
  who want to mix in the same tree should wait for the follow-up RFC.
- **(C) Split the umbrella**. Ship `<cudnn_frontend.h>` (v9-only) and a
  separate `<cudnn_frontend_v8.h>` (stubs). Functionally the same as
  (A) but with explicit consumer opt-in for the v0.x stub surface.
  **Deferred** with (A) — the difference matters only when (A) ships.
