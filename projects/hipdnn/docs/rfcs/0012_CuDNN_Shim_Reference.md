# RFC 0012 — Supporting Reference: Verified cuDNN FE v1.24 Surface

This document is the long-form companion to
[RFC 0012: cuDNN shim for hipDNN](./0012_CuDNN_Shim.md). It collects the
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
    // Scalar tensor() convenience overloads (int64_t / float / double / etc.)
    // that wrap a pass-by-value constant — also exposed.
    std::shared_ptr<Tensor_attributes> tensor(int64_t value);
    std::shared_ptr<Tensor_attributes> tensor(float value);
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
    // Convenience one-shot build (validate + build_operation_graph +
    // create_execution_plans + check_support + build_plans). MOST upstream
    // samples drive the graph through build(); high priority to support.
    error_t build(cudnnHandle_t, std::vector<HeurMode_t> const&,
                  BuildPlanPolicy_t = BuildPlanPolicy_t::HEURISTICS_CHOICE);

    // Engine / knob query + targeted plan construction
    error_t get_engine_count(int64_t& count);
    error_t get_knobs_for_engine(int64_t engine,
                                 std::vector<Knob>& knobs);
    error_t create_execution_plan(int64_t engine_id,
                                  std::unordered_map<KnobType_t, int64_t> const& knobs);
    Graph&  deselect_engines(std::vector<std::string> const& engine_names);
    // Plan-name introspection
    error_t get_plan_name(std::string&) const;
    error_t get_plan_name_at_index(int64_t index, std::string&) const;
    // Per-plan workspace
    error_t get_workspace_size_plan_at_index(int64_t index, int64_t&) const;
    // Targeted execution
    error_t execute_plan_at_index(cudnnHandle_t,
        std::unordered_map<int64_t, void*>&, void* workspace, int64_t index) const;
    // (shared_ptr-keyed execute_plan_at_index overload also exists)

    // Autotuning / warmup (relates to hipDNN auto-tuning work, PR #7217)
    error_t autotune(cudnnHandle_t,
                     std::unordered_map<int64_t, void*>&, void* workspace);
    error_t warmup(cudnnHandle_t,
                   std::unordered_map<int64_t, void*>&, void* workspace);

    // CUDA-graph capture — no in-scope HIP-graph analogue; Tier-2
    // error-wrappers (compile, fail at use) per RFC §4.3.1.
    error_t populate_cuda_graph(cudnnHandle_t,
        std::unordered_map<int64_t, void*>&, void* workspace, cudaGraph_t&);
    error_t update_cuda_graph(cudnnHandle_t,
        std::unordered_map<int64_t, void*>&, void* workspace, cudaGraph_t&);

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
    // No auxiliary identity map is needed: hipdnn_frontend already honors
    //   set UIDs, auto-assigns unset UIDs uniquely at build time, and ships
    //   a shared_ptr<Tensor_attributes>-keyed execute() overload that lowers
    //   to the UID variant pack internally (see RFC §4.4.1). The Graph
    //   wrapper only holds composition state for non-1:1 surface such as
    //   notes / knobs / heuristics.
};

} // namespace
```

Disposition of the `build()`, engine/knob-query, plan-name, per-index
workspace, `execute_plan_at_index`, `autotune`/`warmup`,
`populate_cuda_graph` / `update_cuda_graph`, `deselect_engines`, and scalar
`tensor()` members above: `build()` and the scalar `tensor()` overloads are
high-priority forwards (samples rely on them); the engine/knob/autotune
surface forwards to hipDNN where supported and ties into the auto-tuning
work (PR #7217, see §4); the CUDA-graph-capture methods are Tier-2
error-wrappers (no in-scope HIP-graph analogue); the rest forward or
error per the three-tier rule.

`Graph` is a Tier-3 composition wrapper (RFC §4.3.1) because its
plan-filtering / notes / knobs / heuristics surface is not 1:1 with hipDNN.
The `*_attributes` classes, by contrast, follow the three-tier rule: those
that are exactly 1:1 with their hipDNN counterpart (most of them — see §2)
are **aliased** with a `using` declaration, not wrapped; only the
non-matching or unsupported ones get a wrapper. Where a Tier-3 wrapper *is*
needed, its setters perform the enum/type translation on the way in and
materialize the inner hipDNN attribute object on calls into hipDNN.

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
| `Layernorm_backward_attributes` | 1154 | ⚠️ **confirm** — hipDNN has `LayernormAttributes` (fwd); whether a distinct LayerNorm-backward attribute exists or is planned needs confirmation (cf. RMSNorm, which has a distinct `RMSNormBackwardAttributes`) before treating this as unsupported. |
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

Disposition follows the three-tier strategy (RFC §4.3.1):

- For each `*_attributes` class that is **exactly 1:1** with its hipDNN
  equivalent, the shim **aliases** the hipDNN type into `<shim_ns>::graph`
  (`using hipdnn_frontend::graph::Conv_fprop_attributes;`) — no wrapper.
  Each row above with a named hipDNN equivalent is an alias candidate
  pending a per-type 1:1 confirmation against
  `frontend/include/hipdnn_frontend/attributes/` (any field/setter mismatch
  demotes that single type to a Tier-3 composition wrapper).
- For each class **without** a hipDNN equivalent (the "(none)" rows), the
  shim ships a Tier-2 error-stub:
  1. Declare the attribute class and `Graph::*` method with the cuDNN FE
     signature, so source compiles.
  2. Surface `error_code_t::GRAPH_NOT_SUPPORTED` at build/validate time
     (node-adding methods return tensors, not `error_t`, so the error is
     recorded on the graph and returned from the next
     `validate()` / `build_operation_graph()` — see RFC §4.4.2), with a
     message pointing the user to file a GitHub issue.
  3. Document the omission in the shim header's Doxygen.

## 3. Verified PyTorch consumer surface

Verified by direct inspection of the PyTorch source tree
([github.com/pytorch/pytorch](https://github.com/pytorch/pytorch),
`aten/src/ATen/native/cudnn/`) against the v9 graph API namespace
(`fe::graph::*`, `cudnn_frontend::graph::*`):

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
  variant pack throughout. (Note: per RFC §4.4.1 there is no tensor-identity
  problem to sidestep — hipDNN handles both keying styles natively — but
  PyTorch's explicit-UID usage is the simplest path regardless.)
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
does not present heuristic modes to the user. Part of this surface is being
added by the hipDNN auto-tuning work (**PR #7217**) — `build`, autotune, and
the engine/knob plumbing — which the shim builds on as it lands. The shim
will:

- Accept all `HeurMode_t` values without error in
  `create_execution_plans()`. **Interim**: map `A`, `B`, `FALLBACK`,
  `OPENSOURCE` all to hipDNN's fallback/default selection, and **WARN on
  first use per process** that the requested mode was not honored. Remap as
  real hipDNN heuristics arrive.

**Per-note triage.** `select_*` / `deselect_*` notes are not blanket
no-ops; each note is classified by how it can be honored. The full table
must be filled in note-by-note before Phase 4, but the handling buckets are:

| Handling | When | Examples |
|----------|------|----------|
| Map to engine capability | hipDNN can express the constraint via engine/knob metadata | deterministic (non-deterministic filter) — wire through now that the note system exists |
| Warn and ignore | advisory note, safe to drop | informational numerical notes with no correctness impact |
| **Error** | correctness-critical and hipDNN cannot honor it | requesting deterministic when it cannot be guaranteed — running anyway is wrong |

The principle: a note that exists to *exclude unsafe plans* must never be
silently dropped — if the shim can't apply the filter, it errors
(recorded-error mechanism, RFC §4.4.2) rather than returning a plan the user
asked to avoid.

**Resource caps.** For `deselect_workspace_greater_than` and
`deselect_shared_mem_greater_than` (both confirmed present in v1.24
upstream):

- The workspace-size cap is enforced post-hoc by filtering out plans whose
  `get_workspace_size()` exceeds the limit before `build_plans()`. The
  workspace *knobs* still need a concrete mapping plan (tracked with the
  auto-tuning work, PR #7217).
- The shared-memory cap requires per-plan shared-memory-usage metadata from
  hipDNN, which is **not** currently exposed. Pending hipDNN exposing that
  metadata, the shim **rejects a non-zero
  `deselect_shared_mem_greater_than`** (errors via the recorded-error
  mechanism) rather than silently no-op'ing — silently ignoring a resource
  cap could let an over-budget plan run. A zero argument is a no-op.

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
a structural problem:

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

The stub `cudnn.h` covers the C-API **types** the v9 method signatures
reference:

- `cudnnHandle_t` — aliased: `using cudnnHandle_t =
  hipdnn_frontend::hipdnnHandle_t;`
- `cudnnStatus_t` enum (mapped to `hipdnnStatus_t`)
- `cudnnDataType_t`, `cudnnTensorFormat_t`, `cudnnConvolutionMode_t`,
  `cudnnReduceTensorOp_t`, `cudnnNormFwdPhase_t`, `cudnnBackendHeurMode_t`,
  `cudnnBackendNumericalNote_t`, `cudnnBackendBehaviorNote_t`,
  `cudnnBackendDescriptorType_t`

It additionally ships a small set of C **entry points**, each forwarding to
the hipDNN equivalent — required for init / error-handling / version checks
outside the frontend and to build the mirrored samples (§8.3):

- `cudnnCreate`, `cudnnDestroy`, `cudnnSetStream`, `cudnnGetStream`
- `cudnnGetErrorString`
- `cudnnGetVersion`
- `create_cudnn_handle()` helper (mirrors cuDNN FE's convenience helper)

The remainder of the cuDNN C library (convolution descriptor APIs, etc.)
stays out of scope. Open implementation item: confirm which of the above
hipDNN C-API entry points already exist and add any that are missing.

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
