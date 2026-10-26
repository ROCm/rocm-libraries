# cuDNN frontend compatibility shim

Header-only layer that exposes the NVIDIA cuDNN frontend (FE) v9 graph API on top of hipDNN. Source written against cuDNN FE v1.24.0 compiles against it after an include swap and a namespace alias.

- Source compatible, not ABI compatible. Consumers recompile.
- v9 graph API only. The v0.x / v8 builder and backend-descriptor API is not provided.
- C++ only. No Python bindings.

NVIDIA, CUDA, and cuDNN are trademarks of NVIDIA Corporation. hipDNN is not affiliated with or endorsed by NVIDIA.

## Contents

- [Install](#install)
- [Hipify a cuDNN FE program](#hipify-a-cudnn-fe-program)
- [Known divergences](#known-divergences)
- [Unsupported nodes and APIs](#unsupported-nodes-and-apis)
- [Troubleshooting](#troubleshooting)

## Install

Build hipDNN with the shim enabled:

```bash
cmake -DHIPDNN_ENABLE_CUDNN_COMPATIBILITY=ON ...
```

The headers install to `<prefix>/include/hipdnn/frontend/hipdnn_compatibility/cudnn/`. There is no separate library; link `hipdnn_frontend`:

```cmake
find_package(hipdnn_frontend CONFIG REQUIRED COMPONENTS cudnn_compatibility)
target_link_libraries(my_app PRIVATE hipdnn_frontend)
```

`COMPONENTS cudnn_compatibility` fails configuration if hipDNN was built without the shim.

`Graph::sdpa` and `Graph::sdpa_backward` exist only when hipDNN is also built with `-DHIPDNN_ENABLE_SDPA=ON`. The `hipdnn_frontend` target propagates that define.

## Hipify a cuDNN FE program

1. Replace the cuDNN FE include and alias the namespace once per translation unit:

   ```cpp
   // #include <cudnn_frontend.h>
   #include <hipdnn_compatibility/cudnn/cudnn_frontend.h>
   namespace cudnn_frontend = hipdnn_frontend::compatibility::cudnn_frontend;
   ```

   All other cuDNN FE code stays as written: `cudnn_frontend::graph::Graph`, `namespace fe = cudnn_frontend;`, `cudnnHandle_t`, `cudnnCreate`, `cudnnDestroy`, `cudnnSetStream`, `create_cudnn_handle()`.

2. Convert CUDA runtime calls to HIP (`cudaMalloc` → `hipMalloc`, `cudaStream_t` → `hipStream_t`, `<cuda_runtime.h>` → `<hip/hip_runtime.h>`, ...). The shim does not map CUDA runtime symbols. Keep cuDNN identifiers unchanged; the shim expects cuDNN spellings.

   `hipify-perl` also rewrites cuDNN identifiers (`cudnnCreate` → `hipdnnCreate`, `cudnnHandle_t` → `hipdnnHandle_t`, `CUDNN_STATUS_*` → `HIPDNN_STATUS_*`, `<cudnn.h>` → `<hipDNN.h>`) and has no option to skip them. Either convert the CUDA calls by hand, or after running `hipify-perl` revert every `cudnn*` → `hipdnn*` and `CUDNN_*` → `HIPDNN_*` rename and remove any `<hipDNN.h>` include, which hipDNN does not ship. A renamed `hipdnnCreate` bypasses the shim's `cudnnCreate`, so `CUDNN_FRONTEND_LOG_*` variables are not mapped.

3. Link `hipdnn_frontend` (see [Install](#install)).

`cudnnSetStream` takes a `hipStream_t`.

Single-source builds select the header with their own platform macro. Code that must differ can test `HIPDNN_COMPATIBILITY_CUDNN_FRONTEND`, which the shim headers define:

```cpp
#if defined(USE_ROCM)
#include <hipdnn_compatibility/cudnn/cudnn_frontend.h>
namespace cudnn_frontend = hipdnn_frontend::compatibility::cudnn_frontend;
#else
#include <cudnn_frontend.h>
#endif
```

Do not include NVIDIA's `cudnn.h` or `cudnn_frontend.h` in the same translation unit: the shim declares `cudnnHandle_t`, `cudnnStatus_t`, and the `cudnn*` entry points itself.

[`samples/cudnn_shim/HipifyConvFprop.cpp`](../samples/cudnn_shim/HipifyConvFprop.cpp) is a complete hipified convolution program. Each line changed from the cuDNN original is marked `// cuDNN: <original line>`. The other samples in [`samples/cudnn_shim/`](../samples/cudnn_shim/) build and validate one node each.

## Known divergences

Details: [KNOWN_DIVERGENCES.md](./KNOWN_DIVERGENCES.md).

| Area | Shim behavior |
|---|---|
| Heuristic modes | Every `HeurMode_t` is accepted. Non-`FALLBACK` modes log a warning and use hipDNN's default engine selection. |
| Numerical-note filters | Mostly ignored with a warning. `deselect_numeric_notes(NONDETERMINISTIC)` keeps only engines that claim determinism. `deselect_numeric_notes` with `DOWN_CONVERT_INPUTS` or `REDUCED_PRECISION_REDUCTION`, and `select_numeric_notes(STRICT_NAN_PROP)`, record `GRAPH_NOT_SUPPORTED`. |
| Behavior-note filters | hipDNN notes filter engines. cuDNN-only notes: `deselect` is ignored with a warning, `select` records `GRAPH_NOT_SUPPORTED`. |
| Engine filters | Applied after plan creation. `build_plans(HEURISTICS_CHOICE)` builds the top-ranked surviving plan; if none survives, plan creation, `check_support()`, and plan builds return `GRAPH_NOT_SUPPORTED`. |
| Engine indices | Dense graph-local indices mapped to hipDNN engines, in hipDNN's ranking order. |
| Knobs | Only integer hipDNN knobs that map to a `KnobType_t` are reported. A knob choice the target engine cannot take returns `INVALID_VALUE`. |
| `deselect_shared_mem_greater_than` | Non-zero limit records `GRAPH_NOT_SUPPORTED`. |
| `set_sm_count`, `set_sm_version`, `set_device_properties` | Record `INVALID_VALUE`, reported by the next `validate()` or build call. |
| `set_dynamic_shape_enabled`, `set_kernel_cache` | Logged and ignored. |
| `build_operation_graph()` without a handle | Fails for a non-empty graph. |
| `populate_cuda_graph`, `update_cuda_graph` | Return `GRAPH_NOT_SUPPORTED`. |
| `execute(handle, void**, int, void*)` | Returns `INVALID_VALUE`. Use the UID-map overload. |
| `warmup` | Takes the execute arguments and runs plan 0 once. cuDNN FE takes only the handle. |
| Runtime shape overrides | Workspace queries with non-empty overrides return `GRAPH_NOT_SUPPORTED`. `execute` with overrides requires an SDPA-enabled build. |
| `is_supported_ext` | hipDNN extension with no cuDNN FE counterpart. |
| Logging | `cudnnCreate` maps `CUDNN_FRONTEND_LOG_INFO` / `CUDNN_FRONTEND_LOG_FILE` to `HIPDNN_LOG_LEVEL` / `HIPDNN_LOG_FILE`. Any cuDNN FE logging variable overrides `HIPDNN_LOG_*`. |
| Versions | `CUDNN_FRONTEND_VERSION` is 12400 (v1.24.0). `CUDNN_VERSION` / `cudnnGetVersion()` report 92200. `cudnnGetCudartVersion()` reports 12000. |

## Unsupported nodes and APIs

These `Graph` node methods compile but record `GRAPH_NOT_SUPPORTED`, reported by the next `validate()` or `build_operation_graph()`. To request one, open an issue at https://github.com/ROCm/rocm-libraries/issues.

`adalayernorm`, `adalayernorm_backward`, `bn_finalize`, `concatenate`, `dbn_weight`, `diagonal_band_mask`, `genstats`, `instancenorm`, `instancenorm_backward`, `reshape`, `rng`, `rope`, `rope_backward`, `sdpa_fp8` (FP8 and MXFP8), `sdpa_fp8_backward` (FP8 and MXFP8), `slice`, `transpose`.

Their `*_attributes` classes exist so the code compiles. `Matmul_fp8_attributes`, `Softmax_attributes`, and `PagedCacheLoad_attributes` are declared for the same reason.

Supported nodes forward to hipDNN: `batchnorm`, `batchnorm_backward`, `batchnorm_inference`, `block_scale_dequantize`, `block_scale_quantize`, `conv_dgrad`, `conv_fprop`, `conv_wgrad`, `layernorm`, `layernorm_backward`, `matmul`, `moe_grouped_matmul`, `moe_grouped_matmul_bwd`, `pointwise`, `reduction`, `resample`, `rmsnorm`, `rmsnorm_backward`, `sdpa`, `sdpa_backward`. Whether a configuration runs depends on the loaded providers; see [Operation Support](./OperationSupport.md).

Not provided (compile error):

- v0.x / v8 API: `TensorBuilder`, `OperationBuilder`, `cudnnBackend*` descriptors, `cudnnException`, and similar.
- cuDNN C API beyond handle, stream, error-string, and version functions.
- `Graph::scaled_dot_product_flash_attention` and `_backward` (deprecated upstream; use `sdpa` / `sdpa_backward`).
- `Graph::key`, `get_variant_pack_uids_sorted`, `prepare_variant_pack_template`, `apply_oss_slot_indices_to_plans`.
- `Graph::set_override_shape_enabled`, unless hipDNN is built with `HIPDNN_ENABLE_SDPA`.
- `NormMode_t`, `RngDistribution_t`, `DescriptorType_t`.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `cudnn_frontend` is not declared | Missing namespace alias. Add it after the include. |
| Redefinition of `cudnnHandle_t` or `cudnnCreate` | NVIDIA cuDNN headers are also included. Remove them from shim builds. |
| v0.x types such as `TensorBuilder` do not compile | Not supported. Port the file to the v9 graph API or exclude it from the ROCm build. |
| `validate()` returns `cuDNN-shim node '<name>' has no hipDNN equivalent yet` | The graph uses an [unsupported node](#unsupported-nodes-and-apis). To request one, open an issue at https://github.com/ROCm/rocm-libraries/issues. |
| `validate()` returns `Duplicate tensor UID in graph` | Two tensors share a UID. Give each tensor a unique UID. |
| `Deviceless build is unsupported for non-empty graphs` | Call `build_operation_graph(handle)`. |
| `check_support` or `build_plans` returns `GRAPH_NOT_SUPPORTED` | No loaded hipDNN engine supports the graph. Check [Operation Support](./OperationSupport.md) and `HIPDNN_PLUGIN_DIR` ([Environment](./Environment.md)). |
| `validate()` fails with `... is unsupported by this shim` or a note-filter message | A setter or plan filter recorded the error earlier. See [Known divergences](#known-divergences). |
| Plan choice or performance differs from cuDNN | Expected: hipDNN engines and ranking differ. Set `HIPDNN_LOG_LEVEL=info` to see engine selection. |
| No log output | Set either `HIPDNN_LOG_LEVEL=info` with no `CUDNN_FRONTEND_LOG_*` or `NV_CUDNN_FRONTEND_DISABLE_LOGGING` variable in the environment, or `CUDNN_FRONTEND_LOG_INFO=1` and `CUDNN_FRONTEND_LOG_FILE=<path\|stdout\|stderr>` before the first `cudnnCreate`. Any cuDNN FE logging variable overrides `HIPDNN_LOG_*` at every `cudnnCreate`; `CUDNN_FRONTEND_LOG_INFO=0`, or a level without `CUDNN_FRONTEND_LOG_FILE`, turns logging off. See [Logging](./KNOWN_DIVERGENCES.md#logging). |
| `hipDNN.h` not found, or `CUDNN_FRONTEND_LOG_*` ignored after hipify | `hipify-perl` renamed cuDNN identifiers. Revert them; see [step 2](#hipify-a-cudnn-fe-program). |
