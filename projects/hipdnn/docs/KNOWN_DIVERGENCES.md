# Known divergences: cuDNN frontend compatibility shim

Behavior of `<hipdnn_compatibility/cudnn/cudnn_frontend.h>` that differs from NVIDIA cuDNN frontend v1.24.0. Usage and unsupported nodes: [cudnn-shim.md](./cudnn-shim.md).

NVIDIA, CUDA, and cuDNN are trademarks of NVIDIA Corporation. hipDNN is not affiliated with or endorsed by NVIDIA.

Errors that a setter or filter "records" are returned by the next `validate()`, `build_operation_graph()`, `create_execution_plans()`, `check_support()`, or `build_plans()`.

## Heuristic modes

Every `HeurMode_t` (`A`, `B`, `FALLBACK`, `OPENSOURCE`) is accepted. Modes other than `FALLBACK` log a warning and use hipDNN's default engine selection, so plan choice and performance can differ from cuDNN.

## Note filters

hipDNN has no per-plan numerical-note metadata and reports only its own behavior notes, so the shim triages each note by what ignoring it could do. These tables are canonical; `detail/note_triage.h` and `TestCudnnShimNoteTriageTable` follow them.

| Action | Effect |
|---|---|
| no-op | accepted silently |
| warn | logged as a warning, otherwise ignored |
| map | honored by filtering applicable engines; logged at info |
| error | records `GRAPH_NOT_SUPPORTED` |

### Numerical notes

| Note | `select_numeric_notes` | `deselect_numeric_notes` | Why |
|---|---|---|---|
| `NOT_SET` | no-op | no-op | placeholder |
| `TENSOR_CORE` | warn | warn | performance only |
| `DOWN_CONVERT_INPUTS` | warn | error | excluding it is a precision guarantee the shim cannot check |
| `REDUCED_PRECISION_REDUCTION` | warn | error | excluding it is a precision guarantee the shim cannot check |
| `FFT` | warn | warn | algorithm class |
| `NONDETERMINISTIC` | warn | map | deselect keeps only engines that claim determinism (see [Determinism](#determinism)) |
| `WINOGRAD`, `WINOGRAD_TILE_4x4`, `WINOGRAD_TILE_6x6`, `WINOGRAD_TILE_13x13` | warn | warn | algorithm class |
| `STRICT_NAN_PROP` | error | warn | selecting it is a NaN-propagation guarantee the shim cannot check; excluding it only relaxes |
| out-of-range value | error | error | unclassifiable |

Deselecting an out-of-range value is an error here but a no-op in cuDNN: without numerical metadata the shim cannot show the exclusion excludes nothing.

### Behavior notes

Notes hipDNN engines report map for both select and deselect. Any other value, including cuDNN's CUDA-specific notes, is an error on select (no hipDNN engine can satisfy it) and a warning on deselect (it excludes nothing).

| Note | `select_behavior_notes` | `deselect_behavior_notes` | Why |
|---|---|---|---|
| `NOT_SET` | no-op | no-op | placeholder |
| `RUNTIME_COMPILATION` | map | map | hipDNN note |
| `REQUIRES_LAYOUT_TRANSFORM` | map | map | hipDNN note |
| `SUPPORTS_GRAPH_CAPTURE` | map | map | hipDNN note |
| `EXTERNAL_LIBRARY_DEPENDENCY` | map | map | hipDNN note |
| `SUPPORTS_EXECUTION_PLAN_SERIALIZATION` | map | map | hipDNN note |
| `REQUIRES_FILTER_INT8x32_REORDER` | error | warn | cuDNN only |
| `REQUIRES_BIAS_INT8x32_REORDER` | error | warn | cuDNN only |
| `SUPPORTS_CUDA_GRAPH_NATIVE_API` | error | warn | cuDNN only; `populate_cuda_graph` is unsupported, and this is not stream capture (`SUPPORTS_GRAPH_CAPTURE`) |
| `CUBLASLT_DEPENDENCY` | error | warn | cuDNN only; not equivalent to `EXTERNAL_LIBRARY_DEPENDENCY` |
| out-of-range value | error | warn | reported by no hipDNN engine |

### Determinism

`deselect_numeric_notes({NONDETERMINISTIC})` keeps only engines that positively claim deterministic results; hipDNN engines declare no numerical notes, so a missing claim proves nothing. Until engine metadata carries the claim, the shim keeps its own allowlist, currently `MIOPEN_ENGINE_DETERMINISTIC`.

## Engine filtering

`deselect_engines`, the behavior-note filters, and the determinism filter are applied to the applicable engines after plan creation. A filter set after plans were created takes effect at the next `check_support()`, `build_plans()`, or `build_plan_at_index()`, including for plans already created.

If the filters bar only some engines, `build_plans(HEURISTICS_CHOICE)` builds the top-ranked surviving plan, as cuDNN does, instead of failing because the top-ranked engine was barred.

If no created plan survives, the next plan creation (`create_execution_plans`, `create_execution_plan`), `check_support()`, `build_plans()`, or `build_plan_at_index()` returns `GRAPH_NOT_SUPPORTED`; cuDNN returns `GRAPH_EXECUTION_PLAN_CREATION_FAILED` from `check_support()`. This includes `deselect_engines` alone barring every applicable engine.

## Engine indices

Engine indices are dense and graph-local, `0 .. get_engine_count() - 1`, in hipDNN's ranking order (not cuDNN's). `get_engine_count()` refreshes the index-to-engine map. `get_knobs_for_engine`, `create_execution_plan`, and `deselect_engines(std::vector<int64_t>)` translate indices through it. The `int64_t` overload of `deselect_engines` is a hipDNN extension.

## Knobs

hipDNN knob IDs are matched to `KnobType_t` by their last dot-separated segment, case-insensitively (`tile_size` and `miopen.tile_size` both map to `TILE_SIZE`). `workspace_size_limit` maps to `WORKSPACE`.

`get_knobs_for_engine` reports a hipDNN knob only if its ID maps to a `KnobType_t`, its value type is `int64`, and its constraint fits `minValue` / `maxValue` / `stride`. Others are omitted with a warning.

`create_execution_plan` returns `INVALID_VALUE` when a `KnobType_t` key is not exposed by the engine, maps to more than one of its knobs, or has a value the knob cannot hold (string knob, or above 2^53 for a float knob). Integer values are converted to `double` for float knobs.

## Graph setters

| Setter | Behavior |
|---|---|
| `set_dynamic_shape_enabled`, `set_kernel_cache` | Log, ignore. |
| `set_sm_count`, `set_sm_version`, `set_device_properties` | Record `INVALID_VALUE`. |
| `deselect_workspace_greater_than` | Forwarded; reapplied after plan creation. |
| `deselect_shared_mem_greater_than` | `0`: no-op. Non-zero: records `GRAPH_NOT_SUPPORTED` (no shared-memory metadata). |

## Build and execute

- `build_operation_graph()` without a handle succeeds only for a graph with no nodes.
- `build_plans(..., do_multithreaded_builds = true)` builds serially.
- `get_workspace_size*` with non-empty runtime shape overrides returns `GRAPH_NOT_SUPPORTED`. `execute` with overrides requires hipDNN built with `HIPDNN_ENABLE_SDPA`; otherwise it returns `INVALID_VALUE`.
- `execute(cudnnHandle_t, void**, int, void*)` returns `INVALID_VALUE`; the shim cannot recover tensor UIDs from a flat pointer array.
- `warmup(handle, tensor map, workspace)` runs plan 0 once. cuDNN FE's `warmup(handle)` signature is not provided.
- `populate_cuda_graph` and `update_cuda_graph` return `GRAPH_NOT_SUPPORTED`.
- `get_behavior_notes_for_plan_at_index` resolves the plan's engine from its name: built-in engines by name hash, others (including plugin engines) by the `0x`-prefixed hexadecimal ID hipDNN reports for them.
- `print()` on a graph whose operation graph is not built prints an empty object.
- `deserialize` ignores `enforce_precompiled`.

## hipDNN extensions

- `Graph::is_supported_ext(cudnnHandle_t, std::vector<HeurMode_t> = {FALLBACK})`: returns `OK` if any engine supports the graph, `GRAPH_NOT_SUPPORTED` otherwise. Builds the operation graph if needed; does not create or invalidate plans.
- `Graph::get_name`, `get_io_data_type`, `get_compute_data_type`, `get_intermediate_data_type`.
- `Graph::deserialize(data, enforce_precompiled)` without a handle.
- `HIPDNN_COMPATIBILITY_CUDNN_FRONTEND` macro.

Code that must also compile against NVIDIA cuDNN cannot use these.

## Versions

| Macro / function | Value |
|---|---|
| `CUDNN_FRONTEND_VERSION` | 12400 (v1.24.0) |
| `CUDNN_VERSION`, `cudnnGetVersion()`, `detail::get_backend_version()` | 92200 (9.22.0) |
| `CUDNN_CUDART_VERSION`, `cudnnGetCudartVersion()` | 12000 |

These are claims for consumers that gate on version; there is no cuDNN or CUDA runtime behind them.

## Logging

Every `cudnnCreate()` call maps cuDNN FE logging environment variables to hipDNN before creating the handle. If any of `CUDNN_FRONTEND_LOG_INFO`, `CUDNN_FRONTEND_LOG_FILE`, or `NV_CUDNN_FRONTEND_DISABLE_LOGGING` is set, the mapping overrides any `HIPDNN_LOG_LEVEL` / `HIPDNN_LOG_FILE` already set:

| cuDNN FE | hipDNN |
|---|---|
| `CUDNN_FRONTEND_LOG_INFO` > 0 and `CUDNN_FRONTEND_LOG_FILE=<path>` | `HIPDNN_LOG_LEVEL=info`, `HIPDNN_LOG_FILE=<path>` |
| `CUDNN_FRONTEND_LOG_INFO` > 0 and `CUDNN_FRONTEND_LOG_FILE=stdout` or `stderr` | `HIPDNN_LOG_LEVEL=info`, `HIPDNN_LOG_FILE` unset; both log to stderr |
| any other combination, or `NV_CUDNN_FRONTEND_DISABLE_LOGGING` set to a non-zero value | `HIPDNN_LOG_LEVEL=off`, `HIPDNN_LOG_FILE` unset |

Nothing changes when none of these variables is set. The level applies immediately. hipDNN picks the log destination (file or stderr) once, when its logging initializes: if that happens before `cudnnCreate()` (for example, on an earlier hipDNN call), the file mapping is ignored, and if the level was off at that point there is no console output either. `CUDNN_FE_LOG*` macros forward to hipDNN frontend logging at info level; compiling with `NV_CUDNN_FRONTEND_DISABLE_LOGGING` turns them into no-ops.
