# Known divergences: cuDNN frontend compatibility shim

Behavior of `<hipdnn_compatibility/cudnn/cudnn_frontend.h>` that differs from NVIDIA cuDNN frontend v1.24.0: the v9 graph API and, under [v8 backend C API](#v8-backend-c-api), the v0.x builders and the backend C API they use. Usage and unsupported nodes: [cudnn-shim.md](./cudnn-shim.md).

NVIDIA, CUDA, and cuDNN are trademarks of NVIDIA Corporation. hipDNN is not affiliated with or endorsed by NVIDIA.

Errors that a setter or filter "records" are returned by the next `validate()`, `build_operation_graph()`, `create_execution_plans()`, `check_support()`, or `build_plans()`.

## Heuristic modes

Every `HeurMode_t` (`A`, `B`, `FALLBACK`, `OPENSOURCE`) is accepted. `A` and `B` select hipDNN's `SelectionHeuristic::ModeA` and `SelectionHeuristic::ModeB` prediction policies, which rank engines by trained UHD models rather than by cuDNN's heuristics and decline to static engine selection when no applicable engine has a usable prediction (see the [Porting Guide](./PortingGuide.md)). `OPENSOURCE` uses hipDNN's default engine selection and logs a warning on its first use in a process. Plan choice and performance can differ from cuDNN.

The v8 backend C API accepts every `CUDNN_HEUR_MODE_*` (`INSTANT`, `A`, `B`, `FALLBACK`); every mode runs hipDNN's default engine selection, and the first use of each non-`FALLBACK` mode in a process logs one warning.

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

hipDNN's native `build_plans(HEURISTICS_CHOICE)` falls back on any plan-build failure. If the active plan fails to finalize, it logs a WARN with the engine ID and the failure, then tries the other ranked plans (scanning the ranked list from the top) and logs a WARN for each that also fails; the first plan that builds becomes the active plan. Plans barred by filters, descriptor-invalid plans, deselected engines, and plans over the workspace limit are skipped with an INFO line. When no plan builds, the error code is `GRAPH_EXECUTION_PLAN_CREATION_FAILED` and its message lists every attempted engine's failure, in attempt order; this includes a graph with a single plan. An active plan that builds but whose workspace exceeds the `deselect_workspace_greater_than` limit is an error, not skipped. A preferred engine (`set_preferred_engine_id_ext`) is a soft preference: if its plan fails to build, the fallback applies. `build_plans(ALL)` and `build_plan_at_index` do not fall back. When the shim's own filters bar the active plan, the shim retargets with `build_plan_at_index`, which does not walk the fallback: if the retarget target fails to build, its error is returned.

If no created plan survives, the next plan creation (`create_execution_plans`, `create_execution_plan`), `check_support()`, `build_plans()`, or `build_plan_at_index()` returns `GRAPH_NOT_SUPPORTED`; cuDNN returns `GRAPH_EXECUTION_PLAN_CREATION_FAILED` from `check_support()`. This includes `deselect_engines` alone barring every applicable engine.

## Engine indices

Engine indices are dense and graph-local, `0 .. get_engine_count() - 1`, in hipDNN's `FALLBACK` ranking of the applicable engines (not cuDNN's, and not the ranking of the requested heuristic mode, `A` and `B` included). `get_engine_count()` refreshes the index-to-engine map. `get_knobs_for_engine`, `create_execution_plan`, and `deselect_engines(std::vector<int64_t>)` translate indices through it, so every index names the same engine. The `int64_t` overload of `deselect_engines` is a hipDNN extension.

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

## v8 backend C API

### Enum values

The C enums in `cudnn_graph.h` (`cudnnBackendAttributeName_t`, `cudnnDataType_t`,
`cudnnPointwiseMode_t`, …) have the cuDNN names but the shim's own values: numbered from
0 in alphabetical order of the enumerator name, with each `*_COUNT` one past the last
value. Values are append-only within a `hipdnn_frontend` major version.
`cudnnBackendTensorReordering_t` follows the order of the shim's `TensorReordering_t`.

Impact: source compatibility only. Code that stores raw cuDNN enum values produced by
real cuDNN, or compares against hard-coded numbers, sees different values.

### Last-error string

`cudnnGetLastErrorString` returns the most recent shim error on the calling thread, falling back to
hipDNN's last error when the shim has none. The buffer lives in the shim's header-only code, so each
DLL (Windows) or shared object (Linux) that includes the shim has its own copy.

Impact: an error raised by shim code in one module is not visible to `cudnnGetLastErrorString`
called from another module.

### Parameter descriptors

NaN propagation on pointwise and resample parameter descriptors is stored and ignored; hipDNN has
no equivalent setting.

### Operations

Operation descriptors are validated at `cudnnBackendFinalize`. These differ from cuDNN:

| Case | Shim behavior |
|---|---|
| `GEN_STATS`, `BN_FINALIZE_STATISTICS`, `BN_BWD_WEIGHTS`, `RNG`, `RESHAPE`, `PAGED_CACHE_LOAD` and other operation types the shim does not lower (including `BLOCK_SCALE_*` and `MOE_GROUPED_MATMUL*`, which hipDNN supports natively) | `NOT_SUPPORTED` at finalize, before any attribute check |
| Alpha ≠ 1 or beta ≠ 0 on convolution, pointwise (`ALPHA1`, `ALPHA2`) or resample | `NOT_SUPPORTED`; hipDNN operations have no scaling factors |
| Batch-norm inference and backward `EPSILON_DESC` | accepted but not used; hipDNN's inference takes inverse variance and its backward takes saved statistics, so neither needs epsilon |
| Batch-norm backward without `MEAN_DESC` / `INV_VARIANCE_DESC` | `BAD_PARAM`; the saved statistics are required |
| Running-statistic descriptors outside batch-norm training; `MEAN_DESC` on RMS norm | `NOT_SUPPORTED` |
| Layer-norm forward without `BIAS_DESC` | `BAD_PARAM`; hipDNN's layer norm requires a bias |
| Norm `INSTANCE`, `GROUP`, `ADA_LAYER` modes; peer statistics | `NOT_SUPPORTED` |
| Max-pool backward resample without `IDXDESC` | `BAD_PARAM`; hipDNN needs the index tensor |
| Backward resample `XDESC` / `YDESC` | `NOT_SUPPORTED`; hipDNN's backward resample has no such inputs |
| Matmul M/N/K override descriptors | `NOT_SUPPORTED` |
| A pointwise tensor slot the mode does not use (for example `TDESC` on a non-select mode, or `YDESC` on a backward activation) | `BAD_PARAM`; upstream builders never set these |

### Operation graphs

`OPERATIONGRAPH` finalize lowers the graph to hipDNN; it needs no device and runs no heuristics.
These differ from cuDNN:

| Case | Shim behavior |
|---|---|
| No `HANDLE` | `BAD_PARAM`; hipDNN needs a handle to build the graph (cuDNN 9.8 and later make it optional) |
| `IS_DYNAMIC_SHAPE_ENABLED` or `IS_OVERRIDE_SHAPE_ENABLED` set to true | `NOT_SUPPORTED` at finalize; the v8 surface has no variant-pack shape overrides |
| Norm forward / backward compute type | cuDNN norm operations carry none; the shim sends `FLOAT` to hipDNN |
| Pointwise `RELU_*_CLIP*`, `ELU_ALPHA`, `SOFTPLUS_BETA`, `SWISH_BETA` | narrowed to `float`; values beyond the float range saturate to ±`FLT_MAX`. Parameters left at their defaults are not sent |
| `RESAMPLE` without `PADDING_MODE` | `BAD_PARAM` at the resample descriptor's finalize (upstream builders always send it) |
| Several `TENSOR` descriptors with one UID | accepted when they agree; `BAD_PARAM` naming the UID when data type, dimensions, strides, virtualness, by-value or constant value, or ragged offset differ. Differing byte alignments use the smallest |
| One UID written by two operations; a cycle | `BAD_PARAM` |
| `CONSTANT_VALUE` on a tensor whose type is not `FLOAT`, `DOUBLE`, `HALF`, `BFLOAT16`, `UINT8`, `INT32`, `INT64` or `BOOLEAN` | `NOT_SUPPORTED` at graph finalize |
| hipDNN operation validation | Graph finalize runs hipDNN's own checks. Beyond cuDNN's: epsilon and the exponential-average factor must be single-element `IS_BY_VALUE` tensors; batch-norm scale, bias and statistics must be per-channel `[1,C,1,…]` (no per-activation mode) with N·spatial > 1 when training; a reduction must reduce at least one dimension; a max-pool index must be `INT8` or `INT32`; RMS-norm scale must match the trailing dimensions of X. A rejection is `BAD_PARAM` naming the operation type, except "no engine supports this graph", which is `NOT_SUPPORTED`. Backend failures during lowering, allocation included, also report `BAD_PARAM` with hipDNN's message |
| `ENGINE_GLOBAL_COUNT` | computed on the first value GET by running hipDNN heuristics, which needs a HIP device; a failure returns its status and the next GET retries. It counts only engines hipDNN reports as applicable whose default configuration finalizes; an engine that fails is logged and left out, so every counted index builds. The result is a per-graph snapshot: later changes to the heuristic configuration or autotune cache do not affect an existing graph |
| Engine indices | `0 … ENGINE_GLOBAL_COUNT-1` enumerate the applicable hipDNN engines in ascending (signed) hipDNN engine-ID order, not cuDNN's global engine numbering. They change if the set of applicable engines changes (provider added, removed or upgraded, different GPU) |

### Engines, plans and execution

| Case | Shim behavior |
|---|---|
| `ENGINEHEUR_MODE` | every mode returns the same results: the graph's engines in hipDNN's fallback rank. See [Heuristic modes](#heuristic-modes) |
| `ENGINEHEUR_SM_COUNT_TARGET`, `ENGINEHEUR_DEVICEPROP` | `NOT_SUPPORTED` when set |
| `ENGINEHEUR_RESULTS` | configurations are finalized when returned and are shared: two fills of the same result return the same configuration |
| Engine indices in errata and in consumer code | errata rules and hard-coded engine numbers (for example PyTorch's cuDNN engines 5, 23, 58 and 63) name NVIDIA engines. Under the shim the same number selects an unrelated hipDNN engine, or none. At `CUDNN_VERSION` 92200 only the 3-D `ConvBwdData` engine-23 rule is active, and it applies only to a graph with more than 23 applicable engines |
| `ENGINE_NUMERICAL_NOTE` | `NONDETERMINISTIC` for every engine except those known to be deterministic (today `MIOPEN_ENGINE_DETERMINISTIC`); no other numerical note is reported, so filters on notes such as `DOWN_CONVERT_INPUTS` or `TENSOR_CORE` never match. Deterministic-only selection keeps only the known-deterministic engines |
| `ENGINE_BEHAVIOR_NOTE` | hipDNN behavior notes with a cuDNN counterpart, matched by name; others (for example `SUPPORTS_EXECUTION_PLAN_SERIALIZATION`) are not reported |
| `ENGINE_KNOB_INFO`, `ENGINE_BEHAVIOR_NOTE` | read from hipDNN on the first GET, not at `ENGINE` finalize; a failure is returned by that GET and a later GET retries |
| `ENGINE_KNOB_INFO` contents | lists only hipDNN knobs that project onto a cuDNN knob type (see [Knobs](#knobs): int64 value, plain minimum/maximum/stride, no list of valid values). A cuDNN type that two of an engine's knobs map onto is not listed (the v9 `get_knobs_for_engine` lists both) |
| `KNOB_INFO` finalized by the caller | `BAD_PARAM`; knob-information descriptors are filled by `ENGINE_KNOB_INFO` only |
| `ENGINECFG_KNOB_CHOICES` | checked at `ENGINECFG` finalize: a type not listed in the engine's `KNOB_INFO` is `NOT_SUPPORTED`; a type chosen twice, or a value outside the knob's range or off its stride, is `BAD_PARAM`. A failure building the hipDNN configuration is reported at `EXECUTION_PLAN` finalize. A choice of `-1` means "unset" (upstream's `EngineConfigBuilder` sends `-1` for every knob it was not asked to set once one is set): it is ignored (not checked, not applied, not counted as a repeat, not in the plan JSON `"knobs"`) unless the knob is listed and its range accepts `-1`. The `KNOB_CHOICES` GET still returns every choice as set. A configuration whose choices are all ignored uses the engine's default hipDNN configuration |
| `EXECUTION_PLAN_HANDLE` | optional for a plan built from `ENGINE_CONFIG` (upstream's `ExecutionPlanBuilder` never sets it): unset or null, the plan runs on the operation graph's `HANDLE`; when set, `BAD_PARAM` at finalize unless it is that handle (hipDNN builds the plan on the graph's handle). Required for a plan loaded from `JSON_REPRESENTATION` (`BAD_PARAM` otherwise), as upstream's `loadFromJson` sets it |
| `EXECUTION_PLAN_KERNEL_CACHE` | `NOT_SUPPORTED` when set |
| `EXECUTION_PLAN_JSON_REPRESENTATION` format | the shim's own versioned JSON, not cuDNN's: engine index, engine ID (a decimal string), engine name, behavior notes, knob choices and, when hipDNN can serialize the plan, a base64 payload. Notes and knob types are the shim's enum values. JSON written by cuDNN or by another tool is `BAD_PARAM` at finalize |
| Loading a plan without `"payload"` | `NOT_SUPPORTED` at finalize; engines whose plans hipDNN cannot serialize cannot be reloaded |
| Loaded plans | the engine ID must match the one inside the payload (`BAD_PARAM` otherwise). Engine index, name, behavior notes and knob choices are reported from the JSON as saved; the index refers to the engine set of the process that saved it. The loaded plan's engine has no `OPERATION_GRAPH` or `KNOB_INFO` (`NOT_SUPPORTED`), and its configuration cannot build a new plan (`NOT_SUPPORTED`) |
| `cudnnBackendExecute` handle | `BAD_PARAM` unless it is the handle the plan was built on (the graph's handle) or, for a loaded plan, the `HANDLE` it was loaded with |
| `VARIANT_PACK` | the hipDNN variant pack is built at finalize; `DATA_POINTERS` and `UNIQUE_IDS` with different element counts are `BAD_PARAM` at finalize |
| `ENGINE_LAYOUT_INFO`, `ENGINECFG_WORKSPACE_SIZE`, `VARIANT_PACK_INTERMEDIATES` | not declared (no supported consumer uses them); code that names them does not compile |

### Builders

The v0.x builders (`TensorBuilder`, `ConvDescBuilder`, `PointWiseDescBuilder`, `MatMulDescBuilder`,
`ReductionDescBuilder`, `ResampleDescBuilder`, `RngDescBuilder`, `OperationBuilder`,
`OperationGraphBuilder`, `VariantPackBuilder`, `EngineHeuristicsBuilder`, `EngineBuilder`,
`EngineConfigBuilder`, `EngineFallbackListBuilder`, `ExecutionPlanBuilder`) and the plan helpers
(`get_heuristics_list`, `cudnnGetPlan`, `time_sorted_plan`, `cudnnFindPlan`,
`cudnnFindPlanAndCache`) are upstream's, running on the C API above, so every C API row applies to
their `build()`. These differ as well:

| Case | Shim behavior |
|---|---|
| Builder error text | `cudnnException::what()` ends in `cudnn_status: HIPDNN_STATUS_…`: `to_string(cudnnStatus_t)` returns `cudnnGetErrorString`, which reports hipDNN's name for the translated status. `getCudnnStatus()` and `get_status()` still carry the cuDNN status |
| nlohmann serialization of frontend enums | `DataType_t`, `PointwiseMode_t` and the other frontend enums are hipDNN's enums, and the shim defines no nlohmann serializer for them, so `json j = DataType_t::FLOAT;` stores an integer, not upstream's `"FLOAT"`. `describe()` is unaffected: it renders enums through the shim's own name table, byte for byte as upstream |
| Descriptor log lines | the body of a builder's log line (`describe()`) is printed only when `CUDNN_FRONTEND_LOG_INFO` is above 0, as upstream, and the line itself goes through hipDNN INFO logging. With hipDNN INFO logging on but `CUDNN_FRONTEND_LOG_INFO` unset, the line is `[cudnn_frontend] ` with nothing after it |
| `getStream()` / `getLogger()` | no shim code calls them. A consumer call opens `CUDNN_FRONTEND_LOG_FILE` for writing, truncating it as upstream does; the `cudnnCreate()` logging bridge may also route hipDNN's log to that file, so the two writers can overwrite each other |
| `NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING` | ignored: the shim never loads cuDNN at run time, and `NV_CUDNN_FE_DYNAMIC_CHECK_*` expand to nothing, so the run-time cuDNN version checks they guard are skipped |
| `get_environment()`, `getLogLevel()` | `get_environment` returns a pointer into a per-thread buffer that the next call on the same thread overwrites (upstream returns `getenv`'s pointer), and returns null for a variable that is set but empty. `getLogLevel` returns 0 when the `NV_CUDNN_FRONTEND_DISABLE_LOGGING` environment variable is set to anything but `0`; upstream honors only the macro |
| Operation and operation-graph `build()` errors | an operation the C API rejects at finalize fails `OperationBuilder::build()` with that row's status, and a graph rejection fails `OperationGraphBuilder::build()` the same way: see [Operations](#operations) and [Operation graphs](#operation-graphs). `GenStats`, `BNFinalize`, `Dgrad_Drelu_BN_Bwd`, `Rng`, paged-cache-load and `Reshape` operations always fail with `NOT_SUPPORTED` |
| `OperationGraphBuilder` without `setHandle` | `build()` fails with `BAD_PARAM`. cuDNN 9.8 and later build the graph without a handle |
| PyTorch fused convolutions | `cudnn_convolution_relu` passes `alpha2 = 0` to the pointwise add, so it always gets `NOT_SUPPORTED`; `cudnn_convolution_add_relu` does whenever its alpha is not 1 (see the scaling row under [Operations](#operations)) |
| Heuristic mode strings | every mode returns the same list (see [Heuristic modes](#heuristic-modes)), so a mode list such as `{"heuristics_instant", "heuristics_fallback"}` given to `get_heuristics_list` or `EngineConfigGenerator` yields every engine configuration once per mode. `cudnnGetPlan` and `time_sorted_plan` build and time the duplicates, and they count against a plan or benchmark limit |
| `get_fallback_engine_list` | returns upstream's table of NVIDIA engine numbers; under the shim those numbers select unrelated hipDNN engines or none (see the engine-indices row under [Engines, plans and execution](#engines-plans-and-execution)) |
| `setKernelCache`, `setDeviceProperties`, `setSMCount` | a non-null kernel cache or device properties, or an SM count of 0 or more, makes `build()` fail with `NOT_SUPPORTED` |
| `cudnnReorderFilterAndBiasInt8x32` | checks its arguments as upstream, then returns `NOT_SUPPORTED`: the shim has no int8x32 reordering |
| Plan tags | the `_k<type>=<value>` segments of `ExecutionPlan::getTag()` use the shim's knob-type integers, so errata `"knob"` entries written against cuDNN's integers do not match |
| `time_sorted_plan` timing | measured with hipDNN profiling events on the handle's stream: device time between the start and stop of each timed execution, as upstream's CUDA events |
| `time_sorted_plan` ties | plans with equal times are all kept; upstream keeps only one of them |
| `cudnnFindPlanAndCache` with no plan that ran | throws `cudnnException` with `NOT_SUPPORTED` (with `NV_CUDNN_DISABLE_EXCEPTION`, returns a plan whose status is not `SUCCESS`); upstream reads the front of an empty list |
| PyTorch quantized operations across threads | PyTorch's quantized convolution, linear and add operations cache plans process-wide, so a plan cached on one thread and executed with another thread's handle is `BAD_PARAM` (see the `cudnnBackendExecute` handle row). PyTorch's `Conv_v8` caches are per-thread and unaffected |
