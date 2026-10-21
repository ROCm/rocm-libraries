# Known divergences: cuDNN frontend compatibility shim

Behavior of `<hipdnn_compatibility/cudnn/cudnn_frontend.h>` that differs from NVIDIA cuDNN frontend v1.24.0: the v9 graph API and, under [cuDNN v8 API](#cudnn-v8-api), the v0.x builders. Usage and unsupported nodes: [cudnn-shim.md](./cudnn-shim.md).

NVIDIA, CUDA, and cuDNN are trademarks of NVIDIA Corporation. hipDNN is not affiliated with or endorsed by NVIDIA.

Errors that a setter or filter "records" are returned by the next `validate()`, `build_operation_graph()`, `create_execution_plans()`, `check_support()`, or `build_plans()`.

## Heuristic modes

Every `HeurMode_t` (`A`, `B`, `FALLBACK`, `OPENSOURCE`) is accepted. `A` and `B` select hipDNN's `SelectionHeuristic::ModeA` and `SelectionHeuristic::ModeB` prediction policies, which rank engines by trained UHD models rather than by cuDNN's heuristics and decline to static engine selection when no applicable engine has a usable prediction (see the [Porting Guide](./PortingGuide.md)). `OPENSOURCE` uses hipDNN's default engine selection and logs a warning on its first use in a process. Plan choice and performance can differ from cuDNN.

The v0.x builders accept every `CUDNN_HEUR_MODE_*` (`INSTANT`, `A`, `B`, `FALLBACK`); every mode runs hipDNN's default engine selection, and the first use of each non-`FALLBACK` mode in a process logs one warning.

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

## cuDNN v8 API

The v0.x builders build hipDNN graphs, engine configurations, plans and variant packs directly. The
cuDNN backend descriptor C API that upstream's builders call is not provided (see
[Backend descriptor C API](#backend-descriptor-c-api)). Only the C identifiers that the v0.x headers,
NVIDIA's legacy samples and PyTorch's v0.x consumers name are declared, less that API; code that names
any other cuDNN C identifier does not compile. The legacy v7-style C API (`cudnnConvolutionForward`,
`cudnnSetTensorNdDescriptor`, …) is not provided.

### Backend descriptor C API

| Case | Shim behavior |
|---|---|
| `cudnnBackendCreateDescriptor`, `cudnnBackendDestroyDescriptor`, `cudnnBackendSetAttribute`, `cudnnBackendGetAttribute`, `cudnnBackendFinalize`; `cudnnBackendAttributeName_t` (`CUDNN_ATTR_*`), `cudnnBackendAttributeType_t` (`CUDNN_TYPE_*`) | not declared: code that calls or names them, such as XLA's autotune persistence, does not compile. `cudnnBackendExecute` and `cudnnGetLastErrorString` are declared and pass straight through to hipDNN |
| `cudnnMoeGroupedMatmulMode_t` | not declared; no v0.x builder takes it |
| `cudnnBackendDescriptor_t` | an alias of `hipdnnBackendDescriptor_t`. `get_raw_desc()` returns the hipDNN operation graph, engine, engine configuration, execution plan or variant pack. Tensors, parameter descriptors, operations, heuristics and fallback lists have no hipDNN descriptor, and their `get_raw_desc()` is null |
| Errata `check_shape` | takes the operation's `ManagedOpaqueDescriptor` and a shim tensor-slot enum instead of a backend descriptor and an attribute name. `check_rule` and `check_errata` are unchanged |

### Enum values

The C enums (`cudnnBackendDescriptorType_t`, `cudnnDataType_t`,
`cudnnPointwiseMode_t`, …) have the cuDNN names but the shim's own values: numbered from
0 in alphabetical order of the enumerator name, with each `*_COUNT` one past the last
value. Values are append-only within a `hipdnn_frontend` major version.
`cudnnBackendTensorReordering_t` follows the order of the shim's `TensorReordering_t`.

Impact: source compatibility only. Code that stores raw cuDNN enum values produced by
real cuDNN, or compares against hard-coded numbers, sees different values.

### Last-error string

`cudnnGetLastErrorString` returns hipDNN's last error on the calling thread and clears it, as
`hipdnnGetLastErrorString` does. A builder failure that the shim detects without a failing hipDNN call
is not recorded there. When a hipDNN call fails, the shim reads (and so clears) hipDNN's error and
appends it to the builder's message.

Impact: the full error of a failed `build()` is in `cudnnException::what()` or the object's
`get_error()`; `cudnnGetLastErrorString` afterwards is normally empty.

### Parameter descriptors

`ConvDescBuilder`, `PointWiseDescBuilder`, `MatMulDescBuilder`, `ReductionDescBuilder`,
`ResampleDescBuilder` and `RngDescBuilder` validate at `build()`; their values reach hipDNN when the
operation graph is built. These differ from cuDNN:

| Case | Shim behavior |
|---|---|
| NaN propagation on pointwise and resample descriptors | stored and ignored; hipDNN has no equivalent setting |
| Pointwise ReLU clips and lower-clip slope, ELU alpha, softplus beta, swish beta | narrowed to `float` when the graph is built; values beyond the float range saturate to ±`FLT_MAX`. Parameters left at their defaults are not sent, except the ReLU lower clip, which is always sent unless a slope is set (hipBLASLt's ReLU epilogue requires it) |
| `MatMulDescBuilder::setPaddingValue` | `NOT_SUPPORTED` at `MatMulDescBuilder::build()` |
| `RngDescBuilder` | builds, but an `RNG` operation is `NOT_SUPPORTED` (see [Operations](#operations)) |

### Tensors

`TensorBuilder::build()` validates the tensor. These differ from cuDNN:

| Case | Shim behavior |
|---|---|
| Byte alignment below 16 (`setAlignment`) | forwarded to hipDNN. A graph with such a tensor needs engine-plugin API 1.3.0 or later, which no current provider reports, so the graph has no engine: `getEngineCount()` is 0 and heuristics return no results, both with `SUCCESS` |
| Vectorized tensors: a vector count other than 1 (`setVectorCountAndDimension`); the data types `INT8x4`, `INT8x32` and `UINT8x4`; a reorder type other than `NONE` | `NOT_SUPPORTED` at `build()` |

### Operations

`OperationBuilder::build()` validates the operation. These differ from cuDNN:

| Case | Shim behavior |
|---|---|
| Operation types the shim does not lower: `GenStats`, `BNFinalize`, `Dgrad_Drelu_BN_Bwd` (BN backward weights), `Rng`, `Reshape`, paged cache load | `NOT_SUPPORTED` at `build()`, after upstream's checks of the operation's required tensors and descriptors |
| Alpha ≠ 1 or beta ≠ 0 on convolution, pointwise (`setAlpha`, `setAlpha2`) or resample | `NOT_SUPPORTED`; hipDNN operations have no scaling factors |
| Batch-norm inference and backward epsilon (`setEpsilonTensor`) | accepted but not used; hipDNN's inference takes inverse variance and its backward takes saved statistics, so neither needs epsilon |
| Batch-norm backward without saved statistics (`setSavedMeanAndInvVar`) | `BAD_PARAM`; the saved statistics are required |
| Running statistics (`setPrevRunningMeanAndVar`, `setNextRunningMeanAndVar`) outside batch-norm training; a saved mean on RMS norm | `NOT_SUPPORTED` |
| Layer-norm forward without a bias (`setScaleAndBias`) | `BAD_PARAM`; hipDNN's layer norm requires a bias |
| Norm `INSTANCE`, `GROUP`, `ADA_LAYER` modes; peer statistics (`addPeerStatTensor`, `setPeerStatTensor`) | `NOT_SUPPORTED` |
| Max-pool backward resample without an index tensor (`setidxDesc`) | `BAD_PARAM`; hipDNN needs the index tensor |
| Backward resample with `setxDesc` / `setyDesc` | `NOT_SUPPORTED`; hipDNN's backward resample has no such inputs |
| Matmul M/N/K overrides (`setmOverrideDesc`, `setnOverrideDesc`, `setkOverrideDesc`) | `NOT_SUPPORTED` |

### Operation graphs

`OperationGraphBuilder::build()` lowers the graph to hipDNN; it needs no device and runs no heuristics.
These differ from cuDNN:

| Case | Shim behavior |
|---|---|
| No handle (`setHandle`) | `BAD_PARAM`; hipDNN needs a handle to build the graph (cuDNN 9.8 and later make it optional) |
| `setIsDynamicShapeEnabled(true)` or `setIsOverrideShapeEnabled(true)` | `NOT_SUPPORTED` at `build()`; the v8 surface has no variant-pack shape overrides |
| Norm forward / backward compute type | cuDNN norm operations carry none; the shim sends `FLOAT` to hipDNN |
| Convolution with `HALF` or `BFLOAT16` tensors and a `FLOAT` compute type | forwarded as `FLOAT`, but the engine chooses the accumulation precision: the MIOpen provider may run a reduced-precision kernel (a 3x3 stride-1 `HALF` forward on gfx1151 runs the Winograd solver `ConvWinoFuryRxS<2-3>` with fp16 accumulation), so results can differ from cuDNN by more than fp32 accumulation would. They match PyTorch's own MIOpen convolution |
| Several tensors with one UID | accepted when they agree; `BAD_PARAM` naming the UID when data type, dimensions, strides, virtualness, by-value or constant value, or ragged offset differ. Differing byte alignments use the smallest |
| One UID written by two operations; a cycle | `BAD_PARAM` |
| A constant value (`setConstValue`) on a tensor whose type is not `FLOAT`, `DOUBLE`, `HALF`, `BFLOAT16`, `UINT8`, `INT32`, `INT64` or `BOOLEAN` | `NOT_SUPPORTED` at graph build |
| hipDNN operation validation | Graph build runs hipDNN's own checks. Beyond cuDNN's: epsilon and the exponential-average factor must be single-element by-value tensors (`setByValue`); batch-norm scale, bias and statistics must be per-channel `[1,C,1,…]` (no per-activation mode) with N·spatial > 1 when training; a reduction must reduce at least one dimension; a max-pool index must be `INT8` or `INT32`; RMS-norm scale must match the trailing dimensions of X. A rejection is `BAD_PARAM` naming the operation type, except "no engine supports this graph", which is `NOT_SUPPORTED`. Backend failures during lowering, allocation included, also report `BAD_PARAM` with hipDNN's message |
| Engine count (`getEngineCount()`) | computed on the graph's first engine query (`getEngineCount()`, heuristics or `EngineBuilder`) by running hipDNN heuristics, which needs a HIP device; a failure returns its status and the next query retries. It counts only engines hipDNN reports as applicable whose default configuration finalizes; an engine that fails is logged and left out, so every counted index builds. The result is a per-graph snapshot: later changes to the heuristic configuration or autotune cache do not affect an existing graph |
| Engine indices | `0 … getEngineCount()-1` enumerate the applicable hipDNN engines in ascending (signed) hipDNN engine-ID order, not cuDNN's global engine numbering. They change if the set of applicable engines changes (provider added, removed or upgraded, different GPU) |

### Engines, plans and execution

These differ from cuDNN:

| Case | Shim behavior |
|---|---|
| Heuristic mode (`setHeurMode`) | every mode returns the same results: the graph's engines in hipDNN's fallback rank. See [Heuristic modes](#heuristic-modes) |
| `EngineHeuristics::getEngineConfig(count)` | returns the first `count` results (at most `getEngineConfigCount()`), the same objects on every call; upstream appends `count` new configurations per call. Every result for a given engine of a graph, from any heuristics or fallback list, shares that engine's default hipDNN configuration: their `get_raw_desc()` is equal |
| Matmul on gfx115x under Windows | hipBLASLt declines every matmul there (a hipBLASLt GEMM crash workaround), and no other provider runs matmul, so a v8 matmul graph has no engine |
| Engine indices in errata and in consumer code | errata rules and hard-coded engine numbers (for example PyTorch's cuDNN engines 5, 23, 58 and 63) name NVIDIA engines. Under the shim the same number selects an unrelated hipDNN engine, or none. At `CUDNN_VERSION` 92200 only the 3-D `ConvBwdData` engine-23 rule is active, and it applies only to a graph with more than 23 applicable engines |
| Numerical notes (`hasNumericalNote`, `ExecutionPlan::getNumericNotes`) | `NONDETERMINISTIC` for every engine except those known to be deterministic (today `MIOPEN_ENGINE_DETERMINISTIC`); no other numerical note is reported, so filters on notes such as `DOWN_CONVERT_INPUTS` or `TENSOR_CORE` never match. Deterministic-only selection keeps only the known-deterministic engines |
| Behavior notes (`hasBehaviorNote`, `ExecutionPlan::getBehaviorNotes`) | hipDNN behavior notes with a cuDNN counterpart, matched by name; others (for example `SUPPORTS_EXECUTION_PLAN_SERIALIZATION`) are not reported |
| Engine knobs (`Engine::getSupportedKnobs`) | lists only hipDNN knobs that project onto a cuDNN knob type (see [Knobs](#knobs): int64 value, plain minimum/maximum/stride, no list of valid values). A cuDNN type that two of an engine's knobs map onto is not listed (the v9 `get_knobs_for_engine` lists both) |
| Knob choices (`EngineConfigBuilder`) | checked at `EngineConfigBuilder::build()`: a type the engine does not list is `NOT_SUPPORTED`; a type chosen twice, or a value outside the knob's range or off its stride, is `BAD_PARAM`. The same `build()` creates the hipDNN configuration, so a failure there is also reported by `EngineConfigBuilder::build()`, not by `ExecutionPlanBuilder::build()`. A choice of `-1` means "unset" (upstream's `EngineConfigBuilder` sends `-1` for every knob it was not asked to set once one is set): it is ignored (not checked, not applied, not counted as a repeat, not in the plan JSON `"knobs"`) unless the knob is listed and its range accepts `-1`; the plan tag keeps it. A configuration whose choices are all ignored uses the engine's default hipDNN configuration |
| Plan JSON format (`getJsonRepresentation`, `loadFromJson`) | the shim's own versioned JSON, not cuDNN's: engine index, engine ID (a decimal string), engine name, behavior notes, knob choices and, when hipDNN can serialize the plan, a base64 payload. Notes and knob types are the shim's enum values. JSON written by cuDNN or by another tool makes `loadFromJson` fail with `BAD_PARAM` |
| Loading a plan without `"payload"` | `NOT_SUPPORTED` from `loadFromJson`; engines whose plans hipDNN cannot serialize cannot be reloaded |
| Loaded plans | the engine ID must match the one inside the payload (`BAD_PARAM` otherwise). Engine index, name, behavior notes and knob choices are reported from the JSON as saved; the index refers to the engine set of the process that saved it |
| `cudnnBackendExecute` handle | not checked: hipDNN runs the plan on the handle it is given, with that handle's stream, including a handle other than the one the plan's graph was built on or, for a loaded plan, the one it was loaded with. `ExecutionPlanBuilder::setHandle` is ignored by `build()`, as upstream never sends it, and required by `loadFromJson` |
| Variant packs (`VariantPackBuilder::build()`) | the hipDNN variant pack is built here; missing or empty data pointers or UIDs, or the two with different element counts, are `BAD_PARAM` |
| Per-call overhead | `cudnnBackendExecute` is one hipDNN call. `VariantPackBuilder::build()` also builds the hipDNN variant pack on every call. Both stay within the budgets of RFC 0021 §2 over calling hipDNN directly on the same plan, as measured by `samples/cudnn_shim/ConvFpropV8.cpp --overhead` |

### Builders

The v0.x builders (`TensorBuilder`, `ConvDescBuilder`, `PointWiseDescBuilder`, `MatMulDescBuilder`,
`ReductionDescBuilder`, `ResampleDescBuilder`, `RngDescBuilder`, `OperationBuilder`,
`OperationGraphBuilder`, `VariantPackBuilder`, `EngineHeuristicsBuilder`, `EngineBuilder`,
`EngineConfigBuilder`, `EngineFallbackListBuilder`, `ExecutionPlanBuilder`) and the plan helpers
(`get_heuristics_list`, `cudnnGetPlan`, `time_sorted_plan`, `cudnnFindPlan`,
`cudnnFindPlanAndCache`) are upstream's, with the members that called the backend rewritten over
hipDNN. The sections above list their rules; these differ as well:

| Case | Shim behavior |
|---|---|
| Builder error text | messages name the builder and the rule (for example `OperationBuilder: alpha2 = 0.5 is not supported by the hipDNN cuDNN shim`), not upstream's attribute names, followed by hipDNN's message when a hipDNN call failed. `cudnnException::what()` ends in `cudnn_status: HIPDNN_STATUS_…`: `to_string(cudnnStatus_t)` returns `cudnnGetErrorString`, which reports hipDNN's name for the translated status. `getCudnnStatus()` and `get_status()` still carry the cuDNN status |
| Status after a retried `build()` | when `TensorBuilder` or a parameter-descriptor builder succeeds after a failed `build()`, the object reports `SUCCESS` and an empty error; upstream keeps the earlier failure's status |
| nlohmann serialization of frontend enums | `DataType_t`, `PointwiseMode_t` and the other frontend enums are hipDNN's enums, and the shim defines no nlohmann serializer for them, so `json j = DataType_t::FLOAT;` stores an integer, not upstream's `"FLOAT"`. `describe()` is unaffected: it renders enums through the shim's own name table, byte for byte as upstream |
| Descriptor log lines | the body of a builder's log line (`describe()`) is printed only when `CUDNN_FRONTEND_LOG_INFO` is above 0, as upstream, and the line itself goes through hipDNN INFO logging. With hipDNN INFO logging on but `CUDNN_FRONTEND_LOG_INFO` unset, the line is `[cudnn_frontend] ` with nothing after it |
| `getStream()` / `getLogger()` | no shim code calls them. A consumer call opens `CUDNN_FRONTEND_LOG_FILE` for writing, truncating it as upstream does; the `cudnnCreate()` logging bridge may also route hipDNN's log to that file, so the two writers can overwrite each other |
| `NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING` | ignored: the shim never loads cuDNN at run time, and `NV_CUDNN_FE_DYNAMIC_CHECK_*` expand to nothing, so the run-time cuDNN version checks they guard are skipped |
| `get_environment()`, `getLogLevel()` | `get_environment` returns a pointer into a per-thread buffer that the next call on the same thread overwrites (upstream returns `getenv`'s pointer), and returns null for a variable that is set but empty. `getLogLevel` returns 0 when the `NV_CUDNN_FRONTEND_DISABLE_LOGGING` environment variable is set to anything but `0`; upstream honors only the macro |
| PyTorch fused convolutions | `cudnn_convolution_relu` passes `alpha2 = 0` to the pointwise add, so it always gets `NOT_SUPPORTED`; `cudnn_convolution_add_relu` does whenever its alpha is not 1 (see the scaling row under [Operations](#operations)) |
| Heuristic mode strings | every mode returns the same list (see [Heuristic modes](#heuristic-modes)), so a mode list such as `{"heuristics_instant", "heuristics_fallback"}` given to `get_heuristics_list` or `EngineConfigGenerator` yields every engine configuration once per mode. `cudnnGetPlan` and `time_sorted_plan` build and time the duplicates, and they count against a plan or benchmark limit |
| `get_fallback_engine_list` | returns upstream's table of NVIDIA engine numbers; under the shim those numbers select unrelated hipDNN engines or none (see the engine-indices row under [Engines, plans and execution](#engines-plans-and-execution)) |
| `setKernelCache`, `setDeviceProperties`, `setSMCount` | a non-null kernel cache or device properties, or an SM count of 0 or more, makes `build()` fail with `NOT_SUPPORTED` |
| `cudnnReorderFilterAndBiasInt8x32` | checks its arguments as upstream, then returns `NOT_SUPPORTED`: the shim has no int8x32 reordering |
| Plan tags | the `_k<type>=<value>` segments of `ExecutionPlan::getTag()` use the shim's knob-type integers, so errata `"knob"` entries written against cuDNN's integers do not match |
| `time_sorted_plan` timing | measured with hipDNN profiling events on the handle's stream: device time between the start and stop of each timed execution, as upstream's CUDA events |
| `time_sorted_plan` ties | plans with equal times are all kept; upstream keeps only one of them |
| `cudnnFindPlanAndCache` with no plan that ran | throws `cudnnException` with `NOT_SUPPORTED`; with `NV_CUDNN_DISABLE_EXCEPTION` it returns an unbuilt plan whose status is `BAD_PARAM` (no engine configuration). Upstream reads the front of an empty list |
| PyTorch quantized operations across threads | PyTorch's quantized convolution, linear and add operations cache plans process-wide, so a plan cached on one thread can be executed with another thread's handle; it runs on that handle (see the `cudnnBackendExecute` handle row). PyTorch's `Conv_v8` caches are per-thread and unaffected |
| `ConvDescBuilder` default mode | upstream's default is `CONVOLUTION`; MIOpen runs only `CROSS_CORRELATION`, so a graph that keeps the default has no engine. Call `setMathMode(CUDNN_CROSS_CORRELATION)` |
| `PointWiseDescBuilder` Swish | β = 1 is upstream's default and is not sent (see [Parameter descriptors](#parameter-descriptors)), so hipBLASLt's Swish epilogue (which requires β present and equal to 1) is unreachable from v8; MIOpen's fused convolution rejects a Swish β of any value |
