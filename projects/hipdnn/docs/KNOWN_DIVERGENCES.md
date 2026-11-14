# Known divergences: cuDNN frontend compatibility shim

Behavior of `<hipdnn_compatibility/cudnn/cudnn_frontend.h>` that differs from NVIDIA cuDNN frontend v1.24.0: the v9 graph API and, under [cuDNN v8 API](#cudnn-v8-api), the v0.x builders. It also lists where hipDNN's SDPA engines differ from cuDNN (dropout random stream, SDPA backward gaps and extensions), outside the shim. Usage and unsupported nodes: [cudnn-shim.md](./cudnn-shim.md).

NVIDIA, CUDA, and cuDNN are trademarks of NVIDIA Corporation. hipDNN is not affiliated with or endorsed by NVIDIA.

Errors that a setter or filter "records" are returned by the next `validate()`, `build_operation_graph()`, `create_execution_plans()`, `check_support()`, or `build_plans()`.

## Heuristic modes

Every `HeurMode_t` (`A`, `B`, `FALLBACK`, `OPENSOURCE`) is accepted. `A` and `B` select hipDNN's `SelectionHeuristic::ModeA` and `SelectionHeuristic::ModeB` prediction policies, which rank engines by trained UHD models rather than by cuDNN's heuristics and decline to static engine selection when no applicable engine has a usable prediction (see the [Porting Guide](./PortingGuide.md)). `OPENSOURCE` uses hipDNN's default engine selection and logs a warning on its first use in a process. Plan choice and performance can differ from cuDNN.

The v0.x builders accept every `CUDNN_HEUR_MODE_*` (`INSTANT`, `A`, `B`, `FALLBACK`). `A` and `B` rank by the `ModeA` and `ModeB` policies, `INSTANT` as `A` (NVIDIA's alias), and `FALLBACK` by hipDNN's default engine selection; no mode logs a warning. Every mode returns the same engines, the ones `FALLBACK` finds, in its own order. With `HIPDNN_HEUR_POLICY_ORDER` set, the environment overrides the requested policy and every mode orders alike.

## Note filters

hipDNN engines report numerical notes (in-tree engines report `NONDETERMINISTIC` or none) and their own behavior notes, per engine and not per plan. The shim filters on `NONDETERMINISTIC` and on the behavior notes hipDNN reports, and triages every other note by what ignoring it could do. These tables are canonical; `detail/note_triage.h` and `TestCudnnShimNoteTriageTable` follow them.

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
| `NONDETERMINISTIC` | map | map | select keeps engines that report it; deselect keeps only engines that do not (see [Determinism](#determinism)) |
| `WINOGRAD`, `WINOGRAD_TILE_4x4`, `WINOGRAD_TILE_6x6`, `WINOGRAD_TILE_13x13` | warn | warn | algorithm class |
| `STRICT_NAN_PROP` | error | warn | selecting it is a NaN-propagation guarantee the shim cannot check; excluding it only relaxes |
| out-of-range value | error | error | unclassifiable |

Deselecting an out-of-range value is an error here but a no-op in cuDNN: the shim filters only on `NONDETERMINISTIC`, so it cannot show the exclusion excludes nothing.

No in-tree engine reports a precision note (`DOWN_CONVERT_INPUTS`, `REDUCED_PRECISION_REDUCTION`), so the numerical-note rows other than `NONDETERMINISTIC` never take effect.

Every engine defaults to `NONDETERMINISTIC`, which means its results are not guaranteed bitwise reproducible, as in cuDNN; a provider reports an empty list only for an engine a test proves reproducible. The native `Graph::get_numerical_notes_for_engine` and the v8 `hasNumericalNote` therefore return `{NONDETERMINISTIC}` for an engine whose provider omitted the field. Both also read `NOT_SUPPORTED` from an older same-major backend, which lacks the attribute, as `{NONDETERMINISTIC}`; so do the v9 filters and the Python binding, which use the native getter. With a newer frontend on an older same-major backend that lacks the attribute, every engine, including `MIOPEN_ENGINE_DETERMINISTIC`, therefore reads `{NONDETERMINISTIC}`, so deselecting `NONDETERMINISTIC` leaves no engine (`GRAPH_NOT_SUPPORTED` in v9, an empty filtered list in v8). The frontend getter and the v8 reader drop note values they do not know, so a list holding only unknown values reads as empty, without `NONDETERMINISTIC`. See [RFC 0025](./rfcs/0025_NumericalNotes.md).

Other differences a cuDNN program can observe:

- A buffer smaller than the note count fails `hipdnnBackendGetAttribute` with `BAD_PARAM`; `cudnnBackendGetAttribute` truncates to the requested count.
- The note type `hipdnnBackendNumericalNote_t` is an `int32_t`, not a C enum, so a value a newer plugin reports passes through an older backend unchanged.
- `Graph::get_numerical_notes_for_engine` is a hipDNN extension; cuDNN's graph API has no per-engine numerical-note getter.
- A UED `numerical_notes` accepts the ten cuDNN note names; a descriptor that names an unknown tag fails to load.

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

Engines verified for stream capture report `SUPPORTS_GRAPH_CAPTURE`, so `select_behavior_notes({SUPPORTS_GRAPH_CAPTURE})` keeps only those engines.

### Determinism

`deselect_numeric_notes({NONDETERMINISTIC})` keeps only engines whose reported numerical notes omit `NONDETERMINISTIC`. A provider claims determinism with an explicit empty list; an engine that declares nothing is reported `NONDETERMINISTIC`, so a missing claim proves nothing. A failed note query for an engine fails the filter instead of passing it.

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
- A built or deserialized plan executes only on handles whose stream is on the device it was built or deserialized for; `execute` with a handle on another device fails, even when the devices have the same properties. cuDNN allows the same finalized plan on another GPU. To use a plan on another device, `serialize` it and `deserialize` it with a handle on that device. See [backend architecture](./user-guides/conceptual/backend-architecture.rst).

## SDPA dropout random stream

Seeded dropout (`dropout_probability`, seed, offset) in every hipDNN SDPA engine that serves it, forward and backward, draws hipDNN's Philox mask, not cuDNN's random stream. Forward and backward must use the same contract: a gradient is correct only when the forward drew its mask under it. The contract is [SEM-015](./SdpaSemanticsRegister.md#sem-015-dropout).

A `dropout_probability` of 0 is the graph without dropout, forward and backward: seed and offset may be given or absent and are not read. cuDNN frontend v1.24.0's backward builds its random-number node from seed and offset whatever the probability, so a cuDNN backward graph at probability 0 needs them. hipDNN treats probability 0 as the graph without dropout in both directions, following the forward, so seed and offset are optional; graphs cuDNN accepts still work. See [SEM-015](./SdpaSemanticsRegister.md#sem-015-dropout).

## SDPA backward engine gaps

The hip-kernel-provider's FlyDSL SDPA backward engines (`hipkernel:flydsl_sdpa_bwd_merged`, `hipkernel:flydsl_sdpa_bwd_split`) are a first version:

- They decline some graphs cuDNN frontend v1.24.0 serves: ALiBi, a bias with a top-left causal or window mask, a window with bottom-right alignment under `padding_mask`, a bias or dBias with `padding_mask`, an explicit dropout mask, a ragged bias or dBias, and a device-resident attention scale. Another engine may serve them. See [COVERAGE.md 9.1](../../../dnn-providers/hip-kernel-provider/flydsl/COVERAGE.md#91-what-the-matcher-accepts-and-declines).
- Their plans cannot be serialized. See [COVERAGE.md 9.3](../../../dnn-providers/hip-kernel-provider/flydsl/COVERAGE.md#93-known-issues).

## hipDNN extensions

- `Graph::is_supported_ext(cudnnHandle_t, std::vector<HeurMode_t> = {FALLBACK})`: returns `OK` if any engine supports the graph, `GRAPH_NOT_SUPPORTED` otherwise. Builds the operation graph if needed; does not create or invalidate plans.
- `Graph::get_name`, `get_io_data_type`, `get_compute_data_type`, `get_intermediate_data_type`.
- `Graph::deserialize(data, enforce_precompiled)` without a handle.
- `HIPDNN_COMPATIBILITY_CUDNN_FRONTEND` macro.
- SDPA forward with a ragged Q over non-paged K/V validates with `seq_len_kv` alone (PyTorch's varlen `seqused_k`); the query lengths come from Q's ragged offset. cuDNN frontend v1.24.0 requires `seq_len_q` as well; cuDNN 9.25 spells the form `cu_seq_len_q` + `seq_len_kv`. See [SEM-028](./SdpaSemanticsRegister.md#sem-028-ragged-q-with-seq_len_kv-alone-non-paged-kv).
- SDPA backward with FP32 Q, K, V, O, dO and gradients: served by the hip-kernel-provider's `hipkernel:flydsl_sdpa_bwd_split` engine. cuDNN documents its SDPA backward for FP16 and BF16 only. See [COVERAGE.md 9.1](../../../dnn-providers/hip-kernel-provider/flydsl/COVERAGE.md#91-what-the-matcher-accepts-and-declines).
- SDPA backward graphs cuDNN frontend v1.24.0 rejects: dropout with a bottom-right causal mask; a bottom-right causal mask or a window with `Sq > Skv` and no `padding_mask`; head dims that are not a multiple of 8, or from 257 to 512; and `s_q = s_kv = 1`. The gfx950 split engine serves them (a head dim that is not a multiple of 8 only with a padded pitch). The merged engine serves such head dims only up to 32 and none above 256, and the gfx11 split engine serves up to 256. See [COVERAGE.md 9.1](../../../dnn-providers/hip-kernel-provider/flydsl/COVERAGE.md#91-what-the-matcher-accepts-and-declines).

Code that must also compile or run against NVIDIA cuDNN cannot use these.

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
`cudnnSetTensorNdDescriptor`, …) is not provided. PyTorch's `ATen/cudnn/Descriptors.h` wraps that API,
and PyTorch's `Conv_v8.cpp` and quantized convolution include it, so they do not compile unmodified.
No CI builds PyTorch against the compatibility layer yet; that is planned for TheRock's PyTorch CI.

### Backend descriptor C API

| Case | Compatibility layer behavior |
|---|---|
| `cudnnBackendCreateDescriptor`, `cudnnBackendDestroyDescriptor`, `cudnnBackendSetAttribute`, `cudnnBackendGetAttribute`, `cudnnBackendFinalize`; `cudnnBackendAttributeName_t` (`CUDNN_ATTR_*`), `cudnnBackendAttributeType_t` (`CUDNN_TYPE_*`) | not declared: code that calls or names them, such as XLA's autotune persistence, does not compile. `cudnnBackendExecute` and `cudnnGetLastErrorString` are declared and pass straight through to hipDNN |
| `cudnnMoeGroupedMatmulMode_t` | not declared; no v0.x builder takes it |
| `cudnnBackendDescriptor_t` | an alias of `hipdnnBackendDescriptor_t`. `get_raw_desc()` returns the hipDNN operation graph, engine, engine configuration, execution plan or variant pack. Tensors, parameter descriptors, operations, heuristics and fallback lists have no hipDNN descriptor, and their `get_raw_desc()` is null. After a failed `build()` (with `NV_CUDNN_DISABLE_EXCEPTION`), `get_raw_desc()` is null too; upstream dereferences a null pointer when the build failed before creating its descriptor |
| Errata `check_shape` | takes the operation's `ManagedOpaqueDescriptor` and a compatibility layer tensor-slot enum instead of a backend descriptor and an attribute name. `check_rule` and `check_errata` are unchanged |

### Enum values

The C enums (`cudnnBackendDescriptorType_t`, `cudnnDataType_t`,
`cudnnPointwiseMode_t`, …) have the cuDNN names but the compatibility layer's own values: numbered from
0 in ASCII order of the enumerator name, with each count sentinel (`*_COUNT`,
`CUDNN_KNOB_TYPE_COUNTS`) one past the last value. Values are append-only within a
`hipdnn_frontend` major version. `cudnnBackendTensorReordering_t` follows the order of the compatibility layer's
`TensorReordering_t`. `CUDNN_RESAMPLE_AVGPOOL` has no value of its own: as in cuDNN, it equals
`CUDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING`. `cudnnStatus_t` (`cudnn_status.h`) is numbered 0 to 14, not
alphabetically (`CUDNN_STATUS_SUCCESS` 0, `CUDNN_STATUS_BAD_PARAM` 3, `CUDNN_STATUS_NOT_SUPPORTED` 9).

Impact: source compatibility only. Code that stores raw cuDNN enum values produced by
real cuDNN, or compares against hard-coded numbers, sees different values.

### Last-error string

`cudnnGetLastErrorString` returns hipDNN's last error on the calling thread and clears it, as
`hipdnnGetLastErrorString` does. A builder failure that the compatibility layer detects without a failing hipDNN call
is not recorded there. When a hipDNN call fails, the compatibility layer reads (and so clears) hipDNN's error and
appends it to the builder's message.
`time_sorted_plan` likewise reads hipDNN's error after a failed profiling call and logs it at info
level instead of leaving it for the next error on the thread.

Impact: the full error of a failed `build()` is in `cudnnException::what()` or the object's
`get_error()`; `cudnnGetLastErrorString` afterwards is normally empty.

### Parameter descriptors

`ConvDescBuilder`, `PointWiseDescBuilder`, `MatMulDescBuilder`, `ReductionDescBuilder`,
`ResampleDescBuilder` and `RngDescBuilder` validate at `build()`; their values reach hipDNN when the
operation graph is built. These differ from cuDNN:

| Case | Compatibility layer behavior |
|---|---|
| NaN propagation on pointwise and resample descriptors | stored and ignored; hipDNN has no equivalent setting |
| Pointwise ReLU clips and lower-clip slope, ELU alpha, softplus beta, swish beta | narrowed to `float` when the graph is built; values beyond the float range saturate to ±`FLT_MAX`, except an upper clip at or above `FLT_MAX` (an infinite one included), which is not sent, as if unset; the result is the same, since such a bound clips no `float`. Parameters left at their defaults are not sent, except Swish β, which is always sent (hipBLASLt's Swish epilogue requires it); plain ReLU therefore reaches hipDNN with no parameters. A matmul followed by plain ReLU or ReLU6 needs a hipBLASLt provider that reads an unset lower clip as 0 (rocm-libraries#14363). A NaN clip, or a non-finite slope, alpha or beta of the descriptor's mode, is `BAD_PARAM` at `PointWiseDescBuilder::build()`; upstream passes it to cuDNN. Infinite clips are accepted |
| Pointwise modes `COS`, `MOD` and `POW` | `NOT_SUPPORTED` at `PointWiseDescBuilder::build()`; hipDNN has no such modes |
| Resample modes `NEAREST` and `BILINEAR`; padding mode `EDGE_VAL_PAD`; a window, padding or stride fraction that is not an integer | `NOT_SUPPORTED` at `ResampleDescBuilder::build()`; a fraction with a zero denominator is `BAD_PARAM` |
| `MatMulDescBuilder::setPaddingValue` | `NOT_SUPPORTED` at `MatMulDescBuilder::build()` |
| `RngDescBuilder` | builds, but an `RNG` operation is `NOT_SUPPORTED` (see [Operations](#operations)) |

### Tensors

`TensorBuilder::build()` validates the tensor. These differ from cuDNN:

| Case | Compatibility layer behavior |
|---|---|
| Byte alignment below 16 (`setAlignment`) | forwarded to hipDNN. A graph with such a tensor needs engine-plugin API 1.3.0 or later, which the MIOpen and hipBLASLt providers do not report; hip-kernel-provider reports it, but its engines decline tensors aligned below 16 bytes. The graph has no engine: `getEngineCount()` is 0 and heuristics return no results, both with `SUCCESS` |
| Vectorized tensors: a vector count above 1 (`setVectorCountAndDimension`); the data types `INT8x4`, `INT8x32` and `UINT8x4`; a reorder type other than `NONE` | `NOT_SUPPORTED` at `build()` |
| Data types `FAST_FLOAT_FOR_FP8`, `COMPLEX_FP32` and `COMPLEX_FP64`, on a tensor or as the compute type of a parameter descriptor (which rejects the vectorized types too) | `NOT_SUPPORTED` at `build()`; hipDNN has no such types |
| A constant (`setConstValue`) whose size is not the element size of the tensor's data type (one byte for `BOOLEAN`) | `BAD_PARAM` at `build()`; upstream passes the stored bytes, and cuDNN reads one element from their start. One case is accepted: a wider integer (for example an `int` literal, or `std::int64_t`) on an `INT8`, `UINT8`, `INT32` or `BOOLEAN` tensor whose value fits that type (`BOOLEAN`: 0 or 1) is stored as one element, the value cuDNN reads. Every other size mismatch stays `BAD_PARAM`: a floating-point one (a `double` on `FLOAT`, a `float` on `HALF`), a narrower constant, an integer of another size on a floating-point tensor and a wider integer that does not fit. Only the size is checked: a constant of the element size is passed as raw bytes, as upstream does, whatever its type, so `setConstValue(1)` on a `FLOAT` tensor builds and reads as the float whose bit pattern is 1 |
| A virtual tensor (`setVirtual`) that is passed by value (`setByValue`) or holds a constant (`setConstValue`) | `TensorBuilder::build()` succeeds, as upstream; `OperationGraphBuilder::build()` fails with `BAD_PARAM` from hipDNN's tensor rule, or first with `NOT_SUPPORTED` for a constant whose type hipDNN cannot carry (see [Operation graphs](#operation-graphs)). Upstream sends both attributes to cuDNN, which does not document the combination |

### Operations

`OperationBuilder::build()` validates the operation. These differ from cuDNN:

| Case | Compatibility layer behavior |
|---|---|
| Operation types the compatibility layer does not lower: `GenStats`, `BNFinalize`, `Dgrad_Drelu_BN_Bwd` (BN backward weights), `Rng`, `Reshape`, paged cache load | `NOT_SUPPORTED` at `build()`, after upstream's checks of the operation's required tensors and descriptors |
| Alpha ≠ 1 or beta ≠ 0 on convolution or resample; alpha ≠ 1 on pointwise (`setAlpha`); alpha2 ≠ 1 (`setAlpha2`) on a pointwise mode with a second input (`B`) or an activation-backward mode | `NOT_SUPPORTED`; hipDNN operations have no scaling factors. A one-input forward pointwise mode ignores alpha2, as cuDNN does |
| Batch-norm backward without saved statistics (`setSavedMeanAndInvVar`) | `BAD_PARAM`; the saved statistics are required |
| Running statistics (`setPrevRunningMeanAndVar`, `setNextRunningMeanAndVar`) outside batch-norm training; a saved mean on RMS norm | `NOT_SUPPORTED` |
| Layer-norm forward without a bias (`setScaleAndBias`) | `BAD_PARAM`; hipDNN's layer norm requires a bias |
| Norm `INSTANCE`, `GROUP`, `ADA_LAYER` modes; peer statistics (`addPeerStatTensor`, `setPeerStatTensor`) | `NOT_SUPPORTED` |
| Max-pool backward resample without an index tensor (`setidxDesc`) | `BAD_PARAM`; hipDNN needs the index tensor |
| Resample with an index tensor (`setidxDesc`), forward or backward, in a mode other than `MAXPOOL` | `NOT_SUPPORTED` at `OperationBuilder::build()`; only max pooling produces and consumes indices |
| Backward resample with `setxDesc` / `setyDesc` | `NOT_SUPPORTED`; hipDNN's backward resample has no such inputs |
| Matmul M/N/K overrides (`setmOverrideDesc`, `setnOverrideDesc`, `setkOverrideDesc`) | `NOT_SUPPORTED` |

### Operation graphs

`OperationGraphBuilder::build()` lowers the graph to hipDNN; it needs no device and runs no heuristics.
These differ from cuDNN:

| Case | Compatibility layer behavior |
|---|---|
| No handle (`setHandle`) | `BAD_PARAM`; hipDNN needs a handle to build the graph (cuDNN 9.8 and later make it optional) |
| `setIsDynamicShapeEnabled(true)` or `setIsOverrideShapeEnabled(true)` | `NOT_SUPPORTED` at `build()`; the v8 surface has no variant-pack shape overrides |
| Norm forward / backward compute type | cuDNN norm operations carry none; the compatibility layer sends `FLOAT` to hipDNN |
| Convolution with `HALF` or `BFLOAT16` tensors and a `FLOAT` compute type | forwarded as `FLOAT`, but the engine chooses the accumulation precision: the MIOpen provider may run a reduced-precision kernel (a 3x3 stride-1 `HALF` forward on gfx1151 runs the Winograd solver `ConvWinoFuryRxS<2-3>` with fp16 accumulation), so results can differ from cuDNN by more than fp32 accumulation would. They match PyTorch's own MIOpen convolution |
| Several tensors with one UID | accepted when they agree; `BAD_PARAM` naming the UID when data type, dimensions, strides, virtualness, by-value or constant value, or ragged offset differ. Differing byte alignments use the smallest |
| One UID written by two operations; a cycle | `BAD_PARAM` |
| A constant value (`setConstValue`) on a tensor whose type is not `FLOAT`, `DOUBLE`, `HALF`, `BFLOAT16`, `UINT8`, `INT32`, `INT64` or `BOOLEAN` | `NOT_SUPPORTED` at graph build |
| hipDNN operation validation | Graph build runs hipDNN's own checks. Beyond cuDNN's: epsilon and the exponential-average factor must be single-element by-value tensors (`setByValue`); batch-norm scale, bias and statistics must be per-channel `[1,C,1,…]` (no per-activation mode) with N·spatial > 1 when training; a reduction must reduce at least one dimension; a max-pool index must be `INT8` or `INT32`; RMS-norm scale must match the trailing dimensions of X. A rejection is `BAD_PARAM` naming the operation type, except "no engine supports this graph", which is `NOT_SUPPORTED`. Backend failures during lowering, allocation included, also report `BAD_PARAM` with hipDNN's message |
| Engine count (`getEngineCount()`) | computed on the graph's first engine query (`getEngineCount()`, heuristics or `EngineBuilder`) by running hipDNN heuristics, which needs a HIP device; a failure returns its status and the next query retries. It counts only engines hipDNN reports as applicable whose default configuration finalizes; an engine that fails is logged and left out, so every counted index builds. The result is a per-graph snapshot: later changes to the heuristic configuration or autotune cache do not affect an existing graph |
| Engine indices | `0 … getEngineCount()-1` enumerate the applicable hipDNN engines in ascending (signed) hipDNN engine-ID order, not cuDNN's global engine numbering. They change if the set of applicable engines changes (provider added, removed or upgraded, different GPU) |

### Engines, plans and execution

These differ from cuDNN:

| Case | Compatibility layer behavior |
|---|---|
| Heuristic mode (`setHeurMode`) | `A` and `INSTANT` rank the graph's engines by hipDNN's `ModeA` policy, `B` by `ModeB`, `FALLBACK` by hipDNN's fallback rank; every mode returns the same engines. Each mode queries hipDNN once per graph, on its first use, so the first `INSTANT` (the builder's default) or `A`/`B` query of a graph costs a second heuristic run. See [Heuristic modes](#heuristic-modes) |
| Heuristic results of `A`, `B` and `INSTANT` | ranked by the policy, but every result uses the engine's default configuration: mode B's predicted knob settings and the ranking metric the policies attach to their configurations are not carried |
| `EngineHeuristics::getEngineConfig(count)` | returns the first `count` results (at most `getEngineConfigCount()`): objects fixed at `build()`, the same on every call. Upstream also returns at most `count` results, but refills its descriptors from cuDNN on each call. Every result for a given engine of a graph, from any heuristics or fallback list, shares that engine's default hipDNN configuration: their `get_raw_desc()` is equal |
| Matmul on gfx115x under Windows | hipBLASLt declines every matmul there (a hipBLASLt GEMM crash workaround), and no other provider runs matmul, so a v8 matmul graph has no engine |
| Engine indices in errata and in consumer code | errata rules and hard-coded engine numbers (for example PyTorch's cuDNN engines 5, 23, 58 and 63) name NVIDIA engines. Under the compatibility layer the same number selects an unrelated hipDNN engine, or none. At `CUDNN_VERSION` 92200 only the 3-D `ConvBwdData` engine-23 rule is active, and it applies only to a graph with more than 23 applicable engines |
| Numerical notes (`hasNumericalNote`, `ExecutionPlan::getNumericNotes`, `ExecutionPlan::getAllNumericNotes`) | read from the backend's per-engine numerical notes. An engine that declares none is reported `NONDETERMINISTIC`; no in-tree engine reports another numerical note, so filters on notes such as `DOWN_CONVERT_INPUTS` or `TENSOR_CORE` never match. Only `hasNumericalNote` answers false when the query fails; for `getNumericNotes`, `getAllNumericNotes` and the plan build, a failed numerical-note query fails `ExecutionPlanBuilder::build()`, as upstream, except that `NOT_SUPPORTED` from an older same-major backend that lacks the attribute reads as `{NONDETERMINISTIC}`. Deterministic-only selection keeps only the engines that claim determinism |
| Behavior notes (`hasBehaviorNote`, `ExecutionPlan::getBehaviorNotes`) | hipDNN behavior notes with a cuDNN counterpart, matched by name; others (for example `SUPPORTS_EXECUTION_PLAN_SERIALIZATION`) are not reported |
| Engine knobs (`Engine::getSupportedKnobs`) | lists only hipDNN knobs that project onto a cuDNN knob type (see [Knobs](#knobs): int64 value, plain minimum/maximum/stride, no list of valid values). A cuDNN type that two of an engine's knobs map onto is not listed (the v9 `get_knobs_for_engine` lists both) |
| Knob choices (`EngineConfigBuilder`) | checked at `EngineConfigBuilder::build()`: a type the engine does not list is `NOT_SUPPORTED`; a type chosen twice, or a value outside the knob's range or off its stride, is `BAD_PARAM`. The same `build()` creates the hipDNN configuration, so a failure there is also reported by `EngineConfigBuilder::build()`, not by `ExecutionPlanBuilder::build()`. A choice of `-1` means "unset" (upstream's `EngineConfigBuilder` sends `-1` for every knob it was not asked to set once one is set): it is ignored (not checked, not applied, not counted as a repeat, not in the plan JSON `"knobs"`) unless the knob is listed and its range accepts `-1`; the plan tag keeps it, also after a JSON round trip (the plan JSON records it under `"requestedKnobs"`). A configuration whose choices are all ignored uses the engine's default hipDNN configuration |
| Plan JSON format (`getJsonRepresentation`, `loadFromJson`) | the compatibility layer's own versioned JSON, not cuDNN's: engine index, engine ID (a decimal string), engine name, behavior notes, numerical notes (always written), knob choices, the requested choices when some were ignored (`"requestedKnobs"`) and, when hipDNN can serialize the plan, a base64 payload. Behavior notes, numerical notes and knob types are all stored as the compatibility layer's own enum values (`detail/c_types.h`; numerical notes are alphabetical, so `NONDETERMINISTIC` is 2), not NVIDIA's numbering; numerical notes map to and from hipDNN's by name. JSON without `numericalNotes` loads as `{NONDETERMINISTIC}`. The format `version` is `1` with `numericalNotes`; a build without that key rejects such a plan (unknown engine key). JSON written by cuDNN or by another tool makes `loadFromJson` fail with `BAD_PARAM` |
| Loading a plan without `"payload"` | `NOT_SUPPORTED` from `loadFromJson`; engines whose plans hipDNN cannot serialize cannot be reloaded |
| Loaded plans | the engine ID must match the one inside the payload (`BAD_PARAM` otherwise). Engine index, name, behavior notes, numerical notes and knob choices are reported from the JSON as saved; the index refers to the engine set of the process that saved it |
| `ExecutionPlan::getWorkspaceSize()` | can differ between builds of the same engine configuration: with the MIOpen provider it follows MIOpen's user find-db, which executions fill. Allocate each plan's workspace from that plan's `getWorkspaceSize()`, not from a size cached by plan tag or engine configuration |
| `cudnnBackendExecute` handle | not checked by the compatibility layer: hipDNN runs the plan on the handle it is given, with that handle's stream, including a handle other than the one the plan's graph was built on or, for a loaded plan, the one it was loaded with. A handle on another device than the one the plan was built or loaded on is rejected by hipDNN's `hipdnnBackendExecute` with `BAD_PARAM` before any provider call. `ExecutionPlanBuilder::setHandle` is ignored by `build()`, as upstream never sends it, and required by `loadFromJson` |
| Variant packs (`VariantPackBuilder::build()`) | the hipDNN variant pack is built here; missing or empty data pointers or UIDs, or the two with different element counts, are `BAD_PARAM` |
| Per-call overhead | `cudnnBackendExecute` is one hipDNN call. `VariantPackBuilder::build()` also builds the hipDNN variant pack on every call. Both stay within the budgets of RFC 0024 §2 over calling hipDNN directly on the same plan, as measured by `samples/cudnn_shim/ConvFpropV8.cpp --overhead` |

### Builders

The v0.x builders (`TensorBuilder`, `ConvDescBuilder`, `PointWiseDescBuilder`, `MatMulDescBuilder`,
`ReductionDescBuilder`, `ResampleDescBuilder`, `RngDescBuilder`, `OperationBuilder`,
`OperationGraphBuilder`, `VariantPackBuilder`, `EngineHeuristicsBuilder`, `EngineBuilder`,
`EngineConfigBuilder`, `EngineFallbackListBuilder`, `ExecutionPlanBuilder`) and the plan helpers
(`get_heuristics_list`, `cudnnGetPlan`, `time_sorted_plan`, `cudnnFindPlan`,
`cudnnFindPlanAndCache`) are upstream's, with the members that called the backend rewritten over
hipDNN. The sections above list their rules; these differ as well:

| Case | Compatibility layer behavior |
|---|---|
| Builder error text | messages name the builder and the rule (for example `OperationBuilder: alpha2 = 0.5 is not supported by the hipDNN cuDNN compatibility layer`), not upstream's attribute names, followed by hipDNN's message when a hipDNN call failed. `cudnnException::what()` ends in `cudnn_status: HIPDNN_STATUS_…`: `to_string(cudnnStatus_t)` returns `cudnnGetErrorString`, which reports hipDNN's name for the translated status. `getCudnnStatus()` and `get_status()` still carry the cuDNN status |
| Retried `build()` | when `TensorBuilder` or a parameter-descriptor builder succeeds after a failed `build()`, the object reports `SUCCESS` and an empty error; upstream keeps the earlier failure's status. Upstream's `TensorBuilder` also refuses to build again after a failure past its prechecks (`BAD_PARAM`, "The tensor already seems to be pointing to something"); the compatibility layer's fixed builder builds, with its ragged offset and constant, also when the failed result was bound by value under `NV_CUDNN_DISABLE_EXCEPTION`. Only a builder that already built a tensor gets `BAD_PARAM` ("the tensor is already built") |
| nlohmann serialization | the compatibility layer defines none of the nlohmann serializers of upstream's `cudnn_frontend_utils.h` (frontend enums, `float`, `half`, `nv_bfloat16`, the scalar variant, `std::optional`, `std::shared_ptr`, `cudnnFraction_t`). `DataType_t`, `PointwiseMode_t` and the other frontend enums are hipDNN's enums, so `json j = DataType_t::FLOAT;` stores an integer, not upstream's `"FLOAT"`, and a `float` is a JSON number, not upstream's 8-hex-digit string; the other types serialize only as far as nlohmann itself supports them. `describe()` is unaffected: it renders enums through the compatibility layer's own name table, byte for byte as upstream |
| `json` alias | declared in the compatibility layer's namespace, as `cudnn_frontend::json` (`nlohmann::json`); upstream declares `json` at global scope. Code that names bare `json`, as upstream's legacy samples do, writes `cudnn_frontend::json` or `nlohmann::json`, or declares its own alias (`using json = nlohmann::json;` compiles with both NVIDIA's header and the compatibility layer). With `using namespace cudnn_frontend;` in scope, an unqualified `json` is ambiguous when the consumer's own global alias names another type (for example `nlohmann::ordered_json`); such code names its alias as `::json` |
| `HIPDNN_FRONTEND_SKIP_JSON_LIB` | a hipDNN built with this option passes the definition to every consumer of `hipdnn_frontend`, and the compatibility layer then defines `CUDNN_FRONTEND_SKIP_JSON_LIB`. The consumer gets upstream's JSON-less surface (no `json`, `load_from_config`, `check_rule` or `check_errata`; descriptor log lines print enums as integers) without defining that macro itself |
| Descriptor log lines | the body of a builder's log line (`describe()`) is printed only when `CUDNN_FRONTEND_LOG_INFO` is above 0, as upstream, and the line itself goes through hipDNN INFO logging. With hipDNN INFO logging on but `CUDNN_FRONTEND_LOG_INFO` unset, the line is `[cudnn_frontend] ` with nothing after it |
| `getStream()` / `getLogger()` | no compatibility layer code calls them. A consumer call opens `CUDNN_FRONTEND_LOG_FILE` for writing, truncating it as upstream does; the `cudnnCreate()` logging bridge may also route hipDNN's log to that file, so the two writers can overwrite each other |
| `NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING` | ignored: the compatibility layer never loads cuDNN at run time, and `NV_CUDNN_FE_DYNAMIC_CHECK_*` expand to nothing, so the run-time cuDNN version checks they guard are skipped |
| `get_environment()`, `getLogLevel()` | `get_environment` returns a pointer into a per-thread buffer that the next call on the same thread overwrites (upstream returns `getenv`'s pointer), and returns null for a variable that is set but empty. `getLogLevel` returns 0 when the `NV_CUDNN_FRONTEND_DISABLE_LOGGING` environment variable is set to anything but `0`; upstream honors only the macro |
| PyTorch fused convolutions | `cudnn_convolution_relu` passes `alpha2 = 0` to the pointwise add, so it always gets `NOT_SUPPORTED`; `cudnn_convolution_add_relu` does whenever its alpha is not 1 (see the scaling row under [Operations](#operations)) |
| Heuristic mode strings | every mode returns the same engines, ordered per mode (see [Heuristic modes](#heuristic-modes)), so a mode list such as `{"heuristics_instant", "heuristics_fallback"}` given to `get_heuristics_list` or `EngineConfigGenerator` yields every engine configuration once per mode. `cudnnGetPlan` and `time_sorted_plan` build and time the duplicates, and they count against a plan or benchmark limit |
| `get_heuristics_list` with a `HeurMode_t` list | `OPENSOURCE` runs fallback selection and logs a warning on its first use in a process; upstream throws `std::out_of_range`. A value outside `HeurMode_t` throws `std::out_of_range`, as upstream |
| `get_fallback_engine_list` | returns upstream's table of NVIDIA engine numbers; under the compatibility layer those numbers select unrelated hipDNN engines or none (see the engine-indices row under [Engines, plans and execution](#engines-plans-and-execution)) |
| `setKernelCache`, `setDeviceProperties`, `setSMCount` | a non-null kernel cache or device properties, or a positive SM count, makes `build()` fail with `NOT_SUPPORTED`; an SM count of 0 is accepted as the default |
| `cudnnReorderFilterAndBiasInt8x32` | checks its arguments as upstream, then returns `NOT_SUPPORTED`: the compatibility layer has no int8x32 reordering |
| Plan tags | the `_k<type>=<value>` segments of `ExecutionPlan::getTag()` use the compatibility layer's knob-type integers, so errata `"knob"` entries written against cuDNN's integers do not match |
| `time_sorted_plan` timing | measured with hipDNN profiling events on the handle's stream: device time between the start and stop of each timed execution, as upstream's CUDA events. A plan whose warm-up execution fails is dropped, as upstream. A plan whose timed execution, or whose timing, fails on any run is kept with time `FLT_MAX`, after every measured plan, so `cudnnFindPlanAndCache` prefers any measured plan to it; upstream ignores the timed execution's status and its CUDA event errors and keeps the measured span, the previous time or 0 |
| `time_sorted_plan` ties | plans with equal times, including plans that could not be timed, are all kept in input order; upstream keeps only one of them |
| `cudnnFindPlanAndCache` with no plan that ran | throws `cudnnException` with `NOT_SUPPORTED`; with `NV_CUDNN_DISABLE_EXCEPTION` it returns an unbuilt plan whose status is `BAD_PARAM` (no engine configuration). Upstream reads the front of an empty list |
| PyTorch quantized operations across threads | PyTorch's quantized convolution, linear and add operations cache plans process-wide, so a plan cached on one thread can be executed with another thread's handle; it runs on that handle (see the `cudnnBackendExecute` handle row). PyTorch's `Conv_v8` caches are per-thread and unaffected |
| `ConvDescBuilder` default mode | upstream's default is `CONVOLUTION`; MIOpen runs only `CROSS_CORRELATION`, so a graph that keeps the default has no engine. Call `setMathMode(CUDNN_CROSS_CORRELATION)` |
| `getMaxValue()` of a knob from `Engine::getSupportedKnobs()`, `RngDesc::getUniformDistMin()` | return the knob's maximum and the uniform distribution's minimum; upstream returns the knob's minimum and the normal distribution's standard deviation |
