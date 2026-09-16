# hipBLASLt horizontal C API and ABI specification

Status: scope inventory for the API and ABI entry gate. This specification covers exported declarations inside `extern "C"` blocks in the current public hipBLASLt headers. It does not cover the bespoke C++ API in `hipblaslt-ext.hpp`.

## Scope rule

The horizontal workstream must give every listed function an explicit disposition before implementation begins:

- **MVP:** implemented by the 13-function Phase 1 slice;
- **Horizontal:** implemented during Phase 2;
- **Remove:** intentionally removed through an approved API and ABI transition; or
- **Retain elsewhere:** remains available through a separately identified compatibility mechanism.

No exported C-linkage declaration may be omitted merely because it is documented as internal. Exported internal-looking functions still require an ABI disposition.

## Inventory summary

| Header | Exported C-linkage functions | Phase 1 MVP | Remaining disposition |
| --- | ---: | ---: | ---: |
| `projects/hipblaslt/library/include/hipblaslt/hipblaslt.h` | 29 | 13 | 16 |
| `projects/hipblaslt/library/include/hipblaslt/hipblaslt-ext-op.h` | 16 | 0 | 16 |
| **Total** | **45** | **13** | **32** |

The API snapshot currently inventories the 29 functions from `hipblaslt.h` but does not inventory the 16 exported C-linkage declarations in `hipblaslt-ext-op.h`. Closing that extraction gap is part of the entry gate.

## Phase 1 MVP functions

| Function | Signature summary | Disposition |
| --- | --- | --- |
| `hipblasLtCreate` | `(hipblasLtHandle_t*) -> hipblasStatus_t` | MVP |
| `hipblasLtDestroy` | `(hipblasLtHandle_t) -> hipblasStatus_t` | MVP |
| `hipblasLtMatrixLayoutCreate` | `(layout*, datatype, rows, columns, ld) -> hipblasStatus_t` | MVP |
| `hipblasLtMatrixLayoutDestroy` | `(layout) -> hipblasStatus_t` | MVP |
| `hipblasLtMatrixLayoutSetAttribute` | `(layout, attribute, buffer, size) -> hipblasStatus_t` | MVP, selected attributes in Phase 1 |
| `hipblasLtMatmulDescCreate` | `(descriptor*, compute_type, scale_type) -> hipblasStatus_t` | MVP |
| `hipblasLtMatmulDescDestroy` | `(descriptor) -> hipblasStatus_t` | MVP |
| `hipblasLtMatmulDescSetAttribute` | `(descriptor, attribute, buffer, size) -> hipblasStatus_t` | MVP, selected attributes in Phase 1 |
| `hipblasLtMatmulPreferenceCreate` | `(preference*) -> hipblasStatus_t` | MVP |
| `hipblasLtMatmulPreferenceDestroy` | `(preference) -> hipblasStatus_t` | MVP |
| `hipblasLtMatmulPreferenceSetAttribute` | `(preference, attribute, buffer, size) -> hipblasStatus_t` | MVP, selected attributes in Phase 1 |
| `hipblasLtMatmulAlgoGetHeuristic` | `(handle, operation, A/B/C/D layouts, preference, requested_count, results, result_count) -> hipblasStatus_t` | MVP, bounded profile in Phase 1 |
| `hipblasLtMatmul` | `(handle, operation, alpha, A/B/C/D, layouts, beta, algorithm, workspace, stream) -> hipblasStatus_t` | MVP, one frozen GEMM in Phase 1 |

Phase 2 must complete the attribute and operation domains of MVP functions; exporting a function does not establish complete behavioral coverage for every value it accepts.

## Remaining base C functions

| Function | Signature summary | Horizontal group |
| --- | --- | --- |
| `hipblasLtMatrixLayoutGetAttribute` | `(layout, attribute, buffer, size, written_size*) -> hipblasStatus_t` | Objects and attributes |
| `hipblasLtMatmulDescGetAttribute` | `(descriptor, attribute, buffer, size, written_size*) -> hipblasStatus_t` | Objects and attributes |
| `hipblasLtMatmulPreferenceGetAttribute` | `(preference, attribute, buffer, size, written_size*) -> hipblasStatus_t` | Objects and attributes |
| `hipblasLtMatrixTransformDescCreate` | `(transform_descriptor*, scale_type) -> hipblasStatus_t` | Matrix transform |
| `hipblasLtMatrixTransformDescDestroy` | `(transform_descriptor) -> hipblasStatus_t` | Matrix transform |
| `hipblasLtMatrixTransformDescSetAttribute` | `(transform_descriptor, attribute, buffer, size) -> hipblasStatus_t` | Matrix transform |
| `hipblasLtMatrixTransformDescGetAttribute` | `(transform_descriptor, attribute, buffer, size, written_size*) -> hipblasStatus_t` | Matrix transform |
| `hipblasLtMatrixTransform` | `(handle, transform_descriptor, alpha, A/layout, beta, B/layout, C/layout, stream) -> hipblasStatus_t` | Matrix transform |
| `hipblasLtGetVersion` | `(handle, version*) -> hipblasStatus_t` | Information and configuration |
| `hipblasLtGetGitRevision` | `(handle, revision_buffer) -> hipblasStatus_t` | Information and configuration |
| `hipblasLtGetArchName` | `(architecture_name**) -> hipblasStatus_t` | Information and configuration |
| `hipblasLtSetSmCountTarget` | `(handle, sm_count) -> hipblasStatus_t` | Information and configuration |
| `hipblasLtGetSmCountTarget` | `(handle, sm_count*) -> hipblasStatus_t` | Information and configuration |
| `hipblasLtSetUniformSummationOrder` | `(handle, enabled) -> hipblasStatus_t` | Information and configuration |
| `hipblasLtGetUniformSummationOrder` | `(handle, enabled*) -> hipblasStatus_t` | Information and configuration |
| `hipblasLtCheckNumericsDrain` | `(handle, abnormal*) -> hipblasStatus_t` | Information and configuration |

## Extension-operation C functions

These functions are exported with C linkage by `hipblaslt-ext-op.h`. The first three are documented extension operations. The remaining thirteen are described by the header as client performance arguments or internal-use queries, but they are still exported declarations and require an API and ABI disposition.

| Function | Signature summary | Required entry-gate decision |
| --- | --- | --- |
| `hipblasltExtSoftmax` | `(datatype, m, n, dim, output, input, stream) -> hipblasStatus_t` | Horizontal implementation or approved exclusion |
| `hipblasltExtLayerNorm` | `(datatype, output, mean, invvar, input, m, n, epsilon, gamma, beta, stream) -> hipblasStatus_t` | Horizontal implementation or approved exclusion |
| `hipblasltExtAMax` | `(input_datatype, output_datatype, output, input, m, n, stream) -> hipblasStatus_t` | Horizontal implementation or approved exclusion |
| `hipblasltSetFlushValue` | `(bool) -> void` | Public, retain-elsewhere, or removal decision |
| `hipblasltSetRotatingBufferSizeValue` | `(int) -> void` | Public, retain-elsewhere, or removal decision |
| `hipblasltSetColdIterationsValue` | `(int) -> void` | Public, retain-elsewhere, or removal decision |
| `hipblasltSetHotIterationsValue` | `(int) -> void` | Public, retain-elsewhere, or removal decision |
| `hipblasltGetTotalGranularityValue` | `() -> double` | Public, retain-elsewhere, or removal decision |
| `hipblasltGetTilesPerCuValue` | `() -> double` | Public, retain-elsewhere, or removal decision |
| `hipblasltGetTile0Granularity` | `() -> double` | Public, retain-elsewhere, or removal decision |
| `hipblasltGetTile1Granularity` | `() -> double` | Public, retain-elsewhere, or removal decision |
| `hipblasltGetCuGranularity` | `() -> double` | Public, retain-elsewhere, or removal decision |
| `hipblasltGetWaveGranularity` | `() -> double` | Public, retain-elsewhere, or removal decision |
| `hipblasltGetCUs` | `() -> int` | Public, retain-elsewhere, or removal decision |
| `hipblasltGetMemWriteBytesD` | `() -> size_t` | Public, retain-elsewhere, or removal decision |
| `hipblasltGetMemReadBytes` | `() -> size_t` | Public, retain-elsewhere, or removal decision |

## Required closure before implementation

For every function above, the entry gate must record:

1. its disposition and owning PR;
2. its exact exported symbol and ABI version;
3. facade-owned versus provider-owned behavior;
4. public state, ownership, lifetime, and concurrency semantics;
5. private protocol operations or records required;
6. legacy behavior and status mapping;
7. positive, negative, ABI, and differential tests; and
8. supported device and platform requirements.

The inventory is complete only when the checked public headers, generated API snapshot, export inventory, this specification, and task graph agree.
