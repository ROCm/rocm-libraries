# RFC: cuDNN shim for hipDNN

- **Status**: Accepted
- **Authors**: Mitch Ousdahl

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Requirements](#2-requirements)
3. [Glossary](#3-glossary)
4. [Proposed Design](#4-proposed-design)
5. [Key Design Decisions](#5-key-design-decisions)
6. [Risks](#6-risks)
7. [Execution Plan](#7-execution-plan)
8. [Testing Plan](#8-testing-plan)
9. [Future Considerations](#9-future-considerations)

## 1. Executive Summary

### Why

There are many barriers to adoption of a new API interface, especially in the
cases where an incumbent already exists. In our case we are not only working
against an incumbent competitor interface (cuDNN), but also our own older
interface (MIOpen). This RFC documents the process to aid in integrating with
consumer code with cuDNN incumbency; another RFC will address MIOpen incumbency.
A direct example from PyTorch is our initial motivator. The difficulty in
creating a hipDNN-native backend for PyTorch has been met with a great deal of
administration friction, whereas using hipify (or an equivalent) to convert an
existing cuDNN backend is seen as much lower risk and therefore more desirable.

### What

This RFC describes the addition of a cuDNN-equivalent frontend for hipDNN. This
will be implemented as an aid to conversion of cuDNN frontend v9 consuming code
to be 'hipified' to instead consume hipDNN. The new cuDNN-style frontend graph
will primarily forward work to a wrapped hipDNN frontend graph object. In cases
where it makes sense, missing cuDNN functionality that can be replicated by
alternative hipDNN interfaces will be implemented inside the wrapper. In cases
where missing cuDNN functionality has no current implementation, one will either
be created in hipDNN to match or provide an adequate error.

### Not

This RFC supports cuDNN frontend **v9 translation units only**. The shim's
`cudnn_frontend.h` is a hand-curated v9-only umbrella; it does **not**
reconstruct the v0.x / v8 interface (not even as compile-only stubs), so
source using v0.x types (`TensorBuilder`, `OperationBuilder`,
`cudnnException`, etc. — e.g. PyTorch's `Conv_v8.cpp` and the quantized
cuDNN ops) will not compile against it. The only carve-out is where a v9
signature itself names a v0.x symbol: the shim provides just enough to keep
v9 TUs compiling — chiefly the small C-API stub `cudnn.h`
(`cudnnHandle_t`, `cudnnStatus_t`, etc.; see §4.7). A v0.x stub layer is a
candidate follow-up RFC (§9).

Also out of scope: **ABI** compatibility (consumers must recompile) and
**Python bindings** — this is a C++-only shim, and the motivating consumer
(PyTorch) uses the C++ API directly. A Python-binding layer would be a
separate future RFC (§9).

## 2. Requirements

- **Header-Only**: The interface must be entirely header-only.
- **Interface parity**: Must supply the cuDNN frontend **v9 graph API**
  surface, **plus all additional API needed to use it** (the C-API types and
  the handful of handle/error/version entry points it depends on — §4.7).
  This is intentionally phrased as "the V9 API and what it needs to run"
  rather than "100% of everything cuDNN FE ships", since the v0.x surface is
  explicitly out of scope (§1).
    - Some small wiggle-room is expected for things that don't map cleanly
      (heuristic mode enums, for example).
- **Functional parity**: Must supply as much of cuDNN's functionality as
  possible.
    - Any un-implemented portions of that surface area must be at least backed
      with a compile-time `static_assert` (or, where the call must remain
      reachable at runtime for hipification to compile, with a runtime error
      that maps to a documented `HIPDNN_FRONTEND_*` status code).
    - No extension methods from hipDNN will be extended into this interface.
    - We target the **cuDNN frontend v1.24 release** (the upstream
      NVIDIA/cudnn-frontend library version — *not* a hipDNN release
      version; the two are independent, see §4.8). Pin details below.
- **Performance parity**: Must minimally add overhead above-and-beyond
  hipDNN's existing overhead. Concrete budget: **< 1% additional
  `build()` time, < 1 µs added per `execute()`**, both measured against
  native `hipdnn_frontend` for the same graph. See §8.4 for how this is
  enforced.
- **MIT Licensed**: With credit to NVIDIA where appropriate (e.g., for any
  comments, enum value names, or documentation that closely mirror upstream
  cuDNN frontend headers).

The pinned upstream reference is cuDNN FE **v1.24.0**, verified in
`include/cudnn_frontend_version.h`:
`CUDNN_FRONTEND_MAJOR_VERSION=1`, `MINOR=24`, `PATCH=0`,
`CUDNN_FRONTEND_VERSION = 12400`.

## 3. Glossary

- **cuDNN**: NVIDIA's implementation of a graph-based API for processing tensor
  data.
- **cuDNN frontend**: NVIDIA's MIT-licensed header-only C++ wrapper around the
  cuDNN backend C API
  ([github.com/NVIDIA/cudnn-frontend](https://github.com/NVIDIA/cudnn-frontend)).
  Hereafter just "cuDNN FE" when context is clear.
- **frontend**: Describes the interface that a consumer of an API would
  primarily interface with. In hipDNN / cuDNN's world, this is a header-only
  library that lifts / lowers data to a backend component.
- **backend**: Describes the component with a common C interface that does the
  heavy lifting of managing object state, plugins, and descriptor components.
- **descriptor**: A descriptor is a consumer-opaque object. The backend provides
  a common C interface for managing lifecycle and properties for any descriptor.
- **plugin**: In hipDNN's case, this refers to an engine provider plugin that
  does the work of processing tensor data.
- **shim**: The translation layer described by this RFC; cuDNN-shaped C++ types
  that wrap and forward to `hipdnn_frontend`.
- **hipify**: AMD's source-to-source CUDA→HIP conversion tooling. In this RFC,
  used loosely to mean "the process of rewriting cuDNN identifiers to hipDNN
  identifiers", regardless of the specific tool.
- **v0.x API / v8 backend descriptor API**: cuDNN frontend's pre-v1 C++
  surface, built around builder classes (`TensorBuilder`,
  `OperationBuilder`, `ExecutionPlanBuilder`, `VariantPackBuilder`,
  `EngineConfigGenerator`, `EngineHeuristicsBuilder`, etc.) that wrap the
  underlying cuDNN v8 backend descriptor C API
  (`cudnnBackendCreateDescriptor`, `cudnnBackendSetAttribute`,
  `cudnnBackendFinalize`). These types still ship in upstream
  `cudnn_frontend.h` as `using` aliases for `*_v8` legacy classes, even
  in v1.24.
- **v9 graph API**: cuDNN frontend's v1.x C++ surface (also referred to
  by NVIDIA as "FE 1.0" and the cuDNN v9 graph API). Built around
  `cudnn_frontend::graph::Graph` and the `*_attributes` family. This is
  the primary (and only) target of this RFC.

## 4. Proposed Design

### 4.1 Layout and packaging

The shim will primarily reside under
`projects/hipdnn/frontend/include/compatibility/cudnn/` and be exposed as an
exported header-only CMake target. The objects will be namespaced
`hipdnn_frontend::compatibility::cudnn_frontend` (referred to below as
`<shim_ns>`), mirroring upstream's `cudnn_frontend` namespace as closely as
possible so a consumer can write `namespace cudnn_frontend =
hipdnn_frontend::compatibility::cudnn_frontend;` and have nested namespaces
(`graph`, `detail`, eventually `experimental`) line up name-for-name. Any
internal details will be placed in a `detail/` subdirectory and additionally
namespaced as `<shim_ns>::detail`.

Proposed source layout (filenames mirror upstream cuDNN FE exactly, all
`.h` per §5.1):

```
projects/hipdnn/frontend/include/compatibility/cudnn/
    cudnn.h                         # Stub: v9-required C-API types only (see §4.7)
    cudnn_frontend.h                # Umbrella; matches upstream filename
    cudnn_frontend_version.h
    cuda_runtime_compat.h           # OPT-IN; not auto-included
    cudnn_frontend/                 # Mirrors upstream include/cudnn_frontend/
        graph_helpers.h
        graph_interface.h
        graph_properties.h
        node_interface.h
        plans.h
        ...
    detail/                         # Shim-internal; namespaced <shim_ns>::detail
        type_mapping.h              # cuDNN enum/struct ↔ hipDNN mapping
        status_translation.h        # cudnnStatus_t ↔ hipdnnStatus_t
        graph_wrapper.h             # Wraps hipdnn_frontend::graph::Graph
        tensor_wrapper.h
        node_wrappers/...
        handle_wrapper.h
        logging_bridge.h
```

Per the aliasing-first strategy (§4.3.1), much of the `detail/` wrapper set
above is expected to be **small**: most public headers
(`cudnn_frontend/*.h`) reduce to thin umbrellas of `using` declarations
that re-export 1:1 hipDNN types, and `detail/` carries real code only for
the composition wrapper (`graph_wrapper.h`), the residual non-aliasable
enum mappings (`type_mapping.h`), and the error/logging bridges. The
`node_wrappers/` directory holds Tier-2 error-stubs and Tier-3 composition
wrappers only — not one file per node.

A separate, **opt-in** header,
`<compatibility/cudnn/cuda_runtime_compat.h>`, will provide `#define`
overrides for the small fixed set of CUDA runtime symbols that frequently
appear in cuDNN-consuming code and have direct HIP equivalents:

| CUDA symbol | HIP equivalent |
|-------------|---------------|
| `cudaMalloc` | `hipMalloc` |
| `cudaFree` | `hipFree` |
| `cudaMemcpy` / `cudaMemcpyAsync` | `hipMemcpy` / `hipMemcpyAsync` |
| `cudaStream_t` / `cudaStreamCreate` / `cudaStreamDestroy` / `cudaStreamSynchronize` | `hipStream_t` / `hipStreamCreate` / `hipStreamDestroy` / `hipStreamSynchronize` |
| `cudaEvent_t` / `cudaEventCreate` / `cudaEventDestroy` / `cudaEventRecord` | `hipEvent_t` / `hipEventCreate` / `hipEventDestroy` / `hipEventRecord` |
| `cudaGetDevice` / `cudaSetDevice` | `hipGetDevice` / `hipSetDevice` |
| `cudaError_t` / `cudaSuccess` | `hipError_t` / `hipSuccess` |

This header is pinned to that set. New symbols are added only on
documented downstream-consumer demand. The header is **not** included by
the umbrella shim header — consumers must `#include` it explicitly to
opt in. This avoids polluting the global macro namespace for projects
that already use a different HIP-shim strategy (e.g., hipify-perl
conversion, or manually `#ifdef`'d code).

### 4.2 Target audience and conversion path

The shim is designed for two distinct workflows, both of which require
the consumer source to be v9-only (per §1):

1. **Textual hipification** — a user (or tool) globally rewrites
   `cudnn_frontend::` → `hipdnn_frontend::compatibility::cudnn_frontend::`
   (or, via `namespace cudnn_frontend =
   hipdnn_frontend::compatibility::cudnn_frontend;`, leaves the in-code
   symbols unchanged) and swaps `#include <cudnn_frontend.h>` for the shim
   header.
2. **Gradual migration** — a downstream project (e.g., PyTorch's
   `aten/src/ATen/native/cudnn/MHA.cpp`) keeps a single source tree that
   compiles against either real cuDNN or the shim, with their existing
   platform-gating mechanism (typically `__HIP_PLATFORM_AMD__` /
   `USE_ROCM`) selecting which include path is active. Wiring this is the
   consumer's responsibility; the shim takes no stance on which flag is
   used.

To support workflow (2), the shim's public symbols must be name-for-name and
signature-for-signature compatible with cuDNN FE within the limits described
in §4.7. To support workflow (1), the umbrella header must be includable by
itself and must not require any additional `#define`s or CMake variables to
function.

Consumer source that mixes v9 graph API calls with v0.x backend
descriptor calls in the same TU (e.g., PyTorch's `Conv_v8.cpp`) is not
supported by this shim. Such files must either be rewritten to v9 first,
or wait for the follow-up v0.x stub RFC (§9). Consumer projects can keep
v0.x TUs in their tree as long as those TUs are excluded from the build
when the shim is in use.

### 4.3 Namespace and type mapping

The top-level cuDNN FE namespace is `cudnn_frontend`. The shim's top-level
namespace is `<shim_ns>` =
`hipdnn_frontend::compatibility::cudnn_frontend`, chosen so the leaf
matches upstream's name and `namespace cudnn_frontend = <shim_ns>;` is the
only aliasing step a consumer needs. Nested namespaces line up
correspondingly: `<shim_ns>::graph` ↔ `cudnn_frontend::graph`,
`<shim_ns>::detail` ↔ `cudnn_frontend::detail`, and (if/when wrapped)
`<shim_ns>::experimental` ↔ `cudnn_frontend::experimental`. Performing the
alias is the consumer's responsibility — the shim does not ship a
`namespace cudnn_frontend = ...` line so as not to collide with consumers
that include real cuDNN FE elsewhere in the same translation unit.

cuDNN FE uses two parallel type families. **C-API types** (`cudnnHandle_t`,
`cudnnStatus_t`, `cudnnDataType_t`, `cudnnTensorFormat_t`,
`cudnnConvolutionMode_t`, `cudnnReduceTensorOp_t`, `cudnnNormFwdPhase_t`,
`cudnnBackendHeurMode_t`, `cudnnBackendNumericalNote_t`,
`cudnnBackendBehaviorNote_t`, `cudnnBackendDescriptorType_t`, ...) come from
`<cudnn.h>`, which `cudnn_frontend.h` includes (`cudnn_frontend.h:106`).
**FE-native C++ enums** in `namespace cudnn_frontend` (verified in
`include/cudnn_frontend_utils.h`) are: `BuildPlanPolicy_t` (line 334),
`TensorReordering_t` (348), `ResampleMode_t` (363), `PaddingMode_t` (383),
`ReshapeMode_t` (391), `ConvolutionMode_t` (405), `NormFwdPhase_t` (426),
`MoeGroupedMatmulMode_t` (440), `DescriptorType_t` (456), `NormMode_t` (503),
`PointwiseMode_t` (525), `HeurMode_t` (635), `BehaviorNote_t` (650),
`NumericalNote_t` (670), `DataType_t` (700), `ReductionMode_t` (751),
`RngDistribution_t` (779), `DiagonalAlignment_t` (996),
`AttentionImplementation_t` (1003).

| Family | Symbol | Shim mapping |
|--------|--------|--------------|
| C-API (`<cudnn.h>`) | `cudnnHandle_t` | Alias declared in the shim's stub `cudnn.h`: `using cudnnHandle_t = hipdnn_frontend::hipdnnHandle_t;` — see §4.7. |
| C-API | `cudnnStatus_t`, `cudnnTensorFormat_t`, `cudnnConvolutionMode_t`, `cudnnDataType_t`, `cudnnBackend*_t` | Defined in the shim's stub `cudnn.h` (see §4.7), mapped to hipDNN equivalents on the way through |
| FE namespace | `cudnn_frontend::DataType_t` | **Aliased** to `hipdnn_frontend::DataType_t`. cuDNN has 20 values incl. FP8/FP4/INT4 variants; any the hipDNN enum is missing are added to it so the alias stays a name-superset (§4.3) |
| FE namespace | `cudnn_frontend::HeurMode_t` | **Aliased** to `hipdnn_frontend::HeurMode_t` — values `A`, `B`, `FALLBACK`, `OPENSOURCE` (added on the hipDNN side if missing). See §4.5 for runtime behavior |
| FE namespace | `cudnn_frontend::PointwiseMode_t` | **Aliased** to `hipdnn_frontend::PointwiseMode_t` |
| FE namespace | `cudnn_frontend::ReductionMode_t` | **Aliased** to `hipdnn_frontend::ReductionMode_t` |
| FE namespace | `cudnn_frontend::ResampleMode_t`, `PaddingMode_t`, `ConvolutionMode_t`, `NormFwdPhase_t`, `NormMode_t`, `RngDistribution_t`, `DiagonalAlignment_t`, `AttentionImplementation_t`, `MoeGroupedMatmulMode_t`, `BuildPlanPolicy_t`, `TensorReordering_t`, `ReshapeMode_t`, `DescriptorType_t` | **Aliased** to the hipDNN `_t` equivalents; missing values added on the hipDNN side to keep each a name-superset |
| FE namespace | `cudnn_frontend::NumericalNote_t` | **Aliased** to `hipdnn_frontend::NumericalNote_t` — values: `NOT_SET`, `TENSOR_CORE`, `DOWN_CONVERT_INPUTS`, `REDUCED_PRECISION_REDUCTION`, `FFT`, `NONDETERMINISTIC`, `WINOGRAD`, `WINOGRAD_TILE_4x4`, `WINOGRAD_TILE_6x6`, `WINOGRAD_TILE_13x13`, `STRICT_NAN_PROP` (added on the hipDNN side as needed). Per-note *handling* is triaged in §4.5 |
| FE namespace | `cudnn_frontend::BehaviorNote_t` | **Aliased** to `hipdnn_frontend::BehaviorNote_t` — values: `NOT_SET`, `RUNTIME_COMPILATION`, `REQUIRES_FILTER_INT8x32_REORDER`, `REQUIRES_BIAS_INT8x32_REORDER`, `SUPPORTS_CUDA_GRAPH_NATIVE_API`, `CUBLASLT_DEPENDENCY` (added on the hipDNN side as needed). These are CUDA-specific; per-note handling is triaged in §4.5 |
| FE namespace | `cudnn_frontend::error_code_t` | **Aliased** to `hipdnn_frontend::error_code_t`; the cuDNN-only values are added to hipDNN's enum so it is a name-superset (§4.6). 16 values incl. `OK`, `ATTRIBUTE_NOT_SET`, `SHAPE_DEDUCTION_FAILED`, `INVALID_TENSOR_NAME`, `INVALID_VARIANT_PACK`, `GRAPH_NOT_SUPPORTED`, `GRAPH_EXECUTION_PLAN_CREATION_FAILED`, `GRAPH_EXECUTION_FAILED`, `HEURISTIC_QUERY_FAILED`, `UNSUPPORTED_GRAPH_FORMAT`, `CUDA_API_FAILED`, `CUDNN_BACKEND_API_FAILED`, `INVALID_CUDA_DEVICE`, `HANDLE_ERROR`, `INVALID_VALUE`, `NVRTC_COMPILATION_FAILED` (verified at `graph_helpers.h:36`) |
| FE namespace | `cudnn_frontend::error_object` (struct) | **Aliased** to `hipdnn_frontend::error_object`, with the full `error_code_t` value set present on the hipDNN side (§4.6) so the type is 1:1. Public members: `code` (field), `err_msg` (field), `get_code()`, `get_message()`, `is_good()`, `is_bad()`, `operator==(error_code_t)`, `operator!=(error_code_t)`. Verified at `graph_helpers.h:55`. |
| FE namespace | `cudnn_frontend::error_t` | **Aliased** to `hipdnn_frontend::error_t` (typedef for `error_object`). Both names must be exposed by the shim. |
| `cudnn_frontend::graph` | `Graph` | Class wrapping `hipdnn_frontend::graph::Graph` (see §4.4) |
| `cudnn_frontend::graph` | `Tensor_attributes` | Wraps `hipdnn_frontend::TensorAttributes`. Verified at `graph_properties.h:49` |
| `cudnn_frontend::graph` | All `*_attributes` classes | Wrappers; defined in upstream `graph_properties.h`. Full list verified there — see §4.4.2 |
| `cudnn_frontend::graph` | `*Node` / `INode` / `NodeCRTP` (`graph_properties.h:455`, `node_interface.h:45`) | Internal; nominally in the public namespace but not user-facing (verified — no sample / consumer constructs them directly). Not wrapped initially. Since hipDNN has matching internals, these *could* be aliased for completeness later, but that is not required for any known consumer. |
| FE namespace, v0.x | `ConvDesc`, `ConvDescBuilder`, `EngineHeuristics`, `EngineConfig`, `EngineFallbackList`, `ResampleDesc`, ... | `using` aliases for `*_v8` legacy types (verified at `cudnn_frontend.h:137-152`). These are pulled into the umbrella header unconditionally. See §4.7 for the v0.x compatibility problem. |

These FE-namespace enums are **not** re-declared by the shim at all — they
are **aliased**. hipDNN already publishes cuDNN-named `_t` typedefs for them
(`hipdnn_frontend::DataType_t`, `HeurMode_t`, `ConvolutionMode_t`,
`PointwiseMode_t`, `ReductionMode_t`, `BehaviorNote_t`, `BuildPlanPolicy_t`,
`NormFwdPhase_t`, …, several carrying an explicit "Matches the
cudnn-frontend … for API compatibility" comment in
`frontend/include/hipdnn_frontend/Types.hpp`). The shim aliases each hipDNN
type into `<shim_ns>` (`using hipdnn_frontend::DataType_t;`) rather than
re-declaring it. See §4.3.1.

Aliasing requires hipDNN's enum to be a **name-superset** of the cuDNN one —
every enumerator a consumer can name must exist on the hipDNN side, with
matching semantics. Where hipDNN is currently missing values (for example
`DataType_t`'s FP8 / FP4 / INT4 variants), **the go-forward is to add the
missing values to the hipDNN enum** so it becomes a strict superset and the
alias holds. This is the same pattern as the error-code enum (§4.6): close
the gap on the hipDNN side and alias, rather than re-declare a parallel enum
and translate. A value that hipDNN's enum carries but no engine yet supports
is a *runtime* `GRAPH_NOT_SUPPORTED` at build time (§4.4.2), not an
enum-mapping concern.

Bidirectional `static constexpr` mapping (`to_hipdnn()` / `from_hipdnn()` in
`detail/type_mapping.h`) is therefore needed only for the **C-API enum
family** — the `<cudnn.h>` stub types (`cudnnDataType_t`, etc., §4.7) that
map to their hipDNN C equivalents — not for the FE-namespace enums, which are
aliases.

The shim **never** `static_cast`s between the cuDNN and hipDNN enum families.
Even where the integer values happen to coincide, the equivalence is only
nominal; a cast like `static_cast<HeurMode_t>(1)` is forbidden because it
silently breaks the moment either side reorders or inserts a value. Identity
is established either by aliasing (the values *are* the same type) or by a
named mapping function — never by a numeric cast.

#### 4.3.1 Three-tier mapping strategy

A central goal is to **minimize wrappers**. For every cuDNN FE symbol the
shim picks the lowest tier that is correct:

1. **Alias (preferred).** The symbol is exactly 1:1 with a hipDNN type
   (same shape, same values, same semantics). Expose it with a `using`
   declaration in the matching shim namespace — no wrapper class, no
   conversion, zero overhead:
   ```cpp
   namespace hipdnn_frontend::compatibility::cudnn_frontend {
       using hipdnn_frontend::ConvolutionMode_t;
       using hipdnn_frontend::DataType_t;
       using hipdnn_frontend::HeurMode_t;
       using hipdnn_frontend::error_t;
   }
   namespace hipdnn_frontend::compatibility::cudnn_frontend::graph {
       using hipdnn_frontend::graph::Tensor_attributes;
       using hipdnn_frontend::graph::Conv_fprop_attributes;
       // …one per 1:1 type
   }
   ```
   All FE-namespace enums (hipDNN already publishes them as `_t` typedefs;
   where it is missing values, those values are added to the hipDNN enum so
   it stays a name-superset — §4.3, §4.6), `error_t`/`error_object`, and
   every `*_attributes` class that maps 1:1 (Reference §2) fall here. The
   guiding move when a type is *almost* 1:1 is to close the gap on the
   hipDNN side so it can be aliased, rather than to introduce a parallel
   re-declared type.

2. **Error-wrapper.** The symbol exists in cuDNN FE but hipDNN has no
   support yet. Declare a minimal type/method with the exact cuDNN FE
   signature so consumer source compiles, but any use returns
   `error_code_t::GRAPH_NOT_SUPPORTED` at runtime (or `static_assert`s for
   non-type template parameters), with a message pointing the user to file
   a GitHub issue for the missing feature. To keep this cheap at scale, the
   shim provides a small macro / generator that stamps out a fail-stub from
   the upstream signature (see Risk §6.5). Every "(none)" row in Reference
   §2 is an error-wrapper.

3. **Composition-wrapper (fallback only).** The symbol exists on both
   sides but the surfaces genuinely don't line up, so neither aliasing nor
   a pure forward is possible. Contain the hipDNN object by value and
   forward method-by-method. In practice the principal case is
   `graph::Graph`, whose plan-filtering, notes, knobs, and heuristics
   surface has no clean hipDNN counterpart (§4.5, §5.2).

The default is to climb to the *lowest* applicable tier; a wrapper is a
signal that the underlying surface isn't yet 1:1, and closing that gap on
the hipDNN side (so the type can be demoted to an alias) is preferred over
carrying the wrapper indefinitely.

### 4.4 Graph and node wrapping

The principal class is `<shim_ns>::graph::Graph`. It is the main Tier-3
composition wrapper (§4.3.1): it contains a `hipdnn_frontend::graph::Graph`
by value and forwards calls, holding extra state only for the surface that
isn't 1:1 (notes, knobs, heuristics — §4.5).

The `*_attributes` classes follow the three-tier rule (§4.3.1), not a
blanket "wrap everything" policy:
- Where a `*_attributes` class is exactly 1:1 with its hipDNN counterpart
  (the majority — see Reference §2), the shim **aliases** the hipDNN type
  into `<shim_ns>::graph` with a `using` declaration. No wrapper, no
  per-setter translation.
- Where hipDNN has **no** equivalent ("(none)" rows in Reference §2), the
  shim ships a Tier-2 error-wrapper that compiles but fails at use.
- Only a class whose surface exists on both sides but doesn't line up gets
  a Tier-3 composition wrapper that translates on the way in and
  materialises the inner hipDNN attribute object on calls into hipDNN.

Three surface details that drive the wrapper implementation:

- Every node-adding method on `Graph` takes its `*_attributes` parameter
  **by value**, not by reference.
- `Graph::execute()` has **four** overloads (two key-type variants of the
  variant pack, plus shape-override and flat-pointer-array forms); the
  variant-pack map is passed by **non-const reference**. The shim wraps
  all four; internally they all funnel through the same hipDNN execute
  path after key-type / pointer-form translation.
- The graph-level configuration setters are **triaged individually**,
  because silently ignoring some of them would change correctness or
  resource behavior. Disposition:

  | Setter | Disposition |
  |--------|-------------|
  | `set_override_shape_enabled` | **Forward** — hipDNN supports it now (matching) |
  | `set_dynamic_shape_enabled` | Forward if supported; else no-op + debug log (re-confirm during impl) |
  | `set_kernel_cache` | **Log and ignore** — kernel reuse is still valid without it |
  | `set_sm_count` | **Error** — ignoring partial-device allocation can cause runtime issues; not safe to silently drop |
  | `set_device_properties` | **Error** until AOT / deviceless compilation support lands (its primary use) |
  | `set_sm_version` | **Investigate** — disposition TBD; lean toward error if it affects codegen target |

  Where a setter errors, it does so via the recorded-error mechanism so the
  fluent chain isn't broken mid-expression (the error surfaces at the next
  `validate()` / `build_operation_graph()`; see §4.4.2). Where it is a
  benign no-op, it logs at debug level and returns `*this`. This keeps
  portable consumer code compiling while refusing to silently mis-execute.

For the full upstream `Graph` signature with line-number citations against
`include/cudnn_frontend/graph_interface.h`, see
[Supporting Reference §1](./0012_CuDNN_Shim_Reference.md#1-graph-class--verified-signature).

#### 4.4.1 Tensor identity

cuDNN FE uses `std::shared_ptr<Tensor_attributes>` as the stable identity for
a tensor across a graph. hipDNN uses a `uid` (`int64_t`) and a node's
input/output role to identify a tensor in the variant pack.

**This requires no special handling in the shim: `hipdnn_frontend`
already replicates cuDNN FE's identity semantics 1:1.** Specifically:

- A UID the user sets via `set_uid()` is honored; a tensor left without a
  UID is auto-assigned a unique one at **build** time — never overlapping a
  pre-set UID — by `hipdnn_frontend::graph::Graph::assignUnsetTensorUids()`
  (verified at `Graph.hpp:293`). This matches cuDNN FE, which is also
  late-bound: `set_uid()` only records the request, and uniqueness is
  enforced during build, not at the call site.
- Duplicate UIDs are detected at build/validation time (UIDs determine
  graph edges, so a collision yields a validation failure), again matching
  cuDNN FE's behavior of failing during `build_operation_graph` on
  duplicate UIDs.
- hipDNN already exposes a `shared_ptr<TensorAttributes>`-keyed `execute()`
  overload that internally lowers to the `map<int64_t, void*>` variant
  pack (verified at `Graph.hpp:1760`), so the shim does **not** need to
  maintain its own `Tensor_attributes*` → `uid` map.

Consequently there is no shim-side UID allocator, no construction-time
assignment, and no auxiliary identity map. UIDs need not be `Graph`-scoped —
a `Tensor_attributes` may be shared across graphs, and the only error
condition is a duplicate UID *within* a single graph, which both libraries
already reject at build time. The shim's `Tensor_attributes` setters
(`set_uid`, `set_output`, etc.) forward directly to the aliased/wrapped
hipDNN type.

#### 4.4.2 Node coverage

Upstream `graph_properties.h` defines 39 `*_attributes` classes for the v9
graph API; roughly half have a current `hipdnn_frontend` equivalent and
half do not. For each class **with** a 1:1 hipDNN equivalent, the shim
**aliases** the hipDNN type (§4.3.1) — no wrapper. For each class
**without** one, the shim declares the attribute class and `Graph::*`
method with the cuDNN FE signature (so source compiles) but reports
`error_code_t::GRAPH_NOT_SUPPORTED`, and the omission is noted in the shim
header's Doxygen.

**Where the error surfaces.** Node-adding methods return
`std::shared_ptr<Tensor_attributes>`, not `error_t`, so an unsupported node
cannot return the error code directly from the method call. Instead, adding
an unsupported node (or using an errored setter, §4.4) **records an error on
the composition `Graph`**, and that error is returned from the next
`validate()` / `build_operation_graph()` call. This matches how a consumer
already checks graph construction (PyTorch wraps these in
`AT_CUDNN_FRONTEND_CHECK`), so no special-casing is needed on the consumer
side. The error message points the user to file a GitHub issue for the
missing node.

The full class-by-class table (39 entries, with line numbers in
`graph_properties.h` and the matching hipDNN attribute type — or "none" —
for each) is in
[Supporting Reference §2](./0012_CuDNN_Shim_Reference.md#2-full-_attributes-coverage-table).

#### 4.4.3 Verified PyTorch consumer surface (cuDNN FE v9 graph API)

`aten/src/ATen/native/cudnn/MHA.cpp` is the **only** file in PyTorch that
consumes the v9 graph API; all other cuDNN consumers (`Conv_v8.cpp`,
`aten/src/ATen/native/quantized/cudnn/*`) use the v0.x / v8 backend
descriptor API and are out of scope per §1 "Not". The v9 surface that
`MHA.cpp` actually exercises is narrow: SDPA forward and backward; a small
set of `Graph` lifecycle, `Tensor_attributes`, `SDPA_attributes`, and
`SDPA_backward_attributes` methods; a handful of `DataType_t` values; and
`HeurMode_t::A`. PyTorch also uses the UID-keyed variant pack throughout —
the simplest path, though per §4.4.1 the shim needs no special tensor-identity
handling for either keying style.

For the precise symbol-by-symbol table with line-number citations, plus
the explicit list of cuDNN FE features PyTorch does **not** use, see
[Supporting Reference §3](./0012_CuDNN_Shim_Reference.md#3-verified-pytorch-consumer-surface).

The shim **ships the full v9 surface so hipified source compiles, and fails
loudly at use** (build/validate time, §4.4.2), with hipDNN node work
prioritized separately. Waiting for full engine support would block
otherwise valid use cases, and adding the frontend surface without engines
would become a hidden runtime error either way — surfacing it as an
explicit, documented `GRAPH_NOT_SUPPORTED` (pointing at a GitHub issue) is
strictly better than a silent gap.

### 4.5 Heuristics and plan selection

v9 exposes plan selection and filtering as methods directly on `Graph`:
`create_execution_plans(vector<HeurMode_t>)`, `check_support`,
`build_plans`, `get_execution_plan_count`, the `select_*` / `deselect_*`
notes filters, the `deselect_workspace_greater_than` /
`deselect_shared_mem_greater_than` resource caps, and the
`get_behavior_notes*` introspection methods. There is no user-facing
`Plans` class (an internal `Execution_plan_list` holds the data but is
not part of the API).

hipDNN's current selection model (RFC 0007) is engine-knob driven and does
not present heuristic modes. Some of this surface is being filled in by the
hipDNN auto-tuning work (PR #7217), which the shim should build on as it
lands.

**HeurMode.** Accept all `HeurMode_t` values; until real hipDNN heuristics
exist, map every mode (`A`, `B`, `FALLBACK`, `OPENSOURCE`) to hipDNN's
fallback/default selection and **log a WARN on first use per process** so a
user debugging a perf regression sees that their heuristic choice was not
honored. As hipDNN heuristics arrive, remap the modes properly. The
benchmarking and auto-tuning paths are explicit user choices and are handled
separately.

**Notes — per-note triage, not blanket no-op.** A blanket "accept and
ignore" is unsafe: silently running without a requested filter can be
*wrong*, not merely slower. Each `NumericalNote_t` / `BehaviorNote_t` is
triaged into one of three handlings:

- **Map to engine capability** — when hipDNN can honor the constraint
  through its engine/knob system. The deterministic ("non-deterministic"
  filter) case is the worked example: now that the note system exists,
  hipDNN can likely express it, and the shim should wire it through.
- **Warn and ignore** — for notes that are advisory and safe to drop
  (logged at WARN on first use).
- **Error** — for correctness-critical notes hipDNN cannot honor. If a user
  requests deterministic behavior and the shim cannot guarantee it, it is
  **wrong** to run anyway; the shim must error rather than silently produce
  non-deterministic results. (Recorded-error mechanism, §4.4.2.)

The per-note classification table lives in
[Supporting Reference §4](./0012_CuDNN_Shim_Reference.md#4-heuristics-and-plan-selection--verified-api)
and must be reviewed note-by-note before Phase 4 lands.

**Resource caps.** The workspace-size cap is enforced post-hoc against
hipDNN's existing `get_workspace_size()` (and the workspace *knobs* need a
concrete mapping plan — Reference §4). The shared-memory cap depends on
per-plan shared-memory metadata hipDNN does not yet expose; until it does,
the shim **rejects a non-zero `deselect_shared_mem_greater_than`** rather
than silently ignoring it (Reference §4).

For the full upstream method enumeration (with `graph_interface.h` line
numbers), behaviour table, and the open question about hipDNN-side
metadata extensions, see
[Supporting Reference §4](./0012_CuDNN_Shim_Reference.md#4-heuristics-and-plan-selection--verified-api).

### 4.6 Error handling and logging

Per the aliasing-first strategy (§4.3.1), the preferred outcome is to make
`hipdnn_frontend`'s error type exactly 1:1 with cuDNN FE's and **alias** it,
rather than ship a wrapper. cuDNN FE's `error_object` exposes public fields
`code` and `err_msg` and methods `get_code`, `get_message`, `is_good`,
`is_bad`, `operator==/!=`, with `error_t` a typedef for it; hipDNN's `Error`
is structurally the same. The one gap is the `error_code_t` enum: cuDNN FE
has 16 values, a few of which (e.g. `NVRTC_COMPILATION_FAILED`,
`INVALID_CUDA_DEVICE`, `CUDNN_BACKEND_API_FAILED`) have no hipDNN
counterpart. **The missing values are added to hipDNN's error-code enum** so
the whole error type is aliasable rather than wrapped. The cuDNN FE
error/log macros
(`CHECK_CUDNN_FRONTEND_ERROR`, `RETURN_CUDNN_FRONTEND_ERROR_IF`,
`CUDNN_FE_LOG*`) are also re-exported, since PyTorch's
`AT_CUDNN_FRONTEND_CHECK` expands to them.

Logging in cuDNN FE is **off by default** (controlled by the integer env
var `CUDNN_FRONTEND_LOG_INFO`; output target via `CUDNN_FRONTEND_LOG_FILE`;
compile-time disable via `NV_CUDNN_FRONTEND_DISABLE_LOGGING`). The shim
re-exports these names and delegates to `hipdnn_frontend::Logging`,
preserving the off-by-default behaviour. Concretely, the shim **reads the
cuDNN FE env vars and forwards them into the equivalent hipDNN Logging API**
(level and output target), rather than only re-exporting the names — so
setting `CUDNN_FRONTEND_LOG_INFO` actually drives hipDNN's logger.
(Alternatively, hipDNN's own env-var handling could be extended to recognize
the cuDNN names directly; the forward-into-API approach is preferred as it
keeps the bridge in the shim.)

For the verified `error_object` structure, the full macro list, and the
detailed logging API surface (with `cudnn_frontend_Logging.h` references),
see
[Supporting Reference §5](./0012_CuDNN_Shim_Reference.md#5-error-handling-and-logging--verified-api).

### 4.7 Compatibility scope (and the `<cudnn.h>` / v0.x problem)

The shim provides **source-level** compatibility for cuDNN frontend v9
TUs only; ABI compatibility with cuDNN-compiled binaries is out of
scope and consumers must recompile.

Upstream `cudnn_frontend.h` unconditionally `#include`s `<cudnn.h>` and 14
v0.x/v8 backend-descriptor headers, and exposes the v0.x class names
(`ConvDesc`, `ConvDescBuilder`, `EngineHeuristics`, `Operation`, `Tensor`,
...) via `using` aliases in `namespace cudnn_frontend`. Any consumer that
includes the upstream umbrella gets these names whether they use them
or not.

**Decision (per §1)**: the shim's `cudnn_frontend.h` is a hand-curated
**v9-only umbrella**. It does not mirror upstream's include pattern, does
not pull in any v0.x descriptor header, and does not declare the v0.x
classes or `using` aliases. The stub `cudnn.h` it ships covers the C-API
**types** that the v9 graph API surface refers to in its own method
signatures:

- `cudnnHandle_t` (used by `Graph::execute`, `build_operation_graph`,
  `check_support`, etc.), declared as an alias:
  `using cudnnHandle_t = hipdnn_frontend::hipdnnHandle_t;`
- `cudnnStatus_t`
- `cudnnDataType_t`, `cudnnTensorFormat_t`, `cudnnConvolutionMode_t`,
  `cudnnReduceTensorOp_t`, `cudnnNormFwdPhase_t`,
  `cudnnBackendHeurMode_t`, `cudnnBackendNumericalNote_t`,
  `cudnnBackendBehaviorNote_t`, `cudnnBackendDescriptorType_t`

**The stub also ships a small set of C-API entry points.** These are used
outside the frontend — for handle init, stream binding, error handling, and
version checks — and are needed to build and run the cuDNN-mirrored samples
(§8.3):

- Handle lifecycle: `cudnnCreate`, `cudnnDestroy`, `cudnnSetStream`,
  `cudnnGetStream` — forwarding to the corresponding hipDNN handle API.
- `cudnnGetErrorString` — forwarding to hipDNN's error-string lookup.
- `cudnnGetVersion` — returning the version the shim claims compatibility
  with (§4.8).

cuDNN FE also provides a `create_cudnn_handle()` helper that calls directly
into the handle API; the shim re-exports an equivalent so sample code that
uses it works unchanged. (Implementation note / open item: confirm which of
these hipDNN C-API entry points already exist to forward to, and add any
that are missing. The full cuDNN C library beyond this set — convolution
descriptor APIs, etc. — remains out of scope.)

If, during implementation, the v9 graph API surface is found to refer to
a v0.x C++ type by name (e.g., `cudnn_frontend::Tensor` as used by some
internal-but-public `INode` signatures), the shim will expose *only that
type*, scoped to make v9 TUs compile. Where that legacy name was simply
reused by the new API and corresponds to a v9 type the shim already
supports, **alias** it to the matching hipDNN type (per §4.3.1) rather than
stubbing it; in practice `Tensor` is the likely — possibly only — case. The
full v0.x C++ surface (`TensorBuilder`, `OperationBuilder`,
`EngineConfigGenerator`, `ExecutionPlanBuilder`, `VariantPackBuilder`,
`EngineHeuristicsBuilder`, `EngineFallbackListBuilder`,
`EngineConfigBuilder`, `ConvDescBuilder`, `MatMulDescBuilder`,
`PointWiseDescBuilder`, `ResampleDescBuilder`, `ReductionDescBuilder`,
the `*_v8` types, the free functions `filter`, `hasNumericalNote<>`,
`time_sorted_plan<>`, `check_errata`, `load_from_config`, and
`cudnnException`) is **out of scope** for this RFC.

Consequence: source files that use v0.x types are not buildable against
this shim. The most prominent affected consumers are PyTorch's
`aten/src/ATen/native/cudnn/Conv_v8.cpp` and
`aten/src/ATen/native/quantized/cudnn/*` — both must be excluded from
the build (or rewritten to v9) for any tree that hipifies against this
shim. The v9 PyTorch consumer `aten/src/ATen/native/cudnn/MHA.cpp` is
fully supported (see §4.4.3, §7.5).

A follow-up RFC for a v0.x compile-only stub layer (or a fuller v0.x
shim) is enumerated in §9 as deferred future work.

For the verbatim upstream `cudnn_frontend.h` include list, the
enumeration of v0.x aliases, and the considered-but-deferred options for
v0.x source compatibility, see
[Supporting Reference §6](./0012_CuDNN_Shim_Reference.md#6-the-cudnnh--v0x-umbrella-header-problem).

### 4.8 Versioning

The shim is part of `hipdnn_frontend`'s versioning surface (see RFC 0005).
That implies:

- Its public headers live under `frontend/include/compatibility/cudnn/` and
  are versioned together with `hipdnn_frontend` (single `version.json`).
- Within a major version of `hipdnn_frontend`, the shim's surface area may
  **only** grow (new node wrappers, new enum values mapped) — never shrink
  or change signatures.
- The shim's public symbols will additionally expose the upstream-style
  version macros (`CUDNN_FRONTEND_MAJOR_VERSION=1`,
  `CUDNN_FRONTEND_MINOR_VERSION=24`, `CUDNN_FRONTEND_PATCH_VERSION=0`,
  `CUDNN_FRONTEND_VERSION=12400`) declaring which cuDNN FE version this
  shim claims source compatibility with. Matching upstream is important
  because PyTorch's `MHA.cpp` actually gates on these (`#if
  CUDNN_FRONTEND_VERSION <= 11200` chooses between `set_is_inference` and
  `set_generate_stats`). The upstream values are in
  `include/cudnn_frontend_version.h`. This is independent of hipDNN's own
  version.

### 4.9 Explored Alternatives

- **hipDNN backend for PyTorch**: **Pros**: Cleanest solution. **Cons**:
  Maintenance burden of an additional backend.
- **Compile-time `#ifdef`'d backend for cuDNN → hipDNN**: **Pros**: Shortest
  path to success, could have something working very quickly. **Cons**: The
  code is deviated from cuDNN enough that maintenance and code quality is
  challenging.
- **hipify build-time generation**: **Pros**: integrates with existing systems
  in some cases. **Cons**: The conversion path is not smooth, and the
  debugging experience is bad.
- **AI-based conversion tools**: **Pros**: Fits in with our current direction
  at AMD. **Cons**: Does not exist. At all.
- **Re-implement cuDNN FE on top of hipDNN backend C API directly** (skipping
  `hipdnn_frontend`): **Pros**: avoids the wrapper-of-a-wrapper overhead;
  could match cuDNN signatures exactly. **Cons**: duplicates a lot of work
  already done in `hipdnn_frontend`; we lose any future improvements to
  `hipdnn_frontend` for free.

Note: once the shim's aliases and wrappers exist, a **direct (non-hipify)
source-conversion path** for the supported API becomes nearly free — the
mapping is already encoded in the shim. This does not compete with the
hipify path; the two can coexist, and offering both is a reasonable future
option for consumers who prefer one workflow over the other.

## 5. Key Design Decisions

### 5.1 Header file extension and naming

**Decision**: Use `.h` extensions, matching cuDNN frontend
convention.  Match cuDNN filenames precisely.

### 5.2 Forwarding strategy

**Decision**: Aliasing is the default; composition is the fallback for
types that are not 1:1 (§4.3.1). Composition is used for the shim's `Graph`,
which *contains* a `hipdnn_frontend::graph::Graph` by value rather than
inheriting from or aliasing it.

**Rationale**: cuDNN FE's `Graph` has a different surface area than hipDNN's
(plan filtering, notes, knobs, heuristics — different method names, return
types, and validation flow), so it cannot be aliased. Composition gives
complete control over that surface and avoids accidentally exposing hipDNN
methods through the cuDNN-shaped class. For the many *other* types that are
already 1:1 (most enums, `error_t`, and the matching `*_attributes`
classes), no wrapper is written at all — they are aliased, which is cheaper
and removes per-call overhead.

**Drawbacks**: The composition path is slightly more verbose — each exposed
method must be forwarded explicitly — but it applies to only the small set
of genuinely non-matching types rather than the whole surface.

### 5.3 Tensor identity — rely on hipDNN's native UID handling

**Decision**: No shim-side UID allocator and no shim-side identity map.
`hipdnn_frontend` already matches cuDNN FE's identity semantics — set UIDs
are honored, unset UIDs are auto-assigned uniquely at build time, duplicates
fail at build, and a `shared_ptr`-keyed `execute()` overload already lowers
to the UID-indexed variant pack internally (see §4.4.1 for the verified
`Graph.hpp` references). The shim simply forwards.

**Rationale**: `hipdnn_frontend::graph::Graph` already provides
cuDNN-equivalent identity natively, so a shim-side UID allocator or
`shared_ptr<Tensor_attributes>` → UID map would be redundant.

**Drawbacks**: None — there is no per-execute map lookup and no collision
semantics for the shim to define.

### 5.4 Failure mode for unimplemented cuDNN features

Runtime
Favor `Error::NOT_SUPPORTED` by default, with optional compile-time `static_assert`
mode if required.

## 6. Risks

### 6.1 cuDNN FE surface drift

**Risk**: cuDNN FE continues to evolve. New nodes, new enum values, and new
attribute setters appear in each release. If we pin to v9.0 and PyTorch
upgrades to a newer point release, hipified PyTorch may stop compiling.

**Mitigation**:
- Pinned to cuDNN FE v1.24.0 (see §2). When upstream releases a new
  version, the shim does not automatically follow — a manual triage step
  evaluates whether to rebase and which new symbols to wrap. New upstream
  nodes appear as `error_code_t::GRAPH_NOT_SUPPORTED` at runtime until
  the wrapper is added.
- CI job that, on a schedule, attempts to build a small set of upstream
  cuDNN FE samples against the shim and reports diffs.
- Establish a quarterly cadence for evaluating upstream changes and triaging
  them.

### 6.2 Performance overhead of the wrapper layer

**Risk**: Even thin wrappers can add measurable overhead in
graph-construction-heavy workloads (e.g., PyTorch eager mode building a fresh
graph per kernel launch).

**Mitigation**:
- Prefer aliasing (§4.3.1): aliased types add **zero** overhead — they are
  the hipDNN type. This keeps most of the surface out of the perf budget,
  and the shim carries no per-execute tensor-identity map (§4.4.1, §5.3).
- All forwarding methods on the residual composition wrappers `inline` and
  header-only.
- Benchmark `Graph::build()` and `Graph::execute()` overhead against native
  `hipdnn_frontend` calls; gate merges on no-regression.
- Avoid heap allocations in the hot path of the composition `Graph`
  wrapper.

### 6.3 Silent behavioral divergence

**Risk**: Two calls — `cudnn_frontend::graph::Graph::build_plans(...,
HeurMode_t::A)` and the shim's equivalent — return success, but the chosen
plan is materially different (different precision, different numerical
behavior). The consumer's accuracy tests may fail in subtle ways.

**Mitigation**:
- Log every mapped-but-not-honored heuristic / mode at `WARN` level on first
  use per process.
- Document the divergences exhaustively in the shim's Doxygen and in a
  `KNOWN_DIVERGENCES.md`.
- Add a sample / test that drives `Graph::build_plans` with every
  `HeurMode_t` value and snapshot-tests the chosen plan's engine ID.

### 6.4 License / attribution

**Risk**: To preserve cuDNN FE's exact signatures and Doxygen, we may end up
copying substantial chunks of comments / enum value names from NVIDIA's
MIT-licensed source. Without correct attribution this is a license violation.

**Mitigation**:
- File-level header in every shim source file: AMD copyright + "Portions
  derived from NVIDIA cuDNN frontend, used under the MIT license", with the
  upstream copyright notice preserved.
- Legal review before initial merge.

### 6.5 Maintenance burden

**Risk**: The shim is a long-term ongoing commitment. Each hipDNN node added
must also be wrapped; each cuDNN FE release must be evaluated; each PyTorch
upgrade tests the shim's adequacy.

**Mitigation**:
- Aliasing-first (§4.3.1) keeps the hand-written surface small: every 1:1
  type is a one-line `using` that needs no per-release maintenance, so the
  ongoing burden concentrates on the few composition wrappers and the
  error-stubs.
- **Owner: the hipDNN team (collective)**, with Mitch Ousdahl leading
  the initial implementation. The shim is treated as a first-class
  component of `hipdnn_frontend`, not a side project.
- Tooling: a generator/macro that takes a cuDNN FE signature and emits a
  Tier-2 fail-stub (returns `GRAPH_NOT_SUPPORTED`, points the user at a
  GitHub issue) for any newly-added upstream symbol we don't yet support,
  reducing per-node boilerplate.
- Out-of-tree consumer integration tests run on PR.

### 6.6 Encouraging hipDNN-native adoption is undermined

**Risk**: The shim is so convenient that no downstream project ever moves to
the hipDNN-native API, leaving us perpetually playing catch-up with cuDNN FE.

**Mitigation**: Accept this. The stated goal is adoption, not API purity. A
hipified-but-running PyTorch is a much better outcome than a clean
hipDNN-native PyTorch that ships in 2027.

## 7. Execution Plan

### 7.1 Phase 1 — Foundations (compiles, builds, has CI)

- Add `projects/hipdnn/frontend/include/compatibility/cudnn/` to the
  source tree.
- Stand up `cudnn_frontend.h` umbrella header (empty but installable).
- Add CMake target `hipdnn_cudnn_shim` (or integrate into `hipdnn_frontend`
  as an additional include directory).
- Implement `detail/type_mapping.h` for the core enums
  (`cudnnStatus_t`, `cudnnDataType_t`, `cudnnTensorFormat_t`,
  `cudnnPointwiseMode_t`, `cudnnReduceTensorOp_t`).
- Implement `detail/status_translation.h` and `error_object`.
- Ship the stub `cudnn.h` (per §4.7) covering the C-API types referenced by
  the v9 graph API (`cudnnHandle_t`, `cudnnStatus_t`, `cudnnDataType_t`,
  `cudnnTensorFormat_t`, `cudnnConvolutionMode_t`, `cudnnReduceTensorOp_t`,
  `cudnnNormFwdPhase_t`, `cudnnBackendHeurMode_t`,
  `cudnnBackendNumericalNote_t`, `cudnnBackendBehaviorNote_t`,
  `cudnnBackendDescriptorType_t`) **and** the small set of C entry points
  needed for init / error handling / version checks and for building the
  mirrored samples: `cudnnCreate`, `cudnnDestroy`, `cudnnSetStream`,
  `cudnnGetStream`, `cudnnGetErrorString`, `cudnnGetVersion`, and a
  `create_cudnn_handle()` helper (each forwarding to the hipDNN equivalent —
  see §4.7). The remainder of the cuDNN C library (convolution descriptor
  APIs, etc.) stays out of scope.
- Unit tests for the type mapping and status translation (round-trip every
  enum value).

**Exit criterion**: a v9-only translation unit — a hand-written test file
that uses only `cudnn_frontend::graph::*` and the C-API types listed
above — compiles cleanly against the shim's `cudnn_frontend.h` umbrella.
Source that mixes v9 and v0.x types is out of scope (§4.7, §9).

### 7.2 Phase 2 — Tensor and Graph skeleton

- Implement `Tensor_attributes` wrapper (all setters, including
  `set_dim`, `set_stride`, `set_data_type`, `set_uid`, `set_is_virtual`
  (`graph_properties.h:327`), `set_is_pass_by_value` (`:348`; note the
  `is_` prefix), `set_output` (`:333`, which is `!set_is_virtual`),
  `set_ragged_offset` (`:446`), `set_name`).
- Implement `Graph` wrapper: constructors, `set_io_data_type`,
  `set_compute_data_type`, `set_intermediate_data_type`, `set_name`,
  `validate`, `build_operation_graph`, `create_execution_plans`,
  `check_support` (both overloads), `build_plans` (both overloads),
  `build_plan_at_index`, `get_execution_plan_count`, the convenience
  `build()` one-shot (validate → build_operation_graph →
  create_execution_plans → check_support → build_plans; most upstream
  samples use it), all **four** `execute` overloads (shared_ptr-keyed
  pack; uid-keyed pack; uid-keyed-with-shape-overrides; flat-pointer-array
  form), `get_workspace_size`, `serialize`, `deserialize`.
- Implement the graph-level configuration setters per the §4.4 triage table:
  `set_override_shape_enabled` forwards (hipDNN supports it);
  `set_dynamic_shape_enabled` forwards if supported, else no-op + debug log;
  `set_kernel_cache` logs and ignores; `set_sm_count` and
  `set_device_properties` error (via the recorded-error mechanism, §4.4.2);
  `set_sm_version` is TBD pending investigation.
- Add `Graph::tensor()` returning `std::shared_ptr<Tensor_attributes>`,
  including the scalar convenience overloads (`tensor(int64_t)`,
  `tensor(float)`, …).
- No tensor-identity map or UID allocator is needed — hipDNN handles UID
  assignment natively (§4.4.1).
- Unit tests covering: building an empty graph, an invalid graph,
  serializing and deserializing the empty graph.

**Exit criteria**: A cuDNN FE sample that constructs a graph with a single
tensor and round-trips it through serialize/deserialize works.

### 7.3 Phase 3 — Node coverage, priority order

Priority is driven by the verified consumer survey in §4.4.3. PyTorch's
only v9 graph-API consumer is `MHA.cpp`, which exercises SDPA fwd/bwd and
nothing else. The remaining nodes have no current PyTorch v9 consumer; their
order is driven by future PyTorch conversions of ops that today still use
the v8 backend descriptor API and by any other downstream consumer that
materialises a concrete ask.

**Tier 1 — PyTorch SDPA unblocked (Phase 3a, minimum-viable shim):**

1. `SDPA_attributes` (full setter surface used by PyTorch — see §4.4.3
   table). Including the `[[deprecated]]` `set_is_inference()` (since
   PyTorch still emits it on `CUDNN_FRONTEND_VERSION <= 11200` paths) and
   the float and shared_ptr overloads of `set_attn_scale`.
2. `Graph::sdpa(shared_ptr<Tensor_attributes>, shared_ptr<Tensor_attributes>, shared_ptr<Tensor_attributes>, SDPA_attributes) -> std::array<shared_ptr<Tensor_attributes>, 2>` (returns `{O, Stats}`, verified at `graph_interface.h:1857`)
3. `SDPA_backward_attributes` (full setter surface used by PyTorch)
4. `Graph::sdpa_backward(shared_ptr<Tensor_attributes>, shared_ptr<Tensor_attributes>, shared_ptr<Tensor_attributes>, shared_ptr<Tensor_attributes>, shared_ptr<Tensor_attributes>, shared_ptr<Tensor_attributes>, SDPA_backward_attributes) -> std::array<shared_ptr<Tensor_attributes>, 3>` (returns `{dQ, dK, dV}`, verified at `:1923`)
5. Tensor-attribute features required by MHA but not yet noted elsewhere:
   `set_ragged_offset`, `set_is_pass_by_value` (these are
   `Tensor_attributes` setters, not new nodes, but they must work for the
   SDPA path).

**Exit criteria for Phase 3a**: a hipified copy of PyTorch's `MHA.cpp`
compiles unchanged against the shim, and the four PyTorch entry points
(`run_cudnn_SDP_fprop`, `run_cudnn_SDP_fprop_nestedtensor`,
`run_cudnn_SDP_bprop`, `run_cudnn_SDP_bprop_nestedtensor`) execute end-to-end
on hipDNN's SDPA provider.

**Tier 2 — Remaining nodes (Phase 3b):**

The remaining v9 nodes with a hipDNN equivalent (Convolution
fprop/dgrad/wgrad, BatchNorm forward/backward/inference, LayerNorm,
RMSNorm fwd/bwd, Matmul, Pointwise, Reduction, ResampleFwd,
BlockScaleQuantize/Dequantize, CustomOp) are implemented in
**consumer-driven order**. Each node lands when a real consumer asks for
it. Until then, the `Graph::*` method and attribute class are declared
(so source compiles) but return `error_code_t::GRAPH_NOT_SUPPORTED` per
§4.4.2.

For each node, the deliverable is: attribute wrapper, `Graph::*` method,
unit tests building/validating/executing the node, and a sample mirroring
the corresponding cuDNN FE sample.

### 7.4 Phase 4 — Heuristics surface, logging, and polish

- Implement `HeurMode_t`, `NumericalNote_t`, `BehaviorNote_t` enum mapping.
- Wire the plan-filter methods on the shim's `Graph`, applying the
  per-note triage (§4.5; classification table in
  [Supporting Reference §4](./0012_CuDNN_Shim_Reference.md#4-heuristics-and-plan-selection--verified-api)):
  - `select_*` / `deselect_*` notes — map to engine capability where
    hipDNN can honor it, warn-and-ignore where advisory, **error** where
    correctness-critical and unsupported (e.g. deterministic). No blanket
    no-op.
  - `deselect_workspace_greater_than` — filter hipDNN plans by
    workspace size before `build_plans()` (hipDNN's
    `get_workspace_size()` is sufficient).
  - `deselect_shared_mem_greater_than` — **reject a non-zero value** until
    hipDNN exposes per-plan shared-memory metadata (see
    [Supporting Reference §4](./0012_CuDNN_Shim_Reference.md#4-heuristics-and-plan-selection--verified-api)).
- Wire the plan/engine introspection surface (Reference §1): `build()`,
  `get_engine_count`, `get_knobs_for_engine`, `create_execution_plan(engine_id,
  knobs)`, `get_plan_name` / `get_plan_name_at_index`,
  `get_workspace_size_plan_at_index`, `execute_plan_at_index`, and
  `deselect_engines` — forwarding to hipDNN where supported (much of this
  arrives with auto-tuning PR #7217), error-wrapper otherwise.
- `autotune` / `warmup` — forward to hipDNN's auto-tuning path (PR #7217).
- `populate_cuda_graph` / `update_cuda_graph` — Tier-2 error-wrappers (no
  in-scope HIP-graph capture analogue).
- Implement environment-variable bridge for cuDNN FE logging vars
  (forward the cuDNN env vars into hipDNN's Logging API — §4.6).
- `KNOWN_DIVERGENCES.md` document.
- Performance benchmarks vs. native `hipdnn_frontend`.

### 7.5 Phase 5 — Real-world validation

- Hipify a small known-good cuDNN FE project (suggestion:
  cuDNN FE's own `samples/cpp/` test programs) and verify they build & run.
- Hipify and run a vertical slice of PyTorch's cuDNN-using ops.

The verified PyTorch target slice (per §4.4.3) is the four entry points in
`aten/src/ATen/native/cudnn/MHA.cpp`:

- `run_cudnn_SDP_fprop`
- `run_cudnn_SDP_fprop_nestedtensor`
- `run_cudnn_SDP_bprop`
- `run_cudnn_SDP_bprop_nestedtensor`

These cover dense and nested-tensor SDPA, forward and backward, with
optional causal masking, padding masking via ragged sequence lengths,
dropout, and additive attention bias. Validation:

1. Hipify `MHA.cpp` and rebuild PyTorch's `_cudnn_attention_*` ops against
   the shim.
2. Run `test_transformers.py` SDPA paths with the cuDNN backend forced
   (`SDPBackend.CUDNN_ATTENTION`) on ROCm.
3. Compare numerical results against the math-reference SDPA backend
   within the test's existing tolerances.

PyTorch's `cudnn_convolution` and quantized cuDNN ops both still consume
the v8 backend descriptor API and are out of scope for this RFC (§4.7,
§9). For the validation build, these files must be excluded from the
PyTorch build (or rewritten to v9 first).

### 7.6 Documentation

- Doxygen on every public shim symbol, including the link to its cuDNN FE
  upstream counterpart.
- `docs/cudnn-shim.md` user guide covering: installation, the hipification
  workflow, known divergences, known unsupported nodes, troubleshooting.
- Inline sample showing the textual-hipify transformation on a small
  cuDNN FE program.

## 8. Testing Plan

### 8.1 Unit tests

Location: `projects/hipdnn/frontend/tests/compatibility/cudnn/`. Same
gtest framework as the rest of the frontend tests; same naming convention
(`TestCudnnShim*.cpp`, suites prefixed `TestCudnnShim*`).

- `TestCudnnShimTypeMapping` — round-trip every mappable enum value in both
  directions; assert unmappable values produce the documented error.
- `TestCudnnShimStatusTranslation` — every cuDNN status code maps to a
  hipDNN one and back.
- `TestCudnnShimHandle` — handle lifecycle, stream binding.
- `TestCudnnShimTensor` — `Tensor_attributes` construction, all setters, UID
  assignment, identity preservation across copies.
- `TestCudnnShimGraph<NodeName>` — one suite per supported node, covering:
  construction, validation (good and bad), serialize/deserialize round-trip,
  variant-pack execution against the in-tree `fake_backend` (see
  `projects/hipdnn/frontend/tests/fake_backend/`).
- `TestCudnnShimError` — error_object construction, code mapping, message
  preservation through `hipdnn_frontend::Error`.

### 8.2 Integration tests (AMD-authored)

Location: `projects/hipdnn/samples/cudnn_shim/` (or under
`projects/hipdnn/samples/` next to the existing `convolution`, `sdpa`, etc.
samples).

These are **AMD-authored** integration tests, distinct from the
upstream-sample compatibility check in §8.3 (which compiles NVIDIA's own
sources). They should resemble the existing hipDNN frontend integration
tests — driving the shim end-to-end against the in-tree test plugins /
providers — so they exercise our wrapper/alias layer specifically rather
than re-proving upstream's samples.

- Per-node test driving the shim API, built and run against a real provider
  (miopen-provider or hipblaslt-provider as applicable) or the in-tree
  `fake_backend`.
- "Full pipeline" test: build → validate → check_support → build_plans →
  execute against real tensor data, asserting numerical correctness against
  a hipDNN-native reference.

### 8.3 Compatibility / "build the cuDNN FE samples" test

This is the **source-compatibility proof** and is deliberately separate
from the AMD-authored integration tests in §8.2: here we compile and run
NVIDIA's *unmodified* sources (only the includes repointed) to demonstrate
the shim is genuinely drop-in. The strongest evidence of source
compatibility is that NVIDIA's own `samples/cpp/` programs from the pinned
cuDNN FE version build and pass when their includes are repointed at the
shim.

- Vendor (as a git submodule or fetched at configure time) a pinned cuDNN
  FE tag.
- CI job that builds those samples against the shim.
- For samples whose dependencies cannot be satisfied (e.g., samples that
  rely on cuBLAS), document the skip and the reason.

### 8.4 Performance / overhead testing

- Microbenchmarks comparing native `hipdnn_frontend` Graph construction +
  execute vs. the shim equivalent for a handful of representative graphs.
- **Gate (from §2)**: CI fails when the shim adds more than **1 %** to
  `build()` time or more than **1 µs** per `execute()` call relative to
  the native `hipdnn_frontend` baseline, measured on the same graphs.
- Baselines re-captured on every hipDNN minor-version bump.

### 8.5 Install / package testing

- The shim headers must be installed by the `Development` CMake component
  alongside `hipdnn_frontend` headers.
- Add a `find_package(hipdnn_frontend)`-using downstream CMake project under
  `projects/hipdnn/tests/install/` (or wherever the existing install tests
  live) that `#include`s the shim umbrella header and compiles a trivial
  Graph.

Python bindings are out of scope (§1, §9); a hipDNN benchmarking/bindings
project would be the natural host for shim-level perf or Python-side testing
*if* such bindings are ever built.

### 8.6 PyTorch end-to-end smoke test

A small set of PyTorch tests run on a nightly schedule against a PyTorch
build whose `aten/src/ATen/native/cudnn/MHA.cpp` has been hipified to use
the shim. Concretely: the `test_transformers.py` SDPA paths forced to
the cuDNN backend (`SDPBackend.CUDNN_ATTENTION`).

⚠️ **OPEN QUESTION**: Hosting and ownership of this CI job. ROCm PyTorch
testing infrastructure already exists; we need to confirm whether (a) the
shim's CI is added there, (b) a hipDNN-side CI host is built that pulls
PyTorch in, or (c) the test is run on-demand by the shim's named owners
(see §6.5) at each shim release. Recommend a focused operational review
(`/rfc-review-ops`) before this is finalized.

## 9. Future Considerations

- **Python bindings (out of scope, possible future RFC)**: cuDNN frontend
  ships Python bindings; this RFC is C++-only (§1). A Python-binding
  compatibility layer — if a consumer materializes a concrete ask — would be
  a separate RFC, and a hipDNN benchmarking/bindings project would be its
  natural home (see §8).
- **v0.x compile-only stub layer (deferred)**: a follow-up RFC covering
  compile-only stubs for the upstream v0.x / v8 C++ surface
  (`TensorBuilder`, `OperationBuilder`, `EngineConfigGenerator`,
  `ExecutionPlanBuilder`, `VariantPackBuilder`, `EngineHeuristicsBuilder`,
  `EngineFallbackListBuilder`, `EngineConfigBuilder`, `ConvDescBuilder`,
  `MatMulDescBuilder`, `PointWiseDescBuilder`, `ResampleDescBuilder`,
  `ReductionDescBuilder`, plus the `*_v8` types, the
  free functions `filter`, `hasNumericalNote<>`, `time_sorted_plan<>`,
  `check_errata`, `load_from_config`, and `cudnnException`). Stubbed
  methods return `error_code_t::GRAPH_NOT_SUPPORTED` at runtime. This
  unblocks consumers that mix v9 and v0.x in the same translation unit
  (PyTorch's `Conv_v8.cpp`, the quantized cuDNN ops). Triggered by a
  concrete consumer ask; scope estimate (PyTorch alone touches 13+
  builder classes plus several free functions) to be filled in at that
  RFC's authoring time. Because v0.x types live in the same upstream
  `cudnn_frontend` namespace and ship from headers adjacent to the v9
  ones, the follow-up is purely **additive** to this shim — new headers
  in the existing `compatibility/cudnn/` directory and new declarations
  in `<shim_ns>` and `<shim_ns>::detail`, with no rename or move of v9
  symbols. The path layout in §4.1 was chosen with that in mind. See
  §4.7 and
  [Supporting Reference §6](./0012_CuDNN_Shim_Reference.md#6-the-cudnnh--v0x-umbrella-header-problem)
  for the underlying constraint and the options considered.
