# hipDNN: Runtime Pass-By-Value Tensors Design Document

- **Contributors**: Samuel Reeder
- **Status**: Draft

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
   - 2.1 [End-user API surface](#21-end-user-api-surface)
   - 2.2 [hipDNN gap](#22-hipdnn-gap)
   - 2.3 [Constraints](#23-constraints)
3. [Current System Overview](#3-current-system-overview)
4. [Proposed Design](#4-proposed-design)
   - 4.1 [Frontend tensor flag](#41-frontend-tensor-flag)
   - 4.2 [Tensor schema addition](#42-tensor-schema-addition)
   - 4.3 [Feature signal (derived)](#43-feature-signal-derived)
   - 4.4 [Execute-time transport](#44-execute-time-transport)
   - 4.5 [Provider contract](#45-provider-contract)
   - 4.6 [Feature detection and version filtering](#46-feature-detection-and-version-filtering)
   - 4.7 [Frontend validation](#47-frontend-validation)
   - 4.8 [Frontend variant-pack injection](#48-frontend-variant-pack-injection)
5. [Key Design Decisions](#5-key-design-decisions)
   - 5.1 [cuDNN API-surface parity](#51-cudnn-api-surface-parity)
   - 5.2 [Reuse the variant-pack pointer map](#52-reuse-the-variant-pack-pointer-map)
   - 5.3 [Derive feature detection from the tensor schema](#53-derive-feature-detection-from-the-tensor-schema)
   - 5.4 [Version-only filtering](#54-version-only-filtering)
   - 5.5 [Compile-time vs runtime mode](#55-compile-time-vs-runtime-mode)
   - 5.6 [Deserialized-plan support via the provider payload](#56-deserialized-plan-support-via-the-provider-payload)
   - 5.7 [Frontend variant-pack injection for RUNTIME_PARAM](#57-frontend-variant-pack-injection-for-runtime_param)
6. [Compatibility, Versioning, and Rollback](#6-compatibility-versioning-and-rollback)
7. [Comparison to Ragged and Override Tensor Support](#7-comparison-to-ragged-and-override-tensor-support)
8. [Risks](#8-risks)
9. [Execution Plan](#9-execution-plan)
10. [Testing Plan](#10-testing-plan)
11. [Glossary](#11-glossary)

---

## 1. Executive Summary

This RFC proposes adding **runtime pass-by-value tensors** to hipDNN.
A pass-by-value tensor is a host-side scalar operand (epsilon, alpha,
beta, an SDPA scale, etc.). Today hipDNN supports such scalars only as
**compile-time constants**: the value is baked into the operation graph
at build time and frozen into the compiled execution plan. This RFC adds
the ability to mark a scalar tensor with `set_is_pass_by_value(true)`
and supply its value through the variant pack at **execute** time,
without rebuilding the graph. The public API mirrors NVIDIA
cuDNN-frontend's pass-by-value model.

The rollout is non-breaking and additive. The new surface consists of:

- A `ScalarType` enum and cuDNN-named constructors, setters, and getters
  on the tensor ([§4.1](#41-frontend-tensor-flag)) mirroring
  cuDNN-frontend.
- Two defaulted booleans appended to the per-tensor flatbuffer schema
  (`is_pass_by_value`, `is_compile_time_constant`); the
  runtime-pass-by-value feature signal is derived from them, with no
  graph-level flag ([§4.3](#43-feature-signal-derived)).
- Two **breaking changes** to reach 1:1 cuDNN parity:
  `get_pass_by_value()` returns the value (not `bool`), and its former
  bool-predicate role moves to `get_is_pass_by_value()`
  ([§6](#6-compatibility-versioning-and-rollback)).
- A `1.2.0` plugin-SDK floor for the two runtime states (#1
  user-supplied, #2 frontend-injected). State #3 (compile-time, the
  default — including the plain constructor and `set_value`) is baked on
  the baseline `1.0.0`, exactly as before.
- Version-only per-graph provider filtering.

There is **no new public backend C-API entry**, **no new plugin SDK
symbol**, and **no new variant-pack attribute**. Runtime scalar values
reuse the existing `uid → void*` variant-pack map: a runtime
pass-by-value tensor's entry is a *host* pointer to the scalar,
delivered to the provider through the existing
`hipdnnEnginePluginExecuteOpGraph` device-buffer array.

**Binary-compatibility scope.** The backend computes the minimum
plugin API version each graph requires from the features the graph uses.
Graphs that do not opt into runtime pass-by-value impose no new
version requirement and continue to be served by existing plugins
unchanged. Graphs that opt in — any tensor marked runtime pass-by-value —
require plugins reporting `>= 1.2.0`; older plugins are filtered out of
the applicable set before they are asked about the graph, so a legacy
plugin can never silently mis-serve a runtime pass-by-value graph. See
[§4.6](#46-feature-detection-and-version-filtering) for the full
versioning model.

---

## 2. Problem Statement

### 2.1 End-user API surface

The desired end-user surface keeps the same execute API and mirrors
cuDNN-frontend. A scalar operand is created in one of the three
pass-by-value states ([§5.5](#55-compile-time-vs-runtime-mode)):

```cpp
// #3 compile-time constant (default) — value baked, no version elevation.
// hipDNN's plain scalar ctor lands in #3, diverging from cuDNN's plain
// ctor which sets pass_by_value (its runtime type-2, graph_properties.h:163-205).
auto k  = graph.tensor(TensorAttributes(0.125f));
auto c3 = graph.tensor(TensorAttributes(0.125f, ScalarType::COMPILE_TIME_CONST));

// #2 runtime, frontend-injected — value carried, delivered at execute.
auto s2 = graph.tensor(TensorAttributes(0.125f, ScalarType::RUNTIME_PARAM));

// #1 runtime, user-supplied — value supplied at execute, not baked.
auto scale = graph.tensor(...);
scale->set_as_runtime_parameter();
```

For states #1 and #2 the host-side value reaches the provider through the
existing variant-pack map, keyed by the tensor UID, exactly as device
buffers are — supplied by the user for #1, injected by the frontend for #2
([§4.8](#48-frontend-variant-pack-injection)). cuDNN-frontend realizes the
injected form with `extend_tensor_map_with_pass_by_value_tensors_`
([`graph_interface.h:198-217`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_interface.h#L198-L217)),
which emplaces a *host* pointer to the scalar into the same
`std::unordered_map<int64_t, void*>` used for device buffers. hipDNN
adopts the same transport.

### 2.2 hipDNN gap

hipDNN bakes scalar pass-by-value values into the operation graph at
compile time. In the frontend, `Tensor_attributes::set_value<T>()`
stores the value in a `std::variant` (`ValueVariant _value`); at build
time `detail::createOrFindTensorDesc`
([`frontend/include/hipdnn_frontend/detail/DescriptorHelpers.hpp`](../../frontend/include/hipdnn_frontend/detail/DescriptorHelpers.hpp))
visits that variant and calls `setDescriptorAttrTensorValue`, which
issues `backendSetAttribute(HIPDNN_ATTR_TENSOR_VALUE_EXT, raw bytes)`.
The value is then memcpy'd into the backend
`TensorDescriptor._data.value` flatbuffer union and frozen into the
serialized plan; nothing re-reads it at execute time.

Serving N values for the same scalar today therefore requires N
distinct compiled graphs and N cached execution plans. The backend
already anticipates this gap: `TensorDescriptor::finalize()`
([`backend/src/descriptors/TensorDescriptor.cpp`](../../backend/src/descriptors/TensorDescriptor.cpp)) carries the comment
"Pass-by-value tensors are currently required to supply a value at
descriptor creation time. In the future, pass-by-value tensors may
also support setting values through variant packs." This RFC realizes
that future direction.

### 2.3 Constraints

The design must:

1. **Preserve binary compatibility with existing plugins.** Plugins
   that do not adopt the new mechanism must continue to load and serve
   graphs that do not opt in. [RFC 0002](0002_PluginSdkDesign.md) commits hipDNN to a stable
   plugin contract; this RFC extends it without breaking that contract.
2. **Preserve the public backend C-API surface.** There is exactly
   one `hipdnnBackendExecute` today, and there will continue to be
   exactly one after this RFC lands.
3. **Keep the graph descriptor read-only after build.** State-#1 values
   are not stored on the graph at all; a state-#2 value is stored in the
   tensor flatbuffer for round-trip yet is variant-pack-delivered
   ([§4.8](#48-frontend-variant-pack-injection)). The graph descriptor
   stays read-only after build.
4. **A pass-by-value tensor is in exactly one of states #1/#2/#3**
   ([§5.5](#55-compile-time-vs-runtime-mode)), selected by the
   constructor/setter used; the four invalid combinations
   ([§4.7](#47-frontend-validation)) are rejected. cuDNN enforces
   analogous exclusivity in `validate()`
   ([`graph_properties.h:67-100`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_properties.h#L67-L100)).

---

## 3. Current System Overview

The hipDNN graph pipeline has four steps:

1. **Create graph.** Frontend builds a graph describing tensors,
   operations, and graph-level attributes.
2. **Validate, finalize, lower.** Frontend validates the graph and
   lowers it into the backend for plugin consumption.
3. **Plugins asked for applicability.** Backend asks each loaded
   plugin which of its engines can execute the finalized graph.
4. **Execute.** A variant pack carries per-execution payload
   (`tensorId → devicePtr`, workspace) into the chosen plugin engine.

The current scalar (pass-by-value) path threads through this pipeline
entirely at compile time:

```text
set_value(scalar)            // frontend: stores into ValueVariant _value
  → createOrFindTensorDesc   // build: std::visit(_value)
  → HIPDNN_ATTR_TENSOR_VALUE_EXT (raw bytes)
  → TensorDescriptor._data.value (flatbuffer TensorValue union)
  → frozen into serialized graph/plan
  → provider reads the value from the op-graph flatbuffer
```

Pass-by-value status is *implicit*: the frontend's
`Tensor_attributes::get_pass_by_value()`
([`frontend/include/hipdnn_frontend/attributes/TensorAttributes.hpp:97`](../../frontend/include/hipdnn_frontend/attributes/TensorAttributes.hpp#L97))
returns `!std::holds_alternative<std::monostate>(_value)`, i.e. "a value
has been set." The backend exposes a *read-only*
`HIPDNN_ATTR_TENSOR_IS_BY_VALUE` (1307) derived from
`_data.value.type != NONE`. At execute time the variant pack carries
only `HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS` (700),
`HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS` (701), intermediates (702),
workspace (703), and — for [RFC 0008](0008_OverridableTensorShapesDesign.md) graphs — the override attributes
(704–707). **No scalar value reaches a provider through the variant
pack today.**

---

## 4. Proposed Design

### 4.1 Frontend tensor flag

Pass-by-value tensors mirror cuDNN-frontend's public surface 1:1, with
two deliberate hipDNN divergences ([§5.1](#51-cudnn-api-surface-parity)).
The umbrella flag `is_pass_by_value` is true whenever the tensor is a
pass-by-value scalar; the second flag `is_compile_time_constant` and value
presence select the three states ([§5.5](#55-compile-time-vs-runtime-mode)).

**Enum.** A `ScalarType` selects a value-carrying tensor's state at
construction, mirroring cuDNN
([`graph_properties.h:42-46`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_properties.h#L42-L46)):

```cpp
namespace hipdnn_frontend::graph {
enum class ScalarType { RUNTIME_PARAM, COMPILE_TIME_CONST };
}
```

**Constructors.**

- `TensorAttributes(const T& scalar)` → **state #3** (compile-time),
  delegating to `set_value`. This diverges from cuDNN, whose plain scalar
  constructor sets `pass_by_value` (its runtime type-2)
  ([`graph_properties.h:163-205`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_properties.h#L163-L205)).
- `TensorAttributes(const T& scalar, ScalarType type)`: `RUNTIME_PARAM`
  → **state #2**, `COMPILE_TIME_CONST` → **state #3**
  ([`graph_properties.h:207-219`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_properties.h#L207-L219)).

**Setters.**

- `set_value<T>(v)` — retained, now **delegates to
  `set_compile_time_constant(v)`** → state #3.
- `set_compile_time_constant(pass_by_values_t v)` → state #3
  ([`graph_properties.h:390-399`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_properties.h#L390-L399)).
- `set_as_runtime_parameter()` → state #1; it **clears any prior value**
  (a deliberate divergence from cuDNN, whose `set_as_runtime_parameter`
  leaves `pass_by_value` set,
  [`graph_properties.h:401-406`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_properties.h#L401-L406)) —
  hipDNN reaches #2 only via the `RUNTIME_PARAM` constructor, so the
  value-less runtime path clears the value.
- `set_is_pass_by_value(bool)` — retained (cuDNN has it,
  [`graph_properties.h:373-377`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_properties.h#L373-L377));
  setting `true` with no value yields state #1.

**Getters.**

- `get_pass_by_value()` returns the **value** (`std::optional`-style /
  value variant), not `bool`, and is mode-gated: the value for #2, empty
  for #1 and #3
  ([`graph_properties.h:358-366`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_properties.h#L358-L366)).
- `get_compile_time_constant()` returns the value for #3, empty otherwise
  ([`graph_properties.h:385-388`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_properties.h#L385-L388)).
- `get_is_pass_by_value()` is the umbrella predicate (true for all three)
  ([`graph_properties.h:368-371`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_properties.h#L368-L371)).

**Breaking changes** (from the current hipDNN API): (a)
`get_pass_by_value()` returns the value instead of `bool`; (b) its former
"is-pass-by-value" bool-predicate role moves to `get_is_pass_by_value()`.
`set_value` now routes to compile-time (#3). State #2 is reachable only
via the `RUNTIME_PARAM` constructor (no dedicated setter).

`Graph::tensor(const TensorAttributes&)`
([`Graph.hpp`](../../frontend/include/hipdnn_frontend/Graph.hpp)) is
unchanged; no `graph.tensor(scalar, ScalarType)` factory is added.

A pass-by-value tensor is a single-element host scalar, so the existing
scalar conventions (dims/strides `{1}`) apply.

Unlike override shapes — whose frontend setters
(`set_override_shape_enabled` and the override execute overload) are
compiled only under `#ifdef HIPDNN_ENABLE_SDPA` in
[`Graph.hpp`](../../frontend/include/hipdnn_frontend/Graph.hpp) — the
pass-by-value frontend API is **not** SDPA-gated. Scalar operands such
as epsilon, alpha, and beta are general, not SDPA-specific, so
`set_is_pass_by_value` is always compiled.

### 4.2 Tensor schema addition

The per-tensor flags are persisted so a provider reading the serialized
op graph can identify pass-by-value scalars and tell the three states
apart. The flatbuffer `TensorAttributes` table gains **two** defaulted
booleans appended as its last fields, re-using the existing `value` union
(no third flag):

```
is_pass_by_value: bool = false;
is_compile_time_constant: bool = false;
```

This is the append-only, defaulted-field pattern used in `develop` by the
override-shape graph and plan flags ([RFC 0008](0008_OverridableTensorShapesDesign.md)), wire-compatible per [RFC 0005](0005_Versioning.md):
a pre-feature graph deserialized in a runtime that understands the fields
reads both as `false` on every tensor. Both flags round-trip through
descriptor pack/unpack alongside the `value`, and both are persisted so
`get_pass_by_value` / `get_compile_time_constant` / `get_is_pass_by_value`
are correct after graph deserialize. cuDNN keeps its analogous
`pass_by_values` in its FE JSON across serialize/deserialize
([`graph_interface.h:1592-1607`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_interface.h#L1592-L1607),
[`graph_interface.h:1671-1710`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_interface.h#L1671-L1710));
hipDNN keeps them in the tensor flatbuffer instead.

The backend `TensorDescriptor` gains two **settable** attributes:
`HIPDNN_ATTR_TENSOR_IS_PASS_BY_VALUE_EXT` (ID 1308) carrying the umbrella
flag and `HIPDNN_ATTR_TENSOR_IS_COMPILE_TIME_CONSTANT_EXT` (ID 1309)
carrying the compile-time-constant flag. The existing **read-only**
`HIPDNN_ATTR_TENSOR_IS_BY_VALUE` (1307) remains derived from
`value.type != NONE` (value-present) but is **demoted** from the baked
discriminator — it is now true for both #2 and #3, so the authoritative
compile-time signal is 1309 (`is_compile_time_constant`). The settable
flags round-trip through descriptor pack/unpack alongside the existing
value-carrying attributes; the precedent is the unpack of
`HIPDNN_ATTR_TENSOR_IS_BY_VALUE` + `HIPDNN_ATTR_TENSOR_VALUE_EXT` in
[`DescriptorUnpackHelpers.hpp`](../../frontend/include/hipdnn_frontend/detail/DescriptorUnpackHelpers.hpp).

Providers read the flags through `isPassByValue()` and
`isCompileTimeConstant()` accessors added to the op-graph tensor wrapper
([`ITensorAttributesWrapper`](../../flatbuffers_sdk/include/hipdnn_flatbuffers_sdk/flatbuffer_utilities/TensorAttributesWrapper.hpp)),
mirroring the existing `isVirtual()`. These are the accessors the provider
example in [§4.5](#45-provider-contract) consumes.

### 4.3 Feature signal (derived)

The backend's runtime-pass-by-value feature signal is **derived from the
per-tensor flags** ([§4.2](#42-tensor-schema-addition)): a graph requires
runtime pass-by-value support iff it contains at least one tensor with
`is_pass_by_value == true && is_compile_time_constant == false` (states #1
or #2). There is **no** graph-level schema field and **no** graph-level
setter — the per-tensor flags are the single source of truth, and a
separate graph boolean would only create a cache that can desync from them
([§5.3](#53-derive-feature-detection-from-the-tensor-schema)).

The backend computes the signal with a `readIsPassByValueEnabled` helper
that scans the serialized op graph at applicability time; a graph with
only compile-time (#3) scalars (every such tensor has
`is_compile_time_constant == true`) yields `false` and needs no version
elevation. This is the single signal feature detection consumes
([§4.6](#46-feature-detection-and-version-filtering)).

### 4.4 Execute-time transport

Runtime scalar values travel in the existing `Graph::execute()` map
with **no new overload and no new variant-pack attribute**. The core
execute overload is unchanged:

```cpp
Error execute(hipdnnHandle_t handle,
              std::unordered_map<int64_t, void*>& variantPack,
              void* workspace) const;
```

For each runtime pass-by-value tensor, the caller inserts an entry whose
value is a **host pointer** to the scalar:

```cpp
float scaleValue = 0.125f;
variantPack[scale->get_uid()] = &scaleValue;   // host pointer
graph.execute(handle, variantPack, workspace);
```

This is exactly cuDNN-frontend's
`extend_tensor_map_with_pass_by_value_tensors_` behavior
([`graph_interface.h:198-217`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_interface.h#L198-L217)).
The frontend
variant-pack builder `detail::populateBaseVariantPackDescriptor`
([`frontend/include/hipdnn_frontend/detail/VariantPackHelpers.hpp:19`](../../frontend/include/hipdnn_frontend/detail/VariantPackHelpers.hpp#L19))
is unchanged: the host pointer is an ordinary
`HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS` entry.

### 4.5 Provider contract

A provider reporting plugin SDK API version `>= 1.2.0` must, for any
tensor matching the runtime discriminator
`isPassByValue() && !isCompileTimeConstant()` (states #1 and #2), read
the scalar from that UID's slot in the `device_buffers` array **as a host
pointer** at execute time. A compile-time constant (#3,
`isCompileTimeConstant() == true`) is read from the op-graph flatbuffer as
today.

The plugin entry point is unchanged:

```c
hipdnnPluginStatus_t
hipdnnEnginePluginExecuteOpGraph(hipdnnEnginePluginHandle_t handle,
                                 hipdnnEnginePluginExecutionContext_t execution_context,
                                 void* workspace,
                                 const hipdnnPluginDeviceBuffer_t* device_buffers,
                                 uint32_t num_device_buffers);
```

The runtime-slot discriminator is the conjunction
`isPassByValue() && !isCompileTimeConstant()`: the umbrella flag alone is
not sufficient, because a compile-time constant (#3) is *also*
pass-by-value. Value presence is **not** the discriminator either — a #2
(`RUNTIME_PARAM`) tensor carries a baked `value` yet is runtime — so the
authoritative signal is `is_compile_time_constant`, exposed as
`isCompileTimeConstant()`. The read-only `HIPDNN_ATTR_TENSOR_IS_BY_VALUE`
(1307) is true for both #2 and #3, so it cannot stand in for this check.

**#2 caveat.** A `RUNTIME_PARAM` tensor carries a baked `value` in the
flatbuffer (and 1307 is true for it), but because
`is_compile_time_constant == false` the provider MUST read the
variant-pack slot and MUST NOT use the baked value; 1307 must not be used
to shortcut to the baked path here. States #1 and #2 reach the provider
identically (a host pointer in `device_buffers`) and are handled the same.

Each `hipdnnPluginDeviceBuffer_t` carries its `uid`, and
[§4.2](#42-tensor-schema-addition) adds the `isPassByValue()` /
`isCompileTimeConstant()` accessors, so a provider builds the set of
runtime pass-by-value UIDs once from the op graph and interprets the
matching `device_buffers` slots as host pointers at execute time:

```cpp
// Setup (once per graph): which UIDs are RUNTIME pass-by-value (#1 or #2)?
std::unordered_set<int64_t> hostScalarUids;
for (auto const& tensor : opGraph.tensors()) {        // ITensorAttributesWrapper
    if (tensor.isPassByValue() && !tensor.isCompileTimeConstant()) {
        hostScalarUids.insert(tensor.uid());
    }
}

// Execute: device_buffers carries {uid, ptr} for every bound tensor.
for (uint32_t i = 0; i < num_device_buffers; ++i) {
    const hipdnnPluginDeviceBuffer_t& buf = device_buffers[i];
    if (hostScalarUids.count(buf.uid)) {
        // buf.ptr is a HOST pointer to the scalar; read it on the host
        // using the tensor's declared data_type, e.g.:
        float scale = *static_cast<const float*>(buf.ptr);
        // ... feed `scale` into the kernel launch as a by-value argument.
    } else {
        // buf.ptr is an ordinary DEVICE pointer, handled as today.
    }
}
```

The loop above runs on the **fresh-build** path, where the op graph is
available. On a **deserialized** plan
([RFC 0009](0009_CompiledPlanSerialization.md)) the op graph and
per-tensor attributes are not reconstructed
([§5.6](#56-deserialized-plan-support-via-the-provider-payload)): the
plan is rebuilt from the engine ID, workspace size, a bare tensor-UID
list, and the provider's opaque `plugin_payload` alone. A provider that
supports runtime pass-by-value (reports `1.2.0`) must therefore:

1. **Persist** the runtime pass-by-value UID set (`hostScalarUids`
   above) into its serialized `plugin_payload` and restore it on
   deserialize; the host-scalar identity is otherwise lost across
   serialization.
2. **Version** that payload. The `plugin_payload` is opaque to hipDNN
   and its versioning is plugin-owned
   ([RFC 0009](0009_CompiledPlanSerialization.md#envelope-format)), so
   the provider must stamp a format version (or kind) it can recognize.
3. **Reject** a payload whose version/kind it cannot interpret,
   returning a deserialize error **before** reading any slot. This is
   the only guard against a downgraded `< 1.2.0` provider, re-bound by
   `engineId` to a newer payload, dereferencing a host pointer as device
   memory — a memory-unsafe failure, not merely a wrong result.

These three obligations are part of what reporting `1.2.0` means — the
same trust-the-version contract as reading the host-scalar slot itself —
because hipDNN core performs no version check on the deserialized path
([§4.6](#46-feature-detection-and-version-filtering)). The hipDNN
envelope `version` field versions the plan layout, not the opaque payload
contents, so it cannot stand in for this provider check.

**Pointer lifetime.** The host pointer is valid for the duration of the
execute call only; the provider must not retain or dereference it after
returning, matching the existing device-buffer contract.

### 4.6 Feature detection and version filtering

The backend maps each graph to the minimum plugin API version it
requires and filters plugins against that mapping. The
runtime-pass-by-value input to that mapping is the derived per-tensor
signal from [§4.3](#43-feature-signal-derived) (`readIsPassByValueEnabled`);
the rest of this section covers how that signal — alongside the override
flag — becomes a version floor.

`computeMinimumPluginApiVersion`
([`backend/src/plugin/EnginePluginResourceManager.cpp:86-99`](../../backend/src/plugin/EnginePluginResourceManager.cpp#L86-L99)) becomes
feature-aware. Today it maps a single override-shape boolean to either
the baseline `1.0.0` or the override minimum `1.1.0`. It is extended to
also account for the pass-by-value flag and return the **maximum** of the
per-feature minimums:

| Enabled feature(s) | Required plugin API version |
|--------------------|-----------------------------|
| None | `1.0.0` (baseline) |
| Override shapes only | `1.1.0` |
| Runtime pass-by-value — any tensor `is_pass_by_value && !is_compile_time_constant` (#1/#2), with or without override | `1.2.0` |

A new version constant is added in
[`plugin_sdk/include/hipdnn_plugin_sdk/PluginVersionConstants.hpp`](../../plugin_sdk/include/hipdnn_plugin_sdk/PluginVersionConstants.hpp):

```cpp
inline constexpr std::string_view K_PASS_BY_VALUE_MIN_API_VERSION = "1.2.0";
```

and the canonical ABI macros in
[`plugin_sdk/include/hipdnn_plugin_sdk/engine_api_version.h`](../../plugin_sdk/include/hipdnn_plugin_sdk/engine_api_version.h) bump
`HIPDNN_ENGINE_API_VERSION_MINOR` from `1` to `2`
(`HIPDNN_ENGINE_API_VERSION = "1.2.0"`).

**Filtering is version-only.** `getApplicableEngineIds`
([`EnginePluginResourceManager.cpp:341-407`](../../backend/src/plugin/EnginePluginResourceManager.cpp#L341-L407)) already skips any plugin
whose `parsedApiVersion() < requiredVersion` (line 362). Once
`requiredVersion` is `1.2.0` for a pass-by-value graph, every plugin
reporting less — including the `1.0.0` fallback assigned to plugins
that do not export `hipdnnPluginGetApiVersion` — is dropped from the
applicable set **before** it is asked about the graph. In other words,
a legacy plugin is rejected for any pass-by-value-enabled graph by the
host on the plugin's behalf, purely from its reported version. When no
plugin qualifies, the graph fails with a clean "no applicable engines"
result; it is never silently mis-served with a garbage scalar.

There is no per-symbol predicate (see
[§5.4](#54-version-only-filtering)) and no dispatch-time re-check in the
core: the applicability filter
([`EnginePluginResourceManager.cpp:362`](../../backend/src/plugin/EnginePluginResourceManager.cpp#L362))
is the single gate, applied when the plan is built. A serialized plan
([RFC 0009](0009_CompiledPlanSerialization.md)) re-binds by the baked
`engineId` with no version re-filter, but its runtime pass-by-value
state lives in the provider's opaque payload, which the provider
versions and validates on deserialize; that path is covered by the
provider contract, not a core gate
([§5.6](#56-deserialized-plan-support-via-the-provider-payload)).

Version parsing and comparison use the existing
`hipdnn_data_sdk::utilities::Version`
([`data_sdk/include/hipdnn_data_sdk/utilities/VersionUtils.hpp`](../../data_sdk/include/hipdnn_data_sdk/utilities/VersionUtils.hpp)).

### 4.7 Frontend validation

`Tensor_attributes::validate()` enforces, in addition to the existing
checks. States #1/#2/#3 are all valid; the rules reject only the four
invalid combinations, all unreachable via the public constructors and
setters:

1. **Value implies the umbrella.** A baked `value` implies
   `is_pass_by_value == true`; `set_value` / `set_compile_time_constant`
   and the constructors set the umbrella flag.
2. `INVALID_VALUE` if `is_compile_time_constant == true` and **no
   value**.
3. `INVALID_VALUE` if `value present` and `!is_pass_by_value`.
4. `INVALID_VALUE` if `is_compile_time_constant == true` and
   `!is_pass_by_value`.
5. **Virtual exclusion.** `INVALID_VALUE` if `virtual &&
   is_pass_by_value`.

There is **no** "value present ⇒ compile-time mode" rule — that was the
old two-state design; #2 is value + runtime. These mirror
cuDNN-frontend's `validate()` (the value⇒umbrella and virtual-exclusion
analogues and the compile-time-constant constraints,
[`graph_properties.h:67-100`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_properties.h#L67-L100)).

`detail::validateScalarParameter`
([`frontend/include/hipdnn_frontend/node/detail/Utilities.hpp`](../../frontend/include/hipdnn_frontend/node/detail/Utilities.hpp), ~line
475) currently requires `get_pass_by_value()` to be true for required
scalar inputs (epsilon, SDPA scale, etc.). It is relaxed to accept any
pass-by-value scalar (`get_is_pass_by_value() == true`), whether the
value is baked (#3), frontend-injected (#2), or user-supplied (#1). As
today, the actual numeric value cannot be checked at build time for
runtime scalars because it is not yet available.

### 4.8 Frontend variant-pack injection

For a state-#2 (`RUNTIME_PARAM`) tensor the frontend owns a host-side
copy of the value: it is snapshot at graph build and stays stable across
every execute, mirroring cuDNN-frontend's `cached_pass_by_value`
(collected once at build, reused every execute,
[`graph_interface.h:978-982`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_interface.h#L978-L982))
and delivered by `extend_tensor_map_with_pass_by_value_tensors_`
([`graph_interface.h:198-217`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_interface.h#L198-L217))
from that build-time cache at execute
([`graph_interface.h:649-659`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_interface.h#L649-L659)).

At `Graph::execute()` the frontend injects `uid → &cachedValue` (a host
pointer to its own copy) into the variant pack before dispatch — the graph
fills the slot, not the user. From the provider's perspective a #2 slot is
identical to a #1 slot: a host pointer under a runtime-discriminated UID
([§4.5](#45-provider-contract)).

**Path scope.** Injection operates on the fresh-build path and the
graph-(de)serialize path, where the tensor flatbuffer carries the value
and both flags (so the frontend restores its cached copy after graph
deserialize). On the [RFC 0009](0009_CompiledPlanSerialization.md)
compiled-plan path, `deserializeBackendPlan` reconstructs no per-tensor
attributes ([§5.6](#56-deserialized-plan-support-via-the-provider-payload);
[`ExecutionPlanDescriptor.cpp:416-477`](../../backend/src/descriptors/ExecutionPlanDescriptor.cpp#L416-L477)),
so the frontend has no value to inject; a #2 tensor round-tripped through
`to_compiled_plan_binary` therefore degrades to #1 semantics on that path
— the caller must supply the value at execute. This limitation is explicit
and covered by test ([§10](#10-testing-plan)).

**Version-cost divergence from cuDNN.** cuDNN's type-2 runs on any engine
because its runtime-fusion JIT can fold the value at build; hipDNN #2 is
variant-pack-delivered and therefore requires `1.2.0` even though the
value is build-fixed. This is observable only as provider availability,
never a wrong result.

---

## 5. Key Design Decisions

### 5.1 cuDNN API-surface parity

**Decision**: mirror cuDNN-frontend's pass-by-value surface 1:1 — the
`ScalarType` enum
([`graph_properties.h:42-46`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_properties.h#L42-L46)),
the two constructors
([`graph_properties.h:163-205`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_properties.h#L163-L205),
[`graph_properties.h:207-219`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_properties.h#L207-L219)),
`set_is_pass_by_value` / `get_is_pass_by_value`
([`graph_properties.h:373-377`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_properties.h#L373-L377),
[`graph_properties.h:368-371`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_properties.h#L368-L371)),
`set_compile_time_constant` / `get_compile_time_constant`
([`graph_properties.h:390-399`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_properties.h#L390-L399),
[`graph_properties.h:385-388`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_properties.h#L385-L388)),
`set_as_runtime_parameter`
([`graph_properties.h:401-406`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_properties.h#L401-L406)),
and `get_pass_by_value` returning the value
([`graph_properties.h:358-366`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_properties.h#L358-L366)) —
with two deliberate hipDNN divergences.

**Rationale**: adopting cuDNN's names and shapes lets cuDNN users port
with no concept translation, and the three-state model
([§5.5](#55-compile-time-vs-runtime-mode)) covers cuDNN's full
fused-constant-vs-execute-time surface
([`graph_properties.h:58-64`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_properties.h#L58-L64)).

**Divergences (decisions, not gaps)**:

1. The single-value constructor `TensorAttributes(v)` and `set_value(v)`
   default to #3 (compile-time), not cuDNN's value-carrying runtime type-2
   ([`graph_properties.h:163-205`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_properties.h#L163-L205));
   hipDNN's default stays on the backward-compatible baked path.
2. `get_pass_by_value()` is mode-gated over a **single** re-used value
   member, whereas cuDNN keys the mode on two separate value members
   (`pass_by_value`, `compile_time_constant_value`,
   [`graph_properties.h:122-127`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_properties.h#L122-L127)).

**Breaking changes** (from the current hipDNN API): `get_pass_by_value()`
returns the value instead of `bool`, and its former bool-predicate role
moves to `get_is_pass_by_value()` ([§4.1](#41-frontend-tensor-flag)).

### 5.2 Reuse the variant-pack pointer map

**Decision**: transport runtime scalar values as host pointers in the
existing `uid → void*` variant-pack map, delivered through the existing
`hipdnnEnginePluginExecuteOpGraph` `device_buffers` array. No new
variant-pack attribute and no new plugin symbol.

**Rationale**: this is exactly cuDNN-frontend's model, so consumers
port with no signature changes, and it keeps the new surface minimal —
no additions to `populateBaseVariantPackDescriptor`, the plugin C ABI,
or the 700–799 attribute range. A pass-by-value scalar is logically
just another per-execution pointer.

**Trade-off**: the provider must consult the two per-tensor flags via the
discriminator `isPassByValue() && !isCompileTimeConstant()` to know that a
given `device_buffers` slot holds a *host* pointer rather than a device
pointer. That discriminator is authoritative
([§5.5](#55-compile-time-vs-runtime-mode)).

### 5.3 Derive feature detection from the tensor schema

**Decision**: detect the runtime-pass-by-value feature by deriving it
from the per-tensor flags (any tensor with `is_pass_by_value == true &&
is_compile_time_constant == false`, i.e. states #1 or #2) rather than
adding a separate graph-level `is_pass_by_value_enabled` flag. The
per-tensor schema fields are the single source of truth.

**Rationale**: the per-tensor flags are already **mandatory** — the
two-flag discriminator `is_pass_by_value && !is_compile_time_constant` is
the only signal distinguishing a runtime host-scalar slot from an
ordinary device buffer or a baked #3 constant
([§4.5](#45-provider-contract)). A graph-level flag would therefore be
a *denormalized cache* of a tensor-derived predicate, and a cache that
can disagree with its source. The disagreement is not symmetric: if the
graph flag were `false` while a runtime tensor existed (e.g. a raw
backend C-API caller sets the tensor attribute but not the graph
attribute), the applicability filter would **not** elevate the required
version, a sub-`1.2.0` plugin could be selected, and it would read a host
pointer as a device pointer — the exact memory-unsafe failure the version
gate exists to prevent. Deriving makes that desync unrepresentable, drops
a schema field and a backend enum, and removes the frontend
serialize/deserialize/reset plumbing a graph flag would need.

**Trade-off**: a one-time `O(tensors)` walk of the already-materialized
serialized graph per applicability query, instead of a single bool read.
This is negligible — the walk runs once per query (not per plugin) on a
non-hot path, over a flatbuffer already in hand
([`EnginePluginResourceManager.cpp:341-407`](../../backend/src/plugin/EnginePluginResourceManager.cpp#L341-L407)).

**Divergence from override**: [RFC 0008](0008_OverridableTensorShapesDesign.md)
uses a graph-level flag because override has **no** per-tensor schema
field to derive from — its graph flag is the sole source of truth. Runtime
pass-by-value does have a per-tensor source of truth, so mirroring
override here would import a desync risk override never had.

### 5.4 Version-only filtering

**Decision**: gate providers on reported version alone; do not add an
optional plugin symbol or a `hasPassByValueExecute()` predicate.

**Rationale**: [RFC 0008](0008_OverridableTensorShapesDesign.md) needed a per-symbol check because it introduced
a new plugin entry point (`hipdnnEnginePluginExecuteOpGraphWithOverrides`)
that a plugin could fail to export even at the right version. Runtime
pass-by-value introduces no new entry point — the value arrives through
the unchanged `device_buffers` array — so there is nothing to probe.
The version contract fully expresses capability: a provider that reports
`1.2.0` asserts it reads host-scalar slots correctly.

**Trade-off**: a provider that bumps its reported version to `1.2.0`
but mishandles host pointers cannot be caught by a symbol check; the
host trusts the version contract. This is covered by integration tests
([§10](#10-testing-plan)) rather than a runtime guard.

### 5.5 Compile-time vs runtime mode

**Decision**: a pass-by-value tensor is in exactly one of three states,
selected by the two flags `is_pass_by_value` / `is_compile_time_constant`
and value presence over a single re-used `ValueVariant _value` member.
All three carry the umbrella `is_pass_by_value == true`:

| State | Creation (frontend) | `value` | `is_compile_time_constant` | `get_pass_by_value()` | `get_compile_time_constant()` | `get_is_pass_by_value()` | Delivery | Provider floor |
|---|---|---|---|---|---|---|---|---|
| **#1** runtime, user-supplied | `set_as_runtime_parameter()`; or `set_is_pass_by_value(true)` with no value | ∅ | false | ∅ | ∅ | true | user supplies host ptr in variant pack | `1.2.0` |
| **#2** runtime, frontend-injected | `TensorAttributes(v, ScalarType::RUNTIME_PARAM)` | v | false | v | ∅ | true | frontend injects host ptr in variant pack ([§4.8](#48-frontend-variant-pack-injection)) | `1.2.0` |
| **#3** compile-time constant **(default)** | `TensorAttributes(v)`; `set_value(v)`; `TensorAttributes(v, ScalarType::COMPILE_TIME_CONST)`; `set_compile_time_constant(v)` | v | true | ∅ | v | true | baked in op-graph flatbuffer, read via existing path | baseline `1.0.0` |

(∅ = empty / `std::monostate`.)

**Rationale**: the three states mirror cuDNN-frontend's public surface
while re-using a single value member instead of cuDNN's two separate value
slots (`pass_by_value` + `compile_time_constant_value`,
[`graph_properties.h:122-127`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_properties.h#L122-L127)).
State is a function of the two flags plus value presence, so no second
value slot is needed. The getters are mode-gated exactly as in cuDNN:
`get_pass_by_value()` returns the value only for #2 (value present &&
`!is_compile_time_constant`) and is empty for #1 and #3
([`graph_properties.h:358-366`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_properties.h#L358-L366));
`get_compile_time_constant()` returns the value only for #3
([`graph_properties.h:385-388`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_properties.h#L385-L388));
`get_is_pass_by_value()` is true for all three
([`graph_properties.h:368-371`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_properties.h#L368-L371)).

**Provider runtime discriminator**: `isPassByValue() &&
!isCompileTimeConstant()` selects the host-pointer slots (states #1 and
#2). It replaces value-presence checks, which would misclassify #2 — a
runtime scalar that nonetheless carries a value. The provider contract
([§4.5](#45-provider-contract)) spells this out.

**Deliberate divergence from cuDNN**: the single-value constructor
`TensorAttributes(v)` (and `set_value(v)`) defaults to #3 (compile-time),
whereas cuDNN's plain scalar constructor sets `pass_by_value` (its
value-carrying runtime type-2,
[`graph_properties.h:163-205`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_properties.h#L163-L205)).
hipDNN keeps the default on the baked/old-API path so existing callers
need no version elevation. Observable consequence: `get_pass_by_value()`
on a plain-constructed tensor returns empty in hipDNN but the value in
cuDNN. State #2 is the explicit parity path for a value-carrying runtime
scalar and is reached only via the `RUNTIME_PARAM` constructor
([§4.8](#48-frontend-variant-pack-injection)).

### 5.6 Deserialized-plan support via the provider payload

**Decision**: add no execution-plan schema field and no hipDNN
dispatch-time version gate. A provider that supports runtime
pass-by-value (a) persists the set of runtime pass-by-value UIDs into
its opaque `plugin_payload` when a plan is serialized and reconstructs
it on deserialize, and (b) versions that payload so a plugin that cannot
interpret it fails the deserialize rather than mis-reading it.

**Rationale**: on the fresh-build path the provider derives the
runtime-pass-by-value UID set from the op graph
(`isPassByValue() && !isCompileTimeConstant()`,
[§4.5](#45-provider-contract)). On the
deserialized path that op graph does not exist:
`ExecutionPlanDescriptor::deserializeBackendPlan`
([`ExecutionPlanDescriptor.cpp:416-477`](../../backend/src/descriptors/ExecutionPlanDescriptor.cpp#L416-L477))
reconstructs the plan from the serialized plan alone — `engineId`,
`workspace_size`, the opaque `plugin_payload`, and a bare `tensor_uids`
list — and rebuilds the provider's execution context via
`createExecutionContextFromSerialized` with only that payload. The bare
`tensor_uids` list does not distinguish host scalars from device
tensors, so the only place host-scalar identity can survive
serialization is inside the provider's own payload, which the provider
already owns and versions. Skew safety reduces to that existing
payload-versioning contract: a downgraded plugin rebound by `engineId`
to a payload it does not understand rejects it, exactly as for any other
payload-format change — no pass-by-value-specific core mechanism is
needed.

This follows [RFC 0009](0009_CompiledPlanSerialization.md)'s explicit
payload-ownership rule: the serialized envelope deliberately omits the
op graph, and *"plugins that need graph-derived data must store it in
their own payload"*
([RFC 0009, Envelope Format](0009_CompiledPlanSerialization.md#envelope-format)).
The runtime pass-by-value UID set is precisely such graph-derived data —
needed after deserialize, when the op graph is gone — so persisting it
in the payload extends the established serialization architecture rather
than bolting on a parallel mechanism.

**Trade-off**: serialized/deserialized-path safety depends on the
provider obeying the §4.5 payload-versioning contract; hipDNN core no
longer enforces a version floor on that path, and the skew failure is
memory-unsafe rather than merely wrong-result. This is the same
trust-the-version posture already accepted for the host-pointer read
itself ([§8.1](#81-provider-reports-120-but-mishandles-host-pointers)):
the core trusts a provider that reports `1.2.0`. It is preferred over the
alternative execution-plan-field design — a sibling
`is_pass_by_value_enabled` on `SerializedExecutionPlan`, peer to the
existing `is_override_shape_enabled`
([`execution_plan.fbs:12`](../../flatbuffers_sdk/schemas/execution_plan.fbs#L12))
— not because it avoids new schema surface (that precedent already
exists) but because it keeps host-scalar identity where RFC 0009 says
graph-derived data belongs: in the provider's own versioned payload,
rather than in a core gate the provider does not control.

**Retrofit limit**: the skew window closes only when the *older*
same-`engineId` release already rejected payloads it could not
interpret. A provider release that shipped before runtime pass-by-value
and did not practice defensive payload versioning cannot be made safe
retroactively — it will read the newer payload's bytes regardless. The
guarantee therefore covers only providers that versioned-and-rejected
from the start; adopting that discipline is a precondition of shipping
runtime pass-by-value ([§4.5](#45-provider-contract),
[Step 7](#step-7-provider-adoption)).

### 5.7 Frontend variant-pack injection for RUNTIME_PARAM

**Decision**: build a frontend variant-pack injection path for state #2
(`RUNTIME_PARAM`) tensors ([§4.8](#48-frontend-variant-pack-injection)):
the `Graph` snapshots the tensor's value into host-side storage at build
and injects `uid → &cachedValue` (a host pointer) into the variant pack
at `Graph::execute()` before dispatch — the graph fills the slot, not the
user.

**Rationale**: this reproduces cuDNN-frontend's type-2 delivery exactly —
a `cached_pass_by_value` collected once at build
([`graph_interface.h:978-982`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_interface.h#L978-L982))
and re-injected every execute via
`extend_tensor_map_with_pass_by_value_tensors_`
([`graph_interface.h:198-217`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_interface.h#L198-L217),
[`graph_interface.h:649-659`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_interface.h#L649-L659)) —
so a value-carrying runtime scalar reaches the provider through the same
host-pointer slot as a user-supplied #1 tensor, with no user action.

**Trade-off**: adds frontend host-storage plus an execute-time injection
step, and imposes the `1.2.0` provider floor on a value fixed at build
(the §4.8 version-cost divergence from cuDNN, whose type-2 runs on any
engine). It is observable only as provider availability, never a wrong
result. On the [RFC 0009](0009_CompiledPlanSerialization.md) compiled-plan
path the frontend has no reconstructed value to inject, so a #2 tensor
degrades to caller-supplied (#1 semantics) after `to_compiled_plan_binary`
([§4.8](#48-frontend-variant-pack-injection)).

---

## 6. Compatibility, Versioning, and Rollback

**Upgrade path.** Existing plugins (in-tree and out-of-tree) continue
to serve non-pass-by-value graphs and state-#3 (compile-time, default)
graphs unchanged; they require no rebuild. A plugin adopts runtime
pass-by-value by reading host-scalar slots for tensors matching
`isPassByValue() && !isCompileTimeConstant()` (states #1/#2) and bumping
its reported API version to `1.2.0`. State #3 needs no new provider;
#1/#2 need `1.2.0`. Each plugin migrates on its own schedule.

**Breaking changes.** Reaching 1:1 cuDNN API parity changes two frontend
accessors:

- `get_pass_by_value()` now returns the value (a value/optional variant),
  not `bool`; its former "is-pass-by-value" bool-predicate role moves to
  the new `get_is_pass_by_value()`
  ([§4.1](#41-frontend-tensor-flag)).
- Internal callers that used `get_pass_by_value()` to mean "a baked value
  exists" must switch to a value-presence check via `get_value_variant()`;
  the known callsite is `createOrFindTensorDesc` in
  [`DescriptorHelpers.hpp`](../../frontend/include/hipdnn_frontend/detail/DescriptorHelpers.hpp).

These are source-level changes to a frontend header API; no wire format
or plugin ABI is affected.

**Version skew.** An older plugin paired with a #1/#2 graph is filtered
out by the per-graph version gate
([§4.6](#46-feature-detection-and-version-filtering)). If no plugin
reports `>= 1.2.0`, applicability returns a clean "no applicable
engines" result. A legacy plugin never receives a #1/#2 graph and
therefore never reads a host pointer where it expects a baked value
or a device buffer, so on the applicability-filtered fresh-build path
there is no silent wrong-result path.

The serialized/deserialized path is not core-gated. A plan re-bound by
`engineId` to a downgraded plugin relies on the provider versioning its
opaque `plugin_payload`: a provider that versions it correctly rejects
the mismatched payload at deserialize rather than mis-reading a host
pointer, while a provider that does not is the residual risk
[§5.6](#56-deserialized-plan-support-via-the-provider-payload) discloses,
since hipDNN core no longer enforces a version floor on that path.

**Non-breaking schema compatibility.** The two new schema fields
(`TensorAttributes.is_pass_by_value` and
`TensorAttributes.is_compile_time_constant`) are appended, defaulted
`false`, and wire-compatible per [RFC 0005](0005_Versioning.md). A graph
serialized before this feature, deserialized in a runtime that
understands the fields, reads `false` on every tensor — i.e. it is
treated as a non-pass-by-value graph and served by any plugin, exactly
as before.

**Rollback.** The feature is inert unless a caller creates a state-#1 or
state-#2 tensor. Reverting a caller to the default compile-time
`set_value(scalar)` / plain constructor (state #3) restores the existing
baked-value path with zero schema migration: the value is baked, no
version is elevated, and `computeMinimumPluginApiVersion` returns the
baseline. No data migration or plan invalidation is required.

---

## 7. Comparison to Ragged and Override Tensor Support

Runtime pass-by-value reuses the **shipped** compatibility machinery from
override shapes ([RFC 0008](0008_OverridableTensorShapesDesign.md)) — the
only tensor-feature template currently in `develop` — and takes the
lightest-weight choice at each axis. Ragged tensors
([RFC 0014](0014_RaggedTensors.md)) are included as a **proposed** design
point (that RFC is not yet implemented in `develop`) because their shape
is the closest structural analogue.

| Axis | Runtime pass-by-value (this RFC) | Ragged tensors ([RFC 0014](0014_RaggedTensors.md), proposed) | Override shapes ([RFC 0008](0008_OverridableTensorShapesDesign.md)) |
|------|----------------------------------|---------------------------|----------------------------|
| Tensor schema change | append `is_pass_by_value`, `is_compile_time_constant` | append `ragged_offset_tensor_uid`, `alignment` | none (graph-level only) |
| Graph schema change | none (derived from tensor flag) | append `is_ragged_tensor_enabled: bool` | append `is_override_shape_enabled: bool` |
| Execute transport | reuse `uid → void*` map (host pointer) | variant pack unchanged | new variant-pack attrs 704–707 |
| New plugin SDK symbol | none | none | `hipdnnEnginePluginExecuteOpGraphWithOverrides` |
| Provider filtering | `computeMinimumPluginApiVersion`, version-only | `computeMinimumPluginApiVersion` | `computeMinimumPluginApiVersion` + per-symbol `hasOverrideExecute()` |
| Required plugin floor | `1.2.0` | its own minimum | `1.1.0` |

Structurally, runtime pass-by-value is **closest to the proposed ragged
design**: both are declarative per-tensor schema additions feeding a
`computeMinimumPluginApiVersion` mapping, with no new plugin entry point
and no new variant-pack attribute. It is lighter even than ragged on one
axis — pass-by-value derives its feature predicate from the per-tensor
flag ([§4.3](#43-feature-signal-derived)) instead of adding a graph-level
enable flag. The machinery it actually reuses in code is **override
shapes'** (the only shipped template) — `computeMinimumPluginApiVersion`
and the version-filter path — which it adopts while declining override's
new plugin symbol and variant-pack transport.

A graph that enables both features is filtered by the union of their
requirements: the `1.2.0` floor from pass-by-value plus override's
`hasOverrideExecute()` per-symbol gate. Unlike ragged and override, a
state-#2 (`RUNTIME_PARAM`) pass-by-value tensor also adds **frontend
variant-pack injection** ([§4.8](#48-frontend-variant-pack-injection)) —
a build-time value snapshot the graph injects at execute — a mechanism
neither ragged nor override has.

---

## 8. Risks

### 8.1 Provider reports 1.2.0 but mishandles host pointers

**Risk**: a provider bumps its reported version to `1.2.0` but reads a
runtime pass-by-value slot as a device pointer (or otherwise mishandles
the host scalar), producing wrong results.

**Mitigation**: the version contract is the sole *capability* signal —
by design there is no per-symbol safety net
([§5.4](#54-version-only-filtering)). The applicability filter rejects a
plugin whose *reported version* is too low, but no core check can catch
a plugin that truthfully reports `1.2.0` yet mishandles the host
pointer; that residual risk is covered by the integration suite's fake
`1.2.0` plugin, which asserts the value it receives equals what the
caller supplied ([§10](#10-testing-plan)).

### 8.2 Caller marks a tensor pass-by-value but omits the value at execute

**Risk**: a tensor is marked `is_pass_by_value` but the caller does not
insert its UID into the variant-pack map, so the provider reads an
unset or garbage slot.

**Mitigation**: documented contract. The frontend cannot validate the
numeric value at build time (it does not yet exist); it validates only
that the tensor is structurally a scalar. This matches the existing
behavior for required scalar parameters
(`validateScalarParameter`).

### 8.3 Host vs device pointer confusion in the shared map

**Risk**: the same variant-pack map carries both device pointers and
host scalar pointers, so a provider could dereference the wrong kind.

**Mitigation**: on the fresh-build path the two-flag discriminator
`isPassByValue() && !isCompileTimeConstant()` (states #1 and #2) is the
authoritative marker of a host-pointer slot; on a deserialized plan the
provider relies on the runtime pass-by-value UID set persisted in its
payload
([§5.6](#56-deserialized-plan-support-via-the-provider-payload)). The
three-state model ([§5.5](#55-compile-time-vs-runtime-mode)) keeps the
discriminator unambiguous. Round-trip and end-to-end tests cover both
slot kinds in one graph.

### 8.4 Downgraded provider mis-reads a serialized pass-by-value plan

**Risk**: a compiled pass-by-value plan
([RFC 0009](0009_CompiledPlanSerialization.md)) is serialized against a
`1.2.0` provider, then deserialized where the same `engineId` resolves
to a downgraded `< 1.2.0` build of that provider. Deserialize re-binds
by `engineId` with no core version check
([§4.6](#46-feature-detection-and-version-filtering)), so the older build
receives a payload it did not produce and could read a host scalar as a
device pointer — a memory-unsafe failure.

**Mitigation**: the §4.5 provider contract requires a `1.2.0` provider
to version its `plugin_payload` and reject a version/kind it cannot
interpret before reading any slot, so a provider that practiced
defensive payload versioning fails the deserialize cleanly. hipDNN core
adds no gate here by design
([§5.6](#56-deserialized-plan-support-via-the-provider-payload)); the
residual case — a provider release that predates pass-by-value and never
rejected unknown payloads — cannot be closed retroactively (§5.6
retrofit limit) and is covered by the integration suite's payload
round-trip and rejection tests ([§10](#10-testing-plan)).

### 8.5 Provider reads the baked value of a RUNTIME_PARAM (#2) tensor

**Risk**: a state-#2 (`RUNTIME_PARAM`) tensor carries a baked `value` in
the flatbuffer, so the read-only `HIPDNN_ATTR_TENSOR_IS_BY_VALUE` (1307)
is true for it — yet its value must come from the frontend-injected
variant-pack slot, not the baked value. A provider that shortcuts on 1307
(or on value presence) would read the stale baked value instead of the
injected one.

**Mitigation**: the authoritative discriminator is
`is_compile_time_constant`, not 1307; a #2 tensor has
`is_compile_time_constant == false`, so the two-flag discriminator
`isPassByValue() && !isCompileTimeConstant()` correctly routes it to the
variant-pack slot ([§4.5](#45-provider-contract)). The #2 end-to-end test
asserts the injected value — not the baked one — reaches the provider.
Note a #2 tensor's value is **not** carried across compiled-plan
([RFC 0009](0009_CompiledPlanSerialization.md)) serialization; on that
path it degrades to caller-supplied (#1 semantics), covered by the
compiled-plan round-trip test
([§4.8](#48-frontend-variant-pack-injection)).

---

## 9. Execution Plan

Implementation plan for the work this RFC enables, ordered so the tree
builds and existing tests pass after each step:

### Step 1: Schema fields

Append two defaulted booleans, `is_pass_by_value: bool = false;` and
`is_compile_time_constant: bool = false;`, to the `TensorAttributes`
flatbuffer table (re-using the existing `value` union); regenerate. Add
round-trip coverage of both flags plus the value, including the
default-`false` case.

### Step 2: Backend descriptor and enums

Add the settable tensor attributes
`HIPDNN_ATTR_TENSOR_IS_PASS_BY_VALUE_EXT = 1308` and
`HIPDNN_ATTR_TENSOR_IS_COMPILE_TIME_CONSTANT_EXT = 1309`; wire both
through the existing `TensorDescriptor` get/set-attribute and pack/unpack
paths. The read-only `HIPDNN_ATTR_TENSOR_IS_BY_VALUE` (1307) is retained
unchanged (value-present) but is no longer the baked discriminator. Add
`isPassByValue()` and `isCompileTimeConstant()` accessors to
`ITensorAttributesWrapper`. No operation-graph attribute is added.

### Step 3: Version constant and filter

Add `K_PASS_BY_VALUE_MIN_API_VERSION = "1.2.0"` to
[`PluginVersionConstants.hpp`](../../plugin_sdk/include/hipdnn_plugin_sdk/PluginVersionConstants.hpp); bump [`engine_api_version.h`](../../plugin_sdk/include/hipdnn_plugin_sdk/engine_api_version.h) minor to `2`.
Add a `readIsPassByValueEnabled(graphDesc)` helper that scans the
serialized op graph and returns `true` iff any tensor has
`is_pass_by_value == true && is_compile_time_constant == false` (states
#1 or #2; no graph-level attribute is read; the per-tensor flags are the
source of truth). Extend
`computeMinimumPluginApiVersion(bool isOverride, bool isPassByValue)` to
take the second flag and return the maximum required version (the
applicability-time filter).

### Step 4: Frontend API and validation

Add `enum class ScalarType { RUNTIME_PARAM, COMPILE_TIME_CONST };`; the
`TensorAttributes(const T&, ScalarType)` constructor; `set_compile_time_constant`
and `set_as_runtime_parameter`; and `get_compile_time_constant` /
`get_is_pass_by_value`. Route the plain scalar constructor and
`set_value` through `set_compile_time_constant` (state #3). Change
`get_pass_by_value()` to return the value variant (mode-gated to #2),
moving its former bool-predicate role to `get_is_pass_by_value()`, and
migrate every internal value-presence caller off `get_pass_by_value()`
onto `get_value_variant()` — the known callsite is `createOrFindTensorDesc`
in [`DescriptorHelpers.hpp`](../../frontend/include/hipdnn_frontend/detail/DescriptorHelpers.hpp)
(writes the baked value); `search` for `get_pass_by_value()` to enumerate
the rest. Add the four invalid-combination validations and the relaxed
`validateScalarParameter` ([§4.7](#47-frontend-validation)). No
graph-level setter or graph schema field is added — the
runtime-pass-by-value feature signal is derived from the per-tensor flags
([§4.3](#43-feature-signal-derived)).

### Step 5: Frontend variant-pack injection (state #2)

Add the state-#2 machinery ([§4.8](#48-frontend-variant-pack-injection)):
the `Graph` snapshots each `RUNTIME_PARAM` tensor's value into host-side
storage at build (stable across execute), and `Graph::execute()` injects
`uid → &cachedValue` (host pointer) into the variant pack before dispatch
on the fresh-build and graph-(de)serialize paths. On the RFC 0009
compiled-plan path no per-tensor value is reconstructed, so a #2 tensor
degrades to caller-supplied (#1 semantics); document the limitation.

### Step 6: Cross-cutting tests

Fake plugins (one at `1.2.0` consuming the host scalar, one below it)
and the three-state / version-filter / end-to-end matrix
([§10](#10-testing-plan)).

### Step 7: Provider adoption

A shipping provider that adopts runtime pass-by-value **MUST**: read
host-scalar slots for any tensor matching the runtime discriminator
`isPassByValue() && !isCompileTimeConstant()` (states #1 and #2,
identically); persist its runtime pass-by-value UID set into its
serialized `plugin_payload` and restore it on deserialize
([§5.6](#56-deserialized-plan-support-via-the-provider-payload)); version
that payload and reject a payload whose version/kind it cannot interpret
before reading any slot ([§4.5](#45-provider-contract)); and bump its
reported version to `1.2.0`. The reject-on-unknown-payload requirement
is what keeps a downgraded re-bind from dereferencing a host pointer as
device memory, and it only protects releases that practiced this
versioning from the start (§5.6 retrofit limit). Provider work is
independent of Steps 1–6 and lands on its own schedule.

---

## 10. Testing Plan

Test conventions follow [RFC 0006](0006_PluginAgnosticIntegrationTests.md). The plan exercises:

- **Three-state matrix.** Each state is exercised independently:
  - **State #3 (compile-time, default):** a plain-constructed or
    `set_value`/`set_compile_time_constant` scalar is served by baseline
    plugins with **no** version elevation; `get_compile_time_constant()`
    returns the value and `get_pass_by_value()` returns empty.
  - **State #2 (runtime, frontend-injected):** a
    `TensorAttributes(v, ScalarType::RUNTIME_PARAM)` scalar elevates the
    required version to `1.2.0`; at execute the frontend-injected value
    reaches the provider's `device_buffers` slot and **equals** the
    supplied value (end-to-end); it survives graph
    serialize → deserialize → execute; and after a compiled-plan
    (`to_compiled_plan_binary`) round-trip it degrades to caller-supplied
    (#1 semantics), so the caller must supply the value at execute
    ([§4.8](#48-frontend-variant-pack-injection)).
  - **State #1 (runtime, user-supplied):** a `set_as_runtime_parameter()`
    scalar elevates to `1.2.0`; the user-supplied value reaches the
    provider and equals what the caller supplied.

- **Getter-return assertions.** Per state, assert the getter semantics of
  the state table ([§5.5](#55-compile-time-vs-runtime-mode)):
  `get_is_pass_by_value()` is `true` for all three; `get_pass_by_value()`
  returns the value only for #2 and is empty for #1 and #3;
  `get_compile_time_constant()` returns the value only for #3 and is empty
  for #1 and #2. Explicitly assert the plain constructor lands in #3
  (`get_pass_by_value()` empty, `get_compile_time_constant()` == value) —
  hipDNN's deliberate divergence from cuDNN's plain ctor.

- **Breaking-change assertions.** `get_pass_by_value()` returns the
  *value* variant (not a `bool`) and yields it for a #2 tensor;
  `get_is_pass_by_value()` carries the umbrella-predicate role.

- **Schema round-trip.** Both `TensorAttributes.is_pass_by_value` and
  `is_compile_time_constant` survive descriptor → serialize → deserialize
  → read-back alongside the `value`, including the default-`false` case
  for tensors that never opt in, so `get_pass_by_value` /
  `get_compile_time_constant` / `get_is_pass_by_value` are correct after
  graph deserialize.

- **Version filtering (unit).** A graph with any tensor
  `is_pass_by_value && !is_compile_time_constant` (#1 or #2) elevates the
  required version to `1.2.0`; plugins reporting `< 1.2.0` (including the
  `1.0.0` no-symbol fallback) are dropped from the applicable set; a graph
  with no qualifying plugin returns "no applicable engines." A #3-only
  graph imposes no floor. Serialized/deserialized path: a `1.2.0` fake
  plugin that persists its runtime pass-by-value UID set into its
  `plugin_payload` serializes a pass-by-value plan; after
  deserialize-and-execute (no op graph available) the host scalar is
  still read correctly, proving host-scalar identity survives via the
  payload. A plugin that cannot interpret a newer payload version rejects
  it at deserialize.

- **Invalid-combination rejection.** All four invalid combinations
  ([§4.7](#47-frontend-validation)) return `INVALID_VALUE`:
  `is_compile_time_constant && no value`; `value present &&
  !is_pass_by_value`; `is_compile_time_constant && !is_pass_by_value`;
  `virtual && is_pass_by_value`. States #1/#2/#3 validate cleanly, and a
  required scalar in any of the three passes `validateScalarParameter`.

- **Serialization parity.** A graph serialized without the feature loads
  in a feature-aware runtime with both new flags `false` and is served by
  a baseline plugin unchanged.

---

## 11. Glossary

- **Pass-by-value tensor**: a host-side scalar operand (e.g. epsilon,
  alpha, beta, SDPA scale) carried as a single-element tensor with the
  umbrella flag `is_pass_by_value == true`. The umbrella term covering
  all three states below.
- **`ScalarType`**: the frontend enum
  `enum class ScalarType { RUNTIME_PARAM, COMPILE_TIME_CONST };`
  selecting a value-carrying tensor's state at construction, mirroring
  cuDNN-frontend
  ([`graph_properties.h:42-46`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_properties.h#L42-L46)).
- **`RUNTIME_PARAM`**: the `ScalarType` alternative producing state #2 —
  a value-carrying runtime scalar delivered by frontend variant-pack
  injection ([§4.8](#48-frontend-variant-pack-injection)).
- **`COMPILE_TIME_CONST`**: the `ScalarType` alternative producing state
  #3 — a value baked into the op-graph flatbuffer.
- **State #1 — runtime, user-supplied** (`is_pass_by_value == true`,
  `is_compile_time_constant == false`, no value): created by
  `set_as_runtime_parameter()` (or `set_is_pass_by_value(true)` with no
  value); the user supplies the host pointer in the variant pack.
  Requires plugin floor `1.2.0`.
- **State #2 — runtime, frontend-injected** (`is_pass_by_value == true`,
  `is_compile_time_constant == false`, value present): created by
  `TensorAttributes(v, ScalarType::RUNTIME_PARAM)`; the frontend injects
  a host pointer to its cached value into the variant pack
  ([§4.8](#48-frontend-variant-pack-injection)). Requires plugin floor
  `1.2.0`.
- **State #3 — compile-time constant (default)** (`is_pass_by_value ==
  true`, `is_compile_time_constant == true`, value present): created by
  the plain scalar constructor `TensorAttributes(v)`, `set_value(v)`,
  `TensorAttributes(v, ScalarType::COMPILE_TIME_CONST)`, or
  `set_compile_time_constant(v)`; the value is frozen into the op-graph
  flatbuffer via `HIPDNN_ATTR_TENSOR_VALUE_EXT` and read from it, exactly
  as before this RFC. Imposes no version floor (baseline `1.0.0`). The
  only mode hipDNN supported before this RFC.
- **`is_compile_time_constant`**: the per-tensor flatbuffer boolean
  distinguishing state #3 (`true`) from states #1/#2 (`false`); it is
  the authoritative baked/compile-time discriminator, replacing the old
  `HIPDNN_ATTR_TENSOR_IS_BY_VALUE` (1307) role (1307 is now true for both
  #2 and #3 and is no longer the discriminator).
- **Frontend variant-pack injection**: the state-#2 delivery mechanism
  by which `Graph::execute()` inserts `uid → &cachedValue` (a host
  pointer to the frontend's build-time snapshot of the value) into the
  variant pack before dispatch, mirroring cuDNN's
  `extend_tensor_map_with_pass_by_value_tensors_`
  ([`graph_interface.h:198-217`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_interface.h#L198-L217))
  and `cached_pass_by_value`
  ([`graph_interface.h:978-982`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_interface.h#L978-L982)).
  See [§4.8](#48-frontend-variant-pack-injection).
- **Variant pack**: the runtime-only carrier of per-execution payload
  (data pointers, unique IDs, workspace). New in this RFC: a runtime
  pass-by-value tensor's `uid → void*` entry is a *host* pointer to the
  scalar rather than a device pointer (supplied by the user for #1, by
  the frontend for #2). The variant pack has no flatbuffer schema and is
  never serialized.
- **Feature predicate (derived)**: the backend's runtime-pass-by-value
  feature signal, derived by scanning the op graph for any tensor with
  `is_pass_by_value == true && is_compile_time_constant == false` (states
  #1 or #2). There is no separate graph-level flag; the per-tensor schema
  fields are the single source of truth. State #3 does not raise the
  signal.
- **Supported plugin SDK API version**: a per-plugin declaration of the
  Plugin SDK API version the plugin was built against, reported via
  `hipdnnPluginGetApiVersion(const char**)` as a `"MAJOR.MINOR.PATCH"`
  string and parsed with `hipdnn_data_sdk::utilities::Version`. Plugins
  that do not export the symbol fall back to `"1.0.0"`.
- **Required plugin SDK API version**: the per-graph minimum the backend
  computes from the features a graph uses; `1.2.0` for runtime
  pass-by-value (#1/#2). A plugin stays in a graph's applicable set only
  when its supported version is `>=` the graph's required version.
- **Version-only filtering**: the applicability model used by
  this RFC, in which provider eligibility is decided by reported API
  version alone, with no per-symbol predicate, because the feature adds
  no new plugin entry point.
