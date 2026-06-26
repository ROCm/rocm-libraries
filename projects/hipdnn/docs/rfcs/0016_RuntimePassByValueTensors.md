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
5. [Key Design Decisions](#5-key-design-decisions)
   - 5.1 [Boolean flag mirroring cuDNN](#51-boolean-flag-mirroring-cudnn)
   - 5.2 [Reuse the variant-pack pointer map](#52-reuse-the-variant-pack-pointer-map)
   - 5.3 [Derive feature detection from the tensor schema](#53-derive-feature-detection-from-the-tensor-schema)
   - 5.4 [Version-only filtering](#54-version-only-filtering)
   - 5.5 [Compile-time vs runtime mode](#55-compile-time-vs-runtime-mode)
   - 5.6 [Deserialized-plan support via the provider payload](#56-deserialized-plan-support-via-the-provider-payload)
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

- One defaulted boolean appended to the per-tensor flatbuffer schema
  (`is_pass_by_value`); the runtime-pass-by-value feature signal is
  derived from it, with no graph-level flag
  ([§4.3](#43-feature-signal-derived)).
- One minor plugin-SDK API version bump (`1.1.0` → `1.2.0`).
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

The desired end-user surface keeps the same execute API and matches
cuDNN-frontend. A scalar operand is declared pass-by-value on the
tensor:

```cpp
auto scale = graph.tensor(...);
scale->set_is_pass_by_value(true);   // value supplied at execute, not baked
```

At execute time the host-side value is supplied through the existing
variant-pack map, keyed by the tensor UID, exactly as device buffers
are. cuDNN-frontend realizes this with
`extend_tensor_map_with_pass_by_value_tensors_`
([`graph_interface.h:190-212`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_interface.h#L190-L212)), which emplaces a *host* pointer to the
scalar into the same `std::unordered_map<int64_t, void*>` used for
device buffers. hipDNN adopts the same transport.

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
3. **Keep the graph descriptor read-only after build.** The
   pass-by-value *value* is not stored on the graph for runtime
   tensors; it travels via the variant pack only.
4. **A pass-by-value tensor has exactly one mode.** Value presence
   determines it — a baked value ⇒ compile-time, no value ⇒ runtime — so
   no tensor is simultaneously baked and runtime-overridable. cuDNN
   enforces the same exclusivity
   ([`graph_properties.h:62-105`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_properties.h#L62-L105)),
   though it keys on an explicit flag rather than value presence.

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

A per-tensor boolean marks a scalar tensor as **pass-by-value**. It is
the **umbrella** flag — true whenever the tensor is a pass-by-value
scalar — mirroring cuDNN-frontend's `is_pass_by_value`
([`graph_properties.h:368`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_properties.h#L368)):

```cpp
class Tensor_attributes {
public:
    auto set_is_pass_by_value(bool value) -> Tensor_attributes&;
    bool get_is_pass_by_value() const;
};
```

The two modes are distinguished by whether a value is also baked:

- **Compile-time constant** (existing behavior): a value is baked with
  `set_value(scalar)`, which implies `is_pass_by_value == true`. The
  value is frozen into the graph and served by any plugin.
- **Runtime** (new): `set_is_pass_by_value(true)` with **no** baked value
  declares a scalar whose value is supplied through the variant pack at
  execute time.

So the mode follows value presence: `is_pass_by_value && value == NONE`
⇒ runtime; `is_pass_by_value && value != NONE` ⇒ compile-time. The
ticket's `set_pass_by_value_tensor` refers to this same setter.

The existing `get_pass_by_value()` predicate (a `bool` meaning "a
compile-time value is present") is retained unchanged; under the umbrella
model it is simply the compile-time-mode indicator — a *subset* of
`get_is_pass_by_value()`, not an opposing signal. We deliberately do
**not** repurpose `get_pass_by_value()` to return the value as
cuDNN-frontend does, to avoid breaking the existing accessor.

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

The per-tensor pass-by-value flag is persisted so a provider reading the
serialized op graph can identify pass-by-value scalars and, with value
presence, tell the two modes apart. The
flatbuffer `TensorAttributes` table gains one defaulted boolean
appended as its last field:

```
is_pass_by_value: bool = false;
```

This is the append-only, defaulted-field pattern used in `develop` by the
override-shape graph and plan flags ([RFC 0008](0008_OverridableTensorShapesDesign.md)), wire-compatible per [RFC 0005](0005_Versioning.md):
a pre-feature graph deserialized in a runtime that understands the
field reads `is_pass_by_value == false` on every tensor.

The backend `TensorDescriptor` gains a **settable** attribute
`HIPDNN_ATTR_TENSOR_IS_PASS_BY_VALUE_EXT` (ID 1308) carrying the umbrella
flag. It is related to — not a duplicate of — the existing **read-only**
`HIPDNN_ATTR_TENSOR_IS_BY_VALUE` (1307), which remains derived from
`value.type != NONE` and therefore reports specifically the
*compile-time* mode (a baked value present). The settable flag
round-trips through descriptor pack/unpack alongside the existing
value-carrying attributes; the precedent is the unpack of
`HIPDNN_ATTR_TENSOR_IS_BY_VALUE` + `HIPDNN_ATTR_TENSOR_VALUE_EXT` in
[`frontend/include/hipdnn_frontend/detail/DescriptorUnpackHelpers.hpp`](../../frontend/include/hipdnn_frontend/detail/DescriptorUnpackHelpers.hpp).

Providers read the flag through an `isPassByValue()` accessor added to
the op-graph tensor wrapper
([`ITensorAttributesWrapper`](../../flatbuffers_sdk/include/hipdnn_flatbuffers_sdk/flatbuffer_utilities/TensorAttributesWrapper.hpp)),
mirroring the existing `isVirtual()`. This is the accessor the provider
example in [§4.5](#45-provider-contract) consumes.

### 4.3 Feature signal (derived)

The backend's runtime-pass-by-value feature signal is **derived from the
per-tensor flag** ([§4.2](#42-tensor-schema-addition)): a graph requires
runtime pass-by-value support iff it contains at least one tensor with
`is_pass_by_value == true` and no baked `value`. There is **no**
graph-level schema field and **no** graph-level setter — the per-tensor
`is_pass_by_value` flag is the single source of truth, and a separate
graph boolean would only create a cache that can desync from it
([§5.3](#53-derive-feature-detection-from-the-tensor-schema)).

The backend computes the signal with a `readIsPassByValueEnabled` helper
that scans the serialized op graph at applicability time; a graph with
only compile-time pass-by-value scalars (every such tensor carries a
baked value) yields `false` and needs no version elevation. This is the
single signal feature detection consumes
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
`extend_tensor_map_with_pass_by_value_tensors_` behavior. The frontend
variant-pack builder `detail::populateBaseVariantPackDescriptor`
([`frontend/include/hipdnn_frontend/detail/VariantPackHelpers.hpp:19`](../../frontend/include/hipdnn_frontend/detail/VariantPackHelpers.hpp#L19))
is unchanged: the host pointer is an ordinary
`HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS` entry.

### 4.5 Provider contract

A provider reporting plugin SDK API version `>= 1.2.0` must, for any
tensor that is pass-by-value with **no** baked value
(`is_pass_by_value == true && value == NONE` — the runtime mode), read
the scalar from that UID's slot in the `device_buffers` array **as a host
pointer** at execute time. A pass-by-value tensor *with* a baked value is
a compile-time constant and is read from the op-graph flatbuffer as today.

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
`isPassByValue() && valueType() == NONE`: the umbrella flag alone is not
sufficient, because a compile-time constant is *also* pass-by-value (with
a value present). `valueType()` returns
`hipdnn_flatbuffers_sdk::data_objects::TensorValue`, and `NONE` is that
union's absent state (no baked value), not a separately-defined enum. No
marker travels on the buffer itself, and the existing read-only
`HIPDNN_ATTR_TENSOR_IS_BY_VALUE` (1307) reports the opposite mode (a
baked value present), so it cannot stand in for this check.

Each `hipdnnPluginDeviceBuffer_t` carries its `uid`, and
[§4.2](#42-tensor-schema-addition) adds the `isPassByValue()` accessor,
so a provider builds the set of runtime pass-by-value UIDs once from the
op graph and interprets the matching `device_buffers` slots as host
pointers at execute time:

```cpp
// Setup (once per graph): which UIDs are RUNTIME pass-by-value?
std::unordered_set<int64_t> hostScalarUids;
for (auto const& tensor : opGraph.tensors()) {        // ITensorAttributesWrapper
    if (tensor.isPassByValue()                         // umbrella flag
        && tensor.valueType() == TensorValue::NONE) {  // runtime mode: value union absent (no baked value)
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
| Runtime pass-by-value (with or without override) | `1.2.0` |

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
checks:

1. **Value implies the umbrella.** A baked `value` implies
   `is_pass_by_value == true` (a compile-time constant is a pass-by-value
   scalar); `set_value` sets the umbrella flag. A tensor that carries a
   value but has `is_pass_by_value` explicitly cleared returns
   `ErrorCode::INVALID_VALUE`. There is **no** error for a tensor that is
   both pass-by-value and has a value — that is precisely the
   compile-time mode.
2. **Virtual exclusion.** The existing virtual + pass-by-value
   rejection is extended to the runtime flag: a virtual tensor must not
   be marked `is_pass_by_value`.

`detail::validateScalarParameter`
([`frontend/include/hipdnn_frontend/node/detail/Utilities.hpp`](../../frontend/include/hipdnn_frontend/node/detail/Utilities.hpp), ~line
475) currently requires `get_pass_by_value()` to be true for required
scalar inputs (epsilon, SDPA scale, etc.), i.e. a baked value present.
It is relaxed to accept any pass-by-value scalar
(`get_is_pass_by_value() == true`), whether the value is baked
(compile-time) or absent (runtime). As today, the
actual numeric value cannot be checked at build time for runtime
scalars because it is not yet available.

---

## 5. Key Design Decisions

### 5.1 Boolean flag mirroring cuDNN

**Decision**: model pass-by-value with an explicit umbrella flag
`set_is_pass_by_value(bool)` matching cuDNN-frontend, and distinguish the
two modes internally by whether a value is baked.

**Rationale**: the umbrella flag mirrors cuDNN-frontend's `is_pass_by_value`,
so the user-facing concept is familiar to cuDNN users. Internally we
**simplify** cuDNN's scheme rather than copy it: cuDNN keys the mode on an
explicit `has_compile_time_constant` flag with two separate value slots
because its runtime-fusion JIT folds compile-time constants; hipDNN
dispatches to precompiled providers and has no consumer for that
distinction, so the mode is read from value-presence on the existing
`value` union — needing no second value slot and no separate
compile-time-constant flag.

**Trade-off**: `get_is_pass_by_value()` (umbrella) and the existing
`get_pass_by_value()` (baked value present) coexist as a superset/subset
pair; we deliberately keep `get_pass_by_value()`'s `bool` return rather
than cuDNN's value-returning form, to avoid breaking the existing accessor.

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

**Trade-off**: the provider must consult the per-tensor pass-by-value
flag together with value presence to know that a given `device_buffers`
slot holds a *host* pointer rather than a device pointer. That
conjunction is the authoritative discriminator
([§5.5](#55-compile-time-vs-runtime-mode)).

### 5.3 Derive feature detection from the tensor schema

**Decision**: detect the runtime-pass-by-value feature by deriving it
from the per-tensor `is_pass_by_value` flag (any tensor with
`is_pass_by_value == true` and `value == NONE`) rather than adding a
separate graph-level `is_pass_by_value_enabled` flag. The per-tensor
schema field is the single source of truth.

**Rationale**: the per-tensor flag is already **mandatory** — it is the
only signal distinguishing a runtime host-scalar slot from an ordinary
device buffer ([§4.5](#45-provider-contract)), since `value == NONE`
alone matches every device tensor. A graph-level flag would therefore be
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

**Decision**: a pass-by-value tensor is in one of two modes, selected by
whether a value is baked: value present ⇒ compile-time constant; value
absent ⇒ runtime. Both states are valid; there is no mutual-exclusivity
error.

**Rationale**: it yields a clean, single-axis distinction — simpler than
cuDNN's separate-flag form (cuDNN keys the mode on an explicit
`has_compile_time_constant`; we key it on value presence). The umbrella
flag `is_pass_by_value` says "this
is a pass-by-value scalar"; the value union says "baked or not." The
union's `NONE` state is distinct from a baked zero (e.g.
`Float32Value{0}`), so the mode test is unambiguous for every scalar
type.

**Trade-off**: identifying a *runtime* slot is a two-term check
(`is_pass_by_value && value == NONE`) rather than a single dedicated
flag; the provider contract ([§4.5](#45-provider-contract)) spells this
out so implementations do not diverge.

The full cuDNN-frontend per-mode methods (`set_compile_time_constant`,
`set_as_runtime_parameter`, `get_has_compile_time_constant`,
`get_compile_time_constant`) are **out of scope** for this PR; they can
later be added as thin source-level drop-in wrappers over these two modes,
with no schema or backend change.

### 5.6 Deserialized-plan support via the provider payload

**Decision**: add no execution-plan schema field and no hipDNN
dispatch-time version gate. A provider that supports runtime
pass-by-value (a) persists the set of runtime pass-by-value UIDs into
its opaque `plugin_payload` when a plan is serialized and reconstructs
it on deserialize, and (b) versions that payload so a plugin that cannot
interpret it fails the deserialize rather than mis-reading it.

**Rationale**: on the fresh-build path the provider derives the
runtime-pass-by-value UID set from the op graph (`isPassByValue() &&
valueType() == NONE`, [§4.5](#45-provider-contract)). On the
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
[Step 6](#step-6-provider-adoption)).

---

## 6. Compatibility, Versioning, and Rollback

**Upgrade path.** Existing plugins (in-tree and out-of-tree) continue
to serve non-pass-by-value graphs unchanged; they require no rebuild. A
plugin adopts runtime pass-by-value by reading host-scalar slots for
`is_pass_by_value` tensors and bumping its reported API version to
`1.2.0`. Each plugin migrates on its own schedule.

**Version skew.** An older plugin paired with a pass-by-value graph is
filtered out by the per-graph version gate
([§4.6](#46-feature-detection-and-version-filtering)). If no plugin
reports `>= 1.2.0`, applicability returns a clean "no applicable
engines" result. A legacy plugin never receives a pass-by-value graph
and therefore never reads a host pointer where it expects a baked value
or a device buffer, so on the applicability-filtered fresh-build path
there is no silent wrong-result path.

The serialized/deserialized path is not core-gated. A plan re-bound by
`engineId` to a downgraded plugin relies on the provider versioning its
opaque `plugin_payload`: a provider that versions it correctly rejects
the mismatched payload at deserialize rather than mis-reading a host
pointer, while a provider that does not is the residual risk
[§5.6](#56-deserialized-plan-support-via-the-provider-payload) discloses,
since hipDNN core no longer enforces a version floor on that path.

**Non-breaking schema compatibility.** The new schema field
(`TensorAttributes.is_pass_by_value`) is appended, defaulted `false`, and
wire-compatible per [RFC 0005](0005_Versioning.md). A graph serialized
before this feature, deserialized in a runtime that understands the
field, reads `false` on every tensor — i.e. it is treated as a
non-pass-by-value graph and served by any plugin, exactly as before.

**Rollback.** The feature is inert unless a caller marks a tensor
runtime pass-by-value. Reverting a caller to compile-time
`set_value(scalar)` (or never setting the flag) restores the existing
baked-value path with zero schema migration: the defaulted fields stay
`false`, and `computeMinimumPluginApiVersion` returns the baseline. No
data migration or plan invalidation is required.

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
| Tensor schema change | append `is_pass_by_value: bool` | append `ragged_offset_tensor_uid`, `alignment` | none (graph-level only) |
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
`hasOverrideExecute()` per-symbol gate.

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

**Mitigation**: on the fresh-build path the per-tensor pass-by-value
flag plus value presence is the authoritative discriminator; on a
deserialized plan the provider relies on the runtime pass-by-value UID
set persisted in its payload
([§5.6](#56-deserialized-plan-support-via-the-provider-payload)). The two-mode rule
([§5.5](#55-compile-time-vs-runtime-mode)) keeps the discriminator
unambiguous. Round-trip and end-to-end tests cover both
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

---

## 9. Execution Plan

Implementation plan for the work this RFC enables, ordered so the tree
builds and existing tests pass after each step:

### Step 1: Schema fields

Append `is_pass_by_value: bool = false;` to the `TensorAttributes`
flatbuffer table; regenerate. Add round-trip coverage, including the
default-`false` case.

### Step 2: Backend descriptor and enums

Add the settable tensor attribute
`HIPDNN_ATTR_TENSOR_IS_PASS_BY_VALUE_EXT = 1308`; wire it through the
existing `TensorDescriptor` get/set-attribute and pack/unpack paths. No
operation-graph attribute is added.

### Step 3: Version constant and filter

Add `K_PASS_BY_VALUE_MIN_API_VERSION = "1.2.0"` to
[`PluginVersionConstants.hpp`](../../plugin_sdk/include/hipdnn_plugin_sdk/PluginVersionConstants.hpp); bump [`engine_api_version.h`](../../plugin_sdk/include/hipdnn_plugin_sdk/engine_api_version.h) minor to `2`.
Add a `readIsPassByValueEnabled(graphDesc)` helper that scans the
serialized op graph and returns `true` iff any tensor has
`is_pass_by_value == true` and `value == NONE` (no graph-level attribute
is read; the per-tensor flag is the source of truth). Extend
`computeMinimumPluginApiVersion(bool isOverride, bool isPassByValue)` to
take the second flag and return the maximum required version (the
applicability-time filter).

### Step 4: Frontend API and validation

Add `Tensor_attributes::set_is_pass_by_value` /
`get_is_pass_by_value`; make `set_value` imply the umbrella flag; add the
value-implies-umbrella and virtual-exclusion validation; relax
`validateScalarParameter` to accept any pass-by-value scalar. No
graph-level setter or graph schema field is added — the
runtime-pass-by-value feature signal is derived from the per-tensor flags
([§4.3](#43-feature-signal-derived)).

### Step 5: Cross-cutting tests

Fake plugins (one at `1.2.0` consuming the host scalar, one below it)
and the version-filter / end-to-end matrix
([§10](#10-testing-plan)).

### Step 6: Provider adoption

A shipping provider that adopts runtime pass-by-value **MUST**: read
host-scalar slots for runtime pass-by-value tensors; persist its runtime
pass-by-value UID set into its serialized `plugin_payload` and restore
it on deserialize
([§5.6](#56-deserialized-plan-support-via-the-provider-payload)); version
that payload and reject a payload whose version/kind it cannot interpret
before reading any slot ([§4.5](#45-provider-contract)); and bump its
reported version to `1.2.0`. The reject-on-unknown-payload requirement
is what keeps a downgraded re-bind from dereferencing a host pointer as
device memory, and it only protects releases that practiced this
versioning from the start (§5.6 retrofit limit). Provider work is
independent of Steps 1–5 and lands on its own schedule.

---

## 10. Testing Plan

Test conventions follow [RFC 0006](0006_PluginAgnosticIntegrationTests.md). The plan exercises:

- **Schema round-trip.** `TensorAttributes.is_pass_by_value` survives
  descriptor → serialize → deserialize → read-back, including the
  default-`false` case for tensors that never opt in.

- **Version filtering (unit).** A pass-by-value-enabled graph elevates
  the required version to `1.2.0`; plugins reporting `< 1.2.0`
  (including the `1.0.0` no-symbol fallback) are dropped from the
  applicable set; a graph with no qualifying plugin returns "no
  applicable engines." Serialized/deserialized path: a `1.2.0` fake
  plugin that persists its runtime pass-by-value UID set into its
  `plugin_payload` serializes a pass-by-value plan; after
  deserialize-and-execute (no op graph available) the host scalar is
  still read correctly, proving host-scalar identity survives via the
  payload. A plugin that cannot interpret a newer payload version rejects
  it at deserialize.

- **New-behavior end-to-end.** A fake plugin reporting `1.2.0` reads
  the host scalar from its `device_buffers` slot and records it. The
  test supplies a known value through the variant-pack map at execute
  and asserts the value the plugin received **equals** the value the
  caller supplied — exercising the runtime supply path, not just
  build/round-trip.

- **Mode classification / rejection.** A pass-by-value tensor with a
  baked value is treated as compile-time (served by baseline plugins, no
  version elevation); the same tensor with no value is treated as runtime
  (elevates to `1.2.0`). A tensor with a value but `is_pass_by_value`
  explicitly cleared returns `INVALID_VALUE`; a virtual tensor marked
  `is_pass_by_value` is rejected; a required scalar in either mode passes
  `validateScalarParameter`.

- **Serialization parity.** A graph serialized without the feature
  loads in a feature-aware runtime with both new flags `false` and is
  served by a baseline plugin unchanged.

---

## 11. Glossary

- **Pass-by-value tensor**: a host-side scalar operand (e.g. epsilon,
  alpha, beta, SDPA scale) carried as a single-element tensor and marked
  `is_pass_by_value == true`. The umbrella term covering both modes below.
- **Compile-time constant scalar** (compile-time mode): a pass-by-value
  tensor with a baked value (`is_pass_by_value == true`, `value != NONE`),
  frozen into the compiled plan via `HIPDNN_ATTR_TENSOR_VALUE_EXT`. The
  only mode hipDNN supports before this RFC.
- **Runtime pass-by-value tensor** (runtime mode): a pass-by-value tensor
  with no baked value (`is_pass_by_value == true`, `value == NONE`), whose
  scalar is supplied through the variant pack at execute time. New in this
  RFC.
- **Variant pack**: the runtime-only carrier of per-execution payload
  (data pointers, unique IDs, workspace). New in this RFC: a runtime
  pass-by-value tensor's `uid → void*` entry is a *host* pointer to the
  scalar rather than a device pointer. The variant pack has no
  flatbuffer schema and is never serialized.
- **Feature predicate (derived)**: the backend's runtime-pass-by-value
  feature signal, derived by scanning the op graph for any tensor with
  `is_pass_by_value == true` and `value == NONE`. There is no separate
  graph-level flag; the per-tensor schema field is the single source of
  truth.
- **Supported plugin SDK API version**: a per-plugin declaration of the
  Plugin SDK API version the plugin was built against, reported via
  `hipdnnPluginGetApiVersion(const char**)` as a `"MAJOR.MINOR.PATCH"`
  string and parsed with `hipdnn_data_sdk::utilities::Version`. Plugins
  that do not export the symbol fall back to `"1.0.0"`.
- **Required plugin SDK API version**: the per-graph minimum the backend
  computes from the features a graph uses; `1.2.0` for runtime
  pass-by-value. A plugin stays in a graph's applicable set only when
  its supported version is `>=` the graph's required version.
- **Version-only filtering**: the applicability model used by
  this RFC, in which provider eligibility is decided by reported API
  version alone, with no per-symbol predicate, because the feature adds
  no new plugin entry point.
