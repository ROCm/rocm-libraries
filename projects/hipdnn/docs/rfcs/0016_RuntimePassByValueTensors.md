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
   - 4.8 [Execute-time variant-pack filter](#48-execute-time-variant-pack-filter)
   - 4.9 [State reference](#49-state-reference)
5. [Key Design Decisions](#5-key-design-decisions)
   - 5.1 [cuDNN API-surface parity](#51-cudnn-api-surface-parity)
   - 5.2 [Reuse the variant-pack pointer map](#52-reuse-the-variant-pack-pointer-map)
   - 5.3 [Derive feature detection from the tensor schema](#53-derive-feature-detection-from-the-tensor-schema)
   - 5.4 [Version-only filtering](#54-version-only-filtering)
   - 5.5 [Deserialized-plan support via the provider payload](#55-deserialized-plan-support-via-the-provider-payload)
   - 5.6 [2-bit encoding (why no compile-time flag)](#56-2-bit-encoding-why-no-compile-time-flag)
6. [Compatibility, Versioning, and Rollback](#6-compatibility-versioning-and-rollback)
7. [Risks](#7-risks)
8. [Execution Plan](#8-execution-plan)
9. [Testing Plan](#9-testing-plan)
10. [Glossary](#10-glossary)

---

## 1. Executive Summary

This RFC proposes adding **runtime pass-by-value tensors** to hipDNN.
A pass-by-value tensor is a host-side scalar operand (epsilon, alpha,
beta, an SDPA scale, etc.). Today hipDNN supports such scalars only as
**compile-time constants**: the value is baked into the operation graph
at build time and frozen into the compiled execution plan. This RFC adds
the ability to mark a scalar tensor with `set_as_runtime_parameter()`
and supply its value through the variant pack at **execute** time,
without rebuilding the graph. The public API mirrors NVIDIA
cuDNN-frontend's pass-by-value model.

The rollout is non-breaking and additive. The new surface consists of:

- A `ScalarType` enum and cuDNN-named constructors, setters, and getters
  on the tensor ([§4.1](#41-frontend-tensor-flag)) mirroring
  cuDNN-frontend.
- **One** defaulted boolean appended to the per-tensor flatbuffer schema
  (`is_runtime_pass_by_value`); combined with value presence it selects the
  state (a 2-bit encoding, [§4.9](#49-state-reference)), and the
  runtime-pass-by-value feature signal is derived from it, with no
  graph-level flag ([§4.3](#43-feature-signal-derived)).
- Two **breaking changes** to reach 1:1 cuDNN parity:
  `get_pass_by_value()` returns the value (not `bool`), and its former
  bool-predicate role moves to `get_is_pass_by_value()`
  ([§6](#6-compatibility-versioning-and-rollback)).
- A `1.2.0` plugin-SDK floor for the pure user-supplied state only. Every
  value-carrying state — the plain constructor, `set_value`
  (runtime-with-default), and `set_compile_time_constant` — bakes on the
  baseline `1.0.0`, exactly as before.
- Version-only per-graph provider filtering.

There is **no new public backend C-API entry**, **no new plugin SDK
symbol**, and **no new variant-pack attribute**. A runtime user-supplied
scalar value reuses the existing `uid → void*` variant-pack map: its entry
is a *host* pointer to the scalar, delivered to the provider through the
existing `hipdnnEnginePluginExecuteOpGraph` device-buffer array.

**Binary-compatibility scope.** The backend computes the minimum
plugin API version each graph requires from the features the graph uses.
Graphs whose scalars carry a baked value (compile-time constant or runtime
default) impose no new version requirement and continue to be served by
existing plugins unchanged. Graphs with a pure user-supplied scalar (no baked
value) require plugins reporting `>= 1.2.0`; older
plugins are filtered out of the applicable set before they are asked about
the graph, so a legacy plugin can never silently mis-serve such a graph. See
[§4.6](#46-feature-detection-and-version-filtering) for the full
versioning model.

---

## 2. Problem Statement

### 2.1 End-user API surface

The desired end-user surface keeps the same execute API and mirrors
cuDNN-frontend. A scalar operand is created in one of the by-value
states ([§4.9](#49-state-reference)):

```cpp
// compile-time constant — value baked, never overridable, no version elevation.
auto c3 = graph.tensor(0.125f, ScalarType::COMPILE_TIME_CONST);

// runtime with default (the default path) — value baked as a default,
// overridable via the variant pack in a future release (override filtered today).
auto k  = graph.tensor(TensorAttributes(0.125f));   // plain ctor, matches cuDNN
auto s2 = graph.tensor(0.125f, ScalarType::RUNTIME_PARAM);

// runtime, user-supplied — value supplied at execute, not baked.
auto scale = graph.tensor(...);
scale->set_as_runtime_parameter();
```

For the user-supplied state the host-side value reaches the provider
through the existing variant-pack map, keyed by the tensor UID, exactly as
device buffers are ([§4.8](#48-execute-time-variant-pack-filter)); the
value-carrying states bake the value in the tensor flatbuffer. cuDNN-frontend
delivers its runtime value with `extend_tensor_map_with_pass_by_value_tensors_`
([`graph_interface.h:190-212`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_interface.h#L190-L212)),
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
3. **Keep the graph descriptor read-only after build.** A user-supplied
   value is not stored on the graph at all; a value-carrying scalar (runtime
   default or compile-time constant) is baked in the tensor flatbuffer, yet
   the user-supplied path is variant-pack-delivered
   ([§4.8](#48-execute-time-variant-pack-filter)). The graph descriptor stays
   read-only after build.

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

Pass-by-value tensors mirror cuDNN-frontend's public surface 1:1, with the
deliberate hipDNN divergences in [§5.1](#51-cudnn-api-surface-parity). One
stored runtime flag `is_runtime_pass_by_value` (true for the two runtime
states) plus value presence select the state over a single re-used value
member ([§4.9](#49-state-reference)); the cuDNN-named getters below are
derived from those two bits. **Every cuDNN-mirrored method is preserved
verbatim** — the "runtime" rename applies only to the flatbuffer field,
backend attr 1307, and the provider wrapper, never to these frontend methods.

**Enum.** A `ScalarType` selects a value-carrying tensor's state at
construction, mirroring cuDNN
([`graph_properties.h:42-45`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_properties.h#L42-L45)):

```cpp
namespace hipdnn_frontend::graph {
enum class ScalarType { RUNTIME_PARAM, COMPILE_TIME_CONST };
}
```

**Constructors.**

- `TensorAttributes(const T& scalar)` → the **runtime-with-default** state,
  delegating to `set_value` — matching cuDNN's plain scalar
  constructor, which sets `pass_by_value` (its runtime type-2)
  ([`graph_properties.h:158-198`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_properties.h#L158-L198)).
- `TensorAttributes(const T& scalar, ScalarType type)`: `RUNTIME_PARAM`
  → **runtime-with-default**, `COMPILE_TIME_CONST` → **compile-time constant**
  ([`graph_properties.h:200-271`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_properties.h#L200-L271)).

**Setters** (all retained, cuDNN names unchanged).

- `set_value<T>(v)` → **runtime-with-default** (value stored, runtime
  flag true) — the default path, matching cuDNN's plain scalar constructor;
  the value is baked as a default, overridable later.
- `set_compile_time_constant(pass_by_values_t v)` → compile-time constant
  (value stored, runtime flag false)
  ([`graph_properties.h:384-392`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_properties.h#L384-L392)).
- `set_as_runtime_parameter()` → runtime user-supplied: sets the runtime
  flag true and **clears** any prior value (a deliberate divergence from
  cuDNN, whose `set_as_runtime_parameter` leaves `pass_by_value` set,
  [`graph_properties.h:394-400`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_properties.h#L394-L400)).
- `set_is_pass_by_value(bool)` — retained (cuDNN has it,
  [`graph_properties.h:367-371`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_properties.h#L367-L371));
  sets the runtime flag. `true` with no value yields the user-supplied state.

**Getters** (derived from the runtime flag + value presence; cuDNN names
unchanged).

- `get_pass_by_value()` returns the **value** (`std::optional`-style /
  value variant), not `bool`, present iff `runtime flag && value present`
  (the runtime-with-default state), empty otherwise
  ([`graph_properties.h:357-360`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_properties.h#L357-L360)).
- `get_compile_time_constant()` returns the value iff `!runtime flag &&
  value present` (the compile-time constant), empty otherwise
  ([`graph_properties.h:379-382`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_properties.h#L379-L382)).
- `get_is_pass_by_value()` is the derived umbrella predicate =
  `runtime flag || value present` (true for all three by-value states)
  ([`graph_properties.h:362-365`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_properties.h#L362-L365)).
- `get_has_compile_time_constant()` is the derived bool = `!runtime flag &&
  value present`; mirrors cuDNN
  ([`graph_properties.h:374-377`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_properties.h#L374-L377)).

**Breaking changes** (from the current hipDNN API): (a)
`get_pass_by_value()` returns the value instead of `bool`; (b) its former
"is-pass-by-value" bool-predicate role moves to `get_is_pass_by_value()`.

`Graph::tensor(const TensorAttributes&)`
([`Graph.hpp`](../../frontend/include/hipdnn_frontend/Graph.hpp)) is
retained, and `graph.tensor(scalar, ScalarType)` overloads are added (one per supported
scalar type), each delegating to the `TensorAttributes(scalar, ScalarType)`
constructor: `graph.tensor(v, ScalarType::RUNTIME_PARAM)` → runtime-with-default and
`graph.tensor(v, ScalarType::COMPILE_TIME_CONST)` → compile-time. Like cuDNN there is
no bare-scalar `graph.tensor(v)` overload; the plain default is reached
via `graph.tensor(TensorAttributes(v))` (runtime-with-default).

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

The per-tensor flag is persisted so a provider reading the serialized op
graph can identify runtime pass-by-value scalars and distinguish them from
baked constants. The flatbuffer `TensorAttributes` table gains **one**
defaulted boolean appended as its last field, re-using the existing `value`
union:

```
is_runtime_pass_by_value: bool = false;
```

This is the append-only, defaulted-field pattern used in `develop` by the
override-shape graph and plan flags ([RFC 0008](0008_OverridableTensorShapesDesign.md)), wire-compatible per [RFC 0005](0005_Versioning.md):
a pre-feature graph deserialized in a runtime that understands the field
reads `false` on every tensor. The flag round-trips through descriptor
pack/unpack alongside the `value`, and both are persisted so
`get_pass_by_value` / `get_compile_time_constant` / `get_is_pass_by_value`
are correct after graph deserialize. cuDNN keeps its analogous
`pass_by_values` in its FE JSON across serialize/deserialize
([`graph_interface.h:1588-1593`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_interface.h#L1588-L1593),
[`graph_interface.h:1666-1673`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_interface.h#L1666-L1673));
hipDNN keeps it in the tensor flatbuffer instead.

**Backend attributes.** `HIPDNN_ATTR_TENSOR_VALUE_EXT` (1306) is kept as-is
(the value union; it holds a compile-time constant OR a runtime default).
The existing read-only `HIPDNN_ATTR_TENSOR_IS_BY_VALUE` (1307) — today
derived from `value.type != NONE` — is **renamed**
`HIPDNN_ATTR_TENSOR_IS_RUNTIME_PASS_BY_VALUE` and made **settable**, true
only for the two runtime states; its wire identity stays the integer
`1307`. No separate compile-time-constant flag is introduced; the runtime
flag and value presence already distinguish all states
([§5.6](#56-2-bit-encoding-why-no-compile-time-flag)).

**Deserialize invariant.** The value is read from the union whenever it is
present (`value.type != NONE`), **not gated on the flag**; the flag is read
independently. This is what makes a legacy baked scalar (value present, flag
absent → `false`) deserialize correctly as a compile-time constant. It
changes the current descriptor unpack gate in
[`DescriptorUnpackHelpers.hpp`](../../frontend/include/hipdnn_frontend/detail/DescriptorUnpackHelpers.hpp),
which keys the value-read on `IS_BY_VALUE == true`, to a value-presence gate.

Providers read the flag through the `isRuntimePassByValue()` accessor added
to the op-graph tensor wrapper
([`ITensorAttributesWrapper`](../../flatbuffers_sdk/include/hipdnn_flatbuffers_sdk/flatbuffer_utilities/TensorAttributesWrapper.hpp)),
mirroring the existing `isVirtual()`. It is the accessor the provider
example in [§4.5](#45-provider-contract) consumes.

### 4.3 Feature signal (derived)

The backend's runtime-pass-by-value feature signal is **derived from the
per-tensor flag** ([§4.2](#42-tensor-schema-addition)): a graph requires
runtime pass-by-value support iff it contains at least one tensor with
`is_runtime_pass_by_value == true` **and no baked value** (the user-supplied
state). A value-carrying scalar bakes into `VALUE_EXT`, so a baseline plugin
serves it by reading it — it imposes no floor. The per-tensor flag is the
single source of truth; the feature is derived from it rather than a
graph-level flag ([§5.3](#53-derive-feature-detection-from-the-tensor-schema)).

The backend computes the signal with a `readIsRuntimePassByValueEnabled`
helper that scans the serialized op graph at applicability time; a graph
whose runtime-flagged tensors all carry a baked value (or has none) yields
`false` and needs no version elevation. This is the single signal feature
detection consumes ([§4.6](#46-feature-detection-and-version-filtering)).
This version-floor predicate is separate from the runtime-slot discriminator
`isRuntimePassByValue()` alone ([§4.5](#45-provider-contract)), which marks
both runtime states.

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

### 4.5 Provider contract

A provider reporting plugin SDK API version `>= 1.2.0` must, for any tensor
matching the runtime discriminator `isRuntimePassByValue()` (the
user-supplied and runtime-with-default states), read the scalar from that
UID's slot in the `device_buffers` array **as a host pointer** at execute
time, using it to override any seeded default. A compile-time constant
(`isRuntimePassByValue() == false`) is read from the op-graph flatbuffer as
today.

**Capability assertion.** Reporting `>= 1.2.0` asserts runtime
pass-by-value capability. A provider that reports `>= 1.2.0` but cannot
serve a particular runtime pass-by-value graph MUST decline it (return a
not-applicable / reject result at execute) rather than mis-serve it by
reading the host-scalar slot as a device pointer. This is the
fresh-build-path analogue of the deserialize reject obligation below.

The plugin entry point is unchanged:

```c
hipdnnPluginStatus_t
hipdnnEnginePluginExecuteOpGraph(hipdnnEnginePluginHandle_t handle,
                                 hipdnnEnginePluginExecutionContext_t execution_context,
                                 void* workspace,
                                 const hipdnnPluginDeviceBuffer_t* device_buffers,
                                 uint32_t num_device_buffers);
```

The runtime-slot discriminator is `isRuntimePassByValue()` alone (true for
the user-supplied and runtime-with-default states). It is **distinct** from
the version-floor feature signal ([§4.3](#43-feature-signal-derived)), which
fires only for the user-supplied state (`flag && no baked value`); a
runtime-with-default value is baked in `VALUE_EXT`, so a baseline plugin
serves it by reading that value.

**Defaulted-tensor caveat.** A runtime-with-default tensor carries a
baked value read as its default; its variant-pack override is filtered and
warned at the frontend today ([§4.8](#48-execute-time-variant-pack-filter)),
so no host pointer for its UID reaches `device_buffers` and the provider
uses the baked default. A compile-time constant is
`isRuntimePassByValue() == false` and never appears in `device_buffers` as a
host scalar. Only a user-supplied scalar arrives as a host pointer.

Each `hipdnnPluginDeviceBuffer_t` carries its `uid`, and
[§4.2](#42-tensor-schema-addition) adds the `isRuntimePassByValue()`
accessor, so a provider seeds each scalar from its schema value, records the
runtime-flagged UIDs once from the op graph, and overrides those slots from
`device_buffers` at execute:

```cpp
// Setup (once per graph): seed each scalar from its schema value and record
// which UIDs are runtime pass-by-value. isRuntimePassByValue() is true for
// both runtime states. A compile-time constant is flag=false + value
// present: read here, baked, never delivered in device_buffers.
Plan p;
std::unordered_set<int64_t> runtimeUids;
for (auto const& tensor : opGraph.tensors()) {          // ITensorAttributesWrapper
    if (tensor.valueType() != TensorValue::NONE) {
        p.set_scalar(tensor.uid(), tensor.value<float>());  // baked default or compile-time constant
    }
    if (tensor.isRuntimePassByValue()) {
        runtimeUids.insert(tensor.uid());               // user-supplied (no default) or runtime-with-default
    }
}
p.finalize();

// Execute: device_buffers carries {uid, ptr} for every bound tensor. A
// runtime pass-by-value slot is a HOST pointer whose value overrides the
// seeded default. Today only a user-supplied scalar (no baked default)
// arrives here; a runtime-with-default override is filtered out and warned
// by the frontend, and a compile-time constant is never runtime — so both
// keep their baked value.
for (uint32_t i = 0; i < num_device_buffers; ++i) {
    const hipdnnPluginDeviceBuffer_t& buf = device_buffers[i];
    if (runtimeUids.count(buf.uid)) {
        p.set_scalar(buf.uid, *static_cast<const float*>(buf.ptr));  // host ptr overrides default
    }
    // else: ordinary DEVICE pointer, handled as today.
}
p.execute();
```

The `Plan` / `set_scalar` / `finalize` / `execute` names are the provider's
own kernel-config object (illustrative); `valueType()` / `value<T>()` /
`isRuntimePassByValue()` are the real `ITensorAttributesWrapper` accessors.
Keep the value-read typed by the tensor's declared `data_type` (the `float`
above is illustrative).

The flow above runs on the **fresh-build** path, where the op graph is
available. On a **deserialized** execution plan the op graph and
per-tensor attributes are not reconstructed
([§5.5](#55-deserialized-plan-support-via-the-provider-payload)): the
plan is rebuilt from the engine ID, workspace size, a bare tensor-UID
list, and the provider's opaque `plugin_payload` alone. A provider that
supports runtime pass-by-value (reports `1.2.0`) must therefore:

1. **Persist** the runtime pass-by-value UID set (`runtimeUids`
   above) into its serialized `plugin_payload` and restore it on
   deserialize; the host-scalar identity is otherwise lost across
   serialization.
2. **Version** that payload. The `plugin_payload` is opaque to hipDNN
   and its versioning is plugin-owned, so
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
signal from [§4.3](#43-feature-signal-derived) (`readIsRuntimePassByValueEnabled`);
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
| Runtime pass-by-value — any tensor `is_runtime_pass_by_value && no baked value` (user-supplied), with or without override | `1.2.0` |

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
re-binds by the baked
`engineId` with no version re-filter, but its runtime pass-by-value
state lives in the provider's opaque payload, which the provider
versions and validates on deserialize; that path is covered by the
provider contract, not a core gate
([§5.5](#55-deserialized-plan-support-via-the-provider-payload)).

Version parsing and comparison use the existing
`hipdnn_data_sdk::utilities::Version`
([`data_sdk/include/hipdnn_data_sdk/utilities/VersionUtils.hpp`](../../data_sdk/include/hipdnn_data_sdk/utilities/VersionUtils.hpp)).

### 4.7 Frontend validation

`TensorAttributes::validate()` runs at build time and enforces, in addition
to the existing checks, the virtual-exclusion rules below. The 2-bit
encoding is orthogonal — every flag/value quadrant is otherwise valid — so a
virtual tensor (an internal graph edge) is the only inconsistency to reject:

1. `INVALID_VALUE` if `virtual && is_runtime_pass_by_value` — a virtual
   tensor cannot be a runtime host scalar.
2. `INVALID_VALUE` if `virtual && value present` — a virtual tensor cannot
   carry a baked value.

Both are reachable and frontend-testable, and mirror cuDNN-frontend's
`validate()`
([`graph_properties.h:70-94`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_properties.h#L70-L94)).
There is no rule keying "value present" to a particular mode: a value with
the runtime flag set is a runtime default, a value without it is a
compile-time constant, and both are valid.

**Post-build immutability.** The compiled plan is frozen at build
(`backendFinalize`, [§2.3](#23-constraints)): a scalar's baked value and
pass-by-value flag are captured in the backend `TensorDescriptor` and the
serialized plan, and nothing re-reads the frontend `TensorAttributes`
after build. Post-build calls to the value/mode setters (`set_value`,
`set_compile_time_constant`, `set_as_runtime_parameter`,
`set_is_pass_by_value`) are therefore inert: they mutate only the detached
frontend object and cannot alter the compiled plan. The only way to vary a
user-supplied scalar after build is the variant pack at execute
([§4.4](#44-execute-time-transport)).

`detail::validateScalarParameter`
([`frontend/include/hipdnn_frontend/node/detail/Utilities.hpp`](../../frontend/include/hipdnn_frontend/node/detail/Utilities.hpp), ~line
475) currently requires `get_pass_by_value()` to be true for required
scalar inputs (epsilon, SDPA scale, etc.). It is relaxed to accept any
by-value scalar (`get_is_pass_by_value() == true`, the derived umbrella),
whether the value is baked (compile-time or runtime-with-default) or
user-supplied. As today, the actual numeric value cannot be checked at
build time for a user-supplied scalar because it is not yet available.

### 4.8 Execute-time variant-pack filter

A user-supplied scalar's value reaches the provider as a host pointer
in the variant pack at `Graph::execute()` ([§4.4](#44-execute-time-transport)).
A runtime-with-default or compile-time scalar instead carries its
value baked in the tensor flatbuffer (`VALUE_EXT`); the provider reads that
baked value as the default.

**Forwarded-UID filter.** `Graph::execute()` builds the set of variant-pack
UIDs it forwards to the provider and **filters out any UID whose tensor
carries a baked value**, leaving only the pure user-supplied UIDs —
mirroring cuDNN's `variant_pack_uids` set and its `emplace` precedence
([`graph_interface.h:190-212`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_interface.h#L190-L212),
[`graph_interface.h:2858-2859`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_interface.h#L2858-L2859)).
If the user supplies a variant-pack value for a defaulted UID, the frontend
**logs a warning, drops that override, and continues** — the ignored-override
behavior is visible, matching cuDNN's warn-and-continue (no error). Because a
runtime-with-default override is filtered today, that state is served at
baseline `1.0.0` from its baked default; only the user-supplied state (no
baked value — the value MUST come from the variant pack) requires `1.2.0`. A
future release that honors the override elevates it to `1.2.0` then.

**Compiled-plan path.** On the compiled-plan path `deserializeBackendPlan`
reconstructs no per-tensor attributes
([§5.5](#55-deserialized-plan-support-via-the-provider-payload);
[`ExecutionPlanDescriptor.cpp:416-477`](../../backend/src/descriptors/ExecutionPlanDescriptor.cpp#L416-L477)),
so a runtime-with-default tensor round-tripped through `to_compiled_plan_binary`
loses its baked default and degrades to user-supplied semantics — the caller must
supply the value at execute. cuDNN avoids this: its serialize bundles
`pass_by_values` with the plan JSON, so the value survives
([`graph_interface.h:1583-1593`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_interface.h#L1583-L1593),
[`graph_interface.h:1666-1693`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_interface.h#L1666-L1693)).
This limitation is explicit and covered by test ([§9](#9-testing-plan)).

**Usage guidance.** `set_value` / the plain constructor is the default path
(runtime-with-default, baseline `1.0.0`, override deferred). Use
`set_compile_time_constant` for a value that must never be overridable.
The user-supplied path (`set_as_runtime_parameter`) is the only one
that needs a `1.2.0` provider.

### 4.9 State reference

A pass-by-value tensor is in one of the by-value states below, selected by
the stored runtime flag `is_runtime_pass_by_value` (default `false`) plus
value presence over a single re-used `ValueVariant _value` member — a 2-bit
orthogonal encoding ([§5.6](#56-2-bit-encoding-why-no-compile-time-flag)):

| State | Creation (frontend) | runtime flag | `value` | `get_is_pass_by_value()` | `get_pass_by_value()` | `get_compile_time_constant()` | Delivery | Provider floor |
|---|---|---|---|---|---|---|---|---|
| **Runtime, user-supplied** | `set_as_runtime_parameter()`; or `set_is_pass_by_value(true)` with no value | true | ∅ | true | ∅ | ∅ | user supplies host ptr in variant pack | `1.2.0` |
| **Runtime with default (default path)** | `TensorAttributes(v)`; `set_value(v)`; `TensorAttributes(v, ScalarType::RUNTIME_PARAM)` | true | v | true | v | ∅ | baked default in `VALUE_EXT`; overridable via variant pack (override filtered + warned today, [§4.8](#48-execute-time-variant-pack-filter)) | baseline `1.0.0` |
| **Compile-time constant** | `set_compile_time_constant(v)`; `TensorAttributes(v, ScalarType::COMPILE_TIME_CONST)` | false | v | true | ∅ | v | baked in op-graph flatbuffer, read via existing path | baseline `1.0.0` |

(∅ = empty / `std::monostate`.)

---

## 5. Key Design Decisions

### 5.1 cuDNN API-surface parity

**Decision**: mirror cuDNN-frontend's pass-by-value surface 1:1 — the
`ScalarType` enum, both constructors, the cuDNN-named setters/getters, and
the `graph.tensor(scalar, ScalarType)` factory, each enumerated with its
cuDNN citation in [§4.1](#41-frontend-tensor-flag) — with the divergences
below.

**Rationale**: adopting cuDNN's names and shapes lets cuDNN users port
with no concept translation, and the 2-bit model
([§4.9](#49-state-reference)) covers cuDNN's full
fused-constant-vs-execute-time surface
([`graph_properties.h:53-57`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_properties.h#L53-L57)).
The plain scalar constructor now **matches** cuDNN (runtime-with-default),
removing the earlier "plain-ctor bakes a constant" divergence.

**Divergences (decisions, not gaps)**:

1. hipDNN stores a single `is_runtime_pass_by_value` flag (runtime-only)
   over a **single** re-used value member, whereas cuDNN stores an
   `is_pass_by_value` umbrella plus a separate `has_compile_time_constant`
   and two value members (`pass_by_value`, `compile_time_constant_value`,
   [`graph_properties.h:118-123`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_properties.h#L118-L123)).
   The frontend getters (`get_is_pass_by_value`, `get_compile_time_constant`,
   `get_has_compile_time_constant`) are derived from the runtime flag + value
   presence for porting parity.
2. `set_as_runtime_parameter()` **clears** any prior value, whereas cuDNN
   leaves `pass_by_value` set
   ([`graph_properties.h:394-400`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_properties.h#L394-L400)).

### 5.2 Reuse the variant-pack pointer map

**Decision**: transport a user-supplied scalar as a host
pointer in the existing `uid → void*` variant-pack map, delivered through
the unchanged `hipdnnEnginePluginExecuteOpGraph` `device_buffers` array. The
caller fills the slot. No new variant-pack attribute and no new plugin
symbol.

**Rationale**: this is cuDNN-frontend's model — one host-pointer slot for a
runtime scalar, keyed by UID in the same map as device buffers. Consumers
port with no signature changes; nothing is added to
`populateBaseVariantPackDescriptor`, the plugin C ABI, or the 700-799
attribute range.

**Trade-off**: the provider must consult the discriminator
`isRuntimePassByValue()` to know a slot holds a *host* pointer
([§4.9](#49-state-reference)). A runtime-with-default value carries the
`1.2.0` floor only once its override is honored; until then its baked
default serves at baseline `1.0.0`, so for a build-fixed value prefer the
compile-time constant or accept the default — the cost is observable
only as provider availability, never a wrong result.

### 5.3 Derive feature detection from the tensor schema

**Decision**: detect the feature from the per-tensor flag (any tensor with
`is_runtime_pass_by_value && no baked value`, i.e. the user-supplied state)
rather than a separate graph-level `is_pass_by_value_enabled` flag.

**Rationale**: the per-tensor flag is already the mandatory discriminator
separating a runtime host-scalar slot from a device buffer or a baked
constant ([§4.5](#45-provider-contract)), so it is the single source of
truth. A graph-level flag would be a denormalized cache that can disagree —
and unsafely: a raw backend C-API caller could set the tensor attribute but
not the graph one, leaving the filter reading `false` while a runtime
tensor exists, so a sub-`1.2.0` plugin reads a host pointer as a device
pointer. Deriving makes that desync unrepresentable. [RFC 0008](0008_OverridableTensorShapesDesign.md)
uses a graph-level flag only because override shapes have no per-tensor
field to derive from; runtime pass-by-value does, so mirroring it would
import a desync risk override never had.

**Trade-off**: a one-time `O(tensors)` walk of the already-materialized
serialized graph per applicability query instead of a bool read —
negligible, on a non-hot path
([`EnginePluginResourceManager.cpp:341-407`](../../backend/src/plugin/EnginePluginResourceManager.cpp#L341-L407)).

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
([§9](#9-testing-plan)) rather than a runtime guard.

### 5.5 Deserialized-plan support via the provider payload

**Decision**: add no execution-plan schema field and no hipDNN
dispatch-time version gate; the provider's opaque `plugin_payload` carries
the runtime pass-by-value UID set across serialization, under the
persist/version/reject obligations of
[§4.6](#46-feature-detection-and-version-filtering).

**Rationale**: on the deserialized path the op graph is gone
(`deserializeBackendPlan`, [§4.8](#48-execute-time-variant-pack-filter)), so
the only place host-scalar identity can survive is the provider's own
payload, which it already owns and versions — skew safety reduces to that
existing payload-versioning contract. This is exactly RFC 0009's
payload-ownership rule: the envelope omits the op graph, so *"plugins that
need graph-derived data must store it in their own payload"*
([RFC 0009, Envelope Format](0009_CompiledPlanSerialization.md#envelope-format)).

**Trade-off**: deserialized-path safety rests on the provider obeying the
§4.5 payload-versioning contract; core enforces no version floor there,
and the skew failure is memory-unsafe, not merely wrong-result — the same
trust-the-version posture already accepted for the host-pointer read
itself ([§7.1](#71-provider-reports-120-but-mishandles-host-pointers)). It
is preferred over a sibling `is_pass_by_value_enabled` on
`SerializedExecutionPlan` (peer to `is_override_shape_enabled`,
[`execution_plan.fbs:12`](../../flatbuffers_sdk/schemas/execution_plan.fbs#L12))
because it keeps host-scalar identity where the plugin payload owns
graph-derived data, not in a core gate the provider does not control.

**Retrofit limit**: the skew window closes only when the older
same-`engineId` release already rejected payloads it could not interpret.
A provider that shipped before runtime pass-by-value without defensive
payload versioning cannot be made safe retroactively, so adopting that
discipline is a precondition of shipping runtime pass-by-value
([§4.5](#45-provider-contract), [Step 7](#step-7-provider-adoption)).

### 5.6 2-bit encoding (why no compile-time flag)

**Decision**: encode the by-value state from one stored runtime flag
(`is_runtime_pass_by_value`) plus value presence — four orthogonal
quadrants — rather than adding a second stored `is_compile_time_constant`
flag.

**Rationale**: the runtime flag and value presence are already orthogonal
and together cover every state, so a separate compile-time flag is
redundant.
Worse, a second flag reintroduces a legacy-graph ambiguity: a pre-feature
baked scalar deserializes with the new flags defaulting `false`, which the
two-flag rules would reject or misclassify. Under the 2-bit encoding that
same tensor — `value present, is_runtime_pass_by_value == false` — is
unambiguously a compile-time constant by definition, so the legacy graph
round-trips correctly with no normalization sentinel. This is what closes
the backward-compatibility hole the earlier two-flag design carried.

---

## 6. Compatibility, Versioning, and Rollback

**Upgrade path.** Existing plugins (in-tree and out-of-tree) continue
to serve ordinary graphs and graphs whose scalars carry a baked value
unchanged; they require no rebuild. A plugin adopts runtime pass-by-value by
reading host-scalar slots for tensors matching `isRuntimePassByValue()` and
bumping its reported API version to `1.2.0`. Only the user-supplied state (no
baked value) requires `1.2.0`; the value-carrying states stay baseline
`1.0.0`. Each plugin migrates on its own schedule.

**Breaking changes.** `get_pass_by_value()` becomes value-returning and
its bool-predicate role moves to `get_is_pass_by_value()`
([§4.1](#41-frontend-tensor-flag)); source-level only, no wire format or
plugin ABI change.

**Version skew.** An older plugin paired with a user-supplied
graph is filtered out by the per-graph version gate
([§4.6](#46-feature-detection-and-version-filtering)). If no plugin
reports `>= 1.2.0`, applicability returns a clean "no applicable
engines" result. A legacy plugin never receives such a graph and
therefore never reads a host pointer where it expects a device buffer,
so on the applicability-filtered fresh-build path there is no silent
wrong-result path. Graphs whose scalars all carry a baked value impose no
floor, so a legacy plugin serves them by reading `VALUE_EXT`.

The serialized/deserialized path is not core-gated. A plan re-bound by
`engineId` to a downgraded plugin relies on the provider versioning its
opaque `plugin_payload`: a provider that versions it correctly rejects
the mismatched payload at deserialize rather than mis-reading a host
pointer, while a provider that does not is the residual risk
[§5.5](#55-deserialized-plan-support-via-the-provider-payload) discloses,
since hipDNN core no longer enforces a version floor on that path.

**Non-breaking schema compatibility & the legacy story.** The one new
schema field (`TensorAttributes.is_runtime_pass_by_value`) is appended,
defaulted `false`, and wire-compatible per [RFC 0005](0005_Versioning.md). A
**legacy graph with a baked scalar** (serialized before this feature)
deserializes as `value present, is_runtime_pass_by_value == false` — a
compile-time constant by definition — validates, imposes no version floor,
and is served by any existing plugin exactly as before. A legacy graph with
**no value** deserializes as an ordinary tensor. Note the intentional
asymmetry: a *new* `set_value(v)` graph is runtime-with-default (`flag true`)
while a *legacy* baked scalar is compile-time (`flag false`) — both baseline
`1.0.0`, both read the same baked `VALUE_EXT`, so behavior is identical on
existing plugins; the flag only marks new graphs as override-capable in a
future release.

**Rollback.** The feature is inert unless a caller creates a
user-supplied tensor. Reverting a caller to `set_value(scalar)` / the plain
constructor keeps the value baked (runtime-with-default), or use
`set_compile_time_constant`; either restores
the baked-value path with zero schema migration: the value is baked, no
version is elevated, and `computeMinimumPluginApiVersion` returns the
baseline. No data migration or plan invalidation is required.

---

## 7. Risks

### 7.1 Provider reports 1.2.0 but mishandles host pointers

**Risk**: a provider bumps its reported version to `1.2.0` but reads a
runtime pass-by-value slot as a device pointer (or otherwise mishandles
the host scalar), producing wrong results.

**Mitigation**: this is a plugin implementation bug, not a hipDNN defect —
a provider that reports `1.2.0` asserts it reads host-scalar slots
correctly. The version contract is the sole *capability* signal — by
design there is no per-symbol safety net
([§5.4](#54-version-only-filtering)). The applicability filter rejects a
plugin whose *reported version* is too low, but no core check can catch
a plugin that truthfully reports `1.2.0` yet mishandles the host
pointer; that residual risk is covered by the integration suite's fake
`1.2.0` plugin, which asserts the value it receives equals what the
caller supplied ([§9](#9-testing-plan)).

### 7.2 Caller marks a tensor pass-by-value but omits the value at execute

**Risk**: a tensor is marked runtime pass-by-value but the caller does not
insert its UID into the variant-pack map, so the provider reads an
unset or garbage slot.

**Mitigation**: documented contract. The frontend cannot validate the
numeric value at build time (it does not yet exist); it validates only
that the tensor is structurally a scalar. This matches the existing
behavior for required scalar parameters
(`validateScalarParameter`).

### 7.3 Host vs device pointer confusion in the shared map

**Risk**: the same variant-pack map carries both device pointers and
host scalar pointers, so a provider could dereference the wrong kind.

**Mitigation**: on the fresh-build path the discriminator
`isRuntimePassByValue()` is the authoritative marker of a
host-pointer slot; on a deserialized plan the provider relies on the runtime
pass-by-value UID set persisted in its payload
([§5.5](#55-deserialized-plan-support-via-the-provider-payload)). The 2-bit
model ([§4.9](#49-state-reference)) keeps the discriminator unambiguous.
Round-trip and end-to-end tests cover both slot kinds in one graph.

### 7.4 Downgraded provider mis-reads a serialized pass-by-value plan

**Risk**: a compiled pass-by-value plan is serialized against a
`1.2.0` provider, then deserialized where the same `engineId` resolves
to a downgraded `< 1.2.0` build of that provider. Deserialize re-binds
by `engineId` with no core version check
([§4.6](#46-feature-detection-and-version-filtering)), so the older build
receives a payload it did not produce and could read a host scalar as a
device pointer — a memory-unsafe failure.

**Mitigation**: the §4.5 provider contract requires a `1.2.0` provider
to version its `plugin_payload` and reject a version/kind it cannot
interpret before reading any slot, so a provider that practiced
defensive payload versioning fails the deserialize cleanly. The build-time
version filter ([§4.6](#46-feature-detection-and-version-filtering)) runs
only on the fresh-build path, not on deserialize, so the safety of this
path rests solely on plugin payload versioning.

### 7.5 Provider reads the stored value of a runtime-with-default tensor

**Risk**: a runtime-with-default tensor carries a value
(`HIPDNN_ATTR_TENSOR_VALUE_EXT`) in the flatbuffer as its default — a
provider that honors a runtime override must apply the variant-pack value
over that default rather than silently using the stored default when an
override was supplied.

**Mitigation**: benign today. The override is filtered and warned at the
frontend ([§4.8](#48-execute-time-variant-pack-filter)), so no override
reaches the provider and the baked default is the intended value; the
provider reads it from `VALUE_EXT` exactly as for a compile-time constant.
When a future release honors the override, this state elevates to `1.2.0`
and the provider reads the variant-pack slot as a host pointer over the
default — a version-gated behavior change, not a schema change. The
end-to-end filter test asserts the warning fires and the default (not the
attempted override) reaches the provider.

### 7.6 Deserialize value-read must gate on value presence, not the flag

**Risk**: if the descriptor unpack path gated the value-read on the runtime
flag (as the pre-RFC `IS_BY_VALUE` gate did), a compile-time constant
(`value present, flag false`) — and every legacy baked scalar — would lose
its value on deserialize.

**Mitigation**: the deserialize invariant reads the value whenever the
`value` union is present, independent of the flag
([§4.2](#42-tensor-schema-addition)); the flag is read separately. This is
the change that makes the legacy baked scalar round-trip correctly as a
compile-time constant.

---

## 8. Execution Plan

Implementation plan for the work this RFC enables, ordered so the tree
builds and existing tests pass after each step:

### Step 1: Schema field

Append one defaulted boolean, `is_runtime_pass_by_value: bool = false;`, to
the `TensorAttributes` flatbuffer table (re-using the existing `value`
union); regenerate. Add round-trip coverage of the flag plus the value,
including the default-`false` case.

### Step 2: Backend descriptor and enums

Rename the existing read-only `HIPDNN_ATTR_TENSOR_IS_BY_VALUE` (1307) to a
**settable** `HIPDNN_ATTR_TENSOR_IS_RUNTIME_PASS_BY_VALUE` carrying the
`is_runtime_pass_by_value` flag (true for the runtime states), and update the
`BackendEnumStringUtils` string. Keep `HIPDNN_ATTR_TENSOR_VALUE_EXT` (1306)
as-is (value union). No 1308 / `IS_COMPILE_TIME_CONSTANT` attribute is
introduced. Change the descriptor unpack value-read gate
([`DescriptorUnpackHelpers.hpp`](../../frontend/include/hipdnn_frontend/detail/DescriptorUnpackHelpers.hpp))
from `IS_BY_VALUE`-gated to **value-presence-gated** (read the value
whenever the union is present), and read the runtime flag independently.
Wire both through the existing `TensorDescriptor` get/set-attribute and
pack/unpack paths. Add the `isRuntimePassByValue()` accessor to
`ITensorAttributesWrapper`. No operation-graph attribute is added.

### Step 3: Version constant and filter

Add `K_PASS_BY_VALUE_MIN_API_VERSION = "1.2.0"` to
[`PluginVersionConstants.hpp`](../../plugin_sdk/include/hipdnn_plugin_sdk/PluginVersionConstants.hpp); bump [`engine_api_version.h`](../../plugin_sdk/include/hipdnn_plugin_sdk/engine_api_version.h) minor to `2`.
Add a `readIsRuntimePassByValueEnabled(graphDesc)` helper that scans the
serialized op graph and returns `true` iff any tensor has
`is_runtime_pass_by_value == true` **and no baked value** (user-supplied only; no
graph-level attribute is read; the per-tensor flag is the source of truth).
Extend `computeMinimumPluginApiVersion(bool isOverride, bool isRuntimePassByValue)`
to take the second flag and return the maximum required version (the
applicability-time filter). Add the execute-time filter+warn: drop a
variant-pack override for a UID with a baked value, log a warning,
continue.

### Step 4: Frontend API and validation

Add `enum class ScalarType { RUNTIME_PARAM, COMPILE_TIME_CONST };`; the
`TensorAttributes(const T&, ScalarType)` constructor; `set_compile_time_constant`
and `set_as_runtime_parameter`; and `get_compile_time_constant` /
`get_has_compile_time_constant` / `get_is_pass_by_value`. **Keep every
cuDNN frontend method name verbatim** — no renames; only their internal
derivation changes (from the single runtime bit + value presence). There is
no stored `is_compile_time_constant` member; `get_compile_time_constant` /
`get_has_compile_time_constant` are derived from `!runtime bit && value
present`. Route the plain scalar constructor and `set_value` through the
runtime-with-default state (runtime bit true, value stored). Change
`get_pass_by_value()` to return the value variant (present iff `runtime bit
&& value present`, i.e. runtime-with-default), moving its former bool-predicate role to
`get_is_pass_by_value()`, and migrate every internal value-presence caller
off `get_pass_by_value()` onto `get_value_variant()` — the known callsite is
`createOrFindTensorDesc` in
[`DescriptorHelpers.hpp`](../../frontend/include/hipdnn_frontend/detail/DescriptorHelpers.hpp)
(writes the baked value); `search` for `get_pass_by_value()` to enumerate
the rest. Add the `graph.tensor(scalar, ScalarType)` factory overloads
([§4.1](#41-frontend-tensor-flag)). Add the frontend validations
([§4.7](#47-frontend-validation)): the two virtual-exclusion checks and the
relaxed `validateScalarParameter`. No graph-level setter or graph schema
field is added — the runtime-pass-by-value feature signal is derived from
the per-tensor flag ([§4.3](#43-feature-signal-derived)).

### Step 5: Execute-time variant-pack filter

Build the forwarded variant-pack UID set ([§4.8](#48-execute-time-variant-pack-filter)):
`Graph::execute()` forwards only pure user-supplied UIDs and filters
out any UID whose tensor carries a baked value. A user-supplied
value for a filtered (defaulted) UID logs a warning and is dropped, then
execution continues. The provider reads the schema value as the default for
the value-carrying states. On the compiled-plan path no per-tensor value is
reconstructed, so a runtime-with-default tensor degrades to user-supplied
semantics; document the limitation.

### Step 6: Cross-cutting tests

Fake plugins (one at `1.2.0` consuming the host scalar, one below it)
and the 2-bit state / version-floor / filter-and-warn / end-to-end matrix
([§9](#9-testing-plan)).

### Step 7: Provider adoption

A shipping provider that adopts runtime pass-by-value **MUST**: read
host-scalar slots for any tensor matching the runtime discriminator
`isRuntimePassByValue()`; persist its runtime pass-by-value
UID set into its serialized `plugin_payload` and restore it on deserialize
([§5.5](#55-deserialized-plan-support-via-the-provider-payload)); version
that payload and reject a payload whose version/kind it cannot interpret
before reading any slot ([§4.5](#45-provider-contract)); and bump its
reported version to `1.2.0`. The reject-on-unknown-payload requirement
is what keeps a downgraded re-bind from dereferencing a host pointer as
device memory, and it only protects releases that practiced this
versioning from the start (§5.5 retrofit limit). Provider work is
independent of Steps 1-6 and lands on its own schedule.

---

## 9. Testing Plan

Test conventions follow [RFC 0006](0006_PluginAgnosticIntegrationTests.md). The plan exercises:

- **2-bit state round-trip.** A graph with one tensor per by-value state —
  user-supplied (flag true, no value), runtime-with-default (flag true, value
  present), compile-time constant (flag false, value present) — survives
  descriptor → serialize → deserialize → read-back with the
  `is_runtime_pass_by_value` flag and the `value` intact, including the
  default-`false` case for tensors that never opt in, so
  `get_pass_by_value` / `get_compile_time_constant` / `get_is_pass_by_value`
  are correct after graph deserialize.

- **Legacy round-trip.** A pre-feature `TensorAttributes` with a baked
  `value` and no flag deserializes as a compile-time constant (`value
  present, is_runtime_pass_by_value == false`), validates, imposes no
  version floor, and executes on a baseline `1.0.0` plugin — the reviewer's
  legacy-compat case, served unchanged.

- **Getter-return assertions.** Per state, assert the getter semantics of
  the state table ([§4.9](#49-state-reference)):
  `get_is_pass_by_value()` is `true` for all three by-value states;
  `get_pass_by_value()` returns the value only for the runtime-with-default
  state (the `set_value`/plain-ctor path) and is empty for the other two;
  `get_compile_time_constant()` returns the value only for the compile-time
  constant and is empty for the runtime states. The plain constructor /
  `set_value` land in runtime-with-default (`get_pass_by_value()` == value),
  matching cuDNN's plain scalar constructor. `get_pass_by_value()` returns
  the *value* variant (not a `bool`), the breaking change from today's API.

- **Version floor.** A graph with a user-supplied tensor elevates the
  required version to `1.2.0`; plugins reporting `< 1.2.0` (including the
  `1.0.0` no-symbol fallback) are dropped from the applicable set; a graph
  with no qualifying plugin returns "no applicable engines." A graph whose
  scalars all carry a baked value stays baseline `1.0.0` and is served
  unchanged. Serialized/deserialized path: a `1.2.0` fake plugin that
  persists its runtime pass-by-value UID set into its `plugin_payload`
  serializes a user-supplied plan; after deserialize-and-execute (no op graph
  available) the host scalar is still read correctly, proving host-scalar
  identity survives via the payload. A plugin that cannot interpret a newer
  payload version rejects it at deserialize.

- **Execute-time filter + warning.** Supplying a variant-pack value for a
  defaulted (value-carrying) UID drops the override, **emits a warning, and
  executes with the baked default** — assert both the warning fires and that
  the default value (not the attempted override) reached the provider
  ([§4.8](#48-execute-time-variant-pack-filter)).

- **User-supplied delivery.** A user-supplied host scalar placed in the
  variant pack reaches the provider's `device_buffers` slot and **equals**
  what the caller supplied (end-to-end).

- **Validation.** All three by-value states validate cleanly, and a
  required scalar in any of the three passes `validateScalarParameter`. The
  virtual-exclusion rules reject `virtual && is_runtime_pass_by_value` and
  `virtual && value present` ([§4.7](#47-frontend-validation)).

- **Serialization parity.** A graph serialized without the feature loads
  in a feature-aware runtime with the new flag `false` and is served by a
  baseline plugin unchanged.

---

## 10. Glossary

- **Pass-by-value tensor**: a host-side scalar operand (e.g. epsilon,
  alpha, beta, SDPA scale) carried as a single-element tensor that is
  either runtime pass-by-value (`is_runtime_pass_by_value == true`) or
  carries a stored value. The umbrella term covering the three by-value
  states below.
- **`ScalarType`**: the frontend enum
  `enum class ScalarType { RUNTIME_PARAM, COMPILE_TIME_CONST };`
  selecting a value-carrying tensor's state at construction, mirroring
  cuDNN-frontend
  ([`graph_properties.h:42-45`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_properties.h#L42-L45)).
- **`RUNTIME_PARAM`**: the `ScalarType` alternative producing the
  runtime-with-default state — a value-carrying runtime scalar whose
  stored value is a **default** that a variant-pack entry may override
  ([§4.8](#48-execute-time-variant-pack-filter)).
- **`COMPILE_TIME_CONST`**: the `ScalarType` alternative producing the
  compile-time constant — a value baked into the op-graph flatbuffer,
  never overridable.
- **Runtime, user-supplied** (`is_runtime_pass_by_value == true`,
  no value): created by `set_as_runtime_parameter()` (or
  `set_is_pass_by_value(true)` with no value); the user supplies the host
  pointer in the variant pack. The only state requiring plugin floor
  `1.2.0`.
- **Runtime with default** (`is_runtime_pass_by_value == true`,
  value present): created by the plain scalar constructor
  `TensorAttributes(v)`, `set_value(v)`,
  `TensorAttributes(v, ScalarType::RUNTIME_PARAM)`, or
  `graph.tensor(v, ScalarType::RUNTIME_PARAM)`; the value is baked in
  `HIPDNN_ATTR_TENSOR_VALUE_EXT` as a default and read from there, so
  baseline plugins serve it unchanged (`1.0.0`). A variant-pack override
  is filtered and warned today ([§4.8](#48-execute-time-variant-pack-filter)).
- **Compile-time constant** (`is_runtime_pass_by_value == false`,
  value present): created by `set_compile_time_constant(v)`,
  `TensorAttributes(v, ScalarType::COMPILE_TIME_CONST)`, or
  `graph.tensor(v, ScalarType::COMPILE_TIME_CONST)`; the value is frozen
  into the op-graph flatbuffer via `HIPDNN_ATTR_TENSOR_VALUE_EXT` and read
  from it, exactly as before this RFC. Imposes no version floor (baseline
  `1.0.0`). The only mode hipDNN supported before this RFC.
- **`is_runtime_pass_by_value`**: the per-tensor flatbuffer boolean
  (backend attribute `HIPDNN_ATTR_TENSOR_IS_RUNTIME_PASS_BY_VALUE`, ID
  1307), true for the two runtime states and false for the compile-time
  constant and ordinary tensors. Combined with value presence it selects
  the state (2-bit encoding, [§4.9](#49-state-reference)).
- **Execute-time variant-pack filter**: the delivery mechanism by which
  `Graph::execute()` forwards to the variant pack only the pure
  user-supplied UIDs, filtering out any UID whose tensor carries a
  baked value; a user-supplied value for a defaulted UID is
  **warned and dropped**, mirroring cuDNN's `variant_pack_uids` set
  ([`graph_interface.h:190-212`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_interface.h#L190-L212),
  [`graph_interface.h:2858-2859`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_interface.h#L2858-L2859)).
  See [§4.8](#48-execute-time-variant-pack-filter).
- **Variant pack**: the runtime-only carrier of per-execution payload
  (data pointers, unique IDs, workspace). New in this RFC: a runtime
  pass-by-value tensor's `uid → void*` entry is a *host* pointer to the
  scalar rather than a device pointer. The variant pack has no flatbuffer
  schema and is never serialized.
- **Feature predicate (derived)**: the backend's runtime-pass-by-value
  feature signal, derived by scanning the op graph for any tensor with
  `is_runtime_pass_by_value == true` **and no baked value** (the
  user-supplied state). The value-carrying states do not raise the signal.
- **Supported plugin SDK API version**: a per-plugin declaration of the
  Plugin SDK API version the plugin was built against, reported via
  `hipdnnPluginGetApiVersion(const char**)` as a `"MAJOR.MINOR.PATCH"`
  string and parsed with `hipdnn_data_sdk::utilities::Version`. Plugins
  that do not export the symbol fall back to `"1.0.0"`.
- **Required plugin SDK API version**: the per-graph minimum the backend
  computes from the features a graph uses; `1.2.0` for a runtime
  user-supplied tensor. A plugin stays in a graph's applicable set
  only when its supported version is `>=` the graph's required version.
- **Version-only filtering**: the applicability model used by
  this RFC, in which provider eligibility is decided by reported API
  version alone, with no per-symbol predicate, because the feature adds
  no new plugin entry point.
