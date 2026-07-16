# hipDNN: Runtime Pass-By-Value Tensors Design Document

- **Contributors**: Samuel Reeder
- **Status**: Draft

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
   - 2.1 [hipDNN gap](#21-hipdnn-gap)
   - 2.2 [End-user API surface](#22-end-user-api-surface)
   - 2.3 [Constraints](#23-constraints)
3. [Current System Overview](#3-current-system-overview)
4. [Proposed Design](#4-proposed-design)
   - 4.1 [Tensor schema addition](#41-tensor-schema-addition)
   - 4.2 [State reference](#42-state-reference)
   - 4.3 [Frontend surface](#43-frontend-surface)
   - 4.4 [Feature signal (derived)](#44-feature-signal-derived)
   - 4.5 [Execute-time transport](#45-execute-time-transport)
   - 4.6 [Provider contract](#46-provider-contract)
   - 4.7 [Feature detection and version filtering](#47-feature-detection-and-version-filtering)
   - 4.8 [Frontend validation](#48-frontend-validation)
   - 4.9 [Execute-time variant-pack forwarding](#49-execute-time-variant-pack-forwarding)
   - 4.10 [Serialized-graph reader-version guard](#410-serialized-graph-reader-version-guard)
5. [Key Design Decisions](#5-key-design-decisions)
   - 5.1 [cuDNN API-surface parity](#51-cudnn-api-surface-parity)
   - 5.2 [Reuse the variant-pack pointer map](#52-reuse-the-variant-pack-pointer-map)
   - 5.3 [Derive feature detection from the tensor schema](#53-derive-feature-detection-from-the-tensor-schema)
   - 5.4 [Version-only filtering](#54-version-only-filtering)
   - 5.5 [Deserialized-plan support via the provider payload](#55-deserialized-plan-support-via-the-provider-payload)
   - 5.6 [No `is_compile_time_constant` flag](#56-no-is_compile_time_constant-flag)
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

The rollout is additive at the schema and ABI level — no new backend entry, plugin symbol, or variant-pack attribute — with one source-level breaking change, below. The new surface consists of:

- A `ScalarType` enum and cuDNN-named constructors, setters, and getters
  on the tensor ([§4.3](#43-frontend-surface)) mirroring
  cuDNN-frontend, with one deliberate divergence: the plain constructor and
  `set_value` bake a compile-time constant rather than matching cuDNN's
  runtime-with-default plain scalar, so existing callers keep their `1.0.0`
  floor unchanged ([§5.1](#51-cudnn-api-surface-parity)).
- **One** defaulted boolean appended to the per-tensor flatbuffer schema
  (`is_runtime_pass_by_value`); combined with value presence it selects the
  state (a 2-bit encoding, [§4.2](#42-state-reference)). The feature signal
  is derived s.t. a graph needs runtime pass-by-value iff some tensor has it
  enabled ([§4.4](#44-feature-signal-derived)).
- One **breaking change**, for cuDNN parity: `get_pass_by_value()` returns
  the value (not `bool`), its former bool-predicate role moving to
  `get_is_pass_by_value()` ([§4.3](#43-frontend-surface)).
- A `1.2.0` plugin-SDK floor whenever a tensor is runtime pass-by-value
  (`set_as_runtime_parameter()`, `set_is_pass_by_value(true)`, or the
  `(scalar, ScalarType::RUNTIME_PARAM)` constructor/setter). A compile-time
  constant — including the plain constructor and `set_value`, which now bake
  one by default — stays on the baseline `1.0.0`.
- Version-only per-graph provider filtering.

There is **no new public backend C-API entry**, **no new plugin SDK
symbol**, and **no new variant-pack attribute**. A runtime user-supplied
scalar value reuses the existing `uid → void*` variant-pack map: its entry
is a *host* pointer to the scalar, delivered to the provider through the
existing `hipdnnEnginePluginExecuteOpGraph` device-buffer array.

**Binary-compatibility scope.** The backend computes the minimum plugin API
version each graph requires from the features it uses. Graphs that do not use
runtime pass-by-value impose no new version requirement and are served by
existing plugins unchanged; a graph that does requires plugins reporting
`>= 1.2.0`, and older plugins are filtered out of the applicable set before
they are asked about the graph, so a legacy plugin can never silently
mis-serve one. See [§4.7](#47-feature-detection-and-version-filtering) for
the full versioning model.

---

## 2. Problem Statement

### 2.1 hipDNN gap

hipDNN bakes scalar pass-by-value values into the operation graph at build
time: `set_value` stores the scalar, and it is frozen into the compiled plan,
with nothing re-reading it at execute time (the build path is detailed in
[§3](#3-current-system-overview)).

Serving N values for the same scalar today therefore requires N
distinct compiled graphs and N cached execution plans. The backend
already anticipates this gap: `TensorDescriptor::finalize()`
([`backend/src/descriptors/TensorDescriptor.cpp`](https://github.com/ROCm/rocm-libraries/blob/ce7ea204012bd0e0013485b919f86b7f071c6aa2/projects/hipdnn/backend/src/descriptors/TensorDescriptor.cpp)) carries the comment
"Pass-by-value tensors are currently required to supply a value at
descriptor creation time. In the future, pass-by-value tensors may
also support setting values through variant packs." This RFC realizes
that future direction.

### 2.2 End-user API surface

The desired end-user surface keeps the same execute API and mirrors
cuDNN-frontend. A scalar operand is created in one of the by-value
states ([§4.2](#42-state-reference)):

```cpp
// compile-time constant — value baked, never overridable, no version elevation.
// The plain ctor and set_value produce this state (a deliberate divergence
// from cuDNN, whose plain scalar constructor is runtime-with-default;
// see 5.1); use ScalarType::RUNTIME_PARAM or set_as_runtime_parameter() for
// a runtime-with-default or user-supplied scalar instead.
auto c1 = graph.tensor(TensorAttributes(0.125f));  // plain ctor → compile-time constant
auto c2 = graph.tensor(0.125f, ScalarType::COMPILE_TIME_CONST);
c2->set_value(0.125f);                             // set_value → compile-time constant
c2->set_compile_time_constant(0.125f);             // equivalent, explicit spelling
// getters: get_compile_time_constant() -> value; get_is_pass_by_value() -> true.

// runtime with default — the (scalar, ScalarType::RUNTIME_PARAM) constructor
// produces this state, matching cuDNN's plain-scalar semantics; the value is
// a baked default. The override cannot be supplied via the variant pack today
// (an attempted override errors); a future release may honor it. Needs 1.2.0.
// For a value that never changes prefer the plain ctor / set_value (baseline
// 1.0.0).
auto s1 = graph.tensor(0.125f, ScalarType::RUNTIME_PARAM);
// getters: get_pass_by_value() -> value; get_compile_time_constant() -> empty;
// get_is_pass_by_value() -> true.

// runtime, user-supplied — value supplied at execute, not baked.
auto scale = graph.tensor(...);
scale->set_as_runtime_parameter();
// getters: get_pass_by_value() -> empty; get_compile_time_constant() -> empty;
// get_is_pass_by_value() -> true. set_is_pass_by_value(true) on a valueless
// tensor is the equivalent setter path.

// at execute: the value is a HOST pointer in the variant pack, keyed by uid.
float scaleValue = 0.125f;
variantPack[scale->get_uid()] = &scaleValue;
graph.execute(handle, variantPack, workspace);
```

cuDNN-frontend similarily
delivers its runtime value with `extend_tensor_map_with_pass_by_value_tensors_`
([`graph_interface.h:190-212`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_interface.h#L190-L212)),
which emplaces a *host* pointer to the scalar into the same
`std::unordered_map<int64_t, void*>` used for device buffers.

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
   ([§4.9](#49-execute-time-variant-pack-forwarding)). The graph descriptor stays
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
([`frontend/include/hipdnn_frontend/attributes/TensorAttributes.hpp:97`](https://github.com/ROCm/rocm-libraries/blob/ce7ea204012bd0e0013485b919f86b7f071c6aa2/projects/hipdnn/frontend/include/hipdnn_frontend/attributes/TensorAttributes.hpp#L97))
returns `!std::holds_alternative<std::monostate>(_value)`, i.e. "a value
has been set." The backend exposes a *read-only*
`HIPDNN_ATTR_TENSOR_IS_BY_VALUE` (1307) derived from
`_data.value.type != NONE`. At execute time the variant pack carries only
unique IDs, data pointers, intermediates, and workspace (plus [RFC 0008](0008_OverridableTensorShapesDesign.md)'s
override attributes) — **no scalar value reaches a provider through the
variant pack today.**

---

## 4. Proposed Design

### 4.1 Tensor schema addition

This RFC adds a single per-tensor flag, `is_runtime_pass_by_value`, to the
tensor schema. It is persisted so a provider reading the serialized op graph
can identify runtime pass-by-value scalars and distinguish them from baked
constants. The flatbuffer `TensorAttributes` table gains a defaulted
boolean appended as its last field:

```
is_runtime_pass_by_value: bool = false;
```

This is the append-only, defaulted-field pattern per [RFC 0005](0005_Versioning.md):
a pre-feature graph deserialized in a runtime that understands the field
reads `false` on every tensor.

The reverse direction — an old core reading a *new* graph — is guarded by
the reader-version stamp in
[§4.10](#410-serialized-graph-reader-version-guard).

The scalar itself keeps using the table's existing `value: TensorValue` union (`tensor_attributes.fbs:61`).

The flag round-trips through descriptor
pack/unpack alongside the `value`, so
`TensorAttributes` methods
are correct after graph deserialize. cuDNN keeps its analogous
`pass_by_values` in its FE JSON across serialize/deserialize
([`graph_interface.h:1588-1593`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_interface.h#L1588-L1593),
[`graph_interface.h:1666-1673`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_interface.h#L1666-L1673));
hipDNN keeps it in the tensor flatbuffer instead.

**Backend attributes.** `HIPDNN_ATTR_TENSOR_VALUE_EXT` (1306) is kept as-is;
`HIPDNN_ATTR_TENSOR_CONSTANT_VALUE` is added as an alias for it (same integer
`1306`, no wire change) to ease porting cuDNN code that queries the constant
`HIPDNN_ATTR_TENSOR_IS_BY_VALUE` (1307) is **kept** but is now
read-only and **derived** as **value-presence** (`value.type != NONE`) — i.e.
"does this tensor carry a baked scalar readable via `VALUE_EXT`." This is the
back-compatible C-API meaning: a caller querying 1307 to decide whether to read
`VALUE_EXT` still gets the right answer, and a pure runtime user-supplied tensor
(flag set, no value) reads `false`. A new
integer `HIPDNN_ATTR_TENSOR_IS_RUNTIME_PASS_BY_VALUE_EXT` (1308) is true only for
the runtime pass-by-value states. Only the runtime bit is stored; 1307 and the
has-constant query are derived. The "any by-value" umbrella
(`value present || runtime flag`, true for every by-value state) is **not** 1307;
it lives only at the wrapper's `isByValue()` query (below), which a direct
backend-API caller does not see. The two therefore differ by design for a pure
runtime user-supplied tensor: C-API 1307 is `false` (no baked value), wrapper
`isByValue()` is `true` (it is a by-value tensor).

**Deserialize invariant.** The value is read from the union whenever it is
present (`value.type != NONE`), and the flag is read
independently. This is what makes a legacy baked scalar (value present, flag
absent → `false`) deserialize correctly as a compile-time constant. The
descriptor unpack gate in
[`DescriptorUnpackHelpers.hpp`](https://github.com/ROCm/rocm-libraries/blob/ce7ea204012bd0e0013485b919f86b7f071c6aa2/projects/hipdnn/frontend/include/hipdnn_frontend/detail/DescriptorUnpackHelpers.hpp)
keys the value-read on `IS_BY_VALUE == true`; because 1307 now derives
value-presence, that is exactly a value-presence gate — a pure runtime
user-supplied tensor (1307 `false`) skips the `VALUE_EXT` read, and a tensor
carrying a value reads it.

Providers read the flag through a `isRuntimePassByValue()` accessor added
to the op-graph tensor wrapper
([`ITensorAttributesWrapper`](https://github.com/ROCm/rocm-libraries/blob/ce7ea204012bd0e0013485b919f86b7f071c6aa2/projects/hipdnn/flatbuffers_sdk/include/hipdnn_flatbuffers_sdk/flatbuffer_utilities/TensorAttributesWrapper.hpp)). It is the accessor the provider
example in [§4.6](#46-provider-contract) consumes. For cuDNN porting parity
the wrapper also exposes two **derived** queries — `isByValue()` (the umbrella,
`value present || is_runtime_pass_by_value`, true for every by-value state) and
`hasCompileTimeConstant()` (`!is_runtime_pass_by_value && value present`) —
both computed from the flag and value presence, not stored, so no second
discriminator can desync from the flag.

### 4.2 State reference

A pass-by-value tensor is in one of the by-value states below, selected by
the stored runtime flag `is_runtime_pass_by_value` (default `false`) plus
value presence over a single re-used `ValueVariant _value` member — a 2-bit
orthogonal encoding ([§5.6](#56-no-is_compile_time_constant-flag)):

| State | Creation (frontend) | runtime flag | `value` | `get_is_pass_by_value()` | `get_pass_by_value()` | `get_compile_time_constant()` | Delivery | Provider floor |
|---|---|---|---|---|---|---|---|---|
| **Runtime, user-supplied** | `set_as_runtime_parameter()`; or `set_is_pass_by_value(true)` with no value | true | ∅ | true | ∅ | ∅ | user supplies host ptr in variant pack | `1.2.0` |
| **Runtime with default** (future override) | `TensorAttributes(v, ScalarType::RUNTIME_PARAM)` | true | v | true | v | ∅ | baked default in `VALUE_EXT`; override inert today ([§4.9](#49-execute-time-variant-pack-forwarding)) | `1.2.0` |
| **Compile-time constant** | `TensorAttributes(v)` (plain ctor); `set_value(v)`; `set_compile_time_constant(v)`; `TensorAttributes(v, ScalarType::COMPILE_TIME_CONST)` | false | v | true | ∅ | v | baked in op-graph flatbuffer, read via existing path | baseline `1.0.0` |

(∅ = empty / `std::monostate`.)

**Runtime-with-default is one state, read differently at each layer.** The
backend/provider contract treats it as override-capable — a variant-pack
value, if present, overrides the baked default ([§4.6](#46-provider-contract)).
The frontend does not yet exercise that: a variant-pack override for a scalar
carrying a value is forwarded to the provider but **inert**
([§4.9](#49-execute-time-variant-pack-forwarding)) — the provider reads the baked
value from the flatbuffer and ignores the `device_buffers` slot for a
value-carrying tensor, so only the baked default is used. It is thus reachable
only by a direct backend-API caller today; the frontend may expose the
override in a future release. Wherever behavior is described below, "frontend"
and "backend" qualify which layer is meant.

### 4.3 Frontend surface

Pass-by-value tensors mirror cuDNN-frontend's public surface 1:1, with the
deliberate hipDNN divergences in [§5.1](#51-cudnn-api-surface-parity). The
cuDNN-named constructors, setters, and getters below map onto the runtime
flag + value encoding of [§4.2](#42-state-reference), the getters deriving
their results from those two bits. **Every cuDNN-mirrored method is preserved
verbatim**.

**Enum.** A `ScalarType` selects a value-carrying tensor's state at
construction, mirroring cuDNN
([`graph_properties.h:42-45`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_properties.h#L42-L45)):

```cpp
namespace hipdnn_frontend::graph {
enum class ScalarType { RUNTIME_PARAM, COMPILE_TIME_CONST };
}
```

**Constructors.**

- `TensorAttributes(const T& scalar)` → the **compile-time constant** state,
  delegating to `set_value`. **Deliberately diverges from cuDNN**, whose plain
  scalar constructor is runtime pass-by-value ([§5.1](#51-cudnn-api-surface-parity));
  hipDNN keeps the plain constructor on the pre-existing baseline-`1.0.0`
  baked-constant behavior so it is source- and version-compatible with
  pre-RFC-0016 callers. Use `(scalar, ScalarType::RUNTIME_PARAM)` for a
  runtime-with-default scalar.
- `TensorAttributes(const T& scalar, ScalarType type)`: `RUNTIME_PARAM`
  → **runtime-with-default**, `COMPILE_TIME_CONST` → **compile-time constant**
  ([`graph_properties.h:200-271`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_properties.h#L200-L271)).

**Setters**.

- `set_value<T>(v)` → **compile-time constant** (value baked, runtime flag
  clear); floor stays baseline `1.0.0`. **Deliberately diverges from cuDNN**,
  which has no `set_value` and whose closest equivalent (the plain
  constructor) is runtime pass-by-value; hipDNN keeps `set_value` on its
  pre-existing baked-constant behavior for source compatibility
  ([§5.1](#51-cudnn-api-surface-parity)).
- `set_compile_time_constant(const pass_by_values_t& v)` → compile-time
  constant; matches cuDNN's signature
  ([`graph_properties.h:384-392`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_properties.h#L384-L392)).
  Delegates to the typed `set_value` (via `std::visit`) so the per-scalar data
  type is derived from the active alternative (a `std::monostate` variant is
  a no-op); now a thin alias over `set_value` with an ergonomic
  `pass_by_values_t`-variant signature, since `set_value` itself bakes a
  compile-time constant.
- `set_as_runtime_parameter()` → runtime user-supplied: sets the runtime
  flag true and **clears** any prior value (a deliberate divergence from
  cuDNN, whose `set_as_runtime_parameter` leaves `pass_by_value` set,
  [`graph_properties.h:394-400`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_properties.h#L394-L400)).
- `set_is_pass_by_value(bool)` — retained (cuDNN source:
  [`graph_properties.h:367-371`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_properties.h#L367-L371));
  sets **only** the runtime flag and leaves any stored value untouched. So `true` on a value-carrying
  tensor yields runtime-with-default; `true` with no value yields user-supplied.

**Getters** (derived from the runtime flag + value presence).

- `get_pass_by_value()` returns `std::optional<pass_by_values_t>` (the value
  variant, not `bool`), present iff `runtime flag && value present` (the
  runtime-with-default state), `std::nullopt` otherwise — byte-for-byte cuDNN
  ([`graph_properties.h:357-360`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_properties.h#L357-L360)).
  hipDNN adds a typed convenience wrapper `get_pass_by_value<T>()` →
  `std::optional<T>` (delegates to the primary form; `std::nullopt` when absent
  OR the stored scalar is not a `T`). hipDNN also exposes the ungated raw read
  `get_value_variant()` → `const ValueVariant&` (`std::monostate` = empty),
  which ignores the runtime flag; it is a hipDNN-only accessor with no cuDNN
  equivalent.
- `get_compile_time_constant()` returns `std::optional<pass_by_values_t>` iff
  `!runtime flag && value present` (the compile-time constant), `std::nullopt`
  otherwise ([`graph_properties.h:379-382`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_properties.h#L379-L382));
  hipDNN adds the same typed convenience wrapper `get_compile_time_constant<T>()`
  → `std::optional<T>`.
- `get_is_pass_by_value()` is the derived umbrella predicate =
  `runtime flag || value present` (true for all three by-value states)
  ([`graph_properties.h:362-365`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_properties.h#L362-L365)).
- `get_has_compile_time_constant()` is the derived bool = `!runtime flag &&
  value present`; mirrors cuDNN
  ([`graph_properties.h:374-377`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_properties.h#L374-L377)).
- `get_is_runtime_pass_by_value()` is a hipDNN-only raw accessor returning the
  stored runtime flag verbatim (no value-presence derivation). It has no cuDNN
  equivalent and distinguishes the two runtime states from the compile-time
  constant regardless of whether a default value is baked.

**Breaking change** (from the current hipDNN API, source-level):
`get_pass_by_value()` returns the value instead of `bool`; its former
"is-pass-by-value" bool-predicate role moves to `get_is_pass_by_value()`. The
plain constructor and `set_value` keep their pre-existing compile-time-constant,
baseline-`1.0.0` behavior — no version floor changes for any existing caller
([§6](#6-compatibility-versioning-and-rollback)).

`Graph::tensor(const TensorAttributes&)`
([`Graph.hpp`](https://github.com/ROCm/rocm-libraries/blob/ce7ea204012bd0e0013485b919f86b7f071c6aa2/projects/hipdnn/frontend/include/hipdnn_frontend/Graph.hpp)) is
retained, and `graph.tensor(scalar, ScalarType)` overloads are added (one per supported
scalar type), each delegating to the `TensorAttributes(scalar, ScalarType)`
constructor: `graph.tensor(v, ScalarType::RUNTIME_PARAM)` → runtime-with-default and
`graph.tensor(v, ScalarType::COMPILE_TIME_CONST)` → compile-time. Like cuDNN there is
no bare-scalar `graph.tensor(v)` overload; the plain default is reached
via `graph.tensor(TensorAttributes(v))` (compile-time constant).

A pass-by-value tensor is a single-element host scalar, so the existing
scalar conventions (dims/strides `{1}`) apply.

Unlike override shapes — whose frontend setters
(`set_override_shape_enabled` and the override execute overload) are
compiled only under `#ifdef HIPDNN_ENABLE_SDPA` in
[`Graph.hpp`](https://github.com/ROCm/rocm-libraries/blob/ce7ea204012bd0e0013485b919f86b7f071c6aa2/projects/hipdnn/frontend/include/hipdnn_frontend/Graph.hpp) — the
pass-by-value frontend API is **not** SDPA-gated. Scalar operands such
as epsilon, alpha, and beta are general, not SDPA-specific, so
`set_is_pass_by_value` is always compiled.

### 4.4 Feature signal (derived)

The backend's runtime-pass-by-value feature signal is **derived from the
per-tensor flag** ([§4.1](#41-tensor-schema-addition)): a graph requires
runtime pass-by-value support iff it contains at least one tensor with
`is_runtime_pass_by_value == true` (either runtime state). A compile-time
constant has the flag clear, so a baseline plugin serves it by reading the
baked `VALUE_EXT` — it imposes no floor. The per-tensor flag is the single
source of truth; the feature is derived from it rather than a graph-level
flag ([§5.3](#53-derive-feature-detection-from-the-tensor-schema)).

The backend computes the signal with a `readIsRuntimePassByValueEnabled`
helper that scans the serialized op graph at applicability time; a graph
with no runtime-flagged tensor yields `false` and needs no version
elevation. This is the single signal feature detection consumes
([§4.7](#47-feature-detection-and-version-filtering)), and it is exactly the
runtime-slot discriminator `isRuntimePassByValue()` ([§4.6](#46-provider-contract)).

### 4.5 Execute-time transport

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
```

A compile-time constant or runtime-with-default carries a baked value and
need **not** be placed in the variant pack; supplying one for such a UID is
inert — the provider ignores it and uses the baked value
([§4.9](#49-execute-time-variant-pack-forwarding)).

### 4.6 Provider contract

A provider reporting plugin SDK API version `>= 1.2.0` must, for any tensor
matching the runtime discriminator `isRuntimePassByValue()`, read the scalar from that
UID's slot in the `device_buffers` array **as a host pointer** whenever a
variant-pack entry is present, using it to override any baked default; when
no variant-pack entry is present it uses the baked default (`VALUE_EXT`). A
compile-time constant (`isRuntimePassByValue() == false`) is read from the
op-graph flatbuffer as today.

**Capability assertion.** Reporting `>= 1.2.0` asserts runtime
pass-by-value capability. A provider that reports `>= 1.2.0` but cannot
serve a particular runtime pass-by-value graph MUST decline it rather than mis-serve it by
reading the host-scalar slot as a device pointer.

**Override capability.** For any `isRuntimePassByValue()` tensor the contract
is that a variant-pack value, when present, overrides a baked default. An
engine that folds the scalar into a compiled kernel and cannot re-read it
from the variant-pack pointer at execute therefore MUST decline any graph
with `isRuntimePassByValue() == true`, **even when the flatbuffer carries a
value** — it cannot guarantee the required override, so serving the graph
would silently ignore a user's runtime value.

**Defaulted-tensor caveat.** A runtime-with-default tensor carries a baked
value read as its default. The frontend and backend diverge here: through
`Graph::execute()` the in-tree providers read the baked default and ignore any
`device_buffers` slot for such a UID today
([§4.9](#49-execute-time-variant-pack-forwarding)), so a variant-pack override for
it is inert — while the backend/provider contract above already honors an
override if a provider chooses to read the slot (e.g. from a direct
backend-API caller).
Either way the tensor's flag puts the graph at the `1.2.0` floor. A
compile-time constant is `isRuntimePassByValue() == false` and never appears
in `device_buffers` as a host scalar. Only a user-supplied scalar arrives as
a host pointer through the frontend.

Each `hipdnnPluginDeviceBuffer_t` carries its `uid`, and
[§4.1](#41-tensor-schema-addition) adds the `isRuntimePassByValue()`
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
// seeded default. Today only a pure user-supplied scalar
// arrives here as an override; a runtime-with-default override is inert (the
// provider below reads the baked default and ignores the slot for a
// value-carrying tensor), and a compile-time constant is never runtime — so
// both keep their baked value. Runtime-with-default override could be enabled
// later by a provider reading the slot for such a UID.
for (uint32_t i = 0; i < num_device_buffers; ++i) {
    const hipdnnPluginDeviceBuffer_t& buf = device_buffers[i];
    if (runtimeUids.count(buf.uid)) {
        p.set_scalar(buf.uid, *static_cast<const float*>(buf.ptr));  // host ptr overrides default
    }
    // else: ordinary DEVICE pointer
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
list, and the provider's opaque `plugin_payload` alone. These obligations
apply only to a provider that supports runtime pass-by-value (reports
`1.2.0`) **and implements compiled-plan save/load**; a provider that never
serializes plans has no deserialized path and carries none of them. Such a
provider must therefore:

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

These deserialization obligations are only part of what reporting `1.2.0`
means — they carry the
same trust-the-version contract as reading the host-scalar slot itself —
because hipDNN core performs no version check on the deserialized path
([§4.7](#47-feature-detection-and-version-filtering)). The hipDNN
envelope `version` field versions the plan layout, not the opaque payload
contents, so it cannot stand in for this provider check.

**Pointer lifetime.** The host pointer is valid for the duration of the
execute call only; the provider must not retain or dereference it after
returning, matching the existing device-buffer contract.

### 4.7 Feature detection and version filtering

The backend maps each graph to the minimum plugin API version it
requires and filters plugins against that mapping. The
runtime-pass-by-value input to that mapping is the derived per-tensor
signal from [§4.4](#44-feature-signal-derived) (`readIsRuntimePassByValueEnabled`);
the rest of this section covers how that signal — alongside the override
flag — becomes a version floor.

`computeMinimumPluginApiVersion`
([`backend/src/plugin/EnginePluginResourceManager.cpp:86-99`](https://github.com/ROCm/rocm-libraries/blob/ce7ea204012bd0e0013485b919f86b7f071c6aa2/projects/hipdnn/backend/src/plugin/EnginePluginResourceManager.cpp#L86-L99)) becomes
feature-aware. Today it maps a single override-shape boolean to either
the baseline `1.0.0` or the override minimum `1.1.0`. It is extended to
also account for the pass-by-value flag and return the **maximum** of the
per-feature minimums:

| Enabled feature(s) | Required plugin API version |
|--------------------|-----------------------------|
| None | `1.0.0` (baseline) |
| Override shapes only | `1.1.0` |
| Runtime pass-by-value — any tensor `is_runtime_pass_by_value == true` (user-supplied or runtime-with-default), with or without override | `1.2.0` |

A new version constant is added in
[`plugin_sdk/include/hipdnn_plugin_sdk/PluginVersionConstants.hpp`](https://github.com/ROCm/rocm-libraries/blob/ce7ea204012bd0e0013485b919f86b7f071c6aa2/projects/hipdnn/plugin_sdk/include/hipdnn_plugin_sdk/PluginVersionConstants.hpp):

```cpp
inline constexpr std::string_view K_PASS_BY_VALUE_MIN_API_VERSION = "1.2.0";
```

and the canonical ABI macros in
[`plugin_sdk/include/hipdnn_plugin_sdk/engine_api_version.h`](https://github.com/ROCm/rocm-libraries/blob/ce7ea204012bd0e0013485b919f86b7f071c6aa2/projects/hipdnn/plugin_sdk/include/hipdnn_plugin_sdk/engine_api_version.h) bump
`HIPDNN_ENGINE_API_VERSION_MINOR` from `1` to `2`
(`HIPDNN_ENGINE_API_VERSION = "1.2.0"`).

**Filtering is version-only.** `getApplicableEngineIds`
([`EnginePluginResourceManager.cpp:341-407`](https://github.com/ROCm/rocm-libraries/blob/ce7ea204012bd0e0013485b919f86b7f071c6aa2/projects/hipdnn/backend/src/plugin/EnginePluginResourceManager.cpp#L341-L407)) already skips any plugin
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
core. A serialized execution plan
re-binds by the baked
`engineId` with no version re-filter, but its runtime pass-by-value
state lives in the provider's opaque payload, which the provider
versions and validates on deserialize; that path is covered by the
provider contract, not a core gate
([§5.5](#55-deserialized-plan-support-via-the-provider-payload)).

### 4.8 Frontend validation

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
compile-time constant.

**Post-build immutability.** The compiled plan is frozen at build
(`backendFinalize`, [§2.3](#23-constraints)): a scalar's baked value and
pass-by-value flag are captured in the backend `TensorDescriptor` and the
serialized plan, and nothing re-reads the frontend `TensorAttributes`
after build. Post-build calls to the value/mode setters are therefore inert:
they mutate only the detached frontend object and cannot alter the compiled
plan, so such a call logs a warning. The only way to vary a user-supplied
scalar after build is the variant pack at execute
([§4.5](#45-execute-time-transport)).

`detail::validateScalarParameter`
([`frontend/include/hipdnn_frontend/node/detail/Utilities.hpp`](https://github.com/ROCm/rocm-libraries/blob/ce7ea204012bd0e0013485b919f86b7f071c6aa2/projects/hipdnn/frontend/include/hipdnn_frontend/node/detail/Utilities.hpp), ~line
475) today gates required scalar inputs (epsilon, SDPA scale, etc.) on the
pre-RFC bool `get_pass_by_value()` ("a value has been set"). Since that name
is repurposed to return the value, the check moves to the derived umbrella
`get_is_pass_by_value()`, accepting any variant of pass-by-value tensors.

### 4.9 Execute-time variant-pack forwarding

A user-supplied scalar's value reaches the provider as a host pointer
in the variant pack at `Graph::execute()` ([§4.5](#45-execute-time-transport)).
A runtime-with-default or compile-time scalar instead carries its
value baked in the tensor flatbuffer (`VALUE_EXT`).

**No frontend filter; the provider is the uniform backstop.** `Graph::execute()`
forwards the caller's entire variant pack to the backend unchanged — it does
**not** filter out UIDs that carry a baked value, nor does it pre-validate that
a pure user-supplied scalar's UID is present. This matches how hipDNN treats
**every** tensor: the frontend performs no per-tensor variant-pack
presence-validation for ordinary inputs/outputs on the execute path either
(`populateBaseVariantPackDescriptor` copies whatever the caller supplied), and
the backend `VariantDescriptor::finalize()` checks only pack self-consistency
(pointer/UID counts match, non-empty), not graph-required-UID coverage. The
single enforcement point for "is this required scalar present?" is the
provider's `findDeviceBuffer(uid)` ([§4.6](#46-provider-contract)), which
throws `HIPDNN_PLUGIN_STATUS_INVALID_VALUE` (surfaced as `INVALID_VALUE`) when
an engine reads a UID the pack lacks — identical treatment to any other missing
input/output tensor. A pure user-supplied scalar whose UID is omitted is thus
caught by the provider at execute, not by the frontend.

**Baked-value UIDs are inert, not rejected.** Supplying a variant-pack entry
for a UID that carries a baked value (compile-time constant or
runtime-with-default) is not an error: the provider reads such a tensor's value
from the op-graph flatbuffer and ignores its `device_buffers` slot
([§4.6](#46-provider-contract)), so the entry is silently inert and the baked
value wins. The frontend override of a value-carrying scalar is therefore
**deferred, not rejected** — `Graph::execute()` never has to reject it because
delivering it changes nothing. A runtime-with-default tensor still requires
`1.2.0` (its flag is set). The compiled-plan path reconstructs no per-tensor
attributes (see below), which is consistent with this model: the core never
needed those attributes to police the pack.

**Rationale for no frontend filter.** An earlier draft had `Graph::execute()`
gather the op-graph tensors and reject a missing pure user-supplied scalar
eagerly. It was dropped: (1) it duplicated the provider's `findDeviceBuffer`
guarantee, producing the same `INVALID_VALUE` one layer earlier; (2) it was
asymmetric — no equivalent frontend check exists for ordinary required tensors,
so pass-by-value scalars were being held to a stricter standard than every
other tensor for no contract reason; and (3) it added a per-execute
full-subtree tensor gather on the hot path guarding a case the provider already
covers. Leaving validation to the provider keeps one enforcement point and one
error for all missing-tensor cases.

**Compiled-plan path.** `deserializeBackendPlan` reconstructs no per-tensor
attributes ([§5.5](#55-deserialized-plan-support-via-the-provider-payload);
[`ExecutionPlanDescriptor.cpp:425-486`](https://github.com/ROCm/rocm-libraries/blob/ce7ea204012bd0e0013485b919f86b7f071c6aa2/projects/hipdnn/backend/src/descriptors/ExecutionPlanDescriptor.cpp#L425-L486)),
so the core cannot identify which UIDs carry a baked value; a variant-pack
entry is accepted as an ordinary binding (a provider MAY reject an unexpected
override from its own payload; the core does not). This is the same
no-frontend-filter model as the fresh-build path above — the provider remains
the sole enforcement point. A runtime-with-default tensor round-tripped through
`to_compiled_plan_binary` also loses its baked default at the hipDNN layer and
degrades to user-supplied semantics unless the provider persists the value in
its `plugin_payload` ([§4.6](#46-provider-contract)) and restores it on
deserialize. Graph serialization is unaffected — the baked `VALUE_EXT`
survives a graph serialize/deserialize; only the compiled-plan path drops it,
and cuDNN avoids even that by bundling `pass_by_values` with the plan JSON.
This limitation is explicit and covered by test ([§9](#9-testing-plan)).

**Usage guidance.** `set_compile_time_constant` — and equivalently the plain
constructor / `set_value`, which bake the same state — fix a build-known
value at baseline `1.0.0`: the choice for a value that never changes. The
tagged `(scalar, ScalarType::RUNTIME_PARAM)` constructor/setter produces a
runtime-with-default scalar (a baked default at the `1.2.0` floor whose
override is deferred today), so prefer the compile-time-constant path for a
fixed value to stay on the baseline. `set_as_runtime_parameter` supplies the
value at execute with no baked default.

### 4.10 Serialized-graph reader-version guard

Appending `is_runtime_pass_by_value` to `TensorAttributes` is
forward-compatible ([§4.1](#41-tensor-schema-addition)): a *new* core reading
an *old* graph reads the absent field as `false`. The reverse is not safe on
its own — an *old* core reading a *new* graph silently drops the unknown
field and reads `false`, so a runtime pass-by-value tensor is mis-read as an
ordinary tensor and its host scalar is dereferenced as a device pointer.
Plugin version filtering ([§4.7](#47-feature-detection-and-version-filtering))
does not cover this: it gates the *plugin*, not the *core* that deserializes
the graph flatbuffer.

The `Graph` table therefore gains a reader-version guard (`graph.fbs`):

```
min_reader_version: uint = 0;
```

The serializer sets it to `1` — the value this RFC introduces — for any graph
containing a runtime pass-by-value tensor (the same per-tensor signal as
[§4.4](#44-feature-signal-derived)); a graph with only compile-time constants
or ordinary tensors leaves it `0` and stays readable by every existing core.

The guard is **two numbers**: the graph's stamped `min_reader_version` and a
`K_GRAPH_READER_VERSION` constant the backend is built with (its highest
supported stamp, `1` as of this RFC). On deserialize, the core rejects any
graph whose `min_reader_version` exceeds its `K_GRAPH_READER_VERSION`,
returning a clean error before interpreting any tensor — never silently
mis-reading a gated field. This mirrors the existing plan-version check with `PLAN_SERIALIZATION_VERSION`.

**Bootstrapping limit.** A core predating this field cannot read it, so it
cannot self-guard against this first stamped graph; that residual exposure is
inherent to introducing the guard. Every core from this release forward
rejects a graph stamped newer than it understands, closing the gap for this
and all future gated fields.

## 5. Key Design Decisions

### 5.1 cuDNN API-surface parity

**Decision**: mirror cuDNN-frontend's pass-by-value surface 1:1 — the
`ScalarType` enum, both constructors, the cuDNN-named setters/getters, and
the `graph.tensor(scalar, ScalarType)` factory, each enumerated with its
cuDNN citation in [§4.3](#43-frontend-surface) — with the divergences
below.

**Rationale**: adopting cuDNN's names and shapes lets cuDNN users port
with no concept translation, and the 2-bit model
([§4.2](#42-state-reference)) covers cuDNN's full
fused-constant-vs-execute-time surface
([`graph_properties.h:53-57`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_properties.h#L53-L57)).
The surface mirrors cuDNN name-for-name for the enum, the tagged
constructor/setters/getters, and the `graph.tensor(scalar, ScalarType)`
factory. It diverges for the plain scalar constructor and `set_value`
(divergence 3, below): source compatibility with hipDNN's pre-RFC-0016
baked-constant behavior was judged more important than matching cuDNN's
plain-scalar default for these two specific entry points.

**Divergences**:

1. hipDNN stores a single `is_runtime_pass_by_value` flag (runtime-only)
   and a single re-used value member, whereas cuDNN stores an
   `is_pass_by_value` umbrella plus a separate `has_compile_time_constant`
   and two value members (`pass_by_value`, `compile_time_constant_value`,
   [`graph_properties.h:118-123`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_properties.h#L118-L123)),
   both `std::optional<pass_by_values_t>` over cuDNN's value-variant alias
   (`std::variant<int64_t, int32_t, half, float, double, nv_bfloat16>`,
   [`graph_properties.h:59`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_properties.h#L59)).
   hipDNN mirrors that alias as `pass_by_values_t` for the setter/getter
   signatures ([§4.3](#43-frontend-surface), [§8](#8-execution-plan)) but folds
   emptiness into the variant via `std::monostate` instead of an outer
   `std::optional`, over hipDNN's own dtype set. The frontend getters are
   derived from the runtime flag + value presence for porting parity.
2. `set_as_runtime_parameter()` **clears** any prior value, whereas cuDNN
   leaves `pass_by_value` set
   ([`graph_properties.h:394-400`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_properties.h#L394-L400)). We could later match this API by collecting an optional default value for the `set_as_runtime_parameter()` route without any added plugin changes.
3. **The plain scalar constructor `TensorAttributes(const T&)` and
   `set_value(v)` bake a compile-time constant** (runtime flag clear),
   whereas cuDNN's plain scalar constructor is runtime pass-by-value
   ([`graph_properties.h:158-198`](https://github.com/NVIDIA/cudnn-frontend/blob/c4ec01a28a26aa57021862de809cc257619f7516/include/cudnn_frontend/graph_properties.h#L158-L198)),
   and cuDNN has no `set_value` at all. hipDNN kept the pre-existing
   ([RFC-0016](0016_RuntimePassByValueTensors.md)-predating) baked-constant
   behavior for these two entry points rather than reaching full cuDNN
   parity, because flipping them to runtime-with-default would silently
   floor every existing caller of the plain constructor / `set_value` — the
   samples, the Python `set_value` binding, and any out-of-tree code — at
   plugin API `1.2.0` with no source change and no compiler diagnostic. Use
   the tagged `(scalar, ScalarType::RUNTIME_PARAM)` constructor for cuDNN's
   plain-scalar semantics.

### 5.2 Reuse the variant-pack pointer map

**Decision**: transport a user-supplied scalar as a host
pointer in the existing `uid → void*` variant-pack map, delivered through
the unchanged `hipdnnEnginePluginExecuteOpGraph` `device_buffers` array. The
caller fills the slot. No new variant-pack attribute and no new plugin
symbol.

**Rationale**: reusing the existing map keeps the plugin boundary unchanged —
nothing is added to `populateBaseVariantPackDescriptor`, the plugin C ABI, or
the 700-799 attribute range, and consumers port with no signature changes.

**Trade-off**: the provider must consult the discriminator
`isRuntimePassByValue()` to know a slot holds a *host* pointer.

### 5.3 Derive feature detection from the tensor schema

**Decision**: detect the feature from the per-tensor flag (any tensor with
`is_runtime_pass_by_value == true`, either runtime state)
rather than a separate graph-level `is_pass_by_value_enabled` flag.

**Rationale**: the per-tensor flag is already the mandatory discriminator
separating a runtime host-scalar slot from a device buffer or a baked
constant ([§4.6](#46-provider-contract)), so it is the single source of
truth. A graph-level flag would be a denormalized cache that can disagree —
and unsafely: a raw backend C-API caller could set the tensor attribute but
not the graph one, leaving the filter reading `false` while a runtime
tensor exists, so a sub-`1.2.0` plugin reads a host pointer as a device
pointer. Deriving makes that desync unrepresentable. [RFC 0008](0008_OverridableTensorShapesDesign.md)
uses a graph-level flag only because override shapes have no per-tensor
field to derive from; runtime pass-by-value does.

**Trade-off**: a one-time `O(tensors)` walk of the already-materialized
serialized graph per applicability query instead of a bool read —
negligible, on a non-hot path
([`EnginePluginResourceManager.cpp:341-407`](https://github.com/ROCm/rocm-libraries/blob/ce7ea204012bd0e0013485b919f86b7f071c6aa2/projects/hipdnn/backend/src/plugin/EnginePluginResourceManager.cpp#L341-L407)).

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
[§4.7](#47-feature-detection-and-version-filtering).

**Rationale**: on the deserialized path the op graph is gone
(`deserializeBackendPlan`, [§4.9](#49-execute-time-variant-pack-forwarding)), so
the only place host-scalar identity can survive is the provider's own
payload, which it already owns and versions — skew safety reduces to that
existing payload-versioning contract. This is exactly RFC 0009's
payload-ownership rule: the envelope omits the op graph, so *"plugins that
need graph-derived data must store it in their own payload"*
([RFC 0009, Envelope Format](0009_CompiledPlanSerialization.md#envelope-format)).

**Trade-off**: deserialized-path safety rests on the provider obeying the
§4.6 payload-versioning contract; core enforces no version floor there,
and the skew failure is memory-unsafe, not merely wrong-result — the same
trust-the-version posture already accepted for the host-pointer read
itself ([§7.1](#71-provider-reports-120-but-mishandles-host-pointers)).

**Retrofit limit**: the skew window closes only when the older
same-`engineId` release already rejected payloads it could not interpret.
A provider that shipped before runtime pass-by-value without defensive
payload versioning cannot be made safe retroactively, so adopting that
discipline is a precondition of shipping runtime pass-by-value
([§4.6](#46-provider-contract), [Step 7](#step-7-provider-adoption)).

### 5.6 No `is_compile_time_constant` flag

**Decision**: encode the by-value state from one stored runtime flag
(`is_runtime_pass_by_value`) plus value presence — four orthogonal
quadrants — rather than adding a second stored `is_compile_time_constant`
flag.

**Rationale**: the runtime flag and value presence are already orthogonal
and together cover every state, so a separate compile-time flag would be
redundant — and it would reintroduce a legacy-graph ambiguity: a pre-feature
baked scalar deserializes with both flags defaulting `false`, which a
two-flag rule set would reject or misclassify. Under the single-flag 2-bit
encoding that same tensor —
`value present, is_runtime_pass_by_value == false` — is unambiguously a
compile-time constant by definition, so a legacy graph round-trips correctly
with no normalization sentinel.

---

## 6. Compatibility, Versioning, and Rollback

**Upgrade path.** Existing plugins (in-tree and out-of-tree) serve
compile-time-constant scalars (`set_compile_time_constant`) unchanged with no
rebuild. A plugin adopts runtime pass-by-value by reading host-scalar slots
for tensors matching `isRuntimePassByValue()` and bumping its reported API
version to `1.2.0`. Any runtime tensor — user-supplied or runtime-with-default
— requires `1.2.0`; version skew is handled by the per-graph filter on the
fresh-build path ([§4.7](#47-feature-detection-and-version-filtering)) and by
provider payload versioning on the deserialized path
([§5.5](#55-deserialized-plan-support-via-the-provider-payload)). Each plugin
migrates asynchronously. The one new schema field is appended and defaulted
`false`, wire-compatible per [RFC 0005](0005_Versioning.md), so a pre-feature
*serialized* graph (baked value, no flag) loads as a compile-time constant
unchanged.

**No breaking change to version floors.** The plain constructor and
`set_value` keep their pre-existing compile-time-constant, baseline-`1.0.0`
behavior ([§5.1](#51-cudnn-api-surface-parity), divergence 3) — no existing
caller's graph gains a new plugin-version requirement from this RFC. Only a
caller that explicitly opts in via `set_as_runtime_parameter()`,
`set_is_pass_by_value(true)`, or the `(scalar, ScalarType::RUNTIME_PARAM)`
constructor/setter floors at `1.2.0`.

**Rollback.** The feature is inert unless a caller creates a runtime
pass-by-value tensor. Reverting a caller to `set_compile_time_constant`
(or simply not opting into `ScalarType::RUNTIME_PARAM` /
`set_as_runtime_parameter()`) restores the baked compile-time path with zero
schema migration: the value is baked, no version is elevated, and
`hipdnn_plugin_sdk::computeMinimumEnginePluginApiVersion` returns the
baseline. No data migration or plan invalidation is required.

---

## 7. Risks

### 7.1 Provider reports 1.2.0 but mishandles host pointers

**Risk**: a provider bumps its reported version to `1.2.0` but reads a
runtime pass-by-value slot as a device pointer (or otherwise mishandles
the host scalar), producing wrong results.

**Mitigation**: this is a plugin implementation bug, not a hipDNN defect —
a provider that reports `1.2.0` asserts it reads host-scalar slots
correctly.

### 7.2 Caller marks a tensor pass-by-value but omits the value at execute

**Risk**: a tensor is marked runtime pass-by-value but the caller does not
insert its UID into the variant-pack map, so the provider reads an
unset or garbage slot.

**Mitigation**: the provider's `findDeviceBuffer(uid)` is the backstop. When an
engine goes to read a runtime pass-by-value scalar whose UID the caller omitted
from the variant pack, `findDeviceBuffer` throws
`HIPDNN_PLUGIN_STATUS_INVALID_VALUE` (surfaced as `INVALID_VALUE`) before any
unset slot is dereferenced — the same treatment as any other missing
input/output tensor ([§4.9](#49-execute-time-variant-pack-forwarding)). The
frontend performs no separate presence-check for this (nor for any other
required tensor on the execute path), and cannot validate the numeric value at
build (it does not yet exist); it only checks at build that the tensor is
structurally a scalar (`validateScalarParameter`). This holds on both the
fresh-build and compiled-plan paths, since the provider guarantee does not
depend on per-tensor attributes surviving deserialize.

### 7.3 Host vs device pointer confusion in the shared map

**Risk**: the same variant-pack map carries both device pointers and
host scalar pointers, so a provider could dereference the wrong kind.

**Mitigation**: on the fresh-build path the discriminator
`isRuntimePassByValue()` is the authoritative marker of a
host-pointer slot; on a deserialized plan the provider relies on the runtime
pass-by-value UID set persisted in its payload
([§5.5](#55-deserialized-plan-support-via-the-provider-payload)). The 2-bit
model ([§4.2](#42-state-reference)) keeps the discriminator unambiguous.
Round-trip and end-to-end tests cover both slot kinds in one graph.

### 7.4 Downgraded provider mis-reads a serialized pass-by-value plan

**Risk**: a compiled pass-by-value plan is serialized against a
`1.2.0` provider, then deserialized where the same `engineId` resolves
to a downgraded `< 1.2.0` build of that provider. Deserialize re-binds
by `engineId` with no core version check
([§4.7](#47-feature-detection-and-version-filtering)), so the older build
receives a payload it did not produce and could read a host scalar as a
device pointer — a memory-unsafe failure.

**Mitigation**: the payload persist/version/reject contract of
[§5.5](#55-deserialized-plan-support-via-the-provider-payload) — a `1.2.0`
provider versions its `plugin_payload` and rejects a version/kind it cannot
interpret before reading any slot — makes a defensively-versioned provider
fail the deserialize cleanly; the build-time version filter does not run on
this path. The risk is latent today: no in-tree provider implements the
RFC 0009 payload round-trip, so the skew window opens only once one does.

---

## 8. Execution Plan

Implementation plan for the work this RFC enables, ordered so the tree
builds and existing tests pass after each step:

### Step 1: Schema field

Append one defaulted boolean, `is_runtime_pass_by_value: bool = false;`, to
the `TensorAttributes` flatbuffer table (re-using the existing `value`
union); regenerate. Add round-trip coverage of the flag plus the value,
including the default-`false` case.

Add `min_reader_version: uint = 0;` to the `Graph` table and a
`K_GRAPH_READER_VERSION` core constant (= `1`)
([§4.10](#410-serialized-graph-reader-version-guard)): the serializer sets the
stamp to `1` when the graph contains any runtime pass-by-value tensor; the
graph deserializer rejects a graph whose `min_reader_version` exceeds
`K_GRAPH_READER_VERSION` (mirroring the plan-version check in
`ExecutionPlanDescriptor.cpp:444-447`). Regenerate; add round-trip plus
reject coverage.

### Step 2: Backend descriptor and enums

`HIPDNN_ATTR_TENSOR_IS_BY_VALUE` (1307) is **kept** but is now
read-only-**derived** as **value-presence** (`value.type != NONE`) — the
back-compatible C-API "has a baked `VALUE_EXT`" meaning — so existing readers
that key `VALUE_EXT` reads on it are unaffected. Add a new
**settable** attribute `HIPDNN_ATTR_TENSOR_IS_RUNTIME_PASS_BY_VALUE_EXT` (1308)
carrying the `is_runtime_pass_by_value` flag (true for the runtime states),
and update the `BackendEnumStringUtils` string. Keep
`HIPDNN_ATTR_TENSOR_VALUE_EXT` (1306) as-is (value union) and add
`HIPDNN_ATTR_TENSOR_CONSTANT_VALUE` as an alias for it (same integer). 1307 and
the has-constant query stay **derived**, not
separately stored; only the runtime bit (1308) is stored. The "any by-value"
umbrella (`value present || runtime flag`) is not 1307 — it is the wrapper-only
`isByValue()` query. The
descriptor unpack value-read gate
([`DescriptorUnpackHelpers.hpp`](https://github.com/ROCm/rocm-libraries/blob/ce7ea204012bd0e0013485b919f86b7f071c6aa2/projects/hipdnn/frontend/include/hipdnn_frontend/detail/DescriptorUnpackHelpers.hpp))
keys the value-read on 1307, which — now that 1307 derives value-presence — is a
**value-presence gate** (read the value whenever the union is present); the
runtime flag is read independently.
Wire both through the existing `TensorDescriptor` get/set-attribute and
pack/unpack paths. Add the `isRuntimePassByValue()` accessor to
`ITensorAttributesWrapper`, plus the derived `isByValue()` and
`hasCompileTimeConstant()` queries (computed from the flag + value presence,
not stored). No operation-graph attribute is added.

### Step 3: Version constant and filter

Add `K_PASS_BY_VALUE_MIN_API_VERSION = "1.2.0"` to
[`PluginVersionConstants.hpp`](https://github.com/ROCm/rocm-libraries/blob/ce7ea204012bd0e0013485b919f86b7f071c6aa2/projects/hipdnn/plugin_sdk/include/hipdnn_plugin_sdk/PluginVersionConstants.hpp); bump [`engine_api_version.h`](https://github.com/ROCm/rocm-libraries/blob/ce7ea204012bd0e0013485b919f86b7f071c6aa2/projects/hipdnn/plugin_sdk/include/hipdnn_plugin_sdk/engine_api_version.h) minor to `2`.
Add a `readIsRuntimePassByValueEnabled(graphDesc)` helper that scans the
serialized op graph and returns `true` iff any tensor has
`is_runtime_pass_by_value == true` (either runtime state; no
graph-level attribute is read; the per-tensor flag is the source of truth).
Extend `computeMinimumPluginApiVersion(bool isOverride, bool isRuntimePassByValue)`
to take the second flag and return the maximum required version (the
applicability-time filter). No execute-time filter is added: `Graph::execute()`
forwards the variant pack unchanged and the provider's `findDeviceBuffer` is the
backstop for a missing pure user-supplied scalar (see Step 5).

### Step 4: Frontend API and validation

Add `enum class ScalarType { RUNTIME_PARAM, COMPILE_TIME_CONST };`.
Add the cuDNN-parity value-variant alias
`using pass_by_values_t = ValueVariant;` on `TensorAttributes` (the existing
`ValueVariant` is `std::variant<std::monostate, double, float, half, bfloat16,
uint8_t, int32_t, int64_t, bool>`), so the setter/getter signatures read the
same as cuDNN. Note the shape difference: hipDNN folds the empty state into
the variant (`std::monostate`), where cuDNN wraps it (`std::optional<pass_by_values_t>`);
the type set is hipDNN's dtypes, not cuDNN's.
Also add the `TensorAttributes(const T&, ScalarType)` constructor; `set_compile_time_constant`
and `set_as_runtime_parameter`; and `get_compile_time_constant` /
`get_has_compile_time_constant` / `get_is_pass_by_value`. **Keep every
cuDNN frontend method name verbatim** — no renames; only their internal
derivation changes (from the single runtime bit + value presence). There is
no stored `is_compile_time_constant` member; `get_compile_time_constant` /
`get_has_compile_time_constant` are derived from `!runtime bit && value
present`. Route
the compile-time constant (runtime bit false, value stored) via the plain
scalar constructor, `set_value`, `set_compile_time_constant`, and
`COMPILE_TIME_CONST`, and route `RUNTIME_PARAM` through the
runtime-with-default state (runtime bit true, value stored). Change
`get_pass_by_value()` to
return the value variant (present iff `runtime bit && value present`, i.e.
runtime-with-default), moving its former bool-predicate role to
`get_is_pass_by_value()`, and migrate every internal value-presence caller
off `get_pass_by_value()` onto `get_value_variant()` — the known callsite is
`createOrFindTensorDesc` in
[`DescriptorHelpers.hpp`](https://github.com/ROCm/rocm-libraries/blob/ce7ea204012bd0e0013485b919f86b7f071c6aa2/projects/hipdnn/frontend/include/hipdnn_frontend/detail/DescriptorHelpers.hpp)
(writes the baked value); `search` for `get_pass_by_value()` to enumerate
the rest. Add the `graph.tensor(scalar, ScalarType)` factory overloads
([§4.3](#43-frontend-surface)). Add the frontend validations
([§4.8](#48-frontend-validation)): the two virtual-exclusion checks and the
relaxed `validateScalarParameter`. No graph-level setter or graph schema
field is added — the runtime-pass-by-value feature signal is derived from
the per-tensor flag ([§4.4](#44-feature-signal-derived)).

### Step 5: Execute-time variant-pack forwarding

`Graph::execute()` forwards the caller's variant pack to the backend unchanged
([§4.9](#49-execute-time-variant-pack-forwarding)): no frontend filter and no
frontend presence-validation. A variant-pack entry for a UID whose tensor
carries a baked value is inert — the provider reads the schema value for the
value-carrying states and ignores the slot. A missing pure user-supplied
scalar is caught by the provider's `findDeviceBuffer(uid)` (`INVALID_VALUE`),
the same backstop as any other missing tensor. On the compiled-plan path no
per-tensor value is reconstructed, so a runtime-with-default tensor degrades to
user-supplied semantics; document the limitation.

### Step 6: Test harness, references, and cross-cutting tests

Extend the shared integration harness
([RFC 0006](0006_PluginAgnosticIntegrationTests.md)) to populate **host**
scalar values into the variant pack keyed by uid
(`variantPack[t.uid()] = &hostVal`) for runtime pass-by-value tensors — it
today only generates device pointers or skips scalars as constants. Update
the CPU and GPU reference implementations to read a pass-by-value scalar
(host value for a runtime tensor, baked value for a constant) so expected
results match. Add fake plugins (one at `1.2.0` consuming the host scalar,
one below it) and the 2-bit state / version-floor / override-inert /
end-to-end matrix ([§9](#9-testing-plan)).

### Step 7: Provider adoption

A shipping provider that adopts runtime pass-by-value **MUST**: read
host-scalar slots for any tensor matching the runtime discriminator
`isRuntimePassByValue()`; persist its runtime pass-by-value
UID set into its serialized `plugin_payload` and restore it on deserialize
([§5.5](#55-deserialized-plan-support-via-the-provider-payload)); version
that payload and reject a payload whose version/kind it cannot interpret
before reading any slot ([§4.6](#46-provider-contract)); and bump its
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
  the state table ([§4.2](#42-state-reference)):
  `get_is_pass_by_value()` is `true` for all three by-value states;
  `get_pass_by_value()` returns the value only for the runtime-with-default
  state (the tagged `RUNTIME_PARAM` path) and is empty
  for the other two;
  `get_compile_time_constant()` returns the value only for the compile-time
  constant (the plain ctor / `set_value` / `set_compile_time_constant` /
  `COMPILE_TIME_CONST` paths)
  and is empty for the runtime states. `get_pass_by_value()` returns the
  *value* variant (not a `bool`), the breaking change from today's API.

- **Version floor.** A graph with any runtime tensor (user-supplied or
  runtime-with-default) elevates the required version to `1.2.0`; plugins
  reporting `< 1.2.0` (including the `1.0.0` no-symbol fallback) are dropped
  from the applicable set; a graph with no qualifying plugin returns "no
  applicable engines." A graph whose scalars are all compile-time constants
  stays baseline `1.0.0` and is served unchanged. Serialized/deserialized
  path: a `1.2.0` fake plugin that persists its runtime pass-by-value UID set
  into its `plugin_payload` serializes a user-supplied plan; after
  deserialize-and-execute (no op graph available) the host scalar is still
  read correctly, proving host-scalar identity survives via the payload. A
  plugin that cannot interpret a newer payload version rejects it at
  deserialize.

- **Non-supporting plugin, both paths.** A single baseline (`< 1.2.0`) plugin,
  against a graph with a runtime pass-by-value tensor, is dropped from the
  applicable set (the graph is declined for it — never dispatched to read a
  host scalar it cannot handle); the *same* plugin, against a graph whose
  scalars are all compile-time constants, remains applicable and serves it
  unchanged. Confirms a plugin that never adopts the feature keeps working for
  compile-time scalars and can never silently mis-serve a runtime one.

- **Override inert for defaulted UID.** Supplying a variant-pack value for a
  defaulted (value-carrying) UID is inert — assert that, with or without such a
  variant-pack entry, the baked default reaches the provider unchanged
  ([§4.9](#49-execute-time-variant-pack-forwarding)).

- **User-supplied delivery.** A user-supplied host scalar placed in the
  variant pack reaches the provider's `device_buffers` slot and **equals**
  what the caller supplied (end-to-end).

- **Validation.** All three by-value states validate cleanly, and a
  required scalar in any of the three passes `validateScalarParameter`. The
  virtual-exclusion rules reject `virtual && is_runtime_pass_by_value` and
  `virtual && value present` ([§4.8](#48-frontend-validation)).

- **Serialization parity.** A graph serialized without the feature loads
  in a feature-aware runtime with the new flag `false` and is served by a
  baseline plugin unchanged.

- **Reader-version guard.** A graph with a runtime pass-by-value tensor is
  stamped `min_reader_version = 1`; a core supporting a lower reader version
  rejects it with a clean error at deserialize (never silently dropping the
  flag), while a graph of only compile-time constants stamps `0` and loads on
  every core ([§4.10](#410-serialized-graph-reader-version-guard)).

- **Missing required value.** A graph with a user-supplied (#1) tensor whose
  UID is absent from the variant pack at execute is rejected by the provider's
  `findDeviceBuffer` with `INVALID_VALUE` when the engine reads the slot,
  matching a missing input/output tensor. *(Plugin integration; exercised once
  a `1.2.0` provider is available.)*

- **Runtime-default vs runtime-only provider paths** *(plugin integration;
  skipped until a plugin reports `1.2.0` runtime pass-by-value support, then
  run against that plugin)*. Once a plugin adopts the feature, the harness
  feeds it host scalars (below); exercise both a runtime-with-default graph
  (baked default, no override delivered) and a pure user-supplied graph (host
  scalar in the variant pack) end-to-end. The two carry different provider
  expectations (baked-value read vs host-pointer read), so both are needed to
  confirm each is handled correctly.

- **Constant-operation runtime-value regression** *(plugin integration;
  skipped until a plugin reports `1.2.0` runtime pass-by-value support, then
  run against that plugin)*. For each operation that takes a
  compile-time-constant scalar today, add a variant that sneaks in a runtime
  pass-by-value tensor with an obvious sentinel value; assert the result
  reflects the runtime value (or the graph is declined), catching providers
  updated incorrectly or mishandling pass-by-value scalars.

- **Harness host-value injection.** The shared integration harness
  ([RFC 0006](0006_PluginAgnosticIntegrationTests.md)) populates a **host**
  scalar into the variant pack keyed by uid (`variantPack[t.uid()] =
  &hostVal`) for a runtime pass-by-value tensor — the capability this epic
  adds; today the harness only generates device pointers or skips scalars as
  constants. Assert the CPU and GPU references read the pass-by-value scalar
  (host value for a runtime tensor, baked value for a constant) and produce
  matching expected results (e.g. a Batchnorm `epsilon`). Every runtime
  delivery test above depends on this.

- **Mixed multi-scalar graph.** A single op carrying more than one
  pass-by-value scalar in different states (e.g. a compile-time-constant
  `alpha` plus a user-supplied runtime `beta`) executes with each delivered
  or baked correctly and independently.

- **Backend attribute derivation.** After a descriptor round-trip,
  `HIPDNN_ATTR_TENSOR_IS_BY_VALUE` (1307) reads `true` for the two
  value-carrying states (compile-time constant, runtime-with-default) and
  `false` for the pure runtime user-supplied state — it derives value-presence
  (`value.type != NONE`), not the umbrella. `HIPDNN_ATTR_TENSOR_IS_RUNTIME_PASS_BY_VALUE_EXT`
  (1308) reads `true` only for the two runtime states. Confirms 1307 stays
  derived-as-value-presence and only the runtime bit is stored; the by-value
  umbrella is a wrapper-only query (`isByValue()`), which for a pure runtime
  user-supplied tensor reads `true` while C-API 1307 reads `false`
  ([§4.1](#41-tensor-schema-addition)).

- **Post-build setter warning.** Calling a value/mode setter after
  `backendFinalize` logs a warning and does not alter the compiled plan
  ([§4.8](#48-frontend-validation) post-build immutability).

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
  ([§4.9](#49-execute-time-variant-pack-forwarding)).
- **`COMPILE_TIME_CONST`**: the `ScalarType` alternative producing the
  compile-time constant — a value baked into the op-graph flatbuffer,
  never overridable.
- **Runtime, user-supplied** (`is_runtime_pass_by_value == true`,
  no value): created by `set_as_runtime_parameter()` (or
  `set_is_pass_by_value(true)` with no value); the user supplies the host
  pointer in the variant pack. Requires plugin floor `1.2.0`.
- **Runtime with default** (future override) (`is_runtime_pass_by_value == true`,
  value present): created by `TensorAttributes(v, ScalarType::RUNTIME_PARAM)` or
  `graph.tensor(v, ScalarType::RUNTIME_PARAM)`; the value is baked in
  `HIPDNN_ATTR_TENSOR_VALUE_EXT` as a default, overridable via the variant
  pack in a future release (override inert today,
  [§4.9](#49-execute-time-variant-pack-forwarding)). Requires plugin floor
  `1.2.0` (its flag is set).
- **Compile-time constant** (`is_runtime_pass_by_value == false`,
  value present): created by the plain scalar constructor
  `TensorAttributes(v)`, `set_value(v)`, `set_compile_time_constant(v)`,
  `TensorAttributes(v, ScalarType::COMPILE_TIME_CONST)`, or
  `graph.tensor(v, ScalarType::COMPILE_TIME_CONST)`; the value is frozen
  into the op-graph flatbuffer via `HIPDNN_ATTR_TENSOR_VALUE_EXT` and read
  from it, exactly as before this RFC. Imposes no version floor (baseline
  `1.0.0`). The only mode hipDNN supported before this RFC, and still what
  the plain constructor / `set_value` produce (a deliberate divergence from
  cuDNN, [§5.1](#51-cudnn-api-surface-parity)).
- **Execute-time variant-pack forwarding**: `Graph::execute()` forwards the
  caller's variant pack to the backend unchanged — no frontend filter or
  presence-validation. A variant-pack entry for a UID whose tensor carries a
  baked value is inert (the provider uses the baked value); a missing pure
  user-supplied scalar is caught by the provider's `findDeviceBuffer`.
- **Variant pack**: the runtime-only carrier of per-execution payload
  (data pointers, unique IDs, workspace). New in this RFC: a runtime
  pass-by-value tensor's `uid → void*` entry is a *host* pointer to the
  scalar rather than a device pointer. The variant pack has no flatbuffer
  schema and is never serialized.
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
