# Execution Plan Serialization

- Contributors: hipDNN maintainers
- **Status**: Implemented
- **Status notes**: Plan-only serialization shipped in #6970. The envelope later gained `is_override_shape_enabled` (#7279) and `tensor_alignments` (#8728), and #7975 added a graph-and-plan container. The hip-kernel-provider exports the plugin hooks for its kernel ingestor engines (#13315), and the hipDNN test plugins also implement them.
- **Last verified**: 2026-10-22 against `develop` (0bb1c36e8f7) with #13315 applied
- Scope: frontend, backend, engine plugin API, test plugin implementation, hip-kernel-provider
  ingestor implementation

## Summary

hipDNN can serialize a compiled execution plan to bytes, deserialize those bytes later, and execute
the restored plan with UID-based variant packs. The feature is intentionally plan-only: it restores
the executable plan state, not the frontend operation graph.

The serialized object has two layers:

1. A hipDNN-owned FlatBuffer envelope rooted at `SerializedExecutionPlan`.
2. An opaque plugin-owned byte payload.

hipDNN owns routing metadata and tensor UID metadata. Plugins own their internal serialization
format. hipDNN never inspects or constrains the plugin payload beyond copying bytes into and out of
the envelope.

## User Flow

Build and serialize a compiled plan:

```cpp
graph.validate();
graph.build_operation_graph(handle);
graph.create_execution_plans();
graph.check_support();
graph.build_plans();

std::vector<uint8_t> compiledPlan;
graph.serialize_compiled_plan(compiledPlan);
```

Deserialize and execute:

```cpp
hipdnn_frontend::graph::Graph restored;
restored.deserialize_compiled_plan(handle, compiledPlan);
restored.execute(handle, variantPack, workspace);
```

The restored `Graph` contains only an execution plan descriptor. APIs that depend on source graph
nodes, engine configs, or graph mutation should not be assumed to work after compiled-plan
deserialization.

## Frontend API

`hipdnn_frontend::graph::Graph` exposes:

```cpp
Error serialize_compiled_plan(std::vector<uint8_t>& data) const;
std::pair<std::vector<uint8_t>, Error> to_compiled_plan_binary() const;

Error deserialize_compiled_plan(hipdnnHandle_t handle, const std::vector<uint8_t>& data);
Error from_compiled_plan_binary(hipdnnHandle_t handle, const std::vector<uint8_t>& data);
```

`serialize_compiled_plan()` requires a finalized execution plan. `deserialize_compiled_plan()`
creates a finalized backend execution plan descriptor, stores it in the frontend graph, clears graph
build descriptors, and leaves execution available through the normal UID variant-pack path.

## Backend API

The frontend calls these backend extension APIs:

```cpp
hipdnnStatus_t hipdnnBackendGetSerializedExecutionPlan_ext(
    hipdnnBackendDescriptor_t descriptor,
    size_t requestedByteSize,
    size_t* planByteSize,
    uint8_t* serializedPlan);

hipdnnStatus_t hipdnnBackendCreateAndDeserializeExecutionPlan_ext(
    hipdnnHandle_t handle,
    hipdnnBackendDescriptor_t* descriptor,
    const uint8_t* serializedPlan,
    size_t planByteSize);
```

Serialization follows the normal two-call size-query pattern. Deserialization uses the supplied
`hipdnnHandle_t` to find the engine plugin resource manager, restore the plugin execution context,
and return a finalized execution plan descriptor.

`hipdnnBackendCreateAndDeserializeExecutionPlan_ext()` also accepts a graph-and-plan container
blob made by `hipdnnBackendGetSerializedBinaryGraphAndPlan_ext()` (#7975). It extracts the embedded
execution plan. A graph-only container fails with `HIPDNN_STATUS_BAD_PARAM`.

## Envelope Format

The hipDNN envelope is a FlatBuffer schema in `flatbuffers_sdk/schemas/execution_plan.fbs`:

```flatbuffers
namespace hipdnn_flatbuffers_sdk.data_objects;

table SerializedExecutionPlan {
    version: uint;
    engine_id: int64;
    workspace_size: int64;
    tensor_uids: [int64];
    plugin_payload: [ubyte];
    is_override_shape_enabled: bool = false;
    // Required byte alignment of each tensor's device pointer. Parallel to
    // tensor_uids: tensor_alignments[i] applies to tensor_uids[i].
    tensor_alignments: [int64];
}

root_type SerializedExecutionPlan;
```

Field ownership:

- `version`: hipDNN envelope version. The current version is 2 (`PLAN_SERIALIZATION_VERSION` in
  `backend/src/descriptors/ExecutionPlanDescriptor.cpp`). Version 1 plans have no
  `tensor_alignments` and still deserialize.
- `engine_id`: selected backend engine ID.
- `workspace_size`: workspace size reported for the compiled plan.
- `tensor_uids`: hipDNN-level tensor UID metadata needed for UID-based execution.
- `plugin_payload`: opaque plugin-specific execution context bytes.
- `is_override_shape_enabled`: whether the plan was built with override shapes enabled (RFC 0008).
- `tensor_alignments`: required device-pointer byte alignment per tensor, parallel to `tensor_uids`.

There is deliberately no operation graph in the envelope. A backend may not use graph execution, and
plugins that need graph-derived data must store it in their own payload.

## Plugin API

Engine plugins may implement these optional C API entry points
(`plugin_sdk/include/hipdnn_plugin_sdk/EnginePluginApi.h`):

```cpp
hipdnnPluginStatus_t hipdnnEnginePluginSerializeExecutionContext(
    hipdnnEnginePluginHandle_t handle,
    hipdnnEnginePluginExecutionContext_t execution_context,
    hipdnnPluginConstData_t* serialized_context);

hipdnnPluginStatus_t hipdnnEnginePluginDestroySerializedExecutionContext(
    hipdnnEnginePluginHandle_t handle,
    hipdnnPluginConstData_t* serialized_context);

hipdnnPluginStatus_t hipdnnEnginePluginCreateExecutionContextFromSerialized(
    hipdnnEnginePluginHandle_t handle,
    const hipdnnPluginConstData_t* serialized_context,
    hipdnnEnginePluginExecutionContext_t* execution_context);
```

These functions are optional. Plugins that do not export them still load normally, but compiled-plan
serialization returns `HIPDNN_STATUS_NOT_SUPPORTED` for plans using those plugins. hipDNN uses the
hooks only when the plugin exports all three.

The plugin payload contract is byte-level:

- The plugin allocates the `serialized_context` bytes and owns them. The bytes must stay valid
  until hipDNN calls the destroy hook.
- hipDNN copies the bytes into `plugin_payload`. hipDNN refuses a null or empty payload with
  `HIPDNN_STATUS_PLUGIN_ERROR`.
- After each successful serialize call, hipDNN calls the destroy hook one time, also when it
  refuses the payload. After a failed serialize call, hipDNN does not call the destroy hook. The
  plugin must then free its own memory.
- hipDNN passes the same bytes back during deserialization.
- Plugin payload versioning and compatibility are plugin-owned.

## Backend Object Model

`ExecutionPlanDescriptor` supports two construction paths:

1. Normal build path: finalize from an `EngineConfigDescriptor`, graph, and plugin resource manager.
2. Deserialized path: finalize from `SerializedExecutionPlan` bytes and a restored plugin execution
   context.

The descriptor stores enough state to execute without `EngineConfig -> Engine -> Graph`:

- Engine ID.
- Workspace size.
- Tensor UIDs and tensor alignments.
- Override-shape-enabled flag.
- Plugin resource manager.
- Plugin execution context wrapper.

Execution reads the engine ID and execution context directly from the execution plan descriptor.
This is what makes a graph-less restored plan executable.

## Plugin Implementations

The hip-kernel-provider exports the serialization hooks for its kernel ingestor engines. See
[hip-kernel-provider Ingestor Implementation](#hip-kernel-provider-ingestor-implementation). The
hipDNN test plugins also implement them in `tests/test_plugins/TestPluginCommon.hpp`. Their
`plugin_payload` is a fixed one-byte placeholder. The test plugins also advertise
`HIPDNN_BEHAVIOR_NOTE_SUPPORTS_EXECUTION_PLAN_SERIALIZATION`.

Round-trip coverage lives in `frontend/tests/TestGraph.cpp`,
`tests/frontend/IntegrationOverrideExecute.cpp`, and
`backend/tests/TestGraphAndPlanSerializationApi.cpp`.

## hip-kernel-provider Ingestor Implementation

The hip-kernel-provider implements the three plugin hooks for its kernel ingestor engines. The
provider exports the hooks only when the build sets `HIPDNN_ENABLE_KERNEL_INGESTOR=ON`. The hooks
are in `dnn-providers/hip-kernel-provider/src/core/PluginSerialization.cpp`.

### Supported Engines

An ingestor engine supports serialization when each kernel in its descriptor set is a kpack code
object. Each dispatch handler of the set must also report `supportsSaving()`. The provider
calculates this from the descriptor set and its dispatch handlers. A supported engine reports the
behavior note `HIPDNN_BEHAVIOR_NOTE_SUPPORTS_EXECUTION_PLAN_SERIALIZATION`. The other engines of the
provider do not report this note.

The save hook refuses a plan that no ingestor engine built. Such a plan has no single-kernel ingestor
plan to save.

### Payload Layout

The `plugin_payload` starts with a 48-byte header. All header integers are little-endian.

```text
offset  size  field
0       4     marker: "HKIP"
4       2     format_major: uint16
6       2     format_minor: uint16
8       2     kind: uint16 (1 = single-kernel plan)
10      2     header_size: uint16
12      4     reserved: zero
16      32    body_sha256: SHA-256 of the body
```

The body starts at `header_size`. The body is a FlatBuffer with the file identifier `HKSP`. The
schema is `dnn-providers/hip-kernel-provider/src/engines/kernel_ingestor_engine/serialization/ingestor_plan.fbs`.
The body contains this data:

- The kernel descriptor ID.
- The workspace size in bytes.
- The UIDs of the runtime pass-by-value tensors.
- The dispatch data: the versioned dispatch name and the named, typed launch values.
- The kernel image: the source kind, the kernel symbol, the target (the base architecture of the
  GPU that saved the plan), the SHA-256 of the code object, the recorded argument signature, and
  the uncompressed code object.
- The provider version. The provider version is for diagnostics only.

The launch values are the inputs from which the dispatch handler calculates a launch. The payload
does not hold calculated grid sizes or kernel argument values.

A loaded plan runs the code it was saved with. Upgrading the provider does not change the kernel
inside a saved plan, even when the upgrade fixes that kernel. To use the new kernel, build the plan
again from its graph and save it.

### Versioning

The current format version is 1.0. A reader reads a payload when these conditions are true:

- The major version of the payload is equal to the major version of the reader.
- The minor version of the payload is not more than the minor version of the reader.

A new minor version can make only these changes to the schema:

- Add an optional table field with the next unused field ID. Do not make the field `(required)`.
  Give a new scalar field `= null`, or a default value that means "absent".
- Add a member at the end of a union.
- Add a value at the end of an enum.
- Declare a new table or enum that only a new field or a new union member uses.
- Retire a field: mark the field `(deprecated)`. Keep its field ID.

The reader of the new minor version accepts a payload without the new field, and a payload of an
older minor version.

Never make these changes in a minor version:

- Change the type or the default value of a field.
- Add or remove `(required)` on a field that exists.
- Change or reuse a field ID.
- Delete a field.

All other changes need a new major version. The reader refuses a payload of a different major
version before it reads the body. Thus a new major version can change the schema freely.

Never change the fields of the first 48 bytes of the header, in any version. A reader uses them to
refuse a payload.

`IngestorPlanPayload.hpp` holds `INGESTOR_PLAN_SCHEMA_SHA256` next to the format version constants.
This value is the SHA-256 of `ingestor_plan.fbs` with LF line ends. The build calculates the
SHA-256 of the schema file, and a unit test compares it with this value. After each schema change,
decide the version change and then update the value.

The dispatch handler contract has a separate version. That version is part of the dispatch name.
It is not the format version of the saved file. The dispatch version identifies the contract between
the saved launch values and the launch. A saved plan can pair an older code object with the current
launch. Use a new dispatch version when one of these items changes:

- The names, types, or meanings of the launch values.
- The kernel arguments that the launch passes: their order, count, kinds, sizes, or meanings.
- How the launch calculates the grid, the block, or the dynamic LDS from the launch values.

Keep the dispatch handler of the old version registered while plans saved with that version must
load.

The provider version never controls a load. A reader accepts a header larger than 48 bytes when its
size is a multiple of 16.

### Load Checks

hipDNN uses the envelope engine ID to find the plugin. When no loaded plugin serves that engine ID,
hipDNN refuses the plan with `HIPDNN_STATUS_INTERNAL_ERROR`. The plugin does not run. Nothing checks
that the payload belongs to the engine of the envelope engine ID.

The load hook then does these checks in this order. The first check that fails stops the load.

1. The payload holds the fixed header fields (`INVALID_VALUE`).
2. The marker is `HKIP` (`INVALID_VALUE`).
3. The provider reads the format version (`NOT_APPLICABLE`).
4. The plan kind is known (`NOT_APPLICABLE`).
5. The header size is valid for the payload (`INVALID_VALUE`).
6. The body size is in the FlatBuffers limits (`INVALID_VALUE`).
7. The SHA-256 of the body is equal to the header value (`INVALID_VALUE`).
8. The FlatBuffers verifier accepts the body, and the body has the `HKSP` identifier
   (`INVALID_VALUE`).
9. The workspace size and the source kind are valid. The kernel SHA-256 text is valid and matches
   the code object. The launch values are valid. Each of these refusals is `INVALID_VALUE`.
10. The plan has no runtime pass-by-value tensors (`NOT_APPLICABLE`).
11. The device of the handle can run the stored GPU target (`NOT_APPLICABLE`). For a kpack plan,
    the stored target is the base architecture of the GPU that saved the plan, for example
    `gfx942`. A device matches when its architecture name is that target, with or without feature
    flags. A generic archive entry does not extend this match. A plan whose build read a generic
    entry loads only on the base architecture that saved it. Other source kinds need the exact
    target.
12. A dispatch handler is registered under the dispatch name (`NOT_APPLICABLE`).
13. The handler supports restore (`NOT_APPLICABLE`).
14. The recorded argument signature matches the arguments that the handler launches
    (`NOT_APPLICABLE`).
15. The handler accepts the launch values (`NOT_APPLICABLE`).
16. The driver loads the code object on the device of the handle and finds the kernel symbol
    (`NOT_APPLICABLE`).

The load hook loads the code object. As a result, a code object that the driver refuses fails the
load, not the first execute. Another device loads its own module from the stored bytes at its first use.

### Statuses and Messages

The plugin uses two statuses for a refusal. Each status has one message prefix for a load and one
for a save. One save refusal uses the load prefix: a code object that is too large for the payload
format gives `HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE` with the prefix
`ingestor plan is valid but cannot be saved or loaded here: `.

| Cause | Plugin status | Load message prefix | Save message prefix |
| --- | --- | --- | --- |
| The data is damaged. | `HIPDNN_PLUGIN_STATUS_INVALID_VALUE` | `ingestor plan data is damaged: ` | `cannot save this plan: its kernel source does not match its descriptor: ` |
| The plan is valid, but the provider cannot save or load it here. | `HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE` | `ingestor plan is valid but cannot be saved or loaded here: ` | `cannot save this plan here: ` |

hipDNN reports both plugin statuses as `HIPDNN_STATUS_PLUGIN_ERROR`. The hipDNN error message keeps
the text of the plugin. The prefix identifies the cause category. The remaining text identifies the
cause. A fault in the provider gives `HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR`.

### Save Rules

The save hook refuses these plans with `HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE`. It does the checks in
this order:

1. A context without a plan.
2. A plan that the provider loaded from a payload.
3. A benchmarking plan that has not selected a kernel.
4. A plan that no ingestor engine built.
5. A kernel with a source kind other than kpack.
6. A plan whose dispatch handler does not support save.

Rule 2 has a result for the frontend: `to_binary()` of a graph that `from_binary()` loaded fails.
`to_compiled_plan_binary()` of a loaded plan also fails. Keep the original bytes of the payload.

For rule 3: a benchmarking plan usually selects its kernel at the first execute. A stored benchmark
result can cover all candidates. In that case, the build makes a plain plan, and the plan saves before
the first execute. To save a benchmarking plan, execute the plan one time, then save it. Or, build the
plan with benchmarking off.

The save hook reads the code object from the kpack archive that the plan build used. The save hook
can read the archive through the handle that the plan build opened. It also checks the SHA-256 of
the code object. If the archive changes after the plan build, the save can fail. When the save
fails because the archive is not readable, the status is `HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE`.
When the save fails because the bytes do not match the SHA-256, the status is
`HIPDNN_PLUGIN_STATUS_INVALID_VALUE`. Do not change the archive between the plan build and the save.

### Security

A saved plan contains executable GPU code. The load hook loads it, and the restored plan executes
it. Load only payloads from a trusted source.

- The SHA-256 values detect accidental damage. They do not detect a crafted payload.
- A load does not do the path checks of the descriptor tree. A save also does not repeat them.
- The driver function that loads the code object, `hipModuleLoadData`, takes no length. A crafted
  code object goes to the driver.

## Current Limitations

1. There is no frontend convenience API yet for querying `tensor_uids`; the backend exposes them via
   `HIPDNN_ATTR_EXECUTION_PLAN_TENSOR_UIDS_EXT`.
2. Deserialized execution-plan attribute semantics are only partially defined. In particular,
   `HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG` may not be reconstructable for graph-less plans.
3. The envelope does not include plugin identity. The current deserialization path relies on the
   handle's loaded engine plugin resource manager and the serialized engine ID.

## Proposed Improvements

### Add Plan Binding Query Helpers

Add frontend helpers to query tensor UID metadata from a compiled or deserialized plan:

```cpp
std::pair<std::vector<int64_t>, Error> get_compiled_plan_tensor_uids() const;
```

This avoids requiring users to call backend descriptor attributes directly.

### Define Deserialized Attribute Semantics

Document and enforce which execution-plan attributes are valid after deserialization. Attributes
that require the original graph or a reconstructed `EngineConfigDescriptor` should return a clear
unsupported or not-available status.

### Add File-Level Compatibility Tests

Add integration tests that persist compiled-plan files across process boundaries and plugin
versions. The current unit coverage checks
malformed envelopes, missing and empty required fields, invalid workspace metadata, unsupported
plugin serialization, and plugin callback failures.

### Add Minimal Plugin Identity Metadata

Consider adding a small hipDNN-owned plugin identity section in a future envelope version. It should
be used only for routing and diagnostics, not for structuring the plugin payload. A conservative
first version could include plugin name, plugin version, and engine ID compatibility expectations.

## Non-Goals

- Serialize the hipDNN operation graph as part of compiled-plan serialization.
- Require graph-executor semantics from all backends.
- Define a common plugin payload schema.
- Rebuild from a source graph when direct compiled-plan deserialization fails.
