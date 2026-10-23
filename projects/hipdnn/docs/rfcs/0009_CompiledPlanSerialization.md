# Execution Plan Serialization

- Contributors: hipDNN maintainers
- **Status**: Implemented
- **Status notes**: Plan-only serialization shipped in #6970. The envelope later gained `is_override_shape_enabled` (#7279) and `tensor_alignments` (#8728), and #7975 added a graph-and-plan container. No shipped provider exports the plugin hooks; only the hipDNN test plugins implement them.
- **Last verified**: 2026-10-15 against `develop` (2501d0b28cc)
- Scope: frontend, backend, engine plugin API, test plugin implementation

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
serialization returns `HIPDNN_STATUS_NOT_SUPPORTED` for plans using those plugins.

The plugin payload contract is byte-level:

- The plugin allocates and owns `serialized_context` for the duration of the serialize call.
- hipDNN copies the bytes into `plugin_payload`.
- hipDNN calls the plugin destroy hook after copying.
- hipDNN passes the same bytes back during deserialization.
- Plugin payload versioning and compatibility are plugin-owned.

## Backend Object Model

`ExecutionPlanDescriptor` now supports two construction paths:

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

No provider in `dnn-providers/` exports the serialization hooks. The hipDNN test plugins implement
them in `tests/test_plugins/TestPluginCommon.hpp`. Their `plugin_payload` is a fixed one-byte
placeholder. The test plugins also advertise
`HIPDNN_BEHAVIOR_NOTE_SUPPORTS_EXECUTION_PLAN_SERIALIZATION`.

Round-trip coverage lives in `frontend/tests/TestGraph.cpp`,
`tests/frontend/IntegrationOverrideExecute.cpp`, and
`backend/tests/TestGraphAndPlanSerializationApi.cpp`.

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
versions once a provider implements the serialization hooks. The current unit coverage checks
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
