# Execution Plan Serialization

- Contributors: hipDNN maintainers
- Status: Implemented initial slice
- Scope: frontend, backend, engine plugin API, fusilli sample implementation

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

## Envelope Format

The hipDNN envelope is a FlatBuffer schema in `flatbuffers_sdk/schemas/execution_plan.fbs`:

```flatbuffers
namespace hipdnn_flatbuffers_sdk.data_objects;

table SerializedExecutionPlan {
    version: uint;
    engine_id: int64;
    workspace_size: int64;
    tensor_uids: [int64];
    engine_config: [ubyte];
    plugin_payload: [ubyte];
}

root_type SerializedExecutionPlan;
```

Field ownership:

- `version`: hipDNN envelope version.
- `engine_id`: selected backend engine ID.
- `workspace_size`: workspace size reported for the compiled plan.
- `tensor_uids`: hipDNN-level tensor UID metadata needed for UID-based execution.
- `engine_config`: serialized engine config bytes used to recreate the plugin execution context.
- `plugin_payload`: opaque plugin-specific execution context bytes.

There is deliberately no operation graph in the envelope. A backend may not use graph execution, and
plugins that need graph-derived data must store it in their own payload.

## Plugin API

Engine plugins may implement these optional C API entry points:

```cpp
hipdnnPluginStatus_t hipdnnEnginePluginSerializeExecutionContext_ext(
    hipdnnEnginePluginHandle_t handle,
    hipdnnEnginePluginExecutionContext_t executionContext,
    const hipdnnPluginConstData_t* engineConfig,
    hipdnnPluginConstData_t* serializedContext);

hipdnnPluginStatus_t hipdnnEnginePluginDestroySerializedExecutionContext_ext(
    hipdnnEnginePluginHandle_t handle,
    hipdnnPluginConstData_t* serializedContext);

hipdnnPluginStatus_t hipdnnEnginePluginCreateExecutionContextFromSerialized_ext(
    hipdnnEnginePluginHandle_t handle,
    const hipdnnPluginConstData_t* engineConfig,
    const hipdnnPluginConstData_t* serializedContext,
    hipdnnEnginePluginExecutionContext_t* executionContext);
```

These functions are optional. Plugins that do not export them still load normally, but compiled-plan
serialization returns `HIPDNN_STATUS_NOT_SUPPORTED` for plans using those plugins.

The plugin payload contract is byte-level:

- The plugin allocates and owns `serializedContext` for the duration of the serialize call.
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
- Tensor UIDs.
- Serialized engine config bytes.
- Plugin resource manager.
- Plugin execution context wrapper.

Execution reads the engine ID and execution context directly from the execution plan descriptor.
This is what makes a graph-less restored plan executable.

## Fusilli Implementation

The fusilli provider implements the optional plugin hooks as the sample backend. Its
`plugin_payload` is fusilli-owned and starts with a small fusilli compatibility header:

```text
magic: "FUSPLAN\0"
format_major: uint16
format_minor: uint16
payload_kind: uint32
header_size: uint32
payload_bytes: uint8[]
```

The initial `payload_kind` stores the hipDNN graph FlatBuffer bytes needed to reconstruct and
compile a fusilli execution context. Fusilli rejects payloads with an unknown magic value, an
unsupported major format version, an unsupported payload kind, or an empty/malformed payload before
attempting graph import. This keeps the hipDNN envelope byte-level while giving fusilli an explicit
breaking-change boundary for future native compiled-artifact serialization.

The sample at
`dnn-providers/fusilli-provider/test/integration/matmul/serialized_matmul.cpp`:

1. Builds a simple matmul graph.
2. Builds the compiled execution plan.
3. Serializes the compiled plan.
4. Deserializes it into a fresh frontend `Graph`.
5. Executes with UID bindings.
6. Validates the output against the CPU reference.

This sample demonstrates the intended contract: the original graph performs compilation; the
restored graph owns only an executable compiled plan.

## Current Limitations

1. There is no frontend convenience API yet for querying `tensor_uids`; the backend exposes them via
   `HIPDNN_ATTR_EXECUTION_PLAN_TENSOR_UIDS_EXT`.
2. Deserialized execution-plan attribute semantics are only partially defined. In particular,
   `HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG` may not be reconstructable for graph-less plans.
3. The envelope does not include plugin identity. The current deserialization path relies on the
   handle's loaded engine plugin resource manager and the serialized engine ID/config.

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
versions once the fusilli payload compatibility policy is stable. The current unit coverage checks
malformed envelopes, missing and empty required fields, invalid workspace metadata, unsupported
plugin serialization, and plugin callback failures.

### Add Minimal Plugin Identity Metadata

Consider adding a small hipDNN-owned plugin identity section in a future envelope version. It should
be used only for routing and diagnostics, not for structuring the plugin payload. A conservative
first version could include plugin name, plugin version, and engine ID compatibility expectations.

### Improve Fusilli Payload Stability

The fusilli sample should eventually serialize a stable compiled artifact when fusilli exposes one.
Until then, fusilli owns compatibility for its payload bytes and may reject incompatible payloads.

## Non-Goals

- Serialize the hipDNN operation graph as part of compiled-plan serialization.
- Require graph-executor semantics from all backends.
- Define a common plugin payload schema.
- Rebuild from a source graph when direct compiled-plan deserialization fails.
