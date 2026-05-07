# hipDNN - Per-Engine Behavior Notes RFC

- Contributors: TBD
- **Status**: Draft
- **Implementation Version**: TBD

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Current System Overview](#3-current-system-overview)
4. [Proposed Design](#4-proposed-design)
5. [API Contract](#5-api-contract)
6. [Key Design Decisions](#6-key-design-decisions)
7. [Risks](#7-risks)
8. [Execution Plan](#8-execution-plan)
9. [Testing Plan](#9-testing-plan)
10. [Future Considerations](#10-future-considerations)
11. [Glossary](#11-glossary)

## 1. Executive Summary

This RFC proposes per-engine behavior-note queries for hipDNN. A behavior note is advisory metadata reported by an engine, such as runtime compilation or an external library dependency. The first implementation will let users request behavior notes for a particular engine ID after an operation graph has been finalized.

The proposed first API is:

```cpp
Error Graph::get_behavior_notes_for_engine(int64_t engineId,
                                           std::vector<BehaviorNote>& notes);
```

This RFC intentionally does not add graph-level filtering, plan-index note queries, numerical notes, or compiled execution-plan serialization changes. Behavior notes are engine/plugin metadata, not execution-plan payload.

## 2. Problem Statement

hipDNN currently reserves backend attribute/type IDs for behavior and numerical notes, but it does not expose behavior-note data from engines.

Existing public backend constants:

- `HIPDNN_TYPE_BEHAVIOR_NOTE`
- `HIPDNN_ATTR_ENGINE_BEHAVIOR_NOTE`
- `HIPDNN_TYPE_NUMERICAL_NOTE`
- `HIPDNN_ATTR_ENGINE_NUMERICAL_NOTE`

Current limitations:

1. `EngineDescriptor` rejects `HIPDNN_ATTR_ENGINE_BEHAVIOR_NOTE`.
2. There is no public shared `hipdnnBackendBehaviorNote_t` enum.
3. `EngineDetails` contains only `engine_id` and `knobs`.
4. The frontend has no `BehaviorNote` enum or per-engine note query.
5. Python bindings do not expose behavior-note metadata.

Users need a way to inspect advisory behavior for a chosen engine before creating an execution plan, especially when selecting engines manually through `get_ranked_engine_ids()` and `create_execution_plan_ext()`.

## 3. Current System Overview

The current engine path is:

1. `EngineHeuristicDescriptor` asks `EnginePluginResourceManager::getApplicableEngineIds()` for candidate engine IDs.
2. `EngineHeuristicDescriptor::getEngineConfigs()` creates `EngineConfigDescriptor` objects.
3. `EngineDescriptor::finalize()` fetches plugin `EngineDetails`.
4. `Graph::create_execution_plans()` selects a single engine config.

`EngineDetails` is the correct metadata boundary for behavior notes because plugins already use it to report per-engine details such as knobs.

cuDNN frontend supports broader note behavior, including candidate filtering through `select_behavior_notes()` and `deselect_behavior_notes()`. hipDNN will not implement that broader filtering in this RFC. The first hipDNN behavior is a direct per-engine query.

## 4. Proposed Design

### 4.1 Public Types

Add one backend public enum, duplicated in the plugin SDK for plugin use, for example:

```cpp
typedef enum
{
    HIPDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION = 0,
    HIPDNN_BEHAVIOR_NOTE_REQUIRES_LAYOUT_TRANSFORM = 1,
    HIPDNN_BEHAVIOR_NOTE_SUPPORTS_GRAPH_CAPTURE = 2,
    HIPDNN_BEHAVIOR_NOTE_EXTERNAL_LIBRARY_DEPENDENCY = 3,
    HIPDNN_BEHAVIOR_NOTE_SUPPORTS_EXECUTION_PLAN_SERIALIZATION = 4,
    HIPDNN_BEHAVIOR_NOTE_TYPE_COUNT = 5
} hipdnnBackendBehaviorNote_t;
```

The backend and plugin SDK definitions should use the same `HIPDNN_BEHAVIOR_NOTE_TYPES_DEFINED` include guard, matching the existing callback-type pattern. This keeps backend public headers self-contained and avoids adding a backend package dependency on the plugin SDK, while still letting plugin SDK users obtain the same numeric values through the plugin SDK dependency path.

The values must remain synchronized between:

- `backend/include/HipdnnBackendBehaviorNote.h`
- `plugin_sdk/include/hipdnn_plugin_sdk/BehaviorNote.h`

Add one frontend wrapper enum:

```cpp
enum class BehaviorNote
{
    RUNTIME_COMPILATION,
    REQUIRES_LAYOUT_TRANSFORM,
    SUPPORTS_GRAPH_CAPTURE,
    EXTERNAL_LIBRARY_DEPENDENCY,
    SUPPORTS_EXECUTION_PLAN_SERIALIZATION
};
```

The final names should follow hipDNN naming conventions. The enum set should stay small and only include notes with stable semantics.

### 4.2 EngineDetails Metadata

Append one optional field to `flatbuffers_sdk/schemas/engine_details.fbs`:

```fbs
table EngineDetails {
    engine_id: int64;
    knobs: [Knob];
    behavior_notes: [int32];
}
```

Compatibility rules:

- Missing `behavior_notes` means an empty note list.
- Empty means "the engine reports no behavior notes", not "metadata unavailable".
- Existing plugin binaries should remain readable because the field is optional and appended.
- Existing plugin source should usually continue to compile because generated FlatBuffers builders default later fields, but CI must verify this.
- The field must not be marked `required`.
- `EngineDetailsWrapper` should expose raw `int32` values so the FlatBuffers SDK does not depend on backend headers.
- `EngineDescriptor::finalize()` should convert raw values to `hipdnnBackendBehaviorNote_t` and reject invalid values.

Plugins may initially emit empty behavior-note lists. Non-empty notes should only be emitted where the engine can make a true statement, for example runtime compilation or an external library dependency.

### 4.3 Backend EngineDescriptor Query

Implement `HIPDNN_ATTR_ENGINE_BEHAVIOR_NOTE` on `EngineDescriptor`.

Backend behavior:

- `EngineDescriptor::finalize()` reads behavior notes from `EngineDetailsWrapper` and caches them.
- `getAttribute()` supports `HIPDNN_ATTR_ENGINE_BEHAVIOR_NOTE` with `HIPDNN_TYPE_BEHAVIOR_NOTE`.
- Count-only query works with `requestedElementCount == 0` and `arrayOfElements == nullptr`.
- A second query copies values when `requestedElementCount >= elementCount`.
- Wrong type, too-small output buffer, and invalid note values fail clearly.
- Setting behavior notes on `EngineDescriptor` remains unsupported.

### 4.4 Frontend Query API

Add:

```cpp
Error Graph::get_behavior_notes_for_engine(int64_t engineId,
                                           std::vector<BehaviorNote>& notes);
```

Implementation behavior:

- Clear `notes` on entry.
- Require a finalized graph descriptor.
- Create/finalize an engine descriptor with `detail::createEngineDescriptorForGraph`.
- Query backend behavior-note count.
- Query backend behavior-note values.
- Convert backend notes to frontend `BehaviorNote` values.

### 4.5 Python API

Bind:

- `BehaviorNote`
- `Graph.get_behavior_notes_for_engine(engine_id) -> list[BehaviorNote]`

## 5. API Contract

`Graph::get_behavior_notes_for_engine(int64_t engineId, std::vector<BehaviorNote>& notes)` has the following contract:

- `engineId` is the backend global engine ID to query for the graph currently held by the `Graph`.
- The operation graph descriptor must be finalized with a handle. Practically, `build_operation_graph(handle)` or `build(handle)` must have succeeded.
- `notes` is caller-owned and is cleared on entry.
- On success, `notes` contains exactly the behavior notes reported by that engine.
- On failure, `notes` remains empty to avoid stale results.
- Success with no notes returns `ErrorCode::OK` and leaves `notes` empty.
- Invalid or inapplicable engine IDs return a frontend error wrapping the backend failure from creating/finalizing the engine descriptor.
- Backend query failures return a frontend error with context that `HIPDNN_ATTR_ENGINE_BEHAVIOR_NOTE` query failed.
- The method must keep `createEngineDescriptorForGraph()` finalization in the path so only engines applicable to the finalized graph can be queried.

## 6. Key Design Decisions

- Behavior notes are engine/plugin metadata.
- `SerializedExecutionPlan` remains unchanged.
- Per-engine query is the first API because it fits hipDNN's current single-selected-plan model.
- Filtering by notes is a future frontend selection feature, not part of this RFC.
- Numerical notes are not part of this RFC. They can reuse the same pattern later.

### 6.1 Included Behavior Notes

- `RUNTIME_COMPILATION`: the engine may compile, generate, or specialize code at runtime. This is useful for callers that need predictable startup or plan-build latency.
- `REQUIRES_LAYOUT_TRANSFORM`: the engine may require tensor layout conversion. This helps callers reason about workspace, memory traffic, and latency costs.
- `SUPPORTS_GRAPH_CAPTURE`: the engine is compatible with graph-capture workflows. This is a hard deployment requirement for some repeated execution paths.
- `EXTERNAL_LIBRARY_DEPENDENCY`: the engine depends on an external runtime or library beyond the core hipDNN API surface. This helps deployment and reproducibility checks.
- `SUPPORTS_EXECUTION_PLAN_SERIALIZATION`: the engine supports binary execution-plan serialization. This lets callers prefer engines that can participate in plan cache save/load flows without changing `SerializedExecutionPlan` in this RFC.

## 7. Risks

1. **Ambiguous semantics**: Plugins may emit notes inconsistently. Mitigation: keep the initial enum small and document each value precisely.
2. **FlatBuffers compatibility mistakes**: Inserting fields or marking fields required would break old buffers. Mitigation: append optional fields only and add old-shape buffer tests.
3. **Invalid raw metadata**: Plugins can emit invalid `int32` values. Mitigation: validate during `EngineDescriptor::finalize()`.
4. **Future API pressure**: Users may ask for cuDNN-style filtering. Mitigation: document filtering as future work and keep this RFC scoped to per-engine queries.

## 8. Execution Plan

### 8.1 Files To Edit

`backend/include/HipdnnBackendBehaviorNote.h`

- Add the backend public `hipdnnBackendBehaviorNote_t` values.
- Guard the definition with `HIPDNN_BEHAVIOR_NOTE_TYPES_DEFINED`.
- Add a comment noting the definition is duplicated by the plugin SDK header and must stay synced.

`plugin_sdk/include/hipdnn_plugin_sdk/BehaviorNote.h`

- Add the same `hipdnnBackendBehaviorNote_t` values for plugin SDK users.
- Guard the definition with `HIPDNN_BEHAVIOR_NOTE_TYPES_DEFINED`.
- Add a reciprocal comment noting the definition is duplicated by the backend header and must stay synced.

`plugin_sdk/include/hipdnn_plugin_sdk/PluginApiDataTypes.h`

- Do not include behavior-note definitions transitively. Plugin API users that emit behavior notes should explicitly include `<hipdnn_plugin_sdk/BehaviorNote.h>`.

`backend/include/hipdnn_backend.h`

- Include `HipdnnBackendBehaviorNote.h`.

`backend/src/BackendEnumStringUtils.hpp`

- Add behavior-note enum-to-string support if this remains the local convention for public enum helpers.

`flatbuffers_sdk/schemas/engine_details.fbs`

- Append `behavior_notes: [int32];`.
- Do not edit `flatbuffers_sdk/schemas/execution_plan.fbs`.

Generated FlatBuffer files

- Regenerate, do not hand-edit:
  - `flatbuffers_sdk/include/hipdnn_flatbuffers_sdk/data_objects/engine_details_generated.h`

`flatbuffers_sdk/include/hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineDetailsWrapper.hpp`

- Add a raw behavior-note accessor.
- Return empty for missing `behavior_notes`.
- Leave enum conversion and invalid-value rejection to backend descriptors.

`backend/src/descriptors/EngineDescriptor.hpp`

- Add cached behavior notes:

```cpp
std::vector<hipdnnBackendBehaviorNote_t> _behaviorNotes;
```

`backend/src/descriptors/EngineDescriptor.cpp`

- Read and cache notes during `finalize()`.
- Implement `HIPDNN_ATTR_ENGINE_BEHAVIOR_NOTE` in `getAttribute()`.
- Keep behavior-note setting unsupported.

`frontend/include/hipdnn_frontend/Types.hpp`

- Add `BehaviorNote`.
- Add backend/frontend conversion helpers.

`frontend/include/hipdnn_frontend/Graph.hpp`

- Add `get_behavior_notes_for_engine()`.

`python/src/types_bindings.cpp`

- Bind `BehaviorNote`.

`python/src/graph_bindings.cpp`

- Bind `Graph.get_behavior_notes_for_engine(engine_id)`.

Plugin implementations

- Update `CreateEngineDetails(...)` call for the new schema.
- Emit empty behavior notes unless a sample or test engine has a truthful note.

### 8.2 Implementation Order

1. Add the duplicated guarded `hipdnnBackendBehaviorNote_t` definitions and string conversion.
2. Append `behavior_notes` to `EngineDetails` and update wrappers.
3. Update plugin/sample metadata creation.
4. Implement `EngineDescriptor` behavior-note query.
5. Add backend and FlatBuffers compatibility tests.
6. Add frontend `BehaviorNote` and `Graph::get_behavior_notes_for_engine()`.
7. Add frontend tests.
8. Add Python binding and tests.

## 9. Testing Plan

Backend tests:

- Behavior-note count query returns the expected count.
- Behavior-note value query returns expected notes.
- Empty note list returns count zero.
- Wrong attribute type fails.
- Insufficient output count fails.
- Invalid raw enum value in `EngineDetails` fails during `EngineDescriptor::finalize()`.

FlatBuffers and plugin compatibility tests:

- Old-shape `EngineDetails` bytes containing only `engine_id` and `knobs` read as an empty behavior-note list.
- New `EngineDetails` buffers with behavior notes read correctly.
- A sample or metadata test plugin can emit empty and non-empty note lists.

Frontend tests:

- Query before graph finalization fails.
- Valid engine ID returns expected notes.
- Valid engine with no notes returns empty vector.
- Invalid or inapplicable engine ID fails.
- Output vector is cleared on entry for success, empty success, and failure.

Python tests:

- `BehaviorNote` enum is exposed.
- `get_behavior_notes_for_engine(engine_id)` returns expected enum values.

## 10. Future Considerations

Future work may include:

- `select_behavior_notes()` and `deselect_behavior_notes()` candidate filtering.
- `get_behavior_notes()` for a selected plan.
- `get_behavior_notes_for_plan_at_index()` if the frontend stores multiple candidate plans.
- Numerical-note support using the same per-engine metadata pattern.

These are not part of this RFC.

## 11. Glossary

- **Behavior note**: Advisory metadata describing non-numerical execution behavior of an engine.
- **EngineDetails**: Plugin-provided FlatBuffer metadata for an engine.
- **Per-engine query**: A query that asks for metadata for a specific engine ID rather than the currently selected plan.
