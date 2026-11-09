# hipDNN - Engine Configuration Knobs Design Document

- Contributors: Mitch Ousdahl
- **Status**: Implemented
- **Status notes**: Schemas, backend knob descriptors, frontend `Knob`/`KnobSetting`/Graph APIs, and MIOpen knobs shipped (#3802, #3923, #3901, #5057, #4328). Plugins define knobs with `KnobFactory` and `GlobalKnobDefines.hpp` in `plugin_sdk`. The frontend validates knob values; `hipdnnBackendFinalize(engineCfg)` does not.
- **Last verified**: 2026-10-15 against `develop` (2501d0b28cc)

**User Documentation**: For practical usage information, see:
- [hipDNN Knobs Documentation](../Knobs.md) - Complete user guide with examples
- [HowTo Guide - Configuring Knobs](../HowTo.md#configuring-engine-knobs) - Quick start guide
- [Plugin Development - Providing Knobs](../PluginDevelopment.md#providing-knobs) - Plugin developer guide
- [MIOpen Provider Knobs](../../../../dnn-providers/miopen-provider/docs/Knobs.md) - MIOpen-specific knobs

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [System Context](#3-system-context)
4. [Proposed Design](#4-proposed-design)
5. [Key Design Decisions](#5-key-design-decisions)
6. [Risks](#6-risks)
7. [Execution Plan](#7-execution-plan)
8. [Testing Plan](#8-testing-plan)
9. [Future Considerations](#9-future-considerations)
10. [Glossary](#10-glossary)
11. [Implementation Reference](#11-implementation-reference)

## 1. Executive Summary

This RFC proposes a flexible engine configuration knobs system for hipDNN that allows plugin developers to expose custom runtime settings and enables end-users to adjust these settings. Unlike a more limited int64_t-based knob system with min/max/stride constraints, this design leverages Flatbuffers' union types to support multiple value types (integers, floats, strings) while maintaining type safety.

The knobs system is designed to be:
- **Optional**: Plugins can opt-in to exposing knobs
- **Flexible**: Support for multiple data types beyond int64_t
- **Namespace-safe**: Plugin-specific human-readable knob identifiers
- **Extensible**: New knob types can be added without breaking existing code

## 2. Problem Statement

### 2.1 Current Limitations

Currently, hipDNN engines have limited runtime configurability. While the backend supports the concept of engine configurations, there is no standardized mechanism for:

1. **Plugin developers** to expose custom tuning parameters (e.g., tile sizes, algorithmic choices, memory layout preferences)
2. **End-users** to discover available settings for a given engine

### 2.2 The int64 Range Model and Its Limitations

A cuDNN-style knob model describes each knob with an enumerated knob type and three int64 values: a minimum, a maximum, and a stride (step) between valid values.

**Limitations:**
1. **Type Restriction**: All knob values must be int64_t, making it awkward for floating-point parameters or other choices
2. **Forced Range Semantics**: Every knob must specify min/max/stride, even when these concepts don't apply
3. **Global Namespace**: Knob types are globally enumerated, making it difficult for custom plugins to add new knobs without upstream changes
4. **Limited Expressiveness**: Cannot represent constraints such as "valid values are {1, 4, 8, 16}"

### 2.3 Requirements

A robust knobs system for hipDNN must support:

1. **Multiple Value Types**: int64, float64, and string types
2. **Flexible Validation**: Per-knob validation rules appropriate to the value type
3. **Plugin Autonomy**: Plugins can define custom knobs without modifying core hipDNN
4. **Namespace Isolation**: Different plugins can have knobs with the same semantic meaning without conflicts
5. **Discovery**: Users can query available knobs and their valid ranges/values

## 3. System Context

### 3.1 Backend Attributes

The knobs system uses the cuDNN-compatible backend attributes for engine knobs:

```cpp
// Backend attributes (from HipdnnBackendAttributeName.h)
HIPDNN_ATTR_ENGINE_KNOB_INFO       = 1002,  // Query knob metadata
HIPDNN_ATTR_ENGINECFG_KNOB_CHOICES = 202,   // Set knob values
```

The `EngineDetails` and `EngineConfig` flatbuffers carry the engine ID between hipDNN and plugins. The knobs system adds knob metadata to `EngineDetails` and knob values to `EngineConfig` (see [4.1](#41-flatbuffer-schema-design)).

### 3.2 Engine Configuration Workflow

The engine configuration workflow is:

1. User queries available engines for an operation graph
2. User selects an engine by ID
3. User queries the knobs that the engine supports
4. User creates an EngineConfig with the engine ID and optional knob settings
5. User finalizes an ExecutionPlan with the EngineConfig
6. User executes the plan

### 3.3 Plugin SDK Context

The Plugin SDK (RFC 0002) provides the infrastructure for plugins to implement engines. The knobs system integrates with this architecture as follows:

- Plan builders declare their knobs through `IPlanBuilder::getCustomKnobs()`
- `KnobFactory` builds knob flatbuffers with value constraints
- Engines return knob metadata in `EngineDetails.knobs`
- Plan builders read knob settings from the engine config

## 4. Proposed Design

### 4.1 Flatbuffer Schema Design

We propose using Flatbuffers' union types to create a flexible, type-safe knob system. The schemas live in `flatbuffers_sdk/schemas/` under namespace `hipdnn_flatbuffers_sdk.data_objects`.

#### 4.1.1 Core Knob Schema

```flatbuffers
// flatbuffers_sdk/schemas/knob_value.fbs
namespace hipdnn_flatbuffers_sdk.data_objects;

table IntValue {
    value: int64;
}

table FloatValue {
    value: float64;
}

table StringValue {
    value: string;
}

// Union of possible knob value types
union KnobValue {
    IntValue,
    FloatValue,
    StringValue,
}

table IntConstraint {
    min_value: int64;
    max_value: int64;
    step: int64 = 1;              // Step size (default 1)
    valid_values: [int64];        // Optional: explicit list of valid values
}

table FloatConstraint {
    min_value: float64;
    max_value: float64;
}

table StringConstraint {
    max_length: int32;            // Maximum string length
    valid_values: [string];       // Optional: enum-like string choices
}

// Type-specific constraint
union KnobConstraint {
    IntConstraint,
    FloatConstraint,
    StringConstraint,
}

// Knob metadata
table Knob {
    knob_id: string;                      // Unique identifier (namespaced by plugin)
    description: string;                  // Help text
    default_value: KnobValue (required);  // Default setting - must be provided (not NONE)
    constraint: KnobConstraint;           // Validation rules - may be NONE (unconstrained)
    deprecated: bool;                     // Allow knob deprecation
}
```

The type of `default_value` sets the value type of the knob. An integer knob uses `valid_values` for constraints that a range cannot express. For example, a "power of 2" knob uses `valid_values = {8, 16, 32, 64, 128}`. When `valid_values` is not empty, validation uses it and ignores `min_value`, `max_value`, and `step`.

#### 4.1.2 Updated Engine Schemas

```flatbuffers
// flatbuffers_sdk/schemas/engine_details.fbs
include "knob_value.fbs";

namespace hipdnn_flatbuffers_sdk.data_objects;

table EngineDetails {
    engine_id: int64;
    knobs: [Knob];             // Knobs this engine supports
    behavior_notes: [int32];
    name: string;
}

// flatbuffers_sdk/schemas/engine_config.fbs
include "knob_value.fbs";

namespace hipdnn_flatbuffers_sdk.data_objects;

table KnobSetting {
    knob_id: string;
    value: KnobValue;
}

table EngineConfig {
    engine_id: int64;
    knobs: [KnobSetting];      // User-specified knob values
}
```

### 4.2 Backend API Extensions

#### 4.2.1 Knob Discovery

Users query knob metadata from a finalized engine descriptor. `HIPDNN_ATTR_ENGINE_KNOB_INFO` returns one `HIPDNN_BACKEND_KNOB_INFO_DESCRIPTOR` for each knob:

```cpp
// Query the number of knobs
int64_t knobCount = 0;
hipdnnBackendGetAttribute(
    engine,
    HIPDNN_ATTR_ENGINE_KNOB_INFO,
    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
    0,
    &knobCount,
    nullptr
);

// Get the knob info descriptors
std::vector<hipdnnBackendDescriptor_t> knobInfos(knobCount);
hipdnnBackendGetAttribute(
    engine,
    HIPDNN_ATTR_ENGINE_KNOB_INFO,
    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
    knobCount,
    &knobCount,
    knobInfos.data()
);
```

Each knob info descriptor exposes these attributes:

| Attribute | Value |
|-----------|-------|
| `HIPDNN_ATTR_KNOB_INFO_TYPE` (900) | Knob ID string (`HIPDNN_TYPE_CHAR`) |
| `HIPDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE` (901) | Maximum value (int64 or double) |
| `HIPDNN_ATTR_KNOB_INFO_MINIMUM_VALUE` (902) | Minimum value (int64 or double) |
| `HIPDNN_ATTR_KNOB_INFO_STRIDE` (903) | Step size for integer knobs |
| `HIPDNN_ATTR_KNOB_INFO_DESCRIPTION` (904) | Description string |
| `HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE` (905) | Default value (int64, double, or char) |
| `HIPDNN_ATTR_KNOB_INFO_DEPRECATED` (906) | Deprecated flag |
| `HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_INT` (907) | Explicit valid integer values |
| `HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_STRING` (908) | Explicit valid string values |
| `HIPDNN_ATTR_KNOB_INFO_STRING_MAX_LENGTH` (909) | Maximum string length |
| `HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE_TYPE` (910) | Attribute type to use for the default value |

The extension attribute `HIPDNN_ATTR_KNOB_INFO_SERIALIZED_VALUE` on the engine descriptor returns each knob as a serialized `Knob` flatbuffer (`HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT`).

#### 4.2.2 Knob Configuration

Users set knob values in the engine config with `HIPDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR` descriptors. Each knob choice descriptor holds `HIPDNN_ATTR_KNOB_CHOICE_KNOB_TYPE` (the knob ID string) and `HIPDNN_ATTR_KNOB_CHOICE_KNOB_VALUE` (an int64, double, or char value). Finalize each knob choice descriptor before you set it on the engine config:

```cpp
// Create and finalize one knob choice descriptor
hipdnnBackendDescriptor_t knobChoice;
hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR, &knobChoice);

const std::string knobId = "global.workspace_size_limit";
hipdnnBackendSetAttribute(
    knobChoice,
    HIPDNN_ATTR_KNOB_CHOICE_KNOB_TYPE,
    HIPDNN_TYPE_CHAR,
    static_cast<int64_t>(knobId.size()),
    knobId.c_str()
);

int64_t limit = 64 * 1024 * 1024;
hipdnnBackendSetAttribute(
    knobChoice,
    HIPDNN_ATTR_KNOB_CHOICE_KNOB_VALUE,
    HIPDNN_TYPE_INT64,
    1,
    &limit
);

hipdnnBackendFinalize(knobChoice);

// Create engine config
hipdnnBackendDescriptor_t engineCfg;
hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, &engineCfg);

// Set engine
hipdnnBackendSetAttribute(
    engineCfg,
    HIPDNN_ATTR_ENGINECFG_ENGINE,
    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
    1,
    &engine
);

// Set knob choices
hipdnnBackendSetAttribute(
    engineCfg,
    HIPDNN_ATTR_ENGINECFG_KNOB_CHOICES,
    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
    1,
    &knobChoice
);

hipdnnBackendFinalize(engineCfg);
```

The extension attribute `HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE` on the engine config accepts serialized `KnobSetting` flatbuffers (`HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT`). `KnobSettingFactory` in `plugin_sdk` builds these buffers.

#### 4.2.3 Validation

The frontend validates knob settings. The backend does not check knob constraints. During `hipdnnBackendFinalize(engineCfg)`, the backend passes the serialized `EngineConfig`, with its knob settings, to the plugin to get the workspace size.

The frontend validates each setting with `Knob::validate()`, which checks the setting against the knob constraint:

- `Graph::create_execution_plan_ext()` ignores a setting for an unknown knob and logs a warning. It logs a warning for a deprecated knob. It returns an error for a value that fails the knob constraint.
- `Graph::add_engine()` and `Graph::add_engine_variants()` return an error for an unknown knob or an invalid value. They remove `global.benchmarking`, because `Graph::autotune()` controls that knob.

Plugins apply default values for knobs that the user does not set. Plugins also check combinations of knob values that a single-knob constraint cannot express.

> [!NOTE]
> `create_execution_plan_ext()` does not fail for a misspelled or unsupported knob ID. It only logs a warning and drops the setting. Use `add_engine()` to get an error instead.

### 4.3 Plugin SDK Extensions

#### 4.3.1 Knob Definition Helpers

`hipdnn_plugin_sdk::KnobFactory` (`plugin_sdk/include/hipdnn_plugin_sdk/KnobFactory.hpp`) builds `Knob` flatbuffers with a default value and a constraint:

```cpp
namespace hipdnn_plugin_sdk {

class KnobFactory {
public:
    static flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Knob>
        createIntKnob(flatbuffers::FlatBufferBuilder& builder,
                      const std::string& name,
                      const std::string& description,
                      int64_t defaultValue,
                      int64_t min,
                      int64_t max,
                      int64_t step,
                      const std::vector<int64_t>& options,
                      bool deprecated = false);

    static flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Knob>
        createFloatKnob(flatbuffers::FlatBufferBuilder& builder,
                        const std::string& name,
                        const std::string& description,
                        float defaultValue,
                        float min,
                        float max,
                        bool deprecated = false);

    static flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Knob>
        createStringKnob(flatbuffers::FlatBufferBuilder& builder,
                         const std::string& name,
                         const std::string& description,
                         const std::string& defaultValue,
                         const std::vector<std::string>& options,
                         bool deprecated = false);
};

} // namespace hipdnn_plugin_sdk
```

`plugin_sdk/include/hipdnn_plugin_sdk/GlobalKnobDefines.hpp` defines the names of the global knobs (`BENCHMARKING_KNOB_NAME`, `WORKSPACE_SIZE_LIMIT_KNOB_NAME`).

#### 4.3.2 Exposing Knobs

A plan builder declares its knobs with `IPlanBuilder::getCustomKnobs()`. It returns the knobs as `KnobT` objects:

```cpp
// plugin_sdk/include/hipdnn_plugin_sdk/interfaces/IPlanBuilder.hpp
virtual std::vector<hipdnn_flatbuffers_sdk::data_objects::KnobT>
    getCustomKnobs(const THandle& handle,
                   const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
    = 0;
```

The engine collects these knobs, adds any engine-wide knobs, and returns them in `EngineDetails.knobs`. For example, the MIOpen engine adds `global.benchmarking` to the knobs of the applicable plan builder (`dnn-providers/miopen-provider/engines/MiopenEngine.cpp`).

#### 4.3.3 Reading Knob Settings

Plan builders read knob settings from the engine config through `IEngineConfig` (`flatbuffers_sdk/include/hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp`). `hasKnobSetting()` tells if the user set a knob. `getKnobSettingByName()` returns the setting. If the user did not set a knob, the plan builder uses the knob default value.

#### 4.3.4 Example Plugin Usage

This example is from `samples/example_engine_plugin/src/engines/plans/ReluPlanBuilder.cpp`:

```cpp
// Declare a float knob with range [0.0, 1.0]
std::vector<hipdnn_flatbuffers_sdk::data_objects::KnobT> ReluPlanBuilder::getCustomKnobs(
    const ExampleProviderHandle& /*handle*/,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& /*opGraph*/) const
{
    std::vector<hipdnn_flatbuffers_sdk::data_objects::KnobT> knobs;

    hipdnn_flatbuffers_sdk::data_objects::KnobT knob;
    knob.knob_id = "example.relu.negative_slope";
    knob.description = "Negative slope for leaky ReLU (0.0 = standard ReLU)";

    hipdnn_flatbuffers_sdk::data_objects::FloatValueT defaultValue;
    defaultValue.value = 0.0;
    knob.default_value.Set(defaultValue);

    hipdnn_flatbuffers_sdk::data_objects::FloatConstraintT constraint;
    constraint.min_value = 0.0;
    constraint.max_value = 1.0;
    knob.constraint.Set(constraint);

    knobs.push_back(std::move(knob));
    return knobs;
}

// Read the knob setting if the user set it
if(engineConfig.hasKnobSetting("example.relu.negative_slope"))
{
    const auto& knobSetting = engineConfig.getKnobSettingByName("example.relu.negative_slope");
    const auto& floatVal
        = knobSetting.valueAs<hipdnn_flatbuffers_sdk::data_objects::FloatValue>();
    executionSettings.reluNegativeSlope = floatVal.value();
}
```

### 4.4 Frontend API Extensions

The frontend provides extension methods that align with the frontend API pattern while adding hipDNN-specific functionality. The frontend reads knobs through the backend knob descriptors (see [4.2](#42-backend-api-extensions)).

```cpp
// frontend/include/hipdnn_frontend/knob/KnobSetting.hpp
namespace hipdnn_frontend {

using KnobValueVariant = std::variant<int64_t, double, std::string>;
typedef std::string KnobType_t;

// Knob setting class - represents a configured knob value
class KnobSetting {
public:
    KnobSetting(std::string knobId, KnobValueVariant value);
    template <typename T> KnobSetting(std::string knobId, const T& value);

    const std::string& knobId() const;
    const KnobValueVariant& value() const;
    template <typename T> void setValue(const T& value);
};

// frontend/include/hipdnn_frontend/knob/KnobConstraint.hpp
enum class ConstraintKind { INT, FLOAT, STRING, EMPTY };

class IConstraint {
public:
    virtual ConstraintKind kind() const = 0;
    virtual Error validateKnobSetting(const KnobSetting& setting) const = 0;
    virtual std::string toString() const = 0;
};

class IntConstraint;     // min, max, step, optional valid values
class FloatConstraint;   // min, max
class StringConstraint;  // max length, optional valid values
class EmptyConstraint;   // no constraint

// frontend/include/hipdnn_frontend/knob/Knob.hpp
// Knob information class - describes available knobs for an engine
class Knob {
public:
    static std::pair<Error, Knob> tryCreate(std::string knobIdStr,
                                            std::string description,
                                            KnobValueVariant defaultValue,
                                            bool deprecated,
                                            std::shared_ptr<IConstraint> constraint
                                            = std::make_shared<EmptyConstraint>());

    const std::string& knobId() const;
    const std::string& description() const;
    bool isDeprecated() const;
    KnobValueType valueType() const;   // INT64, FLOAT64, or STRING
    const KnobValueVariant& defaultValue() const;
    const IConstraint* constraint() const;
    Error validate(const KnobSetting& setting) const;
};

} // namespace hipdnn_frontend

// frontend/include/hipdnn_frontend/Graph.hpp
namespace hipdnn_frontend::graph {

class Graph : public INode {
public:
    // Existing methods...

    Error get_ranked_engine_ids(std::vector<int64_t>& rankedEngineIds,
                                const std::vector<HeuristicMode>& modes = {HeuristicMode::FALLBACK});

    Error get_knobs_for_engine(int64_t engineId, std::vector<Knob>& knobs) const;
    Error get_knob_lookup_for_engine(int64_t engineId,
                                     std::unordered_map<KnobType_t, Knob>& knobs) const;

    Error create_execution_plan_ext(int64_t engineId, const std::vector<KnobSetting>& settings);

    Error add_engine(int64_t engineId, const std::vector<KnobSetting>& knobSettings = {});
    Error add_engine_variants(const std::vector<EngineVariant>& variants);
};

} // namespace hipdnn_frontend::graph
```

The `_ext` suffix marks `create_execution_plan_ext()` as a hipDNN extension to the frontend API. It takes a vector of `KnobSetting`, so one method accepts all value types.

#### Usage Example

This example follows `samples/knobs/KnobsUsage.cpp`:

```cpp
using namespace hipdnn_frontend;

// Create and build graph
graph::Graph graph;

// ... setup graph ...

std::vector<int64_t> engineIds;
graph.get_ranked_engine_ids(engineIds);
const int64_t engineId = engineIds[0];

// Discover knobs
std::vector<Knob> knobs;
graph.get_knobs_for_engine(engineId, knobs);

// Option 1: Use default knob values (no settings)
graph.create_execution_plan_ext(engineId, {});

// Option 2: Customize specific knobs
std::vector<KnobSetting> settings;
settings.emplace_back("global.benchmarking", static_cast<int64_t>(1));
settings.emplace_back("global.workspace_size_limit", static_cast<int64_t>(64 * 1024 * 1024));
graph.create_execution_plan_ext(engineId, settings);
```

The extension methods provide:
- **Richer metadata**: `Knob` includes descriptions, constraints, and default values
- **Type safety**: `KnobSetting` holds one of three value types in a `std::variant`
- **Flexibility**: Support for several value types beyond int64_t
- **Early Validation**: Since we know the constraints for the knobs, we can error out early if they defy constraints

### 4.5 Knob Naming Convention

We recommend a hierarchical naming scheme:

```
<plugin_name>.<engine_name>.<knob_name>

Examples:
- example.relu.negative_slope
- test.engine_b.block_size
- test.engine_b.algorithm
```

The plugin name prefix ensures that different plugins can have semantically similar knobs.

The "global." namespace contains commonly used knobs. `GlobalKnobDefines.hpp` defines them:

```
- global.benchmarking
- global.workspace_size_limit
```

Plugins must use the `global.` prefix only for these knobs. The code does not reject a custom knob in the `global.` namespace.

hipDNN defines no `global.deterministic` knob. The MIOpen provider supplies deterministic behavior as a separate engine (`MIOPEN_ENGINE_DETERMINISTIC`).

## 5. Key Design Decisions

### 5.1 Flatbuffer Unions vs. Original Approach

**Decision**: Use Flatbuffer union types instead of an int64_t-only approach.

**Rationale**:
- **Type Safety**: Unions provide compile-time and runtime type checking
- **Expressiveness**: Can represent string parameters, floating-point parameters, and integers naturally
- **Flexibility**: Easy to add new value types without breaking existing code
- **Efficiency**: Flatbuffers are zero-copy and highly efficient
- **Validation**: Type-specific constraints are more intuitive (e.g., enum valid values vs. min/max/stride)

**Trade-off**: Slightly more complex schema, but significantly better developer experience.

### 5.2 Optional Plugin Support

**Decision**: Knobs are optional. An engine that has no knobs returns an empty `knobs` list in `EngineDetails`.

**Rationale**:
- **Backward Compatibility**: Plugins that expose no knobs work without modification
- **Simplicity**: Simple engines don't need to expose knobs
- **Flexibility**: Plugins can add knob support incrementally

Users can also ignore knobs. Every knob has a required default value, and the plugin uses it when the user sets no value.

### 5.3 Namespace-Based Knob IDs

**Decision**: Use string-based hierarchical knob IDs instead of global enumerations. Knob IDs stay strings in the schemas, the backend, and the frontend. hipDNN does not hash them to integers.

**Rationale**:
- **Plugin Autonomy**: Plugins can define custom knobs without modifying core hipDNN
- **Readability**: String IDs are self-documenting and appear directly in logs
- **Extensibility**: New plugins can be added without coordination

**Trade-off**: String comparison is slower than integer comparison, but knob operations are infrequent (configuration time, not execution time).

### 5.4 Validation in the Frontend

**Decision**: The frontend validates knob settings against the knob constraints before it creates the engine config. See [4.2.3 Validation](#423-validation).

**Rationale**:
- **Early Error Detection**: Catch invalid settings before execution
- **Clear Error Messages**: `Knob::validate()` returns an error that describes the failed constraint
- **Plugin Responsibility**: Plugins check combinations of knob values and apply defaults

## 6. Risks

### 6.1 API Complexity

**Risk**: The knobs API adds complexity to the already extensive backend API.

**Mitigation**:
- Provide high-level frontend wrappers for common use cases
- Create comprehensive examples and documentation
- Make knobs optional (plugins and users can ignore if not needed)
- Provide sensible defaults so knobs are only needed for tuning

### 6.2 Performance Overhead

**Risk**: Knob validation could add overhead.

**Mitigation**:
- Validation occurs once, when the frontend creates the engine config
- Flatbuffers are zero-copy and highly efficient
- Knob settings are applied once per engine config, not per execution

### 6.3 Plugin Compatibility

**Risk**: Plugins may expose incompatible knobs.

**Mitigation**:
- Provide `KnobFactory` and `GlobalKnobDefines.hpp` in the Plugin SDK
- Document best practices for knob design

### 6.4 Testing Complexity

**Risk**: Testing all combinations of knob values is impractical.

**Mitigation**:
- Focus testing on boundary conditions and invalid values

## 7. Execution Plan

### Phase 1: Core Infrastructure

1. **Flatbuffer Schema Design**
   - Create `knob_value.fbs` with union types for KnobValue and KnobConstraint
   - Update `engine_config.fbs` to include KnobSetting array
   - Update `engine_details.fbs` to include Knob array
   - Generate C++ code from schemas using flatc
   - Write unit tests for schema handling

2. **Backend API Implementation**
   - Add `HIPDNN_BACKEND_KNOB_INFO_DESCRIPTOR` and `HIPDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR`
   - Add `HIPDNN_ATTR_ENGINE_KNOB_INFO` support in EngineDescriptor
   - Add `HIPDNN_ATTR_ENGINECFG_KNOB_CHOICES` support in EngineConfigDescriptor
   - Add the serialized extension attributes `HIPDNN_ATTR_KNOB_INFO_SERIALIZED_VALUE` and `HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE`
   - Write backend unit tests for knob descriptors

### Phase 2: Frontend API

5. **Knob Class Implementation**
   - Implement `Knob` class built from backend knob info descriptors
   - Add accessors for knob metadata (ID, description, default value, deprecated flag)
   - Implement constraint classes (`IntConstraint`, `FloatConstraint`, `StringConstraint`, `EmptyConstraint`)
   - Add `Knob::validate()`
   - Write unit tests for Knob class

6. **KnobSetting Class Implementation**
   - Implement `KnobSetting` class with type-safe constructors
   - Validate settings against Knob constraints in the Graph methods (and warn on deprecated knobs)
   - Write unit tests for KnobSetting class

7. **Graph Extension Methods**
   - Implement `Graph::get_knobs_for_engine()` and `Graph::get_knob_lookup_for_engine()`
     - Query backend for engine knob info descriptors
     - Return Knob objects
   - Implement `Graph::create_execution_plan_ext(int64_t engineId, const std::vector<KnobSetting>&)`
     - Validate knob settings
     - Create knob choice descriptors and an EngineConfig with them
     - Create the ExecutionPlan
   - Write frontend integration tests

8. **Documentation**
   - Write user guide for knobs API
   - Create plugin developer guide for knob registration
       - Define some best practices
           - Document how execution plans should and shouldn't be serialized, regarding how it interacts with knob settings
           - If a knob setting isn't supplied by the user, it's up to the plugin author to supply default values
           - If a knob is no longer needed, deprecate it rather than removing it.  There is a deprecated flag on knobs that can be used.  Deprecated knobs can be removed during a major version update.
   - Document naming conventions and best practices
   - Provide code examples (`samples/knobs/KnobsUsage.cpp`)

### Phase 3: Plugin Migration

9. **MIOpen Plugin Update**
   - Identify tunable parameters in MIOpen engines
   - Implement knobs with `KnobFactory` and plan builder `getCustomKnobs()`
   - Add plugin-specific tests for knob functionality

### Phase 4: Testing & Refinement

11. **Integration Testing**
    - End-to-end tests with real plugins and knobs
    - Test usage patterns (defaults, customization)
    - Error handling and validation edge cases

12. **Documentation & Review**
    - Complete API reference documentation
    - Create tutorial examples showing common workflows

## 8. Testing Plan

### 8.1 Unit Tests

#### Flatbuffer Schema Tests
- Handle each knob value type
- Validate constraint handling
- Test union type discrimination
- Verify default value handling

#### Validation Tests
- Valid values pass validation
- Invalid values fail with appropriate errors
- Boundary conditions (min/max values)
- Type mismatches detected

### 8.2 Integration Tests

#### Backend Integration
- Query knobs from engine descriptors
- Set knobs in engine config
- Error handling and reporting

#### Plugin SDK Integration
- Engines expose knobs
- Knob settings reach the plan builder
- Multiple engines with different knobs

#### Frontend Integration
- Type-safe knob setting
- Knob value retrieval
- Error propagation to user

### 8.3 Plugin-Specific Tests

#### MIOpen Plugin
- Verify all exposed knobs
- Test knob application affects behavior

## 9. Future Considerations

### 9.1 Knob identifier headers

`GlobalKnobDefines.hpp` in `plugin_sdk` defines the global knob names for plugins. The frontend has no public header for knob names.

- **Global Knob Identifier Header** - A header that lives in the frontend that defines an enum or consts for common knob types, preventing the need for knowing "special strings"
- **Plugin Knob Identifier Header** - Plug devs could have their plugin knob id consts / enums in a header that installs to a known frontend include location, providing the same features as the Global Knob Identifier Header

## 10. Glossary

- **Knob**: A runtime-configurable parameter that affects engine behavior
- **Knob ID**: A unique string identifier for a knob (e.g., "global.benchmarking")
- **Knob Value**: The actual setting for a knob (int64, double, or string)
- **Knob Constraint**: Validation rules for a knob (min/max, valid values, etc.)
- **Engine Config**: A configuration object that includes engine ID and knob settings
- **Engine Details**: Metadata about an engine, including supported knobs
- **Knob Setting**: A (knob_id, value) pair in an engine configuration
- **Namespace**: A prefix for knob IDs to prevent conflicts (e.g., "example.")
- **Validation**: Checking that knob values satisfy constraints
- **Finalization**: The process of locking a backend descriptor configuration
- **Auto-Tuning**: Automatically searching for optimal knob values

## 11. Implementation Reference

### 11.1 Components

1. **FlatBuffer Schemas** (Phase 1.1)
   - `flatbuffers_sdk/schemas/knob_value.fbs` - Core knob schema with union types
   - `engine_details.fbs` - Knob metadata
   - `engine_config.fbs` - Knob settings
   - All schemas support int64, float64, and string value types

2. **Backend API** (Phase 1.2)
   - `HIPDNN_ATTR_ENGINE_KNOB_INFO` - Query knob metadata as `HIPDNN_BACKEND_KNOB_INFO_DESCRIPTOR` descriptors
   - `HIPDNN_ATTR_KNOB_INFO_SERIALIZED_VALUE` - Query knob metadata as serialized flatbuffers
   - `HIPDNN_ATTR_ENGINECFG_KNOB_CHOICES` - Set knob values as `HIPDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR` descriptors
   - `HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE` - Set knob values as serialized flatbuffers

3. **Frontend API** (Phase 2)
   - `Knob` class - Wraps knob metadata with type-safe accessors and `validate()`
   - `KnobSetting` class - Type-safe knob value setting
   - Constraint classes: `IntConstraint`, `FloatConstraint`, `StringConstraint`, `EmptyConstraint`
   - `Graph::get_knobs_for_engine()` - Query available knobs
   - `Graph::get_knob_lookup_for_engine()` - Query knobs as map
   - `Graph::create_execution_plan_ext()` - Create plan with knob settings
   - `Graph::add_engine()` - Add a plan spec with knob settings

4. **Plugin SDK**
   - `KnobFactory` - Build `Knob` flatbuffers
   - `KnobSettingFactory` - Build serialized `KnobSetting` flatbuffers
   - `GlobalKnobDefines.hpp` - Global knob names

5. **MIOpen Provider** (Phase 3)
   - `global.benchmarking` knob on all engines
   - `global.workspace_size_limit` knob for convolution operations
   - Plan builders add knobs for their operation type
   - Plan builders read knob settings from the engine config

6. **Documentation** (Phase 4)
   - Core hipDNN knobs documentation
   - User guide (HowTo) with quick start
   - Plugin development guide for knob providers
   - MIOpen provider knobs documentation with examples

### 11.2 Tests

**Unit Tests**:
- `frontend/tests/TestKnob.cpp` - Knob, KnobSetting, and constraint class tests
- `frontend/tests/TestKnobUnpacker.cpp` - Knob descriptor unpacking tests
- `frontend/tests/TestGraph.cpp` - Graph knob API tests
- `backend/tests/descriptors/TestKnobDescriptor.cpp`, `TestKnobSettingDescriptor.cpp` - Backend descriptor tests
- `plugin_sdk/tests/TestKnobFactory.cpp`, `TestKnobSettingFactory.cpp` - Plugin SDK helper tests

**Integration Tests**:
- `tests/frontend/IntegrationGraphKnobsApi.cpp` - End-to-end frontend knob tests
- `tests/backend/IntegrationKnobsApi.cpp` - Backend knob API tests
- `tests/test_plugins/TestKnobsPlugin.cpp`, `TestKnobConstraintValidationPlugin.cpp` - Test plugins that expose knobs

**Plugin Tests**:
- `dnn-providers/miopen-provider/integration_tests/IntegrationGpuBenchmarkingKnob.cpp` - MIOpen benchmarking knob tests
- `dnn-providers/miopen-provider/tests/engines/plans/TestMiopenConvPlanBuilder.cpp` - MIOpen workspace limit knob tests

### 11.3 Knob Users

1. **MIOpen Provider**
   - `global.benchmarking` - Enable/disable kernel benchmarking
   - `global.workspace_size_limit` - Control convolution workspace memory

2. **Autotuning**
   - `Graph::autotune()` and `Graph::add_engine_sweep()` sweep knob values and time candidate plans. See [RFC 0013](./0013_Autotune.md) and `frontend/include/hipdnn_frontend/autotune/`.

3. **Samples and Test Plugins**
   - `samples/knobs/KnobsUsage.cpp` - Frontend knob usage
   - `samples/example_engine_plugin/` - Plugin knob example
   - `tests/test_plugins/` - Test plugins that expose knobs

### 11.4 Related Documentation

For implementation details and usage examples, refer to:

- **User Documentation**:
  - [hipDNN Knobs](../Knobs.md) - Complete user guide
  - [HowTo - Configuring Knobs](../HowTo.md#configuring-engine-knobs) - Quick start
  - [MIOpen Knobs](../../../../dnn-providers/miopen-provider/docs/Knobs.md) - Provider-specific knobs

- **Developer Documentation**:
  - [Plugin Development - Providing Knobs](../PluginDevelopment.md#providing-knobs) - Plugin author guide

- **Implementation Files**:
  - `frontend/include/hipdnn_frontend/knob/Knob.hpp`
  - `frontend/include/hipdnn_frontend/knob/KnobSetting.hpp`
  - `frontend/include/hipdnn_frontend/knob/KnobConstraint.hpp`
  - `flatbuffers_sdk/schemas/knob_value.fbs`
  - `plugin_sdk/include/hipdnn_plugin_sdk/KnobFactory.hpp`
  - `plugin_sdk/include/hipdnn_plugin_sdk/GlobalKnobDefines.hpp`

---

**Last Updated**: 2026-10-15
**Implemented By**: Mitch Ousdahl and contributors
