# hipDNN - Engine Configuration Knobs Design Document

- Contributors: Mitch Ousdahl
- **Status**: Implemented ✅
- **Implementation Version**: hipDNN 1.0+

> [!NOTE]
> **Implementation Status**: This RFC has been implemented in hipDNN. The design described here reflects the current implementation with minor deviations noted in sections below.
>
> **User Documentation**: For practical usage information, see:
> - [hipDNN Knobs Documentation](../Knobs.md) - Complete user guide with examples
> - [HowTo Guide - Configuring Knobs](../HowTo.md#configuring-engine-knobs) - Quick start guide
> - [Plugin Development - Providing Knobs](../PluginDevelopment.md#providing-knobs) - Plugin developer guide
> - [MIOpen Provider Knobs](../../../dnn-providers/miopen-provider/docs/Knobs.md) - MIOpen-specific knobs

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Current System Overview](#3-current-system-overview)
4. [Proposed Design](#4-proposed-design)
5. [Key Design Decisions](#5-key-design-decisions)
6. [Risks](#6-risks)
7. [Execution Plan](#7-execution-plan)
8. [Testing Plan](#8-testing-plan)
9. [Future Considerations](#9-future-considerations)
10. [Glossary](#10-glossary)
11. [Implementation Notes](#11-implementation-notes)

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

### 2.2 The Initial Approach and Its Limitations

The default implementation specifies a knob system with the following characteristics:

```cpp
typedef struct {
    hipdnnBackendKnobType_t type;  // Enumerated knob type
    int64_t minimum_value;         // Minimum valid value
    int64_t maximum_value;         // Maximum valid value
    int64_t step;                // Step size between valid values
} hipdnnBackendKnobInfo_t;
```

**Limitations:**
1. **Type Restriction**: All knob values must be int64_t, making it awkward for floating-point parameters or other choices
2. **Forced Range Semantics**: Every knob must specify min/max/stride, even when these concepts don't apply
3. **Global Namespace**: Knob types are globally enumerated, making it difficult for custom plugins to add new knobs without upstream changes
4. **Limited Expressiveness**: Cannot represent complex constraints (e.g., "value must be a power of 2" or "valid values are {1, 4, 8, 16}")

### 2.3 Requirements

A robust knobs system for hipDNN must support:

1. **Multiple Value Types**: int64, float64, and string types
2. **Flexible Validation**: Per-knob validation rules appropriate to the value type
3. **Plugin Autonomy**: Plugins can define custom knobs without modifying core hipDNN
4. **Namespace Isolation**: Different plugins can have knobs with the same semantic meaning without conflicts
5. **Discovery**: Users can query available knobs and their valid ranges/values

## 3. Current System Overview

### 3.1 Existing Infrastructure

hipDNN already has foundational support for engine configurations:

```cpp
// Backend attributes (from HipdnnBackendAttributeName.h)
HIPDNN_ATTR_ENGINE_KNOB_INFO       = 1302,  // Query knob metadata
HIPDNN_ATTR_ENGINECFG_KNOB_CHOICES = 302,   // Set knob values

// Flatbuffer schemas
// engine_config.fbs
table EngineConfig {
    engine_id: int64;
    // No knob support yet
}

// engine_details.fbs
table EngineDetails {
    engine_id: int64;
    // No knob metadata yet
}
```

### 3.2 Current Workflow

The current engine configuration workflow is:

1. User queries available engines for an operation graph
2. User selects an engine by ID
3. User creates an EngineConfig with the engine ID
4. User finalizes an ExecutionPlan with the EngineConfig
5. User executes the plan

**Missing**: Steps to discover, configure, and persist engine-specific settings.

### 3.3 Plugin SDK Context

The Plugin SDK (RFC 0002) provides the infrastructure for plugins to implement engines. The knobs system will integrate with this architecture by:

- Allowing `EngineBase` implementations to declare supported knobs
- Providing helper classes for knob validation
- Extending the plugin API to expose knob metadata

## 4. Proposed Design

### 4.1 Flatbuffer Schema Design

We propose using Flatbuffers' union types to create a flexible, type-safe knob system.

#### 4.1.1 Core Knob Schema

```flatbuffers
// data_sdk/schemas/knob_value.fbs
namespace hipdnn_data_sdk.data_objects;

// Union of possible knob value types
union KnobValue {
    IntValue,
    FloatValue,
    StringValue,
}

table IntValue {
    value: int64;
}

table FloatValue {
    value: float64;
}

table StringValue {
    value: string;
}

// Knob metadata for discovery
table Knob {
    knob_id: int64;               // Hashed from knob_id_str
    knob_id_str: string;          // Unique identifier (namespaced by plugin)
    description: string;          // Help text
    value_type: KnobValue;        // Discriminator for union
    default_value: KnobValue;     // Default setting
    constraints: KnobConstraint;  // Validation rules
    deprecated: bool;             // Allow knob deprecation
}

// Type-specific constraints
union KnobConstraint {
    IntConstraint,
    FloatConstraint,
    StringConstraint,
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
    max_length: int32;
    valid_values: [string];       // Optional: enum-like string choices
}
```

#### 4.1.2 Updated Engine Schemas

```flatbuffers
// data_sdk/schemas/engine_details.fbs
namespace hipdnn_data_sdk.data_objects;

include "knob_value.fbs";

table EngineDetails {
    engine_id: int64;
    knobs: [Knob];  // NEW: Knobs this engine supports
}

// data_sdk/schemas/engine_config.fbs
namespace hipdnn_data_sdk.data_objects;

include "knob_value.fbs";

table KnobSetting {
    knob_id: int64;
    value: KnobValue;
}

table EngineConfig {
    engine_id: int64;
    knobs: [KnobSetting];  // NEW: User-specified knob values
}
```

### 4.2 Backend API Extensions

#### 4.2.1 Knob Discovery

Users can query knob metadata from engine details:

```cpp
// Query engine details (existing API)
hipdnnBackendDescriptor_t engineDetails;
hipdnnBackendGetAttribute(
    engine,
    HIPDNN_ATTR_ENGINE_KNOB_INFO,
    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
    requestedCount,
    &actualCount,
    &engineDetails
);

// The engineDetails descriptor contains serialized EngineDetails flatbuffer
// which includes the supported_knobs array
```

#### 4.2.2 Knob Configuration

Users set knob values in engine config:

```cpp
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

// Set knob choices (serialized KnobSetting flatbuffers)
hipdnnBackendSetAttribute(
    engineCfg,
    HIPDNN_ATTR_ENGINECFG_KNOB_CHOICES,
    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
    numKnobs,
    knobSettingDescriptors
);

hipdnnBackendFinalize(engineCfg);
```

#### 4.2.3 Validation

During `hipdnnBackendFinalize(engineCfg)`, the backend will:

1. Deserialize the knob settings from the EngineConfig
2. Query the engine for its supported knobs
3. Validate each setting against the knob's constraints
4. Use defaults for any settings that weren't specified by the end-user
5. Return `HIPDNN_STATUS_BAD_PARAM` if validation fails

### 4.3 Plugin SDK Extensions

⚠️ These changes are predicated on the existence of the plugin helper classes as defined in [RFC2 - Plugin SDK Design](./0002_PluginSdkDesign.md)  As an interim step, flatbuffer wrapper classes can add as temporary aids to plugin developers.

#### 4.3.1 Knob Registration Interface

```cpp
// plugin_sdk/include/hipdnn_plugin_sdk/KnobRegistry.hpp
namespace hipdnn_plugin_sdk {

class KnobRegistry {
public:
    // Register an integer knob
    void registerIntKnob(
        const std::string& knobId,
        const std::string& description,
        int64_t defaultValue,
        int64_t minValue,
        int64_t maxValue,
        int64_t step = 1
    );

    // Register an integer knob with explicit valid values
    void registerIntKnob(
        const std::string& knobId,
        const std::string& description,
        int64_t defaultValue,
        const std::vector<int64_t>& validValues
    );

    // Register a float knob
    void registerFloatKnob(
        const std::string& knobId,
        const std::string& description,
        double defaultValue,
        double minValue,
        double maxValue
    );

private:
    std::vector<Knob> _knobs;
};

} // namespace hipdnn_plugin_sdk
```

#### 4.3.2 Engine Base Class Extension

```cpp
// plugin_sdk/include/hipdnn_plugin_sdk/EngineBase.hpp
namespace hipdnn_plugin_sdk {

class EngineBase : public IEngine {
public:
    // Existing methods...

    // NEW: Register some common global namespace knobs
    void registerDeterministicKnob();
    void registerBenchmarkingKnob();

    // NEW: Override to register knobs
    virtual void registerKnobs(KnobRegistry& registry) {
        // Default: no knobs
    }
    }

protected:
    KnobRegistry _knobRegistry;
};

} // namespace hipdnn_plugin_sdk
```

#### 4.3.3 PlanBuilderBase Class Extension
```cpp
// plugin_sdk/include/hipdnn_plugin_sdk/EngineBase.hpp
namespace hipdnn_plugin_sdk {

class PlanBuilderBase : public IPlanBuilder {
public:
    // Existing methods...

    virtual ExecutionContext initializeExecutionContext(handle, graph, engine_config) {

        // KnobSettings can be pulled from the engine_config, and resultant values should be stored
        // on the IPlan that is attached to the execution context

        // Knobs have already been validated against constraints on the frontend, so rather than
        // check them here, this is the place to validate logical knob combinations,
        // for example: knob A can only have a value of 1 if knob B is zero
    }
};

} // namespace hipdnn_plugin_sdk
```

#### 4.3.3 Example Plugin Usage

```cpp
// Example: Convolution engine with tile size knob
class ConvolutionEngine : public EngineBase {
public:
    void registerKnobs(KnobRegistry& registry) override {
        // Register tile size knob with power-of-2 values
        // Throw if the user tries to register a knob that starts with "global."
        registry.registerIntKnob(
            "miopen.conv.tile_size",
            "Size of computation tile (must be power of 2)",
            16,  // default
            {8, 16, 32, 64, 128}  // valid values
        );

        // Register the built-in deterministic knob
        // This will register a knob called global.deterministic
        registry.registerDeterministicKnob();
    }
};
```

### 4.4 Frontend API Extensions

The frontend will provide extension methods that align with the more limited frontend API pattern while adding hipDNN-specific functionality:

```cpp
// frontend/include/hipdnn_frontend/Graph.hpp
namespace hipdnn::frontend {

// Type information
enum class KnobValueType {
    INT64,
    FLOAT64,
    STRING,
};

// Knob information class - describes available knobs for an engine
class Knob {
public:
    Knob(const std::string& knobIdStr, // knob_id hashed from this string internally
         const std::string& description);

    // Accessors
    int64_t getKnobId() const; // A hash id of Knob ID String
    const std::string& getKnobIdStr() const;
    const std::string& getDescription() const;
    const bool isDeprecated() const;

    KnobValueType getValueType() const;

    // Get default value (type-specific)
    std::optional<int64_t> getDefaultInt() const;
    std::optional<double> getDefaultFloat() const;
    std::optional<std::string> getDefaultString() const;

    // Get constraints (type-specific)
    struct IntConstraint {
        int64_t minValue;
        int64_t maxValue;
        int64_t step;
        std::vector<int64_t> validValues;  // Optional explicit list
    };
    struct FloatConstraint {
        double minValue;
        double maxValue;
    };

    std::optional<IntConstraint> getIntConstraint() const;
    std::optional<FloatConstraint> getFloatConstraint() const;

    // Convert to Knob with default value
    KnobSetting toDefaultKnob() const;

    // flatbuffer pack and unpack methods
    pack()
    unpack()

    // Helper, to hash the string ID to the int ID
    static int64_t make_knob_id(const std::string& strID);
};

// Knob setting class - represents a configured knob value
class KnobSetting {
public:
    // Constructors for different value types
    KnobSetting(int64_t knobId, int64_t value);
    KnobSetting(int64_t knobId, double value);
    KnobSetting(int64_t knobId, const std::string& value);

    // Accessors
    int64_t getKnobId() const;
    Knob::ValueType getValueType() const;

    // Get value (type-specific)
    std::optional<int64_t> getIntValue() const;
    std::optional<double> getFloatValue() const;
    std::optional<std::string> getStringValue() const;

    // flatbuffer pack and unpack methods
    pack()
    unpack()

};

typedef int64_t KnobType_t;

class Graph {
public:
    // Existing methods... only supports int64_t values for knobs

    // The key is KnobID
    // Supplied for added compatibility
    error_t Graph::create_execution_plan(int64_t engineId,
                                         std::unordered_map<KnobType_t, int64_t> const &knobs) const;
    error_t Graph::get_knobs_for_engine(int64_t engineId,
                                        std::vector<Knob> &knobs) const;

    // New methods
    error_t Graph::create_execution_plan(int64_t engineId,
                                         std::unordered_map<int64_t, KnobSetting> const &user_knobs
                                         bool filterByKnobs = false) const;

    error_t Graph::get_ranked_engine_ids(std::vector<int64_t>& rankedEngineIds,
                                         const std::vector<HeuristicMode>& modes = {HeuristicMode::FALLBACK});

private:
    // Internal method for validating knob settings against knob constraints
    error_t Graph::validate_knob_settings(std::unordered_map<int64_t, KnobSetting> const &user_knobs);

};

} // namespace hipdnn::frontend
```

#### Usage Example

```cpp
using namespace hipdnn::frontend;

// Create and build graph
Graph graph;

// ... setup graph ...

std::vector<Knob> knobs;
graph.get_knobs_for_engine(MIOPEN_ENGINE, knobs);

// Option 1: Use default knob values
auto defaultKnobSettings = Graph::knobToDefaultKnobSettings(knobs);
graph.create_execution_plan_ext(MIOPEN_ENGINE, defaultKnobSettings);

// Option 2: Customize specific knobs
std::unordered_map<int64_t, KnobSetting> customKnobSettings;
customKnobSettings.insert(Knob::make_knob_id("miopen.conv.tile_size"), 32);
graph.create_execution_plan_ext(MIOPEN_ENGINE, customKnobSettings);
```

The extension methods provide:
- **Richer metadata**: `Knob` includes display names, descriptions, constraints, and default values
- **Type safety**: `KnobSetting` class enforces type-safe value setting
- **Flexibility**: Support for several value types beyond int64_t
- **Early Validation**: Since we know the constraints for the knobs, we can error out early if they defy constraints

The existing methods will use the flatbuffer variants under the hood.

### 4.5 Knob Naming Convention

We recommend a hierarchical naming scheme:

```
<plugin_name>.<engine_name>.<knob_name>

Examples:
- miopen.conv.tile_size
- miopen.conv.algorithm
- rocblas.gemm.transpose_algorithm
- custom_plugin.matmul.block_size
```

The plugin name prefix ensures that different plugins can have semantically similar knobs.

Additionally, we will define a "global." namespace which will contain commonly used knobs.  Custom knobs registered in the global namespace will be rejected.

```
Examples:
- global.deterministic
- global.workspace
- global.benchmarking
```

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

**Decision**: Knobs are optional; plugins can opt-in by overriding `registerKnobs()`.

**Rationale**:
- **Backward Compatibility**: Existing plugins continue to work without modification
- **Simplicity**: Simple engines don't need to expose knobs
- **Flexibility**: Plugins can add knob support incrementally

### 5.3 Namespace-Based Knob IDs

**Decision**: Use string-based hierarchical knob IDs instead of global enumerations.

**Rationale**:
- **Plugin Autonomy**: Plugins can define custom knobs without modifying core hipDNN
- **Readability**: String IDs are self-documenting
- **Extensibility**: New plugins can be added without coordination

**Trade-off**: String comparison is slower than integer comparison, but knob operations are infrequent (configuration time, not execution time).

### 5.4 Validation at Finalization

**Decision**: Validate knob settings during `hipdnnBackendFinalize(engineCfg)`.

**Rationale**:
- **Early Error Detection**: Catch invalid settings before execution
- **Clear Error Messages**: Can provide detailed validation errors
- **Consistency**: Matches existing finalization pattern in hipDNN


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
- Validation only occurs during finalization (one-time cost)
- Flatbuffers are zero-copy and highly efficient
- Knob settings are applied once per engine config, not per execution
- Profile and optimize hot paths if needed

### 6.3 Plugin Compatibility

**Risk**: Plugins may expose incompatible knobs.

**Mitigation**:
- Provide validation helpers in Plugin SDK
- Document best practices for knob design

### 6.4 Testing Complexity

**Risk**: Testing all combinations of knob values is impractical.

**Mitigation**:
- Focus testing on boundary conditions and invalid values
- Use property-based testing for validation logic
- Require plugins to provide test cases for their knobs

## 7. Execution Plan

The execution plan has two branches.  One is a "get it in and get it done" version that doesn't leverage any plugin SDK helper classes.  The goal with this path is to get to implementation as quickly as possible.  The second path can be done once we migrate to the plugin SDK helper classes.

### Phase 1: Core Infrastructure

1. **Flatbuffer Schema Design**
   - Create `knob_value.fbs` with union types for KnobValue and KnobConstraints
   - Update `engine_config.fbs` to include KnobSetting array
   - Update `engine_details.fbs` to include Knob array
   - Generate C++ code from schemas using flatc
   - Write unit tests for schema handling
   - Verify union type handling and constraint validation

2. **Backend API Implementation**
   - Implement knob-related attribute handling in backend descriptors
   - Add `HIPDNN_ATTR_ENGINE_KNOB_INFO` support in EngineDescriptor
   - Add `HIPDNN_ATTR_ENGINECFG_KNOB_CHOICES` support in EngineConfigDescriptor
   - Implement validation logic for knob settings during finalization
   - Update `hipdnnBackendFinalize(engineCfg)` to validate knobs against constraints
   - Add error reporting for invalid knob values
   - Write backend unit tests for knob validation
   - Rough in backend support using wrapper classes for now, until the Plugin SDK is  available

### Phase 2: Frontend API

5. **Knob Class Implementation**
   - Implement `Knob` class wrapping flatbuffer Knob
   - Add accessors for knob metadata (ID, display name, description)
   - Implement type-specific default value getters
   - Implement type-specific constraint getters
   - Add `toDefaultKnobSettings()` converter method
   - Write unit tests for Knob class

6. **KnobSettings Class Implementation**
   - Implement `KnobSettings` class with type-safe constructors
   - Add type-specific value getters
   - Add validation against Knob constraints (and warn on deprecated knobs)
   - Write unit tests for Knob class

7. **Graph Extension Methods**
   - Implement `Graph::get_knobs_for_engine_ext(int64_t engineId)`
     - Query backend for engine details
     - Extract Knob array from flatbuffer
     - Return vector of Knob objects
   - Implement `Graph::create_execution_plan_ext(int64_t engineId, const std::unordered_map<int64_t, KnobSetting>&)`
     - Convert KnobSetting vector to flatbuffer
     - Create EngineConfig with knob settings
     - Validate knobs during finalization
     - Create and return ExecutionPlan
   - Implement the int64 versions of the base methods
   - Implement `Graph::knobToDefaultKnobSettings()` static helper
   - Write frontend integration tests

8. **Documentation**
   - Write user guide for knobs API
   - Create plugin developer guide for knob registration
       - Define some best practices
           - Document how execution plans should and shouldn't be serialized, regarding how it interacts with knob settings
           - If a knob setting isn't supplied by the user, it's up to the plugin author to supply default values
           - If a knob is no longer needed, deprecate it rather than removing it.  There is a deprecated flag on knobs that can be used.  Deprecated knobs can be removed during a major version udpate.
   - Document naming conventions and best practices
   - Provide code examples for all three usage patterns

### Phase 3: Plugin Migration

9. **MIOpen Plugin Update**
   - Identify tunable parameters in MIOpen convolution engines
   - Implement knobs using flatbuffer wrapper classes
   - Add plugin-specific tests for knob functionality
   - Verify backward compatibility (engines work without knobs)

### Phase 4: Testing & Refinement

11. **Integration Testing**
    - End-to-end tests with real plugins and knobs
    - Test usage patterns (defaults, customization)
    - Cross-plugin knob namespace isolation tests
    - Error handling and validation edge cases
    - Performance benchmarking (knob overhead should be negligible)

12. **Documentation & Review**
    - Complete API reference documentation
    - Create tutorial examples showing common workflows
    - Document migration path for existing plugins
    - Team review and feedback incorporation
    - Address any issues discovered during testing
    - Prepare release notes highlighting new functionality

### Phase X: Plugin SDK Support
⚠️ This phase would occur if the plugin SDK helper classes are available.

3. **KnobRegistry Implementation**
   - Implement `KnobRegistry` class with registration methods:
     - `registerIntKnob()` (both range-based and explicit values)
     - `registerFloatKnob()`
   - Implement `validateKnobSettings()` against registered constraints
   - Add helper methods for knob lookup and type checking
   - Write comprehensive unit tests for all knob types

4. **EngineBase and PlanBuilderBase Integration**
   - Add `registerKnobs(KnobRegistry&)` virtual method to `EngineBase`
   - Add `validateKnobSettings(const std::vector<KnobSetting>&)` virtual method to `EngineBase`
   - Implement `KnobSetting` helper class for extracting typed values
   - Extract knob settings during initializeExecutionContext
   - Update engine lifecycle to call knob registration during initialization
   - Update engine configuration to call knob application before execution
   - Provide utility methods for common knob patterns
   - Write integration tests for engine knob lifecycle

5. **Migrate MIOpen plugin**
    - Switch from using flatbuffer knob wrappers to `register***Knob()` calls

## 8. Testing Plan

### 8.1 Unit Tests

#### Flatbuffer Schema Tests
- Handle each knob value type
- Validate constraint handling
- Test union type discrimination
- Verify default value handling

#### KnobRegistry Tests
⚠️ This phase would occur if the plugin SDK helper classes are available.
- Register each knob type
- Validate constraint enforcement
- Test duplicate knob ID detection
- Verify conversion to EngineDetails

#### Validation Tests
- Valid values pass validation
- Invalid values fail with appropriate errors
- Boundary conditions (min/max values)
- Type mismatches detected
- Missing required knobs detected

### 8.2 Integration Tests

#### Backend Integration
- Query knobs from engine details
- Set knobs in engine config
- Validate during finalization
- Error handling and reporting

#### Plugin SDK Integration
- Engine registers knobs
- Knob settings applied to engine
- Multiple engines with different knobs

#### Frontend Integration
- Type-safe knob setting
- Knob value retrieval
- Error propagation to user

### 8.3 Plugin-Specific Tests

#### MIOpen Plugin
- Verify all registered knobs
- Test knob application affects behavior
- Validate performance with different settings
- Ensure backward compatibility

## 9. Future Considerations

### 9.1 Auto-Tuning Integration

Future work could integrate knobs with auto-tuning:

- **Knob Search Space**: Use knob metadata to define search space
- **Performance Feedback**: Collect timing data for different knob settings
- **Optimal Configuration**: Automatically find best knob values for a workload
- **Caching**: Store optimal configurations for reuse

### 9.2 Knob identifier headers

- **Global Knob Identifier Header** - A header that lives in the frontend that defines an enum or consts for common knob types, preventing the need for knowing "special strings"
- **Plugin Knob Identifier Header** - Plug devs could have their plugin knob id consts / enums in a header that installs to a known frontend include location, providing the same features as the Global Knob Identifier Header

## 10. Glossary

- **Knob**: A runtime-configurable parameter that affects engine behavior
- **Knob ID**: A unique string identifier for a knob (e.g., "miopen.conv.tile_size")
- **Knob Value**: The actual setting for a knob (int, float, bool, string, or enum)
- **Knob Constraint**: Validation rules for a knob (min/max, valid values, etc.)
- **Knob Registry**: A collection of knob metadata for an engine
- **Engine Config**: A configuration object that includes engine ID and knob settings
- **Engine Details**: Metadata about an engine, including supported knobs
- **Knob Setting**: A (knob_id, value) pair in an engine configuration
- **Namespace**: A prefix for knob IDs to prevent conflicts (e.g., "miopen.")
- **Validation**: Checking that knob values satisfy constraints
- **Finalization**: The process of validating and locking a configuration
- **Auto-Tuning**: Automatically searching for optimal knob values
- **Preset**: A predefined collection of knob settings for common use cases

## 11. Implementation Notes

This section documents the actual implementation status and any deviations from the proposed design.

### 11.1 Implementation Status

**Status**: ✅ **Implemented**

The knobs system has been fully implemented in hipDNN as of version 1.0+. The implementation follows the design proposed in this RFC with the following components completed:

#### ✅ Completed Components

1. **FlatBuffer Schemas** (Phase 1.1)
   - `data_sdk/schemas/knob_value.fbs` - Core knob schema with union types
   - `engine_details.fbs` - Updated with knob metadata support
   - `engine_config.fbs` - Updated with knob settings support
   - All schemas support int64, float64, and string value types

2. **Backend API** (Phase 1.2)
   - `HIPDNN_ATTR_KNOB_INFO_SERIALIZED_VALUE` - Query knob metadata
   - `HIPDNN_ATTR_ENGINECFG_KNOB_CHOICES` - Set knob values
   - Validation during `hipdnnBackendFinalize(engineCfg)`
   - Error reporting for invalid knob values

3. **Frontend API** (Phase 2)
   - `Knob` class - Wraps knob metadata with type-safe accessors
   - `KnobSetting` class - Type-safe knob value setting
   - Constraint classes: `IntConstraint`, `FloatConstraint`, `StringConstraint`, `EmptyConstraint`
   - `Graph::get_knobs_for_engine()` - Query available knobs
   - `Graph::get_knob_lookup_for_engine()` - Query knobs as map
   - `Graph::create_execution_plan_ext()` - Create plan with knob settings

4. **MIOpen Provider** (Phase 3)
   - Implemented `global.benchmarking` knob
   - Implemented `global.workspace_size_limit` custom knob for convolution operations
   - Dynamic knob registration based on operation type
   - Knob validation and application in plan builders

5. **Documentation** (Phase 4)
   - Core hipDNN knobs documentation
   - User guide (HowTo) with quick start
   - Plugin development guide for knob providers
   - MIOpen provider knobs documentation with examples
   - This RFC updated with implementation status

### 11.2 Implementation Deviations

The following deviations from the original RFC were made during implementation:

#### 1. Knob ID Hashing

**RFC Proposed**: Hash knob ID strings to int64_t for efficient lookup

**Implemented**: Knob IDs remain as strings throughout the system

**Rationale**:
- String IDs provide better debuggability and logging
- String comparison performance is acceptable for configuration-time operations
- Easier integration with FlatBuffers
- No measurable performance impact since knobs are set at plan creation, not execution

#### 2. Frontend API Naming

**RFC Proposed**: `Graph::create_execution_plan(...)` with knob overloads

**Implemented**: `Graph::create_execution_plan_ext(...)` with `KnobSetting` vector

**Rationale**:
- `_ext` suffix distinguishes the extended API from potential future simplified APIs
- Vector of `KnobSetting` is more flexible than individual overloads
- Consistent with other extended APIs in hipDNN

#### 3. Constraint Representation

**RFC Proposed**: Complex constraint expressions (e.g., "power of 2" constraints)

**Implemented**: Explicit valid values list for complex constraints

**Rationale**:
- Simpler to implement and validate
- More explicit and easier to document
- Covers all practical use cases
- Example: For "power of 2", use `validValues = {8, 16, 32, 64, 128}` instead of a complex predicate

### 11.3 Known Limitations

1. **String-based Knob IDs**: Slightly less efficient than integer IDs, but acceptable for configuration-time operations.

2. **No Auto-Tuning Integration**: Auto-tuning integration (section 9.1) is planned for a future release.

### 11.4 Actual API Signatures

The implemented API signatures differ slightly from the RFC proposal:

#### Frontend Knob Class
```cpp
class Knob {
public:
    static std::pair<Error, Knob> tryFromFlatbuffer(hipdnnBackendFlatbufferData_t fbData);

    const std::string& knobId() const;
    const std::string& description() const;
    bool isDeprecated() const;
    KnobValueType valueType() const;
    const KnobValueVariant& defaultValue() const;
    const IConstraint* constraint() const;
    Error validate(const KnobSetting& setting) const;
};
```

#### Frontend KnobSetting Class
```cpp
class KnobSetting {
public:
    KnobSetting(std::string knobId, KnobValueVariant value);
    template <typename T> KnobSetting(std::string knobId, const T& value);

    const std::string& knobId() const;
    const KnobValueVariant& value() const;
    template <typename T> void setValue(const T& value);
};
```

#### Graph Methods
```cpp
class Graph {
public:
    Error get_knobs_for_engine(int64_t engineId, std::vector<Knob>& knobs) const;
    Error get_knob_lookup_for_engine(int64_t engineId,
                                     std::unordered_map<std::string, Knob>& knobs) const;
    Error create_execution_plan_ext(int64_t engineId,
                                    const std::vector<KnobSetting>& settings);
};
```

### 11.5 Testing Status

**Unit Tests**: ✅ Completed
- `frontend/tests/TestKnob.cpp` - Knob class tests
- `frontend/tests/TestKnobSetting.cpp` - KnobSetting class tests
- `frontend/tests/TestGraph.cpp` - Graph knob API tests

**Integration Tests**: ✅ Completed
- MIOpen provider knob tests
- End-to-end knob setting and validation tests

**Plugin Tests**: ✅ Completed
- MIOpen benchmarking knob tests
- MIOpen workspace limit knob tests

### 11.6 Production Usage

The knobs system is currently used in production by:

1. **MIOpen Provider**
   - `global.benchmarking` - Enable/disable kernel benchmarking
   - `global.workspace_size_limit` - Control convolution workspace memory

2. **Internal Tests and Samples**
   - Test suites use knobs for configuring test behavior
   - Samples demonstrate knob usage patterns

### 11.7 Migration Path

For users migrating from earlier versions without knobs:

1. **No action required for default behavior**: All knobs have sensible defaults
2. **Opt-in configuration**: Knobs are optional; existing code continues to work
3. **Progressive enhancement**: Users can gradually adopt knobs for specific tuning needs

### 11.8 Related Documentation

For implementation details and usage examples, refer to:

- **User Documentation**:
  - [hipDNN Knobs](../Knobs.md) - Complete user guide
  - [HowTo - Configuring Knobs](../HowTo.md#configuring-engine-knobs) - Quick start
  - [MIOpen Knobs](../../../dnn-providers/miopen-provider/docs/Knobs.md) - Provider-specific knobs

- **Developer Documentation**:
  - [Plugin Development - Providing Knobs](../PluginDevelopment.md#providing-knobs) - Plugin author guide

- **Implementation Files**:
  - `frontend/include/hipdnn_frontend/knob/Knob.hpp`
  - `frontend/include/hipdnn_frontend/knob/KnobSetting.hpp`
  - `frontend/include/hipdnn_frontend/knob/KnobConstraint.hpp`
  - `data_sdk/schemas/knob_value.fbs`

---

**Last Updated**: 2026-02-18
**Implemented By**: Mitch Ousdahl and contributors
