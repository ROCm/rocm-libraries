# hipDNN - Engine Configuration Knobs Design Document

- Contributors: Mitch Ousdahl

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

## 1. Executive Summary

This RFC proposes a flexible engine configuration knobs system for hipDNN that allows plugin developers to expose custom runtime settings and enables end-users to adjust and serialize these settings. Unlike a more limited int64_t-based knob system with min/max/stride constraints, this design leverages Flatbuffers' union types to support multiple value types (integers, floats, booleans, strings, enums) while maintaining type safety and efficient serialization.

The knobs system is designed to be:
- **Optional**: Plugins can opt-in to exposing knobs
- **Flexible**: Support for multiple data types beyond int64_t
- **Namespace-safe**: Plugin-specific knob identifiers prevent conflicts
- **Serializable**: Engine configurations with knob settings can be saved and restored
- **Extensible**: New knob types can be added without breaking existing code

## 2. Problem Statement

### 2.1 Current Limitations

Currently, hipDNN engines have limited runtime configurability. While the backend supports the concept of engine configurations, there is no standardized mechanism for:

1. **Plugin developers** to expose custom tuning parameters (e.g., tile sizes, algorithmic choices, memory layout preferences)
2. **End-users** to discover available settings for a given engine
3. **Applications** to persist and restore optimal engine configurations across runs
4. **Different plugins** to safely expose settings without naming conflicts

### 2.2 The Initial Approach and Its Limitations

The default implementation specifies a knob system with the following characteristics:

```cpp
typedef struct {
    hipdnnBackendKnobType_t type;  // Enumerated knob type
    int64_t minimum_value;         // Minimum valid value
    int64_t maximum_value;         // Maximum valid value
    int64_t stride;                // Step size between valid values
} hipdnnBackendKnobInfo_t;
```

**Limitations:**
1. **Type Restriction**: All knob values must be int64_t, making it awkward for boolean flags, floating-point parameters, or enumerated choices
2. **Forced Range Semantics**: Every knob must specify min/max/stride, even when these concepts don't apply (e.g., boolean flags, enum selections)
3. **Global Namespace**: Knob types are globally enumerated, making it difficult for custom plugins to add new knobs without upstream changes
4. **Limited Expressiveness**: Cannot represent complex constraints (e.g., "value must be a power of 2" or "valid values are {1, 4, 8, 16}")

### 2.3 Requirements

A robust knobs system for hipDNN must support:

1. **Multiple Value Types**: int64, float64, bool, string, and enumerated types
2. **Flexible Validation**: Per-knob validation rules appropriate to the value type
3. **Plugin Autonomy**: Plugins can define custom knobs without modifying core hipDNN
4. **Namespace Isolation**: Different plugins can have knobs with the same semantic meaning without conflicts
5. **Discovery**: Users can query available knobs and their valid ranges/values
6. **Serialization**: Engine configurations with knob settings can be saved and restored

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
- Providing helper classes for knob validation and serialization
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
    BoolValue,
    StringValue,
    EnumValue
}

table IntValue {
    value: int64;
}

table FloatValue {
    value: float64;
}

table BoolValue {
    value: bool;
}

table StringValue {
    value: string;
}

table EnumValue {
    selected_index: int32;  // Index into valid_values array
}

// Knob metadata for discovery
table KnobInfo {
    knob_id: int64;               // Hashed from knob_id_str
    knob_id_str: string;          // Unique identifier (namespaced by plugin)
    display_name: string;         // Human-readable name
    description: string;          // Help text
    value_type: KnobValue;        // Discriminator for union
    default_value: KnobValue;     // Default setting
    constraints: KnobConstraints; // Validation rules
}

// Type-specific constraints
union KnobConstraints {
    IntConstraints,
    FloatConstraints,
    BoolConstraints,
    StringConstraints,
    EnumConstraints
}

table IntConstraints {
    min_value: int64;
    max_value: int64;
    stride: int64 = 1;            // Step size (default 1)
    valid_values: [int64];        // Optional: explicit list of valid values
}

table FloatConstraints {
    min_value: float64;
    max_value: float64;
    valid_values: [float64];      // Optional: explicit list
}

table BoolConstraints {
    // No constraints needed for boolean
}

table StringConstraints {
    max_length: int32;
    valid_values: [string];       // Optional: enum-like string choices
}

table EnumConstraints {
    valid_values: [string];       // Required: list of valid enum strings
}
```

#### 4.1.2 Updated Engine Schemas

```flatbuffers
// data_sdk/schemas/engine_details.fbs
namespace hipdnn_data_sdk.data_objects;

include "knob_value.fbs";

table EngineDetails {
    engine_id: int64;
    supported_knobs: [KnobInfo];  // NEW: Knobs this engine supports
}

// data_sdk/schemas/engine_config.fbs
namespace hipdnn_data_sdk.data_objects;

include "knob_value.fbs";

table Knob {
    knob_id: int64;
    value: KnobValue;
}

table EngineConfig {
    engine_id: int64;
    knobs: [Knob];  // NEW: User-specified knob values
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
        const std::string& displayName,
        const std::string& description,
        int64_t defaultValue,
        int64_t minValue,
        int64_t maxValue,
        int64_t stride = 1
    );

    // Register an integer knob with explicit valid values
    void registerIntKnob(
        const std::string& knobId,
        const std::string& displayName,
        const std::string& description,
        int64_t defaultValue,
        const std::vector<int64_t>& validValues
    );

    // Register a float knob
    void registerFloatKnob(
        const std::string& knobId,
        const std::string& displayName,
        const std::string& description,
        double defaultValue,
        double minValue,
        double maxValue
    );

    // Register a boolean knob
    void registerBoolKnob(
        const std::string& knobId,
        const std::string& displayName,
        const std::string& description,
        bool defaultValue
    );

    // Register an enum knob
    void registerEnumKnob(
        const std::string& knobId,
        const std::string& displayName,
        const std::string& description,
        const std::string& defaultValue,
        const std::vector<std::string>& validValues
    );

    // Serialize to EngineDetails flatbuffer
    std::vector<uint8_t> serializeKnobInfo() const;

    // Validate knob settings
    bool validateKnobSettings(
        const std::vector<uint8_t>& serializedSettings,
        std::string& errorMessage
    ) const;

private:
    std::vector<KnobInfo> _knobs;
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

    // NEW: Override to register knobs
    virtual void registerKnobs(KnobRegistry& registry) {
        // Default: no knobs
    }

    // NEW: Override to apply knob settings
    virtual void applyKnobSettings(const KnobSettings& settings) {
        // Default: no-op
    }

protected:
    KnobRegistry _knobRegistry;
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
        registry.registerIntKnob(
            "miopen.conv.tile_size",
            "Tile Size",
            "Size of computation tile (must be power of 2)",
            16,  // default
            {8, 16, 32, 64, 128}  // valid values
        );

        // Register algorithm selection
        registry.registerEnumKnob(
            "miopen.conv.algorithm",
            "Algorithm",
            "Convolution algorithm to use",
            "winograd",  // default
            {"direct", "gemm", "winograd", "fft"}
        );

        // Register workspace reuse flag
        registry.registerBoolKnob(
            "miopen.conv.reuse_workspace",
            "Reuse Workspace",
            "Reuse workspace memory across calls",
            true  // default
        );
    }

    void applyKnobSettings(const KnobSettings& settings) override {
        // Extract and apply settings
        if (auto tileSize = settings.getInt("miopen.conv.tile_size")) {
            _tileSize = *tileSize;
        }
        if (auto algo = settings.getEnum("miopen.conv.algorithm")) {
            _algorithm = parseAlgorithm(*algo);
        }
        if (auto reuse = settings.getBool("miopen.conv.reuse_workspace")) {
            _reuseWorkspace = *reuse;
        }
    }

private:
    int64_t _tileSize = 16;
    Algorithm _algorithm = Algorithm::Winograd;
    bool _reuseWorkspace = true;
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
    BOOL,
    STRING,
    ENUM
};

// Knob information class - describes available knobs for an engine
class Knob {
public:
    Knob(const int64_t knobId,
             const std::string& knobIdStr,
             const std::string& displayName,
             const std::string& description);

    // Accessors
    int64_t getKnobId() const;
    const std::string& getKnobIdStr() const;
    const std::string& getDisplayName() const;
    const std::string& getDescription() const;

    KnobValueType getValueType() const;

    // Get default value (type-specific)
    std::optional<int64_t> getDefaultInt() const;
    std::optional<double> getDefaultFloat() const;
    std::optional<bool> getDefaultBool() const;
    std::optional<std::string> getDefaultString() const;

    // Get constraints (type-specific)
    struct IntConstraints {
        int64_t minValue;
        int64_t maxValue;
        int64_t stride;
        std::vector<int64_t> validValues;  // Optional explicit list
    };
    struct FloatConstraints {
        double minValue;
        double maxValue;
        std::vector<double> validValues;  // Optional explicit list
    };
    struct EnumConstraints {
        std::vector<std::string> validValues;
    };

    std::optional<IntConstraints> getIntConstraints() const;
    std::optional<FloatConstraints> getFloatConstraints() const;
    std::optional<EnumConstraints> getEnumConstraints() const;

    // Convert to Knob with default value
    Knob toDefaultKnob() const;

    // flatbuffer pack and unpack methods
    pack()
    unpack()

    // Helper: Convert Knob vector to KnobSettings vector with defaults
    static std::vector<Knob> knobToDefaultKnobSettings(
        const std::vector<KnobInfo>& knobInfos);

    // Helper, to hash the string ID to the int ID
    static int64_t make_knob_id(const std::string& strID);
};

// Knob setting class - represents a configured knob value
class KnobSetting {
public:
    // Constructors for different value types
    KnobSetting(int64_t knobId, int64_t value);
    KnobSetting(int64_t knobId, double value);
    KnobSetting(int64_t knobId, bool value);
    KnobSetting(int64_t knobId, const std::string& value);

    // Accessors
    int64_t getKnobId() const;
    KnobInfo::ValueType getValueType() const;

    // Get value (type-specific)
    std::optional<int64_t> getIntValue() const;
    std::optional<double> getFloatValue() const;
    std::optional<bool> getBoolValue() const;
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
                                        std::vector<Knob> &knobinfos) const;

    // New methods
    error_t Graph::create_execution_plan(int64_t engineId,
                                         std::vector<KnobSetting> const &user_knobs) const;

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
std::vector<Knob> customKnobSettings;
for(const auto& info : knobInfos)
{
    if(info.getKnobIdStr() == "miopen.conv.tile_size")
    {
        // Override tile size
        customKnobSettings.emplace_back(info.getKnobId(), 32);
    }
    else
    {
        // Use default for other knobs
        customKnobSettings.push_back(info.toDefaultKnob());
    }
}
graph.create_execution_plan_ext(MIOPEN_ENGINE, customKnobSettings);
```

The extension methods provide:
- **Richer metadata**: `Knob` includes display names, descriptions, constraints, and default values
- **Type safety**: `KnobSetting` class enforces type-safe value setting
- **Flexibility**: Support for multiple value types beyond int64_t

The existing methods will use the flatbuffer variants under the hood.

### 4.5 Knob Naming Convention

To prevent conflicts between plugins, we recommend a hierarchical naming scheme:

```
<plugin_name>.<engine_name>.<knob_name>

Examples:
- miopen.conv.tile_size
- miopen.conv.algorithm
- rocblas.gemm.transpose_algorithm
- custom_plugin.matmul.block_size
```

The plugin name prefix ensures that different plugins can have semantically similar knobs without collision.

## 5. Key Design Decisions

### 5.1 Flatbuffer Unions vs. Original Approach

**Decision**: Use Flatbuffer union types instead of an int64_t-only approach.

**Rationale**:
- **Type Safety**: Unions provide compile-time and runtime type checking
- **Expressiveness**: Can represent boolean flags, floating-point parameters, and enumerations naturally
- **Flexibility**: Easy to add new value types without breaking existing code
- **Efficiency**: Flatbuffers are zero-copy and highly efficient for serialization
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
- **Collision Avoidance**: Namespace prefixes prevent conflicts
- **Readability**: String IDs are self-documenting
- **Extensibility**: New plugins can be added without coordination

**Trade-off**: String comparison is slower than integer comparison, but knob operations are infrequent (configuration time, not execution time).

### 5.4 Validation at Finalization

**Decision**: Validate knob settings during `hipdnnBackendFinalize(engineCfg)`.

**Rationale**:
- **Early Error Detection**: Catch invalid settings before execution
- **Clear Error Messages**: Can provide detailed validation errors
- **Consistency**: Matches existing finalization pattern in hipDNN

### 5.5 Serialization Support

**Decision**: Engine configurations with knob settings are serializable via Flatbuffers.

**Rationale**:
- **Persistence**: Users can save optimal configurations
- **Reproducibility**: Configurations can be shared across runs/systems
- **Debugging**: Serialized configs can be inspected and modified
- **Integration**: Works naturally with existing Flatbuffer infrastructure

## 6. Risks

### 6.1 API Complexity

**Risk**: The knobs API adds complexity to the already extensive backend API.

**Mitigation**:
- Provide high-level frontend wrappers for common use cases
- Create comprehensive examples and documentation
- Make knobs optional (plugins and users can ignore if not needed)
- Provide sensible defaults so knobs are only needed for tuning

### 6.2 Performance Overhead

**Risk**: Knob validation and serialization could add overhead.

**Mitigation**:
- Validation only occurs during finalization (one-time cost)
- Flatbuffers are zero-copy and highly efficient
- Knob settings are applied once per engine config, not per execution
- Profile and optimize hot paths if needed

### 6.3 Plugin Compatibility

**Risk**: Plugins may expose incompatible or conflicting knobs.

**Mitigation**:
- Enforce namespace conventions in documentation
- Provide validation helpers in Plugin SDK
- Consider adding a knob registry service for conflict detection
- Document best practices for knob design

### 6.4 Serialization Versioning

**Risk**: Serialized configurations may become incompatible across hipDNN versions.

**Mitigation**:
- Include version information in serialized data (a separate RFC will address versioning)
- Design schemas to be forward/backward compatible where possible
- Document serialization format stability guarantees

### 6.5 Testing Complexity

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
   - Update `engine_details.fbs` to include KnobInfo array
   - Generate C++ code from schemas using flatc
   - Write unit tests for schema serialization/deserialization
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

5. **KnobInfo Class Implementation**
   - Implement `KnobInfo` class wrapping flatbuffer KnobInfo
   - Add accessors for knob metadata (ID, display name, description)
   - Implement type-specific default value getters
   - Implement type-specific constraint getters
   - Add `toDefaultKnob()` converter method
   - Write unit tests for KnobInfo class

6. **Knob Class Implementation**
   - Implement `Knob` class with type-safe constructors
   - Add type-specific value getters
   - Implement serialization/deserialization methods
   - Add validation against KnobInfo constraints
   - Write unit tests for Knob class

7. **Graph Extension Methods**
   - Implement `Graph::get_knobs_for_engine_ext(int64_t engineId)`
     - Query backend for engine details
     - Deserialize KnobInfo array from flatbuffer
     - Return vector of KnobInfo objects
   - Implement `Graph::create_execution_plan_ext(int64_t engineId, const std::vector<Knob>&)`
     - Serialize Knob vector to flatbuffer
     - Create EngineConfig with knob settings
     - Validate knobs during finalization
     - Create and return ExecutionPlan
   - Implement the int64 versions of the base methods
   - Implement `Graph::knobInfoToDefaultKnobs()` static helper
   - Write frontend integration tests

8. **Documentation**
   - Write user guide for knobs API
   - Create plugin developer guide for knob registration
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
    - Test all three usage patterns (defaults, customization, serialization)
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
     - `registerBoolKnob()`
     - `registerEnumKnob()`
   - Add `serializeKnobInfo()` to generate EngineDetails flatbuffer
   - Implement `validateKnobSettings()` against registered constraints
   - Add helper methods for knob lookup and type checking
   - Write comprehensive unit tests for all knob types

4. **EngineBase Integration**
   - Add `registerKnobs(KnobRegistry&)` virtual method to `EngineBase`
   - Add `applyKnobSettings(const KnobSettings&)` virtual method to `EngineBase`
   - Implement `KnobSettings` helper class for extracting typed values
   - Update engine lifecycle to call knob registration during initialization
   - Update engine configuration to call knob application before execution
   - Provide utility methods for common knob patterns
   - Write integration tests for engine knob lifecycle

5. **Migrate MIOpen plugin**
    - Switch from using flatbuffer knob wrappers to `register***Knob()` calls

## 8. Testing Plan

### 8.1 Unit Tests

#### Flatbuffer Schema Tests
- Serialize/deserialize each knob value type
- Validate constraint serialization
- Test union type discrimination
- Verify default value handling

#### KnobRegistry Tests
⚠️ This phase would occur if the plugin SDK helper classes are available.
- Register each knob type
- Validate constraint enforcement
- Test duplicate knob ID detection
- Verify serialization to EngineDetails

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
- Knob namespace isolation

#### Frontend Integration
- Type-safe knob setting
- Knob value retrieval
- Serialization round-trip
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

## 10. Glossary

- **Knob**: A runtime-configurable parameter that affects engine behavior
- **Knob ID**: A unique string identifier for a knob (e.g., "miopen.conv.tile_size")
- **Knob Value**: The actual setting for a knob (int, float, bool, string, or enum)
- **Knob Constraints**: Validation rules for a knob (min/max, valid values, etc.)
- **Knob Registry**: A collection of knob metadata for an engine
- **Engine Config**: A configuration object that includes engine ID and knob settings
- **Engine Details**: Metadata about an engine, including supported knobs
- **Knob Setting**: A (knob_id, value) pair in an engine configuration
- **Namespace**: A prefix for knob IDs to prevent conflicts (e.g., "miopen.")
- **Serialization**: Converting knob settings to/from binary format for persistence
- **Validation**: Checking that knob values satisfy constraints
- **Finalization**: The process of validating and locking a configuration
- **Auto-Tuning**: Automatically searching for optimal knob values
- **Preset**: A predefined collection of knob settings for common use cases
