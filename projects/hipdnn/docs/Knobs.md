# hipDNN Engine Configuration Knobs

This document provides comprehensive information about hipDNN's engine configuration knobs system.

## Table of Contents

- [Overview](#overview)
- [What are Knobs?](#what-are-knobs)
- [Knob Types](#knob-types)
- [Naming Conventions](#naming-conventions)
- [Using Knobs](#using-knobs)
  - [Querying Available Knobs](#querying-available-knobs)
  - [Setting Knob Values](#setting-knob-values)
  - [Using Default Values](#using-default-values)
- [Standard Global Knobs](#standard-global-knobs)
- [Provider-Specific Knobs](#provider-specific-knobs)
- [API Reference](#api-reference)
- [Best Practices](#best-practices)

---

## Overview

Engine configuration knobs provide a flexible mechanism for controlling runtime behavior of hipDNN engines. They allow you to tune performance, configure algorithmic choices, and adjust memory usage without recompiling code.

## What are Knobs?

**Knobs** are runtime-configurable parameters that affect engine behavior. Each knob has:

- **Unique identifier**: A string-based ID (e.g., `"global.benchmarking"`)
- **Type**: Integer (int64), Float (double), or String
- **Default value**: The value used when not explicitly set
- **Constraints**: Valid ranges or allowed values
- **Description**: Human-readable explanation of purpose

Knobs enable you to:
- Enable or disable features (e.g., benchmarking mode)
- Tune performance parameters (e.g., tile sizes, workspace limits)
- Select algorithmic variants (e.g., solver selection)
- Control memory usage (e.g., workspace size limits)

## Knob Types

hipDNN supports two categories of knobs:

### Global Knobs

**Global knobs** are standard knobs available across all or most engines. They are defined in the `global.*` namespace and provide common functionality.

**Characteristics**:
- Namespace prefix: `global.*`
- Consistent behavior across engines
- Defined by the hipDNN specification
- Cannot be registered by custom plugins

**Examples**:
- `global.benchmarking` - Enable/disable solver benchmarking
- `global.workspace_size_limit` - Maximum workspace memory

### Custom Knobs

**Custom knobs** are engine-specific or plugin-specific parameters. Plugin developers can register custom knobs to expose their own tuning parameters.

**Characteristics**:
- Namespace prefix: `<plugin>.*` or `<plugin>.<operation>.*`
- Engine-specific behavior
- Defined by plugin developers
- Extend hipDNN functionality for specific use cases

**Examples**:
- `miopen.conv.tile_size` - Convolution tile size for MIOpen
- `rocblas.gemm.algo` - GEMM algorithm selection for ROCm BLAS

## Naming Conventions

Knobs follow a hierarchical naming scheme to avoid conflicts and improve organization:

```
<namespace>.<category>.<knob_name>
```

**Global namespace** (reserved):
```
global.benchmarking
global.workspace_size_limit
global.deterministic
```

**Plugin-specific namespace**:
```
miopen.conv.tile_size
rocblas.gemm.transpose_algorithm
custom_plugin.matmul.block_size
```

> [!IMPORTANT]
> The `global.*` namespace is reserved for standard knobs. Custom plugins cannot register knobs in this namespace.

---

## Using Knobs

### Querying Available Knobs

Before setting knobs, you typically want to discover what knobs an engine supports and their constraints.

#### C++ Frontend API

```cpp
#include <hipdnn_frontend.hpp>

using namespace hipdnn_frontend;

// After building the graph
Graph graph;
// ... setup and build graph ...

// Get available knobs for an engine
std::vector<Knob> knobs;
auto error = graph.get_knobs_for_engine(engineId, knobs);

if (error.is_good()) {
    for (const auto& knob : knobs) {
        std::cout << "Knob ID: " << knob.knobId() << "\n";
        std::cout << "Description: " << knob.description() << "\n";
        std::cout << "Type: " << static_cast<int>(knob.valueType()) << "\n";

        // Access default value (it's a variant)
        const auto& defaultVal = knob.defaultValue();
        if (std::holds_alternative<int64_t>(defaultVal)) {
            std::cout << "Default: " << std::get<int64_t>(defaultVal) << "\n";
        } else if (std::holds_alternative<double>(defaultVal)) {
            std::cout << "Default: " << std::get<double>(defaultVal) << "\n";
        } else if (std::holds_alternative<std::string>(defaultVal)) {
            std::cout << "Default: " << std::get<std::string>(defaultVal) << "\n";
        }

        // Check constraints
        const IConstraint* constraint = knob.constraint();
        if (constraint) {
            std::cout << "Constraint: " << constraint->toString() << "\n";
        }

        std::cout << "---\n";
    }
} else {
    std::cerr << "Error getting knobs: " << error.get_message() << "\n";
}
```

#### Using Knob Lookup Map

For easier access by knob ID, use the lookup method:

```cpp
std::unordered_map<std::string, Knob> knobMap;
auto error = graph.get_knob_lookup_for_engine(engineId, knobMap);

if (error.is_good()) {
    auto it = knobMap.find("global.benchmarking");
    if (it != knobMap.end()) {
        const Knob& benchmarkingKnob = it->second;
        // Use the knob...
    }
}
```

### Setting Knob Values

Once you know what knobs are available, you can set them when creating an execution plan.

#### Basic Example

```cpp
#include <hipdnn_frontend.hpp>

using namespace hipdnn_frontend;

Graph graph;
// ... setup and build graph ...

// Create knob settings
std::vector<KnobSetting> settings;

// Set integer knob
settings.emplace_back("global.benchmarking", 1);

// Set int64 knob
settings.emplace_back("global.workspace_size_limit", 1024000LL);

// Set float knob
settings.emplace_back("some.float_knob", 0.5);

// Set string knob
settings.emplace_back("some.string_knob", std::string("value"));

// Create execution plan with these settings
auto error = graph.create_execution_plan_ext(engineId, settings);

if (error.is_good()) {
    std::cout << "Execution plan created successfully with custom knob settings\n";
} else {
    std::cerr << "Error: " << error.get_message() << "\n";
}
```

#### Type-Safe Knob Setting

```cpp
// KnobSetting constructor is type-safe
KnobSetting intSetting("test.knob", 42);                    // int64_t
KnobSetting floatSetting("test.knob", 3.14);                // double
KnobSetting stringSetting("test.knob", std::string("val")); // string

// You can also update values later
intSetting.setValue(100);
```

### Using Default Values

If you don't specify a knob setting, the engine will use the default value defined by the knob. To explicitly use defaults for all knobs:

```cpp
// Option 1: Don't specify any settings (simplest)
auto error = graph.create_execution_plan_ext(engineId, {});

// Option 2: Specify only the knobs you want to customize
std::vector<KnobSetting> settings;
settings.emplace_back("global.benchmarking", 1);  // Only customize this one
auto error = graph.create_execution_plan_ext(engineId, settings);
```

> [!NOTE]
> **Validation**: All knob settings are validated against their constraints when creating an execution plan. Invalid values will result in an error with a descriptive message.

> [!NOTE]
> **Unknown Knobs**: If you specify a knob that doesn't exist for the engine, hipDNN will log a warning but continue. This allows forward compatibility when new knobs are added.

> [!NOTE]
> **Deprecated Knobs**: If a knob is marked as deprecated, hipDNN will log a warning when you use it, but the knob will still function normally.

---

## Standard Global Knobs

The following global knobs are available in hipDNN:

| Knob Name | Type | Default | Description |
|-----------|------|---------|-------------|
| `global.benchmarking` | int64 | 0 (disabled) | Enable benchmarking mode for kernel selection. When enabled, engines may run multiple kernel variants and select the fastest. First run may be slower due to benchmarking overhead. |

> [!NOTE]
> Additional global knobs may be available depending on the engine. Use `get_knobs_for_engine()` to discover all available knobs for a specific engine.

---

## Provider-Specific Knobs

Different engine providers may expose their own custom knobs. Refer to the provider-specific documentation for details:

- **[MIOpen Provider Knobs](../../../dnn-providers/miopen-provider/docs/Knobs.md)** - Configuration knobs for the MIOpen GPU acceleration plugin

> [!TIP]
> When developing with multiple providers, use `get_knobs_for_engine()` to programmatically discover available knobs rather than hard-coding knob names.

---

## API Reference

### Knob Class

Describes metadata for an available knob.

**Key Methods**:
- `const std::string& knobId()` - Get the knob identifier
- `const std::string& description()` - Get human-readable description
- `bool isDeprecated()` - Check if knob is deprecated
- `KnobValueType valueType()` - Get value type (INT64, FLOAT64, or STRING)
- `const KnobValueVariant& defaultValue()` - Get default value as variant
- `const IConstraint* constraint()` - Get constraint validator
- `Error validate(const KnobSetting& setting)` - Validate a setting

### KnobSetting Class

Represents a knob value setting to apply.

**Constructors**:
```cpp
KnobSetting(std::string knobId, KnobValueVariant value);
template <typename T> KnobSetting(std::string knobId, const T& value);
```

**Key Methods**:
- `const std::string& knobId()` - Get knob identifier
- `const KnobValueVariant& value()` - Get the value
- `template <typename T> void setValue(const T& value)` - Update value

### Graph Methods

**Querying Knobs**:
```cpp
Error get_knobs_for_engine(int64_t engineId, std::vector<Knob>& knobs) const;
Error get_knob_lookup_for_engine(int64_t engineId, std::unordered_map<std::string, Knob>& knobs) const;
```

**Setting Knobs**:
```cpp
Error create_execution_plan_ext(int64_t engineId, const std::vector<KnobSetting>& settings);
```

### Constraint Classes

**Base Interface**:
```cpp
class IConstraint {
    virtual Error validateKnobSetting(const KnobSetting& setting) const = 0;
    virtual std::string toString() const = 0;
};
```

**Implementations**:
- `IntConstraint(int64_t minValue, int64_t maxValue, int64_t step, std::unordered_set<int64_t> validValues)`
- `FloatConstraint(double minValue, double maxValue)`
- `StringConstraint(int32_t maxLength, std::unordered_set<std::string> validValues)`
- `EmptyConstraint()` - No constraints

### Type Definitions

```cpp
using KnobValueVariant = std::variant<int64_t, double, std::string>;
typedef std::string KnobType_t;

enum class KnobValueType {
    NOT_SET = 0,
    INT64 = 1,
    FLOAT64 = 2,
    STRING = 3,
};
```

---

## Best Practices

### For Users

1. **Query before setting**: Always call `get_knobs_for_engine()` to understand available knobs and their constraints before setting values.

2. **Validate constraints**: Check the constraint object to ensure your values are valid before creating execution plans.

3. **Use default values when possible**: Only customize knobs when you have a specific performance or behavior requirement.

4. **Handle errors gracefully**: Always check the error return value when setting knobs or creating execution plans.

5. **Be aware of deprecated knobs**: Watch for deprecation warnings and update your code to use recommended alternatives.

6. **Profile before tuning**: Measure performance impact when changing knob values to ensure improvements.

### For Plugin Developers

Plugin developers can expose custom knobs using the Plugin SDK utilities:

- **[`KnobFactory`](../plugin_sdk/include/hipdnn_plugin_sdk/KnobFactory.hpp)** - Helper class to create knob definitions
- **[`IPlanBuilder::getCustomKnobs()`](../plugin_sdk/include/hipdnn_plugin_sdk/interfaces/IPlanBuilder.hpp)** - Interface method for exposing knobs
- **[`GlobalKnobDefines`](../plugin_sdk/include/hipdnn_plugin_sdk/GlobalKnobDefines.hpp)** - Constants for standard global knob names

For comprehensive guidance on exposing knobs in your plugin, see the [Plugin Development Guide](./PluginDevelopment.md#providing-knobs).

---

## Related Documentation

- [HowTo Guide - Configuring Engine Knobs](./HowTo.md#configuring-engine-knobs) - Quick start guide for using knobs
- [Plugin Development Guide - Providing Knobs](./PluginDevelopment.md#providing-knobs) - How to expose knobs in plugins
- [RFC 0004 - Engine Configuration Knobs Design](./rfcs/0004_EngineConfigKnobs.md) - Detailed design document
- [MIOpen Provider Knobs](../../../dnn-providers/miopen-provider/docs/Knobs.md) - MIOpen-specific knobs

---

## Examples

For complete working examples, see:
- [Knobs Usage Sample](../samples/knobs/) - Comprehensive example demonstrating knob discovery and configuration
- [Frontend Tests](../frontend/tests/TestKnob.cpp) - Unit tests showing knob API usage
