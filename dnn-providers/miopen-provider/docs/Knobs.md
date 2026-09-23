# MIOpen Provider - Configuration Knobs

This document describes the configuration knobs available for the MIOpen Provider Plugin for hipDNN.

For general information about hipDNN's knobs system, see the [hipDNN Knobs Documentation](../../../projects/hipdnn/docs/Knobs.md).

## Table of Contents

- [Overview](#overview)
- [Available Knobs](#available-knobs)
- [Knob Details](#knob-details)
  - [Benchmarking](#benchmarking)
  - [Workspace Size Limit](#workspace-size-limit)
- [Usage Examples](#usage-examples)
- [Performance Considerations](#performance-considerations)
- [Best Practices](#best-practices)

---

## Overview

The MIOpen Provider Plugin supports configuration knobs that control kernel selection, performance tuning, and memory usage. These knobs allow you to optimize MIOpen's behavior for your specific workload and hardware configuration.

### Knob Types

The MIOpen Provider supports two types of knobs:

- **Global Knobs**: Standard knobs available for all engines (namespace: `global.*`)
- **Custom Knobs**: Operation-specific knobs provided dynamically based on the graph (no custom knobs currently)

---

## Available Knobs

The following table lists all configuration knobs supported by the MIOpen Provider:

| Knob Name | Type | Scope | Default | Valid Range | Description |
|-----------|------|-------|---------|-------------|-------------|
| `global.benchmarking` | int64 | All Operations | 0 (disabled) | 0-1 | Enable benchmarking mode for kernel selection |
| `global.workspace_size_limit` | int64 | Convolution Operations Only | Maximum | Dynamic (solver-dependent) | Maximum workspace memory in bytes |

> [!NOTE]
> The `global.workspace_size_limit` knob is **only available for convolution operations** (Forward, Backward Data, Backward Weights). It is not supported for batchnorm or other operations.

---

## Knob Details

### Benchmarking

**Knob Name**: `global.benchmarking`

**Purpose**: Controls whether MIOpen performs kernel benchmarking to find the optimal solver for a given operation.

**Type**: Integer (int64)

**Valid Values**:
- `0` - Benchmarking disabled (default)
- `1` - Benchmarking enabled

**Behavior**:

When **disabled** (default):
- MIOpen uses heuristics to select a kernel
- First execution is fast
- May not use the optimal kernel for your specific configuration

When **enabled**:
- MIOpen benchmarks multiple solver candidates
- First execution is significantly slower (seconds to minutes)
- Subsequent executions use the cached optimal solver
- Provides best performance for production workloads

**Caching**:
- Benchmark results are cached in MIOpen's performance database
- Default location: `~/.config/miopen/` on Linux
- Cache persists across application runs
- Cache is specific to: GPU model, operation parameters, tensor dimensions, and data types

**Performance Impact**:
- **First run with benchmarking**: Can take seconds to minutes depending on operation complexity
- **Subsequent runs**: Minimal overhead (cache lookup)
- **Production benefit**: Typically 10-50% performance improvement over heuristic selection

**Example**:
```cpp
// Enable benchmarking
std::vector<KnobSetting> settings;
settings.emplace_back("global.benchmarking", 1);
graph.create_execution_plan_ext(engineId, settings);
```

### Workspace Size Limit

**Knob Name**: `global.workspace_size_limit`

**Purpose**: Limits the maximum workspace memory that MIOpen solvers can use for convolution operations.

**Type**: Integer (int64) - workspace size in bytes

**Scope**: **Convolution operations only** (Forward, Backward Data, Backward Weights)

**Valid Range**: Dynamic, determined at runtime based on available MIOpen solvers

The valid range is **operation-specific** and depends on:
- Convolution type (Forward/Backward Data/Backward Weights)
- Tensor dimensions and data types
- Available MIOpen solvers for the configuration
- GPU memory constraints

**How the Range is Determined**:

1. MIOpen queries all available solvers for the specific operation and tensor configuration
2. Each solver reports its workspace memory requirement
3. The minimum workspace is the smallest requirement across all solvers
4. The maximum workspace is the largest requirement across all solvers
5. Default is set to the maximum for optimal performance

**Example Range**:
For a specific convolution forward operation:
- **Minimum**: 512 KB (lightweight kernel with minimal workspace)
- **Maximum**: 128 MB (high-performance kernel with large workspace)
- **Default**: 128 MB (use maximum for best performance)

**Behavior**:

Setting this knob to a value lower than the maximum:
- **Constrains solver selection** to only those requiring ≤ the specified workspace
- **May reduce performance** if optimal solvers require more workspace
- **Useful for memory-constrained systems** where total GPU memory is limited

Setting this knob to the maximum (or not setting it):
- **Allows all solvers** to be considered
- **Provides best performance**
- **Uses more GPU memory**

**Important Notes**:

> [!IMPORTANT]
> The `global.workspace_size_limit` knob is dynamically provided only when applicable. It will not appear in the knobs list for non-convolution operations (e.g., batchnorm, pointwise).

> [!WARNING]
> Setting a workspace limit below the minimum required by all solvers will result in an error when creating the execution plan.

**Example**:
```cpp
// Query knobs to find valid range
std::vector<Knob> knobs;
graph.get_knobs_for_engine(engineId, knobs);

for (const auto& knob : knobs) {
    if (knob.knobId() == "global.workspace_size_limit") {
        // Check constraints to see valid range
        const auto* constraint = knob.constraint();
        std::cout << "Workspace range: " << constraint->toString() << "\n";
    }
}

// Limit workspace to 32 MB
std::vector<KnobSetting> settings;
settings.emplace_back("global.workspace_size_limit", 32LL * 1024 * 1024);
graph.create_execution_plan_ext(engineId, settings);
```

---

## Usage Examples

### Example 1: Query Available Knobs

```cpp
#include <hipdnn_frontend.hpp>

using namespace hipdnn_frontend;

// After building the graph
Graph graph;
// ... setup and build graph ...

// Query knobs for MIOpen engine
std::vector<Knob> knobs;
auto error = graph.get_knobs_for_engine(MIOPEN_ENGINE_ID, knobs);

if (error.is_good()) {
    std::cout << "Available knobs for MIOpen engine:\n";
    for (const auto& knob : knobs) {
        std::cout << "  " << knob.knobId() << ": " << knob.description() << "\n";
        std::cout << "  Default: ";

        const auto& defaultVal = knob.defaultValue();
        if (std::holds_alternative<int64_t>(defaultVal)) {
            std::cout << std::get<int64_t>(defaultVal);
        }
        std::cout << "\n";

        if (const auto* constraint = knob.constraint()) {
            std::cout << "  Constraint: " << constraint->toString() << "\n";
        }
        std::cout << "\n";
    }
}
```

### Example 2: Combined Knob Settings

```cpp
#include <hipdnn_frontend.hpp>

using namespace hipdnn_frontend;

Graph graph;
// ... setup convolution graph ...
graph.build_operation_graph(handle);

// Enable benchmarking and set workspace limit
std::vector<KnobSetting> settings;
settings.emplace_back("global.benchmarking", 1);
settings.emplace_back("global.workspace_size_limit", 128LL * 1024 * 1024);

auto error = graph.create_execution_plan_ext(MIOPEN_ENGINE_ID, settings);

if (error.is_good()) {
    std::cout << "Execution plan created with custom knob settings\n";
}
```

---

## Best Practices

### For Development

1. **Start with defaults**: Use default knob values during initial development
2. **Profile first**: Measure baseline performance before tuning knobs
3. **Query knobs**: Always check available knobs and their constraints using `get_knobs_for_engine()`
4. **Test incremental changes**: Modify one knob at a time to understand impact

### For Production

1. **Enable benchmarking during warm-up**:
   ```cpp
   // Warm-up phase
   std::vector<KnobSetting> warmupSettings;
   warmupSettings.emplace_back("global.benchmarking", 1);
   graph.create_execution_plan_ext(engineId, warmupSettings);

   // Execute a few times to populate cache
   for (int i = 0; i < 5; i++) {
       graph.execute(handle, variantPack, workspace);
   }
   ```

2. **Use cached results in production**: After warm-up, benchmarking can be disabled as results are cached

3. **Document knob settings**: Keep a record of knob configurations used in production for reproducibility

### For Memory-Constrained Environments

1. **Query workspace ranges**:
   ```cpp
   // Find minimum and maximum workspace for the operation
   std::vector<Knob> knobs;
   graph.get_knobs_for_engine(engineId, knobs);

   for (const auto& knob : knobs) {
       if (knob.knobId() == "global.workspace_size_limit") {
           // Log constraint to understand valid range
       }
   }
   ```

2. **Set conservative limits**: Start with a lower workspace limit and increase if performance is insufficient

3. **Balance batch size and workspace**: Reducing workspace allows larger batch sizes, which may offset performance loss

### Error Handling

Always check for errors when setting knobs:

```cpp
auto error = graph.create_execution_plan_ext(engineId, settings);
if (!error.is_good()) {
    std::cerr << "Failed to create execution plan: " << error.get_message() << "\n";

    // Common errors:
    // - Workspace limit below minimum
    // - Invalid knob ID
    // - Value outside valid range
}
```

---

## Related Documentation

- [hipDNN Knobs Documentation](../../../projects/hipdnn/docs/Knobs.md) - Comprehensive knobs system guide
- [MIOpen Provider Operation Support](./OperationSupport.md) - Supported operations
- [MIOpen Documentation](https://rocmsoftwareplatform.github.io/MIOpen/doc/html/) - MIOpen library documentation
- [Plugin Development Guide](../../../projects/hipdnn/docs/PluginDevelopment.md#providing-knobs) - How to implement knobs in plugins
