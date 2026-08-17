# hipDNN Knobs Usage Sample

This sample demonstrates how to use hipDNN's engine configuration knobs system.

## What are Knobs?

Knobs are runtime-configurable parameters that control engine behavior, performance tuning, and feature selection. They allow you to:
- Enable/disable features (e.g., benchmarking mode)
- Tune performance parameters (e.g., workspace size limits)
- Select algorithmic variants
- Control memory usage

## What This Sample Demonstrates

1. **Querying Available Knobs**: How to discover what knobs an engine supports
2. **Knob Metadata**: Understanding knob types, constraints, and default values
3. **Setting Knob Values**: Creating execution plans with custom knob settings
4. **Knob Validation**: Validating settings against knob constraints
5. **Knob Value Types**: Using integer, float, and string knobs
6. **Real Execution**: Running graphs with different knob configurations

## Building

The sample is built as part of the hipDNN samples:

```bash
cd /path/to/hipdnn/samples/build
cmake ..
make knobs_usage
```

## Running

### Basic Usage

```bash
./knobs_usage
```

This runs all demonstrations including actual graph executions.

### Skip Execution Examples

To see knob API demonstrations without running graphs:

```bash
./knobs_usage --skip-execution
```

This is faster and useful when you only want to learn about the knob API.

### Help

```bash
./knobs_usage --help
```

## Sample Output

The sample produces detailed output organized into sections:

```
=========================================
  hipDNN Knobs Usage Sample
=========================================

1. Querying Available Knobs
   - Lists all knobs for an engine
   - Shows knob metadata (type, description, constraints, defaults)

2. Using Knob Lookup Map
   - Demonstrates map-based knob lookup
   - Shows how to find specific knobs by ID

3. Using Default Knob Values
   - Creates execution plan with defaults
   - Demonstrates simplest knob usage

4. Setting Custom Knob Values
   - Shows how to set integer knob values
   - Demonstrates creating plans with custom settings

5. Knob Validation
   - Validates settings against constraints
   - Shows both valid and invalid examples

6. Different Knob Value Types
   - Integer knobs (int64_t)
   - Float knobs (double)
   - String knobs

7. Execution with Default Knobs
   - Runs actual batchnorm graph with defaults

8. Execution with Benchmarking Enabled
   - Runs graph with global.benchmarking=1
```

## Key Code Patterns

### Query Available Knobs

```cpp
std::vector<Knob> knobs;
auto error = graph.get_knobs_for_engine(engineId, knobs);

for (const auto& knob : knobs) {
    std::cout << knob.knobId() << ": " << knob.description() << "\n";
}
```

### Set Custom Knob Values

```cpp
std::vector<KnobSetting> settings;
settings.emplace_back("global.benchmarking", 1);
settings.emplace_back("global.workspace_size_limit", 64 * 1024 * 1024);

auto error = graph.create_execution_plan_ext(engineId, settings);
```

### Validate Knob Settings

```cpp
KnobSetting setting("global.benchmarking", 1);
auto validationError = knob.validate(setting);

if (validationError.is_good()) {
    std::cout << "Valid!\n";
}
```

## Common Knobs

### global.benchmarking

- **Type**: int64
- **Default**: 0 (disabled)
- **Range**: 0-1
- **Purpose**: Enable MIOpen kernel benchmarking for optimal performance

### global.workspace_size_limit

- **Type**: int64
- **Default**: Maximum
- **Range**: Dynamic (operation-specific)
- **Purpose**: Limit workspace memory for convolution operations
- **Note**: Only available for convolution operations

## Related Documentation

For comprehensive information about knobs:

- **[hipDNN Knobs Documentation](../../docs/Knobs.md)** - Complete guide with API reference
- **[HowTo Guide](../../docs/HowTo.md#configuring-engine-knobs)** - Quick start guide
- **[Plugin Development Guide](../../docs/PluginDevelopment.md#providing-knobs)** - For plugin authors
- **[MIOpen Provider Knobs](../../../../dnn-providers/miopen-provider/docs/Knobs.md)** - MIOpen-specific knobs

## Notes

- The sample uses a batchnorm operation for demonstration, but knobs work with all operations
- Not all engines support all knobs; always query available knobs first
- Unknown knobs are ignored with a warning (not an error)
- Invalid knob values will cause execution plan creation to fail
- Knobs are set at execution plan creation time, not at execution time

## Tips

1. **Always query first**: Use `get_knobs_for_engine()` to understand what's available
2. **Check constraints**: Look at the constraint object to understand valid ranges
3. **Use defaults when possible**: Only customize knobs when needed
4. **Handle errors**: Always check error return values when setting knobs
5. **Read the docs**: See the comprehensive documentation for advanced usage
