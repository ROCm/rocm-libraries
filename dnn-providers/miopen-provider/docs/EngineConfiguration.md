# MIOpen Provider Plugin - Engine Configuration

This document describes the configuration knobs available for the MIOpen Provider Plugin.

## Available Knobs

The MIOpen Provider supports the following configuration knobs:

| Knob Name | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `global.benchmarking` | Integer | 0 (disabled) | 0-1 | Enable benchmarking for kernel selection |
| `global.workspace_size_limit` | Integer | INT64_MAX | 0 to INT64_MAX | Maximum workspace size in bytes |

## Knob Details

### Benchmarking

The `global.benchmarking` knob enables benchmarking for kernel selection:

- **0 (default)**: Disabled - benchmarking disabled
- **1**: Enabled - use benchmarking for kernel selection

> [!NOTE]
> This knob currently only stores the value. The benchmarking functionality for kernel selection will be implemented in a future update.

### Workspace Size Limit

The `global.workspace_size_limit` knob controls the maximum amount of workspace memory that MIOpen operations can use.

Different MIOpen algorithms require different amounts of workspace memory, varying based on the operation type and tensor dimensions.

**Current Implementation:**
- **Minimum**: 0 bytes (no workspace allowed)
- **Maximum**: INT64_MAX bytes
- **Default**: INT64_MAX bytes (no limit - allows algorithms to use whatever workspace they need)

> [!NOTE]
> The knob currently uses INT64_MAX as a conservative default. In a future update, the minimum, maximum, and default values will be dynamically determined by querying MIOpen for the actual workspace requirements of the specific operation being executed. These values will range from 0 to INT64_MAX bytes, with the default set to the maximum workspace size that MIOpen can utilize for optimal performance.

> [!NOTE]
> This knob currently only stores the limit value. Enforcement of the limit during operation execution will be implemented in a future update.
