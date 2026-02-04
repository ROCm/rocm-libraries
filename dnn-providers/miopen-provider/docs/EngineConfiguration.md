# MIOpen Provider Plugin - Engine Configuration

This document describes the configuration knobs available for the MIOpen Provider Plugin.

## Available Knobs

The MIOpen Provider supports the following configuration knobs:

| Knob Name | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `global.benchmarking` | Integer | 0 (disabled) | 0-1 | Enable benchmarking for kernel selection |
| `global.workspace_size_limit` | Integer | Operation-specific (max) | Operation-specific | Maximum workspace size in bytes |

## Knob Details

### Benchmarking

The `global.benchmarking` knob enables benchmarking for kernel selection:

- **0 (default)**: Disabled - benchmarking disabled
- **1**: Enabled - use benchmarking for kernel selection

### Workspace Size Limit

The `global.workspace_size_limit` knob controls the maximum amount of workspace memory that MIOpen operations can use.
The knob values are **operation-specific** and dynamically determined based on the operation type and tensor dimensions:

- **Minimum**: The minimum workspace size required by the operation
- **Maximum**: The maximum workspace size that can be utilized by the operation
- **Default**: Set to the maximum workspace size for optimal performance
