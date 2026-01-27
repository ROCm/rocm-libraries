# MIOpen Provider Plugin - Engine Configuration

This document describes the configuration knobs available for the MIOpen Provider Plugin.

## Available Knobs

The MIOpen Provider supports the following configuration knobs:

| Knob Name | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `global.benchmarking` | Integer | 0 (disabled) | 0-1 | Enable benchmarking for kernel selection |
| `global.workspace_size_limit` | Integer | -1 (unlimited) | -1 to INT64_MAX | Maximum workspace size in bytes. -1 means unlimited, 0 means no workspace allowed |

## Knob Details

### Benchmarking

The `global.benchmarking` knob enables benchmarking for kernel selection:

- **0 (default)**: Disabled - benchmarking disabled
- **1**: Enabled - use benchmarking for kernel selection

> [!NOTE]
> This knob currently only stores the value. The benchmarking functionality for kernel selection will be implemented in a future update.

### Workspace Size Limit

The `global.workspace_size_limit` knob controls the maximum amount of workspace memory that MIOpen operations can use:

- **-1 (default)**: No limit - operations can request any workspace size they need
- **0**: No workspace allowed - only operations that don't require workspace can execute
- **Positive values**: Maximum workspace size in bytes

> [!NOTE]
> This knob currently only stores the limit value. Enforcement of the limit during operation execution will be implemented in a future update.
