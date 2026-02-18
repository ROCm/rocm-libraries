# MIOpen Provider Plugin - Engine Configuration

This document describes the configuration knobs available for the MIOpen Provider Plugin.

## Knob Types

The MIOpen Provider supports two types of configuration knobs:

- **Global Knobs**: Always available for all operations
- **Custom Knobs**: Operation-specific knobs provided by plan builders when applicable to a given graph

## Available Knobs

The MIOpen Provider supports the following configuration knobs:

| Knob Name | Type | Knob Type | Default | Range | Description |
|-----------|------|-----------|---------|-------|-------------|
| `global.benchmarking` | Integer | Global | 0 (disabled) | 0-1 | Enable benchmarking for kernel selection |
| `global.workspace_size_limit` | Integer | Custom | Maximum | Dynamic | Maximum workspace size in bytes (convolution operations only) |

## Knob Details

### Benchmarking

The `global.benchmarking` knob enables benchmarking for kernel selection:

- **0 (default)**: Benchmarking disabled
- **1**: Enabled - use benchmarking for kernel selection

### Workspace Size Limit

The `global.workspace_size_limit` knob controls the maximum amount of workspace memory that MIOpen operations can use.

**Important**: This is a **custom knob** available only for **convolution operations** (Forward, Backward Data, Backward Weights). It is not available for batchnorm or other operations.

The knob values are **dynamically determined** at runtime by querying all available MIOpen solutions for the specific operation and tensor configuration:

- **Minimum**: The smallest workspace size required by any available kernel for the operation
- **Maximum**: The largest workspace size that can be utilized by any available kernel for the operation
- **Default**: Set to the maximum workspace size for optimal performance

The valid range varies based on:
- Operation type (Forward/Backward Data/Backward Weights)
- Tensor dimensions and data types
- Available MIOpen kernels for the specific configuration

**Example**: For a specific convolution forward operation, MIOpen might report:
- Minimum: 512 KB (lightweight kernel)
- Maximum: 128 MB (high-performance kernel)
- Default: 128 MB

Reducing the workspace size limit may force selection of kernels that require less memory but may have lower performance.
