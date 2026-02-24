# MIOpen Provider - Engine Configuration

## Quick Reference

The MIOpen Provider supports the following configuration knobs:

| Knob Name | Type | Default | Description |
|-----------|------|---------|-------------|
| `global.benchmarking` | int64 | 0 (disabled) | Enable benchmarking for optimal kernel selection |
| `global.workspace_size_limit` | int64 | Maximum | Limit workspace memory (convolution operations only) |

## See Also

- **[Knobs Documentation](./Knobs.md)** - Complete MIOpen knobs guide
- **[hipDNN Knobs](../../../projects/hipdnn/docs/Knobs.md)** - General knobs system documentation
- **[Operation Support](./OperationSupport.md)** - Supported operations
