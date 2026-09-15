# MIOpen Provider - Engine Configuration

## Quick Reference

The MIOpen Provider supports the following configuration knobs:

| Knob Name | Type | Default | Description |
|-----------|------|---------|-------------|
| `global.benchmarking` | int64 | 0 (disabled) | Enable benchmarking for optimal kernel selection |
| `global.workspace_size_limit` | int64 | Maximum | Limit workspace memory (convolution operations only) |

## Engine predictions

MIOpen exposes graph-only L1 predictions through the shared UHD runtime, without
requiring kernel descriptors or an L2 catalog. Predictions describe ordinary
untuned execution with `global.benchmarking=0`, not a separate cache-bypassing
selection mode. Existing tuning results may still be reused, and the process-wide
benchmarking override retains its normal precedence over the knob.

Missing or incompatible models leave ordinary engine execution available.
The selector revision identifies the untuned execution policy and the provider
and MIOpen versions; models collected for the removed isolated cache-bypass path
are not compatible.
See the [UHD generation guide](../../../projects/hipdnn/tools/uhd_gen/README.md)
for collection, training, promotion, and Mode A/B selection.

## See Also

- **[Knobs Documentation](./Knobs.md)** - Complete MIOpen knobs guide
- **[hipDNN Knobs](../../../projects/hipdnn/docs/Knobs.md)** - General knobs system documentation
- **[Operation Support](./OperationSupport.md)** - Supported operations
