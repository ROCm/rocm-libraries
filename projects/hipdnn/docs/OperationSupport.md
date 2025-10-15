# hipDNN Operation Support

This document provides a comprehensive overview of the operations currently supported in hipDNN and the details of their implementation support.

## Current Operation Support

The following table lists all operations currently supported in hipDNN, along with their supported data types, layouts, sparse support status, and the plugin that provides the implementation.

| Graph Pattern | Datatypes | Layouts | Sparse Support | Plugin with Support |
|--------------|-----------|---------|----------------|-------------------|
| Batchnorm Inference | Fp16, BFp16, Float32 | NHWC, NCHW, NDHWC, NCDHW | No | MIOpen Legacy Plugin |
| Batchnorm Backwards | Fp16, BFp16, Float32 | NHWC, NCHW, NDHWC, NCDHW | No | MIOpen Legacy Plugin |
| Convolution Forward | Fp16, BFp16, Float32 | NHWC, NCHW, NDHWC, NCDHW | No | MIOpen Legacy Plugin |

## Notes

> [!IMPORTANT]
> ⚠️ **hipDNN is in the early phase of development.** The operation support table above reflects the current state of the library. We are actively working on expanding support for additional operations and features.

For information about upcoming operations and features, please refer to the [Roadmap.md](./Roadmap.md) document.

## Legend

### Datatypes
- **Fp16**: Half-precision floating point (16-bit)
- **BFp16**: Brain floating point (16-bit)
- **Float32**: Single-precision floating point (32-bit)

### Layouts
- **NHWC**: Batch, Height, Width, Channels
- **NCHW**: Batch, Channels, Height, Width
- **NDHWC**: Batch, Depth, Height, Width, Channels (for 3D operations)
- **NCDHW**: Batch, Channels, Depth, Height, Width (for 3D operations)

### Plugins
- **MIOpen Legacy Plugin**: Integration with AMD's MIOpen library for GPU-accelerated operations

## Contributing

As hipDNN evolves, this document will be updated to reflect new operation support.

If you're interested in contributing to expand operation support, please see our [CONTRIBUTING.md](../CONTRIBUTING.md) guide.
