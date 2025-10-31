# hipDNN Operation Support

This document provides a comprehensive overview of the operations currently supported in hipDNN and the details of their implementation support.

> [!IMPORTANT]
> ⚠️ **hipDNN is in the early phase of development.** The operation support table below reflects the current state of the library. We are actively working on expanding support for additional operations and features.

## Current Operation Support

The following table lists all operations currently supported in hipDNN:

| Operation | Datatypes | Layouts | Plugin | Notes |
|-----------|-----------|---------|--------|-------|
| Batchnorm Inference | FP16, BFP16, FP32 | NCHW, NHWC, NCDHW, NDHWC | MIOpen |  |
| Batchnorm Training  | FP16, BFP16, FP32 | NCHW, NHWC, NCDHW, NDHWC | MIOpen | Forward pass with statistics |
| Batchnorm Backward  | FP16, BFP16, FP32 | NCHW, NHWC, NCDHW, NDHWC | MIOpen | Gradients for scale, bias, input |
| Convolution Forward | FP16, BFP16, FP32 | NCHW, NHWC, NCDHW, NDHWC | MIOpen | Cross-correlation only¹ |
| Convolution Dgrad   | FP16, BFP16, FP32 | NCHW, NHWC, NCDHW, NDHWC | MIOpen | Cross-correlation only¹ |

¹ See Convolution Operations note below

## Operation Notes

> [!NOTE]
> **Convolution Operations:** Currently, only cross-correlation convolutions are supported. True mathematical convolution (with kernel flipping) is not yet implemented. In practice, cross-correlation is the standard operation used in modern deep learning frameworks.

> [!NOTE]
> **Sparse Support:** All operations currently work with dense tensors only. Sparse tensor support is planned for future releases.

## Legend

### Datatypes
- **FP16**: Half-precision floating point (16-bit)
- **BFP16**: Brain floating point (16-bit)
- **FP32**: Single-precision floating point (32-bit)

### Layouts
- **NCHW**: Batch, Channels, Height, Width (2D, channel-first)
- **NHWC**: Batch, Height, Width, Channels (2D, channel-last)
- **NCDHW**: Batch, Channels, Depth, Height, Width (3D, channel-first)
- **NDHWC**: Batch, Depth, Height, Width, Channels (3D, channel-last)

### Plugins
- **MIOpen**: MIOpen Legacy Plugin - Integration with AMD's MIOpen library for GPU-accelerated operations

## Future Operations

For information about upcoming operations and features, please refer to the [Roadmap](./Roadmap.md) document.

## Contributing

As hipDNN evolves, this document will be updated to reflect new operation support.

If you're interested in contributing to expand operation support, please see our [Contributing Guide](../CONTRIBUTING.md).
