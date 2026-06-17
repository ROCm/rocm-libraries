# hipDNN Operation Support

This document provides an overview of operation support in hipDNN and guides you to detailed information about specific plugin implementations.

## About hipDNN Operation Support

hipDNN is a plugin-based deep learning library that provides graph-based operation support through various backend plugins. Each plugin implements specific operations with support for different datatypes, layouts, and features.

> [!IMPORTANT]
> ⚠️ **hipDNN is in the early phase of development.** Operation support continues to expand as the library matures. Check individual plugin documentation for current support details.

## Plugin-Specific Operation Support

hipDNN operations are implemented through plugins. Each plugin provides its own set of supported operations. For detailed information about what operations are available, please refer to the plugin-specific documentation:

### Available Plugins

- **[MIOpen Provider Plugin](../../../dnn-providers/miopen-provider/docs/OperationSupport.md)** - Integration with AMD's MIOpen library for GPU-accelerated deep learning operations
  - Convolution operations (Forward, Dgrad, Wgrad)
  - Batchnorm operations (Training, Backward, Inference)
  - Fused operation graphs

- **[hipBLASLt Provider Plugin](../../../dnn-providers/hipblaslt-provider/docs/OperationSupport.md)** - Integration with AMD's hipBLASLt library that provides optimized GEMM operations.

### Reference Implementation

- **[CPU Reference Implementation](./OperationSupport-ReferenceImpl.md)** - CPU-based reference implementation for validation and testing
  - Provides ground-truth results for validating GPU implementations
  - Supports core operations (Convolution, Batchnorm, Pointwise)
  - Not intended for performance or production use

## Roadmap

For information about planned features, upcoming operations, and the development roadmap, please see:

- **[hipDNN Roadmap](./Roadmap.md)** - Comprehensive roadmap covering all hipDNN components, including planned plugin enhancements and new operation support

## Contributing

We welcome contributions to expand operation support in hipDNN!

For detailed contribution guidelines, please see:

- **[Contributing Guide](../CONTRIBUTING.md)** - Complete guide to contributing to hipDNN
- **[Plugin Development](./PluginDevelopment.md)** - Guide for creating and extending plugins

### Getting Started

1. Review the [hipDNN Design](./Design.md) to understand the architecture
2. Check the [Roadmap](./Roadmap.md) for planned features and contribution opportunities
3. Open a GitHub issue to discuss your planned contribution
4. Follow the [Contributing Guide](../CONTRIBUTING.md) for code quality and testing requirements
