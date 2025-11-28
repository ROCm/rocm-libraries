# hipDNN Roadmap

This document outlines the development roadmap for hipDNN, a comprehensive graph-based deep learning library for AMD GPUs. For current operation support details, refer to the [Operation Support documentation](./OperationSupport.md).

> [!NOTE]
> 📝 This roadmap is subject to change based on project priorities, community feedback, and technical requirements.

## hipDNN Core

The following improvements represent foundational changes spanning all hipDNN components.

### Near-Term Priorities

- **Version Management**: Implement consistent version numbering across Frontend, SDK, and Backend components
- **EngineId Management**: Update engineId registration to support plugin development workflows
- **Enhanced Logging**: Add detailed logging for API calls and runtime graph serialization capture
- **Persistence**: Implement graph save and load functionality
- **Benchmarking & Validation**: Develop Python-based tool suite for graph execution, performance measurement, and validation

### Longer-Term Priorities

- **Execution Plan Persistence**: Implement save and load functionality for execution plans
- **API Extensions**: Add support for behavioral notes and tunable knobs
- **Plugin Systems**:
  - Benchmarking and tuning plugin system (see [Design.md](./Design.md#high-level-architecture))
  - Heuristic plugin system (see [Design.md](./Design.md#high-level-architecture))

## Frontend

The Frontend provides the user-facing C++ API, focusing on usability and feature completeness.

### Near-Term Priorities

- **Python API**: Add Python frontend for broader accessibility
- **Engine Selection**: Enhance documentation and headers for available engines and preferred source selection

### Longer-Term Priorities

- **Extended Operations**: Support additional operation types
- **Dynamic Loading**: Enable runtime loading of hipDNN backend libraries

## Backend

The Backend manages plugins and orchestrates graph execution. See [Design.md](./Design.md) for detailed architecture information.

### Longer-Term Priorities

- **C API Support**: Add graph building C API for language interoperability
- **Custom Schema Support**: Allow users to extend graphs without recompiling hipDNN backend

## SDK

The SDK provides shared utilities and interfaces ensuring compatibility between Frontend, Backend, and Plugins.

### Near-Term Priorities

- **SDK Refactoring**: Split SDK into focused sub-components:
  - Testing SDK
  - Plugin Development SDK
  - Graph Data Object SDK
- **Testing SDK**: Extract testing utilities from MIOpen plugin and reference implementation into a unified SDK for plugin integration testing
- **Plugin Development SDK**: Provide plugin headers and utilities for bootstrapping plugin development

## Plugins

Plugins extend hipDNN's computational capabilities. See [Design.md](./Design.md#engine-plugins) for plugin architecture details.

### MIOpen Plugin

#### Near-Term Priorities

- **Fusion Support**: Complete integration for Convolution fusions
- **Batchnorm Inference**: Re-enable support after resolving [known issues](https://github.com/ROCm/rocm-libraries/issues/2459)
- **Batchnorm Running Stats**: Add support for running statistics in Batchnorm operations
- **Code Refactoring**: Extract common MIOpen plugin code into reusable SDK components

### Fusilli IREE Plugin

#### Near-Term Priorities

- **Batchnorm Inference**: Implement Batchnorm inference with fusion support
- **Convolution**: Implement Forward Convolution with fusion support

#### Mid-Term Priorities

- **Batchnorm Training**: Implement Batchnorm training with fusion support
- **Convolution**: Implement Backward Convolution (Data & Weight) with fusion support
- **GEMM**: Implement GEMM operations with fusion support

#### Longer-Term Priorities

- **Attention**: Implement Attention operations with fusion support

### General Plugin Ecosystem

#### Longer-Term Priorities

- **Extended Plugin Support**: Develop additional plugins to broaden graph support

## Testing and Performance

### Near-Term Priorities

- **Sample Validation**: Integrate [samples](../samples/README.md) into CI to validate installation and ensure functionality
- **Reference Plugin**: Create a reference plugin using the Test SDK for validation against other plugins

### Longer-Term Priorities

- **ASAN Integration**: Add AddressSanitizer as an automated CI step

## Contributing

hipDNN is an open-source project that welcomes community contributions. Your feedback shapes the project's direction.

For contribution guidelines, see [CONTRIBUTING.md](../CONTRIBUTING.md). For questions or suggestions, please open an issue in the hipDNN repository.
