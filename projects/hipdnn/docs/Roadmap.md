# hipDNN Roadmap

This document outlines the development roadmap for hipDNN, including near-term priorities and future goals for each component. hipDNN is actively evolving to provide a comprehensive graph-based deep learning library for AMD GPUs. For a table of active operation support, see the [Operation Support doc](./OperationSupport.md).

> [!NOTE]
> 📝 This roadmap is subject to change based on project priorities, community feedback, and technical requirements.

## hipDNN Core

The following improvements span across all hipDNN components and represent foundational changes to the project.

### What's Next
*The following items are current priorities but are subject to change:*

- **Build Platform Integration**: Ongoing integration into [https://github.com/ROCm/TheRock](https://github.com/ROCm/TheRock) build platform
- **Version Management**: Implementing consistent version numbering support across SDK and Backend components

### Future Roadmap
*The following items are longer-term goals that are not yet scheduled:*

- **Performance Infrastructure**: Adding a comprehensive benchmarking and performance tracking project for monitoring performance and accuracy across the full hipDNN installation (including all plugins)

## Frontend

The Frontend provides the user-facing C++ API for hipDNN, focusing on ease of use and feature completeness.

### What's Next
*The following items are current priorities but are subject to change:*

- **Fusion Support**: Adding operation support for Convolution & BatchNorm Fusions
- **Backend Selection**: Adding API to select preferred backend engines

### Future Roadmap
*The following items are longer-term goals that are not yet scheduled:*

- **Python API**: Adding Python frontend API for broader accessibility
- **Extended Operations**: Supporting additional operation types beyond current capabilities
- **Persistence**: Implementing functionality to save and load graphs and execution plans
- **Dynamic Loading**: Supporting runtime loading of hipDNN backend libraries
- **Filtering engines**: Adding API to control filtering of engines based off behavorial notes
- **Configurable engines**: Adding API to configure tunable settings provided by engines
- **Specify compute type**: Allow users to specify the accumulator compute types for plugins to use

## Backend

The Backend serves as the core engine of hipDNN, managing plugins and orchestrating graph execution. See [Design.md](./Design.md) for detailed architecture information.

### What's Next
*The following items are current priorities but are subject to change:*

### Future Roadmap
*The following items are longer-term goals that are not yet scheduled:*

- **Unique Engine IDs**: Mechanism to improve or ensure the global uniqueness of engine IDs
- **API Extensions**: Adding support for behavioral notes and tunable knobs
- **Improved Logging**: Enhancing logging capabilities for better debugging and monitoring
- **Plugin Systems**:
  - Benchmarking and tuning plugin system (see [Design.md](./Design.md#high-level-architecture) for architecture details)
  - Heuristic plugin system (see [Design.md](./Design.md#high-level-architecture) for architecture details)
- **Persistence**: Implementing functionality to save and load graphs and execution plans
- **C API Support**: Adding graph building C API support for language interoperability
- **Custom schema support**: Adding support to allow users to extend graphs without needing to recompile hipDNN backend

## SDK

The SDK provides shared utilities and interfaces that ensure compatibility between Frontend, Backend, and Plugins.

### What's Next
*The following items are current priorities but are subject to change:*

- **Fusion Support**: Validating the reference fusion support for Pointwise operations

### Future Roadmap
*The following items are longer-term goals that are not yet scheduled:*

The SDK may eventually be split into more focused sub-projects:
  - Core plugin interfaces
  - Graph manipulation utilities
  - Reference implementations
  - Performance utilities

With extended support for:
  - Caching mechanisms
  - Graph matching and manipulation
  - Additional operation schemas
  - Benchmarking and tuning utilities

## Plugins

Plugins extend hipDNN's capabilities by providing computational implementations. See [Design.md](./Design.md#engine-plugins) for plugin architecture details.

### MIOpen Plugin

#### What's Next
*The following items are current priorities but are subject to change:*

- **Convolution Integration**: Adding integration to support the remaining Convolution WRW operation
- **Batch Normalization**: Adding integration to support the remaining BatchNorm training operation
- **Fusion Support**: Adding integration to support Convolution and BatchNorm fusions

### General Plugin Ecosystem

#### Future Roadmap
*The following items are longer-term goals that are not yet scheduled:*

- **Extended Plugin Support**: Developing additional plugins to extend graph support
- **MIOpen Plugin Completion**: Finishing MIOpen plugin integration
- **Advanced Features**:
  - Adding support for behavioral notes and tunable knobs
  - Implementing heuristic plugins (see [Design.md](./Design.md#high-level-architecture) for details)
  - Implementing tuning and benchmarking plugins (see [Design.md](./Design.md#high-level-architecture) for details)

## Testing and Performance

This section covers testing infrastructure improvements and performance benchmarking capabilities for hipDNN.

### What's Next
*The following items are current priorities but are subject to change:*

- **Golden Reference Data**: Leverage our golden reference data exhaustively in unit testing at plugin level & to verify reference implementations
- **CI Platform**: Swap to leverage TheRock for CI
- **ASAN Integration**: Add ASAN as an automatic step to CI
- **Samples**: Run [samples](../samples/README.md) in CI to validate the installation.

### Future Roadmap
*The following items are longer-term goals that are not yet scheduled:*


- **Performance/Benchmarking Project**: Create a separate benchmarking and performance project for capturing performance and accuracy for full hipDNN graphs
  - Track performance and accuracy of hipDNN & installed plugins across a broad set of graphs
  - **Test Categories**:
    - Quick Suite: A quick running set of graphs to run per PR to flag severe regressions
    - Full Suite: A long running set of graphs to run on demand to flag broad regressions (nightly or weekly frequency)
  - **Requirements**:
    - Minimal set of graphs maintained for pre-checkin performance checks
    - Full set of graphs maintained for on-demand performance & accuracy checks
    - Requires GPU
    - Validates correctness and performance of graphs
  - **Testing Environments**:
    - Windows & supported Linux distros
    - Test on all ASICs supported by hipDNN
    - Note: Certain plugins/graphs may have ASIC restrictions

## Contributing to the Roadmap

As an open-source project, hipDNN welcomes community input. Your feedback helps shape the future direction of the project.

Please refer to [CONTRIBUTING.md](../CONTRIBUTING.md) for information on how to contribute to hipDNN development.

For questions or suggestions, please open an issue for hipDNN.
