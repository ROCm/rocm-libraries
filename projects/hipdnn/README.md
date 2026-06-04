# hipDNN

> [!CAUTION]
> **hipDNN is in the early stages of development. There is currently limited functionality available to execute graphs. See [Operation Support](./docs/OperationSupport.md) for reference.**

> [!NOTE]
> The published hipDNN documentation is available [here](https://rocm.docs.amd.com/projects/hipdnn/en/latest/index.html) in an organized, easy-to-read format, with search and a table of contents. The documentation source files reside in the `docs` folder of this repository. As with all ROCm projects, the documentation is open source. For more information on contributing to the documentation, see [Contribute to ROCm documentation](https://rocm.docs.amd.com/en/latest/contribute/contributing.html).

## Overview

hipDNN is a graph-based deep learning library for AMD GPUs that leverages a flexible plugin architecture to provide optimized implementations and utilities for various routines.

---

## Table of Contents

- [Getting Started](#getting-started)
- [Documentation](#documentation)
  - [User Guides](#user-guides)
  - [Developer Guides](#developer-guides)
  - [Testing](#testing)
- [Project Structure](#project-structure)
- [API Documentation](#api-documentation)
- [Contributing](#contributing)

---

## Getting Started

The fastest way to get started with hipDNN is to follow the [quick start steps in the build guide](./docs/Building.md#quick-start-guide).

---

## Documentation

### User Guides
- **[Building](./docs/Building.md)** - Prerequisites, build configurations, and platform-specific instructions
- **[Consumer Quick Start](./docs/ConsumerQuickStart.md)** - Using an installed hipDNN in your CMake project
- **[How-To](./docs/HowTo.md)** - Using hipDNN components and extending the framework
- **[Environment Configuration](./docs/Environment.md)** - Runtime configuration and logging setup
- **[Operation Support](./docs/OperationSupport.md)** - Currently supported operations and their status
- **[Samples](./samples/README.md)** - Frontend usage examples
- **[API Reference](#api-documentation)** - Doxygen-generated API documentation

### Developer Guides
- **[Design Overview](./docs/Design.md)** - Architecture and design descriptions and diagrams
- **[Extending hipDNN](./docs/HowTo.md#extending-hipdnn)** - How to extend hipDNN functionality
- **[Plugin Development](./docs/PluginDevelopment.md)** - Creating and using custom plugins for hipDNN
- **[Roadmap](./docs/Roadmap.md)** - Feature priorities and development plans

### Testing
- **[Testing](./docs/Testing.md)** - Synopsis of testing information
- **[Testing Strategy](./docs/testing/TestingStrategy.md)** - Specific testing approach
- **[Test Plan](./docs/testing/TestPlan.md)** - Detailed test planning
- **[Test Run Template](./docs/testing/TestRunTemplate.md)** - Guidelines for test execution

---

## Project Structure

hipDNN is organized into several key components. For detailed architecture descriptions, see the [Design Overview](./docs/Design.md).

| Component | Description |
|-----------|-------------|
| **[Backend](./backend/)** | Core shared library providing C API for operation graphs and managing plugins |
| **[Data SDK](./data_sdk/)** | Header-only library with shared types, tensor utilities, logging, and the engine name registry |
| **[Flatbuffers SDK](./flatbuffers_sdk/)** | Header-only library with FlatBuffers schemas, generated headers, graph wrappers, and optional JSON helpers |
| **[Frontend](./frontend/)** | Header-only C++ API wrapper around the backend |
| **[Plugin SDK](./plugin_sdk/)** | Header-only library for plugin development |
| **[Samples](./samples/)** | Example implementations demonstrating hipDNN usage |
| **[Tests](./tests/)** | Tests for the public API (incl. frontend integration tests) |
| **[Tools](./tools/)** | Experimental utilities (e.g., benchmarking, engine listing) — subject to change |

> [!NOTE]
> Official hipDNN plugins can be found in the [dnn-providers](../../../dnn-providers/) folder (e.g., [MIOpen Plugin](../../../dnn-providers/miopen-provider/)).

### Docker Support
See [Docker README](./dockerfiles/README.md) for containerized development environments.

---

## API Documentation

hipDNN includes Doxygen-generated API documentation for the public C++ frontend.

### Building the Documentation

1. Install Doxygen:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install doxygen
   ```
   For Windows, download the installer from [doxygen.nl/download](https://www.doxygen.nl/download.html).

2. Generate the documentation:
   ```bash
   cd <project-root>
   doxygen Doxyfile
   ```

3. Open the generated documentation:
   ```bash
   # Linux
   xdg-open build/docs/html/index.html

   # Windows
   start build\docs\html\index.html
   ```

The documentation covers the frontend API including:
- Graph construction and execution
- Tensor and operation attributes
- Engine configuration and knobs
- Error handling

---

## Contributing

For information about contributing to the hipDNN project, please see the [Contributing Guide](./CONTRIBUTING.md).
