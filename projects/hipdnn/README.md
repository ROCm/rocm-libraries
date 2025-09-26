# hipDNN

> [!CAUTION]
> **hipDNN is in the early stages of development. There is currently limited functionality available to solve problems. See [Operation Support](./docs/OperationSupport.md) for reference.**

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
- [Contributing](#contributing)

---

## Getting Started

The fastest way to get started with hipDNN is to follow the [quick start steps in the build guide](./docs/Building.md#quick-start-guide).

---

## Documentation

### User Guides
- **[Building](./docs/Building.md)** - Prerequisites, build configurations, and platform-specific instructions
- **[How-To](./docs/HowTo.md)** - Using hipDNN components and extending the framework
- **[Environment Configuration](./docs/Environment.md)** - Runtime configuration and logging setup
- **[Operation Support](./docs/OperationSupport.md)** - Currently supported operations and their status
- **[Samples](./samples/README.md)** - Frontend usage examples

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
| **[Frontend](./frontend/)** | Header-only C++ API wrapper around the backend |
| **[SDK](./sdk/)** | Header-only library for plugin development and utilities |
| **[Plugins](./plugins/)** | Plugin implementations, including [MIOpen Legacy Plugin](./plugins/miopen_legacy_plugin/) |
| **[Samples](./samples/)** | Example implementations demonstrating hipDNN usage |
| **[Tests](./tests/)** | Tests for the public API (incl. frontend integration tests) |

### Docker Support
See [Docker README](./dockerfiles/README.md) for containerized development environments.

---

## Contributing

For information about contributing to the hipDNN project, please see the [Contributing Guide](./CONTRIBUTING.md).
