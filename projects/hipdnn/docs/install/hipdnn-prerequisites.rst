.. meta::
  :description: Component install prerequisites
  :keywords: Component, ROCm, install, prerequisites

********************
hipDNN prerequisites
********************

System requirements
===================

- **GPU**: AMD GPU with ROCm support
- **Operating System**:

  - **Linux**: Any distribution supported by [TheRock](https://github.com/ROCm/TheRock), such as Ubuntu 24
  - **Windows**: Windows 11 (limited support, see [Windows section](#windows))

Dependencies
============

Prebuilt binaries and Docker files are available to provide a consistent development environment with all dependencies pre-installed. 
This is the recommended approach for most users. For more details about these Docker images, see the [Docker README](../dockerfiles/README.md). 
Dockerfile development environments are not available for Windows. Refer to the [Windows](#windows) section for details on building under Windows.

Required dependencies
---------------------

| Dependency | Version | Description |
|------------|---------|-------------|
| ROCm | Matching TheRock (ROCm version 7.0+) | AMD GPU programming stack (see [TheRock releases](https://github.com/ROCm/TheRock/releases)) |
| CMake | 3.25.2+ | Build system generator |
| Ninja | 1.12.1+ | Faster build system (recommended) |
| C++ Compiler | C++17 compatible | hipDNN requires C++17 compatible AMD Clang (plugins using device code may require C++20)|
| HIP | Matching TheRock | GPU programming interface (included with ROCm/TheRock) |
| clang-format | 18.x | Code formatting tool |
| clang-tidy | 20.x | Static analysis tool |
| LLVM Tools | 20.x | LLVM tools for code_coverage, and ASAN enabled builds |

Optional dependencies
---------------------

| Dependency | Version | Description |
|------------|---------|-------------|
| Docker | Latest | For containerized build environment |
| Python3 | Latest | For test name validation |

.. tip::

  See [Docker README](../dockerfiles/README.md) for details on using prebuilt binaries in Docker containers to ensure a consistent build environment.

Third-party libraries
---------------------

The following libraries are automatically managed by CMake (see [Dependencies.cmake](../cmake/Dependencies.cmake)):
- [FlatBuffers](https://github.com/google/flatbuffers) - Serialization library
- [Google Test](https://github.com/google/googletest) - Unit testing framework
- [spdlog](https://github.com/gabime/spdlog) - Logging library