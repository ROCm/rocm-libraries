# Consumer Quick Start

This guide shows how to consume an already-installed hipDNN in a CMake project. For building hipDNN from source, see the [Building Guide](./Building.md).

## Table of Contents

- [Prerequisites](#prerequisites)
- [Project Setup](#project-setup)
- [Building](#building)
- [Next Steps](#next-steps)

---

## Prerequisites

- **ROCm / TheRock** installed with hipDNN packages (see [TheRock Releases](https://github.com/ROCm/TheRock/blob/main/RELEASES.md))
- **CMake**
- **Ninja** (recommended)
- **C++17** compatible compiler

## Project Setup

### CMakeLists.txt

```cmake
cmake_minimum_required(VERSION <your_minimum>)
project(my_app LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(hipdnn_frontend CONFIG REQUIRED)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE hipdnn_frontend)
```

> [!NOTE]
> `find_package(hipdnn_frontend)` transitively brings in `hipdnn_backend`, `hipdnn_data_sdk`, and `hip` — only one `find_package` call is needed. HIP runtime APIs (`hipMalloc`, `hipFree`, etc.) are available through the transitive `hip::host` link dependency.

> [!NOTE]
> If CMake cannot find the packages, set `CMAKE_PREFIX_PATH` to the install location. hipDNN CMake files are installed to `/opt/rocm/lib/cmake` by default, which CMake may already search automatically depending on your system configuration.

### Source File

Include the frontend header to access the hipDNN graph API:

```cpp
#include <hipdnn_frontend.hpp>
```

See the [Samples](../samples/README.md) for complete working examples of building and executing operation graphs.

## Building

```bash
mkdir build && cd build
cmake -GNinja -DCMAKE_PREFIX_PATH=/opt/rocm ..
ninja
```

## Next Steps

- **[Samples](../samples/README.md)** — Working examples including convolution, batch normalization, and fused operations
- **[Operation Support](./OperationSupport.md)** — Currently supported operations and their status
- **[How-To Guide](./HowTo.md)** — Detailed component usage, CMake integration for all packages, and extending hipDNN
- **[Environment Configuration](./Environment.md)** — Runtime configuration and logging setup
