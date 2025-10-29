# Building hipDNN

## Table of Contents
- [Prerequisites](#prerequisites)
  - [System Requirements](#system-requirements)
  - [Dependencies](#dependencies)
- [Quick Start Guide](#quick-start-guide)
  - [Using Docker (Recommended)](#using-docker-recommended)
  - [Native Build](#native-build)
- [Build Configurations](#build-configurations)
- [Build Targets](#build-targets)
- [Platform-Specific Instructions](#platform-specific-instructions)
  - [Linux](#linux)
  - [Windows](#windows)
- [Troubleshooting](#troubleshooting)
- [Verifying Installation](#verifying-installation)

## Prerequisites

### System Requirements
- **GPU**: AMD GPU with ROCm support
- **Operating System**: 
  - Linux: Matching support to [TheRock](https://github.com/ROCm/TheRock)
  - Windows: Windows 11 (limited support, see [Windows section](#windows))

### Dependencies

#### Required Dependencies
| Dependency | Version | Description |
|------------|---------|-------------|
| ROCm | Matching TheRock (ROCm version 7.0+) | AMD GPU programming stack (see [TheRock releases](https://github.com/ROCm/TheRock/releases)) |
| CMake | 3.25.2+ | Build system generator |
| C++ Compiler | C++17 compatible | hipDNN requires C++17 compatible AMD Clang (plugins using device code may require C++20)|
| HIP | Matching TheRock | GPU programming interface (included with ROCm/TheRock) |
| clang-format | 18.x | Code formatting tool |
| clang-tidy | 20.x | Static analysis tool |
| LLVM Tools | 20.x | LLVM tools for code_coverage, and ASAN enabled builds |

#### Optional Dependencies
| Dependency | Version | Description |
|------------|---------|-------------|
| Ninja | 1.12.1+ | Faster build system (recommended) |
| Docker | Latest | For containerized builds |
| Python3 | Latest | For test name validation | 

#### Third-Party Libraries
The following libraries are automatically managed by CMake (see [Dependencies.cmake](../cmake/Dependencies.cmake)):
- [FlatBuffers](https://github.com/google/flatbuffers) - Serialization library
- [Google Test](https://github.com/google/googletest) - Unit testing framework
- [spdlog](https://github.com/gabime/spdlog) - Logging library

## Quick Start Guide

### Using Docker (Recommended)

> [!TIP]
> 💡 Docker provides a consistent development environment with all dependencies pre-installed. This is the recommended approach for most users. For more details about Docker images, see the [Docker README](../dockerfiles/README.md).

1. **Clone hipDNN**
   ```bash
   git clone https://github.com/ROCm/hipDNN.git
   ```

2. **Build the Development Container**
   ```bash
   cd hipDNN/dockerfiles/
   
   # For Ubuntu 24.04 using prebuilt tarballs with gfx94X support (recommended)
   # (see Docker README for other options if needed)
   docker build -f ./Dockerfile.ubuntu24 -t hipdnn-dev:ubuntu24 .
   ```

3. **Run the Container**
   ```bash
   # Replace <path/to/hipDNN> with your hipDNN repository path
   docker run -it \
     -v <path/to/hipDNN>:/workspace/hipDNN \
     --privileged \
     --rm \
     --device=/dev/kfd \
     --device=/dev/dri:/dev/dri:rw \
     --volume=/dev/dri:/dev/dri:rw \
     -v /var/lib/docker:/var/lib/docker \
     --group-add video \
     --cap-add=SYS_PTRACE \
     --security-opt seccomp=unconfined \
     hipdnn-dev:ubuntu24
   ```

4. **Build and Test**
   ```bash
   cd /workspace/hipDNN
   mkdir build && cd build
   cmake -GNinja ..
   ninja check
   ```

5. **Install**
   ```bash
   # Default installation to /opt/rocm
   sudo ninja install
   ```

### Native Build

1. **Install ROCm** (follow [official ROCm installation guide](https://rocm.docs.amd.com/))

2. **Clone and Build**
   ```bash
   git clone https://github.com/ROCm/hipDNN.git
   cd hipDNN
   mkdir build && cd build
   
   # Configure with Ninja (recommended)
   cmake -GNinja ..
   
   # Build and run tests
   ninja check

   # Install
   sudo ninja install
   ```

## Build Configurations

### Release Build (Default)
```bash
cmake -GNinja ..
```

### Debug Build
```bash
cmake -GNinja -DCMAKE_BUILD_TYPE=Debug ..
```

### Code Coverage Build
```bash
cmake -GNinja -DCODE_COVERAGE=ON ..
ninja code_coverage
# Coverage reports will be generated in build/coverage/
```

### Address Sanitizer Build
```bash
cmake -GNinja -DBUILD_ADDRESS_SANITIZER=ON ..
ninja check
# Note: Some HIP-related tests may be skipped due to AddressSanitizer incompatibility
```

### Building Specific Components
```bash
# Build without plugins
cmake -GNinja -DHIP_DNN_BUILD_PLUGINS=OFF ..

# Build only the backend
cmake -GNinja -DHIP_DNN_BUILD_FRONTEND=OFF ..

# Build without samples
cmake -GNinja -DHIP_DNN_BUILD_SAMPLES=OFF ..
```

### ROCM_PATH & CMAKE_INSTALL_PREFIX

- **`ROCM_PATH`**: Specifies where the build system looks for ROCm dependencies (default: `/opt/rocm`)
- **`CMAKE_INSTALL_PREFIX`**: Specifies where hipDNN will be installed (defaults to `ROCM_PATH`)

These variables can be set independently:

```bash
# Default: Both use /opt/rocm
cmake -GNinja ..

# Install hipDNN to custom location, find ROCm dependencies in /opt/rocm
cmake -GNinja -DCMAKE_INSTALL_PREFIX=/custom/install/path ..

# Both custom (if ROCm is in a non-standard location)
cmake -GNinja -DROCM_PATH=/custom/rocm -DCMAKE_INSTALL_PREFIX=/another/path ..
```
## Build Targets

> [!NOTE]
> 📝 Make is supported for all targets. Configure with `cmake -G "Unix Makefiles" ..` if it is not the default generator in your environment. For parallel builds, use `make -j$(nproc)` on Linux. Unlike `ninja`, `make` does not build in parallel by default.

All targets support parallel builds with ninja:

| Target | Description |
|--------|-------------|
| `ninja` | Build all components |
| `ninja check` | Build and run all tests (see [Testing](./Testing.md)) |
| `ninja check-ctest` | Build and run all tests with CTest |
| `ninja unit-check` | Build and run exclusively the unit tests and API tests (minimal version of `ninja check`) |
| `ninja integration-check` | Build and run exclusively the E2E integration tests (this is the bulk of the testing time) |
| `ninja install` | Install libraries and headers |
| `ninja format` | Auto-format all C++ source files |
| `ninja check_format` | Check code formatting compliance |
| `ninja code_coverage` | Generate test coverage reports (requires `-DCODE_COVERAGE=ON`) |
| `ninja clean` | Clean build artifacts |
| `ninja validate_test_names` | Validates test names conform to naming rules |
| `ninja generate_hipdnn_sdk_headers` | Generate C++ headers from schema (`.fbs`) files |

## Platform-Specific Instructions

### Linux
The standard build instructions above work for all supported Linux distributions. Ensure ROCm is properly installed and configured for your distribution.

### Windows

> [!WARNING]
> GPU functionality and HIP-related tests are not currently supported on Windows. Only CPU tests can be run.

1. **Prerequisites**
   - Visual Studio 2022 with C++ workload
   - [TheRock](https://github.com/ROCm/TheRock) (ROCm Windows port)
   - CMake 3.25.2+
   - Ninja (recommended)

2. **Setup Environment**
   ```cmd
   # Open "x64 Native Tools Command Prompt for VS 2022"
   
   # Set HIP platform
   set HIP_PLATFORM=amd
   
   # Clone and build TheRock (see TheRock documentation)
   ```

3. **Build hipDNN**
   ```cmd
   cd <path\to\hipDNN>
   mkdir build
   cd build
   
   # Configure without plugins (not supported on Windows)
   cmake -GNinja -DHIP_DNN_BUILD_PLUGINS=OFF ..
   
   # Build and test (CPU tests only)
   ninja check
   ```

4. **Path Configuration**
   Add the following to your PATH:
   ```cmd
   set PATH=<hipDNN_build_dir>\backend\src;<TheRock_dist>\rocm\bin;%PATH%
   ```
   
   If using custom paths, you may need to modify [ClangToolChain.cmake](../cmake/ClangToolChain.cmake).

## Troubleshooting

### Common Build Issues

1. **Out of memory during build**
   ```bash
   # Reduce parallel jobs
   ninja -j4  # or even -j2 for systems with limited RAM
   ```

2. **Docker GPU access issues**
   - Ensure ROCm is installed on the host system
   - Verify GPU is visible: `rocm-smi` or `rocminfo`
   - Check user is in `video` and `render` groups:
     ```bash
     sudo usermod -a -G video,render $USER
     # Log out and back in for changes to take effect
     ```

## Verifying Installation

After installation, verify hipDNN is correctly installed:

1. **Check installed files**
   ```bash
   # Default installation
   ls /opt/rocm/include/hipdnn*
   ls /opt/rocm/lib/libhipdnn*
   ```

2. **Build and run samples**
   ```bash
   cd samples
   # See [samples README](../samples/README.md) for detailed instructions
   ```

3. **Test with a simple program**
   ```cpp
   #include <hipdnn.h>
   #include <iostream>
   
   int main() {
       size_t version;
       hipdnnGetVersion(&version);
       std::cout << "hipDNN version: " << version << std::endl;
       return 0;
   }
   ```
   
   Compile with:
   ```bash
   hipcc TestHipdnn.cpp -lhipdnn -o TestHipdnn
   ./TestHipdnn
   ```
