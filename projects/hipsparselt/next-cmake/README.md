# hipSPARSELt Next-Generation Build System

This directory contains the next-generation build system for hipSPARSELt that addresses the major issues with the legacy build system.

## Key Improvements

### Fixed Issues
- ✅ **Removed Tensile-tag consumption mechanism** - Now uses `add_subdirectory` for `hipblaslt/tensilelite`
- ✅ **Fixed Python invocations** - Uses `Python3_EXECUTABLE` properly
- ✅ **Updated TensileCreateLibrary** - Uses modern `TensileCreateLibrary` instead of legacy `TensileCreateLibraryFiles`
- ✅ **Target-level compiler features** - No global `CMAKE_CXX_STANDARD` settings
- ✅ **Target-level flags** - No global `CMAKE_CXX_FLAGS` modifications
- ✅ **Fixed shared/static handling** - Uses `HIPSPARSELT_BUILD_SHARED_LIBS` without modifying `BUILD_SHARED_LIBS`
- ✅ **Removed hardcoded install prefix** - Uses standard CMake install prefix handling
- ✅ **Eliminated legacy commands** - No `include_directories()` or `add_definitions()`
- ✅ **Fixed cuSPARSELt detection** - Proper library finding and consumption

### Modern CMake Features
- Target-level property management
- Proper generator expression usage
- Modern dependency management
- Component-based installation
- Proper imported target usage

## Build Options

### Core Options
- `HIPSPARSELT_BUILD_SHARED_LIBS` - Build shared library (default: ON)
- `HIPSPARSELT_ENABLE_CUDA` - Build with CUDA backend (default: OFF)
- `HIPSPARSELT_ENABLE_TENSILELITE` - Build with TensileLite backend (default: ON)

### Client Options
- `HIPSPARSELT_ENABLE_CLIENT` - Build client applications (default: ON)
- `HIPSPARSELT_ENABLE_SAMPLES` - Build sample programs (default: ON)
- `HIPSPARSELT_ENABLE_BENCHMARKS` - Build benchmark client (default: ON)
- `HIPSPARSELT_ENABLE_TESTS` - Build test suite (default: ON)
- `HIPSPARSELT_ENABLE_FORTRAN` - Build Fortran clients (default: OFF)
- `HIPSPARSELT_ENABLE_BLIS` - Enable BLIS support for reference implementations (default: ON)

### Development Options
- `HIPSPARSELT_BUILD_COVERAGE` - Build with code coverage (default: OFF)
- `HIPSPARSELT_ENABLE_MARKER` - Enable rocTracer markers (default: OFF)
- `HIPSPARSELT_ENABLE_ASAN` - Build with address sanitizer (default: OFF)
- `HIPSPARSELT_ENABLE_VERBOSE` - Enable verbose build output (default: OFF)
- `HIPSPARSELT_ENABLE_ROCM_SMI` - Require rocm_smi (default: ON, except Windows)

## Usage Examples

### Basic ROCm Build
```bash
mkdir build && cd build
cmake ../next-cmake
make -j$(nproc)
```

### CUDA Build
```bash
mkdir build && cd build
cmake ../next-cmake -DHIPSPARSELT_ENABLE_CUDA=ON
make -j$(nproc)
```

### Static Library Build
```bash
mkdir build && cd build
cmake ../next-cmake -DHIPSPARSELT_BUILD_SHARED_LIBS=OFF
make -j$(nproc)
```

### Client-Only Build
```bash
mkdir build && cd build
cmake ../next-cmake -DHIPSPARSELT_ENABLE_CLIENT=ON -DHIPSPARSELT_ENABLE_SAMPLES=OFF
make -j$(nproc)
```

### Debug Build with Coverage
```bash
mkdir build && cd build
cmake ../next-cmake -DCMAKE_BUILD_TYPE=Debug -DHIPSPARSELT_BUILD_COVERAGE=ON
make -j$(nproc)
```

### Custom GPU Targets
```bash
mkdir build && cd build
cmake ../next-cmake -DAMDGPU_TARGETS="gfx942;gfx950;gfx1100"
make -j$(nproc)
```

## Directory Structure

```
next-cmake/
├── CMakeLists.txt          # Main build configuration
├── cmake/                  # CMake helper modules
│   ├── FetchROCmCMake.cmake
│   └── FindBLIS.cmake
├── library/                # Library build configuration
│   ├── CMakeLists.txt
│   └── src/
│       └── CMakeLists.txt
├── clients/                # Client applications
│   └── CMakeLists.txt
└── README.md              # This file
```

## Dependencies

### Required
- CMake 3.25.2 or newer
- Python 3 (for build scripts)
- HIP runtime
- hipSPARSE library

### Optional
- CUDA Toolkit (for CUDA backend)
- cuSPARSELt (for CUDA backend)
- TensileLite (for ROCm backend)
- GTest (for testing)
- OpenMP (for reference implementations)
- BLIS (for optimized reference implementations)
- rocm_smi (for ROCm system management)
- rocTracer (for marker support)

## TensileLite Integration

The build system expects TensileLite to be available at:
```
<project_root>/deps/hipblaslt/tensilelite
```

This should be set up as a Git submodule or by extracting hipBLASLt to the deps directory.

## Installation

The build system creates proper installation packages with multiple components:

- `runtime` - Core library
- `samples` - Sample programs  
- `benchmarks` - Benchmark client
- `tests` - Test suite
- `clients-common` - Common client files

Install all components:
```bash
make install
```

Install specific components:
```bash
make install/runtime
make install/samples
```

## Package Creation

The build system creates proper packages for distribution:

```bash
make package
```

This creates packages named:
- `hipsparselt` (for ROCm backend)
- `hipsparselt-cuda` (for CUDA backend)

## Troubleshooting

### TensileLite Not Found
Ensure the TensileLite directory exists:
```bash
# From project root
ls deps/hipblaslt/tensilelite
```

### Python Executable Issues
The build system uses `Python3_EXECUTABLE`. Ensure Python 3 is available:
```bash
python3 --version
```

### CUDA Backend Issues
For CUDA builds, ensure:
1. CUDA Toolkit is installed
2. cuSPARSELt library is available
3. Environment variables are set correctly

### Build Failures
Enable verbose output for debugging:
```bash
cmake ../next-cmake -DHIPSPARSELT_ENABLE_VERBOSE=ON
```

## Migration from Legacy Build

The new build system uses different option names:

| Legacy Option | New Option |
|---------------|------------|
| `BUILD_SHARED_LIBS` | `HIPSPARSELT_BUILD_SHARED_LIBS` |
| `BUILD_CLIENTS_TESTS` | `HIPSPARSELT_ENABLE_TESTS` |
| `BUILD_CLIENTS_BENCHMARKS` | `HIPSPARSELT_ENABLE_BENCHMARKS` |
| `BUILD_CLIENTS_SAMPLES` | `HIPSPARSELT_ENABLE_SAMPLES` |
| `BUILD_CUDA` | `HIPSPARSELT_ENABLE_CUDA` |
| `BUILD_WITH_TENSILE` | `HIPSPARSELT_ENABLE_TENSILELITE` |
| `BUILD_CODE_COVERAGE` | `HIPSPARSELT_BUILD_COVERAGE` |
| `BUILD_ADDRESS_SANITIZER` | `HIPSPARSELT_ENABLE_ASAN` |
| `BUILD_VERBOSE` | `HIPSPARSELT_ENABLE_VERBOSE` |
| `HIPSPARSELT_ENABLE_MARKER` | `HIPSPARSELT_ENABLE_MARKER` |

## Contributing

When making changes to the build system:

1. Follow the target-level configuration pattern
2. Use modern CMake features (3.25+)
3. Avoid global variables and legacy commands
4. Test both ROCm and CUDA backends
5. Update this README for new options 