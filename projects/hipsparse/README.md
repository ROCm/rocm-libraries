# hipSPARSE

hipSPARSE is a SPARSE marshalling library with multiple supported backends. It sits between your
application and a 'worker' SPARSE library, where it marshals inputs to the backend library and marshals
results to your application. hipSPARSE exports an interface that doesn't require the client to change,
regardless of the chosen backend. Currently, hipSPARSE supports
[rocSPARSE](https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocsparse) and
[NVIDIA CUDA cuSPARSE](https://developer.nvidia.com/cusparse) backends.

## Documentation

> [!NOTE]
> The published hipSPARSE documentation is available at [https://rocm.docs.amd.com/projects/hipSPARSE/en/latest/](https://rocm.docs.amd.com/projects/hipSPARSE/en/latest/index.html) in an organized, easy-to-read format, with search and a table of contents. The documentation source files reside in the `rocm-libraries/projects/hipsparse` folder of this repository. As with all ROCm projects, the documentation is open source. For more information, see [Contribute to ROCm documentation](https://rocm.docs.amd.com/en/latest/contribute/contributing.html).

To build our documentation locally, run the following code:

```bash
cd docs

pip3 install -r sphinx/requirements.txt

python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html
```

Alternatively, build with CMake:

```bash
cmake -DHIPSPARSE_BUILD_DOCS=ON ...
```

## Installing pre-built packages

Download pre-built packages from
[ROCm's package servers](https://rocm.docs.amd.com/en/latest/deploy/linux/index.html) using the
following code:

```bash
`sudo apt update && sudo apt install hipsparse`
```

## Build hipSPARSE

hipSPARSE provides modern CMake support and relies on native CMake functionality, with the exception of
some project specific options. As such, users are advised to consult the CMake documentation for
general usage questions. For details on all configuration options see the [Options](#options) section.

We assume the user has a ROCm installation (conventionally installed to `/opt/rocm`), Python 3.9 or newer, 
and a CMake version greater than or equal to the `cmake_minimum_required` defined in [CMakeLists.txt](./CMakeLists.txt#L4).

### Using CMake presets

> [!NOTE]
> When using presets, assumptions are made about search paths, built-in CMake variables, and output directories. 
> Consult [CMakePresets.json](./CMakePresets.json) to understand which variables are set, 
> or refer to [Using CMake variables directly](#using-cmake-variables-directly) for a fully custom configuration.

**Release build**

```bash
# show available presets
cmake --list-presets
# configure
cmake --preset default:release
# build
cmake --build build --parallel
```

**Debug build for development**

```bash
cmake --preset debug
cmake --build build/debug
```

**Build with coverage**

```bash
cmake --preset coverage
cmake --build build/coverage --target coverage
```

### Using CMake variables directly

**Full build**

```bash
# configure
cmake -B build -S .                                  \
      -D CMAKE_BUILD_TYPE=Release                    \
      -D CMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ \
      -D CMAKE_C_COMPILER=/opt/rocm/bin/amdclang     \
      -D CMAKE_PREFIX_PATH=/opt/rocm
# build
cmake --build build --parallel
```

### Using the installation script

The `install.sh` script builds and installs hipSPARSE on Ubuntu with a single command.

```bash
# Command line options:
#   -h|--help            - prints help message
#   -i|--install         - install after build
#   -d|--dependencies    - install build dependencies
#   -c|--clients         - build library clients too (combines with -i & -d)
#   -g|--debug           - build with debug flag
#   -k|--relwithdebinfo  - build with RelWithDebInfo

# build and install library with clients, and fetch dependencies
./install.sh -idc
```

## Options

> [!NOTE]
> When using the install script these variables are either hardcoded or set via its command line options.

*CMake options*:

* `CMAKE_BUILD_TYPE`: Any of Release, Debug, RelWithDebInfo, MinSizeRel
* `CMAKE_INSTALL_PREFIX`: Base installation directory (defaults to `/opt/rocm` on Linux, `C:/hipSDK` on Windows)
* `CMAKE_PREFIX_PATH`: Find package search path (consider setting to `$ROCM_PATH`)
* `CMAKE_EXPORT_COMPILE_COMMANDS`: Export compile_commands.json for clang tooling support (default: `ON`)

*Build control options*:

* `HIPSPARSE_BUILD_SHARED_LIBS`: Build hipSPARSE as a shared library (default: `ON`)
* `HIPSPARSE_BUILD_TESTING`: Build test client (default: `OFF`)
* `HIPSPARSE_ENABLE_COVERAGE`: Build with code coverage enabled (default: `OFF`)
* `HIPSPARSE_BUILD_DOCS`: Build documentation (default: `OFF`)

*Backend options*:

* `HIPSPARSE_ENABLE_HIP`: Build hipSPARSE with HIP backend (default: `ON`)
* `HIPSPARSE_ENABLE_CUDA`: Build hipSPARSE with CUDA backend (default: `OFF`)

*Client options*:

* `HIPSPARSE_ENABLE_CLIENT`: Build hipSPARSE clients (default: `ON`)
* `HIPSPARSE_ENABLE_BENCHMARKS`: Build benchmark client (default: `ON`)
* `HIPSPARSE_ENABLE_SAMPLES`: Build sample programs (default: `ON`)
* `HIPSPARSE_ENABLE_FORTRAN`: Enable Fortran client support (default: `ON` on Linux, `OFF` on Windows)
* `HIPSPARSE_ENABLE_OPENMP`: Enable OpenMP support in clients (default: `ON`)

*Advanced options*:

* `HIPSPARSE_ENABLE_ASAN`: Build with address sanitizer enabled (default: `OFF`)
* `HIPSPARSE_ENABLE_HOST`: Build hipSPARSE library (default: `ON`)

## CMake Targets

*Libraries*:

* `roc::hipsparse` - Main library target

*Executables*:

* `hipsparse-test` - Test executable (when `HIPSPARSE_BUILD_TESTING=ON`)
* `hipsparse-bench` - Benchmark executable (when `HIPSPARSE_ENABLE_BENCHMARKS=ON`)
* `example_*` - Sample executables (when `HIPSPARSE_ENABLE_SAMPLES=ON`)
* `coverage` - Code coverage target (when `HIPSPARSE_ENABLE_COVERAGE=ON`)

## Supported functions

See the hipSPARSE documentation for a list of
[exported functions](https://rocm.docs.amd.com/projects/hipSPARSE/en/latest/reference/api.html).

## Interface examples

The hipSPARSE interface is compatible with rocSPARSE and CUDA cuSPARSE-v2 APIs. Porting a CUDA
application that calls the CUDA cuSPARSE API to an application that calls the hipSPARSE API is relatively
straightforward. For example, the hipSPARSE SCSRMV interface is:

### CSRMV API

```c
hipsparseStatus_t
hipsparseScsrmv(hipsparseHandle_t handle,
                hipsparseOperation_t transA,
                int m, int n, int nnz, const float *alpha,
                const hipsparseMatDescr_t descrA,
                const float *csrValA,
                const int *csrRowPtrA, const int *csrColIndA,
                const float *x, const float *beta,
                float *y);
```

hipSPARSE assumes matrix A and vectors x, y are allocated in GPU memory space filled with data. Users
are responsible for copying data to and from the host and device memory.
