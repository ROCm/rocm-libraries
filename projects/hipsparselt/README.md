# hipSPARSELt

hipSPARSELt is a SPARSE marshalling library, with multiple supported backends.
It sits between the application and a 'worker' SPARSE library, marshalling
inputs into the backend library and marshalling results back to the
application. hipSPARSELt exports an interface that does not require the client
to change, regardless of the chosen backend. Currently, hipSPARSELt supports
[rocSPARSELt](library/src/hcc_detail/rocsparselt) and [NVIDIA CUDA cuSPARSELt v0.6.3](https://docs.nvidia.com/cuda/cusparselt)
as backends.

> [!NOTE]
> The published hipSPARSELt documentation is available at [hipSPARSELt](https://rocm.docs.amd.com/projects/hipSPARSELt/en/latest/index.html) in an organized, easy-to-read format, with search and a table of contents. The documentation source files reside in the hipsparselt/docs folder of this repository. As with all ROCm projects, the documentation is open source. For more information, see [Contribute to ROCm documentation](https://rocm.docs.amd.com/en/latest/contribute/contributing.html).

## Installing pre-built packages

Download pre-built packages either from the
[ROCm package servers](https://rocm.docs.amd.com/projects/hipSPARSELt/en/latest/tutorials/install/linux.html#building-hipsparselt-from-source)
or by clicking the GitHub releases tab and manually downloading. Release notes are available for each release on the releases tab.

* `sudo apt update && sudo apt install hipsparselt`

## Getting started

> [!NOTE]
> The steps in this section are intended to help users get started building hipsparselt. However, it is recommended to consult the
> [hipSPARSELt installation documentation](https://rocm.docs.amd.com/projects/hipSPARSELt/en/latest/tutorials/install/linux.html)
> for complete setup and installation instructions.

### Setup

The simplest option is to clone all of [rocm-libraries](https://github.com/ROCm/rocm-libraries) and navigate to the hipsparselt project:

```bash
# Clone rocm-libraries
git clone https://github.com/ROCm/rocm-libraries.git
# Go to hipSPARSELt directory
cd rocm-libraries/projects/hipsparselt
```

For a shorter download process, use sparse checkout to only clone the hipsparselt project:

```bash
git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-libraries.git
cd rocm-libraries
git sparse-checkout init --cone
git sparse-checkout set projects/hipsparselt
git checkout develop # or the branch you are starting from
```

### Configure and build

hipSPARSELt provides modern CMake support and relies on native CMake functionality, with the exception of
some project specific options. As such, users are advised to consult the CMake documentation for
general usage questions. For details on all configuration options see the [Options](#options) section.

This section provides usage examples on how to configure, build and install hipSPARSELt using various supported methods.
We assume the user has a ROCm installation (conventionally installed to `/opt/rocm`), Python 3.8 or newer, 
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
cmake --preset default-release
# build
cmake --build _build --parallel
# install
cmake --install _build
```

**Debug build for development**

```bash
cmake --preset debug
cmake --build _build
```

**Build with coverage**

```bash
cmake --preset coverage
cmake --build _build --target coverage
```

### Using CMake variables directly

**Full build for gfx1201**

```bash
# configure
cmake -B build -S .                                  \
      -D CMAKE_BUILD_TYPE=Release                    \
      -D CMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ \
      -D CMAKE_C_COMPILER=/opt/rocm/bin/amdclang     \
      -D CMAKE_PREFIX_PATH=/opt/rocm                 \
      -D GPU_TARGETS=gfx1201
# build
cmake --build build --parallel
```

> [!TIP]
> **For Developers**
>
> View debugging info by adding `--log-level=VERBOSE` to the configure command.

### Using the installation script

Refer to the available build options using `./install.sh --help`

```bash
# Command line options:
#   -h|--help            - prints help message
#   -i|--install         - install after build
#   -d|--dependencies    - install build dependencies
#   -c|--clients         - build library clients too (combines with -i & -d)
#   -g|--debug           - build with debug flag
#   -k|--relwithdebinfo  - build with RelWithDebInfo

# build and install all libraries and clients, and fetch dependencies
./install.sh -idc
# build for gfx1201 only without installation
./install.sh -c -a gfx1201
```

### Options

> [!NOTE]
> When using the install script these variables are either hardcoded or set via its command line options.

*CMake options*:

* `CMAKE_BUILD_TYPE`: Any of Release, Debug, RelWithDebInfo, MinSizeRel
* `CMAKE_INSTALL_PREFIX`: Base installation directory (defaults to `/opt/rocm` on Linux, `C:/hipSDK` on Windows)
* `CMAKE_PREFIX_PATH`: Find package search path (consider setting to `$ROCM_PATH`)
* `CMAKE_EXPORT_COMPILE_COMMANDS`: Export compile_commands.json for clang tooling support (default: `ON`)

*Build control options*:

* `GPU_TARGETS`: AMD GFX targets to cross-compile for (default: `all`)
* `HIPSPARSELT_BUILD_SHARED_LIBS`: Build the hipSPARSELt shared or static library (default: `ON`)
* `HIPSPARSELT_BUILD_TESTING`: Build test client (default: `ON`)
* `HIPSPARSELT_BUILD_COVERAGE`: Build tests with coverage support (default: `OFF`)

*Backend options*:

* `HIPSPARSELT_ENABLE_HIP`: Build hipSPARSELt with HIP backend (default: `ON`)
* `HIPSPARSELT_ENABLE_CUDA`: Build hipSPARSELt with CUDA backend (default: `OFF`)

*Client options*:

* `HIPSPARSELT_ENABLE_CLIENT`: Build hipSPARSELt clients (default: `ON`)
* `HIPSPARSELT_ENABLE_BENCHMARKS`: Build benchmark client (default: `ON`)
* `HIPSPARSELT_ENABLE_SAMPLES`: Build client samples (default: `ON`)
* `HIPSPARSELT_ENABLE_FORTRAN`: Build Fortran clients (default: `OFF`)
* `HIPSPARSELT_ENABLE_BLIS`: Enable BLIS support for reference implementations (default: `ON`)

*Advanced options*:

* `HIPSPARSELT_ENABLE_MARKER`: Enable rocTracer marker support (default: `OFF`)
* `HIPSPARSELT_ENABLE_ASAN`: Build with address sanitizer enabled (default: `OFF`)
* `HIPSPARSELT_HIPBLASLT_PATH`: Path to hipblaslt directory (default: `${CMAKE_CURRENT_SOURCE_DIR}/../../hipblaslt/next-cmake`)
* `HIPSPARSELT_COVERAGE_GTEST_FILTER`: GTest filter for coverage tests (default: empty)
* `TENSILE_ENABLE_MARKER`: Enables or disables the Tensile marker functionality (default: `OFF`)

## CMake Targets

*Libraries*:

* `roc::hipsparselt` - Main library target

*Executables*:

* `hipsparselt-test` - Test executable (when HIPSPARSELT_BUILD_TESTING=ON)
* `hipsparselt-bench` - Benchmark executable (when HIPSPARSELT_ENABLE_BENCHMARKS=ON)
* `example_spmm_strided_batched` - Sample executable (when HIPSPARSELT_ENABLE_SAMPLES=ON)
* `example_prune_strip` - Sample executable (when HIPSPARSELT_ENABLE_SAMPLES=ON)
* `example_compress` - Sample executable (when HIPSPARSELT_ENABLE_SAMPLES=ON)
* `coverage` - Code coverage target (when HIPSPARSELT_BUILD_COVERAGE=ON)

## Functions supported

* ROCm
  * AMD sparse MFMA matrix core support
    * Mixed-precision computation support:
      * FP16 input/output, FP32 Matrix Core accumulate
      * BFLOAT16 input/output, FP32 Matrix Core accumulate
      * INT8 input/output, INT32 Matrix Core accumulate
      * INT8 input, FP16 output, INT32 Matrix Core accumulate
      * FP8(E4M3) input, FP32 output, FP32 Matrix Core accumulate (LLVM target: gfx950)
      * BF8(E5M2) input, FP32 output, FP32 Matrix Core accumulate (LLVM target: gfx950)
    * Matrix pruning and compression functionalities
    * Auto-tuning functionality (see hipsparseLtMatmulSearch())
    * Batched Sparse Gemm support:
      * Single sparse matrix / Multiple dense matrices (Broadcast)
      * Multiple sparse and dense matrices
      * Batched bias vector
    * Activation function fuse in spmm kernel support:
      * ReLU
      * ClippedReLU (ReLU with upper bound and threshold setting)
      * GeLU
      * GeLU Scaling (Implied enable GeLU)
      * Abs
      * LeakyReLU
      * Sigmoid
      * Tanh
    * On-going feature development
      * Add support for Mixed-precision computation
        * FP8 input/output, FP32 Matrix Core accumulate
        * BF8 input/output, FP32 Matrix Core accumulate
      * Add kernel selection and generator, used to provide the appropriate
        solution for the specific problem.
* CUDA
  * Support cusparseLt v0.6.3

## Documentation

Full documentation for hipSPARSELt is available at
[rocm.docs.amd.com/projects/hipSPARSELt](https://rocm.docs.amd.com/projects/hipSPARSELt/en/latest/index.html).

Run the steps below to build documentation locally.

```bash
cd docs

pip3 install -r sphinx/requirements.txt

python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html
```

Alternatively, build with CMake:

```bash
cmake -DBUILD_DOCS=ON ...
```

## hipSPARSELt interface examples

The hipSPARSELt interface is compatible with cuSPARSELt APIs. Porting a CUDA
application which originally calls the cuSPARSELt API to an application calling
hipSPARSELt API should be relatively straightforward. For example, the
hipSPARSELt matmul interface is

### matmul API

```c
hipsparseStatus_t hipsparseLtMatmul(const hipsparseLtHandle_t*     handle,
                                    const hipsparseLtMatmulPlan_t* plan,
                                    const void*                    alpha,
                                    const void*                    d_A,
                                    const void*                    d_B,
                                    const void*                    beta,
                                    const void*                    d_C,
                                    void*                          d_D,
                                    void*                          workspace,
                                    hipStream_t*                   streams,
                                    int32_t                        numStreams);

```

hipSPARSELt assumes matrix A, B, C, D and workspace are allocated in GPU memory
space filled with data. Users are responsible for copying data from/to the host
and device memory.

## Unit tests

All unit tests are located in the build directory. To build these tests, you must build
hipSPARSELt with `--clients` or `-c` flag.

To run unit tests:

```bash
# Go to hipSPARSELt build directory
cd build/release

# Run all tests
./clients/staging/hipsparselt-test
```

To run benchmarks:

```bash
# Go to hipSPARSELt build directory
cd build/release

# Run benchmark, e.g.
./clients/staging/hipsparselt-bench -f spmm -i 200 -m 256 -n 256 -k 256
```
