# rocPRIM

The rocPRIM is a header-only library providing HIP parallel primitives for developing
performant GPU-accelerated code on AMD ROCm platform.

## Requirements

* Git
* CMake (3.5.1 or later)
* AMD [ROCm](https://rocm.github.io/install.html) platform (1.8.2 or later)
  * Including [HCC](https://github.com/RadeonOpenCompute/hcc) compiler, which must be
    set as C++ compiler on ROCm platform.

Optional:

* [GTest](https://github.com/google/googletest)
  * Required only for tests. Building tests is enabled by default.
  * It will be automatically downloaded and built by cmake script.
* [Google Benchmark](https://github.com/google/benchmark)
  * Required only for benchmarks. Building benchmarks is off by default.
  * It will be automatically downloaded and built by cmake script.

## Build and Install

```shell
git clone https://github.com/ROCmSoftwarePlatform/rocPRIM.git

# Go to rocPRIM directory, create and go to the build directory.
cd rocPRIM; mkdir build; cd build

# Configure rocPRIM, setup options for your system.
# Build options:
#   BUILD_TEST - on by default,
#   BUILD_BENCHMARK - off by default.
#
# ! IMPORTANT !
# On ROCm platform set C++ compiler to HCC. You can do it by adding 'CXX=<path-to-hcc>'
# before 'cmake' or setting cmake option 'CMAKE_CXX_COMPILER' to path to the HCC compiler.
#
[CXX=hcc] cmake -DBUILD_BENCHMARK=ON ../. # or cmake-gui ../.

# Build
make -j4

# Optionally, run tests if they're enabled.
ctest --output-on-failure

# Install
[sudo] make install
```

### Using rocPRIM

Include `<rocprim/rocprim.hpp>` header:

```cpp
#include <rocprim/rocprim.hpp>
```

Recommended way of including rocPRIM into a CMake project is by using its package
configuration files. rocPRIM package name is `rocprim`.

```cmake
# "/opt/rocm" - default install prefix
find_package(rocprim REQUIRED CONFIG PATHS "/opt/rocm/rocprim")

...

# Includes only rocPRIM headers, HIP libraries have
# to be linked manually by user
target_link_libraries(<your_target> roc::rocprim)

# Includes rocPRIM headers and required HIP dependencies
target_link_libraries(<your_target> roc::rocprim_hip)
```

## Running Unit Tests

```shell
# Go to rocPRIM build directory
cd rocPRIM; cd build

# To run all tests
ctest

# To run unit tests for rocPRIM
./test/rocprim/<unit-test-name>
```

## Running Benchmarks

```shell
# Go to rocPRIM build directory
cd rocPRIM; cd build

# To run benchmark for warp functions:
# Further option can be found using --help
# [] Fields are optional
./benchmark/benchmark_warp_<function_name> [--size <size>] [--trials <trials>]

# To run benchmark for block functions:
# Further option can be found using --help
# [] Fields are optional
./benchmark/benchmark_block_<function_name> [--size <size>] [--trials <trials>]

# To run benchmark for device functions:
# Further option can be found using --help
# [] Fields are optional
./benchmark/benchmark_device_<function_name> [--size <size>] [--trials <trials>]
```

### Performance configuration

Most of device-wide primitives provided by rocPRIM can be tuned for different AMD device,
different types or different operations using compile-time configuration structures passed
to them as a template parameter. Main "knobs" are usually size of the block and number of
items processed by a single thread.

rocPRIM has built-in default configurations for each of its primitives. In order to use
included configurations user should define macro `ROCPRIM_TARGET_ARCH` to `803` if algorithms
should be optimized for gfx803 GCN version, or to `900` for gfx900.

## Documentation

```shell
# go to rocPRIM doc directory
cd rocPRIM; cd doc

# run doxygen
doxygen Doxyfile

# open html/index.html
```

## hipCUB

[hipCUB](https://github.com/ROCmSoftwarePlatform/hipCUB/) is a thin wrapper library on top of
[rocPRIM](https://github.com/ROCmSoftwarePlatform/rocPRIM) or [CUB](https://github.com/NVlabs/cub).
It enables developers to port project that uses CUB library to the
[HIP](https://github.com/ROCm-Developer-Tools/HIP) layer and to run them on AMD hardware. In [ROCm](https://rocm.github.io/)
environment hipCUB uses rocPRIM library as the backend, however, on CUDA platforms it uses CUB instead.

## Support

Bugs and feature requests can be reported through [the issue tracker](https://github.com/ROCmSoftwarePlatform/rocPRIM/issues).

## Contributions and License

Contributions of any kind are most welcome! More details are found at [CONTRIBUTING](./CONTRIBUTING.md)
and [LICENSE](./LICENSE.txt).
