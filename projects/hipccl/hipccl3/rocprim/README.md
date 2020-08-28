# rocPRIM

The rocPRIM is a header-only library providing HIP parallel primitives for developing
performant GPU-accelerated code on AMD ROCm platform.

## Requirements

* Git
* CMake (3.5.1 or later)
* AMD [ROCm](https://rocm.github.io/install.html) platform (1.8.2 or later)
  * Including [HIP-clang](https://github.com/ROCm-Developer-Tools/HIP/blob/master/INSTALL.md#hip-clang) compiler

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
#   DISABLE_WERROR - ON by default, This flag disable the -Werror compiler flag
#   ONLY_INSTALL - OFF by default, If this flag is on, the build ignore the BUILD_* flags
#   BUILD_TEST - OFF by default,
#   BUILD_EXAMPLE - OFF by default,
#   BUILD_BENCHMARK - OFF by default.
#   BENCHMARK_CONFIG_TUNING - OFF by default. The purpose of this flag to find the best kernel config parameters.
#     At ON the compilation time can be increased significantly.
#   AMDGPU_TARGETS - list of AMD architectures, default: gfx803;gfx900;gfx906;gfx908.
#     You can make compilation faster if you want to test/benchmark only on one architecture,
#     for example, add -DAMDGPU_TARGETS=gfx906 to 'cmake' parameters.
#   AMDGPU_TEST_TARGETS - list of AMD architectures, default: "" (default system device)
#     If you want to detect failures on a per GFX IP basis, setting it to some set of ips will create
#     separate tests with the ip name embedded into the test name. Building for all, but selecting
#     tests only of a specific architecture is possible for eg: ctest -R gfx803|gfx900
#
# ! IMPORTANT !
# Set C++ compiler to HIP-clang. You can do it by adding 'CXX=<path-to-compiler>'
# before 'cmake' or setting cmake option 'CMAKE_CXX_COMPILER' to path to the compiler.
# Using HIP-clang:
[CXX=hipcc] cmake -DBUILD_BENCHMARK=ON ../.

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

Unit tests are implemented in terms of Google Test and collections of tests are wrapped to be invoked from CTest for convenience.

```shell
# Go to rocPRIM build directory
cd rocPRIM; cd build

# List available tests
ctest --show-only

# To run all tests
ctest

# Run specific test(s)
ctest -R <regex>

# To run the Google Test manually
./test/rocprim/test_<unit-test-name>
```

### Using multiple GPUs concurrently for testing

This feature requires CMake 3.16+ to be used for building / testing. _(Prior versions of CMake cannot assign ids to tests when running in parallel. Assigning tests to distinct devices could only be done at the cost of extreme complexity._)

The unit tests can make use of [CTest Resource Allocation](https://cmake.org/cmake/help/latest/manual/ctest.1.html#resource-allocation) feature enabling distributing tests across multiple GPUs in an intelligent manner. The feature can accelerate testing when multiple GPUs of the same family are in a system as well as test multiple family of products from one invocation without having to resort to `HIP_VISIBLE_DEVICES` environment variable. The feature relies on the presence of a resource spec file.

> IMPORTANT: trying to use `RESOURCE_GROUPS` and `--resource-spec-file` with CMake/CTest respectively of versions prior to 3.16 omits the feature silently. No warnings issued about unknown properties or command-line arguments. Make sure that `cmake`/`ctest` invoked are sufficiently recent.

#### Auto resource spec generation

There is a utility script in the repo that may be called independently:

```shell
# Go to rocPRIM build directory
cd rocPRIM; cd build

# Invoke directly or use CMake script mode via cmake -P
../cmake/GenerateResourceSpec.cmake

# Assuming you have 2 compatible GPUs in the system
ctest --resource-spec-file ./resources.json --parallel 2
```

#### Manual

Assuming the user has 2 GPUs from the gfx900 family and they are the first devices enumerated by the system one may specify during configuration `-D AMDGPU_TEST_TARGETS=gfx900` stating only one family will be tested. Leaving this var empty (default) results in targeting the default device in the system. To let CMake know there are 2 GPUs that should be targeted, one has to feed CTest a JSON file via the `--resource-spec-file <path_to_file>` flag. For example:

```json
{
  "version": {
    "major": 1,
    "minor": 0
  },
  "local": [
    {
      "gfx900": [
        {
          "id": "0"
        },
        {
          "id": "1"
        }
      ]
    }
  ]
}
```

Invoking CTest as `ctest --resource-spec-file <path_to_file> --parallel 2` will allow two tests to run concurrently which will be distributed among the two GPUs.

### Using custom seeds for the tests

Go to the `rocPRIM/test/rocprim/test_seed.hpp` file.
```cpp
//(1)
static constexpr int random_seeds_count = 10;

//(2)
static constexpr unsigned int seeds [] = {0, 2, 10, 1000};

//(3)
static constexpr size_t seed_size = sizeof(seeds) / sizeof(seeds[0]);
```

(1) defines a constant that sets how many passes over the tests will be done with runtime-generated seeds. Modify at will.

(2) defines the user generated seeds. Each of the elements of the array will be used as seed for all tests. Modify at will. If no static seeds are desired, the array should be left empty.

```cpp
static constexpr unsigned int seeds [] = {};
```

(3) this line should never be modified.

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
The latest rocPRIM documentation and API description can be found [here](https://codedocs.xyz/ROCmSoftwarePlatform/rocPRIM/).

It can also be build using the following commands

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
