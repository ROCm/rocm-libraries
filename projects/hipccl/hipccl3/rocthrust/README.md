# rocThrust

## Introduction

Thrust is a parallel algorithm library. This library has been ported to [HIP](https://github.com/ROCm-Developer-Tools/HIP)/[ROCm](https://rocm.github.io/) platform, which uses the [rocPRIM](https://github.com/ROCmSoftwarePlatform/rocPRIM) library. The HIP ported library works on HIP/ROCm platforms. Currently there is no CUDA backend in place.

## Requirements

### Software

* CMake (3.5.1 or later)
* AMD [ROCm](https://rocm.github.io/install.html) platform (1.8.0 or later)
  * Including [HipCC](https://github.com/ROCm-Developer-Tools/HIP) compiler, which must be
    set as C++ compiler on ROCm platform.
* [rocPRIM](https://github.com/ROCmSoftwarePlatform/rocPRIM) library
  * It will be automatically downloaded and built by CMake script.
* Python 3.6 or higher (HIP on Windows only, only required for install scripts)
* Visual Studio 2019 with clang support (HIP on Windows only)
* Strawberry Perl (HIP on Windows only)

Optional:

* [GTest](https://github.com/google/googletest)
  * Required only for tests. Building tests is enabled by default.
  * It will be automatically downloaded and built by CMake script.

### Hardware
Visit the following link for ROCm hardware requirements:
https://github.com/RadeonOpenCompute/ROCm/blob/master/README.md#supported-cpus


## Build And Install

```sh
git clone https://github.com/ROCmSoftwarePlatform/rocThrust

# Go to rocThrust directory, create and go to the build directory.
cd rocThrust; mkdir build; cd build

# Configure rocThrust, setup options for your system.
# Build options:
#   DISABLE_WERROR   - ON  by default, This flag disable the -Werror compiler flag
#   BUILD_TEST       - OFF by default,
#   BUILD_EXAMPLES   - OFF by default,
#   BUILD_BENCHMARKS - OFF by default,
#   DOWNLOAD_ROCPRIM - OFF by default, when ON rocPRIM will be downloaded to the build folder,
#   RNG_SEED_COUNT   - 0 by default, controls non-repeatable random dataset count
#   PRNG_SEEDS       - 1 by default, reproducible seeds to generate random data
#
# ! IMPORTANT !
# On ROCm platform set C++ compiler to HipCC. You can do it by adding 'CXX=<path-to-hipcc>'
# before 'cmake' or setting cmake option 'CMAKE_CXX_COMPILER' with the path to the HipCC compiler.
#
[CXX=hipcc] cmake ../. # or cmake-gui ../.

# Build
make -j4

# Optionally, run tests if they're enabled.
ctest --output-on-failure

# Package
make package

# Install
[sudo] make install
```

### HIP on Windows

Initial support for HIP on Windows has been added.  To install, use the provided rmake.py python script:
```shell
git clone https://github.com/ROCmSoftwarePlatform/rocThrust.git
cd rocThrust

# the -i option will install rocPRIM to C:\hipSDK by default
python rmake.py -i

# the -c option will build all clients including unit tests
python rmake.py -c
```

### Macro options

```
# Performance improvement option. If you define THRUST_HIP_PRINTF_ENABLED before
# thrust includes to 0, you can disable printfs on device side and improve
# performance. The default value is 1
#define THRUST_HIP_PRINTF_ENABLED 0
```

### Using rocThrust In A Project

Recommended way of including rocThrust into a CMake project is by using its package
configuration files.

```cmake
# On ROCm rocThrust requires rocPRIM
find_package(rocprim REQUIRED CONFIG PATHS "/opt/rocm/rocprim")

# "/opt/rocm" - default install prefix
find_package(rocthrust REQUIRED CONFIG PATHS "/opt/rocm/rocthrust")

...
includes rocThrust headers and roc::rocprim_hip target
target_link_libraries(<your_target> roc::rocthrust)
```

## Running Unit Tests

```sh
# Go to rocThrust build directory
cd rocThrust; cd build

# Configure with examples flag on
CXX=hipcc cmake -DBUILD_TEST=ON ..

# Build tests
make -j4

# To run all tests
ctest

# To run unit tests for rocThrust
./test/<unit-test-name>
```

### Using multiple GPUs concurrently for testing

This feature requires CMake 3.16+ to be used for building / testing. _(Prior versions of CMake cannot assign ids to tests when running in parallel. Assigning tests to distinct devices could only be done at the cost of extreme complexity._)

The unit tests can make use of [CTest Resource Allocation](https://cmake.org/cmake/help/latest/manual/ctest.1.html#resource-allocation) feature enabling distributing tests across multiple GPUs in an intelligent manner. The feature can accelerate testing when multiple GPUs of the same family are in a system as well as test multiple family of products from one invocation without having to resort to `HIP_VISIBLE_DEVICES` environment variable. The feature relies on the presence of a resource spec file.

> IMPORTANT: trying to use `RESOURCE_GROUPS` and `--resource-spec-file` with CMake/CTest respectively of versions prior to 3.16 omits the feature silently. No warnings issued about unknown properties or command-line arguments. Make sure that `cmake`/`ctest` invoked are sufficiently recent.

#### Auto resource spec generation

There is a utility script in the repo that may be called independently:

```shell
# Go to rocThrust build directory
cd rocThrust; cd build

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

## Using custom seeds for the tests

There are 2 CMake configuration-time options that control random data fed to unit tests.

- `RNG_SEED_COUNT`, (0 by default) controls non-repeatable random dataset count. It draws values from a default constructed `std::random_device`. Should tests fail, the actual seed producing the failure are reported by Gtest, enabling reproducibility.
- `PRNG_SEEDS`, (1 by default) controls repeatable dataset seeds. It is a CMake formatted (semi-colon delimited) array of 32-bit unsigned integrals.
  - _(Note: semi-colons often collide with shell command parsing. It is advised to escape the entire CMake CLI argument to avoid the variable itself picking up quotation marks. Pass `cmake "-DPRNG_SEEDS=1;2;3;4"` instead of `cmake -DPRNG_SEEDS="1;2;3;4"`, the two cases differ in how the CMake executable receives its arguments from the OS.)_

## Running Examples
```sh
# Go to rocThrust build directory
cd rocThrust; cd build

# Configure with examples flag on
CXX=hipcc cmake -DBUILD_EXAMPLES=ON ..

# Build examples
make -j4

# Run the example you want to run
# ./examples/example_thrust_<example-name>
# For example:
./examples/example_thrust_version

# Example for linking with cpp files
./examples/cpp_integration/example_thrust_linking
```

## Running Benchmarks
```sh
# Go to rocThrust build directory
cd rocThrust; cd build

# Configure with benchmarks flag on
CXX=hipcc cmake -DBUILD_BENCHMARKS=ON ..

# Build benchmarks
make -j4

# Run the benchmarks
./benchmarks/benchmark_thrust_bench
```


## Documentation

Documentation is available [here](https://rocthrust.readthedocs.io/en/latest/).

## Support

Bugs and feature requests can be reported through [the issue tracker](https://github.com/ROCmSoftwarePlatform/rocThrust/issues).

## License

rocThrust is distributed under the [Apache 2.0 LICENSE](./LICENSE).
