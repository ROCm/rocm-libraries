# Thrust

HIP back-end for Thrust (alpha release).

## Introduction

Thrust is a parallel algorithm library. This library has been ported to [HIP](https://github.com/ROCm-Developer-Tools/HIP)/[ROCm](https://rocm.github.io/) platform, which uses the [rocPRIM](https://github.com/ROCmSoftwarePlatform/rocPRIM) library. The HIP ported library works on HIP/ROCm platforms. Currently there is no CUDA backend in place.

## Requirements

### Software

* Git
* CMake (3.5.1 or later)
* AMD [ROCm](https://rocm.github.io/install.html) platform (1.8.0 or later)
  * Including [HipCC](https://github.com/ROCm-Developer-Tools/HIP) compiler, which must be
    set as C++ compiler on ROCm platform.
* [rocPRIM](https://github.com/ROCmSoftwarePlatform/rocPRIM) library
  * It will be automatically downloaded and built by CMake script.

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

This code sample computes the sum of 100 random numbers in parallel:

```c++
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <algorithm>
#include <cstdlib>

int main(void)
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

CMake Support
-------------

Thrust provides CMake configuration files that make it easy to include Thrust
from other CMake projects. See the [CMake README](thrust/cmake/README.md)
for details.

Development process
-------------------

For information on development process, see [this document](doc/development_model.md).
