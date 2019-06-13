# Thrust

HIP back-end for Thrust (alpha release).

## Introduction

Thrust is a parallel algorithm library. This library has been ported to [HIP](https://github.com/ROCm-Developer-Tools/HIP)/[ROCm](https://rocm.github.io/) platform, which uses the [rocPRIM](https://github.com/ROCmSoftwarePlatform/rocPRIM) library. The HIP ported library works on HIP/ROCm platforms. Currently there is no CUDA backend in place.

## Requirements

### Software

* Git
* CMake (3.5.1 or later)
* AMD [ROCm](https://rocm.github.io/install.html) platform (1.8.0 or later)
  * Including [HCC](https://github.com/RadeonOpenCompute/hcc) compiler, which must be
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
#   BUILD_TEST - ON by default,
#
# ! IMPORTANT !
# On ROCm platform set C++ compiler to HCC. You can do it by adding 'CXX=<path-to-hcc>'
# before 'cmake' or setting cmake option 'CMAKE_CXX_COMPILER' with the path to the HCC compiler.
#
[CXX=hcc] cmake ../. # or cmake-gui ../.

# Build
make -j4

# Optionally, run tests if they're enabled.
ctest --output-on-failure

# Package
make package

# Install
[sudo] make install
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
target_link_libraries(<your_target> rocthrust)
```

## Running Unit Tests

```sh
# Go to rocThrust build directory
cd rocThrust; cd build

# To run all tests
ctest

# To run unit tests for rocThrust
./test/<unit-test-name>
```

## Documentation

Documentation is available [here](https://rocthrust.readthedocs.io/en/latest/).

## Support

Bugs and feature requests can be reported through [the issue tracker](https://github.com/ROCmSoftwarePlatform/rocThrust/issues).

## Contributions and License

Contributions of any kind are most welcome! More details are found at [CONTRIBUTING](./CONTRIBUTING.md)
and [LICENSE](./LICENSE.txt).
