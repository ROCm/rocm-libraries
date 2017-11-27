# rocPRIM

The rocPRIM is a header-only library providing HIP and HC parallel primitives for developing
performant GPU-accelerated code on AMD ROCm platform.

## Requirements

* Git
* CMake (3.10.0 or later)
* AMD [ROCm](https://rocm.github.io/install.html) platform (1.6.4 or later)

Optional:

* [GTest](https://github.com/google/googletest)
  * Required only for tests. Building tests is enabled by default.
  * Use `GTEST_ROOT` to specify GTest location (also see [FindGTest](https://cmake.org/cmake/help/latest/module/FindGTest.html)).
  * If GTest is not already installed, it will be automatically downloaded and built.
* [Google Benchmark](https://github.com/google/benchmark)
  * Required only for benchmarks. Building benchmarks is off by default.
  * If Google Benchmark is not already installed, it will be automatically downloaded and built.

If some dependencies are missing, CMake script automatically downloads, builds and
installs them. Setting `DEPENDENCIES_FORCE_DOWNLOAD` option `ON` forces script to
not to use system-installed libraries, and to download all dependencies.

## Build and Install

```
git clone https://github.com/ROCmSoftwarePlatform/rocPRIM.git

# Go to rocPRIM directory, create and go to the build directory.
cd rocPRIM; mkdir build; cd build

# Configure rocPRIM, setup options for your system.
# Build options:
#   BUILD_TEST - on by default,
#   BUILD_BENCHMARK - off by default.
cmake -DBUILD_BENCHMARK=ON ../. # or cmake-gui ../.

# Build
make -j4

# Optionally, run tests if they're enabled.
ctest --output-on-failure

# Install
[sudo] make install
```
## Running Unit Tests

## Running Benchmarks

## Documentation

## Support

Bugs and feature requests can be reported through [the issue tracker](https://github.com/ROCmSoftwarePlatform/rocPRIM/issues).

## Contributions and License

Contributions of any kind are most welcome! More details are found at [CONTRIBUTING](./CONTRIBUTING.md)
and [LICENSE](./LICENSE.txt).
