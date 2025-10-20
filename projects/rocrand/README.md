# rocRAND

> [!NOTE]
> The published rocRAND documentation is available [here](https://rocm.docs.amd.com/projects/rocRAND/en/latest/) in an organized, easy-to-read format, with search and a table of contents. The documentation source files reside in the `docs` folder of this repository. As with all ROCm projects, the documentation is open source. For more information on contributing to the documentation, see [Contribute to ROCm documentation](https://rocm.docs.amd.com/en/latest/contribute/contributing.html).

The rocRAND project provides functions that generate pseudorandom and quasirandom numbers.
The rocRAND library is implemented in the [HIP](https://github.com/ROCm/rocm-systems/tree/develop/projects/hip)
programming language and optimized for AMD's latest discrete GPUs. It is designed to run on top
of AMD's [ROCm](https://rocm.docs.amd.com) runtime.

Prior to ROCm version 5.0, this project included the
[hipRAND](https://github.com/ROCm/rocm-libraries/tree/develop/projects/hiprand) wrapper. As of version 5.0, it was
split into a separate library. As of version 6.0, hipRAND can no longer be built from rocRAND.

## Supported random number generators

* XORWOW
* MRG31k3p
* MRG32k3a
* Mersenne Twister (MT19937)
* Mersenne Twister for Graphic Processors (MTGP32)
* Philox (4x32, 10 rounds)
* LFSR113
* Sobol32
* Scrambled Sobol32
* Sobol64
* Scrambled Sobol64
* ThreeFry

## Requirements

* CMake (3.16 or later)
* C++ compiler with C++17 support to build the library.
  * Recommended to use at least gcc 9
  * clang uses the development headers and libraries from gcc, so a recent version of it must still
    be installed when compiling with clang
* C++ compiler with C++11 support to consume the library.
* For AMD platforms:
  * [ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/native-install/index.html) (1.7 or later)
  * [HIP-clang](https://rocm.docs.amd.com/projects/HIP/en/latest/install/install.html) compiler, which must be
    set as the C++ compiler for the ROCm platform.
* Python 3.6 or higher (HIP on Windows only, only required for install script)
* Visual Studio 2019 with clang support (HIP on Windows only)
* Strawberry Perl (HIP on Windows only)

Optional:

* [GoogleTest](https://github.com/google/googletest) (required only for tests; building tests is enabled
  by default)
  * Use `GTEST_ROOT` to specify the GoogleTest location (see also
    [FindGTest](https://cmake.org/cmake/help/latest/module/FindGTest.html))
  * Note: If GoogleTest is not already installed, it will be automatically downloaded and built
* Fortran compiler (required only for Fortran wrapper)
  * `gfortran` is recommended
* Python 3.5+ (required only for Python wrapper)
* [doxygen](https://www.doxygen.nl/) to build the documentation

If some dependencies are missing, the CMake script automatically downloads, builds, and installs them.
Setting the `DEPENDENCIES_FORCE_DOWNLOAD` option to `ON` forces the script to download all
dependencies, rather than using the system-installed libraries.

## Build and install

> [!NOTE]
> The following clone command downloads all components in the [rocm-libraries](https://github.com/ROCm/rocm-libraries) GitHub repository.
This is recommended for working with multiple library components, but can take a very long time to
download. For a shorter download process that only clones the rocRAND library, see the
[rocRAND installation documentation](https://rocm.docs.amd.com/projects/rocRAND/en/latest/install/installing.html)
for ROCm 7.0 or later.

```shell
git clone https://github.com/ROCm/rocm-libraries.git

# Go to rocRAND directory, create and go to build directory
cd rocm-libraries/projects/rocrand; mkdir build; cd build

# Configure rocRAND, setup options for your system
# Build options:
#   BUILD_SHARED_LIBS - ON by default.
#   BUILD_TEST        - OFF by default.
#   BUILD_BENCHMARK   - OFF by default.
#   USE_SYSTEM_LIB    - OFF by default. Setting it to ON will build tests using the existing ``rocrand``
#                       library installation from the system. This only takes effect when BUILD_TEST is ON
#                       and the ``rocrand`` installation must be compatible with the version of the tests.
#                       This option can be used to build tests exclusively when you do not intend to build
#                       the library nor the benchmarks.
# Additionally, the ROCm installation prefix should be passed using CMAKE_PREFIX_PATH or by setting the ROCM_PATH environment variable.
#
# ! IMPORTANT !
# Set C++ compiler to HIP-clang. You can do it by adding 'CXX=<path-to-compiler>'
# before 'cmake' or setting cmake option 'CMAKE_CXX_COMPILER' to path to the compiler.
#
# The python interface do not work with static library.
#
[CXX=hipcc] cmake -DBUILD_BENCHMARK=ON ../. -DCMAKE_PREFIX_PATH=/opt/rocm # or cmake-gui ../.

# Build
make -j4

# Optionally, run tests if they're enabled
ctest --output-on-failure

# Install
[sudo] make install
```

### SPIR-V

rocRAND supports the `amdgcnspirv` target, but it should be built with `USE_DEVICE_DISPATCH`
turned off like `-DUSE_DEVICE_DISPATCH=0`.

### HIP on Windows

We've added initial support for HIP on Windows, which you can install using the `rmake.py` python
script:

```shell
git clone https://github.com/ROCm/rocm-libraries.git
cd rocm-libraries/projects/rocrand

# the -i option will install rocPRIM to C:\hipSDK by default
python rmake.py -i

# the -c option will build all clients including unit tests
python rmake.py -c
```

The existing GoogleTest library in the system (especially static GoogleTest libraries built with other
compilers) may cause a build failure; if you encounter errors with the existing GoogleTest library or
other dependencies, you can pass the `DEPENDENCIES_FORCE_DOWNLOAD` flag to CMake, which can
help to solve the problem.

## Running unit tests

```shell
# Go to rocRAND build directory
cd rocm-libraries/projects/rocrand; cd build

# To run all tests
ctest

# To run unit tests
./test/<unit-test-name>
```

## Running benchmarks

```shell
# Go to rocRAND build directory
cd rocm-libraries/projects/rocrand; cd build

# To run benchmark for the host generate functions:
# The benchmarks are registered with Google Benchmark as `device_generate<engine,distribution>`, where
# engine -> xorwow, mrg31k3p, mrg32k3a, mtgp32, philox, lfsr113, mt19937,
#           threefry2x32, threefry2x64, threefry4x32, threefry4x64,
#           sobol32, scrambled_sobol32, sobol64, scrambled_sobol64
# distribution -> uniform-uint, uniform-uchar, uniform-ushort,
#                 uniform-half, uniform-float, uniform-double,
#                 normal-half, normal-float, normal-double,
#                 log-normal-half, log-normal-float, log-normal-double, poisson
# Further option can be found using --help
./benchmark/benchmark_rocrand_host_api
# To run specific benchmarks:
./benchmark/benchmark_rocrand_host_api --benchmark_filter=<regex>
# For example to run benchmarks with engine sobol64:
./benchmark/benchmark_rocrand_host_api --benchmark_filter="device_generate<sobol64*"
# To view all registered benchmarks:
./benchmark/benchmark_rocrand_host_api --benchmark_list_tests=true
# The benchmark also supports user input:
./benchmark/benchmark_rocrand_host_api --size <number> --trials <number> --offset <number> --dimensions <number> --lambda <float float float ...>
# And can print output in different formats:
./benchmark/benchmark_rocrand_host_api --benchmark_format=<console|json|csv>

# To run benchmark for device kernel functions:
# The benchmarks are registered with Google Benchmark as `device_kernel<engine,distribution>`, where
# engine -> xorwow, mrg31k3p, mrg32k3a, mtgp32, philox, lfsr113,
#           threefry2x32, threefry2x64, threefry4x32, threefry4x64,
#           sobol32, scrambled_sobol32, sobol64, scrambled_sobol64
# distribution -> uniform-uint or uniform-ullong, uniform-float, uniform-double, normal-float, normal-double,
#                 log-normal-float, log-normal-double, poisson, discrete-poisson, discrete-custom
# Further option can be found using --help
./benchmark/benchmark_rocrand_device_api
# To run specific benchmarks:
./benchmark/benchmark_rocrand_device_api --benchmark_filter=<regex>
# For example to run benchmarks with engine sobol64:
./benchmark/benchmark_rocrand_device_api --benchmark_filter="device_kernel<sobol64*"
# To view all registered benchmarks:
./benchmark/benchmark_rocrand_device_api --benchmark_list_tests=true
# The benchmark also supports user input:
./benchmark/benchmark_rocrand_device_api --size <number> --trials <number> --dimensions <number> --lambda <float float float ...>
# And can print output in different formats:
./benchmark/benchmark_rocrand_device_api --benchmark_format=<console|json|csv>

# To compare against cuRAND (cuRAND must be supported):
./benchmark/benchmark_curand_host_api [google benchmark options]
./benchmark/benchmark_curand_device_api [google benchmark options]
```

### Legacy benchmarks

You can disable legacy benchmarks (those used prior to Google Benchmark) by setting the
CMake option `BUILD_LEGACY_BENCHMARK` to `OFF`. For compatibility, the default setting is `ON`
when `BUILD_BENCHMARK` is set.

Legacy benchmarks are deprecated and will be removed in a future version once all benchmarks have
been migrated to the new framework.

## Wrappers

* C++ wrappers for the rocRAND host API are in [`rocrand.hpp`](./library/include/rocrand/rocrand.hpp).
* [Fortran wrappers](./library/src/fortran/).
* [Python wrappers](./python/): [rocRAND](./python/rocrand).

## Building the documentation locally

### Requirements

#### Doxygen

The build system uses Doxygen [version 1.9.4](https://github.com/doxygen/doxygen/releases/tag/Release_1_9_4). You can try using a newer version, but that might cause issues.

After you have downloaded Doxygen version 1.9.4:

```shell
# Add doxygen to your PATH
echo 'export PATH=<doxygen 1.9.4 path>/bin:$PATH' >> ~/.bashrc

# Apply the updated .bashrc
source ~/.bashrc

# Confirm that you are using version 1.9.4
doxygen --version
```

#### Python

The build system uses Python version 3.10. You can try using a newer version, but that might cause issues.

You can install Python 3.10 alongside your other Python versions using [pyenv](https://github.com/pyenv/pyenv?tab=readme-ov-file#installation):

```shell
# Install Python 3.10
pyenv install 3.10

# Create a Python 3.10 virtual environment
pyenv virtualenv 3.10 venv_rocrand

# Activate the virtual environment
pyenv activate venv_rocrand
```

### Building

After cloning this repository, and `cd`ing into it:

```shell
# Install Python dependencies
python3 -m pip install -r docs/sphinx/requirements.txt

# Build the documentation
python3 -m sphinx -T -E -b html -d docs/_build/doctrees -D language=en docs docs/_build/html
```

You can then open `docs/_build/html/index.html` in your browser to view the documentation.

## Support

Bugs and feature requests can be reported through the
[issue tracker](https://github.com/ROCm/rocm-libraries/issues).

## Contributions and license

Contributions of any kind are most welcome! You can find more information at
[CONTRIBUTING](./CONTRIBUTING.md).

Licensing information is located at [LICENSE](./LICENSE.txt).
