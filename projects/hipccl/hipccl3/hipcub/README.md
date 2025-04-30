# hipCUB

> [!NOTE]
> The published hipCUB documentation is available [here](https://rocm.docs.amd.com/projects/hipCUB/en/latest/) in an organized, easy-to-read format, with search and a table of contents. The documentation source files reside in the `docs` folder of this repository. As with all ROCm projects, the documentation is open source. For more information on contributing to the documentation, see [Contribute to ROCm documentation](https://rocm.docs.amd.com/en/latest/contribute/contributing.html).

hipCUB is a thin wrapper library on top of
[rocPRIM](https://github.com/ROCm/rocPRIM) or
[CUB](https://github.com/thrust/cub). You can use it to port a CUB project into
[HIP](https://github.com/ROCm/HIP) so you can use AMD hardware (and
[ROCm](https://rocm.docs.amd.com/en/latest/) software).

In the [ROCm](https://rocm.docs.amd.com/en/latest/)
environment, hipCUB uses the rocPRIM library as the backend. On CUDA platforms, it uses CUB as the
backend.

## Requirements

* Git
* CMake (3.16 or later)
* For AMD GPUs:
  * AMD [ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/native-install/index.html) software (1.8.0 or later)
    * The [HIP-clang](https://github.com/ROCm/HIP/blob/master/INSTALL.md#hip-clang) compiler (you
      must, set this as the C++ compiler for ROCm)
  * The [rocPRIM](https://github.com/ROCm/rocPRIM) library
    * Automatically downloaded and built by the CMake script
    * Requires CMake 3.16.9 or later
* For NVIDIA GPUs:
  * CUDA Toolkit
  * CCCL library (>= 2.6.0)
    * Automatically downloaded and built by the CMake script
    * Requires CMake 3.15.0 or later
* Python 3.6 or higher (for HIP on Windows only; this is only required for install scripts)
* Visual Studio 2019 with Clang support (HIP on Windows only)
* Strawberry Perl (HIP on Windows only)

Optional:

* [GoogleTest](https://github.com/google/googletest)
* [Google Benchmark](https://github.com/google/benchmark)

GoogleTest and Google Benchmark are automatically downloaded and built by the CMake script.

## Build and install

To build and install hipCub, run the following code:

```shell
git clone https://github.com/ROCm/hipCUB.git

# Go to hipCUB directory, create and go to the build directory.
cd hipCUB; mkdir build; cd build

# Configure hipCUB, setup options for your system.
# Build options:
#   BUILD_TEST - OFF by default,
#   BUILD_BENCHMARK - OFF by default.
#   DEPENDENCIES_FORCE_DOWNLOAD - OFF by default and at ON the dependencies will be downloaded to build folder,
#
# ! IMPORTANT !
# Set C++ compiler to HIP-aware clang. You can do it by adding 'CXX=<path-to-compiler>'
# before 'cmake' or setting cmake option 'CMAKE_CXX_COMPILER' to path to the compiler.
#
[CXX=hipcc] cmake ../. # or cmake-gui ../.

# To configure hipCUB for Nvidia platforms, 'CXX=<path-to-nvcc>', `CXX=nvcc` or omitting the flag
# entirely before 'cmake' is sufficient
[CXX=nvcc] cmake -DBUILD_TEST=ON ../. # or cmake-gui ../.
# or
cmake -DBUILD_TEST=ON ../. # or cmake-gui ../.
# or to build benchmarks
cmake -DBUILD_BENCHMARK=ON ../.

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

Initial support for HIP on Windows is available. You can install it using the provided `rmake.py` Python
script:

```shell
git clone https://github.com/ROCm/hipCUB.git
cd hipCUB

# the -i option will install rocPRIM to C:\hipSDK by default
python rmake.py -i

# the -c option will build all clients including unit tests
python rmake.py -c
```

### Using hipCUB

To use hipCUB in a CMake project, we recommended using the package configuration files.

```cmake
# On ROCm hipCUB requires rocPRIM
find_package(rocprim REQUIRED CONFIG PATHS "/opt/rocm/lib/cmake/rocprim")

# "/opt/rocm" - default install prefix
find_package(hipcub REQUIRED CONFIG PATHS "/opt/rocm/lib/cmake/hipcub")

...
# On ROCm: includes hipCUB headers and roc::rocprim_hip target
# On CUDA: includes only hipCUB headers, user has to include CUB directory
target_link_libraries(<your_target> hip::hipcub)
```

Include only the main header file:

```cpp
#include <hipcub/hipcub.hpp>
```

Depending on your current HIP platform, hipCUB includes CUB or rocPRIM headers.

## Running unit tests

```shell
# Go to hipCUB build directory
cd hipCUB; cd build

# To run all tests
ctest

# To run unit tests for hipCUB
./test/hipcub/<unit-test-name>
```

### Using custom seeds for the tests

Go to the `hipCUB/test/hipcub/test_seed.hpp` file.

```cpp
//(1)
static constexpr int random_seeds_count = 10;

//(2)
static constexpr unsigned int seeds [] = {0, 2, 10, 1000};

//(3)
static constexpr size_t seed_size = sizeof(seeds) / sizeof(seeds[0]);
```

(1) Defines a constant that sets how many passes are performed over the tests with runtime-generated
 seeds. Modify at will.

(2) Defines the user-generated seeds. Each of the elements of the array are used as seeds for all tests.
 Modify at will. If no static seeds are desired, leave the array empty.

  ```cpp
  static constexpr unsigned int seeds [] = {};
  ```

(3) Never modified this line.

## Running benchmarks

```shell
# Go to hipCUB build directory
cd hipCUB; cd build

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
pyenv virtualenv 3.10 venv_hipcub

# Activate the virtual environment
pyenv activate venv_hipcub
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
[GitHub issue tracker](https://github.com/ROCm/hipCUB/issues).

## Contributing

Contributions are most welcome! Learn more at [CONTRIBUTING](./CONTRIBUTING.md).
