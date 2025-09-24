# hipCUB

> [!NOTE]
> The published hipCUB documentation is available [here](https://rocm.docs.amd.com/projects/hipCUB/en/latest/) in an organized, easy-to-read format, with search and a table of contents. The documentation source files reside in the `docs` folder of this repository. As with all ROCm projects, the documentation is open source. For more information on contributing to the documentation, see [Contribute to ROCm documentation](https://rocm.docs.amd.com/en/latest/contribute/contributing.html).

hipCUB is a thin wrapper library on top of
[rocPRIM](https://github.com/ROCm/rocm-libraries) or
[CUB](https://github.com/nvidia/cccl). You can use it to port a CUB project into
[HIP](https://github.com/ROCm/HIP) so you can use AMD hardware (and
[ROCm](https://rocm.docs.amd.com/en/latest/) software).

In the [ROCm](https://rocm.docs.amd.com/en/latest/)
environment, hipCUB uses the rocPRIM library as the backend.

## Requirements

* Git
* CMake (3.18 or later)
* For AMD GPUs:
  * AMD [ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/native-install/index.html) software (1.8.0 or later)
    * The [HIP-clang](https://github.com/ROCm/HIP/blob/master/INSTALL.md#hip-clang) compiler (you
      must, set this as the C++ compiler for ROCm)
  * The [rocPRIM](https://github.com/ROCm/rocm-libraries) library
    * Automatically downloaded and built by the CMake script
    * Requires CMake 3.16.9 or later
* Python 3.6 or higher (for HIP on Windows only; this is only required for install scripts)
* Visual Studio 2019 with Clang support (HIP on Windows only)
* Strawberry Perl (HIP on Windows only)

Optional:

* [GoogleTest](https://github.com/google/googletest)
* [Google Benchmark](https://github.com/google/benchmark)

GoogleTest and Google Benchmark are automatically downloaded and built by the CMake script.

## Build and install

### Obtaining the source code

hipCUB can be cloned in two ways:

1.  Clone hipCUB along with other ROCm libraries that are frequently used together (note that this may take some time to complete):
```sh
git clone https://github.com/ROCm/rocm-libraries.git
cd rocm-libraries
```

2. To clone hipCUB individually (faster, but requires git version 2.25+):
```sh
git clone --no-checkout --depth=1 --filter=tree:0 https://github.com/ROCm/rocm-libraries.git
cd rocm-libraries
git sparse-checkout init --cone
git sparse-checkout set projects/hipcub
git checkout develop
```

### Building the library

```shell
# Go to the hipCUB directory.
cd projects/hipcub

# Create a directory for the build and navigate to it.
mkdir build; cd build

# Configure hipCUB, setup options for your system.
# Build options:
#   BUILD_TEST                   - OFF by default,
#   BUILD_BENCHMARK              - OFF by default.
#   ROCPRIM_FETCH_METHOD         - One of PACKAGE (default), DOWNLOAD, and MONOREPO. See below for a description of each.
#   EXTERNAL_DEPS_FORCE_DOWNLOAD - OFF by default, forces download for non-ROCm dependencies (eg. Google Test / Benchmark).
#   DOWNLOAD_CUB                 - OFF by default, (Nvidia CUB backend only) forces download of CUB instead of searching for an installed package.
#   BUILD_OFFLOAD_COMPRESS       - ON by default, compresses device code to reduce the size of the generated binary.
#   BUILD_EXAMPLE                - OFF by default, builds examples.
#   BUILD_ADDRESS_SANITIZER      - OFF by default, builds with clang address sanitizer enabled.
#   BUILD_COMPUTE_SANITIZER      - OFF by default, (Nvidia CUB backend only) builds tests with CUDA's compute sanitizer enabled.
#   USE_SYSTEM_LIB               - OFF by default, builds tests using the installed hipCUB provided by the system. This only takes effect when BUILD_TEST is ON.
#   USE_HIPCXX                   - OFF by default, builds with CMake HIP language support. This eliminates the need to set CXX.
#
# ! IMPORTANT !
# Set C++ compiler to HIP-aware clang. You can do it by adding 'CXX=<path-to-compiler>'
# before 'cmake' or setting cmake option 'CMAKE_CXX_COMPILER' to path to the compiler.
#
[CXX=hipcc] cmake ../. # or cmake-gui ../.
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

`ROCPRIM_FETCH_METHOD` can be used to control how hipCUB obtains the rocPRIM dependency. It must be set to one of the following values:
* `PACKAGE` (default) - Searches for an installed package on the system that meets the minimum version requirement. If it is not found, the build will fall back using option `DOWNLOAD`.
* `DOWNLOAD` - Clones rocPRIM from the upstream repository. If git >= 2.25 is present, this option uses a sparse checkout that avoids downloading more than it needs to. If not, the whole monorepo is downloaded (this may take some time).
* `MONOREPO` - This value is intended to be used if you are building hipCUB from within a copy of the rocm-libraries repository that you have cloned (and therefore already contains rocPRIM). When selected, the build will try find the dependency in the local repository tree. If it cannot be found, the build will attempt to use git to perform a sparse-checkout of rocPRIM. If that also fails, it will fall back to using the `DOWNLOAD` option described above.

### HIP on Windows

Initial support for HIP on Windows is available. You can install it using the provided `rmake.py` Python
script. To do this, first, clone rocThrust using the steps described in [obtaining the source code](#obtaining-the-source-code).
Next:

```shell
cd projects/hipcub

# the -i option will install rocPRIM to C:\hipSDK by default
python rmake.py -i

# the -c option will build all clients including unit tests
python rmake.py -c

# to build for a specific architecture only, use the -a option
python rmake.py -ci -a gfx1100

# for a full list of available options, please refer to the help documentation
python rmake.py -h
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
```

Include only the main header file:

```cpp
#include <hipcub/hipcub.hpp>
```

Depending on your current HIP platform, hipCUB includes CUB or rocPRIM headers.

## Running unit tests

```shell
# Go to hipCUB build directory
cd projects/hipcub; cd build

# To run all tests
ctest

# To run unit tests for hipCUB
./test/hipcub/<unit-test-name>
```

### Using custom seeds for the tests

Go to the `projects/hipcub/test/hipcub/test_seed.hpp` file.

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
cd projects/hipcub; cd build

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

After cloning this repository (see [obtaining the source code](#obtaining-the-source-code)):

```shell
cd rocm-libraries/projects/hipcub

# Install Python dependencies
python3 -m pip install -r docs/sphinx/requirements.txt

# Build the documentation
python3 -m sphinx -T -E -b html -d docs/_build/doctrees -D language=en docs docs/_build/html
```

You can then open `docs/_build/html/index.html` in your browser to view the documentation.

## Support

You can report bugs and feature requests through the GitHub
[issue tracker](https://github.com/ROCm/rocm-libraries/issues).
To help ensure that your issue is seen by the right team more quickly, when creating your issue, please apply the label `project: hipcub`.
Similarly, to filter the exising issue list down to only those affecting rocThrust, you can add the filter `label:"project: hipcub"`,
or follow [this link](https://github.com/ROCm/rocm-libraries/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22project%3A%20hipcub%22).

## Contributing

Contributions are most welcome! Learn more at [CONTRIBUTING](./CONTRIBUTING.md).
