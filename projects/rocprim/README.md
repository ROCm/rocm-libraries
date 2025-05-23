# rocPRIM

> [!NOTE]
> The published rocPRIM documentation is available [here](https://rocm.docs.amd.com/projects/rocPRIM/en/latest/) in an organized, easy-to-read format, with search and a table of contents. The documentation source files reside in the `docs` folder of this repository. As with all ROCm projects, the documentation is open source. For more information on contributing to the documentation, see [Contribute to ROCm documentation](https://rocm.docs.amd.com/en/latest/contribute/contributing.html).

rocPRIM is a header-only library that provides HIP parallel primitives. You can use this library to
develop performant GPU-accelerated code on AMD ROCm platforms.

## Requirements

* Git
* CMake (3.16 or later)
* AMD [ROCm](https://rocm.docs.amd.com/en/latest/) platform (1.8.2 or later)
  * Including
    [HIP-clang](https://github.com/ROCm/HIP/blob/master/INSTALL.md#hip-clang)
    compiler
* C++17
* Python 3.6 or higher (HIP on Windows only, required only for install script)
* Visual Studio 2019 with Clang support (HIP on Windows only)
* Strawberry Perl (HIP on Windows only)

Optional:

* [GoogleTest](https://github.com/google/googletest)
  * Required only for tests. Building tests is on by default.
  * This is automatically downloaded and built by the CMake script.
* [Google Benchmark](https://github.com/google/benchmark)
  * Required only for benchmarks. Building benchmarks is off by default.
  * This is automatically downloaded and built by the CMake script.

## Build and install

You can build and install rocPRIM on Linux or Windows.

* Linux:

  ```shell
  git clone https://github.com/ROCm/rocPRIM.git

  # Go to rocPRIM directory, create and go to the build directory.
  cd rocPRIM; mkdir build; cd build

  # Configure rocPRIM, setup options for your system.
  # Build options:
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

* Windows:

  We've added initial support for HIP on Windows; to install, use the provided `rmake.py` python script:

  ```shell
  git clone https://github.com/ROCm/rocPRIM.git
  cd rocPRIM

  # the -i option will install rocPRIM to C:\hipSDK by default
  python rmake.py -i

  # the -c option will build all clients including unit tests
  python rmake.py -c
  ```

### Using rocPRIM

Include the `<rocprim/rocprim.hpp>` header:

```cpp
#include <rocprim/rocprim.hpp>
```

We recommended including rocPRIM into a CMake project by using the package configuration files.
The rocPRIM package name is `rocprim`.

```cmake
# "/opt/rocm" - default install prefix
find_package(rocprim REQUIRED CONFIG PATHS "/opt/rocm/rocprim")

...

# Includes only rocPRIM headers, HIP libraries have
# to be linked manually by user
target_link_libraries(<your_target> roc::rocprim)

# Include rocPRIM headers and required HIP dependencies
# - If using HIP language support (USE_HIPCXX=ON):
target_link_libraries(<your_target> hip::host)

# - Otherwise:
target_link_libraries(<your_target> hip::device)
```

For more information on `hip::host` and `hip::device`, please see the [ROCm documentation](https://rocm.docs.amd.com/en/latest/conceptual/cmake-packages.html#consuming-the-hip-api-in-c-code).

## Running unit tests

Unit tests are implemented in terms of GoogleTest. Collections of tests are wrapped and invoked from
CTest.

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

This feature requires using CMake 3.16+ for building and testing.

```note
Prior versions of CMake can't assign IDs to tests when running in parallel. Assigning tests to distinct
devices could only be done at the cost of extreme complexity.
```

Unit tests can make use of the
[CTest resource allocation](https://cmake.org/cmake/help/latest/manual/ctest.1.html#resource-allocation)
feature, which you can use to distribute tests across multiple GPUs in an intelligent manner. This
feature can accelerate testing when multiple GPUs of the same family are in a system. It can also test
multiple product families from one invocation without having to use the `HIP_VISIBLE_DEVICES`
environment variable. The feature relies on the presence of a resource specifications file.

```important
Trying to use `RESOURCE_GROUPS` and `--resource-spec-file` with CMake and CTest for versions prior
to 3.16 silently omits the feature. No warnings are issued about unknown properties or command-line
arguments. Make sure that the `cmake` and `ctest` versions you invoke are sufficiently recent.
```

#### Auto resource specification generation

You can independently call the utility script located in the repository using the following code:

```shell
# Go to rocPRIM build directory
cd rocPRIM; cd build

# Invoke directly or use CMake script mode via cmake -P
../cmake/GenerateResourceSpec.cmake

# Assuming you have 2 compatible GPUs in the system
ctest --resource-spec-file ./resources.json --parallel 2
```

#### Manual

Assuming you have two GPUs from the gfx900 family and they are the first devices enumerated by the
system, you can use `-D AMDGPU_TEST_TARGETS=gfx900` during configuration to specify that only
one family will be tested. Leaving this var empty (default) results in targeting the default device in the
system. To let CMake know there are two GPUs that should be targeted, you have to provide a `JSON`
file to CTest via the `--resource-spec-file <path_to_file>` flag. For example:

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

Invoking CTest as `ctest --resource-spec-file <path_to_file> --parallel 2` allows two tests to run
concurrently, distributed between the two GPUs.

### Using custom seeds for the tests

Modify the `rocPRIM/test/rocprim/test_seed.hpp` file.

```cpp
//(1)
static constexpr int random_seeds_count = 10;

//(2)
static constexpr unsigned int seeds [] = {0, 2, 10, 1000};

//(3)
static constexpr size_t seed_size = sizeof(seeds) / sizeof(seeds[0]);
```

(1) Defines a constant that sets how many passes over the tests will be done with runtime-generated
seeds. Modify at will.

(2) Defines the user-generated seeds. Each of the array elements will be used as seed for all tests.
Modify at will. If you don't want any static seeds, leave the array empty.

```cpp
static constexpr unsigned int seeds [] = {};
```

(3) Never modify this line.

## Running benchmarks

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

Most device-specific primitives provided by rocPRIM can be tuned for other AMD devices, and
different types and operations, by passing compile-time configuration structures as a template
parameter. The main "knobs" are usually the size of the block and the number of items processed by a
single thread.

rocPRIM has built-in default configurations for each of its primitives, these will be used automatically
based on the input types and the target architecture from the stream used.

## hipCUB

[hipCUB](https://github.com/ROCm/hipCUB/) is a thin wrapper library on top of
[rocPRIM](https://github.com/ROCm/rocPRIM) or
[CUB](https://github.com/NVlabs/cub). You can use it to port projects that use the CUB library to the
[HIP](https://github.com/ROCm/HIP) layer and run them on AMD hardware. In the
[ROCm](https://rocm.docs.amd.com/en/latest/) environment, hipCUB uses the rocPRIM library as a
backend; on CUDA platforms, it uses CUB as a backend.

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
pyenv virtualenv 3.10 venv_rocprim

# Activate the virtual environment
pyenv activate venv_rocprim
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

### Build documentation via CMake

Install [rocm-cmake](https://github.com/ROCm/rocm-cmake/)

```shell
# Change directory to rocPRIM
cd rocPRIM

# Install documentation dependencies
python3 -m pip install -r docs/sphinx/requirements.txt

# Set C++ compiler
# This example uses hipcc and assumes it is at the path /usr/bin
export CXX=hipcc
export PATH=/usr/bin:$PATH

# Configure the project
cmake -S . -B ./build -D BUILD_DOCS=ON

# Build the documentation
cmake --build ./build --target doc

# To serve the HTML docs locally
cd ./build/docs/html
python3 -m http.server
```

## Support

You can report bugs and feature requests through our GitHub
[issue tracker](https://github.com/ROCm/rocPRIM/issues).

## Contributions and license

Contributions of any kind are most welcome! Contribution instructions are in
[CONTRIBUTING](./CONTRIBUTING.md).

Licensing information is in [LICENSE](./LICENSE.txt).
