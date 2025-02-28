# rocThrust

> [!NOTE]
> The published rocThrust documentation is available [here](https://rocm.docs.amd.com/projects/rocThrust/en/latest/) in an organized, easy-to-read format, with search and a table of contents. The documentation source files reside in the `docs` folder of this repository. As with all ROCm projects, the documentation is open source. For more information on contributing to the documentation, see [Contribute to ROCm documentation](https://rocm.docs.amd.com/en/latest/contribute/contributing.html).

Thrust is a parallel algorithm library. It has been ported to
[HIP](https://github.com/ROCm/HIP) and [ROCm](https://www.github.com/ROCm/ROCm), which use
the [rocPRIM](https://github.com/ROCm/rocPRIM) library. The HIP-ported library
works on HIP and ROCm software. Currently there is no CUDA backend in place.

## Requirements

Software requirements include:

* CMake (3.10.2 or later)
* AMD [ROCm](https://rocm.docs.amd.com) Software (1.8.0 or later)
  * Including the [HipCC](https://github.com/ROCm/HIP) compiler, which must be set
    as your C++ compiler for ROCm
* [rocPRIM](https://github.com/ROCm/rocPRIM) library
  * This is automatically downloaded and built by the CMake script
* Python 3.6 or higher (for HIP on Windows; only required for install scripts)
* Visual Studio 2019 with Clang support (for HIP on Windows)
* Strawberry Perl (for HIP on Windows)

Optional:

* [GoogleTest](https://github.com/google/googletest)
  * Required only for tests; building tests is enabled by default
  * This is automatically downloaded and built by the CMake script
* [doxygen](https://www.doxygen.nl/)
  * Required for building the documentation

For ROCm hardware requirements, refer to:

* [Linux support](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)
* [Windows support](https://rocm.docs.amd.com/projects/install-on-windows/en/latest/reference/system-requirements.html)

## Build and install

```sh
git clone https://github.com/ROCm/rocThrust

# Go to rocThrust directory, create and go to the build directory.
cd rocThrust; mkdir build; cd build

# Configure rocThrust, setup options for your system.
# Build options:
#   DISABLE_WERROR        - ON  by default, This flag disable the -Werror compiler flag
#   BUILD_TEST            - OFF by default,
#   BUILD_HIPSTDPAR_TEST  - OFF by default,
#   BUILD_EXAMPLES        - OFF by default,
#   BUILD_BENCHMARKS      - OFF by default,
#   DOWNLOAD_ROCPRIM      - OFF by default, when ON rocPRIM will be downloaded to the build folder,
#   RNG_SEED_COUNT        - 0 by default, controls non-repeatable random dataset count
#   PRNG_SEEDS            - 1 by default, reproducible seeds to generate random data
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

We've added initial support for HIP on Windows. To install, use the provided `rmake.py` Python script:

```shell
git clone https://github.com/ROCm/rocThrust.git
cd rocThrust

# the -i option will install rocPRIM to C:\hipSDK by default
python rmake.py -i

# the -c option will build all clients including unit tests
python rmake.py -c
```

### Macro options

```cpp
# Performance improvement option. If you define THRUST_HIP_PRINTF_ENABLED before
# thrust includes to 0, you can disable printfs on device side and improve
# performance. The default value is 1
#define THRUST_HIP_PRINTF_ENABLED 0
```

### Using rocThrust in a project

We recommended including rocThrust into a CMake project by using its package configuration files.

```cmake
# On ROCm rocThrust requires rocPRIM
find_package(rocprim REQUIRED CONFIG PATHS "/opt/rocm/rocprim")

# "/opt/rocm" - default install prefix
find_package(rocthrust REQUIRED CONFIG PATHS "/opt/rocm/rocthrust")

...
includes rocThrust headers and roc::rocprim_hip target
target_link_libraries(<your_target> roc::rocthrust)
```

## Running unit tests

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

This feature requires CMake 3.16+ to be used for building and testing. *(Prior versions of CMake can't
assign IDs to tests when running in parallel. Assigning tests to distinct devices could only be done at
the cost of extreme complexity.)*

Unit tests can make use of the
[CTest Resource Allocation](https://cmake.org/cmake/help/latest/manual/ctest.1.html#resource-allocation) feature, which enables distributing tests across multiple GPUs in an intelligent manner. This feature can
accelerate testing when multiple GPUs of the same family are in a system. It can also test multiple
product families from one invocation without having to use the `HIP_VISIBLE_DEVICES` environment
variable. CTest Resource Allocation requires a resource spec file.

```important
Using `RESOURCE_GROUPS` and `--resource-spec-file` with CMake and CTest, respectively for versions
prior to 3.16 omits the feature silently. Therefore, you must ensure that the `cmake` and `ctest` you
invoke are sufficiently recent.
```

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

Assuming you have two GPUs from the gfx900 family and they are the first devices enumerated by the
system, you can specify `-D AMDGPU_TEST_TARGETS=gfx900` during configuration to specify that you
want only one family to be tested. If you leave this var empty (default), the default device in the system
is targeted. To specify that there are two GPUs that should be targeted, you must feed a JSON file to
CTest using the `--resource-spec-file <path_to_file>` flag. For example:

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

There are two CMake configuration-time options that control random data fed to unit tests.

* `RNG_SEED_COUNT`: 0 by default, controls non-repeatable random dataset count.
  * Draws values from a default constructed `std::random_device`.
  * Should tests fail, the actual seed producing the failure is reported by Googletest, which allows for
    reproducibility.

* `PRNG_SEEDS`: 1 by default, controls repeatable dataset seeds.
  * This is a CMake formatted (semicolon delimited) array of 32-bit unsigned integers. Note that
    semicolons often collide with shell command parsing. We advise escaping the entire CMake CLI
    argument to avoid having the variable pick up quotation marks. For example, pass
    `cmake "-DPRNG_SEEDS=1;2;3;4"` instead of `cmake -DPRNG_SEEDS="1;2;3;4"` (these cases differ in
    how the CMake executable receives arguments from the operating system).

## Running examples

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

## Running benchmarks

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

## HIPSTDPAR

rocThrust also hosts the header files for [HIPSTDPAR](https://rocm.blogs.amd.com/software-tools-optimization/hipstdpar/README.html#c-17-parallel-algorithms-and-hipstdpar).
Within these headers, a great part of the C++ Standard Library parallel algorithms are overloaded so that rocThrust's and rocPRIM's implementations of those algorithms are used when they are invoked with the `parallel_unsequenced_policy` policy.
When compiling with the proper flags (see [LLVM (AMD's fork) docs](https://github.com/ROCm/llvm-project/blob/rocm-6.2.x/clang/docs/HIPSupport.rst#implementation-driver)[^1] for the complete list), the HIPSTDPAR headers are implicitly included by the compiler, and therefore the execution of these parallel algorithms will be offloaded to AMD devices.

[^1]: Altough currently only AMD's fork of LLVM contains the docs for the [C++ Standard Parallelism Offload Support](https://github.com/ROCm/llvm-project/blob/rocm-6.2.x/clang/docs/HIPSupport.rst#c-standard-parallelism-offload-support-compiler-and-runtime), both of them (the upstream LLVM and AMD's fork) do support it.

### Install

HIPSTDPAR is currently packaged along rocThrust. The `hipstdpar` package is set up as a virtual package provided by `rocthrust`, so the latter needs to be installed entirely for getting HIPSTDPAR's headers. Conversely, installing the `rocthrust` package will also include HIPSTDPAR's headers in the system.

### Tests
rocThrust also includes tests to check the correct building of HIPSTDPAR implementations. They are located in the [tests/hipstdpar](/test/hipstdpar/) folder. When configuring the project with the `BUILD_TEST` option, these tests will not be enabled by default. To enable them, set `BUILD_HIPSTDPAR_TEST=ON`. Additionally, you can configure only HIPSTDPAR's tests by disabling `BUILD_TEST` and enabling `BUILD_HIPSTDPAR_TEST`. In general, the following steps can be followed for building and running the tests:

```sh
git clone https://github.com/ROCm/rocThrust

# Go to rocThrust directory, create and go to the build directory.
cd rocThrust; mkdir build; cd build

# Configure rocThrust.
[CXX=hipcc] cmake ../. -D BUILD_TEST=ON # Configure rocThrust's tests.
[CXX=hipcc] cmake ../. -D BUILD_TEST=ON -D BUILD_HIPSTDPAR_TEST=ON # Configure both rocThrust's tests and HIPSTDPAR's tests.
[CXX=hipcc] cmake ../. -D BUILD_TEST=OFF -D BUILD_HIPSTDPAR_TEST=ON # Only configure HIPSTDPAR's tests.

# Build
make -j4

# Run tests.
ctest --output-on-failure
```

#### Requirements

* [rocPRIM](https://github.com/ROCm/rocPRIM) and [rocThrust](https://github.com/ROCm/rocThrust) libraries
* [TBB](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onetbb.html) library
  * Notice that oneTBB (oneAPI TBB) may fail to compile when libstdc++-9 or -10 is used, due to them using legacy TBB interfaces that are incompatible with the oneTBB ones (see the [release notes](https://www.intel.com/content/www/us/en/developer/articles/release-notes/intel-oneapi-threading-building-blocks-release-notes.html)).
* CMake (3.10.2 or later)

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
pyenv virtualenv 3.10 venv_rocthrust

# Activate the virtual environment
pyenv activate venv_rocthrust
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

You can report bugs and feature requests through the GitHub
[issue tracker](https://github.com/ROCm/rocThrust/issues).

## License

rocThrust is distributed under the [Apache 2.0 LICENSE](./LICENSE).
