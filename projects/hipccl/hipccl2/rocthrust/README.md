# Thrust

HIP back-end for Thrust (alpha release).

Thrust is included in the NVIDIA HPC SDK and the CUDA Toolkit.

Refer to the [Quick Start Guide](http://github.com/thrust/thrust/wiki/Quick-Start-Guide) page for further information and examples.

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

Releases
--------

Thrust is distributed with the NVIDIA HPC SDK and the CUDA Toolkit in addition
to GitHub.

See the [changelog](CHANGELOG.md) for details about specific releases.

| Thrust Release    | Included In                             |
| ----------------- | --------------------------------------- |
| 1.9.10-1          | NVIDIA HPC SDK 20.7 & CUDA Toolkit 11.1 |
| 1.9.10            | NVIDIA HPC SDK 20.5                     |
| 1.9.9             | CUDA Toolkit 11.0                       |
| 1.9.8-1           | NVIDIA HPC SDK 20.3                     |
| 1.9.8             | CUDA Toolkit 11.0 Early Access          |
| 1.9.7-1           | CUDA Toolkit 10.2 for Tegra             |
| 1.9.7             | CUDA Toolkit 10.2                       |
| 1.9.6-1           | NVIDIA HPC SDK 20.3                     |
| 1.9.6             | CUDA Toolkit 10.1 Update 2              |
| 1.9.5             | CUDA Toolkit 10.1 Update 1              |
| 1.9.4             | CUDA Toolkit 10.1                       |
| 1.9.3             | CUDA Toolkit 10.0                       |
| 1.9.2             | CUDA Toolkit 9.2                        |
| 1.9.1-2           | CUDA Toolkit 9.1                        |
| 1.9.0-5           | CUDA Toolkit 9.0                        |
| 1.8.3             | CUDA Toolkit 8.0                        |
| 1.8.2             | CUDA Toolkit 7.5                        |
| 1.8.1             | CUDA Toolkit 7.0                        |
| 1.8.0             |                                         |
| 1.7.2             | CUDA Toolkit 6.5                        |
| 1.7.1             | CUDA Toolkit 6.0                        |
| 1.7.0             | CUDA Toolkit 5.5                        |
| 1.6.0             |                                         |
| 1.5.3             | CUDA Toolkit 5.0                        |
| 1.5.2             | CUDA Toolkit 4.2                        |
| 1.5.1             | CUDA Toolkit 4.1                        |
| 1.5.0             |                                         |
| 1.4.0             | CUDA Toolkit 4.0                        |
| 1.3.0             |                                         |
| 1.2.1             |                                         |
| 1.2.0             |                                         |
| 1.1.1             |                                         |
| 1.1.0             |                                         |
| 1.0.0             |                                         |

Adding Thrust To A CMake Project
--------------------------------

Since Thrust is a header library, there is no need to build or install Thrust
to use it. The `thrust` directory contains a complete, ready-to-use Thrust
package upon checkout.

We provide CMake configuration files that make it easy to include Thrust
from other CMake projects. See the [CMake README](thrust/cmake/README.md)
for details.

Development Process
-------------------

Thrust uses the [CMake build system](https://cmake.org/) to build unit tests,
examples, and header tests. To build Thrust as a developer, the following
recipe should be followed:

```
# Clone Thrust and CUB repos recursively:
git clone --recursive https://github.com/thrust/thrust.git
cd thrust

# Create build directory:
mkdir build
cd build

# Configure -- use one of the following:
cmake ..   # Command line interface.
ccmake ..  # ncurses GUI (Linux only)
cmake-gui  # Graphical UI, set source/build directories in the app

# Build:
cmake --build . -j <num jobs>   # invokes make (or ninja, etc)

# Run tests and examples:
ctest
```

By default, a serial `CPP` host system, `CUDA` accelerated device system, and
C++14 standard are used. This can be changed in CMake. More information on
configuring your Thrust build and creating a pull request can be found in
[CONTRIBUTING.md](CONTRIBUTING.md).
