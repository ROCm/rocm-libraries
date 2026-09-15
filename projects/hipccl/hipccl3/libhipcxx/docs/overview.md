<!-- MIT License
  -- 
  -- Modifications Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
  -- 
  -- Permission is hereby granted, free of charge, to any person obtaining a copy
  -- of this software and associated documentation files (the "Software"), to deal
  -- in the Software without restriction, including without limitation the rights
  -- to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  -- copies of the Software, and to permit persons to whom the Software is
  -- furnished to do so, subject to the following conditions:
  -- 
  -- The above copyright notice and this permission notice shall be included in all
  -- copies or substantial portions of the Software.
  -- 
  -- THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  -- IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  -- FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  -- AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  -- LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  -- OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  -- SOFTWARE.
  -->

# libhipcxx: The C++ Standard Library for Your Entire System

**libhipcxx, is a HIP enabled C++ Standard Library for your entire system.**
It provides a heterogeneous implementation of the C++ Standard Library that
can be used in and between CPU and GPU code.

Libhipcxx provides a compatible interface to the C++ Standard Library, so you only need to add `hip/std` to the start of your Standard Library
include and `cuda::` before any uses of `std::`:

```hip
#include <hip/hip_runtime.h>
#include <cuda/std/atomic>
cuda::std::atomic<int> x;
```

*IMPORTANT*: Please make sure to always include the header `hip/hip_runtime.h` *before* including any header from libhipcxx.

## `cuda::` and `cuda::std::`

When used with hipcc, libhipcxx facilities live in their own
  header hierarchy and namespace with the same structure as, but distinct from,
  the host compiler's Standard Library:

* `std::`/`<*>`: When using hipcc, this is your host compiler's Standard Library
      that works in `__host__` code only.
    With hipcc, libhip++ does not replace or interfere with host compiler's
      Standard Library.
* `cuda::std::`/`<cuda/std/*>`: Strictly conforming implementations of
      facilities from the Standard Library that work in `__host__ __device__`
      code.
* `cuda::`/`<cuda/*>`: Conforming extensions to the Standard Library that
      work in `__host__ __device__` code.
* `cuda::device`/`<cuda/device/*>`: Conforming extensions to the Standard
      Library that work only in `__device__` code.

```hip
// Standard C++, __host__ only.
#include <atomic>
std::atomic<int> x;

// HIP C++, __host__ __device__.
// Strictly conforming to the C++ Standard.
#include <cuda/std/atomic>
cuda::std::atomic<int> x;

// HIP C++, __host__ __device__.
// Conforming extensions to the C++ Standard.
#include <cuda/atomic>
cuda::atomic<int, cuda::thread_scope_block> x;
```

## libhipcxx is Heterogeneous

libhipcxx works across your entire codebase, both in and
  across host and device code.
libhipcxx is a C++ Standard Library for your entire system, not just your CPU or
  GPU.
Everything in `cuda::` is `__host__ __device__`.

libhipcxx facilities are designed to be passed between host and device code.
Unless otherwise noted, any libhipcxx object which is copyable or movable can be
  copied or moved between host and device code.

### `cuda::device::`

A small number of libhipcxx facilities only work in device code, usually because
  there is no sensible implementation in host code.

Such facilities live in `cuda::device::`.

## libhipcxx is Incremental

Today, the libhipcxx delivers a high-priority subset of the
  C++ Standard Library today, and each release increases the feature set.
But it is a subset; not everything is available today.

## Conformance

libhipcxx aims to be a conforming implementation of the
  C++ Standard, [ISO/IEC IS 14882], Clause 16 through 32.

# Requirements 
- CMake >=3.12
- ROCm with HIP >=6.2.0 
- AMD Architectures CDNA, CDNA2, CDNA3, RDNA2, RDNA3, RDNA4. See [ROCm docs](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html) for more details about supported architectures. (NVIDIA GPUs are currently not supported)
- Linux OS (Windows is currently not supported)

For running the integrated LIT unit tests:
- Python 3
- pip (for installing lit)
- lit 18.1.8+
- sccache
- ninja

# Build and Installation

It is assumed that the test dependencies have been installed (e.g., `pip3 install lit==18.1.8` for the lit dependency).
The following commands can be run from the root directory to configure libhipcxx, build it and run the unit tests:

1) Create build directory
`mkdir build && cd build`
2) Run CMake to configure LIT testing
`cmake -DCMAKE_INSTALL_PREFIX=<path to install directory> ..`
3) Compile all headers that are part of the library
`make`
4) Install the headers locally: `make install`

## Build with HIPRTC
If you would like to build libhipcxx with HIPRTC support, you can modify step 2 as follows:
   - Run CMake with HIPRTC support enabled:
     `cmake -DLIBHIPCXX_TEST_WITH_HIPRTC=ON -DCMAKE_INSTALL_PREFIX=<path-to-install-directory> ..`

# Optional: Creating Distribution Packages

Libhipcxx can be packaged using cpack:

1) Switch to the build directory
`cd build`
2) Run CPack to generate packaged archives of libhipcxx:
`cpack .`

Note: Per default, this will only generate TGZ, ZIP and DEB packages. You can generate other package formats, e.g., with:
```
cpack -G RPM .
```

# Running the Tests
To run the tests on host and device based on LIT, you can use
`make check-hipcxx`.

Alternatively, there is a helper script at `utils/amd/linux/perform_tests.bash` which can be used as follows:
1) Change directory to build directory: `cd build`
2) `bash ../utils/amd/linux/perform_tests.bash`

Please note the following regarding test behavior:
- If you pass the `LIBHIPCXX_TEST_WITH_HIPRTC` option to CMake, the tests will run with HIPRTC support enabled.
- If you do not pass this option, the tests will run without HIPRTC support.

For automated testing in continuous integration environments or when you need a predefined testing workflow, it is recommended to use the CI scripts provided in the `ci` directory. These scripts can run tests with both HIP and HIPRTC configurations.
## Using CI Scripts for Automated Testing
1) Change directory to ci directory: `cd ci`
2) If you want to run tests without HIPRTC: `bash ./test_libhipcxx.sh`
3) If you want to run tests with HIPRTC: `bash ./hiprtc_libhipcxx.sh`

# How to use libhipcxx in your CMake Project

Example `CMakeLists.txt`:
```
...
find_package(libhipcxx)
...
target_link_libraries(<your_target> PRIVATE libhipcxx::libhipcxx)
```
Make sure to set `CMAKE_PREFIX_PATH` when running CMake for your project, in case you installed libhipcxx in a non-default installation directory.

# Limitations/Unsupported Features/APIs
- Libhipcxx does not support for CUDA backend/NVIDIA hardware.
- Libhipcxx does not support the Windows OS.
- `cuda::std::chrono::system_clock::now()` does not return a UNIX timestamp, host system clock and device system clock are not synchronized and they may run at different clock rates.
- The following APIs from [libcudacxx] are *NOT* supported in libhipcxx:

| Group                   | API Header                 | Description                                             |
| ----------------------- | -------------------------  | ------------------------------------------------------- |
| Synchronization Library | `<cuda/std/latch>`       | Single-phase asynchronous thread-coordination mechanism |
| Synchronization Library | `<cuda/std/barrier>`      | Multi-phase asynchronous thread-coordination mechanism  | 
| Synchronization Library | `<cuda/std/semaphore>`    | Primitives for constraining concurrent access           |
| Extended Synchronization Library | `<cuda/latch>`    | System-wide `cuda::std::latch` single-phase asynchronous thread coordination mechanism.|
| Extended Synchronization Library | `<cuda/barrier>`    | System-wide `cuda::std::barrier` multi-phase asynchronous thread coordination mechanism.|
| Extended Synchronization Library | `<cuda/semaphore>`    | System-wide primitives for constraining concurrent access.|
| Extended Synchronization Library | `<cuda/pipeline>`    |  Coordination mechanisms to sequence asynchronous operations.|
| Extended Memory Access Properties Library  | `<cuda/annotated_ptr>`         | Memory access properties for pointers. |

# License

libhipcxx is an open source project. It is derived from [libcudacxx] 
and [LLVM's libc++]. The original [libcudacxx] and [LLVM's libc++] are distributed under the [Apache License v2.0 with LLVM Exceptions]. Any new files and modifications made to exisiting files by AMD are distributed under MIT.

[libcudacxx]: https://github.com/nvidia/libcudacxx
[LLVM's libc++]: https://libcxx.llvm.org
[Apache License v2.0 with LLVM Exceptions]: https://llvm.org/LICENSE.txt
