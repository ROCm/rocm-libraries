# hipfort: Fortran Interface For GPU Kernel Libraries

This repository contains the source and testing for hipfort.  
This is a FORTRAN interface library for accessing GPU Kernels.

## Documentation

> [!NOTE]
> The published hipfort documentation is available at [hipfort](https://rocm.docs.amd.com/projects/hipfort/en/latest/index.html) in an organized, easy-to-read format, with search and a table of contents. The documentation source files reside in the hipfort/docs folder of this repository. As with all ROCm projects, the documentation is open source. For more information, see [Contribute to ROCm documentation](https://rocm.docs.amd.com/en/latest/contribute/contributing.html).

## Known issues

* `hipSOLVER` interfaces will only work for AMD GPUs.
* We recommend `gfortran` version 7.5.0 or newer as we have observed problems with older versions.

## Build and test hipfort from source

Install `gfortran`, `git`, `cmake`, and HIP, if not yet installed.
Then build, install, and test hipfort from source with the commands below:

```shell
git clone https://github.com/ROCm/hipfort.git
cd hipfort
cmake -S. -Bbuild -DCMAKE_INSTALL_PREFIOX=/tmp/hipfort -DBUILD_TESTING=ON
make -C build
make -C build check
```

## Fortran interfaces

`hipfort` provides interfaces to the following HIP and ROCm libraries:

* **HIP:**   HIP runtime, hipBLAS, hipSPARSE, hipFFT, hipRAND, hipSOLVER
* **ROCm:** rocBLAS, rocSPARSE, rocFFT, rocRAND, rocSOLVER

While the HIP interfaces and libraries allow to write portable code for both AMD and CUDA devices, the ROCm ones 
can only be used with AMD devices.

The available interfaces depend on the Fortran compiler that is used to compile the `hipfort` modules and libraries.
As the interfaces make use of the `iso_c_binding` module, the minimum requirement is a Fortran compiler 
that supports the Fortran 2003 standard (`f2003`).
These interfaces typically require to pass `type(c_ptr)` variables and the number of bytes to memory
management (e.g. `hipMalloc`) and math library routines (e.g. `hipblasDGEMM`).

If your compiler understands the Fortran 2008 (`f2008`) code constructs that occur in `hipfort`'s source and test files, 
additional interfaces are compiled into the `hipfort` modules and libraries. 
These directly take Fortran (array) variables and the number of
elements instead of `type(c_ptr)` variables and the number of bytes, respectively. 
Therefore, they reduce the chance to introduce compile-time and runtime errors
into your code and makes it easier to read too.

> **NOTE**: If you plan to use the `f2008` interfaces, we recommend `gfortran` version `7.5.0` or newer
as we have observed problems with older versions.

### Example

While you could write the following using the `f2003` interfaces:

```Fortran
use iso_c_binding
use hipfort
integer     :: ierr        ! error code
real        :: a_h(5,6)    ! host array
type(c_ptr) :: a_d         ! device array pointer
!
ierr = hipMalloc(a_d,size(a_h)*4_c_size_t) ! real has 4 bytes
                                           ! append suffix '_c_size_t' to write '4' 
                                           ! as 'integer(c_size_t)'
ierr = hipMemcpy(a_d,c_loc(a_h),size(a_h)*4_c_size_t,hipMemcpyHostToDevice)
```

you could express the same with the `f2008` interfaces as follows:

```Fortran
use hipfort
integer     :: ierr        ! error code
real        :: a_h(5,6)    ! host array
real,pointer :: a_d(:,:)   ! device array pointer
!
ierr = hipMalloc(a_d,shape(a_h))      ! or hipMalloc(a_d,[5,6]) or hipMalloc(a_d,5,6) or hipMalloc(a_d,mold=a_h)
ierr = hipMemcpy(a_d,a_h,size(a_h),hipMemcpyHostToDevice)
```

The `f2008` interfaces also overload `hipMalloc` similar to the Fortran 2008 `ALLOCATE` intrinsic. 
So you could write the whole code as shown below:

```Fortran
integer     :: ierr        ! error code
real        :: a_h(5,6)    ! host array
real,pointer :: a_d(:,:)   ! device array pointer
!
ierr = hipMalloc(a_d,source=a_h)       ! take shape (incl. bounds) of a_h and perform a blocking copy to device
```

In addition to `source`, there is also `dsource` in case the source is a device array.

### Supported HIP and ROCm API

The current batch of HIPFORT interfaces is derived from ROCm 4.5.0.
The following tables list the supported API:

* [HIP](https://github.com/ROCmSoftwarePlatform/hipfort/blob/master/lib/hipfort/SUPPORTED_API_HIP.md)
* [hipBLAS](https://github.com/ROCmSoftwarePlatform/hipfort/blob/master/lib/hipfort/SUPPORTED_API_HIPBLAS.md)
* [hipFFT](https://github.com/ROCmSoftwarePlatform/hipfort/blob/master/lib/hipfort/SUPPORTED_API_HIPFFT.md)
* [hipRAND](https://github.com/ROCmSoftwarePlatform/hipfort/blob/master/lib/hipfort/SUPPORTED_API_HIPRAND.md)
* [hipSOLVER](https://github.com/ROCmSoftwarePlatform/hipfort/blob/master/lib/hipfort/SUPPORTED_API_HIPSOLVER.md)
* [hipSPARSE](https://github.com/ROCmSoftwarePlatform/hipfort/blob/master/lib/hipfort/SUPPORTED_API_HIPSPARSE.md)
* [rocBLAS](https://github.com/ROCmSoftwarePlatform/hipfort/blob/master/lib/hipfort/SUPPORTED_API_ROCBLAS.md)
* [rocFFT](https://github.com/ROCmSoftwarePlatform/hipfort/blob/master/lib/hipfort/SUPPORTED_API_ROCFFT.md)
* [rocRAND](https://github.com/ROCmSoftwarePlatform/hipfort/blob/master/lib/hipfort/SUPPORTED_API_ROCRAND.md)
* [rocSOLVER](https://github.com/ROCmSoftwarePlatform/hipfort/blob/master/lib/hipfort/SUPPORTED_API_ROCSOLVER.md)
* [rocSPARSE](https://github.com/ROCmSoftwarePlatform/hipfort/blob/master/lib/hipfort/SUPPORTED_API_ROCSPARSE.md)

You may further find it convenient to directly use the search function on
[HIPFORT's documentation page](https://rocm.docs.amd.com/projects/hipfort/en/latest/) to get information on the arguments of an interface

## hipfc wrapper compiler and Makefile.hipfort

The hipfc wrapper compiler is deprecated and will be removed in a future release. Users are
encouraged to call their Fortran or HIP compilers directly instead of relying on the hipfc wrapper.
The hipfort project provides exported CMake targets that can be used for linking to the appropriate
ROCm libraries.

hipfort currently ships the `hipfc` wrapper compiler and a `Makefile.hipfort` that can be included
in a project's build system. hipfc is located in the `bin/` subdirectory and Makefile.hipfort in
share/hipfort of the repository. While both can be configured via a number of environment variables,
`hipfc` also understands a greater number of command line options that you can print to the screen via
`hipfc -h`.

Among the environment variables, the most important are:

| Variable | Description | Default |
|---|---|---|
| `HIP_PLATFORM` | The platform to compile for (either 'amd' or 'nvidia') | `amd` |
| `ROCM_PATH` | Path to ROCm installation | `/opt/rocm` |
| `CUDA_PATH` | Path to CUDA installation | `/usr/local/cuda` | 
| `CMAKE_Fortran_COMPILER` | Fortran compiler to be used | `gfortran` | 

## Examples and tests

The examples, which simultaneously serve as tests, are located in the 
`f2003` and `f2008` subdirectories of the repo's `test/` folder.
Both test collections implement the same tests but require
that the used Fortran compiler supports at least the respective Fortran standard.
There are further subcategories per `hip*` or `roc*` library that is tested.

### Building a single test

> **NOTE**: Only the `hip*` tests can be compiled for CUDA devices. The `roc*` tests cannot.
> **NOTE**: The make targets append the linker flags for AMD devices to the `CFLAGS` variable per default.

To compile for AMD devices you can simply call `make` in the test directories.

If you want to compile for CUDA devices, you need to build as follows:

```shell
make CFLAGS="--offload-arch=sm_70 <libs>"
```

where you must substitute `<libs>` by `-lcublas`, `-lcusparse`, ... as needed.
Compilation typically boils down to calling `hipfc` as follows:

```shell
hipfc <CFLAGS> <test_name>.f03 -o <test_name>
```

The `vecadd` test is the exception as the additional HIP C++ source must be supplied too:

```shell
hipfc <CFLAGS> hip_implementation.cpp main.f03 -o main
```

### Building and running all tests

You can build and run the whole test collection from the `build/` folder (see [Build and test hipfort from source](#build-and-test-hipfort-from-source)) or
from the `test/` folder. The instructions are given below.

#### AMD devices

> **NOTE**: Running all tests as below requires that all ROCm math libraries can be found at `/opt/rocm`.
Specify a different ROCm location via the `ROCM_PATH` environment variable.
> **NOTE**: When using older ROCm versions, you might need to manually set the environment variable `HIP_PLATFORM` to `hcc`
before running the tests.

```shell
cd build/
make all-tests-run
```

Alternatively:

```shell
cd test/
make run_all
```

#### CUDA devices

> **NOTE**: Running all tests as below requires that CUDA can be found at `/usr/local/cuda`. Specify a different CUDA location via the `CUDA_PATH` environment variable
> or supply it to the `CFLAGS` variable by appending `-cuda-path <path_to_cuda>`.
> **NOTE**: Choose offload architecture value according to used device.

```shell
cd build/
make all-tests-run CFLAGS="--offload-arch=sm_70 -lcublas -lcusolver -lcufft"
```

Alternatively:

```shell
cd test/
make run_all CFLAGS="--offload-arch=sm_70 -lcublas -lcusolver -lcufft"
```

## Copyright, License, and Disclaimer

<A NAME="Copyright">

Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
[MITx11 License]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
