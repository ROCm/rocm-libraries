.. meta::
  :description: Install rocDecode
  :keywords: install, rocDecode, AMD, ROCm

*********************************
Installation
*********************************


Prerequisites
===============

* gfortran version 7.5.0 or newer

Building and testing hipFORT from source
==========================================

1. Ensure you have ``gfortran``, ``git``, ``cmake``, and HIP installed.
2. Build, install, and test hipFORT from source with the following commands:

   .. code-block:: 

      git clone https://github.com/ROCm/hipfort.git
      cd hipfort
      cmake -S. -Bbuild -DHIPFORT_INSTALL_DIR=/tmp/hipfort -DBUILD_TESTING=ON
      make -C build
      make -C build check

Fortran interfaces
===================

`hipfort` provides interfaces to the following HIP and ROCm libraries:

* **HIP:**   HIP runtime, hipBLAS, hipSPARSE, hipFFT, hipRAND, hipSOLVER

* **ROCm:** rocBLAS, rocSPARSE, rocFFT, rocRAND, rocSOLVER

.. note:: 

hipSOLVER interfaces will only work with AMD GPUs.

While the HIP interfaces and libraries allow the writing of portable code for both AMD and CUDA devices, ROCm can only be used with AMD devices.

The available interfaces depend on the Fortran compiler used to compile the `hipfort` modules and libraries. As the interfaces use the `iso_c_binding` module, the minimum requirement is a Fortran compiler that supports the Fortran 2003 standard (`f2003`). These interfaces typically require passing `type(c_ptr)` variables and the number of bytes to memory management. For example, `hipMalloc` and math library routines like `hipblasDGEMM`.

If your compiler understands the Fortran 2008 (`f2008`) code constructs in `hipfort`'s source and test files, additional interfaces are compiled into the `hipfort` modules and libraries. 
These directly take Fortran (array) variables, the number of elements instead of `type(c_ptr)` variables, and the number of bytes, respectively. Therefore, they reduce the chance of introducing compile-time and runtime errors into your code and make it easier to read, too.

.. note:: 

If you plan to use the `f2008` interfaces, we recommend `gfortran` version `7.5.0` or newer as we have observed problems with older versions.

Examples
--------

Use the following examples to express `f2003` interfaces:

**Example 1**


.. code-block:: 

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


**Example 2**

.. code-block::

        use hipfort
        integer     :: ierr        ! error code
        real        :: a_h(5,6)    ! host array
        real,pointer :: a_d(:,:)   ! device array pointer
        !
        ierr = hipMalloc(a_d,shape(a_h))      ! or hipMalloc(a_d,[5,6]) or hipMalloc(a_d,5,6) or hipMalloc(a_d,mold=a_h)
        ierr = hipMemcpy(a_d,a_h,size(a_h),hipMemcpyHostToDevice)



.. note::

The `f2008` interfaces also overload `hipMalloc`, similar to the Fortran 2008 `ALLOCATE` intrinsic. For example,

.. code-block:: 

        integer     :: ierr        ! error code
        real        :: a_h(5,6)    ! host array
        real,pointer :: a_d(:,:)   ! device array pointer
        !
        ierr = hipMalloc(a_d,source=a_h)       ! take shape (incl. bounds) of a_h and perform a blocking copy to device

In addition to `source`, there is also `dsource` in case the source is a device array.

Supported HIP and ROCm API
---------------------------

The current batch of HIPFORT interfaces is derived from ROCm 4.5.0. The following tables list the supported API:

* :doc:`HIP API <../doxygen/html/md_input_supported_api_hip>`
* :doc:`hipBLAS API <../doxygen/html/md_input_supported_api_hipblas>` 
* :doc:`hipFFT API <../doxygen/html/md_input_supported_api_hipfft>` 
* :doc:`hipRAND API <../doxygen/html/md_input_supported_api_hiprand>`
* :doc:`hipSOLVER API <../doxygen/html/md_input_supported_api_hipsolver>`
* :doc:`hipSPARSE API <../doxygen/html/md_input_supported_api_hipsparse>`
* :doc:`rocBLAS API <../doxygen/html/md_input_supported_api_rocblas>`
* :doc:`rocFFT API <../doxygen/html/md_input_supported_api_rocfft>`
* :doc:`rocRAND API <../doxygen/html/md_input_supported_api_rocrand>`
* :doc:`rocSOLVER API <../doxygen/html/md_input_supported_api_rocsolver>`
* :doc:`rocSPARSE API <../doxygen/html/md_input_supported_api_rocsparse>`


You may further find it convenient to directly use the ``Search`` function from the hipFORT TOC to get information on the arguments of an interface.

hipfc wrapper compiler and Makefile.hipfort
================================================

Aside from Fortran interfaces to the HIP and ROCm libraries, hipFORT ships the `hipfc` wrapper compiler and a `Makefile. fort` that can be included in a project's build system. hipfc is in the `bin/` subdirectory, and Makefile.hipfort is in share/hipfort of the repository. While both can be configured using a number of environment variables, ` hipfc` also understands a greater number of command line options that you can print to the screen using `hipfc -h.`

.. note::

   The hipfc wrapper compiler is deprecated and will be removed in a future release. Users are
   encouraged to call the Fortran or HIP compilers directly instead of relying on the hipfc wrapper.
   The hipfort component provides exported CMake targets that can be used to link to the appropriate
   ROCm libraries.

Among the environment variables, the most important are:

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Environment variable
     - Description
     - Default
   * - `HIP_PLATFORM` 
     - The platform to compile for (either 'amd' or 'nvidia')
     - `amd` 
   * - `ROCM_PATH`
     - Path to ROCm installation
     - `/opt/rocm`
   * - `CUDA_PATH` 
     - Path to CUDA installation
     - `/usr/local/cuda` 
   * - `HIPFORT_COMPILER` 
     -  Fortran compiler to be used
     - `gfortran`


Examples and tests
====================

The examples simultaneously serve as tests in the `f2003` and `f2008` subdirectories of the repository’s `test/` folder. Both test collections implement the same tests but require that the used Fortran compiler support at least the respective Fortran standard. Further subcategories per the `hip*` or `roc*` library are tested.

Building a single test
-----------------------

.. note:: 

        Only `hip*` tests can be compiled for CUDA devices. The `roc*` tests cannot be compiled for CUDA devices. 

.. note::

        The `make` targets append the linker flags for AMD devices to the `CFLAGS` variable per default.


To compile for AMD devices, you can simply call `make` in the test directories.

If you want to compile for CUDA devices, you need to build as follows:

.. code-block::

        make CFLAGS="--offload-arch=sm_70 <libs>"


where you must substitute `<libs>` by `-lcublas`, `-lcusparse`, ... as needed.

Compilation typically boils down to calling `hipfc` as follows:

.. code-block::

        hipfc <CFLAGS> <test_name>.f03 -o <test_name>


The `vecadd` test is the exception as the additional HIP C++ source must be supplied too:

.. code-block::

        hipfc <CFLAGS> hip_implementation.cpp main.f03 -o main


Building and running all tests
-------------------------------

You can build and run the whole test collection from the `build/` folder (see [Build and test hipfort from source](#build-and-test-hipfort-from-source)) or
from the `test/` folder. The instructions are given below.

AMD devices
############

Running all tests as below requires all ROCm math libraries to be found at `/opt/rocm`. You can specify a different ROCm location via the `ROCM_PATH` environment variable.

.. note::

When using older ROCm versions, you might need to manually set the environment variable `HIP_PLATFORM` to `hcc` before running the tests.

.. code-block::

        cd build/
        make all-tests-run


Alternatively:

.. code-block::

        cd test/
        make run_all


CUDA devices
##############

.. note::

Running the following tests requires that CUDA to be found at `/usr/local/cuda`. You can specify a different CUDA location via the `CUDA_PATH` environment variable or supply it to the `CFLAGS` variable by appending `-cuda-path <path_to_cuda>`. 

.. note::

Choose offload architecture value according to the used device.

.. code-block::

        cd build/
        make all-tests-run CFLAGS="--offload-arch=sm_70 -lcublas -lcusolver -lcufft"


Alternatively:

.. code-block::

        cd test/
        make run_all CFLAGS="--offload-arch=sm_70 -lcublas -lcusolver -lcufft"


