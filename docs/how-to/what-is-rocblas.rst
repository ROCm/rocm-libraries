.. meta::
  :description: rocBLAS documentation and API reference library
  :keywords: rocBLAS, ROCm, API, Linear Algebra, documentation

.. _what-is-rocblas:

********************************************************************
What is rocBLAS?
********************************************************************

rocBLAS is the AMD library for Basic Linear Algebra Subprograms (BLAS) on the :doc:`ROCm platform <rocm:index>`.
It is implemented in the :doc:`HIP programming language <hip:index>` and optimized for AMD GPUs.

The aim of rocBLAS is to provide:

* Functionality similar to legacy BLAS, adapted to run on GPUs
* A high-performance robust implementation

rocBLAS is written in C++17 and HIP and uses the AMD ROCm runtime to run on GPU devices.

The rocBLAS API is a thin C99 API that uses the hourglass pattern. It contains:

* :ref:`level-1`, :ref:`level-2`, and :ref:`level-3` with batched and strided_batched versions
* Extensions to legacy BLAS, including functions for mixed precision
* Auxiliary functions
* Device memory functions

.. note::

   * The official rocBLAS API is the C99 API defined in ``rocblas.h``. Therefore, the use of any other
     public symbols is discouraged. Other C/C++ interfaces might not follow a deprecation model and
     could change without warning from one release to the next.
   * The rocBLAS array storage format is column major and one-based.
     This is to maintain compatibility with the legacy BLAS code, which is written in Fortran.
   * rocBLAS calls the AMD :doc:`Tensile <tensile:src/index>` and :doc:`hipBLASLt <hipblaslt:index>` libraries
     for Level-3 GEMMs (matrix matrix multiplication).
