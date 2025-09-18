.. meta::
    :description: rocBLAS environment variables
    :keywords: AMD, rocBLAS, environment variables, environment

.. _env-variables:

********************************************************************
Environment variables
********************************************************************

The rocBLAS environment variables are collected in the following
tables.

.. list-table::
    :header-rows: 1
    :widths: 30, 15, 20, 35

    * - **Environment variable**
      - **Default value**
      - **Links**
      - **Value**

    * - | ``ROCBLAS_USE_HIPBLASLT``
        | Provides manual control over which GEMM backend is used.
      - Unset by default.
      - :ref:`Control the GEMM backend <rocblas-tensile-hipblaslt>`
      - | **Unset**: GEMM backend is automatically selected.
        | **0**: Tensile is always used as the GEMM backend.
        | **1**: hipBLASLt is preferred as the GEMM backend, but will fallback to Tensile on problems for which hipBLASLt does not provide a solution or when errors are encountered using the hipBLASLt backend.

    * - | ``ROCBLAS_USE_HIPBLASLT_BATCHED``
        | Manual control to selectively disable the use of hipBlasLt only for the batched GEMMs. ``ROCBLAS_USE_HIPBLASLT=0`` disables the ``ROCBLAS_USE_HIPBLASLT_BATCHED`` variable, because hipBlasLt would not be enabled.
      - 1
      - :ref:`Control the GEMM backend <rocblas-tensile-hipblaslt>`
      - | **Unset**: GEMM batched default backend.
        | **0**: Tensile is always used as the GEMM batched backend.
        | **1**: hipBLASLt will be used as the GEMM batched backend when applicable, but will fallback to Tensile on problems for which hipBLASLt does not provide a solution or when errors are encountered using the hipBLASLt backend.

    * - | ``ROCBLAS_DEVICE_MEMORY_SIZE``
        | Sets how much memory to preallocate.
      - Unset by default.
      - :ref:`rocblas_device_memory_size`
      - | **0 or unset**: Lets rocBLAS manage the device memory.
        | **Bigger than 0**: Sets the default handle device memory size to the specified size (in bytes).

    * - | ``ROCBLAS_DEFAULT_ATOMICS_MODE``
        | Sets the default atomics mode during the creation of ``rocblas_handle``.
      - Unset by default.
      - :ref:`Atomic Operations`
      - | **0**: Sets the default to :any:`rocblas_atomics_not_allowed`
        | **1**: Sets the default to :any:`rocblas_atomics_allowed`

    * - | ``ROCBLAS_STREAM_ORDER_ALLOC``
        | Allows memory allocation and deallocation to be stream ordered.
      - 0
      - :ref:`stream order alloc`
      - | **0**: Disable
        | **1**: Enable

    * - | ``ROCBLAS_BENCH_STREAM_SYNC``
        | Benchmark timing based on ``hipStreamSynchronize``, otherwise uses default ``hipEvent_t`` based timing.
      - 0
      - :ref:`rocblas_bench_stream_sync`
      - | **0**: Disable
        | **1**: Enable

Logging environment variables
--------------------------------------------------------------------------------

The logging environment variables in rocBLAS are collected in the following
table. For information on how to use these variables, see :ref:`logging`.

.. include:: ../data/reference/env-variables/logging-env.rst
