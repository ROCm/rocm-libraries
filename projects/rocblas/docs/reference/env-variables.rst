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

    * - | ``ROCBLAS_DEVICE_MEMORY_SIZE``
        | Sets how much memory to preallocate.
      - Unset by default.
      - :ref:`Preallocate memory <memory-alloc>`
      - | **0 or unset**: Lets rocBLAS manage the device memory.
        | **Bigger than 0**: Sets the default handle device memory size to the specified size (in bytes).

    * - | ``ROCBLAS_EMULATE_DOUBLE_PRECISION``
        | ``ROCBLAS_EMULATE_SINGLE_PRECISION``
        | Requests hipBLASLt fixed-point emulation for ``f64_r`` or ``f32_r`` GEMM that runs on the hipBLASLt backend (rocBLAS built against hipBLASLt 1.5.1 or later and hipblas-common 1.5.0 or later). In builds with Tensile, this includes ``gemm`` and the GEMMs used inside other Level 3 functions; in builds without Tensile, ``gemm`` and most Level 3 functions use rocBLAS's own GEMM kernels, and mainly the ``gemm_ex`` functions run on the hipBLASLt backend. Read when a handle is created.
      - Unset by default.
      - :ref:`Control the GEMM backend <rocblas-tensile-hipblaslt>`
      - | **1**: rocBLAS passes the emulated compute type (``HIPBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT`` or ``HIPBLAS_COMPUTE_32F_EMULATED_FIXEDPOINT``) to hipBLASLt.
        | **Any other value or unset**: rocBLAS passes the usual compute type.
        | Applies only to calls that run on the hipBLASLt backend, are non-batched or strided-batched with ``batch_count`` = 1 (including strided-batched calls that rocBLAS splits into single GEMMs because a batch stride reaches 2^32 elements), and give no explicit hipBLASLt solution index; with ``rocblas_xf32_xdl_math_op``, ``f32_r`` keeps its xf32 compute type. hipBLASLt still decides each call (see ``HIPBLAS_EMULATION_STRATEGY`` in the hipBLASLt documentation) and runs native GEMM where emulation does not apply, including on devices it has no emulation data for. hipBLASLt's own ``HIPBLAS_EMULATE_DOUBLE_PRECISION`` and ``HIPBLAS_EMULATE_SINGLE_PRECISION`` also enable emulation for rocBLAS calls, and setting them to 0 does not turn this variable off. To choose how hipBLASLt decides between emulation and native GEMM for a handle, use ``rocblas_set_emulation_strategy``; its non-default strategies override ``HIPBLAS_EMULATION_STRATEGY`` for that handle.
        | Level 3 functions that run many smaller GEMMs, for example ``trmm``, can be much slower with emulation than without, especially with ``HIPBLAS_EMULATION_STRATEGY=eager``, under which hipBLASLt emulates calls even where emulation is slower than native GEMM. Functions that already hold rocBLAS device memory when they call GEMM, for example ``trsm``, cannot grow the pool (see ``ROCBLAS_EMULATION_WORKSPACE_MAX_BYTES``): with the default pool, hipBLASLt runs their GEMMs natively. With ``HIPBLAS_EMULATION_STRATEGY=eager`` and a larger pool set through ``ROCBLAS_DEVICE_MEMORY_SIZE``, they emulate and can also be much slower.

    * - | ``ROCBLAS_EMULATION_WORKSPACE_MAX_BYTES``
        | Limits how far rocBLAS grows its device memory pool for hipBLASLt fixed-point emulation (rocBLAS built against hipBLASLt 1.5.1 or later). Read once, when it is first needed.
      - Unset by default.
      - :ref:`Device memory allocation <memory-alloc>`
      - | **Unset**: Limits each growth request to a quarter of the headroom, clamped to between 256 MiB and 8 GiB. The headroom is free device memory minus rocBLAS's default pool size and a 64 MiB margin.
        | **0**, or a value that does not start with digits: Never grows the pool for emulation.
        | **Bigger than 0**: Limits each growth request to the specified size (in bytes).
        | The value is read as an unsigned integer with ``strtoull``: decimal, hexadecimal with ``0x``, or octal with a leading ``0``, after optional whitespace and an optional sign. Text after the number is ignored. A negative value, or one too large to represent, becomes a very large number, so only the headroom limits growth.
        | In every case the request is never more than the headroom, and the pool holds rocBLAS's default pool size on top of it. The grown pool is kept until the handle is destroyed or its device memory size is reset. When the pool cannot grow, or growth reaches the limit, rocBLAS prints a warning, once per process for each case; emulation then runs slower with less workspace than it asks for, and below its minimum hipBLASLt runs native GEMM instead.

    * - | ``ROCBLAS_DEFAULT_ATOMICS_MODE``
        | Sets the default atomics mode during the creation of ``rocblas_handle``.
      - Unset by default.
      - :ref:`Atomic Operations`
      - | **0**: Sets the default to :any:`rocblas_atomics_not_allowed`
        | **1**: Sets the default to :any:`rocblas_atomics_allowed`

    * - | ``ROCBLAS_STREAM_ORDER_ALLOC``
        | Allows memory allocation and deallocation to be stream ordered.
      - 0
      - :ref:`Device memory allocation <memory-alloc>`
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

.. include:: ../data/reference/logging-env.rst
