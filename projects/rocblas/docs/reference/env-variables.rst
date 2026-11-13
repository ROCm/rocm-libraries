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
        | Requests hipBLASLt fixed-point emulation of ``f64_r`` or ``f32_r`` GEMM (rocBLAS built with hipBLASLt fixed-point emulation support, hipBLASLt 1.5.1 or later). Read when a handle is created.
      - Unset by default.
      - :ref:`Control the GEMM backend <rocblas-tensile-hipblaslt>`
      - | **1**: Requests emulation, by passing the emulated compute type to hipBLASLt.
        | **Any other value or unset**: No request.
        | Applies to GEMM on the hipBLASLt backend, including the GEMMs inside other Level 3 functions, that is non-batched or strided-batched with ``batch_count`` = 1 and has no explicit solution index; ``f32_r`` with ``rocblas_xf32_xdl_math_op`` stays xf32. In builds without Tensile, mainly the ``gemm_ex`` functions use the hipBLASLt backend.
        | hipBLASLt still decides each call. With the default (performant) strategy it runs native GEMM where it predicts emulation to be slower, and on devices it has no emulation data for. To measure emulation, use the eager strategy: ``rocblas_set_emulation_strategy``, ``HIPBLAS_EMULATION_STRATEGY=eager``, or ``rocblas-bench --emulation 1 --emulation_strategy 2``. A handle's non-default strategy overrides ``HIPBLAS_EMULATION_STRATEGY``.
        | hipBLASLt's ``HIPBLAS_EMULATE_DOUBLE_PRECISION`` and ``HIPBLAS_EMULATE_SINGLE_PRECISION`` also request emulation; setting them to 0 does not turn this variable off.
        | Bench logging (``ROCBLAS_LAYER=2``) records a handle's non-default emulation strategy but not this variable; set it again, or add ``--emulation 1``, when replaying a logged ``rocblas-bench`` command. Kernel-select logging (``ROCBLAS_LAYER=0x10``) adds ``--emulation 1`` to every real GEMM line of a precision the handle requested; it records the request, not whether that call was emulated.
        | Level 3 functions that run many small GEMMs, such as ``trmm``, can be much slower with emulation. Functions that hold rocBLAS device memory when they call GEMM, such as ``trsm``, cannot grow the pool and run native GEMM unless ``ROCBLAS_DEVICE_MEMORY_SIZE`` is large enough.

    * - | ``ROCBLAS_EMULATION_WORKSPACE_MAX_BYTES``
        | Limits how far rocBLAS grows its device memory pool for hipBLASLt fixed-point emulation (same build requirements as ``ROCBLAS_EMULATE_DOUBLE_PRECISION``). Read once, when first needed.
      - Unset by default.
      - :ref:`Device memory allocation <memory-alloc>`
      - | **Unset**: The emulation workspace of each growth is limited to a quarter of the headroom (free device memory minus rocBLAS's default pool size and 64 MiB), clamped to between 256 MiB and 8 GiB.
        | **0**: Never grows the pool for emulation.
        | **Bigger than 0**: The emulation workspace of each growth is limited to this many bytes, written in decimal, ``0x`` hexadecimal or leading-``0`` octal. The grown pool is that workspace plus rocBLAS's default pool size.
        | **Any other value**, for example a negative number, a unit suffix, or a value out of range: rocBLAS prints a warning and uses the default limit. An empty value counts as unset.
        | Growth is never more than the headroom. The grown pool is kept until the handle is destroyed or its device memory size is reset. When the pool cannot grow, rocBLAS prints a warning naming the reason, once per process for each reason; when the limit is reached, it prints a warning once per process. Emulation then runs with less workspace, or hipBLASLt runs native GEMM below its minimum.

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
