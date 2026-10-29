.. Shared fragment: FP32/FP64 fixed-point emulation environment variables.
   Included by docs/reference/env-variables.rst and docs/how-to/fp-emulation.rst.
   Keep this file free of section headings and labels so it nests under whatever
   heading the including page provides and can be included more than once.

The ``HIPBLAS_*`` variables below use the same names as their cuBLAS counterparts,
with the ``CUBLAS_`` prefix replaced by ``HIPBLAS_``. hipBLASLt ignores invalid
values and uses the default instead.

.. list-table::
    :header-rows: 1
    :widths: 70,30

    * - **Environment variable**
      - **Value**

    * - | ``HIPBLAS_EMULATE_DOUBLE_PRECISION``
        | Enables FP64 emulation.
        | CUDA equivalent: ``CUBLAS_EMULATE_DOUBLE_PRECISION``.
      - | 0 or unset: Disabled (default)
        | 1: Enabled

    * - | ``HIPBLAS_EMULATE_SINGLE_PRECISION``
        | Enables FP32 emulation.
        | CUDA equivalent: ``CUBLAS_EMULATE_SINGLE_PRECISION``.
      - | 0 or unset: Disabled (default)
        | 1: Enabled

    * - | ``HIPBLAS_EMULATION_STRATEGY``
        | Controls when an enabled emulation path is taken.
        | CUDA equivalent: ``CUBLAS_EMULATION_STRATEGY``.
      - | ``performant``: Emulate only when predicted to be faster than the
          native GEMM (default)
        | ``eager``: Always emulate, regardless of problem size

    * - | ``HIPBLAS_FIXEDPOINT_EMULATION_FP64_MANTISSA_BIT_COUNT``
        | Sets the Automatic Dynamic Precision target for FP64, in mantissa bits.
        | Lower values need fewer moduli and run faster.
        | Takes precedence over ``HIPBLAS_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT``.
      - | 1 to 52
        | Default: 52 (full FP64 precision)

    * - | ``HIPBLAS_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT``
        | Same as ``HIPBLAS_FIXEDPOINT_EMULATION_FP64_MANTISSA_BIT_COUNT``, and
          used only when that variable is unset. Applies to FP64 only.
        | CUDA equivalent: ``CUBLAS_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT``.
          In cuBLAS this selects a fixed mantissa bit count; in hipBLASLt it sets
          the Automatic Dynamic Precision target, so the number of moduli is still
          chosen from the input data.
      - | 1 to 52
        | Default: 52 (full FP64 precision)

    * - | ``HIPBLAS_FIXEDPOINT_EMULATION_FP32_MANTISSA_BIT_COUNT``
        | Sets the Automatic Dynamic Precision target for FP32, in mantissa bits.
        | Lower values need fewer moduli and run faster.
        | No CUDA equivalent: cuBLAS emulates FP32 with BF16x9, not fixed-point
          emulation.
      - | 1 to 23
        | Default: 23 (full FP32 precision)

    * - | ``HIPBLAS_EMULATION_SPECIAL_VALUES_SUPPORT_MASK``
        | Bitmask controlling Inf and NaN detection in the input matrices.
        | Setting it to 0 skips detection, which is faster but silently
          corrupts the output for inputs that contain Inf or NaN.
        | Ignored for a matmul that has an emulation descriptor attached. The
          descriptor's ``HIPBLASLT_EMULATION_DESC_SPECIAL_VALUES_SUPPORT``
          attribute applies instead.
        | CUDA equivalent: ``CUBLAS_EMULATION_SPECIAL_VALUES_SUPPORT_MASK``.
      - | Bit 0: Inf detection
        | Bit 1: NaN detection
        | Default: 0x3 (both)

    * - | ``HIPBLASLT_EMULATION_NUM_MODULI``
        | Forces a fixed moduli count, bypassing Automatic Dynamic Precision.
        | Bypasses the adaptive precision check, so accuracy is not assured.
        | Leave this unset unless you are investigating a specific problem.
        | hipBLASLt-specific; no CUDA equivalent.
      - | 2 to 20
        | Default: unset (Automatic Dynamic Precision)

    * - | ``HIPBLASLT_EMULATION_PROFILE``
        | Appends per-call emulation profiling rows, in CSV format, to the
          given file.
        | Synchronizes the stream after each emulated call, so use it for
          diagnosis only.
        | hipBLASLt-specific; no CUDA equivalent.
      - | Path to a CSV file
        | Default: unset (no profiling output)
