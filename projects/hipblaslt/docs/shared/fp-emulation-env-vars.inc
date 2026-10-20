.. Shared fragment: FP32/FP64 fixed-point emulation environment variables.
   Included by docs/reference/env-variables.rst and docs/how-to/fp-emulation.rst.
   Keep this file free of section headings and labels so it nests under whatever
   heading the including page provides and can be included more than once.

.. list-table::
    :header-rows: 1
    :widths: 70,30

    * - **Environment variable**
      - **Value**

    * - | ``HIPBLASLT_EMULATE_DOUBLE_PRECISION``
        | Enables FP64 emulation.
      - | 0 or unset: Disabled (default)
        | 1: Enabled

    * - | ``HIPBLASLT_EMULATE_SINGLE_PRECISION``
        | Enables FP32 emulation.
      - | 0 or unset: Disabled (default)
        | 1: Enabled

    * - | ``HIPBLASLT_EMULATION_STRATEGY``
        | Controls when an enabled emulation path is taken.
      - | ``performant``: Emulate only when predicted to be faster than the
          native GEMM (default)
        | ``eager``: Always emulate, regardless of problem size

    * - | ``HIPBLASLT_EMULATION_FP64_MANTISSA_BIT_COUNT``
        | Sets the Automatic Dynamic Precision target for FP64, in mantissa bits.
        | Lower values need fewer moduli and run faster.
      - | 1 to 52
        | Default: 52 (full FP64 precision)

    * - | ``HIPBLASLT_EMULATION_FP32_MANTISSA_BIT_COUNT``
        | Sets the Automatic Dynamic Precision target for FP32, in mantissa bits.
        | Lower values need fewer moduli and run faster.
      - | 1 to 23
        | Default: 23 (full FP32 precision)

    * - | ``HIPBLASLT_EMULATION_NUM_MODULI``
        | Forces a fixed moduli count, bypassing Automatic Dynamic Precision.
        | Bypasses the adaptive precision check, so accuracy is not assured.
        | Leave this unset unless you are investigating a specific problem.
      - | 2 to 20
        | Default: unset (Automatic Dynamic Precision)

    * - | ``HIPBLASLT_EMULATION_SPECIAL_VALUES_SUPPORT_MASK``
        | Bitmask controlling Inf and NaN detection in the input matrices.
        | Setting it to 0 skips detection, which is faster but silently
          corrupts the output for inputs that contain Inf or NaN.
      - | Bit 0: Inf detection
        | Bit 1: NaN detection
        | Default: 0x3 (both)

    * - | ``HIPBLASLT_EMULATION_PROFILE``
        | Appends per-call emulation profiling rows, in CSV format, to the
          given file.
      - | Path to a CSV file
        | Default: unset (no profiling output)
