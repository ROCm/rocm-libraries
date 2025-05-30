.. meta::
  :description: rocSPARSE environment variables reference
  :keywords: rocSPARSE, ROCm, API, environment variables, environment, reference

.. _rocsparse_environment_variables_:

********************************************************************
rocSPARSE environment variables
********************************************************************

This section describes the important rocSPARSE environment variables,
which are grouped by functionality.

Logging variables
================================================================================

The logging environment variables for rocSPARSE are collected in the following table.
For more information, see :ref:`rocsparse_logging`.

.. list-table::
    :header-rows: 1
    :widths: 35,14,51

    * - **Environment variable**
      - **Default value**
      - **Value**

    * - | ``ROCSPARSE_LAYER``
        | Bit mask that enables logging modes.
      - Unset by default.
      - | Bitwise OR of zero or more bit masks:
        | 0: No logging (if not set)
        | 1: Trace logging
        | 2: Bench logging
        | 3: Trace and bench logging
        | 4: Debug logging
        | 5: Trace and debug logging
        | 6: Bench and debug logging
        | 7: Trace, bench, and debug logging

    * - | ``ROCSPARSE_LOG_TRACE_PATH``
        | Specifies path and file name to capture trace logging output.
      - stderr output
      - | Full path name for trace log files.
        | If not set or file cannot be opened, output streams to stderr.

    * - | ``ROCSPARSE_LOG_BENCH_PATH``
        | Specifies path and file name to capture bench logging output.
      - stderr output
      - | Full path name for bench log files.
        | If not set or file cannot be opened, output streams to stderr.

    * - | ``ROCSPARSE_LOG_DEBUG_PATH``
        | Specifies path and file name to capture debug logging output.
      - stderr output
      - | Full path name for debug log files.
        | If not set or file cannot be opened, output streams to stderr.
