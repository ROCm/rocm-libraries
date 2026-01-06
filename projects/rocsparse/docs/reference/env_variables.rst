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

Debug variables
================================================================================

The debug environment variables for rocSPARSE are collected in the following table.
For more information, see :doc:`Debugging rocSPARSE functions <./debugging>`.

.. list-table::
    :header-rows: 1
    :widths: 35,14,51

    * - **Environment variable**
      - **Default value**
      - **Value**

    * - | ``ROCSPARSE_DEBUG``
        | Generates code traces for unsuccessful status returns.
        | The level of information depends on the ``ROCSPARSE_DEBUG_VERBOSE`` setting.
      - No debug
      - | **0**: Disable (no debug)
        | **1**: Enable
    * - | ``ROCSPARSE_DEBUG_ARGUMENTS``
        | Generates debug messages when errors occur during argument checking.
        | The level of information depends on the value of ``ROCSPARSE_DEBUG_ARGUMENTS_VERBOSE``.
      - No debug
      - | **0**: Disable
        | **1**: Enable
    * - | ``ROCSPARSE_DEBUG_ARGUMENTS_VERBOSE``
        | Generates verbose-level debug messages for argument checking.
      - Verbose level off for argument checking debug messages
      - | **0**: Disable
        | **1**: Enable
    * - | ``ROCSPARSE_DEBUG_KERNEL_LAUNCH``
        | Generates kernel launch debug messages.
        | Checks for HIP errors before and after every kernel launch.
      - No kernel launch debug
      - | **0**: Disable
        | **1**: Enable
    * - | ``ROCSPARSE_DEBUG_VERBOSE``
        | Generates verbose-level debug messages.
        | Displays a stack of code traces that shows where the code handled a unsuccessful status.
      - Verbose level off for debug messages
      - | **0**: Disable
        | **1**: Enable
    * - | ``ROCSPARSE_DEBUG_WARNINGS``
        | Enable debug warnings.
        | Prints specific debug warnings during execution.
      - No debug warnings
      - | **0**: Disable
        | **1**: Enable
