.. meta::
  :description: rocSOLVER environment variables reference
  :keywords: rocSOLVER, ROCm, API, environment variables, environment, reference

.. _rocsolver-environment-variables:

********************************************************************
rocSOLVER environment variables
********************************************************************

This section describes the important rocSOLVER environment variables,
which are grouped by functionality.

Logging variables
================================================================================

The logging environment variables for rocSOLVER are collected in the following table.
For more information, see :doc:`Use multi-level logging <../howto/logging>`.

.. list-table::
    :header-rows: 1
    :widths: 70,30

    * - **Environment variable**
      - **Value**

    * - | ``ROCSOLVER_LAYER``
        | Specifies which logging types, if any, are activated.
      - | Bitwise OR of zero or more bit masks:
        | 0: No logging (if not set)
        | 1: Trace logging
        | 2: Bench logging
        | 4: Profile logging
        | 16: Kernel logging
        | 17: Trace logging with kernel calls
        | 20: Profile logging with kernel calls

    * - | ``ROCSOLVER_LEVELS``
        | Specifies the default maximum depth of nested function calls for trace and profile logging.
      - | Integer value representing maximum nesting depth.

    * - | ``ROCSOLVER_LOG_PATH``
        | Sets the full path for logging output when specific log path variables are not set.
      - | Full path name for log files.
        | If not set, output streams to standard error.

    * - | ``ROCSOLVER_LOG_TRACE_PATH``
        | Sets the full path name for trace logging output.
      - | Full path name for trace log files.
        | Falls back to ``ROCSOLVER_LOG_PATH`` if not set.

    * - | ``ROCSOLVER_LOG_BENCH_PATH``
        | Sets the full path name for bench logging output.
      - | Full path name for bench log files.
        | Falls back to ``ROCSOLVER_LOG_PATH`` if not set.

    * - | ``ROCSOLVER_LOG_PROFILE_PATH``
        | Sets the full path name for profile logging output.
      - | Full path name for profile log files.
        | Falls back to ``ROCSOLVER_LOG_PATH`` if not set.
