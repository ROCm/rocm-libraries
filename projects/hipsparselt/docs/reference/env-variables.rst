.. meta::
   :description: hipSPARSELt environment variables reference
   :keywords: hipSPARSELt, ROCm, API, environment variables, environment, reference

.. _env-variables:

******************************************
Environment variables
******************************************

This section describes the important hipSPARSELt environment variables,
which are grouped by functionality.

Logging and debugging
=====================

The logging and debugging environment variables for hipSPARSELt are collected in the following table.

.. list-table::
    :header-rows: 1
    :widths: 70,30

    * - **Environment variable**
      - **Value**

    * - | ``HIPSPARSELT_LOG_LEVEL``
        | Controls the verbosity level of logging output.
      - | ``0``: Off - logging is disabled (default)
        | ``1``: Error - only errors will be logged
        | ``2``: Trace - API calls that launch HIP kernels will log parameters
        | ``3``: Hints - suggestions that can potentially improve performance
        | ``4``: Info - general information about library execution
        | ``5``: API Trace - API calls will log parameters and information

    * - | ``HIPSPARSELT_LOG_MASK``
        | Controls which types of messages are logged using bitmask values.
      - | ``0``: Off
        | ``1``: Error
        | ``2``: Trace
        | ``4``: Hints
        | ``8``: Info
        | ``16``: API Trace
        | Values can be combined by adding multiple values together

    * - | ``HIPSPARSELT_LOG_FILE``
        | Specifies the file path for logging output.
      - | String path to log file
        | File name can contain ``%i``, which is replaced with the process ID
        | If not defined, log messages are printed to stdout

Performance monitoring
======================

The performance monitoring environment variables for hipSPARSELt are collected in the following table.

.. list-table::
    :header-rows: 1
    :widths: 35,14,51

    * - **Environment variable**
      - **Default value**
      - **Value**

    * - | ``HIPSPARSELT_ENABLE_MARKER``
        | Enables ROCTracer logging for performance analysis.
      - ``0``
      - | ``0``: Disable ROCTracer logging
        | ``1``: Enable ROCTracer logging
