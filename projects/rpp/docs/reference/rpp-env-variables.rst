.. meta::
  :description: ROCm Performance Primitives (RPP) reference
  :keywords: RPP, ROCm, Performance Primitives, reference, environment variable, environment

********************************************************************
ROCm Performance Primitives environment variables
********************************************************************

This section describes the most important ROCm Performance Primitives (RPP) environment variables,
which are grouped by functionality.

Logging and debugging
=====================

The logging and debugging environment variables for RPP are collected in the following table.

.. list-table::
    :header-rows: 1
    :widths: 35,14,51

    * - **Environment variable**
      - **Default value**
      - **Value**

    * - | ``RPP_ENABLE_LOGGING``
        | Enables logging of the most important function calls.
      - ``0``
      - | ``0``, "no", "false", "disable", "disabled": Disable function call logging
        | ``1``, "yes", "true", "enable", "enabled": Enable function call logging


    * - | ``RPP_ENABLE_LOGGING_CMD``
        | Prints driver command lines into log for reproducing use cases.
      - ``0``
      - | ``0``, "no", "false", "disable", "disabled": Disable command logging
        | ``1``, "yes", "true", "enable", "enabled": Enable command logging


    * - | ``RPP_ENABLE_LOGGING_MPMT``
        | Prefixes log lines with process/thread identification for multi-process/multi-threaded debugging.
      - ``0``
      - | ``0``, "no", "false", "disable", "disabled": Disable prefixes
        | ``1``, "yes", "true", "enable", "enabled": Enable process/thread prefixes


    * - | ``RPP_LOG_LEVEL``
        | Controls the verbosity level of RPP logging output.
      - ``0``
      - | ``0``: Default (Warning level for release, Info level for debug builds)
        | ``1``: Quiet (no logging)
        | ``2``: Fatal errors only
        | ``3``: Error level
        | ``4``: Warning level
        | ``5``: Info level
        | ``6``: Info2 level (detailed info)
        | ``7``: Trace level (most verbose)

Compiler and assembly
=====================

The compiler and assembly environment variables for RPP are collected in the following table. These
variables are primarily intended for debugging and development purposes.

.. list-table::
    :header-rows: 1
    :widths: 50,50

    * - **Environment variable**
      - **Value**

    * - | ``RPP_EXPERIMENTAL_GCN_ASM_PATH``
        | Overrides the default path to the AMDGCN assembler for experimental features and development.
      - | String path to AMDGCN assembler executable
        | Used for custom assembler locations during development

    * - | ``RPP_DEBUG_AMD_ROCM_PRECOMPILED_BINARIES``
        | Controls debugging behavior for precompiled binaries in ROCm.
      - | ``0``, "no", "false", "disable", "disabled": Disable precompiled binary debugging
        | ``1``, "yes", "true", "enable", "enabled": Enable precompiled binary debugging


Cache control
=============

The cache control environment variables for RPP are collected in the following table. These variables
are primarily intended for debugging and development purposes.

.. list-table::
    :header-rows: 1
    :widths: 50,50

    * - **Environment variable**
      - **Value**

    * - | ``RPP_DISABLE_CACHE``
        | Disables binary caching functionality to force recompilation. Useful for development and debugging.
      - | ``0``, "no", "false", "disable", "disabled": Enable binary cache
        | ``1``, "yes", "true", "enable", "enabled": Disable binary cache
