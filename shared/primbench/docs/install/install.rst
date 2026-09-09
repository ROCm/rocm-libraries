.. meta::
   :description: Installation instructions for Primbench, a single-header C++ benchmarking library for HIP and CUDA GPU workloads.
   :keywords: Primbench, install, HIP, CUDA, ROCm, GPU benchmarking, single-header, AMD


***********************
Install Primbench
***********************

Primbench is a single-header library. No build step, package installation, or linking against a Primbench binary is required.

Copy `primbench.hpp <https://github.com/ROCm/rocm-libraries/blob/develop/shared/primbench/primbench.hpp>`_ to your include directory or use the ``-I`` compiler option to point to the path to ``primbench.hpp``.

Include the header in your project and compile with ``hipcc``.

Prerequisites
=============

- ``hipcc``
- A C++17-capable compiler
- `AMD SMI <https://rocm.docs.amd.com/projects/amdsmi/en/latest/>`_ for GPU temperature monitoring

.. note::

   Because AMD SMI isn't available on Windows, GPU temperature monitoring is unavailable on Windows. Disable GPU monitoring on Windows by compiling with ``PRIMBENCH_NO_MONITORING``.

Compile options
===============

Primbench behavior is controlled by preprocessor macros passed at compile time.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Macro
     - Effect
   * - ``-DPRIMBENCH_NO_MONITORING``
     - Disables GPU temperature monitoring. Required on Windows. Use when AMD SMI isn't available.
   * - ``-DPRIMBENCH_NO_TEST``
     - Disables correctness-test execution. ``state.test()`` lambdas aren't called.
   * - ``-DPRIMBENCH_GPU_CACHE_SIZE=n``
     - Sets the size in bytes of the buffer used to evict GPU caches before each kernel launch. Defaults to ``256 * MiB``.
   * - ``-DBRANCH_NAME=\"name\"``
     - Embeds the Git branch name in ``context.general.branch_name`` of the JSON output.
   * - ``-DCOMMIT_HASH=\"hash\"``
     - Embeds the Git commit hash in ``context.general.commit_hash`` of the JSON output.

For full descriptions of each macro, see :doc:`Primbench API </reference/primbench-api>`.
