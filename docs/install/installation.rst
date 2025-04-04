.. meta::
   :description: C++ library for accelerating mixed precision matrix multiply-accumulate operations
    leveraging specialized GPU matrix cores on AMD's latest discrete GPUs
   :keywords: rocWMMA, ROCm, library, API, tool

.. _installation:

==============
Installation
==============

This document provides instructions for installing and configuring the rocWMMA library.
The quickest way to install is using prebuilt packages that are released with ROCm.
Alternatively, there are instructions to build from source.

Available ROCm packages are:

* rocwmma-dev (sources files for development).
* rocwmma-samples (sample executables).
* rocwmma-samples-dbgsym (sample executables with debug symbols).
* rocwmma-tests (test executables).
* rocwmma-tests-dbgsym (test executables with debug symbols).
* rocwmma-clients (samples, tests and benchmarks executables).

-------------
Prerequisites
-------------

* A ROCm 6.0 enabled platform. More information `here <https://github.com/ROCm/ROCm>`_.
* rocBLAS 4.0.0 for ROCm 6.0, if rocWMMA is configured to validate with rocBLAS (see below).

-----------------------------
Installing pre-built packages
-----------------------------

To install rocWMMA on Ubuntu or Debian, use:

::

   sudo apt-get update
   sudo apt-get install rocwmma-dev rocwmma-samples rocwmma-tests

To install rocWMMA on CentOS, use:

::

    sudo yum update
    sudo yum install rocwmma-dev rocwmma-samples rocwmma-tests

To install rocWMMA on SLES, use:

::

    sudo dnf upgrade
    sudo dnf install rocwmma-dev rocwmma-samples rocwmma-tests

Once installed, rocWMMA can be used just like any other library with a C++ API.

.. note::
 The prebuilt package supports the following targets: ``gfx908``; ``gfx90a``; ``gfx942``; ``gfx950``; ``gfx1100``; ``gfx1101``; ``gfx1102``; ``gfx1200``; ``gfx1201``


Once rocWMMA is installed, you can see the ``rocwmma.hpp`` header file in the ``/opt/rocm/include/rocwmma`` directory.
You must include only ``rocwmma.hpp``, ``rocwmma_coop.hpp`` and ``rocwmma_transforms.hpp`` in the user code to make calls into rocWMMA.
Don't directly include other rocWMMA files that are found in ``/opt/rocm/include/internal``.

-------------------------------
Building and installing rocWMMA
-------------------------------

For most users building from source is not necessary, as rocWMMA can be used after installing the pre-built
packages as described above. If still desired, here are the instructions to build rocWMMA from source:

System requirements
^^^^^^^^^^^^^^^^^^^
As a general rule, a minimum of 8GB of system memory is required for a full rocWMMA build. This value can be lower if rocWMMA is built without tests.
This value may also increase in the future as more features are added to rocWMMA.


GPU support
^^^^^^^^^^^
AMD CDNA class GPU featuring matrix core support: `gfx908`, `gfx90a`, `gfx942`, `gfx950` as `gfx9`

.. note::
    Double precision FP64 datatype support requires gfx90a, gfx942 or gfx950.

    F8 and BF8 datatype support requires gfx942 or gfx950

Or

AMD RDNA3 class GPU featuring AI acceleration support: `gfx1100`, `gfx1101`, `gfx1102` as `gfx11`; `gfx1200`, `gfx1201` as `gfx12`.

.. note::
    F8 and BF8 datatype support requires gfx1200 or gfx1201

Dependencies
^^^^^^^^^^^^
rocWMMA is designed to have minimal external dependencies such that it is light-weight and portable.

* Minimum ROCm version support is 6.4.
* Minimum cmake version support is 3.14.
* Minimum ROCm-cmake version support is 0.8.0.
* Minimum rocBLAS version support is rocBLAS 4.0.0 for ROCm 6.0* (or ROCm packages rocblas and rocblas-dev).
* Minimum HIP runtime version support is 4.3.0 (or ROCm package ROCm hip-runtime-amd).
* Minimum LLVM OpenMP runtime dev package version support is 10.0 (available as ROCm package rocm-llvm-dev).

.. note::
    \* = if using rocBLAS for validation.

    It is best to use available ROCm packages from the same release where applicable.

Download rocWMMA
^^^^^^^^^^^^^^^^^

The rocWMMA source code is available at the `rocWMMA github page <https://github.com/ROCm/rocWMMA>`_. rocWMMA has a minimum ROCm support version 6.0.
To check the ROCm version on an Ubuntu system, use:

::

    apt show rocm-libs -a

On Centos, use:

::

    yum info rocm-libs

The ROCm version has major, minor, and patch fields, possibly followed by a build specific identifier. For example, a ROCm version 6.0.0.40000-23 corresponds to major = 6, minor = 0, patch = 0, and build identifier 40000-23.
There are GitHub branches at the rocWMMA site with names ``rocm-major.minor.x`` where major and minor are the same as in the ROCm version. To download rocWMMA on ROCm version 6.0.0.40000-23, use:

::

   git clone -b release/rocm-rel-x.y https://github.com/ROCmSoftwarePlatform/rocWMMA.git
   cd rocWMMA

Replace ``x.y`` in the above command with the version of ROCm installed on your machine. For example, if you have ROCm 6.0 installed, then replace release/rocm-rel-x.y with release/rocm-rel-6.0.

Build Documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To build documentation locally as a pdf, run:

.. code-block:: bash

    cd docs

    sudo apt-get update
    sudo apt-get install doxygen
    sudo apt-get install texlive-latex-base texlive-latex-extra

    pip3 install -r sphinx/requirements.txt

    python3 -m sphinx -T -E -b latex -d _build/doctrees -D language=en . _build/latex

    cd _build/latex

    pdflatex rocwmma.tex

Running the above commands generates ``rocwmma.pdf``.

To build documentation locally as html, run:

.. code-block:: bash

    cd docs

    pip3 install -r sphinx/requirements.txt

    python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html

The HTML documentation can be viewed in your browser by opening the ``docs/_build/html/index.html`` result.

Build Configuration
^^^^^^^^^^^^^^^^^^^^

You can choose to build any of the following:

* library only
* library and samples
* library and tests (validation and / or benchmarks)
* library, samples, tests, and (optionally) assembly

Since rocWMMA is a header library, you only need the header includes for calling rocWMMA from your code.
The client contains the test samples and benchmark code.

Below are the project options available to build rocWMMA library with or without clients.

.. list-table::

    *   -   **Option**
        -   **Description**
        -   **Default Value**
    *   -   GPU_TARGETS
        -   Build code for specific GPU target(s)
        -   ``gfx908``; ``gfx90a``; ``gfx942``; ``gfx950``; ``gfx1100``; ``gfx1101``; ``gfx1102``; ``gfx1200``; ``gfx1201``
    *   -   ROCWMMA_BUILD_TESTS
        -   Build Tests
        -   ON
    *   -   ROCWMMA_BUILD_SAMPLES
        -   Build Samples
        -   ON
    *   -   ROCWMMA_BUILD_ASSEMBLY
        -   Generate assembly files
        -   OFF
    *   -   ROCWMMA_BUILD_VALIDATION_TESTS
        -   Build validation tests
        -   ON (requires ROCWMMA_BUILD_TESTS=ON)
    *   -   ROCWMMA_BUILD_BENCHMARK_TESTS
        -   Build benchmark tests
        -   OFF (requires ROCWMMA_BUILD_TESTS=ON)
    *   -   ROCWMMA_BUILD_EXTENDED_TESTS
        -   Build extended testing coverage
        -   OFF (requires ROCWMMA_BUILD_TESTS=ON)
    *   -   ROCWMMA_VALIDATE_WITH_ROCBLAS
        -   Use rocBLAS for validation tests
        -   ON (requires ROCWMMA_BUILD_VALIDATION_TESTS=ON)
    *   -   ROCWMMA_BENCHMARK_WITH_ROCBLAS
        -   Include rocBLAS benchmarking data
        -   OFF (requires ROCWMMA_BUILD_BENCHMARK_TESTS=ON)
    *   -   ROCWMMA_USE_SYSTEM_GOOGLETEST
        -   Use system Google Test library instead of downloading and building it
        -   OFF (requires ROCWMMA_BUILD_TESTS=ON)

Build library
^^^^^^^^^^^^^^^^^^

By default, the project is configured in Release mode.

To build the library alone, run:

.. code-block:: bash

    CC=/opt/rocm/bin/amdclang CXX=/opt/rocm/bin/amdclang++ cmake -B <build_dir> . -DROCWMMA_BUILD_TESTS=OFF -DROCWMMA_BUILD_SAMPLES=OFF

Here are some other example project configurations:

.. tabularcolumns::
   |\X{1}{4}|\X{3}{4}|

+-----------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------+
|           Configuration           |                                                                          Command                                                                               |
+===================================+================================================================================================================================================================+
|               Basic               |                                      :code:`CC=/opt/rocm/bin/amdclang CXX=/opt/rocm/bin/amdclang++ cmake -B <build_dir>`                                       |
+-----------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------+
|         Targeting gfx908          |                      :code:`CC=/opt/rocm/bin/amdclang CXX=/opt/rocm/bin/amdclang++ cmake -B <build_dir> . -DGPU_TARGETS=gfx908:xnack-`                         |
+-----------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------+
|            Debug build            |                         :code:`CC=/opt/rocm/bin/amdclang CXX=/opt/rocm/bin/amdclang++ cmake -B <build_dir> . -DCMAKE_BUILD_TYPE=Debug`                         |
+-----------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Build without rocBLAS(default on) | :code:`CC=/opt/rocm/bin/amdclang CXX=/opt/rocm/bin/amdclang++ cmake -B <build_dir> . -DROCWMMA_VALIDATE_WITH_ROCBLAS=OFF -DROCWMMA_BENCHMARK_WITH_ROCBLAS=OFF` |
+-----------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------+

After configuration, build using:

.. code-block:: bash

    cmake --build <build_dir> -- -j<nproc>

.. note::
    We recommend using a minimum of 16 threads to build rocWMMA with any tests (-j16).

Build library and samples
^^^^^^^^^^^^^^^^^^^^^^^^^^^

To build library and samples, run:

.. code-block:: bash

    CC=/opt/rocm/bin/amdclang CXX=/opt/rocm/bin/amdclang++ cmake -B <build_dir> . -DROCWMMA_BUILD_TESTS=OFF -DROCWMMA_BUILD_SAMPLES=ON

After configuration, build using:

.. code-block:: bash

    cmake --build <build_dir> -- -j<nproc>

The samples folder in ``<build_dir>`` contains executables as given in the table below.

================ ==============================================================================================================================
Executable Name  Description
================ ==============================================================================================================================
``simple_sgemm``      A simple GEMM operation [D = alpha * (A x B) + beta * C] using rocWMMA API for single-precision floating point types
``simple_dgemm``      A simple GEMM operation [D = alpha * (A x B) + beta * C] using rocWMMA API for double-precision floating point types
``simple_hgemm``      A simple GEMM operation [D = alpha * (A x B) + beta * C] using rocWMMA API for half-precision floating point types

``perf_sgemm``        An optimized GEMM operation [D = alpha * (A x B) + beta * C] using rocWMMA API for single-precision floating point types
``perf_dgemm``        An optimized GEMM operation [D = alpha * (A x B) + beta * C] using rocWMMA API for double-precision floating point types
``perf_hgemm``        An optimized GEMM operation [D = alpha * (A x B) + beta * C] using rocWMMA API for half-precision floating point types

``simple_sgemv``      A simple GEMV operation [y = alpha * (A) * x + beta * y] using rocWMMA API for single-precision floating point types
``simple_dgemv``      A simple GEMV operation [y = alpha * (A) * x + beta * y] using rocWMMA API for double-precision floating point types

``simple-dlrm``       A simple DLRM operation using rocWMMA API

``hipRTC_gemm``       A simple GEMM operation [D = alpha * (A x B) + beta * C] demonstrating runtime compilation (hipRTC) compatibility
================ ==============================================================================================================================


Build library and tests
^^^^^^^^^^^^^^^^^^^^^^^^^
rocWMMA provides the following test suites:

- DLRM tests: Cover the dot product interactions between embeddings used in Deep Learning Recommendation Model (DLRM) implemented with rocWMMA.
- GEMM tests: Cover block-wise Generalized Matrix Multiplication (GEMM) implemented with rocWMMA.
- Unit tests: Cover various aspects of rocWMMA API and internal functionality.

rocWMMA can build both validation and benchmark tests. Validation tests verify the rocWMMA implementations against a reference model, giving a PASS
or FAIL result. Benchmark tests invoke the tests multiple times, returning average compute throughput in tera-flop/sec (TFlops) and may guage efficiency
as a percentage of expected peak performance. The library uses CPU or rocBLAS methods for validation (when available) and benchmark
comparisons based on the provided selected project configurations. By default, the project is linked against rocBLAS for validating results more efficiently.

To build library and tests, run:

.. code-block:: bash

    CC=/opt/rocm/bin/amdclang CXX=/opt/rocm/bin/amdclang++ cmake -B <build_dir> . -DROCWMMA_BUILD_TESTS=ON

After configuration, build using:

.. code-block:: bash

    cmake --build <build_dir> -- -j<nproc>

The tests in ``<build_dir>`` contain executables as given in the table below.

============================================= ===================================================================================================================================================
Executable Name                               Description
============================================= ===================================================================================================================================================
``dlrm/dlrm_dot_test-*``                        A DLRM implementation using rocWMMA API
``dlrm/dlrm_dot_lds_test-*``                    A DLRM implementation using rocWMMA API with LDS shared memory
``gemm/gemm_PGR0_LB0_MP0_SB_NC-*``              A simple GEMM operation [D = alpha * (A x B) + beta * C] using rocWMMA API
``gemm/gemm_PGR0_LB0_MP0_MB_NC-*``              A modified GEMM operation where each wave targets a sub-grid of output blocks using rocWMMA API
``gemm/gemm_PGR1_LB2_MP0_MB_CP_BLK-*``          A modified GEMM operation where each wave targets a sub-grid of output blocks using LDS memory, rocWMMA API, and block-level collaboration
``gemm/gemm_PGR1_LB2_MP0_MB_CP_WV-*``           A modified GEMM operation where each wave targets a sub-grid of output blocks using LDS memory, rocWMMA API, and wave-level collaboration
``gemm/gemm_PGR1_LB2_MP0_MB_CP_WG-*``           A modified GEMM operation where each wave targets a sub-grid of output blocks using LDS memory, rocWMMA API, and workgroup-level collaboration
``gemm/gemm_PGR0_LB0_MP0_SB_NC_ad_hoc-*``       An adhoc version of ``gemm_PGR0_LB0_MP0_SB_NC-*``
``gemm/gemm_PGR0_LB0_MP0_MB_NC_ad_hoc-*``       An adhoc version of ``gemm_PGR0_LB0_MP0_MB_NC-*``
``gemm/gemm_PGR1_LB2_MP0_MB_CP_BLK_ad_hoc-*``   An adhoc version of ``gemm_PGR1_LB2_MP0_MB_CP_BLK-*``
``gemm/gemm_PGR1_LB2_MP0_MB_CP_WV_ad_hoc-*``    An adhoc version of ``gemm_PGR1_LB2_MP0_MB_CP_WV-*``
``gemm/gemm_PGR1_LB2_MP0_MB_CP_WG_ad_hoc-*``    An adhoc version of ``gemm_PGR1_LB2_MP0_MB_CP_WG-*``
``unit/contamination_test``                     Tests against contamination of pristine data for loads and stores
``unit/cross_lane_ops_test``                    Tests cross-lane vector operations
``unit/fill_fragment_test``                     Tests fill_fragment API function
``unit/io_shape_test``                          Tests input and output shape meta data
``unit/io_traits_test``                         Tests input and output logistical meta data
``unit/layout_test``                            Tests accuracy of internal matrix layout patterns
``unit/layout_traits_test``                     Tests accuracy of internal matrix layout traits
``unit/load_store_matrix_sync_test``            Tests ``load_matrix_sync`` and ``store_matrix_sync`` API functions
``unit/load_store_matrix_coop_sync_test``       Tests ``load_matrix_coop_sync`` and ``store_matrix_coop_sync`` API functions
``unit/map_util_test``                          Tests mapping utilities used in rocWMMA implementations
``unit/pack_util_test``                         Tests vector packing utilities used in rocWMMA implementations
``unit/transforms_test``                        Tests transform utilities used in rocWMMA implementations
``unit/tuple_test``                             Tests additional transform utilities used in rocWMMA implementations
``unit/unpack_util_test``                       Tests vector un-packing utilities used in rocWMMA implementations
``unit/vector_iterator_test``                   Tests internal vector storage iteration implementation
``unit/vector_test``                            Tests internal vector storage implementation
``unit/vector_util_test``                       Tests internal vector manipulation utilities implementation
============================================= ===================================================================================================================================================

.. note::

    \*= validate: Executables that compare outputs for correctness against reference sources such as CPU or rocBLAS calculations.

    \*= bench: Executables that measure kernel execution speeds and may compare against those of rocBLAS references.

Build library, tests, and assembly
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To build the library and tests with assembly code generation, run:

.. code-block:: bash

    CC=/opt/rocm/bin/amdclang CXX=/opt/rocm/bin/amdclang++ cmake -B <build_dir> . -DROCWMMA_BUILD_ASSEMBLY=ON -DROCWMMA_BUILD_TESTS=ON

After configuration, build using:

.. code-block:: bash

    cmake --build <build_dir> -- -j<nproc>

.. note::
    The ``assembly`` folder within ``<build_dir>`` contains a hierarchy of assembly files generated the executables in the format ``test_executable_name.s``.
    These may be viewed from your favorite text editor.

Make targets list
^^^^^^^^^^^^^^^^^

When building rocWMMA during the ``make`` step, we can specify make targets instead of defaulting ``make all``. The following table highlights relationships between high level grouped targets and individual targets.

.. tabularcolumns::
   |\X{1}{4}|\X{3}{4}|

+-----------------------------------+------------------------------------------+
|           Group Target            |            Individual Targets            |
+===================================+==========================================+
|                                   | simple_sgemm                             |
|                                   +------------------------------------------+
| rocwmma_samples                   | simple_dgemm                             |
|                                   +------------------------------------------+
|                                   | simple_hgemm                             |
|                                   +------------------------------------------+
|                                   | perf_sgemm                               |
|                                   +------------------------------------------+
|                                   | perf_dgemm                               |
|                                   +------------------------------------------+
|                                   | perf_hgemm                               |
|                                   +------------------------------------------+
|                                   | simple_sgemv                             |
|                                   +------------------------------------------+
|                                   | simple_dgemv                             |
|                                   +------------------------------------------+
|                                   | simple_dlrm                              |
|                                   +------------------------------------------+
|                                   | hipRTC_gemm                              |
+-----------------------------------+------------------------------------------+
|                                   | gemm_PGR0_LB0_MP0_SB_NC-validate         |
|                                   +------------------------------------------+
|                                   | gemm_PGR0_LB0_MP0_SB_NC_ad_hoc-validate  |
|                                   +------------------------------------------+
|                                   | gemm_PGR0_LB0_MP0_MB_NC-validate         |
|                                   +------------------------------------------+
|                                   | gemm_PGR0_LB0_MP0_MB_NC_ad_hoc-validate  |
|                                   +------------------------------------------+
|     rocwmma_gemm_tests_validate   | gemm_PGR1_LB2_MP0_MB_CP_BLK-validate     |
|                                   +------------------------------------------+
|                                   | gemm_PGR1_LB2_MP0_MB_CP_WV-validate      |
|                                   +------------------------------------------+
|                                   | gemm_PGR1_LB2_MP0_MB_CP_WG-validate      |
|                                   +------------------------------------------+
|                                   | gemm_PGR1_LB2_MP0_MB_CP_ad_hoc-validate  |
+-----------------------------------+------------------------------------------+
|                                   | gemm_PGR0_LB0_MP0_SB_NC-bench            |
|                                   +------------------------------------------+
|                                   | gemm_PGR0_LB0_MP0_SB_NC_ad_hoc-bench     |
|                                   +------------------------------------------+
|                                   | gemm_PGR0_LB0_MP0_MB_NC-bench            |
|                                   +------------------------------------------+
|                                   | gemm_PGR0_LB0_MP0_MB_NC_ad_hoc-bench     |
|                                   +------------------------------------------+
|     rocwmma_gemm_tests_bench      | gemm_PGR1_LB2_MP0_MB_CP_BLK-bench        |
|                                   +------------------------------------------+
|                                   | gemm_PGR1_LB2_MP0_MB_CP_WV-bench         |
|                                   +------------------------------------------+
|                                   | gemm_PGR1_LB2_MP0_MB_CP_WG-bench         |
|                                   +------------------------------------------+
|                                   | gemm_PGR1_LB2_MP0_MB_CP_ad_hoc-bench     |
+-----------------------------------+------------------------------------------+
|                                   | dlrm_dot_test-validate                   |
|    rocwmma_dlrm_tests_validate    +------------------------------------------+
|                                   | dlrm_dot_lds_test-validate               |
+-----------------------------------+------------------------------------------+
|                                   | dlrm_dot_test-bench                      |
|    rocwmma_dlrm_tests_bench       +------------------------------------------+
|                                   | dlrm_dot_lds_test-bench                  |
+-----------------------------------+------------------------------------------+
|                                   | contamination_test                       |
|                                   +------------------------------------------+
|                                   | layout_test                              |
|                                   +------------------------------------------+
|                                   | layout_traits_test                       |
|                                   +------------------------------------------+
|                                   | map_util_test                            |
|                                   +------------------------------------------+
|                                   | load_store_matrix_sync_test              |
|                                   +------------------------------------------+
|     rocwmma_unit_tests            | load_store_matrix_coop_sync_test         |
|                                   +------------------------------------------+
|                                   | fill_fragment_test                       |
|                                   +------------------------------------------+
|                                   | vector_iterator_test                     |
|                                   +------------------------------------------+
|                                   | vector_test                              |
|                                   +------------------------------------------+
|                                   | vector_util_test                         |
|                                   +------------------------------------------+
|                                   | pack_util_test                           |
|                                   +------------------------------------------+
|                                   | io_traits_test                           |
|                                   +------------------------------------------+
|                                   | cross_lane_ops_test                      |
|                                   +------------------------------------------+
|                                   | io_shape_test                            |
|                                   +------------------------------------------+
|                                   | tuple_test                               |
|                                   +------------------------------------------+
|                                   | transforms_test                          |
|                                   +------------------------------------------+
|                                   | unpack_util_test                         |
+-----------------------------------+------------------------------------------+

Build performance
^^^^^^^^^^^^^^^^^

Depending on the resources available to the build machine and the build configuration selected, rocWMMA build times can be on the order of an hour or more. Here are some things you can do to reduce build times:

* Target a specific GPU (e.g., ``-D GPU_TARGETS=gfx908:xnack-``)
* Use lots of threads (e.g., ``-j32``)
* Select ``ROCWMMA_BUILD_ASSEMBLY=OFF``
* Select ``ROCWMMA_BUILD_DOCS=OFF``.
* Select ``ROCWMMA_BUILD_EXTENDED_TESTS=OFF``.
* Specify either ``ROCWMMA_BUILD_VALIDATION_TESTS`` or ``ROCWMMA_BUILD_BENCHMARK_TESTS`` as ON, and the other as OFF instead of doing both.
* During the ``make`` command, build a specific target, e.g: ``rocwmma_gemm_tests``.

Test runtime
^^^^^^^^^^^^^^^^^

Depending on the resources available to the machine running the selected tests, rocWMMA test runtimes can be on the order of an hour or more. Here are some things you can do to reduce run-times:

* CTest will invoke the entire test suite. You may invoke tests individually by name.
* Use GoogleTest filters, targeting specific test cases:

.. code-block:: bash

    <test_exe> --gtest_filter=\*name_filter\*

* Use ad hoc tests to focus on a specific set of parameters.
* Manually adjust the test cases coverage.

Test verbosity and output redirection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

GEMM tests support logging arguments that can be used to control verbosity and output redirection.

.. code-block:: bash

    <test_exe> --output_stream "output.csv" --omit 1

.. tabularcolumns::
   |C|C|C|

+------------------------+-------------------------------------+--------------------------------------------+
|Compact                 |Verbose                              |  Description                               |
+========================+=====================================+============================================+
| -os <output_file>.csv  | --output_stream <output_file>.csv   |  redirect GEMM testing output to CSV file  |
+------------------------+-------------------------------------+--------------------------------------------+
|                        |                                     |  code = 1: Omit gtest SKIPPED tests        |
|                        |                                     +--------------------------------------------+
|                        | --omit <code>                       |  code = 2: Omit gtest FAILED tests         |
|                        |                                     +--------------------------------------------+
|                        |                                     |  code = 4: Omit gtest PASSED tests         |
|                        |                                     +--------------------------------------------+
|                        |                                     |  code = 8: Omit all gtest output           |
|                        |                                     +--------------------------------------------+
|                        |                                     |  code = <N>: OR'd combination of 1, 2, 4   |
+------------------------+-------------------------------------+--------------------------------------------+
