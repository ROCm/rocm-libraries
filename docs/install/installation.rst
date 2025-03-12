.. meta::
   :description: A high-performance HIP library for tensor primitives
   :keywords: hipTensor, ROCm, library, API, tool, installation

.. _installation:

===============================
Installation
===============================

The quickest way to install is using prebuilt packages that are released with ROCm.
Alternatively, there are instructions to build from source.

Available ROCm packages are:

* hiptensor (library + header files for development).
* hiptensor-dev (library development package).
* hiptensor-samples (sample executables).
* hiptensor-tests (test executables).
* hiptensor-clients (samples and test executables).

-------------
Prerequisites
-------------

* A ROCm 6.0 enabled platform. More information at `ROCm Github <https://github.com/ROCm/ROCm>`_.

-----------------------------
Installing pre-built packages
-----------------------------

To install hipTensor on Ubuntu or Debian, use:

::

   sudo apt-get update
   sudo apt-get install hiptensor hiptensor-dev hiptensor-samples hiptensor-tests

To install hipTensor on CentOS, use:

::

    sudo yum update
    sudo yum install hiptensor hiptensor-dev hiptensor-samples hiptensor-tests

To install hipTensor on SLES, use:

::

    sudo dnf upgrade
    sudo dnf install hiptensor hiptensor-dev hiptensor-samples hiptensor-tests

Once installed, hipTensor can be used just like any other library with a C++ API.

---------------------------------
Building and installing hipTensor
---------------------------------

For most users building from source is not necessary, as hipTensor can be used after installing the pre-built
packages as described above. If still desired, here are the instructions to build hipTensor from source:

System requirements
^^^^^^^^^^^^^^^^^^^
As a general rule, 8GB of system memory is required for a full hipTensor build. This value can be lower if hipTensor is built without tests. This value may also increase in the future as more functions are added.

GPU support
^^^^^^^^^^^
AMD CDNA class GPU featuring matrix core support: `gfx908`, `gfx90a`, `gfx940`, `gfx941`, `gfx942`, `gfx950` labeled as `gfx9`.

.. note::
    Double precision FP64 datatype support requires `gfx90a`, `gfx940`, `gfx941`, `gfx942` or `gfx950`.

Dependencies
^^^^^^^^^^^^
hipTensor is designed to have minimal external dependencies such that it is light-weight and portable.

.. <!-- spellcheck-disable -->

* Minimum ROCm version support is 6.4.
* Minimum cmake version support is 3.14.
* Minimum ROCm-cmake version support is 0.8.0.
* Minimum HIP runtime version support is 4.3.0 (or ROCm package ROCm hip-runtime-amd).
* Minimum LLVM dev package version support is 7.0 (available as ROCm package rocm-llvm-dev).
* Hiptensor leverages the amd-master branch of the composable kernel, a stable and widely adopted version for development.

.. <!-- spellcheck-enable -->

.. note::

    It is best to use available ROCm packages from the same release where applicable.

Download hipTensor
^^^^^^^^^^^^^^^^^^

The hipTensor source code is available on `hipTensor Github <https://github.com/ROCmSoftwarePlatform/hipTensor>`_. hipTensor has a minimum ROCm support version 6.4.
To check the ROCm Version on your system, use:

::

    apt show rocm-libs -a

For Centos use

::

    yum info rocm-libs

The ROCm version has major, minor, and patch fields, possibly followed by a build specific identifier. For example, a ROCm version 4.0.0.40000-23 corresponds to major = 4, minor = 0, patch = 0, and build identifier 40000-23.
There are GitHub branches at the hipTensor site with names ``rocm-major.minor.x`` where major and minor are the same as in the ROCm version. To download hipTensor on ROCm version 4.0.0.40000-23, use:

::

   git clone -b release/rocm-rel-x.y https://github.com/ROCmSoftwarePlatform/hipTensor.git
   cd hipTensor

Replace ``x.y`` in the above command with the version of ROCm installed on your machine. For example, if you have ROCm 5.0 installed, then replace release/rocm-rel-x.y with release/rocm-rel-5.0.

Build documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To build documentation locally, run:

.. code-block:: bash

    cd docs

    sudo apt-get update
    sudo apt-get install doxygen
    sudo apt-get install texlive-latex-base texlive-latex-extra

    pip3 install -r sphinx/requirements.txt

    python3 -m sphinx -T -E -b latex -d _build/doctrees -D language=en . _build/latex

    cd _build/latex

    pdflatex hiptensor.tex

Running the above commands generates ``hiptensor.pdf``. Alternatively, the latest docs build can be found at `hipTensor docs <https://rocm.docs.amd.com/projects/hipTensor/en/latest/index.html>`_.

Build configuration
^^^^^^^^^^^^^^^^^^^^^

You can choose to build any of the following:

* library only
* library and samples
* library and tests
* library, samples and tests

You only need the hipTensor library for calling and linking to hipTensor API from your code.
The clients contain the tests and sample codes.

Below are the project options available to build hipTensor library with or without clients.

.. list-table::

    *   -   **Option**
        -   **Description**
        -   **Default Value**
    *   -   GPU_TARGETS
        -   Build code for specific GPU target(s)
        -   ``gfx908``; ``gfx90a``; ``gfx942``; ``gfx950``
    *   -   HIPTENSOR_BUILD_TESTS
        -   Build Tests
        -   ON
    *   -   HIPTENSOR_BUILD_SAMPLES
        -   Build Samples
        -   ON
    *   -   HIPTENSOR_BUILD_COMPRESSED_DBG
        -   Enable compressed debug symbols
        -   ON
    *   -   HIPTENSOR_DATA_LAYOUT_COL_MAJOR
        -   Set hiptensor default data layout to column major
        -   ON

Here are some example project configurations:

.. tabularcolumns::
   |\X{1}{4}|\X{3}{4}|

+-----------------------------------+--------------------------------------------------------------------------------------------------------------------+
|         Configuration             |                                          Command                                                                   |
+===================================+====================================================================================================================+
|            Basic                  | :code:`CC=/opt/rocm/bin/amdclang CXX=/opt/rocm/bin/amdclang++ cmake -B<build_dir> .`                               |
+-----------------------------------+--------------------------------------------------------------------------------------------------------------------+
|        Targeting gfx908           | :code:`CC=/opt/rocm/bin/amdclang CXX=/opt/rocm/bin/amdclang++ cmake -B<build_dir> . -DGPU_TARGETS=gfx908`          |
+-----------------------------------+--------------------------------------------------------------------------------------------------------------------+
|          Debug build              | :code:`CC=/opt/rocm/bin/amdclang CXX=/opt/rocm/bin/amdclang++ cmake -B<build_dir> . -DCMAKE_BUILD_TYPE=Debug`      |
+-----------------------------------+--------------------------------------------------------------------------------------------------------------------+

Build library
^^^^^^^^^^^^^^^^^^

By default, the project is configured in Release mode.

To build the library alone, run:

.. code-block:: bash

    CC=/opt/rocm/bin/amdclang CXX=/opt/rocm/bin/amdclang++ cmake -B <build_dir> . -DHIPTENSOR_BUILD_TESTS=OFF -DHIPTENSOR_BUILD_SAMPLES=OFF

After configuration, build using:

.. code-block:: bash

    cmake --build <build_dir> -- -j<nproc>

.. note::
    We recommend using a minimum of 16 threads to build hipTensor with any tests (-j16).

Build library and samples
^^^^^^^^^^^^^^^^^^^^^^^^^^^

To build library and samples, run:

.. code-block:: bash

    CC=/opt/rocm/bin/amdclang CXX=/opt/rocm/bin/amdclang++ cmake -B <build_dir> . -DHIPTENSOR_BUILD_TESTS=OFF -DHIPTENSOR_BUILD_SAMPLES=ON

After configuration, build using:

.. code-block:: bash

    cmake --build <build_dir> -- -j<nproc>

The samples folder in ``<build_dir>`` contains executables in the table below.

.. tabularcolumns::
   |\X{2}{4}|\X{2}{4}|

================================================================== =====================================================================================================================================================================
Executable Name                                                    Description
================================================================== =====================================================================================================================================================================
``simple_bilinear_contraction_bf16_bf16_bf16_bf16_compute_bf16``   A simple bilinear contraction [D = alpha * (A x B) + beta * C] using half-precision brain float inputs, output and compute types
``simple_bilinear_contraction_f16_f16_f16_f16_compute_f16``        A simple bilinear contraction [D = alpha * (A x B) + beta * C] using half-precision floating point inputs, output and compute types
``simple_bilinear_contraction_f32_f32_f32_f32_compute_bf16``       A simple bilinear contraction [D = alpha * (A x B) + beta * C] using single-precision floating point input and output, half-precision brain float compute types
``simple_bilinear_contraction_f32_f32_f32_f32_compute_f16``        A simple bilinear contraction [D = alpha * (A x B) + beta * C] using single-precision floating point input and output, half-precision floating point compute types
``simple_bilinear_contraction_f32_f32_f32_f32_compute_f32``        A simple bilinear contraction [D = alpha * (A x B) + beta * C] using single-precision floating point input, output and compute types
``simple_bilinear_contraction_cf32_cf32_cf32_cf32_compute_cf32``   A simple bilinear contraction [D = alpha * (A x B) + beta * C] using complex single-precision floating point input, output and compute types
``simple_bilinear_contraction_f64_f64_f64_f64_compute_f32``        A simple bilinear contraction [D = alpha * (A x B) + beta * C] using double-precision floating point input, output and single precision floating point compute types
``simple_bilinear_contraction_f64_f64_f64_f64_compute_f64``        A simple bilinear contraction [D = alpha * (A x B) + beta * C] using double-precision floating point input, output and compute types
``simple_scale_contraction_bf16_bf16_bf16_compute_bf16``           A simple scale contraction [D = alpha * (A x B) ] using half-precision brain float inputs, output and compute types
``simple_scale_contraction_f16_f16_f16_compute_f16``               A simple scale contraction [D = alpha * (A x B) ] using half-precision floating point inputs, output and compute types
``simple_scale_contraction_f32_f32_f32_compute_bf16``              A simple scale contraction [D = alpha * (A x B) ] using single-precision floating point input and output, half-precision brain float compute types
``simple_scale_contraction_f32_f32_f32_compute_f16``               A simple scale contraction [D = alpha * (A x B) ] using single-precision floating point input and output, half-precision floating point compute types
``simple_scale_contraction_f32_f32_f32_compute_f32``               A simple scale contraction [D = alpha * (A x B) ] using single-precision floating point input, output and compute types
``simple_scale_contraction_cf32_cf32_cf32_compute_cf32``           A simple scale contraction [D = alpha * (A x B) ] using complex single-precision floating point input, output and compute types
``simple_scale_contraction_f64_f64_f64_compute_f32``               A simple scale contraction [D = alpha * (A x B) ] using double-precision floating point input, output and single precision floating point compute types
``simple_scale_contraction_f64_f64_f64_compute_f64``               A simple scale contraction [D = alpha * (A x B) ] using double-precision floating point input, output and compute types
``simple_permutation``                                             A simple permutation using single-precision floating point input and output types
``simple_reduction``                                               A simple reduction using single-precision floating point input and output types
================================================================== =====================================================================================================================================================================

Build library and tests
^^^^^^^^^^^^^^^^^^^^^^^^^

To build library and tests, run:

.. code-block:: bash

    CC=/opt/rocm/bin/amdclang CXX=/opt/rocm/bin/amdclang++ cmake -B <build_dir> . -DHIPTENSOR_BUILD_TESTS=ON -DHIPTENSOR_BUILD_SAMPLES=OFF

After configuration, build using:

.. code-block:: bash

    cmake --build <build_dir> -- -j<nproc>

The tests in ``<build_dir>`` contain executables as given in the table below.

.. tabularcolumns::
   |\X{2}{4}|\X{2}{4}|

================================================ ===========================================================================================================================
Executable name                                  Description
================================================ ===========================================================================================================================
``logger_test``                                  Unit test to validate hipTensor Logger APIs
``yaml_test``                                    Unit test to validate the YAML functionality used to bundle and run test suites
``bilinear_contraction_test_m1n1k1``             Bilinear contraction test [D = alpha * (A x B) + beta * C] with  half, single and mixed precision datatypes of rank 2
``bilinear_contraction_test_m2n2k2``             Bilinear contraction test [D = alpha * (A x B) + beta * C] with  half, single and mixed precision datatypes of rank 4
``bilinear_contraction_test_m3n3k3``             Bilinear contraction test [D = alpha * (A x B) + beta * C] with  half, single and mixed precision datatypes of rank 6
``bilinear_contraction_test_m4n4k4``             Bilinear contraction test [D = alpha * (A x B) + beta * C] with  half, single and mixed precision datatypes of rank 8
``bilinear_contraction_test_m5n5k5``             Bilinear contraction test [D = alpha * (A x B) + beta * C] with  half, single and mixed precision datatypes of rank 10
``bilinear_contraction_test_m6n6k6``             Bilinear contraction test [D = alpha * (A x B) + beta * C] with  half, single and mixed precision datatypes of rank 12
``complex_bilinear_contraction_test_m1n2k1``     Bilinear contraction test [D = alpha * (A x B) + beta * C] with  complex single and double precision datatypes of rank 2
``complex_bilinear_contraction_test_m2n2k2``     Bilinear contraction test [D = alpha * (A x B) + beta * C] with  complex single and double precision datatypes of rank 4
``complex_bilinear_contraction_test_m3n3k3``     Bilinear contraction test [D = alpha * (A x B) + beta * C] with  complex single and double precision datatypes of rank 6
``complex_bilinear_contraction_test_m4n4k4``     Bilinear contraction test [D = alpha * (A x B) + beta * C] with  complex single and double precision datatypes of rank 8
``complex_bilinear_contraction_test_m5n5k5``     Bilinear contraction test [D = alpha * (A x B) + beta * C] with  complex single and double precision datatypes of rank 10
``complex_bilinear_contraction_test_m6n6k6``     Bilinear contraction test [D = alpha * (A x B) + beta * C] with  complex single and double precision datatypes of rank 12
``scale_contraction_test_m1n1k1``                Scale contraction test [D = alpha * (A x B)] with  half, single and mixed precision datatypes of rank 2
``scale_contraction_test_m2n2k2``                Scale contraction test [D = alpha * (A x B)] with  half, single and mixed precision datatypes of rank 4
``scale_contraction_test_m3n3k3``                Scale contraction test [D = alpha * (A x B)] with  half, single and mixed precision datatypes of rank 6
``scale_contraction_test_m4n4k4``                Scale contraction test [D = alpha * (A x B)] with  half, single and mixed precision datatypes of rank 8
``scale_contraction_test_m5n5k5``                Scale contraction test [D = alpha * (A x B)] with  half, single and mixed precision datatypes of rank 10
``scale_contraction_test_m6n6k6``                Scale contraction test [D = alpha * (A x B)] with  half, single and mixed precision datatypes of rank 12
``complex_scale_contraction_test_m1n1k1``        Scale contraction test [D = alpha * (A x B)] with  complex single and double precision datatypes of rank 2
``complex_scale_contraction_test_m2n2k2``        Scale contraction test [D = alpha * (A x B)] with  complex single and double precision datatypes of rank 4
``complex_scale_contraction_test_m3n3k3``        Scale contraction test [D = alpha * (A x B)] with  complex single and double precision datatypes of rank 6
``complex_scale_contraction_test_m4n4k4``        Scale contraction test [D = alpha * (A x B)] with  complex single and double precision datatypes of rank 8
``complex_scale_contraction_test_m5n5k5``        Scale contraction test [D = alpha * (A x B)] with  complex single and double precision datatypes of rank 10
``complex_scale_contraction_test_m6n6k6``        Scale contraction test [D = alpha * (A x B)] with  complex single and double precision datatypes of rank 12
``rank2_permutation_test``                       Permutation test with half and single precision datatypes of rank 2
``rank3_permutation_test``                       Permutation test with half and single precision datatypes of rank 3
``rank4_permutation_test``                       Permutation test with half and single precision datatypes of rank 4
``rank5_permutation_test``                       Permutation test with half and single precision datatypes of rank 5
``rank6_permutation_test``                       Permutation test with half and single precision datatypes of rank 6
``rank1_reduction_test``                         Reduction test with half, single and double precision datatypes of rank 1
``rank2_reduction_test``                         Reduction test with half, single and double precision datatypes of rank 2
``rank3_reduction_test``                         Reduction test with half, single and double precision datatypes of rank 3
``rank4_reduction_test``                         Reduction test with half, single and double precision datatypes of rank 4
``rank5_reduction_test``                         Reduction test with half, single and double precision datatypes of rank 5
``rank6_reduction_test``                         Reduction test with half, single and double precision datatypes of rank 6
================================================ ===========================================================================================================================

Make targets list
^^^^^^^^^^^^^^^^^

When building hipTensor during the ``make`` step, we can specify make targets instead of defaulting ``make all``. The following table highlights relationships between high level grouped targets and individual targets.

.. tabularcolumns::
   |\X{1}{4}|\X{3}{4}|

+-----------------------------------+-----------------------------------------------------------------------------+
|           Group Target            |            Individual Targets                                               |
+===================================+=============================================================================+
|                                   |simple_bilinear_contraction_bf16_bf16_bf16_bf16_compute_bf16                 |
|                                   +-----------------------------------------------------------------------------+
|                                   |simple_bilinear_contraction_f16_f16_f16_f16_compute_f16                      |
|                                   +-----------------------------------------------------------------------------+
| hiptensor_samples                 |simple_bilinear_contraction_f32_f32_f32_f32_compute_bf16                     |
|                                   +-----------------------------------------------------------------------------+
|                                   |simple_bilinear_contraction_f32_f32_f32_f32_compute_f16                      |
|                                   +-----------------------------------------------------------------------------+
|                                   |simple_bilinear_contraction_f32_f32_f32_f32_compute_f32                      |
|                                   +-----------------------------------------------------------------------------+
|                                   |simple_bilinear_contraction_cf32_cf32_cf32_cf32_compute_cf32                 |
|                                   +-----------------------------------------------------------------------------+
|                                   |simple_bilinear_contraction_f64_f64_f64_f64_compute_f32                      |
|                                   +-----------------------------------------------------------------------------+
|                                   |simple_bilinear_contraction_f64_f64_f64_f64_compute_f64                      |
|                                   +-----------------------------------------------------------------------------+
|                                   |simple_scale_contraction_bf16_bf16_bf16_compute_bf16                         |
|                                   +-----------------------------------------------------------------------------+
|                                   |simple_scale_contraction_f16_f16_f16_compute_f16                             |
|                                   +-----------------------------------------------------------------------------+
|                                   |simple_scale_contraction_f32_f32_f32_compute_bf16                            |
|                                   +-----------------------------------------------------------------------------+
|                                   |simple_scale_contraction_f32_f32_f32_compute_f16                             |
|                                   +-----------------------------------------------------------------------------+
|                                   |simple_scale_contraction_f32_f32_f32_compute_f32                             |
|                                   +-----------------------------------------------------------------------------+
|                                   |simple_scale_contraction_cf32_cf32_cf32_compute_cf32                         |
|                                   +-----------------------------------------------------------------------------+
|                                   |simple_scale_contraction_f64_f64_f64_compute_f32                             |
|                                   +-----------------------------------------------------------------------------+
|                                   |simple_scale_contraction_f64_f64_f64_compute_f64                             |
|                                   +-----------------------------------------------------------------------------+
|                                   |simple_permutation                                                           |
|                                   |simple_reduction                                                             |
+-----------------------------------+-----------------------------------------------------------------------------+
|                                   |logger_test                                                                  |
|                                   +-----------------------------------------------------------------------------+
|                                   |yaml_test                                                                    |
|                                   +-----------------------------------------------------------------------------+
|                                   |bilinear_contraction_test_m1n1k1                                             |
|                                   +-----------------------------------------------------------------------------+
| hiptensor_tests                   |bilinear_contraction_test_m2n2k2                                             |
|                                   +-----------------------------------------------------------------------------+
|                                   |bilinear_contraction_test_m3n3k3                                             |
|                                   +-----------------------------------------------------------------------------+
|                                   |bilinear_contraction_test_m4n4k4                                             |
|                                   +-----------------------------------------------------------------------------+
|                                   |bilinear_contraction_test_m5n5k5                                             |
|                                   +-----------------------------------------------------------------------------+
|                                   |bilinear_contraction_test_m6n6k6                                             |
|                                   +-----------------------------------------------------------------------------+
|                                   |complex_bilinear_contraction_test_m1n1k1                                     |
|                                   +-----------------------------------------------------------------------------+
|                                   |complex_bilinear_contraction_test_m2n2k2                                     |
|                                   +-----------------------------------------------------------------------------+
|                                   |complex_bilinear_contraction_test_m3n3k3                                     |
|                                   +-----------------------------------------------------------------------------+
|                                   |complex_bilinear_contraction_test_m4n4k4                                     |
|                                   +-----------------------------------------------------------------------------+
|                                   |complex_bilinear_contraction_test_m5n5k5                                     |
|                                   +-----------------------------------------------------------------------------+
|                                   |complex_bilinear_contraction_test_m6n6k6                                     |
|                                   +-----------------------------------------------------------------------------+
|                                   |scale_contraction_test_m1n1k1                                                |
|                                   +-----------------------------------------------------------------------------+
|                                   |scale_contraction_test_m2n2k2                                                |
|                                   +-----------------------------------------------------------------------------+
|                                   |scale_contraction_test_m3n3k3                                                |
|                                   +-----------------------------------------------------------------------------+
|                                   |scale_contraction_test_m4n4k4                                                |
|                                   +-----------------------------------------------------------------------------+
|                                   |scale_contraction_test_m5n5k5                                                |
|                                   +-----------------------------------------------------------------------------+
|                                   |scale_contraction_test_m6n6k6                                                |
|                                   +-----------------------------------------------------------------------------+
|                                   |complex_scale_contraction_test_m1n1k1                                        |
|                                   +-----------------------------------------------------------------------------+
|                                   |complex_scale_contraction_test_m2n2k2                                        |
|                                   +-----------------------------------------------------------------------------+
|                                   |complex_scale_contraction_test_m3n3k3                                        |
|                                   +-----------------------------------------------------------------------------+
|                                   |complex_scale_contraction_test_m4n4k4                                        |
|                                   +-----------------------------------------------------------------------------+
|                                   |complex_scale_contraction_test_m5n5k5                                        |
|                                   +-----------------------------------------------------------------------------+
|                                   |complex_scale_contraction_test_m6n6k6                                        |
|                                   +-----------------------------------------------------------------------------+
|                                   |rank2_permutation_test                                                       |
|                                   +-----------------------------------------------------------------------------+
|                                   |rank3_permutation_test                                                       |
|                                   +-----------------------------------------------------------------------------+
|                                   |rank4_permutation_test                                                       |
|                                   +-----------------------------------------------------------------------------+
|                                   |rank5_permutation_test                                                       |
|                                   +-----------------------------------------------------------------------------+
|                                   |rank6_permutation_test                                                       |
|                                   +-----------------------------------------------------------------------------+
|                                   |rank1_reduction_test                                                         |
|                                   +-----------------------------------------------------------------------------+
|                                   |rank2_reduction_test                                                         |
|                                   +-----------------------------------------------------------------------------+
|                                   |rank3_reduction_test                                                         |
|                                   +-----------------------------------------------------------------------------+
|                                   |rank4_reduction_test                                                         |
|                                   +-----------------------------------------------------------------------------+
|                                   |rank5_reduction_test                                                         |
|                                   +-----------------------------------------------------------------------------+
|                                   |rank6_reduction_test                                                         |
+-----------------------------------+-----------------------------------------------------------------------------+

Benchmarking scripts
^^^^^^^^^^^^^^^^^^^^

Benchmarking scripts located at ``<project root>/scripts/performance/``

.. tabularcolumns::
   |\X{2}{4}|\X{2}{4}|

================================================================== =====================================================================================================================================================================
Script Name                                                        Description
================================================================== =====================================================================================================================================================================
``BenchmarkContraction.sh``                                        Benchmarking script for contraction
``BenchmarkPermutation.sh``                                        Benchmarking script for permutation
``BenchmarkReduction.sh``                                          Benchmarking script for reduction
================================================================== =====================================================================================================================================================================

Build performance
^^^^^^^^^^^^^^^^^

Depending on the resources available to the build machine and the build configuration selected, hipTensor build times can be on the order of an hour or more. Here are some things you can do to reduce build times:

* Target a specific GPU (e.g., ``-D GPU_TARGETS=gfx908``)
* Use lots of threads (e.g., ``-j32``)
* If they aren't needed, specify either ``HIPTENSOR_BUILD_TESTS`` or ``HIPTENSOR_BUILD_SAMPLES`` as OFF to disable client builds.
* During the ``make`` command, build a specific target, e.g: ``logger_test``.

Test run lengths
^^^^^^^^^^^^^^^^^

Depending on the resources available to the machine running the selected tests, hipTensor test runtimes can be on the order of an hour or more. Here are some things you can do to reduce run-times:

* CTest will invoke the entire test suite. You may invoke tests individually by name.
* Use GoogleTest filters, targeting specific test cases:

.. code-block:: bash

    <test_exe> --gtest_filter=*name_filter*

* Manually adjust the test cases coverage. Using your favorite text editor, you can modify test YAML configs to affect the test parameter coverage.
* Alternatively, use your own testing YAML config with a reduced parameter set.
* For tests with large tensor ranks, avoid using larger lengths to reduce computational load.

Test verbosity and file redirection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Tests support logging arguments that can be used to control verbosity and output redirection.

.. code-block:: bash

    <test_exe> -y "testing_params.yaml" -o "output.csv" --omit 1

.. tabularcolumns::
   |C|C|C|

+------------------------+-------------------------------------+--------------------------------------------------+
|Compact                 |Verbose                              |  Description                                     |
+========================+=====================================+==================================================+
| -y <input_file>.yaml   |                                     | override read testing parameters from input file |
+------------------------+-------------------------------------+--------------------------------------------------+
| -o <output_file>.csv   |                                     | redirect gtest output to file                    |
+------------------------+-------------------------------------+--------------------------------------------------+
|                        |                                     | code = 1: Omit gtest SKIPPED tests               |
|                        |                                     +--------------------------------------------------+
|                        | --omit <code>                       | code = 2: Omit gtest FAILED tests                |
|                        |                                     +--------------------------------------------------+
|                        |                                     | code = 4: Omit gtest PASSED tests                |
|                        |                                     +--------------------------------------------------+
|                        |                                     | code = 8: Omit all gtest output                  |
|                        |                                     +--------------------------------------------------+
|                        |                                     | code = <N>: OR combination of 1, 2, 4            |
+------------------------+-------------------------------------+--------------------------------------------------+
