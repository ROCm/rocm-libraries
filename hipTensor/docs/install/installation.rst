.. meta::
   :description: installation instructions for the hipTensor library
   :keywords: hipTensor, ROCm, library, API, tool, installation

.. _installation:

*********************************
Installing and building hipTensor
*********************************

The quickest way to install hipTensor is to use the prebuilt packages that are released with ROCm.
Alternatively, this topic includes instructions to build from source.

The available ROCm packages are:

*  hiptensor (library and header files for development)
*  hiptensor-dev (library development package)
*  hiptensor-samples (sample executables)
*  hiptensor-tests (test executables)
*  hiptensor-clients (samples and test executables)

Prerequisites
=============

hipTensor requires a ROCm-enabled platform, using ROCm version 6.4 or later.
For more information, see :doc:`ROCm installation <rocm-install-on-linux:index>`
and the `ROCm GitHub <https://github.com/ROCm/ROCm>`_.

Installing prebuilt packages
============================

To install hipTensor on Ubuntu or Debian, use these commands:

.. code-block:: shell

   sudo apt-get update
   sudo apt-get install hiptensor hiptensor-dev hiptensor-samples hiptensor-tests

To install hipTensor on RHEL-based systems, use:

.. code-block:: shell

   sudo yum update
   sudo yum install hiptensor hiptensor-dev hiptensor-samples hiptensor-tests

To install hipTensor on SLES, use:

.. code-block:: shell

   sudo dnf upgrade
   sudo dnf install hiptensor hiptensor-dev hiptensor-samples hiptensor-tests

After hipTensor is installed, it can be used just like any other library with a C++ API.

Building and installing hipTensor from source
=============================================

It isn't necessary to build hipTensor from source because it's ready to use after installing
the prebuilt packages, as described above.
To build hipTensor from source, follow the instructions in this section.

System requirements
-------------------------------------------

8GB of system memory is required for a full hipTensor build.
This value might be lower if hipTensor is built without tests.

GPU support
-------------------------------------------

hipTensor is supported on the AMD CDNA class GPUs featuring matrix core support,
including the gfx908, gfx90a, gfx942, and gfx950 GPUs (collectively labeled as gfx9).

.. note::

   Double precision ``FP64`` datatype support requires the gfx90a, gfx942, or gfx950.

Dependencies
-------------------------------------------

hipTensor is designed to have minimal external dependencies so it's lightweight and portable.
The following dependencies are required:

.. <!-- spellcheck-disable -->

*  `ROCm <https://github.com/ROCm/ROCm>`_ (Version 6.4 or later)
*  `CMake <https://cmake.org/>`_ (Version 3.14 or later)
*  `rocm-cmake <https://github.com/ROCm/rocm-cmake>`_ (Version 0.8.0 or later)
*  `HIP runtime <https://github.com/ROCm/hip>`_ (Version 6.4.0 or later) (Or the ROCm hip-runtime-amd package)
*  LLVM dev package (Version 7.0 or later) (Also available as the ROCm rocm-llvm-dev package)
*  `composable kernel <https://github.com/ROCm/composable_kernel>`_ (hipTensor uses the amd-master branch, which is a stable and widely adopted version for development.)

.. <!-- spellcheck-enable -->

.. note::

   It's best to use ROCm packages from the same release where applicable.

Downloading hipTensor
-------------------------------------------

The hipTensor source code is available from the `hipTensor GitHub <https://github.com/ROCm/hipTensor>`_.
ROCm version 6.4 or later is required.

To verify the ROCm version installed on an Ubuntu system, use this command:

.. code-block:: shell

   apt show rocm-libs -a

For RHEL-related systems, use:

.. code-block:: shell

   yum info rocm-libs

The ROCm version has major, minor, and patch fields, possibly followed by a build-specific identifier.
For example, a ROCm version of ``4.0.0.40000-23`` corresponds to major release = ``4``, minor release = ``0``,
patch = ``0``, and build identifier ``40000-23``.
The hipTensor GitHub has branches with names like ``rocm-major.minor.x``,
where ``major`` and ``minor`` are the same as for the ROCm version.
To download hipTensor on ROCm version `x.y`, use this command:

.. code-block:: shell

   git clone -b release/rocm-rel-x.y https://github.com/ROCm/hipTensor.git
   cd hipTensor

Replace ``x.y`` in the above command with the version of ROCm installed on your machine.
For example, if you have ROCm 6.4 installed, then replace ``release/rocm-rel-x.y`` with ``release/rocm-rel-6.4``.

Build configuration
-------------------------------------------

You can choose to build any of the following combinations:

* The hipTensor library only
* The library and samples
* The library and tests
* The library, samples, and tests

You only need the hipTensor library to call and link to hipTensor API from your code.
The clients contain the tests and sample code.

Here are the available options to build the hipTensor library, with or without clients.

.. list-table::

    *   -   **Option**
        -   **Description**
        -   **Default value**
    *   -   ``GPU_TARGETS``
        -   Build the code for specific GPU target(s)
        -   ``gfx908``; ``gfx90a``; ``gfx942``; ``gfx950``
    *   -   ``HIPTENSOR_BUILD_TESTS``
        -   Build the tests
        -   ``ON``
    *   -   ``HIPTENSOR_BUILD_SAMPLES``
        -   Build the samples
        -   ``ON``
    *   -   ``HIPTENSOR_BUILD_COMPRESSED_DBG``
        -   Enable compressed debug symbols
        -   ``ON``
    *   -   ``HIPTENSOR_DEFAULT_STRIDES_COL_MAJOR``
        -   Set the hipTensor default data layout to column major
        -   ``ON``

Here are some example project configurations:

.. csv-table::
   :header: "Configuration","Command"
   :widths: 20, 110

   "Basic", "``CC=/opt/rocm/bin/amdclang CXX=/opt/rocm/bin/amdclang++ cmake -B<build_dir> .``"
   "Targeting gfx908", "``CC=/opt/rocm/bin/amdclang CXX=/opt/rocm/bin/amdclang++ cmake -B<build_dir> . -DGPU_TARGETS=gfx908``"
   "Debug build", "``CC=/opt/rocm/bin/amdclang CXX=/opt/rocm/bin/amdclang++ cmake -B<build_dir> . -DCMAKE_BUILD_TYPE=Debug``"

Building the library alone
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, the project is configured for Release mode.

To build the library alone, run this command:

.. code-block:: bash

   CC=/opt/rocm/bin/amdclang CXX=/opt/rocm/bin/amdclang++ cmake -B <build_dir> . -DHIPTENSOR_BUILD_TESTS=OFF -DHIPTENSOR_BUILD_SAMPLES=OFF

After configuration, build the library using this command:

.. code-block:: bash

   cmake --build <build_dir> -- -j<nproc>

.. note::
   
   It's recommended to use a minimum of 16 threads to build hipTensor with any tests, for example, using ``-j16``.

Building the library and samples
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To build the library and samples, run the following command:

.. code-block:: bash

   CC=/opt/rocm/bin/amdclang CXX=/opt/rocm/bin/amdclang++ cmake -B <build_dir> . -DHIPTENSOR_BUILD_TESTS=OFF -DHIPTENSOR_BUILD_SAMPLES=ON

After configuration, build the library using this command:

.. code-block:: bash

   cmake --build <build_dir> -- -j<nproc>

The samples folder in ``<build_dir>`` contains the executables in the table below.

.. tabularcolumns::
   |\X{2}{4}|\X{2}{4}|

================================================================== =======================================================================================================================================================================
Executable name                                                    Description
================================================================== =======================================================================================================================================================================
``simple_bilinear_contraction_bf16_bf16_bf16_bf16_compute_bf16``   A simple bilinear contraction [D = alpha * (A x B) + beta * C] using half-precision brain float inputs, output, and compute types
``simple_bilinear_contraction_f16_f16_f16_f16_compute_f16``        A simple bilinear contraction [D = alpha * (A x B) + beta * C] using half-precision floating point inputs, output, and compute types
``simple_bilinear_contraction_f32_f32_f32_f32_compute_bf16``       A simple bilinear contraction [D = alpha * (A x B) + beta * C] using single-precision floating point input and output, and half-precision brain float compute types
``simple_bilinear_contraction_f32_f32_f32_f32_compute_f16``        A simple bilinear contraction [D = alpha * (A x B) + beta * C] using single-precision floating point input and output, and half-precision floating point compute types
``simple_bilinear_contraction_f32_f32_f32_f32_compute_f32``        A simple bilinear contraction [D = alpha * (A x B) + beta * C] using single-precision floating point input, output, and compute types
``simple_bilinear_contraction_cf32_cf32_cf32_cf32_compute_cf32``   A simple bilinear contraction [D = alpha * (A x B) + beta * C] using complex single-precision floating point input, output, and compute types
``simple_bilinear_contraction_f64_f64_f64_f64_compute_f32``        A simple bilinear contraction [D = alpha * (A x B) + beta * C] using double-precision floating point input and output and single-precision floating point compute types
``simple_bilinear_contraction_f64_f64_f64_f64_compute_f64``        A simple bilinear contraction [D = alpha * (A x B) + beta * C] using double-precision floating point input, output, and compute types
``simple_scale_contraction_bf16_bf16_bf16_compute_bf16``           A simple scale contraction [D = alpha * (A x B) ] using half-precision brain float inputs, output, and compute types
``simple_scale_contraction_f16_f16_f16_compute_f16``               A simple scale contraction [D = alpha * (A x B) ] using half-precision floating point inputs, output, and compute types
``simple_scale_contraction_f32_f32_f32_compute_bf16``              A simple scale contraction [D = alpha * (A x B) ] using single-precision floating point input and output and half-precision brain float compute types
``simple_scale_contraction_f32_f32_f32_compute_f16``               A simple scale contraction [D = alpha * (A x B) ] using single-precision floating point input and output and half-precision floating point compute types
``simple_scale_contraction_f32_f32_f32_compute_f32``               A simple scale contraction [D = alpha * (A x B) ] using single-precision floating point input, output, and compute types
``simple_scale_contraction_cf32_cf32_cf32_compute_cf32``           A simple scale contraction [D = alpha * (A x B) ] using complex single-precision floating point input, output, and compute types
``simple_scale_contraction_f64_f64_f64_compute_f32``               A simple scale contraction [D = alpha * (A x B) ] using double-precision floating point input and output and single-precision floating point compute types
``simple_scale_contraction_f64_f64_f64_compute_f64``               A simple scale contraction [D = alpha * (A x B) ] using double-precision floating point input, output, and compute types
``simple_permutation``                                             A simple permutation using single-precision floating point input and output types
``simple_elementwise_binary``                                      A simple element-wise binary operation using single-precision floating point input and output types
``simple_elementwise_trinary``                                     A simple element-wise trinary operation using single-precision floating point input and output types
``simple_reduction``                                               A simple reduction using single-precision floating point input and output types
================================================================== =======================================================================================================================================================================

Building the library and tests
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To build the library and tests, run the following command:

.. code-block:: bash

   CC=/opt/rocm/bin/amdclang CXX=/opt/rocm/bin/amdclang++ cmake -B <build_dir> . -DHIPTENSOR_BUILD_TESTS=ON -DHIPTENSOR_BUILD_SAMPLES=OFF

After configuration, build using this command:

.. code-block:: bash

   cmake --build <build_dir> -- -j<nproc>

The tests in ``<build_dir>`` contain executables, as shown in the table below.

.. tabularcolumns::
   |\X{2}{4}|\X{2}{4}|

================================================ ===========================================================================================================================
Executable name                                  Description
================================================ ===========================================================================================================================
``logger_test``                                  Unit test to validate hipTensor Logger APIs
``yaml_test``                                    Unit test to validate the YAML functionality used to bundle and run test suites
``bilinear_contraction_test_m1n1k1``             Bilinear contraction test [D = alpha * (A x B) + beta * C] with half, single, and mixed precision datatypes of rank 2
``bilinear_contraction_test_m2n2k2``             Bilinear contraction test [D = alpha * (A x B) + beta * C] with half, single, and mixed precision datatypes of rank 4
``bilinear_contraction_test_m3n3k3``             Bilinear contraction test [D = alpha * (A x B) + beta * C] with half, single, and mixed precision datatypes of rank 6
``bilinear_contraction_test_m4n4k4``             Bilinear contraction test [D = alpha * (A x B) + beta * C] with half, single, and mixed precision datatypes of rank 8
``bilinear_contraction_test_m5n5k5``             Bilinear contraction test [D = alpha * (A x B) + beta * C] with half, single, and mixed precision datatypes of rank 10
``bilinear_contraction_test_m6n6k6``             Bilinear contraction test [D = alpha * (A x B) + beta * C] with half, single, and mixed precision datatypes of rank 12
``complex_bilinear_contraction_test_m1n2k1``     Bilinear contraction test [D = alpha * (A x B) + beta * C] with complex single and double precision datatypes of rank 2
``complex_bilinear_contraction_test_m2n2k2``     Bilinear contraction test [D = alpha * (A x B) + beta * C] with complex single and double precision datatypes of rank 4
``complex_bilinear_contraction_test_m3n3k3``     Bilinear contraction test [D = alpha * (A x B) + beta * C] with complex single and double precision datatypes of rank 6
``complex_bilinear_contraction_test_m4n4k4``     Bilinear contraction test [D = alpha * (A x B) + beta * C] with complex single and double precision datatypes of rank 8
``complex_bilinear_contraction_test_m5n5k5``     Bilinear contraction test [D = alpha * (A x B) + beta * C] with complex single and double precision datatypes of rank 10
``complex_bilinear_contraction_test_m6n6k6``     Bilinear contraction test [D = alpha * (A x B) + beta * C] with complex single and double precision datatypes of rank 12
``scale_contraction_test_m1n1k1``                Scale contraction test [D = alpha * (A x B)] with half, single, and mixed precision datatypes of rank 2
``scale_contraction_test_m2n2k2``                Scale contraction test [D = alpha * (A x B)] with half, single, and mixed precision datatypes of rank 4
``scale_contraction_test_m3n3k3``                Scale contraction test [D = alpha * (A x B)] with half, single, and mixed precision datatypes of rank 6
``scale_contraction_test_m4n4k4``                Scale contraction test [D = alpha * (A x B)] with half, single, and mixed precision datatypes of rank 8
``scale_contraction_test_m5n5k5``                Scale contraction test [D = alpha * (A x B)] with half, single, and mixed precision datatypes of rank 10
``scale_contraction_test_m6n6k6``                Scale contraction test [D = alpha * (A x B)] with half, single, and mixed precision datatypes of rank 12
``complex_scale_contraction_test_m1n1k1``        Scale contraction test [D = alpha * (A x B)] with complex single and double precision datatypes of rank 2
``complex_scale_contraction_test_m2n2k2``        Scale contraction test [D = alpha * (A x B)] with complex single and double precision datatypes of rank 4
``complex_scale_contraction_test_m3n3k3``        Scale contraction test [D = alpha * (A x B)] with complex single and double precision datatypes of rank 6
``complex_scale_contraction_test_m4n4k4``        Scale contraction test [D = alpha * (A x B)] with complex single and double precision datatypes of rank 8
``complex_scale_contraction_test_m5n5k5``        Scale contraction test [D = alpha * (A x B)] with complex single and double precision datatypes of rank 10
``complex_scale_contraction_test_m6n6k6``        Scale contraction test [D = alpha * (A x B)] with complex single and double precision datatypes of rank 12
``rank2_permutation_test``                       Permutation test with half and single precision datatypes of rank 2
``rank3_permutation_test``                       Permutation test with half and single precision datatypes of rank 3
``rank4_permutation_test``                       Permutation test with half and single precision datatypes of rank 4
``rank5_permutation_test``                       Permutation test with half and single precision datatypes of rank 5
``rank6_permutation_test``                       Permutation test with half and single precision datatypes of rank 6
``rank2_elementwise_binary_op_test``             Element-wise binary operation test with half, single, and double precision datatypes of rank 2
``rank3_elementwise_binary_op_test``             Element-wise binary operation test with half, single, and double precision datatypes of rank 3
``rank4_elementwise_binary_op_test``             Element-wise binary operation test with half, single, and double precision datatypes of rank 4
``rank5_elementwise_binary_op_test``             Element-wise binary operation test with half, single, and double precision datatypes of rank 5
``rank6_elementwise_binary_op_test``             Element-wise binary operation test with half, single, and double precision datatypes of rank 6
``rank2_elementwise_trinary_op_test``            Element-wise trinary operation test with half, single, and double precision datatypes of rank 2
``rank3_elementwise_trinary_op_test``            Element-wise trinary operation test with half, single, and double precision datatypes of rank 3
``rank4_elementwise_trinary_op_test``            Element-wise trinary operation test with half, single, and double precision datatypes of rank 4
``rank5_elementwise_trinary_op_test``            Element-wise trinary operation test with half, single, and double precision datatypes of rank 5
``rank6_elementwise_trinary_op_test``            Element-wise trinary operation test with half, single, and double precision datatypes of rank 6
``rank1_reduction_test``                         Reduction test with half, single, and double precision datatypes of rank 1
``rank2_reduction_test``                         Reduction test with half, single, and double precision datatypes of rank 2
``rank3_reduction_test``                         Reduction test with half, single, and double precision datatypes of rank 3
``rank4_reduction_test``                         Reduction test with half, single, and double precision datatypes of rank 4
``rank5_reduction_test``                         Reduction test with half, single, and double precision datatypes of rank 5
``rank6_reduction_test``                         Reduction test with half, single, and double precision datatypes of rank 6
================================================ ===========================================================================================================================

Make targets list
^^^^^^^^^^^^^^^^^

When building hipTensor during the ``make`` step, you can specify the Make targets instead of defaulting to ``make all``.
The following table highlights the relationships between high-level grouped targets and individual targets.


+-----------------------------------+---------------------------------------------------------------------------------+
|           Group target            |            Individual targets                                                   |
+===================================+=================================================================================+
|                                   |``simple_bilinear_contraction_bf16_bf16_bf16_bf16_compute_bf16``                 |
|                                   +---------------------------------------------------------------------------------+
|                                   |``simple_bilinear_contraction_f16_f16_f16_f16_compute_f16``                      |
|                                   +---------------------------------------------------------------------------------+
| ``hiptensor_samples``             |``simple_bilinear_contraction_f32_f32_f32_f32_compute_bf16``                     |
|                                   +---------------------------------------------------------------------------------+
|                                   |``simple_bilinear_contraction_f32_f32_f32_f32_compute_f16``                      |
|                                   +---------------------------------------------------------------------------------+
|                                   |``simple_bilinear_contraction_f32_f32_f32_f32_compute_f32``                      |
|                                   +---------------------------------------------------------------------------------+
|                                   |``simple_bilinear_contraction_cf32_cf32_cf32_cf32_compute_cf32``                 |
|                                   +---------------------------------------------------------------------------------+
|                                   |``simple_bilinear_contraction_f64_f64_f64_f64_compute_f32``                      |
|                                   +---------------------------------------------------------------------------------+
|                                   |``simple_bilinear_contraction_f64_f64_f64_f64_compute_f64``                      |
|                                   +---------------------------------------------------------------------------------+
|                                   |``simple_scale_contraction_bf16_bf16_bf16_compute_bf16``                         |
|                                   +---------------------------------------------------------------------------------+
|                                   |``simple_scale_contraction_f16_f16_f16_compute_f16``                             |
|                                   +---------------------------------------------------------------------------------+
|                                   |``simple_scale_contraction_f32_f32_f32_compute_bf16``                            |
|                                   +---------------------------------------------------------------------------------+
|                                   |``simple_scale_contraction_f32_f32_f32_compute_f16``                             |
|                                   +---------------------------------------------------------------------------------+
|                                   |``simple_scale_contraction_f32_f32_f32_compute_f32``                             |
|                                   +---------------------------------------------------------------------------------+
|                                   |``simple_scale_contraction_cf32_cf32_cf32_compute_cf32``                         |
|                                   +---------------------------------------------------------------------------------+
|                                   |``simple_scale_contraction_f64_f64_f64_compute_f32``                             |
|                                   +---------------------------------------------------------------------------------+
|                                   |``simple_scale_contraction_f64_f64_f64_compute_f64``                             |
|                                   +---------------------------------------------------------------------------------+
|                                   |``simple_permutation``                                                           |
|                                   +---------------------------------------------------------------------------------+
|                                   |``simple_elementwise_binary``                                                    |
|                                   +---------------------------------------------------------------------------------+
|                                   |``simple_elementwise_trinary``                                                   |
|                                   +---------------------------------------------------------------------------------+
|                                   |``simple_reduction``                                                             |
+-----------------------------------+---------------------------------------------------------------------------------+
|                                   |``logger_test``                                                                  |
|                                   +---------------------------------------------------------------------------------+
|                                   |``yaml_test``                                                                    |
|                                   +---------------------------------------------------------------------------------+
|                                   |``bilinear_contraction_test_m1n1k1``                                             |
|                                   +---------------------------------------------------------------------------------+
| ``hiptensor_tests``               |``bilinear_contraction_test_m2n2k2``                                             |
|                                   +---------------------------------------------------------------------------------+
|                                   |``bilinear_contraction_test_m3n3k3``                                             |
|                                   +---------------------------------------------------------------------------------+
|                                   |``bilinear_contraction_test_m4n4k4``                                             |
|                                   +---------------------------------------------------------------------------------+
|                                   |``bilinear_contraction_test_m5n5k5``                                             |
|                                   +---------------------------------------------------------------------------------+
|                                   |``bilinear_contraction_test_m6n6k6``                                             |
|                                   +---------------------------------------------------------------------------------+
|                                   |``complex_bilinear_contraction_test_m1n1k1``                                     |
|                                   +---------------------------------------------------------------------------------+
|                                   |``complex_bilinear_contraction_test_m2n2k2``                                     |
|                                   +---------------------------------------------------------------------------------+
|                                   |``complex_bilinear_contraction_test_m3n3k3``                                     |
|                                   +---------------------------------------------------------------------------------+
|                                   |``complex_bilinear_contraction_test_m4n4k4``                                     |
|                                   +---------------------------------------------------------------------------------+
|                                   |``complex_bilinear_contraction_test_m5n5k5``                                     |
|                                   +---------------------------------------------------------------------------------+
|                                   |``complex_bilinear_contraction_test_m6n6k6``                                     |
|                                   +---------------------------------------------------------------------------------+
|                                   |``scale_contraction_test_m1n1k1``                                                |
|                                   +---------------------------------------------------------------------------------+
|                                   |``scale_contraction_test_m2n2k2``                                                |
|                                   +---------------------------------------------------------------------------------+
|                                   |``scale_contraction_test_m3n3k3``                                                |
|                                   +---------------------------------------------------------------------------------+
|                                   |``scale_contraction_test_m4n4k4``                                                |
|                                   +---------------------------------------------------------------------------------+
|                                   |``scale_contraction_test_m5n5k5``                                                |
|                                   +---------------------------------------------------------------------------------+
|                                   |``scale_contraction_test_m6n6k6``                                                |
|                                   +---------------------------------------------------------------------------------+
|                                   |``complex_scale_contraction_test_m1n1k1``                                        |
|                                   +---------------------------------------------------------------------------------+
|                                   |``complex_scale_contraction_test_m2n2k2``                                        |
|                                   +---------------------------------------------------------------------------------+
|                                   |``complex_scale_contraction_test_m3n3k3``                                        |
|                                   +---------------------------------------------------------------------------------+
|                                   |``complex_scale_contraction_test_m4n4k4``                                        |
|                                   +---------------------------------------------------------------------------------+
|                                   |``complex_scale_contraction_test_m5n5k5``                                        |
|                                   +---------------------------------------------------------------------------------+
|                                   |``complex_scale_contraction_test_m6n6k6``                                        |
|                                   +---------------------------------------------------------------------------------+
|                                   |``rank2_permutation_test``                                                       |
|                                   +---------------------------------------------------------------------------------+
|                                   |``rank3_permutation_test``                                                       |
|                                   +---------------------------------------------------------------------------------+
|                                   |``rank4_permutation_test``                                                       |
|                                   +---------------------------------------------------------------------------------+
|                                   |``rank5_permutation_test``                                                       |
|                                   +---------------------------------------------------------------------------------+
|                                   |``rank6_permutation_test``                                                       |
|                                   +---------------------------------------------------------------------------------+
|                                   |``rank2_elementwise_binary_op_test``                                             |
|                                   +---------------------------------------------------------------------------------+
|                                   |``rank3_elementwise_binary_op_test``                                             |
|                                   +---------------------------------------------------------------------------------+
|                                   |``rank4_elementwise_binary_op_test``                                             |
|                                   +---------------------------------------------------------------------------------+
|                                   |``rank5_elementwise_binary_op_test``                                             |
|                                   +---------------------------------------------------------------------------------+
|                                   |``rank6_elementwise_binary_op_test``                                             |
|                                   +---------------------------------------------------------------------------------+
|                                   |``rank2_elementwise_trinary_op_test``                                            |
|                                   +---------------------------------------------------------------------------------+
|                                   |``rank3_elementwise_trinary_op_test``                                            |
|                                   +---------------------------------------------------------------------------------+
|                                   |``rank4_elementwise_trinary_op_test``                                            |
|                                   +---------------------------------------------------------------------------------+
|                                   |``rank5_elementwise_trinary_op_test``                                            |
|                                   +---------------------------------------------------------------------------------+
|                                   |``rank6_elementwise_trinary_op_test``                                            |
|                                   +---------------------------------------------------------------------------------+
|                                   |``rank1_reduction_test``                                                         |
|                                   +---------------------------------------------------------------------------------+
|                                   |``rank2_reduction_test``                                                         |
|                                   +---------------------------------------------------------------------------------+
|                                   |``rank3_reduction_test``                                                         |
|                                   +---------------------------------------------------------------------------------+
|                                   |``rank4_reduction_test``                                                         |
|                                   +---------------------------------------------------------------------------------+
|                                   |``rank5_reduction_test``                                                         |
|                                   +---------------------------------------------------------------------------------+
|                                   |``rank6_reduction_test``                                                         |
+-----------------------------------+---------------------------------------------------------------------------------+

Benchmarking scripts
-------------------------------------------

The benchmarking scripts are located at ``<project root>/scripts/performance/``.

.. csv-table::
   :header: "Script name","Description"
   :widths: 40, 60

   "``BenchmarkContraction.sh``", "Benchmarking script for contraction"
   "``BenchmarkPermutation.sh``", "Benchmarking script for permutation"
   "``BenchmarkReduction.sh``", "Benchmarking script for reduction"

Build performance
-------------------------------------------

Depending on the resources available to the build machine and the selected build configuration,
hipTensor build times can take an hour or more. Here are some things you can do to reduce build times:

*  Target a specific GPU, for instance, with ``-D GPU_TARGETS=gfx908``.
*  Use a large number of threads, for instance, ``-j32``.
*  If the client builds aren't needed, specify ``HIPTENSOR_BUILD_TESTS`` or ``HIPTENSOR_BUILD_SAMPLES`` as ``OFF`` to disable them.
*  During the ``make`` command, build a specific target, for instance, ``logger_test``.

Test runtimes
-------------------------------------------

Depending on the resources available to the machine running the selected tests,
hipTensor test runtimes can last an hour or more. Here are some things you can do to reduce test runtimes:

*  CTest runs the entire test suite, but you can invoke tests individually by name.
*  Use GoogleTest filters to target specific test cases:

   .. code-block:: bash

      <test_exe> --gtest_filter=*name_filter*

*  Manually adjust the test case coverage. Use a text editor to modify the test YAML configs to adjust the test parameter coverage.
*  Alternatively, use your own YAML config for testing with a reduced parameter set.
*  For tests with large tensor ranks, avoid using larger lengths to reduce the computational load.

Test verbosity and file redirection
-------------------------------------------

The tests support logging arguments to control verbosity and output redirection.

.. code-block:: bash

   <test_exe> -y "testing_params.yaml" -o "output.csv" --omit 1

.. tabularcolumns::
   |C|C|C|

+----------------------------+-------------------------------------+--------------------------------------------------+
|Compact                     |Verbose                              |  Description                                     |
+============================+=====================================+==================================================+
| ``-y <input_file>.yaml``   |                                     | Override read testing parameters from input file |
+----------------------------+-------------------------------------+--------------------------------------------------+
| ``-o <output_file>.csv``   |                                     | Redirect GoogleTest output to file               |
+----------------------------+-------------------------------------+--------------------------------------------------+
|                            |                                     | code = 1: Omit gtest SKIPPED tests               |
|                            |                                     +--------------------------------------------------+
|                            | ``--omit <code>``                   | code = 2: Omit gtest FAILED tests                |
|                            |                                     +--------------------------------------------------+
|                            |                                     | code = 4: Omit gtest PASSED tests                |
|                            |                                     +--------------------------------------------------+
|                            |                                     | code = 8: Omit all gtest output                  |
|                            |                                     +--------------------------------------------------+
|                            |                                     | code = <N>: OR combination of 1, 2, 4            |
+----------------------------+-------------------------------------+--------------------------------------------------+
