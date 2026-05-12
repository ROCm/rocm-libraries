.. meta::
  :description: Learn how to install hipDNN.
  :keywords: hipDNN, ROCm, install, prerequisites, Windows, Linux

.. _prerequisites:

*******************
hipDNN installation
*******************

The hipDNN frontend API is distributed as a header-only library. It requires a development ROCm package installation.
These development packages contain the ``-dev`` prefix.

Dependencies
============

Before you begin, verify that your system is supported. For more information,
see :ref:`ROCm Core SDK components <rocm:release-components>`.

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Dependency
     - Version
     - Description
   * - ROCm development installation
     - ROCm 7.12.0 or later
     - AMD GPU programming stack
   * - CMake
     - 3.25.2 or later
     - Build system generator
   * - Ninja (recommended)
     - 1.12.1 or later
     - Faster build system
   * - C++ Compiler
     - C++17 compatible
     - hipDNN API supports C++17

.. _install-rocm:

Install the ROCm Core SDK
=========================

hipDNN is included with the ROCm Core SDK on Linux and Windows. For the most
complete installation on Linux, we recommend that developers use the
``amdrocm-core-sdk`` meta package.

For instructions, see :doc:`Install AMD ROCm <rocm:install/rocm>`. Use the
selector panel on that page to view instructions appropriate for your system
environment.

.. _install-base:

Install ROCm DNN libraries on Linux
===================================

Alternatively, if you want to install hipDNN as part of the ROCm
DNN package (a subset of the ROCm Core SDK ``amdrocm-core-sdk``) without
additional ROCm libraries and tools, install the ``amdrocm-dnn`` package.

1. Complete the :doc:`ROCm installation prerequisites <rocm:install/rocm>` to
   install dependencies and configure GPU access permissions.

2. Install the ROCm DNN package that matches your desired ROCm version,
   development package needs, and AMD GPU architecture. Package names use the
   following format:

   .. code-block:: shell-session

      amdrocm-dnn<dev/devel><rocm_version>-<llvm_target>

   Where:

   * ``<rocm_version>`` is the ROCm Core SDK version to install. Omit this
     suffix to install the latest available version.

   * ``<dev/devel>`` specifies whether to install library files and
     headers. Omit this suffix to only install runtime packages.

     * ``-dev`` is used on Debian-based distributions, including Ubuntu.

     * ``-devel`` is used on RPM-based distributions, including RHEL and SLES.

   * ``<llvm_target>`` (starting with ``gfx``) is used if you are installing
     for a single AMD GPU architecture. Omit this suffix to install for all
     architectures at the cost of disk space.

   For example, to install the latest DNN development package release for
   supported GPU architectures:

   .. tab-set::

      .. tab-item:: Debian-based distros

         .. code-block:: bash

            sudo apt install amdrocm-dnn-dev

      .. tab-item:: RHEL-based distros

         .. code-block:: bash

            sudo dnf install amdrocm-dnn-devel

      .. tab-item:: SLES

         .. code-block:: bash

            sudo zypper install amdrocm-dnn-devel

.. _install-nightly:

Install a nightly build
=======================

The `TheRock <https://github.com/ROCm/TheRock>`__ build system also publishes
nightly builds for the ROCm Core SDK and its components, including MIOpen.
See `Nightly release status
<https://github.com/ROCm/TheRock#nightly-release-status>`__ for details.

Verify hipDNN installation
==========================

The hipDNN library ships with executables that can be used to test the hipDNN library and plugins when the GPU hardware is available.
These executables can be run individually or as a suite using the ``ctest`` driver program. (``ctest`` is isntalled as part of the CMake package.)

Run hipDNN unit tests
---------------------

A suite of unit test executables are included with hipDNN. These test executables don't require the GPU hardware to run.

Here's an example that demonstrates running these tests using ``ctest``:

.. code-block:: console

  $ ctest --test-dir /opt/rocm/bin/hipdnn
  Internal ctest changing into directory: /opt/rocm/bin/hipdnn
  Test project /opt/rocm/bin/hipdnn
      Start 1: hipdnn_data_sdk_tests
  1/7 Test #1: hipdnn_data_sdk_tests ............   Passed    0.27 sec
      Start 2: hipdnn_backend_tests
  2/7 Test #2: hipdnn_backend_tests .............   Passed    1.25 sec
      Start 3: hipdnn_frontend_tests
  3/7 Test #3: hipdnn_frontend_tests ............   Passed    0.04 sec
      Start 4: hipdnn_test_sdk_tests
  4/7 Test #4: hipdnn_test_sdk_tests ............   Passed    3.48 sec
      Start 5: hipdnn_plugin_sdk_tests
  5/7 Test #5: hipdnn_plugin_sdk_tests ..........   Passed    0.03 sec
      Start 6: public_hipdnn_backend_tests
  6/7 Test #6: public_hipdnn_backend_tests ......   Passed    0.27 sec
      Start 7: public_hipdnn_frontend_tests
  7/7 Test #7: public_hipdnn_frontend_tests .....   Passed    0.23 sec

  100% tests passed, 0 tests failed out of 7

Replace ``/opt/rocm`` with your ROCm install folder when running this ``ctest`` command.

Run hipDNN samples
------------------

The sample programs shipped with hipDNN require AMD GPU hardware to execute.
The samples execute supported tensor operations on GPU hardware using the plugins installed with hipDNN.

This example demonstrates running the entire set of samples using ``ctest``:

.. code-block:: console

  $ ctest --test-dir /opt/rocm/bin/hipdnn_samples
  Internal ctest changing into directory: /opt/rocm/bin/hipdnn_samples
  Test project /opt/rocm/bin/hipdnn_samples
        Start  1: conv_fprop
  1/15 Test  #1: conv_fprop .............................   Passed    5.15 sec
        Start  2: conv_dgrad
  2/15 Test  #2: conv_dgrad .............................   Passed    5.80 sec
        Start  3: conv_wgrad
  3/15 Test  #3: conv_wgrad .............................   Passed    8.03 sec
        Start  4: fused_conv_fprop_activ
  4/15 Test  #4: fused_conv_fprop_activ .................   Passed    1.01 sec
        Start  5: fused_conv_fprop_bias_activ
  5/15 Test  #5: fused_conv_fprop_bias_activ ............   Passed    1.43 sec
        Start  6: conv_fprop_deterministic
  6/15 Test  #6: conv_fprop_deterministic ...............   Passed    9.14 sec
        Start  7: bn_inference
  7/15 Test  #7: bn_inference ...........................   Passed    2.78 sec
        Start  8: bn_inference_with_variance
  8/15 Test  #8: bn_inference_with_variance .............   Passed    0.45 sec
        Start  9: bn_training
  9/15 Test  #9: bn_training ............................   Passed    3.26 sec
        Start 10: fused_bn_training_activ
  10/15 Test #10: fused_bn_training_activ ................   Passed    3.30 sec
        Start 11: bn_backward
  11/15 Test #11: bn_backward ............................   Passed    3.52 sec
        Start 12: fused_bn_inference_activ
  12/15 Test #12: fused_bn_inference_activ ...............   Passed    2.81 sec
        Start 13: fused_bn_inference_drelu_bn_backward
  13/15 Test #13: fused_bn_inference_drelu_bn_backward ...   Passed    3.08 sec
        Start 14: fused_bn_inference_variance_activ
  14/15 Test #14: fused_bn_inference_variance_activ ......   Passed    0.48 sec
        Start 15: serialization_roundtrip
  15/15 Test #15: serialization_roundtrip ................   Passed    6.49 sec

  100% tests passed, 0 tests failed out of 15

Replace ``/opt/rocm`` with your ROCm install folder when running this ``ctest`` command.

See `hipDNN Samples <https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipdnn/samples/README.md>`_ for further details on running these sample programs and their operation.
