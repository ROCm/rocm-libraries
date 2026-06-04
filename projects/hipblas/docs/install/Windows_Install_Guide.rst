.. meta::
  :description: Installing and Building hipBLAS for Windows
  :keywords: hipBLAS, rocBLAS, BLAS, ROCm, API, Linear Algebra, documentation, Windows installation, build

.. _windows-install:

*****************************************
Install and build for hipBLAS for Windows
*****************************************

To build hipBLAS as part of the ROCm Core SDK, see `TheRock build
instructions
<https://github.com/ROCm/TheRock/blob/main/docs/development/README.md>`__.
TheRock is the recommended way to build ROCm components from source.

Alternatively, you can build hipBLAS standalone using the following
instructions.

This topic covers how to install hipBLAS on Microsoft Windows from a package and how to build and install it from the source code.
For a list of installation prerequisites, see :doc:`hipBLAS prerequisites <prerequisites>`.

Prerequisites for Microsoft Windows
===================================

*  Here are the prerequisites required to use the rocBLAS backend with AMD components:

   * A ROCm-enabled platform. For more information, see :ref:`ROCm Core SDK components <rocm:release-components>`.
   * A compatible version of `hipblas-common <https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipblas-common>`_.
   * A compatible version of :doc:`rocBLAS <rocblas:index>`.
   * For full functionality, a compatible version of :doc:`rocSOLVER <rocsolver:index>` and its dependencies.
   * hipBLAS is supported on the same Windows versions and toolchains that HIP SDK supports.

* hipBLAS does not support the cuBLAS backend on Windows.

Building and installing hipBLAS
===============================

Follow these instructions to build hipBLAS from source.

Download the hipBLAS source code
--------------------------------

The hipBLAS source code is available from the `hipBLAS folder <https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipblas>`_
of the `rocm-libraries GitHub <https://github.com/ROCm/rocm-libraries>`_.The ROCm HIP SDK version might be shown in the default installation path,
but you can run the HIP SDK compiler to display the version from the ``bin/`` folder:

::

    hipcc --version

The HIP version includes the major, minor, and patch fields, and possibly a build-specific identifier.
As an example, the HIP version might be ``5.4.22880-135e1ab4``.
This corresponds to major release = ``5``, minor release = ``4``, patch = ``22880``, and the build identifier ``135e1ab4``.
The rocm-libraries GitHub contains branches with names like ``release/rocm-rel-major.minor``. Select the branch where the major and minor
fields correspond to the HIP version.

To download hipBLAS, including all projects in the rocm-libraries repository, use the following commands.

::

   git clone -b release/rocm-rel-x.y https://github.com/ROCm/rocm-libraries.git
   cd  rocm-libraries/projects/hipblas

Replace ``x.y`` in this command with the version of the HIP SDK installed on your machine.
For example, if you have HIP 7.0 installed, then use ``-b release/rocm-rel-7.0``.
You can add the SDK tools to your path with an entry such as:

::

    %HIP_PATH%\bin


To limit your local checkout to only the hipBLAS project, configure ``sparse-checkout`` before cloning.
This uses the Git partial clone feature (``--filter=blob:none``) to reduce how much data is downloaded.
Use the following commands for a sparse checkout:

.. note::

   To include the hipBLAS dependencies, set the projects for the sparse checkout using
   ``git sparse-checkout set projects/hipblas projects/rocsolver projects/rocblas projects/hipblas-common``.

.. code-block:: shell

   git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-libraries.git
   cd rocm-libraries
   git sparse-checkout init --cone
   git sparse-checkout set projects/hipblas # add projects/rocsolver projects/rocblas, projects/hipblas-common to include dependencies
   git checkout develop # or use the branch you want to work with

.. note::

   To build ROCm 6.4.3 and older, use the hipBLAS repository at `<https://github.com/ROCm/hipBLAS>`_.
   For more information, see the documentation associated with the release you want to build.

Building the library and library dependencies
---------------------------------------------

The root of the hipBLAS repository includes a helper Python script named ``rmake.py`` which lets you build and install hipBLAS
with a single command. It accepts several options but has a hard-coded configuration
that you can override by invoking ``cmake`` directly. However, it's a great way to get started quickly and serves
as an example of how to build and install hipBLAS.
A few commands in the script require administrator access, so you might be prompted for a password.

.. note::

   You can run the ``rmake.py`` script from the ``projects/hipblas`` directory.

Common examples showing how to use ``rmake.py`` to build the library dependencies and library are listed
in this table:

.. csv-table::
   :header: "Command","Description"
   :widths: 30, 100

   "``python3 rmake.py -h``", "Help information."
   "``python3 rmake.py -d``", "Build the library dependencies and library in your local directory. The ``-d`` flag only has to be used once. For subsequent invocations of ``rmake.py`` it's not necessary to rebuild the dependencies."
   "``python3 rmake.py``", "Build the library in your local directory. It is assumed the dependencies have been built."
   "``python3 rmake.py -i``", "Build the library, then build and install the hipBLAS package in ``C:\\hipSDK``. You will be prompted for admin access. This installs it for all users. To restrict hipBLAS to your local directory, do not use the ``-i`` flag."
   "``python3 rmake.py -n``", "Build the library without the functionality provided by rocSOLVER. The rocSOLVER, rocSPARSE, and rocPRIM dependencies will not be required. This flag has no effect when building with a NVIDIA CUDA backend."


Building the library, client, and library and client dependencies
-------------------------------------------------------------------

The client contains the executables listed in the table below.

================= ====================================================
Executable name   Description
================= ====================================================
hipblas-test      Runs Google Tests to test the library
hipblas-bench     An executable to benchmark or test individual functions
hipblas-example-* Various examples showing how to use hipBLAS
================= ====================================================

Common ways to use ``rmake.py`` to build the dependencies, library, and client are
listed in this table.

.. csv-table::
   :header: "Command","Description"
   :widths: 33, 97

   "``python3 rmake.py -dc``", "Build the library dependencies, client dependencies, library, and client in your local directory. The ``-d`` flag only has to be used once. For subsequent invocations of ``rmake.py``, it is not necessary to rebuild the dependencies."
   "``python3 rmake.py -c``", "Build the library and client in your local directory. It is assumed the dependencies have been built."
   "``python3 rmake.py -idc``", "Build the library dependencies, client dependencies, library, and client, then build and install the hipBLAS package. You will be prompted for administrator access. To install hipBLAS for all users, use the ``-i`` flag. To restrict hipBLAS to your local directory, do not use the ``-i`` flag."
   "``python3 rmake.py -ic``", "Build and install the hipBLAS package and build the client. You will be prompted for administrator access. This installs it for all users. To restrict hipBLAS to your local directory, do not use the ``-i`` flag."

Dependencies for building the library
=====================================

Use ``rmake.py`` with the ``-d`` option to install the dependencies required to build the library.
This does not install the hipblas-common, rocBLAS, rocSOLVER, rocSPARSE, and rocPRIM dependencies.
When building hipBLAS, it is important to take note of the version dependencies for other libraries. The hipblas-common,
rocBLAS, and rocSOLVER versions required to build for an AMD backend are listed in the top-level ``CMakeLists.txt`` file.
rocSPARSE and rocPRIM are currently dependencies of rocSOLVER. To build these libraries from source,
see the :doc:`rocBLAS <rocblas:index>`,
:doc:`rocSOLVER <rocsolver:index>`, :doc:`rocSPARSE <rocsparse:index>`,
and :doc:`rocPRIM <rocprim:index>` documentation..

The minimum version of CMake is currently 3.16.8. See the ``--cmake_install`` flag in ``rmake.py`` to
upgrade automatically.

To use the test and benchmark clients' host reference functions, you must manually download and install
AMD's `ILP64 version of the AOCL libraries <https://www.amd.com/en/developer/aocl.html>`_ version 4.2.
If you download and run the full Windows AOCL installer to the default location (``C:\Program Files\AMD\AOCL-Windows\``),
then the AOCL reference BLAS (``amd-blis``) should be found by the clients' ``CMakeLists.txt`` file.

.. note::

   If you only use the ``rmake.py -d`` dependency script and change the default CMake option ``LINK_BLIS=ON``,
   you might experience ``hipblas-test`` stress test failures due to a 32-bit integer overflow
   on the host. To resolve this issue, exclude the stress tests using the command line argument ``--gtest_filter=-*stress*``.
