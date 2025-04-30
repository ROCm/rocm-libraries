.. meta::
  :description: hipSOLVER installation guide
  :keywords: hipSOLVER, rocSOLVER, ROCm, API, documentation, install

.. _install-linux:

*********************************
Installing and building hipSOLVER
*********************************

This topic explains how to install and build the hipSOLVER library on Linux.

Install using prebuilt packages
===============================

Download the ROCm packages from the package servers following the :doc:`ROCm installation dcocumentation <rocm-install-on-linux:index>`.
Run the following command to install hipSOLVER using the package manager.

.. code-block:: shell

   sudo apt update && sudo apt install hipsolver

.. note::

   Updates for each release are listed in the ``CHANGELOG.md`` file under the releases section of the
   `hipSOLVER GitHub page <https://github.com/ROCm/hipSOLVER>`_.


Build and install the library using a script (Ubuntu only)
==========================================================

The root directory of the `hipSOLVER GitHub repository <https://github.com/ROCm/hipSOLVER>`_ includes the ``install.sh`` bash
script for building and installing hipSOLVER on Ubuntu with a single command.
The script does not accept many options and hardcodes most configuration attributes,
but it's a great way to get started quickly and serves as an example of how to build and install hipSOLVER.
A more extensive set of options can be specified by invoking ``cmake`` directly.
A few commands in the script require ``sudo`` access,
so it might prompt you for a password.

Here are some typical examples of using the script:

*  ``./install.sh -id``: Build the library and dependencies and install them (the ``-d`` flag only needs to be passed once on a system).
*  ``./install.sh -ic``: Build library and clients (tests, benchmarks, and samples) and install them.
*  ``./install.sh --cuda``: Build the library on a NVIDIA CUDA-enabled machine, with CUDA cuSOLVER as the backend.

To list more options, use the ``-h`` (help) option of the install script.

.. code-block:: shell

   ./install.sh -h

Build and install the library manually
======================================

For a standard library installation, follow these steps:

.. code-block:: bash

   mkdir -p <HIPSOLVER_BUILD_DIR_PATH>/release
   cd <HIPSOLVER_BUILD_DIR_PATH>/release
   CXX=/opt/rocm/bin/hipcc cmake <HIPSOLVER_SOURCE_DIR_PATH>
   make -j$(nproc)
   sudo make install

``sudo`` is required to install hipSOLVER to a system directory, such as ``/opt/rocm``,
which is the default location.

*  Use ``-DCMAKE_INSTALL_PREFIX=<other_path>`` to specify a different install directory.
*  Use ``-DCMAKE_BUILD_TYPE=<other_configuration>`` to specify a build configuration, such as ``Debug``.
   The default build configuration is ``Release``.

Library dependencies
---------------------

The hipSOLVER library has two separate sets of dependencies, depending on the backend being used.

The NVIDIA CUDA backend has a dependency on cuSOLVER.

The ROCm (rocSOLVER) backend has the following dependencies:

*  `rocSOLVER <https://github.com/ROCm/rocSOLVER>`_
*  `rocBLAS <https://github.com/ROCm/rocBLAS>`_
*  `rocSPARSE <https://github.com/ROCm/rocSPARSE>`_ (optional)
*  `SuiteSparse <https://github.com/DrTimothyAldenDavis/SuiteSparse>`_:
   ``CHOLMOD`` and ``SuiteSparse_config`` modules (optional)

rocSOLVER itself depends on rocBLAS and rocSPARSE, therefore all three libraries should be installed
as part of a standard rocSOLVER installation. For more information
about building and installing rocSOLVER, see the :doc:`rocSOLVER installation guide <rocsolver:installation/installlinux>`.

SuiteSparse is a third-party library which can be installed using the package managers of most distributions.
Together with rocSPARSE, it is used to provide
functionality for the ``hipsolverSp`` API. By default, both libraries are run-time dependencies.
They are dynamically loaded by ``dlopen`` if they are
present on the system. If the ``BUILD_WITH_SPARSE`` option is set to ``ON``,
they become build-time dependencies instead and must be present on the
system to build hipSOLVER.

.. code-block:: shell

   DBUILD_WITH_SPARSE=ON

Build the library, tests, benchmarks, and samples manually
==========================================================

The repository contains source code for client programs that serve as tests, benchmarks, and samples.
The client source code can be found in the ``clients`` subdirectory.

Client dependencies
--------------------

The hipSOLVER samples have no external dependencies, but the unit test and benchmarking applications do.
These clients introduce the following dependencies:

*  `LAPACK <https://github.com/Reference-LAPACK/lapack-release>`_ (Adds a dependency on a Fortran compiler)
*  `GoogleTest <https://github.com/google/googletest>`_
*  `hipBLAS <https://github.com/ROCm/hipBLAS>`_ (Optional)
*  `hipSPARSE <https://github.com/ROCm/hipSPARSE>`_ (Optional, but required with the default settings)

Many distributions do not provide a GoogleTest package with precompiled libraries,
and the LAPACK packages do not have the necessary CMake configuration files to
link to the CBLAS library. hipSOLVER provides a CMake script that builds
LAPACK and GoogleTest from source. This is an optional step because you can provide your own builds
of these dependencies, setting ``CMAKE_PREFIX_PATH`` to let CMake find them.

The following sequence of steps builds the dependencies and installs them to the default CMake directory ``/usr/local``:

.. code-block:: bash

   mkdir -p <HIPSOLVER_BUILD_DIR_PATH>/release/deps
   cd <HIPSOLVER_BUILD_DIR_PATH>/release/deps
   cmake -DBUILD_BOOST=OFF <HIPSOLVER_SOURCE_PATH>/deps   #assuming boost is installed through package manager as above
   make -j$(nproc) install

hipBLAS is only required if the ``BUILD_HIPBLAS_TESTS`` option is set to ``ON``. It's used to ensure
compatibility between the hipBLAS enumerations defined
separately by hipBLAS and hipSOLVER. hipSPARSE is required by default but the dependency is
ignored if the ``BUILD_HIPSPARSE_TESTS`` option is set to ``OFF``. It's used
to create objects required by tests for the ``hipsolverSp`` API.

.. code-block:: shell

   DBUILD_HIPBLAS_TESTS=ON
   DBUILD_HIPSPARSE_TESTS=OFF

Both libraries can be installed the same way as hipSOLVER. For example, the install scripts for
hipBLAS and hipSPARSE can be invoked to build and
install those libraries using the following command:

.. code-block:: shell

   ./install.sh -i

For more details, see the :doc:`hipBLAS <hipblas:index>`
and :doc:`hipSPARSE <hipsparse:index>` documentation.

Library and clients
--------------------

After the dependencies are installed on the system, you can configure which clients to build.
This requires adding a few extra flags to the CMake configure script for the library.
If the dependencies are not installed into the default system locations, like ``/usr/local``,
pass the ``CMAKE_PREFIX_PATH`` to CMake so it can find them.

.. code-block:: bash

   -DCMAKE_PREFIX_PATH="<semicolon separated paths>"

Follow this example to build the library and clients:

.. code-block:: bash

   CXX=/opt/rocm/bin/hipcc cmake -DBUILD_CLIENTS_TESTS=ON -DBUILD_CLIENTS_BENCHMARKS=ON [HIPSOLVER_SOURCE]
   make -j$(nproc)
   sudo make install   # sudo required if installing into system directory such as /opt/rocm
