.. meta::
   :description: Installation instructions for the hipBLASLt library
   :keywords: hipBLASLt, ROCm, library, API, installation, build

.. _installation:

*********************************
Building and installing hipBLASLt
*********************************

This topic describes how to build and install hipBLASLt on Linux systems.

Prerequisites
=============

To install hipBLASLt, your system must include these components:

*  A ROCm-enabled platform. For more information, see the :doc:`ROCm documentation <rocm:index>`.
*  A compatible version of :doc:`hipBLAS <hipblas:index>`.

Installing prebuilt packages
=============================

Download the prebuilt packages from the native package manager for your distribution.
For more information, see the :doc:`ROCm quick start installation guide <rocm-install-on-linux:install/quick-start>`.

.. code-block:: bash

   sudo apt update && sudo apt install hipblaslt

Building hipBLASLt using invoke
================================

hipBLASLt provides an `invoke <https://www.pyinvoke.org/>`_-based task runner for building and
installing hipBLASLt and its dependencies. This supports Linux and Windows (ROCm 7.0+).

.. note::

   To build ROCm 6.4 and older, use the hipBLASLt repository at `<https://github.com/ROCm/hipBLASLt>`_.
   Select the documentation associated with the release you want to build.

Setting up the environment
--------------------------

Create a virtual environment and install the Python build dependencies:

.. code-block:: bash

   python3 -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt

Building the library dependencies and library
---------------------------------------------

Here are some typical examples showing how to build the library:

.. csv-table::
   :header: "Command","Description"
   :widths: 40, 100

   "``inv --help build``", "Help information."
   "``inv build --install-deps``", "Install system dependencies and build the library."
   "``inv build``", "Build the library. Assumes dependencies are already installed."
   "``inv build --install-pkg``", "Build the library and install the hipBLASLt package."

Building the library, client, and all dependencies
---------------------------------------------------

The client contains the executables listed in the table below.

============================= ========================================================
Executable Name                Description
============================= ========================================================
``hipblaslt-test``             Runs GoogleTest tests to test the library
``hipblaslt-bench``            Executable to benchmark or test individual functions
============================= ========================================================

Here are some common ways to build the dependencies, library, and client:

.. csv-table::
   :header: "Command","Description"
   :widths: 40, 100

   "``inv --help build``", "Help information."
   "``inv build --install-deps --clients``", "Install system dependencies and build the library and client."
   "``inv build --clients``", "Build the library and client. Assumes dependencies are already installed."
   "``inv build --install-deps --clients --install-pkg``", "Build everything and install the hipBLASLt package."

Static library
--------------

To build a static library, use the ``--static`` option.
This produces a non-standard static library build. This means it has an additional runtime dependency
consisting of the entire ``hipblaslt/`` subdirectory, which is located in the ``/opt/rocm/lib`` folder.
You can move this folder, but you must set the environment variable ``HIPBLASLT_TENSILE_LIBPATH``
to the new location.

Dependencies
------------

Python dependencies are listed in ``requirements.txt`` and installed via ``pip install -r requirements.txt``.
System dependencies (such as compilers and NUMA) can be installed by passing ``--install-deps`` to ``inv build``.

Manual build for all supported platforms
========================================

This section provides information on how to configure CMake and build manually using individual commands.

Building the library manually
----------------------------------------

Before building hipBLASLt manually, ensure the following dependencies are installed on your system:

*  The `hipBLAS-common <https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipblas-common>`_ header files.
*  The `ROC-tracer (ROC-TX) <https://github.com/ROCm/rocm-systems/tree/develop/projects/roctracer>`_ library (this is typically pre-installed).

Building hipBLASLt
^^^^^^^^^^^^^^^^^^^^

To build hipBLASLt, run these commands:

.. code-block:: bash

   mkdir -p [HIPBLASLT_BUILD_DIR]/release
   cd [HIPBLASLT_BUILD_DIR]/release
   # Default install location is in /opt/rocm, define -DCMAKE_INSTALL_PREFIX=<path> to specify other
   # Default build config is 'Release', define -DCMAKE_BUILD_TYPE=<config> to specify other
   CXX=/opt/rocm/bin/amdclang++ ccmake [HIPBLASLT_SOURCE]
   make -j$(nproc)
   sudo make install # sudo required if installing into system directory such as /opt/rocm

Building the library, tests, benchmarks, and samples manually
-------------------------------------------------------------

The repository contains source code for clients that serve as samples, tests, and benchmarks.
You can find this code in the ``clients`` subdirectory.

Dependencies for the hipBLASLt clients
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The hipBLASLt samples have no external dependencies, but the unit test and benchmarking applications do.
These clients introduce the following dependencies:

- `LAPACK <https://github.com/Reference-LAPACK/lapack-release>`_,  which adds a dependency on a Fortran compiler
- `GoogleTest <https://github.com/google/googletest>`_

.. _building-hipblaslt-clients:

Building the hipBLASLt clients
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

GoogleTest and LAPACK are not easy to install. Many Linux distributions don't provide a GoogleTest package
with precompiled libraries and the LAPACK packages don't have the necessary CMake configuration files
to allow the ``cmake`` command to configure links with the ``cblas`` library. hipBLASLt provides an optional CMake script that builds
the above dependencies from source. You can provide your own builds for
these dependencies and help ``cmake`` find them by setting the ``CMAKE_PREFIX_PATH`` definition.
Follow this sequence of steps to build the dependencies and install them to the default CMake directory ``/usr/local``.

#. Build the dependencies from source (optional).

   .. code-block:: bash

      mkdir -p [HIPBLASLT_BUILD_DIR]/release/deps
      cd [HIPBLASLT_BUILD_DIR]/release/deps
      ccmake -DBUILD_BOOST=OFF [HIPBLASLT_SOURCE]/deps   # assuming boost is installed through package manager as above
      make -j$(nproc) install

#. After the dependencies are available on the system, configure the clients to build.
   This requires adding a few extra flags to the library CMake configuration script.
   If the dependencies are not installed in the system default directories, like ``/usr/local``,
   pass the ``CMAKE_PREFIX_PATH`` to ``cmake`` to help CMake find them.

   .. code-block:: bash

      -DCMAKE_PREFIX_PATH="<semicolon separated paths>"
      # Default install location is in /opt/rocm, use -DCMAKE_INSTALL_PREFIX=<path> to specify other
      CXX=/opt/rocm/bin/amdclang++ ccmake -DBUILD_CLIENTS_TESTS=ON -DBUILD_CLIENTS_BENCHMARKS=ON [HIPBLASLT_SOURCE]
      make -j$(nproc)
      sudo make install   # sudo required if installing into system directory such as /opt/rocm
