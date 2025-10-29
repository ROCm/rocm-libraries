.. meta::
  :description: How to build and install rocSPARSE on Linux
  :keywords: rocSPARSE, ROCm, API, documentation, installation, Linux, build

.. _linux-install:

********************************************************************
Installing and building rocSPARSE for Linux
********************************************************************

This topic describes how to install or build rocSPARSE on Linux by using prebuilt packages or building from source.
For information on installing and building rocSPARSE on Microsoft Windows, see :doc:`rocSPARSE for Windows <./Windows_Install_Guide>`.

Prerequisites
=============

rocSPARSE requires a ROCm enabled platform. For more information, including a list of supported
GPUs and Linux distributions, see the :doc:`ROCm on Linux install guide <rocm-install-on-linux:index>`.

Installing pre-built packages
=============================

Use the following commands to install rocSPARSE on Ubuntu or Debian:

.. code-block:: shell

   sudo apt-get update
   sudo apt-get install rocsparse

Use the following commands to install rocSPARSE on RHEL-based platforms:

.. code-block:: shell

   sudo yum update
   sudo yum install rocsparse

Use the following commands to install rocSPARSE on SLES:

.. code-block:: shell

   sudo dnf upgrade
   sudo dnf install rocsparse

After rocSPARSE is installed, it can be used just like any other library with a C API.
To call rocSPARSE, the header file must be included in the user code.
This means the rocSPARSE shared library becomes a link-time and run-time dependency for the user application.

Building rocSPARSE from source
==============================

It isn't necessary to build rocSPARSE from source because it's ready to use after installing
the prebuilt packages, as described above.
To build rocSPARSE from source, follow the instructions in this section.

Requirements
------------

To compile and run rocSPARSE, the `AMD ROCm Platform <https://github.com/ROCm/ROCm>`_ is required.
Building rocSPARSE from source also requires the following components and dependencies:

*  `git <https://git-scm.com/>`_
*  `CMake <https://cmake.org/>`_ (Version 3.5 or later)
*  `rocPRIM <https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocprim>`_
*  `rocBLAS <https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocblas>`_ (Optional: for the library)
*  `GoogleTest <https://github.com/google/googletest>`_ (Optional: only required to build the clients)
*  `Python <https://www.python.org/>`_
*  `PyYAML <https://pypi.org/project/PyYAML/>`_

When building rocSPARSE from source, select supported versions of the math library
dependencies (:doc:`rocPRIM <rocprim:index>` and optionally :doc:`rocBLAS <rocblas:index>`). Given a version of rocSPARSE,
you must use a version of rocPRIM (and optionally rocBLAS) that is the same or later. For example, it's
possible to build rocSPARSE 3.2.0 with any future rocPRIM 3.Y.Z version (with the same major version
and where 3.Y.Z is 3.2.0 or later), but compiling rocSPARSE with an older version of
rocPRIM, such as 3.1.0, is not supported.

Downloading rocSPARSE
----------------------

The rocSPARSE source code is available from the `rocSPARSE folder <https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocsparse>`_
of the `rocm-libraries GitHub <https://github.com/ROCm/rocm-libraries>`_.

To limit your local checkout to only the rocSPARSE project, configure ``sparse-checkout`` before cloning.
This uses the Git partial clone feature (``--filter=blob:none``) to reduce how much data is downloaded.
Use the following commands for a sparse checkout:

.. note::

   To include the rocPRIM and rocBLAS dependencies, set the projects for the sparse checkout using
   ``git sparse-checkout set projects/rocsparse projects/rocprim projects/rocblas``.

.. code-block:: shell

   git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-libraries.git
   cd rocm-libraries
   git sparse-checkout init --cone
   git sparse-checkout set projects/rocsparse
   git checkout develop # or use the branch you want to work with

To download the develop branch for all projects in rocm-libraries, use these commands. This process takes
longer but is recommended for those working with a large number of libraries.

.. code-block:: shell

   git clone -b develop https://github.com/ROCm/rocm-libraries.git
   cd rocm-libraries/projects/rocsparse

.. note::

   To build ROCm 6.4.3 and earlier, use the rocSPARSE repository at `<https://github.com/ROCm/rocSPARSE>`_.
   For more information, see the documentation associated with the release you want to build.

Building rocSPARSE using the install script
-------------------------------------------

It's recommended to use the ``install.sh`` script to install rocSPARSE.
Here are the steps required to build different packages of the library, including the dependencies and clients.

.. note::

   You can run the ``install.sh`` script from the ``projects/rocsparse`` directory.

Using install.sh to build rocSPARSE with dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following table lists the common ways to use ``install.sh`` to build the rocSPARSE dependencies and library.

.. csv-table::
   :header: "Command","Description"
   :widths: 40, 90

   "``./install.sh -h``", "Print the help information."
   "``./install.sh -d``", "Build the dependencies and library in your local directory. The ``-d`` flag only needs to be used once. For subsequent invocations of the script, it isn't necessary to rebuild the dependencies."
   "``./install.sh``", "Build the library in your local directory. The script assumes the dependencies are available."
   "``./install.sh -i``", "Build the library, then build and install the rocSPARSE package in ``/opt/rocm/rocsparse``. The script prompts you for ``sudo`` access. This installs rocSPARSE for all users."
   "``./install.sh -i -a gfx908``", "Build the library specifically for the gfx908 architecture, then build and install the rocSPARSE package in ``/opt/rocm/rocsparse``. The script prompts you for ``sudo`` access. This installs rocSPARSE for all users."

Using install.sh to build rocSPARSE with dependencies and clients
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The clients contain example code, unit tests, and benchmarks. Common use cases of ``install.sh`` to build
the library, dependencies, and clients are listed in the table below.

.. csv-table::
   :header: "Command","Description"
   :widths: 40, 90

   "``./install.sh -h``", "Print the help information."
   "``./install.sh -dc``", "Build the dependencies, library, and client in your local directory. The ``-d`` flag only needs to be used once. For subsequent invocations of the script, it isn't necessary to rebuild the dependencies"
   "``./install.sh -c``", "Build the library and client in your local directory. The script assumes the dependencies are available."
   "``./install.sh -idc``", "Build the library, dependencies, and client, then build and install the rocSPARSE package in ``/opt/rocm/rocsparse``. The script prompts you for ``sudo`` access. This installs rocSPARSE for all users."
   "``./install.sh -ic``", "Build the library and client, then build and install the rocSPARSE package in ``opt/rocm/rocsparse``. The script prompts you for ``sudo`` access. This installs rocSPARSE for all users."
   "``./install.sh -idc -a gfx908``", "Build the library specifically for the gfx908 architecture, build the dependencies and client, then build and install the rocSPARSE package in ``/opt/rocm/rocsparse``. The script prompts you for ``sudo`` access. This installs rocSPARSE for all users."
   "``./install.sh -ic -a gfx908``", "Build the library specifically for the gfx908 architecture, build the client, then build and install the rocSPARSE package in ``opt/rocm/rocsparse``. The script prompts you for ``sudo`` access. This installs rocSPARSE for all users."
   "``./install.sh -o``", "Build the client executables using an already installed version of the library."

Building rocSPARSE using individual make commands
-------------------------------------------------

The rocSPARSE library contains both host and device code, therefore, the HIP compiler
must be specified during the CMake configuration process.

You can build rocSPARSE using the following commands:

.. note::

   Run these commands from the ``projects/rocsparse`` directory.

.. note::

   CMake 3.5 or later is required to build rocSPARSE.

.. code-block:: shell

   # Create and change to build directory
   mkdir -p build/release ; cd build/release

   # Default install path is /opt/rocm, use -DCMAKE_INSTALL_PREFIX=<path> to adjust it
   CXX=/opt/rocm/bin/amdclang++ cmake ../..

   # Compile rocSPARSE library
   make -j$(nproc)

   # Install rocSPARSE to /opt/rocm
   make install


You can build rocSPARSE with the dependencies and clients using the following commands:

.. note::

   GoogleTest is required to build rocSPARSE clients.

.. code-block:: shell

   # Install GoogleTest
   mkdir -p build/release/deps ; cd build/release/deps
   cmake ../../../deps
   make -j$(nproc) install

   # Change to build directory
   cd ..

   # Default install path is /opt/rocm, use -DCMAKE_INSTALL_PREFIX=<path> to adjust it
   CXX=/opt/rocm/bin/amdclang++ cmake ../.. -DBUILD_CLIENTS_TESTS=ON \
                                          -DBUILD_CLIENTS_BENCHMARKS=ON \
                                          -DBUILD_CLIENTS_SAMPLES=ON

   # Compile rocSPARSE library
   make -j$(nproc)

   # Install rocSPARSE to /opt/rocm
   make install

Resolving common build problems
-------------------------------------------------

If you encounter the error message
"Could not find a package configuration file provided by "ROCm" with any of the following names: ROCMConfig.cmake, rocm-config.cmake"
during the build, install the `ROCm CMake modules <https://github.com/ROCm/rocm-cmake>`_.

Testing rocSPARSE
==============================

You can test the rocSPARSE installation by running one of the rocSPARSE examples
after successfully compiling the library with the clients.

.. code-block:: shell

      # Navigate to clients binary directory
      cd build/release/clients/staging

      # Execute rocSPARSE example
      ./example_csrmv 1000
