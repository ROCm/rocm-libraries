.. meta::
  :description: How to install rocBLAS on Windows
  :keywords: rocBLAS, ROCm, API, Linear Algebra, documentation, installation, building on Windows

.. _windows-install:

********************************************************************
Installing and building on Microsoft Windows
********************************************************************

This topic discusses how to install rocBLAS on Microsoft Windows from a prebuilt package or from source.

=====================================
Prerequisites
=====================================

rocBLAS requires an AMD HIP SDK-enabled platform. For more information,
see :doc:`System requirements for Windows <rocm-install-on-windows:reference/system-requirements>`.
rocBLAS is supported on the same Windows versions and toolchains that HIP SDK supports.

.. note::

   The :doc:`AMD HIP SDK documentation <rocm-install-on-windows:index>` contains more up-to-date Windows install information.

============================
Installing prebuilt packages
============================

rocBLAS can be installed on Windows 10 or 11 using the AMD HIP SDK installer.

The simplest way to use rocBLAS in your code is to use CMake. To install rocBLAS on Windows, follow these steps:

#. Add the SDK installation location to ``CMAKE_PREFIX_PATH`` in the CMake configuration command.

   .. code-block:: shell

      -DCMAKE_PREFIX_PATH="C:\Program Files\AMD\ROCm\5.5"

   .. note::

      You must use quotes around the path because it contains a space.

#. Add the following lines to ``CMakeLists.txt``:

   .. code-block:: shell

      find_package(rocblas)
      target_link_libraries( your_exe PRIVATE roc::rocblas )

Examples demonstrating how to build rocBLAS on Windows with CMake can be found at the
`rocBLAS-Examples GitHub page <https://github.com/ROCm/rocBLAS-Examples>`_.

After installation, use rocBLAS like any other library with a C API.
The ``rocblas.h`` header file must be included in the user code to make calls
into rocBLAS, while the rocBLAS shared library is link-time and run-time
dependent for the user application.

.. note::

   Run-time dependencies include the dynamic link library (``.dll``) file and the entire ``rocblas/``
   subdirectory, which is found in the HIP SDK ``bin`` folder. This folder
   can either reside in the same directory as ``rocblas.dll``
   or be moved elsewhere provided that the environment variable ``ROCBLAS_TENSILE_LIBPATH`` is set to the
   new location. The contents are read at execution time much like additional DLL files.

After installation, you can find ``rocblas.h`` in the HIP SDK ``\include\rocblas``
directory. Only use these two installed files when needed in user code.
You can find the other rocBLAS included files included in the HIP SDK in ``\include\rocblas\internal`` but
do not include these files directly into source code.

===============================
Building and installing rocBLAS
===============================

Most users do not need to build rocBLAS from source because it can be used after installing the prebuilt packages as described above.
If necessary, follow these instructions to build rocBLAS from source.
The rocBLAS codebase for the HIP SDK is the same as the one used for the Linux ROCm distribution.
However, because these two distributions have different stacks, the code and build process have subtle variations.


Requirements
------------

64GB of system memory is normally required for a full rocBLAS build. This value can be lower if
you build rocBLAS with a different Tensile logic target. See the ``--logic`` command from ``rmake.py --help``.


Download rocBLAS
----------------

The rocBLAS source code, which is the same as the ROCm Linux version, is available from the
`rocBLAS folder <https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocblas>`_
of the `rocm-libraries GitHub <https://github.com/ROCm/rocm-libraries>`_.
The ROCm HIP SDK version might appear in the default installation path,
but you can run the HIP SDK compiler to display the version from the ``bin/`` folder:

::

    hipcc --version

The HIP version has major, minor, and patch fields, possibly followed by a build-specific identifier.
For example, the HIP version might be ``5.4.22880-135e1ab4``.
This corresponds to major release = ``5``, minor release = ``4``, patch = ``22880``, and build identifier ``135e1ab4``.
The GitHub branches at the rocBLAS site have names like ``release/rocm-rel-major.minor``,
where the major and minor releases have the same meaning as in the ROCm version.


To download rocBLAS, including all projects in the rocm-libraries repository, use the following commands.

::

   git clone -b release/rocm-rel-x.y https://github.com/ROCm/rocm-libraries.git
   cd rocm-libraries/projects/rocblas

Replace ``x.y`` in the above command with the version of HIP SDK installed on your machine.
For example, if you have HIP 7.0 installed, use ``-b release/rocm-rel-7.0``.
You can add the SDK tools to your path using an entry like this:

::

   %HIP_PATH%\bin

To limit your local checkout to only the rocBLAS and Tensile projects, configure ``sparse-checkout`` before cloning.
This uses the Git partial clone feature (``--filter=blob:none``) to reduce how much data is downloaded.
Use the following commands for a sparse checkout:

::

   git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-libraries.git
   cd rocm-libraries
   git sparse-checkout init --cone
   git sparse-checkout set projects/rocblas shared/tensile
   git checkout develop # or use the branch you want to work with

.. note::

   To build ROCm 6.4 and older, use the rocBLAS repository at `<https://github.com/ROCm/rocBLAS>`_.
   For more information, see the documentation associated with the release you want to build.

Building rocBLAS
----------------

The following sections list the steps to build rocBLAS using the ``rmake.py`` script, which can install the dependencies.
You can build either:

* The dependencies and library

* The dependencies, library, and client

You only need the dependencies and library to call rocBLAS from your code.
The client contains the test and benchmark code.
``rmake.py`` prints the full ``cmake`` command used to configure rocBLAS to the screen
based on your ``rmake`` command line options.
The full ``cmake`` command can be used in build scripts to bypass the
Python helper script and use a fixed set of build options.

Library dependencies
--------------------

The dependencies installed by the Python script ``rdeps.py`` are listed in the ``rdeps.xml`` configuration file.
Passing the ``-d`` flag to ``rmake.py`` installs the dependencies the same way that
running ``rdeps.py`` directly does.
``rdeps.py`` uses ``vcpkg`` and ``pip`` to install the build dependencies.
``vcpkg`` is cloned into either the location defined by the environment variable ``VCPKG_PATH``
or the default ``C:\github\vckpg`` directory if the variable is undefined.
``pip`` is installed into your current Python 3 environment.

The top-level ``CMakeLists.txt`` file lists the minimum version requirement for CMake.
The version of CMake installed with Visual Studio 2022 meets this requirement.
The ``vcpkg`` version tag is specified at the top of the ``rdeps.py`` file.

However, for the host reference BLAS test and benchmark clients,
it is recommended that you manually download and install the AMD `ILP64 version of
AOCL-BLAS 4.2 <https://www.amd.com/en/developer/aocl.html>`_.
If you download and run the full Windows AOCL installer into the default location
(``C:\Program Files\AMD\AOCL-Windows\``), then the ``CMakeLists.txt`` file for the client can find the reference BLAS.

.. note::

   If you use OpenBLAS with the ``vcpkg`` version
   from ``rdeps.py`` instead of the AOCL reference library, you might experience ``rocblas-test`` stress test failures due to 32-bit integer overflow
   on the host reference code. If this occurs, exclude the ILP64 stress tests
   using the command line argument ``--gtest_filter=-*stress*``.


Building the library dependencies and library
---------------------------------------------


Common examples of how to use ``rmake.py`` to build the library dependencies and library are
shown in the table below:

.. note::

   You can run ``rmake.py`` from the ``projects\rocblas`` directory.

.. csv-table::
   :header: "Command","Description"
   :widths: 30, 100

   "``./rmake.py -h``", "Help information."
   "``./rmake.py -d``", "Build the library dependencies and library in your local directory. The ``-d`` flag only needs to be used once."
   "``./rmake.py``", "Build the library. It is assumed the dependencies have been built."
   "``./rmake.py -i``", "Build the library, then build and install the rocBLAS package. To keep rocBLAS in your local tree, do not use the ``-i`` flag."

Building the library, client, and all dependencies
-------------------------------------------------------------------

The client executables (``.exe`` files) are listed in the table below:

====================== ========================================================
Executable name        Description
====================== ========================================================
rocblas-test           Runs GoogleTest tests to validate the library
rocblas-bench          An executable to benchmark or test the functions
rocblas-example-sscal  Example C code that calls the ``rocblas_sscal`` function
====================== ========================================================

Common ways to use ``rmake.py`` to build the dependencies, library, and client are
listed in this table.

.. csv-table::
   :header: "Command","Description"
   :widths: 33, 97

   "``./rmake.py -h``", "Help information."
   "``./rmake.py -dc``", "Build the library dependencies, client dependencies, library, and client in your local directory. The ``-d`` flag only needs to be used once. For subsequent invocations of ``rmake.py``, it is not necessary to rebuild the dependencies."
   "``./rmake.py -c``", "Build the library and client in your local directory. It is assumed the dependencies have been installed."
   "``./rmake.py -idc``", "Build the library dependencies, client dependencies, library, and client, then build and install the rocBLAS package. To keep rocBLAS in your local directory, do not use the ``-i`` flag."
   "``./rmake.py -ic``", "Build and install the rocBLAS package and build the client. To keep rocBLAS in your local directory, do not use the ``-i`` flag."
   "``./rmake.py -t /path/to/Tensile``", "``tensile_tag.txt`` has been deprecated so use this option to build a folder that has a different Tensile commit than the rocBLAS commit."

Building the clients without the library
----------------------------------------

The rocBLAS clients can be built on their own by using ``rmake.py`` with a pre-existing rocBLAS library.

The version of the rocBLAS clients being built should match the version of the installed rocBLAS.
You can determine the version of the installed rocBLAS in the HIP SDK directory
from the file ``include\rocblas\internal\rocblas-version.h``.
If you have installed the ``grep`` utility, you can find the version of rocBLAS being built
by running the ``grep "VERSION_STRING" CMakeLists.txt`` command in the
rocBLAS directory where you are building the clients.

.. csv-table::
   :header: "Command","Description"
   :widths: 53, 77

   "``./rmake.py --clients-only``", "Build the rocBLAS clients and use the installed rocBLAS library at ``HIP_PATH`` if no ``--library-path`` is specified."
   "``./rmake.py --clients-only --library-path /path/to/rocBLAS``", "Build the rocBLAS clients and use the rocBLAS library at the specified location."
