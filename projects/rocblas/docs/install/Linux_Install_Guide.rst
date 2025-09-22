.. meta::
  :description: How to install rocBLAS on Linux
  :keywords: rocBLAS, ROCm, API, Linear Algebra, documentation, installation, building on Linux

.. _linux-install:

********************************************************************
Installing and building on Linux
********************************************************************

This topic discusses how to install rocBLAS on Linux from a prebuilt package or from source.

Prerequisites
===================================

rocBLAS requires a ROCm-enabled platform. For more information,
see the :doc:`system requirements <rocm-install-on-linux:reference/system-requirements>`.

Installing prebuilt packages
===================================

Install rocBLAS on Linux using the appropriate package manager for your distribution.
For example, on Ubuntu or Debian, use these commands:

.. code-block:: shell

   sudo apt-get update
   sudo apt-get install rocblas

For Fedora, CentOS, or RHEL, use these commands:

.. code-block:: shell

   sudo dnf update
   sudo dnf install rocblas

If you are compiling a program that will link against rocBLAS, instead install the ``rocblas-dev`` package.
Similarly, at the top level, install ``rocm-dev`` instead of ``rocm``.
For example, on Ubuntu or Debian, use these commands:

.. code-block:: shell

   sudo apt-get update
   sudo apt-get install rocblas-dev

For Fedora, CentOS, or RHEL, use these commands:

.. code-block:: shell

   sudo dnf update
   sudo dnf install rocblas-devel

If you are using a different distribution, use the appropriate package manager to install the ``rocblas`` or ``rocblas-dev`` package.

The ``rocblas`` package contains the shared library and runtime only requirements.
The ``rocblas-dev`` package includes the header files and depends on the ``rocblas`` package.
The development package will also include the static library when using static library packages.
The ``rocblas-dev`` package is not required to run the library but is required to compile code that uses rocBLAS.

After installation of the development package which includes the headers, use rocBLAS like any other library with a C API.
The ``rocblas.h`` header file must be included in the user code to make calls
into rocBLAS, while the rocBLAS shared library is link-time and run-time
dependent for the user application.

The header files ``rocblas.h`` and ``rocblas_module.f90`` are installed in ``/opt/rocm/include/rocblas``.
The library file ``librocblas.so`` is installed in ``/opt/rocm/lib``.


Static library
----------------

Non-standard static library builds have the additional runtime dependency of
the entire ``rocblas/`` subdirectory, which is located in the ``/opt/rocm/lib`` folder.
This runtime folder can be moved elsewhere, but the environment variable
``ROCBLAS_TENSILE_LIBPATH`` must be set to the new location. In addition, if you are running an executable
linked against the static library ``librocblas.a``, the build searches for the ``rocblas`` subdirectory in
the same directory as the executable.
The contents of the ``rocblas/`` subdirectory are read at execution time
in the same way as shared library files.
These files contain GPU code objects and metadata.

Building and installing rocBLAS
===================================

For most users, it isn't necessary to build rocBLAS from source. They can use
the prebuilt packages as described above. However, if necessary, you can use the following instructions to build
rocBLAS from source.  You need to install ``rocm-dev`` instead of ``rocm`` (or in addition to ``rocm``), because ``rocm`` contains the runtime-only packages.
For example, on Ubuntu or Debian, use these commands:

.. code-block:: shell

   sudo apt-get update
   sudo apt-get install rocm-dev

For Fedora, CentOS, or RHEL, use these commands:

.. code-block:: shell

   sudo dnf update
   sudo dnf install rocm-dev

Requirements
------------

Normally, a full rocBLAS fat binary build requires 64 GB of system memory. This value might be lower if
you build rocBLAS for specific architectures using the ``-a`` option for ``install.sh``. For more information,
run the following help command:

.. code-block:: shell

   ./install.sh --help

.. note::

   You can run the ``install.sh`` script from the ``projects/rocblas`` directory.

Download rocBLAS
----------------

The rocBLAS source code is available from the `rocBLAS folder <https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocblas>`_
of the `rocm-libraries GitHub <https://github.com/ROCm/rocm-libraries>`_.
Verify the ROCm version on your system. On an Ubuntu distribution, use:

.. code-block:: shell

   apt show rocm-libs -a

For distributions that use the ``yum`` package manager, run this command:

.. code-block:: shell

    yum info rocm-libs

The ROCm version has major, minor, and patch fields, possibly followed by a build-specific identifier.
For example, the ROCm version might be ``4.0.0.40000-23``. This corresponds to major release = ``4``,
minor release = ``0``, patch = ``0``, and build identifier ``40000-23``.
The GitHub branches at the rocBLAS site have names like ``rocm-major.minor.x``,
where the major and minor releases have the same meaning as in the ROCm version.

To download rocBLAS, including all projects in the rocm-libraries repository, use the following commands.

.. code-block:: shell

   git clone -b release/rocm-rel-x.y https://github.com/ROCm/rocm-libraries.git
   cd  rocm-libraries/projects/rocblas

Replace ``x.y`` in the above command with the ROCm version installed on your machine.
For example, if you have ROCm 7.0 installed, replace ``release/rocm-rel-x.y`` with ``release/rocm-rel-7.0``.

To limit your local checkout to only the rocBLAS and Tensile projects, configure ``sparse-checkout`` before cloning.
This uses the Git partial clone feature (``--filter=blob:none``) to reduce how much data is downloaded.
Use the following commands for a sparse checkout:

.. code-block:: shell

   git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-libraries.git
   cd rocm-libraries
   git sparse-checkout init --cone
   git sparse-checkout set projects/rocblas shared/tensile
   git checkout develop # or use the branch you want to work with

.. note::

   To build ROCm 6.4 and older, use the rocBLAS repository at `<https://github.com/ROCm/rocBLAS>`_.
   For more information, see the documentation associated with the release you want to build.

The rocBLAS source code is found in the ``projects/rocblas`` directory.
The following sections list the steps to build rocBLAS using the ``install.sh`` script.
You can build either:

* The dependencies and library

* The dependencies, library, and client

You only need the dependencies and library to call rocBLAS from your code.
The client contains the test and benchmark code.

Library dependencies
--------------------

CMake has a minimum version requirement, which is listed in the ``install.sh`` script.
See the ``--cmake_install`` flag in ``install.sh`` to upgrade automatically.

The ROCm hipBLASLt dependency is not installed when using ``-d`` flag with ``install.sh``, so install it
manually using the native package manager for your distribution.
For example, on Ubuntu or Debian, use:

.. code-block:: shell

   sudo apt-get update
   sudo apt-get install hipblaslt-dev

For Fedora, CentOS, or RHEL, use:

.. code-block:: shell

   sudo dnf update
   sudo dnf install hipblaslt-devel

For other distributions, use the appropriate package manager to install the ``hipblaslt-dev`` or ``hipblaslt-devel`` package.

Other dependencies are listed in the ``install.sh`` script.
Pass the ``-d`` flag to ``install.sh`` to install these dependencies.


However, for the host reference BLAS test and benchmark clients,
it is recommended that you manually download and install the AMD `ILP64 version of
AOCL-BLAS 4.2 <https://www.amd.com/en/developer/aocl.html>`_.
If you download and install the full AOCL packages into their default locations
or download the BLIS archive files and extract them into the build directory ``deps`` subfolder,
then the client's ``CMakeLists.txt`` should find the reference BLAS.

.. note::

   If you only use the ``install.sh -d`` script-based BLIS download and install,
   you might experience ``rocblas-test`` stress test failures due to 32-bit integer overflow on the host.
   If this occurs, exclude the stress tests using the command line argument ``--gtest_filter=-*stress*``.

Building the library dependencies and library
---------------------------------------------

Common examples of how to use ``install.sh`` to build the library dependencies and library are
shown in the table below:

.. csv-table::
   :header: "Command","Description"
   :widths: 30, 100

   "``./install.sh -h``", "Help information."
   "``./install.sh -d``", "Build the library dependencies and library in your local directory. The ``-d`` flag only needs to be used once. For subsequent invocations of ``install.sh``, it is not necessary to rebuild the dependencies."
   "``./install.sh --upgrade_tensile_venv_pip``", "On RPM-based Linux distributions it may be required to upgrade ``pip`` or ``packaging`` for the build to succeed."
   "``./install.sh``", "Build the library in your local directory. It is assumed the dependencies have been built."
   "``./install.sh -i``", "Build the library, then build and install the rocBLAS package in ``/opt/rocm/rocblas``. You will be prompted for ``sudo`` access. This installs it for all users. To keep rocBLAS in your local directory, do not use the ``-i`` flag."


Building the library, client, and all dependencies
-------------------------------------------------------------------

This section explains how to build the library, client, library dependencies, and client dependencies.
The client contains the executables listed in the table below.

====================== ========================================================
Executable name        Description
====================== ========================================================
rocblas-test           Runs GoogleTest tests to validate the library
rocblas-bench          An executable to benchmark or test the functions
rocblas-example-sscal  Example C code that calls the ``rocblas_sscal`` function
====================== ========================================================

Common ways to use ``install.sh`` to build the dependencies, library, and client are
listed in this table.

.. csv-table::
   :header: "Command","Description"
   :widths: 33, 97

   "``./install.sh -h``", "Help information."
   "``./install.sh -dc``", "Build the library dependencies, client dependencies, library, and client in your local directory. The ``-d`` flag only has to be used once. For subsequent invocations of ``install.sh``, it is not necessary to rebuild the dependencies."
   "``./install.sh -c``", "Build the library and client in your local directory. It is assumed the dependencies have been built."
   "``./install.sh -idc``", "Build the library  dependencies, client dependencies, library, and client, then build and install the rocBLAS package. You will be prompted for ``sudo`` access. To install rocBLAS for all users, use the ``-i`` flag. To restrict it to your local directory, do not use the ``-i`` flag."
   "``./install.sh -ic``", "Build and install the rocBLAS package and build the client. You will be prompted for ``sudo`` access. This installs it for all users. To restrict rocBLAS to your local directory, do not use the ``-i`` flag."
   "``./install.sh -t /path/to/Tensile``", "``tensile_tag.txt`` has been deprecated so use this option to build a folder that has a different Tensile commit than the rocBLAS commit."

Building the clients without the library
------------------------------------------

You can use ``install.sh`` to build the rocBLAS clients on their own with a pre-existing rocBLAS library using
one of these commands.

.. note::

   The version of the rocBLAS clients being built should match the installed rocBLAS version.
   You can find the installed rocBLAS version in ``include/internal/rocblas-version.h`` in the
   directory where rocBLAS is installed. To find the version of the rocBLAS clients being built,
   run ``grep "VERSION_STRING" CMakeLists.txt`` in the directory where you are building rocBLAS.

.. csv-table::
   :header: "Command","Description"
   :widths: 53, 77

   "``./install.sh --clients-only``", "Build the rocBLAS clients and use the installed rocBLAS library at ``ROCM_PATH`` (defaults to ``/opt/rocm`` if not specified)."
   "``./install.sh --clients-only --library-path /path/to/rocBLAS``", "Build the rocBLAS clients and use the rocBLAS library at the specified location."
