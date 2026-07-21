.. meta::
  :description: How to install rocBLAS on Linux
  :keywords: rocBLAS, ROCm, API, Linear Algebra, documentation, installation, building on Linux

.. _linux-install:

**************************
Build and install on Linux
**************************

To build rocBLAS as part of the ROCm Core SDK, see `TheRock build
instructions
<https://github.com/ROCm/TheRock/blob/main/docs/development/README.md>`__.
TheRock is the recommended way to build ROCm components from source.

Alternatively, you can build rocBLAS standalone using the following
instructions.

Prerequisites
=============

rocBLAS requires a ROCm-enabled platform. For more information,
see :ref:`ROCm Core SDK components <rocm:release-components>`.

Static library
--------------

Non-standard static library builds have the additional runtime dependency of
the entire ``rocblas/`` subdirectory, which is located in the ``/opt/rocm/lib`` folder.
This runtime folder can be moved elsewhere, but the environment variable
``ROCBLAS_TENSILE_LIBPATH`` must be set to the new location. In addition, if you are running an executable
linked against the static library ``librocblas.a``, the build searches for the ``rocblas`` subdirectory in
the same directory as the executable.
The contents of the ``rocblas/`` subdirectory are read at execution time
in the same way as shared library files.
These files contain GPU code objects and metadata.

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

The checkout above omits other top-level trees (for example ``shared/ctest``). If you build the test
client (``BUILD_CLIENTS_TESTS=ON``) and want YAML-based CTest labels and the installed
``CTestTestfile.cmake`` from the shared categorization helpers, add ``shared/ctest`` to the
``git sparse-checkout set`` list (or use a full clone). Without ``shared/ctest`` present under
``ROCM_LIBRARIES_ROOT``, ``ROCBLAS_ENABLE_CTEST`` defaults to OFF. If you turn
``ROCBLAS_ENABLE_CTEST`` ON explicitly, configuration requires both ``clients/gtest/test_categories.yaml``
and ``shared/ctest/TestCategories.cmake`` to exist.

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


Client dependencies: BLAS library
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The rocBLAS test and benchmark clients require a host reference BLAS library. When building
clients with ``-dc`` flags, ``install.sh`` automatically builds AOCL 5.2 (AMD Optimizing CPU Libraries)
from source with ILP64 support. The build searches for BLAS libraries in this order:

1. AOCL 5.x (built or pre-installed)
2. AOCL 4.x BLIS
3. Bundled BLIS
4. System BLAS

To skip the automatic AOCL build and use an alternative, add ``--skip-aocl`` to the install command.
To clean and rebuild all dependencies, use ``--clean-deps``. You can also set ``AOCL_ROOT``
to specify a custom AOCL location, or manually install AOCL from
`AMD Developer Central <https://www.amd.com/en/developer/aocl.html>`_.

.. note::

   If using a BLAS library without ILP64 support, some stress tests might fail.
   To exclude these tests, use the ``--gtest_filter=-*stress*`` option.

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
   :widths: 40, 90

   "``./install.sh -h``", "Help information."
   "``./install.sh -dc``", "Build the library dependencies, client dependencies, library, and client in your local directory. Automatically builds AOCL 5.2. The ``-d`` flag only has to be used once. For subsequent invocations of ``install.sh``, it is not necessary to rebuild the dependencies."
   "``./install.sh -dc --skip-aocl``", "Build clients without AOCL, using the next available BLAS library."
   "``./install.sh -dc --clean-deps``", "Clean and rebuild all dependencies before building."
   "``./install.sh -c``", "Build the library and client in your local directory. It is assumed the dependencies have been built."
   "``./install.sh -idc``", "Build the library dependencies, client dependencies, library, and client, then build and install the rocBLAS package. You will be prompted for ``sudo`` access. To install rocBLAS for all users, use the ``-i`` flag. To restrict it to your local directory, do not use the ``-i`` flag."
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

Using the rocBLAS Docker images
------------------------------------------

The rocBLAS Docker images provide a reproducible, ready-to-use development environment to simplify
the set-up process. The Dockerfiles install all system dependencies, such as Clang, LLVM,
zstd, and the development libraries, schedule update alternatives, and configure the environment.
Two Dockerfiles are available for Ubuntu 24.04:

*  ``Dockerfile.ubuntu24.prebuilt``: Downloads a prebuilt ROCm nightly tarball from the `ROCm nightly builds <https://rocm.nightlies.amd.com>`_.
   This solution is faster to build and suitable for most development work. It lets you configure the target ASIC,
   nightly tag, tarball filename, and tarball source URL.
*  ``Dockerfile.ubuntu24.fullbuild``: Clones `<https://github.com/ROCm/TheRock>`_ and builds ROCm from source.
   It's the best choice for when a prebuilt tarball is unavailable or custom build options are required.
   The configuration options include the target ASIC, specific commit hash, build type (release, debug, or preset), CMake presets,
   and parallel job count.
   
For more information on how to download and use these images, see the
`rocBLAS Docker documentation <https://github.com/ROCm/rocm-libraries/blob/develop/projects/rocblas/docker/README.md>`_.
