.. meta::
  :description: How to build and install rocSPARSE on Windows
  :keywords: rocSPARSE, ROCm, API, documentation, installation, Windows, build

.. _windows-install:

********************************************************************
Installing and building rocSPARSE for Microsoft Windows
********************************************************************

This topic describes how to install or build rocSPARSE on Microsoft Windows by using prebuilt packages or building from source.
For information on installing and building rocSPARSE on Linux, see :doc:`rocSPARSE for Linux <./Linux_Install_Guide>`.

Prerequisites
=============

rocSPARSE on Windows requires an AMD HIP SDK-enabled platform. It's supported on the
same Windows versions and toolchains that the HIP SDK supports. For more information, see
:doc:`HIP SDK installation for Windows <rocm-install-on-windows:index>`.

Installing prebuilt packages
============================

rocSPARSE can be installed on Windows 10 or 11 using the AMD HIP SDK installer.

The simplest way to add rocSPARSE to your code is to use CMake.
Add the SDK installation location to your ``CMAKE_PREFIX_PATH``.

.. note::

   You must use quotes because the path contains a space.

.. code-block:: shell

   -DCMAKE_PREFIX_PATH="C:\Program Files\AMD\ROCm\7.0"

In your ``CMakeLists.txt`` file, use these lines:

.. code-block:: shell

   find_package(rocsparse)
   target_link_libraries( your_exe PRIVATE roc::rocsparse )

After rocSPARSE is installed, it can be used just like any other library with a C API.
To call rocSPARSE, the ``rocsparse.h`` header file must be included in the user code.
This means the rocSPARSE  import library and dynamic link library respectively become link-time and run-time dependencies
for the user application.

After the installation, you can find ``rocsparse.h`` in the HIP SDK ``\\include\\rocsparse``
directory. When you need to include rocSPARSE in your application code, you must only use these two files.
You can find the other rocSPARSE files included in the HIP SDK ``\\include\\rocsparse\\internal`` directory, but
do not include these files directly in your source code.

Building rocSPARSE from source
=================================

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
*  `vcpkg <https://github.com/Microsoft/vcpkg.git>`_
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

The rocSPARSE source code for Windows, which is the same as for Linux, is available
from the `rocSPARSE folder <https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocsparse>`_
of the `rocm-libraries GitHub <https://github.com/ROCm/rocm-libraries>`_.
The ROCm HIP SDK version might be shown in the default installation path, but
you can run the HIP SDK compiler from the ``bin/`` folder to display the version using this command:

.. code-block:: shell

   hipcc --version

The HIP version number consists of major, minor, and patch fields, and is sometimes followed by a build-specific identifier.
For example, the HIP version might be ``5.4.22880-135e1ab4``.
This corresponds to major release ``5``, minor release ``4``, patch ``22880``, and build identifier ``135e1ab4``.
The rocSPARSE GitHub includes branches with names like ``release/rocm-rel-major.minor``,
where major and minor have the same meaning as the HIP version.

To limit your local checkout to only the rocSPARSE project, configure ``sparse-checkout`` before cloning.
This uses the Git partial clone feature (``--filter=blob:none``) to reduce the data download.
Use the following commands for a sparse checkout:

.. note::

   To include the rocPRIM and rocBLAS dependencies, set the projects for the sparse checkout using
   ``git sparse-checkout set projects/rocsparse projects/rocprim projects/rocblas``.

.. code-block:: shell

   git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-libraries.git
   cd rocm-libraries
   git sparse-checkout init --cone
   git sparse-checkout set projects/rocsparse
   git checkout release/rocm-rel-x.y  # or use the branch you want to work with

Replace ``x.y`` in the above command with the version of HIP SDK installed on your machine.
For example, if you have HIP 7.0 installed, use ``-b release/rocm-rel-7.0``.

To download all projects in rocm-libraries, use these commands. This process takes
longer but is recommended for those working with a large number of libraries.

.. code-block:: shell

   git clone -b release/rocm-rel-x.y https://github.com/ROCm/rocm-libraries.git
   cd rocm-libraries/projects/rocsparse

.. note::

   To build ROCm 6.4.3 and earlier, use the rocSPARSE repository at `<https://github.com/ROCm/rocSPARSE>`_.
   For more information, see the documentation associated with the release you want to build.

Add the SDK tools to your path with an entry like the following:

.. code-block:: shell

   %HIP_PATH%\bin

Building using the Python script
---------------------------------

This section describes the steps required to build rocSPARSE using the ``rmake.py`` script. You can build:

*  The library
*  The library and client

To call rocSPARSE from your code, you only need the library. The client contains testing and benchmark tools.
``rmake.py`` prints the full ``cmake`` command being used to configure rocSPARSE based on the ``rmake`` command line options.
This full ``cmake`` command can be used in your own build scripts to bypass the Python helper script for a fixed set of build options.

Building the library from source
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following table lists the common ways to use ``rmake.py`` to build the rocSPARSE library only.

.. note::

   You can run ``rmake.py`` from the ``projects\rocsparse`` directory.

.. csv-table::
   :header: "Command","Description"
   :widths: 40, 90

   "``./rmake.py -h``","Print the help information."
   "``./rmake.py``","Build the library."
   "``./rmake.py -i``","Build the library, then build and install the rocSPARSE package. To keep rocSPARSE in your local tree, do not use the ``-i`` flag."
   "``./rmake.py -in``","Build the library without rocBLAS, then build and install the rocSPARSE package. To keep rocSPARSE in your local tree, do not use the ``-i`` flag."
   "``./rmake.py -i -a gfx900``","Build the library using only the gfx900 architecture, then build and install the rocSPARSE package. To keep rocSPARSE in your local tree, do not use the ``-i`` flag."

Building the library and client from source
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The client executables (``.exe`` files) are listed in the table below:

.. csv-table::
   :header: "Executable name","Description"
   :widths: 40, 90

   "rocsparse-test","Runs Google Tests to test the library"
   "rocsparse-bench","An executable to benchmark and test functions"
   "rocsparse_axpyi","Example C code that calls the ``rocsparse_axpyi`` function"

The following table lists the common ways to use ``rmake.py`` to build the rocSPARSE library and client.

.. csv-table::
   :header: "Command","Description"
   :widths: 40, 90

   "``./rmake.py -h``","Print the help information."
   "``./rmake.py -c``","Build the library and client in your local directory."
   "``./rmake.py -ic``","Build and install the rocSPARSE package and build the client. To keep rocSPARSE in your local directory, do not use the ``-i`` flag."
   "``./rmake.py -icn``","Build and install the rocSPARSE package without rocBLAS and build the client. To keep rocSPARSE in your local tree, do not use the ``-i`` flag."
   "``./rmake.py -ic -a gfx900``","Build and install the rocSPARSE package using only the gfx900 architecture and build the client. To keep rocSPARSE in your local tree, do not use the ``-i`` flag."
