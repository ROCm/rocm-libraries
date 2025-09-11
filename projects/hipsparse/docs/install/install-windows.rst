.. meta::
  :description: How to build and install hipSPARSE on Windows
  :keywords: hipSPARSE, ROCm, API, documentation, installation, Windows, build

.. _windows-install:

********************************************************************
Installing and building hipSPARSE for Microsoft Windows
********************************************************************

This topic describes how to install or build hipSPARSE on Microsoft Windows by using prebuilt packages or building from source.
For information on installing and building hipSPARSE on Linux, see :doc:`hipSPARSE for Linux <./install>`.

Prerequisites
=============

hipSPARSE on Windows requires an AMD HIP SDK-enabled platform. It's supported on the
same Windows versions and toolchains that the HIP SDK supports. For more information, see
:doc:`HIP SDK installation for Windows <rocm-install-on-windows:index>`.

Installing prebuilt packages
============================

hipSPARSE can be installed on Windows 10 or 11 using the AMD HIP SDK installer.

To add hipSPARSE to your code, use CMake.
Add the SDK installation location to your ``CMAKE_PREFIX_PATH``.

.. note::

   You must use quotes because the path contains a space.

.. code-block:: shell

   -DCMAKE_PREFIX_PATH="C:\Program Files\AMD\ROCm\7.0"

In your ``CMakeLists.txt`` file, use these lines:

.. code-block:: shell

   find_package(hipsparse)
   target_link_libraries( your_exe PRIVATE roc::hipsparse )

After hipSPARSE is installed, it can be used just like any other library with a C API.
To call hipSPARSE, the ``hipsparse.h`` header file must be included in the user code.
This means the hipSPARSE  import library and dynamic link library respectively become link-time and run-time dependencies
for the user application.

After the installation, you can find ``hipsparse.h`` in the HIP SDK ``\\include\\hipsparse``
directory. When you need to include hipSPARSE in your application code, you must only use this file.
You can find the other hipSPARSE files included in the HIP SDK ``\\include\\hipsparse\\internal`` directory, but
do not include these files directly in your source code.

Building hipSPARSE from source
=================================

It isn't necessary to build hipSPARSE from source because it's ready to use after installing
the prebuilt packages, as described above.
To build hipSPARSE from source, follow the instructions in this section.

Requirements
------------

To compile and run hipSPARSE, the `AMD ROCm Platform <https://github.com/ROCm/ROCm>`_ is required.
Building hipSPARSE from source also requires the following components and dependencies:

*  `rocSPARSE <https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocsparse>`_
*  `git <https://git-scm.com/>`_
*  `CMake <https://cmake.org/>`_ (Version 3.5 or later)
*  `vcpkg <https://github.com/Microsoft/vcpkg.git>`_
*  `GoogleTest <https://github.com/google/googletest>`_ (Optional: only required to build the clients)
*  `Python <https://www.python.org/>`_
*  `PyYAML <https://pypi.org/project/PyYAML/>`_

When building hipSPARSE from source, select a supported version of the :doc:`rocSPARSE <rocsparse:index>` library
dependency. Given a specific version of hipSPARSE,
it's best to use the same version of rocSPARSE.

Downloading hipSPARSE
----------------------

The hipSPARSE source code for Windows, which is the same as for Linux, is available
from the `hipSPARSE folder <https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipsparse>`_
of the `rocm-libraries GitHub <https://github.com/ROCm/rocm-libraries>`_.
The ROCm HIP SDK version might be shown in the default installation path, but
you can run the HIP SDK compiler from the ``bin/`` folder to display the version using this command:

.. code-block:: shell

   hipcc --version

The HIP version number consists of major, minor, and patch fields, and is sometimes followed by a build-specific identifier.
For example, the HIP version might be ``6.4.22880-135e1ab4``.
This corresponds to major release ``6``, minor release ``4``, patch ``22880``, and build identifier ``135e1ab4``.
The rocm-libraries GitHub includes branches with names like ``release/rocm-rel-major.minor``,
where major and minor have the same meaning as the HIP version.

To limit your local checkout to only the hipSPARSE project, configure ``sparse-checkout`` before cloning.
This uses the Git partial clone feature (``--filter=blob:none``) to reduce the data download.
Use the following commands for a sparse checkout:

.. note::

   To include the rocSPARSE dependency, set the projects for the sparse checkout using
   ``git sparse-checkout set projects/hipsparse projects/rocsparse``.

.. code-block:: shell

   git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-libraries.git
   cd rocm-libraries
   git sparse-checkout init --cone
   git sparse-checkout set projects/hipsparse # add projects/rocsparse to include dependency
   git checkout release/rocm-rel-x.y  # or use the branch you want to work with

Replace ``x.y`` in the above command with the version of HIP SDK installed on your machine.
For example, if you have HIP 7.0 installed, use ``-b release/rocm-rel-7.0``.

To download all projects in rocm-libraries, use these commands. This process takes
longer but is recommended for those working with a large number of libraries.

.. code-block:: shell

   git clone -b release/rocm-rel-x.y https://github.com/ROCm/rocm-libraries.git
   cd rocm-libraries/projects/hipsparse

.. note::

   To build ROCm 6.4.3 and earlier, use the hipSPARSE repository at `<https://github.com/ROCm/hipSPARSE>`_.
   For more information, see the documentation associated with the release you want to build.

Add the SDK tools to your path with an entry like the following:

.. code-block:: shell

   %HIP_PATH%\bin

Building using the Python script
---------------------------------

This section describes the steps required to build hipSPARSE using the ``rmake.py`` script. You can build:

*  The library
*  The library and client

To call hipSPARSE from your code, you only need the library. The client contains testing and benchmark tools.
``rmake.py`` prints the full ``cmake`` command being used to configure hipSPARSE based on the ``rmake`` command line options.
This full ``cmake`` command can be used in your own build scripts to bypass the Python helper script for a fixed set of build options.

Building the library from source
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following table lists the common ways to use ``rmake.py`` to build the hipSPARSE library only.

.. note::

   You can run ``rmake.py`` from the ``projects\hipsparse`` directory.

.. csv-table::
   :header: "Command","Description"
   :widths: 40, 90

   "``./rmake.py -h``","Print the help information."
   "``./rmake.py``","Build the library."
   "``./rmake.py -i``","Build the library, then build and install the hipSPARSE package. To keep hipSPARSE in your local tree, do not use the ``-i`` flag."
   "``./rmake.py -in``","Build the library without rocBLAS, then build and install the hipSPARSE package. To keep hipSPARSE in your local tree, do not use the ``-i`` flag."
   "``./rmake.py -i -a gfx900``","Build the library using only the gfx900 architecture, then build and install the hipSPARSE package. To keep hipSPARSE in your local tree, do not use the ``-i`` flag."

Building the library and client from source
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following table lists the common ways to use ``rmake.py`` to build the hipSPARSE library and clients.

.. csv-table::
   :header: "Command","Description"
   :widths: 40, 90

   "``./rmake.py -h``","Print the help information."
   "``./rmake.py -c``","Build the library and client in your local directory."
   "``./rmake.py -ic``","Build and install the hipSPARSE package and build the client. To keep hipSPARSE in your local directory, do not use the ``-i`` flag."
   "``./rmake.py -icn``","Build and install the hipSPARSE package without rocBLAS and build the client. To keep hipSPARSE in your local tree, do not use the ``-i`` flag."
   "``./rmake.py -ic -a gfx900``","Build and install the hipSPARSE package using only the gfx900 architecture and build the client. To keep hipSPARSE in your local tree, do not use the ``-i`` flag."
