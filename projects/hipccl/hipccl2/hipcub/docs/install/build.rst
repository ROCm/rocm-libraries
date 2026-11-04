.. meta::
  :description: Build and install hipCUB from source
  :keywords: install, building, hipCUB, AMD, ROCm, source code, cmake, Windows

.. _build-from-source:

************************
Build hipCUB from source
************************

To build hipCUB as part of the ROCm Core SDK, see `TheRock build
instructions
<https://github.com/ROCm/TheRock/blob/main/docs/development/README.md>`__.
TheRock is the recommended way to build ROCm components from source.

Alternatively, you can build hipCUB standalone using the following
instructions.

.. _hipcub-prerequisites:

Prerequisites
=============

hipCUB has the following prerequisites on Linux:

* `CMake <https://cmake.org/>`_ version 3.18 or higher
* `ROCm <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/index.html>`_
* `amdclang++ <https://rocm.docs.amd.com/projects/llvm-project/en/latest/index.html>`_
* `rocPRIM <https://rocm.docs.amd.com/projects/rocPRIM/en/latest/index.html>`_

amdclang++ is installed with ROCm. rocPRIM is automatically downloaded and installed by the CMake script.

hipCUB has the following prerequisites on Microsoft Windows:

* `HIP SDK for Windows <https://rocm.docs.amd.com/projects/install-on-windows/en/latest/>`_
* Python version 3.6 or later
* Visual Studio 2019 with Clang support
* Strawberry Perl

.. _hipcub-get-source:

Get the hipCUB source code
==========================

The hipCUB source code is available from the `ROCm libraries GitHub repository <https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipcub>`_.
Use sparse checkout when cloning the hipCUB project:

.. code-block:: shell

  git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-libraries.git
  cd rocm-libraries
  git sparse-checkout init --cone
  git sparse-checkout set projects/hipcub

Then use ``git checkout`` to check out the branch you need.

The develop branch is the default branch. The develop branch is intended for users who want to preview new features or contribute to the hipCUB code base.

If you don't intend to contribute to the hipCUB code base and won't be previewing features, use a branch that matches the version of ROCm installed on your system.

For build instructions, see :doc:`./build`.

.. _build-with-cmake:

Build with CMake
================

You can build and install hipCUB with CMake on Windows or Linux.

Before you begin, set ``CXX`` to ``amdclang++`` or ``hipcc``, and set ``CMAKE_CXX_COMPILER`` to the compiler's absolute path. For example:

.. code-block:: shell

    CXX=amdclang++
    CMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++

Create the ``build`` directory under the ``hipcub`` root directory, then change directory to the ``build`` directory:

.. code-block:: shell

    mkdir build
    cd build

Generate the makefile using the ``cmake`` command:

.. code-block:: shell

    cmake ../. [-D<OPTION1=VALUE1> [-D<OPTION2=VALUE2>] ...]

The available build options are:

* ``BUILD_BENCHMARK``: Set this to ``ON`` to build benchmark tests. Off by default.
* ``BUILD_TEST``: Set this to ``ON`` to build tests. Off by default.
* ``BUILD_EXAMPLE``: Set this to ``ON`` to build the hipCUB examples. Default is ``OFF``.
* ``USE_SYSTEM_LIB``: Set to ``ON`` to use the installed ``hipCUB`` from the system when building the tests. Off by default. For this option to take effect, ``BUILD_TEST`` must be ``ON`` and the ``hipCUB`` install (with its dependencies) must be compatible with the version of the tests.
* ``BUILD_ADDRESS_SANITIZER``: Set this to ``ON`` to build with the Clang address sanitizer enabled. Default is ``OFF``.
* ``EXTERNAL_DEPS_FORCE_DOWNLOAD``: Set this to ``ON`` to download the non-ROCm dependencies such as Google Test even if they're already installed. Default is ``OFF``.
* ``BUILD_OFFLOAD_COMPRESS``: Set this to ``OFF`` to prevent the ``--offload-compress`` switch from being passed to the compiler and compressing the binary. On by default.
* ``USE_HIPCXX``: Set this to ``ON`` to build with CMake HIP language support. Setting this to ``ON`` eliminates the need to use ``CXX=hipcc``. Default is ``OFF``.
* ``ROCPRIM_FETCH_METHOD``: Set this to the method to use to download rocPRIM. Can be set to ``PACKAGE``, ``DOWNLOAD``, or ``MONOREPO``. Set to ``MONOREPO`` if rocPRIM isn't already installed and you're building hipCUB from within a clone of the `rocm-libraries <https://github.com/ROCm/rocm-libraries/>`_ repository that also includes rocPRIM. Set to ``DOWNLOAD`` if rocPRIM isn't installed and you aren't in a clone of the ``rocm-libraries`` repository that includes rocPRIM. ``DOWNLOAD`` will clone the repository using sparse checkout so that only the necessary files are downloaded. Set to ``PACKAGE`` if rocPRIM is already installed. If you specify ``PACKAGE`` but rocPRIM isn't installed, the files will be downloaded using the same method as the ``DOWNLOAD`` option. The default method is ``PACKAGE``.

.. note::

    If you're using a version of git earlier than 2.25, ``-DROCPRIM_FETCH_METHOD=DOWNLOAD`` will download the entire ``rocm-libraries`` repository.

Build hipCUB using the generated make file:

.. code-block:: shell

    make -j4

After you've built hipCUB, you can optionally generate tar, zip, and deb packages:

.. code-block:: shell

    make package

Finally, install hipCUB:

.. code-block:: shell

    make install

.. _build-on-windows:

Build on Windows
================

You can use ``rmake.py`` to build and install hipCUB on Microsoft Windows. You can also use :ref:`CMake <build-with-cmake>` if you want more build and installation options.

``rmake.py`` is located in the ``hipcub`` root directory. To build and install hipCUB, run:

.. code-block:: shell

    python rmake.py -i

This command also downloads `rocPRIM <https://rocm.docs.amd.com/projects/rocPRIM/en/latest/index.html>`_ and installs it in ``C:\hipSDK``.

The ``-c`` option builds all clients, including the unit tests:

.. code-block:: shell

    python rmake.py -c

To see a complete list of ``rmake.py`` options, run:

.. code-block:: shell

    python rmake.py --help
