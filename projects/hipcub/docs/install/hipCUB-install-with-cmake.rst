.. meta::
  :description: Build and install hipCUB with CMake
  :keywords: install, building, hipCUB, AMD, ROCm, source code, cmake

.. _install-with-cmake:

********************************************************************
Building and installing hipCUB with CMake
********************************************************************

You can build and install hipCUB with CMake on Windows or Linux.

Before you begin, set ``CXX`` to ``amdclang++`` or ``hipcc``, and set ``CMAKE_CXX_COMPILER`` to the compiler's absolute path. For example: 

.. code:: shell

    CXX=amdclang++
    CMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++

After :doc:`cloning the project <./hipCUB-install-overview>`, create the ``build`` directory under the ``hipcub`` root directory, then change directory to the ``build`` directory:

.. code:: shell

    mkdir build
    cd build

Generate the makefile using the ``cmake`` command: 

.. code:: shell

    cmake ../. [-D<OPTION1=VALUE1> [-D<OPTION2=VALUE2>] ...]

The available build options are:


* ``BUILD_BENCHMARK``. Set this to ``ON`` to build benchmark tests. Off by default.
* ``BUILD_TEST``. Set this to ``ON`` to build tests. Off by default. 
* ``BUILD_EXAMPLE``. Set this to ``ON`` to build the hipCUB examples. Default is ``OFF``.
* ``USE_SYSTEM_LIB``: Set to ``ON`` to use the installed ``hipCUB`` from the system when building the tests. Off by default. For this option to take effect, ``BUILD_TEST`` must be ``ON`` and the ``hipCUB`` install (with its dependencies) must be compatible with the version of the tests.
* ``BUILD_ADDRESS_SANITIZER``. Set this to ``ON`` to build with the Clang address sanitizer enabled. Default is ``OFF``.
* `` EXTERNAL_DEPS_FORCE_DOWNLOAD``. Set this to ``ON`` to download the non-ROCm dependencies such as Google Test even if they're already installed. Default is ``OFF``.
* ``BUILD_OFFLOAD_COMPRESS``. Set this to ``OFF`` to prevent the ``--offload-compress`` switch from being passed to the compiler and compressing the binary. On by default.
* ``USE_HIPCXX``. Set this to ``ON`` to build with CMake HIP language support. Setting this to ``ON`` eliminates the need to use ``CXX=hipcc``. Default is ``OFF``.
* ``ROCPRIM_FETCH_METHOD``. Set this to the method to use to download rocPRIM. Can be set to ``PACKAGE``, ``DOWNLOAD``, or ``MONOREPO``. Set to ``MONOREPO`` if rocPRIM isn't already installed and you're building hipCUB from within a clone of the `rocm-libraries <https://github.com/ROCm/rocm-libraries/>`_ repository that also includes rocPRIM. Set to ``DOWNLOAD`` if rocPRIM isn't installed and you aren't in a clone of the ``rocm-libraries`` repository that includes rocPRIM. ``DOWNLOAD`` will clone the repository using sparse checkout so that only the necessary files are downloaded. Set to ``PACKAGE`` if rocPRIM is already installed. If you specify ``PACKAGE`` but rocPRIM isn't installed, the files will be downloaded using the same method as the ``DOWNLOAD`` option. The default method is ``PACKAGE``.

.. note::

    If you're using a version of git earlier than 2.25, ``-DROCPRIM_FETCH_METHOD=DOWNLOAD`` will download the entire ``rocm-libraries`` repository.

Build hipCUB using the generated make file:

.. code:: shell

    make -j4

After you've built hipCUB, you can optionally generate tar, zip, and deb packages:

.. code:: shell

    make package

Finally, install hipCUB:

.. code:: shell

    make install
