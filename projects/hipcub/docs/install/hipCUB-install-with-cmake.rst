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
* ``USE_SYSTEM_LIB``: Set to ``ON`` to use the installed ``hipCUB`` from the system when building the tests. Off by default. For this option to take effect, ``BUILD_TEST`` must be ``ON`` and the ``hipCUB`` install (with its dependencies) must be compatible with the version of the tests.
* ``DEPENDENCIES_FORCE_DOWNLOAD``. Set this to ``ON`` to download the dependencies regardless of whether or not they are already installed. Off by default.
* ``BUILD_OFFLOAD_COMPRESS``. Set this to ``OFF`` to prevent the ``--offload-compress`` switch from being passed to the compiler and compressing the binary. On by default.

Build hipCUB using the generated make file:

.. code:: shell

    make -j4

After you've built hipCUB, you can optionally generate tar, zip, and deb packages:

.. code:: shell

    make package

Finally, install hipCUB:

.. code:: shell

    make install
