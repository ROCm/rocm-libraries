.. meta::
  :description: Build and install rocThrust with CMake
  :keywords: install, building, rocThrust, AMD, ROCm, source code, cmake

.. _install-with-cmake:

********************************************************************
Building and installing rocThrust on Windows and Linux with CMake 
********************************************************************

You can build and install rocThrust with CMake on either Windows or Linux.

Before you begin, set ``CXX`` to ``amdclang++`` or ``hipcc`` depending on the compiler you'll be using, and set ``CMAKE_CXX_COMPILER`` to the compiler's absolute path. For example: 

.. code:: shell

    CXX=amdclang++
    CMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++

Create the ``build`` directory inside the ``rocThrust`` directory, then change directory to the ``build`` directory:

.. code:: shell

    mkdir build
    cd build

Generate the rocThrust makefile using the ``cmake`` command: 

.. code:: shell

    cmake ../. [-D<OPTION1=VALUE1> [-D<OPTION2=VALUE2>] ...]

The available build options are:

* ``BUILD_BENCHMARK``. Set this to ``ON`` to build benchmark tests. Off by default.
* ``BUILD_EXAMPLES``. Set this to ``ON`` to build rocThrust examples. Off by default.
* ``BUILD_TEST`` and ``BUILD_HIPSTDPAR_TEST``. Set ``BUILD_TEST`` to ``ON`` to enable both rocThrust and HIPSTDPAR tests. Set ``BUILD_HIPSTDPAR_TEST`` to ``ON`` to enable only the HIPSTDPAR tests. Both options are Off by default.
* ``DISABLE_WERROR``. Set this to ``OFF`` to pass ``-Werror`` to the compiler. On by default.
* ``DOWNLOAD_ROCPRIM``. Set this to ``ON`` to download rocPRIM regardless of whether or not rocPRIM is already installed. Off by default.
* ``RNG_SEED_COUNT``. Set this to the non-repeatable random dataset count. Set to 0 by default.
* ``PRNG_SEEDS``. Set this to the RNG seeds. The seeds are passed as a semicolon-delimited array of 32-bit unsigned integers. To avoid command line parsing errors, enclose the entire option in quotation marks. For example, ``cmake "-DPRNG_SEEDS=1;2;3;4"``. Set to 1 by default.

Build rocThrust using the generated make file:

.. code:: shell

    make -j4
    
After you've built rocThrust, you can optionally generate tar, zip, and deb packages:

.. code:: shell

    make package

Finally, install rocThrust:

.. code:: shell

    make install
