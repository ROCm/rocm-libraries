.. meta::
   :description: Build and install hipThreads from source
   :keywords: install, building, hipThreads, AMD, ROCm, source code, cmake, Windows, Linux

.. _source-build:

****************************
Build hipThreads from source
****************************

To build hipThreads as part of the ROCm Core SDK, see `TheRock build instructions <https://github.com/ROCm/TheRock/blob/main/docs/development/README.md>`__.
TheRock is the recommended way to build ROCm components from source.

Alternatively, you can build hipThreads standalone using the following instructions.

.. _hipthreads-prerequisites:

Prerequisites
=============

On Linux, :doc:`ROCm <rocm:install/rocm>` must be installed before hipThreads is built.

hipThreads has the following prerequisites on Linux and Microsoft Windows:

* `CMake <https://cmake.org/>`_ version 3.21 or higher
* `hipcc <https://rocm.docs.amd.com/projects/HIPCC/en/latest/index.html>`_
* ROCm 7.12 or later, which provides HIP and libhipcxx
* A build tool such as ``make`` or `Ninja <https://ninja-build.org/>`_

hipThreads has these additional prerequisites on Windows:

* `HIP SDK for Windows <https://rocm.docs.amd.com/projects/install-on-windows/en/latest/>`_ (or a TheRock build), with ``HIP_PATH`` and ``ROCM_PATH`` set to its root using forward slashes
* `Visual Studio 2022 Build Tools <https://visualstudio.microsoft.com/>`_ with the *Desktop development with C++* workload and the Windows SDK
* `Ninja <https://ninja-build.org/>`_

.. _hipthreads-get-source:

Get the hipThreads source code
==============================

The hipThreads source code is available from the `ROCm libraries GitHub repository <https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipthreads>`_.
Use sparse checkout when cloning the hipThreads project:

.. code-block:: shell

  git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-libraries.git
  cd rocm-libraries
  git sparse-checkout init --cone
  git sparse-checkout set projects/hipthreads

Then use ``git checkout`` to check out the branch you need.

The develop branch is intended for users who want to preview new features or contribute to the hipThreads code base.

If you don't intend to contribute to the hipThreads code base and won't be previewing features, use a branch that matches the version of ROCm installed on your system.

.. _hipthreads-build-linux:

Build on Linux
==============

By default, hipThreads installs under ``$ROCM_PATH`` to match other ROCm components.
Override this by passing ``-DCMAKE_INSTALL_PREFIX=<path>`` to the CMake configure step.

From the ``projects/hipthreads`` directory, configure, build, and install:

.. code-block:: bash

   cmake -B build
   cmake --build ./build
   sudo cmake --install ./build

The GPU architecture is auto-detected on Linux, so you do not need to set ``-DCMAKE_HIP_ARCHITECTURES``.

.. note::

  Installing to ``$ROCM_PATH`` usually requires ``sudo``.

.. _hipthreads-build-windows:

Build on Windows
================

Run all of the following from the **x64 Native Tools Command Prompt for VS 2022** so that CMake can find the MSVC toolchain and the Windows SDK.

Unlike Linux, the GPU architecture is not auto-detected on Windows and must be passed with ``-DCMAKE_HIP_ARCHITECTURES``.
It must match between the hipThreads build and every consumer, or you will get undefined device-symbol errors.
For example, ``gfx1201`` targets the Radeon RX 9070 XT.

.. code-block:: bat

   cmake -B build -G Ninja ^
     -DCMAKE_CXX_COMPILER="clang++" -DCMAKE_C_COMPILER="clang" ^
     -DCMAKE_INSTALL_PREFIX="%HIP_PATH%" ^
     -DHIP_PLATFORM=amd ^
     -DCMAKE_HIP_ARCHITECTURES=gfx1201 ^
     -DCMAKE_BUILD_TYPE=Release .
   cmake --build build
   cmake --install build

.. _hipthreads-run-tests:

Build and run the tests
=======================

The test suite lives in the ``test/`` directory as ``test/*.cxx`` files and is run with `lit <https://llvm.org/docs/CommandGuide/lit.html>`_.
lit compiles each ``test/*.cxx`` file with ``hipcc`` and runs the whole suite, so it works against any build (``Debug`` or ``Release``):

.. code-block:: bash

   pip install lit
   HIPTHREADS_SOURCE_DIR=$PWD HIPTHREADS_BUILD_DIR=$PWD/build lit -j 1 test/

Run with ``-j 1`` so the tests run one at a time, since they share the GPU.

For quick iteration on a single test, you can also build the tests through CMake in a ``Debug`` build, which compiles each ``test/*.cxx`` file into an executable named after its source file:

.. code-block:: bash

   cmake -B build-debug -DCMAKE_BUILD_TYPE=Debug -DDISABLE_WERROR=ON
   cmake --build build-debug -j$(nproc)
   ./build-debug/bin/hip_thread_mutex_test
   ./build-debug/bin/hip_thread_condvar_test

.. _hipthreads-build-examples:

Build and run the examples
==========================

The ``examples/`` directory contains standalone CMake projects that each :doc:`find and link hipThreads <../how-to/hipThreads-in-a-project>`.
Each example is organized as a series of ``stepN-*`` directories showing an incremental port from CPU ``std::thread`` code to hipThreads.

Each example is built and run on its own.
For instance, to build and run the SIMD-optimized SAXPY example on Linux:

.. code-block:: bash

   cd examples/saxpy/step3-simdize
   cmake -B build
   cmake --build ./build
   ./build/bin/saxpy

On Windows, use the same Ninja, clang, and ``-DCMAKE_HIP_ARCHITECTURES`` flags as the library build above.
The exact configure, build, and run commands for each step, on both Linux and Windows, are recorded in a comment at the bottom of that step's ``CMakeLists.txt``.
Some examples need extra setup — for example, the sparse matrix multiply data is pulled with ``git lfs``, and llama3.c takes a model path as an argument — so check the ``CMakeLists.txt`` footer for the step you are building.

After installing, see :doc:`../how-to/hipThreads-in-a-project` to consume hipThreads from your own CMake project.
