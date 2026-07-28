.. meta::
  :description: Build and install hipThreads from source
  :keywords: install, building, hipThreads, AMD, ROCm, source code, cmake, Windows, Linux

.. _build-from-source:

********************************
Build hipThreads from source
********************************

To build hipThreads as part of the ROCm Core SDK, see `TheRock build instructions <https://github.com/ROCm/TheRock/blob/main/docs/development/README.md>`__.
TheRock is the recommended way to build ROCm components from source.

Alternatively, you can build hipThreads standalone using the following instructions.

.. _hipthreads-prerequisites:

Prerequisites
=============

Install :doc:`ROCm <rocm:install/rocm>` before you build hipThreads on Linux.

hipThreads has the following prerequisites on Linux and Microsoft Windows:

- `CMake <https://cmake.org/>`_ version 3.21 or higher
- A C++17-capable compiler
- `hipcc <https://rocm.docs.amd.com/projects/HIPCC/en/latest/index.html>`_
- ROCm 7.12 or later, which provides HIP and libhipcxx
- `lld <https://lld.llvm.org/>`_ linker
- A build tool such as ``make`` or `Ninja <https://ninja-build.org/>`_

CMake locates ``hip`` and ``libhipcxx`` through ``CMAKE_PREFIX_PATH``, which the project prepends from ``ROCM_PATH`` and ``/opt/rocm``.
If ROCm is installed elsewhere, pass ``-DCMAKE_PREFIX_PATH=rocm_install_prefix`` at configure time.

hipThreads has these additional prerequisites on Windows:

- `HIP SDK for Windows <https://rocm.docs.amd.com/projects/install-on-windows/en/latest/>`_, or a TheRock build, with ``HIP_PATH`` and ``ROCM_PATH`` set to its root using forward slashes
- `Visual Studio 2022 Build Tools <https://visualstudio.microsoft.com/>`_ with the Desktop development with C++ workload and the Windows SDK
- `Ninja <https://ninja-build.org/>`__

.. _hipthreads-get-source:

Get the hipThreads source code
==============================

You can clone the hipThreads source code from the `ROCm libraries GitHub repository <https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipthreads>`_.
Use sparse checkout when cloning the hipThreads project:

.. code:: shell

    git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-libraries.git
    cd rocm-libraries
    git sparse-checkout init --cone
    git sparse-checkout set projects/hipthreads

Then use ``git checkout`` to check out the branch you need.

Use the develop branch if you want to preview new features or contribute to the hipThreads code base.

If you don't intend to contribute to the hipThreads code base and won't be previewing features, use a branch that matches the version of ROCm installed on your system.

.. _hipthreads-build-linux:

Build on Linux
==============

By default, hipThreads installs under ``$ROCM_PATH`` to match other ROCm components.
Override this by passing ``-DCMAKE_INSTALL_PREFIX=install_prefix`` to the CMake configure step.

From the ``projects/hipthreads`` directory, configure, build, and install:

.. code:: shell

   cmake -B build
   cmake --build ./build
   sudo cmake --install ./build

Linux builds auto-detect the GPU architecture, so you don't need to set ``-DCMAKE_HIP_ARCHITECTURES``.

.. note::

  Installing to ``$ROCM_PATH`` usually requires ``sudo``.

.. _hipthreads-build-windows:

Build on Windows
================

Run every command in this procedure from the x64 Native Tools Command Prompt for VS 2022 so that CMake can find the MSVC toolchain and the Windows SDK.

On Windows, you must pass the GPU architecture with ``-DCMAKE_HIP_ARCHITECTURES``.
Unlike Linux, it isn't auto-detected.
It must match between the hipThreads build and every consumer, or you'll get undefined device-symbol errors.
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

.. _hipthreads-cmake-options:

CMake options
=============

The following table lists project-specific CMake options.
Pass them on the configure command line with ``-Doption_name=value``.

.. list-table::
  :header-rows: 1
  :widths: 35 50 15

  * - Option
    - Description
    - Default
  * - ``CMAKE_BUILD_TYPE``
    - Build configuration (``Release``, ``Debug``, ``RelWithDebInfo``, and so on).
      A ``Debug`` build adds the CMake test targets under ``test/``.
    - ``Release``
  * - ``CMAKE_INSTALL_PREFIX``
    - Root for ``cmake --install``.
      When left at the CMake default, the project sets this from ``ROCM_PATH``, or ``/opt/rocm`` when that variable is unset.
    - ``$ROCM_PATH`` or ``/opt/rocm``
  * - ``HIPTHREADS_COPY_TO_BUILD``
    - Copy the source tree into the build directory.
      TheRock's tester pipeline sets this so ``lit`` can compile tests from the packaged artifact.
      Leave it off for local builds.
    - ``OFF``
  * - ``HIPTHREADS_DEFAULT_VCORES_PER_WGP``
    - Compiled-in default vcore count per :term:`WGP`.
      When unset at configure time, ``src/hip/thread.cxx`` uses ``16``.
      You can also change the count at runtime without rebuilding.
      See :doc:`../how-to/tune-scheduler-concurrency`.
    - ``16`` in ``src/hip/thread.cxx`` when unset

Install layout
--------------

``cmake --install`` places headers under ``install_prefix/include/hipthreads/hip/``, the static library under ``install_prefix/lib/hipthreads/``, and CMake package files under ``install_prefix/lib/cmake/hipthreads/``.
Downstream projects consume the install with ``find_package(hipthreads REQUIRED)``.

Unsupported-architecture compile definition
-------------------------------------------

The build adds ``_LIBCUDACXX_ALLOW_UNSUPPORTED_ARCHITECTURE`` as a ``PUBLIC`` compile definition on the ``hipthreads`` target.
This is required until libhipcxx resolves an upstream architecture-guard issue.
Targets that link ``hipthreads::hipthreads`` inherit the definition automatically.

Post-build tuning
-----------------

After installation, you can adjust vcores per WGP at runtime with the ``HIPTHREADS_VCORES_PER_WGP`` environment variable without rebuilding.
See :doc:`../how-to/tune-scheduler-concurrency` for guidance.

.. _hipthreads-run-tests:

Build and run the tests
=======================

The main test suite is under ``test/std/`` and ``test/libcxx/`` as ``*.pass.cpp`` files, configured by ``test/lit.cfg``.
Run it with `lit <https://llvm.org/docs/CommandGuide/lit.html>`_.
``lit`` compiles and runs each test with ``hipcc``, so it works against either ``Debug`` or ``Release`` builds:

.. code:: shell

   pip install lit
   HIPTHREADS_SOURCE_DIR=$PWD HIPTHREADS_BUILD_DIR=$PWD/build lit -j 1 test/

Because the tests share the GPU, run with ``-j 1`` so they run one at a time.

``lit.cfg`` defaults ``HIPTHREADS_SOURCE_DIR`` to the repository root and ``HIPTHREADS_BUILD_DIR`` to ``build`` under it, so you only need to set them when your build directory is somewhere else.

A separate, smaller set of ``test/*.cxx`` files builds through CMake in a ``Debug`` build, which is useful for quick iteration on a single test.
CMake compiles each ``test/*.cxx`` file into an executable named after its source file:

.. code:: shell

   cmake -B build-debug -DCMAKE_BUILD_TYPE=Debug -DDISABLE_WERROR=ON
   cmake --build build-debug -j$(nproc)
   ./build-debug/bin/hip_thread_mutex_test
   ./build-debug/bin/hip_thread_condvar_test

.. _hipthreads-build-examples:

Build and run the examples
==========================

The ``examples/`` directory contains standalone CMake projects.
Each example uses a series of ``stepN-*`` directories showing an incremental port from CPU ``std::thread`` code to hipThreads.
The hipThreads steps :doc:`find and link hipThreads <../how-to/use-hipthreads-in-a-project>`, and the baseline steps are CPU-only.

Each example is built and run on its own.
For instance, to build and run the SIMD-optimized SAXPY example on Linux:

.. code:: shell

   cd examples/saxpy/step3-simdize
   cmake -B build
   cmake --build ./build
   ./build/bin/saxpy

On Windows, use the same Ninja, clang, and ``-DCMAKE_HIP_ARCHITECTURES`` flags as in :ref:`Build on Windows <hipthreads-build-windows>`.
Each step records the exact configure, build, and run commands in a comment in that step's ``CMakeLists.txt``, at the bottom for most examples and at the top for llama3.c.
Some examples need extra setup.
For example, pull the sparse matrix multiply data with ``git lfs``, and pass a model path to llama3.c.
Check the ``CMakeLists.txt`` comment block for the step you're building.

After installing, see :doc:`../how-to/use-hipthreads-in-a-project` to consume hipThreads from your own CMake project.
