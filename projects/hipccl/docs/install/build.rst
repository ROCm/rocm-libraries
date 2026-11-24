.. meta::
  :description: Build and install hipCCL from source
  :keywords: install, building, hipCCL, AMD, ROCm, source code, cmake, Linux, Windows

.. _build-from-source:

****************************
Build hipCCL from source
****************************

To build hipCCL as part of the ROCm Core SDK, use the `TheRock build instructions <https://github.com/ROCm/TheRock/blob/main/docs/development/README.md>`__.
TheRock is the recommended way to build ROCm components from source.

You can also build hipCCL on its own.

.. _hipccl-prerequisites:

Prerequisites
=============

hipCCL on Linux requires `ROCm <https://rocm.docs.amd.com/en/latest/>`_.
hipCCL uses `HIPCC <https://rocm.docs.amd.com/projects/HIPCC/en/latest/index.html>`_ to build and run examples, tests, and benchmarks.

`CMake 3.25.2 or later <https://cmake.org/>`_ and C++17 are required.

.. _hipccl-get-source:

Get the hipCCL source code
==========================

The hipCCL source code is in the `ROCm libraries GitHub repository <https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipccl>`_.
Use a sparse checkout when cloning hipCCL.
The project is self-contained and doesn't need any other directory under ``projects/`` alongside it.

.. code:: shell

   git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-libraries.git
   cd rocm-libraries
   git sparse-checkout init --cone
   git sparse-checkout set projects/hipccl

Then run ``git checkout`` for the branch you need.

The develop branch is for previewing new features or contributing to hipCCL.
If you aren't contributing and aren't previewing features, use a branch that matches the ROCm version installed on your system.

.. _hipccl-build:

Build on Linux
==============

hipCCL is built on Linux with CMake.
There are two entry points.

- The layout selector lets one ``cmake`` invocation choose the ``hipccl3`` layout or the ``hipccl2`` layout.
  ``hipccl3`` is the default.

  .. code:: shell

     mkdir build
     cd build
     export CXX=hipcc
     cmake -S ../projects/hipccl -B .
     make install

  Pass ``-DHIPCCL_BUILD_LEGACY=ON`` to CMake to build the ``hipccl2`` layout instead.

- You can also build a layout directly.

  .. code:: shell

     cd projects/hipccl/hipccl3
     mkdir build
     cd build
     export CXX=hipcc
     cmake ../.
     make install

  The difference between the two layouts is documented in :doc:`../conceptual/hipccl-layout`.

Both layouts and the selector share these CMake options.

``HIPCCL_BUILD_LEGACY``
    Set to ``ON`` to build the ``hipccl2`` layout instead of the default ``hipccl3`` layout.
    This option applies only when building through the ``projects/hipccl`` selector.
    The default is ``OFF``.

``HIPCCL_BUILD_ROCPRIM``
    Set to ``OFF`` to exclude rocPRIM from the build.
    The default is ``ON``.

``HIPCCL_BUILD_HIPCUB``
    Set to ``OFF`` to exclude hipCUB from the build.
    The default is ``ON``.

``HIPCCL_BUILD_ROCTHRUST``
    Set to ``OFF`` to exclude rocThrust from the build.
    The default is ``ON``.

``HIPCCL_BUILD_ALL``
    Set to ``ON`` to force ``HIPCCL_BUILD_ROCPRIM``, ``HIPCCL_BUILD_HIPCUB``, and ``HIPCCL_BUILD_ROCTHRUST`` on, including any that were set to ``OFF``.
    The default is ``OFF``.

``HIPCCL_BUILD_LIBHIPCXX``
    This option exists only for ``hipccl3``.
    Set to ``ON`` to also build the ``libhipcxx`` git submodule.
    Initialize the submodule first with ``git submodule update --init projects/hipccl/hipccl3/libhipcxx``.
    The default is ``OFF``.

``HIPCCL_INSTALL_LEGACY_HEADER_SYMLINKS``
    Set to ``ON`` to install the temporary compatibility symlinks described in :doc:`../conceptual/hipccl-layout`.
    Set to ``OFF`` to skip them.
    The default is ``ON``.

``BUILD_DOCS``
    Set to ``ON`` to build a local copy of the hipCCL documentation.
    The default is ``OFF``.

rocPRIM, hipCUB, and rocThrust also accept their own ``BUILD_TEST``, ``BUILD_BENCHMARK``, and ``BUILD_EXAMPLE`` CMake options.
Those options are forwarded unchanged.
Each component's build documentation lists them.

Run ``make`` after ``cmake`` to build, then run ``make install``.
For example, to build only rocPRIM and hipCUB, and to skip rocThrust:

.. code:: shell

   export CXX=hipcc
   cmake -DHIPCCL_BUILD_ROCTHRUST=OFF ../.
   make
   sudo make install
