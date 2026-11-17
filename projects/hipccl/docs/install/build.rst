.. meta::
  :description: Build and install hipCCL from source
  :keywords: install, building, hipCCL, AMD, ROCm, source code, cmake, Linux, Windows

.. _build-from-source:

****************************
Build hipCCL from source
****************************

To build hipCCL as part of the ROCm Core SDK, see `TheRock build
instructions
<https://github.com/ROCm/TheRock/blob/main/docs/development/README.md>`__.
TheRock is the recommended way to build ROCm components from source.

Alternatively, you can build hipCCL standalone using the following
instructions.

.. _hipccl-prerequisites:

Prerequisites
=============

hipCCL on Linux requires `ROCm <https://rocm.docs.amd.com/en/latest/>`_.
hipCCL uses `HIPCC <https://rocm.docs.amd.com/projects/HIPCC/en/latest/index.html>`_
to build and run examples, tests, and benchmarks.

`CMake version 3.25.2 or later <https://cmake.org/>`_ and C++17 are required.

.. _hipccl-get-source:

Get the hipCCL source code
============================

The hipCCL source code is available from the `ROCm libraries GitHub
repository <https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipccl>`_.
Use sparse checkout when cloning the hipCCL project - it's self-contained and
doesn't need any other ``projects/`` directory alongside it:

.. code-block:: shell

  git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-libraries.git
  cd rocm-libraries
  git sparse-checkout init --cone
  git sparse-checkout set projects/hipccl

Then use ``git checkout`` to check out the branch you need.

The develop branch is intended for users who want to preview new features or
contribute to the hipCCL code base.

If you don't intend to contribute to the hipCCL code base and won't be
previewing features, use a branch that matches the version of ROCm installed
on your system.

.. _hipccl-build:

Build on Linux
==============

hipCCL is built on Linux using CMake, through one of two entry points:

* The layout selector, which lets a single ``cmake`` invocation choose
  between the ``hipccl3`` (default) and ``hipccl2`` layouts:

  .. code-block:: shell

      mkdir build
      cd build
      export CXX=hipcc
      cmake -S ../projects/hipccl -B .
      make install

  Pass ``-DHIPCCL_BUILD_LEGACY=ON`` to build the ``hipccl2`` layout instead.

* Building a layout directly, e.g.:

  .. code-block:: shell

      cd projects/hipccl/hipccl3
      mkdir build
      cd build
      export CXX=hipcc
      cmake ../.
      make install

  See :doc:`../conceptual/hipccl-layout` for the difference between the two
  layouts.

The available CMake options (shared by both layouts and the selector) are:

* ``HIPCCL_BUILD_LEGACY``: Set to ``ON`` to build the ``hipccl2`` layout
  instead of the default, unified ``hipccl3`` layout. Only meaningful when
  building through the ``projects/hipccl`` selector. ``OFF`` by default.
* ``HIPCCL_BUILD_ROCPRIM``: Set to ``OFF`` to exclude rocPRIM from the build.
  ``ON`` by default.
* ``HIPCCL_BUILD_HIPCUB``: Set to ``OFF`` to exclude hipCUB from the build.
  ``ON`` by default.
* ``HIPCCL_BUILD_ROCTHRUST``: Set to ``OFF`` to exclude rocThrust from the
  build. ``ON`` by default.
* ``HIPCCL_BUILD_ALL``: Set to ``ON`` to force all three of the above
  ``ON``, overriding any of them individually set to ``OFF``. ``OFF`` by
  default.
* ``HIPCCL_BUILD_LIBHIPCXX``: (``hipccl3`` only) Set to ``ON`` to also build
  the ``libhipcxx`` git submodule. Requires the submodule to be initialized
  first (``git submodule update --init
  projects/hipccl/hipccl3/libhipcxx``). ``OFF`` by default.
* ``HIPCCL_INSTALL_LEGACY_HEADER_SYMLINKS``: Set to install (or skip
  installing) the temporary, old-flat-path compatibility symlinks described
  in :doc:`../conceptual/hipccl-layout`. ``ON`` by default in ``hipccl3``;
  ``OFF`` by default in ``hipccl2``.
* ``BUILD_DOCS``: Set to ``ON`` to build a local copy of the hipCCL
  documentation. ``OFF`` by default.

Each of rocPRIM, hipCUB, and rocThrust also accepts its own usual
``BUILD_TEST``/``BUILD_BENCHMARK``/``BUILD_EXAMPLE`` options, forwarded
through unchanged - see each component's own build documentation for the
full list.

Run ``make`` after ``cmake`` to build, then run ``make install``. For
example, to build only rocPRIM and hipCUB, skipping rocThrust:

.. code-block:: shell

    export CXX=hipcc
    cmake -DHIPCCL_BUILD_ROCTHRUST=OFF ../.
    make
    sudo make install
