.. meta::
  :description: Build and install rocPRIM from source
  :keywords: install, building, rocPRIM, AMD, ROCm, source code, cmake, Linux, Windows

.. _build-from-source:

****************************
Build rocPRIM from source
****************************

To build rocPRIM as part of the ROCm Core SDK, see `TheRock build
instructions
<https://github.com/ROCm/TheRock/blob/main/docs/development/README.md>`__.
TheRock is the recommended way to build ROCm components from source.

Alternatively, you can build rocPRIM standalone using the following
instructions.

.. _rocprim-prerequisites:

Prerequisites
=============

rocPRIM on Linux requires `ROCm <https://rocm.docs.amd.com/en/latest/>`_. rocPRIM on Windows requires `HIP SDK for Windows <https://rocm.docs.amd.com/projects/install-on-windows/en/latest/>`_.

rocPRIM uses `HIPCC <https://rocm.docs.amd.com/projects/HIPCC/en/latest/index.html>`_ to build and run examples, tests, and benchmarks.

`CMake version 3.16 or later <https://cmake.org/>`_ and C++17 are required on both Linux and Windows.

The following additional prerequisites are needed on Windows only:

* `Python version 3.6 or higher <https://www.python.org/downloads/>`_
* `Microsoft Visual Studio 2019 with Clang support <https://visualstudio.microsoft.com/>`_
* `Strawberry Perl <https://www.strawberryperl.com/>`_

.. _rocprim-get-source:

Get the rocPRIM source code
============================

The rocPRIM source code is available from the `ROCm libraries GitHub repository <https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocprim>`_.
Use sparse checkout when cloning the rocPRIM project:

.. code-block:: shell

  git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-libraries.git
  cd rocm-libraries
  git sparse-checkout init --cone
  git sparse-checkout set projects/rocprim

Then use ``git checkout`` to check out the branch you need.

The develop branch is intended for users who want to preview new features or contribute to the rocPRIM code base.

If you don't intend to contribute to the rocPRIM code base and won't be previewing features, use a branch that matches the version of ROCm installed on your system.

For build instructions, see :doc:`./build`.

.. _rocprim-build-linux:

Build on Linux
==============

rocPRIM is built on Linux using CMake. CMake is also used to build rocPRIM examples, tests, and benchmarks.

Create the ``build`` directory under the cloned ``rocprim`` directory, then change directory to ``build``:

.. code-block:: shell

    mkdir build
    cd build

Set the ``CXX`` environment variable to ``hipcc``:

.. code-block:: shell

    export CXX=hipcc

You can build and install the rocPRIM library without any examples, tests, or benchmarks by running ``cmake`` followed by ``make install``:

.. code-block:: shell

    cmake ../.
    make install

The available CMake options are:

* ``BUILD_TEST``: Set to ``ON`` to build the CTests. ``OFF`` by default.
* ``BUILD_EXAMPLE``: Set to ``ON`` to build examples. ``OFF`` by default.
* ``BUILD_DOCS``: Set to ``ON`` to build a local copy of the rocPRIM documentation. ``OFF`` by default.
* ``BUILD_BENCHMARK``: Set to ``ON`` to build benchmarking tests. ``OFF`` by default.
* ``BENCHMARK_CONFIG_TUNING``: Set to ``ON`` to find the best kernel configuration parameters for benchmarking. Turning this on might increase compilation time significantly. ``OFF`` by default.
* ``AMDGPU_TARGETS``: Set this to build the library, examples, tests, and benchmarks for specific architecture targets. When not set, the examples, tests, and benchmarks are built for gfx906:xnack-, gfx908:xnack-, gfx90a:xnack-, gfx90a:xnack+, gfx942, gfx950, gfx1030, gfx1100, gfx1101, gfx1102, gfx1150, gfx1151, gfx1152, gfx1153, gfx1200, and gfx1201 architectures. The list of targets must be separated by a semicolon (``;``).
* ``AMDGPU_TEST_TARGETS``: Set this to build tests for a subset of the architectures specified by ``AMDGPU_TARGETS``. When set, copies of the same test will be generated for each of the architectures listed. These tests can be run using ``ctest -R "TARGET_ARCHITECTURE"``. The list of targets must be separated by a semicolon (``;``).
* ``USE_SYSTEM_LIB``: Set to ``ON`` to use the installed ``ROCm`` libraries when building the tests. Off by default. For this option to take effect, ``BUILD_TEST`` must be ``ON``.
* ``ONLY_INSTALL``: Set to ``ON`` to ignore any example, test, or benchmark build instructions. ``OFF`` by default.

Run ``make`` after ``cmake`` to build the examples, tests, and benchmarks, then run ``make install``. For example, to build tests run:

.. code-block:: shell

    export CXX=hipcc
    cmake -DBUILD_TEST=ON ../.
    make
    sudo make install

.. _rocprim-build-windows:

Build on Windows
================

rocPRIM is built on Windows using the ``rmake.py`` Python script. ``rmake.py`` is also used to build rocPRIM examples, tests, and benchmarks.

In the cloned ``rocprim`` directory, run ``rmake.py -i`` to install rocPRIM to ``C:\hipSDK\include\``:

.. code-block:: shell

    cd rocPRIM
    python3 rmake.py -i

Use the ``-c`` option to build the examples, tests, and benchmarks:

.. code-block:: shell

    python3 rmake.py -c

You can also build Microsoft Visual Studio projects for the examples, tests, and benchmarks.

Change directory to the ``example``, ``test``, or ``benchmark`` directory, and create the ``build`` directory. For example:

.. code-block:: shell

    cd benchmark
    mkdir build

Change directory to the ``build`` directory, and run ``cmake``:

.. code-block:: shell

    cd build
    cmake ../.

The Visual Studio projects and solutions will be created in the ``build`` directory.
