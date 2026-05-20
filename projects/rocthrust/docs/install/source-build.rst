.. meta::
   :description: Build and install rocThrust from source
   :keywords: install, building, rocThrust, AMD, ROCm, source code, cmake, Windows, Linux

.. _build-from-source:

***************************
Build rocThrust from source
***************************

To build rocThrust as part of the ROCm Core SDK, see `TheRock build
instructions
<https://github.com/ROCm/TheRock/blob/main/docs/development/README.md>`__.
TheRock is the recommended way to build ROCm components from source.

Alternatively, you can build rocThrust standalone using the following
instructions.

.. _rocthrust-prerequisites:

Prerequisites
=============

On Linux, :doc:`ROCm <rocm:install/rocm>` must be installed before rocThrust is installed.

rocThrust has the following prerequisites on Linux and Microsoft Windows:

* `CMake <https://cmake.org/>`_ version 3.10.2 or higher
* `hipcc <https://rocm.docs.amd.com/projects/HIPCC/en/latest/index.html>`_
* `rocPRIM <https://rocm.docs.amd.com/projects/rocPRIM/en/latest/index.html>`_
* `rocRAND <https://rocm.docs.amd.com/projects/rocRAND/en/latest/index.html>`_

rocPRIM can be automatically downloaded and installed when rocThrust is built.

rocThrust has these additional prerequisites on Windows:

* `HIP SDK for Windows <https://rocm.docs.amd.com/projects/install-on-windows/en/latest/>`_
* `Python version 3.6 or later <https://www.python.org/>`_
* `Visual Studio 2019 with Clang support <https://visualstudio.microsoft.com/>`_
* `Strawberry Perl <https://strawberryperl.com/>`_

.. _rocthrust-get-source:

Get the rocThrust source code
==============================

The rocThrust source code is available from the `ROCm libraries GitHub repository <https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocthrust>`_.
Use sparse checkout when cloning the rocThrust project:

.. code-block:: shell

  git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-libraries.git
  cd rocm-libraries
  git sparse-checkout init --cone
  git sparse-checkout set projects/rocthrust

Then use ``git checkout`` to check out the branch you need.

The develop branch is intended for users who want to preview new features or contribute to the rocThrust code base.

If you don't intend to contribute to the rocThrust code base and won't be previewing features, use a branch that matches the version of ROCm installed on your system.

For build instructions, see :doc:`./build`.

.. _rocthrust-build-linux:

Build on Linux
==============

You can use the ``install`` script to build and install rocThrust on Linux. You can also use :ref:`CMake <rocthrust-build-cmake>` if you want more build and installation options.

The ``install`` script is located in the ``rocthrust`` root directory.

To build and install rocThrust, run:

.. code-block:: shell

  ./install --install

This command also downloads and installs rocPRIM.

To build rocThrust and generate tar, zip, and debian packages, run:

.. code-block:: shell

  ./install --package

To see a complete list of options, run:

.. code-block:: shell

  ./install --help

.. _rocthrust-build-windows:

Build on Windows
================

You can use ``rmake.py`` to build and install rocThrust on Windows. You can also use :ref:`CMake <rocthrust-build-cmake>` if you want more build and installation options.

``rmake.py`` is located in the ``rocthrust`` root directory.

To build and install rocThrust, run:

.. code-block:: shell

    python rmake.py -i

This command also downloads `rocPRIM <https://rocm.docs.amd.com/projects/rocPRIM/en/latest/index.html>`_ and installs it in ``C:\hipSDK``.

The ``-c`` option builds all clients, including the unit tests:

.. code-block:: shell

    python rmake.py -c

CMake build options can be passed to the ``rmake.py`` script using the ``--cmake-darg`` option:

.. code-block:: shell

    python rmake.py -ci --cmake-darg THRUST_HOST_SYSTEM=OMP --cmake-darg THRUST_DEVICE_SYSTEM=OMP

To see a complete list of ``rmake.py`` options, run:

.. code-block:: shell

    python rmake.py --help

.. _rocthrust-build-cmake:

Build with CMake
================

You can build and install rocThrust with CMake on either Windows or Linux.

Set ``CXX`` to ``hipcc`` and set ``CMAKE_CXX_COMPILER`` to hipcc's absolute path. For example:

.. code-block:: shell

    CXX=hipcc
    CMAKE_CXX_COMPILER=/usr/bin/hipcc

Create the ``build`` directory under the ``rocthrust`` root directory, then change directory to the ``build`` directory:

.. code-block:: shell

    mkdir build
    cd build

Generate the rocThrust makefile using the ``cmake`` command:

.. code-block:: shell

    cmake ../. [-D<OPTION1=VALUE1> [-D<OPTION2=VALUE2>] ...]

The build options are:

* ``DISABLE_WERROR``: Set this to ``OFF`` to pass ``-Werror`` to the compiler. Default is ``ON``.
* ``BUILD_TEST``: Set this to ``ON`` to enable rocThrust tests. Default is ``OFF``.
* ``BUILD_HIPSTDPAR_TEST``: Set this to ``ON`` to enable HIPSTDPAR tests. Default is ``OFF``.
* ``BUILD_BENCHMARK``: Set this to ``ON`` to build rocThrust benchmarks. Default is ``OFF``.
* ``BUILD_EXAMPLE``: Set this to ``ON`` to build the rocThrust examples. Default is ``OFF``.
* ``BUILD_OFFLOAD_COMPRESS``: Set this to ``OFF`` to prevent the ``--offload-compress`` switch from being passed to the compiler and compressing the binary. On by default.
* ``USE_SYSTEM_LIB``: Set this to ``ON`` to use the installed ``ROCm`` libraries when building the tests. For this option to take effect, ``BUILD_TEST`` must be set to ``ON``. Default is ``OFF``.
* ``RNG_SEED_COUNT``: Set this to the non-repeatable random dataset count. Default is 0.
* ``PRNG_SEEDS``: Set this to the RNG seeds. The seeds must be passed as a semicolon-delimited array of 32-bit unsigned integers. To avoid command line parsing errors, enclose the entire option in quotation marks. For example, ``cmake "-DPRNG_SEEDS=1;2;3;4"``. ``-DPRNG_SEEDS=1`` is used by default.
* ``BUILD_ADDRESS_SANITIZER``: Set this to ``ON`` to build with the Clang address sanitizer enabled. Default is ``OFF``.
* ``EXTERNAL_DEPS_FORCE_DOWNLOAD``: Set this to ``ON`` to download the non-ROCm dependencies such as Google Test even if they're already installed. Default is ``OFF``.
* ``USE_HIPCXX``: Set this to ``ON`` to build with CMake HIP language support. Setting this to ``ON`` eliminates the need to use ``CXX=hipcc``. Default is ``OFF``.
* ``AMDGPU_TARGETS``: Set this to build the library, examples, tests, and benchmarks for specific architecture targets. 
* ``AMDGPU_TEST_TARGETS``: Set this to build tests for a subset of the architectures specified by ``AMDGPU_TARGETS``. When set, copies of the same test will be generated for each of the architectures listed. These tests can be run using ``ctest -R "TARGET_ARCHITECTURE"``. The list of targets must be separated by a semicolon (``;``).
* ``ROCPRIM_FETCH_METHOD`` and ``ROCRAND_FETCH_METHOD``: Set these to the method to use to download the rocPRIM and rocRAND components, respectively. Can be set to ``PACKAGE``, ``DOWNLOAD``, or ``MONOREPO``. Set to ``MONOREPO`` if the component isn't already installed and you're building rocThrust from within a clone of the `rocm-libraries <https://github.com/ROCm/rocm-libraries/>`_ repository that includes the component. Set to ``DOWNLOAD`` if the component isn't installed and you aren't in a clone of the ``rocm-libraries`` repository that includes the component. ``DOWNLOAD`` will clone the repository using sparse checkout so that only the necessary files are downloaded. Set to ``PACKAGE`` if the component is already installed. If the component isn't installed, it'll be downloaded from the repository in the same way as using the ``DOWNLOAD`` option. The default method is ``PACKAGE``.
* ``LINK_HIP_DEVICE_LIBS``: Set to ``OFF`` to prevent linking against device dependencies, including rocPRIM, and use the build for host-side computation only. Must be set to ``OFF`` when ``THRUST_DEVICE_SYSTEM`` is set to anything other than ``HIP``. Default is ``ON``.
* ``THRUST_DEVICE_SYSTEM``: Sets the device backend that will be used. Can be set to ``HIP`` for AMD GPU acceleration, ``CPP`` (C++) for sequential host-side compute, ``TBB`` (Thread Building Blocks) for parallel host-side compute, and ``OMP`` (OpenMP) for parallel host-side compute. For values other than ``HIP``,  ``LINK_HIP_DEVICE_LIBS`` must be set to ``OFF``.
* ``THRUST_HOST_SYSTEM``: Sets the host backend that will be used. If unspecified, this option defaults to ``CPP``. Valid values include ``CPP`` (C++), ``OMP`` (OpenMP), and ``TBB`` (Thread Building Blocks).

.. note::

    If you're using a version of git earlier than 2.25, ``-DROCPRIM_FETCH_METHOD=DOWNLOAD`` and ``-DROCRAND_FETCH_METHOD=DOWNLOAD`` will download the entire ``rocm-libraries`` repository.

Build rocThrust using the generated make file:

.. code-block:: shell

    make -j4

After you've built rocThrust, you can optionally generate tar, zip, and deb packages:

.. code-block:: shell

    make package

Finally, install rocThrust:

.. code-block:: shell

    make install
