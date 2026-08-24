.. meta::
  :description: Build and install RPP from source
  :keywords: install, building, RPP, AMD, ROCm, source code, cmake, Linux

.. _build-from-source:

*********************
Build RPP from source
*********************

To build RPP as part of the ROCm Core SDK, see `TheRock build
instructions
<https://github.com/ROCm/TheRock/blob/main/docs/development/README.md>`__.
TheRock is the recommended way to build ROCm components from source.

Alternatively, you can build RPP standalone using the following
instructions.

.. _rpp-prerequisites:

Prerequisites
=============

RPP on Linux requires `ROCm <https://rocm.docs.amd.com/en/latest/>`_ for HIP
backends.

RPP has been tested on the following Linux environments:

* Ubuntu 22.04 and 24.04
* RHEL 8 and 9
* SLES 15 SP7

See `Supported operating systems
<https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html#supported-operating-systems>`__
for the complete list of ROCm supported Linux environments.

On HIP backends, RPP runs on ROCm-supported AMD GPUs, including AMD Instinct
(CDNA) and AMD Radeon (RDNA) graphics. See `Supported GPUs
<https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html>`__
for details.

On CPU-only backends, also referred to as HOST backends, RPP requires CPUs that
support PCIe atomics.

`CMake version 3.10 or later <https://cmake.org/>`_ and C++17 are required.

The following compilers and libraries are also required:

* HIP
* OpenMP
* half, the half-precision floating-point library, version 1.12.0 or later
* libstdc++-12-dev for Ubuntu 22.04 only
* Clang version 5.0.1 or later for CPU-only backends
* AMD Clang++ version 18.0.0 or later for HIP backends

With the following compiler support:

* OpenMP
* Threads

The `test suite prerequisites
<https://github.com/ROCm/rocm-libraries/blob/develop/projects/rpp/utilities/test_suite/README.md>`__
are required to build and run the RPP test suite.

.. _rpp-get-source:

Get the RPP source code
=======================

The RPP source code is available from the `ROCm libraries GitHub repository
<https://github.com/ROCm/rocm-libraries/tree/develop/projects/rpp>`_.
Use sparse checkout when cloning the RPP project:

.. code-block:: shell

  git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-libraries.git
  cd rocm-libraries
  git sparse-checkout init --cone
  git sparse-checkout set projects/rpp

Then use ``git checkout`` to check out the branch you need.

The develop branch is intended for users who want to preview new features or
contribute to the RPP code base.

If you don't intend to contribute to the RPP code base and won't be previewing
features, use a branch that matches the version of ROCm installed on your
system.

.. _rpp-build-linux:

Build on Linux
==============

RPP is built on Linux using CMake.

Create a build directory under the cloned ``rpp`` directory, then change
directory to the build directory:

.. tab-set::

  .. tab-item:: HIP

    .. code-block:: shell

        cd projects/rpp
        mkdir build-hip
        cd build-hip
        cmake ..
        make -j8
        sudo make install

  .. tab-item:: CPU-only

    .. code-block:: shell

        cd projects/rpp
        mkdir build-cpu
        cd build-cpu
        cmake -DBACKEND=CPU ..
        make -j8
        sudo make install

The available CMake options are:

* ``BACKEND``: Set to ``CPU`` for CPU-only builds. ``HIP`` by default.
* ``CMAKE_BUILD_TYPE``: Set to ``Debug`` or ``Release``. ``Release`` by default.
* ``RPP_AUDIO_SUPPORT``: Set to ``ON`` to enable audio augmentations. ``OFF`` by
  default.

.. _verify-install:

Verify the installation
=======================

After installation, verify that RPP files are installed in the expected
locations:

* Libraries: ``${ROCM_PATH}/lib``
* Header files: ``${ROCM_PATH}/include/rpp``
* Test suite: ``${ROCM_PATH}/share/rpp/test``
* License files: ``${ROCM_PATH}/share/doc/rpp``

To run the installed test suite, install the `test suite prerequisites
<https://github.com/ROCm/rocm-libraries/blob/develop/projects/rpp/utilities/test_suite/README.md>`__
first.

.. code-block:: shell

    mkdir rpp-test
    cd rpp-test
    cmake ${ROCM_PATH}/share/rpp/test/
    ctest -VV
