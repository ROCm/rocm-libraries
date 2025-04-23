.. meta::
  :description: Install guide for hipFORT
  :keywords: install, hipFORT, AMD, ROCm, building, tests

*********************************
Installing and building hipFORT
*********************************

This topic discusses how to build hipFORT from source and use the Makefile.
It also provides information on how to build and run the tests.

Prerequisites
===============

hipFORT requires GFortran version 7.5.0 or newer.
For more information, see the `GFortran website. <https://fortran-lang.org/learn/os_setup/install_gfortran/>`_

.. _build-test-hipfort-from-source:

Building and testing hipFORT from source
==========================================

#. Ensure you have installed ``gfortran``, ``git``, ``cmake``, and :doc:`HIP <hip:index>`.
#. Build, install, and test hipFORT from source using the following commands:

   .. code-block:: shell

      git clone https://github.com/ROCm/hipfort.git
      cd hipfort
      cmake -S. -Bbuild -DHIPFORT_INSTALL_DIR=/tmp/hipfort -DBUILD_TESTING=ON
      make -C build
      make -C build check

   .. note::

      The hipFORT installation compiles backends for both ROCm (``hipfort-amdgcn``) and NVIDIA CUDA (``hipfort-nvptx``).
      When installing hipFORT from source, you do not need to specify the ``HIP_PLATFORM`` environment variable.

Customizing the build
-----------------------

You can customize the build by setting the following environment variables:

*  ``FC``: The Fortran compiler to use
*  ``FFLAGS``: Compiler flags for building hipFORT

or by setting the CMake cache variables:

*  ``CMAKE_BUILD_TYPE``: Set to ``RELEASE``, ``TESTING``, or ``DEBUG``
*  ``CMAKE_AR``: Static archive command
*  ``CMAKE_RANLIB``: The ``ranlib`` used to create the static archive
*  ``CMAKE_INSTALL_PREFIX``: The install directory

hipfc wrapper compiler and Makefile.hipfort
================================================

Along with Fortran interfaces for the HIP and ROCm libraries, hipFORT ships the hipfc wrapper compiler
and a ``Makefile.fort`` file that can be included in a project's build system.
hipfc can be found in the ``bin/`` directory, while ``Makefile.hipfort`` is in the ``share/hipfort`` directory
of the repository.

Both build mechanisms can be configured using a number of environment variables, but hipfc
includes a greater number of command-line options. You can list these options using the following command:

.. code-block:: shell

   hipfc -h

.. note::

   The hipfc wrapper compiler is deprecated and will be removed in a future release. Users are
   encouraged to call the Fortran or HIP compilers directly instead of relying on the hipfc wrapper.
   The hipFORT component provides exported CMake targets that can be used to link to the appropriate
   ROCm libraries.

The following table lists the most important environment variables:

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Environment variable
     - Description
     - Default
   * - ``HIP_PLATFORM``
     - The platform to compile for (``amd`` or ``nvidia``)
     - ``amd``
   * - ``ROCM_PATH``
     - Path to the ROCm installation
     - ``/opt/rocm``
   * - ``CUDA_PATH``
     - Path to the CUDA installation
     - ``/usr/local/cuda``
   * - ``FC``
     -  Fortran compiler to use
     - ``gfortran``


Examples and tests
====================

The examples in the ``f2003`` and ``f2008`` subdirectories of the ``test`` folder in the repository
also serve as tests. Both test collections implement the same tests. However, the ``f2008`` tests require the
Fortran compiler to support the Fortran 2008 standard or newer.
The ``f2003`` tests only require support for the Fortran 2003 (`f2003`) standard.
The ``f2003`` and ``f2008`` subdirectories are further subdivided into tests for the various hip* and roc* libraries.

Building a single test
-----------------------

To compile for AMD devices, call the ``make`` command from within the test directories.

.. note::

   The ``make`` targets append the linker flags for AMD devices to the ``CFLAGS`` variable by default.

To compile for CUDA devices, use the ``make`` command to build as follows:

.. code-block:: shell

   make CFLAGS="--offload-arch=sm_70 <libs>"


Substitute ``<libs>`` for the actual library, such as ``-lcublas`` or ``-lcusparse``.

.. note:: 

        Only the hip* tests can be compiled for CUDA devices. The roc* tests cannot be compiled for CUDA devices. 

To compile using hipfc, run the following command:

.. code-block:: shell

   hipfc <CFLAGS> <test_name>.f03 -o <test_name>


The ``vecadd`` test is the only exception. It also requires the HIP C++ source.

.. code-block:: shell

   hipfc <CFLAGS> hip_implementation.cpp main.f03 -o main


Building and running all tests
-------------------------------

You can build and run the whole test collection from the ``build/`` folder
(see :ref:`build-test-hipfort-from-source`) or
from the ``test/`` folder. Follow the instructions below corresponding to the platform you are building for.

AMD devices
^^^^^^^^^^^^

The command to run all tests, as shown below, expects the ROCm math libraries to be found at ``/opt/rocm``.
To specify a different ROCm location, use the ``ROCM_PATH`` environment variable.

.. note::

   When using older ROCm versions, you might need to manually set the environment variable ``HIP_PLATFORM``
   to ``hcc`` before running the tests.

To run the tests from the ``build`` subdirectory, use these commands:

.. code-block:: shell

   cd build/
   make all-tests-run

Alternatively, run the following commands from the ``test`` directory:

.. code-block:: shell

   cd test/
   make run_all

CUDA devices
^^^^^^^^^^^^

To run the tests as shown below, CUDA must be found at ``/usr/local/cuda``.
To specify a different CUDA location, use the ``CUDA_PATH`` environment variable or provide it as part of the ``CFLAGS`` variable
by appending ``-cuda-path <path_to_cuda>``. 

.. note::

   Choose the offload architecture value according to the device being used.

To run the tests from the ``build`` subdirectory, use these commands:

.. code-block:: shell

   cd build/
   make all-tests-run CFLAGS="--offload-arch=sm_70 -lcublas -lcusolver -lcufft"


Alternatively, run the following commands from the ``test`` directory:

.. code-block:: shell

   cd test/
   make run_all CFLAGS="--offload-arch=sm_70 -lcublas -lcusolver -lcufft"


