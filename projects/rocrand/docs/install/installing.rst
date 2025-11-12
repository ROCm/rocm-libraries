.. meta::
   :description: rocRAND installation guide
   :keywords: rocRAND, ROCm, API, documentation, installation

.. _installing:

*******************************************************************
Installing and building rocRAND
*******************************************************************

This topic describes how to install or build rocRAND. The easiest method is to install the prebuilt
packages from the ROCm repositories, but this chapter also describes how to build rocRAND from source.

Requirements
===============================

rocRAND has the following prerequisites:

*  CMake (version 3.16 or later)
*  C++ compiler with C++17 support to build the library

   *  gcc version 9 or later is recommended
   *  Clang uses the development headers and libraries from gcc, so a recent version of it must still be installed when compiling with clang

*  C++ compiler with C++11 support to use the library

*  (Optional) Fortran compiler (This is only required for the Fortran wrapper. GFortran is recommended.)

*  (Optional) GoogleTest (This is only required to build and use the tests. Building the tests is enabled by default.)

   Use ``GTEST_ROOT`` to specify the GoogleTest location. For more information,
   see `FindGTest <https://cmake.org/cmake/help/latest/module/FindGTest.html>`_.
   
   .. note::

      If GoogleTest is not already installed, it will be automatically downloaded and built.

*  ROCm (see the :doc:`ROCm installation guide <rocm-install-on-linux:install/quick-start>`)
*  A HIP-clang compiler, which must be set as the C++ compiler on the ROCm platform.


Install using prebuilt packages
===============================

To install the prebuilt rocRAND packages, you require a ROCm-enabled platform.
For information on installing ROCm, see the :doc:`ROCm installation guide <rocm-install-on-linux:install/quick-start>`.
After installing ROCm or enabling the ROCm repositories, use the system package manager to install rocRAND.

For Ubuntu and Debian:

.. code-block:: shell

   sudo apt-get install rocrand

For CentOS-based systems:

.. code-block:: shell

   sudo yum install rocrand

For SLES:

.. code-block:: shell

   sudo dnf install rocrand

These commands install rocRAND in the ``/opt/rocm`` directory.

Build rocRAND from source
===============================

This section provides the information required to build rocRAND from source.


Obtaining the rocRAND source code
---------------------------------

The rocRAND source code is available from the `rocrand folder <https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocrand>`_ of
the `rocm-libraries <https://github.com/ROCm/rocm-libraries>`_ GitHub repository.
Use the branch that matches the ROCm version installed on the system.
The rocRAND source code can be cloned in two different ways.

.. note::

   For both methods, replace all occurrences of "x.y" in the commands with the version number matching your ROCm installation.
   For example, if you have ROCm 7.0 installed, clone the ``release/rocm-rel-7.0`` branch.

*  Clone the entire `rocm-libraries <https://github.com/ROCm/rocm-libraries>`_ repository.
   This is the default method and is the recommended option if you need to install other
   ROCm libraries alongside rocRAND. However, due to the download size, ``git clone``
   might take a significant amount of time to complete.

   On a system with ROCm x.y installed, use the following command to obtain the source code
   for rocRAND version x.y. Replace x.y with the actual version:

   .. code-block:: shell

      git clone -b release/rocm-rel-x.y https://github.com/ROCm/rocm-libraries.git

*  Clone the individual rocRAND project folder. This option only fetches the rocRAND source code,
   without any additional ROCm libraries. This significantly reduces the amount of time required
   to complete the clone operation. However, it requires Git 2.25 or later.
   To use this method to obtain the source code for rocRAND version x.y, run the following commands.
   Replace x.y with the actual version:

   .. code-block:: shell

      git clone -b release/rocm-rel-x.y --no-checkout --depth=1 --filter=tree:0 https://github.com/ROCm/rocm-libraries.git
      cd rocm-libraries
      git sparse-checkout set --cone projects/rocrand
      git checkout release/rocm-rel-x.y

.. note::

   To build ROCm 6.4 and earlier, use the rocRAND repository at `<https://github.com/ROCm/rocRAND>`_.
   For more information, see the documentation associated with the release you want to build.

Building the library
--------------------

After downloading the source code, use the installation script to build rocRAND:

.. code-block:: shell

   cd rocm-libraries/projects/rocrand
   ./install --install

This automatically builds all required dependencies, excluding HIP and Git, and installs the project
to ``/opt/rocm``. For further information, run the ``./install --help`` command.

Building with CMake
--------------------

For a more detailed installation process, build rocRAND manually using CMake.
This enables certain configuration options that are not available through the ``install`` script.
To build rocRAND, use CMake with the following configuration:

.. code-block:: shell

   cd rocm-libraries/projects/rocrand; mkdir build; cd build
   # Configure the project
   CXX=<compiler> cmake [options] ..
   # Build
   make -j4
   # Optionally, run the tests
   ctest --output-on-failure
   # Install
   [sudo] make install

To build for the ROCm platform, ``<compiler>`` should be set to ``hipcc``.
Additionally, the directory where ``FindHIP.cmake`` is installed needs to be passed explicitly
using ``-DCMAKE_MODULE_PATH``. By default, this file is installed in ``/opt/rocm/hip/cmake``.

In addition to the built-in CMake options, the following configuration options are available:

* ``BUILD_FORTRAN_WRAPPER``: Controls whether to build the Fortran wrapper. Defaults to ``OFF``.
* ``BUILD_TEST``: Controls whether to build the rocRAND tests. Defaults to ``OFF``.
* ``BUILD_BENCHMARK``: Controls whether to build the rocRAND benchmarks. Defaults to ``OFF``.
* ``BUILD_ADDRESS_SANITIZER`` Controls whether to build with address sanitization enabled. Defaults to ``OFF``.
* ``USE_SYSTEM_LIB``. Defaults to ``OFF``. Set to ``ON`` to use the installed ``ROCm`` libraries when building the
  tests. For this option to take effect, ``BUILD_TEST`` must be ``ON`` and the version of the installed ``rocrand``
  library must be compatible with the version of the tests.

To install rocRAND with a non-standard installation location of ROCm, pass ``-DCMAKE_PREFIX_PATH=</path/to/opt/rocm/>``
or set the environment variable ``ROCM_PATH`` to ``path/to/opt/rocm``.

rocRAND with HIP on Windows
===============================

rocRAND with HIP on Microsoft Windows has the following additional prerequisites:

*  Python 3.6 or higher (Only required for the install script)
*  Visual Studio 2019 with Clang support
*  Strawberry Perl


To install support for rocRAND and HIP on Windows, use the ``rmake.py`` Python script as follows:

.. code-block:: shell

   git clone https://github.com/ROCm/rocm-libraries.git 
   cd rocm-libraries/projects/rocrand

   # the -i option will install rocRAND to C:\hipSDK by default
   python rmake.py -i

   # the -c option will build all clients including unit tests
   python rmake.py -c

Any existing GoogleTest library in the system (especially static GoogleTest libraries built with other compilers)
might cause a build failure. If you encounter errors with the existing GoogleTest library or other dependencies,
pass the ``DEPENDENCIES_FORCE_DOWNLOAD`` flag to CMake to help solve the problem.

To disable inline assembly optimizations in rocRAND for both the host library and the device functions provided in ``rocrand_kernel.h``,
set the CMake option ``ENABLE_INLINE_ASM`` to ``OFF``.

Building the Python API wrapper
===============================

This section provides the information required to build the rocRAND Python API wrapper.

Requirements
--------------------

The rocRAND Python API Wrapper requires the following dependencies:

* rocRAND
* Python 3.5
* NumPy (will be installed automatically as a dependency if necessary)

.. note::

   If rocRAND is built from source but is either not installed or installed in a
   non-standard directory, set the ``ROCRAND_PATH`` environment variable
   to the library location. For example:

   .. code-block:: shell

      export ROCRAND_PATH=~rocm-libraries/projects/rocrand/build/library/

Installation
--------------------

The Python rocRAND module can be installed using ``pip``:

.. code-block:: shell

   cd rocm-libraries/projects/rocrand/python/rocrand
   pip install .

The tests can be executed as follows:

.. code-block:: shell

   cd rocm-libraries/projects/rocrand/python/rocrand
   python tests/rocrand_test.py

