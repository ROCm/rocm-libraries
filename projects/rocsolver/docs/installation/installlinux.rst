.. meta::
  :description: rocSOLVER documentation and API reference library
  :keywords: rocSOLVER, ROCm, API, documentation

.. _install-linux:

*********************************
Installing and building rocSOLVER
*********************************

This topic explains how to install and build the rocSOLVER library on the Linux platform.

Prerequisites
=================

rocSOLVER requires a ROCm-enabled platform. For more information, see the
:doc:`ROCm install guide <rocm-install-on-linux:index>`.

rocSOLVER also requires a compatible version of rocBLAS installed on the system and might require
rocSPARSE, depending on the build options. For more information, see
:doc:`rocBLAS <rocblas:index>` and
:doc:`rocSPARSE <rocsparse:index>`.

In terms of compatibility, it's best to always use rocSOLVER with the
matching rocBLAS and rocSPARSE versions. For example, to install the rocSOLVER version from ROCm 6.4,
ensure the ROCm 6.4 versions of rocBLAS and rocSPARSE are also installed.

Install using prebuilt packages
====================================

If you have added the ROCm repositories to your Linux system, you can install the latest release version of
rocSOLVER using a package manager. For example, use these commands on Ubuntu:

.. code-block:: bash

    sudo apt-get update
    sudo apt-get install rocsolver

.. _linux-install-source:

Build and install from source
=====================================

The `rocSOLVER source code <https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocsolver>`_ is hosted
in the `rocm-libraries GitHub <https://github.com/ROCm/rocm-libraries>`_.
To download the rocSOLVER source code, use either a sparse checkout or a full clone of the rocm-libraries repository,
as follows.

.. note::

   To build rocSOLVER for ROCm 7.0.2 and earlier, use the rocSOLVER repository at `<https://github.com/ROCm/rocSOLVER>`_.
   For more information, see the documentation associated with the release you want to build.

To limit your local checkout to only the rocSOLVER project, configure ``sparse-checkout`` before cloning.
This uses the Git partial clone feature (``--filter=blob:none``) to reduce how much data is downloaded.
Use the following commands for a sparse checkout:

.. code-block:: bash

   git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-libraries.git
   cd rocm-libraries
   git sparse-checkout init --cone
   git sparse-checkout set projects/rocsolver # add projects/rocblas projects/rocsparse to include dependencies
   git checkout develop # or use the branch you want to work with
   cd projects/rocsolver

To clone the entire rocm-libraries repository, use the following commands. This process takes more time,
but is recommended if you want to work with a large number of libraries.

.. code-block:: bash

   git clone https://github.com/ROCm/rocm-libraries.git
   cd rocm-libraries/projects/rocsolver

To build from source, external dependencies, such as CMake and Python, are required. Additionally, if 
you are building the library clients (which are not built by default), LAPACK and GoogleTest are also required.

.. note::

   The library clients, which include rocsolver-test and rocsolver-bench, provide the infrastructure for testing and benchmarking rocSOLVER.
   For more details, see :doc:`rocSOLVER clients <../howto/clients>`.

Using the install script
-------------------------------

The recommended method of building and installing rocSOLVER is the ``install.sh`` script.
The ``help`` command provides detailed information on how to use the script.

.. note::

   Run these commands from the ``rocm-libraries/projects/rocsolver`` directory.

.. code-block:: bash

   ./install.sh --help

This section discusses how to use the install script for some common use cases.

*  The following command builds rocSOLVER and places the generated library files, such as headers and
   ``librocsolver.so``, in the output directory ``rocSOLVER/build/release/rocsolver-install``.
   Other output files from the configuration and build process can be found in the
   ``rocSOLVER/build`` and ``rocSOLVER/build/release`` directories. This command assumes that all
   external library dependencies have been installed and that the rocBLAS library
   is located at ``/opt/rocm/rocblas``.

   .. code-block:: bash

      ./install.sh

*  The ``--no-sparse`` option builds rocSOLVER without rocSPARSE as a dependency. This
   disables the SPARSE functionality within rocSOLVER and causes all related methods to
   return ``rocblas_status_not_implemented``.

   .. code-block:: bash

      ./install.sh --no-sparse

*  Use the ``-g`` flag to build rocSOLVER in debug mode. In this case, the generated library files can be found at
   ``rocSOLVER/build/debug/rocsolver-install``.
   Other output files from the configuration and build process can be found
   in the ``rocSOLVER/build`` and ``rocSOLVER/build/debug`` directories.

   .. code-block:: bash

      ./install.sh -g

*  Use ``--lib_dir`` and ``--build_dir`` to change output directories.
   In this example, the installer places the headers and library files in
   ``/home/user/rocsolverlib`` and the outputs
   from the configuration and build processes
   in ``rocSOLVER/buildoutput`` and ``rocSOLVER/buildoutput/release``.
   The designated output directories must be
   local. Otherwise, you might require ``sudo`` privileges.
   For a system-wide rocSOLVER installation,
   use the ``-i`` flag, as shown below.

   .. code-block:: bash

      ./install.sh --lib_dir /home/user/rocsolverlib --build_dir buildoutput

*  Use ``--rocblas_dir`` to change where the build system searches for the rocBLAS
   library. In this case, the installer looks for the rocBLAS library at
   ``/alternative/rocblas/location``. Similarly, you can use ``--rocsparse_dir`` to specify
   an alternative location for the rocSPARSE library.

   .. code-block:: bash

      ./install.sh --rocblas_dir /alternative/rocblas/location

*  When the ``-s`` flag is provided, the installer generates a static library
   (``librocsolver.a``) instead.

   .. code-block:: bash

      ./install.sh -s

*  The ``-d`` flag installs all the external dependencies
   required by the rocSOLVER library in
   ``/usr/local``.
   This flag only needs to be used once.
   Subsequent invocations of ``install.sh`` do
   not have to rebuild the dependencies.

   .. code-block:: bash

      ./install.sh -d

*  The ``-c`` flag
   also builds the library clients
   ``rocsolver-bench`` and ``rocsolver-test``.
   The binaries are located at
   ``rocSOLVER/build/release/clients/staging``.
   The script assumes that all external dependencies
   for the clients have been installed.

   .. code-block:: bash

      ./install.sh -c

*  Combining the ``-c`` and ``-d`` flags
   installs all external
   dependencies required by the rocSOLVER clients.
   The ``-d`` flag only needs to be used once.

   .. code-block:: bash

      ./install.sh -dc

*  The ``-i`` flag generates a prebuilt rocSOLVER package and
   installs it, using the relevant package
   manager, at the standard ``/opt/rocm/rocsolver`` location.
   This is the preferred approach for installing
   rocSOLVER on a system because it allows
   the library to be safely removed using the
   package manager.

   .. code-block:: bash

      ./install.sh -i

*  With the ``-p`` flag, the installer
   generates the rocSOLVER package but doesn't install it.

   .. code-block:: bash

      ./install.sh -p

*  When generating a package, use ``--install_dir`` to change the directory where
   the package is installed.
   In this case, the rocSOLVER
   package is installed at ``/package/install/path``.

   .. code-block:: bash

      ./install.sh -i --install_dir /package/install/path

Manual building and installation
--------------------------------------

Manual installation of all the external dependencies is a complex task. For more information on
how to install each dependency, see the corresponding documentation:

*  `CMake <https://cmake.org/>`_ (Version 3.16 is recommended)
*  `LAPACK <https://github.com/Reference-LAPACK/lapack-release>`_ (Depends internally on a Fortran compiler)
*  `GoogleTest <https://github.com/google/googletest>`_
*  `fmt <https://github.com/fmtlib/fmt>`_

After all dependencies are installed (including ROCm, rocBLAS, and rocSPARSE), you can manually
build rocSOLVER by using a combination of CMake and Make commands. The CMake options
provide more flexibility to modify the building and installation process.
This list provides some examples of common use cases. (See the CMake documentation for more
information about the options.)

This is equivalent to ``./install.sh``.

.. note::

   Run these commands from the ``rocm-libraries/projects/rocsolver`` directory.

.. code-block:: bash

    mkdir -p build/release && cd build/release
    cmake --toolchain=toolchain-linux.cmake -DCMAKE_INSTALL_PREFIX=rocsolver-install ../..
    make install

This is equivalent to ``./install.sh --lib_dir /home/user/rocsolverlib --build_dir buildoutput``.

.. code-block:: bash

    mkdir -p buildoutput/release && cd buildoutput/release
    cmake --toolchain=toolchain-linux.cmake -DCMAKE_INSTALL_PREFIX=/home/user/rocsolverlib ../..
    make install

This is equivalent to ``./install.sh --no-sparse``.

.. code-block:: bash

    mkdir -p build/release && cd build/release
    cmake --toolchain=toolchain-linux.cmake -DCMAKE_INSTALL_PREFIX=rocsolver-install -DBUILD_WITH_SPARSE=OFF ../..
    make install

This is equivalent to ``./install.sh --rocblas_dir /alternative/rocblas/location``.

.. code-block:: bash

    mkdir -p build/release && cd build/release
    cmake --toolchain=toolchain-linux.cmake -DCMAKE_INSTALL_PREFIX=rocsolver-install -Drocblas_DIR=/alternative/rocblas/location ../..
    make install

This is equivalent to ``./install.sh -g``.

.. code-block:: bash

    mkdir -p build/debug && cd build/debug
    cmake --toolchain=toolchain-linux.cmake -DCMAKE_INSTALL_PREFIX=rocsolver-install -DCMAKE_BUILD_TYPE=Debug ../..
    make install

This is equivalent to ``./install.sh -s``.

.. code-block:: bash

    mkdir -p build/release && cd build/release
    cmake --toolchain=toolchain-linux.cmake -DCMAKE_INSTALL_PREFIX=rocsolver-install -DBUILD_SHARED_LIBS=OFF ../..
    make install

This is equivalent to ``./install.sh -c``.

.. code-block:: bash

    mkdir -p build/release && cd build/release
    cmake --toolchain=toolchain-linux.cmake -DCMAKE_INSTALL_PREFIX=rocsolver-install -DBUILD_CLIENTS_TESTS=ON -DBUILD_CLIENTS_BENCHMARKS=ON ../..
    make install

This is equivalent to ``./install.sh -p``.

.. code-block:: bash

    mkdir -p build/release && cd build/release
    cmake --toolchain=toolchain-linux.cmake -DCMAKE_INSTALL_PREFIX=rocsolver-install -DCPACK_SET_DESTDIR=OFF -DCPACK_PACKAGING_INSTALL_PREFIX=/opt/rocm ../..
    make install
    make package

On an Ubuntu system, this is equivalent to ``./install.sh -i --install_dir /package/install/path``.

.. code-block:: bash

    mkdir -p build/release && cd build/release
    cmake --toolchain=toolchain-linux.cmake -DCMAKE_INSTALL_PREFIX=rocsolver-install -DCPACK_SET_DESTDIR=OFF -DCPACK_PACKAGING_INSTALL_PREFIX=/package/install/path ../..
    make install
    make package
    sudo dpkg -i rocsolver[-\_]*.deb
