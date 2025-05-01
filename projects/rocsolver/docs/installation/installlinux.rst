.. meta::
  :description: rocSOLVER documentation and API reference library
  :keywords: rocSOLVER, ROCm, API, documentation

.. _install-linux:

********************************************
Installing rocSOLVER on Linux
********************************************

Prerequisites
=================

rocSOLVER requires a ROCm-enabled platform. For more information, see the
:doc:`ROCm install guide <rocm-install-on-linux:index>`.

rocSOLVER also requires a compatible versions of rocBLAS installed on the system and may require
rocSPARSE depending on the build options. For more information, see the
:doc:`rocBLAS install guide <rocblas:install/Linux_Install_Guide>` and
:doc:`rocSPARSE install guide <rocsparse:install/install>`.

rocBLAS, rocSPARSE, and rocSOLVER are still under active development, and it is hard to define minimal
compatibility versions. For now, a good rule of thumb is to always use rocSOLVER together with the
matching rocBLAS and rocSPARSE version. For example, if you want to install rocSOLVER from the ROCm 5.6
release, then be sure that the ROCm 5.6 versions of rocBLAS and rocSPARSE are also installed. If you are
building the rocSOLVER branch tip, then you will need to build and install the rocBLAS and rocSPARSE branch
tips as well.


Installing from pre-built packages
====================================

If you have added the ROCm repositories to your Linux distribution, the latest release version of
rocSOLVER can be installed using a package manager. On Ubuntu, for example, use the commands:

.. code-block:: bash

    sudo apt-get update
    sudo apt-get install rocsolver


.. _linux-install-source:

Building and installing from source
=====================================

The `rocSOLVER source code <https://github.com/ROCm/rocSOLVER.git>`_ is hosted
on GitHub. Download the code and checkout the desired branch using:

.. code-block:: bash

    git clone -b <desired_branch_name> https://github.com/ROCmSoftwarePlatform/rocSOLVER.git
    cd rocSOLVER

To build from source, some external dependencies such as CMake and Python are required. Additionally, if the library clients
are to be built (by default they are not), then LAPACK and GoogleTest will be also required.
(The library clients, rocsolver-test and rocsolver-bench, provide the infrastructure for testing and benchmarking rocSOLVER.
For more details see the :ref:`rocSOLVER clients How-To <clients>`).

Using the install.sh script
-------------------------------

It is recommended that the provided install.sh script be used to build and install rocSOLVER. The command

.. code-block:: bash

    ./install.sh --help

gives detailed information on how to use this installation script.

Next, some common use cases are listed:

.. code-block:: bash

    ./install.sh

This command builds rocSOLVER and puts the generated library files, such as headers and
``librocsolver.so``, in the output directory: ``rocSOLVER/build/release/rocsolver-install``.
Other output files from the configuration and building process can also be found in the
``rocSOLVER/build`` and ``rocSOLVER/build/release`` directories. It is assumed that all
external library dependencies have been installed. It also assumes that the rocBLAS library
is located at ``/opt/rocm/rocblas``.

.. code-block:: bash

    ./install.sh --no-sparse

Use the ``--no-sparse`` to build rocSOLVER without rocSPARSE as a dependency. This will
disable sparse functionality within rocSOLVER and will cause all relevant methods to
return ``rocblas_status_not_implemented``.

.. code-block:: bash

    ./install.sh -g

Use the ``-g`` flag to build in debug mode. In this case the generated library files will be located at
``rocSOLVER/build/debug/rocsolver-install``.
Other output files from the configuration
and building process can also be found
in the ``rocSOLVER/build`` and ``rocSOLVER/build/debug`` directories.

.. code-block:: bash

    ./install.sh --lib_dir /home/user/rocsolverlib --build_dir buildoutput

Use ``--lib_dir`` and ``--build_dir`` to
change output directories.
In this case, for example, the installer
will put the headers and library files in
``/home/user/rocsolverlib``, while the outputs
of the configuration and building processes will
be in ``rocSOLVER/buildoutput`` and ``rocSOLVER/buildoutput/release``.
The selected output directories must be
local, otherwise the user may require sudo
privileges.
To install rocSOLVER system-wide, we
recommend the use of the ``-i`` flag as shown
below.

.. code-block:: bash

    ./install.sh --rocblas_dir /alternative/rocblas/location

Use ``--rocblas_dir`` to change where the build system will search for the rocBLAS
library. In this case, for example, the installer will look for the rocBLAS library at
``/alternative/rocblas/location``. Similarly, you may use ``--rocsparse_dir`` to specify
an alternative location for the rocSPARSE library.

.. code-block:: bash

    ./install.sh -s

With the ``-s`` flag, the installer will
generate a static library
(``librocsolver.a``) instead.

.. code-block:: bash

    ./install.sh -d

With the ``-d`` flag, the installer will first
install all the external dependencies
required by the rocSOLVER library in
``/usr/local``.
This flag only needs to be used once. For
subsequent invocations of install.sh it is
not necessary to rebuild the dependencies.

.. code-block:: bash

    ./install.sh -c

With the ``-c`` flag, the installer will
additionally build the library clients
``rocsolver-bench`` and
``rocsolver-test``.
The binaries will be located at
``rocSOLVER/build/release/clients/staging``.
It is assumed that all external dependencies
for the client have been installed.

.. code-block:: bash

    ./install.sh -dc

By combining the ``-c`` and ``-d`` flags, the installer
will also install all the external
dependencies required by rocSOLVER clients.
Again, the ``-d`` flag only needs to be used once.

.. code-block:: bash

    ./install.sh -i

With the ``-i`` flag, the installer will
additionally
generate a pre-built rocSOLVER package and
install it, using a suitable package
manager, at the standard location:
``/opt/rocm/rocsolver``.
This is the preferred approach to install
rocSOLVER on a system, as it will allow
the library to be safely removed using the
package manager.

.. code-block:: bash

    ./install.sh -p

With the ``-p`` flag, the installer will also
generate the rocSOLVER package, but it will
not be installed.

.. code-block:: bash

    ./install.sh -i --install_dir /package/install/path

When generating a package, use ``--install_dir`` to change the directory where
it will be installed.
In this case, for example, the rocSOLVER
package will be installed at
``/package/install/path``.

Manual building and installation
--------------------------------------

Manual installation of all the external dependencies is not an easy task. Get more information on
how to install each dependency at the corresponding documentation sites:

* `CMake <https://cmake.org/>`_ (version 3.16 is recommended).
* `LAPACK <https://github.com/Reference-LAPACK/lapack-release>`_ (which internally depends on a Fortran compiler), and
* `GoogleTest <https://github.com/google/googletest>`_
* `fmt <https://github.com/fmtlib/fmt>`_

Once all dependencies are installed (including ROCm, rocBLAS, and rocSPARSE), rocSOLVER
can be manually built using a combination of CMake and Make commands. Using CMake options
can provide more flexibility in tailoring the building and installation process. Here we
provide a list of examples of common use cases (see the CMake documentation for more
information on CMake options).

.. code-block:: bash

    mkdir -p build/release && cd build/release
    cmake --toolchain=toolchain-linux.cmake -DCMAKE_INSTALL_PREFIX=rocsolver-install ../..
    make install

This is equivalent to ``./install.sh``.

.. code-block:: bash

    mkdir -p buildoutput/release && cd buildoutput/release
    cmake --toolchain=toolchain-linux.cmake -DCMAKE_INSTALL_PREFIX=/home/user/rocsolverlib ../..
    make install

This is equivalent to ``./install.sh --lib_dir /home/user/rocsolverlib --build_dir buildoutput``.

.. code-block:: bash

    mkdir -p build/release && cd build/release
    cmake --toolchain=toolchain-linux.cmake -DCMAKE_INSTALL_PREFIX=rocsolver-install -DBUILD_WITH_SPARSE=OFF ../..
    make install

This is equivalent to ``./install.sh --no-sparse``.

.. code-block:: bash

    mkdir -p build/release && cd build/release
    cmake --toolchain=toolchain-linux.cmake -DCMAKE_INSTALL_PREFIX=rocsolver-install -Drocblas_DIR=/alternative/rocblas/location ../..
    make install

This is equivalent to ``./install.sh --rocblas_dir /alternative/rocblas/location``.

.. code-block:: bash

    mkdir -p build/debug && cd build/debug
    cmake --toolchain=toolchain-linux.cmake -DCMAKE_INSTALL_PREFIX=rocsolver-install -DCMAKE_BUILD_TYPE=Debug ../..
    make install

This is equivalent to ``./install.sh -g``.

.. code-block:: bash

    mkdir -p build/release && cd build/release
    cmake --toolchain=toolchain-linux.cmake -DCMAKE_INSTALL_PREFIX=rocsolver-install -DBUILD_SHARED_LIBS=OFF ../..
    make install

This is equivalent to ``./install.sh -s``.

.. code-block:: bash

    mkdir -p build/release && cd build/release
    cmake --toolchain=toolchain-linux.cmake -DCMAKE_INSTALL_PREFIX=rocsolver-install -DBUILD_CLIENTS_TESTS=ON -DBUILD_CLIENTS_BENCHMARKS=ON ../..
    make install

This is equivalent to ``./install.sh -c``.

.. code-block:: bash

    mkdir -p build/release && cd build/release
    cmake --toolchain=toolchain-linux.cmake -DCMAKE_INSTALL_PREFIX=rocsolver-install -DCPACK_SET_DESTDIR=OFF -DCPACK_PACKAGING_INSTALL_PREFIX=/opt/rocm ../..
    make install
    make package

This is equivalent to ``./install.sh -p``.

.. code-block:: bash

    mkdir -p build/release && cd build/release
    cmake --toolchain=toolchain-linux.cmake -DCMAKE_INSTALL_PREFIX=rocsolver-install -DCPACK_SET_DESTDIR=OFF -DCPACK_PACKAGING_INSTALL_PREFIX=/package/install/path ../..
    make install
    make package
    sudo dpkg -i rocsolver[-\_]*.deb

On an Ubuntu system, for example, this would be equivalent to ``./install.sh -i --install_dir /package/install/path``.
