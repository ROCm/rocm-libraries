.. meta::
   :description: Build rocALUTION from source
   :keywords: rocALUTION, ROCm, library, API, build, install, HIP, source

.. _build-rocalution-from-source:

******************************
Build rocALUTION from source
******************************

You don't need to build rocALUTION from source if you install the prebuilt
packages described in :doc:`Install rocALUTION <./rocALUTION-install>`. To build
rocALUTION from source, follow the instructions in this section.

When you build rocALUTION from source, select supported versions of the math
library dependencies. Given a version of rocALUTION, you must use versions of
these dependencies that are the same or later.

.. _build-rocalution-prerequisites:

Prerequisites
=============

You need the following to build rocALUTION:

* `git <https://git-scm.com/>`__
* `CMake <https://cmake.org/>`__ (version 3.5 or later)
* A ROCm installation providing the ``amdclang++`` compiler

If you build with HIP support enabled, which is the default when HIP is
available, you also need:

* :doc:`rocBLAS <rocblas:index>`
* :doc:`rocSPARSE <rocsparse:index>`
* :doc:`rocPRIM <rocprim:index>`
* :doc:`rocRAND <rocrand:index>`

For multi-node configurations, you must also install:

* `OpenMP <https://www.openmp.org/>`__
* `MPI <https://www.mcs.anl.gov/research/projects/mpi/>`__

Complete the `ROCm installation prerequisites <https://rocm.docs.amd.com/en/latest/install/rocm.html?fam=all&w=compute&os=ubuntu&ubuntu-ver=26.04&i=pkgman>`__ to
install prerequisites and configure GPU access permissions.

Install the build dependencies with your distribution's package manager:

.. tab-set::

   .. tab-item:: Debian-based distros

      .. code-block:: bash

         sudo apt install amdrocm-llvm-dev amdrocm-runtime-dev amdrocm-blas-dev \
                          amdrocm-sparse-dev amdrocm-rand-dev amdrocm-ccl-dev

   .. tab-item:: RHEL-based distros

      .. code-block:: bash

         sudo dnf install amdrocm-llvm-devel amdrocm-runtime-devel amdrocm-blas-devel \
                          amdrocm-sparse-devel amdrocm-rand-devel amdrocm-ccl-devel

   .. tab-item:: SLES

      .. code-block:: bash

         sudo zypper install amdrocm-llvm-devel amdrocm-runtime-devel amdrocm-blas-devel \
                             amdrocm-sparse-devel amdrocm-rand-devel amdrocm-ccl-devel

.. note::

   rocPRIM is provided by the ``amdrocm-ccl-dev`` package on Debian-based
   distributions and ``amdrocm-ccl-devel`` on RPM-based distributions. There is
   no package named for rocPRIM directly. Without this package, CMake
   configuration fails with ``Could not find a package configuration file
   provided by "rocprim"``.

.. _download-rocalution:

Download rocALUTION
===================

You can find the rocALUTION source code in the
`rocALUTION folder <https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocalution>`__
of the `rocm-libraries GitHub <https://github.com/ROCm/rocm-libraries>`__
repository.

To limit your local checkout to only the rocALUTION project, configure
``sparse-checkout`` before you clone. The partial clone feature
(``--filter=blob:none``) reduces how much data you download. Use the following
commands for a sparse checkout:

.. code-block:: shell

   git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-libraries.git
   cd rocm-libraries
   git sparse-checkout init --cone
   git sparse-checkout set projects/rocalution
   git checkout develop
   cd projects/rocalution

.. note::

   To include the rocBLAS, rocSPARSE, rocRAND, and rocPRIM dependencies, set
   the projects for the sparse checkout using ``git sparse-checkout set
   projects/rocalution projects/rocblas projects/rocsparse projects/rocrand
   projects/rocprim``.

To download the ``develop`` branch for all projects in rocm-libraries, use these
commands. This process takes longer, but use it if you work with a large number
of libraries.

.. code-block:: shell

   git clone -b develop https://github.com/ROCm/rocm-libraries.git
   cd rocm-libraries/projects/rocalution

.. _build-rocalution-install-script:

Build rocALUTION using the install script
=========================================

Use the ``install.sh`` script to build and install rocALUTION. The following
tables describe how to build different packages of the library, including the
dependencies and clients.

.. note::

   Run the ``install.sh`` script from the ``projects/rocalution`` directory.

Use install.sh to build rocALUTION with dependencies
----------------------------------------------------

The following table lists the common ways to use ``install.sh`` to build the
rocALUTION dependencies and library.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Command
     - Description
   * - ``./install.sh -h``
     - Print the help information.
   * - ``./install.sh -d``
     - Install the build dependencies, then build the library in your local directory. The script installs packages with your distribution's package manager and prompts you for sudo access. Use the ``-d`` flag only once. For subsequent invocations of ``install.sh``, you don't need to reinstall the dependencies.
   * - ``./install.sh``
     - Build the library in your local directory. The script assumes the dependencies are available.
   * - ``./install.sh -i``
     - Build the library, then build and install the rocALUTION package in ``/opt/rocm``. The script prompts you for sudo access. This installs rocALUTION for all users.
   * - ``./install.sh --mpi=<dir> -i``
     - Build the library with MPI support enabled, then build and install the rocALUTION package in ``/opt/rocm``.

Use install.sh to build rocALUTION with dependencies and clients
----------------------------------------------------------------

The clients contain example code, unit tests, and benchmarks. The following
table lists common ways to use ``install.sh`` to build the library,
dependencies, and clients.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Command
     - Description
   * - ``./install.sh -dc``
     - Install the dependencies, then build the library and clients in your local directory. Use the ``-d`` flag only once. For subsequent invocations of ``install.sh``, you don't need to reinstall the dependencies.
   * - ``./install.sh -c``
     - Build the library and clients in your local directory. The script assumes the dependencies are available.
   * - ``./install.sh -idc``
     - Install the dependencies, build the library and clients, then build and install the rocALUTION package in ``/opt/rocm``. The script prompts you for sudo access. This installs rocALUTION for all users.
   * - ``./install.sh -ic``
     - Build the library and clients, then build and install the rocALUTION package in ``/opt/rocm``. The script prompts you for sudo access. This installs rocALUTION for all users.
   * - ``./install.sh -o``
     - Build the client executables using an already installed version of the library. Client-only builds place the sample binaries in ``build/release/staging``.

.. _build-rocalution-make:

Build rocALUTION using individual make commands
===============================================

The rocALUTION library contains both host and device code, so specify the HIP
compiler during the CMake configuration process.

.. note::

   Run these commands from the ``projects/rocalution`` directory.

.. note::

   You need CMake 3.5 or later to build rocALUTION.

Set ``ROCM_PATH`` to your ROCm installation directory. The following example uses
ROCm 7.14:

You can build rocALUTION using the following commands:

.. code-block:: shell

   # Set the ROCm installation path
   export ROCM_PATH=/opt/rocm/core-7.14

   # Create and change to build directory
   mkdir -p build/release
   cd build/release

   # The install path defaults to ROCM_PATH; use -DCMAKE_INSTALL_PREFIX= to adjust it
   CXX=$ROCM_PATH/bin/amdclang++ cmake ../.. -DROCM_PATH=$ROCM_PATH

   # Compile rocALUTION library
   make -j$(nproc)

   # Install rocALUTION
   make install

You can also configure the following optional CMake directives:

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Directive
     - Default
     - Description
   * - ``SUPPORT_HIP``
     - ``ON`` when HIP is found
     - Enable HIP support.
   * - ``SUPPORT_OMP``
     - ``ON``
     - Enable OpenMP support.
   * - ``SUPPORT_MPI``
     - ``OFF``
     - Enable MPI support for multi-node execution.
   * - ``BUILD_SHARED_LIBS``
     - ``ON``
     - Build rocALUTION as a shared library. This is recommended.
   * - ``BUILD_CLIENTS_SAMPLES``
     - ``ON``
     - Build the example programs.

.. note::

   ``SUPPORT_HIP`` is enabled only if HIP is found during configuration. If HIP
   is not found, the build completes successfully as a CPU-only library and
   prints ``HIP not found. Compiling WITHOUT HIP support.`` Check the
   configuration output to confirm that HIP support is enabled.

For example, to build rocALUTION with MPI support enabled:

.. code-block:: shell

   CXX=$ROCM_PATH/bin/amdclang++ cmake ../.. -DSUPPORT_MPI=ON -DROCM_PATH=$ROCM_PATH

.. _testing-rocalution:

Testing rocALUTION
==================

Verify that you built and installed rocALUTION correctly by running the
Conjugate Gradient (CG) solver client on a sample Laplacian matrix.

These steps assume you built rocALUTION with client applications enabled, which
is the default configuration.

1. Open a terminal and ensure the ROCm environment is available (for example,
   ``rocminfo`` and ``hipcc`` are in your ``PATH``).

2. Change to the directory containing the built CG client. For a Release build:

   .. code-block:: shell

      cd build/release/clients/staging

   For a Debug build:

   .. code-block:: shell

      cd build/debug/clients/staging

3. Download a test matrix in Matrix Market format:

   .. code-block:: shell

      wget https://math.nist.gov/pub/MatrixMarket2/Harwell-Boeing/laplace/gr_30_30.mtx.gz

4. Extract the matrix file:

   .. code-block:: shell

      gzip -d gr_30_30.mtx.gz

5. Run the CG solver client:

   .. code-block:: shell

      ./cg gr_30_30.mtx

If your installation succeeded, the solver prints iteration and residual
information and converges without errors:

.. code-block:: text

   IterationControl initial residual = 33.2866
   IterationControl RELATIVE criteria has been reached: res norm=2.03206e-05; rel val=6.10474e-07; iter=36
   PCG ends
   ||e - x||_2 = 8.01194e-06

.. note::

   This test also runs on systems without an AMD GPU. rocALUTION prints
   ``HIP is not initialized``, falls back to the CPU backend, and the solver
   still converges. You can use this to verify a build before you have hardware
   available.
