.. meta::
   :description: Install rocALUTION
   :keywords: rocALUTION, ROCm, library, API, install, linux, build, HIP SDK

.. _install-rocalution:

********************************
Install rocALUTION
********************************

.. _linux-installation:

You can install rocALUTION as part of the AMD ROCm software stack on Linux,
or build it from source. Install ROCm before you build or use rocALUTION.

When you install rocALUTION through ROCm, you get a single-node, accelerator-enabled library.
If you need a different configuration, such as multi-node or distributed execution, build rocALUTION from source.

.. _windows-installation:

Prerequisites
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You need a ROCm enabled platform to use rocALUTION. For more information, including a list of supported GPUs and Linux distributions, see the `ROCm on Linux install guide <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/index.html>`__.

You need the following prerequisites to build rocALUTION from source:

- `git <https://git-scm.com/>`__
- `CMake <https://cmake.org/>`__ (version 3.5 or later)
- `rocBLAS <https://rocm.docs.amd.com/projects/rocBLAS/en/latest/index.html>`__
- `rocSPARSE <https://rocm.docs.amd.com/projects/rocSPARSE/en/latest/index.html>`__
- `rocRAND <https://rocm.docs.amd.com/projects/rocRAND/en/latest/index.html>`__
- `rocPRIM <https://rocm.docs.amd.com/projects/rocPRIM/en/latest/index.html>`__
- `Python <https://www.python.org/>`__
- `PyYAML <https://pypi.org/project/PyYAML/>`__

Install these components before you build rocALUTION. Refer to their respective documentation for install instructions.

For multi-node configurations, you must also install:

- `OpenMP <https://www.openmp.org/>`__
- `MPI <https://www.mcs.anl.gov/research/projects/mpi/>`__

Install pre-built packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the following commands to install rocALUTION on Ubuntu or Debian:

.. code-block:: shell

   sudo apt-get update
   sudo apt-get install rocalution

Use the following commands to install rocALUTION on RHEL-based platforms:

.. code-block:: shell

   sudo yum update
   sudo yum install rocalution

Use the following commands to install rocALUTION on SLES:

.. code-block:: shell

   sudo zypper refresh
   sudo zypper install rocalution

.. note::

   On SLES, use ``zypper`` instead of ``dnf``. The ROCm repositories must be
   configured first, and installing the full ``rocm`` package is usually
   recommended because rocALUTION depends on other ROCm components.

After you install rocALUTION, you can use it like any other library with a C++ API.
Include the header file in your code to call rocALUTION.
The rocALUTION shared library becomes a link-time and run-time dependency for your application.

Build rocALUTION from source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You don't need to build rocALUTION from source if you install the prebuilt packages described above.
To build rocALUTION from source, follow the instructions in this section.

You need the `AMD ROCm Platform <https://github.com/ROCm/ROCm>`__ to compile and run rocALUTION.
When you build rocALUTION from source, select supported versions of the math library dependencies (rocBLAS, rocSPARSE, rocRAND, and rocPRIM).
Given a version of rocALUTION, you must use versions of these dependencies that are the same or later.

Download rocALUTION
^^^^^^^^^^^^^^^^^^^^^

You can find the rocALUTION source code in the `https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocalution <https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocalution>`__ repository.

To limit your local checkout to only the rocALUTION project, configure ``sparse-checkout`` before you clone.
The partial clone feature (``--filter=blob:none``) reduces how much data you download.
Use the following commands for a sparse checkout:

.. note::

   To include the rocBLAS, rocSPARSE, rocRAND, and rocPRIM dependencies, set the projects for the sparse checkout using ``git sparse-checkout set projects/rocalution projects/rocblas projects/rocsparse projects/rocrand projects/rocprim``.

.. code-block:: shell

   git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-libraries.git
   cd rocm-libraries
   git sparse-checkout init --cone
   git sparse-checkout set projects/rocalution
   git checkout develop
   cd projects/rocalution

.. note::

   For ROCm preview releases (7.11, 7.12, 7.13, 7.14), use the ``develop`` branch.
   For production ROCm 7.2.x, use ``release/rocm-rel-7.2``.
   To build ROCm 6.4.3 and earlier, use the standalone `rocALUTION repository <https://github.com/ROCm/rocALUTION>`__.
   For more information, see the documentation associated with the release you want to build.

To download the ``develop`` branch for all projects in rocm-libraries, use these commands.
This process takes longer, but use it if you work with a large number of libraries.

.. code-block:: shell

   git clone -b develop https://github.com/ROCm/rocm-libraries.git
   cd rocm-libraries/projects/rocalution

Build rocALUTION using the install script
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use the ``install.sh`` script to install rocALUTION.
The following tables describe how to build different packages of the library, including the dependencies and clients.

.. note::

   Run the ``install.sh`` script from the ``projects/rocalution`` directory.

Use install.sh to build rocALUTION with dependencies
"""""""""""""""""""""""""""""""""""""""""""""""""""""

The following table lists the common ways to use ``install.sh`` to build the rocALUTION dependencies and library.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Command
     - Description
   * - ``./install.sh -h``
     - Print the help information.
   * - ``./install.sh -d``
     - Build the dependencies and library in your local directory. Use the ``-d`` flag only once. For subsequent invocations of ``install.sh``, you don't need to rebuild the dependencies.
   * - ``./install.sh``
     - Build the library in your local directory. The script assumes the dependencies are available.
   * - ``./install.sh -i``
     - Build the library, then build and install the rocALUTION package in ``/opt/rocm``. The script prompts you for sudo access. This installs rocALUTION for all users.
   * - ``./install.sh --mpi=<dir> -i``
     - Build the library with MPI support enabled, then build and install the rocALUTION package in ``/opt/rocm``.

Use install.sh to build rocALUTION with dependencies and clients
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

The clients contain example code, unit tests, and benchmarks.
The following table lists common ways to use ``install.sh`` to build the library, dependencies, and clients.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Command
     - Description
   * - ``./install.sh -dc``
     - Build the dependencies, library, and client in your local directory. Use the ``-d`` flag only once. For subsequent invocations of ``install.sh``, you don't need to rebuild the dependencies.
   * - ``./install.sh -c``
     - Build the library and client in your local directory. The script assumes the dependencies are available.
   * - ``./install.sh -idc``
     - Build the library, dependencies, and client, then build and install the rocALUTION package in ``/opt/rocm``. The script prompts you for sudo access. This installs rocALUTION for all users.
   * - ``./install.sh -ic``
     - Build the library and client, then build and install the rocALUTION package in ``/opt/rocm``. The script prompts you for sudo access. This installs rocALUTION for all users.
   * - ``./install.sh -o``
     - Build the client executables using an already installed version of the library.

Build rocALUTION using individual make commands
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The rocALUTION library contains both host and device code, so specify the HIP compiler during the CMake configuration process.

You can build rocALUTION using the following commands:

.. note::

   Run these commands from the ``projects/rocalution`` directory.

.. note::

   You need CMake 3.5 or later to build rocALUTION.

.. code-block:: shell

   # Create and change to build directory
   mkdir -p build/release
   cd build/release

   # Default install path is /opt/rocm, use -DCMAKE_INSTALL_PREFIX= to adjust it
   CXX=/opt/rocm/bin/amdclang++ cmake ../..

   # Compile rocALUTION library
   make -j$(nproc)

   # Install rocALUTION to /opt/rocm
   make install

You can also configure the following optional CMake directives:

- ``SUPPORT_HIP``: Enable HIP support. This option is ``ON`` by default.
- ``SUPPORT_OMP``: Enable OpenMP support. This option is ``ON`` by default.
- ``SUPPORT_MPI``: Enable MPI support for multi-node execution. This option is ``OFF`` by default.
- ``BUILD_SHARED_LIBS``: Build rocALUTION as a shared library. This option is ``ON`` by default and is recommended.
- ``BUILD_EXAMPLES``: Build the example programs. This option is ``ON`` by default.

For example, to build rocALUTION with MPI support enabled:

.. code-block:: shell

   CXX=/opt/rocm/bin/amdclang++ cmake ../.. -DSUPPORT_MPI=ON -DROCM_PATH=/opt/rocm/

Test your rocALUTION installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Verify that you built and installed rocALUTION correctly by running the Conjugate Gradient (CG) solver client on a sample Laplacian matrix.

These steps assume you built rocALUTION with client applications enabled (the default configuration).

1. Open a terminal and ensure the ROCm environment is available
   (for example, ``rocminfo`` and ``hipcc`` are in your ``PATH``).

2. Change to the directory containing the built CG client.
   For a Release build:

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

If your installation succeeded, the solver prints iteration and residual information and converges without errors.
