.. meta::
   :description: Install rocALUTION
   :keywords: rocALUTION, ROCm, library, API, install, windows, linux, HIP SDK, building, installing


********************************
Install rocALUTION
********************************

You can install rocALUTION as part of the AMD ROCm software stack or `HIP SDK <https://rocm.docs.amd.com/projects/install-on-windows/en/latest/>`__ for Windows, or you can build it directly from source.
The installation method you choose depends on your operating system and whether you need a custom configuration, such as multi-node execution.

Install on Linux
--------------------------------

On Linux systems, rocALUTION is typically installed as part of ROCm. 
You must install ROCm before building or using rocALUTION. 

When installed through ROCm, rocALUTION is provided as a single-node, accelerator-enabled library.
If you require a different configuration, such as multi-node or distributed execution, you can build rocALUTION from source.


Prerequisites
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Building rocALUTION from source on Linux requires the following prerequisites:

- `CMake <https://cmake.org/>`__
- `rocBLAS <https://rocm.docs.amd.com/projects/rocBLAS/en/latest/index.html>`__
- `rocSPARSE <https://rocm.docs.amd.com/projects/rocSPARSE/en/latest/index.html>`__
- `rocRAND <https://rocm.docs.amd.com/projects/rocRAND/en/latest/index.html>`__
- `rocPRIM <https://rocm.docs.amd.com/projects/rocPRIM/en/latest/index.html>`__

Ensure that these components are installed before building rocALUTION. Refer to their respective documentation for installation instructions.

For multi-node configurations, you must also install:

- `OpenMP <https://www.openmp.org/>`__
- `MPI <https://www.mcs.anl.gov/research/projects/mpi/>`__

Build from source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Obtain the source code from the `rocALUTION GitHub repository <https://github.com/ROCm/rocALUTION>`__.
   Use the branch that matches the version of ROCm installed on your system.

2. Create a ``build`` directory in the rocALUTION root directory and change into it:

   .. code:: shell

      mkdir build
      cd build

3. Use CMake to generate the build files. You must set the ``ROCM_PATH`` directive to point to the ROCm installation directory.
   The following optional directives can also be configured:

   - ``SUPPORT_HIP``: Enable HIP support. This option is ``ON`` by default.
   - ``SUPPORT_OMP``: Enable OpenMP support. This option is ``ON`` by default.
   - ``SUPPORT_MPI``: Enable MPI support for multi-node execution. This option is ``OFF`` by default.
   - ``BUILD_SHARED_LIBS``: Build rocALUTION as a shared library. This option is ``ON`` by default and is recommended.
   - ``BUILD_EXAMPLES``: Build the example programs. This option is ``ON`` by default.

   For example, to build rocALUTION with MPI support enabled:

   .. code:: shell

      cmake .. -DSUPPORT_MPI=ON -DROCM_PATH=/opt/rocm/

4. Build and install rocALUTION:

   .. code:: shell

      make
      make install

   The library is installed under the ROCm installation directory.

Test your rocALUTION installation on Linux
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To verify that rocALUTION was built and installed correctly on Linux, run the Conjugate Gradient (CG) solver client on a sample Laplacian matrix.

These steps assume that rocALUTION was built with client applications enabled (the default configuration).

1. Open a terminal and ensure the ROCm environment is available
   (for example, ``rocminfo`` and ``hipcc`` are in your ``PATH``).

2. Change to the directory containing the built CG client.
   For a Release build:

   .. code-block:: shell

      cd rocALUTION\build\release\clients\staging

   For a Debug build:

   .. code-block:: shell

      cd rocALUTION\build\debug\clients\staging

3. Download a test matrix in Matrix Market format:

   .. code-block:: shell

      wget https://math.nist.gov/pub/MatrixMarket2/Harwell-Boeing/laplace/gr_30_30.mtx.gz

4. Extract the matrix file:

   .. code-block:: shell

      gzip -d gr_30_30.mtx.gz

5. Run the CG solver client:

   .. code-block:: shell

      ./cg gr_30_30.mtx

If the installation is successful, the solver prints iteration and residual information and converges without errors.

Install on Windows
--------------------------------

On Microsoft Windows, rocALUTION is built and used with the HIP SDK for Windows.
You must install the HIP SDK for Windows before building or using rocALUTION.

When installed through the HIP SDK, rocALUTION is provided as a single-node, accelerator-enabled library.
If you require a different configuration, such as multi-node or distributed execution, you can build rocALUTION from source.


Prerequisites
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Building rocALUTION from source on Windows requires the following prerequisites:

- `CMake <https://cmake.org/>`_
- `rocBLAS <https://rocm.docs.amd.com/projects/rocBLAS/en/latest/index.html>`_
- `rocSPARSE <https://rocm.docs.amd.com/projects/rocSPARSE/en/latest/index.html>`_
- `rocRAND <https://rocm.docs.amd.com/projects/rocRAND/en/latest/index.html>`_
- `rocPRIM <https://rocm.docs.amd.com/projects/rocPRIM/en/latest/index.html>`_
- `Python 3 <https://www.python.org/downloads/>`_
- `Ninja <https://ninja-build.org/>`_
- `Strawberry Perl <https://strawberryperl.com/>`_


Build from source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Obtain the source code from the `rocALUTION GitHub repository <https://github.com/ROCm/rocALUTION>`__.
   Use the branch that matches the installed version of the HIP SDK for Windows.

2. Verify the installed HIP SDK version by running:

   .. code:: shell

      hipcc --version

   .. note::

      If ``hipcc`` is not found, add ``%HIP_PATH%\\bin`` to your ``PATH`` environment variable.

3. Use the ``rmake.py`` script to build rocALUTION without installing it:

   .. code:: shell

      python3 rmake.py

   The built library files are placed in ``build\\release\\include\\rocalution``.

4. To build and install the library, use the ``-i`` option:

   .. code:: shell

      python3 rmake.py -i

   The library files are installed under ``%HIP_PATH%\\include\\rocalution``.

5. To build the library and its clients and install the library files, use:

   .. code:: shell

      python3 rmake.py -ci

   You can also omit the ``i`` option to build the library and clients without installing the library.


Test your rocALUTION installation on Windows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To verify that rocALUTION was built correctly, run the Conjugate Gradient (CG) solver client on a sample Laplacian matrix.

1. Open a **Command Prompt** or **PowerShell** with your HIP environment initialized
   (ensure ``hipcc`` is available in your ``PATH``).

2. Change to the directory containing the built CG client.
   For a Release build:

   .. code-block:: shell

      cd rocALUTION\build\release\clients\staging

   For a Debug build:

   .. code-block:: shell

      cd rocALUTION\build\debug\clients\staging

3. Download a test matrix in Matrix Market format:

   .. code-block:: shell

      curl -LO https://math.nist.gov/pub/MatrixMarket2/Harwell-Boeing/laplace/gr_30_30.mtx.gz

4. Decompress the matrix file:

   .. code-block:: shell

      gunzip gr_30_30.mtx.gz

5. Run the CG solver:

   .. code-block:: shell

      .\cg.exe gr_30_30.mtx

If the installation is successful, the solver prints iteration information and converges to a solution without errors.
