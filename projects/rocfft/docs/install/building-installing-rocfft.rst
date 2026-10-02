.. meta::
  :description: Building and installing rocFFT
  :keywords: rocFFT, ROCm, API, documentation, install, build from source

.. _building-installing-rocfft:

************************************
Build and install rocFFT from source
************************************

To build rocFFT as part of the ROCm Core SDK, see `TheRock build
instructions
<https://github.com/ROCm/TheRock/blob/main/docs/development/README.md>`__.
TheRock is the recommended way to build ROCm components from source.

Alternatively, you can build rocFFT standalone using the following
instructions.

Building rocFFT from source
=============================

You can use the GitHub releases tab to download the source code. This might provide you with a more recent version
than the prebuilt packages.

rocFFT uses the AMD clang++ compiler and CMake. You can specify several options to customize your build.
Use the following commands to build a shared library for the supported AMD GPUs.
Run these commands from the ``rocm-libraries/projects/rocfft`` directory:

.. code-block:: shell

   mkdir build && cd build
   cmake -DCMAKE_CXX_COMPILER=amdclang++ -DCMAKE_C_COMPILER=amdclang ..
   make -j

.. note::

   To compile a static library, use the ``-DBUILD_SHARED_LIBS=off`` option.

In ROCm version 4.3 or higher, indirect function calls are enabled for rocFFT by default.
Use ``-DROCFFT_CALLBACKS_ENABLED=off`` with CMake to prevent these calls on older ROCm versions.

.. note::

   If rocFFT is built with this configuration, callbacks won't work correctly.

rocFFT clients
=============================

rocFFT includes the following clients and utilities:

*  **rocfft-bench**: Runs general transforms and performance analysis

*  **rocfft-test**: Runs a series of regression tests

*  Various samples

The following table includes the CMake option to build each client and the client dependencies.

.. csv-table::
   :header: "Client","CMake option","Dependencies"
   :widths: 20, 30, 30

   "rocfft-bench","``-DBUILD_CLIENTS_BENCH=on``","hipRAND"
   "rocfft-test","``-DBUILD_CLIENTS_TESTS=on``","hipRAND, FFTW, GoogleTest"
   "samples","``-DBUILD_CLIENTS_SAMPLES=on``","none"

The clients are not built by default. To build them, use ``-DBUILD_CLIENTS=on``.
The build process downloads and builds GoogleTest and FFTW if they are not already installed.
rocFFT uses version 1.11 of GoogleTest.

MPI and RCCL
=============================

MPI distributed transforms require ``-DROCFFT_MPI_ENABLE=ON`` and a GPU-aware MPI.
On Cray systems also pass ``-DROCFFT_CRAY_MPI_ENABLE=ON`` and link the GTL library
(see the Cray MPI example below). Leave ``ROCFFT_RCCL_ENABLE`` off for an MPI-only
build; that is the default.

The optional RCCL transpose backend is a rocFFT 1.0.39 / ROCm 10.0 feature. Enable it
with ``-DROCFFT_RCCL_ENABLE=ON`` only when CMake can ``find_package(rccl)`` from the
same ROCm prefix used to compile rocFFT (typically ``$ROCM_PATH/lib/cmake``).
RCCL itself ships in older ROCm releases, but this rocFFT tree (1.0.40) is not
intended to be built against ROCm 6.2.

Example: MPI-only on a Cray node (no RCCL)::

   cmake -DCMAKE_CXX_COMPILER=hipcc -DCMAKE_C_COMPILER=hipcc \
     -DCMAKE_BUILD_TYPE=Release \
     -DBUILD_CLIENTS=ON -DBUILD_FFTW=ON -DBUILD_GTEST=ON \
     -DROCFFT_MPI_ENABLE=ON -DROCFFT_CRAY_MPI_ENABLE=ON \
     -DROCFFT_RCCL_ENABLE=OFF \
     -DGPU_TARGETS=gfx90a \
     -DCMAKE_INSTALL_PREFIX=rocFFT_install \
     -DCMAKE_PREFIX_PATH="${ROCM_PATH}/lib/cmake;${ROCM_PATH}/lib/cmake/hip;${MPICH_DIR}" \
     -DCMAKE_SHARED_LINKER_FLAGS="-L${MPICH_DIR}/lib -lmpi ${PE_MPICH_GTL_DIR_amd_gfx90a} ${PE_MPICH_GTL_LIBS_amd_gfx90a} ${CRAY_XPMEM_POST_LINK_OPTS} -lxpmem" \
     ..

Example: same build with RCCL (requires a ROCm that provides ``roc::rccl`` and
matches this rocFFT, ROCm 10.x preferred)::

   cmake ... -DROCFFT_RCCL_ENABLE=ON ...

``AMDGPU_TARGETS`` still works but is deprecated; use ``GPU_TARGETS``.

You can build the clients separately from the main library.
For example, to build all the clients with an existing rocFFT library, invoke CMake from
within the ``rocm-libraries/projects/rocfft/rocFFT-src/clients`` folder using these commands:

.. code-block:: shell

   mkdir build && cd build
   cmake -DCMAKE_CXX_COMPILER=amdclang++ -DCMAKE_PREFIX_PATH=/path/to/rocFFT-libb ..
   make -j

To install the client dependencies on Ubuntu, run the following command:

.. code-block:: shell

   sudo apt install libgtest-dev libfftw3-dev

.. note::

   On Red Hat-related distributions, these packages are named ``gtest-devel`` and ``fftw-devel``.
