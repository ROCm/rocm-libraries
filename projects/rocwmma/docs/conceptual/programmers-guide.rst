.. meta::
   :description: C++ library for accelerating mixed precision matrix multiply-accumulate operations
    leveraging specialized GPU matrix cores on AMD's latest discrete GPUs
   :keywords: rocWMMA, ROCm, library, API, tool

.. _programmers-guide:

===================
Programming guide
===================

This document outlines the library design choices, source code organization and helpful information for new development.

--------------------------------
Infrastructure
--------------------------------

- Doxygen and Sphinx are used to generate the project's documentation.
- Jenkins is used to automate Continuous Integration (CI) testing, with configurations stored in the ``.jenkins`` folder.
- rocWMMA is hosted and maintained by AMD on `GitHub  <https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocwmma>`_.

  .. note::

    The rocWMMA repository for ROCm 7.1.1 and earlier is located at `<https://github.com/ROCm/rocWMMA>`_.

- The rocWMMA project is organized and configured using ``CMake``, with ``CMakeLists.txt`` files in the root of each directory.
- ``clang-format`` is used to format C++ code. ``.githooks/install`` ensures that a clang-format pass will run on each committed file.
- ``GTest`` is used to implement test suite organization and execution.
- ``CTest`` is used to consolidate and invoke multiple test targets. The ``<rocWMMA_install_dir>/bin/rocWMMA/CTestTestfile.cmake`` file lists the testing targets executed when ``ctest`` is invoked.
- The preferred compiler for rocWMMA is ``CC=<path_to_rocm>/bin/amdclang and CXX=<path_to_rocm>/bin/amdclang++``. ``hipcc`` is also supported, however may be deprecated in future ROCm releases.

--------------------------------
hipRTC support
--------------------------------

The HIP runtime compilation (hipRTC) environment enables on-the-fly runtime compilation, loading, and execution of device code on AMD GPUs. The rocWMMA library is compatible with hipRTC, so it can be used for runtime-generated kernels.
A simple GEMM sample is included to demonstrate compatibility.

For more information, refer to the `HIP API Reference  <https://rocm.docs.amd.com/projects/HIP/en/latest/index.html>`_

--------------------------------
Design concepts
--------------------------------

rocWMMA is a header-only library written in C++17 and contained in the ``rocwmma`` namespace. It leverages template meta-programming to optimize code at compile time and generate efficient GPU kernels. 
rocWMMA offers full implementation visibility, allowing developers to integrate rocWMMA API calls directly into their own kernels. 
The integrated code is more visible to compiler optimizations for generating efficient device code. The API also avoids expensive host-device transfers or external kernel invokations.

The programming model best suited for the rocWMMA API is wavefront-centric. Data loading, storing, and MMA functions are assumed to involve the entire wavefront (or warp). Undefined behaviour is expected if not all threads in each wavefront are active while using rocWMMA. Small block sizes representing edge cases will be automatically padded and will not affect thread masking.

The data of collaborative fragments is distributed across participating waves in the same thread block.
Collaborative fragments optimize collective data movement between memory locations, such as data movement from global memory to LDS, to balance shared data responsibilities across wavefronts. However, collaborative fragments are not supported in MMA functions. Fragment instances express wavefront collaboration with fragment scheduler meta-data which is specified by the developer.

In general, larger fragment sizes are better able to maximize memory bandwidth when transferring data between memory spaces, as well as ordering MMA functions. Beginning with rocWMMA 2.0.0 for ROCm 7.0, the mma_sync API can handle partial or large tiles in a piece-wise manner automatically. This enhanced functionality is demonstrated in the performance samples, which feature simplified workflow with larger tile sizes and good performance.

The rocWMMA API reduces the boiler-plate code by providing tools to decompose matrix multiply-accumulate based problems into blocks, or fragments of data that individual wavefronts can efficiently process. Wavefronts abstract blocks of data into ``fragments``, which encapsulate meta-data properties about the blocks in different contexts and the data itself, including:

- a general geometric shape, for example, the BlockM/N/K dimensions.
- a context of the provenance of the data, for example, ``matrix_a`` or ``matrix_b``.
- a context of how the data is stored, for example, row-major or column-major.
- a data type, for example, single or half-precision floating point.
- a fragment scheduler, for example, a class that specifies thread block collaboration, the number of participating waves, and their execution order.

These basic traits determine storage requirements and how rocWMMA organizes individual threads in a layout to fetch or store data.
``Fragments`` are powerful objects because they are versatile in configuring and representing data. Their meta-properties propagate easily via Argument
Dependent Lookup (ADL) and other deduction techniques. In practice, users configure fragments for their use case, and rocWMMA handles the underlying details.

The implementation code is encapsulated into layered objects, fulfilling specific interface communications from low level functions to high-level API interactions:

- **Unit backend operations:** Act as wrappers around device-specific functions, such as intrinsics typically prefixed with ``amdgcn_*``. These functions translate inputs into raw vector forms and addresses required by the low-level intrinsics, and handle architecture or device-specific behavior.
- **Vector operations:** This level of objects such as ``OpaqueLoad`` or ``OpaqueStore`` handle variable-sized vector inputs and marshall them into unrolled unit backend operations. They encompass thread-wise details of vector operations. These classes provide a consistent functional interface for input vectors of all sizes, independent of device architecture, whose details are handled at a lower level.
- **Fragment operations:** At the API level, user data is stored and managed in ``fragment`` objects. Fragment objects can be visualized as geometric blocks of data from the perspective of a wavefront and stored as vectors. rocWMMA's load, store, and MMA operations operate at the wavefront level, assuming that all threads in the wavefront are participating under the hood. This layer's implementation translates wavefront fragment operations into vector operations. rocWMMA's layered design enables seamless API experience across diverse device architectures and platforms.

--------------------------------
Nomenclature
--------------------------------

GEMM
^^^^^

Generalized Matrix-Matrix multiplication (GEMM) is a fundamental application for rocWMMA, solving the equation ``D = alpha * A x B + beta * C`` , where ``A``, ``B``, ``C``, and ``D`` are matrices, and ``alpha`` and ``beta`` are scalars.
Matrices are sized by ``M x N x K``, where ``A = M x K``, ``B = K x N`` and ``C, D = M x N``.
rocWMMA includes a range of test and sample kernels covering various parameters. Test kernels are grouped into executables named using parameter strings that describe their specific implementations.

.. code-block:: bash

    PGR# - Prefetch Global Read lookup stages. PGR0 = no global read prefetch. PGR1 = 1 stage global read prefetch.
    LB# - LDS buffer count. LB0 = no LDS usage, LB2 = 2 LDS buffers used for swap.
    MP# - MFMA instruction priority. MP0 = default MFMA instruction priority of 0. MP1 = raise MFMA instruction priority to 1.
    MB - Multiple output blocks targeted per wave
    SB - Single output block target per wave
    NC - Non-Cooperative load / store
    CP - Cooperative load / store
    BLK - Cooperative load / store per block tile
    WV - Cooperative load / store per wave tile
    WG - Cooperative load / store per macro tile

* ``gemm_PGR0_LB0_MP0_SB_NC``: The simplest blocked GEMM example, which targets one output
  block of matrix multiplication per wave. No prefetch, no LDS usage, default MFMA prioritization, single
  block output and non-collaborative.

* ``gemm_PGR0_LB0_MP0_MB_NC``: Implements a multi-block GEMM where each wave is responsible
  for a BlocksX x BlocksY grid of output blocks. No prefetch, no LDS usage, default MFMA prioritization,
  multiple blocks output, and non-collaborative.

* ``gemm_PGR1_LB2_MP0_MB_CP_BLK``: Implements a multi-block GEMM where each wave is
  responsible for a BlocksX x BlocksY grid of output blocks. This kernel leverages shared memory to
  implement a data prefetching pipeline and collaborates with other waves to improve performance.
  Implements single stage prefetch, double LDS buffer, default MFMA prioritization, multiple blocks
  output, and is block-tile collaborative in global read and local write.

* ``gemm_PGR1_LB2_MP0_MB_CP_WV``: Implements a multi-block GEMM where each wave is
  responsible for a BlocksX x BlocksY grid of output blocks. This kernel leverages shared memory to
  implement a data prefetching pipeline and collaborates with other waves to improve performance.
  Implements single stage prefetch, double LDS buffer, default MFMA prioritization, multiple blocks
  output, and is wave-tile collaborative in global read and local write.

* ``gemm_PGR1_LB2_MP0_MB_CP_WG``: Implements a multi-block GEMM where each wave is
  responsible for a BlocksX x BlocksY grid of output blocks. This kernel leverages shared memory to
  implement a data prefetching pipeline and collaborates with other waves to improve performance.
  Implements single stage prefetch, double LDS buffer, default MFMA prioritization, multiple blocks
  output and is macro-tile collaborative in global read and local write.

* ``Ad Hoc Test``: An executable targeting a specific set of kernel parameters. This is used as a
  quick mock-up for investigating a particular GEMM kernel scenario.

Validation tests are postfixed with ``-validate``. Benchmark tests are postfixed with ``-bench``.

Sample kernels are built with minimal infrastructure as possible and use more approachable names to appeal to a broader audience.

* ``simple_sgemm``: a simple GEMM kernel with ``s`` denoting single-precision floating-point data type.
* ``simple_dgemm``: a simple GEMM kernel with ``d`` denoting double-precision floating-point data type.
* ``simple_hgemm``: a simple GEMM kernel with ``h`` denoting half-precision floating-point data type.
* ``perf_sgemm``: a performant GEMM kernel with ``s`` denoting single-precision floating-point data type.
* ``perf_dgemm``: a performant GEMM kernel with ``d`` denoting double-precision floating-point data type.
* ``perf_hgemm``: a performant GEMM kernel with ``h`` denoting half-precision floating-point data type.

Community Samples
^^^^^^^^^^^^^^^^^^

rocWMMA provides a ``samples/community/`` directory for community-contributed samples that
demonstrate advanced techniques, specialized use cases, or experimental approaches extending
beyond the core competency demonstrations in the official samples directory.

Community samples have reduced review requirements compared to official samples and may not be
maintained with the same rigor. They are provided as-is and may not work on all GPU architectures
or with all data types. Users are encouraged to adapt these examples for their specific use cases.

The community samples directory is currently available for contributions. As samples accumulate,
they may be organized into subdirectories by technique or domain (e.g., fusion/, ml-models/,
optimizations/, advanced/, experimental/).

To build community samples, configure with ``-DROCWMMA_BUILD_COMMUNITY_SAMPLES=ON``.

GEMV
^^^^^

Generalized Matrix-Vector multiplication (GEMV) is another application for rocWMMA that solves the equation ``y = alpha * A * x + beta * y``, where ``A`` is a matrix, ``x and y`` are vectors and ``alpha and beta`` are scalars.
``Matrix A`` is sized as ``M x K``, vector ``X`` is ``K x 1``, and vector ``Y`` is ``M x 1``.

rocWMMA implements the following simple GEMV demonstration samples:

* ``simple_sgemv``: Simple GEMV kernel with ``s`` denoting single-precision floating-point data type.
* ``simple_dgemv``: Simple GEMV kernel with ``d`` denoting double-precision floating-point data type.

DLRM
^^^^

rocWMMA implements a simple component of Deep Learning Recommendation Model (DLRM) for machine learning. Both forward and backward passes using half-precision inputs and outputs are demonstrated.

* ``simple_dlrm``: Simple GEMV kernel with ``s`` denoting single-precision floating-point data type.

--------------------------------
Library source code organization
--------------------------------

The rocWMMA code consists of four major parts:

- The ``library`` directory contains the header library API and its implementation.
- The ``samples`` directory contains real-world sample use-cases using the rocWMMA API.
- The ``test`` directory contains testing infrastructure for rocWMMA.
- The ``docs`` directory contains sources for documentation generation.

``library`` directory
^^^^^^^^^^^^^^^^^^^^^^^

The ``library`` directory is structured as follows:

- ``library/include/rocwmma/``: C++ include files for the rocWMMA API. These files also contain Doxygen content that documents the API.

  - ``rocwmma.hpp``: The main API for rocWMMA, defining fragment data abstractions, wave-wise storing, loading, matrix multiply-accumulate (mma) and thread block synchronization. This API offers function signatures highly compatible with common CUDA WMMA interfaces.
  - ``rocwmma_transforms.hpp``: A complimentary API for rocWMMA, defining functionality to manipulate fragment data, for example, transpose and data layout changes. These are unique to rocWMMA.

- ``library/include/internal``: Internal include files which define the main infrastructure driving the rocWMMA API:

  - Configuration of platforms and architectures.
  - Type support.
  - Input and output configuration, shapes and traits.
  - Loading and storing utilities.
  - Layouts of memory and registers.
  - Mapping utilities.
  - Intrinsic wrappers and hardware abstraction.
  - Vector class implementations.
  - Vector conversion, permutation, and transform utilities.
  - Vector packing and unpacking.
  - Matrix multiply-accumulate.
  - Cooperative loading and storing.
  - Thread block synchronization and flow control.
  - Utility code.
  - Data layout transformation utilities.

``samples`` directory
^^^^^^^^^^^^^^^^^^^^^^^

The ``samples`` directory contains the sample codes for the following use cases:

- ``samples/hipRTC_gemm.cpp``: Simple General Matrix Multiply (GEMM) algorithm demonstration without LDS memory usage or transposition, running within the hipRTC environment.
- ``samples/simple_sgemv.cpp``: Simple matrix multiply-accumulate with a vector demonstration without LDS or transposition for single-precision floating-point types.
- ``samples/simple_dgemv.cpp``: Simple matrix multiply-accumulate with a vector demonstration without LDS or transposition for double-precision floating-point types.
- ``samples/simple_sgemm.cpp``: Simple GEMM algorithm demonstration without LDS memory usage or transposition for single-precision floating-point types.
- ``samples/simple_dgemm.cpp``: Simple GEMM algorithm demonstration without LDS memory usage or transposition for double-precision floating-point types.
- ``samples/simple_hgemm.cpp``: Simple GEMM algorithm demonstration without LDS memory usage or transposition for half-precision floating-point types.
- ``samples/perf_sgemm.cpp``: High performing multi-block GEMM algorithm demonstration with LDS memory, macro tile collaboration, data reuse and optimized pipeline for single-precision floating-point types.
- ``samples/perf_dgemm.cpp``: High performing multi-block GEMM algorithm demonstration with LDS memory, macro tile collaboration, data reuse and optimized pipeline for double-precision floating-point types.
- ``samples/perf_hgemm.cpp``: High performant multi-block GEMM algorithm demonstration with LDS memory, macro tile collaboration, data reuse and optimized pipeline for half-precision floating-point types.
- ``samples/simple_dlrm.cpp``: Simple Deep Learning Recommendation Model (DLRM) for machine learning.
- ``samples/common.hpp``: Common code used by all the above rocWMMA sample files.
- ``samples/community/``: Community-contributed samples demonstrating advanced techniques.
  See the Community Samples section above for details.

``test`` directory
^^^^^^^^^^^^^^^^^^^^^^^

The ``test`` directory contains the following test code support:

- ``test/bin``: To generate benchmark plots from the ``gtest`` output dumps of rocWMMA's benchmark tests.
- ``test/device``: Device utility kernels to support test setup and validation on GPU.
- ``test/dlrm``: For various strategies of DLRM application. This test is used to validate DLRM functions using rocWMMA API.
- ``test/gemm``: For various strategies of GEMM application. This test is used to validate and benchmark GEMM functions using rocWMMA API.
- ``test/unit``: For testing the basic functional units of rocWMMA library.

``docs`` directory
^^^^^^^^^^^^^^^^^^^

- Sphinx and Doxygen are used to generate the project's documentation.
- ``api-reference-guide.rst`` pulls from Doxygen documentation to format the API documentation.
- ``installation.rst`` builds installation and build instructions for rocWMMA.
- ``license.rst`` includes information about rocWMMA licensing.
- ``programmers-guide.rst`` includes information about project organization and expectations.
- ``what-is-rocwmma.rst`` includes a description of rocWMMA.

Contributing
^^^^^^^^^^^^

To contribute to the project, see `Contributing to rocWMMA  <https://github.com/ROCm/rocm-libraries/blob/develop/projects/rocwmma/CONTRIBUTING.md>`_.
