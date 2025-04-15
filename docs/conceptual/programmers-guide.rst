.. meta::
   :description: C++ library for accelerating mixed precision matrix multiply-accumulate operations
    leveraging specialized GPU matrix cores on AMD's latest discrete GPUs
   :keywords: rocWMMA, ROCm, library, API, tool

.. _programmers-guide:

===================
Programming guide
===================

This document provides insight into the library design choices, source code organization and helpful information for new development.

--------------------------------
Infrastructure
--------------------------------

- Doxygen and Sphinx are used to generate the project's documentation.
- Jenkins is used to automate Continuous Integration (CI) testing (``.jenkins`` folder has configurations).
- rocWMMA is hosted and maintained by AMD on `Github  <https://github.com/ROCm/rocWMMA>`_.
- The rocWMMA project is organized and configured via ``CMake`` and the collection of ``CMakeLists.txt`` in the base of each directory.
- ``clang-format`` is used to format C++ code. ``.githooks/install`` ensures that a clang-format pass will run on each committed file.
- ``GTest`` is used to implement test suite organization and execution.
- ``CTest`` is used to consolidate and invoke multiple test targets. In the ``<rocWMMA_install_dir>/bin/rocWMMA/CTestTestfile.cmake`` file, testing targets are listed that will be run when ``ctest`` is invoked.
- The preferred compiler for rocWMMA is ``CC=<path_to_rocm>/bin/amdclang and CXX=<path_to_rocm>/bin/amdclang++``. ``hipcc`` is also supported, however may be deprecated in future ROCm releases.

--------------------------------
hipRTC Support
--------------------------------

The HIP runtime compilation (hipRTC) environment allows on-the-fly runtime compilation, loading, and execution of device code on AMD GPUs. The rocWMMA library is compatible with hipRTC, so it can be leveraged for runtime-generated kernels.
A simple GEMM sample is included to demonstrate compatibility.

For more information, refer to the `HIP API Reference  <https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/index.html>`_

--------------------------------
General Design Concepts
--------------------------------

rocWMMA is developed with the ``C++17`` language standard. The library takes advantage of several meta-programming techniques that help to statically
optimize code at compile time and generate more efficient GPU kernels. All of the code is contained within the ``rocwmma`` namespace. Due to the templated
nature of the code, the rocWMMA API is a header-only library. The developer has the advantage of full visibility of the implementation, as well as the ability to
integrate rocWMMA API calls directly within their own kernels. An important feature is that the integrated code has higher visibility
to the compiler's optimization passes where there is higher potential to generate more efficient device code. Moreover, no expensive host-device transfers or
external kernel invokations are imposed by the API.

The programming model that is the most useful with the rocWMMA API is wavefront-centric. In a more general
sense, loading and storing data or mma calls are assumed to involve the entire wavefront (or, warp). In the collaborative API, we can assume that other wavefronts in the
same threadblock may collaborate moving data from one location to another.

The rocWMMA API can eliminate a very large amount of boiler-plate code as it provides tools to decompose matrix multiply-accumulate based problems into
blocks, or fragments of data that may be efficiently processed by individual wavefronts. Wavefronts will abstract blocks of data into ``fragments``, which are designed to encapsulate meta-data properties about the blocks in different contexts, along with the data itself:

- a general geometric shape (e.g. BlockM/N/K dimensions)
- a context of the provenance of the data (e.g. ``matrix_a`` or ``matrix_b``)
- a context of how the data was stored (e.g. row or column major)
- a datatype (e.g. single or half-precision floating point)

These basic traits are then used to decide things like how much storage is needed, and how rocWMMA will organize individual threads in a layout to fetch or store the data.
``Fragments`` are powerful objects because they are versatile in configuring and representing data, and their meta-properties propagate easily via Argument
Dependent Lookup (ADL) and other deduction techniques. In practice, the user must simply configure the fragments correctly for their problem and rocWMMA conveniently takes care of the rest.

The implementation code is generally encapsulated into different layers of objects, fulfilling specific interface communications (from low level functions to API level):

- Unit backend operations. These are often wrappers to device-specific functions such as intrinsics that usually prefixed with ``amdgcn_*``. These functions translate inputs into raw vector forms and addresses required by the low-level intrinsics. This backend-area of the code usually handles architecture or device-specific behavior differences.
- Vector operations. This level of objects such as ``OpaqueLoad`` or ``OpaqueStore`` handle variable-sized vector inputs and marshall them into unrolled unit backend operations. They encompass thread-wise details of vector operations. These classes provide a consistent functional interface for input vectors of all sizes, independent of device architecture, whose details are handled at a lower level.
- Fragment operations. This is the API level of rocWMMA where user data is stored and manipulated in ``fragment`` objects. Fragment objects can be visualized as geometric 'blocks' of data in the perspective of the current wavefront, which are to be stored as vectors. Each of the loading / storing and mma operations provided by rocWMMA are in the perspective of the wavefront, assuming that all threads in the wavefront are participating under the hood. This layer's implementation translates wavefront fragment operations into vector operations and so on. The encapsulation of rocWMMA into these separate layers allows for an API experience that is seamless across different device architectures and platforms.

--------------------------------
Nomenclature
--------------------------------

GEMM
^^^^^

GEneralized Matrix-Matrix multiplication (or, GEMM) is one of the fundamental applications for rocWMMA. In general, GEMM solves the equation ``D = alpha * A x B + beta * C`` , where ``A, B, C, D`` are matrices and ``alpha`` and ``beta`` are scalars.
Matrices are generally sized by ``M x N x K``, such that ``A = M x K``, ``B = K x N`` and ``C, D = M x N``.
rocWMMA implements many varieties of testing and sample kernels for this purpose and encompasses a wide variety of parameters. Testing kernels are grouped into executables that are named as a string of parameters that describe their implementations.

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

* ``Ad Hoc Test``: An executable that focuses on a specific set of kernel parameters. This is used as a
  quick mock-up of a situational investigation of a particular GEMM kernel.

Validation tests are postfixed with ``-validate``. Benchmark tests are postfixed with ``-bench``.

Sample kernels are constructed with as minimal infrastructure as possible. Their namings are much different to appeal to a broader audience.

* ``simple_sgemm``: a simple GEMM kernel with ``s`` denoting single-precision floating point datatype.
* ``simple_dgemm``: a simple GEMM kernel with ``d`` denoting double-precision floating point datatype.
* ``simple_hgemm``: a simple GEMM kernel with ``h`` denoting half-precision floating point datatype.
* ``perf_sgemm``: a performant GEMM kernel with ``s`` denoting single-precision floating point datatype.
* ``perf_dgemm``: a performant GEMM kernel with ``d`` denoting double-precision floating point datatype.
* ``perf_hgemm``: a performant GEMM kernel with ``h`` denoting half-precision floating point datatype.

GEMV
^^^^^

GEneralized Matrix-Vector multiplication (or, GEMV) is another application for rocWMMA. In general, GEMV solves the equation ``y = alpha * A * x + beta * y``, where ``A`` is a matrix, ``x and y`` are vectors and ``alpha and beta`` are scalars.
``Matrix A`` is generally sized as ``M x K``, vector ``X`` as ``K x 1`` and vector ``Y`` as ``M x 1``.
rocWMMA implements some samples of simple GEMV demonstrations as below:

* ``simple_sgemv``: Simple GEMV kernel with ``s`` denoting single-precision floating point datatype.
* ``simple_dgemv``: Simple GEMV kernel with ``d`` denoting double-precision floating point datatype.

DLRM
^^^^

rocWMMA implements a simple component of Deep Learning Recommendation Model (DLRM) for machine learning. Both forward and backwards passes on half-precision inputs and outputs are demonstrated.

* ``simple_dlrm``: Simple GEMV kernel with ``s`` denoting single-precision floating point datatype.

--------------------------------
Library source code organization
--------------------------------

The rocWMMA code is split into four major parts:

- The ``library`` directory contains the header library API and implementation.
- The ``samples`` directory contains real-world sample use-cases of the rocWMMA API.
- The ``test`` directory contains testing infrastructure for rocWMMA.
- The ``docs`` directory contains documentation generation sources.

``library`` directory
^^^^^^^^^^^^^^^^^^^^^^^

The ``library`` directory contains the following structure:

- ``library/include/rocwmma/``: C++ include files for the rocWMMA API. These files also contain Doxygen content that documents the API.

The API currently has three API contexts:

  - ``rocwmma.hpp``: The main API for rocWMMA, defining fragment data abstractions, wave-wise storing, loading, matrix multiply-accumulate (mma) and threadblock synchronization. This API's function signatures are portable from nvcuda::wmma.
  - ``rocwmma_coop.hpp``: A complimentary API for rocWMMA, defining functionality that allows GPU wavefronts to collaborate in the loading / storing of fragment data. These are unique to rocWMMA.
  - ``rocwmma_transforms.hpp``: A complimentary API for rocWMMA, defining functionality to manipulate fragment data (e.g. transpose and data layout changes). These are unique to rocWMMA.

- ``library/include/internal``: Internal include files define the main infrastructure driving the rocWMMA API:

  - Configuration of platforms and architectures
  - Type support
  - Input and output configuration, shapes and traits
  - Loading and storing utilities
  - Layouts of memory and registers
  - Mapping utilities
  - Intrinsic wrappers
  - Vector class implementations
  - Vector conversion, permutation and transform utilities
  - Vector packing and unpacking
  - Matrix multiply-accumulate
  - Cooperative loading and storing
  - Threadblock synchronization and flow control
  - Utility code
  - Data layout transformation utilities

``samples`` directory
^^^^^^^^^^^^^^^^^^^^^^^

The ``samples`` directory contains the sample codes for the following use cases:

- ``samples/hipRTC_gemm.cpp``: For calling simple General Matrix Multiply (GEMM) algorithm demonstration without LDS memory usage and no transpose, from within the hipRTC environment.
- ``samples/simple_sgemv.cpp``: For calling simple matrix multiply-accumulate with a vector demonstration, without LDS and no transpose for single-precision floating point types.
- ``samples/simple_dgemv.cpp``: For calling simple matrix multiply-accumulate with a vector demonstration, without LDS and no transpose for double-precision floating point types.
- ``samples/simple_sgemm.cpp``: For calling simple GEMM algorithm demonstration without LDS memory usage and no transpose for single-precision floating point types.
- ``samples/simple_dgemm.cpp``: For calling simple GEMM algorithm demonstration without LDS memory usage and no transpose for double-precision floating point types.
- ``samples/simple_hgemm.cpp``: For calling simple GEMM algorithm demonstration without LDS memory usage and no transpose for half-precision floating point types.
- ``samples/perf_sgemm.cpp``: For calling the high performing multi-block GEMM algorithm demonstration with LDS memory, macro tile collaboration, data reuse and optimized pipeline for single-precision floating point types.
- ``samples/perf_dgemm.cpp``: For calling the high performing multi-block GEMM algorithm demonstration with LDS memory, macro tile collaboration, data reuse and optimized pipeline for double-precision floating point types.
- ``samples/perf_hgemm.cpp``: For calling the high performant multi-block GEMM algorithm demonstration with LDS memory, macro tile collaboration, data reuse and optimized pipeline for half-precision floating point types.
- ``samples/simple_dlrm.cpp``: For calling simple Deep Learning Recommendation Model (DLRM) for machine learning.
- ``samples/common.hpp``: Common code used by all the above rocWMMA samples files.

``test`` directory
^^^^^^^^^^^^^^^^^^^^^^^

The ``test`` directory contains the test code support:

- ``test/bin``: To generate benchmark plots from the ``gtest`` output dumps of rocWMMA's benchmark tests.
- ``test/device``: Device utility kernels to support test setup and validation on GPU.
- ``test/dlrm``: For various strategies of DLRM application. This test is used to validate DLRM functions using rocWMMA API.
- ``test/gemm``: For various strategies of GEMM application. This test is used to validate and benchmark GEMM functions using rocWMMA API.
- ``test/unit``: For testing the basic functional units of rocWMMA library.

``docs`` directory
^^^^^^^^^^^^^^^^^^^

- Sphinx and Doxygen are used to generate project documentation.
- ``api-reference-guide.rst`` pulls from Doxygen documentation to format API documentation.
- ``installation.rst`` builds installation / build instructions for rocWMMA.
- ``license.rst`` includes information pertaining to rocWMMA licensing.
- ``programmers-guide.rst`` includes information about project organization and expectations.
- ``what-is-rocwmma.rst`` includes a description of rocWMMA.

Contributing
^^^^^^^^^^^^

For those wishing to contribute to the project, please see `Contributing to rocWMMA  <https://github.com/ROCm/rocWMMA/blob/develop/CONTRIBUTING.md>`_.
