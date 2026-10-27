.. meta::
   :description: A high-performance HIP library for tensor primitives
   :keywords: hipTensor, cuTensor, ROCm, library, API, tool

.. _programmers-guide:

===================
Programming guide
===================

This document provides insight into the library source code organization, design implementation, development guidance, and testing and benchmarking details.

--------------------------------
Infrastructure
--------------------------------

- Doxygen and Sphinx are used to generate the project's documentation.
- Jenkins is used to automate Continuous Integration (CI) testing, with configurations stored in the ``.jenkins`` folder.
- hipTensor is hosted and maintained by AMD on `GitHub  <https://github.com/ROCm/rocm-libraries/tree/develop/projects/hiptensor>`_.

  .. note::

    The hipTensor repository for ROCm 7.1.1 and earlier is located at `<https://github.com/ROCm/hipTensor>`_.

- The hipTensor project is organized and configured using ``CMake``, with ``CMakeLists.txt`` files in the root of each directory.
- ``clang-format`` is used to format C++ code. ``.githooks/install`` ensures that a clang-format pass will run on each committed file.
- ``GTest`` is used to implement test suite organization and execution.
- ``CTest`` is used to consolidate and invoke multiple test targets. The ``<hipTensor_install_dir>/CTestTestfile.cmake`` file lists the testing targets executed when ``ctest`` is invoked.
- The preferred compiler for hipTensor is ``CC=<path_to_rocm>/bin/amdclang and CXX=<path_to_rocm>/bin/amdclang++``. ``hipcc`` is also supported, however may be deprecated in future ROCm releases.

--------------------------------
Design concepts
--------------------------------

hipTensor is a library developed with the ``C++20`` language standard. It uses meta-programming techniques to optimize code at compile time and generate efficient GPU kernels.
hipTensor employs Composable Kernel as a functional backend and is written in different layers.

The outer API layer serves as a functional interface for users to define tensor data abstractions and manipulations.
The second layer, the hipTensor solution interface, connects the API objects with the intended functionality.
The solution layer handles translating input problem parameters into solution candidates, selecting candidates, managing resources, and logging.
Solution candidates provide interface abstractions into functional backends, such as Composable Kernel objects, which can be invoked with results returned through the API.
The Composable Kernel library is used as a header library, where all kernel instances are customized by hipTensor and statically bundled by the hipTensor functional backend layer. This allows future backends to remain isolated in their own modules, as they are now.
The hipTensor solution layer is divided into functional components, such as contraction, permutation, and reduction.
Each component contains a registry of backend instances as potential solution candidates for given input parameters.
These instances are selected based on API-provided hints and populated with the appropriate arguments for invocation by the API.

Backends
^^^^^^^^

The solution layer reaches kernels only through a backend interface (``library/src/backend``). When a plan is
created, hipTensor describes the operation to the backends as a problem with 64-bit extents and strides, selects
the backend that serves it, and lets that backend prepare the plan. Executing the plan launches the prepared
kernels of that backend on the caller's stream. Plan-cache entries and plan-cache files are keyed by backend.

The Composable Kernel backend (``library/src/backends/ck``) is always built and serves every plan by default.
The rocKE backend (``library/src/backends/rocke``) is built with ``-DHIPTENSOR_ENABLE_ROCKE_BACKEND=ON``.
These environment variables, read at every plan creation and workspace estimate, select the backend:

.. list-table::

    *   -   **Variable**
        -   **Description**
    *   -   ``HIPTENSOR_BACKEND``
        -   ``auto`` (default), ``ck``, or ``rocke``: the backend that serves plans. ``auto`` selects the Composable
            Kernel backend. Other values make ``hiptensorCreatePlan`` and ``hiptensorEstimateWorkspaceSize`` return
            ``HIPTENSOR_STATUS_INVALID_VALUE``.
    *   -   ``HIPTENSOR_BACKEND_FALLBACK``
        -   ``1`` (default) or ``0``. When the selected backend declines a problem, or isn't built, the Composable
            Kernel backend serves it. With ``0``, ``hiptensorCreatePlan`` returns ``HIPTENSOR_STATUS_NOT_SUPPORTED``
            instead.
    *   -   ``HIPTENSOR_BACKEND_AB``
        -   ``<primary>,<shadow>``, for example ``rocke,ck``, for debugging. Plans prepare both backends. Every
            execution runs the shadow backend into a buffer of its own, then the primary backend into the caller's
            output, and logs the difference between the two results at the ``HIPTENSOR_LOG_LEVEL_HEURISTICS_TRACE``
            level. It overrides ``HIPTENSOR_BACKEND``, allocates device memory, and synchronizes the stream.
    *   -   ``HIPTENSOR_ROCKE_PACK_PATH``
        -   ``:``-separated directories with rocKE kernel packs (``<dir>/<target>/rocke.htpk``). When set, they are the
            only places the rocKE backend looks; otherwise it looks next to the library
            (``../share/hiptensor/rocke``), in the install prefix and in the build tree. Without a pack for the device,
            the rocKE backend declines every problem.
    *   -   ``HIPTENSOR_ROCKE_CONTRACTION_ROUTE``
        -   ``f64``, for debugging and testing: the rocKE backend runs every contraction in f64 (complex f64), with the
            operands converted to f64 and the result rounded once to the output type.
    *   -   ``HIPTENSOR_ROCKE_DISABLE_TUNED``
        -   Any value, for debugging and testing: the rocKE backend does not use the launches it has for the
            contractions of its tuned route table, and plans them like any other contraction.
    *   -   ``HIPTENSOR_ROCKE_SELECT``
        -   ``static``, for debugging and testing: the rocKE backend selects kernels by rocKE's rules instead of
            the selection model of its kernel pack.
    *   -   ``HIPTENSOR_ROCKE_AUTOTUNE_TOPK``
        -   The number of candidates ``HIPTENSOR_ALGO_DEFAULT_PATIENT`` times on the rocKE backend (default 4), or
            ``all``.
    *   -   ``HIPTENSOR_ROCKE_AUTOTUNE_REPS``
        -   The timed repetitions per candidate of ``HIPTENSOR_ALGO_DEFAULT_PATIENT`` on the rocKE backend
            (default 5).
    *   -   ``HIPTENSOR_ROCKE_SELECT_DUMP``
        -   A file to which ``HIPTENSOR_ALGO_DEFAULT_PATIENT`` on the rocKE backend appends, for every timed
            candidate, one line of JSON with its features, the selector's view of it and its median time. It
            is the training data of the selection models. ``HIPTENSOR_ROCKE_SELECT_DUMP_TAG`` adds a tag to the
            lines.
    *   -   ``HIPTENSOR_ROCKE_JIT``
        -   ``all`` or ``0``, for plans created with ``HIPTENSOR_JIT_MODE_DEFAULT`` on the rocKE backend: ``all``
            also specializes the elementwise launches the default rule leaves on the kernel pack (no vector
            form, or no input read with vector loads); ``0`` turns the JIT off.
    *   -   ``HIPTENSOR_JIT_MAX_MS``
        -   The compile time, in milliseconds, one ``hiptensorCreatePlan`` may spend on JIT compiles. Past it, the
            remaining launches of the plan keep their kernel-pack kernels (kernels already in the kernel cache are
            still used). Unset or ``0``: no limit.
    *   -   ``HIPTENSOR_KERNEL_CACHE_DIR``
        -   A directory (of the user, not writable by others unless sticky) where plans created with
            ``HIPTENSOR_JIT_MODE_DEFAULT`` look for a kernel before they compile it and store every kernel they
            compile, one kernel cache file per kernel. Processes that share the directory compile a kernel once.
            Unset (default): no directory.
    *   -   ``HIPTENSOR_KERNEL_CACHE_ACCEPT_STALE``
        -   ``1``: kernel cache entries read from a file that another compiler version compiled are used. By
            default they are kept and written back, but not used.
    *   -   ``HIPTENSOR_ROCKE_JIT_NO_COMPILE``
        -   Any value but ``0``, for testing: the JIT never compiles; only kernel cache entries are used.

The selected backend and kernel of every plan are logged at the ``HIPTENSOR_LOG_LEVEL_HEURISTICS_TRACE`` level.

The rocKE backend selects kernels when the plan is created. It lists the candidates of the problem, the
programs that its kernel pack can run the problem with within the plan's workspace limit (for a contraction, the
universal GEMM bodies of the data type's route and the f64 strided GEMM classes; for a reduction, the row and
column kernels with several splits of the reduced modes), checking every capability on the host without
launching anything. It then orders them:

1. A contraction of rocKE's tuned route table takes the launches of that table.
2. Otherwise, the selection model of the kernel pack for the problem's class (gradient-boosted trees over
   integer features of the problem and the candidate, trained on measured timings of every candidate) ranks
   the candidates. Selection takes no timing, no allocation and no device call, and is the same on every host
   for the same problem, device and pack.
3. Without a model for the class, rocKE's rules order the candidates.

The plan preference changes this:

* ``HIPTENSOR_PLAN_PREFERENCE_KERNEL_RANK`` = *k* selects the *k*-th candidate of that order (0 is the
  default). A rank past the candidates the plan admits returns ``HIPTENSOR_STATUS_NOT_SUPPORTED``.
* ``HIPTENSOR_ALGO_DEFAULT_PATIENT`` times the first candidates of the order on the device at plan creation
  (on buffers it allocates and initializes, which it frees before ``hiptensorCreatePlan`` returns) and keeps
  the fastest one. With ``HIPTENSOR_CACHE_MODE_PEDANTIC`` (the default) the winner is stored in the handle's plan
  cache, and later plans of the same problem on that handle, or on a handle that read the plan cache file,
  take it from there without timing, as long as their workspace limit admits it.

``hiptensorEstimateWorkspaceSize`` reports, for ``HIPTENSOR_WORKSPACE_DEFAULT``, the workspace of the candidate
a plan selects (of the candidates ``HIPTENSOR_ALGO_DEFAULT_PATIENT`` times, of every candidate for a
``KERNEL_RANK`` preference, and at least that of the plan cache's kernel); for ``HIPTENSOR_WORKSPACE_MIN``, the
least workspace some candidate runs with; for ``HIPTENSOR_WORKSPACE_MAX``, the workspace every candidate runs with.
The plan's ``HIPTENSOR_PLAN_REQUIRED_WORKSPACE`` is the workspace of the selected candidate.

The rocKE backend serves contractions (f16, bf16, and f32 with an f32, f16 or bf16 compute type; f64; complex f32;
complex f64), permutations, elementwise binary and trinary operations, and reductions on the GPU targets it has a
kernel pack for (gfx90a and gfx942: every contraction on the f64 route; gfx950: the routes of each data type, and
the f64 route when those decline). Its kernels take their sizes and strides at run time. Tensors with more than eight modes
after merging contiguous modes, unary operators other than identity, square root, ReLU, conjugate, reciprocal, sigmoid,
tanh, exp, log, abs and negation, trinary contractions, and problems past the kernels' 32-bit offsets are declined, and
the fallback backend serves them. Contractions need workspace for packed copies of A and B and an f32 (f64) result.

JIT and kernel cache
^^^^^^^^^^^^^^^^^^^^

A build with ``-DHIPTENSOR_ENABLE_ROCKE_JIT=ON`` (the default when the rocKE backend is built) links the COMGR
library and rocKE's recipe engine. Plans created on the rocKE backend with ``HIPTENSOR_JIT_MODE_DEFAULT`` then
replace the elementwise launches of the plan (permutations, elementwise binary and trinary operations, and the
operand packing and epilogue launches of contractions) by kernels compiled for the problem at plan creation: every
size, stride and operator a constant, and, when the output and an input are contiguous in the output's fastest
mode, 16-byte vector loads and stores along that mode. A launch without such an input (for example a
permutation that moves the fastest mode) keeps its kernel-pack kernel (``HIPTENSOR_ROCKE_JIT=all``
specializes it too). The kernels come from recipes that the build
records from rocKE's kernel builder (``share/hiptensor/rocke/rocke.recipes``); the library replays them in C++ and
compiles the result with COMGR, without Python. A specialized launch keeps its kernel-pack launch, which execution
takes when a buffer is not aligned to the vector width or the plan runs on another device than the one it was
created on.

Plans created with ``HIPTENSOR_JIT_MODE_NONE`` (the default) never compile and never use JIT kernels; every
problem the rocKE backend serves runs without JIT. Plan creation with ``HIPTENSOR_JIT_MODE_DEFAULT`` can take as
long as the compiles of its new kernels and can't be captured in a HIP graph; execution can.

Compiled kernels go to a process-wide kernel cache. An entry is identified by the SHA-256 of its LLVM IR, the
target, the compile options and the compiler version, and is found again by its recipe and values, so a second
plan of the same problem, from any handle, compiles nothing. ``hiptensorWriteKernelCacheToFile`` writes the entries
of the handle's device class (architecture, target features, compute unit count) to a file, replacing it;
``hiptensorReadKernelCacheFromFile`` adds the entries of a file to the cache. Reading returns
``HIPTENSOR_STATUS_INVALID_VALUE`` for a file of a newer format, another device class or another kernel ABI,
``HIPTENSOR_STATUS_IO_ERROR`` for a file that is missing, damaged in its header, not a kernel cache file, not owned
by the user (or root), or writable by others, and ``HIPTENSOR_STATUS_SUCCESS`` for an empty file. Damaged entries
are skipped. Entries are used only by plans created with ``HIPTENSOR_JIT_MODE_DEFAULT``. A build without the JIT
keeps no kernel cache: both calls then succeed without touching the file.

hipTensor tests and samples are consumers of the hipTensor library and demonstrate the usages of the API in different contexts, such as tensor contractions, permutations and reductions.

--------------------------------
Nomenclature
--------------------------------

Tensor contraction
^^^^^^^^^^^^^^^^^^^

In general, a tensor contraction is a multiply-accumulate problem over elements between two or more multi-dimensional tensors.
hipTensor uses Einstein notation, where repeated indices are summed and each index appears at most twice in each mathematical term.
In the process of accumulating over summation dimensions, they are effectively collapsed or contracted.

hipTensor supports the following contraction forms:

* **Binary bilinear contraction**: :math:`D = \alpha \, A \, B + \beta \, C`,
  where :math:`A`, :math:`B`, :math:`C` are input tensors and :math:`D` is the
  output tensor. Created with ``hiptensorCreateContraction`` and executed with
  ``hiptensorContract``.

* **Binary scale contraction**: :math:`D = \alpha \, A \, B`. The scale form is
  selected by passing a null tensor descriptor (and a null pointer at execution
  time) for :math:`C`; the bilinear API is reused with the bias term omitted.

* **Trinary bilinear contraction**: :math:`E = \alpha \, A \, B \, C + \beta \, D`,
  where :math:`A`, :math:`B`, :math:`C` are input tensors, :math:`D` is the
  bias input, and :math:`E` is the output. Created with
  ``hiptensorCreateContractionTrinary`` and executed with
  ``hiptensorContractTrinary``.

* **Trinary scale contraction**: :math:`E = \alpha \, A \, B \, C`. The scale
  form is selected by passing a null tensor descriptor (and a null pointer at
  execution time) for :math:`D`; the trinary API is reused with the bias term
  omitted.

All four contraction forms support per-input-tensor unary element-wise
operators (for example, ``HIPTENSOR_OP_RELU``, ``HIPTENSOR_OP_EXP``,
``HIPTENSOR_OP_COS``) applied before the contraction, in addition to the
default ``HIPTENSOR_OP_IDENTITY``.

Tensor permutation
^^^^^^^^^^^^^^^^^^^

Tensor permutation reorders stride indices, changing the data dimensional locality relationships.

Tensor reduction
^^^^^^^^^^^^^^^^^^^

Tensor reduction transforms a higher-dimensional tensor into a lower-dimensional one by performing some operations on the original tensor.

Tensor rank
^^^^^^^^^^^

Tensor rank refers to the data dimensionality, such as the number of modes. In Einstein notation, repeated modes indicate the dimensions being contracted during tensor contractions.

In contractions, modes can be categorized as M, N, and K, defined as follows:

* Tensor A modes [M0, ..., Mn, K0, ..., Kn]
* Tensor B modes [N0, ..., Nn, K0, ..., Kn]
* Tensor C/D modes [M0, ..., Mn, N0, ..., Nn]

Repeated indices K0, ..., Kn are indices that are contracted. Contractions currently support up to M6N6K6, allowing up to 6 dimensions for each M, N and K.
A tensor contraction with A = [M0, ..., M5, K0, ..., K5], B = [N0, ..., N5, K0, ..., K5], and C/D [M0, ..., M5, N0, ..., N5] is considered rank 12.

Tensor mode
^^^^^^^^^^^

Tensor modes let users specify the order or labels of the input strides that define the dimensional data relationship.
They describe the data's memory layout and spatial relationships.

--------------------------------
Library source code organization
--------------------------------

The hipTensor code is split into four major parts:

- The ``library`` directory contains the library source code.
- The ``samples`` directory contains real-world use-cases of the hipTensor API.
- The ``test`` directory contains validation tests for hipTensor API.
- Infrastructure.

``library`` directory
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``library`` directory contains the following include and source files:

- ``library/include/hiptensor/``: C++ include files for the hipTensor API. These files also contain Doxygen comments that document the API.
- ``library/include/hiptensor/internal``: Include files for utility code and tensor utility generation.
- ``library/src/``: Source files for logging, device management, and performance functions.
- ``library/src/backend/``: The backend interface: the problem description, backend selection, and plan execution.
- ``library/src/backends/ck/``: The Composable Kernel backend.
- ``library/src/backends/rocke/``: The rocKE backend.
- ``library/src/contraction/``: Source files for core initialization and management of contraction module.
- ``library/src/contraction/device``: Source files for composable kernel backend bilinear and scale instances.
- ``library/src/elementwise/``: Source files for core initialization and management of permutation module.
- ``library/src/elementwise/device``: Source files for composable kernel backend permute instances.
- ``library/src/reduction/``: Source files for core initialization and management of reduction module.
- ``library/src/reduction/device``: Source files for composable kernel backend reduction instances.
- ``library/src/include``: Infrastructure support for backend and logging management.

``samples`` directory
^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``samples`` directory contains the sample codes for the following demonstrations:

- ``01_contraction/simple_bilinear_contraction``: Abstract base code for bilinear contractions.
- ``01_contraction/simple_scale_contraction``: Abstract base code for scale contractions.
- ``01_contraction/simple_bilinear_contraction_*``: Specialized bilinear contraction demonstrations per data type.
- ``01_contraction/simple_scale_contraction_*``: Specialized bilinear contraction demonstrations per data type.
- ``01_contraction/simple_trinary_bilinear_contraction_*``: Specialized trinary bilinear contraction demonstrations per data type.
- ``01_contraction/simple_trinary_scale_contraction_*``: Specialized trinary scale contraction demonstrations per data type.
- ``01_contraction/simple_contraction_plan_cache``: Simple contraction with plan cache demonstration.
- ``01_contraction/simple_contraction_c``: Simple contraction demonstration in C.
- ``02_elementwise/elementwise_permute``: Simple permutation demonstration.
- ``02_elementwise/elementwise_binary``: Simple element-wise binary operation demonstration.
- ``02_elementwise/elementwise_trinary``: Simple element-wise trinary operation demonstration.
- ``02_elementwise/elementwise_binary_c``: Simple element-wise binary operation demonstration in C.
- ``03_reduction/reduction``: Simple reduction demonstration.
- ``03_reduction/reduction_c``: Simple reduction demonstration in C.

``test`` directory
^^^^^^^^^^^^^^^^^^^^^^^

The ``test`` directory contains the test codes for testing the following functionalities:

- ``00_unit/logger_test``: Tests logger API functions of hipTensor.
- ``00_unit/yaml_test``: Tests the YAML serialization and de-serialization for testing parameters.
- ``01_contraction/bilinear_contraction_test_*``: Testing harness for the bilinear contractions.
- ``01_contraction/scale_contraction_test_*``: Testing harness for the scale contractions.
- ``01_contraction/complex_bilinear_contraction_test_*``: Testing harness for the bilinear contractions with complex data types.
- ``01_contraction/complex_scale_contraction_test_*``: Testing harness for the scale contractions with complex data types.
- ``01_contraction/trinary_contraction_test``: Testing harness for the trinary contractions.
- ``01_contraction/trinary_bilinear_contraction_test``: Testing harness for the trinary bilinear contractions.
- ``01_contraction/trinary_scale_contraction_test``: Testing harness for the trinary scale contractions.
- ``01_contraction/bilinear_contraction_with_unary_ops_test``: Testing harness for binary bilinear contractions with non-identity unary operators.
- ``01_contraction/scale_contraction_with_unary_ops_test``: Testing harness for binary scale contractions with non-identity unary operators.
- ``01_contraction/contraction_resource``: Shared resource infrastructure for testing contractions.
- ``01_contraction/trinary_contraction_resource``: Shared resource infrastructure for testing trinary contractions.
- ``01_contraction/configs``: YAML files with actual contraction testing parameters.
- ``02_elementwise/elementwise_*``: Testing infrastructure for element-wise operation tests.
- ``02_elementwise/rank?_elementwise_permute_*``: Testing harnesses for permutation of a particular rank.
- ``02_elementwise/rank?_elementwise_binary_op*``: Testing harnesses for elementwise binary operation of a particular rank.
- ``02_elementwise/rank?_elementwise_trinary_op*``: Testing harnesses for elementwise trinary operation of a particular rank.
- ``02_elementwise/configs``: YAML files with actual permutation testing parameters.
- ``03_conduction/conduction*``: Testing infrastructure for conduction tests.
- ``03_conduction/rank*``: Testing harnesses for conduction of a particular rank.
- ``03_conduction/configs``: YAML files with actual conduction testing parameters.

``performance`` directory
^^^^^^^^^^^^^^^^^^^^^^^^^

The ``scripts/performance`` directory contains the benchmarking scripts:

- ``BenchmarkContraction.sh``: Benchmarking script for contraction.
- ``BenchmarkPermutation.sh``: Benchmarking script for permutation.
- ``BenchmarkReduction.sh``: Benchmarking script for reduction.

``emulation test`` script
^^^^^^^^^^^^^^^^^^^^^^^^^

The emulation test script ``rtest.py`` is located in the project's root directory.

Contributing
^^^^^^^^^^^^

To contribute to the project, see :ref:`contributors-guide`.
