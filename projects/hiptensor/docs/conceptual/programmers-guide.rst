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
    *   -   ``HIPTENSOR_ROCKE_F32_RECIPE``
        -   ``native`` (default), ``x3``, ``x4``, ``x6``, ``x9`` or ``f64``: how the rocKE backend multiplies
            contractions with ``HIPTENSOR_COMPUTE_DESC_32F`` or ``HIPTENSOR_COMPUTE_DESC_C32F`` whose operands are f32
            or complex f32 (or mixed 16-bit and f32), see :ref:`rocke-f32-recipes`. Other values select ``native``,
            and the plan log says so.
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
            form, or no input read with vector loads: every input gathered or broadcast along the output's
            fastest mode); ``0`` turns the JIT off.
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
ring GETT, the universal GEMM bodies of the data type's route, the f64 strided GEMM classes and the direct GETT;
for a reduction, the row and column kernels with several splits of the reduced modes; for a permutation or an
elementwise operation, the tiled kernel and the generic kernel), checking every capability on the host without
launching anything. It then orders them:

1. A contraction of rocKE's tuned route table takes the launches of that table.
2. Otherwise, a contraction the ring GETT serves takes it (with the operands whose fastest free run is not a
   multiple of 8 elements first packed into the workspace when the limit allows, else read in place), and on a
   target without the GEMM routes (gfx90a, gfx942) the direct GETT of a 16-bit or f32 compute class takes it.
3. Otherwise, the selection model of the kernel pack for the problem's class (gradient-boosted trees over
   integer features of the problem and the candidate, trained on measured timings of every candidate) ranks
   the candidates of the routes it was trained on; the direct GETT follows them. Selection takes no timing, no
   allocation and no device call, and is the same on every host for the same problem, device and pack.
4. Without a model for the class, rocKE's rules order the candidates (a permutation or elementwise operation:
   the tiled kernel, then the vector form of the generic kernel, then the generic kernel).

The plan preference changes this:

* ``HIPTENSOR_PLAN_PREFERENCE_KERNEL_RANK`` = *k* selects the *k*-th candidate of that order (0 is the
  default). A rank past the candidates the plan admits returns ``HIPTENSOR_STATUS_NOT_SUPPORTED``.
* ``HIPTENSOR_ALGO_DEFAULT_PATIENT`` times the first candidates of the order on the device at plan creation
  (on buffers it allocates and initializes, which it frees before ``hiptensorCreatePlan`` returns) and keeps
  the fastest one.

With ``HIPTENSOR_CACHE_MODE_PEDANTIC`` (the default) the kernel a plan takes (the winner of the timing, or the
selected candidate) is stored in the handle's plan cache, and later plans of the same problem on that handle, or on
a handle that read the plan cache file, take it from there without timing or selecting again, as long as their
workspace limit admits it.

``hiptensorEstimateWorkspaceSize`` reports, for ``HIPTENSOR_WORKSPACE_DEFAULT``, the workspace of the candidate
a plan selects (of the candidates ``HIPTENSOR_ALGO_DEFAULT_PATIENT`` times, of every candidate for a
``KERNEL_RANK`` preference, and at least that of the plan cache's kernel); for ``HIPTENSOR_WORKSPACE_MIN``, the
least workspace some candidate runs with; for ``HIPTENSOR_WORKSPACE_MAX``, the workspace every candidate runs with.
The plan's ``HIPTENSOR_PLAN_REQUIRED_WORKSPACE`` is the workspace of the selected candidate.

The rocKE backend serves contractions and trinary contractions, permutations, elementwise binary and trinary
operations, and reductions on the GPU targets it has a kernel pack for. Its kernels take their sizes and strides
at run time. Every unary operator is served on real tensors (see "Unary operators" below); on complex tensors
identity and conjugate. Contractions of real with complex tensors and other operators of complex tensors are
declined; the fallback backend serves what it supports of them. A contraction runs on one of these routes:

* The ring GETT (gfx950): rocKE's tensor-contraction body reads A and B and writes D (and reads C) through their
  own strides in one launch, with no workspace. It serves f16 or bf16 data (A, B, C and D of one type) with f32
  accumulation and no operators, when each operand's fastest free mode runs over an even number of elements and
  the K modes split at its K tile of 32 elements (one K mode may be padded inside the tile), without batch modes,
  with C laid out as D.
* The GEMM routes (gfx950): A and B are packed into the workspace, a GEMM computes an f32 result in the
  workspace, and an elementwise epilogue writes D.
* The native f32 route (gfx942 and gfx950): contractions of f32 or complex f32 data with
  ``HIPTENSOR_COMPUTE_DESC_32F`` or ``HIPTENSOR_COMPUTE_DESC_C32F`` (the default recipe, see
  :ref:`rocke-f32-recipes`): A and B are packed into the workspace and an f32 strided GEMM with f32 products and
  f32 accumulation writes D (or an f32 result that an epilogue writes to D).
* The f64 route (every target): A and B are converted to f64 into the workspace and an f64 GEMM writes D (or
  an f64 result that an epilogue writes to D).
* The direct GETT (every target): one launch reads A, B and C and writes D through their own strides, with no
  workspace, for every data type, compute type, operator and layout.

A plan created with a workspace limit of zero takes the ring GETT or the direct GETT. The ring GETT and rocKE's
tuned plans run the direct GETT instead when alpha is zero, or beta is zero for a contraction with C (their
epilogue would combine the term of the zero scale, so that a NaN or an infinity of a tensor that is not read would
reach D), and when a buffer is not aligned to their vector accesses.

A permutation or an elementwise operation whose output is fastest along one mode and an input along another
runs on a tiled kernel: the inputs that are fastest along the other mode are read with 16-byte loads into a tile
in LDS and read back along the output's mode, and D is written with 16-byte stores. It applies when the output's
extents and strides keep those vectors aligned (otherwise, and for buffers that are not aligned, the generic
kernel runs). Otherwise, when an input is contiguous along the output's fastest mode, a vector form of the generic
kernel handles 16 bytes of the output per thread, with vector loads of those inputs (the classes of the JIT's
vector kernels, with the sizes and strides as kernel arguments). When the output's two tiled modes span less
than half a tile, one workgroup of the tiled kernel computes the planes of several consecutive batch indices.

The launches that pack a contraction's operands into the workspace and the epilogue that writes D from the GEMM
result (real data) use the tiled kernel when they transpose a tensor, and also when the generic kernel would need
several launches for them (more modes than its index slots): those run as one launch of the tiled kernel, with
the output's two fastest modes as the tile. Each such launch keeps the generic launches it replaces, which
execution runs instead when a buffer, or the workspace, is not aligned to the tiled kernel's 16-byte vectors.
Kernels compiled by the JIT for a plan exist only on the device the plan was created on: executed on another
device, the plan runs the kernel pack's launches they replace. The other kernels of the pack are loaded on the
executing device. The workspace passed to ``hiptensorContract`` should be aligned to 256 bytes (as memory from
``hipMalloc`` is): the rocKE backend places the packed operands and the GEMM result at multiples of 256 bytes
from its start, and its GEMM routes read them with vector accesses.

The data types of a contraction's A, B, C and D may differ: each is f16, bf16, f32 or f64, or each is complex f32
or complex f64. The compute type sets the precision of the products. With ``HIPTENSOR_COMPUTE_DESC_16F`` and
``HIPTENSOR_COMPUTE_DESC_16BF`` every operand is rounded once to that type (to nearest even, from any type
including f64; values past its range become infinities). With ``HIPTENSOR_COMPUTE_DESC_32F`` and
``HIPTENSOR_COMPUTE_DESC_C32F`` the operands keep their values (every f16 and bf16 value is an f32 value) and
operands of f64 data run on the f64 route. With ``HIPTENSOR_COMPUTE_DESC_64F`` and ``HIPTENSOR_COMPUTE_DESC_C64F``
every contraction runs on the f64 route or, in the direct GETT, in f64. Products accumulate in f32 (f64 on the
f64 route, in the direct GETT's f64 kernels, which also take f32 data on gfx90a, and for f64 operands with a
16-bit compute type), C is read with its own type, and the result is rounded once to D's type. The error of an element of D against the exact result of the
rounded operands is then at most ``2 (K + 4) u_c |alpha| sum |a b| + 2 u_d |d| + 4 u_c |beta c|``, with K the
number of products, ``u_c`` the unit roundoff of the accumulation (2^-24 for f32, 2^-53 for f64) and ``u_d`` that
of D's type.

Unary operators apply to an operand in the type the kernel computes in (f32 for f16, bf16 and f32 data, f64 when
an operand or the output is f64, or with ``HIPTENSOR_COMPUTE_DESC_64F``), before the operand is scaled and combined,
and their result has that type (for the operands of a contraction with a 16-bit compute type, it's then rounded
once to that type). The trigonometric and hyperbolic operators, their inverses, ceil and floor are evaluated in f64;
for f32 the result is rounded once to f32. The table gives the largest error of each operator in units in the last
place (ulp) of the compute type, against the exact result of the operand. The f32 bounds hold for every one of the
2^32 inputs; the f64-evaluated operators are correctly rounded in f32 except for at most six inputs per operator,
which lie within 2^-29 ulp of the midpoint between two f32 values. The f64 column is measured, not proven: it is
the largest error over at least 100,000 inputs per operator over its domain, including its boundaries and, for
sine, cosine and tangent, the arguments closest to multiples of pi/2.

.. list-table::
   :header-rows: 1

   *   -   Operator
       -   f32
       -   f64 (measured)
   *   -   identity, conjugate (real data), negation, abs, ReLU, ceil, floor
       -   exact
       -   exact
   *   -   square root, reciprocal
       -   0.5
       -   0.5
   *   -   exp
       -   1
       -   1
   *   -   log
       -   2.25
       -   1
   *   -   tanh
       -   2
       -   1
   *   -   sigmoid
       -   3
       -   2
   *   -   sin, cos, tan, sinh, cosh, asin, acos, atan, asinh, acosh, atanh
       -   0.5 (see above)
       -   0.65

Special values follow C: NaN gives NaN, an argument outside the domain gives NaN (sqrt and log of a negative
number, asin and acos past 1 in magnitude, acosh below 1, atanh past 1 in magnitude), log of zero is -inf and
atanh of +-1 is +-inf, sin, cos and tan of an infinity are NaN, and odd operators keep the sign of zero. ReLU is
``x > 0 ? x : 0``, so ReLU of NaN and of -0 is +0. Sigmoid is evaluated as ``1 / (1 + e^-x)`` for x >= 0 and
``e^x / (1 + e^x)`` below, so large negative arguments keep their small results.

.. _rocke-f32-recipes:

Contractions of f32 data on the rocKE backend
"""""""""""""""""""""""""""""""""""""""""""""

With ``HIPTENSOR_COMPUTE_DESC_32F`` or ``HIPTENSOR_COMPUTE_DESC_C32F``, the rocKE backend multiplies f32 and complex
f32 operands (and mixed f16, bf16 and f32 operands) by the recipe ``HIPTENSOR_ROCKE_F32_RECIPE`` names. No recipe
admits a kernel less accurate than its own: every recipe also lists the f64 route, which is at least as accurate,
and a target without the recipe's kernels serves the contraction on the f64 route. The direct GETT is a candidate
of every recipe too (f32 products and accumulation; under ``f64`` f64 products and accumulation), and the ring GETT
of f16 and bf16 data (exact products, f32 accumulation) of every recipe but ``f64``. f16 or bf16 operands of the
same type without an operator keep their 16-bit GEMM route (exact products) under every recipe but ``f64``.

.. list-table::

    *   -   **Recipe**
        -   **Kernels**
        -   **Products**
        -   **Accuracy class**
    *   -   ``native`` (default)
        -   f32 (complex f32) strided GEMM on f32 matrix instructions, gfx942 and gfx950
        -   every product enters unrounded into a chain of correctly rounded f32 fmas in K order; complex
            products as four real products (4M)
        -   f32
    *   -   ``x9``
        -   universal GEMM, every operand split into three bf16 parts, gfx950
        -   all nine part products (each exact), summed in f32: a product carries a few f32 roundings
        -   f32
    *   -   ``x6``
        -   as ``x9``, without the three smallest part products
        -   the dropped part products are at most about 2^-23 of a product, summed in f32
        -   f32
    *   -   ``x4``
        -   universal GEMM, every operand split into two bf16 parts, gfx950
        -   operands kept to about 16 significant bits: relative error of a product at most about 2^-15
        -   between bf16 and f32 (about 16 bits)
    *   -   ``x3``
        -   as ``x4``, without the product of the two low parts
        -   about 2^-15, as ``x4``
        -   between bf16 and f32 (about 16 bits)
    *   -   ``f64``
        -   f64 (complex f64) strided GEMM, every target with a kernel pack
        -   operands converted to f64 exactly, f64 accumulation, the result rounded once to D's type
        -   f64

For the f32-class recipes, the error of an element of D against the exact result is at most
``2 (n K + 4) u |alpha| sum |a b| + e |alpha| sum |a b| + 2 u_d |d| + 4 u |beta c|``, with ``u`` = 2^-24, ``n`` the part
products per product (1 for ``native``, 9, 6, 4 or 3) and ``e`` the relative error of a product's representation (0
for ``native`` and ``x9``, 2^-21 for ``x6``, 2^-14 for ``x4`` and ``x3``); for ``f64`` the bound above with
``u_c`` = 2^-53 applies. ``x6`` ranks its candidates with the selection models of the kernel pack; the other recipes
rank them by rocKE's rules (the models are trained on ``x6``). A plan-cache entry written under one recipe names a
kernel the other recipes do not list, so plans under another recipe select their kernel as if the entry were not
there.

A trinary contraction runs as the library's two contractions: T = opA(A) opB(B), stored with E's data type at the
start of the workspace (its size is part of the plan's required workspace), then
E = alpha T opC(C) + beta opD(D). The first step writes all of T without reading it, so earlier contents of the
workspace never reach E. Each step's candidates are ranked by that step's selection, and the candidates of the
trinary contraction pair them: every candidate of a step with the best one of the other step, and the first three
of each step with each other, ordered by the sum of their ranks. ``KERNEL_RANK`` and
``HIPTENSOR_ALGO_DEFAULT_PATIENT`` apply to these pairs.

Tensors of 2 GiB and more are served. The elementwise and reduction kernels have 64-bit-offset variants, the
contraction routes of gfx950 have a GEMM body with 64-bit tile addresses (one launch per batch and per 65535 rows
of tiles), and the f64 GEMM runs as several launches over ranges of its rows, columns and batches. When one row
of A or B is past 2 GiB, the f64 GEMM splits K into ranges that run as the batches of one launch, each writing an
f64 partial result, and a reduction adds the partial results in f64 before the epilogue. What one launch can't
address (a mode of more than 2^31 elements, more than eight modes after merging contiguous modes, more than 2^31
rows of a kernel) runs as launches over pieces of the tensors, and the partial results of a reduction's pieces are
combined by its final kernel. Each of these steps has at most 4096 launches; larger problems are declined.

JIT and kernel cache
^^^^^^^^^^^^^^^^^^^^

A build with ``-DHIPTENSOR_ENABLE_ROCKE_JIT=ON`` (the default when the rocKE backend is built) links the COMGR
library and rocKE's C++ kernel builders. Plans created on the rocKE backend with ``HIPTENSOR_JIT_MODE_DEFAULT`` then
replace the elementwise launches of the plan (permutations, elementwise binary and trinary operations, and the
operand packing and epilogue launches of contractions) by kernels compiled for the problem at plan creation: every
size, stride and operator a constant, and, when the output and an input are contiguous in the output's fastest
mode, 16-byte vector loads and stores along that mode. A launch without such an input (every input gathered or
broadcast along the output's fastest mode, for example a permutation that moves the fastest mode) keeps its
kernel-pack kernel (``HIPTENSOR_ROCKE_JIT=all`` specializes it too). The choice is made when the plan is created
and is kept by the plans the plan cache restores. The library builds the kernels with rocKE's C++ kernel builders
(the same builders that wrote the kernel pack) and compiles them with COMGR, without Python and without any data
file besides the kernel pack. The operand packing and epilogue launches of mixed-type contractions keep their
kernel-pack kernels, and so do launches with the operators ``HIPTENSOR_OP_SIN`` to ``HIPTENSOR_OP_FLOOR`` unless
the output and every input are contiguous in the output's fastest mode and addressed with 32-bit offsets. A
specialized launch keeps its kernel-pack launch, which execution takes when a buffer is not aligned to the vector
width or the plan runs on another device than the one it was created on.

Plans created with ``HIPTENSOR_JIT_MODE_NONE`` (the default) never compile and never use JIT kernels; every
problem the rocKE backend serves runs without JIT. Plan creation with ``HIPTENSOR_JIT_MODE_DEFAULT`` can take as
long as the compiles of its new kernels and can't be captured in a HIP graph; execution can.

Compiled kernels go to a process-wide kernel cache. An entry is identified by the SHA-256 of its LLVM IR, the
target, the compile options and the compiler version, and is found again by its kernel class, the build of
rocKE's kernel builders and the values, so a second plan of the same problem, from any handle, compiles nothing.
``hiptensorWriteKernelCacheToFile`` writes the entries of the handle's device class (architecture, target features,
compute unit count) to a file, replacing it; ``hiptensorReadKernelCacheFromFile`` adds the entries of a file to the
cache. Reading returns
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
