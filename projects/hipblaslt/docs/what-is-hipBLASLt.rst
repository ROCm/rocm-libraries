.. meta::
   :description: Introduction to hipBLASLt, covering the matmul programming model, layouts and orders, precision and scaling, epilogues, batching, kernel selection, tuning, and tooling
   :keywords: hipBLASLt, ROCm, library, API, GEMM, introduction, matmul, scaling, epilogue, tuning

.. _what-is-hipblaslt:

====================
What is hipBLASLt?
====================

hipBLASLt is a General Matrix-Matrix Multiplication (GEMM) library for AMD GPUs, written in the
:doc:`HIP programming language<hip:index>`. A GEMM multiplies two matrices, optionally scales the
result, and optionally adds a third matrix. It's the operation at the heart of dense linear algebra
and of nearly every neural network layer. That's why it gets a dedicated, optimized library.

hipBLASLt differs from a traditional BLAS ``gemm`` in two ways. Both recover performance that a
plain ``gemm`` leaves on the table.

First, hipBLASLt can **fuse** the small operations that normally follow a matrix multiply into the
multiply kernel itself. Adding a bias vector and applying an activation function are the common
cases. Run separately, each step reads the whole result back from memory and writes it out again.
Fused, they happen while the values are still in GPU registers.

Second, hipBLASLt separates **describing** a problem from **running** it. Instead of one function
with a long, fixed argument list, you create a few small descriptor objects and set attributes on
them. You then ask the library which kernel it recommends for that problem, and launch. Choosing a
kernel is work the library does on the CPU, so paying for it once and reusing the answer matters
when you call the same shape thousands of times.

That structure also keeps the API stable as hardware evolves. In a positional interface, every new
capability tends to need a new entry point: a new data type, a new fused operation, a new scaling
scheme. Here each one becomes another attribute on an existing descriptor.

Consider hipBLASLt when you want to fuse bias or activation work into the multiply, use 8-bit or
4-bit data types, run many small or differently shaped multiplications together, or control which
kernel runs. If you only need a conventional ``gemm`` at standard precision, use
:doc:`hipBLAS <hipblas:index>`.

The rest of this page introduces the concepts and vocabulary that appear throughout the
documentation. Each concept points to the topic that covers it in full.

The operation hipBLASLt computes
================================

:ref:`hipblasltmatmul` is the central call. The library calls the operation a *matmul*, short for
matrix multiplication; GEMM and matmul are used interchangeably here. It computes:

.. math::

   D = Activation(alpha \cdot op(A) \cdot op(B) + beta \cdot op(C) + bias)

Reading it piece by piece:

* :math:`A` and :math:`B` are the matrices being multiplied. :math:`C` is an optional matrix added
  to the product, and :math:`D` receives the result. :math:`C` and :math:`D` can be the same memory.
* :math:`op(A)` and :math:`op(B)` indicate that each input can be used either as stored or
  transposed. You state the choice on the descriptor instead of rearranging data beforehand.
* :math:`alpha` scales the product and :math:`beta` scales :math:`C`. Setting :math:`beta = 1`
  accumulates into an existing matrix. Setting :math:`beta = 0` ignores :math:`C` entirely.
* :math:`bias` is an optional vector with one element per row of :math:`D`, added to every column.
* :math:`Activation` is an optional element-wise function applied last.

The bias and the activation aren't a second kernel. They're the matmul's **epilogue**, the final
stage of the same kernel, and they're the fusion described earlier.
`Epilogues: fusing the work after the multiply`_ lists the available functions.

The formula doesn't show **scaling**. Each operand can carry its own scale factor, applied around
the multiply, so values held in a narrow type such as FP8 or FP4 still cover a useful numeric range.
FP32 rarely needs scaling. For the narrow types, scaling is what makes them usable at all.
`Precision: types, compute modes, and scaling`_ explains how to configure it.

The plan-then-run programming model
===================================

A matmul is assembled from four kinds of objects. Each answers a different question:

* A **handle** (``hipblasLtCreate()``) holds the library's per-process state and its association
  with a device. Create one and keep it.
* A **matrix layout**, one each for ``A``, ``B``, ``C``, and ``D``, describes *where the data is and
  how it's arranged*: its dimensions, data type, memory order, leading dimension, and batching.
* A **matmul descriptor** describes *what to compute*: the compute type, the transpose choices, the
  epilogue, the scaling modes and their pointers, and how ``alpha`` and ``beta`` are supplied.
* A **preference** describes *what you're willing to accept*, most often a cap on workspace size.

With those in place, the sequence is:

#. :ref:`hipblasltmatmulalgogetheuristic` returns candidate kernels. Each candidate names an
   algorithm and the workspace it requires. **Workspace** means scratch device memory that some
   kernels need for intermediate results; it's neither an input nor an output.
#. Allocate at least the reported workspace, then call :ref:`hipblasltmatmul` with the algorithm you
   chose.
#. Destroy the descriptor, the layouts, the preference, and the handle when you're done.

Step 1 runs on the CPU and can take tens of microseconds, comparable to a small GEMM on the GPU. A
**heuristic**, in this library, means its ranking of the available kernels for your problem shape
and device. hipBLASLt caches those rankings. The dependable pattern is to request a heuristic once
per problem shape and reuse the returned algorithm for every call with that shape. See
:doc:`Use logging and heuristics <./how-to/use-logging-heuristics>`.

By default, hipBLASLt reads ``alpha`` and ``beta`` from host memory.
``HIPBLASLT_MATMUL_DESC_POINTER_MODE`` lets you keep them in device memory instead, or makes
``alpha`` a device vector with one entry per row of ``D``. Use it when the scalars are themselves
produced on the GPU, so you don't copy them back to the host to pass them in.

The :doc:`hipBLASLt API <./reference/api-reference>` and
:doc:`hipBLASLt datatypes <./reference/datatypes>` references list every function and attribute.

Describing your data: layouts and orders
========================================

In BLAS, a matrix is described by its dimensions plus a *leading dimension*, the distance in memory
between consecutive columns. That lets a matrix be a window into a larger array. A hipBLASLt layout
keeps that idea and adds three more:

* A **memory order**. ``HIPBLASLT_ORDER_COL`` (column-major) is the default, and
  ``HIPBLASLT_ORDER_ROW`` is also supported. Some architectures additionally accept **swizzled**
  orders, such as ``HIPBLASLT_ORDER_COL16_4R8`` and ``HIPBLASLT_ORDER_COL16_4R16``, in which the
  data is pre-arranged into the exact pattern the kernel reads. That saves the kernel from
  rearranging values itself. However, a swizzled order is valid only for particular type, transpose,
  and architecture combinations, which the :doc:`API reference <./reference/api-reference>` lists.
  ``hipblaslt-bench`` exposes it as ``--swizzleA`` and ``--swizzleB``.
* An **offset** (``HIPBLASLT_MATRIX_LAYOUT_OFFSET``), measured in elements, so a layout can point at
  a sub-matrix inside a larger allocation. :doc:`Use general batched GEMM
  <./how-to/how-to-general-batched-gemm>` describes when offsets apply.
* **Batch** information, which `Running many GEMMs at once`_ covers.

Precision: types, compute modes, and scaling
============================================

hipBLASLt keeps the type the data is *stored* in separate from the type it's *computed* in. That
separation turns mixed precision into a matter of configuration rather than a different API.

**Storage types** are set on each layout. They include ``int8``, FP8 and BF8, FP4 as an input, FP16,
BF16, and FP32. Support is hardware-dependent, so the library's list is broader than what any one
GPU implements. FP8 is an example: the FNUZ encodings are available on gfx942, and the OCP encodings
are available on gfx950 and gfx12.

**Compute modes** are set on the descriptor. Some match the storage type, such as
``HIPBLAS_COMPUTE_32F``. Others deliberately compute at lower precision than the data is stored in,
including ``HIPBLAS_COMPUTE_32F_FAST_16F``, ``FAST_16BF``, and ``FAST_TF32``. Those modes trade some
accuracy for more throughput on FP32 data. You can go further and down-convert the inputs themselves
with ``HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_A_EXT`` and its ``B`` counterpart. A few compute
types can also be redirected at run time without changing the application; see
:ref:`Type overrides <env-type_overrides>`.

**Scaling** is the mechanism introduced earlier. FP8 and FP4 hold so few bits that typical tensor
values would overflow or collapse to zero. The values are stored scaled, and the scale factors are
reapplied during the multiply. What varies is how finely the factors are applied.
``HIPBLASLT_MATMUL_DESC_A_SCALE_MODE`` and ``B_SCALE_MODE`` select the granularity:

* One **scalar** for the entire tensor (``HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F``, the default
  for FP8). Cheapest, and adequate when a tensor's values share a range.
* An **outer vector** of per-row and per-column factors (``..._OUTER_VEC_32F``), which tolerates
  rows or columns whose magnitudes differ.
* **Block scaling** (``..._VEC32_UE8M0``), where every 32-element block carries its own 8-bit
  exponent. This is the finest granularity and the approach used for FP8 and FP4 operands. It also
  carries the tightest restrictions: it constrains the alignment of ``m``, ``n``, and ``k``, the
  transpose choices, and epilogue support. The :doc:`API reference <./reference/api-reference>`
  states the exact rules.

Scaling is a loop, not a one-time setup. Each layer's output becomes the next layer's input and
needs its own factor. ``HIPBLASLT_MATMUL_DESC_AMAX_D_POINTER`` closes that loop. It has the kernel
return the largest absolute value it wrote, so you can derive the next scale factor without a
separate pass over the data.

For supported types and the combinations that go together, start with
:doc:`hipBLASLt precision support <./reference/data-type-support>`.

Epilogues: fusing the work after the multiply
=============================================

In compiler terminology, an epilogue is the code at the end of a function. hipBLASLt applies the
same idea to a GEMM kernel. After the multiply-accumulate finishes, and before the results are
written to ``D``, the kernel can perform a short element-wise step. ``hipblasLtEpilogue_t`` selects
that step. ``HIPBLASLT_EPILOGUE_DEFAULT`` means no extra math, though results might still be scaled
or quantized on the way out.

The available functions come from neural networks. A network layer computes a linear transform, the
GEMM, then applies a non-linear function to the result. Without that non-linear function, stacking
layers would be no more expressive than a single layer. hipBLASLt provides the usual choices:

* **ReLU** (rectified linear unit), :math:`x := \max(x, 0)`. Negative values become zero and
  positive values pass through unchanged. It's the cheapest option.
* **GELU** (Gaussian error linear unit), a smooth version of ReLU that tapers near zero instead of
  cutting off sharply. It's standard in transformer models and often trains more stably.
* **Swish**, also called SiLU (sigmoid linear unit), :math:`x \cdot \mathrm{sigmoid}(x)`, using a
  sigmoid scale of 1. Also smooth, and common in more recent architectures.
* **Sigmoid**, :math:`1 / (1 + e^{-x})`, which compresses any value into the range
  :math:`(0, 1)`.
* **Clamp**, which limits each value to an interval,
  :math:`x := \max(\mathrm{lo}, \min(x, \mathrm{hi}))`. You supply the bounds as descriptor
  arguments.

A **bias** means a per-row additive offset, the :math:`b` in :math:`Wx + b`. The ``_BIAS`` epilogue
variants add the bias vector first, then apply the activation.

The remaining epilogues support training rather than inference. Training runs each layer forward to
produce a prediction, then works backward to compute how each parameter should change. The backward
pass requires values the forward pass has to save.

* **Auxiliary output**, the ``_AUX`` modes, writes the result *before* the activation to a separate
  ``E`` buffer, which has its own data type, leading dimension, and batch stride. The backward pass
  needs that tensor to evaluate the activation's derivative.
* **Gradient** modes do the backward step. ``DRELU`` and ``DGELU`` apply the derivative of the
  activation, reading the auxiliary tensor saved earlier. The ``BGRAD`` variants additionally sum a
  **bias gradient**, this layer's contribution to updating the bias, and ``BGRADA`` and ``BGRADB``
  produce that sum with respect to ``A`` or ``B``.

The two directions are meant to be paired. Choose an ``_AUX`` epilogue on the forward pass so the
matching ``D``-prefixed epilogue can consume its output on the backward pass.

Running many GEMMs at once
==========================

A single small matrix multiplication can't fill a GPU. Most of the device sits idle, and the launch
overhead rivals the arithmetic. The remedy is to hand the library many problems in one call.
hipBLASLt offers three ways to do that. The right one depends on where your matrices are already
allocated and whether they all have the same shape.

* **Strided batched** (``HIPBLASLT_BATCH_MODE_STRIDED``) expects one contiguous buffer per operand,
  with a fixed distance between one batch entry and the next. It takes the least setup and is
  usually the fastest, because the kernel computes each address arithmetically. Choose it when you
  control the allocation.
* **General batched**, also called pointer-array batched
  (``HIPBLASLT_BATCH_MODE_POINTER_ARRAY``), lets every batch entry sit at an unrelated device
  address. You pass an array of pointers that itself resides in device memory. Choose it when the
  matrices already exist as separate allocations, or when memory fragmentation rules out one large
  buffer. The shapes and leading dimensions must still be identical across the batch. See
  :doc:`Use general batched GEMM <./how-to/how-to-general-batched-gemm>`.
* **Grouped GEMM** is the one case the batch modes can't express: problems with *different* ``m``,
  ``n``, or ``k`` submitted together. It's available through the C++ extension API rather than as a
  layout attribute.

Grouped GEMM can also take its arguments from device memory. You fill in a ``UserArguments``
structure on the GPU, which lets a preceding kernel decide the sizes and pointers, so the host never
has to learn them. The "Fixed MK" pattern in the
:doc:`hipBLASLtExt API <./reference/ext-reference>` reference documents the constraints, including
declaring a total for ``N`` in advance.

The C++ extension API
=====================

The ``hipblaslt_ext`` namespace is a C++-only layer over the same machinery. Instead of a set of
descriptors, it gives you a ``Gemm`` or ``GroupedGemm`` object. You configure it, call
``initialize()``, then call ``run()`` as often as you like. That suits a framework that holds on to
a problem across many iterations.

It also exposes algorithm handling the C API doesn't. ``getAllAlgos()`` enumerates every kernel that
exists for a problem type. ``isAlgoSupported()`` tests a candidate against your specific problem and
reports the workspace it would need. ``getIndexFromAlgo()`` and ``getAlgosFromIndex()`` translate
between an algorithm and a **solution index**, a stable identifier you can record now and replay in
a later run.

Use the extension API when you need grouped GEMM, want to enumerate algorithms yourself, or intend
to pin kernels by index. Stay with the C API for portability and for the familiar
handle-and-descriptor style. See :doc:`hipBLASLtExt API <./reference/ext-reference>`.

How a kernel gets chosen
========================

hipBLASLt doesn't generate a kernel for your problem while your application runs. Kernels are
produced ahead of time, during the library build, and shipped as a **device library**: a collection
of compiled GPU code objects that accompanies the host library. At run time, hipBLASLt loads the
selection data for your architecture. It then uses **predicates** to determine which kernel
libraries are eligible for the GPU in the machine. A predicate is a condition on a device property
such as the processor name, the compute-unit count, or the PCI chip ID. hipBLASLt then applies its
heuristic to rank the eligible candidates for your problem shape.

Two practical consequences follow. First, a build of only the host library can't run a matmul; it
needs a matching device library. Clients look for one in ``/opt/rocm/lib`` unless
``HIPBLASLT_TENSILE_LIBPATH`` points somewhere else. Second, selection depends on both the device
and the contents of that device library. A solution index is therefore meaningful only for the
library build and architecture that produced it. Don't carry indices across releases or GPUs.

Two descriptor attributes deliberately constrain this selection.
``HIPBLASLT_MATMUL_DESC_UNIFORM_SUMMATION_ORDER_EXT``, or the equivalent setting on the handle,
restricts hipBLASLt to kernels that add up their partial results in a consistent order across the
``M`` dimension. Identical input rows then yield bitwise identical output rows. Fewer kernels
qualify, which can cost performance, and the call returns an error rather than quietly giving you a
non-uniform result. ``HIPBLASLT_MATMUL_DESC_SM_COUNT_TARGET`` steers selection and grid sizing
toward a compute-unit budget, which is useful when a GEMM has to share the device with other work.

:doc:`hipBLASLt PCI chip ID predicates <./conceptual/pci-chip-id-predicates-walkthrough>` describes
the eligibility mechanism in detail.

Tuning and Stream-K
===================

The built-in heuristic is a good general answer, but it can't anticipate your exact problem sizes.
Three mechanisms let you do better, in increasing order of effort.

* **Origami with Stream-K** is an alternative kernel library, selected by setting
  ``TENSILE_SOLUTION_SELECTION_METHOD=2``. Conventional GEMM kernels assign each workgroup a tile of
  the output, so when the work doesn't divide evenly, some compute units finish early and idle.
  Stream-K instead divides the total inner-loop iterations evenly among the compute units, which
  keeps utilization high. The effect is most visible when one dimension is much larger than the
  others. The practical benefit is more consistent performance across a wide range of shapes from
  far fewer tuned kernels. On the AMD Instinct MI350 series, Stream-K is the only strategy, and the
  variable has no effect. Additional variables cap the number of workgroups or compute units so a
  GEMM doesn't monopolize the device. See
  :doc:`Use Stream-K with hipBLASLt <./how-to/how-to-use-streamk>`.
* **Offline tuning** records the best kernel per problem and replays it, with no application changes.
  Set ``HIPBLASLT_LOG_MASK=32`` so each call logs a ready-made ``hipblaslt-bench`` command.
  Benchmark the candidates to find the fastest solution index, save the results to
  ``HIPBLASLT_TUNING_FILE``, and have later runs read them through
  ``HIPBLASLT_TUNING_OVERRIDE_FILE``. The indices are specific to one library build and
  architecture. See
  :doc:`Use hipBLASLt offline tuning <./how-to/how-to-use-hipblaslt-offline-tuning>`.
* **The tuning utility** goes one level deeper. ``find_exact.py`` benchmarks the existing kernel pool
  against the problem sizes you list. It writes an *equality logic* file: a YAML mapping from exact
  problem shapes to kernels that the library consults during selection, rather than a per-process
  override. See
  :doc:`Use the hipBLASLt tuning utility <./how-to/how-to-use-hipblaslt-tuning-utility>`.

Operations beyond GEMM
======================

Two parts of the library don't multiply matrices at all.

``hipblasLtMatrixTransform()`` computes :math:`C = alpha \cdot op(A) + beta \cdot op(B)`. Use it to
change a matrix's memory order or data type, to scale or shift its values, or to combine two
matrices. It's the tool for preparing data in the layout a matmul wants, or for converting results
afterward, without writing a kernel yourself.

``hipblasltExtSoftmax()``, ``hipblasltExtLayerNorm()``, and ``hipblasltExtAMax()`` are standalone
routines over 2D tensors. They're included because these reductions normally sit immediately before
or after a GEMM in a model, so calling them here avoids adding another library dependency. See
:doc:`hipBLASLtExt operation API <./reference/ext-ops>` and
:doc:`Code samples <./samples/samples>`.

Getting hipBLASLt and trying it out
===================================

hipBLASLt ships as part of the ROCm Core SDK, so installing a package is usually all you need; see
:doc:`Install hipBLASLt <./install/install>`. Build from source when you want a specific GPU
architecture, a static library, or the client programs; see
:doc:`Build from source <./install/building-installing-hipblaslt>`. A source build takes a while,
because it generates the device library described earlier. The build options let you narrow that
work to the architectures you care about.

A build configured with clients produces two executables. They're the quickest way to find out
whether something works and how fast it is before you write any code.

* ``hipblaslt-test`` is the correctness suite, built on GoogleTest, and can be narrowed with
  ``--gtest_filter``.
* ``hipblaslt-bench`` runs one configuration from the command line, optionally checks the result
  against a CPU reference, and prints the problem and its timing as a row of CSV.

.. code-block:: bash

   ./hipblaslt-bench --precision f32_r -v

Most of the concepts on this page have a corresponding ``hipblaslt-bench`` flag. That makes the tool
a convenient way to experiment with scaling modes, epilogues, batching, and specific solution
indices before committing to them in code. See
:doc:`hipBLASLt clients <./conceptual/hipblaslt-clients>`.

Seeing what the library is doing
================================

hipBLASLt reports on its own activity through environment variables, so you can investigate a
running application without rebuilding it. ``HIPBLASLT_LOG_LEVEL`` sets how much is reported, and
``HIPBLASLT_LOG_MASK`` selects which categories. Those categories include the bench-style command
lines that feed offline tuning and the profile lines that name the selected kernel.
``HIPBLASLT_LOG_FILE`` sends the output to a file, where ``%i`` expands to the process ID so
concurrent processes don't overwrite one another. ``HIPBLASLT_ENABLE_MARKER`` annotates the timeline
for :doc:`ROCProfiler <rocprofiler:index>`.

The same logging channel supports a more specific investigation. An opt-in scanner samples matmul
outputs and reports the first call whose result contained a NaN, along with a hint for narrowing
the search on a subsequent run. That turns "training diverged somewhere" into a specific GEMM, again
with no code changes. See :doc:`Detect NaN in GEMM output <./how-to/how-to-detect-nan>`, and
:doc:`Use logging and heuristics <./how-to/use-logging-heuristics>` for logging itself.
:doc:`hipBLASLt environment variables <./reference/env-variables>` catalogues every variable.

Summary
=======

hipBLASLt is a descriptor-driven GEMM library. You describe the layouts, the computation, and the
epilogue as attributes, ask once for a kernel, and reuse that decision. Fusion, scaling, batching,
and mixed precision all serve the same end: keeping work on the GPU and in registers instead of
shuttling intermediate results through memory. Kernel selection is automatic by default. Stream-K,
offline tuning, and the tuning utility are available when the default isn't good enough for your
shapes.

Everything else in this documentation is the detailed form of these same ideas. The how-to topics
cover the workflows, and the API reference covers the individual attributes.
:doc:`hipBLASLt library organization <./conceptual/hipblaslt-library-organization>` is the starting
point if you intend to work on the library itself rather than call it.
