.. meta::
   :description: How to use general batched GEMM (pointer-array batched GEMM) with hipBLASLt
   :keywords: hipBLASLt, ROCm, library, API, batched GEMM, pointer array

.. _general-batched-gemm:

**********************************
Using general batched GEMM
**********************************

General batched GEMM is a hipBLASLt feature that performs multiple independent
GEMM operations in a single API call.
Each batch references its own matrices through device-resident pointer arrays
instead of a single contiguous buffer with uniform strides.

Operation: ``D = alpha * op(A) * op(B) + beta * C``

Each batch can use separate memory allocations for ``A``, ``B``, ``C``, and ``D``.
This guide covers bench usage, the API workflow, a complete standalone example,
and common pitfalls.

Prerequisites: A ROCm installation with hipBLASLt built for your target GPU
architecture. For build instructions, see
:doc:`Build from source <../install/building-installing-hipblaslt>`.

What is general batched GEMM?
=============================

In GEMM, general is standard BLAS terminology for the usual rectangular matrix multiply (as
opposed to symmetric, triangular, or other specialized variants). In general batched GEMM, general
refers to the batching layout: each batch matrix may live at a separate device address,
referenced through a pointer array, rather than in one contiguous strided buffer. This is also
known as pointer-array batched GEMM.

General batched GEMM runs multiple GEMM operations where:

* Each batch has its own matrices in separate memory locations.
* Matrices are referenced through device pointer arrays (``A[]``, ``B[]``, ``C[]``, ``D[]``).
* Batch mode (``hipblasLtBatchMode_t``) is set to ``HIPBLASLT_BATCH_MODE_POINTER_ARRAY`` (or ``1``).
* Strided-batch offset attributes are not used; each pointer in the array
  identifies the base address of one matrix.

This mode is useful when:

* Matrices for different batches cannot be stored contiguously.
* You are integrating with code that already owns separate allocations.
* You need maximum flexibility in where each batch's data lives in device memory.

General batched vs. strided batched GEMM
=========================================

hipBLASLt supports two batching modes. Choosing the right one affects both setup
and performance.

Strided batched GEMM (``HIPBLASLT_BATCH_MODE_STRIDED``)
--------------------------------------------------------------------

Memory layout:

.. code-block:: text

   A: [A0 | A1 | A2 | A3]  <- single contiguous allocation
   B: [B0 | B1 | B2 | B3]  <- single contiguous allocation
   C: [C0 | C1 | C2 | C3]  <- single contiguous allocation
   D: [D0 | D1 | D2 | D3]  <- single contiguous allocation

Characteristics:

* All matrices live in one contiguous buffer per operand.
* Batch mode (``hipblasLtBatchMode_t``) is set to ``HIPBLASLT_BATCH_MODE_STRIDED`` (or ``0``).
* Uniform stride between batches set with ``HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET``.
* More memory-efficient and often slightly better cache locality.
* Less flexible: requires contiguous allocation.

Use when:

* All batches share identical dimensions and leading dimensions.
* You can allocate one buffer per operand.
* Memory is limited or you want the simplest batched setup.

General batched GEMM (``HIPBLASLT_BATCH_MODE_POINTER_ARRAY``)
---------------------------------------------------------------------------

Memory layout:

.. code-block:: text

   A: [ptr->A0, ptr->A1, ptr->A2, ptr->A3]  <- array of pointers
   B: [ptr->B0, ptr->B1, ptr->B2, ptr->B3]
   C: [ptr->C0, ptr->C1, ptr->C2, ptr->C3]
   D: [ptr->D0, ptr->D1, ptr->D2, ptr->D3]

   A0, A1, A2, A3 may reside anywhere in device memory.

Characteristics:

* Each batch matrix is independently allocated.
* No strided-batch offset attributes; pointer arrays carry the per-batch bases.
* Works with pre-existing, separately allocated matrices.
* Requires extra setup to build and copy pointer arrays to the device.

Use when:

* Matrices are already allocated separately.
* Contiguous batch buffers are impractical (for example, due to fragmentation).
* You need to integrate with legacy or modular code paths.

Quick comparison
----------------

.. csv-table::
   :header: "Feature", "Strided batched", "General batched"
   :widths: 30, 35, 35

   "Batch mode value (``hipblasLtBatchMode_t``)", "``0`` (``HIPBLASLT_BATCH_MODE_STRIDED``)", "``1`` (``HIPBLASLT_BATCH_MODE_POINTER_ARRAY``)"
   "Memory layout", "Single contiguous buffer", "Separate allocations"
   "Stride attributes", "Required (uniform stride)", "Not used"
   "Setup complexity", "Simple", "Moderate (pointer arrays on device)"
   "Memory efficiency", "Higher", "Moderate"
   "Flexibility", "Limited", "Maximum (non-contiguous bases)"
   "Best for", "Uniform batch problems", "Pre-allocated or non-contiguous batches"

.. important::

   A single ``hipblasLtMatmul`` call uses one matrix layout descriptor per
   operand. All batches in that call must therefore share the same ``m``,
   ``n``, ``k``, and leading dimensions (``lda``, ``ldb``, ``ldc``, and ``ldd``). General batched mode
   lets each batch live at a different address; it does not let each batch
   use different problem sizes in one call. For variable problem sizes, use
   grouped GEMM instead (see below).

Not the same as grouped GEMM
============================

The grouped GEMM API runs multiple GEMMs with different ``m``, ``n``, and/or ``k``
values in one launch. General batched GEMM runs multiple GEMMs that share the
same dimensions and layout metadata but use separate memory for each batch.

For strided batching examples in the hipBLASLt repository, see
``clients/samples/02_hipblaslt_gemm_batched/``.

When to use general batched GEMM
=================================

Choose general batched GEMM when you have:

1. Pre-allocated matrices already living in separate device buffers.
2. Dynamic batch management where batches are added or removed independently.
3. Memory fragmentation that prevents large contiguous allocations.
4. Legacy integration with code paths that already use per-batch pointers.
5. Different data sources where matrices originate from separate modules or structures.

Choose strided batched GEMM when you:

1. Are allocating memory specifically for batched GEMM.
2. Have uniform batch dimensions and can use contiguous buffers.
3. Want the simplest setup and typically the best performance for uniform problems.

Using hipblaslt-bench
=====================

Basic command structure
---------------------

.. code-block:: bash

   ./hipblaslt-bench [options]

Key options for general batched GEMM
------------------------------------

.. csv-table::
   :header: "Option", "Description", "Default", "Example"
   :widths: 20, 45, 15, 20

   "``--batch_mode``", "Batch mode: ``0`` = strided, ``1`` = general (pointer array)", "``0``", "``--batch_mode 1``"
   "``--batch_count``", "Number of batches", "``1``", "``--batch_count 4``"
   "``-m``, ``--sizem``", "Rows of ``op(A)`` and ``D``", "``128``", "``-m 512``"
   "``-n``, ``--sizen``", "Columns of ``op(B)`` and ``D``", "``128``", "``-n 256``"
   "``-k``, ``--sizek``", "Columns of ``op(A)`` / rows of ``op(B)``", "``128``", "``-k 128``"
   "``--alpha``", "Scalar alpha", "``1.0``", "``--alpha 1.5``"
   "``--beta``", "Scalar beta", "``0.0``", "``--beta 0.5``"
   "``--transA``", "Transpose ``A``: ``N``/``T``/``C``", "``N``", "``--transA N``"
   "``--transB``", "Transpose ``B``: ``N``/``T``/``C``", "``N``", "``--transB T``"
   "``-r``, ``--precision``", "Data type", "``f16_r``", "``--precision f32_r``"
   "``--lda``", "Leading dimension of ``A``", "auto", "``--lda 512``"
   "``--ldb``", "Leading dimension of ``B``", "auto", "``--ldb 128``"
   "``--ldc``", "Leading dimension of ``C``", "auto", "``--ldc 512``"
   "``--ldd``", "Leading dimension of ``D``", "auto", "``--ldd 512``"
   "``-v``, ``--verify``", "Enable CPU verification", "disabled", "``-v``"
   "``-i``, ``--iters``", "Timing iterations", "``10``", "``-i 100``"

Example: Basic general batched GEMM (FP32)
------------------------------------------

.. code-block:: bash

   ./hipblaslt-bench --batch_mode 1 --batch_count 4 \
     -m 128 -n 128 -k 128 \
     --precision f32_r \
     --verify

This runs four independent ``128 x 128 x 128`` FP32 GEMMs in pointer-array mode
and verifies the results against a CPU reference.

The first output line is a CSV header; subsequent lines report problem parameters
and timing. The ``batch_count`` column reflects the number of pointer-array
batches, and ``grouped_gemm`` remains ``0`` for this mode.

API overview
============

Key steps
---------

The following outline shows the essential workflow. Steps omitted here (matmul
preference, workspace, stream, and cleanup) are included in the complete example
below.

.. code-block:: cpp

   // 1. Create hipBLASLt handle and stream
   hipblasLtHandle_t handle;
   hipblasLtCreate(&handle);

   // 2. Allocate individual device matrices for each batch
   for(int i = 0; i < batch_count; ++i) {
       hipMalloc(&d_a[i], size_a * sizeof(float));
       // ... B, C, D ...
   }

   // 3. Build device-resident pointer arrays
   float** d_ptr_array_a;
   hipMalloc(&d_ptr_array_a, sizeof(float*) * batch_count);
   hipMemcpy(d_ptr_array_a, d_a.data(),
             sizeof(float*) * batch_count, hipMemcpyHostToDevice);
   // Repeat for B, C, D.

   // 4. Create matrix layouts (same m, n, k, lda for all batches)
   hipblasLtMatrixLayoutCreate(&mat_a, HIP_R_32F, m, k, lda);

   // 5. Set batch attributes on all four layouts
   hipblasLtBatchMode_t batch_mode = HIPBLASLT_BATCH_MODE_POINTER_ARRAY;
   hipblasLtMatrixLayoutSetAttribute(
       mat_a, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
       &batch_count, sizeof(batch_count));
   hipblasLtMatrixLayoutSetAttribute(
       mat_a, HIPBLASLT_MATRIX_LAYOUT_BATCH_MODE,
       &batch_mode, sizeof(batch_mode));

   // 6. Create matmul descriptor (set transposes and epilogue)
   hipblasLtMatmulDescCreate(&matmul_desc, HIPBLAS_COMPUTE_32F, HIP_R_32F);

   // 7. Query a heuristic and allocate workspace
   hipblasLtMatmulAlgoGetHeuristic(
       handle, matmul_desc, mat_a, mat_b, mat_c, mat_d,
       pref, 1, &heuristic_result, &returned_count);

   // 8. Execute with pointer arrays passed as the matrix pointers
   hipblasLtMatmul(handle, matmul_desc,
       &alpha,
       d_ptr_array_a, mat_a,
       d_ptr_array_b, mat_b,
       &beta,
       d_ptr_array_c, mat_c,
       d_ptr_array_d, mat_d,
       &heuristic_result.algo,
       workspace, workspace_size,
       stream);

Required layout attributes
--------------------------

For general batched GEMM, set both of the following on all four matrix
layouts (``A``, ``B``, ``C``, ``D``):

1. ``HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT``, number of batches.
2. ``HIPBLASLT_MATRIX_LAYOUT_BATCH_MODE``, ``HIPBLASLT_BATCH_MODE_POINTER_ARRAY``.

Do not set ``HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET`` in this mode.

Optional sub-matrix offset
--------------------------

When a batch matrix is a sub-region of a larger allocation, set
``HIPBLASLT_MATRIX_LAYOUT_OFFSET`` (in elements from the base pointer) on the
layout. Offsets are only valid when batch mode is pointer-array on all four
operands. See :doc:`hipBLASLt datatypes <../reference/datatypes>`.

Known limitations
-----------------

* Uniform dimensions per call: All batches in one ``hipblasLtMatmul`` must
  share the same ``m``, ``n``, ``k``, and leading dimensions.
* Scaling formats: When using ``hipblaslt-bench`` with ``--batch_mode 1``,
  only tensor-wide scaling is supported for matrices ``A`` and ``B``; vector and
  block scaling modes are rejected.
* Pointer arrays must be on device: Pass the device pointer to the pointer
  array, not a host-side array of addresses.

Complete example
================

The following standalone program demonstrates general batched GEMM from scratch:
per-batch allocation, device pointer arrays, layout configuration, heuristic
selection, execution, CPU verification, and cleanup.

Compile and run
---------------

From the ``docs/data/how-to`` directory (adjust ``ROCM_PATH`` if hipBLASLt is installed
elsewhere):

.. code-block:: bash

   ROCM_PATH=$(hipconfig --rocm-path)
   hipcc -o general_batched_gemm \
       example_general_batched_gemm_standalone.cpp \
       -I${ROCM_PATH}/include \
       -L${ROCM_PATH}/lib -lhipblaslt

   ./general_batched_gemm

Expected output:

.. code-block:: text

   === general batched GEMM example ===
   Problem size: M=128, N=64, K=96
   Batch count: 4
   Alpha=1.5, Beta=-0.5

   Algorithm selected with workspace size: XXXXX bytes
   Executing general batched GEMM...
   GEMM execution completed.
   Batch 0: PASSED
   Batch 1: PASSED
   Batch 2: PASSED
   Batch 3: PASSED

   SUCCESS: All batches passed verification!

Source code
-----------

.. literalinclude:: ../data/how-to/example_general_batched_gemm_standalone.cpp
   :language: c++

Performance considerations
==========================

* Strided vs. general: For uniform batches where you control allocation,
  strided batched GEMM is usually simpler and can be faster because the runtime
  avoids pointer-array indirection and extra device memory for the arrays.
* Pointer-array overhead: general batched mode allocates four pointer arrays
  plus ``batch_count`` separate buffers. Factor that into memory planning.
* Heuristic reuse: Query heuristics once with
  :ref:`hipblasltmatmulalgogetheuristic` and reuse the selected algorithm for
  repeated calls with the same problem descriptor. See also
  :doc:`Use logging and heuristics <./use-logging-heuristics>`.
* Workspace: Always honor ``workspaceSize`` returned by the heuristic; some
  algorithms require non-zero workspace for correct results.

Troubleshooting
===============

``returned_algo_count == 0`` (no suitable algorithm)
   Widen ``HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES``, verify data types and
   transpose combinations are supported on your GPU, or try a different precision.

Verification failures
   Confirm ``batch_count`` and ``HIPBLASLT_MATRIX_LAYOUT_BATCH_MODE`` match on
   all four layouts. Ensure pointer arrays were copied to the device with
   ``hipMemcpyHostToDevice``.

Invalid value / status errors
   Mixing batch modes across ``A``/``B``/``C``/``D`` layouts is invalid. Do not
   set strided-batch offsets when using pointer-array mode.

Segfaults or garbage results
   The matrix pointer arguments to ``hipblasLtMatmul`` must be device pointers
   to pointer arrays (``float**`` on device), not host arrays of ``float*``.

Summary
=======

General batched GEMM is the right choice when you need separate device allocations
per batch and cannot rely on contiguous strided buffers.

Key points:

1. Set ``--batch_mode 1`` (or ``HIPBLASLT_BATCH_MODE_POINTER_ARRAY``) for pointer-array batching.
2. Pass device-resident pointer arrays to ``hipblasLtMatmul``.
3. Do not use strided-batch offset attributes in this mode.
4. All batches in one call share the same dimensions and leading dimensions.
5. For variable ``m``/``n``/``k`` per problem, use grouped GEMM instead.

Quick-start bench command:

.. code-block:: bash

   ./hipblaslt-bench --batch_mode 1 --batch_count 4 \
     -m 512 -n 512 -k 512 --precision f32_r --verify
