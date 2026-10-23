.. meta::
   :description: Gate residual epilogue in hipSPARSELt
   :keywords: hipSPARSELt, ROCm, gate residual, epilogue, SpMM

.. _gate-residual:

**********************************************************************
Gate residual epilogue
**********************************************************************

The gate residual epilogue applies an element-wise gate matrix :math:`G` to the result of the
sparse matrix multiplication, fused into the same kernel:

.. math::

   D_{ij} = G_{ij} \cdot X_{ij} + G_{ij}

where :math:`X` is the result after alpha/beta scaling, bias, and activation, before it is converted
to the data type of ``D``.

Operation order
===============

When the gate residual is enabled, the epilogue is executed in the following order:

1. Sparse-dense matrix multiplication
2. Scaling by alpha/beta
3. Bias addition (if configured)
4. Activation function (if configured)
5. Gate residual: :math:`X \leftarrow G \cdot X + G`
6. Conversion to the data type of ``D``

All steps up to and including the gate residual are computed in ``FP32``. The gate values are
converted to ``FP32`` before they are applied.

Gate matrix requirements
========================

The gate matrix is described by a dense matrix descriptor with the following properties:

* The number of rows and columns must match matrix ``D``.
* The memory order must match matrix ``D``.
* The leading dimension can differ from ``D``.
* For batched operations, each batch uses its own gate matrix by default. A batch stride of ``0`` uses
  the same gate matrix for every batch.

Supported data types
====================

The gate residual is supported on ``gfx942`` and ``gfx950`` for the following data types:

.. list-table::
   :header-rows: 1

   * - Input A/B
     - Output D
     - Gate G
   * - ``FP16``
     - ``FP16``
     - ``FP16``
   * - ``BF16``
     - ``BF16``
     - ``BF16``
   * - ``FP8``
     - ``FP32``
     - ``FP32`` or ``FP8``

``FP8`` refers to ``HIP_R_8F_E4M3`` on ``gfx950`` and ``HIP_R_8F_E4M3_FNUZ`` on ``gfx942``.

Other combinations, including ``BF8`` and ``INT8`` inputs, are not supported.

Configuration
=============

Create a dense descriptor for the gate matrix, then attach it and the gate pointer to the matmul
descriptor. Set both before calling ``hipsparseLtMatmulAlgSelectionInit()``: the algorithm selection
chooses the kernel based on the gate configuration, so a gate attached later might not be applied, and no error
is returned. To change the gate configuration, create a new algorithm selection and plan.

.. code-block:: cpp

    // Describe the gate matrix: same size and memory order as matrix D
    hipsparseLtMatDescriptor_t matGate;
    CHECK_HIPSPARSELT_ERROR(hipsparseLtDenseDescriptorInit(
        &handle, &matGate, row_d, col_d, ldg, 16, HIP_R_16F, HIPSPARSE_ORDER_COL));

    // Batched operations only. Set these before attaching the descriptor, because it is copied when attached.
    // By default each batch uses its own gate matrix; a batch stride of 0 shares one gate matrix.
    CHECK_HIPSPARSELT_ERROR(hipsparseLtMatDescSetAttribute(
        &handle, &matGate, HIPSPARSELT_MAT_NUM_BATCHES, &num_batches, sizeof(num_batches)));
    int64_t gate_batch_stride = 0;
    CHECK_HIPSPARSELT_ERROR(hipsparseLtMatDescSetAttribute(
        &handle, &matGate, HIPSPARSELT_MAT_BATCH_STRIDE, &gate_batch_stride, sizeof(gate_batch_stride)));

    // Attach the gate descriptor: pass the descriptor itself, with the size of a pointer
    CHECK_HIPSPARSELT_ERROR(hipsparseLtMatmulDescSetAttribute(
        &handle,
        &matmul,
        HIPSPARSELT_MATMUL_GATE_RESIDUAL_DESC,
        &matGate,
        sizeof(hipsparseLtMatDescriptor_t*)));

    // Attach the gate matrix
    void* gate_ptr = d_gate;
    CHECK_HIPSPARSELT_ERROR(hipsparseLtMatmulDescSetAttribute(
        &handle,
        &matmul,
        HIPSPARSELT_MATMUL_GATE_RESIDUAL_MAT_POINTER,
        &gate_ptr,
        sizeof(void*)));

Setting the gate pointer to ``nullptr`` disables the gate residual.
