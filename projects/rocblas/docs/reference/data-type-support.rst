.. meta::
  :description: rocBLAS library data type support
  :keywords: rocBLAS, ROCm, API, Linear Algebra, documentation, data type support

.. _data-types-support:

********************************************************************
Data type support
********************************************************************

This topic lists the data type support for the rocBLAS library on AMD GPUs for
different levels of BLAS operations :ref:`Level 1 <level-1>`,
:ref:`2 <level-2>`, and :ref:`3 <level-3>`.

The rocBLAS library functions are also available with ILP64 interfaces. With
these interfaces, all ``rocblas_int`` arguments are replaced by the type name
``int64_t``. For more information on these ``_64`` functions, see the
:ref:`ILP64 API` section.

The icons representing different levels of support are explained in the
following table.

.. list-table::
    :header-rows: 1

    *
      -  Icon
      - Definition

    *
      - NA
      - Not applicable

    *
      - ❌
      - Not supported

    *
      - ⚠️
      - Partial support

    *
      - ✅
      - Full support


For more information about data type support for the other ROCm libraries, see
:doc:`Data types and precision support page <rocm:reference/precision-support>`.

Level 1 functions - vector operations
=====================================

Level-1 functions perform scalar, vector, and vector-vector operations.

.. tab-set::

  .. tab-item:: Float types
    :sync: float-type

    .. list-table::
        :header-rows: 1
        :widths: 15, 45, 10, 10, 10, 10

        *
          - Function
          - Description
          - float16
          - bfloat16
          - float
          - double

        *
          - :ref:`AMax <rocblas_amax>`, :ref:`AMin <rocblas_amin>`, :ref:`ASum <rocblas_asum>`
          - Finds the first index of the element of minimum or maximum magnitude of a vector x or computes the sum of the magnitudes of elements of a real vector x.
          - ❌
          - ❌
          - ✅
          - ✅

        *
          - :ref:`AXPY <rocblas_axpy>`
          - Scales a vector and adds it to another: :math:`y = \alpha x + y`
          - ✅
          - ❌
          - ✅
          - ✅

        *
          - :ref:`Copy <rocblas_copy>`
          - Copies vector x to y: :math:`y = x`
          - ❌
          - ❌
          - ✅
          - ✅

        *
          - :ref:`Dot <rocblas_dot>`
          - Computes the dot product: :math:`result = x^T y`
          - ✅
          - ✅
          - ✅
          - ✅

        *
          - :ref:`NRM2 <rocblas_nrm2>`
          - Computes the Euclidean norm of a vector.
          - ❌
          - ❌
          - ✅
          - ✅

        *
          - :ref:`Rot <rocblas_rot>`, :ref:`Rotg <rocblas_rotg>`
          - Applies and generates a Givens rotation matrix.
          - ❌
          - ❌
          - ✅
          - ✅

        *
          - :ref:`Rotm <rocblas_rotm>`, :ref:`Rotmg <rocblas_rotmg>`
          - Applies and generates a modified Givens rotation matrix.
          - ❌
          - ❌
          - ✅
          - ✅

        *
          - :ref:`Scal <rocblas_scal>`
          - Scales a vector by a scalar: :math:`x = \alpha x`
          - ❌
          - ❌
          - ✅
          - ✅

        *
          - :ref:`Swap <rocblas_swap>`
          - Swaps corresponding elements of two vectors x and y: :math:`x_i \leftrightarrow y_i \quad \text{for} \quad i = 0, 1, 2, \ldots, n - 1`
          - ❌
          - ❌
          - ✅
          - ✅

  .. tab-item:: Complex types
    :sync: complex-type

    .. list-table::
        :header-rows: 1
        :widths: 15, 55, 10, 20

        *
          - Function
          - Description
          - complex
          - double complex

        *
          - :ref:`AMax <rocblas_amax>`, :ref:`AMin <rocblas_amin>`, :ref:`ASum <rocblas_asum>`
          - Finds the first index of the element of minimum or maximum magnitude of a vector x or computes the sum of the magnitudes of elements of a real vector x.
          - ✅
          - ✅

        *
          - :ref:`AXPY <rocblas_axpy>`
          - Scales a vector and adds it to another: :math:`y = \alpha x + y`
          - ✅
          - ✅

        *
          - :ref:`Copy <rocblas_copy>`
          - Copies vector x to y: :math:`y = x`
          - ✅
          - ✅

        *
          - :ref:`Dot <rocblas_dot>`
          - Computes the dot product: :math:`result = x^T y`
          - ✅
          - ✅

        *
          - :ref:`NRM2 <rocblas_nrm2>`
          - Computes the Euclidean norm of a vector.
          - ✅
          - ✅

        *
          - :ref:`Rot <rocblas_rot>`, :ref:`Rotg <rocblas_rotg>`
          - Applies and generates a Givens rotation matrix.
          - ✅
          - ✅

        *
          - :ref:`Rotm <rocblas_rotm>`, :ref:`Rotmg <rocblas_rotmg>`
          - Applies and generates a modified Givens rotation matrix.
          - ❌
          - ❌

        *
          - :ref:`Scal <rocblas_scal>`
          - Scales a vector by a scalar: :math:`x = \alpha x`
          - ✅
          - ✅

        *
          - :ref:`Swap <rocblas_swap>`
          - Swaps corresponding elements of two vectors x and y: :math:`x_i \leftrightarrow y_i \quad \text{for} \quad i = 0, 1, 2, \ldots, n - 1`
          - ✅
          - ✅

Level 2 functions - matrix-vector operations
============================================

Level-2 functions perform matrix-vector operations.

.. tab-set::

  .. tab-item:: Float types
    :sync: float-type

    .. list-table::
        :header-rows: 1
        :widths: 15, 45, 10, 10, 10, 10

        *
          - Function
          - Description
          - float16
          - bfloat16
          - float
          - double

        *
          - :ref:`GBMV <rocblas_gbmv>`
          - General band matrix-vector multiplication: :math:`y = \alpha A x + \beta y`
          - ❌
          - ❌
          - ✅
          - ✅

        *
          - :ref:`GEMV <rocblas_gemv>`
          - General matrix-vector multiplication: :math:`y = \alpha A x + \beta y`
          - ⚠️ [#gemv]_
          - ⚠️ [#gemv]_
          - ✅
          - ✅

        *
          - :ref:`GER <rocblas_ger>`
          - Generalized rank-1 update: :math:`A = \alpha x y^T + A`
          - ❌
          - ❌
          - ✅
          - ✅

        *
          - :ref:`GERU and GERC <rocblas_ger>`
          - Generalized rank-1 update for unconjugated or conjugated complex numbers: :math:`A = \alpha x y^T + A`
          - ❌
          - ❌
          - ❌
          - ❌

        *
          - :ref:`SBMV <rocblas_sbmv>`, :ref:`SPMV <rocblas_spmv>`
          - Symmetric Band matrix-vector multiplication: :math:`y = \alpha A x + \beta y`
          - ❌
          - ❌
          - ✅
          - ✅

        *
          - :ref:`SPR <rocblas_spr>`
          - Symmetric packed rank-1 update.
          - ❌
          - ❌
          - ✅
          - ✅

        *
          - :ref:`SPR2 <rocblas_spr2>`
          - Symmetric packed rank-2 update.
          - ❌
          - ❌
          - ✅
          - ✅

        *
          - :ref:`SYMV <rocblas_symv>`
          - Symmetric matrix-vector multiplication: :math:`y = \alpha A x + \beta y`
          - ❌
          - ❌
          - ✅
          - ✅

        *
          - :ref:`SYR <rocblas_syr>`, :ref:`SYR2 <rocblas_syr2>`
          - Symmetric matrix rank-1 or rank-2 update.
          - ❌
          - ❌
          - ✅
          - ✅

        *
          - :ref:`TBMV <rocblas_tbmv>`, :ref:`TBSV <rocblas_tbsv>`
          - Triangular band matrix-vector multiplication, triangular band solve.
          - ❌
          - ❌
          - ✅
          - ✅

        *
          - :ref:`TPMV <rocblas_tpmv>`, :ref:`TPSV <rocblas_tpsv>`
          - Triangular packed matrix-vector multiplication, triangular packed solve.
          - ❌
          - ❌
          - ✅
          - ✅

        *
          - :ref:`TRMV <rocblas_trmv>`, :ref:`TRSV <rocblas_trsv>`
          - Triangular matrix-vector multiplication, triangular solve.
          - ❌
          - ❌
          - ✅
          - ✅

        *
          - :ref:`HEMV <rocblas_hemv>`, :ref:`HBMV <rocblas_hbmv>`, :ref:`HPMV <rocblas_hpmv>`
          - Hermitian matrix-vector multiplication.
          - NA
          - NA
          - NA
          - NA

        *
          - :ref:`HER <rocblas_her>`, :ref:`HER2 <rocblas_her2>`
          - Hermitian rank-1 and rank-2 update.
          - NA
          - NA
          - NA
          - NA


        *
          - :ref:`HPR <rocblas_hpr>`, :ref:`HPR2 <rocblas_hpr2>`
          - Hermitian packed rank-1 and rank-2 update of packed.
          - NA
          - NA
          - NA
          - NA



  .. tab-item:: Complex types
    :sync: complex-type

    .. list-table::
        :header-rows: 1
        :widths: 15, 55, 10, 20

        *
          - Function
          - Description
          - complex
          - double complex

        *
          - :ref:`GBMV <rocblas_gbmv>`
          - General band matrix-vector multiplication: :math:`y = \alpha A x + \beta y`
          - ✅
          - ✅

        *
          - :ref:`GEMV <rocblas_gemv>`
          - General matrix-vector multiplication: :math:`y = \alpha A x + \beta y`
          - ✅
          - ✅

        *
          - :ref:`GER <rocblas_ger>`
          - Generalized rank-1 update: :math:`A = \alpha x y^T + A`
          - ❌
          - ❌

        *
          - :ref:`GERU and GERC <rocblas_ger>`
          - Generalized rank-1 update for unconjugated or conjugated complex numbers: :math:`A = \alpha x y^T + A`
          - ✅
          - ✅

        *
          - :ref:`SBMV <rocblas_sbmv>`, :ref:`SPMV <rocblas_spmv>`
          - Symmetric Band matrix-vector multiplication: :math:`y = \alpha A x + \beta y`
          - ❌
          - ❌

        *
          - :ref:`SPR <rocblas_spr>`
          - Symmetric packed rank-1 update.
          - ✅
          - ✅

        *
          - :ref:`SPR2 <rocblas_spr2>`
          - Symmetric packed rank-2 update.
          - ❌
          - ❌

        *
          - :ref:`SYMV <rocblas_symv>`
          - Symmetric matrix-vector multiplication: :math:`y = \alpha A x + \beta y`
          - ✅
          - ✅

        *
          - :ref:`SYR <rocblas_syr>`, :ref:`SYR2 <rocblas_syr2>`
          - Symmetric matrix rank-1 or rank-2 update.
          - ✅
          - ✅

        *
          - :ref:`TBMV <rocblas_tbmv>`, :ref:`TBSV <rocblas_tbsv>`
          - Triangular band matrix-vector multiplication, triangular band solve.
          - ✅
          - ✅

        *
          - :ref:`TPMV <rocblas_tpmv>`, :ref:`TPSV <rocblas_tpsv>`
          - Triangular packed matrix-vector multiplication, triangular packed solve.
          - ✅
          - ✅

        *
          - :ref:`TRMV <rocblas_trmv>`, :ref:`TRSV <rocblas_trsv>`
          - Triangular matrix-vector multiplication, triangular solve.
          - ✅
          - ✅

        *
          - :ref:`HEMV <rocblas_hemv>`, :ref:`HBMV <rocblas_hbmv>`, :ref:`HPMV <rocblas_hpmv>`
          - Hermitian matrix-vector multiplication.
          - ✅
          - ✅

        *
          - :ref:`HER <rocblas_her>`, :ref:`HER2 <rocblas_her2>`
          - Hermitian rank-1 and rank-2 update.
          - ✅
          - ✅

        *
          - :ref:`HPR <rocblas_hpr>`, :ref:`HPR2 <rocblas_hpr2>`
          - Hermitian packed rank-1 and rank-2 update.
          - ✅
          - ✅

Level 3 functions - matrix-matrix operations
============================================

Level-3 functions perform matix-matrix operations. rocBLAS calls the AMD
:doc:`Tensile <tensile:src/index>` and :doc:`hipBLASLt <hipblaslt:index>`
libraries for Level-3 GEMMs (matrix matrix multiplication).

.. tab-set::

  .. tab-item:: Float types
    :sync: float-type

    .. list-table::
        :header-rows: 1
        :widths: 15, 45, 10, 10, 10, 10

        *
          - Function
          - Description
          - float16
          - bfloat16
          - float
          - double

        *
          - :ref:`GEMM <rocblas_gemm>`
          - General matrix-matrix multiplication: :math:`C = \alpha A B + \beta C`
          - ✅
          - ❌
          - ✅
          - ✅

        *
          - :ref:`SYMM <rocblas_symm>`
          - Symmetric matrix-matrix multiplication: :math:`C = \alpha A B + \beta C`
          - ❌
          - ❌
          - ✅
          - ✅

        *
          - :ref:`SYRK <rocblas_syrk>`, :ref:`SYR2K <rocblas_syr2k>`
          - Update symmetric matrix with one matrix product or by using two matrices.
          - ❌
          - ❌
          - ✅
          - ✅

        *
          - :ref:`SYRKX <rocblas_syrkx>`
          - SYRKX adds an extra matrix multiplication step before updating the symmetric matrix.
          - ❌
          - ❌
          - ✅
          - ✅

        *
          - :ref:`TRMM <rocblas_trmm>`
          - Triangular matrix-matrix multiplication.
          - ❌
          - ❌
          - ✅
          - ✅

        *
          - :ref:`TRSM <rocblas_trsm>`
          - Triangular solve with multiple right-hand sides.
          - ❌
          - ❌
          - ✅
          - ✅
        *
          - :ref:`HEMM <rocblas_hemm>`
          - Hermitian matrix-matrix multiplication.
          - NA
          - NA
          - NA
          - NA

        *
          - :ref:`HERK <rocblas_herk>`, :ref:`HER2K <rocblas_her2k>`
          - Update Hermitian matrix with one matrix product or by using two matrices.
          - NA
          - NA
          - NA
          - NA

        *
          - :ref:`HERKX <rocblas_herkx>`
          - HERKX adds an extra matrix multiplication step before updating the Hermitian matrix.
          - NA
          - NA
          - NA
          - NA
        *
          - :ref:`TRTRI <rocblas_trtri>`
          - Triangular matrix inversion.
          - ❌
          - ❌
          - ✅
          - ✅


  .. tab-item:: Complex types
    :sync: complex-type

    .. list-table::
        :header-rows: 1
        :widths: 15, 55, 10, 20

        *
          - Function
          - Description
          - complex
          - double complex

        *
          - :ref:`GEMM <rocblas_gemm>`
          - General matrix-matrix multiplication: :math:`C = \alpha A B + \beta C`
          - ✅
          - ✅

        *
          - :ref:`SYMM <rocblas_symm>`
          - Symmetric matrix-matrix multiplication: :math:`C = \alpha A B + \beta C`
          - ✅
          - ✅

        *
          - :ref:`SYRK <rocblas_syrk>`, :ref:`SYR2K <rocblas_syr2k>`
          - Update symmetric matrix with one matrix product or by using two matrices.
          - ✅
          - ✅

        *
          - :ref:`SYRKX <rocblas_syrkx>`
          - SYRKX adds an extra matrix multiplication step before updating the symmetric matrix.
          - ✅
          - ✅

        *
          - :ref:`TRMM <rocblas_trmm>`
          - Triangular matrix-matrix multiplication.
          - ✅
          - ✅

        *
          - :ref:`TRSM <rocblas_trsm>`
          - Triangular solve with multiple right-hand sides.
          - ✅
          - ✅
        *
          - :ref:`HEMM <rocblas_hemm>`
          - Hermitian matrix-matrix multiplication.
          - ✅
          - ✅

        *
          - :ref:`HERK <rocblas_herk>`, :ref:`HER2K <rocblas_her2k>`
          - Update Hermitian matrix with one matrix product or by using two matrices.
          - ✅
          - ✅

        *
          - :ref:`HERKX <rocblas_herkx>`
          - HERKX adds an extra matrix multiplication step before updating the Hermitian matrix.
          - ✅
          - ✅
        *
          - :ref:`TRTRI <rocblas_trtri>`
          - Triangular matrix inversion.
          - ❌
          - ❌


Extensions
==========

The extension function data type support is listed for every function separately
on the :ref:`Extensions reference page <extension>`.

.. rubric:: Footnotes

.. [#gemv] Only the batched and strided batched GEMV functions support the ``half`` and ``bfloat16`` types.
