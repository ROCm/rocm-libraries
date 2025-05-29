.. meta::
  :description: hipSPARSE library precision support overview
  :keywords: hipSPARSE, rocSPARSE, ROCm, API, Sparse Linear Algebra, documentation, precision support, data types

.. _hipsparse-precision-support:

********************************************************************
hipSPARSE precision support
********************************************************************

This section provides an overview of the numerical precision types supported by the hipSPARSE library for sparse
matrix operations.

Supported precision types
=========================

hipSPARSE supports the following precision types across its functions:

.. list-table::
    :header-rows: 1

    *
      - Type prefix
      - C++ type
      - Description

    *
      - ``s``
      - ``float``
      - Single-precision real (32-bit)

    *
      - ``d``
      - ``double``
      - Double-precision real (64-bit)

    *
      - ``c``
      - ``hipFloatComplex``
      - Single-precision complex (32-bit real, 32-bit imaginary)

    *
      - ``z``
      - ``hipDoubleComplex``
      - Double-precision complex (64-bit real, 64-bit imaginary)

Function naming convention
--------------------------

hipSPARSE follows a naming convention where the precision type is included in the function name:

* Functions containing ``hipsparseS`` operate on single-precision real data.
* Functions containing ``hipsparseD`` operate on double-precision real data.
* Functions containing ``hipsparseC`` operate on single-precision complex data.
* Functions containing ``hipsparseZ`` operate on double-precision complex data.

For example, the dense matrix sparse vector multiplication function is implemented as:

* ``hipsparseSgemvi`` - For single-precision real sparse matrices.
* ``hipsparseDgemvi`` - For double-precision real sparse matrices.
* ``hipsparseCgemvi`` - For single-precision complex sparse matrices.
* ``hipsparseZgemvi`` - For double-precision complex sparse matrices.

In the documentation, these functions are often represented generically as ``hipsparseXgemvi()``, where ``X`` is a
placeholder for the precision type character.

Understanding precision in function signatures
----------------------------------------------

In the function signatures throughout the documentation, precision information is indicated directly in
the parameter types. For example:

.. code-block:: c

    hipsparseStatus_t hipsparseSgemvi(hipsparseHandle_t handle,
                                      hipsparseOperation_t transA,
                                      int m,
                                      int n,
                                      const float *alpha,
                                      const float *A,
                                      int lda,
                                      int nnz,
                                      const float *x,
                                      const int *xInd,
                                      const float *beta,
                                      float *y,
                                      hipsparseIndexBase_t idxBase,
                                      void *pBuffer)

The parameter types (``float``, ``double``, ``hipFloatComplex``, or ``hipDoubleComplex``)
correspond to the function precision type character and indicate the precision used by that specific function
variant.

Generic functions and mixed precision
-------------------------------------

hipSPARSE provides generic functions that allow for mixed-precision operations. These functions use data type
enumerations to specify precision when creating matrix or vector descriptors and during computation.

For example, when creating a sparse matrix descriptor:

.. code-block:: c

    // Create CSR matrix with single-precision values
    hipsparseCreateCsr(&matA, m, n, nnz,
                       rowPtr, colInd, values,
                       HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I,
                       HIPSPARSE_INDEX_BASE_ZERO, HIP_R_32F);

Or when creating a dense vector:

.. code-block:: c

    // Create dense vector with double-precision values
    hipsparseCreateDnVec(&vecX, size, xValues, HIP_R_64F);

Functions like ``hipsparseSpMV`` use a separate parameter for computation precision:

.. code-block:: c

    // Matrix-vector multiplication with computation in single precision
    hipsparseSpMV(handle, transA, &alpha, matA, vecX, &beta, vecY,
                  HIP_R_32F, HIPSPARSE_MV_ALG_DEFAULT, buffer);

This approach enables:

* Using different precision types for matrices and vectors.
* Specifying computation precision independently of storage precision.
* Supporting mixed-precision workflows with a unified API.

The advantage of using different data types is to save on memory bandwidth and storage when a user application
allows for it while performing the actual computation in a higher precision.

Real versus complex precision
-----------------------------

Most sparse matrix operations in hipSPARSE support both real and complex precisions, however certain algorithms
or optimizations might be specific to real or complex data:

* Some functions might have different performance characteristics for complex data compared to real data.
* Certain conversion routines might operate differently depending on whether the data is real or complex.

Core precision types
--------------------

The four core precision types (s, d, c, z) are supported across most functions in hipSPARSE. Some specialized
functions might only support a subset of these types. See the specific function documentation to confirm
which precision types are supported for a particular operation.
