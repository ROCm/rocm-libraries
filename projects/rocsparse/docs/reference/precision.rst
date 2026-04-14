.. meta::
  :description: rocSPARSE library precision support overview
  :keywords: rocSPARSE, ROCm, API, Sparse Linear Algebra, documentation, precision support, data types

.. _rocsparse_precision_support_:

********************************************************************
rocSPARSE precision support
********************************************************************

This section provides an overview of the numerical precision types supported by the rocSPARSE library for sparse
matrix operations.

Supported precision types
=========================

rocSPARSE supports the following precision types across its functions:

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
      - ``rocsparse_float_complex``
      - Single-precision complex (32-bit real, 32-bit imaginary)

    *
      - ``z``
      - ``rocsparse_double_complex``
      - Double-precision complex (64-bit real, 64-bit imaginary)

Function naming convention
--------------------------

rocSPARSE follows a naming convention where the precision type is included in the function name:

* Functions containing ``_s`` operate on single-precision real data.
* Functions containing ``_d`` operate on double-precision real data.
* Functions containing ``_c`` operate on single-precision complex data.
* Functions containing ``_z`` operate on double-precision complex data.

For example, the dense matrix sparse vector multiplication function is implemented as:

* ``rocsparse_sgemvi`` - For single-precision real sparse matrices.
* ``rocsparse_dgemvi`` - For double-precision real sparse matrices.
* ``rocsparse_cgemvi`` - For single-precision complex sparse matrices.
* ``rocsparse_zgemvi`` - For double-precision complex sparse matrices.

In the documentation, these are often represented generically as ``rocsparse_Xgemvi()``, where ``X`` is a
placeholder for the precision type character.

Understanding precision in function signatures
----------------------------------------------

In the function signatures throughout the documentation, precision information is indicated directly in
the parameter types. For example:

.. code-block:: c

    rocsparse_status rocsparse_sgemvi(rocsparse_handle handle,
                                      rocsparse_operation trans,
                                      rocsparse_int m,
                                      rocsparse_int n,
                                      const float *alpha,
                                      const float *A,
                                      rocsparse_int lda,
                                      rocsparse_int nnz,
                                      const float *x_val,
                                      const rocsparse_int *x_ind,
                                      const float *beta,
                                      float *y,
                                      rocsparse_index_base idx_base,
                                      void *temp_buffer)

The parameter types (``float``, ``double``, ``rocsparse_float_complex``, or ``rocsparse_double_complex``)
correspond to the function precision type character and indicate the precision used by that specific function
variant.

Generic functions and mixed precision
-------------------------------------

rocSPARSE provides generic functions that allow for mixed-precision operations. These functions use data type
enumerations to specify precision when creating matrix/vector descriptors and during computation.

For example, when creating a sparse matrix descriptor:

.. code-block:: c

    // Create CSR matrix with single-precision values
    rocsparse_create_csr_descr(&matA, m, n, nnz,
                               dcsr_row_ptr, dcsr_col_ind, dcsr_val,
                               rocsparse_indextype_i32, rocsparse_indextype_i32,
                               rocsparse_index_base_zero, rocsparse_datatype_f32_r);

Or when creating a dense vector:

.. code-block:: c

    // Create dense vector with double-precision values
    rocsparse_create_dnvec_descr(&vecX, n, dx, rocsparse_datatype_f64_r);

Functions like ``rocsparse_spmv`` use a separate parameter for computation precision:

.. code-block:: c

    // Matrix-vector multiplication with computation in single precision
    rocsparse_spmv(handle, trans, &alpha, matA, vecX, &beta, vecY,
                   rocsparse_datatype_f32_r, rocsparse_spmv_alg_csr_adaptive,
                   rocsparse_spmv_stage_compute, &buffer_size, temp_buffer);

This approach enables:

* Using different precision types for matrices and vectors.
* Specifying computation precision independently of storage precision.
* Supporting mixed-precision workflows with a unified API.

The advantage of using different data types is to save on memory bandwidth and storage when a user application
allows while performing the actual computation in a higher precision.

Real versus complex precision
-----------------------------

Most sparse matrix operations in rocSPARSE support both real and complex precisions, however certain algorithms
or optimizations might be specific to real or complex data:

* Some functions might have different performance characteristics for complex data compared to real data.
* Certain conversion routines might operate differently depending on whether the data is real or complex.

Core precision types
--------------------

The four core precision types (s, d, c, z) are supported across most functions in rocSPARSE. Some specialized
functions might only support a subset of these types. See the specific function documentation to confirm
which precision types are supported for a particular operation.
