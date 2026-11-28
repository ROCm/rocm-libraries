.. meta::
  :description: rocSPARSE generic functions API documentation
  :keywords: rocSPARSE, ROCm, API, documentation, generic functions

.. _rocsparse_generic_functions_:

********************************************************************
Sparse generic functions
********************************************************************

This module contains all sparse generic routines.

The sparse generic routines describe some of the most common operations that manipulate sparse matrices and
vectors. The generic API is more flexible than the other rocSPARSE APIs because it is easy to set
different index types, data types, and compute types. For some generic routines, for example, SpMV, the generic
API also lets users select different algorithms which have different performance characteristics depending
on the sparse matrix being operated on.

Each operation is documented on its own page:

* :ref:`rocsparse_generic_axpby_` scales a sparse vector and adds it to a scaled dense vector
  (:cpp:func:`rocsparse_axpby()`).
* :ref:`rocsparse_generic_gather_` gathers elements from a dense vector and stores them in a sparse vector
  (:cpp:func:`rocsparse_gather()`).
* :ref:`rocsparse_generic_scatter_` scatters elements from a sparse vector into a dense vector
  (:cpp:func:`rocsparse_scatter()`).
* :ref:`rocsparse_generic_rot_` applies a Givens rotation to a sparse vector and a dense vector
  (:cpp:func:`rocsparse_rot()`).
* :ref:`rocsparse_generic_spvv_` computes the inner dot product of a sparse vector and a dense vector
  (:cpp:func:`rocsparse_spvv()`).
* :ref:`rocsparse_generic_spmv_` multiplies a sparse matrix by a dense vector
  (:cpp:func:`rocsparse_spmv()`, :cpp:func:`rocsparse_v2_spmv_buffer_size()`,
  :cpp:func:`rocsparse_v2_spmv()`, :cpp:func:`rocsparse_spmv_set_extra()`,
  :cpp:func:`rocsparse_spmv_clear_extra()`).
* :ref:`rocsparse_generic_spsv_` solves a sparse triangular linear system
  (:cpp:func:`rocsparse_spsv()`, :cpp:func:`rocsparse_sptrsv_buffer_size()`,
  :cpp:func:`rocsparse_sptrsv()`).
* :ref:`rocsparse_generic_spilu0_` computes an incomplete LU factorization with 0 fill-ins and no pivoting
  (:cpp:func:`rocsparse_spilu0_buffer_size()`, :cpp:func:`rocsparse_spilu0()`).
* :ref:`rocsparse_generic_spic0_` computes an incomplete Cholesky factorization with 0 fill-ins and
  no pivoting (:cpp:func:`rocsparse_spic0_buffer_size()`, :cpp:func:`rocsparse_spic0()`).
* :ref:`rocsparse_generic_spildlt0_` computes an incomplete :math:`LDL^H` factorization with 0 fill-ins
  and no pivoting (:cpp:func:`rocsparse_spildlt0_buffer_size()`, :cpp:func:`rocsparse_spildlt0()`).
* :ref:`rocsparse_generic_spsm_` solves a sparse triangular linear system with multiple right-hand sides
  (:cpp:func:`rocsparse_spsm()`, :cpp:func:`rocsparse_sptrsm_buffer_size()`,
  :cpp:func:`rocsparse_sptrsm()`).
* :ref:`rocsparse_generic_spmm_` multiplies a sparse matrix by a dense matrix
  (:cpp:func:`rocsparse_spmm()`).
* :ref:`rocsparse_generic_spgemm_` multiplies two sparse matrices
  (:cpp:func:`rocsparse_spgemm()`).
* :ref:`rocsparse_generic_spgeam_` adds two sparse matrices
  (:cpp:func:`rocsparse_spgeam_buffer_size()`, :cpp:func:`rocsparse_spgeam()`).
* :ref:`rocsparse_generic_spsort_` sorts the entries of a sparse matrix
  (:cpp:func:`rocsparse_spsort_buffer_size()`, :cpp:func:`rocsparse_spsort()`).
* :ref:`rocsparse_generic_spmat_scale_` scales the values of a sparse matrix
  (:cpp:func:`rocsparse_spmat_scale()`).
* :ref:`rocsparse_generic_sddmm_` multiplies two dense matrices and samples the product on a sparse
  pattern (:cpp:func:`rocsparse_sddmm_buffer_size()`, :cpp:func:`rocsparse_sddmm_preprocess()`,
  :cpp:func:`rocsparse_sddmm()`).
* :ref:`rocsparse_generic_dense_to_sparse_` converts a dense matrix to a sparse matrix
  (:cpp:func:`rocsparse_dense_to_sparse()`).
* :ref:`rocsparse_generic_sparse_to_dense_` converts a sparse matrix to a dense matrix
  (:cpp:func:`rocsparse_sparse_to_dense()`).
* :ref:`rocsparse_generic_sparse_to_sparse_` converts a sparse matrix from one sparse format to another
  (:cpp:func:`rocsparse_sparse_to_sparse_buffer_size()`, :cpp:func:`rocsparse_sparse_to_sparse()`).
* :ref:`rocsparse_generic_extract_` extracts the lower or upper triangle of a sparse matrix
  (:cpp:func:`rocsparse_extract_buffer_size()`, :cpp:func:`rocsparse_extract_nnz()`,
  :cpp:func:`rocsparse_extract()`).
* :ref:`rocsparse_generic_check_spmat_` checks whether a sparse matrix is valid
  (:cpp:func:`rocsparse_check_spmat()`).
* :ref:`rocsparse_generic_spitsv_` solves a sparse triangular linear system with the Jacobi iterative
  method (:cpp:func:`rocsparse_spitsv()`).
