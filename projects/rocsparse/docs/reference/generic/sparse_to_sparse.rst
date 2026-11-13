.. meta::
  :description: rocSPARSE sparse matrix to sparse matrix conversion API documentation
  :keywords: rocSPARSE, ROCm, API, documentation, sparse to sparse conversion

.. _rocsparse_generic_sparse_to_sparse_:

********************************************************************
Sparse matrix to sparse matrix conversion
********************************************************************

These routines convert a sparse matrix from one sparse format to another.
Call ``rocsparse_sparse_to_sparse_buffer_size()`` to obtain the temporary storage size
for a stage, then ``rocsparse_sparse_to_sparse()`` to convert the matrix.

rocsparse_sparse_to_sparse_buffer_size()
----------------------------------------

.. doxygenfunction:: rocsparse_sparse_to_sparse_buffer_size

rocsparse_sparse_to_sparse()
----------------------------

.. doxygenfunction:: rocsparse_sparse_to_sparse
