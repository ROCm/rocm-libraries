.. meta::
  :description: rocSPARSE sparse matrix-dense matrix multiplication API documentation
  :keywords: rocSPARSE, ROCm, API, documentation, SpMM, sparse matrix dense matrix multiplication

.. _rocsparse_generic_spmm_:

********************************************************************
Sparse matrix-dense matrix multiplication
********************************************************************

``rocsparse_spmm()`` multiplies a sparse matrix by a dense matrix and adds the result
to a scaled dense matrix:

.. math::

   C := \alpha \cdot op(A) \cdot op(B) + \beta \cdot C

rocsparse_spmm()
----------------

.. doxygenfunction:: rocsparse_spmm
