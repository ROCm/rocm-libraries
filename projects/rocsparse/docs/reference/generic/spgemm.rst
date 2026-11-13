.. meta::
  :description: rocSPARSE sparse matrix-sparse matrix multiplication API documentation
  :keywords: rocSPARSE, ROCm, API, documentation, SpGEMM, sparse matrix sparse matrix multiplication

.. _rocsparse_generic_spgemm_:

********************************************************************
Sparse matrix-sparse matrix multiplication
********************************************************************

``rocsparse_spgemm()`` multiplies two sparse matrices and adds the result to a scaled
sparse matrix:

.. math::

   C := \alpha \cdot op(A) \cdot op(B) + \beta \cdot D

rocsparse_spgemm()
------------------

.. doxygenfunction:: rocsparse_spgemm
