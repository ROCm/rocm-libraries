.. meta::
  :description: rocSPARSE sparse iterative triangular solve API documentation
  :keywords: rocSPARSE, ROCm, API, documentation, SpITSV, iterative triangular solve

.. _rocsparse_generic_spitsv_:

********************************************************************
Sparse iterative triangular solve
********************************************************************

``rocsparse_spitsv()`` solves a sparse triangular linear system with the Jacobi
iterative method:

.. math::

   op(A) \cdot y = \alpha \cdot x

rocsparse_spitsv()
------------------

.. doxygenfunction:: rocsparse_spitsv
