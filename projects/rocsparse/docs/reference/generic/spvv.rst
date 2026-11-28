.. meta::
  :description: rocSPARSE sparse vector dot product API documentation
  :keywords: rocSPARSE, ROCm, API, documentation, SpVV, sparse dot product

.. _rocsparse_generic_spvv_:

********************************************************************
Sparse vector dot product
********************************************************************

``rocsparse_spvv()`` computes the inner dot product of a sparse vector and a dense vector:

.. math::

   \mathrm{result} := op(x) \cdot y

rocsparse_spvv()
----------------

.. doxygenfunction:: rocsparse_spvv
