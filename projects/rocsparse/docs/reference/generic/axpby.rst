.. meta::
  :description: rocSPARSE sparse vector axpby API documentation
  :keywords: rocSPARSE, ROCm, API, documentation, axpby, sparse vector

.. _rocsparse_generic_axpby_:

********************************************************************
Sparse vector axpby
********************************************************************

``rocsparse_axpby()`` scales a sparse vector and adds it to a scaled dense vector:

.. math::

   y := \alpha \cdot x + \beta \cdot y

rocsparse_axpby()
-----------------

.. doxygenfunction:: rocsparse_axpby
