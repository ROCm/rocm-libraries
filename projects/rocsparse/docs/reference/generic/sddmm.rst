.. meta::
  :description: rocSPARSE sampled dense-dense matrix multiplication API documentation
  :keywords: rocSPARSE, ROCm, API, documentation, SDDMM, sampled dense-dense matrix multiplication

.. _rocsparse_generic_sddmm_:

********************************************************************
Sampled dense-dense matrix multiplication
********************************************************************

Sampled dense-dense matrix multiplication multiplies two dense matrices and keeps
the entries that match the sparsity pattern of the sparse result:

.. math::

   C := \alpha \cdot (op(A) \cdot op(B)) \circ \mathrm{spy}(C) + \beta \cdot C

Call ``rocsparse_sddmm_buffer_size()`` to obtain the temporary storage size and
``rocsparse_sddmm_preprocess()`` to preprocess the inputs.
``rocsparse_sddmm()`` performs the multiplication.

rocsparse_sddmm_buffer_size()
-----------------------------

.. doxygenfunction:: rocsparse_sddmm_buffer_size

rocsparse_sddmm_preprocess()
----------------------------

.. doxygenfunction:: rocsparse_sddmm_preprocess

rocsparse_sddmm()
-----------------

.. doxygenfunction:: rocsparse_sddmm
