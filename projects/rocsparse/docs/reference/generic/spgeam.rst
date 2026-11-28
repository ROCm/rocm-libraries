.. meta::
  :description: rocSPARSE sparse matrix addition API documentation
  :keywords: rocSPARSE, ROCm, API, documentation, SpGEAM, sparse matrix addition

.. _rocsparse_generic_spgeam_:

********************************************************************
Sparse matrix addition
********************************************************************

Sparse matrix addition computes

.. math::

   C := \alpha \cdot op(A) + \beta \cdot op(B)

Call ``rocsparse_spgeam_buffer_size()`` to obtain the temporary storage size for a stage,
then ``rocsparse_spgeam()`` to compute the sum.

rocsparse_spgeam_buffer_size()
------------------------------

.. doxygenfunction:: rocsparse_spgeam_buffer_size

rocsparse_spgeam()
------------------

.. doxygenfunction:: rocsparse_spgeam
