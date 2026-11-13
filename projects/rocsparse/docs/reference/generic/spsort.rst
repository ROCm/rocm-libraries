.. meta::
  :description: rocSPARSE sparse matrix sorting API documentation
  :keywords: rocSPARSE, ROCm, API, documentation, sparse matrix sorting

.. _rocsparse_generic_spsort_:

********************************************************************
Sparse matrix sorting
********************************************************************

These routines sort the entries of a sparse matrix.
Call ``rocsparse_spsort_buffer_size()`` to obtain the temporary storage size for a stage,
then ``rocsparse_spsort()`` to sort the matrix.

rocsparse_spsort_buffer_size()
------------------------------

.. doxygenfunction:: rocsparse_spsort_buffer_size

rocsparse_spsort()
------------------

.. doxygenfunction:: rocsparse_spsort
