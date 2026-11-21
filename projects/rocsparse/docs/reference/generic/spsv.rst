.. meta::
  :description: rocSPARSE sparse triangular solve API documentation
  :keywords: rocSPARSE, ROCm, API, documentation, SpSV, SpTrSV, triangular solve

.. _rocsparse_generic_spsv_:

********************************************************************
Sparse triangular solve
********************************************************************

Sparse triangular solve computes the solution of

.. math::

   op(A) \cdot y = \alpha \cdot x

``rocsparse_spsv()`` solves the system in stages selected by its stage argument.
``rocsparse_sptrsv()`` solves the same system using an SpTrSV descriptor.
Call ``rocsparse_sptrsv_buffer_size()`` to obtain the temporary storage size for a stage
of ``rocsparse_sptrsv()``.

rocsparse_spsv()
----------------

.. doxygenfunction:: rocsparse_spsv

rocsparse_sptrsv_buffer_size()
------------------------------

.. doxygenfunction:: rocsparse_sptrsv_buffer_size

rocsparse_sptrsv()
------------------

.. doxygenfunction:: rocsparse_sptrsv
