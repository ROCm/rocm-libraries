.. meta::
  :description: rocSPARSE sparse triangular solve with multiple right-hand sides API documentation
  :keywords: rocSPARSE, ROCm, API, documentation, SpSM, SpTrSM, triangular solve

.. _rocsparse_generic_spsm_:

********************************************************************
Sparse triangular solve with multiple right-hand sides
********************************************************************

Sparse triangular solve with multiple right-hand sides computes the solution of

.. math::

   op(A) \cdot C = \alpha \cdot op(B)

``rocsparse_spsm()`` solves the system in stages selected by its stage argument.
``rocsparse_sptrsm()`` solves the same system using an SpTrSM descriptor.
Call ``rocsparse_sptrsm_buffer_size()`` to obtain the temporary storage size for a stage
of ``rocsparse_sptrsm()``.

rocsparse_spsm()
----------------

.. doxygenfunction:: rocsparse_spsm

rocsparse_sptrsm_buffer_size()
------------------------------

.. doxygenfunction:: rocsparse_sptrsm_buffer_size

rocsparse_sptrsm()
------------------

.. doxygenfunction:: rocsparse_sptrsm
