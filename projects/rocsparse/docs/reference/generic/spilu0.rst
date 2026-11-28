.. meta::
  :description: rocSPARSE incomplete LU factorization API documentation
  :keywords: rocSPARSE, ROCm, API, documentation, SpILU0, incomplete LU

.. _rocsparse_generic_spilu0_:

********************************************************************
Incomplete LU factorization
********************************************************************

These routines compute an incomplete LU factorization with 0 fill-ins and no pivoting.
Call ``rocsparse_spilu0_buffer_size()`` to obtain the temporary storage size, then
``rocsparse_spilu0()`` to compute the factorization.

rocsparse_spilu0_buffer_size()
------------------------------

.. doxygenfunction:: rocsparse_spilu0_buffer_size

rocsparse_spilu0()
------------------

.. doxygenfunction:: rocsparse_spilu0
