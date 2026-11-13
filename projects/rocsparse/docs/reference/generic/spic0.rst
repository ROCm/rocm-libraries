.. meta::
  :description: rocSPARSE incomplete Cholesky factorization API documentation
  :keywords: rocSPARSE, ROCm, API, documentation, SpIC0, incomplete Cholesky

.. _rocsparse_generic_spic0_:

********************************************************************
Incomplete Cholesky factorization
********************************************************************

These routines compute an incomplete Cholesky factorization with 0 fill-ins and no pivoting.
Call ``rocsparse_spic0_buffer_size()`` to obtain the temporary storage size, then
``rocsparse_spic0()`` to compute the factorization.

rocsparse_spic0_buffer_size()
-----------------------------

.. doxygenfunction:: rocsparse_spic0_buffer_size

rocsparse_spic0()
-----------------

.. doxygenfunction:: rocsparse_spic0
