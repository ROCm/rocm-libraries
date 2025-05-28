.. meta::
  :description: hipSOLVER sparse matrix functions API documentation
  :keywords: hipSOLVER, rocSOLVER, ROCm, API, documentation, sparse matrix

.. _sparse_sparsefunc:

************************
Sparse matrix functions
************************

Sparse matrix routines solve complex numerical linear algebra problems for sparse matrices.

.. _sparse_factlinears:

Combined factorization and linear-system solvers
=================================================

.. contents:: List of combined factorization and linear-system solvers
   :local:
   :backlinks: top

.. _sparse_csrlsvchol:

hipsolverSp<type>csrlsvchol()
---------------------------------------------------
.. doxygenfunction:: hipsolverSpDcsrlsvchol
   :outline:
.. doxygenfunction:: hipsolverSpScsrlsvchol

.. _sparse_csrlsvcholHost:

hipsolverSp<type>csrlsvcholHost()
---------------------------------------------------
.. doxygenfunction:: hipsolverSpDcsrlsvcholHost
   :outline:
.. doxygenfunction:: hipsolverSpScsrlsvcholHost

.. _sparse_csrlsvqr:

hipsolverSp<type>csrlsvqr()
---------------------------------------------------
.. doxygenfunction:: hipsolverSpDcsrlsvqr
   :outline:
.. doxygenfunction:: hipsolverSpScsrlsvqr

