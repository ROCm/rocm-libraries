.. meta::
  :description: hipSOLVER documentation and API reference library
  :keywords: hipSOLVER, rocSOLVER, ROCm, API, documentation

.. _sparse_sparsefunc:

************************
Sparse matrix functions
************************

Sparse matrix routines to solve complex Numerical Linear Algebra problems for sparse matrices.
These functions are organized in the following categories:

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

