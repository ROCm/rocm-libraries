.. meta::
  :description: hipSOLVER refactorization functions API documentation
  :keywords: hipSOLVER, rocSOLVER, ROCm, API, documentation, refactorization

.. _refactor_refactorfunc:

**************************
Refactorization functions
**************************

Refactoring routines are used to solve complex numerical linear algebra problems for sparse matrices.
These functions are organized into the following categories:

*  :ref:`refactor_triangular`
*  :ref:`refactor_linears`

.. _refactor_triangular:

Triangular factorizations
================================

.. contents:: List of triangular factorizations
   :local:
   :backlinks: top

.. _refactor_refactor:

hipsolverRfRefactor()
---------------------------------------------------
.. doxygenfunction:: hipsolverRfRefactor


.. _refactor_batch_refactor:

hipsolverRfBatchRefactor()
---------------------------------------------------
.. doxygenfunction:: hipsolverRfBatchRefactor



.. _refactor_linears:

Linear-systems solvers
================================

.. contents:: List of linear solvers
   :local:
   :backlinks: top

.. _refactor_solve:

hipsolverRfSolve()
---------------------------------------------------
.. doxygenfunction:: hipsolverRfSolve


.. _refactor_batch_solve:

hipsolverRfBatchSolve()
---------------------------------------------------
.. doxygenfunction:: hipsolverRfBatchSolve

