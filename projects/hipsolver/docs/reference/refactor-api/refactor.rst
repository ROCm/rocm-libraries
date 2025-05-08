.. meta::
  :description: hipSOLVER documentation and API reference library
  :keywords: hipSOLVER, rocSOLVER, ROCm, API, documentation

.. _refactor_refactorfunc:

**************************
Refactorization Functions
**************************

Refactoring routines to solve complex Numerical Linear Algebra problems for sparse matrices.
These functions are organized in the following categories:

* :ref:`refactor_triangular`.
* :ref:`refactor_linears`. Based on triangular factorizations.



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

