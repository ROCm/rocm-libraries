.. meta::
  :description: hipSOLVER refactorization data types documentation
  :keywords: hipSOLVER, rocSOLVER, ROCm, API, documentation, refactorization, data types

.. _refactor_types:

********************************************************************
Refactorization data types
********************************************************************

hipSOLVER defines types and enumerations that are internally converted to the corresponding backend 
types at runtime. Here is a list of the types used in this compatibility API.

hipSOLVER compatibility API types
====================================

.. contents:: List of types in the compatibility API
   :local:
   :backlinks: top

hipsolverRfHandle_t
---------------------------------
.. doxygentypedef:: hipsolverRfHandle_t

hipsolverRfFactorization_t
---------------------------------
.. doxygenenum:: hipsolverRfFactorization_t

hipsolverRfMatrixFormat_t
---------------------------------
.. doxygenenum:: hipsolverRfMatrixFormat_t

hipsolverRfNumericBoostReport_t
---------------------------------
.. doxygenenum:: hipsolverRfNumericBoostReport_t

hipsolverRfResetValuesFastMode_t
---------------------------------
.. doxygenenum:: hipsolverRfResetValuesFastMode_t

hipsolverRfTriangularSolve_t
---------------------------------
.. doxygenenum:: hipsolverRfTriangularSolve_t

hipsolverRfUnitDiagonal_t
---------------------------------
.. doxygenenum:: hipsolverRfUnitDiagonal_t

hipsolverStatus_t
--------------------
See :ref:`hipsolverStatus_t <status_t>`.

