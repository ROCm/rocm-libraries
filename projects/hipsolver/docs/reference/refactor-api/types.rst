.. meta::
  :description: hipSOLVER documentation and API reference library
  :keywords: hipSOLVER, rocSOLVER, ROCm, API, documentation

.. _refactor_types:

********************************************************************
Refactorization datatypes
********************************************************************

hipSOLVER defines types and enumerations that are internally converted to the corresponding backend 
types at runtime. Here we list the types used in the this compatibility API.

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

