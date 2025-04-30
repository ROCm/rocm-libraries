.. meta::
  :description: hipSOLVER documentation and API reference library
  :keywords: hipSOLVER, rocSOLVER, ROCm, API, documentation

.. _sparse_types:

*******************************
Sparse matrix datatypes
*******************************

hipSOLVER defines types and enumerations that are internally converted to the corresponding backend
types at runtime. Here we list the types used in this compatibility API.

hipSOLVER compatibility API types
====================================

hipsolverSpHandle_t
---------------------------------
.. doxygentypedef:: hipsolverSpHandle_t

hipsparseMatDescr_t
---------------------------------
.. doxygentypedef:: hipsparseMatDescr_t

hipsolverStatus_t
--------------------
See :ref:`hipsolverStatus_t <status_t>`.

