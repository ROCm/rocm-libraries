.. meta::
  :description: hipSOLVER sparse matrix data types documentation
  :keywords: hipSOLVER, rocSOLVER, ROCm, API, documentation, sparse matrix, data types

.. _sparse_types:

*******************************
Sparse matrix data types
*******************************

hipSOLVER defines types and enumerations that are internally converted to the corresponding backend
types at runtime. Here is a list of the types used in the compatibility API.

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

