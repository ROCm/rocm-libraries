.. meta::
  :description: hipSOLVER sparse matrix helper functions API documentation
  :keywords: hipSOLVER, rocSOLVER, ROCm, API, documentation

.. _sparse_helpers:

*******************************
Sparse matrix helper functions
*******************************

These helper functions control aspects of the hipSOLVER library. They are divided
into the following categories:

* :ref:`sparse_initialize`: Functions to initialize and cleanup the library handle.
* :ref:`sparse_stream`: Functions to manipulate streams.


.. _sparse_initialize:

Handle setup and teardown
===============================

.. contents:: List of handle initialization functions
   :local:
   :backlinks: top

hipsolverSpCreate()
-----------------------------------------
.. doxygenfunction:: hipsolverSpCreate

hipsolverSpDestroy()
-----------------------------------------
.. doxygenfunction:: hipsolverSpDestroy



.. _sparse_stream:

Stream manipulation
==============================

.. contents:: List of stream manipulation functions
   :local:
   :backlinks: top

hipsolverSpSetStream()
---------------------------------
.. doxygenfunction:: hipsolverSpSetStream

