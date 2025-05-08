.. meta::
  :description: hipSOLVER documentation and API reference library
  :keywords: hipSOLVER, rocSOLVER, ROCm, API, documentation

.. _sparse_helpers:

*******************************
Sparse matrix helper functions
*******************************

These are helper functions that control aspects of the hipSOLVER library. They are divided
into the following categories:

* :ref:`sparse_initialize` functions. Used to initialize and cleanup the library handle.
* :ref:`sparse_stream` functions. Provide functionality to manipulate streams.


.. _sparse_initialize:

Handle set-up and tear-down
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

