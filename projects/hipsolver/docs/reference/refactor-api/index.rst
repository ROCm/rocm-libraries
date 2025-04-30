.. meta::
  :description: hipSOLVER documentation and API reference library
  :keywords: hipSOLVER, rocSOLVER, ROCm, API, documentation

.. _library_refactor:

********************************************************************
hipSOLVER compatibility API: refactorization
********************************************************************

This document provides the method signatures for the wrapper functions that are currently implemented in hipSOLVER.
For a complete description of the functions' behavior and arguments, see the corresponding backend documentation
at `cuSOLVER API <https://docs.nvidia.com/cuda/cusolver/index.html#cuds-api>`_ and/or :doc:`rocSOLVER API <rocsolver:reference/intro>`.

For ease of porting from existing cuSOLVER applications to hipSOLVER, functions in the hipsolverRf compatibility API are designed to have
method signatures that are consistent with the cusolverRf interface. At present, equivalent functions have not been added to hipSOLVER's regular API.

  * :ref:`refactor_types`
  * :ref:`refactor_helpers`
  * :ref:`refactor_refactorfunc`
