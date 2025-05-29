.. meta::
  :description: hipSOLVER compatibility API documentation
  :keywords: hipSOLVER, rocSOLVER, ROCm, API, documentation

.. _library_refactor:

********************************************************************
hipSOLVER compatibility API: refactorization
********************************************************************

This section lists the method signatures for the wrapper functions that hipSOLVER implements.
For a complete description of the behavior and arguments of the functions, see the corresponding backend documentation
at `cuSOLVER API <https://docs.nvidia.com/cuda/cusolver/index.html#cuds-api>`_ and :doc:`rocSOLVER API <rocsolver:reference/intro>`.

For ease of porting from existing cuSOLVER applications to hipSOLVER, functions in the hipsolverRf compatibility API are designed to have
method signatures that are consistent with the cusolverRf interface. Equivalent functions
have not yet been added to the regular hipSOLVER API.

*  :ref:`refactor_types`
*  :ref:`refactor_helpers`
*  :ref:`refactor_refactorfunc`
