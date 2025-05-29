.. meta::
  :description: hipSOLVER sparse matrices compatibility API documentation
  :keywords: hipSOLVER, rocSOLVER, ROCm, API, documentation, sparse matrix, compatibility

.. _library_sparse:

********************************************************************
hipSOLVER compatibility API: sparse matrices
********************************************************************

Here are the method signatures for the wrapper functions that hipSOLVER implements.
For a complete description of the behavior and arguments of the functions,
see the corresponding backend documentation
at `cuSOLVER API <https://docs.nvidia.com/cuda/cusolver/index.html#cuds-api>`_ and :doc:`rocSOLVER API <rocsolver:reference/intro>`.

For ease of porting from existing cuSOLVER applications to hipSOLVER,
functions in the hipsolverSp compatibility API are designed to have
method signatures that are consistent with the cusolverSp interface.
The equivalent functions have not been added to the regular hipSOLVER
API.

.. note::

   There are :ref:`some performance limitations <sparse_performance>` when using the rocSOLVER backend because not all
   functionality required for optimal performance has been implemented yet.

* :ref:`sparse_types`
* :ref:`sparse_helpers`
* :ref:`sparse_sparsefunc`
