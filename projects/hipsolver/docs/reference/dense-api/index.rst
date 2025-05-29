.. meta::
  :description: hipSOLVER compatibility API for dense matrices documentation
  :keywords: hipSOLVER, rocSOLVER, ROCm, API, documentation, compatibility, dense matrices

.. _library_dense:

********************************************************************
hipSOLVER compatibility API: dense matrices
********************************************************************

Here are the method signatures for the wrapper functions that hipSOLVER  implements.
For a complete description of the behavior and arguments of the functions, see the corresponding backend documentation
at `cuSOLVER API <https://docs.nvidia.com/cuda/cusolver/>`_ and :doc:`rocSOLVER API <rocsolver:reference/intro>`.

For ease of porting from existing cuSOLVER applications to hipSOLVER, functions in the hipsolverDn compatibility API are designed to have
method signatures that are consistent with the cusolverDn interface. However, :ref:`performance issues <dense_performance>` might arise when
using the rocSOLVER backend due to differing workspace requirements. If you are interested in achieving the best performance with
the rocSOLVER backend, review the :ref:`regular API documentation <library_api>` and transition from the compatibility API to
the regular API at the earliest convenience. See :ref:`usage_label` for additional :ref:`considerations regarding the use of
the compatibility API <dense_api_differences>`.

*  :ref:`dense_types`
*  :ref:`dense_helpers`
*  :ref:`dense_auxiliary`
*  :ref:`dense_lapackfunc`
*  :ref:`dense_lapacklike`
