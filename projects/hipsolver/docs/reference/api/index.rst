.. meta::
  :description: Introduction to the regular hipSOLVER API
  :keywords: hipSOLVER, rocSOLVER, ROCm, API, documentation

.. _library_api:

********************************************************************
hipSOLVER regular API
********************************************************************

This topic provides the method signatures for the wrapper functions that hipSOLVER implements.
For a complete description of the behavior and arguments of the functions, see the corresponding backend documentation
at `cuSOLVER API <https://docs.nvidia.com/cuda/cusolver/>`_ and :doc:`rocSOLVER API <rocsolver:reference/intro>`.

The hipSOLVER API is designed to be similar to the cuSOLVER and rocSOLVER interfaces, but it requires some minor adjustments to ensure
the best performance out of both backends. This involves the addition of workspace parameters and some additional API methods.
See :ref:`usage_label` for a complete list of :ref:`API differences <api_differences>`.

If you're interested in using hipSOLVER without these adjustments, so that the interface matches cuSOLVER, consult the
:ref:`Compatibility API documentation <library_dense>` instead. See :ref:`the porting section <porting>` for more details.

*  :ref:`library_types`
*  :ref:`api_helpers`
*  :ref:`library_auxiliary`
*  :ref:`lapackfunc`
*  :ref:`lapacklike`
