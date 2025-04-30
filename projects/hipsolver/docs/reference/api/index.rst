.. meta::
  :description: hipSOLVER documentation and API reference library
  :keywords: hipSOLVER, rocSOLVER, ROCm, API, documentation

.. _library_api:

********************************************************************
hipSOLVER regular API
********************************************************************

This document provides the method signatures for wrapper functions that are currently implemented in hipSOLVER.
For a complete description of the functions' behavior and arguments, see the corresponding backend documentation
at `cuSOLVER API <https://docs.nvidia.com/cuda/cusolver/>`_ and/or :doc:`rocSOLVER API <rocsolver:reference/intro>`.

The hipSOLVER API is designed to be similar to the cuSOLVER and rocSOLVER interfaces, but it requires some minor adjustments to ensure
the best performance out of both backends. Generally, this involves the addition of workspace parameters and some additional API methods.
Refer to :ref:`usage_label` for a complete list of :ref:`API differences <api_differences>`.

Users interested in using hipSOLVER without these adjustments, so that the interface matches cuSOLVER, should instead consult the
:ref:`Compatibility API documentation <library_dense>`. See also :ref:`the porting section <porting>` for more details.

  * :ref:`library_types`
  * :ref:`api_helpers`
  * :ref:`library_auxiliary`
  * :ref:`lapackfunc`
  * :ref:`lapacklike`
