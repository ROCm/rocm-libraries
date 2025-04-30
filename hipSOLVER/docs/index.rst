.. meta::
  :description: hipSOLVER documentation and API reference library
  :keywords: hipSOLVER, rocSOLVER, ROCm, API, documentation

.. _hipsolver:

********************************************************************
hipSOLVER documentation
********************************************************************

hipSOLVER is a LAPACK marshalling library with multiple supported backends.
It sits between the application and a "worker" LAPACK library,
marshalling inputs into the backend library and results back to the application.
hipSOLVER supports rocSOLVER and NVIDIA CUDA cuSOLVER as backends.
It exports an interface that does not require the client to change, regardless of the chosen backend.

The hipSOLVER public repository is located at `<https://github.com/ROCm/hipSOLVER>`_.

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Install

    * :doc:`Installation guide <./installation/install>`

  .. grid-item-card:: How to

    * :doc:`Use hipSOLVER <./howto/usage>`

  .. grid-item-card:: Examples

    * `Client samples <https://github.com/ROCm/hipSOLVER/tree/develop/clients/samples>`_

  .. grid-item-card:: Reference

    * :ref:`api-intro`
    * :ref:`precision-support`
    * :ref:`library_api`
    * :ref:`library_dense`
    * :ref:`library_sparse`
    * :ref:`library_refactor`

To contribute to the documentation, see `Contributing to ROCm <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

You can find licensing information on the `Licensing <https://rocm.docs.amd.com/en/latest/about/license.html>`_ page.

