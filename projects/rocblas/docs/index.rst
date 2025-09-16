.. meta::
  :description: rocBLAS documentation and API reference library
  :keywords: rocBLAS, ROCm, API, Linear Algebra, documentation

.. _rocblas:

********************************************************************
rocBLAS documentation
********************************************************************

rocBLAS is the ROCm Basic Linear Algebra Subprograms (BLAS) library. rocBLAS is implemented in :doc:`HIP C++ <hip:index>` and optimized for AMD GPUs.
This documentation set contains instructions for installing, understanding, and using the rocBLAS library.
To learn more, see :doc:`./what-is-rocblas`

The rocBLAS public repository is located at
`<https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocblas>`_.

.. note::

   The rocBLAS repository for ROCm 6.4 and earlier is located at `<https://github.com/ROCm/rocBLAS>`_.

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Installation

    * :doc:`Install rocBLAS on Linux <./install/Linux_Install_Guide>`
    * :doc:`Install rocBLAS on Windows <./install/Windows_Install_Guide>`

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Conceptual

    * :doc:`rocBLAS design notes <./conceptual/rocblas-design-notes>`
    * :doc:`BLAS operations introduction <./conceptual/blas-operations-intro>`

  .. grid-item-card:: How To

    * :doc:`Program with rocBLAS <./how-to/Programmers_Guide>`
    * :doc:`Use logging with rocBLAS <./how-to/logging-in-rocblas>`
    * :doc:`Contribute to rocBLAS <./how-to/Contributors_Guide>`

  .. grid-item-card:: Examples

    * `rocBLAS sample code <https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocblas/clients/samples>`_

  .. grid-item-card:: Reference

    * :ref:`env-variables`
    * :ref:`data-types-support`
    * :ref:`api-reference-guide`

To contribute to the documentation, see `Contributing to ROCm  <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

You can find licensing information on the `Licensing <https://rocm.docs.amd.com/en/latest/about/license.html>`_ page.
