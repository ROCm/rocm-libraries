.. meta::
  :description: introduction to the rocSPARSE library and API reference library
  :keywords: rocSPARSE, ROCm, API, documentation, intro

.. _rocsparse:

********************************************************************
rocSPARSE documentation
********************************************************************

rocSPARSE is a library that provides basic linear algebra subroutines for sparse matrices and vectors.
It's created using the HIP programming language, implemented on top of the ROCm runtime and toolchains,
and optimized for AMD discrete GPUs.

The rocSPARSE public repository is located at `<https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocsparse>`_.

.. note::

   The rocSPARSE repository for ROCm 6.4.3 and earlier is located at `<https://github.com/ROCm/rocSPARSE>`_.

.. note::
   For portability, ROCm provides the `hipSPARSE <https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipsparse>`_ library.
   hipSPARSE includes a comprehensive, portable interface that supports multiple backends (including rocSPARSE and cuSPARSE).
   For documentation and examples, see the `hipSPARSE documentation <https://rocm.docs.amd.com/projects/hipSPARSE/en/latest/>`_.

For ROCm code examples, see `<https://github.com/ROCm/rocm-examples>`_.

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Install

    * :doc:`Linux installation guide <./install/Linux_Install_Guide>`
    * :doc:`Windows installation guide <./install/Windows_Install_Guide>`

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Conceptual

    * :doc:`rocSPARSE design <./conceptual/rocsparse-design>`

  .. grid-item-card:: How to

    * :doc:`Use rocSPARSE <./how-to/using-rocsparse>`
    * :doc:`Contribute to rocSPARSE <./how-to/contribute>`

  .. grid-item-card:: Examples

   * `Client samples <https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocsparse/clients/samples>`_

  .. grid-item-card:: API reference

    * :ref:`api`
    * :ref:`debugging_`
    * :ref:`rocsparse_types_`
    * :ref:`rocsparse_precision_support_`
    * :ref:`rocsparse_environment_variables_`
    * :ref:`rocsparse_roctx_`
    * :ref:`rocsparse_enumerations_`
    * :ref:`rocsparse_auxiliary_functions_`
    * :ref:`rocsparse_level1_functions_`
    * :ref:`rocsparse_level2_functions_`
    * :ref:`rocsparse_level3_functions_`
    * :ref:`rocsparse_extra_functions_`
    * :ref:`rocsparse_precond_functions_`
    * :ref:`rocsparse_conversion_functions_`
    * :ref:`rocsparse_reordering_functions_`
    * :ref:`rocsparse_utility_functions_`
    * :ref:`rocsparse_generic_functions_`
    * :ref:`reproducibility`

To contribute to the documentation, see `Contributing to ROCm <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

You can find licensing information on the `Licensing <https://rocm.docs.amd.com/en/latest/about/license.html>`_ page.
