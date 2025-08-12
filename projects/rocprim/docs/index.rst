.. meta::
  :description: rocPRIM documentation and API reference library
  :keywords: rocPRIM, ROCm, API, documentation

.. _rocprim:

********************************************************************
 rocPRIM documentation
********************************************************************

rocPRIM is a header-only library that provides HIP parallel primitives. The purpose of the library is to provide a set of portable, commonly used, GPU-accelerated parallel primitive algorithms. rocPRIM is written in HIP and has been optimized for AMD's latest discrete GPUs.

The rocPRIM project is located in https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocprim.

.. grid:: 2

  .. grid-item-card:: Installation

    * :doc:`rocPRIM prerequisites <install/rocPRIM-prerequisites>`
    * :doc:`rocPRIM installation overview <install/rocPRIM-install-overview>`
    * :doc:`Install rocPRIM on Linux <install/rocPRIM-build-install-linux>`
    * :doc:`Install rocPRIM on Windows <install/rocPRIM-build-install-windows>`
  
.. grid:: 2

  .. grid-item-card:: Conceptual

    * :doc:`Scope of rocPRIM operations <./conceptual/rocPRIM-scope>`
    * :doc:`rocPRIM operations <./conceptual/rocPRIM-operations>`
    * :doc:`rocPRIM performance tuning <./conceptual/rocPRIM-performance-tuning>`
    * :doc:`Block and stripe arrangements <./conceptual/rocPRIM-stripe-block>`

  .. grid-item-card:: How-to

    * :doc:`Use the SPIR-V target with rocPRIM <./how-to/rocPRIM-spir-v>`
   
  .. grid-item-card:: Reference

    * :ref:`glossary`
    * :ref:`data-type-support`
    * :ref:`types`
    * :ref:`type_traits`
    * :ref:`iterators` 
    * :ref:`intrinsics` 
    * :ref:`dev-index`
    * :ref:`block-index`
    * :ref:`warp-index` 
    * :ref:`thread-index`
    * :ref:`developer`


To contribute to the documentation refer to `Contributing to ROCm  <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

You can find licensing information on the `Licensing <https://rocm.docs.amd.com/en/latest/about/license.html>`_ page.



