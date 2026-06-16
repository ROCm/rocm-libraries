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
  :gutter: 3

  .. grid-item-card:: Install

    * :doc:`Install rocPRIM <install/install>`
    * :doc:`Build from source <install/build>`

  .. grid-item-card:: Conceptual

    * :doc:`Block and stripe arrangements <./conceptual/rocPRIM-stripe-block>`
    * :doc:`Types of rocPRIM operations <./conceptual/rocPRIM-operations>`
    * :doc:`Scope of rocPRIM operations <./conceptual/rocPRIM-scope>`
    * :doc:`rocPRIM performance tuning <./conceptual/rocPRIM-performance-tuning>`

  .. grid-item-card:: How to

    * :doc:`Use the SPIR-V target with rocPRIM <./how-to/rocPRIM-spir-v>`
    * :doc:`Use primbench for benchmarking <./how-to/rocPRIM-primbench>`
    * :doc:`Run unit tests on multiple GPUs <./how-to/rocPRIM-run-tests>`

   
  .. grid-item-card:: Reference

    * :ref:`glossary`
    * :ref:`data-type-support`
    * :ref:`reference`

      * :ref:`Iterators <iterators>` 
      * :ref:`Intrinsics <intrinsics>` 
      * :ref:`Utility types <types>`
      * :ref:`Custom type traits <type_traits>`
      * :ref:`Data movement functions <data_mov_funcs>`
      * :ref:`Device-level operations <dev-index>`
      * :ref:`Block-level operations <block-index>`
      * :ref:`Warp-level operations <warp-index>` 
      * :ref:`Thread-level operations <thread-index>`

To contribute to the documentation refer to `Contributing to ROCm  <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

You can find licensing information on the `Licensing <https://rocm.docs.amd.com/en/latest/about/license.html>`_ page.



