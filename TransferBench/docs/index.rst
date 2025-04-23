.. meta::
  :description: TransferBench is a utility to benchmark simultaneous transfers between user-specified devices (CPUs or GPUs)
  :keywords: Benchmarking utility, Memory transfers, Device transfers

****************************
TransferBench documentation
****************************

TransferBench is a utility to benchmark simultaneous transfers between user-specified devices (CPUs or GPUs). A transfer is a single operation where an executor reads and adds values from source (SRC) memory locations, then writes the sum to destination (DST) memory locations.
This simplifies to a simple copy operation when dealing with a single SRC or DST.

The code is open and hosted at `<https://github.com/ROCm/TransferBench>`_.

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Install

    * :ref:`install-transferbench`

  .. grid-item-card:: API reference

    * :ref:`transferbench-api`

  .. grid-item-card:: How to

    * :ref:`using-transferbench`

To contribute to the documentation, refer to
`Contributing to ROCm <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

You can find licensing information on the
`Licensing <https://rocm.docs.amd.com/en/latest/about/license.html>`_ page.
