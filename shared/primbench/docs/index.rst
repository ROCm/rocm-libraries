.. meta::
   :description: Primbench is a single-header C++ benchmarking library for HIP GPU workloads. It provides an API for writing GPU microbenchmarks with built-in noise reduction, GPU warming and cooling, cache clearing, correctness assertions, and detailed JSON and CSV output.
   :keywords: Primbench, AMD, ROCm, GPU

*************************
Primbench documentation
*************************

Primbench is a single-header C++ benchmarking library for HIP GPU workloads. It provides an API for writing GPU microbenchmarks with built-in noise reduction, GPU warming and cooling, cache clearing, correctness assertions, and detailed JSON and CSV output.

The Primbench public repository is at `ROCm/rocm-libraries <https://github.com/ROCm/rocm-libraries/tree/develop/shared/primbench>`_.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Install

      * :doc:`Install Primbench <install/install>`

   .. grid-item-card:: Conceptual

      * :doc:`Noise reduction in Primbench <conceptual/primbench-noise-reduction>`

   .. grid-item-card:: How to

      * :doc:`Configure benchmark settings <how-to/configure-settings>`
      * :doc:`Validate benchmark output <how-to/validate-output>`
      * :doc:`Use Primbench <how-to/use-primbench>`

   .. grid-item-card:: Reference

      * :doc:`Primbench API <reference/primbench-api>`
      * :doc:`Command-line options <reference/cli-options>`
      * :doc:`JSON output format <reference/primbench-json-output>`
      * :doc:`CSV output <reference/primbench-csv-output>`

To contribute to the documentation, refer to
`Contributing to ROCm <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

Licensing information can be found on the :doc:`License <license>` page.
