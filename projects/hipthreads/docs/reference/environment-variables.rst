.. meta::
   :description: hipThreads environment variables
   :keywords: hipThreads, environment variables, ROCm, AMD, vcores, concurrency

.. _environment-variables:

******************************************
hipThreads environment variables
******************************************

The following environment variables affect the hipThreads runtime.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Environment variable
     - Values
   * - ``HIPTHREADS_VCORES_PER_WGP``
     - | Sets the number of scheduler virtual cores (vcores) launched per workgroup processor (WGP).
       | Must be a positive integer greater than zero. Non-numeric, zero, or empty values are treated as invalid and ignored.
       | Default: ``16``, or the value of ``-DHIPTHREADS_DEFAULT_VCORES_PER_WGP`` when hipThreads is built from source.
