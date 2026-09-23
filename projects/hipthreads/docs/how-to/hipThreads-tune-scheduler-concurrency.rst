.. meta::
  :description: Tuning the number of hipThreads scheduler vcores per WGP
  :keywords: hipThreads, ROCm, vcores, concurrency, hardware_concurrency, performance, tuning, WGP

.. _tune-scheduler-concurrency:

**************************************
Tuning scheduler concurrency
**************************************

hipThreads launches a fixed number of execution slots, called virtual cores (vcores), on each workgroup processor (WGP). The default number of vcores per WGP is 16. You can change this value to accommodate different workloads and architectures.

The total vcore count is calculated by multiplying the number of WGPs on the device (``multiprocessorCount``) by the number of vcores per WGP:

.. code-block:: text

   hardware_concurrency() = multiprocessorCount * vcoresPerWgp

Increasing the number of vcores per WGP raises the number of work items that can run concurrently, which can
improve throughput.

.. note::

   Setting the number of vcores per WGP too high can reduce performance. Monitor your throughput when tuning this setting. See :ref:`limitations <limitations>` for more information.

You can change the number of vcores per WGP at compile time with the ``HIPTHREADS_DEFAULT_VCORES_PER_WGP`` CMake option, or at runtime with the ``HIPTHREADS_VCORES_PER_WGP`` environment variable.

For example, to set the default number of vcores per WGP to 20:

.. code-block:: bash

   cmake -B build -DHIPTHREADS_DEFAULT_VCORES_PER_WGP=20
   cmake --build ./build

To set the number of vcores per WGP at runtime without changing the default value:

.. code-block:: bash

   HIPTHREADS_VCORES_PER_WGP=20

``HIPTHREADS_VCORES_PER_WGP`` must be set before launching the application to take effect. ``HIPTHREADS_VCORES_PER_WGP`` always overrides the compile-time value.
