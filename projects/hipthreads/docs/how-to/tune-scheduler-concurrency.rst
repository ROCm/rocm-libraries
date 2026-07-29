.. meta::
  :description: Tuning the number of hipThreads scheduler vcores per WGP
  :keywords: hipThreads, ROCm, vcores, concurrency, hardware_concurrency, performance, tuning, WGP

.. _tune-scheduler-concurrency:

**************************************
How to tune scheduler concurrency
**************************************

The hipThreads scheduler launches a fixed number of execution slots ("vcores"), and
``hip::wthread::hardware_concurrency()`` reports that number.
It is computed as:

.. code-block:: text

   hardware_concurrency() = multiprocessorCount * vcoresPerWgp

where ``multiprocessorCount`` is the number of WGPs/compute units on the device (queried at runtime)
and ``vcoresPerWgp`` is the number of scheduler vcores launched per WGP.
Increasing ``vcoresPerWgp`` raises the number of work items that can run concurrently, which can
improve throughput; setting it too high wastes resources and can reduce performance.
For background on what these slots are and how work is scheduled onto them, see the
:ref:`execution model <execution-model>` and :ref:`limitations <limitations>`.

Default value
=============

``vcoresPerWgp`` defaults to **16**.
This value was chosen because it performs well across most of the bundled samples on Navi (Radeon)
cards.
The best value is architecture- and workload-dependent, so you can often extract more performance by
tuning it for your specific GPU and application.

Configure it at runtime
=======================

Set the ``HIPTHREADS_VCORES_PER_WGP`` environment variable before running your application to override
the number of vcores per WGP without rebuilding:

.. code-block:: bash

   HIPTHREADS_VCORES_PER_WGP=20 ./your_application

The value must be a positive integer.
If the variable is unset or cannot be parsed, hipThreads falls back to the compiled-in default.
This is the quickest way to experiment with different values and find the best setting for your GPU
and workload.

Configure the default at build time
===================================

If you build hipThreads from source, you can change the compiled-in default by passing
``-DHIPTHREADS_DEFAULT_VCORES_PER_WGP=<n>`` to the CMake configure step:

.. code-block:: bash

   cmake -B build -DHIPTHREADS_DEFAULT_VCORES_PER_WGP=20
   cmake --build ./build

The runtime ``HIPTHREADS_VCORES_PER_WGP`` environment variable still takes precedence over this
build-time default, so a value baked in at build time can always be overridden per run.
