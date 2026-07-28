.. meta::
  :description: Tuning the number of hipThreads scheduler vcores per WGP
  :keywords: hipThreads, ROCm, vcores, concurrency, hardware_concurrency, performance, tuning, WGP

.. _tune-scheduler-concurrency:

**************************************
How to tune scheduler concurrency
**************************************

hipThreads runs GPU-side work through a :term:`persistent scheduler kernel`.
The scheduler launches a fixed number of execution slots ("vcores").

Follow these instructions when the default vcore count doesn't match your GPU or workload and you want to tune concurrency without changing application code.

``hip::wthread::hardware_concurrency()`` reports the vcore count:

.. code-block:: text

   hardware_concurrency() = multiprocessorCount * vcoresPerWgp

``multiprocessorCount`` is the number of :term:`workgroup processors<workgroup processor>`, or WGPs, on the device, queried at run time, and ``vcoresPerWgp`` is the number of scheduler vcores launched per WGP. By default, ``vcoresPerWgp`` is set to 16.

Increasing ``vcoresPerWgp`` raises the number of :term:`work items<work item>` that can run concurrently, which can improve throughput.

``vcoresPerWgp`` can be set either at compile time when building hipThreads from source  or at run time with the ``HIPTHREADS_VCORES_PER_WGP`` environment variable. For example, to set it to 20:

.. code:: shell

   HIPTHREADS_VCORES_PER_WGP=20

``HIPTHREADS_VCORES_PER_WGP`` must be set a positive integer without a trailing whitespace.

..note::

   Setting ``HIPTHREADS_VCORES_PER_WGP=0`` will not change the vcoresPerGwp value.

Use the ``-DHIPTHREADS_DEFAULT_VCORES_PER_WGP`` CMake option to set ``vcoresPerWgp`` at compile time. For example, to set ``vcoresPerWgp`` to 20:

.. code:: shell

   cmake -B build -DHIPTHREADS_DEFAULT_VCORES_PER_WGP=20
   cmake --build ./build

.. note::

   Setting ``HIPTHREADS_VCORES_PER_WGP`` will override the value set at compile time.
