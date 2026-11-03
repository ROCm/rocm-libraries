.. meta::
  :description: Tuning the number of hipThreads scheduler vcores per WGP
  :keywords: hipThreads, ROCm, vcores, concurrency, hardware_concurrency, performance, tuning, WGP

.. _tune-scheduler-concurrency:

**************************************
Tuning scheduler concurrency
**************************************

hipThreads launches a fixed number of execution slots, called virtual cores (vcores), on each workgroup processor (WGP). The default number of vcores per WGP is 16. You can change this value to accommodate different workloads and architectures.

The total vcore count, returned by ``hip::wthread::hardware_concurrency()``, is the number of WGPs on the current device multiplied by the number of vcores per WGP, capped at the number of vcores the device can run at once:

.. code-block:: text

   hardware_concurrency() = min(WGPs * vcoresPerWgp, maximum resident vcores)

* On Instinct (CDNA) GPUs, which have no WGPs, every two compute units count as one WGP, so the same setting gives the same density on every architecture.
* The maximum resident vcores is the number of scheduler blocks the device can hold at the same time, as reported by ``hipOccupancyMaxActiveBlocksPerMultiprocessor``.
  Vcores beyond it could not start until others exit, and ``hardware_concurrency()`` must not promise more simultaneous wthreads than can actually run (see :ref:`limitations <limitations>`).
  When the cap applies, hipThreads prints a message such as ``[hipthreads] device 0: requested 3072 vcores, but the occupancy-based maximum is 2112 ...; using 2112.``

Increasing the number of vcores per WGP raises the number of work items that can run concurrently, which might improve throughput.
Requesting more than the device can hold has the same effect as requesting the maximum.

.. note::

   Setting the number of vcores per WGP too high can reduce performance. Monitor your throughput when tuning this setting. See :ref:`limitations <limitations>` for more information.

You can change the number of vcores per WGP at compile time with the ``HIPTHREADS_DEFAULT_VCORES_PER_WGP`` CMake option, or at runtime with the ``HIPTHREADS_VCORES_PER_WGP`` environment variable.

For example, to set the default number of vcores per WGP to 20:

.. code:: shell

   cmake -B build -DHIPTHREADS_DEFAULT_VCORES_PER_WGP=20
   cmake --build ./build

To set the number of vcores per WGP at runtime without changing the default value:

.. code:: shell

   export HIPTHREADS_VCORES_PER_WGP=20

Prefixing the application command also works: ``HIPTHREADS_VCORES_PER_WGP=20 ./application``.

``HIPTHREADS_VCORES_PER_WGP`` must be set before launching the application to take effect. ``HIPTHREADS_VCORES_PER_WGP`` always overrides the compile-time value.
