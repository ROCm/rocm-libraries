.. meta::
  :description: hipThreads scheduler configuration reference
  :keywords: hipThreads, scheduler, vcores, WGP, HIPTHREADS_VCORES_PER_WGP, hardware_concurrency, ROCm

.. _scheduler-configuration:

*********************************
Scheduler configuration reference
*********************************

The scheduler exposes two configuration points for ``vcoresPerWgp``, the number of :term:`vcores<vcore>` launched per :term:`WGP`.
``hip::wthread::hardware_concurrency()`` reports the total vcore pool size.

.. code-block:: text

   hardware_concurrency() = multiprocessorCount * vcoresPerWgp

``multiprocessorCount`` is the device WGP count, queried at run time.
``vcoresPerWgp`` defaults to 16 when unset.

Runtime environment variable
============================

``HIPTHREADS_VCORES_PER_WGP`` overrides the compiled-in default at process start.
See :doc:`hipThreads environment variables <hipThreads-env-vars>` for accepted values, defaults, and parsing rules.

For step-by-step tuning, see :ref:`how to tune scheduler concurrency <tune-scheduler-concurrency>`.

Build-time CMake option
=======================

``HIPTHREADS_DEFAULT_VCORES_PER_WGP``
  Sets the compiled-in default when building hipThreads from source.
  Pass ``-DHIPTHREADS_DEFAULT_VCORES_PER_WGP=vcores_per_wgp`` to the CMake configure step.
  When omitted, the default of 16 in ``src/hip/thread.cxx`` applies.

For build instructions, see :doc:`Build from source <../install/build-from-source>`.
