.. meta::
  :description: Mapping from C++ standard threading primitives to hipThreads equivalents
  :keywords: hipThreads, ROCm, std, hip, mapping, thread, mutex, condition_variable

.. _std-to-hip-mapping:

******************************************
Mapping ``std::`` to ``hip::``
******************************************

hipThreads mirrors the C++ Concurrency Support Library.
In most cases, porting CPU code means replacing the ``std::`` qualifier with ``hip::`` and adding ``__device__`` or ``__host__ __device__`` to the callables that run on the GPU.
The tables below list the supported primitives and how they correspond to the standard library.

For behavioral differences that have no standard-library analogue, see :ref:`limitations`.

Threads
=======

.. list-table::
  :header-rows: 1
  :widths: 40 40 20

  * - C++ standard
    - hipThreads
    - Header
  * - ``std::thread``
    - ``hip::wthread``
    - ``<hip/thread>``
  * - ``std::thread::id``
    - ``hip::wthread::id``
    - ``<hip/thread>``
  * - ``std::this_thread::get_id``
    - ``hip::this_thread::get_id``
    - ``<hip/thread>``
  * - ``std::this_thread::yield``
    - ``hip::this_thread::pseudo_yield``
    - ``<hip/thread>``
  * - ``std::this_thread::sleep_for``
    - ``hip::this_thread::sleep_for``
    - ``<hip/thread>``
  * - *(no equivalent)*
    - ``hip::this_thread::get_fiber_id``
    - ``<hip/thread>``

Mutexes and locks
=================

.. list-table::
  :header-rows: 1
  :widths: 40 40 20

  * - C++ standard
    - hipThreads
    - Header
  * - ``std::mutex``
    - ``hip::spin_mutex`` (also ``hip::pseudo_mutex``)
    - ``<hip/mutex>``, ``<hip/pseudo_mutex>``
  * - ``std::lock_guard``
    - ``hip::lock_guard``
    - ``<hip/mutex>``
  * - ``std::unique_lock``
    - ``hip::unique_lock``
    - ``<hip/mutex>``

Condition variables
===================

.. list-table::
  :header-rows: 1
  :widths: 40 40 20

  * - C++ standard
    - hipThreads
    - Header
  * - ``std::condition_variable_any``
    - ``hip::condition_variable_any`` (also
      ``hip::spin_condition_variable``, ``hip::pseudo_condition_variable``)
    - ``<hip/condition_variable>``, ``<hip/pseudo_condition_variable>``

.. note::

  The ``spin_`` variants busy-wait.
  The ``pseudo_`` variants busy-wait but periodically call ``pseudo_yield`` to let other GPU work progress.
  Choose a ``pseudo_`` primitive only when you can guarantee no yield-loops occur; see :ref:`limitations`.
