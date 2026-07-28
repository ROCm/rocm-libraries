.. meta::
  :description: The hipThreads execution model
  :keywords: hipThreads, ROCm, scheduler, persistent kernel, fiber, width, yield

.. _execution-model:

**********************************
hipThreads execution model
**********************************

hipThreads exposes C++ Standard Library threading types with the same constructor and member-function patterns as ``std::thread`` and its synchronization types.

hipThreads is different than ``std::thread`` in the way threads are launched. Unlike ``std::threads`` where operating-system threads are scheduled by the OS, ``hip::wthread`` enqueues its constructor as a :term:`work item` on one long-lived :term:`persistent scheduler kernel`.

In a traditional HIP program, each unit of parallel work is a separate kernel launch with its own grid and block configuration.
Workloads that spawn and join many short-lived tasks pay launch overhead on every iteration. hipThreads uses a submit-to-scheduler model instead. A single persistent kernel runs for the lifetime of the application's ``hip::wthread`` usage.
Each ``hip::wthread`` construction enqueues a :term:`work item`, and waits for that item to finish without tearing down the kernel.

The kernel loops, polling work queues and running submitted work items, until the last handle is destroyed.

A loop that creates and joins several ``hip::wthread`` objects will hold the default ``hip::wthread`` in scope to keep the scheduler alive across iterations.

The persistent scheduler kernel keeps running until the last host-side ``hip::wthread`` handle is destroyed.

``hipDeviceSynchronize()`` and synchronous ``hipMemcpy()`` block until all GPU work on the device finishes. Because the scheduler can't exit while any handle remains, those calls block indefinitely. See :ref:`Limitations <limitations>` for information on how to avoid this deadlock.

The scheduler dispatches work onto a fixed pool of :term:`vcores<vcore>`, with one block of the :term:`persistent scheduler kernel` per vcore.

``hip::wthread::hardware_concurrency()`` returns the total vcore count across all :term:`WGPs<workgroup processor>` on the device.
The per-WGP vcore count defaults to 16 and is :ref:`tunable <tune-scheduler-concurrency>`.

Logical threads are scheduled cooperatively. ``hip::this_thread::pseudo_yield()`` runs an additional ready work item nested inside the current one and only resumes the caller once the :term:`yieldee` finishes.

If no work item is ready, ``pseudo_yield()`` returns immediately and the caller continues. There's no preemption and no hardware blocking, so synchronization primitives such as ``hip::condition_variable_any`` spin and yield rather than block.

A single ``hip::wthread`` can run as multiple :term:`fibers<fiber>`, one per hardware lane. The ``width`` constructor parameter sets the fiber count up to a fixed compile-time maximum.

The work item runs on each active lane, which supports cooperative SIMD-style work partitioning within one wthread.
