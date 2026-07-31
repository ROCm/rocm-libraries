.. meta::
  :description: The hipThreads execution model
  :keywords: hipThreads, ROCm, scheduler, persistent kernel, fiber, width, yield

.. _execution-model:

**********************************
hipThreads execution model
**********************************

hipThreads looks like the C++ standard threading library, but it runs on a GPU.
Understanding the execution model explains both why the API behaves the way it does and why the :ref:`limitations <limitations>` exist.

Persistent scheduler kernel
===========================

Creating the first ``hip::wthread`` launches a long-lived *scheduler kernel* onto a dedicated stream.
This kernel loops, polling work queues and running submitted callables, until the last ``hip::wthread`` is destroyed.
A ``hip::wthread`` is therefore not a kernel launch of its own.
It is a work item submitted to an already-running scheduler, which keeps per-thread launch overhead low.

Because the scheduler kernel stays resident while any thread is alive, any host call that waits for *all* GPU work to finish (such as ``hipDeviceSynchronize`` or a synchronous ``hipMemcpy``) will wait for the scheduler too, and deadlock.
See :ref:`limitations <limitations>` for how to avoid this.

Cooperative multitasking, no preemption
=======================================

Logical threads are scheduled cooperatively.
``hip::this_thread::pseudo_yield`` runs another ready work item nested inside the current one and only resumes the caller once the yieldee finishes.
There is no preemption and no hardware blocking, so synchronization primitives such as ``condition_variable`` spin and yield rather than block.

Fibers and width
================

A single ``hip::wthread`` can run as multiple fibers (one per hardware lane) by setting its ``width`` parameter, up to ``wthread::max_width()`` (currently the warp size, 32).
The callable runs on each active lane, which enables cooperative, SIMD-style work partitioning within one wthread.
``hip::this_thread::get_fiber_id`` returns the current lane index.
