.. meta::
  :description: Known limitations and unsupported facilities in hipThreads
  :keywords: hipThreads, ROCm, limitations, unsupported, deadlock, constraints

.. _limitations:

******************************************
Limitations
******************************************

hipThreads presents a standard-library-like interface, but GPU hardware imposes constraints that have no counterpart on the CPU.
The rules recorded here aren't compile-time errors.
Ignoring them causes deadlocks, crashes, or undefined behavior at run time.
For background on why these apply, see the :ref:`execution model <execution-model>`.

No synchronous HIP calls while threads are alive
================================================

Creating a ``hip::wthread`` launches a persistent scheduler kernel that stays resident until the last wthread is destroyed.
Synchronous HIP calls such as ``hipDeviceSynchronize()``, synchronous ``hipMemcpy()``, or ``thrust::copy()`` wait for all GPU work, including the scheduler, and therefore deadlock.

- Use asynchronous APIs, such as ``hipMemcpyAsync()`` and ``hipMemsetAsync()``, instead.
- Or wrap your ``hip::wthread`` objects in a scoped block ``{ ... }`` so you join them and tear down the scheduler before any synchronous call.

Callables must be ``__device__`` extended lambdas
=================================================

A ``hip::wthread`` constructed on the host can't accept host function pointers or ordinary host lambdas.

- Pass extended lambdas annotated with ``__device__``.
- Host code can't reference a ``__device__`` function directly.
  To run one, wrap it in a ``[] __device__ { ... }`` lambda.

Restrictions on the callable and its arguments
==============================================

The constraints on the callable and its arguments depend on where the ``hip::wthread`` is constructed.

Constructing on the host
------------------------

hipThreads transfers the callable and arguments to the device by a bitwise ``memcpy()`` copy, not by invoking a copy constructor.
They must therefore be ``TriviallyCopyable``.
A type with a non-trivial copy constructor would be copied bitwise without its constructor ever running, leaving the device-side object in an invalid state.
In addition:

- Don't pass standard containers such as ``std::vector``.
  Most aren't usable in device code to begin with because they provide no ``__device__`` member functions.
  Even if you make one compile, for example via relaxed ``constexpr`` support, it still won't work.
  Its internal data is in host memory, and the bitwise copy transfers only the container's pointers and size, not the elements they refer to.
  The device then holds pointers into host memory, which fails for the reason given in the next item.
- A raw pointer argument must point to GPU-accessible memory allocated with ``hipMalloc()`` or similar.
  Copying a host pointer to the device succeeds, but dereferencing a non-pinned host pointer from device code crashes.

Constructing on the device
--------------------------

No host-to-device transfer happens.
The work node is allocated and constructed directly in device memory, so hipThreads relaxes the ``TriviallyCopyable`` requirement.
The callable and arguments must instead be move constructible and trivially destructible.
hipThreads move-constructs the callable into the work node rather than copying it bitwise, so a non-trivial move constructor runs normally.
A non-trivial copy constructor never runs on this path.
The trivial-destructor requirement remains because ``hip::wthread`` erases the callable's concrete type and can't invoke a non-trivial destructor when the work completes.
In addition:

- GPU threads have private stacks.
  Never capture by reference, ``[&]``, a variable on the launching thread's stack.
  Shared data must be in heap or global memory.
- Don't capture a pointer or reference to shared memory, LDS, or ``__shared__`` memory.
  Shared memory is private to a single block or workgroup, but wthreads may run in different blocks, so such an address is meaningless to another thread.
  Shared data must be in global device memory, for example memory from ``hipMalloc()``.

No true blocking or preemption
==============================

GPU synchronization primitives emulate their CPU counterparts rather than reproducing them exactly.

- ``hip::condition_variable_any::wait()`` and ``hip::spin_condition_variable::wait()`` spin rather than blocking the hardware.
- ``hip::pseudo_condition_variable::wait()`` spins and periodically calls ``hip::this_thread::pseudo_yield()``.
- ``hip::this_thread::pseudo_yield()`` returns control to the caller only after the :term:`yieldee` finishes.
  The yieldee isn't interrupted and can't yield back to the caller.

Concurrency is bounded by ``hardware_concurrency()``
====================================================

The persistent scheduler runs a fixed number of execution slots ("vcores"), equal to ``hip::wthread::hardware_concurrency()``.
Each slot runs one work item to completion before pulling the next ready item from the queue.
The number of slots can be tuned; see :ref:`how to tune scheduler concurrency <tune-scheduler-concurrency>`.

If you spawn more wthreads than there are slots, the excess threads do not run immediately.
They sit in the work queue and only start once a running wthread finishes and frees its slot.
This has two consequences:

- ``hip::wthread`` construction isn't a guarantee that the work has started. It only means the work has been queued.
- Designs that assume all spawned threads make progress simultaneously can deadlock.
  If every slot is occupied by a wthread that is blocked waiting on a not-yet-started wthread (for example on a ``mutex`` or ``condition_variable``), the waited-for wthread can never be scheduled.
  Keep the number of mutually dependent threads at or below ``hip::wthread::hardware_concurrency()``, or structure the work so that queued threads aren't prerequisites for running ones.

Note that ``hip::this_thread::pseudo_yield()`` runs another ready work item nested inside the caller, so a yielding wthread can let queued work progress. It still holds its own slot until it returns (see :ref:`the execution model <execution-model>`).

Yield-loops deadlock
====================

``hip::this_thread::pseudo_yield()`` doesn't suspend the caller and place it back on a ready queue the way a preemptive scheduler would.
Instead it runs an additional ready work item nested inside the current one, and the caller can't resume until that yieldee runs to completion.
When no work item is ready, it runs nothing and returns immediately, so a spin-and-yield loop degenerates into plain busy-waiting.
This creates a strict caller-waits-for-yieldee dependency.

A yield-loop is any cycle in that dependency, and it deadlocks unconditionally:

- Thread A holds a ``pseudo_mutex`` or ``pseudo_condition_variable`` and yields to thread B.
  B then tries to acquire the same mutex.
  B spins and yields waiting for the lock, but A can't resume to release it until B completes.
  B can't complete until A releases the lock.
- More generally, A yields to B and B, directly or transitively, waits on anything that only A can produce.

Follow these guidelines to avoid yield-loops:

- Don't call a blocking or spinning primitive from a yieldee on a resource held by one of its transitive callers.
- Prefer ``hip::spin_mutex`` or ``hip::spin_condition_variable`` over the ``pseudo_`` variants when a yield-loop is possible.
  The spinning variants don't yield, so they can't form this cycle.
  They busy-wait instead.
  Use a ``pseudo_`` primitive only when you can guarantee no yield-loop occurs.

Static storage duration is unsupported
======================================

Defining a ``hip::wthread`` with static storage duration is undefined behavior.

Creating the first ``hip::wthread`` automatically launches the persistent scheduler kernel, and destroying the last one tears it down (see the :ref:`execution model <execution-model>`).
A ``hip::wthread`` at static storage duration breaks this lifecycle: its constructor would run during static initialization, before ``main`` and before the HIP runtime is guaranteed to be ready to launch the scheduler, and its destructor would run during static teardown, after ``main`` when the runtime may already have been shut down.
In both cases the automatic launch and teardown cannot run correctly.
Give every ``hip::wthread`` automatic (block-scope) or dynamic storage duration instead.

A single function can't create threads from both host and device
=================================================================

Constructing a ``hip::wthread`` inside a ``__host__ __device__`` function is currently unsupported when that function can be called from the host.

Unsupported example:

.. code-block:: cpp

   __host__ __device__ void f()
   {
       hip::wthread t(1, [] __device__() {});
       t.join();
   }

   int main()
   {
       f();
   }

This pattern may fail at runtime with an error similar to:

.. code-block:: text

   Cannot find Symbol ... getWrapperFn ...

The failure happens because HIP compiles ``__host__ __device__`` functions in separate host and device compilation passes.
The host pass uses the host-side hipThreads implementation, which launches a wrapper kernel for the work node.
The device pass uses the device-side hipThreads implementation, which queues work from within the GPU and doesn't instantiate the same host-launch wrapper kernel.

Therefore, the host code may try to launch a generated wrapper kernel symbol that isn't present in the device image.

Applications should use separate host-only and device-only functions instead:

.. code-block:: cpp

   struct Work
   {
       __device__ void operator()() const {}
   };

   __host__ void f_host()
   {
       hip::wthread t(1, Work{});
       t.join();
   }

   __device__ void f_device()
   {
       hip::wthread t(1, Work{});
       t.join();
   }

Avoid placing the ``hip::wthread`` construction itself in a shared ``__host__ __device__`` function.

Unsupported standard library facilities
=======================================

This release doesn't provide the following standard threading facilities:

- ``std::recursive_mutex``, ``std::timed_mutex``, ``std::recursive_timed_mutex``, and the timed locking operations, such as ``try_lock_for()`` and ``try_lock_until()``.
- ``std::shared_mutex`` and ``std::shared_lock``, and reader-writer locking.
- ``std::scoped_lock``.
  Use ``hip::lock_guard`` or ``hip::lock()`` for multi-mutex cases instead.
- ``std::condition_variable`` bound specifically to ``std::unique_lock<mutex>``.
  Use ``hip::condition_variable_any`` instead.
  ``std::notify_all_at_thread_exit`` is also unsupported.
- Timed condition-variable waits, such as ``wait_for()`` and ``wait_until()``, and ``hip::unique_lock::try_lock_for()`` and ``try_lock_until()``.
- ``std::this_thread::sleep_until()``.
- ``std::jthread``, ``std::stop_token``, and cooperative cancellation.
- ``std::future``, ``std::promise``, ``std::packaged_task``, and ``std::async``.
- ``std::latch``, ``std::barrier``, and ``std::semaphore``.
- ``thread_local`` storage and exception propagation across threads.
