.. meta::
  :description: Known limitations and unsupported facilities in hipThreads
  :keywords: hipThreads, ROCm, limitations, unsupported, deadlock, constraints

.. _limitations:

******************************************
Limitations
******************************************

hipThreads presents a standard-library-like interface, but GPU hardware imposes constraints that have no counterpart on the CPU.
The rules below are not compile-time errors; ignoring them causes deadlocks, crashes, or undefined behavior at run time.
For background on *why* these apply, see the :ref:`execution model <execution-model>`.

No synchronous HIP calls while threads are alive
================================================

Creating a ``hip::wthread`` launches a persistent scheduler kernel that stays resident until the last wthread is destroyed.
Synchronous HIP calls such as ``hipDeviceSynchronize``, synchronous ``hipMemcpy``, or ``thrust::copy`` wait for *all* GPU work, including the scheduler, and therefore deadlock.

* Use asynchronous APIs (``hipMemcpyAsync``, ``hipMemsetAsync``) instead.
* Or wrap your ``hip::wthread`` objects in a scoped block ``{ ... }`` so they are joined and the scheduler is torn down before any synchronous call.

Callables must be ``__device__`` extended lambdas
=================================================

A ``hip::wthread`` constructed on the host cannot accept host function pointers or ordinary host lambdas.

* Pass extended lambdas annotated with ``__device__``.
* Host code cannot reference a ``__device__`` function directly.
  To run one, wrap it in a ``[] __device__ { ... }`` lambda.

Restrictions on the callable and its arguments
==============================================

The constraints on the callable and its arguments depend on where the ``hip::wthread`` is constructed.

**Constructing on the host.**
The callable and arguments are transferred to the device by a bitwise copy (``memcpy``), not by invoking a copy constructor.
They must therefore be ``TriviallyCopyable``: a type with a non-trivial copy constructor would be copied bitwise without its constructor ever running, leaving the device-side object in an invalid state.
In addition:

* Do not pass standard containers such as ``std::vector``.
  Most are not usable in device code to begin with (they provide no ``__device__`` member functions), and even if you make one compile — for example via relaxed ``constexpr`` support — it still won't work:
  its internal data lives in host memory, and the bitwise copy transfers only the container's pointers/size, not the elements they refer to.
  The device then holds pointers into host memory (see the next point).
* A raw pointer argument must point to GPU-accessible memory (allocated with ``hipMalloc`` or similar).
  Passing a host pointer is fine; *dereferencing* a non-pinned host pointer from device code will crash.

**Constructing on the device.**
No host-to-device transfer happens — the work node is allocated and constructed directly in device memory — so the requirement is relaxed.
The callable and arguments only need to be **trivially destructible** (any non-trivial copy constructor still runs normally).
The trivial-destructor requirement remains because ``hip::wthread`` erases the callable's concrete type and cannot invoke a non-trivial destructor when the work completes.
In addition:

* GPU threads have private stacks.
  Never capture by reference (``[&]``) a variable that lives on the launching thread's stack; shared data must live in heap or global memory.
* Do not capture a pointer or reference to shared memory (LDS, ``__shared__``).
  Shared memory is private to a single block/workgroup, but ``hip::wthread``s may run in different blocks, so such an address is meaningless to another thread.
  Shared data must live in global device memory (for example, memory from ``hipMalloc``).

No true blocking or preemption
============================================

GPU synchronization primitives emulate their CPU counterparts rather than reproducing them exactly.

* **No blocking or preemption.**
  ``condition_variable::wait`` spins or yields rather than blocking the hardware.
* **Cooperative yield only.**
  ``hip::this_thread::pseudo_yield`` returns control to the caller only after the yieldee finishes.
  The yieldee is not interrupted and cannot yield back to the caller.

Concurrency is bounded by ``hardware_concurrency()``
====================================================

The persistent scheduler runs a fixed number of execution slots ("vcores"), equal to ``hip::wthread::hardware_concurrency()``.
Each slot runs one work item to completion before pulling the next ready item from the queue.
The number of slots can be tuned; see :ref:`how to tune scheduler concurrency <tune-scheduler-concurrency>`.

If you spawn more ``hip::wthread`` s than there are slots, the excess threads do not run immediately.
They sit in the work queue and only start once a running wthread finishes and frees its slot.
This has two consequences:

* ``hip::wthread`` construction is not a guarantee that the work has started — only that it has been queued.
* Designs that assume all spawned threads make progress simultaneously can deadlock.
  If every slot is occupied by a wthread that is blocked waiting on a not-yet-started wthread (for example on a ``mutex`` or ``condition_variable``), the waited-for wthread can never be scheduled.
  Keep the number of mutually dependent threads at or below ``hardware_concurrency()``, or structure the work so that queued threads are not prerequisites for running ones.

Note that ``hip::this_thread::pseudo_yield`` runs another ready work item nested inside the caller, so a yielding wthread can let queued work progress — but it still holds its own slot until it returns (see :ref:`the execution model <execution-model>`).

Yield-loops deadlock
====================

``hip::this_thread::pseudo_yield`` does not suspend the caller and place it back on a ready queue the way a preemptive scheduler would.
Instead it runs another ready work item *nested* inside the current one, and the caller cannot resume until that yieldee runs to completion.
This creates a strict caller-waits-for-yieldee dependency.

A **yield-loop** is any cycle in that dependency, and it deadlocks unconditionally:

* Thread A holds a ``pseudo_mutex`` (or ``pseudo_condition_variable``) and yields to thread B.
  B then tries to acquire the same mutex.
  B spins and yields waiting for the lock, but A cannot resume to release it until B completes — and B cannot complete until A releases the lock.
* More generally, A yields to B and B (directly or transitively) waits on anything that only A can produce.

Guidelines:

* Do not call a blocking/spinning primitive from a yieldee on a resource held by one of its (transitive) callers.
* Prefer ``hip::spin_mutex`` / ``hip::spin_condition_variable`` over the ``pseudo_`` variants when a yield-loop is possible.
  The spinning variants do not yield, so they cannot form this cycle (at the cost of busy-waiting).
  Use a ``pseudo_`` primitive only when you can guarantee no yield-loop occurs.

Static storage duration is unsupported
======================================

Defining a ``hip::wthread`` with static storage duration is undefined behavior.

Creating the first ``hip::wthread`` automatically launches the persistent scheduler kernel, and destroying the last one tears it down (see the :ref:`execution model <execution-model>`).
A ``hip::wthread`` at static storage duration breaks this lifecycle: its constructor would run during static initialization, before ``main`` and before the HIP runtime is guaranteed to be ready to launch the scheduler, and its destructor would run during static teardown, after ``main`` when the runtime may already have been shut down.
In both cases the automatic launch and teardown cannot run correctly.
Give every ``hip::wthread`` automatic (block-scope) or dynamic storage duration instead.

A single function cannot create threads from both host and device
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
The device pass uses the device-side hipThreads implementation, which queues work from within the GPU and does not instantiate the same host-launch wrapper kernel.

Therefore, the host code may try to launch a generated wrapper kernel symbol that is not present in the device image.

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

The following standard threading facilities are not provided in this release:

* ``std::recursive_mutex``, ``std::timed_mutex``, ``std::recursive_timed_mutex``, and the timed locking operations (``try_lock_for``, ``try_lock_until``).
* ``std::shared_mutex`` / ``std::shared_lock`` and reader-writer locking.
* ``std::scoped_lock`` and the variadic ``std::lock`` / ``std::try_lock`` helpers.
* ``std::condition_variable`` bound specifically to ``std::unique_lock<mutex>`` (use ``hip::condition_variable_any``), and ``std::notify_all_at_thread_exit``.
* ``std::jthread``, ``std::stop_token``, and cooperative cancellation.
* ``std::future``, ``std::promise``, ``std::packaged_task``, and ``std::async``.
* ``std::latch``, ``std::barrier``, ``std::semaphore``.
* ``thread_local`` storage and exception propagation across threads.
