.. meta::
  :description: The hipThreads execution model
  :keywords: hipThreads, ROCm, scheduler, persistent kernel, fiber, width, yield

.. _execution-model:

**********************************
hipThreads execution model
**********************************

hipThreads provides a threads library that runs on AMD GPUs. This threads library is different from typical GPU programming models.

In a typical GPU programming model, each unit of parallel work is expressed as a kernel launch. For workloads that create many short-lived parallel tasks, such as iterative algorithms that spawn and join threads in a loop, kernel launch overhead is incurred on every iteration.

hipthreads replaces this pattern with a submit-to-scheduler model:

- A single persistent kernel runs for the lifetime of the application's ``hip::wthread`` usage.
- Each ``hip::wthread`` construction submits a work item to the scheduler, which dispatches it to an available virtual core (vcore).
- Joining a ``hip::wthread`` waits for that work item to complete without tearing down the kernel.

The execution model for hipThreads relies on the persistent scheduling kernel. When the first ``hip::wthread`` object is created, a scheduler kernel is launched on a dedicated stream.

Each subsequent ``hip::wthread`` object is submitted to this persistent scheduler kernel as a work item. The scheduler loops, polling work items in the queue, and running their workload.

The scheduler kernel persists until the last ``hip::wthread`` object is destroyed.

The scheduler manages a fixed grid of vcores. Each workgroup processor (WGP) on the GPU hosts a configurable number of vcores.

The logical thread executes across multiple single instruction, multiple data (SIMD) lanes called fibers within a single GPU wavefront. All fibers in a thread execute in lockstep.

A single ``hip::wthread`` can run as multiple fibers, with one fiber per hardware lane. The workload runs on each active lane, which enables cooperative, SIMD-style work partitioning within one ``hip::wthread``.

Logical threads are scheduled cooperatively. ``hip::this_thread::pseudo_yield`` will run the next work item nested inside the current one. Once the nested work item completes, the original workload can continue. There is no preemption or hardware blocking in this model, and synchronization primitives such as ``condition_variable`` spin and yield rather than block.

.. note::

  Because the scheduler kernel persists as long as any ``hip::wthread`` object exists, any call that waits for GPU work to finish will also wait for scheduler to finish. As a result, calls such as ``hipDeviceSynchronize`` or a synchronous ``hipMemcpy`` will deadlock.

Stack space for callables
=========================

The submit-to-scheduler model also determines how much stack a ``hip::wthread`` callable has available, which is a design consideration when deciding what a callable keeps in local variables.

For an ordinary HIP kernel, the compiler sees every function the kernel calls, adds up the local variables along the deepest call path, and records the total in the kernel. The runtime then allocates exactly that much stack per thread. The scheduler kernel is compiled once, ahead of time, and runs whichever callable is submitted to it later, so it invokes the callable indirectly through a function pointer. Because the callable is not known at that call site when the kernel is compiled, there is no frame size to compute, and the runtime instead applies the HIP stack limit ``hipLimitStackSize`` to the whole call chain. This limit defaults to 1024 bytes per thread.

That budget covers every local variable in the callable and in anything it calls. Objects with inline storage dominate it, and what matters is the storage class rather than the container type: a ``hip::std::inplace_vector`` and a plain array of the same element count occupy the same stack space. A callable that spawns child threads and keeps their handles in a local array is a common case, since ``sizeof(hip::wthread)`` is 16 bytes on the device, so roughly 63 handles fit in the default budget, independently of ``hip::wthread::hardware_concurrency()``.

There are two ways to design around this. The first is to keep large objects off the stack by allocating them on the device heap, so the callable's own stack usage stays constant no matter how much data it manages. The device heap is sized by ``hipLimitMallocHeapSize``, which defaults to 8 MB.

.. code-block:: cpp

   hip::wthread coordinator(1, [=] __device__ {
       // On the device heap: the callable's stack holds only a pointer.
       hip::wthread* workers = new hip::wthread[count];
       for (int i = 0; i < count; ++i)
           workers[i] = hip::wthread(1, [=] __device__ { /* ... */ });
       for (int i = 0; i < count; ++i)
           workers[i].join();
       delete[] workers;
   });

The second is to raise ``hipLimitStackSize``. Set it before constructing the first ``hip::wthread``, because that construction is what launches the scheduler kernel, and the limit is applied at launch. Size the value against actual usage: the budget applies per thread, so raising it increases the scratch memory reserved for the scheduler. Raising it to 2 KB, for example, only moves the handle ceiling above from roughly 63 to roughly 127.

.. code-block:: cpp

   hipDeviceSetLimit(hipLimitStackSize, 128 * 1024);

   // Any hip::wthread created from here on runs with the larger budget.

.. note::

  Exceeding the budget is undefined behavior with no dedicated diagnostic, and the symptom depends on how much scratch memory the runtime reserved, which varies with occupancy and GPU architecture. It can appear as an abort with ``HSA_STATUS_ERROR_EXCEPTION`` (surfacing as ``an illegal memory access was encountered`` or ``unspecified launch failure``), or as a workload that runs correctly for a while and then hangs or produces wrong results after corrupting another thread's scratch memory. The latter is easy to mistake for a scheduling problem, but it is unrelated to scheduler capacity: a workload can exhaust its stack budget while running far below ``hardware_concurrency()``. Re-running with a substantially larger ``hipLimitStackSize`` confirms which one it is.
