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
