.. meta::
  :description: hipThreads glossary of terms
  :keywords: hipThreads, glossary, ROCm, AMD, fiber, vcore, work item, yieldee

.. _hipthreads-glossary:

******************************************
hipThreads glossary
******************************************

.. glossary::
    :sorted:

    fiber
        A fiber is one :term:`lane` of the SIMD :term:`wave` that runs a ``hip::wthread`` callable.
        A ``hip::wthread`` runs as one fiber per active lane, and its ``width`` constructor parameter sets that count.
        Unlike a userspace fiber on the CPU, a fiber isn't scheduled independently and has no stack of its own.
        All fibers of a ``hip::wthread`` share one :term:`work item` and one instruction stream.

    lane
        A lane is one element of a SIMD :term:`wave`, and it runs a single :term:`fiber` of a ``hip::wthread``.

    wave
        A wave is a group of :term:`lanes<lane>` that run the same instruction stream.
        hipThreads uses "wave" and "warp" interchangeably.
        ``hip::wthread::max_width()`` returns 32, which matches the wave size on wave32 architectures.

    warp
        See :term:`wave`.

    persistent scheduler kernel
        The persistent scheduler kernel is the long-running GPU kernel that runs every ``hip::wthread`` callable.
        It launches onto a dedicated stream when the first ``hip::wthread`` handle is constructed, then loops, polling work queues and dispatching ready :term:`work items<work item>` onto :term:`vcores<vcore>`.
        It keeps running until the last host-side handle is destroyed, so it counts as outstanding GPU work for the whole lifetime of the program's threads.

    vcore
        A vcore is one execution slot on the :term:`persistent scheduler kernel`, and it corresponds to one block of that kernel.
        Each vcore runs one :term:`work item` to completion before pulling the next ready item from the queue.
        ``hip::wthread::hardware_concurrency()`` returns the vcore count, computed as the device multiprocessor count multiplied by the vcores launched per :term:`WGP`.
        The per-WGP figure defaults to 16 and is tunable through ``HIPTHREADS_VCORES_PER_WGP``.

    slot
        See :term:`vcore`. hipThreads uses "slot" and "vcore" interchangeably for a scheduler execution slot.

    workgroup processor
        A workgroup processor, or WGP, is the hardware unit that hipThreads launches scheduler :term:`vcores<vcore>` onto.
        hipThreads derives the count from the ``hipDeviceAttributeMultiprocessorCount`` device attribute at run time.

    WGP
        See :term:`workgroup processor`.

    work item
        A work item is one unit of queued work on the :term:`persistent scheduler kernel`, holding the callable that a ``hip::wthread`` was constructed with.
        Constructing a ``hip::wthread`` enqueues a work item, and the scheduler dispatches it onto a :term:`vcore` once one is free.
        A work item is backed internally by a :term:`work node`.

        In hipThreads a work item is a scheduled unit of work, not a single :term:`lane`.
        This differs from the wider ROCm and OpenCL meaning, where a work-item is the smallest unit of parallel execution and is equivalent to an NVIDIA thread.

    work node
        A work node is the internal structure that backs a :term:`work item`, declared as ``WorkNode_Header`` and ``WorkNode<Callable_t>``.
        It stores the type-erased callable, the thread width and base thread id, and the scheduler's queue linkage.
        Work nodes are an implementation detail rather than public API.

    cooperative multitasking
        Cooperative multitasking means that a running :term:`work item` holds its :term:`vcore` until it finishes or calls ``hip::this_thread::pseudo_yield()``.
        The scheduler can't preempt a work item, and no primitive blocks the hardware, so ``hip::condition_variable_any`` and the other synchronization types spin rather than block.

    pseudo_yield
        ``hip::this_thread::pseudo_yield()`` runs an additional ready :term:`work item` nested inside the calling one.
        The caller resumes only after the :term:`yieldee` finishes, and it holds its own :term:`vcore` throughout.
        If no work item is ready, ``pseudo_yield()`` returns immediately and the caller continues.

    yieldee
        The yieldee is the :term:`work item` that ``hip::this_thread::pseudo_yield()`` runs nested inside the caller.
        A yieldee isn't interrupted and can't yield back to its caller, which makes any cycle in the caller-waits-for-yieldee dependency deadlock unconditionally.

    yield-loop
        A yield-loop is a cycle in the caller-waits-for-yieldee dependency, such as thread A yielding to thread B while B waits on a lock that A holds.
        Because the caller can't resume until the :term:`yieldee` finishes, a yield-loop deadlocks unconditionally.
        For the cases that produce one and the rules for avoiding them, see :ref:`limitations <limitations>`.

    pseudo_mutex
        ``hip::pseudo_mutex`` is an exclusive, non-recursive mutex that spins on ``atomicCAS`` and calls ``hip::this_thread::pseudo_yield()`` every ``0x10000`` failed attempts.
        It tracks ownership by block id, so re-acquiring it from a block that already holds it livelocks and asserts in debug builds.
        Only one :term:`fiber` may be active while acquiring the lock, because extra fibers spin on a lock already held by a fiber in the same wave.
        Prefer ``hip::spin_mutex`` when a :term:`yield-loop` is possible, because it doesn't yield.
