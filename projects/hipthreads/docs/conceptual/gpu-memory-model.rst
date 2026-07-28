.. meta::
   :description: Explains the GPU memory model for hipthreads, covering separate host and device memory pools, device-side allocation, data transfer, TriviallyCopyable constraints, and a placement-new workaround.
   :keywords: hipthreads, GPU memory, device memory, host memory, hip::wthread, TriviallyCopyable, thrust, hipMemcpy, ROCm, AMD GPU
x

*******************************
GPU memory model for hipthreads
*******************************

The C++ Standard Library and hipthreads expose similar threading interfaces but operate on different memory domains. ``std::thread`` runs callables on host CPUs against host memory. Ordinary pointers, standard containers, and heap allocations on the host remain valid for the lifetime of the thread. ``hip::wthread`` runs callables as GPU device code inside a :term:`persistent scheduler kernel`.
Device code reads and writes device memory only.
Host pointers aren't valid inside ``__device__`` lambdas, and device pointers aren't valid on the host.
When you construct a ``hip::wthread`` on the host, hipThreads copies arguments from host memory into device-accessible storage with a bitwise copy, so those arguments must be ``TriviallyCopyable``.
Unlike ``std::thread``, hipThreads device callables can't invoke synchronous HIP APIs without risking deadlock while the scheduler is alive.

Separate host and device memory
*******************************

Host memory is allocated through standard C++ mechanisms such as ``new``, ``std::make_unique``, and ``std::vector``. Device memory occupies a separate physical address space. A ``hip::wthread`` lambda executes on the GPU and cannot dereference host pointers. Host code cannot dereference device pointers.

Data that device threads read or write must reside in device memory before those threads start. Results intended for the host require an explicit copy back after the threads complete.

.. code-block:: text

   Host memory -- thrust::copy / hipMemcpy --> Device memory
   Device memory -- hipMemcpy --> Host memory
   Device memory -- hip::wthread lambdas --> GPU execution

Device-side allocation with ``hip::std::make_unique``
*****************************************************

Device code allocates dynamic memory through ``hip::std::make_unique`` from the libhipcxx header ``<hip/std/memory>``. The facility mirrors ``std::make_unique`` but draws from the GPU heap:

.. code-block:: cpp

   RunState __device__ malloc_run_state(Config *p) {
     RunState s;
     s.x = hip::std::make_unique<float[]>(p->dim);
     s.xb = hip::std::make_unique<float[]>(p->dim);
     s.logits = hip::std::make_unique<float[]>(p->vocab_size);
     return s;
   }

The returned ``hip::std::unique_ptr`` releases the underlying device allocation when destroyed on the device. The pattern supports constructing complex data structures entirely in device memory without host round-trips.

Transferring data between host and device
*****************************************

Host-to-device movement typically uses one of two mechanisms.

Thrust
------

The thrust library exposes ``thrust::device_malloc``, ``thrust::copy_n``, and ``thrust::device_free`` for bulk transfers. The API suits large contiguous buffers, such as model weights, transferred before any ``hip::wthread`` object enters scope:

.. code-block:: cpp

   thrust::device_ptr<float> weights_ptr_dev = thrust::device_malloc<float>(num_floats);
   thrust::copy_n(reinterpret_cast<float*>(((char*)data) + sizeof(Config)),
                  num_floats, weights_ptr_dev);

.. warning::

   Thrust APIs issue synchronous HIP calls. Calls from inside a ``__device__`` lambda, or while a ``hip::wthread`` remains on the host call stack, can deadlock because synchronous HIP calls block inside the device-thread context.

hipMemcpy
---------

When thrust is unavailable, for example while a ``hip::wthread`` remains alive on the host stack, transfers proceed through ``hipMemcpyAsync`` on a HIP stream:

.. code-block:: cpp

   // Can't use thrust APIs here because there is still a hip::wthread on the stack.
   float *logits_device_ptr;
   hipMemcpyAsync(&logits_device_ptr, &transformer->state.logits_raw,
                  sizeof(float*), hipMemcpyDeviceToHost, g_stream);
   hipStreamSynchronize(g_stream);
   hipMemcpyAsync(logits.get(), logits_device_ptr,
                  vocab_size * sizeof(float), hipMemcpyDeviceToHost, g_stream);

Thrust fits setup and teardown phases, before or after all ``hip::wthread`` objects have joined. ``hipMemcpy`` and ``hipMemcpyAsync`` handle transfers that must occur while the persistent scheduler kernel remains active.

``TriviallyCopyable`` arguments on the host
*******************************************

When you construct a ``hip::wthread`` on the host, hipThreads copies lambda arguments from host memory into device memory with a bitwise copy.
Each argument must satisfy the C++ ``TriviallyCopyable`` requirement.
Accepted categories include:

- Raw pointers to device memory, as trivially copyable scalars.
- Scalar types such as ``int``, ``float``, and ``size_t``.
- Plain-old-data structs without virtual functions or non-trivial copy constructors.

Standard library containers such as ``std::vector``, ``std::string``, and ``std::unique_ptr`` are not trivially copyable. Call sites pass device pointers in their place.

A non-trivially-copyable argument yields undefined behavior.
The compiler doesn't currently emit a diagnostic for this constraint on the host path.

Constructing a ``hip::wthread`` on the device relaxes the ``TriviallyCopyable`` requirement.
The callable and arguments must instead be move constructible and trivially destructible.
For ``__device__`` annotation requirements, synchronous HIP call restrictions, and the full constraint list, see :ref:`limitations <limitations>`.

Placement-new workaround for device-side initialization
*******************************************************

Direct move-assignment into a struct that contains ``hip::std::unique_ptr`` fields on the device can interact incorrectly with uninitialized unique pointers that the compiler treats as already owning memory. The documented workaround constructs the struct in place with placement new rather than assigning into it:

.. code-block:: cpp

   hip::wthread(
     [] __device__ (Transformer *t, float *weights_ptr, int shared_weights) {
       t->weights = memory_map_weights(&t->config, weights_ptr, shared_weights);
       // Use placement-new so we don't try to assign to unique_ptrs
       // that think they already point at something
       new (&t->state) RunState(malloc_run_state(&t->config));
     }, transformer, weights_ptr, shared_weights
   ).join();

The CHANGELOG records this technique as a workaround for a ROCm 7.12 compiler issue. It applies to device-resident objects that manage memory through smart pointers.

Related resources
*****************

- :doc:`/reference/gpu-memory-utilities`, API reference for device-side allocation and copy functions
- :ref:`limitations <limitations>`, complete list of constraints for ``hip::wthread`` lambdas
- :doc:`/conceptual/porting-from-std-thread`, ``std::thread`` to ``hip::wthread`` API mapping and memory management differences
