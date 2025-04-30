.. meta::
  :description: rocSOLVER documentation and API reference library
  :keywords: rocSOLVER, ROCm, API, documentation

.. _memory:

*******************************
rocSOLVER memory model
*******************************

Almost all LAPACK and rocSOLVER routines require workspace memory in order to compute their results. In contrast to LAPACK,
however, pointers to the workspace are not explicitly passed to rocSOLVER functions as arguments; instead, they are
managed behind-the-scenes using a configurable device memory model.

rocSOLVER makes use of and is integrated with :doc:`the rocBLAS memory model <rocblas:reference/memory-alloc>`. Workspace memory,
and the scheme used to manage it, is tracked on a per- ``rocblas_handle`` basis. The same functionality that is used to
manipulate rocBLAS's workspace memory will also affect rocSOLVER's workspace memory.
You can also refer to the rocBLAS :ref:`rocblas:Device Memory allocation in detail` documentation.

There are two schemes for device memory management:

* Automatic (managed by rocSOLVER/rocBLAS): The default scheme. Device memory persists between function
  calls and will be automatically reallocated if more memory is required by a function.
* User-owned: The user manually allocates device memory and calls a rocBLAS helper function to use it
  as the workspace.

Automatic workspace
================================================

By default, rocSOLVER will automatically allocate device memory to be used as internal workspace
using the rocBLAS memory model, and will increase the amount of allocated memory as needed by rocSOLVER functions.
If this scheme is in use, the function ``rocblas_is_managing_device_memory`` will return
``true``. In order to re-enable this scheme if it is not in use, a ``nullptr`` or zero size can be passed to the
helper function ``rocblas_set_workspace``. For more details on these rocBLAS APIs, see :doc:`Device Memory Allocation Functions in rocBLAS <rocblas:reference/memory-alloc>`.

This scheme has the disadvantage that automatic reallocation is synchronizing, and the user cannot control when this synchronization happens.

User-owned workspace
================================================

Alternatively, the user can opt to manage the workspace memory manually using HIP. This involves querying rocSOLVER to determine the minimum amount of memory required,
allocating the memory with ``hipMalloc``, and then passing the resulting pointer to rocBLAS.

Minimum required size
------------------------------

In order to choose an appropriate amount of memory to allocate, rocSOLVER can be queried to determine the minimum amount of memory required for functions to complete.
The query can be started by calling ``rocblas_start_device_memory_size_query``, followed by calls to the desired functions with appropriate problem sizes (a null pointer
can be passed to the device pointer arguments). A final call to ``rocblas_stop_device_memory_size_query`` will return the minimum required size.

For example, the following code snippet will return the memory size required to solve a 1024*1024 linear system with 1 right-hand side (involving calls to ``getrf`` and ``getrs``):

.. code-block:: cpp

    size_t memory_size;
    rocblas_start_device_memory_size_query(handle);
    rocsolver_dgetrf(handle, 1024, 1024, nullptr, lda, nullptr, nullptr);
    rocsolver_dgetrs(handle, rocblas_operation_none, 1024, 1, nullptr, lda, nullptr, nullptr, ldb);
    rocblas_stop_device_memory_size_query(handle, &memory_size);

For more details on the rocBLAS APIs, see :doc:`Device Memory Allocation Functions in rocBLAS <rocblas:reference/memory-alloc>`.

Setting the workspace
------------------------------

By calling the function ``rocblas_set_workspace``, the user can pass a pointer to device memory to rocBLAS that will be used as the workspace for rocSOLVER. For example:

.. code-block:: cpp

    void* device_memory;
    hipMalloc(&device_memory, memory_size);
    rocblas_set_workspace(handle, device_memory, memory_size);

    // perform computations here
    rocblas_set_workspace(handle, nullptr, 0);
    hipFree(device_memory);

For more details on the rocBLAS APIs, see :doc:`Device Memory Allocation Functions in rocBLAS <rocblas:reference/memory-alloc>`.

.. _the rocBLAS memory model: https://rocm.docs.amd.com/projects/rocBLAS/en/latest/API_Reference_Guide.html#device-memory-allocation-in-rocblas
.. _Device Memory Allocation Functions in rocBLAS: https://rocm.docs.amd.com/projects/rocBLAS/en/latest/API_Reference_Guide.html#device-memory-allocation-in-rocblas
