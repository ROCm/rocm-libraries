.. meta::
  :description: rocSOLVER documentation and API reference library
  :keywords: rocSOLVER, ROCm, API, documentation

.. _memory:

*******************************
rocSOLVER Memory Model
*******************************

Almost all LAPACK and rocSOLVER routines require workspace memory in order to compute their results. In contrast to LAPACK, however, pointers to the workspace are not explicitly passed to rocSOLVER functions as arguments; instead, they are managed behind-the-scenes using a configurable device memory model.

rocSOLVER makes use of and is integrated with :doc:`the rocBLAS memory model <rocblas:reference/memory-alloc>`. Workspace memory, and the scheme used to manage it, is tracked on a per- ``rocblas_handle`` basis. The same functionality that is used to manipulate rocBLAS's workspace memory will also affect rocSOLVER's workspace memory. 
You can also refer to the rocBLAS :ref:`rocblas:Device Memory allocation in detail` documentation.

There are four schemes for device memory management:

* Automatic (managed by rocSOLVER/rocBLAS): The default scheme. Device memory persists between function
  calls and will be automatically reallocated if more memory is required by a function.
* User-managed (preallocated): The desired workspace size is specified by the user as an environment variable before handle creation, and cannot be altered after the handle is created.
* User-managed (manual): The desired workspace size can be manipulated using rocBLAS helper functions.
* User-owned: The user manually allocates device memory and calls a rocBLAS helper function to use it
  as the workspace.

Automatic workspace
================================================

By default, rocSOLVER will automatically allocate device memory to be used as internal workspace
using the rocBLAS memory model, and will increase the amount of allocated memory as needed by rocSOLVER functions. If this scheme is in use, the function ``rocblas_is_managing_device_memory`` will return
``true``. In order to re-enable this scheme if it is not in use, a ``nullptr`` or zero size can be passed to the helper functions ``rocblas_set_device_memory_size`` or ``rocblas_set_workspace``. For more details on these rocBLAS APIs, see :doc:`Device Memory Allocation Functions in rocBLAS <rocblas:reference/memory-alloc>`.

This scheme has the disadvantage that automatic reallocation is synchronizing, and the user cannot control when this synchronization happens.

User-managed workspace
================================================

Alternatively, the user can manually specify an amount of memory to be allocated by rocSOLVER/rocBLAS. This allows the user to control when and if memory is reallocated and synchronization occurs. However, function calls will fail if there is not enough allocated memory.

Minimum required size
------------------------------

In order to choose an appropriate amount of memory to allocate, rocSOLVER can be queried to determine the minimum amount of memory required for functions to complete. The query can be started by calling ``rocblas_start_device_memory_size_query``, followed by calls to the desired functions with appropriate problem sizes (a null pointer can be passed to the device pointer arguments). A final call to ``rocblas_stop_device_memory_size_query`` will return the minimum required size.

For example, the following code snippet will return the memory size required to solve a 1024*1024 linear system with 1 right-hand side (involving calls to ``getrf`` and ``getrs``):

.. code-block:: cpp

    size_t memory_size;
    rocblas_start_device_memory_size_query(handle);
    rocsolver_dgetrf(handle, 1024, 1024, nullptr, lda, nullptr, nullptr);
    rocsolver_dgetrs(handle, rocblas_operation_none, 1024, 1, nullptr, lda, nullptr, nullptr, ldb);
    rocblas_stop_device_memory_size_query(handle, &memory_size);

For more details on the rocBLAS APIs, see :doc:`Device Memory Allocation Functions in rocBLAS <rocblas:reference/memory-alloc>`.

Using an environment variable
------------------------------

The desired workspace size can be provided before creation of the ``rocblas_handle`` by setting the value of environment variable ``ROCBLAS_DEVICE_MEMORY_SIZE``. If this variable is unset or the value is == 0, then it will be ignored. Note that a workspace size set in this way cannot be changed once the handle has been created.

Using helper functions
------------------------------

Another way to set the desired workspace size is by using the helper function ``rocblas_set_device_memory_size``.
This function is called after handle creation and can be called multiple times; however, it is
recommended to first synchronize the handle stream if a rocSOLVER or rocBLAS routine has already been called. For example:

.. code-block:: cpp

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);
    hipStreamSynchronize(stream);

    rocblas_set_device_memory_size(handle, memory_size);

For more details on the rocBLAS APIs, see :doc:`Device Memory Allocation Functions in rocBLAS <rocblas:reference/memory-alloc>`.

User-owned workspace
================================================

Finally, the user may opt to manage the workspace memory manually using HIP. By calling the function ``rocblas_set_workspace``, the user may pass a pointer to device memory to rocBLAS that will be used as the workspace for rocSOLVER. For example:

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
