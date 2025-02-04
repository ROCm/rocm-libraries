.. meta::
  :description: A guide to memory allocation in rocBLAS
  :keywords: rocBLAS, ROCm, API, Linear Algebra, documentation, memory

.. _memory-alloc:
.. _Device Memory Allocation Usage:

********************************************************************
Device memory allocation in rocBLAS
********************************************************************

For temporary device memory, rocBLAS uses a per-handle memory allocation with out-of-band management.
The temporary device memory is stored in the handle. This allows for the recycling of temporary device memory
across multiple computational kernels that use the same handle. Each handle has a single stream, and
kernels execute in order in the stream, with each kernel completing before the next kernel in the
stream starts. There are four schemes for temporary device memory:

*  **rocBLAS_managed**: This is the default scheme. If there is not enough memory in the handle, computational functions allocate the memory
   they require. Any memory that is allocated persists in the handle, so it is available for later computational functions that use the handle.
*  **user_managed, preallocate**: An environment variable is set before the rocBLAS handle is created.
   Thereafter there are no more allocations or deallocations.
*  **user_managed, manual**:  The user calls helper functions to get or set the memory size throughout the program,
   thereby controlling when allocation and deallocation occur.
*  **user_owned**:  The user allocates workspace and calls a helper function to allow rocBLAS to access the workspace.

The default scheme has the disadvantage that allocation is synchronizing, so if there is not enough memory in the handle,
a synchronizing deallocation and allocation occur.

The following computational functions use temporary device memory.

+------------------------------------------------+------------------------------------------------------+
|Function                                        |Use of temporary device memory                        |
+================================================+======================================================+
|L1 reduction functions                          | Reduction array                                      |
|                                                |                                                      |
| - rocblas_Xasum                                |                                                      |
| - rocblas_Xasum_batched                        |                                                      |
| - rocblas_Xasum_strided_batched                |                                                      |
| - rocblas_Xdot                                 |                                                      |
| - rocblas_Xdot_batched                         |                                                      |
| - rocblas_Xdot_strided_batched                 |                                                      |
| - rocblas_Xmax                                 |                                                      |
| - rocblas_Xmax_batched                         |                                                      |
| - rocblas_Xmax_strided_batched                 |                                                      |
| - rocblas_Xmin                                 |                                                      |
| - rocblas_Xmin_batched                         |                                                      |
| - rocblas_Xmin_strided_batched                 |                                                      |
| - rocblas_Xnrm2                                |                                                      |
| - rocblas_Xnrm2_batched                        |                                                      |
| - rocblas_Xnrm2_strided_batched                |                                                      |
| - rocblas_dot_ex                               |                                                      |
| - rocblas_dot_batched_ex                       |                                                      |
| - rocblas_dot_strided_batched_ex               |                                                      |
| - rocblas_nrm2_ex                              |                                                      |
| - rocblas_nrm2_batched_ex                      |                                                      |
| - rocblas_nrm2_strided_batched_ex              |                                                      |
+------------------------------------------------+------------------------------------------------------+
|L2 functions                                    | Result array before overwriting input                |
|                                                |                                                      |
| - rocblas_Xgemv (optional)                     | Column reductions of skinny transposed matrices      |
| - rocblas_Xgemv_batched                        | applicable for ``gemv`` functions                    |
| - rocblas_Xgemv_strided_batched                |                                                      |
| - rocblas_Xtbmv                                |                                                      |
| - rocblas_Xtbmv_batched                        |                                                      |
| - rocblas_Xtbmv_strided_batched                |                                                      |
| - rocblas_Xtpmv                                |                                                      |
| - rocblas_Xtpmv_batched                        |                                                      |
| - rocblas_Xtpmv_strided_batched                |                                                      |
| - rocblas_Xtrmv                                |                                                      |
| - rocblas_Xtrmv_batched                        |                                                      |
| - rocblas_Xtrmv_strided_batched                |                                                      |
| - rocblas_Xtrsv                                |                                                      |
| - rocblas_Xtrsv_batched                        |                                                      |
| - rocblas_Xtrsv_strided_batched                |                                                      |
| - rocblas_Xhemv                                |                                                      |
| - rocblas_Xhemv_batched                        |                                                      |
| - rocblas_Xhemv_strided_batched                |                                                      |
| - rocblas_Xsymv                                |                                                      |
| - rocblas_Xsymv_batched                        |                                                      |
| - rocblas_Xsymv_strided_batched                |                                                      |
| - rocblas_Xtrsv_ex                             |                                                      |
| - rocblas_Xtrsv_batched_ex                     |                                                      |
| - rocblas_Xtrsv_strided_batched_ex             |                                                      |
+------------------------------------------------+------------------------------------------------------+
|L3 GEMM based functions                         | Block of matrix                                      |
|                                                |                                                      |
| - rocblas_Xtrsm                                |                                                      |
| - rocblas_Xtrsm_batched                        |                                                      |
| - rocblas_Xtrsm_strided_batched                |                                                      |
| - rocblas_Xsymm                                |                                                      |
| - rocblas_Xsymm_batched                        |                                                      |
| - rocblas_Xsymm_strided_batched                |                                                      |
| - rocblas_Xsyrk                                |                                                      |
| - rocblas_Xsyrk_batched                        |                                                      |
| - rocblas_Xsyrk_strided_batched                |                                                      |
| - rocblas_Xsyr2k                               |                                                      |
| - rocblas_Xsyr2k_batched                       |                                                      |
| - rocblas_Xsyr2k_strided_batched               |                                                      |
| - rocblas_Xsyrkx                               |                                                      |
| - rocblas_Xsyrkx_batched                       |                                                      |
| - rocblas_Xsyrkx_strided_batched               |                                                      |
| - rocblas_Xtrmm                                |                                                      |
| - rocblas_Xtrmm_batched                        |                                                      |
| - rocblas_Xtrmm_strided_batched                |                                                      |
| - rocblas_Xhemm                                |                                                      |
| - rocblas_Xhemm_batched                        |                                                      |
| - rocblas_Xhemm_strided_batched                |                                                      |
| - rocblas_Xherk                                |                                                      |
| - rocblas_Xherk_batched                        |                                                      |
| - rocblas_Xherk_strided_batched                |                                                      |
| - rocblas_Xher2k                               |                                                      |
| - rocblas_Xher2k_batched                       |                                                      |
| - rocblas_Xher2k_strided_batched               |                                                      |
| - rocblas_Xherkx                               |                                                      |
| - rocblas_Xherkx_batched                       |                                                      |
| - rocblas_Xherkx_strided_batched               |                                                      |
| - rocblas_Xgemm                                |                                                      |
| - rocblas_Xgemm_batched                        |                                                      |
| - rocblas_Xgemm_strided_batched                |                                                      |
| - rocblas_gemm_ex                              |                                                      |
| - rocblas_gemm_ex_batched                      |                                                      |
| - rocblas_gemm_ex_strided_batched              |                                                      |
| - rocblas_Xtrtri                               |                                                      |
| - rocblas_Xtrtri_batched                       |                                                      |
| - rocblas_Xtrtri_strided_batched               |                                                      |
+------------------------------------------------+------------------------------------------------------+

Environment variable for preallocating memory
=============================================

The environment variable ``ROCBLAS_DEVICE_MEMORY_SIZE`` is used to set how much memory to preallocate:

*  If it is greater than 0, it sets the default handle device memory size to the specified size (in bytes).
*  If it is equal to 0 or unset, it lets rocBLAS manage the device memory. It uses a default size, like 32MiB or 128MiB, and expands it when necessary.

Functions for manually setting the memory size
==============================================

*  ``rocblas_set_device_memory_size``
*  ``rocblas_get_device_memory_size``
*  ``rocblas_is_user_managing_device_memory``

Function for setting a user-owned workspace
===========================================

- ``rocblas_set_workspace``

Functions for determining memory requirements
==================================================

- ``rocblas_start_device_memory_size_query``
- ``rocblas_stop_device_memory_size_query``
- ``rocblas_is_managing_device_memory``

See the API section for information about these functions.

rocBLAS function return values for insufficient device memory
=============================================================

If the user preallocates or manually allocates, that size is used as the limit and no resizing or synchronizing ever occurs.
The following two function return values indicate insufficient memory:

*  ``rocblas_status == rocblas_status_memory_error`` : indicates there is insufficient device memory for a rocBLAS function.
*  ``rocblas_status == rocblas_status_perf_degraded`` : indicates that a slower algorithm was used because of insufficient device memory for the optimal algorithm.

.. _stream order alloc:

Stream-ordered memory allocation
========================================

Stream-ordered device memory allocation is added to rocBLAS. The asynchronous allocators
``hipMallocAsync()`` and ``hipFreeAsync()`` are used to allow allocation and free to happen in stream order.

This is a non-default beta option enabled by setting the environment variable ``ROCBLAS_STREAM_ORDER_ALLOC``.

You can check whether the device supports stream-order allocation by calling ``hipDeviceGetAttribute()`` with the
device attribute ``hipDeviceAttributeMemoryPoolsSupported``.

Enabling stream-ordered memory allocation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

On supported platforms, the environment variable ``ROCBLAS_STREAM_ORDER_ALLOC`` is used to enable stream-ordered memory allocation.

*  If it is greater than 0 (> 0), it sets the allocation to be stream-ordered and uses ``hipMallocAsync/hipFreeAsync`` to manage device memory.
*  If it is equal to zero (= 0) or unset, it uses ``hipMalloc`` and ``hipFree`` to manage device memory.

Switching streams without synchronization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Stream-order memory allocation allows switching of streams without the need to call ``hipStreamSynchronize()``.

