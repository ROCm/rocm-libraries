.. meta::
  :description: A guide to memory allocation in rocBLAS
  :keywords: rocBLAS, ROCm, API, Linear Algebra, documentation, memory

.. _memory-alloc:
.. _Device Memory Allocation Usage:

********************************************************************
Device memory allocation in rocBLAS
********************************************************************
rocBLAS uses per-handle device memory allocation to manage temporary memory efficiently. Each handle maintains its own memory and executes kernels sequentially in a single stream, allowing memory reuse across kernels.

There are two memory allocation schemes:

*  **rocBLAS_managed(default)**:  By default, rocBLAS internally manages memory, allocating more if needed. Allocated memory persists with the handle for reuse.

*  **user_owned**: Users allocate memory and provide it to rocBLAS via ``rocblas_set_workspace``.

``rocBLAS_managed`` is the default scheme. This scheme uses ``hipMallocAsync`` and ``hipFreeAsync`` (stream-order allocation) to allocate and free memory in stream order, avoiding global synchronization. This enables seamless stream switching without needing ``hipStreamSynchronize()``.

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
|L3 GEMM-based functions                         | Block of matrix                                      |
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

Memory allocation functions
==============================================

rocBLAS includes functions for manually setting the memory size and determining the memory requirements.

Function for setting a user-owned workspace
----------------------------------------------

* ``rocblas_set_workspace``

Functions for determining memory requirements
----------------------------------------------

*  ``rocblas_start_device_memory_size_query``
*  ``rocblas_stop_device_memory_size_query``
*  ``rocblas_is_managing_device_memory``

See the API section for information about these functions.

rocBLAS function return values for insufficient device memory
=============================================================

If the user manually allocates (user-owned scheme) using ``rocblas_set_workspace(rocblas_handle handle, void* addr, size_t size)``, that size is used as the limit and no resizing or synchronizing ever occurs.
The following two function return values indicate insufficient memory:

*  ``rocblas_status == rocblas_status_memory_error`` : indicates there is insufficient device memory for a rocBLAS function.
*  ``rocblas_status == rocblas_status_perf_degraded`` : indicates that a slower algorithm was used because of insufficient device memory for the optimal algorithm.

Switching streams without synchronization
----------------------------------------------

Stream-order memory allocation lets the application switch streams without having to call ``hipStreamSynchronize()``.
