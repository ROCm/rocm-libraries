.. meta::
  :description: rocBLAS documentation and API reference library
  :keywords: rocBLAS, ROCm, API, Linear Algebra, documentation

.. _beta-features:

********************************************************************
rocBLAS beta features
********************************************************************

To allow for future growth and changes, the features in this section are not subject to the same
level of backwards compatibility and support as the normal rocBLAS API. These features are subject
to change and removal in future release of rocBLAS.

To use the following beta API features, define ``ROCBLAS_BETA_FEATURES_API`` before including ``rocblas.h``.

rocblas_gemm_ex_get_solutions + batched, strided_batched
=========================================================

.. doxygenfunction:: rocblas_gemm_ex_get_solutions
.. doxygenfunction:: rocblas_gemm_ex_get_solutions_by_type
.. doxygenfunction:: rocblas_gemm_batched_ex_get_solutions
.. doxygenfunction:: rocblas_gemm_batched_ex_get_solutions_by_type
.. doxygenfunction:: rocblas_gemm_strided_batched_ex_get_solutions

Graph support for rocBLAS
=========================================================

Most of the rocBLAS functions can be captured into a graph node via Graph Management HIP APIs,
except those listed in :ref:`Functions Unsupported with Graph Capture`.
For a list of graph related HIP APIs, see
`Graph Management HIP API <https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/group___graph.html#graph-management>`_.

The following code creates a graph with ``rocblas_function()`` as graph node.

.. code-block:: c++

      CHECK_HIP_ERROR((hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
      rocblas_<function>(<arguments>);
      CHECK_HIP_ERROR(hipStreamEndCapture(stream, &graph));

The captured graph can be launched as shown below:

.. code-block:: c++

      CHECK_HIP_ERROR(hipGraphInstantiate(&instance, graph, NULL, NULL, 0));
      CHECK_HIP_ERROR(hipGraphLaunch(instance, stream));

Graph support requires asynchronous HIP APIs, so users must use the ``rocBLAS_managed`` (default) memory allocation scheme which uses stream-order memory allocation.
For more details, see :ref:`Device Memory Allocation Usage`.

During stream capture, rocBLAS stores the allocated host and device memory in the handle.
The allocated memory is freed when the handle is destroyed.

.. _Functions Unsupported with Graph Capture:

Functions unsupported with Graph Capture
=========================================================

The following Level-1 functions place results into host buffers (in pointer mode host) which enforces synchronization.

*  ``dot``
*  ``asum``
*  ``nrm2``
*  ``imax``
*  ``imin``

BLAS Level-3 and BLAS-EX functions in pointer mode device do not support HIP Graph. Support will be added in a future release.

HIP Graph known issues in rocBLAS
=========================================================

On the Windows platform, batched functions (Level-1, Level-2, and Level-3) produce incorrect results.
