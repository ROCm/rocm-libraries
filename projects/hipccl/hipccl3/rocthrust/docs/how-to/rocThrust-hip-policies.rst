.. meta::
  :description: rocThrust documentation and API reference
  :keywords: rocThrust, ROCm, API, reference, execution policy

.. _hip-execution-policies:

*********************************************************
Avoiding synchronization barriers
*********************************************************

The ``hip_rocprim::par_nosync`` execution policy provides a way to avoid synchronization barriers when running algorithms.

``hip_rocprim::par_nosync`` and ``hip_rocprim::par`` are both parallel non-deterministic policies. ``hip_rocprim::par`` is :ref:`synchronous and blocking with respect to the host <synchronization-and-blocking>`. Under ``hip_rocprim::par``, algorithms are launched in parallel on the device, but the host blocks on each algorithm. The next algorithm won't be launched until each algorithm finishes.

The ``hip_rocprim::par_nosync`` policy can be used to avoid this synchronization barrier. Synchronization can be skipped when possible under the ``hip_rocprim::par_nosync`` policy. Under this policy, the host has the possibility of not blocking on the algorithms running on the GPU. The CPU can then perform other work while waiting for the GPU to finish running the algorithms. The host and device should be explicitly synchronized before accessing results. 

.. note:: 

  rocThrust doesn't support `hipGraphs <https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/hipgraph.html>`_. The operations inside a hipGraph must be asynchronous and the rocThrust API is synchronous by default.  While there are asynchronous versions of the algorithms in the ``thrust::async`` namespace, these algorithms operate asynchronously by returning futures, which is different from the form of asynchronous execution required within hipGraphs. It is currently impossible to guarantee that synchronization doesn't occur within any rocThrust algorithm. 
