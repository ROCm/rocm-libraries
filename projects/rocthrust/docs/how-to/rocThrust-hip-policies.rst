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

You can test the ``hip_rocprim::par_nosync`` and ``hip_rocprim::par`` policies using the following code:

.. code:: cpp

  #include <hip/hip_runtime_api.h>
  #include <thrust/host_vector.h> 
  #include <thrust/device_vector.h>
  #include <thrust/random.h>
  #include <thrust/count.h>
  #include <thrust/reduce.h>
  #include <thrust/system/hip/execution_policy.h>
  #include <ctime>
  #include <iostream>

  int main(int argc, char* argv[])
  {
    // Allocate host and device vectors.
    const size_t size = 100;
    thrust::host_vector<int> h_vec(size);
    thrust::device_vector<int> d_vec1(size);
    thrust::device_vector<int> d_vec2(size);

    // Fill host vector with random values.
    const int limit = 100;
    auto seed = std::time(nullptr);
    thrust::default_random_engine rng(seed);
    for (int i = 0; i < size; i++)
        h_vec[i] = rng() % limit;

    // Copy data to device vectors.
    d_vec1 = h_vec;
    d_vec2 = h_vec;

    // Launch some algorithms using the hip_rocprim::par policy.
    // The calls below are blocking with respect to the host.
    // However, internally, each algorithm will run in parallel.
    auto par_policy = thrust::hip_rocprim::par;
    int count = thrust::count(par_policy, d_vec1.begin(), d_vec1.end(), 50);
    int reduction = thrust::reduce(par_policy, d_vec2.begin(), d_vec2.end());

    // Print out the results.
    std::cout << "par results:" << std::endl;
    std::cout << "count: " << count << std::endl;
    std::cout << "reduction: " << reduction << std::endl;

    // Launch the algorithms using the hip_rocprim::par_nosync policy.
    // These calls may not be blocking with respect to the host.
    auto nosync_policy = thrust::hip_rocprim::par_nosync;
    int count2 = thrust::count(nosync_policy, d_vec1.begin(), d_vec1.end(), 50);
    int reduction2 = thrust::reduce(nosync_policy, d_vec2.begin(), d_vec2.end());

    // We can perform other host-side work here, and it may overlap with the
    // algorithms launched above.
    DoHostSideWork();

    // We must synchronize before accessing the results on the host.
    hipDeviceSynchronize();

    // Print out the results.
    std::cout << "par_nosync results:" << std::endl;
    std::cout << "count: " << count2 << std::endl;
    std::cout << "reduction: " << reduction2 << std::endl;

    return 0;
  } 
