.. meta::
  :description: rocThrust documentation and API reference
  :keywords: rocThrust, ROCm, API, reference, execution policy

.. _hip-execution-policies:

*********************************************************
rocThrust parallel non-deterministic execution policies
*********************************************************

rocThrust provides two parallel non-deterministic execution policies:

* ``hip_rocprim::par``: Algorithms run in parallel on the device. The host blocks on each algorithm running on the GPU, waiting for each to finish before launching the next.

* ``hip_rocprim::par_nosync``: Algorithms run in parallel on the device. The host doesn't block on the algorithms running on the GPU and can perform other work while waiting for the GPU to finish running the algorithms. The host and device must be explicitly synchronized if results are left on the device.

For example, when using the ``hip_rocprim::par`` policy, ``thrust::count`` and ``thrust::reduce`` are both blocking with respect to the host, and their results on the host are available without any explicit synchronization:

.. code:: cpp

  auto par_policy = thrust::hip_rocprim::par;
  int count = thrust::count(par_policy, d_vec1.begin(), d_vec1.end(), 50);
  int reduction = thrust::reduce(par_policy, d_vec2.begin(), d_vec2.end());

  std::cout << "par results:" << std::endl;
  std::cout << "count: " << count << std::endl;
  std::cout << "reduction: " << reduction << std::endl;

When using the ``hip_rocprim::par_nosync`` policy, ``thrust::count`` and ``thrust::reduce`` are asynchronous with respect to the host. The host can do other work while the algorithms are running on the device. ``hipDeviceSynchronize()`` should be called to ensure synchronization before accessing results:

.. code:: cpp

  auto nosync_policy = thrust::hip_rocprim::par_nosync;
  int count2 = thrust::count(nosync_policy, d_vec1.begin(), d_vec1.end(), 50);
  int reduction2 = thrust::reduce(nosync_policy, d_vec2.begin(), d_vec2.end());
  
  DoHostSideWork();

  hipDeviceSynchronize();

  std::cout << "par_nosync results:" << std::endl;
  std::cout << "count: " << count2 << std::endl;
  std::cout << "reduction: " << reduction2 << std::endl;

You can test out these two policies using the following code:

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
