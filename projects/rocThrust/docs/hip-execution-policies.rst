:orphan:

.. meta::
    :description: rocThrust documentation and API reference
    :keywords: rocThrust, ROCm, API, reference, execution policy

.. _hip-execution-policies:

******************************************
Execution Policies
******************************************

In addition to the standard Thrust execution policies (eg. ``thrust::host``, ``thrust::device``, ``thrust::seq``),
rocThrust's HIP backend provides the following:

* ``hip_rocprim::par`` - This policy causes algorithms to be launched in a parallel configuration.
  API calls are blocking (synchronous with respect to the host).

* ``hip_rocprim::par_nosync`` - This policy tells Thrust that algorithms may avoid synchronization
  barriers when it is possible to do so. As a result, algorithms may be launched asynchronously with
  respect to the host. This can allow you to perform other host-side work while the algorithms
  are running on the device. If you use this policy, you must synchronize before accessing results
  on the host side.

The example below illustrates the behaviour of these two policies.

.. code-block:: cpp

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