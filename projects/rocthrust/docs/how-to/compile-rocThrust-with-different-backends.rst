.. meta::
  :description: How to compile rocThrust with different backends
  :keywords: rocThrust, ROCm, cmake, CUDA, TBB, OpenMP, CPP

*******************************************
How to compile rocThrust with different backends
*******************************************

We will be using this example file ``binary_search_example.cpp`` to demonstrate how to compile for
different rocThrust backends. This program will use ``thrust::binary_search`` to search for the range -10, 10
within a source vector and prints the result.

.. code:: c++
  #include <iostream>
  #include <numeric>
  #include <thrust/binary_search.h>
  #include <thrust/device_vector.h>
  #include <thrust/host_vector.h>



  template <class T> inline void print_arr(T &arr) {
    for (const auto &x : arr)
      std::cout << x << " ";
    std::cout << std::endl;
  }

  int main() {
    thrust::host_vector<int> h_src(11);
    std::iota(h_src.begin(), h_src.end(), -5);

    thrust::host_vector<int> h_search(21);
    std::iota(h_search.begin(), h_search.end(), -10);

    thrust::device_vector<int> d_src = h_src;
    thrust::device_vector<int> d_search = h_search;
    thrust::device_vector<bool> d_output(h_search.size());

    thrust::binary_search(d_src.begin(), d_src.end(), d_search.begin(),
                          d_search.end(), d_output.begin());

    thrust::host_vector<bool> h_output = d_output;

    std::cout << "Source Numbers:\n";
    print_arr(h_src);

    std::cout << "Numbers to search:\n";
    print_arr(h_search);

    std::cout << "Output:\n";
    print_arr(h_output);
    return 0;
  }

Host Side
=========
rocThrust supports ``CPP (default)``, ``OpenMP`` and ``TBB``. 

To compile for CPP:
.. code::
  hipcc ./binary_search_example.cpp -DTHRUST_HOST_SYSTEM=CPP -D__THRUST_HOST_SYSTEM_NAMESPACE=cpp

To compile for OpenMP:
.. code::
  hipcc -fopenmp ./binary_search_example.cpp -DTHRUST_HOST_SYSTEM=OMP -D__THRUST_HOST_SYSTEM_NAMESPACE=omp -D_OPENMP=202011

To compile for TBB:
.. code::
  hipcc ./binary_search_example.cpp -DTHRUST_HOST_SYSTEM=TBB -D__THRUST_HOST_SYSTEM_NAMESPACE=tbb -ltbb

To compile using cmake, you can use the same ``THRUST_HOST_SYSTEM`` and ``__THRUST_HOST_SYSTEM_NAMESPACE`` flags like so:
.. code::
  CXX=hipcc cmake -DTHRUST_HOST_SYSTEM=CPP -D__THRUST_HOST_SYSTEM_NAMESPACE=cpp ..

Device Side
===========
rocThrust supports ````

