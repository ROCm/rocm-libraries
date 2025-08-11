
.. meta::
   :description: How to compile rocThrust with different backends
   :keywords: rocThrust, ROCm, cmake, CUDA, TBB, OpenMP, CPP

===========================================================
How to Compile rocThrust with Different Backends
===========================================================

This guide demonstrates how to compile ``rocThrust`` using various host and device backends. We'll use the example file ``binary_search_example.cpp``, which utilizes ``thrust::binary_search`` to search for values in the range [-10, 10] within a source vector and prints the results.

Example Code
============

.. code-block:: c++

   #include <iostream>
   #include <numeric>
   #include <thrust/binary_search.h>
   #include <thrust/device_vector.h>
   #include <thrust/host_vector.h>

   template <class T>
   inline void print_arr(T &arr) {
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

     std::cout << "Numbers to Search:\n";
     print_arr(h_search);

     std::cout << "Output:\n";
     print_arr(h_output);

     return 0;
   }

Host Side Compilation
=====================

rocThrust supports the following host backends:

- ``CPP`` (default)
- ``OpenMP``
- ``TBB``

To specify the host backend, set the CMake flags ``THRUST_HOST_SYSTEM`` and ``__THRUST_HOST_SYSTEM_NAMESPACE``.

Compile with hipcc
-------------------

**CPP:**

.. code-block:: bash

   hipcc binary_search_example.cpp -DTHRUST_HOST_SYSTEM=1 -D__THRUST_HOST_SYSTEM_NAMESPACE=cpp

**OpenMP:**

.. code-block:: bash

   hipcc -fopenmp binary_search_example.cpp -xc++ -DTHRUST_HOST_SYSTEM=2 -D__THRUST_HOST_SYSTEM_NAMESPACE=omp -D_OPENMP=202011

**TBB:**

.. code-block:: bash

   hipcc binary_search_example.cpp -DTHRUST_HOST_SYSTEM=3 -D__THRUST_HOST_SYSTEM_NAMESPACE=tbb -ltbb

Compile with CMake
-------------------

.. code-block:: bash

   CXX=hipcc cmake -DTHRUST_HOST_SYSTEM=CPP -D__THRUST_HOST_SYSTEM_NAMESPACE=cpp ..

Device Side Compilation
=======================

rocThrust supports the following device backends:

- ``CPP``
- ``CUDA`` (default)
- ``OpenMP``
- ``TBB``

To specify the device backend, set the CMake flags ``THRUST_DEVICE_SYSTEM`` and ``__THRUST_DEVICE_SYSTEM_NAMESPACE``.

Compile with hipcc
-------------------

**CPP:**

.. code-block:: bash

   hipcc binary_search_example.cpp -DTHRUST_DEVICE_SYSTEM=4 -D__THRUST_DEVICE_SYSTEM_NAMESPACE=cpp

**CUDA:**

.. code-block:: bash

   hipcc binary_search_example.cpp -DTHRUST_DEVICE_SYSTEM=1 -D__THRUST_DEVICE_SYSTEM_NAMESPACE=cuda

**OpenMP:**

.. code-block:: bash

   hipcc -fopenmp binary_search_example.cpp -xc++ -DTHRUST_DEVICE_SYSTEM=2 -D__THRUST_DEVICE_SYSTEM_NAMESPACE=omp -D_OPENMP=202011

**TBB:**

.. code-block:: bash

   hipcc binary_search_example.cpp -DTHRUST_DEVICE_SYSTEM=3 -D__THRUST_DEVICE_SYSTEM_NAMESPACE=tbb -ltbb



Compile with CMake
-------------------

.. code-block:: bash

   CXX=hipcc cmake -DTHRUST_DEVICE_SYSTEM=CPP -D__THRUST_DEVICE_SYSTEM_NAMESPACE=cpp ..


Compile for host and device backend at the same time
----------------------------------------------------
You can set both the host and device backend at the same time. For example compiling for TBB:

.. code-block:: bash
    hipcc binary_search_example.cpp \
    -DTHRUST_DEVICE_SYSTEM=3 \
    -D__THRUST_DEVICE_SYSTEM_NAMESPACE=tbb \
    -DTHRUST_HOST_SYSTEM=3 \
    -D__THRUST_HOST_SYSTEM_NAMESPACE=tbb \
    -ltbb 
