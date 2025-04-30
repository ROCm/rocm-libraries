.. meta::
  :description: rocRAND documentation and API reference library
  :keywords: rocRAND, ROCm, API, documentation

.. _cpp-api:

===================
C/C++ API reference
===================

This chapter describes the rocRAND C and C++ API.

API index
=========

To search an API, refer to the API :ref:`genindex`.

Device functions
================

To use the device API, include the file ``rocrand_kernel.h`` in files that define kernels that use rocRAND device functions. The typical usage of device functions consists of the following operations in the device kernel definition:

1. Create a new generator state object of the desired generator type.

2. Initialize the generator state parameters using ``rocrand_init``.

3. Generate random numbers by calling the generation function on the generator state.

4. Use the results.

Since the rocRAND device functions are invoked from inside the user kernel, the generated numbers can be used right away in the kernel without the need to copy them to the host memory.

In the below example, random number generation is using the XORWOW generator.

..  code-block:: cpp

    #include <hip/hip_runtime.h>
    #include <rocrand/rocrand_kernel.h>
    
    __global__
    void test()
    {
        uint                 tid = blockDim.x * blockIdx.x + threadIdx.x;
        rocrand_state_xorwow state;
        rocrand_init(123, tid, 0, &state);
    
        for(int i = 0; i < 3; ++i)
        {
            const auto value = rocrand(&state);
            printf("thread %d, index %u: %u\n", tid, i, value);
        }
    }
    
    int main()
    {
        test<<<dim3(1), dim3(32)>>>();
        hipDeviceSynchronize();
    }

.. doxygengroup:: rocranddevice

C host API
==========

The C host API allows encapsulation of the internal generator state. Random numbers may be produced either on the host or device, depending on the created generator object. The typical sequence of operations for device generation consists of the following steps:

1. Allocate memory on the device with ``hipMalloc``.

2. Create a new generator of the desired type with ``rocrand_create_generator``.

3. Set the generator options, for example, use ``rocrand_set_seed`` to set the seed.

4. Generate random numbers with ``rocrand_generate`` or another generation function.

5. Use the results.

6. Clean up with ``rocrand_destroy_generator`` and ``hipFree``.

To generate random numbers on the host, the memory allocation in step one should be made using a host memory allocation call. In step two ``rocrand_create_generator_host`` should be called instead. In the last step, the appropriate memory release should be made using the ``rocrand_destroy_generator``. All other calls work identically whether you are generating random numbers on the device or on the host CPU. 

In the example below, the C host API is used to generate 10 random floats using GPU capabilities.

..  code-block:: c

    #include <hip/hip_runtime.h>
    #include <rocrand.h>
    #include <stdio.h>
    
    int main()
    {
        size_t n = 10;
    
        rocrand_generator gen;
        float *           d_rand, *h_rand;
    
        h_rand = (float*)malloc(sizeof(float) * n);
        hipMalloc((void**)&d_rand, n * sizeof(float));
    
        rocrand_create_generator(&gen, ROCRAND_RNG_PSEUDO_DEFAULT);
        rocrand_set_seed(gen, 123);
        rocrand_generate_uniform(gen, d_rand, n);
    
        hipMemcpy(h_rand, d_rand, n * sizeof(float), hipMemcpyDeviceToHost);
    
        for(int i = 0; i < n; i++)
        {
            printf("%f\n", h_rand[i]);
        }
    
        rocrand_destroy_generator(gen);
        hipFree(d_rand);
    
        return 0;
    }

.. doxygengroup:: rocrandhost

C++ host API wrapper
====================

The C++ host API wrapper provides resource management and an object-oriented interface for random number generation facilities.

In the example below C++ host API wrapper is used to produce a random number using the default generation parameters.

..  code-block:: cpp

    #include <hip/hip_runtime.h>
    #include <rocrand/rocrand.hpp>
    
    #include <iostream>
    
    int main()
    {
        float* d_rand;
        float  h_rand;
        hipMalloc((void**)&d_rand, sizeof(float));
    
        rocrand_cpp::xorwow                gen;
        rocrand_cpp::normal_distribution<> dist;
    
        dist(gen, d_rand, 1);
    
        hipMemcpy(&h_rand, d_rand, sizeof(float), hipMemcpyDeviceToHost);
    
        std::cout << h_rand << std::endl;
    
        hipFree(d_rand);
    
        return 0;
    }

.. doxygengroup:: rocrandhostcpp
