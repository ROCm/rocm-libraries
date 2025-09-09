.. meta::
  :description: rocPRIM documentation and API reference library
  :keywords: rocPRIM, ROCm, API, documentation

.. _spirv:

***************************
Using SPIR-V with rocPRIM
***************************

rocPRIM supports building with target-agnostic SPIR-V.

.. note::

    SPIR-V is in an early access state. Using it in production is not recommended.

To build with SPIR-V, set the ``ROCPRIM_EXPERIMENTAL_SPIRV`` parameter to any non-zero value, and set ``--offload-arch`` to ``amdgcnspirv``.

For example, with hipcc:

.. code:: shell

    hipcc -DROCPRIM_EXPERIMENTAL_SPIRV=1 --offload-arch=amdgcnspirv

For example, with cmake:

.. code:: shell

    cmake -DCMAKE_CXX_FLAGS="-DROCPRIM_EXPERIMENTAL_SPIRV=1 --offload-arch=amdgcnspirv"


.. note::

    rocPRIM does not support mixed compilation targets. No other target can be set when ``--offload-arch=amdgcnspirv`` is used.
    
    Setting ``ROCPRIM_EXPERIMENTAL_SPIRV`` will disable all config dispatching.


When targeting SPIR-V, the hardware wavefront size (also known as warp size) is not known
at compile time. 

Because the hardware wavefront size is not known at compilation-time, rocPRIM will add implicit dispatching where it is needed. This provides a way for the same SPIR-V code to work hardware wavefront sizes of both 32 and 64.

Adding SPIR-V wavefront dependent dispatching to an existing warp or block level algorithm can 
be done using the APIs in ``rocprim::arch::wavefront``. 

For example, given the following:

.. code:: cpp

    template<typename T>
    class my_alg
    {
    private:
        static constexpr unsigned int wave_size = 32;
    public:
        __device__ void some_api(T& a, T& b)
        {
            some_opaque_impl<wave_size>(a, b);
        }
    };

The wavefront size can be found using ``rocprim::arch::wavefront::get_target()``. A partial specialization needs to be defined to handle the dispatching for dynamic wavefront sizes.

.. code:: cpp

    template<
        typename T, 
        rocprim::arch::wavefront::target TargetWaveSize = rocprim::arch::wavefront::get_target()
    >
    class my_alg
    {
    private:
        constexpr unsigned int wave_size = rocprim::arch::wavefront::size_from_target<TargetWaveSize>();
    public:
        __device__ void some_api(T& a, T& b)
        {
            some_opaque_impl<wave_size>(a, b);
        }
    };

    template<typename T>
    class my_alg<T, ::rocprim::arch::wavefront::target::dynamic>
    {
    private:
        using impl32 = my_alg<T, ::rocprim::arch::wavefront::target::size32>;
        using impl64 = my_alg<T, ::rocprim::arch::wavefront::target::size64>;
    public:
        __device__ void some_api(T& a, T& b)
        {
            if (rocprim::arch::wavefront::size() == 32) {
                impl32().some_api(a, b);
            } else {
                impl64().some_api(a, b);
            }
        }
    };

    .. note::
    
        Developers who are modifying the rocPRIM code base can use ``rocprim::detail::dispatch_wave_size``. This function also manages exposed ``storage_type``- types to handle and map shared memory. Variadic templates are used to capture all signatures for a given member function.

        Developers who are only intending to use the rocPRIM library should not use ``dispatch_wave_size``.


``rocprim::arch::wavefront::get_target()`` will resolve to ``target::dynamic`` and be handled through  partial specialization. A downside of this implementation is that more shared memory is allocated than is used. 

An algorithm that runs on a device with wavefront size 32 will have a different shared memory footprint than one that runs on a device with wavefront size 64. Using the dispatcher will then result in suboptimal occupancy.

If it's known that a kernel will only run on hardware with a specific wavefront size, then the wavefront size can be passed to the kernel:

.. code:: cpp

    using key_type   = int;
    using value_type = rocprim::empty_type;

    // Will use dispatch internally when targeting SPIR-V.
    using sort_dispatch = rocprim::warp_sort<
        key_type,
        value_type
    >;

    // Will not use dispatch.
    // Will only work proper with hardware with wavefront size of 32.
    // Undefined behavior on hardware with wavefront size of 64.
    using sort_wave32   = rocprim::warp_sort<
        key_type,
        value_type,
        rocprim::arch::wavefront::target::wave32
    >;

    // Will not use dispatch.
    // Will only work proper with hardware with wavefront size of 64.
    // Undefined behavior on hardware with wavefront size of 32.
    using sort_wave64   = rocprim::warp_sort<
        key_type, value_type,
        rocprim::arch::wavefront::target::wave64
    >;

When compiling for a specific architecture and not SPIR-V, dispatch will not be used by default because ``rocprim::arch::wavefront::get_target()`` will already resolve to the wavefront size of the architecture.

.. code:: cpp

    #ifdef ROCPRIM_TARGET_CDNA3
        static_assert(rocprim::arch::wavefront::target::wave64 == rocprim::arch::wavefront::get_target());
    #endif