.. meta::
  :description: rocPRIM documentation and API reference library
  :keywords: rocPRIM, ROCm, API, documentation

.. _spirv:

******
SPIR-V
******

.. warning::
    This is an **experimental compile target** and may be subject to change without notice.
    Use at your own risk. Features may be incomplete, unstable, or incompatible with future
    versions. It is **not recommended for production use**.

rocPRIM has experimental support for AMD GPU target agnostic SPIR-V.

Requirements
============

The define ``ROCPRIM_EXPERIMENTAL_SPIRV`` must be set to a non-zero value. For example:

.. code:: shell

    hipcc -DROCPRIM_EXPERIMENTAL_SPIRV=1 --offload-arch=amdgcnspirv

.. warning::
    Setting `ROCPRIM_EXPERIMENTAL_SPIRV` will disable all config dispatching.

.. note::
    When using CMake, this flag needs to be set in the `CMAKE_CXX_FLAGS`. For example:
    
    .. code:: shell

        cmake -DCMAKE_CXX_FLAGS="-DROCPRIM_EXPERIMENTAL_SPIRV=1 --offload-arch=amdgcnspirv"

No other target apart from ``amdgcnspirv`` should be targeted. rocPRIM does not support mixed
compilation targets.

.. code:: shell

    # !!! This does not work !!!
    hipcc -DROCPRIM_EXPERIMENTAL_SPIRV=1 --offload-arch=amdgcnspirv --offload-arch=gfx942

Warp- and block-level dispatch
==============================

When targeting SPIR-V, the hardware wavefront size (also known as warp size) is not known
at compile time. It is only known during the execution of the host program and thus at the
runtime compilation of SPIR-V code to the ISA of the actual target, e.g. gfx942 (MI300).
rocPRIM will add implicit dispatching where it is needed when targeting SPIR-V. This allows
the same SPIR-V code to work with hardware wavefront sizes of both 32 and 64.

Adding dispatching
------------------

Adding SPIR-V wavefront dependent dispatching to an existing warp or block level algorithm can 
be done using infrastructure provided by rocPRIM (see: ``rocprim::arch::wavefront``). This does
require some extra boiler plate. For example, for an algorithm with some API:

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

We need to know the current hardware wavefront size. This can be done with
``rocprim::arch::wavefront::get_target()``. A partial specialization needs to be defined to
handle the dispatching for dynamic wavefront sizes (either 32 or 64).

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

    Developers of rocPRIM can use the the ``rocprim::detail::dispatch_wave_size``. This function will also manage
    exposed ``storage_type``- types to handle and map shared memory. Variadic templates are used to capture all
    signatures for a given member function.

    .. warning::
        ``rocprim::detail::dispatch_wave_size`` is used internally by rocPRIM. Usage by downstream users is not
        recommended because its behaviour and signature may change at any moment.

    .. code:: cpp

        template<typename T>
        class my_alg<T, ::rocprim::arch::wavefront::target::dynamic>
        {
        private:
            using impl32 = my_alg<T, ::rocprim::arch::wavefront::target::size32>;
            using impl64 = my_alg<T, ::rocprim::arch::wavefront::target::size64>;
            using dispatch = rocprim::detail::dispatch_wave_size<impl32, impl64>;
        public:
            template<typename... Args>
            __device__ void some_api(Args&&... args)
            {
                dispatch{}([](auto impl, auto&&... args) { impl.some_api(args...); }, args...);
            }
        };

Invoking a dispatchable algorithm
---------------------------------

When a rocPRIM provided algorithm is dependent on the wavefront size, the dispatcher will handle most of the 
cases. ``rocprim::arch::wavefront::get_target()`` will resolve to ``target::dynamic`` and be handled via 
partial specialization. However, there are cases where this is not desired. A downside of the dispatched
implementation is that more shared memory is allocated than actually used. An algorithm that runs on a device
with wavefront size 32 may have a different shared memory footprint than one with wavefront size 64. Using the
dispatcher will then result in suboptimal occupancy.

If it's known that a kernel is only invoked on hardware with a specific wavefront size, then the wavefront size can
be passed to it.

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
    // Undefined behaviour on hardware with wavefront size of 64.
    using sort_wave32   = rocprim::warp_sort<
        key_type,
        value_type,
        rocprim::arch::wavefront::target::wave32
    >;

    // Will not use dispatch.
    // Will only work proper with hardware with wavefront size of 64.
    // Undefined behaviour on hardware with wavefront size of 32.
    using sort_wave64   = rocprim::warp_sort<
        key_type, value_type,
        rocprim::arch::wavefront::target::wave64
    >;

When compiling for a specific architecture (and thus not SPIR-V), dispatch will not be used by default since
``rocprim::arch::wavefront::get_target()`` will already resolve to the wavefront size of the architecture.

.. code:: cpp

    #ifdef ROCPRIM_TARGET_CDNA3
        static_assert(rocprim::arch::wavefront::target::wave64 == rocprim::arch::wavefront::get_target());
    #endif
