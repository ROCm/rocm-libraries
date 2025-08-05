.. meta::
  :description: rocPRIM documentation and API reference library
  :keywords: rocPRIM, ROCm, API, documentation

.. _developer:

********************************************************************
rocPRIM Developer guidelines
********************************************************************

Overview
========

As explained in :ref:`rocprim-intro`, rocPRIM's operations are part of one of four different hierarchical scopes: *Device/Grid*, :term:`Block`, :term:`Warp`, or *Thread*. This division facilitates re-use in the codebase and provides flexibility for users. Additional developer considerations are:

* *Device/Grid*: algorithms called from host code, executed on the entire device. The input size is variable and passed as an argument to the function.
* :term:`Block`: algorithms that are called from device code, executed by one thread block. All threads in a thread block should participate in the function call, and the threads together perform the algorithm. They are defined as structures, to group similar overloads and provide associated types such as the ``storage_type`` defining shared memory storage requirements. The maximum input size is defined by template arguments. Optionally, an actual size can be defined by ``valid_items`` overloads.
* :term:`Warp`: algorithms called from device code, executed by one warp. In many ways these algorithms are similar to the block-level algorithms, the key difference is that all threads in a warp collectively perform the algorithm. Through template arguments, a logical warp size can be specified.
* *Thread*: algorithms called from the device and perform work in a single thread without communicating with other threads.

See the contributing guide `CONTRIBUTING.md <https://github.com/ROCm/rocPRIM/blob/develop/CONTRIBUTING.md>`_ for information on file structure and how test and benchmarks should be implemented.

General rules
=============

Code should be modular, and when possible broader scoped to facilitate reuse. If there is no adverse effect on performance, extract common functionality. The different hierarchies of the API are not only for the user, algorithm implementations use these endpoints as well. For instance, device-level algorithms typically use the block-level algorithms for loading and storing data. 

It should be clear from function template parameters whether they are tuning options that do not affect behavior, or are algorithmic parameters that change behavior. For instance, tuning options may be block size, items per thread, or the block-level scan method (``block_scan_algorithm``). An algorithmic parameter could be whether a scan has an initial value, or whether a reduction is inclusive or exclusive. An example of an enumeration that violates this rule is ``block_load_method``, where the different members make different orders of the elements.

Between minor ROCm versions, breaking changes in the public API MUST NOT be introduced. Everything in the namespace ``rocprim`` is considered public API, based on the assumption that a user may in theory depend on it. Pay special attention not to break backward-compatibility, as it can be done in subtle ways. For example, many functions allow user-defined types, which behave differently in many ways from fundamental types. Be defensive in what is placed in the public API as sustaining backward-compatibility is a burden on maintenance. If it is not necessary to be exposed, place it in ``rocprim::detail`` (or lower) instead. A common additional check is to make sure downstream libraries still compile and execute tests successfully (hipCUB, rocThrust, Tensorflow, and PyTorch).

HIP Graphs are a way to capture all stream-ordered HIP API calls into a graph without executing, and then replaying the graph many times afterwards. Supporting graph capture makes rocPRIM more flexible to use, and all device-level algorithms should strive to allow it. Among other things, one general requirement is that the number of kernel calls and the launch parameters of kernel calls should not depend on input data. If support is not possible for a specific algorithm, the documentation should state this clearly.

Configurations and architecture dispatch
========================================

One of the most complex parts of rocPRIM is the mechanism that allows for the user-provided configuration and defaulted automatic configuration of device-level algorithms.

Default and user-specified configuration
----------------------------------------

As explained in :ref:`tuning`, device-level algorithms may be configured by accepting a tuning config. It may be provided by the caller, or defaulted to ``default_config``, which selects a suitable default configuration. 

The number of threads in a block (the "block size") is a quintessential configuration parameter for kernels. It needs to be known at the host side to launch the kernel and at the device side at compile-time for the generation of algorithmic functions. HIP code is compiled in multiple passes, one pass for the host and one pass for each targeted device architecture. When a kernel is launched on the host, the HIP runtime selects the binary based on the device associated with the HIP stream. Since the configuration, and thus the block size, depends on this device architecture, rocPRIM must have a similar mechanism to infer the architecture of the device based on the the HIP stream.

To facilitate a dispatching mechanism supporting the above requirements, several standardized structures need to be defined for each algorithm, which is outlined in this section. These structures depend on a generalized dispatching mechanism.

The algorithm's configuration struct is defined in ``rocprim/device/detail/device_config_helper.hpp``. The reason for putting all configurations in one file is to make the configuration templates simpler (generating configurations is explained :ref:`tuning`). The tuning config has the name ``ALGO_config``, and no members (unless for backward-compatibility reasons), only template parameters. 

The config struct derives from a non-public parameter struct holding the actual parameters. This separation between structs is done to facilitate change without breaking public API.

.. code:: cpp

    namespace detail
    {

    struct ALGO_config_params
    {
        unsigned int BlockSize; 
        unsigned int ItemsPerThread;
    };

    } // namespace detail

    template<unsigned int BlockSize, unsigned int ItemsPerThread>
    struct ALGO_config : public detail::ALGO_config_params
    {
        constexpr ALGO_config() : detail::ALGO_config_params{BlockSize, ItemsPerThread}
        {}
    }

In order to accept either ``default_config`` or ``ALGO_config`` as the device-level configuration template type and convert it to a parameter instance, a non-public config wrapper is defined in ``rocprim/device/device_ALGO_config.hpp``.

.. code:: cpp

    namespace detail {

    // generic struct that instantiates custom configurations
    template<typename ALGOConfig, typename>
    struct wrapped_ALGO_config
    {
        template<target_arch Arch>
        struct architecture_config
        {
            static constexpr ALGO_config_params params = ALGOConfig();
        };
    };

    // specialized for rocprim::default_config, which instantiates the default_ALGO_config
    template<typename Type>
    struct wrapped_ALGO_config<default_config, Type>
    {
        template<target_arch Arch>
        struct architecture_config
        {
            static constexpr ALGO_config_params params = default_ALGO_config<static_cast<unsigned int>(Arch), Type>();
        };
    };

    } // namespace detail

Selecting the default configuration is done based on the target architecture ``target_arch`` and typically also on the input types of the algorithm (in the example above, a single type ``Type`` is used). The ``default_ALGO_config`` is defined in ``rocprim/include/device/detail/config/device_ALGO.hpp``. This file will be generated by the autotuning process, as explained in :ref:`tuning`. The files look like this:

.. code:: cpp

    namespace detail
    {

    // base configuration in case no specific configuration exists
    template<unsigned int arch, typename Type, class enable = void>
    struct default_ALGO_config : default_ALGO_config_base<Type>::type
    {};

    // generated configuration for architecture gfx1030, based on float
    template<class Type>
    struct default_ALGO_config<
        static_cast<unsigned int>(target_arch::gfx1030),
        Type,
        std::enable_if_t<bool(rocprim::traits::get<value_type>().is_floating_point()) && (sizeof(value_type) <= 4) && (sizeof(value_type) > 2)>>
        : ALGO_config<256, 16>
    {};

    // many generated configurations..

    } // namespace detail

It is up to the implementer to specify a suitable and generic base configuration. This base configuration is not placed in the template to make the template simpler. Instead, it is defined in ``rocprim/device/detail/device_config_helper.hpp``:

.. code:: cpp

    namespace detail
    {

    template<typename Type>
    struct default_ALGO_config_base
    {
        using type = ALGO_config<256, 4>;
    };

    } // namespace detail

Finally, the kernel is templatized with the ``wrapped_ALGO_config`` and not the actual configuration parameters. It is done so that the architecture enumeration value (or any dependent configuration parameters) does not appear in the function signature. This prevents a host-side switch statement over the architecture values to select the right kernel to launch. Instead, this selection is done at compile time in device code.

Config dispatch
---------------

The default configuration depends on the types of the input values of the algorithm, as well as the device architecture. The device architecture is determined at runtime, based on the HIP stream. At the host side, the configuration parameters are selected at runtime using the following pattern:

.. code:: cpp

    using config = wrapped_ALGO_config<config, Type>;

    detail::target_arch target_arch;
    hipError_t          result = host_target_arch(stream, target_arch);
    if(result != hipSuccess)
    {
        return result;
    }
    const ALGO_config_params params = dispatch_target_arch<config>(target_arch);

In device code the device architecture is known at compile time, and thus the configuration can also be selected at compile time. All that is needed, is the following pattern:

.. code:: cpp

    constexpr ALGO_CONFIG_PARAMS params = device_params<config>();

The ``device_params`` function selects the configuration based on the predefined compiler macro ``__amdgcn_processor__``. In the example, ``config`` is of type ``wrapped_ALGO_config`` as in the host example.

Common patterns
===============

There are several patterns throughout rocPRIM's codebase for uniformity and enforcing good practice.

Temporary storage allocation
----------------------------

If a device-level function requires temporary storage, ``void* temporary_storage`` and ``size_t& storage_size`` will be the first two parameters. When calling the function with ``nullptr`` for ``temporary_storage``, the function will set ``storage_size`` to the required number of temporary device memory bytes. If no temporary storage is required under specific circumstances, ``storage_size`` should be set to a small non-zero value, to prevent the users from having to check before making a zero-sized allocation.

Common functionality in the ``detail::temp_storage`` namespace is used to calculate the required storage on the first function call and assign pointers in the second function call. The below example allocates and assigns a temporary array of ten integers.

.. code:: cpp

    hipError_t function(void* temporary_storage, size_t& storage_size)
    {
        int* d_tmp{};

        // if temporary_storage is nullptr, sets storage_size to the required size
        // else, assigns the pointer d_tmp
        const hipError_t partition_result = detail::temp_storage::partition(
            temporary_storage,
            storage_size,
            detail::temp_storage::make_linear_partition(
                detail::temp_storage::ptr_aligned_array(&d_tmp, 10)));
        if(partition_result != hipSuccess || temporary_storage == nullptr)
        {
            return partition_result;
        }

        // perform the function with temporary memory
        return function_impl(d_tmp);
    }

Reusing shared memory
---------------------

Shared memory reuse in a kernel is facilitated by placing multiple ``storage_type`` declarations in a union.

.. code:: cpp

    using block_load_t = block_load<T, block_size>;
    using block_scan_t = block_scan<T, block_size>;
    using block_store_t = block_store<T, block_size>;

    ROCPRIM_SHARED_MEMORY union
    {
        typename block_load_t::storage_type  load;
        typename block_scan_t::storage_type  scan;
        typename block_store_t::storage_type store;
    } storage;

    T value;
    block_load_t().load(input, value, storage.load);

    syncthreads();

    block_scan_t().scan(value, storage.scan);

    syncthreads();

    block_store_t().store(output, value, storage.store);

Partial block idiom
-------------------

Since thread blocks have uniform sizes, bounds checking is necessary to prevent out-of-bounds loads and stores. Applying a check to every loaded and stored value may become a performance bottleneck. A typical solution is to have a block-wide check, whether a per-item check is necessary. A simple example is below.

.. code:: cpp

    // slow, adds a check for every stored item in each block
    const unsigned int thread_id = detail::block_thread_id<0>();
    const unsigned int block_id  = detail::block_id<0>();
    const auto num_valid_in_last_block = input_size - block_offset;
    block_store_t().store(
        output,
        values,
        num_valid_in_last_block,
        storage);

    // fast, adds a check only for incomplete blocks (which can only be the last block)
    constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;
    const bool is_incomplete_block = block_id == (input_size / items_per_block);
    if(is_incomplete_block)
    {
        block_store_t().store(
            output,
            values,
            num_valid_in_last_block,
            storage);
    }
    else
    {
        block_store_t().store(
            output,
            values,
            storage);
    }

Large indices
-------------

Typically, each thread handles a fixed amount of elements and HIP limits how many threads can be in a single launch. This means there is a hard limit to the number of elements that can be handled in a single kernel call. Special attention must be paid to how input sizes beyond this limit are handled. This is commonly handled by launching multiple kernels in a loop and combining results.

Naming of device-level functions
--------------------------------

Typically, multiple overloads of device-level functions exist, that call into a common implementation. Below is an example of this pattern and what the naming should look like

.. code:: cpp

    BEGIN_ROCPRIM_NAMESPACE

    namespace detail
    {

    ROCPRIM_KERNEL reduce_kernel(...)
    {
        // reduce_kernel_impl defined in rocprim/device/detail/device_reduce.hpp
        reduce_kernel_impl(...);
    }

    template<bool HasInitialValue>
    hipError_t reduce_impl(...)
    {
        reduce_kernel<<<...>>>(...);
    }

    } // namespace detail

    // default reduce
    hipError_t reduce(...)
    {
        return detail::reduce_impl<false>(...);
    }

    // reduce overload with initial value
    hipError_t reduce(...)
    {
        return detail::reduce_impl<true>(...);
    }

    END_ROCPRIM_NAMESPACE

Synchronous debugging
---------------------

All device-level functions have as a last parameter ``bool debug_synchronous``, which defaults to ``false``. This parameter toggles synchronization after kernel launches for debugging purposes. Typically, additional debugging information is printed as well.

Items per thread
----------------

Most device functions operate on a fixed number of elements and are templatized based on the element type. These functions will have an ``unsigned int ItemsPerThread`` template parameter, which specifies how many elements each thread should process. The main purpose of this parameter is to tune the performance of such a function. As different types are of different sizes, it is likely that there is no single ``ItemsPerThread`` value that gives good performance for types of all sizes. The ``ItemsPerThread`` value often directly influences register usage of a kernel, which influences the kernel's occupancy.

Kernel launch bounds
--------------------

To guide the code generation process, it is possible to specify the maximum block size for a kernel with ``__launch_bounds__()``. Since most kernels are templatized based on a configuration, a common pattern is the following:

.. code:: cpp

    template<class Config>
    ROCPRIM_KERNEL __launch_bounds__(device_params<Config>().block_size) void kernel(...)
    {}

Pitfalls and common mistakes
----------------------------

HIP code is compiled in multiple passes: one for the host and one for each targeted device architecture. As such, host code is agnostic of device architecture, and should be designed as such. Only with a ``hipStream`` can the device be inferred and can certain properties be obtained. Since device code is compiled for a specific architecture, it can contain compile-time optimizations for specific architectures. Note that AMD GPUs have a warp size of 32 or 64, and unless specialized, algorithms should work for both warp sizes.

All variables with the ``__shared__`` memory space specifier should either be in a function with the ``__global__`` (``ROCPRIM_KERNEL``) execution space specifier or in a function with the ``__device__`` (``ROCPRIM_DEVICE``) execution space specifier marked with ``__forceinline__`` (``ROCPRIM_FORCE_INLINE``). The reason for this is that without forcing the inlining of the function, the compiler may choose not to optimize shared memory allocations, leading to exceeding the limit dictated by hardware.

Documenting algorithms
======================

Documenting algorithms requires updating several files.

See `Contributing to the ROCm documentation <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_ for general guidelines.

The ``docs/`` directory contains the ``.rst`` files that form this website. These ``.rst`` files use `Breathe directives <https://breathe.readthedocs.io/en/latest/directives.html>`_ to display `Doxygen's special commands <https://www.doxygen.nl/manual/commands.html>`_, which are used in C++ comments to document code.

Algorithms should be documented in their appropriate ``docs/`` subdirectories, like ``block_ops/`` and ``device_ops/``, based on their scope. :ref:`dev-nth_element` for example is a :ref:`device-wide operation <dev-index>`, documented in ``docs/device_ops/nth_element.rst``.

There are several pages that link to the :ref:`dev-nth_element` documentation:

- ``docs/device_ops/index.rst``, which adds :ref:`dev-nth_element` to :ref:`dev-index`.
- ``docs/reference/ops_summary.rst``, which adds :ref:`dev-nth_element` to :ref:`ops-summary`.
- ``docs/sphinx/_toc.yml.in``, which adds :ref:`dev-nth_element` to Sphinx its `Table Of Contents <https://github.com/ROCm/rocPRIM/blob/develop/docs/sphinx/_toc.yml.in>`_, and ``docs/sphinx/_toc.yml`` is automatically generated from this file.

Any newly added algorithm should also update these files. As one can easily forget to update one of them, it is recommended to `build and preview the documentation locally <https://github.com/ROCm/rocPRIM/blob/develop/README.md#building-the-documentation-locally>`_.
