// Copyright (c) 2017-2021 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef ROCPRIM_DEVICE_DEVICE_SCAN_HPP_
#define ROCPRIM_DEVICE_DEVICE_SCAN_HPP_

#include <type_traits>
#include <iterator>

#include "../config.hpp"
#include "../functional.hpp"
#include "../type_traits.hpp"
#include "../types/future_value.hpp"
#include "../detail/various.hpp"

#include "device_scan_config.hpp"
#include "device_transform.hpp"
#include "detail/device_scan_reduce_then_scan.hpp"
#include "detail/device_scan_lookback.hpp"

BEGIN_ROCPRIM_NAMESPACE

/// \addtogroup devicemodule
/// @{

namespace detail
{

// Single kernel scan (performs scan on one thread block only)
template<
    bool Exclusive,
    class Config,
    class InputIterator,
    class OutputIterator,
    class BinaryFunction,
    class InitValueType
>
ROCPRIM_KERNEL
__launch_bounds__(ROCPRIM_DEFAULT_MAX_BLOCK_SIZE)
void single_scan_kernel(InputIterator input,
                        const size_t size,
                        const InitValueType initial_value,
                        OutputIterator output,
                        BinaryFunction scan_op)
{
    single_scan_kernel_impl<Exclusive, Config>(
        input, size, get_input_value(initial_value), output, scan_op
    );
}

// Reduce-then-scan kernels

// Calculates block prefixes that will be used in final_scan_kernel
// when performing block scan operations.
template<
    class Config,
    class InputIterator,
    class BinaryFunction,
    class ResultType
>
ROCPRIM_KERNEL
__launch_bounds__(ROCPRIM_DEFAULT_MAX_BLOCK_SIZE)
void block_reduce_kernel(InputIterator input,
                         BinaryFunction scan_op,
                         ResultType * block_prefixes)
{
    block_reduce_kernel_impl<Config>(
        input, scan_op, block_prefixes
    );
}

template<
    bool Exclusive,
    class Config,
    class InputIterator,
    class OutputIterator,
    class BinaryFunction,
    class InitValueType
>
ROCPRIM_KERNEL
__launch_bounds__(ROCPRIM_DEFAULT_MAX_BLOCK_SIZE)
void final_scan_kernel(InputIterator input,
                       const size_t size,
                       OutputIterator output,
                       const InitValueType initial_value,
                       BinaryFunction scan_op,
                       input_type_t<InitValueType>* block_prefixes,
                       input_type_t<InitValueType>* previous_last_element = nullptr,
                       input_type_t<InitValueType>* new_last_element = nullptr,
                       bool override_first_value = false,
                       bool save_last_value = false)
{
    final_scan_kernel_impl<Exclusive, Config>(
        input, size, output, get_input_value(initial_value),
        scan_op, block_prefixes,
        previous_last_element, new_last_element,
        override_first_value, save_last_value
    );
}

// Single pass (look-back kernels)

template<
    bool Exclusive,
    class Config,
    class InputIterator,
    class OutputIterator,
    class BinaryFunction,
    class InitValueType,
    class LookBackScanState
>
ROCPRIM_KERNEL
__launch_bounds__(ROCPRIM_DEFAULT_MAX_BLOCK_SIZE)
void lookback_scan_kernel(InputIterator input,
                          OutputIterator output,
                          const size_t size,
                          const InitValueType initial_value,
                          BinaryFunction scan_op,
                          LookBackScanState lookback_scan_state,
                          const unsigned int number_of_blocks,
                          ordered_block_id<unsigned int> ordered_bid,
                          input_type_t<InitValueType>* previous_last_element = nullptr,
                          input_type_t<InitValueType>* new_last_element = nullptr,
                          bool override_first_value = false,
                          bool save_last_value = false)
{
    lookback_scan_kernel_impl<Exclusive, Config>(
        input, output, size, get_input_value(initial_value), scan_op,
        lookback_scan_state, number_of_blocks, ordered_bid,
        previous_last_element, new_last_element,
        override_first_value, save_last_value
    );
}

template<class LookBackScanState>
ROCPRIM_KERNEL
__launch_bounds__(ROCPRIM_DEFAULT_MAX_BLOCK_SIZE)
void init_lookback_scan_state_kernel(LookBackScanState lookback_scan_state,
                                     const unsigned int number_of_blocks,
                                     ordered_block_id<unsigned int> ordered_bid)
{
    init_lookback_scan_state_kernel_impl(
        lookback_scan_state, number_of_blocks, ordered_bid
    );
}

#define ROCPRIM_DETAIL_HIP_SYNC(name, size, start) \
    if(debug_synchronous) \
    { \
        std::cout << name << "(" << size << ")"; \
        auto error = hipStreamSynchronize(stream); \
        if(error != hipSuccess) return error; \
        auto end = std::chrono::high_resolution_clock::now(); \
        auto d = std::chrono::duration_cast<std::chrono::duration<double>>(end - start); \
        std::cout << " " << d.count() * 1000 << " ms" << '\n'; \
    }

#define ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR(name, size, start) \
    { \
        auto _error = hipGetLastError(); \
        if(_error != hipSuccess) return _error; \
        if(debug_synchronous) \
        { \
            std::cout << name << "(" << size << ")"; \
            auto __error = hipStreamSynchronize(stream); \
            if(__error != hipSuccess) return __error; \
            auto _end = std::chrono::high_resolution_clock::now(); \
            auto _d = std::chrono::duration_cast<std::chrono::duration<double>>(_end - start); \
            std::cout << " " << _d.count() * 1000 << " ms" << '\n'; \
        } \
    }

template<
    bool Exclusive,
    class Config,
    class InputIterator,
    class OutputIterator,
    class InitValueType,
    class BinaryFunction
>
inline
auto scan_impl(void * temporary_storage,
               size_t& storage_size,
               InputIterator input,
               OutputIterator output,
               const InitValueType initial_value,
               const size_t size,
               BinaryFunction scan_op,
               const hipStream_t stream,
               bool debug_synchronous)
    -> typename std::enable_if<!Config::use_lookback, hipError_t>::type
{
    using config = Config;
    using real_init_value_type = input_type_t<InitValueType>;

    constexpr unsigned int block_size = config::block_size;
    constexpr unsigned int items_per_thread = config::items_per_thread;
    constexpr auto items_per_block = block_size * items_per_thread;

    static constexpr size_t size_limit = config::size_limit;
    static constexpr size_t aligned_size_limit = ::rocprim::max<size_t>(size_limit - size_limit % items_per_block, items_per_block);
    size_t limited_size = std::min<size_t>(size, aligned_size_limit);
    const bool use_limited_size = limited_size == aligned_size_limit;
    size_t nested_prefixes_size_bytes = scan_get_temporary_storage_bytes<real_init_value_type>(limited_size, items_per_block);

    // Calculate required temporary storage
    if(temporary_storage == nullptr)
    {
        storage_size = nested_prefixes_size_bytes;

        if(use_limited_size)
            storage_size += 4 * sizeof(real_init_value_type);

        // Make sure user won't try to allocate 0 bytes memory, because
        // hipMalloc will return nullptr when size is zero.
        storage_size = storage_size == 0 ? 4 : storage_size;
        return hipSuccess;
    }

    // Start point for time measurements
    std::chrono::high_resolution_clock::time_point start;

    auto number_of_blocks = (size + items_per_block - 1)/items_per_block;

    if( number_of_blocks == 0u )
        return hipSuccess;

    if(number_of_blocks > 1)
    {
        unsigned int number_of_launch = (size + limited_size - 1)/limited_size;
        for (size_t i = 0, offset = 0; i < number_of_launch; i++, offset+=limited_size )
        {
            size_t current_size = std::min<size_t>(size - offset, limited_size);
            number_of_blocks = (current_size + items_per_block - 1)/items_per_block;
            if(debug_synchronous)
            {
                std::cout << "use_limited_size " << use_limited_size << '\n';
                std::cout << "number_of_launch " << number_of_launch << '\n';
                std::cout << "inex " << i << '\n';
                std::cout << "aligned_size_limit " << aligned_size_limit << '\n';
                std::cout << "size " << current_size << '\n';
                std::cout << "block_size " << block_size << '\n';
                std::cout << "number of blocks " << number_of_blocks << '\n';
                std::cout << "items_per_block " << items_per_block << '\n';
                std::cout.flush();
            }

            // Pointer to array with block_prefixes
            char * ptr = reinterpret_cast<char *>(temporary_storage);
            real_init_value_type* block_prefixes = reinterpret_cast<real_init_value_type*>(ptr);
            real_init_value_type* previous_last_element = nullptr;
            real_init_value_type* new_last_element = nullptr;
            if(use_limited_size)
            {
                ptr += nested_prefixes_size_bytes;
                previous_last_element = reinterpret_cast<real_init_value_type*>(ptr);

                ptr += sizeof(real_init_value_type);
                new_last_element = reinterpret_cast<real_init_value_type*>(ptr);
            }

            // Grid size for block_reduce_kernel, we don't need to calculate reduction
            // of the last block as it will never be used as prefix for other blocks
            auto grid_size = number_of_blocks - 1;
            if( grid_size != 0 )
            {
                if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(detail::block_reduce_kernel<
                        config, InputIterator, BinaryFunction, real_init_value_type
                    >),
                    dim3(grid_size), dim3(block_size), 0, stream,
                    input + offset, scan_op, block_prefixes
                );
                ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("block_reduce_kernel", current_size, start)

                if( !Exclusive && i > 0 )
                {
                    hipError_t error = ::rocprim::transform(
                        previous_last_element, block_prefixes, block_prefixes, 1,
                        scan_op, stream, debug_synchronous
                    );
                    if(error != hipSuccess) return error;
                }

                // TODO: Performance may increase if for (number_of_blocks < 8192) (or some other
                // threshold) we would just use CPU to calculate prefixes.

                // Calculate size of temporary storage for nested device scan operation
                void * nested_temp_storage = static_cast<void*>(block_prefixes + number_of_blocks);
                auto nested_temp_storage_size = storage_size - (number_of_blocks * sizeof(real_init_value_type));

                if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
                auto error = scan_impl<false, config>(
                    nested_temp_storage,
                    nested_temp_storage_size,
                    block_prefixes, // input
                    block_prefixes, // output
                    real_init_value_type(), // dummy initial value
                    number_of_blocks, // input size
                    scan_op,
                    stream,
                    debug_synchronous
                );
                if(error != hipSuccess) return error;
                ROCPRIM_DETAIL_HIP_SYNC("nested_device_scan", number_of_blocks, start);

            }

            // Grid size for final_scan_kernel
            grid_size = number_of_blocks;
            if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(detail::final_scan_kernel<
                    Exclusive, // flag for exclusive scan operation
                    config, // kernel configuration (block size, ipt)
                    InputIterator, OutputIterator,
                    BinaryFunction, InitValueType
                >),
                dim3(grid_size), dim3(block_size), 0, stream,
                input + offset,
                current_size,
                output + offset,
                initial_value,
                scan_op,
                block_prefixes,
                previous_last_element,
                new_last_element,
                i != size_t(0) && ((!Exclusive && number_of_blocks == 1) || Exclusive),
                number_of_launch > 1
            );
            ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("final_scan_kernel", size, start);

            // Swap the last_elements if it's necessary
            if(number_of_launch > 1)
            {
                hipError_t error = ::rocprim::transform(
                    new_last_element, previous_last_element, 1,
                    ::rocprim::identity<real_init_value_type>(),
                    stream, debug_synchronous
                );
                if(error != hipSuccess) return error;
            }
        }
    }
    else
    {
        if(debug_synchronous)
        {
            std::cout << "block_size " << block_size << '\n';
            std::cout << "number of blocks " << number_of_blocks << '\n';
            std::cout << "items_per_block " << items_per_block << '\n';
        }

        if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(detail::single_scan_kernel<
                Exclusive, // flag for exclusive scan operation
                config, // kernel configuration (block size, ipt)
                InputIterator, OutputIterator, BinaryFunction
            >),
            dim3(1), dim3(block_size), 0, stream,
            input, size, initial_value, output, scan_op
        );
        ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("single_scan_kernel", size, start);
    }
    return hipSuccess;
}

template<
    bool Exclusive,
    class Config,
    class InputIterator,
    class OutputIterator,
    class InitValueType,
    class BinaryFunction
>
inline
auto scan_impl(void * temporary_storage,
               size_t& storage_size,
               InputIterator input,
               OutputIterator output,
               const InitValueType initial_value,
               const size_t size,
               BinaryFunction scan_op,
               const hipStream_t stream,
               bool debug_synchronous)
    -> typename std::enable_if<Config::use_lookback, hipError_t>::type
{
    using config = Config;
    using real_init_value_type = input_type_t<InitValueType>;

    using scan_state_type = detail::lookback_scan_state<real_init_value_type>;
    using scan_state_with_sleep_type = detail::lookback_scan_state<real_init_value_type, true>;
    using ordered_block_id_type = detail::ordered_block_id<unsigned int>;

    constexpr unsigned int block_size = config::block_size;
    constexpr unsigned int items_per_thread = config::items_per_thread;
    constexpr auto items_per_block = block_size * items_per_thread;

    static constexpr size_t size_limit = config::size_limit;
    static constexpr size_t aligned_size_limit = ::rocprim::max<size_t>(size_limit - size_limit % items_per_block, items_per_block);
    size_t limited_size = std::min<size_t>(size, aligned_size_limit);
    const bool use_limited_size = limited_size == aligned_size_limit;

    unsigned int number_of_blocks = (limited_size + items_per_block - 1)/items_per_block;

    // Calculate required temporary storage
    size_t scan_state_bytes = ::rocprim::detail::align_size(
        // This is valid even with scan_state_with_sleep_type
        scan_state_type::get_storage_size(number_of_blocks)
    );
    size_t ordered_block_id_bytes = ordered_block_id_type::get_storage_size();
    if(temporary_storage == nullptr)
    {
        // storage_size is never zero
        storage_size = scan_state_bytes + ordered_block_id_bytes;

        if(use_limited_size)
            storage_size += 2 * sizeof(real_init_value_type);

        return hipSuccess;
    }

    // Start point for time measurements
    std::chrono::high_resolution_clock::time_point start;

    if( number_of_blocks == 0u )
        return hipSuccess;

    if(number_of_blocks > 1 || use_limited_size)
    {
        // Create and initialize lookback_scan_state obj
        auto scan_state = scan_state_type::create(temporary_storage, number_of_blocks);
        auto scan_state_with_sleep = scan_state_with_sleep_type::create(temporary_storage, number_of_blocks);
        // Create ad initialize ordered_block_id obj
        auto ptr = reinterpret_cast<char*>(temporary_storage);
        auto ordered_bid = ordered_block_id_type::create(
            reinterpret_cast<ordered_block_id_type::id_type*>(ptr + scan_state_bytes)
        );

        // The last element
        real_init_value_type* previous_last_element = nullptr;
        real_init_value_type* new_last_element = nullptr;
        if(use_limited_size)
        {
            ptr += storage_size - sizeof(real_init_value_type);
            new_last_element = reinterpret_cast<real_init_value_type*>(ptr);
            ptr -= sizeof(real_init_value_type);
            previous_last_element = reinterpret_cast<real_init_value_type*>(ptr);
        }

        hipDeviceProp_t prop;
        int deviceId;
        static_cast<void>(hipGetDevice(&deviceId));
        static_cast<void>(hipGetDeviceProperties(&prop, deviceId));

        if(debug_synchronous) start = std::chrono::high_resolution_clock::now();

#if HIP_VERSION >= 307
        int asicRevision = prop.asicRevision;
#else
        int asicRevision = 0;
#endif

        size_t number_of_launch = (size + limited_size - 1)/limited_size;
        for (size_t i = 0, offset = 0; i < number_of_launch; i++, offset+=limited_size )
        {
            size_t current_size = std::min<size_t>(size - offset, limited_size);
            number_of_blocks = (current_size + items_per_block - 1)/items_per_block;
            auto grid_size = (number_of_blocks + block_size - 1)/block_size;

            if(debug_synchronous)
            {
                std::cout << "use_limited_size " << use_limited_size << '\n';
                std::cout << "aligned_size_limit " << aligned_size_limit << '\n';
                std::cout << "number_of_launch " << number_of_launch << '\n';
                std::cout << "index " << i << '\n';
                std::cout << "size " << current_size << '\n';
                std::cout << "block_size " << block_size << '\n';
                std::cout << "number of blocks " << number_of_blocks << '\n';
                std::cout << "items_per_block " << items_per_block << '\n';
            }

            if (prop.gcnArch == 908 && asicRevision < 2)
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(init_lookback_scan_state_kernel<scan_state_with_sleep_type>),
                    dim3(grid_size), dim3(block_size), 0, stream,
                    scan_state_with_sleep, number_of_blocks, ordered_bid
                );
            } else
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(init_lookback_scan_state_kernel<scan_state_type>),
                    dim3(grid_size), dim3(block_size), 0, stream,
                    scan_state, number_of_blocks, ordered_bid
                );
            }
            ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("init_lookback_scan_state_kernel", number_of_blocks, start)

            if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
            grid_size = number_of_blocks;
            if (prop.gcnArch == 908 && asicRevision < 2)
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(lookback_scan_kernel<
                        Exclusive, // flag for exclusive scan operation
                        config, // kernel configuration (block size, ipt)
                        InputIterator, OutputIterator,
                        BinaryFunction, InitValueType, scan_state_with_sleep_type
                    >),
                    dim3(grid_size), dim3(block_size), 0, stream,
                    input + offset, output + offset, current_size, initial_value,
                    scan_op, scan_state_with_sleep, number_of_blocks, ordered_bid,
                    previous_last_element, new_last_element,
                    i != size_t(0), number_of_launch > 1
                );
            }
            else
            {
                if(debug_synchronous)
                {
                    std::cout << "use_limited_size " << use_limited_size << '\n';
                    std::cout << "aligned_size_limit " << aligned_size_limit << '\n';
                    std::cout << "size " << current_size << '\n';
                    std::cout << "block_size " << block_size << '\n';
                    std::cout << "number of blocks " << number_of_blocks << '\n';
                    std::cout << "items_per_block " << items_per_block << '\n';
                }

                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(lookback_scan_kernel<
                        Exclusive, // flag for exclusive scan operation
                        config, // kernel configuration (block size, ipt)
                        InputIterator, OutputIterator,
                        BinaryFunction, InitValueType, scan_state_type
                    >),
                    dim3(grid_size), dim3(block_size), 0, stream,
                    input + offset, output + offset, current_size, initial_value,
                    scan_op, scan_state, number_of_blocks, ordered_bid,
                    previous_last_element, new_last_element,
                    i != size_t(0), number_of_launch > 1
                );
            }
            ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("lookback_scan_kernel", current_size, start)

            // Swap the last_elements
            if(number_of_launch > 1)
            {
                hipError_t error = ::rocprim::transform(
                    new_last_element, previous_last_element, 1,
                    ::rocprim::identity<real_init_value_type>(),
                    stream, debug_synchronous
                );
                if(error != hipSuccess) return error;
            }
        }
    }
    else
    {
        if(debug_synchronous)
        {
            std::cout << "size " << size << '\n';
            std::cout << "block_size " << block_size << '\n';
            std::cout << "number of blocks " << number_of_blocks << '\n';
            std::cout << "items_per_block " << items_per_block << '\n';
        }

        if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(single_scan_kernel<
                Exclusive, // flag for exclusive scan operation
                config, // kernel configuration (block size, ipt)
                InputIterator, OutputIterator, BinaryFunction
            >),
            dim3(1), dim3(block_size), 0, stream,
            input, size, initial_value, output, scan_op
        );
        ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("single_scan_kernel", size, start);
    }
    return hipSuccess;
}

#undef ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR
#undef ROCPRIM_DETAIL_HIP_SYNC

} // end of detail namespace

/// \brief Parallel inclusive scan primitive for device level.
///
/// inclusive_scan function performs a device-wide inclusive prefix scan operation
/// using binary \p scan_op operator.
///
/// \par Overview
/// * Supports non-commutative scan operators. However, a scan operator should be
/// associative. When used with non-associative functions the results may be non-deterministic
/// and/or vary in precision.
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage in a null pointer.
/// * Ranges specified by \p input and \p output must have at least \p size elements.
///
/// \tparam Config - [optional] configuration of the primitive. It can be \p scan_config or
/// a custom class with the same members.
/// \tparam InputIterator - random-access iterator type of the input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam OutputIterator - random-access iterator type of the output range. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
/// \tparam BinaryFunction - type of binary function used for scan. Default type
/// is \p rocprim::plus<T>, where \p T is a \p value_type of \p InputIterator.
///
/// \param [in] temporary_storage - pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the scan operation.
/// \param [in,out] storage_size - reference to a size (in bytes) of \p temporary_storage.
/// \param [in] input - iterator to the first element in the range to scan.
/// \param [out] output - iterator to the first element in the output range. It can be
/// same as \p input.
/// \param [in] size - number of element in the input range.
/// \param [in] scan_op - binary operation function object that will be used for scan.
/// The signature of the function should be equivalent to the following:
/// <tt>T f(const T &a, const T &b);</tt>. The signature does not need to have
/// <tt>const &</tt>, but function object must not modify the objects passed to it.
/// Default is BinaryFunction().
/// \param [in] stream - [optional] HIP stream object. Default is \p 0 (default stream).
/// \param [in] debug_synchronous - [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. Default value is \p false.
///
/// \returns \p hipSuccess (\p 0) after successful scan; otherwise a HIP runtime error of
/// type \p hipError_t.
///
/// \par Example
/// \parblock
/// In this example a device-level inclusive sum operation is performed on an array of
/// integer values (<tt>short</tt>s are scanned into <tt>int</tt>s).
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// size_t input_size;    // e.g., 8
/// short * input;        // e.g., [1, 2, 3, 4, 5, 6, 7, 8]
/// int * output;         // empty array of 8 elements
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::inclusive_scan(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input, output, input_size, rocprim::plus<int>()
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // perform scan
/// rocprim::inclusive_scan(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input, output, input_size, rocprim::plus<int>()
/// );
/// // output: [1, 3, 6, 10, 15, 21, 28, 36]
/// \endcode
/// \endparblock
template<
    class Config = default_config,
    class InputIterator,
    class OutputIterator,
    class BinaryFunction = ::rocprim::plus<typename std::iterator_traits<InputIterator>::value_type>
>
inline
hipError_t inclusive_scan(void * temporary_storage,
                          size_t& storage_size,
                          InputIterator input,
                          OutputIterator output,
                          const size_t size,
                          BinaryFunction scan_op = BinaryFunction(),
                          const hipStream_t stream = 0,
                          bool debug_synchronous = false)
{
    using input_type = typename std::iterator_traits<InputIterator>::value_type;

    // Get default config if Config is default_config
    using config = detail::default_or_custom_config<
        Config,
        detail::default_scan_config<ROCPRIM_TARGET_ARCH, input_type>
        >;

    return detail::scan_impl<false, config>(
        temporary_storage, storage_size,
        // input_type() is a dummy initial value (not used)
        input, output, input_type(), size,
        scan_op, stream, debug_synchronous
    );
}

/// \brief Parallel exclusive scan primitive for device level.
///
/// exclusive_scan function performs a device-wide exclusive prefix scan operation
/// using binary \p scan_op operator.
///
/// \par Overview
/// * Supports non-commutative scan operators. However, a scan operator should be
/// associative. When used with non-associative functions the results may be non-deterministic
/// and/or vary in precision.
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage in a null pointer.
/// * Ranges specified by \p input and \p output must have at least \p size elements.
///
/// \tparam Config - [optional] configuration of the primitive. It can be \p scan_config or
/// a custom class with the same members.
/// \tparam InputIterator - random-access iterator type of the input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam OutputIterator - random-access iterator type of the output range. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
/// \tparam InitValueType - type of the initial value.
/// \tparam BinaryFunction - type of binary function used for scan. Default type
/// is \p rocprim::plus<T>, where \p T is a \p value_type of \p InputIterator.
///
/// \param [in] temporary_storage - pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the scan operation.
/// \param [in,out] storage_size - reference to a size (in bytes) of \p temporary_storage.
/// \param [in] input - iterator to the first element in the range to scan.
/// \param [out] output - iterator to the first element in the output range. It can be
/// same as \p input.
/// \param [in] initial_value - initial value to start the scan.
/// A rocpim::future_value may be passed to use a value that will be later computed.
/// \param [in] size - number of element in the input range.
/// \param [in] scan_op - binary operation function object that will be used for scan.
/// The signature of the function should be equivalent to the following:
/// <tt>T f(const T &a, const T &b);</tt>. The signature does not need to have
/// <tt>const &</tt>, but function object must not modify the objects passed to it.
/// The default value is \p BinaryFunction().
/// \param [in] stream - [optional] HIP stream object. The default is \p 0 (default stream).
/// \param [in] debug_synchronous - [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. The default value is \p false.
///
/// \returns \p hipSuccess (\p 0) after successful scan; otherwise a HIP runtime error of
/// type \p hipError_t.
///
/// \par Example
/// \parblock
/// In this example a device-level exclusive min-scan operation is performed on an array of
/// integer values (<tt>short</tt>s are scanned into <tt>int</tt>s) using custom operator.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // custom scan function
/// auto min_op =
///     [] __device__ (int a, int b) -> int
///     {
///         return a < b ? a : b;
///     };
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// size_t input_size;    // e.g., 8
/// short * input;        // e.g., [4, 7, 6, 2, 5, 1, 3, 8]
/// int * output;         // empty array of 8 elements
/// int start_value;      // e.g., 9
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::exclusive_scan(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input, output, start_value, input_size, min_op
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // perform scan
/// rocprim::exclusive_scan(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input, output, start_value, input_size, min_op
/// );
/// // output: [9, 4, 7, 6, 2, 2, 1, 1]
/// \endcode
/// \endparblock
template<
    class Config = default_config,
    class InputIterator,
    class OutputIterator,
    class InitValueType,
    class BinaryFunction = ::rocprim::plus<typename std::iterator_traits<InputIterator>::value_type>
>
inline
hipError_t exclusive_scan(void * temporary_storage,
                          size_t& storage_size,
                          InputIterator input,
                          OutputIterator output,
                          const InitValueType initial_value,
                          const size_t size,
                          BinaryFunction scan_op = BinaryFunction(),
                          const hipStream_t stream = 0,
                          bool debug_synchronous = false)
{
    using real_init_value_type = detail::input_type_t<InitValueType>;

    // Get default config if Config is default_config
    using config = detail::default_or_custom_config<
        Config,
        detail::default_scan_config<ROCPRIM_TARGET_ARCH, real_init_value_type>
    >;

    return detail::scan_impl<true, config>(
        temporary_storage, storage_size,
        input, output, initial_value, size,
        scan_op, stream, debug_synchronous
    );
}

/// @}
// end of group devicemodule

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DEVICE_SCAN_HPP_
