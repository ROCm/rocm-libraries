// Copyright (c) 2017-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <iostream>
#include <iterator>
#include <type_traits>

#include "../config.hpp"
#include "../detail/temp_storage.hpp"
#include "../detail/various.hpp"
#include "../functional.hpp"
#include "../type_traits.hpp"
#include "../types/future_value.hpp"
#include "detail/config/device_scan.hpp"
#include "detail/device_scan.hpp"
#include "detail/device_scan_common.hpp"
#include "device_scan_config.hpp"
#include "device_transform.hpp"

BEGIN_ROCPRIM_NAMESPACE

/// \addtogroup devicemodule
/// @{

namespace detail
{

// Single kernel scan (performs scan on one thread block only)
template<bool Exclusive,
         class Config,
         class InputIterator,
         class OutputIterator,
         class BinaryFunction,
         class ResultType>
ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE void single_scan_kernel_impl(InputIterator  input,
                                                                 const size_t   input_size,
                                                                 ResultType     initial_value,
                                                                 OutputIterator output,
                                                                 BinaryFunction scan_op)
{
    static constexpr scan_config_params params = device_params<Config>();

    constexpr unsigned int block_size       = params.kernel_config.block_size;
    constexpr unsigned int items_per_thread = params.kernel_config.items_per_thread;

    using result_type = ResultType;

    using block_load_type = ::rocprim::
        block_load<result_type, block_size, items_per_thread, params.block_load_method>;
    using block_store_type = ::rocprim::
        block_store<result_type, block_size, items_per_thread, params.block_store_method>;
    using block_scan_type
        = ::rocprim::block_scan<result_type, block_size, params.block_scan_method>;

    ROCPRIM_SHARED_MEMORY union
    {
        typename block_load_type::storage_type  load;
        typename block_store_type::storage_type store;
        typename block_scan_type::storage_type  scan;
    } storage;

    result_type values[items_per_thread];
    // load input values into values
    block_load_type().load(input, values, input_size, *(input), storage.load);
    ::rocprim::syncthreads(); // sync threads to reuse shared memory

    single_scan_block_scan<Exclusive, block_scan_type>(values, // input
                                                       values, // output
                                                       initial_value,
                                                       storage.scan,
                                                       scan_op);
    ::rocprim::syncthreads(); // sync threads to reuse shared memory

    // Save values into output array
    block_store_type().store(output, values, input_size, storage.store);
}

template<bool Exclusive,
         class Config,
         class InputIterator,
         class OutputIterator,
         class BinaryFunction,
         class InitValueType>
ROCPRIM_KERNEL
    __launch_bounds__(device_params<Config>().kernel_config.block_size) void single_scan_kernel(
        InputIterator       input,
        const size_t        size,
        const InitValueType initial_value,
        OutputIterator      output,
        BinaryFunction      scan_op)
{
    single_scan_kernel_impl<Exclusive, Config>(input,
                                               size,
                                               get_input_value(initial_value),
                                               output,
                                               scan_op);
}

// Single pass (look-back kernels)

template<bool Exclusive,
         class Config,
         class InputIterator,
         class OutputIterator,
         class BinaryFunction,
         class InitValueType,
         class LookBackScanState>
ROCPRIM_KERNEL
    __launch_bounds__(device_params<Config>().kernel_config.block_size) void lookback_scan_kernel(
        InputIterator                input,
        OutputIterator               output,
        const size_t                 size,
        const InitValueType          initial_value,
        BinaryFunction               scan_op,
        LookBackScanState            lookback_scan_state,
        const unsigned int           number_of_blocks,
        input_type_t<InitValueType>* previous_last_element = nullptr,
        input_type_t<InitValueType>* new_last_element      = nullptr,
        bool                         override_first_value  = false,
        bool                         save_last_value       = false)
{
    lookback_scan_kernel_impl<Exclusive, Config>(input,
                                                 output,
                                                 size,
                                                 get_input_value(initial_value),
                                                 scan_op,
                                                 lookback_scan_state,
                                                 number_of_blocks,
                                                 previous_last_element,
                                                 new_last_element,
                                                 override_first_value,
                                                 save_last_value);
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

template<bool Exclusive,
         class Config,
         class InputIterator,
         class OutputIterator,
         class InitValueType,
         class BinaryFunction>
inline auto scan_impl(void*               temporary_storage,
                      size_t&             storage_size,
                      InputIterator       input,
                      OutputIterator      output,
                      const InitValueType initial_value,
                      const size_t        size,
                      BinaryFunction      scan_op,
                      const hipStream_t   stream,
                      bool                debug_synchronous)
{
    using real_init_value_type = input_type_t<InitValueType>;

    using config = wrapped_scan_config<Config, real_init_value_type>;

    detail::target_arch target_arch;
    hipError_t          result = host_target_arch(stream, target_arch);
    if(result != hipSuccess)
    {
        return result;
    }
    const scan_config_params params = dispatch_target_arch<config>(target_arch);

    using scan_state_type            = detail::lookback_scan_state<real_init_value_type>;
    using scan_state_with_sleep_type = detail::lookback_scan_state<real_init_value_type, true>;

    const unsigned int block_size       = params.kernel_config.block_size;
    const unsigned int items_per_thread = params.kernel_config.items_per_thread;
    const auto         items_per_block  = block_size * items_per_thread;

    const size_t size_limit = params.kernel_config.size_limit;
    const size_t aligned_size_limit
        = ::rocprim::max<size_t>(size_limit - size_limit % items_per_block, items_per_block);
    size_t limited_size = std::min<size_t>(size, aligned_size_limit);
    const bool use_limited_size = limited_size == aligned_size_limit;

    unsigned int number_of_blocks = (limited_size + items_per_block - 1)/items_per_block;

    // Pointer to array with block_prefixes
    void*                 scan_state_storage;
    real_init_value_type* previous_last_element;
    real_init_value_type* new_last_element;

    const hipError_t partition_result = detail::temp_storage::partition(
        temporary_storage,
        storage_size,
        detail::temp_storage::make_linear_partition(
            // This is valid even with offset_scan_state_with_sleep_type
            detail::temp_storage::make_partition(
                &scan_state_storage,
                scan_state_type::get_temp_storage_layout(number_of_blocks)),
            detail::temp_storage::ptr_aligned_array(&previous_last_element,
                                                    use_limited_size ? 1 : 0),
            detail::temp_storage::ptr_aligned_array(&new_last_element, use_limited_size ? 1 : 0)));
    if(partition_result != hipSuccess || temporary_storage == nullptr)
    {
        return partition_result;
    }

    // Start point for time measurements
    std::chrono::high_resolution_clock::time_point start;

    if( number_of_blocks == 0u )
        return hipSuccess;

    if(number_of_blocks > 1 || use_limited_size)
    {
        // Create and initialize lookback_scan_state obj
        auto scan_state = scan_state_type::create(scan_state_storage, number_of_blocks);
        auto scan_state_with_sleep
            = scan_state_with_sleep_type::create(scan_state_storage, number_of_blocks);

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

            if(std::string(prop.gcnArchName).find("908") != std::string::npos && asicRevision < 2)
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(init_lookback_scan_state_kernel<scan_state_with_sleep_type>),
                    dim3(grid_size),
                    dim3(block_size),
                    0,
                    stream,
                    scan_state_with_sleep,
                    number_of_blocks);
            } else
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(init_lookback_scan_state_kernel<scan_state_type>),
                    dim3(grid_size),
                    dim3(block_size),
                    0,
                    stream,
                    scan_state,
                    number_of_blocks);
            }
            ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("init_lookback_scan_state_kernel", number_of_blocks, start)

            if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
            grid_size = number_of_blocks;
            if(std::string(prop.gcnArchName).find("908") != std::string::npos && asicRevision < 2)
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(
                        lookback_scan_kernel<Exclusive, // flag for exclusive scan operation
                                             config,
                                             InputIterator,
                                             OutputIterator,
                                             BinaryFunction,
                                             InitValueType,
                                             scan_state_with_sleep_type>),
                    dim3(grid_size),
                    dim3(block_size),
                    0,
                    stream,
                    input + offset,
                    output + offset,
                    current_size,
                    initial_value,
                    scan_op,
                    scan_state_with_sleep,
                    number_of_blocks,
                    previous_last_element,
                    new_last_element,
                    i != size_t(0),
                    number_of_launch > 1);
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
                    HIP_KERNEL_NAME(
                        lookback_scan_kernel<Exclusive, // flag for exclusive scan operation
                                             config,
                                             InputIterator,
                                             OutputIterator,
                                             BinaryFunction,
                                             InitValueType,
                                             scan_state_type>),
                    dim3(grid_size),
                    dim3(block_size),
                    0,
                    stream,
                    input + offset,
                    output + offset,
                    current_size,
                    initial_value,
                    scan_op,
                    scan_state,
                    number_of_blocks,
                    previous_last_element,
                    new_last_element,
                    i != size_t(0),
                    number_of_launch > 1);
            }
            ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("lookback_scan_kernel", current_size, start)

            // Swap the last_elements
            if(number_of_launch > 1)
            {
                hipError_t error = ::rocprim::transform(new_last_element,
                                                        previous_last_element,
                                                        1,
                                                        ::rocprim::identity<real_init_value_type>(),
                                                        stream,
                                                        debug_synchronous);
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
            HIP_KERNEL_NAME(single_scan_kernel<Exclusive, // flag for exclusive scan operation
                                               config,
                                               InputIterator,
                                               OutputIterator,
                                               BinaryFunction>),
            dim3(1),
            dim3(block_size),
            0,
            stream,
            input,
            size,
            initial_value,
            output,
            scan_op);
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
/// * By default, the input type is used for accumulation. A custom type
/// can be specified using <tt>rocprim::transform_iterator</tt>, see the example below.
///
/// \tparam Config - [optional] configuration of the primitive, should be \p scan_config_v2.
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
///
/// The same example as above, but now a custom accumulator type is specified.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// size_t input_size;
/// short * input;
/// int * output;
///
/// // Use a transform iterator to specify a custom accumulator type
/// auto input_iterator = rocprim::make_transform_iterator(
///     input, [] __device__ (T in) { return static_cast<int>(in); });
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Use the transform iterator
/// rocprim::inclusive_scan(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input_iterator, output, input_size, rocprim::plus<int>()
/// );
///
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// rocprim::inclusive_scan(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input_iterator, output, input_size, rocprim::plus<int>()
/// );
/// \endcode
/// \endparblock
template<class Config = default_config,
         class InputIterator,
         class OutputIterator,
         class BinaryFunction
         = ::rocprim::plus<typename std::iterator_traits<InputIterator>::value_type>>
inline hipError_t inclusive_scan(void*             temporary_storage,
                                 size_t&           storage_size,
                                 InputIterator     input,
                                 OutputIterator    output,
                                 const size_t      size,
                                 BinaryFunction    scan_op           = BinaryFunction(),
                                 const hipStream_t stream            = 0,
                                 bool              debug_synchronous = false)
{
    using input_type = typename std::iterator_traits<InputIterator>::value_type;
    // input_type() is a dummy initial value (not used)
    return detail::scan_impl<false, Config>(temporary_storage,
                                            storage_size,
                                            input,
                                            output,
                                            input_type(),
                                            size,
                                            scan_op,
                                            stream,
                                            debug_synchronous);
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
/// \tparam Config - [optional] configuration of the primitive, should be \p scan_config_v2.
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
template<class Config = default_config,
         class InputIterator,
         class OutputIterator,
         class InitValueType,
         class BinaryFunction
         = ::rocprim::plus<typename std::iterator_traits<InputIterator>::value_type>>
inline hipError_t exclusive_scan(void*               temporary_storage,
                                 size_t&             storage_size,
                                 InputIterator       input,
                                 OutputIterator      output,
                                 const InitValueType initial_value,
                                 const size_t        size,
                                 BinaryFunction      scan_op           = BinaryFunction(),
                                 const hipStream_t   stream            = 0,
                                 bool                debug_synchronous = false)
{
    return detail::scan_impl<true, Config>(temporary_storage,
                                           storage_size,
                                           input,
                                           output,
                                           initial_value,
                                           size,
                                           scan_op,
                                           stream,
                                           debug_synchronous);
}

/// @}
// end of group devicemodule

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DEVICE_SCAN_HPP_
