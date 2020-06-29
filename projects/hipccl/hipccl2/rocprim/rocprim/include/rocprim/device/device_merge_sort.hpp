// Copyright (c) 2017-2019 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DEVICE_DEVICE_SORT_HPP_
#define ROCPRIM_DEVICE_DEVICE_SORT_HPP_

#include <type_traits>
#include <iterator>

#include "../config.hpp"
#include "../detail/various.hpp"

#include "detail/device_merge_sort.hpp"
#include "device_transform.hpp"
#include "device_merge_sort_config.hpp"

BEGIN_ROCPRIM_NAMESPACE

/// \addtogroup devicemodule
/// @{

namespace detail
{

template<
    unsigned int BlockSize,
    class KeysInputIterator,
    class KeysOutputIterator,
    class ValuesInputIterator,
    class ValuesOutputIterator,
    class BinaryFunction
>
__global__
__launch_bounds__(BlockSize, ROCPRIM_DEFAULT_MIN_WARPS_PER_EU)
void block_sort_kernel(KeysInputIterator keys_input,
                       KeysOutputIterator keys_output,
                       ValuesInputIterator values_input,
                       ValuesOutputIterator values_output,
                       const size_t size,
                       BinaryFunction compare_function)
{
    block_sort_kernel_impl<BlockSize>(
        keys_input, keys_output, values_input, values_output,
        size, compare_function
    );
}

template<
    class KeysInputIterator,
    class KeysOutputIterator,
    class ValuesInputIterator,
    class ValuesOutputIterator,
    class BinaryFunction
>
__global__
__launch_bounds__(ROCPRIM_DEFAULT_MAX_BLOCK_SIZE, ROCPRIM_DEFAULT_MIN_WARPS_PER_EU)
void block_merge_kernel(KeysInputIterator keys_input,
                        KeysOutputIterator keys_output,
                        ValuesInputIterator values_input,
                        ValuesOutputIterator values_output,
                        const size_t size,
                        unsigned int block_size,
                        BinaryFunction compare_function)
{
    block_merge_kernel_impl(
        keys_input, keys_output, values_input, values_output,
        size, block_size, compare_function
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
        auto error = hipPeekAtLastError(); \
        if(error != hipSuccess) return error; \
        if(debug_synchronous) \
        { \
            std::cout << name << "(" << size << ")"; \
            auto error = hipStreamSynchronize(stream); \
            if(error != hipSuccess) return error; \
            auto end = std::chrono::high_resolution_clock::now(); \
            auto d = std::chrono::duration_cast<std::chrono::duration<double>>(end - start); \
            std::cout << " " << d.count() * 1000 << " ms" << '\n'; \
        } \
    }

template<
    class Config,
    class KeysInputIterator,
    class KeysOutputIterator,
    class ValuesInputIterator,
    class ValuesOutputIterator,
    class BinaryFunction
>
inline
hipError_t merge_sort_impl(void * temporary_storage,
                           size_t& storage_size,
                           KeysInputIterator keys_input,
                           KeysOutputIterator keys_output,
                           ValuesInputIterator values_input,
                           ValuesOutputIterator values_output,
                           const size_t size,
                           BinaryFunction compare_function,
                           const hipStream_t stream,
                           bool debug_synchronous)
{
    using key_type = typename std::iterator_traits<KeysInputIterator>::value_type;
    using value_type = typename std::iterator_traits<ValuesInputIterator>::value_type;
    constexpr bool with_values = !std::is_same<value_type, ::rocprim::empty_type>::value;

    // Get default config if Config is default_config
    using config = default_or_custom_config<
        Config,
        default_merge_sort_config<ROCPRIM_TARGET_ARCH, key_type, value_type>
    >;

    // Block size
    constexpr unsigned int block_size = config::block_size;

    const size_t keys_bytes = ::rocprim::detail::align_size(size * sizeof(key_type));
    const size_t values_bytes =
        with_values ? ::rocprim::detail::align_size(size * sizeof(value_type)) : 0;
    if(temporary_storage == nullptr)
    {
        storage_size = keys_bytes;
        if(with_values)
        {
            storage_size += values_bytes;
        }
        // Make sure user won't try to allocate 0 bytes memory
        storage_size = storage_size == 0 ? 4 : storage_size;
        return hipSuccess;
    }

    auto number_of_blocks = (size + block_size - 1)/block_size;
    if(debug_synchronous)
    {
        std::cout << "block_size " << block_size << '\n';
        std::cout << "number of blocks " << number_of_blocks << '\n';
    }

    char* ptr = reinterpret_cast<char*>(temporary_storage);
    key_type * keys_buffer = reinterpret_cast<key_type*>(ptr);
    ptr += keys_bytes;
    value_type * values_buffer =
        with_values ? reinterpret_cast<value_type*>(ptr) : nullptr;

    // Start point for time measurements
    std::chrono::high_resolution_clock::time_point start;
    if(debug_synchronous) start = std::chrono::high_resolution_clock::now();

    const unsigned int grid_size = number_of_blocks;
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(block_sort_kernel<block_size>),
        dim3(grid_size), dim3(block_size), 0, stream,
        keys_input, keys_output, values_input, values_output,
        size, compare_function
    );
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("block_sort_kernel", size, start);

    bool temporary_store = false;
    for(unsigned int block = block_size; block < size; block *= 2)
    {
        temporary_store = !temporary_store;
        if(temporary_store)
        {
            if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(block_merge_kernel),
                dim3(grid_size), dim3(block_size), 0, stream,
                keys_output, keys_buffer, values_output, values_buffer,
                size, block, compare_function
            );
            ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("block_merge_kernel", size, start);
        }
        else
        {
            if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(block_merge_kernel),
                dim3(grid_size), dim3(block_size), 0, stream,
                keys_buffer, keys_output, values_buffer, values_output,
                size, block, compare_function
            );
            ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("block_merge_kernel", size, start);
        }
    }

    if(temporary_store)
    {
        hipError_t error = ::rocprim::transform(
            keys_buffer, keys_output, size,
            ::rocprim::identity<key_type>(), stream, debug_synchronous
        );
        if(error != hipSuccess) return error;

        if(with_values)
        {
            hipError_t error = ::rocprim::transform(
                values_buffer, values_output, size,
                ::rocprim::identity<value_type>(), stream, debug_synchronous
            );
            if(error != hipSuccess) return error;
        }
    }

    return hipSuccess;
}

#undef ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR
#undef ROCPRIM_DETAIL_HIP_SYNC

} // end of detail namespace

/// \brief Parallel merge sort primitive for device level.
///
/// \p merge_sort function performs a device-wide merge sort
/// of keys. Function sorts input keys based on comparison function.
///
/// \par Overview
/// * The contents of the inputs are not altered by the sorting function.
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage in a null pointer.
/// * Accepts custom compare_functions for sorting across the device.
///
/// \tparam KeysInputIterator - random-access iterator type of the input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam KeysOutputIterator - random-access iterator type of the output range. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
///
/// \param [in] temporary_storage - pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the sort operation.
/// \param [in,out] storage_size - reference to a size (in bytes) of \p temporary_storage.
/// \param [in] keys_input - pointer to the first element in the range to sort.
/// \param [out] keys_output - pointer to the first element in the output range.
/// \param [in] size - number of element in the input range.
/// \param [in] compare_function - binary operation function object that will be used for comparison.
/// The signature of the function should be equivalent to the following:
/// <tt>bool f(const T &a, const T &b);</tt>. The signature does not need to have
/// <tt>const &</tt>, but function object must not modify the objects passed to it.
/// The default value is \p BinaryFunction().
/// \param [in] stream - [optional] HIP stream object. Default is \p 0 (default stream).
/// \param [in] debug_synchronous - [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. Default value is \p false.
///
/// \returns \p hipSuccess (\p 0) after successful sort; otherwise a HIP runtime error of
/// type \p hipError_t.
///
/// \par Example
/// \parblock
/// In this example a device-level ascending merge sort is performed on an array of
/// \p float values.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// size_t input_size;      // e.g., 8
/// float * input;          // e.g., [0.6, 0.3, 0.65, 0.4, 0.2, 0.08, 1, 0.7]
/// float * output;         // empty array of 8 elements
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::merge_sort(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input, output, input_size
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // perform sort
/// rocprim::merge_sort(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input, output, input_size
/// );
/// // keys_output: [0.08, 0.2, 0.3, 0.4, 0.6, 0.65, 0.7, 1]
/// \endcode
/// \endparblock
template<
    class Config = default_config,
    class KeysInputIterator,
    class KeysOutputIterator,
    class BinaryFunction = ::rocprim::less<typename std::iterator_traits<KeysInputIterator>::value_type>
>
inline
hipError_t merge_sort(void * temporary_storage,
                      size_t& storage_size,
                      KeysInputIterator keys_input,
                      KeysOutputIterator keys_output,
                      const size_t size,
                      BinaryFunction compare_function = BinaryFunction(),
                      const hipStream_t stream = 0,
                      bool debug_synchronous = false)
{
    empty_type * values = nullptr;
    return detail::merge_sort_impl<Config>(
        temporary_storage, storage_size,
        keys_input, keys_output, values, values, size,
        compare_function, stream, debug_synchronous
    );
}

/// \brief Parallel ascending merge sort-by-key primitive for device level.
///
/// \p merge_sort function performs a device-wide merge sort
/// of (key, value) pairs. Function sorts input pairs based on comparison function.
///
/// \par Overview
/// * The contents of the inputs are not altered by the sorting function.
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage in a null pointer.
/// * Accepts custom compare_functions for sorting across the device.
///
/// \tparam KeysInputIterator - random-access iterator type of the input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam KeysOutputIterator - random-access iterator type of the output range. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
/// \tparam ValuesInputIterator - random-access iterator type of the input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam ValuesOutputIterator - random-access iterator type of the output range. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
///
/// \param [in] temporary_storage - pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the sort operation.
/// \param [in,out] storage_size - reference to a size (in bytes) of \p temporary_storage.
/// \param [in] keys_input - pointer to the first element in the range to sort.
/// \param [out] keys_output - pointer to the first element in the output range.
/// \param [in] values_input - pointer to the first element in the range to sort.
/// \param [out] values_output - pointer to the first element in the output range.
/// \param [in] size - number of element in the input range.
/// \param [in] compare_function - binary operation function object that will be used for comparison.
/// The signature of the function should be equivalent to the following:
/// <tt>bool f(const T &a, const T &b);</tt>. The signature does not need to have
/// <tt>const &</tt>, but function object must not modify the objects passed to it.
/// The default value is \p BinaryFunction().
/// \param [in] stream - [optional] HIP stream object. Default is \p 0 (default stream).
/// \param [in] debug_synchronous - [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. Default value is \p false.
///
/// \returns \p hipSuccess (\p 0) after successful sort; otherwise a HIP runtime error of
/// type \p hipError_t.
///
/// \par Example
/// \parblock
/// In this example a device-level ascending merge sort is performed where input keys are
/// represented by an array of unsigned integers and input values by an array of <tt>double</tt>s.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// size_t input_size;          // e.g., 8
/// unsigned int * keys_input;  // e.g., [ 6, 3,  5, 4,  1,  8,  2, 7]
/// double * values_input;      // e.g., [-5, 2, -4, 3, -1, -8, -2, 7]
/// unsigned int * keys_output; // empty array of 8 elements
/// double * values_output;     // empty array of 8 elements
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::merge_sort(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     keys_input, keys_output, values_input, values_output,
///     input_size
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // perform sort
/// rocprim::merge_sort(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     keys_input, keys_output, values_input, values_output,
///     input_size
/// );
/// // keys_output:   [ 1,  2, 3, 4,  5,  6, 7,  8]
/// // values_output: [-1, -2, 2, 3, -4, -5, 7, -8]
/// \endcode
/// \endparblock
template<
    class Config = default_config,
    class KeysInputIterator,
    class KeysOutputIterator,
    class ValuesInputIterator,
    class ValuesOutputIterator,
    class BinaryFunction = ::rocprim::less<typename std::iterator_traits<KeysInputIterator>::value_type>
>
inline
hipError_t merge_sort(void * temporary_storage,
                      size_t& storage_size,
                      KeysInputIterator keys_input,
                      KeysOutputIterator keys_output,
                      ValuesInputIterator values_input,
                      ValuesOutputIterator values_output,
                      const size_t size,
                      BinaryFunction compare_function = BinaryFunction(),
                      const hipStream_t stream = 0,
                      bool debug_synchronous = false)
{
    return detail::merge_sort_impl<Config>(
        temporary_storage, storage_size,
        keys_input, keys_output, values_input, values_output, size,
        compare_function, stream, debug_synchronous
    );
}

/// @}
// end of group devicemodule

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DEVICE_SORT_HPP_
