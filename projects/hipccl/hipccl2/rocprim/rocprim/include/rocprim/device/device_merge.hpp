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

#ifndef ROCPRIM_DEVICE_DEVICE_MERGE_HPP_
#define ROCPRIM_DEVICE_DEVICE_MERGE_HPP_

#include <type_traits>
#include <iterator>

#include "../config.hpp"
#include "../detail/various.hpp"

#include "device_merge_config.hpp"
#include "detail/device_merge.hpp"

BEGIN_ROCPRIM_NAMESPACE

/// \addtogroup devicemodule
/// @{

namespace detail
{

template<
    class IndexIterator,
    class KeysInputIterator1,
    class KeysInputIterator2,
    class BinaryFunction
>
__global__
__launch_bounds__(ROCPRIM_DEFAULT_MAX_BLOCK_SIZE, ROCPRIM_DEFAULT_MIN_WARPS_PER_EU)
void partition_kernel(IndexIterator index,
                      KeysInputIterator1 keys_input1,
                      KeysInputIterator2 keys_input2,
                      const size_t input1_size,
                      const size_t input2_size,
                      const unsigned int spacing,
                      BinaryFunction compare_function)
{
    partition_kernel_impl(
        index, keys_input1, keys_input2, input1_size, input2_size,
        spacing, compare_function
    );
}

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class IndexIterator,
    class KeysInputIterator1,
    class KeysInputIterator2,
    class KeysOutputIterator,
    class ValuesInputIterator1,
    class ValuesInputIterator2,
    class ValuesOutputIterator,
    class BinaryFunction
>
__global__
__launch_bounds__(BlockSize, ROCPRIM_DEFAULT_MIN_WARPS_PER_EU)
void merge_kernel(IndexIterator index,
                  KeysInputIterator1 keys_input1,
                  KeysInputIterator2 keys_input2,
                  KeysOutputIterator keys_output,
                  ValuesInputIterator1 values_input1,
                  ValuesInputIterator2 values_input2,
                  ValuesOutputIterator values_output,
                  const size_t input1_size,
                  const size_t input2_size,
                  BinaryFunction compare_function)
{
    merge_kernel_impl<BlockSize, ItemsPerThread>(
        index, keys_input1, keys_input2, keys_output,
        values_input1, values_input2, values_output,
        input1_size, input2_size, compare_function
    );
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
    class KeysInputIterator1,
    class KeysInputIterator2,
    class KeysOutputIterator,
    class ValuesInputIterator1,
    class ValuesInputIterator2,
    class ValuesOutputIterator,
    class BinaryFunction
>
inline
hipError_t merge_impl(void * temporary_storage,
                      size_t& storage_size,
                      KeysInputIterator1 keys_input1,
                      KeysInputIterator2 keys_input2,
                      KeysOutputIterator keys_output,
                      ValuesInputIterator1 values_input1,
                      ValuesInputIterator2 values_input2,
                      ValuesOutputIterator values_output,
                      const size_t input1_size,
                      const size_t input2_size,
                      BinaryFunction compare_function,
                      const hipStream_t stream,
                      bool debug_synchronous)

{
    using key_type = typename std::iterator_traits<KeysInputIterator1>::value_type;
    using value_type = typename std::iterator_traits<ValuesInputIterator1>::value_type;

    // Get default config if Config is default_config
    using config = detail::default_or_custom_config<
        Config,
        detail::default_merge_config<ROCPRIM_TARGET_ARCH, key_type, value_type>
    >;

    constexpr unsigned int block_size = config::block_size;
    constexpr unsigned int half_block = block_size / 2;
    constexpr unsigned int items_per_thread = config::items_per_thread;
    constexpr auto items_per_block = block_size * items_per_thread;

    const unsigned int partitions = ((input1_size + input2_size) + items_per_block - 1) / items_per_block;
    const size_t partition_bytes = (partitions + 1) * sizeof(unsigned int);

    if(temporary_storage == nullptr)
    {
        // storage_size is never zero
        storage_size = partition_bytes;
        return hipSuccess;
    }

    // Start point for time measurements
    std::chrono::high_resolution_clock::time_point start;

    auto number_of_blocks = partitions;
    if(debug_synchronous)
    {
        std::cout << "block_size " << block_size << '\n';
        std::cout << "number of blocks " << number_of_blocks << '\n';
        std::cout << "items_per_block " << items_per_block << '\n';
    }

    unsigned int * index = reinterpret_cast<unsigned int *>(temporary_storage);

    const unsigned partition_blocks = ((partitions + 1) + half_block - 1) / half_block;

    if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(detail::partition_kernel),
        dim3(partition_blocks), dim3(half_block), 0, stream,
        index, keys_input1, keys_input2, input1_size, input2_size,
        items_per_block, compare_function
    );
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("partition_kernel", input1_size, start);

    if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(detail::merge_kernel<block_size, items_per_thread>),
        dim3(number_of_blocks), dim3(block_size), 0, stream,
        index, keys_input1, keys_input2, keys_output,
        values_input1, values_input2, values_output,
        input1_size, input2_size, compare_function
    );
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("merge_kernel", input1_size, start);

    return hipSuccess;
}

#undef ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR

} // end of detail namespace

/// \brief Parallel merge primitive for device level.
///
/// \p merge function performs a device-wide merge.
/// Function merges two ordered sets of input values based on comparison function.
///
/// \par Overview
/// * The contents of the inputs are not altered by the merging function.
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage in a null pointer.
/// * Accepts custom compare_functions for merging across the device.
///
/// \tparam Config - [optional] configuration of the primitive. It can be \p merge_config or
/// a custom class with the same members.
/// \tparam InputIterator1 - random-access iterator type of the first input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam InputIterator2 - random-access iterator type of the second input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam OutputIterator - random-access iterator type of the output range. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
///
/// \param [in] temporary_storage - pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the sort operation.
/// \param [in,out] storage_size - reference to a size (in bytes) of \p temporary_storage.
/// \param [in] input1 - iterator to the first element in the first range to merge.
/// \param [in] input2 - iterator to the first element in the second range to merge.
/// \param [out] output - iterator to the first element in the output range.
/// \param [in] input1_size - number of element in the first input range.
/// \param [in] input2_size - number of element in the second input range.
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
/// In this example a device-level ascending merge is performed on an array of
/// \p int values.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// size_t input_size1;     // e.g., 4
/// size_t input_size2;     // e.g., 4
/// int * input1;           // e.g., [0, 1, 2, 3]
/// int * input2;           // e.g., [0, 1, 2, 3]
/// int * output;           // empty array of 8 elements
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::merge(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input1, input2, output, input_size1, input_size2
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // perform merge
/// rocprim::merge(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input1, input2, output, input_size1, input_size2
/// );
/// // output: [0, 0, 1, 1, 2, 2, 3, 3]
/// \endcode
/// \endparblock
template<
    class Config = default_config,
    class InputIterator1,
    class InputIterator2,
    class OutputIterator,
    class BinaryFunction = ::rocprim::less<typename std::iterator_traits<InputIterator1>::value_type>
>
inline
hipError_t merge(void * temporary_storage,
                 size_t& storage_size,
                 InputIterator1 input1,
                 InputIterator2 input2,
                 OutputIterator output,
                 const size_t input1_size,
                 const size_t input2_size,
                 BinaryFunction compare_function = BinaryFunction(),
                 const hipStream_t stream = 0,
                 bool debug_synchronous = false)
{
    empty_type * values = nullptr;
    return detail::merge_impl<Config>(
        temporary_storage, storage_size,
        input1, input2, output,
        values, values, values,
        input1_size, input2_size, compare_function,
        stream, debug_synchronous
    );
}

/// \brief Parallel merge primitive for device level.
///
/// \p merge function performs a device-wide merge of (key, value) pairs.
/// Function merges two ordered sets of input keys and corresponding values
/// based on key comparison function.
///
/// \par Overview
/// * The contents of the inputs are not altered by the merging function.
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage in a null pointer.
/// * Accepts custom compare_functions for merging across the device.
///
/// \tparam Config - [optional] configuration of the primitive. It can be \p merge_config or
/// a custom class with the same members.
/// \tparam KeysInputIterator1 - random-access iterator type of the first keys input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam KeysInputIterator2 - random-access iterator type of the second keys input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam KeysOutputIterator - random-access iterator type of the keys output range. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
/// \tparam ValuesInputIterator1 - random-access iterator type of the first values input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam ValuesInputIterator2 - random-access iterator type of the second values input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam ValuesOutputIterator - random-access iterator type of the values output range. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
///
/// \param [in] temporary_storage - pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the sort operation.
/// \param [in,out] storage_size - reference to a size (in bytes) of \p temporary_storage.
/// \param [in] keys_input1 - iterator to the first key in the first range to merge.
/// \param [in] keys_input2 - iterator to the first key in the second range to merge.
/// \param [out] keys_output - iterator to the first key in the output range.
/// \param [in] values_input1 - iterator to the first value in the first range to merge.
/// \param [in] values_input2 - iterator to the first value in the second range to merge.
/// \param [out] values_output - iterator to the first value in the output range.
/// \param [in] input1_size - number of element in the first input range.
/// \param [in] input2_size - number of element in the second input range.
/// \param [in] compare_function - binary operation function object that will be used for key comparison.
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
/// In this example a device-level ascending merge is performed on an array of
/// \p int values.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// size_t input_size1;     // e.g., 4
/// size_t input_size2;     // e.g., 4
/// int * keys_input1;      // e.g., [0, 1, 2, 3]
/// int * keys_input2;      // e.g., [0, 1, 2, 3]
/// int * keys_output;      // empty array of 8 elements
/// int * values_input1;    // e.g., [10, 11, 12, 13]
/// int * values_input2;    // e.g., [20, 21, 22, 23]
/// int * values_output;    // empty array of 8 elements
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::merge(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     keys_input1, keys_input2, keys_output,
///     values_input1, values_input2, values_output,
//      input_size1, input_size2
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // perform merge
/// rocprim::merge(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     keys_input1, keys_input2, keys_output,
///     values_input1, values_input2, values_output,
//      input_size1, input_size2
/// );
/// // keys_output: [0, 0, 1, 1, 2, 2, 3, 3]
/// // values_output: [10, 20, 11, 21, 12, 22, 13, 23]
/// \endcode
/// \endparblock
template<
    class Config = default_config,
    class KeysInputIterator1,
    class KeysInputIterator2,
    class KeysOutputIterator,
    class ValuesInputIterator1,
    class ValuesInputIterator2,
    class ValuesOutputIterator,
    class BinaryFunction = ::rocprim::less<typename std::iterator_traits<KeysInputIterator1>::value_type>
>
inline
hipError_t merge(void * temporary_storage,
                 size_t& storage_size,
                 KeysInputIterator1 keys_input1,
                 KeysInputIterator2 keys_input2,
                 KeysOutputIterator keys_output,
                 ValuesInputIterator1 values_input1,
                 ValuesInputIterator2 values_input2,
                 ValuesOutputIterator values_output,
                 const size_t input1_size,
                 const size_t input2_size,
                 BinaryFunction compare_function = BinaryFunction(),
                 const hipStream_t stream = 0,
                 bool debug_synchronous = false)
{
    return detail::merge_impl<Config>(
        temporary_storage, storage_size,
        keys_input1, keys_input2, keys_output,
        values_input1, values_input2, values_output,
        input1_size, input2_size, compare_function,
        stream, debug_synchronous
    );
}

/// @}
// end of group devicemodule

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DEVICE_MERGE_HPP_
