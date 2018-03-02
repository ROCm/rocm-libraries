// Copyright (c) 2017 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DEVICE_DEVICE_SEGMENTED_RADIX_SORT_HIP_HPP_
#define ROCPRIM_DEVICE_DEVICE_SEGMENTED_RADIX_SORT_HIP_HPP_

#include <iostream>
#include <iterator>
#include <type_traits>
#include <utility>

#include "../config.hpp"
#include "../detail/various.hpp"
#include "../detail/radix_sort.hpp"

#include "../intrinsics.hpp"
#include "../functional.hpp"
#include "../types.hpp"

#include "detail/device_segmented_radix_sort.hpp"

/// \addtogroup devicemodule_hip
/// @{

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    unsigned int RadixBits,
    bool Descending,
    class KeysInputIterator,
    class KeysOutputIterator,
    class ValuesInputIterator,
    class ValuesOutputIterator,
    class OffsetIterator
>
__global__
void segmented_sort_kernel(KeysInputIterator keys_input,
                           KeysOutputIterator keys_output,
                           ValuesInputIterator values_input,
                           ValuesOutputIterator values_output,
                           OffsetIterator begin_offsets,
                           OffsetIterator end_offsets,
                           unsigned int bit,
                           unsigned int current_radix_bits)
{
    segmented_sort<BlockSize, ItemsPerThread, RadixBits, Descending>(
        keys_input, keys_output, values_input, values_output,
        begin_offsets, end_offsets,
        bit, current_radix_bits
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
    bool Descending,
    class KeysInputIterator,
    class KeysOutputIterator,
    class ValuesInputIterator,
    class ValuesOutputIterator,
    class OffsetIterator
>
inline
hipError_t segmented_radix_sort_impl(void * temporary_storage,
                                     size_t& temporary_storage_bytes,
                                     KeysInputIterator keys_input,
                                     typename std::iterator_traits<KeysInputIterator>::value_type * keys_tmp,
                                     KeysOutputIterator keys_output,
                                     ValuesInputIterator values_input,
                                     typename std::iterator_traits<ValuesInputIterator>::value_type * values_tmp,
                                     ValuesOutputIterator values_output,
                                     unsigned int size,
                                     bool& is_result_in_output,
                                     unsigned int segments,
                                     OffsetIterator begin_offsets,
                                     OffsetIterator end_offsets,
                                     unsigned int begin_bit,
                                     unsigned int end_bit,
                                     hipStream_t stream,
                                     bool debug_synchronous)
{
    constexpr unsigned int radix_bits = 8;

    using key_type = typename std::iterator_traits<KeysInputIterator>::value_type;
    using value_type = typename std::iterator_traits<ValuesInputIterator>::value_type;

    constexpr bool with_values = !std::is_same<value_type, ::rocprim::empty_type>::value;

    constexpr unsigned int block_size = 256;
    constexpr unsigned int items_per_thread = 11;

    const unsigned int iterations = ::rocprim::detail::ceiling_div(end_bit - begin_bit, radix_bits);
    const bool with_double_buffer = keys_tmp != nullptr;

    const size_t keys_bytes = ::rocprim::detail::align_size(size * sizeof(key_type));
    const size_t values_bytes = with_values ? ::rocprim::detail::align_size(size * sizeof(value_type)) : 0;
    if(temporary_storage == nullptr)
    {
        if(!with_double_buffer)
        {
            temporary_storage_bytes = keys_bytes + values_bytes;
        }
        else
        {
            temporary_storage_bytes = 4;
        }
        return hipSuccess;
    }

    if(debug_synchronous)
    {
        std::cout << "iterations " << iterations << '\n';
        hipError_t error = hipStreamSynchronize(stream);
        if(error != hipSuccess) return error;
    }

    char * ptr = reinterpret_cast<char *>(temporary_storage);
    if(!with_double_buffer)
    {
        keys_tmp = reinterpret_cast<key_type *>(ptr);
        ptr += keys_bytes;
        values_tmp = with_values ? reinterpret_cast<value_type *>(ptr) : nullptr;
    }

    bool to_output = with_double_buffer || (iterations - 1) % 2 == 0;
    for(unsigned int bit = begin_bit; bit < end_bit; bit += radix_bits)
    {
        // Handle cases when (end_bit - bit) is not divisible by radix_bits, i.e. the last
        // iteration has a shorter mask.
        const unsigned int current_radix_bits = ::rocprim::min(radix_bits, end_bit - bit);

        const bool is_first_iteration = (bit == begin_bit);

        std::chrono::high_resolution_clock::time_point start;

        if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
        if(is_first_iteration)
        {
            if(to_output)
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(segmented_sort_kernel<
                        block_size, items_per_thread, radix_bits, Descending
                    >),
                    dim3(segments), dim3(block_size), 0, stream,
                    keys_input, keys_output, values_input, values_output,
                    begin_offsets, end_offsets,
                    bit, current_radix_bits
                );
            }
            else
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(segmented_sort_kernel<
                        block_size, items_per_thread, radix_bits, Descending
                    >),
                    dim3(segments), dim3(block_size), 0, stream,
                    keys_input, keys_tmp, values_input, values_tmp,
                    begin_offsets, end_offsets,
                    bit, current_radix_bits
                );
            }
        }
        else
        {
            if(to_output)
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(segmented_sort_kernel<
                        block_size, items_per_thread, radix_bits, Descending
                    >),
                    dim3(segments), dim3(block_size), 0, stream,
                    keys_tmp, keys_output, values_tmp, values_output,
                    begin_offsets, end_offsets,
                    bit, current_radix_bits
                );
            }
            else
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(segmented_sort_kernel<
                        block_size, items_per_thread, radix_bits, Descending
                    >),
                    dim3(segments), dim3(block_size), 0, stream,
                    keys_output, keys_tmp, values_output, values_tmp,
                    begin_offsets, end_offsets,
                    bit, current_radix_bits
                );
            }
        }
        ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("segmented_sort", segments, start)

        is_result_in_output = to_output;
        to_output = !to_output;
    }

    return hipSuccess;
}

#undef ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR

} // end namespace detail

template<
    class KeysInputIterator,
    class KeysOutputIterator,
    class OffsetIterator,
    class Key = typename std::iterator_traits<KeysInputIterator>::value_type
>
inline
hipError_t segmented_radix_sort_keys(void * temporary_storage,
                                     size_t& temporary_storage_bytes,
                                     KeysInputIterator keys_input,
                                     KeysOutputIterator keys_output,
                                     unsigned int size,
                                     unsigned int segments,
                                     OffsetIterator begin_offsets,
                                     OffsetIterator end_offsets,
                                     unsigned int begin_bit = 0,
                                     unsigned int end_bit = 8 * sizeof(Key),
                                     hipStream_t stream = 0,
                                     bool debug_synchronous = false)
{
    empty_type * values = nullptr;
    bool ignored;
    return detail::segmented_radix_sort_impl<false>(
        temporary_storage, temporary_storage_bytes,
        keys_input, nullptr, keys_output,
        values, nullptr, values,
        size, ignored,
        segments, begin_offsets, end_offsets,
        begin_bit, end_bit,
        stream, debug_synchronous
    );
}

template<
    class KeysInputIterator,
    class KeysOutputIterator,
    class OffsetIterator,
    class Key = typename std::iterator_traits<KeysInputIterator>::value_type
>
inline
hipError_t segmented_radix_sort_keys_desc(void * temporary_storage,
                                          size_t& temporary_storage_bytes,
                                          KeysInputIterator keys_input,
                                          KeysOutputIterator keys_output,
                                          unsigned int size,
                                          unsigned int segments,
                                          OffsetIterator begin_offsets,
                                          OffsetIterator end_offsets,
                                          unsigned int begin_bit = 0,
                                          unsigned int end_bit = 8 * sizeof(Key),
                                          hipStream_t stream = 0,
                                          bool debug_synchronous = false)
{
    empty_type * values = nullptr;
    bool ignored;
    return detail::segmented_radix_sort_impl<true>(
        temporary_storage, temporary_storage_bytes,
        keys_input, nullptr, keys_output,
        values, nullptr, values,
        size, ignored,
        segments, begin_offsets, end_offsets,
        begin_bit, end_bit,
        stream, debug_synchronous
    );
}

template<
    class KeysInputIterator,
    class KeysOutputIterator,
    class ValuesInputIterator,
    class ValuesOutputIterator,
    class OffsetIterator,
    class Key = typename std::iterator_traits<KeysInputIterator>::value_type
>
inline
hipError_t segmented_radix_sort_pairs(void * temporary_storage,
                                      size_t& temporary_storage_bytes,
                                      KeysInputIterator keys_input,
                                      KeysOutputIterator keys_output,
                                      ValuesInputIterator values_input,
                                      ValuesOutputIterator values_output,
                                      unsigned int size,
                                      unsigned int segments,
                                      OffsetIterator begin_offsets,
                                      OffsetIterator end_offsets,
                                      unsigned int begin_bit = 0,
                                      unsigned int end_bit = 8 * sizeof(Key),
                                      hipStream_t stream = 0,
                                      bool debug_synchronous = false)
{
    bool ignored;
    return detail::segmented_radix_sort_impl<false>(
        temporary_storage, temporary_storage_bytes,
        keys_input, nullptr, keys_output,
        values_input, nullptr, values_output,
        size, ignored,
        segments, begin_offsets, end_offsets,
        begin_bit, end_bit,
        stream, debug_synchronous
    );
}

template<
    class KeysInputIterator,
    class KeysOutputIterator,
    class ValuesInputIterator,
    class ValuesOutputIterator,
    class OffsetIterator,
    class Key = typename std::iterator_traits<KeysInputIterator>::value_type
>
inline
hipError_t segmented_radix_sort_pairs_desc(void * temporary_storage,
                                           size_t& temporary_storage_bytes,
                                           KeysInputIterator keys_input,
                                           KeysOutputIterator keys_output,
                                           ValuesInputIterator values_input,
                                           ValuesOutputIterator values_output,
                                           unsigned int size,
                                           unsigned int segments,
                                           OffsetIterator begin_offsets,
                                           OffsetIterator end_offsets,
                                           unsigned int begin_bit = 0,
                                           unsigned int end_bit = 8 * sizeof(Key),
                                           hipStream_t stream = 0,
                                           bool debug_synchronous = false)
{
    bool ignored;
    return detail::segmented_radix_sort_impl<true>(
        temporary_storage, temporary_storage_bytes,
        keys_input, nullptr, keys_output,
        values_input, nullptr, values_output,
        size, ignored,
        segments, begin_offsets, end_offsets,
        begin_bit, end_bit,
        stream, debug_synchronous
    );
}

template<class Key, class OffsetIterator>
inline
hipError_t segmented_radix_sort_keys(void * temporary_storage,
                                     size_t& temporary_storage_bytes,
                                     double_buffer<Key>& keys,
                                     unsigned int size,
                                     unsigned int segments,
                                     OffsetIterator begin_offsets,
                                     OffsetIterator end_offsets,
                                     unsigned int begin_bit = 0,
                                     unsigned int end_bit = 8 * sizeof(Key),
                                     hipStream_t stream = 0,
                                     bool debug_synchronous = false)
{
    empty_type * values = nullptr;
    bool is_result_in_output;
    hipError_t error = detail::segmented_radix_sort_impl<false>(
        temporary_storage, temporary_storage_bytes,
        keys.current(), keys.current(), keys.alternate(),
        values, values, values,
        size, is_result_in_output,
        segments, begin_offsets, end_offsets,
        begin_bit, end_bit,
        stream, debug_synchronous
    );
    if(temporary_storage != nullptr && is_result_in_output)
    {
        keys.swap();
    }
    return error;
}

template<class Key, class OffsetIterator>
inline
hipError_t segmented_radix_sort_keys_desc(void * temporary_storage,
                                          size_t& temporary_storage_bytes,
                                          double_buffer<Key>& keys,
                                          unsigned int size,
                                          unsigned int segments,
                                          OffsetIterator begin_offsets,
                                          OffsetIterator end_offsets,
                                          unsigned int begin_bit = 0,
                                          unsigned int end_bit = 8 * sizeof(Key),
                                          hipStream_t stream = 0,
                                          bool debug_synchronous = false)
{
    empty_type * values = nullptr;
    bool is_result_in_output;
    hipError_t error = detail::segmented_radix_sort_impl<true>(
        temporary_storage, temporary_storage_bytes,
        keys.current(), keys.current(), keys.alternate(),
        values, values, values,
        size, is_result_in_output,
        segments, begin_offsets, end_offsets,
        begin_bit, end_bit,
        stream, debug_synchronous
    );
    if(temporary_storage != nullptr && is_result_in_output)
    {
        keys.swap();
    }
    return error;
}

template<class Key, class Value, class OffsetIterator>
inline
hipError_t segmented_radix_sort_pairs(void * temporary_storage,
                                      size_t& temporary_storage_bytes,
                                      double_buffer<Key>& keys,
                                      double_buffer<Value>& values,
                                      unsigned int size,
                                      unsigned int segments,
                                      OffsetIterator begin_offsets,
                                      OffsetIterator end_offsets,
                                      unsigned int begin_bit = 0,
                                      unsigned int end_bit = 8 * sizeof(Key),
                                      hipStream_t stream = 0,
                                      bool debug_synchronous = false)
{
    bool is_result_in_output;
    hipError_t error = detail::segmented_radix_sort_impl<false>(
        temporary_storage, temporary_storage_bytes,
        keys.current(), keys.current(), keys.alternate(),
        values.current(), values.current(), values.alternate(),
        size, is_result_in_output,
        segments, begin_offsets, end_offsets,
        begin_bit, end_bit,
        stream, debug_synchronous
    );
    if(temporary_storage != nullptr && is_result_in_output)
    {
        keys.swap();
        values.swap();
    }
    return error;
}

template<class Key, class Value, class OffsetIterator>
inline
hipError_t segmented_radix_sort_pairs_desc(void * temporary_storage,
                                           size_t& temporary_storage_bytes,
                                           double_buffer<Key>& keys,
                                           double_buffer<Value>& values,
                                           unsigned int size,
                                           unsigned int segments,
                                           OffsetIterator begin_offsets,
                                           OffsetIterator end_offsets,
                                           unsigned int begin_bit = 0,
                                           unsigned int end_bit = 8 * sizeof(Key),
                                           hipStream_t stream = 0,
                                           bool debug_synchronous = false)
{
    bool is_result_in_output;
    hipError_t error = detail::segmented_radix_sort_impl<true>(
        temporary_storage, temporary_storage_bytes,
        keys.current(), keys.current(), keys.alternate(),
        values.current(), values.current(), values.alternate(),
        size, is_result_in_output,
        segments, begin_offsets, end_offsets,
        begin_bit, end_bit,
        stream, debug_synchronous
    );
    if(temporary_storage != nullptr && is_result_in_output)
    {
        keys.swap();
        values.swap();
    }
    return error;
}

END_ROCPRIM_NAMESPACE

/// @}
// end of group devicemodule_hip

#endif // ROCPRIM_DEVICE_DEVICE_SEGMENTED_RADIX_SORT_HIP_HPP_
