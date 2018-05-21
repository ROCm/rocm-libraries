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

#ifndef ROCPRIM_DEVICE_DEVICE_SORT_HC_HPP_
#define ROCPRIM_DEVICE_DEVICE_SORT_HC_HPP_

#include <type_traits>
#include <iterator>

#include "../config.hpp"
#include "../detail/various.hpp"

#include "detail/device_merge_sort.hpp"

BEGIN_ROCPRIM_NAMESPACE

/// \addtogroup devicemodule_hc
/// @{

namespace detail
{

#define ROCPRIM_DETAIL_HC_SYNC(name, size, start) \
    { \
        if(debug_synchronous) \
        { \
            std::cout << name << "(" << size << ")"; \
            acc_view.wait(); \
            auto end = std::chrono::high_resolution_clock::now(); \
            auto d = std::chrono::duration_cast<std::chrono::duration<double>>(end - start); \
            std::cout << " " << d.count() * 1000 << " ms" << '\n'; \
        } \
    }

template<
    unsigned int BlockSize,
    class KeysInputIterator,
    class KeysOutputIterator,
    class ValuesInputIterator,
    class ValuesOutputIterator,
    class BinaryFunction
>
inline
void merge_sort_impl(void * temporary_storage,
                     size_t& storage_size,
                     KeysInputIterator keys_input,
                     KeysOutputIterator keys_output,
                     ValuesInputIterator values_input,
                     ValuesOutputIterator values_output,
                     const size_t size,
                     BinaryFunction compare_function,
                     hc::accelerator_view acc_view,
                     bool debug_synchronous)

{
    using key_type = typename std::iterator_traits<KeysInputIterator>::value_type;
    using value_type = typename std::iterator_traits<ValuesInputIterator>::value_type;
    constexpr bool with_values = !std::is_same<value_type, ::rocprim::empty_type>::value;
    const size_t keys_bytes = ::rocprim::detail::align_size(size * sizeof(key_type));
    const size_t values_bytes = with_values ? ::rocprim::detail::align_size(size * sizeof(value_type)) : 0;

    constexpr unsigned int block_size = BlockSize;

    if(temporary_storage == nullptr)
    {
        storage_size = keys_bytes;
        if(with_values)
        {
            storage_size += values_bytes;
        }
        // Make sure user won't try to allocate 0 bytes memory
        storage_size = storage_size == 0 ? 4 : storage_size;
        return;
    }

    // Start point for time measurements
    std::chrono::high_resolution_clock::time_point start;

    auto number_of_blocks = (size + block_size - 1)/block_size;
    if(debug_synchronous)
    {
        std::cout << "block_size " << block_size << '\n';
        std::cout << "number of blocks " << number_of_blocks << '\n';
    }

    const unsigned int grid_size = number_of_blocks * block_size;

    bool temporary_store = false;
    char * ptr = reinterpret_cast<char *>(temporary_storage);
    key_type * keys_buffer = reinterpret_cast<key_type *>(ptr);
    ptr += keys_bytes;
    value_type * values_buffer = with_values ?
                                 reinterpret_cast<value_type *>(ptr) :
                                 nullptr;

    if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
    hc::parallel_for_each(
        acc_view,
        hc::tiled_extent<1>(grid_size, block_size),
        [=](hc::tiled_index<1>) [[hc]]
        {
            block_sort_kernel_impl<BlockSize>(
                keys_input, keys_output, values_input, values_output,
                size, compare_function
            );
        }
    );
    ROCPRIM_DETAIL_HC_SYNC("block_sort_kernel", size, start)

    for(unsigned int block = block_size ; block < size; block *= 2)
    {
        temporary_store = !temporary_store;
        if(temporary_store)
        {
            if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
            hc::parallel_for_each(
                acc_view,
                hc::tiled_extent<1>(grid_size, block_size),
                [=](hc::tiled_index<1>) [[hc]]
                {
                    block_merge_kernel_impl(
                        keys_output, keys_buffer, values_output, values_buffer,
                        size, block, compare_function
                    );
                }
            );
            ROCPRIM_DETAIL_HC_SYNC("block_merge_buffer_kernel", size, start)
        }
        else
        {
            if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
            hc::parallel_for_each(
                acc_view,
                hc::tiled_extent<1>(grid_size, block_size),
                [=](hc::tiled_index<1>) [[hc]]
                {
                    block_merge_kernel_impl(
                        keys_buffer, keys_output, values_buffer, values_output,
                        size, block, compare_function
                    );
                }
            );
            ROCPRIM_DETAIL_HC_SYNC("block_merge_kernel", size, start)
        }
    }

    if(temporary_store)
    {
        if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
        hc::parallel_for_each(
            acc_view,
            hc::tiled_extent<1>(grid_size, block_size),
            [=](hc::tiled_index<1>) [[hc]]
            {
                block_copy_kernel_impl<BlockSize>(
                    keys_buffer, keys_output, values_buffer, values_output, size
                );
            }
        );
        ROCPRIM_DETAIL_HC_SYNC("block_copy_kernel", size, start)
    }
}

#undef ROCPRIM_DETAIL_HC_SYNC

} // end of detail namespace

template<
    class KeysInputIterator,
    class KeysOutputIterator,
    class BinaryFunction = ::rocprim::less<typename std::iterator_traits<KeysInputIterator>::value_type>
>
inline
void merge_sort(void * temporary_storage,
                size_t& storage_size,
                KeysInputIterator keys_input,
                KeysOutputIterator keys_output,
                const size_t size,
                BinaryFunction compare_function = BinaryFunction(),
                hc::accelerator_view acc_view = hc::accelerator().get_default_view(),
                bool debug_synchronous = false)
{
    constexpr unsigned int block_size = 256;
    empty_type * values = nullptr;
    return detail::merge_sort_impl<block_size>(
        temporary_storage, storage_size,
        keys_input, keys_output, values, values, size,
        compare_function, acc_view, debug_synchronous
    );
}

template<
    class KeysInputIterator,
    class KeysOutputIterator,
    class ValuesInputIterator,
    class ValuesOutputIterator,
    class BinaryFunction = ::rocprim::less<typename std::iterator_traits<KeysInputIterator>::value_type>
>
inline
void merge_sort(void * temporary_storage,
                size_t& storage_size,
                KeysInputIterator keys_input,
                KeysOutputIterator keys_output,
                ValuesInputIterator values_input,
                ValuesOutputIterator values_output,
                const size_t size,
                BinaryFunction compare_function = BinaryFunction(),
                hc::accelerator_view acc_view = hc::accelerator().get_default_view(),
                bool debug_synchronous = false)
{
    constexpr unsigned int block_size = 256;
    return detail::merge_sort_impl<block_size>(
        temporary_storage, storage_size,
        keys_input, keys_output, values_input, values_output, size,
        compare_function, acc_view, debug_synchronous
    );
}

/// @}
// end of group devicemodule_hc

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DEVICE_SORT_HC_HPP_
