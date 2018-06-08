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

#ifndef ROCPRIM_DEVICE_DEVICE_PARTITION_HC_HPP_
#define ROCPRIM_DEVICE_DEVICE_PARTITION_HC_HPP_

#include <type_traits>
#include <iterator>

#include "../config.hpp"
#include "../functional.hpp"
#include "../type_traits.hpp"
#include "../detail/various.hpp"

#include "detail/device_partition.hpp"

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
    bool UsePredicate,
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class ResultType,
    class InputIterator,
    class FlagIterator,
    class OutputIterator,
    class UnaryPredicate,
    class SelectedCountOutputIterator
>
inline
void partition_impl(void * temporary_storage,
                    size_t& storage_size,
                    InputIterator input,
                    FlagIterator flags,
                    OutputIterator output,
                    SelectedCountOutputIterator selected_count_output,
                    const size_t size,
                    UnaryPredicate predicate,
                    hc::accelerator_view acc_view,
                    bool debug_synchronous)
{
    using offset_type = unsigned int;
    using offset_scan_state_type = detail::lookback_scan_state<offset_type>;
    using ordered_block_id_type = detail::ordered_block_id<unsigned int>;

    constexpr unsigned int block_size = BlockSize;
    constexpr unsigned int items_per_thread = ItemsPerThread;
    constexpr auto items_per_block = block_size * items_per_thread;
    const unsigned int number_of_blocks = (size + items_per_block - 1)/items_per_block;

    // Calculate required temporary storage
    size_t offset_scan_state_bytes = ::rocprim::detail::align_size(
        offset_scan_state_type::get_storage_size(number_of_blocks)
    );
    size_t ordered_block_id_bytes = ordered_block_id_type::get_storage_size();
    if(temporary_storage == nullptr)
    {
        // storage_size is never zero
        storage_size = offset_scan_state_bytes + ordered_block_id_bytes;
        return;
    }

    // Start point for time measurements
    std::chrono::high_resolution_clock::time_point start;
    if(debug_synchronous)
    {
        std::cout << "size " << size << '\n';
        std::cout << "block_size " << block_size << '\n';
        std::cout << "number of blocks " << number_of_blocks << '\n';
        std::cout << "items_per_block " << items_per_block << '\n';
    }

    // Create and initialize lookback_scan_state obj
    auto offset_scan_state = offset_scan_state_type::create(
        temporary_storage, number_of_blocks
    );
    // Create ad initialize ordered_block_id obj
    auto ptr = reinterpret_cast<char*>(temporary_storage);
    auto ordered_bid = ordered_block_id_type::create(
        reinterpret_cast<ordered_block_id_type::id_type*>(ptr + offset_scan_state_bytes)
    );

    if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
    auto grid_size = ((number_of_blocks + block_size - 1)/block_size) * block_size;
    hc::parallel_for_each(
        acc_view,
        hc::tiled_extent<1>(grid_size, block_size),
        [=](hc::tiled_index<1>) [[hc]]
        {
            init_lookback_scan_state_kernel_impl(
                offset_scan_state, number_of_blocks, ordered_bid
            );
        }
    );
    ROCPRIM_DETAIL_HC_SYNC("init_offset_scan_state_kernel", size, start)

    if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
    grid_size = number_of_blocks * block_size;
    hc::parallel_for_each(
        acc_view,
        hc::tiled_extent<1>(grid_size, block_size),
        [=](hc::tiled_index<1>) [[hc]]
        {
            partition_kernel_impl<UsePredicate, BlockSize, ItemsPerThread, ResultType>(
                input, flags, output, selected_count_output, size, predicate,
                offset_scan_state, number_of_blocks, ordered_bid
            );
        }
    );
    ROCPRIM_DETAIL_HC_SYNC("partition_kernel", size, start)
}

#undef ROCPRIM_DETAIL_HC_SYNC

} // end of detail namespace

template<
    class InputIterator,
    class FlagIterator,
    class OutputIterator,
    class SelectedCountOutputIterator
>
inline
void partition(void * temporary_storage,
               size_t& storage_size,
               InputIterator input,
               FlagIterator flags,
               OutputIterator output,
               SelectedCountOutputIterator selected_count_output,
               const size_t size,
               hc::accelerator_view acc_view = hc::accelerator().get_default_view(),
               const bool debug_synchronous = false)
{
    using input_type = typename std::iterator_traits<InputIterator>::value_type;
    using output_type = typename std::iterator_traits<OutputIterator>::value_type;
    // Fix for cases when output_type is void (there's no sizeof(void))
    using value_type = typename std::conditional<
        std::is_same<void, output_type>::value, input_type, output_type
    >::type;
    // Use smaller type for private storage
    using result_type = typename std::conditional<
        (sizeof(value_type) > sizeof(input_type)), input_type, value_type
    >::type;

    // Dummy unary preficate
    using unary_preficate_type = ::rocprim::empty_type;

    constexpr unsigned int block_size = 256;
    constexpr unsigned int items_per_thread =
        ::rocprim::max<unsigned int>(
            (8 * sizeof(unsigned int))/sizeof(result_type), 1
        );
    return detail::partition_impl<false, block_size, items_per_thread, result_type>(
        temporary_storage, storage_size, input, flags, output, selected_count_output,
        size, unary_preficate_type(), acc_view, debug_synchronous
    );
}

template<
    class InputIterator,
    class OutputIterator,
    class SelectedCountOutputIterator,
    class UnaryPredicate
>
inline
void partition(void * temporary_storage,
               size_t& storage_size,
               InputIterator input,
               OutputIterator output,
               SelectedCountOutputIterator selected_count_output,
               const size_t size,
               UnaryPredicate predicate,
               hc::accelerator_view acc_view = hc::accelerator().get_default_view(),
               const bool debug_synchronous = false)
{
    using input_type = typename std::iterator_traits<InputIterator>::value_type;
    using output_type = typename std::iterator_traits<OutputIterator>::value_type;
    // Fix for cases when output_type is void (there's no sizeof(void))
    using value_type = typename std::conditional<
        std::is_same<void, output_type>::value, input_type, output_type
    >::type;
    // Use smaller type for private storage
    using result_type = typename std::conditional<
        (sizeof(value_type) > sizeof(input_type)), input_type, value_type
    >::type;

    // Dummy flag type
    using flag_type = ::rocprim::empty_type;
    flag_type * flags = nullptr;

    constexpr unsigned int block_size = 256;
    constexpr unsigned int items_per_thread =
        ::rocprim::max<unsigned int>(
            (8 * sizeof(unsigned int))/sizeof(result_type), 1
        );
    return detail::partition_impl<true, block_size, items_per_thread, result_type>(
        temporary_storage, storage_size, input, flags, output, selected_count_output,
        size, predicate, acc_view, debug_synchronous
    );
}

/// @}
// end of group devicemodule_hc

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DEVICE_PARTITION_HC_HPP_
