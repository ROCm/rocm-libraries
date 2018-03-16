// Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DEVICE_DEVICE_SELECT_HC_HPP_
#define ROCPRIM_DEVICE_DEVICE_SELECT_HC_HPP_

#include <type_traits>
#include <iterator>

#include "../config.hpp"
#include "../detail/various.hpp"

#include "../iterator/transform_iterator.hpp"

#include "detail/device_select.hpp"
#include "device_scan_hc.hpp"

BEGIN_ROCPRIM_NAMESPACE

/// \addtogroup devicemodule_hc
/// @{

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
    class InputIterator,
    class FlagIterator,
    class OutputIterator,
    class SelectedCountOutputIterator
>
inline
void select(void * temporary_storage,
            size_t& storage_size,
            InputIterator input,
            FlagIterator flags,
            OutputIterator output,
            SelectedCountOutputIterator selected_count_output,
            const size_t size,
            hc::accelerator_view acc_view = hc::accelerator().get_default_view(),
            const bool debug_synchronous = false)
{
    // Get temporary storage required by scan operation
    size_t scan_storage_size = 0;
    unsigned int * dummy_ptr = nullptr;
    ::rocprim::exclusive_scan(
        nullptr, scan_storage_size,
        flags, dummy_ptr, 0U, size, ::rocprim::plus<unsigned int>(),
        acc_view, debug_synchronous
    );
    // Align
    scan_storage_size = ::rocprim::detail::align_size(scan_storage_size);

    // Calculate required temporary storage
    if(temporary_storage == nullptr)
    {
        storage_size = scan_storage_size;
        // Add storage required for indexes
        storage_size += size * sizeof(unsigned int);
        // Make sure user won't try to allocate 0 bytes memory, otherwise
        // user may again pass nullptr as temporary_storage
        storage_size = storage_size == 0 ? 4 : storage_size;
        return;
    }

    // Return for empty input
    if(size == 0) return;

    // Start point for time measurements
    std::chrono::high_resolution_clock::time_point start;

    // Calculate output indices to scatter selected values
    auto indices = reinterpret_cast<unsigned int*>(
        static_cast<unsigned char*>(temporary_storage) + scan_storage_size
    );
    ::rocprim::exclusive_scan(
        temporary_storage, scan_storage_size,
        flags, indices, 0U, size, ::rocprim::plus<unsigned int>(),
        acc_view, debug_synchronous
    );
    ROCPRIM_DETAIL_HC_SYNC("rocprim::exclusive_scan", size, start)

    // TODO: Those values should depend on type size
    constexpr unsigned int block_size = 256;
    constexpr unsigned int items_per_thread = 4;
    constexpr auto items_per_block = block_size * items_per_thread;

    auto number_of_blocks = (size + items_per_block - 1)/items_per_block;
    if(debug_synchronous)
    {
        std::cout << "block_size " << block_size << '\n';
        std::cout << "number of blocks " << number_of_blocks << '\n';
        std::cout << "items_per_block " << items_per_block << '\n';
        std::cout << "temporary storage size " << storage_size << '\n';
    }

    if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
    hc::parallel_for_each(
        acc_view,
        hc::tiled_extent<1>(number_of_blocks * block_size, block_size),
        [=](hc::tiled_index<1>) [[hc]]
        {
            detail::scatter_kernel_impl<block_size, items_per_thread>(
                input, size, flags, indices,
                output, selected_count_output
            );
        }
    );
    ROCPRIM_DETAIL_HC_SYNC("scatter_kernel", size, start)
}

template<
    class InputIterator,
    class OutputIterator,
    class SelectedCountOutputIterator,
    class SelectOp
>
inline
void select(void * temporary_storage,
            size_t& storage_size,
            InputIterator input,
            OutputIterator output,
            SelectedCountOutputIterator selected_count_output,
            const size_t size,
            SelectOp select_op,
            hc::accelerator_view acc_view = hc::accelerator().get_default_view(),
            const bool debug_synchronous = false)
{
    // Get temporary storage required by scan operation
    size_t scan_storage_size = 0;
    unsigned char * dummy_in_ptr = nullptr;
    unsigned int *  dummy_out_ptr = nullptr;
    ::rocprim::exclusive_scan(
        nullptr, scan_storage_size,
        dummy_in_ptr, dummy_out_ptr,
        0U, size, ::rocprim::plus<unsigned int>(),
        acc_view, debug_synchronous
    );
    // Align
    scan_storage_size = ::rocprim::detail::align_size(scan_storage_size);

    // Calculate required temporary storage
    if(temporary_storage == nullptr)
    {
        storage_size = scan_storage_size;
        // Add storage required for indexes
        storage_size += size * sizeof(unsigned int);
        // Make sure user won't try to allocate 0 bytes memory, otherwise
        // user may again pass nullptr as temporary_storage
        storage_size = storage_size == 0 ? 4 : storage_size;
        return;
    }

    // Return for empty input
    if(size == 0) return;

    // Start point for time measurements
    std::chrono::high_resolution_clock::time_point start;

    // Calculate output indices to scatter selected values
    auto indices = reinterpret_cast<unsigned int*>(
        static_cast<unsigned char*>(temporary_storage) + scan_storage_size
    );
    ::rocprim::exclusive_scan(
        temporary_storage, scan_storage_size,
        ::rocprim::make_transform_iterator(input, select_op),
        indices, 0U, size, ::rocprim::plus<unsigned int>(),
        acc_view, debug_synchronous
    );
    ROCPRIM_DETAIL_HC_SYNC("rocprim::exclusive_scan", size, start)

    // TODO: Those values should depend on type size
    constexpr unsigned int block_size = 256;
    constexpr unsigned int items_per_thread = 4;
    constexpr auto items_per_block = block_size * items_per_thread;

    auto number_of_blocks = (size + items_per_block - 1)/items_per_block;
    if(debug_synchronous)
    {
        std::cout << "block_size " << block_size << '\n';
        std::cout << "number of blocks " << number_of_blocks << '\n';
        std::cout << "items_per_block " << items_per_block << '\n';
        std::cout << "temporary storage size " << storage_size << '\n';
    }

    if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
    hc::parallel_for_each(
        acc_view,
        hc::tiled_extent<1>(number_of_blocks * block_size, block_size),
        [=](hc::tiled_index<1>) [[hc]]
        {
            detail::scatter_if_kernel_impl<block_size, items_per_thread>(
                input, size, indices,
                output, selected_count_output,
                select_op
            );
        }
    );
    ROCPRIM_DETAIL_HC_SYNC("scatter_if_kernel", size, start)
}

template<
    class InputIterator,
    class OutputIterator,
    class UniqueCountOutputIterator,
    class EqualityOp = ::rocprim::equal_to<typename std::iterator_traits<InputIterator>::value_type>
>
inline
void unique(void * temporary_storage,
            size_t& storage_size,
            InputIterator input,
            OutputIterator output,
            UniqueCountOutputIterator unique_count_output,
            const size_t size,
            EqualityOp equality_op = EqualityOp(),
            hc::accelerator_view acc_view = hc::accelerator().get_default_view(),
            const bool debug_synchronous = false)
{
    using input_type = typename std::iterator_traits<InputIterator>::value_type;

    // Get temporary storage required by select operation
    size_t select_storage_size = 0;
    unsigned char * dummy_flags_ptr = nullptr;
    ::rocprim::select(
        static_cast<void*>(nullptr), select_storage_size,
        input, dummy_flags_ptr, output, unique_count_output, size,
        acc_view, debug_synchronous
    );
    // Align
    select_storage_size = ::rocprim::detail::align_size(select_storage_size);

    // Calculate required temporary storage
    if(temporary_storage == nullptr)
    {
        storage_size = select_storage_size;
        // Add storage required for flags
        storage_size += size * sizeof(unsigned char);
        // Make sure user won't try to allocate 0 bytes memory, otherwise
        // user may again pass nullptr as temporary_storage
        storage_size = storage_size == 0 ? 4 : storage_size;
        return;
    }

    // Return for empty input
    if(size == 0) return;

    // TODO: Those values should depend on type size
    constexpr unsigned int block_size = 256;
    constexpr unsigned int items_per_thread = 4;
    constexpr auto items_per_block = block_size * items_per_thread;

    auto number_of_blocks = (size + items_per_block - 1)/items_per_block;
    if(debug_synchronous)
    {
        std::cout << "block_size " << block_size << '\n';
        std::cout << "number of blocks " << number_of_blocks << '\n';
        std::cout << "items_per_block " << items_per_block << '\n';
        std::cout << "temporary storage size " << storage_size << '\n';
    }

    // Start point for time measurements
    std::chrono::high_resolution_clock::time_point start;

    auto flags = static_cast<unsigned char*>(temporary_storage) + select_storage_size;
    auto inequality_op =
        [equality_op](const input_type& a, const input_type& b) [[hc]] -> bool
        {
            return !equality_op(a, b);
        };
    hc::parallel_for_each(
        acc_view,
        hc::tiled_extent<1>(number_of_blocks * block_size, block_size),
        [=](hc::tiled_index<1>) [[hc]]
        {
            detail::flag_unique_kernel_impl<block_size, items_per_thread>(
                input, size, flags, inequality_op
            );
        }
    );
    ROCPRIM_DETAIL_HC_SYNC("flag_unique_kernel", size, start)

    if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
    // select unique values
    ::rocprim::select(
        temporary_storage, select_storage_size,
        input, flags, output, unique_count_output, size,
        acc_view, debug_synchronous
    );
    ROCPRIM_DETAIL_HC_SYNC("rocprim::select", size, start)
}

#undef ROCPRIM_DETAIL_HC_SYNC

/// @}
// end of group devicemodule_hc

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DEVICE_SELECT_HC_HPP_
