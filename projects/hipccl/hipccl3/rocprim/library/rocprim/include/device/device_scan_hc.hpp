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

#ifndef ROCPRIM_DEVICE_DEVICE_SCAN_HC_HPP_
#define ROCPRIM_DEVICE_DEVICE_SCAN_HC_HPP_

#include <type_traits>
#include <iterator>

#include "../config.hpp"
#include "../detail/various.hpp"

#include "detail/device_scan_reduce_then_scan.hpp"

BEGIN_ROCPRIM_NAMESPACE

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
    unsigned int ItemsPerThread,
    bool Exclusive,
    class InputIterator,
    class OutputIterator,
    class InitValueType,
    class BinaryFunction
>
inline
void device_scan_impl(void * temporary_storage,
                      size_t& storage_size,
                      InputIterator input,
                      OutputIterator output,
                      const InitValueType initial_value,
                      const size_t size,
                      BinaryFunction scan_op,
                      hc::accelerator_view acc_view,
                      const bool debug_synchronous)
{
    using input_type = typename std::iterator_traits<InputIterator>::value_type;
    #ifdef __cpp_lib_is_invocable
    using result_type = typename std::invoke_result<BinaryFunction, input_type, input_type>::type;
    #else
    using result_type = typename std::result_of<BinaryFunction(input_type, input_type)>::type;
    #endif

    constexpr unsigned int block_size = BlockSize;
    constexpr unsigned int items_per_thread = ItemsPerThread;
    constexpr auto items_per_block = block_size * items_per_thread;

    // Calculate required temporary storage
    if(temporary_storage == nullptr)
    {
        storage_size = scan_get_temporary_storage_bytes<result_type>(size, items_per_block);
        // Make sure user won't try to allocate 0 bytes memory, otherwise
        // user may again pass nullptr as temporary_storage
        storage_size = storage_size == 0 ? 4 : storage_size;
        return;
    }

    // Start point for time measurements
    std::chrono::high_resolution_clock::time_point start;

    auto number_of_blocks = (size + items_per_block - 1)/items_per_block;
    if(debug_synchronous)
    {
        std::cout << "block_size " << block_size << '\n';
        std::cout << "number of blocks " << number_of_blocks << '\n';
        std::cout << "items_per_block " << items_per_block << '\n';
        std::cout << "temporary storage size " << storage_size << '\n';
    }

    if(number_of_blocks > 1)
    {
        // Grid size for block_reduce_kernel and final_scan_kernel
        auto grid_size = number_of_blocks * block_size;

        // Pointer to array with block_prefixes
        result_type * block_prefixes = static_cast<result_type*>(temporary_storage);

        if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
        hc::parallel_for_each(
            acc_view,
            hc::tiled_extent<1>(grid_size, block_size),
            [=](hc::tiled_index<1>) [[hc]]
            {
                block_reduce_kernel_impl<block_size, items_per_thread>(
                    input, size, scan_op, block_prefixes
                );
            }
        );
        ROCPRIM_DETAIL_HC_SYNC("block_reduce_kernel", size, start)

        // TODO: Performance may increase if for (number_of_blocks < 8192) (or some other
        // threshold) we would just use CPU to calculate prefixes.

        // Calculate size of temporary storage for nested device scan operation
        void * nested_temp_storage = static_cast<void*>(block_prefixes + number_of_blocks);
        auto nested_temp_storage_size = storage_size - (number_of_blocks * sizeof(result_type));

        if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
        device_scan_impl<BlockSize, ItemsPerThread, false>(
            nested_temp_storage,
            nested_temp_storage_size,
            block_prefixes, // input
            block_prefixes, // output
            char(0), // dummy initial value
            number_of_blocks, // size
            scan_op,
            acc_view,
            debug_synchronous
        );
        ROCPRIM_DETAIL_HC_SYNC("nested_device_scan", number_of_blocks, start)

        if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
        hc::parallel_for_each(
            acc_view,
            hc::tiled_extent<1>(grid_size, block_size),
            [=](hc::tiled_index<1>) [[hc]]
            {
                final_scan_kernel_impl<block_size, items_per_thread, Exclusive>(
                    input, size, output, initial_value, scan_op, block_prefixes
                );
            }
        );
        ROCPRIM_DETAIL_HC_SYNC("final_scan_kernel", size, start)
    }
    else
    {
        constexpr unsigned int single_scan_block_size = BlockSize;
        constexpr unsigned int single_scan_items_per_thread = ItemsPerThread;

        if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
        hc::parallel_for_each(
            acc_view,
            hc::tiled_extent<1>(block_size, block_size),
            [=](hc::tiled_index<1>) [[hc]]
            {
                single_scan_kernel_impl<single_scan_block_size, single_scan_items_per_thread, Exclusive>(
                    input, size, initial_value, output, scan_op
                );
            }
        );
        ROCPRIM_DETAIL_HC_SYNC("single_scan_kernel", size, start);
    }
}

#undef ROCPRIM_DETAIL_HC_SYNC

} // end of detail namespace

template<
    class InputIterator,
    class OutputIterator,
    class BinaryFunction = ::rocprim::plus<typename std::iterator_traits<InputIterator>::value_type>
>
inline
void device_inclusive_scan(void * temporary_storage,
                           size_t& storage_size,
                           InputIterator input,
                           OutputIterator output,
                           const size_t size,
                           BinaryFunction scan_op = BinaryFunction(),
                           hc::accelerator_view acc_view = hc::accelerator().get_default_view(),
                           const bool debug_synchronous = false)
{
    // TODO: Those values should depend on type size
    constexpr unsigned int block_size = 256;
    constexpr unsigned int items_per_thread = 4;
    return detail::device_scan_impl<block_size, items_per_thread, false>(
        temporary_storage, storage_size,
        // char(0) is a dummy initial value
        input, output, char(0), size,
        scan_op, acc_view, debug_synchronous
    );
}

template<
    class InputIterator,
    class OutputIterator,
    class InitValueType,
    class BinaryFunction = ::rocprim::plus<typename std::iterator_traits<InputIterator>::value_type>
>
inline
void device_exclusive_scan(void * temporary_storage,
                           size_t& storage_size,
                           InputIterator input,
                           OutputIterator output,
                           const InitValueType initial_value,
                           const size_t size,
                           BinaryFunction scan_op = BinaryFunction(),
                           hc::accelerator_view acc_view = hc::accelerator().get_default_view(),
                           const bool debug_synchronous = false)
{
    // TODO: Those values should depend on type size
    constexpr unsigned int block_size = 256;
    constexpr unsigned int items_per_thread = 4;
    return detail::device_scan_impl<block_size, items_per_thread, true>(
        temporary_storage, storage_size,
        input, output, initial_value, size,
        scan_op, acc_view, debug_synchronous
    );
}

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DEVICE_SCAN_HC_HPP_
