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

#ifndef ROCPRIM_DEVICE_DEVICE_SEGMENTED_SCAN_HC_HPP_
#define ROCPRIM_DEVICE_DEVICE_SEGMENTED_SCAN_HC_HPP_

#include <type_traits>
#include <iterator>

#include "../config.hpp"
#include "../detail/various.hpp"

#include "detail/device_segmented_scan.hpp"

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
    bool Exclusive,
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class InputIterator,
    class OutputIterator,
    class OffsetIterator,
    class InitValueType,
    class BinaryFunction
>
inline
void segmented_scan_impl(void * temporary_storage,
                         size_t& storage_size,
                         InputIterator input,
                         OutputIterator output,
                         unsigned int segments,
                         OffsetIterator begin_offsets,
                         OffsetIterator end_offsets,
                         const InitValueType initial_value,
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

    if(temporary_storage == nullptr)
    {
        // Make sure user won't try to allocate 0 bytes memory, because
        // hipMalloc will return nullptr when size is zero.
        storage_size = 4;
        return;
    }

    std::chrono::high_resolution_clock::time_point start;
    if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
    hc::parallel_for_each(
        acc_view,
        hc::tiled_extent<1>(segments * block_size, block_size),
        [=](hc::tiled_index<1>) [[hc]]
        {
            segmented_scan<Exclusive, block_size, items_per_thread, result_type>(
                input, output,
                begin_offsets, end_offsets,
                static_cast<result_type>(initial_value), scan_op
            );
        }
    );
    ROCPRIM_DETAIL_HC_SYNC("segmented_scan", segments, start);
}

#undef ROCPRIM_DETAIL_HC_SYNC

} // end of detail namespace

template<
    class InputIterator,
    class OutputIterator,
    class OffsetIterator,
    class BinaryFunction = ::rocprim::plus<typename std::iterator_traits<InputIterator>::value_type>
>
inline
void segmented_inclusive_scan(void * temporary_storage,
                              size_t& storage_size,
                              InputIterator input,
                              OutputIterator output,
                              unsigned int segments,
                              OffsetIterator begin_offsets,
                              OffsetIterator end_offsets,
                              BinaryFunction scan_op = BinaryFunction(),
                              hc::accelerator_view acc_view = hc::accelerator().get_default_view(),
                              const bool debug_synchronous = false)
{
    using input_type = typename std::iterator_traits<InputIterator>::value_type;
    #ifdef __cpp_lib_is_invocable
    using result_type = typename std::invoke_result<BinaryFunction, input_type, input_type>::type;
    #else
    using result_type = typename std::result_of<BinaryFunction(input_type, input_type)>::type;
    #endif

    // TODO: Those values should depend on type size
    constexpr unsigned int block_size = 256;
    constexpr unsigned int items_per_thread = 8;
    return detail::segmented_scan_impl<false, block_size, items_per_thread>(
        temporary_storage, storage_size,
        input, output, segments, begin_offsets, end_offsets, result_type(),
        scan_op, acc_view, debug_synchronous
    );
}

template<
    class InputIterator,
    class OutputIterator,
    class OffsetIterator,
    class InitValueType,
    class BinaryFunction = ::rocprim::plus<typename std::iterator_traits<InputIterator>::value_type>
>
inline
void segmented_exclusive_scan(void * temporary_storage,
                              size_t& storage_size,
                              InputIterator input,
                              OutputIterator output,
                              unsigned int segments,
                              OffsetIterator begin_offsets,
                              OffsetIterator end_offsets,
                              const InitValueType initial_value,
                              BinaryFunction scan_op = BinaryFunction(),
                              hc::accelerator_view acc_view = hc::accelerator().get_default_view(),
                              const bool debug_synchronous = false)
{
    // TODO: Those values should depend on type size
    constexpr unsigned int block_size = 256;
    constexpr unsigned int items_per_thread = 8;
    return detail::segmented_scan_impl<true, block_size, items_per_thread>(
        temporary_storage, storage_size,
        input, output, segments, begin_offsets, end_offsets, initial_value,
        scan_op, acc_view, debug_synchronous
    );
}

/// @}
// end of group devicemodule_hc

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DEVICE_SEGMENTED_SCAN_HC_HPP_
