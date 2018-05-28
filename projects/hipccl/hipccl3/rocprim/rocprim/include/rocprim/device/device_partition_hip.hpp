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

#ifndef ROCPRIM_DEVICE_DEVICE_PARTITION_HIP_HPP_
#define ROCPRIM_DEVICE_DEVICE_PARTITION_HIP_HPP_

#include <type_traits>
#include <iterator>

#include "../config.hpp"
#include "../functional.hpp"
#include "../type_traits.hpp"
#include "../detail/various.hpp"

#include "detail/device_partition.hpp"

BEGIN_ROCPRIM_NAMESPACE

/// \addtogroup devicemodule_hip
/// @{

namespace detail
{

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class ResultType,
    class InputIterator,
    class FlagIterator,
    class OutputIterator,
    class SelectedCountOutputIterator,
    class OffsetLookbackScanState
>
__global__
void partition_flag_kernel(InputIterator input,
                           FlagIterator flags,
                           OutputIterator output,
                           SelectedCountOutputIterator selected_count_output,
                           const size_t size,
                           OffsetLookbackScanState offset_scan_state,
                           const unsigned int number_of_blocks,
                           ordered_block_id<unsigned int> ordered_bid)
{
    partition_flag_kernel_impl<BlockSize, ItemsPerThread, ResultType>(
        input, flags, output, selected_count_output, size,
        offset_scan_state, number_of_blocks, ordered_bid
    );
}

template<class OffsetLookBackScanState>
__global__
void init_offset_scan_state_kernel(OffsetLookBackScanState offset_scan_state,
                                   const unsigned int number_of_blocks,
                                   ordered_block_id<unsigned int> ordered_bid)
{
    init_lookback_scan_state_kernel_impl(
        offset_scan_state, number_of_blocks, ordered_bid
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
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class ResultType,
    class InputIterator,
    class FlagIterator,
    class OutputIterator,
    class SelectedCountOutputIterator
>
inline
hipError_t partition_flag_impl(void * temporary_storage,
                               size_t& storage_size,
                               InputIterator input,
                               FlagIterator flags,
                               OutputIterator output,
                               SelectedCountOutputIterator selected_count_output,
                               const size_t size,
                               const hipStream_t stream,
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
        return hipSuccess;
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
    auto grid_size = (number_of_blocks + block_size - 1)/block_size;
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(init_offset_scan_state_kernel<offset_scan_state_type>),
        dim3(grid_size), dim3(block_size), 0, stream,
        offset_scan_state, number_of_blocks, ordered_bid
    );
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("init_offset_scan_state_kernel", size, start)

    if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
    grid_size = number_of_blocks;
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(partition_flag_kernel<
            BlockSize, ItemsPerThread,
            ResultType, InputIterator, FlagIterator,
            OutputIterator, SelectedCountOutputIterator,
            offset_scan_state_type
        >),
        dim3(grid_size), dim3(block_size), 0, stream,
        input, flags, output, selected_count_output, size,
        offset_scan_state, number_of_blocks, ordered_bid
    );
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("lookback_scan_kernel", size, start)

    return hipSuccess;
}

#undef ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR
#undef ROCPRIM_DETAIL_HIP_SYNC

} // end of detail namespace

template<
    class InputIterator,
    class FlagIterator,
    class OutputIterator,
    class SelectedCountOutputIterator
>
inline
hipError_t partition(void * temporary_storage,
                     size_t& storage_size,
                     InputIterator input,
                     FlagIterator flags,
                     OutputIterator output,
                     SelectedCountOutputIterator selected_count_output,
                     const size_t size,
                     const hipStream_t stream = 0,
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

    constexpr unsigned int block_size = 256;
    constexpr unsigned int items_per_thread =
        ::rocprim::max<unsigned int>(
            (8 * sizeof(unsigned int))/sizeof(result_type), 1
        );
    return detail::partition_flag_impl<block_size, items_per_thread, result_type>(
        temporary_storage, storage_size, input, flags,
        output, selected_count_output, size, stream, debug_synchronous
    );
}

/// @}
// end of group devicemodule_hip

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DEVICE_PARTITION_HIP_HPP_
