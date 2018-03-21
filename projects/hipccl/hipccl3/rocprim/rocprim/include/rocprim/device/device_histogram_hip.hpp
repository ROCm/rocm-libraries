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

#ifndef ROCPRIM_DEVICE_DEVICE_HISTOGRAM_HIP_HPP_
#define ROCPRIM_DEVICE_DEVICE_HISTOGRAM_HIP_HPP_

#include <cmath>
#include <type_traits>
#include <iterator>

#include "../config.hpp"
#include "../functional.hpp"
#include "../detail/various.hpp"

#include "detail/device_histogram.hpp"

BEGIN_ROCPRIM_NAMESPACE

/// \addtogroup devicemodule_hip
/// @{

namespace detail
{

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class SampleIterator,
    class Counter,
    class SampleToBinOp
>
__global__
void histogram_shared_kernel(SampleIterator samples,
                             unsigned int size,
                             Counter * histogram,
                             SampleToBinOp sample_to_bin_op,
                             unsigned int bins)
{
    HIP_DYNAMIC_SHARED(unsigned int, block_histogram);

    histogram_shared<BlockSize, ItemsPerThread>(
        samples, size,
        histogram, block_histogram,
        sample_to_bin_op, bins
    );
}

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class SampleIterator,
    class Counter,
    class SampleToBinOp
>
__global__
void histogram_global_kernel(SampleIterator samples,
                             unsigned int size,
                             Counter * histogram,
                             SampleToBinOp sample_to_bin_op,
                             unsigned int bins_bits)
{
    histogram_global<BlockSize, ItemsPerThread>(
        samples, size,
        histogram,
        sample_to_bin_op, bins_bits
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
    class SampleIterator,
    class Counter,
    class Level
>
inline
hipError_t histogram_even_impl(void * temporary_storage,
                               size_t& storage_size,
                               SampleIterator samples,
                               unsigned int size,
                               Counter * histogram,
                               unsigned int levels,
                               Level lower_level,
                               Level upper_level,
                               hipStream_t stream,
                               bool debug_synchronous)
{
    constexpr unsigned int block_size = 256;
    constexpr unsigned int items_per_thread = 8;
    constexpr unsigned int max_grid_size = 1024;
    constexpr unsigned int shared_impl_max_bins = 1024;

    constexpr unsigned int items_per_block = block_size * items_per_thread;

    const unsigned int blocks = ::rocprim::detail::ceiling_div(size, items_per_block);

    if(temporary_storage == nullptr)
    {
        // Make sure user won't try to allocate 0 bytes memory, because
        // hipMalloc will return nullptr when size is zero.
        storage_size = 4;
        return hipSuccess;
    }

    if(debug_synchronous)
    {
        std::cout << "blocks " << blocks << '\n';
        hipError_t error = hipStreamSynchronize(stream);
        if(error != hipSuccess) return error;
    }

    const unsigned int bins = levels - 1;
    const unsigned int bins_bits = static_cast<unsigned int>(std::log2(detail::next_power_of_two(bins)));

    std::chrono::high_resolution_clock::time_point start;

    if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
    hipMemset(histogram, 0, bins * sizeof(Counter));
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("init histogram", bins, start);

    sample_to_bin_even<Level> sample_to_bin_op(bins, lower_level, upper_level);
    if(bins <= shared_impl_max_bins)
    {
        const size_t block_histogram_bytes = bins * sizeof(unsigned int);
        if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(histogram_shared_kernel<block_size, items_per_thread>),
            dim3(std::min(max_grid_size, blocks)), dim3(block_size), block_histogram_bytes, stream,
            samples, size, histogram,
            sample_to_bin_op,
            bins
        );
        ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("histogram_shared", size, start);
    }
    else
    {
        if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(histogram_global_kernel<block_size, items_per_thread>),
            dim3(blocks), dim3(block_size), 0, stream,
            samples, size, histogram,
            sample_to_bin_op,
            bins_bits
        );
        ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("histogram_global", size, start);
    }

    return hipSuccess;
}

template<
    class SampleIterator,
    class Counter,
    class Level
>
inline
hipError_t histogram_range_impl(void * temporary_storage,
                                size_t& storage_size,
                                SampleIterator samples,
                                unsigned int size,
                                Counter * histogram,
                                unsigned int levels,
                                const Level * level_values,
                                hipStream_t stream,
                                bool debug_synchronous)
{
    constexpr unsigned int block_size = 256;
    constexpr unsigned int items_per_thread = 8;
    constexpr unsigned int max_grid_size = 1024;
    constexpr unsigned int shared_impl_max_bins = 1024;

    constexpr unsigned int items_per_block = block_size * items_per_thread;

    const unsigned int blocks = ::rocprim::detail::ceiling_div(size, items_per_block);

    if(temporary_storage == nullptr)
    {
        // Make sure user won't try to allocate 0 bytes memory, because
        // hipMalloc will return nullptr when size is zero.
        storage_size = 4;
        return hipSuccess;
    }

    if(debug_synchronous)
    {
        std::cout << "blocks " << blocks << '\n';
        hipError_t error = hipStreamSynchronize(stream);
        if(error != hipSuccess) return error;
    }

    const unsigned int bins = levels - 1;
    const unsigned int bins_bits = static_cast<unsigned int>(std::log2(detail::next_power_of_two(bins)));

    std::chrono::high_resolution_clock::time_point start;

    if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
    hipMemset(histogram, 0, bins * sizeof(Counter));
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("init histogram", bins, start);

    sample_to_bin_range<Level> sample_to_bin_op(bins, level_values);
    if(bins <= shared_impl_max_bins)
    {
        const size_t block_histogram_bytes = bins * sizeof(unsigned int);
        if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(histogram_shared_kernel<block_size, items_per_thread>),
            dim3(std::min(max_grid_size, blocks)), dim3(block_size), block_histogram_bytes, stream,
            samples, size, histogram,
            sample_to_bin_op,
            bins
        );
        ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("histogram_shared", size, start);
    }
    else
    {
        if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(histogram_global_kernel<block_size, items_per_thread>),
            dim3(blocks), dim3(block_size), 0, stream,
            samples, size, histogram,
            sample_to_bin_op,
            bins_bits
        );
        ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("histogram_global", size, start);
    }

    return hipSuccess;
}

#undef ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR

} // end of detail namespace

template<
    class SampleIterator,
    class Counter,
    class Level
>
inline
hipError_t histogram_even(void * temporary_storage,
                          size_t& storage_size,
                          SampleIterator samples,
                          unsigned int size,
                          Counter * histogram,
                          unsigned int levels,
                          Level lower_level,
                          Level upper_level,
                          hipStream_t stream = 0,
                          bool debug_synchronous = false)
{
    return detail::histogram_even_impl(
        temporary_storage, storage_size,
        samples, size,
        histogram,
        levels, lower_level, upper_level,
        stream, debug_synchronous
    );
}

template<
    class SampleIterator,
    class Counter,
    class Level
>
inline
hipError_t histogram_range(void * temporary_storage,
                           size_t& storage_size,
                           SampleIterator samples,
                           unsigned int size,
                           Counter * histogram,
                           unsigned int levels,
                           const Level * level_values,
                           hipStream_t stream = 0,
                           bool debug_synchronous = false)
{
    return detail::histogram_range_impl(
        temporary_storage, storage_size,
        samples, size,
        histogram,
        levels, level_values,
        stream, debug_synchronous
    );
}

/// @}
// end of group devicemodule_hip

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DEVICE_HISTOGRAM_HIP_HPP_
