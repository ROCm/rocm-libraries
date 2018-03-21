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

#ifndef ROCPRIM_DEVICE_DEVICE_HISTOGRAM_HC_HPP_
#define ROCPRIM_DEVICE_DEVICE_HISTOGRAM_HC_HPP_

#include <cmath>
#include <type_traits>
#include <iterator>

#include "../config.hpp"
#include "../functional.hpp"
#include "../detail/various.hpp"

#include "detail/device_histogram.hpp"

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
    class SampleIterator,
    class Counter,
    class Level
>
inline
void histogram_even_impl(void * temporary_storage,
                         size_t& storage_size,
                         SampleIterator samples,
                         unsigned int size,
                         Counter * histogram,
                         unsigned int levels,
                         Level lower_level,
                         Level upper_level,
                         hc::accelerator_view& acc_view,
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
        // Make sure user won't try to allocate 0 bytes memory, otherwise
        // user may again pass nullptr as temporary_storage
        storage_size = 4;
        return;
    }

    if(debug_synchronous)
    {
        std::cout << "blocks " << blocks << '\n';
        acc_view.wait();
    }

    const unsigned int bins = levels - 1;
    const unsigned int bins_bits = static_cast<unsigned int>(std::log2(detail::next_power_of_two(bins)));

    std::chrono::high_resolution_clock::time_point start;

    if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
    hc::parallel_for_each(
        acc_view,
        hc::extent<1>(bins),
        [=](hc::index<1> i) [[hc]]
        {
            histogram[i[0]] = 0;
        }
    );
    ROCPRIM_DETAIL_HC_SYNC("init histogram", bins, start);

    sample_to_bin_even<Level> sample_to_bin_op(bins, lower_level, upper_level);
    if(bins <= shared_impl_max_bins)
    {
        const size_t block_histogram_bytes = bins * sizeof(unsigned int);
        if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
        hc::parallel_for_each(
            acc_view,
            hc::tiled_extent<1>(std::min(max_grid_size, blocks) * block_size, block_size, block_histogram_bytes),
            [=](hc::tiled_index<1>) [[hc]]
            {
                unsigned int * block_histogram = static_cast<unsigned int *>(
                    hc::get_dynamic_group_segment_base_pointer()
                );
                histogram_shared<block_size, items_per_thread>(
                    samples, size, histogram, block_histogram,
                    sample_to_bin_op,
                    bins
                );
            }
        );
        ROCPRIM_DETAIL_HC_SYNC("histogram_shared", size, start);
    }
    else
    {
        if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
        hc::parallel_for_each(
            acc_view,
            hc::tiled_extent<1>(blocks * block_size, block_size),
            [=](hc::tiled_index<1>) [[hc]]
            {
                histogram_global<block_size, items_per_thread>(
                    samples, size, histogram,
                    sample_to_bin_op,
                    bins_bits
                );
            }
        );
        ROCPRIM_DETAIL_HC_SYNC("histogram_global", size, start);
    }
}

template<
    class SampleIterator,
    class Counter,
    class Level
>
inline
void histogram_range_impl(void * temporary_storage,
                          size_t& storage_size,
                          SampleIterator samples,
                          unsigned int size,
                          Counter * histogram,
                          unsigned int levels,
                          const Level * level_values,
                          hc::accelerator_view& acc_view,
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
        // Make sure user won't try to allocate 0 bytes memory, otherwise
        // user may again pass nullptr as temporary_storage
        storage_size = 4;
        return;
    }

    if(debug_synchronous)
    {
        std::cout << "blocks " << blocks << '\n';
        acc_view.wait();
    }

    const unsigned int bins = levels - 1;
    const unsigned int bins_bits = static_cast<unsigned int>(std::log2(detail::next_power_of_two(bins)));

    std::chrono::high_resolution_clock::time_point start;

    if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
    hc::parallel_for_each(
        acc_view,
        hc::extent<1>(bins),
        [=](hc::index<1> i) [[hc]]
        {
            histogram[i[0]] = 0;
        }
    );
    ROCPRIM_DETAIL_HC_SYNC("init histogram", bins, start);

    sample_to_bin_range<Level> sample_to_bin_op(bins, level_values);
    if(bins <= shared_impl_max_bins)
    {
        const size_t block_histogram_bytes = bins * sizeof(unsigned int);
        if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
        hc::parallel_for_each(
            acc_view,
            hc::tiled_extent<1>(std::min(max_grid_size, blocks) * block_size, block_size, block_histogram_bytes),
            [=](hc::tiled_index<1>) [[hc]]
            {
                unsigned int * block_histogram = static_cast<unsigned int *>(
                    hc::get_dynamic_group_segment_base_pointer()
                );
                histogram_shared<block_size, items_per_thread>(
                    samples, size, histogram, block_histogram,
                    sample_to_bin_op,
                    bins
                );
            }
        );
        ROCPRIM_DETAIL_HC_SYNC("histogram_shared", size, start);
    }
    else
    {
        if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
        hc::parallel_for_each(
            acc_view,
            hc::tiled_extent<1>(blocks * block_size, block_size),
            [=](hc::tiled_index<1>) [[hc]]
            {
                histogram_global<block_size, items_per_thread>(
                    samples, size, histogram,
                    sample_to_bin_op,
                    bins_bits
                );
            }
        );
        ROCPRIM_DETAIL_HC_SYNC("histogram_global", size, start);
    }
}

#undef ROCPRIM_DETAIL_HC_SYNC

} // end of detail namespace

template<
    class SampleIterator,
    class Counter,
    class Level
>
inline
void histogram_even(void * temporary_storage,
                    size_t& storage_size,
                    SampleIterator samples,
                    unsigned int size,
                    Counter * histogram,
                    unsigned int levels,
                    Level lower_level,
                    Level upper_level,
                    hc::accelerator_view acc_view = hc::accelerator().get_default_view(),
                    bool debug_synchronous = false)
{
    detail::histogram_even_impl(
        temporary_storage, storage_size,
        samples, size,
        histogram,
        levels, lower_level, upper_level,
        acc_view, debug_synchronous
    );
}

template<
    class SampleIterator,
    class Counter,
    class Level
>
inline
void histogram_range(void * temporary_storage,
                     size_t& storage_size,
                     SampleIterator samples,
                     unsigned int size,
                     Counter * histogram,
                     unsigned int levels,
                     const Level * level_values,
                     hc::accelerator_view acc_view = hc::accelerator().get_default_view(),
                     bool debug_synchronous = false)
{
    detail::histogram_range_impl(
        temporary_storage, storage_size,
        samples, size,
        histogram,
        levels, level_values,
        acc_view, debug_synchronous
    );
}

/// @}
// end of group devicemodule_hc

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DEVICE_HISTOGRAM_HC_HPP_
