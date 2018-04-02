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
    unsigned int Channels,
    unsigned int ActiveChannels,
    class SampleIterator,
    class Counter,
    class SampleToBinOp
>
inline
void histogram_impl(void * temporary_storage,
                    size_t& storage_size,
                    SampleIterator samples,
                    unsigned int columns,
                    unsigned int rows,
                    size_t row_stride_bytes,
                    Counter * histogram[ActiveChannels],
                    unsigned int levels[ActiveChannels],
                    SampleToBinOp sample_to_bin_op[ActiveChannels],
                    hc::accelerator_view& acc_view,
                    bool debug_synchronous)
{
    using sample_type = typename std::iterator_traits<SampleIterator>::value_type;

    constexpr unsigned int block_size = 256;
    constexpr unsigned int items_per_thread = 8;
    constexpr unsigned int max_grid_size = 1024;
    constexpr unsigned int shared_impl_max_bins = 1024;

    constexpr unsigned int items_per_block = block_size * items_per_thread;

    for(unsigned int channel = 0; channel < ActiveChannels; channel++)
    {
        if(levels[channel] < 2)
        {
            // Histogram must have at least 1 bin
            throw hc::runtime_exception("`levels` must be at least 2", 0);
        }
    }

    if(row_stride_bytes % sizeof(sample_type) != 0)
    {
        // Row stride must be a whole multiple of the sample data type size
        throw hc::runtime_exception("Row stride must be a whole multiple of the sample data type size", 0);
    }

    const unsigned int blocks_x = ::rocprim::detail::ceiling_div(columns, items_per_block);
    const unsigned int row_stride = row_stride_bytes / sizeof(sample_type);

    if(temporary_storage == nullptr)
    {
        // Make sure user won't try to allocate 0 bytes memory, otherwise
        // user may again pass nullptr as temporary_storage
        storage_size = 4;
        return;
    }

    if(debug_synchronous)
    {
        std::cout << "columns " << columns << '\n';
        std::cout << "rows " << rows << '\n';
        std::cout << "blocks_x " << blocks_x << '\n';
        acc_view.wait();
    }

    unsigned int bins[ActiveChannels];
    unsigned int bins_bits[ActiveChannels];
    unsigned int total_bins = 0;
    unsigned int max_bins = 0;
    for(unsigned int channel = 0; channel < ActiveChannels; channel++)
    {
        bins[channel] = levels[channel] - 1;
        bins_bits[channel] = static_cast<unsigned int>(std::log2(detail::next_power_of_two(bins[channel])));
        total_bins += bins[channel];
        max_bins = std::max(max_bins, bins[channel]);
    }

    fixed_array<Counter *, ActiveChannels> histogram_fixed(histogram);
    fixed_array<unsigned int, ActiveChannels> bins_fixed(bins);
    fixed_array<unsigned int, ActiveChannels> bins_bits_fixed(bins_bits);

    // Workaround: HCC cannot pass structs with array fields of composite types
    // even with custom serializer and deserializer (see fixed_array).
    // It seems that they work only with primitive types).
    // Hence a new fixed_array is recreated inside kernels using individual values that can be passed correctly.
    auto sample_to_bin_op0 = sample_to_bin_op[std::min(0u, ActiveChannels - 1)];
    auto sample_to_bin_op1 = sample_to_bin_op[std::min(1u, ActiveChannels - 1)];
    auto sample_to_bin_op2 = sample_to_bin_op[std::min(2u, ActiveChannels - 1)];
    auto sample_to_bin_op3 = sample_to_bin_op[std::min(3u, ActiveChannels - 1)];

    std::chrono::high_resolution_clock::time_point start;

    if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
    hc::parallel_for_each(
        acc_view,
        hc::tiled_extent<1>(::rocprim::detail::ceiling_div(max_bins, block_size) * block_size, block_size),
        [=](hc::tiled_index<1>) [[hc]]
        {
            init_histogram<block_size, ActiveChannels>(histogram_fixed, bins_fixed);
        }
    );
    ROCPRIM_DETAIL_HC_SYNC("init_histogram", max_bins, start);

    if(total_bins <= shared_impl_max_bins)
    {
        const unsigned int grid_size_x = std::min(max_grid_size, blocks_x);
        const unsigned int grid_size_y = std::min(rows, max_grid_size / grid_size_x);
        const size_t block_histogram_bytes = total_bins * sizeof(unsigned int);
        if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
        hc::parallel_for_each(
            acc_view,
            hc::tiled_extent<2>(grid_size_y, grid_size_x * block_size, 1, block_size, block_histogram_bytes),
            [=](hc::tiled_index<2>) [[hc]]
            {
                fixed_array<SampleToBinOp, ActiveChannels> sample_to_bin_op_fixed(
                    sample_to_bin_op0, sample_to_bin_op1, sample_to_bin_op2, sample_to_bin_op3
                );

                unsigned int * block_histogram = static_cast<unsigned int *>(
                    hc::get_dynamic_group_segment_base_pointer()
                );

                histogram_shared<block_size, items_per_thread, Channels, ActiveChannels>(
                    samples, columns, rows, row_stride,
                    histogram_fixed,
                    sample_to_bin_op_fixed,
                    bins_fixed,
                    block_histogram
                );
            }
        );
        ROCPRIM_DETAIL_HC_SYNC("histogram_shared", grid_size_x * grid_size_y * block_size, start);
    }
    else
    {
        if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
        hc::parallel_for_each(
            acc_view,
            hc::tiled_extent<2>(rows, blocks_x * block_size, 1, block_size),
            [=](hc::tiled_index<2>) [[hc]]
            {
                fixed_array<SampleToBinOp, ActiveChannels> sample_to_bin_op_fixed(
                    sample_to_bin_op0, sample_to_bin_op1, sample_to_bin_op2, sample_to_bin_op3
                );

                histogram_global<block_size, items_per_thread, Channels, ActiveChannels>(
                    samples, columns, row_stride,
                    histogram_fixed,
                    sample_to_bin_op_fixed,
                    bins_bits_fixed
                );
            }
        );
        ROCPRIM_DETAIL_HC_SYNC("histogram_global", blocks_x * block_size * rows, start);
    }
}

template<
    unsigned int Channels,
    unsigned int ActiveChannels,
    class SampleIterator,
    class Counter,
    class Level
>
inline
void histogram_even_impl(void * temporary_storage,
                         size_t& storage_size,
                         SampleIterator samples,
                         unsigned int columns,
                         unsigned int rows,
                         size_t row_stride_bytes,
                         Counter * histogram[ActiveChannels],
                         unsigned int levels[ActiveChannels],
                         Level lower_level[ActiveChannels],
                         Level upper_level[ActiveChannels],
                         hc::accelerator_view& acc_view,
                         bool debug_synchronous)
{
    sample_to_bin_even<Level> sample_to_bin_op[ActiveChannels];
    for(unsigned int channel = 0; channel < ActiveChannels; channel++)
    {
        sample_to_bin_op[channel] = sample_to_bin_even<Level>(
            levels[channel] - 1,
            lower_level[channel], upper_level[channel]
        );
    }

    histogram_impl<Channels, ActiveChannels>(
        temporary_storage, storage_size,
        samples, columns, rows, row_stride_bytes,
        histogram,
        levels, sample_to_bin_op,
        acc_view, debug_synchronous
    );
}

template<
    unsigned int Channels,
    unsigned int ActiveChannels,
    class SampleIterator,
    class Counter,
    class Level
>
inline
void histogram_range_impl(void * temporary_storage,
                          size_t& storage_size,
                          SampleIterator samples,
                          unsigned int columns,
                          unsigned int rows,
                          size_t row_stride_bytes,
                          Counter * histogram[ActiveChannels],
                          unsigned int levels[ActiveChannels],
                          Level * level_values[ActiveChannels],
                          hc::accelerator_view& acc_view,
                          bool debug_synchronous)
{
    sample_to_bin_range<Level> sample_to_bin_op[ActiveChannels];
    for(unsigned int channel = 0; channel < ActiveChannels; channel++)
    {
        sample_to_bin_op[channel] = sample_to_bin_range<Level>(
            levels[channel] - 1,
            level_values[channel]
        );
    }

    histogram_impl<Channels, ActiveChannels>(
        temporary_storage, storage_size,
        samples, columns, rows, row_stride_bytes,
        histogram,
        levels, sample_to_bin_op,
        acc_view, debug_synchronous
    );
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
    Counter * histogram_single[1] = { histogram };
    unsigned int levels_single[1] = { levels };
    Level lower_level_single[1] = { lower_level };
    Level upper_level_single[1] = { upper_level };

    detail::histogram_even_impl<1, 1>(
        temporary_storage, storage_size,
        samples, size, 1, 0,
        histogram_single,
        levels_single, lower_level_single, upper_level_single,
        acc_view, debug_synchronous
    );
}

template<
    class SampleIterator,
    class Counter,
    class Level
>
inline
void histogram_even(void * temporary_storage,
                    size_t& storage_size,
                    SampleIterator samples,
                    unsigned int columns,
                    unsigned int rows,
                    size_t row_stride_bytes,
                    Counter * histogram,
                    unsigned int levels,
                    Level lower_level,
                    Level upper_level,
                    hc::accelerator_view acc_view = hc::accelerator().get_default_view(),
                    bool debug_synchronous = false)
{
    Counter * histogram_single[1] = { histogram };
    unsigned int levels_single[1] = { levels };
    Level lower_level_single[1] = { lower_level };
    Level upper_level_single[1] = { upper_level };

    detail::histogram_even_impl<1, 1>(
        temporary_storage, storage_size,
        samples, columns, rows, row_stride_bytes,
        histogram_single,
        levels_single, lower_level_single, upper_level_single,
        acc_view, debug_synchronous
    );
}

template<
    unsigned int Channels,
    unsigned int ActiveChannels,
    class SampleIterator,
    class Counter,
    class Level
>
inline
void multi_histogram_even(void * temporary_storage,
                          size_t& storage_size,
                          SampleIterator samples,
                          unsigned int size,
                          Counter * histogram[ActiveChannels],
                          unsigned int levels[ActiveChannels],
                          Level lower_level[ActiveChannels],
                          Level upper_level[ActiveChannels],
                          hc::accelerator_view acc_view = hc::accelerator().get_default_view(),
                          bool debug_synchronous = false)
{
    detail::histogram_even_impl<Channels, ActiveChannels>(
        temporary_storage, storage_size,
        samples, size, 1, 0,
        histogram,
        levels, lower_level, upper_level,
        acc_view, debug_synchronous
    );
}

template<
    unsigned int Channels,
    unsigned int ActiveChannels,
    class SampleIterator,
    class Counter,
    class Level
>
inline
void multi_histogram_even(void * temporary_storage,
                          size_t& storage_size,
                          SampleIterator samples,
                          unsigned int columns,
                          unsigned int rows,
                          size_t row_stride_bytes,
                          Counter * histogram[ActiveChannels],
                          unsigned int levels[ActiveChannels],
                          Level lower_level[ActiveChannels],
                          Level upper_level[ActiveChannels],
                          hc::accelerator_view acc_view = hc::accelerator().get_default_view(),
                          bool debug_synchronous = false)
{
    detail::histogram_even_impl<Channels, ActiveChannels>(
        temporary_storage, storage_size,
        samples, columns, rows, row_stride_bytes,
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
                     Level * level_values,
                     hc::accelerator_view acc_view = hc::accelerator().get_default_view(),
                     bool debug_synchronous = false)
{
    Counter * histogram_single[1] = { histogram };
    unsigned int levels_single[1] = { levels };
    Level * level_values_single[1] = { level_values };

    detail::histogram_range_impl<1, 1>(
        temporary_storage, storage_size,
        samples, size, 1, 0,
        histogram_single,
        levels_single, level_values_single,
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
                     unsigned int columns,
                     unsigned int rows,
                     size_t row_stride_bytes,
                     Counter * histogram,
                     unsigned int levels,
                     Level * level_values,
                     hc::accelerator_view acc_view = hc::accelerator().get_default_view(),
                     bool debug_synchronous = false)
{
    Counter * histogram_single[1] = { histogram };
    unsigned int levels_single[1] = { levels };
    Level * level_values_single[1] = { level_values };

    detail::histogram_range_impl<1, 1>(
        temporary_storage, storage_size,
        samples, columns, rows, row_stride_bytes,
        histogram_single,
        levels_single, level_values_single,
        acc_view, debug_synchronous
    );
}

template<
    unsigned int Channels,
    unsigned int ActiveChannels,
    class SampleIterator,
    class Counter,
    class Level
>
inline
void multi_histogram_range(void * temporary_storage,
                           size_t& storage_size,
                           SampleIterator samples,
                           unsigned int size,
                           Counter * histogram[ActiveChannels],
                           unsigned int levels[ActiveChannels],
                           Level * level_values[ActiveChannels],
                           hc::accelerator_view acc_view = hc::accelerator().get_default_view(),
                           bool debug_synchronous = false)
{
    detail::histogram_range_impl<Channels, ActiveChannels>(
        temporary_storage, storage_size,
        samples, size, 1, 0,
        histogram,
        levels, level_values,
        acc_view, debug_synchronous
    );
}

template<
    unsigned int Channels,
    unsigned int ActiveChannels,
    class SampleIterator,
    class Counter,
    class Level
>
inline
void multi_histogram_range(void * temporary_storage,
                           size_t& storage_size,
                           SampleIterator samples,
                           unsigned int columns,
                           unsigned int rows,
                           size_t row_stride_bytes,
                           Counter * histogram[ActiveChannels],
                           unsigned int levels[ActiveChannels],
                           Level * level_values[ActiveChannels],
                           hc::accelerator_view acc_view = hc::accelerator().get_default_view(),
                           bool debug_synchronous = false)
{
    detail::histogram_range_impl<Channels, ActiveChannels>(
        temporary_storage, storage_size,
        samples, columns, rows, row_stride_bytes,
        histogram,
        levels, level_values,
        acc_view, debug_synchronous
    );
}

/// @}
// end of group devicemodule_hc

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DEVICE_HISTOGRAM_HC_HPP_
