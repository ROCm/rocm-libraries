// Copyright (c) 2017-2019 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DEVICE_DEVICE_HISTOGRAM_HPP_
#define ROCPRIM_DEVICE_DEVICE_HISTOGRAM_HPP_

#include <cmath>
#include <type_traits>
#include <iterator>

#include "../config.hpp"
#include "../functional.hpp"
#include "../detail/various.hpp"

#include "device_histogram_config.hpp"
#include "detail/device_histogram.hpp"

BEGIN_ROCPRIM_NAMESPACE

/// \addtogroup devicemodule
/// @{

namespace detail
{

template<
    unsigned int BlockSize,
    unsigned int ActiveChannels,
    class Counter
>
__global__
__launch_bounds__(BlockSize, ROCPRIM_DEFAULT_MIN_WARPS_PER_EU)
void init_histogram_kernel(fixed_array<Counter *, ActiveChannels> histogram,
                           fixed_array<unsigned int, ActiveChannels> bins)
{
    init_histogram<BlockSize, ActiveChannels>(histogram, bins);
}

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    unsigned int Channels,
    unsigned int ActiveChannels,
    class SampleIterator,
    class Counter,
    class SampleToBinOp
>
__global__
__launch_bounds__(BlockSize, ROCPRIM_DEFAULT_MIN_WARPS_PER_EU)
void histogram_shared_kernel(SampleIterator samples,
                             unsigned int columns,
                             unsigned int rows,
                             unsigned int row_stride,
                             unsigned int rows_per_block,
                             fixed_array<Counter *, ActiveChannels> histogram,
                             fixed_array<SampleToBinOp, ActiveChannels> sample_to_bin_op,
                             fixed_array<unsigned int, ActiveChannels> bins)
{
    HIP_DYNAMIC_SHARED(unsigned int, block_histogram);

    histogram_shared<BlockSize, ItemsPerThread, Channels, ActiveChannels>(
        samples, columns, rows, row_stride, rows_per_block,
        histogram,
        sample_to_bin_op, bins,
        block_histogram
    );
}

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    unsigned int Channels,
    unsigned int ActiveChannels,
    class SampleIterator,
    class Counter,
    class SampleToBinOp
>
__global__
__launch_bounds__(BlockSize, ROCPRIM_DEFAULT_MIN_WARPS_PER_EU)
void histogram_global_kernel(SampleIterator samples,
                             unsigned int columns,
                             unsigned int row_stride,
                             fixed_array<Counter *, ActiveChannels> histogram,
                             fixed_array<SampleToBinOp, ActiveChannels> sample_to_bin_op,
                             fixed_array<unsigned int, ActiveChannels> bins_bits)
{
    histogram_global<BlockSize, ItemsPerThread, Channels, ActiveChannels>(
        samples, columns, row_stride,
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
    unsigned int Channels,
    unsigned int ActiveChannels,
    class Config,
    class SampleIterator,
    class Counter,
    class SampleToBinOp
>
inline
hipError_t histogram_impl(void * temporary_storage,
                          size_t& storage_size,
                          SampleIterator samples,
                          unsigned int columns,
                          unsigned int rows,
                          size_t row_stride_bytes,
                          Counter * histogram[ActiveChannels],
                          unsigned int levels[ActiveChannels],
                          SampleToBinOp sample_to_bin_op[ActiveChannels],
                          hipStream_t stream,
                          bool debug_synchronous)
{
    using sample_type = typename std::iterator_traits<SampleIterator>::value_type;

    using config = default_or_custom_config<
        Config,
        default_histogram_config<ROCPRIM_TARGET_ARCH, sample_type, Channels, ActiveChannels>
    >;

    constexpr unsigned int block_size = config::histogram::block_size;
    constexpr unsigned int items_per_thread = config::histogram::items_per_thread;
    constexpr unsigned int items_per_block = block_size * items_per_thread;

    if(row_stride_bytes % sizeof(sample_type) != 0)
    {
        // Row stride must be a whole multiple of the sample data type size
        return hipErrorInvalidValue;
    }

    const unsigned int blocks_x = ::rocprim::detail::ceiling_div(columns, items_per_block);
    const unsigned int row_stride = row_stride_bytes / sizeof(sample_type);

    if(temporary_storage == nullptr)
    {
        // Make sure user won't try to allocate 0 bytes memory, because
        // hipMalloc will return nullptr.
        storage_size = 4;
        return hipSuccess;
    }

    if(debug_synchronous)
    {
        std::cout << "columns " << columns << '\n';
        std::cout << "rows " << rows << '\n';
        std::cout << "blocks_x " << blocks_x << '\n';
        hipError_t error = hipStreamSynchronize(stream);
        if(error != hipSuccess) return error;
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

    std::chrono::high_resolution_clock::time_point start;

    if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(init_histogram_kernel<block_size, ActiveChannels>),
        dim3(::rocprim::detail::ceiling_div(max_bins, block_size)), dim3(block_size), 0, stream,
        fixed_array<Counter *, ActiveChannels>(histogram),
        fixed_array<unsigned int, ActiveChannels>(bins)
    );
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("init_histogram", max_bins, start);

    if(columns == 0 || rows == 0)
    {
        return hipSuccess;
    }

    if(total_bins <= config::shared_impl_max_bins)
    {
        dim3 grid_size;
        grid_size.x = std::min(config::max_grid_size, blocks_x);
        grid_size.y = std::min(rows, config::max_grid_size / grid_size.x);
        const size_t block_histogram_bytes = total_bins * sizeof(unsigned int);
        const unsigned int rows_per_block = ::rocprim::detail::ceiling_div(rows, grid_size.y);
        if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(histogram_shared_kernel<
                block_size, items_per_thread, Channels, ActiveChannels
            >),
            grid_size, dim3(block_size, 1), block_histogram_bytes, stream,
            samples, columns, rows, row_stride, rows_per_block,
            fixed_array<Counter *, ActiveChannels>(histogram),
            fixed_array<SampleToBinOp, ActiveChannels>(sample_to_bin_op),
            fixed_array<unsigned int, ActiveChannels>(bins)
        );
        ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("histogram_shared", grid_size.x * grid_size.y * block_size, start);
    }
    else
    {
        if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(histogram_global_kernel<
                block_size, items_per_thread, Channels, ActiveChannels
            >),
            dim3(blocks_x, rows), dim3(block_size, 1), 0, stream,
            samples, columns, row_stride,
            fixed_array<Counter *, ActiveChannels>(histogram),
            fixed_array<SampleToBinOp, ActiveChannels>(sample_to_bin_op),
            fixed_array<unsigned int, ActiveChannels>(bins_bits)
        );
        ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("histogram_global", blocks_x * block_size * rows, start);
    }

    return hipSuccess;
}

template<
    unsigned int Channels,
    unsigned int ActiveChannels,
    class Config,
    class SampleIterator,
    class Counter,
    class Level
>
inline
hipError_t histogram_even_impl(void * temporary_storage,
                               size_t& storage_size,
                               SampleIterator samples,
                               unsigned int columns,
                               unsigned int rows,
                               size_t row_stride_bytes,
                               Counter * histogram[ActiveChannels],
                               unsigned int levels[ActiveChannels],
                               Level lower_level[ActiveChannels],
                               Level upper_level[ActiveChannels],
                               hipStream_t stream,
                               bool debug_synchronous)
{
    for(unsigned int channel = 0; channel < ActiveChannels; channel++)
    {
        if(levels[channel] < 2)
        {
            // Histogram must have at least 1 bin
            return hipErrorInvalidValue;
        }
    }

    sample_to_bin_even<Level> sample_to_bin_op[ActiveChannels];
    for(unsigned int channel = 0; channel < ActiveChannels; channel++)
    {
        sample_to_bin_op[channel] = sample_to_bin_even<Level>(
            levels[channel] - 1,
            lower_level[channel], upper_level[channel]
        );
    }

    return histogram_impl<Channels, ActiveChannels, Config>(
        temporary_storage, storage_size,
        samples, columns, rows, row_stride_bytes,
        histogram,
        levels, sample_to_bin_op,
        stream, debug_synchronous
    );
}

template<
    unsigned int Channels,
    unsigned int ActiveChannels,
    class Config,
    class SampleIterator,
    class Counter,
    class Level
>
inline
hipError_t histogram_range_impl(void * temporary_storage,
                                size_t& storage_size,
                                SampleIterator samples,
                                unsigned int columns,
                                unsigned int rows,
                                size_t row_stride_bytes,
                                Counter * histogram[ActiveChannels],
                                unsigned int levels[ActiveChannels],
                                Level * level_values[ActiveChannels],
                                hipStream_t stream,
                                bool debug_synchronous)
{
    for(unsigned int channel = 0; channel < ActiveChannels; channel++)
    {
        if(levels[channel] < 2)
        {
            // Histogram must have at least 1 bin
            return hipErrorInvalidValue;
        }
    }

    sample_to_bin_range<Level> sample_to_bin_op[ActiveChannels];
    for(unsigned int channel = 0; channel < ActiveChannels; channel++)
    {
        sample_to_bin_op[channel] = sample_to_bin_range<Level>(
            levels[channel] - 1,
            level_values[channel]
        );
    }

    return histogram_impl<Channels, ActiveChannels, Config>(
        temporary_storage, storage_size,
        samples, columns, rows, row_stride_bytes,
        histogram,
        levels, sample_to_bin_op,
        stream, debug_synchronous
    );
}

#undef ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR

} // end of detail namespace

/// \brief Computes a histogram from a sequence of samples using equal-width bins.
///
/// \par
/// * The number of histogram bins is (\p levels - 1).
/// * Bins are evenly-segmented and include the same width of sample values:
/// (\p upper_level - \p lower_level) / (\p levels - 1).
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage in a null pointer.
///
/// \tparam Config - [optional] configuration of the primitive. It can be \p histogram_config or
/// a custom class with the same members.
/// \tparam SampleIterator - random-access iterator type of the input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam Counter - integer type for histogram bin counters.
/// \tparam Level - type of histogram boundaries (levels)
///
/// \param [in] temporary_storage - pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the reduction operation.
/// \param [in,out] storage_size - reference to a size (in bytes) of \p temporary_storage.
/// \param [in] samples - iterator to the first element in the range of input samples.
/// \param [in] size - number of elements in the samples range.
/// \param [out] histogram - pointer to the first element in the histogram range.
/// \param [in] levels - number of boundaries (levels) for histogram bins.
/// \param [in] lower_level - lower sample value bound (inclusive) for the first histogram bin.
/// \param [in] upper_level - upper sample value bound (exclusive) for the last histogram bin.
/// \param [in] stream - [optional] HIP stream object. Default is \p 0 (default stream).
/// \param [in] debug_synchronous - [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. Default value is \p false.
///
/// \returns \p hipSuccess (\p 0) after successful histogram operation; otherwise a HIP runtime error of
/// type \p hipError_t.
///
/// \par Example
/// \parblock
/// In this example a device-level histogram of 5 bins is computed on an array of float samples.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// unsigned int size;        // e.g., 8
/// float * samples;          // e.g., [-10.0, 0.3, 9.5, 8.1, 1.5, 1.9, 100.0, 5.1]
/// int * histogram;          // empty array of at least 5 elements
/// unsigned int levels;      // e.g., 6 (for 5 bins)
/// float lower_level;        // e.g., 0.0
/// float upper_level;        // e.g., 10.0
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::histogram_even(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     samples, size,
///     histogram, levels, lower_level, upper_level
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // compute histogram
/// rocprim::histogram_even(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     samples, size,
///     histogram, levels, lower_level, upper_level
/// );
/// // histogram: [3, 0, 1, 0, 2]
/// \endcode
/// \endparblock
template<
    class Config = default_config,
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
    Counter * histogram_single[1] = { histogram };
    unsigned int levels_single[1] = { levels };
    Level lower_level_single[1] = { lower_level };
    Level upper_level_single[1] = { upper_level };

    return detail::histogram_even_impl<1, 1, Config>(
        temporary_storage, storage_size,
        samples, size, 1, 0,
        histogram_single,
        levels_single, lower_level_single, upper_level_single,
        stream, debug_synchronous
    );
}

/// \brief Computes a histogram from a two-dimensional region of samples using equal-width bins.
///
/// \par
/// * The two-dimensional region of interest within \p samples can be specified using the \p columns,
/// \p rows and \p row_stride_bytes parameters.
/// * The row stride must be a whole multiple of the sample data type size,
/// i.e., <tt>(row_stride_bytes % sizeof(std::iterator_traits<SampleIterator>::value_type)) == 0</tt>.
/// * The number of histogram bins is (\p levels - 1).
/// * Bins are evenly-segmented and include the same width of sample values:
/// (\p upper_level - \p lower_level) / (\p levels - 1).
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage in a null pointer.
///
/// \tparam Config - [optional] configuration of the primitive. It can be \p histogram_config or
/// a custom class with the same members.
/// \tparam SampleIterator - random-access iterator type of the input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam Counter - integer type for histogram bin counters.
/// \tparam Level - type of histogram boundaries (levels)
///
/// \param [in] temporary_storage - pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the reduction operation.
/// \param [in,out] storage_size - reference to a size (in bytes) of \p temporary_storage.
/// \param [in] samples - iterator to the first element in the range of input samples.
/// \param [in] columns - number of elements in each row of the region.
/// \param [in] rows - number of rows of the region.
/// \param [in] row_stride_bytes - number of bytes between starts of consecutive rows of the region.
/// \param [out] histogram - pointer to the first element in the histogram range.
/// \param [in] levels - number of boundaries (levels) for histogram bins.
/// \param [in] lower_level - lower sample value bound (inclusive) for the first histogram bin.
/// \param [in] upper_level - upper sample value bound (exclusive) for the last histogram bin.
/// \param [in] stream - [optional] HIP stream object. Default is \p 0 (default stream).
/// \param [in] debug_synchronous - [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. Default value is \p false.
///
/// \returns \p hipSuccess (\p 0) after successful histogram operation; otherwise a HIP runtime error of
/// type \p hipError_t.
///
/// \par Example
/// \parblock
/// In this example a device-level histogram of 5 bins is computed on an array of float samples.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// unsigned int columns;     // e.g., 4
/// unsigned int rows;        // e.g., 2
/// size_t row_stride_bytes;  // e.g., 6 * sizeof(float)
/// float * samples;          // e.g., [-10.0, 0.3, 9.5, 8.1, -, -, 1.5, 1.9, 100.0, 5.1, -, -]
/// int * histogram;          // empty array of at least 5 elements
/// unsigned int levels;      // e.g., 6 (for 5 bins)
/// float lower_level;        // e.g., 0.0
/// float upper_level;        // e.g., 10.0
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::histogram_even(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     samples, columns, rows, row_stride_bytes,
///     histogram, levels, lower_level, upper_level
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // compute histogram
/// rocprim::histogram_even(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     samples, columns, rows, row_stride_bytes,
///     histogram, levels, lower_level, upper_level
/// );
/// // histogram: [3, 0, 1, 0, 2]
/// \endcode
/// \endparblock
template<
    class Config = default_config,
    class SampleIterator,
    class Counter,
    class Level
>
inline
hipError_t histogram_even(void * temporary_storage,
                          size_t& storage_size,
                          SampleIterator samples,
                          unsigned int columns,
                          unsigned int rows,
                          size_t row_stride_bytes,
                          Counter * histogram,
                          unsigned int levels,
                          Level lower_level,
                          Level upper_level,
                          hipStream_t stream = 0,
                          bool debug_synchronous = false)
{
    Counter * histogram_single[1] = { histogram };
    unsigned int levels_single[1] = { levels };
    Level lower_level_single[1] = { lower_level };
    Level upper_level_single[1] = { upper_level };

    return detail::histogram_even_impl<1, 1, Config>(
        temporary_storage, storage_size,
        samples, columns, rows, row_stride_bytes,
        histogram_single,
        levels_single, lower_level_single, upper_level_single,
        stream, debug_synchronous
    );
}

/// \brief Computes histograms from a sequence of multi-channel samples using equal-width bins.
///
/// \par
/// * The input is a sequence of <em>pixel</em> structures, where each pixel comprises
/// a record of \p Channels consecutive data samples (e.g., \p Channels = 4 for <em>RGBA</em> samples).
/// * The first \p ActiveChannels channels of total \p Channels channels will be used for computing histograms
/// (e.g., \p ActiveChannels = 3 for computing histograms of only <em>RGB</em> from <em>RGBA</em> samples).
/// * For channel<sub><em>i</em></sub> the number of histogram bins is (\p levels[i] - 1).
/// * For channel<sub><em>i</em></sub> bins are evenly-segmented and include the same width of sample values:
/// (\p upper_level[i] - \p lower_level[i]) / (\p levels[i] - 1).
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage in a null pointer.
///
/// \tparam Channels - number of channels interleaved in the input samples.
/// \tparam ActiveChannels - number of channels being used for computing histograms.
/// \tparam Config - [optional] configuration of the primitive. It can be \p histogram_config or
/// a custom class with the same members.
/// \tparam SampleIterator - random-access iterator type of the input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam Counter - integer type for histogram bin counters.
/// \tparam Level - type of histogram boundaries (levels)
///
/// \param [in] temporary_storage - pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the reduction operation.
/// \param [in,out] storage_size - reference to a size (in bytes) of \p temporary_storage.
/// \param [in] samples - iterator to the first element in the range of input samples.
/// \param [in] size - number of pixels in the samples range.
/// \param [out] histogram - pointers to the first element in the histogram range, one for each active channel.
/// \param [in] levels - number of boundaries (levels) for histogram bins in each active channel.
/// \param [in] lower_level - lower sample value bound (inclusive) for the first histogram bin in each active channel.
/// \param [in] upper_level - upper sample value bound (exclusive) for the last histogram bin in each active channel.
/// \param [in] stream - [optional] HIP stream object. Default is \p 0 (default stream).
/// \param [in] debug_synchronous - [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. Default value is \p false.
///
/// \returns \p hipSuccess (\p 0) after successful histogram operation; otherwise a HIP runtime error of
/// type \p hipError_t.
///
/// \par Example
/// \parblock
/// In this example histograms for 3 channels (RGB) are computed on an array of 8-bit RGBA samples.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// unsigned int size;        // e.g., 8
/// unsigned char * samples;  // e.g., [(3, 1, 5, 255), (3, 1, 5, 255), (4, 2, 6, 127), (3, 2, 6, 127),
///                           //        (0, 0, 0, 100), (0, 1, 0, 100), (0, 0, 1, 255), (0, 1, 1, 255)]
/// int * histogram[3];       // 3 empty arrays of at least 256 elements each
/// unsigned int levels[3];   // e.g., [257, 257, 257] (for 256 bins)
/// int lower_level[3];       // e.g., [0, 0, 0]
/// int upper_level[3];       // e.g., [256, 256, 256]
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::multi_histogram_even<4, 3>(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     samples, size,
///     histogram, levels, lower_level, upper_level
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // compute histograms
/// rocprim::multi_histogram_even<4, 3>(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     samples, size,
///     histogram, levels, lower_level, upper_level
/// );
/// // histogram: [[4, 0, 0, 3, 1, 0, 0, ..., 0],
/// //             [2, 4, 2, 0, 0, 0, 0, ..., 0],
/// //             [2, 2, 0, 0, 0, 2, 2, ..., 0]]
/// \endcode
/// \endparblock
template<
    unsigned int Channels,
    unsigned int ActiveChannels,
    class Config = default_config,
    class SampleIterator,
    class Counter,
    class Level
>
inline
hipError_t multi_histogram_even(void * temporary_storage,
                                size_t& storage_size,
                                SampleIterator samples,
                                unsigned int size,
                                Counter * histogram[ActiveChannels],
                                unsigned int levels[ActiveChannels],
                                Level lower_level[ActiveChannels],
                                Level upper_level[ActiveChannels],
                                hipStream_t stream = 0,
                                bool debug_synchronous = false)
{
    return detail::histogram_even_impl<Channels, ActiveChannels, Config>(
        temporary_storage, storage_size,
        samples, size, 1, 0,
        histogram,
        levels, lower_level, upper_level,
        stream, debug_synchronous
    );
}

/// \brief Computes histograms from a two-dimensional region of multi-channel samples using equal-width bins.
///
/// \par
/// * The two-dimensional region of interest within \p samples can be specified using the \p columns,
/// \p rows and \p row_stride_bytes parameters.
/// * The row stride must be a whole multiple of the sample data type size,
/// i.e., <tt>(row_stride_bytes % sizeof(std::iterator_traits<SampleIterator>::value_type)) == 0</tt>.
/// * The input is a sequence of <em>pixel</em> structures, where each pixel comprises
/// a record of \p Channels consecutive data samples (e.g., \p Channels = 4 for <em>RGBA</em> samples).
/// * The first \p ActiveChannels channels of total \p Channels channels will be used for computing histograms
/// (e.g., \p ActiveChannels = 3 for computing histograms of only <em>RGB</em> from <em>RGBA</em> samples).
/// * For channel<sub><em>i</em></sub> the number of histogram bins is (\p levels[i] - 1).
/// * For channel<sub><em>i</em></sub> bins are evenly-segmented and include the same width of sample values:
/// (\p upper_level[i] - \p lower_level[i]) / (\p levels[i] - 1).
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage in a null pointer.
///
/// \tparam Channels - number of channels interleaved in the input samples.
/// \tparam ActiveChannels - number of channels being used for computing histograms.
/// \tparam Config - [optional] configuration of the primitive. It can be \p histogram_config or
/// a custom class with the same members.
/// \tparam SampleIterator - random-access iterator type of the input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam Counter - integer type for histogram bin counters.
/// \tparam Level - type of histogram boundaries (levels)
///
/// \param [in] temporary_storage - pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the reduction operation.
/// \param [in,out] storage_size - reference to a size (in bytes) of \p temporary_storage.
/// \param [in] samples - iterator to the first element in the range of input samples.
/// \param [in] columns - number of elements in each row of the region.
/// \param [in] rows - number of rows of the region.
/// \param [in] row_stride_bytes - number of bytes between starts of consecutive rows of the region.
/// \param [out] histogram - pointers to the first element in the histogram range, one for each active channel.
/// \param [in] levels - number of boundaries (levels) for histogram bins in each active channel.
/// \param [in] lower_level - lower sample value bound (inclusive) for the first histogram bin in each active channel.
/// \param [in] upper_level - upper sample value bound (exclusive) for the last histogram bin in each active channel.
/// \param [in] stream - [optional] HIP stream object. Default is \p 0 (default stream).
/// \param [in] debug_synchronous - [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. Default value is \p false.
///
/// \returns \p hipSuccess (\p 0) after successful histogram operation; otherwise a HIP runtime error of
/// type \p hipError_t.
///
/// \par Example
/// \parblock
/// In this example histograms for 3 channels (RGB) are computed on an array of 8-bit RGBA samples.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// unsigned int columns;     // e.g., 4
/// unsigned int rows;        // e.g., 2
/// size_t row_stride_bytes;  // e.g., 5 * sizeof(unsigned char)
/// unsigned char * samples;  // e.g., [(3, 1, 5, 255), (3, 1, 5, 255), (4, 2, 6, 127), (3, 2, 6, 127), (-, -, -, -),
///                           //        (0, 0, 0, 100), (0, 1, 0, 100), (0, 0, 1, 255), (0, 1, 1, 255), (-, -, -, -)]
/// int * histogram[3];       // 3 empty arrays of at least 256 elements each
/// unsigned int levels[3];   // e.g., [257, 257, 257] (for 256 bins)
/// int lower_level[3];       // e.g., [0, 0, 0]
/// int upper_level[3];       // e.g., [256, 256, 256]
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::multi_histogram_even<4, 3>(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     samples, columns, rows, row_stride_bytes,
///     histogram, levels, lower_level, upper_level
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // compute histograms
/// rocprim::multi_histogram_even<4, 3>(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     samples, columns, rows, row_stride_bytes,
///     histogram, levels, lower_level, upper_level
/// );
/// // histogram: [[4, 0, 0, 3, 1, 0, 0, ..., 0],
/// //             [2, 4, 2, 0, 0, 0, 0, ..., 0],
/// //             [2, 2, 0, 0, 0, 2, 2, ..., 0]]
/// \endcode
/// \endparblock
template<
    unsigned int Channels,
    unsigned int ActiveChannels,
    class Config = default_config,
    class SampleIterator,
    class Counter,
    class Level
>
inline
hipError_t multi_histogram_even(void * temporary_storage,
                                size_t& storage_size,
                                SampleIterator samples,
                                unsigned int columns,
                                unsigned int rows,
                                size_t row_stride_bytes,
                                Counter * histogram[ActiveChannels],
                                unsigned int levels[ActiveChannels],
                                Level lower_level[ActiveChannels],
                                Level upper_level[ActiveChannels],
                                hipStream_t stream = 0,
                                bool debug_synchronous = false)
{
    return detail::histogram_even_impl<Channels, ActiveChannels, Config>(
        temporary_storage, storage_size,
        samples, columns, rows, row_stride_bytes,
        histogram,
        levels, lower_level, upper_level,
        stream, debug_synchronous
    );
}

/// \brief Computes a histogram from a sequence of samples using the specified bin boundary levels.
///
/// \par
/// * The number of histogram bins is (\p levels - 1).
/// * The range for bin<sub><em>j</em></sub> is [<tt>level_values[j]</tt>, <tt>level_values[j+1]</tt>).
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage in a null pointer.
///
/// \tparam Config - [optional] configuration of the primitive. It can be \p histogram_config or
/// a custom class with the same members.
/// \tparam SampleIterator - random-access iterator type of the input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam Counter - integer type for histogram bin counters.
/// \tparam Level - type of histogram boundaries (levels)
///
/// \param [in] temporary_storage - pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the reduction operation.
/// \param [in,out] storage_size - reference to a size (in bytes) of \p temporary_storage.
/// \param [in] samples - iterator to the first element in the range of input samples.
/// \param [in] size - number of elements in the samples range.
/// \param [out] histogram - pointer to the first element in the histogram range.
/// \param [in] levels - number of boundaries (levels) for histogram bins.
/// \param [in] level_values - pointer to the array of bin boundaries.
/// \param [in] stream - [optional] HIP stream object. Default is \p 0 (default stream).
/// \param [in] debug_synchronous - [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. Default value is \p false.
///
/// \returns \p hipSuccess (\p 0) after successful histogram operation; otherwise a HIP runtime error of
/// type \p hipError_t.
///
/// \par Example
/// \parblock
/// In this example a device-level histogram of 5 bins is computed on an array of float samples.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// unsigned int size;        // e.g., 8
/// float * samples;          // e.g., [-10.0, 0.3, 9.5, 8.1, 1.5, 1.9, 100.0, 5.1]
/// int * histogram;          // empty array of at least 5 elements
/// unsigned int levels;      // e.g., 6 (for 5 bins)
/// float * level_values;     // e.g., [0.0, 1.0, 5.0, 10.0, 20.0, 50.0]
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::histogram_range(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     samples, size,
///     histogram, levels, level_values
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // compute histogram
/// rocprim::histogram_range(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     samples, size,
///     histogram, levels, level_values
/// );
/// // histogram: [1, 2, 3, 0, 0]
/// \endcode
/// \endparblock
template<
    class Config = default_config,
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
                           Level * level_values,
                           hipStream_t stream = 0,
                           bool debug_synchronous = false)
{
    Counter * histogram_single[1] = { histogram };
    unsigned int levels_single[1] = { levels };
    Level * level_values_single[1] = { level_values };

    return detail::histogram_range_impl<1, 1, Config>(
        temporary_storage, storage_size,
        samples, size, 1, 0,
        histogram_single,
        levels_single, level_values_single,
        stream, debug_synchronous
    );
}

/// \brief Computes a histogram from a two-dimensional region of samples using the specified bin boundary levels.
///
/// \par
/// * The two-dimensional region of interest within \p samples can be specified using the \p columns,
/// \p rows and \p row_stride_bytes parameters.
/// * The row stride must be a whole multiple of the sample data type size,
/// i.e., <tt>(row_stride_bytes % sizeof(std::iterator_traits<SampleIterator>::value_type)) == 0</tt>.
/// * The number of histogram bins is (\p levels - 1).
/// * The range for bin<sub><em>j</em></sub> is [<tt>level_values[j]</tt>, <tt>level_values[j+1]</tt>).
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage in a null pointer.
///
/// \tparam Config - [optional] configuration of the primitive. It can be \p histogram_config or
/// a custom class with the same members.
/// \tparam SampleIterator - random-access iterator type of the input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam Counter - integer type for histogram bin counters.
/// \tparam Level - type of histogram boundaries (levels)
///
/// \param [in] temporary_storage - pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the reduction operation.
/// \param [in,out] storage_size - reference to a size (in bytes) of \p temporary_storage.
/// \param [in] samples - iterator to the first element in the range of input samples.
/// \param [in] columns - number of elements in each row of the region.
/// \param [in] rows - number of rows of the region.
/// \param [in] row_stride_bytes - number of bytes between starts of consecutive rows of the region.
/// \param [out] histogram - pointer to the first element in the histogram range.
/// \param [in] levels - number of boundaries (levels) for histogram bins.
/// \param [in] level_values - pointer to the array of bin boundaries.
/// \param [in] stream - [optional] HIP stream object. Default is \p 0 (default stream).
/// \param [in] debug_synchronous - [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. Default value is \p false.
///
/// \returns \p hipSuccess (\p 0) after successful histogram operation; otherwise a HIP runtime error of
/// type \p hipError_t.
///
/// \par Example
/// \parblock
/// In this example a device-level histogram of 5 bins is computed on an array of float samples.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// unsigned int columns;     // e.g., 4
/// unsigned int rows;        // e.g., 2
/// size_t row_stride_bytes;  // e.g., 6 * sizeof(float)
/// float * samples;          // e.g., [-10.0, 0.3, 9.5, 8.1, 1.5, 1.9, 100.0, 5.1]
/// int * histogram;          // empty array of at least 5 elements
/// unsigned int levels;      // e.g., 6 (for 5 bins)
/// float level_values;       // e.g., [0.0, 1.0, 5.0, 10.0, 20.0, 50.0]
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::histogram_range(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     samples, columns, rows, row_stride_bytes,
///     histogram, levels, level_values
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // compute histogram
/// rocprim::histogram_range(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     samples, columns, rows, row_stride_bytes,
///     histogram, levels, level_values
/// );
/// // histogram: [1, 2, 3, 0, 0]
/// \endcode
/// \endparblock
template<
    class Config = default_config,
    class SampleIterator,
    class Counter,
    class Level
>
inline
hipError_t histogram_range(void * temporary_storage,
                           size_t& storage_size,
                           SampleIterator samples,
                           unsigned int columns,
                           unsigned int rows,
                           size_t row_stride_bytes,
                           Counter * histogram,
                           unsigned int levels,
                           Level * level_values,
                           hipStream_t stream = 0,
                           bool debug_synchronous = false)
{
    Counter * histogram_single[1] = { histogram };
    unsigned int levels_single[1] = { levels };
    Level * level_values_single[1] = { level_values };

    return detail::histogram_range_impl<1, 1, Config>(
        temporary_storage, storage_size,
        samples, columns, rows, row_stride_bytes,
        histogram_single,
        levels_single, level_values_single,
        stream, debug_synchronous
    );
}

/// \brief Computes histograms from a sequence of multi-channel samples using the specified bin boundary levels.
///
/// \par
/// * The input is a sequence of <em>pixel</em> structures, where each pixel comprises
/// a record of \p Channels consecutive data samples (e.g., \p Channels = 4 for <em>RGBA</em> samples).
/// * The first \p ActiveChannels channels of total \p Channels channels will be used for computing histograms
/// (e.g., \p ActiveChannels = 3 for computing histograms of only <em>RGB</em> from <em>RGBA</em> samples).
/// * For channel<sub><em>i</em></sub> the number of histogram bins is (\p levels[i] - 1).
/// * For channel<sub><em>i</em></sub> the range for bin<sub><em>j</em></sub> is
/// [<tt>level_values[i][j]</tt>, <tt>level_values[i][j+1]</tt>).
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage in a null pointer.
///
/// \tparam Channels - number of channels interleaved in the input samples.
/// \tparam ActiveChannels - number of channels being used for computing histograms.
/// \tparam Config - [optional] configuration of the primitive. It can be \p histogram_config or
/// a custom class with the same members.
/// \tparam SampleIterator - random-access iterator type of the input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam Counter - integer type for histogram bin counters.
/// \tparam Level - type of histogram boundaries (levels)
///
/// \param [in] temporary_storage - pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the reduction operation.
/// \param [in,out] storage_size - reference to a size (in bytes) of \p temporary_storage.
/// \param [in] samples - iterator to the first element in the range of input samples.
/// \param [in] size - number of pixels in the samples range.
/// \param [out] histogram - pointers to the first element in the histogram range, one for each active channel.
/// \param [in] levels - number of boundaries (levels) for histogram bins in each active channel.
/// \param [in] level_values - pointer to the array of bin boundaries for each active channel.
/// \param [in] stream - [optional] HIP stream object. Default is \p 0 (default stream).
/// \param [in] debug_synchronous - [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. Default value is \p false.
///
/// \returns \p hipSuccess (\p 0) after successful histogram operation; otherwise a HIP runtime error of
/// type \p hipError_t.
///
/// \par Example
/// \parblock
/// In this example histograms for 3 channels (RGB) are computed on an array of 8-bit RGBA samples.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// unsigned int size;        // e.g., 8
/// unsigned char * samples;  // e.g., [(0, 0, 80, 255), (120, 0, 80, 255), (123, 0, 82, 127), (10, 1, 83, 127),
///                           //        (51, 1, 8, 100), (52, 1, 8, 100), (53, 0, 81, 255), (54, 50, 81, 255)]
/// int * histogram[3];       // 3 empty arrays of at least 256 elements each
/// unsigned int levels[3];   // e.g., [4, 4, 3]
/// int * level_values[3];    // e.g., [[0, 50, 100, 200], [0, 20, 40, 60], [0, 10, 100]]
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::multi_histogram_range<4, 3>(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     samples, size,
///     histogram, levels, level_values
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // compute histograms
/// rocprim::multi_histogram_range<4, 3>(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     samples, size,
///     histogram, levels, level_values
/// );
/// // histogram: [[2, 4, 2], [7, 0, 1], [2, 6]]
/// \endcode
/// \endparblock
template<
    unsigned int Channels,
    unsigned int ActiveChannels,
    class Config = default_config,
    class SampleIterator,
    class Counter,
    class Level
>
inline
hipError_t multi_histogram_range(void * temporary_storage,
                                 size_t& storage_size,
                                 SampleIterator samples,
                                 unsigned int size,
                                 Counter * histogram[ActiveChannels],
                                 unsigned int levels[ActiveChannels],
                                 Level * level_values[ActiveChannels],
                                 hipStream_t stream = 0,
                                 bool debug_synchronous = false)
{
    return detail::histogram_range_impl<Channels, ActiveChannels, Config>(
        temporary_storage, storage_size,
        samples, size, 1, 0,
        histogram,
        levels, level_values,
        stream, debug_synchronous
    );
}

/// \brief Computes histograms from a two-dimensional region of multi-channel samples using the specified bin
/// boundary levels.
///
/// \par
/// * The two-dimensional region of interest within \p samples can be specified using the \p columns,
/// \p rows and \p row_stride_bytes parameters.
/// * The row stride must be a whole multiple of the sample data type size,
/// i.e., <tt>(row_stride_bytes % sizeof(std::iterator_traits<SampleIterator>::value_type)) == 0</tt>.
/// * The input is a sequence of <em>pixel</em> structures, where each pixel comprises
/// a record of \p Channels consecutive data samples (e.g., \p Channels = 4 for <em>RGBA</em> samples).
/// * The first \p ActiveChannels channels of total \p Channels channels will be used for computing histograms
/// (e.g., \p ActiveChannels = 3 for computing histograms of only <em>RGB</em> from <em>RGBA</em> samples).
/// * For channel<sub><em>i</em></sub> the number of histogram bins is (\p levels[i] - 1).
/// * For channel<sub><em>i</em></sub> the range for bin<sub><em>j</em></sub> is
/// [<tt>level_values[i][j]</tt>, <tt>level_values[i][j+1]</tt>).
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage in a null pointer.
///
/// \tparam Channels - number of channels interleaved in the input samples.
/// \tparam ActiveChannels - number of channels being used for computing histograms.
/// \tparam Config - [optional] configuration of the primitive. It can be \p histogram_config or
/// a custom class with the same members.
/// \tparam SampleIterator - random-access iterator type of the input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam Counter - integer type for histogram bin counters.
/// \tparam Level - type of histogram boundaries (levels)
///
/// \param [in] temporary_storage - pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the reduction operation.
/// \param [in,out] storage_size - reference to a size (in bytes) of \p temporary_storage.
/// \param [in] samples - iterator to the first element in the range of input samples.
/// \param [in] columns - number of elements in each row of the region.
/// \param [in] rows - number of rows of the region.
/// \param [in] row_stride_bytes - number of bytes between starts of consecutive rows of the region.
/// \param [out] histogram - pointers to the first element in the histogram range, one for each active channel.
/// \param [in] levels - number of boundaries (levels) for histogram bins in each active channel.
/// \param [in] level_values - pointer to the array of bin boundaries for each active channel.
/// \param [in] stream - [optional] HIP stream object. Default is \p 0 (default stream).
/// \param [in] debug_synchronous - [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. Default value is \p false.
///
/// \returns \p hipSuccess (\p 0) after successful histogram operation; otherwise a HIP runtime error of
/// type \p hipError_t.
///
/// \par Example
/// \parblock
/// In this example histograms for 3 channels (RGB) are computed on an array of 8-bit RGBA samples.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// unsigned int columns;     // e.g., 4
/// unsigned int rows;        // e.g., 2
/// size_t row_stride_bytes;  // e.g., 5 * sizeof(unsigned char)
/// unsigned char * samples;  // e.g., [(0, 0, 80, 0), (120, 0, 80, 0), (123, 0, 82, 0), (10, 1, 83, 0), (-, -, -, -),
///                           //        (51, 1, 8, 0), (52, 1, 8, 0), (53, 0, 81, 0), (54, 50, 81, 0), (-, -, -, -)]
/// int * histogram[3];       // 3 empty arrays
/// unsigned int levels[3];   // e.g., [4, 4, 3]
/// int * level_values[3];    // e.g., [[0, 50, 100, 200], [0, 20, 40, 60], [0, 10, 100]]
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::multi_histogram_range<4, 3>(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     samples, columns, rows, row_stride_bytes,
///     histogram, levels, level_values
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // compute histograms
/// rocprim::multi_histogram_range<4, 3>(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     samples, columns, rows, row_stride_bytes,
///     histogram, levels, level_values
/// );
/// // histogram: [[2, 4, 2], [7, 0, 1], [2, 6]]
/// \endcode
/// \endparblock
template<
    unsigned int Channels,
    unsigned int ActiveChannels,
    class Config = default_config,
    class SampleIterator,
    class Counter,
    class Level
>
inline
hipError_t multi_histogram_range(void * temporary_storage,
                                 size_t& storage_size,
                                 SampleIterator samples,
                                 unsigned int columns,
                                 unsigned int rows,
                                 size_t row_stride_bytes,
                                 Counter * histogram[ActiveChannels],
                                 unsigned int levels[ActiveChannels],
                                 Level * level_values[ActiveChannels],
                                 hipStream_t stream = 0,
                                 bool debug_synchronous = false)
{
    return detail::histogram_range_impl<Channels, ActiveChannels, Config>(
        temporary_storage, storage_size,
        samples, columns, rows, row_stride_bytes,
        histogram,
        levels, level_values,
        stream, debug_synchronous
    );
}

/// @}
// end of group devicemodule

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DEVICE_HISTOGRAM_HPP_
