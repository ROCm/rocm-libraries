// Copyright (c) 2017-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include <iostream>
#include <iterator>
#include <type_traits>

#include "../common.hpp"
#include "../config.hpp"
#include "../detail/various.hpp"
#include "../functional.hpp"

#include "config_types.hpp"
#include "detail/device_histogram.hpp"
#include "device_histogram_config.hpp"

BEGIN_ROCPRIM_NAMESPACE

/// \addtogroup devicemodule
/// @{

namespace detail
{

template<class Config, unsigned int ActiveChannels, class Counter>
inline hipError_t launch_init_histogram(detail::target_arch                       arch,
                                        fixed_array<Counter*, ActiveChannels>     histogram,
                                        const fixed_array<size_t, ActiveChannels> bins,
                                        dim3                                      grid,
                                        dim3                                      block,
                                        size_t                                    shmem,
                                        hipStream_t                               stream)
{
    auto kernel = [=](auto arch_config)
    {
        static constexpr histogram_config_params params = decltype(arch_config)::params;
        init_histogram<params.histogram_config.block_size, ActiveChannels>(histogram, bins);
    };

    return execute_launch_plan<Config, decltype(kernel), histogram_config_selector>(arch,
                                                                                    kernel,
                                                                                    grid,
                                                                                    block,
                                                                                    shmem,
                                                                                    stream);
}

template<class ArchConfig,
         unsigned int Channels,
         unsigned int ActiveChannels,
         class SampleIterator,
         class Counter,
         class SampleToBinOp>
ROCPRIM_DEVICE
void histogram_shared_kernel_impl(SampleIterator                        samples,
                                  unsigned int                          columns,
                                  unsigned int                          rows,
                                  unsigned int                          row_stride,
                                  unsigned int                          rows_per_block,
                                  unsigned int                          shared_histograms,
                                  fixed_array<Counter*, ActiveChannels> histogram,
                                  const fixed_array<SampleToBinOp, ActiveChannels> sample_to_bin_op,
                                  const fixed_array<size_t, ActiveChannels>        bins)
{
    static constexpr histogram_config_params params = ArchConfig::params;

// Temporary fix: issue with dynamic shared memory on windows.
#ifndef _WIN32
    HIP_DYNAMIC_SHARED(unsigned int, block_histogram);
#else
    __shared__ unsigned int block_histogram[params.shared_impl_max_bins];
#endif

    histogram_shared<params.histogram_config.block_size,
                     params.histogram_config.items_per_thread,
                     Channels,
                     ActiveChannels>(samples,
                                     columns,
                                     rows,
                                     row_stride,
                                     rows_per_block,
                                     shared_histograms,
                                     histogram,
                                     sample_to_bin_op,
                                     bins,
                                     block_histogram);
}

template<unsigned int Channels,
         unsigned int ActiveChannels,
         class SampleIterator,
         class Counter,
         class SampleToBinOp>
struct HistogramSharedOp
{
    SampleIterator                             samples;
    unsigned int                               columns;
    unsigned int                               rows;
    unsigned int                               row_stride;
    fixed_array<Counter*, ActiveChannels>      histogram;
    fixed_array<SampleToBinOp, ActiveChannels> sample_to_bin_op;
    fixed_array<size_t, ActiveChannels>        bins;

    unsigned int rows_per_block    = 0;
    unsigned int shared_histograms = 0;

    template<class ArchConfig>
    ROCPRIM_DEVICE
    inline void
        operator()(ArchConfig) const
    {
        histogram_shared_kernel_impl<ArchConfig,
                                     Channels,
                                     ActiveChannels,
                                     SampleIterator,
                                     Counter,
                                     SampleToBinOp>(samples,
                                                    columns,
                                                    rows,
                                                    row_stride,
                                                    rows_per_block,
                                                    shared_histograms,
                                                    histogram,
                                                    sample_to_bin_op,
                                                    bins);
    }
};

template<typename Kernel>
struct histogram_launch_plan
{
    using kernel_type = void (*)(Kernel);

    kernel_type kernel;
    Kernel      device_callback;

    unsigned int shared_impl_histograms = 0;
    unsigned int max_grid_size          = 0;

    void launch(dim3 grid, dim3 block, size_t shmem, hipStream_t stream) const
    {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel), grid, block, shmem, stream, device_callback);
    }
};

template<class Config,
         class Kernel,
         template<typename, rocprim::detail::target_arch>
         class LaunchSelector>
auto make_histogram_launch_plan(rocprim::detail::target_arch arch, Kernel kernel)
    -> histogram_launch_plan<Kernel>
{
    histogram_launch_plan<Kernel> plan{nullptr, std::move(kernel), 0u, 0u};

    bool found = false;

    for_each_arch(
        [&](auto arch_tag)
        {
            constexpr auto Arch = decltype(arch_tag)::value;
            if(Arch != arch || found)
                return;

            plan.kernel = trampoline_kernel<Config, Arch, Kernel, LaunchSelector>;

            constexpr auto params = Config::template architecture_config<Arch>::params;

            plan.shared_impl_histograms = params.shared_impl_histograms;
            plan.max_grid_size          = params.max_grid_size;

            found = true;
        });
    if(!found)
    {
        constexpr auto Arch = rocprim::detail::target_arch::unknown;

        plan.kernel = trampoline_kernel<Config, Arch, Kernel, LaunchSelector>;

        constexpr auto params       = Config::template architecture_config<Arch>::params;
        plan.shared_impl_histograms = params.shared_impl_histograms;
        plan.max_grid_size          = params.max_grid_size;
    }
    return plan;
}

template<class Config,
         unsigned int Channels,
         unsigned int ActiveChannels,
         class SampleIterator,
         class Counter,
         class SampleToBinOp>
inline hipError_t
    launch_histogram_global(detail::target_arch                              arch,
                            SampleIterator                                   samples,
                            unsigned int                                     columns,
                            unsigned int                                     row_stride,
                            fixed_array<Counter*, ActiveChannels>            histogram,
                            const fixed_array<SampleToBinOp, ActiveChannels> sample_to_bin_op,
                            const fixed_array<size_t, ActiveChannels>        bins_bits,
                            dim3                                             grid,
                            dim3                                             block,
                            size_t                                           shmem,
                            hipStream_t                                      stream)
{
    auto kernel = [=](auto arch_config)
    {
        static constexpr histogram_config_params params = decltype(arch_config)::params;

        histogram_global<params.histogram_config.block_size,
                         params.histogram_config.items_per_thread,
                         Channels,
                         ActiveChannels>(samples,
                                         columns,
                                         row_stride,
                                         histogram,
                                         sample_to_bin_op,
                                         bins_bits);
    };

    return execute_launch_plan<Config, decltype(kernel), histogram_config_selector>(arch,
                                                                                    kernel,
                                                                                    grid,
                                                                                    block,
                                                                                    shmem,
                                                                                    stream);
}

template<class Config,
         unsigned int Channels,
         unsigned int ActiveChannels,
         class SampleIterator,
         class Counter,
         class SampleToBinOp>
inline hipError_t
    launch_histogram_private_global(detail::target_arch                        arch,
                                    SampleIterator                             samples,
                                    unsigned int                               columns,
                                    unsigned int                               rows,
                                    unsigned int                               row_stride,
                                    fixed_array<Counter*, ActiveChannels>      histogram,
                                    fixed_array<SampleToBinOp, ActiveChannels> sample_to_bin_op,
                                    fixed_array<size_t, ActiveChannels>        bins_bits,
                                    fixed_array<size_t, ActiveChannels>        bins,
                                    Counter*                                   private_histograms,
                                    unsigned int                               virtual_max_blocks,
                                    unsigned int*                              block_id_count,
                                    dim3                                       grid,
                                    dim3                                       block,
                                    size_t                                     shmem,
                                    hipStream_t                                stream)
{
    auto kernel = [=](auto arch_config)
    {
        static constexpr histogram_config_params params = decltype(arch_config)::params;

        histogram_private_global<params.histogram_global_config.block_size,
                                 params.histogram_global_config.items_per_thread,
                                 Channels,
                                 ActiveChannels>(samples,
                                                 columns,
                                                 rows,
                                                 row_stride,
                                                 histogram,
                                                 sample_to_bin_op,
                                                 bins_bits,
                                                 bins,
                                                 private_histograms,
                                                 virtual_max_blocks,
                                                 block_id_count);
    };

    return execute_launch_plan<Config, decltype(kernel), histogram_global_config_selector>(arch,
                                                                                           kernel,
                                                                                           grid,
                                                                                           block,
                                                                                           shmem,
                                                                                           stream);
}

template<unsigned int Channels,
         unsigned int ActiveChannels,
         class Config,
         class SampleIterator,
         class Counter,
         class SampleToBinOp>
inline hipError_t histogram_impl(void*          temporary_storage,
                                 size_t&        storage_size,
                                 SampleIterator samples,
                                 unsigned int   columns,
                                 unsigned int   rows,
                                 size_t         row_stride_bytes,
                                 Counter*       histogram[ActiveChannels],
                                 unsigned int   levels[ActiveChannels],
                                 SampleToBinOp  sample_to_bin_op[ActiveChannels],
                                 hipStream_t    stream,
                                 bool           debug_synchronous)
{
    using sample_type = typename std::iterator_traits<SampleIterator>::value_type;

    using config = wrapped_histogram_config<Config, sample_type, Channels, ActiveChannels>;

    detail::target_arch target_arch;
    hipError_t          result = host_target_arch(stream, target_arch);
    if(result != hipSuccess)
    {
        return result;
    }
    const histogram_config_params params = dispatch_target_arch<config, false>(target_arch);

    const unsigned int block_size           = params.histogram_config.block_size;
    const unsigned int items_per_thread     = params.histogram_config.items_per_thread;
    const unsigned int shared_impl_max_bins = params.shared_impl_max_bins;
    const auto         items_per_block      = block_size * items_per_thread;

    if(row_stride_bytes % sizeof(sample_type) != 0)
    {
        // Row stride must be a whole multiple of the sample data type size
        return hipErrorInvalidValue;
    }

    const unsigned int blocks_x   = ::rocprim::detail::ceiling_div(columns, items_per_block);
    const unsigned int row_stride = row_stride_bytes / sizeof(sample_type);

    size_t bins[ActiveChannels];
    size_t bins_bits[ActiveChannels];
    size_t total_shared_bins = 0;
    size_t max_bins          = 0;
    size_t total_bins        = 0;
    for(unsigned int channel = 0; channel < ActiveChannels; channel++)
    {
        bins[channel] = levels[channel] - 1;
        bins_bits[channel]
            = static_cast<unsigned int>(std::log2(detail::next_power_of_two(bins[channel])));
        const size_t size = bins[channel];
        // Prevent LDS bank conflicts
        total_shared_bins += rocprim::detail::is_power_of_two(size) ? size + 1 : size;
        total_bins += size;
        max_bins = std::max(max_bins, bins[channel]);
    }

    const bool use_shared_mem        = total_shared_bins <= shared_impl_max_bins;
    const bool use_private_histogram = target_arch == target_arch::gfx942;

    Counter*      private_histograms         = nullptr;
    unsigned int* block_id_count             = nullptr;
    int           global_histogram_grid_size = 0;
    unsigned int  virtual_max_blocks         = 0;

    if(use_shared_mem || !use_private_histogram)
    {
        if(temporary_storage == nullptr)
        {
            // Make sure user won't try to allocate 0 bytes memory, because
            // hipMalloc will return nullptr.
            storage_size = 4;
            return hipSuccess;
        }
    }
    else
    {
        const auto items_per_block = params.histogram_global_config.block_size
                                     * params.histogram_global_config.items_per_thread;

        int device_id = hipGetStreamDeviceId(stream);

        // Get the number of multiprocessors
        int num_multi_processors{};
        ROCPRIM_RETURN_ON_ERROR(
            hipDeviceGetAttribute(&num_multi_processors,
                                  hipDeviceAttribute_t::hipDeviceAttributeMultiprocessorCount,
                                  device_id));

        global_histogram_grid_size = num_multi_processors;

        virtual_max_blocks = ::rocprim::detail::ceiling_div(columns, items_per_block) * rows;
        global_histogram_grid_size
            = rocprim::min(static_cast<unsigned int>(global_histogram_grid_size),
                           virtual_max_blocks);

        const size_t     size_private_histograms = total_bins * global_histogram_grid_size;
        const hipError_t partition_result        = detail::temp_storage::partition(
            temporary_storage,
            storage_size,
            detail::temp_storage::make_linear_partition(
                detail::temp_storage::ptr_aligned_array(&private_histograms,
                                                        size_private_histograms),
                detail::temp_storage::ptr_aligned_array(&block_id_count, 1)));
        if(partition_result != hipSuccess || temporary_storage == nullptr)
        {
            return partition_result;
        }

        // It will not run the kernel if columns or rows are 0.
        if(global_histogram_grid_size > 0)
        {
            ROCPRIM_RETURN_ON_ERROR(hipMemsetAsync(private_histograms,
                                                   0,
                                                   size_private_histograms * sizeof(Counter),
                                                   stream));

            ROCPRIM_RETURN_ON_ERROR(hipMemcpyAsync(block_id_count,
                                                   &global_histogram_grid_size,
                                                   sizeof(unsigned int),
                                                   hipMemcpyHostToDevice,
                                                   stream));
        }
    }

    if(debug_synchronous)
    {
        std::cout << "columns " << columns << '\n';
        std::cout << "rows " << rows << '\n';
        std::cout << "blocks_x " << blocks_x << '\n';
        ROCPRIM_RETURN_ON_ERROR(hipStreamSynchronize(stream));
    }

    std::chrono::steady_clock::time_point start;

    if(debug_synchronous)
    {
        start = std::chrono::steady_clock::now();
    }
    ROCPRIM_RETURN_ON_ERROR(launch_init_histogram<config, ActiveChannels>(
        target_arch,
        fixed_array<Counter*, ActiveChannels>(histogram),
        fixed_array<size_t, ActiveChannels>(bins),
        dim3(::rocprim::detail::ceiling_div(max_bins, block_size)),
        dim3(block_size),
        0,
        stream));

    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("init_histogram", max_bins, start);

    if(columns == 0 || rows == 0)
    {
        return hipSuccess;
    }

    if(use_shared_mem)
    {
        if(debug_synchronous)
        {
            start = std::chrono::steady_clock::now();
        }

        HistogramSharedOp<Channels, ActiveChannels, SampleIterator, Counter, SampleToBinOp> op{
            samples,
            columns,
            rows,
            row_stride,
            fixed_array<Counter*, ActiveChannels>(histogram),
            fixed_array<SampleToBinOp, ActiveChannels>(sample_to_bin_op),
            fixed_array<size_t, ActiveChannels>(bins),
            0,
            0};

        auto plan = make_histogram_launch_plan<config, decltype(op), histogram_config_selector>(
            target_arch,
            op);

        const size_t block_histogram_bytes = total_shared_bins * sizeof(unsigned int);

        // Use up to shared_impl_histograms histograms in shared memory to reduce atomic conflicts
        // for the case of samples concentrated in one bin
        // Limit the number of shared histograms if occupancy drops due to high dynamic shared
        // memory usage
        unsigned int chosen_shared_histograms = 0;
        int          max_blocks_per_mp        = 0;
        for(unsigned int n = plan.shared_impl_histograms; n >= 1; n--)
        {
            int blocks_per_mp;
            ROCPRIM_RETURN_ON_ERROR(hipOccupancyMaxActiveBlocksPerMultiprocessor(
                &blocks_per_mp,
                reinterpret_cast<const void*>(plan.kernel),
                block_size,
                n * block_histogram_bytes));

            if(blocks_per_mp > max_blocks_per_mp)
            {
                chosen_shared_histograms = n;
                max_blocks_per_mp        = blocks_per_mp;
            }
        }

        // Choose minimum grid size needed to achieve the best occupancy
        int min_grid_size, max_block_size;
        ROCPRIM_RETURN_ON_ERROR(
            hipOccupancyMaxPotentialBlockSize(&min_grid_size,
                                              &max_block_size,
                                              reinterpret_cast<const void*>(plan.kernel),
                                              chosen_shared_histograms * block_histogram_bytes,
                                              int(block_size)));

        const unsigned int chosen_grid_size
            = std::min(static_cast<unsigned int>(min_grid_size), params.max_grid_size);

        dim3 grid_size;
        grid_size.x = std::min(chosen_grid_size, blocks_x);
        grid_size.y = std::min(rows, ::rocprim::detail::ceiling_div(chosen_grid_size, grid_size.x));
        const unsigned int rows_per_block = ::rocprim::detail::ceiling_div(rows, grid_size.y);

        plan.device_callback.shared_histograms = chosen_shared_histograms;
        plan.device_callback.rows_per_block    = rows_per_block;

        // 4) 发射（无需再做任何按架构分发）
        plan.launch(grid_size,
                    dim3(block_size, 1),
                    chosen_shared_histograms * block_histogram_bytes,
                    stream);

        ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("histogram_shared",
                                                    grid_size.x * grid_size.y * block_size,
                                                    start);
    }
    else
    {

        if(use_private_histogram)
        {

            if(debug_synchronous)
            {
                start = std::chrono::steady_clock::now();
            }
            ROCPRIM_RETURN_ON_ERROR(
                launch_histogram_private_global<config, Channels, ActiveChannels>(
                    target_arch,
                    samples,
                    columns,
                    rows,
                    row_stride,
                    fixed_array<Counter*, ActiveChannels>(histogram),
                    fixed_array<SampleToBinOp, ActiveChannels>(sample_to_bin_op),
                    fixed_array<size_t, ActiveChannels>(bins_bits),
                    fixed_array<size_t, ActiveChannels>(bins),
                    private_histograms,
                    virtual_max_blocks,
                    block_id_count,
                    dim3(global_histogram_grid_size),
                    dim3(params.histogram_global_config.block_size),
                    0,
                    stream));
            ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("histogram_private_global",
                                                        blocks_x * block_size * rows,
                                                        start);
        }
        else
        {
            if(debug_synchronous)
            {
                start = std::chrono::steady_clock::now();
            }
            ROCPRIM_RETURN_ON_ERROR(launch_histogram_global<config, Channels, ActiveChannels>(
                target_arch,
                samples,
                columns,
                row_stride,
                fixed_array<Counter*, ActiveChannels>(histogram),
                fixed_array<SampleToBinOp, ActiveChannels>(sample_to_bin_op),
                fixed_array<size_t, ActiveChannels>(bins_bits),
                dim3(blocks_x, rows),
                dim3(block_size, 1),
                0,
                stream));
            ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("histogram_global",
                                                        blocks_x * block_size * rows,
                                                        start);
        }
    }

    return hipSuccess;
}

template<unsigned int Channels,
         unsigned int ActiveChannels,
         class Config,
         class SampleIterator,
         class Counter,
         class Level>
inline hipError_t histogram_even_impl(void*          temporary_storage,
                                      size_t&        storage_size,
                                      SampleIterator samples,
                                      unsigned int   columns,
                                      unsigned int   rows,
                                      size_t         row_stride_bytes,
                                      Counter*       histogram[ActiveChannels],
                                      unsigned int   levels[ActiveChannels],
                                      Level          lower_level[ActiveChannels],
                                      Level          upper_level[ActiveChannels],
                                      hipStream_t    stream,
                                      bool           debug_synchronous)
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
        sample_to_bin_op[channel] = sample_to_bin_even<Level>(levels[channel] - 1,
                                                              lower_level[channel],
                                                              upper_level[channel]);
    }

    return histogram_impl<Channels, ActiveChannels, Config>(temporary_storage,
                                                            storage_size,
                                                            samples,
                                                            columns,
                                                            rows,
                                                            row_stride_bytes,
                                                            histogram,
                                                            levels,
                                                            sample_to_bin_op,
                                                            stream,
                                                            debug_synchronous);
}

template<unsigned int Channels,
         unsigned int ActiveChannels,
         class Config,
         class SampleIterator,
         class Counter,
         class Level>
inline hipError_t histogram_range_impl(void*          temporary_storage,
                                       size_t&        storage_size,
                                       SampleIterator samples,
                                       unsigned int   columns,
                                       unsigned int   rows,
                                       size_t         row_stride_bytes,
                                       Counter*       histogram[ActiveChannels],
                                       unsigned int   levels[ActiveChannels],
                                       Level*         level_values[ActiveChannels],
                                       hipStream_t    stream,
                                       bool           debug_synchronous)
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
        sample_to_bin_op[channel]
            = sample_to_bin_range<Level>(levels[channel] - 1, level_values[channel]);
    }

    return histogram_impl<Channels, ActiveChannels, Config>(temporary_storage,
                                                            storage_size,
                                                            samples,
                                                            columns,
                                                            rows,
                                                            row_stride_bytes,
                                                            histogram,
                                                            levels,
                                                            sample_to_bin_op,
                                                            stream,
                                                            debug_synchronous);
}

} // namespace detail

/// \brief Computes a histogram from a sequence of samples using equal-width bins.
///
/// \par
/// * The number of histogram bins is (\p levels - 1).
/// * Bins are evenly-segmented and include the same width of sample values:
/// (\p upper_level - \p lower_level) / (\p levels - 1).
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage in a null pointer.
///
/// \tparam Config [optional] Configuration of the primitive, must be `default_config` or `kernel_config`.
/// \tparam SampleIterator random-access iterator type of the input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam Counter integer type for histogram bin counters.
/// \tparam Level type of histogram boundaries (levels)
///
/// \param [in] temporary_storage pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the reduction operation.
/// \param [in,out] storage_size reference to a size (in bytes) of \p temporary_storage.
/// \param [in] samples iterator to the first element in the range of input samples.
/// \param [in] size number of elements in the samples range.
/// \param [out] histogram pointer to the first element in the histogram range.
/// \param [in] levels number of boundaries (levels) for histogram bins.
/// \param [in] lower_level lower sample value bound (inclusive) for the first histogram bin.
/// \param [in] upper_level upper sample value bound (exclusive) for the last histogram bin.
/// \param [in] stream [optional] HIP stream object. Default is \p 0 (default stream).
/// \param [in] debug_synchronous [optional] If true, synchronization after every kernel
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
template<class Config = default_config, class SampleIterator, class Counter, class Level>
inline hipError_t histogram_even(void*          temporary_storage,
                                 size_t&        storage_size,
                                 SampleIterator samples,
                                 unsigned int   size,
                                 Counter*       histogram,
                                 unsigned int   levels,
                                 Level          lower_level,
                                 Level          upper_level,
                                 hipStream_t    stream            = 0,
                                 bool           debug_synchronous = false)
{
    Counter*     histogram_single[1]   = {histogram};
    unsigned int levels_single[1]      = {levels};
    Level        lower_level_single[1] = {lower_level};
    Level        upper_level_single[1] = {upper_level};

    return detail::histogram_even_impl<1, 1, Config>(temporary_storage,
                                                     storage_size,
                                                     samples,
                                                     size,
                                                     1,
                                                     0,
                                                     histogram_single,
                                                     levels_single,
                                                     lower_level_single,
                                                     upper_level_single,
                                                     stream,
                                                     debug_synchronous);
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
/// \tparam Config [optional] Configuration of the primitive, must be `default_config` or `kernel_config`.
/// \tparam SampleIterator random-access iterator type of the input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam Counter integer type for histogram bin counters.
/// \tparam Level type of histogram boundaries (levels)
///
/// \param [in] temporary_storage pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the reduction operation.
/// \param [in,out] storage_size reference to a size (in bytes) of \p temporary_storage.
/// \param [in] samples iterator to the first element in the range of input samples.
/// \param [in] columns number of elements in each row of the region.
/// \param [in] rows number of rows of the region.
/// \param [in] row_stride_bytes number of bytes between starts of consecutive rows of the region.
/// \param [out] histogram pointer to the first element in the histogram range.
/// \param [in] levels number of boundaries (levels) for histogram bins.
/// \param [in] lower_level lower sample value bound (inclusive) for the first histogram bin.
/// \param [in] upper_level upper sample value bound (exclusive) for the last histogram bin.
/// \param [in] stream [optional] HIP stream object. Default is \p 0 (default stream).
/// \param [in] debug_synchronous [optional] If true, synchronization after every kernel
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
template<class Config = default_config, class SampleIterator, class Counter, class Level>
inline hipError_t histogram_even(void*          temporary_storage,
                                 size_t&        storage_size,
                                 SampleIterator samples,
                                 unsigned int   columns,
                                 unsigned int   rows,
                                 size_t         row_stride_bytes,
                                 Counter*       histogram,
                                 unsigned int   levels,
                                 Level          lower_level,
                                 Level          upper_level,
                                 hipStream_t    stream            = 0,
                                 bool           debug_synchronous = false)
{
    Counter*     histogram_single[1]   = {histogram};
    unsigned int levels_single[1]      = {levels};
    Level        lower_level_single[1] = {lower_level};
    Level        upper_level_single[1] = {upper_level};

    return detail::histogram_even_impl<1, 1, Config>(temporary_storage,
                                                     storage_size,
                                                     samples,
                                                     columns,
                                                     rows,
                                                     row_stride_bytes,
                                                     histogram_single,
                                                     levels_single,
                                                     lower_level_single,
                                                     upper_level_single,
                                                     stream,
                                                     debug_synchronous);
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
/// \tparam Channels number of channels interleaved in the input samples.
/// \tparam ActiveChannels number of channels being used for computing histograms.
/// \tparam Config [optional] Configuration of the primitive, must be `default_config` or `kernel_config`.
/// \tparam SampleIterator random-access iterator type of the input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam Counter integer type for histogram bin counters.
/// \tparam Level type of histogram boundaries (levels)
///
/// \param [in] temporary_storage pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the reduction operation.
/// \param [in,out] storage_size reference to a size (in bytes) of \p temporary_storage.
/// \param [in] samples iterator to the first element in the range of input samples.
/// \param [in] size number of pixels in the samples range.
/// \param [out] histogram pointers to the first element in the histogram range, one for each active channel.
/// \param [in] levels number of boundaries (levels) for histogram bins in each active channel.
/// \param [in] lower_level lower sample value bound (inclusive) for the first histogram bin in each active channel.
/// \param [in] upper_level upper sample value bound (exclusive) for the last histogram bin in each active channel.
/// \param [in] stream [optional] HIP stream object. Default is \p 0 (default stream).
/// \param [in] debug_synchronous [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. Default value is \p false.
///
/// \returns \p hipSuccess (\p 0) after successful histogram operation; otherwise a HIP runtime error of
/// type \p hipError_t.
///
/// \par Notes
/// * Currently the \p Channels template parameter has no strict restriction on its value. However,
///   internally a vector type of elements of type \p SampleIterator and length \p Channels is used
///   to represent the input items, so the amount of local memory available will limit the range of
///   possible values for this template parameter.
/// * \p ActiveChannels must be less or equal than \p Channels.
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
template<unsigned int Channels,
         unsigned int ActiveChannels,
         class Config = default_config,
         class SampleIterator,
         class Counter,
         class Level>
inline hipError_t multi_histogram_even(void*          temporary_storage,
                                       size_t&        storage_size,
                                       SampleIterator samples,
                                       unsigned int   size,
                                       Counter*       histogram[ActiveChannels],
                                       unsigned int   levels[ActiveChannels],
                                       Level          lower_level[ActiveChannels],
                                       Level          upper_level[ActiveChannels],
                                       hipStream_t    stream            = 0,
                                       bool           debug_synchronous = false)
{
    return detail::histogram_even_impl<Channels, ActiveChannels, Config>(temporary_storage,
                                                                         storage_size,
                                                                         samples,
                                                                         size,
                                                                         1,
                                                                         0,
                                                                         histogram,
                                                                         levels,
                                                                         lower_level,
                                                                         upper_level,
                                                                         stream,
                                                                         debug_synchronous);
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
/// \tparam Channels number of channels interleaved in the input samples.
/// \tparam ActiveChannels number of channels being used for computing histograms.
/// \tparam Config [optional] Configuration of the primitive, must be `default_config` or `kernel_config`.
/// \tparam SampleIterator random-access iterator type of the input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam Counter integer type for histogram bin counters.
/// \tparam Level type of histogram boundaries (levels)
///
/// \param [in] temporary_storage pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the reduction operation.
/// \param [in,out] storage_size reference to a size (in bytes) of \p temporary_storage.
/// \param [in] samples iterator to the first element in the range of input samples.
/// \param [in] columns number of elements in each row of the region.
/// \param [in] rows number of rows of the region.
/// \param [in] row_stride_bytes number of bytes between starts of consecutive rows of the region.
/// \param [out] histogram pointers to the first element in the histogram range, one for each active channel.
/// \param [in] levels number of boundaries (levels) for histogram bins in each active channel.
/// \param [in] lower_level lower sample value bound (inclusive) for the first histogram bin in each active channel.
/// \param [in] upper_level upper sample value bound (exclusive) for the last histogram bin in each active channel.
/// \param [in] stream [optional] HIP stream object. Default is \p 0 (default stream).
/// \param [in] debug_synchronous [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. Default value is \p false.
///
/// \returns \p hipSuccess (\p 0) after successful histogram operation; otherwise a HIP runtime error of
/// type \p hipError_t.
///
/// \par Notes
/// * Currently the \p Channels template parameter has no strict restriction on its value. However,
///   internally a vector type of elements of type \p SampleIterator and length \p Channels is used
///   to represent the input items, so the amount of local memory available will limit the range of
///   possible values for this template parameter.
/// * \p ActiveChannels must be less or equal than \p Channels.
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
template<unsigned int Channels,
         unsigned int ActiveChannels,
         class Config = default_config,
         class SampleIterator,
         class Counter,
         class Level>
inline hipError_t multi_histogram_even(void*          temporary_storage,
                                       size_t&        storage_size,
                                       SampleIterator samples,
                                       unsigned int   columns,
                                       unsigned int   rows,
                                       size_t         row_stride_bytes,
                                       Counter*       histogram[ActiveChannels],
                                       unsigned int   levels[ActiveChannels],
                                       Level          lower_level[ActiveChannels],
                                       Level          upper_level[ActiveChannels],
                                       hipStream_t    stream            = 0,
                                       bool           debug_synchronous = false)
{
    return detail::histogram_even_impl<Channels, ActiveChannels, Config>(temporary_storage,
                                                                         storage_size,
                                                                         samples,
                                                                         columns,
                                                                         rows,
                                                                         row_stride_bytes,
                                                                         histogram,
                                                                         levels,
                                                                         lower_level,
                                                                         upper_level,
                                                                         stream,
                                                                         debug_synchronous);
}

/// \brief Computes a histogram from a sequence of samples using the specified bin boundary levels.
///
/// \par
/// * The number of histogram bins is (\p levels - 1).
/// * The range for bin<sub><em>j</em></sub> is [<tt>level_values[j]</tt>, <tt>level_values[j+1]</tt>).
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage in a null pointer.
///
/// \tparam Config [optional] Configuration of the primitive, must be `default_config` or `kernel_config`.
/// \tparam SampleIterator random-access iterator type of the input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam Counter integer type for histogram bin counters.
/// \tparam Level type of histogram boundaries (levels)
///
/// \param [in] temporary_storage pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the reduction operation.
/// \param [in,out] storage_size reference to a size (in bytes) of \p temporary_storage.
/// \param [in] samples iterator to the first element in the range of input samples.
/// \param [in] size number of elements in the samples range.
/// \param [out] histogram pointer to the first element in the histogram range.
/// \param [in] levels number of boundaries (levels) for histogram bins.
/// \param [in] level_values pointer to the array of bin boundaries.
/// \param [in] stream [optional] HIP stream object. Default is \p 0 (default stream).
/// \param [in] debug_synchronous [optional] If true, synchronization after every kernel
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
template<class Config = default_config, class SampleIterator, class Counter, class Level>
inline hipError_t histogram_range(void*          temporary_storage,
                                  size_t&        storage_size,
                                  SampleIterator samples,
                                  unsigned int   size,
                                  Counter*       histogram,
                                  unsigned int   levels,
                                  Level*         level_values,
                                  hipStream_t    stream            = 0,
                                  bool           debug_synchronous = false)
{
    Counter*     histogram_single[1]    = {histogram};
    unsigned int levels_single[1]       = {levels};
    Level*       level_values_single[1] = {level_values};

    return detail::histogram_range_impl<1, 1, Config>(temporary_storage,
                                                      storage_size,
                                                      samples,
                                                      size,
                                                      1,
                                                      0,
                                                      histogram_single,
                                                      levels_single,
                                                      level_values_single,
                                                      stream,
                                                      debug_synchronous);
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
/// \tparam Config [optional] Configuration of the primitive, must be `default_config` or `kernel_config`.
/// \tparam SampleIterator random-access iterator type of the input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam Counter integer type for histogram bin counters.
/// \tparam Level type of histogram boundaries (levels)
///
/// \param [in] temporary_storage pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the reduction operation.
/// \param [in,out] storage_size reference to a size (in bytes) of \p temporary_storage.
/// \param [in] samples iterator to the first element in the range of input samples.
/// \param [in] columns number of elements in each row of the region.
/// \param [in] rows number of rows of the region.
/// \param [in] row_stride_bytes number of bytes between starts of consecutive rows of the region.
/// \param [out] histogram pointer to the first element in the histogram range.
/// \param [in] levels number of boundaries (levels) for histogram bins.
/// \param [in] level_values pointer to the array of bin boundaries.
/// \param [in] stream [optional] HIP stream object. Default is \p 0 (default stream).
/// \param [in] debug_synchronous [optional] If true, synchronization after every kernel
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
template<class Config = default_config, class SampleIterator, class Counter, class Level>
inline hipError_t histogram_range(void*          temporary_storage,
                                  size_t&        storage_size,
                                  SampleIterator samples,
                                  unsigned int   columns,
                                  unsigned int   rows,
                                  size_t         row_stride_bytes,
                                  Counter*       histogram,
                                  unsigned int   levels,
                                  Level*         level_values,
                                  hipStream_t    stream            = 0,
                                  bool           debug_synchronous = false)
{
    Counter*     histogram_single[1]    = {histogram};
    unsigned int levels_single[1]       = {levels};
    Level*       level_values_single[1] = {level_values};

    return detail::histogram_range_impl<1, 1, Config>(temporary_storage,
                                                      storage_size,
                                                      samples,
                                                      columns,
                                                      rows,
                                                      row_stride_bytes,
                                                      histogram_single,
                                                      levels_single,
                                                      level_values_single,
                                                      stream,
                                                      debug_synchronous);
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
/// \tparam Channels number of channels interleaved in the input samples.
/// \tparam ActiveChannels number of channels being used for computing histograms.
/// \tparam Config [optional] Configuration of the primitive, must be `default_config` or `kernel_config`.
/// \tparam SampleIterator random-access iterator type of the input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam Counter integer type for histogram bin counters.
/// \tparam Level type of histogram boundaries (levels)
///
/// \param [in] temporary_storage pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the reduction operation.
/// \param [in,out] storage_size reference to a size (in bytes) of \p temporary_storage.
/// \param [in] samples iterator to the first element in the range of input samples.
/// \param [in] size number of pixels in the samples range.
/// \param [out] histogram pointers to the first element in the histogram range, one for each active channel.
/// \param [in] levels number of boundaries (levels) for histogram bins in each active channel.
/// \param [in] level_values pointer to the array of bin boundaries for each active channel.
/// \param [in] stream [optional] HIP stream object. Default is \p 0 (default stream).
/// \param [in] debug_synchronous [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. Default value is \p false.
///
/// \returns \p hipSuccess (\p 0) after successful histogram operation; otherwise a HIP runtime error of
/// type \p hipError_t.
///
/// \par Notes
/// * Currently the \p Channels template parameter has no strict restriction on its value. However,
///   internally a vector type of elements of type \p SampleIterator and length \p Channels is used
///   to represent the input items, so the amount of local memory available will limit the range of
///   possible values for this template parameter.
/// * \p ActiveChannels must be less or equal than \p Channels.
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
template<unsigned int Channels,
         unsigned int ActiveChannels,
         class Config = default_config,
         class SampleIterator,
         class Counter,
         class Level>
inline hipError_t multi_histogram_range(void*          temporary_storage,
                                        size_t&        storage_size,
                                        SampleIterator samples,
                                        unsigned int   size,
                                        Counter*       histogram[ActiveChannels],
                                        unsigned int   levels[ActiveChannels],
                                        Level*         level_values[ActiveChannels],
                                        hipStream_t    stream            = 0,
                                        bool           debug_synchronous = false)
{
    return detail::histogram_range_impl<Channels, ActiveChannels, Config>(temporary_storage,
                                                                          storage_size,
                                                                          samples,
                                                                          size,
                                                                          1,
                                                                          0,
                                                                          histogram,
                                                                          levels,
                                                                          level_values,
                                                                          stream,
                                                                          debug_synchronous);
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
/// \tparam Channels number of channels interleaved in the input samples.
/// \tparam ActiveChannels number of channels being used for computing histograms.
/// \tparam Config [optional] Configuration of the primitive, must be `default_config` or `kernel_config`.
/// \tparam SampleIterator random-access iterator type of the input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam Counter integer type for histogram bin counters.
/// \tparam Level type of histogram boundaries (levels)
///
/// \param [in] temporary_storage pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the reduction operation.
/// \param [in,out] storage_size reference to a size (in bytes) of \p temporary_storage.
/// \param [in] samples iterator to the first element in the range of input samples.
/// \param [in] columns number of elements in each row of the region.
/// \param [in] rows number of rows of the region.
/// \param [in] row_stride_bytes number of bytes between starts of consecutive rows of the region.
/// \param [out] histogram pointers to the first element in the histogram range, one for each active channel.
/// \param [in] levels number of boundaries (levels) for histogram bins in each active channel.
/// \param [in] level_values pointer to the array of bin boundaries for each active channel.
/// \param [in] stream [optional] HIP stream object. Default is \p 0 (default stream).
/// \param [in] debug_synchronous [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. Default value is \p false.
///
/// \returns \p hipSuccess (\p 0) after successful histogram operation; otherwise a HIP runtime error of
/// type \p hipError_t.
///
/// \par Notes
/// * Currently the \p Channels template parameter has no strict restriction on its value. However,
///   internally a vector type of elements of type \p SampleIterator and length \p Channels is used
///   to represent the input items, so the amount of local memory available will limit the range of
///   possible values for this template parameter.
/// * \p ActiveChannels must be less or equal than \p Channels.
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
template<unsigned int Channels,
         unsigned int ActiveChannels,
         class Config = default_config,
         class SampleIterator,
         class Counter,
         class Level>
inline hipError_t multi_histogram_range(void*          temporary_storage,
                                        size_t&        storage_size,
                                        SampleIterator samples,
                                        unsigned int   columns,
                                        unsigned int   rows,
                                        size_t         row_stride_bytes,
                                        Counter*       histogram[ActiveChannels],
                                        unsigned int   levels[ActiveChannels],
                                        Level*         level_values[ActiveChannels],
                                        hipStream_t    stream            = 0,
                                        bool           debug_synchronous = false)
{
    return detail::histogram_range_impl<Channels, ActiveChannels, Config>(temporary_storage,
                                                                          storage_size,
                                                                          samples,
                                                                          columns,
                                                                          rows,
                                                                          row_stride_bytes,
                                                                          histogram,
                                                                          levels,
                                                                          level_values,
                                                                          stream,
                                                                          debug_synchronous);
}

/// @}
// end of group devicemodule

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DEVICE_HISTOGRAM_HPP_
