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

#ifndef ROCPRIM_DEVICE_DETAIL_DEVICE_HISTOGRAM_HPP_
#define ROCPRIM_DEVICE_DETAIL_DEVICE_HISTOGRAM_HPP_

#include <cmath>
#include <iterator>
#include <type_traits>

#include "../../config.hpp"
#include "../../detail/various.hpp"
#include "../../functional.hpp"
#include "../../intrinsics.hpp"
#include "../../type_traits.hpp"

#include "../../block/block_load.hpp"

#include "uint_fast_div.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

// Special wrapper for passing fixed-length arrays (i.e. T values[Size]) into kernels
template<class T, size_t Size>
class fixed_array
{
private:
    T values[Size];

public:
    ROCPRIM_HOST_DEVICE fixed_array(const T values[Size])
    {
        for(unsigned int i = 0; i < Size; i++)
        {
            this->values[i] = values[i];
        }
    }

    ROCPRIM_HOST_DEVICE
    T& operator[](size_t index)
    {
        return values[index];
    }

    ROCPRIM_HOST_DEVICE
    const T&
        operator[](size_t index) const
    {
        return values[index];
    }
};

template<class Level, class Enable = void>
struct sample_to_bin_even
{
    size_t bins;
    Level  lower_level;
    Level  upper_level;
    Level  scale;

    ROCPRIM_HOST_DEVICE ROCPRIM_INLINE sample_to_bin_even() = default;

    ROCPRIM_HOST_DEVICE ROCPRIM_INLINE sample_to_bin_even(size_t bins,
                                                         Level  lower_level,
                                                         Level  upper_level)
        : bins(bins)
        , lower_level(lower_level)
        , upper_level(upper_level)
        , scale((upper_level - lower_level) / bins)
    {}

    template<class Sample>
    ROCPRIM_HOST_DEVICE ROCPRIM_INLINE
    bool operator()(Sample sample, size_t& bin) const
    {
        const Level s = static_cast<Level>(sample);
        if(s >= lower_level && s < upper_level)
        {
            bin = static_cast<size_t>((s - lower_level) / scale);
            return true;
        }
        return false;
    }
};

// This specialization uses fast division (uint_fast_div) for integers smaller than 64 bit
template<class Level>
struct sample_to_bin_even<
    Level,
    typename std::enable_if<std::is_integral<Level>::value && (sizeof(Level) <= 4)>::type>
{
    size_t        bins;
    Level         lower_level;
    Level         upper_level;
    uint_fast_div scale;

    ROCPRIM_HOST_DEVICE ROCPRIM_INLINE sample_to_bin_even() = default;

    ROCPRIM_HOST_DEVICE ROCPRIM_INLINE sample_to_bin_even(size_t bins,
                                                         Level  lower_level,
                                                         Level  upper_level)
        : bins(bins)
        , lower_level(lower_level)
        , upper_level(upper_level)
        , scale((upper_level - lower_level) / bins)
    {}

    template<class Sample>
    ROCPRIM_HOST_DEVICE
    ROCPRIM_INLINE
    bool operator()(Sample sample, size_t& bin) const
    {
        const Level s = static_cast<Level>(sample);
        if(s >= lower_level && s < upper_level)
        {
            bin = static_cast<size_t>(s - lower_level) / scale;
            return true;
        }
        return false;
    }
};

// This specialization uses multiplication by inv divisor for floats
template<class Level>
struct sample_to_bin_even<Level,
                          typename std::enable_if<rocprim::is_floating_point<Level>::value>::type>
{
    size_t bins;
    Level  lower_level;
    Level  upper_level;
    Level  inv_scale;

    ROCPRIM_HOST_DEVICE ROCPRIM_INLINE sample_to_bin_even() = default;

    ROCPRIM_HOST_DEVICE ROCPRIM_INLINE sample_to_bin_even(size_t bins,
                                                         Level  lower_level,
                                                         Level  upper_level)
        : bins(bins)
        , lower_level(lower_level)
        , upper_level(upper_level)
        , inv_scale(static_cast<Level>(bins) / (upper_level - lower_level))
    {}

    template<class Sample>
    ROCPRIM_HOST_DEVICE
    ROCPRIM_INLINE
    bool operator()(Sample sample, size_t& bin) const
    {
        const Level s = static_cast<Level>(sample);
        if(s >= lower_level && s < upper_level)
        {
            bin = static_cast<size_t>((s - lower_level) * inv_scale);
            return true;
        }
        return false;
    }
};

// Returns index of the first element in values that is greater than value, or count if no such element is found.
template<class T>
ROCPRIM_HOST_DEVICE ROCPRIM_INLINE
unsigned int upper_bound(const T* values, unsigned int count, T value)
{
    unsigned int current = 0;
    while(count > 0)
    {
        const unsigned int step = count / 2;
        const unsigned int next = current + step;
        if(value < values[next])
        {
            count = step;
        }
        else
        {
            current = next + 1;
            count -= step + 1;
        }
    }
    return current;
}

template<class Level>
struct sample_to_bin_range
{
    size_t bins;
    const Level* level_values;

    ROCPRIM_HOST_DEVICE ROCPRIM_INLINE sample_to_bin_range() = default;

    ROCPRIM_HOST_DEVICE ROCPRIM_INLINE sample_to_bin_range(size_t bins, const Level* level_values)
        : bins(bins), level_values(level_values)
    {}

    template<class Sample>
    ROCPRIM_HOST_DEVICE
    ROCPRIM_INLINE
    bool operator()(Sample sample, size_t& bin) const
    {
        const Level s = static_cast<Level>(sample);
        bin           = upper_bound(level_values, bins + 1, s) - 1;
        return bin < bins;
    }
};

template<class T, unsigned int Size>
struct sample_vector
{
    T values[Size];
};

// Checks if it is possible to load 2 or 4 sample_vector<Sample, Channels> as one 32-bit value
template<unsigned int ItemsPerThread, unsigned int Channels, class Sample>
struct is_sample_vectorizable
    : std::integral_constant<bool,
                             ((sizeof(Sample) * Channels == 1) || (sizeof(Sample) * Channels == 2))
                                 && (sizeof(Sample) * Channels * ItemsPerThread % sizeof(int) == 0)
                                 && (sizeof(Sample) * Channels * ItemsPerThread / sizeof(int) > 0)>
{};

template<unsigned int BlockSize, unsigned int ItemsPerThread, unsigned int Channels, class Sample>
ROCPRIM_DEVICE ROCPRIM_INLINE
typename std::enable_if<is_sample_vectorizable<ItemsPerThread, Channels, Sample>::value>::type
    load_samples(unsigned int flat_id,
                 Sample*      samples,
                 sample_vector<Sample, Channels> (&values)[ItemsPerThread])
{
    using packed_samples_type = int[sizeof(Sample) * Channels * ItemsPerThread / sizeof(int)];

    if(reinterpret_cast<uintptr_t>(samples) % sizeof(int) == 0)
    {
        // the pointer is aligned by 4 bytes
        block_load_direct_striped<BlockSize>(flat_id,
                                             reinterpret_cast<const int*>(samples),
                                             reinterpret_cast<packed_samples_type&>(values));
    }
    else
    {
        block_load_direct_striped<BlockSize>(
            flat_id,
            reinterpret_cast<const sample_vector<Sample, Channels>*>(samples),
            values);
    }
}

template<unsigned int BlockSize, unsigned int ItemsPerThread, unsigned int Channels, class Sample>
ROCPRIM_DEVICE ROCPRIM_INLINE
    typename std::enable_if<!is_sample_vectorizable<ItemsPerThread, Channels, Sample>::value>::type
    load_samples(unsigned int flat_id,
                 Sample*      samples,
                 sample_vector<Sample, Channels> (&values)[ItemsPerThread])
{
    block_load_direct_striped<BlockSize>(
        flat_id,
        reinterpret_cast<const sample_vector<Sample, Channels>*>(samples),
        values);
}

template<unsigned int BlockSize,
         unsigned int ItemsPerThread,
         unsigned int Channels,
         class Sample,
         class SampleIterator>
ROCPRIM_DEVICE ROCPRIM_INLINE void
    load_samples(unsigned int   flat_id,
                 SampleIterator samples,
                 sample_vector<Sample, Channels> (&values)[ItemsPerThread])
{
    Sample tmp[Channels * ItemsPerThread];
    block_load_direct_blocked(flat_id, samples, tmp);
    for(unsigned int i = 0; i < ItemsPerThread; i++)
    {
        for(unsigned int channel = 0; channel < Channels; channel++)
        {
            values[i].values[channel] = tmp[i * Channels + channel];
        }
    }
}

template<unsigned int BlockSize,
         unsigned int ItemsPerThread,
         unsigned int Channels,
         class Sample,
         class SampleIterator>
ROCPRIM_DEVICE ROCPRIM_INLINE void
    load_samples(unsigned int   flat_id,
                 SampleIterator samples,
                 sample_vector<Sample, Channels> (&values)[ItemsPerThread],
                 unsigned int valid_count)
{
    Sample tmp[Channels * ItemsPerThread];
    block_load_direct_blocked(flat_id, samples, tmp, valid_count * Channels);
    for(unsigned int i = 0; i < ItemsPerThread; i++)
    {
        for(unsigned int channel = 0; channel < Channels; channel++)
        {
            values[i].values[channel] = tmp[i * Channels + channel];
        }
    }
}

template<unsigned int BlockSize, unsigned int ActiveChannels, class Counter>
ROCPRIM_DEVICE ROCPRIM_INLINE
void init_histogram(fixed_array<Counter*, ActiveChannels>     histogram,
                    const fixed_array<size_t, ActiveChannels> bins)
{
    const unsigned int flat_id  = ::rocprim::detail::block_thread_id<0>();
    const unsigned int block_id = ::rocprim::detail::block_id<0>();

    const size_t index = block_id * BlockSize + flat_id;
    for(unsigned int channel = 0; channel < ActiveChannels; channel++)
    {
        if(index < bins[channel])
        {
            histogram[channel][index] = 0;
        }
    }
}

template<unsigned int BlockSize,
         unsigned int ItemsPerThread,
         unsigned int Channels,
         unsigned int ActiveChannels,
         class SampleIterator,
         class Counter,
         class SampleToBinOp>
ROCPRIM_DEVICE ROCPRIM_INLINE
void histogram_shared(SampleIterator                                   samples,
                      unsigned int                                     columns,
                      unsigned int                                     rows,
                      unsigned int                                     row_stride,
                      unsigned int                                     rows_per_block,
                      unsigned int                                     shared_histograms,
                      fixed_array<Counter*, ActiveChannels>            histogram,
                      const fixed_array<SampleToBinOp, ActiveChannels> sample_to_bin_op,
                      const fixed_array<size_t, ActiveChannels>        bins,
                      unsigned int*                                    block_histogram_start)
{
    using sample_type        = typename std::iterator_traits<SampleIterator>::value_type;
    using sample_vector_type = sample_vector<sample_type, Channels>;

    constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;

    const unsigned int flat_id    = ::rocprim::detail::block_thread_id<0>();
    const unsigned int block_id0  = ::rocprim::detail::block_id<0>();
    const unsigned int block_id1  = ::rocprim::detail::block_id<1>();
    const unsigned int grid_size0 = ::rocprim::detail::grid_size<0>();

    // Store the start of the first histogram for each channel
    unsigned int* block_histogram[ActiveChannels];
    unsigned int  total_bins = 0;
    for(unsigned int channel = 0; channel < ActiveChannels; channel++)
    {
        block_histogram[channel] = block_histogram_start + total_bins;
        const unsigned int size  = bins[channel];
        // Prevent LDS bank conflicts
        total_bins += rocprim::detail::is_power_of_two(size) ? size + 1 : size;
    }

    // partial histogram to work with
    const unsigned int thread_shift = (flat_id % shared_histograms) * total_bins;

    // fill all histograms with 0
    for(unsigned int i = flat_id; i < total_bins * shared_histograms; i += BlockSize)
    {
        block_histogram_start[i] = 0;
    }
    ::rocprim::syncthreads();

    const unsigned int start_row = block_id1 * rows_per_block;
    const unsigned int end_row   = ::rocprim::min(rows, start_row + rows_per_block);
    for(unsigned int row = start_row; row < end_row; row++)
    {
        SampleIterator row_samples = samples + row * row_stride;

        unsigned int block_offset = block_id0 * items_per_block;
        while(block_offset < columns)
        {
            sample_vector_type values[ItemsPerThread];

            if(block_offset + items_per_block <= columns)
            {
                load_samples<BlockSize>(flat_id, row_samples + Channels * block_offset, values);

                for(unsigned int i = 0; i < ItemsPerThread; i++)
                {
                    for(unsigned int channel = 0; channel < ActiveChannels; channel++)
                    {
                        size_t bin;
                        if(sample_to_bin_op[channel](values[i].values[channel], bin))
                        {
                            ::rocprim::detail::atomic_add(block_histogram[channel] + bin
                                                              + thread_shift,
                                                          1);
                        }
                    }
                }
            }
            else
            {
                const unsigned int valid_count = columns - block_offset;
                load_samples<BlockSize>(flat_id,
                                        row_samples + Channels * block_offset,
                                        values,
                                        valid_count);

                for(unsigned int i = 0; i < ItemsPerThread; i++)
                {
                    if(flat_id * ItemsPerThread + i < valid_count)
                    {
                        for(unsigned int channel = 0; channel < ActiveChannels; channel++)
                        {
                            size_t bin;
                            if(sample_to_bin_op[channel](values[i].values[channel], bin))
                            {
                                ::rocprim::detail::atomic_add(block_histogram[channel] + bin
                                                                  + thread_shift,
                                                              1);
                            }
                        }
                    }
                }
            }

            block_offset += grid_size0 * items_per_block;
        }
    }
    ::rocprim::syncthreads();

    for(unsigned int channel = 0; channel < ActiveChannels; channel++)
    {
        for(unsigned int bin = flat_id; bin < bins[channel]; bin += BlockSize)
        {
            unsigned int total = 0;
            for(unsigned int i = 0; i < shared_histograms; i++)
            {
                total += block_histogram[channel][bin + i * total_bins];
            }
            if(total > 0)
            {
                ::rocprim::detail::atomic_add(&histogram[channel][bin], total);
            }
        }
    }
}

template<unsigned int BlockSize,
         unsigned int ItemsPerThread,
         unsigned int Channels,
         unsigned int ActiveChannels,
         class SampleIterator,
         class Counter,
         class SampleToBinOp>
ROCPRIM_DEVICE ROCPRIM_INLINE
void histogram_global(SampleIterator                                   samples,
                      unsigned int                                     columns,
                      unsigned int                                     row_stride,
                      fixed_array<Counter*, ActiveChannels>            histogram,
                      const fixed_array<SampleToBinOp, ActiveChannels> sample_to_bin_op,
                      const fixed_array<size_t, ActiveChannels>        bins_bits)
{
    using sample_type        = typename std::iterator_traits<SampleIterator>::value_type;
    using sample_vector_type = sample_vector<sample_type, Channels>;

    constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;

    const unsigned int flat_id      = ::rocprim::detail::block_thread_id<0>();
    const unsigned int block_id0    = ::rocprim::detail::block_id<0>();
    const unsigned int block_id1    = ::rocprim::detail::block_id<1>();
    const unsigned int block_offset = block_id0 * items_per_block;

    samples += block_id1 * row_stride + Channels * block_offset;

    sample_vector_type values[ItemsPerThread];
    unsigned int       valid_count;
    if(block_offset + items_per_block <= columns)
    {
        valid_count = items_per_block;
        load_samples<BlockSize>(flat_id, samples, values);
    }
    else
    {
        valid_count = columns - block_offset;
        load_samples<BlockSize>(flat_id, samples, values, valid_count);
    }

    ROCPRIM_UNROLL
    for(unsigned int i = 0; i < ItemsPerThread; i++)
    {
        for(unsigned int channel = 0; channel < ActiveChannels; channel++)
        {
            size_t bin;
            if(sample_to_bin_op[channel](values[i].values[channel], bin))
            {
                const unsigned int pos = flat_id * ItemsPerThread + i;
                lane_mask_type     same_bin_lanes_mask
                    = ::rocprim::match_any(bin, bins_bits[channel], pos < valid_count);

                if(::rocprim::group_elect(same_bin_lanes_mask))
                {
                    // Write the number of lanes having this bin,
                    // if the current lane is the first (and maybe only) lane with this bin.
                    ::rocprim::detail::atomic_add(&histogram[channel][bin],
                                                  ::rocprim::bit_count(same_bin_lanes_mask));
                }
            }
        }
    }
}

template<unsigned int BlockSize,
         unsigned int ItemsPerThread,
         unsigned int Channels,
         unsigned int ActiveChannels,
         class SampleIterator,
         class Counter,
         class SampleToBinOp>
ROCPRIM_DEVICE ROCPRIM_INLINE
void histogram_private_global(SampleIterator                                   samples,
                              unsigned int                                     columns,
                              unsigned int                                     rows,
                              unsigned int                                     row_stride,
                              fixed_array<Counter*, ActiveChannels>            histogram,
                              const fixed_array<SampleToBinOp, ActiveChannels> sample_to_bin_op,
                              const fixed_array<size_t, ActiveChannels>        bins_bits,
                              const fixed_array<size_t, ActiveChannels>        bins,
                              Counter*                                         private_histograms,
                              const unsigned int                               virtual_max_blocks,
                              unsigned int*                                    block_id_count)
{
    using sample_type        = typename std::iterator_traits<SampleIterator>::value_type;
    using sample_vector_type = sample_vector<sample_type, Channels>;

    constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;

    const unsigned int flat_id = ::rocprim::flat_block_thread_id();
    const unsigned int flat_block_id = ::rocprim::flat_block_id();

    __shared__ unsigned int block_id_count_shared;

    // Store the start of the first histogram for each channel
    Counter* block_histogram[ActiveChannels];
    size_t   total_bins = 0;
    for(unsigned int channel = 0; channel < ActiveChannels; channel++)
    {
        block_histogram[channel] = private_histograms + total_bins;
        total_bins += bins[channel];
    }

    for(unsigned int channel = 0; channel < ActiveChannels; channel++)
    {
        block_histogram[channel] += flat_block_id * total_bins;
    }

    unsigned int virtual_block_id = flat_block_id;

    while(virtual_block_id < virtual_max_blocks)
    {
        const unsigned int row_id       = virtual_block_id % rows;
        const unsigned int col_id       = virtual_block_id / rows;
        const unsigned int block_offset = col_id * items_per_block;

        SampleIterator samples_block = samples + row_id * row_stride + block_offset * Channels;

        sample_vector_type values[ItemsPerThread];
        unsigned int       valid_count;

        if(block_offset + items_per_block <= columns)
        {
            valid_count = items_per_block;
            load_samples<BlockSize>(flat_id, samples_block, values);
        }
        else
        {
            valid_count = columns - block_offset;
            load_samples<BlockSize>(flat_id, samples_block, values, valid_count);
        }

        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            for(unsigned int channel = 0; channel < ActiveChannels; channel++)
            {
                size_t bin;
                if(sample_to_bin_op[channel](values[i].values[channel], bin))
                {
                    const unsigned int pos = flat_id * ItemsPerThread + i;
                    lane_mask_type     same_bin_lanes_mask
                        = ::rocprim::match_any(bin, bins_bits[channel], pos < valid_count);

                    if(::rocprim::group_elect(same_bin_lanes_mask))
                    {
                        // Write the number of lanes having this bin,
                        // if the current lane is the first (and maybe only) lane with this bin.
                        ::rocprim::detail::atomic_add(&block_histogram[channel][bin],
                                                      ::rocprim::bit_count(same_bin_lanes_mask));
                    }
                }
            }
        }

        if(flat_id == 0)
        {
            block_id_count_shared = ::rocprim::detail::atomic_add(block_id_count, 1u);
        }

        ::rocprim::syncthreads();

        virtual_block_id = block_id_count_shared;

        ::rocprim::syncthreads();
    }

    for(unsigned int channel = 0; channel < ActiveChannels; channel++)
    {
        for(size_t bin = flat_id; bin < bins[channel]; bin += BlockSize)
        {
            Counter total = block_histogram[channel][bin];
            if(total > 0)
            {
                ::rocprim::detail::atomic_add(&histogram[channel][bin], total);
            }
        }
    }
}

} // namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DETAIL_DEVICE_HISTOGRAM_HPP_
