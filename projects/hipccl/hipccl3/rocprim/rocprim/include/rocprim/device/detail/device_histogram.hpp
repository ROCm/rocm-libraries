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

#ifndef ROCPRIM_DEVICE_DETAIL_DEVICE_HISTOGRAM_HPP_
#define ROCPRIM_DEVICE_DETAIL_DEVICE_HISTOGRAM_HPP_

#include <cmath>
#include <type_traits>
#include <iterator>

#include "../../config.hpp"
#include "../../detail/various.hpp"

#include "../../intrinsics.hpp"
#include "../../functional.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<class Level>
struct sample_to_bin_even
{
    unsigned int bins;
    Level lower_level;
    Level upper_level;
    Level scale;

    ROCPRIM_HOST_DEVICE inline
    sample_to_bin_even(unsigned int bins, Level lower_level, Level upper_level)
        : bins(bins),
          lower_level(lower_level),
          upper_level(upper_level),
          scale((upper_level - lower_level) / bins)
    {}

    template<class Sample>
    ROCPRIM_HOST_DEVICE inline
    int operator()(Sample sample) const
    {
        const Level s = static_cast<Level>(sample);
        if(s >= lower_level && s < upper_level)
        {
            return (s - lower_level) / scale;
        }
        else
        {
            return -1;
        }
    }
};

// Returns index of the first element in values that is greater than value, or count if no such element is found.
template<class T>
ROCPRIM_HOST_DEVICE inline
unsigned int upper_bound(const T * values, unsigned int count, T value)
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
    unsigned int bins;
    const Level * level_values;

    ROCPRIM_HOST_DEVICE inline
    sample_to_bin_range(unsigned int bins, const Level * level_values)
        : bins(bins), level_values(level_values)
    {}

    template<class Sample>
    ROCPRIM_HOST_DEVICE inline
    int operator()(Sample sample) const
    {
        const Level s = static_cast<Level>(sample);
        const unsigned int bin = upper_bound(level_values, bins + 1, s) - 1;
        if(bin < bins)
        {
            return bin;
        }
        else
        {
            return -1;
        }
    }
};

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class SampleIterator,
    class Counter,
    class SampleToBinOp
>
ROCPRIM_DEVICE inline
void histogram_shared(SampleIterator samples,
                      unsigned int size,
                      Counter * histogram,
                      unsigned int * block_histogram,
                      SampleToBinOp sample_to_bin_op,
                      unsigned int bins)
{
    using sample_type = typename std::iterator_traits<SampleIterator>::value_type;

    constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;

    const unsigned int flat_id = ::rocprim::detail::block_thread_id<0>();
    const unsigned int block_id = ::rocprim::detail::block_id<0>();
    const unsigned int grid_size = ::rocprim::detail::grid_size<0>();

    for(unsigned int bin = flat_id; bin < bins; bin += BlockSize)
    {
        block_histogram[bin] = 0;
    }
    ::rocprim::syncthreads();

    unsigned int block_offset = block_id * items_per_block;
    while(block_offset < size)
    {
        sample_type values[ItemsPerThread];

        unsigned int valid_count;
        if(block_offset + items_per_block <= size)
        {
            valid_count = items_per_block;
            block_load_direct_striped<BlockSize>(flat_id, samples + block_offset, values);
        }
        else
        {
            valid_count = size - block_offset;
            block_load_direct_striped<BlockSize>(flat_id, samples + block_offset, values, valid_count);
        }

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            if(i * BlockSize + flat_id < valid_count)
            {
                const int bin = sample_to_bin_op(values[i]);
                if(bin != -1)
                {
                    ::rocprim::atomic_add(&block_histogram[bin], 1);
                }
            }
        }

        block_offset += grid_size * items_per_block;
    }
    ::rocprim::syncthreads();

    for(unsigned int bin = flat_id; bin < bins; bin += BlockSize)
    {
        if(block_histogram[bin] > 0)
        {
            ::rocprim::atomic_add(&histogram[bin], block_histogram[bin]);
        }
    }
}

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class SampleIterator,
    class Counter,
    class SampleToBinOp
>
ROCPRIM_DEVICE inline
void histogram_global(SampleIterator samples,
                      unsigned int size,
                      Counter * histogram,
                      SampleToBinOp sample_to_bin_op,
                      unsigned int bins_bits)
{
    using sample_type = typename std::iterator_traits<SampleIterator>::value_type;

    constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;

    const unsigned int flat_id = ::rocprim::detail::block_thread_id<0>();
    const unsigned int block_id = ::rocprim::detail::block_id<0>();
    const unsigned int block_offset = block_id * items_per_block;

    sample_type values[ItemsPerThread];

    unsigned int valid_count;
    if(block_offset + items_per_block <= size)
    {
        valid_count = items_per_block;
        block_load_direct_striped<BlockSize>(flat_id, samples + block_offset, values);
    }
    else
    {
        valid_count = size - block_offset;
        block_load_direct_striped<BlockSize>(flat_id, samples + block_offset, values, valid_count);
    }

    for(unsigned int i = 0; i < ItemsPerThread; i++)
    {
        const int bin = sample_to_bin_op(values[i]);
        if(bin != -1)
        {
            const unsigned int pos = i * BlockSize + flat_id;
            unsigned long long same_bin_lanes_mask = ::rocprim::ballot(pos < valid_count);
            for(unsigned int b = 0; b < bins_bits; b++)
            {
                const unsigned int bit_set = bin & (1u << b);
                const unsigned long long bit_set_mask = ::rocprim::ballot(bit_set);
                same_bin_lanes_mask &= (bit_set ? bit_set_mask : ~bit_set_mask);
            }
            const unsigned int same_bin_count = ::rocprim::bit_count(same_bin_lanes_mask);
            const unsigned int prev_same_bin_count = ::rocprim::masked_bit_count(same_bin_lanes_mask);
            if(prev_same_bin_count == 0)
            {
                // Write the number of lanes having this bin,
                // if the current lane is the first (and maybe only) lane with this bin.
                ::rocprim::atomic_add(&histogram[bin], same_bin_count);
            }
        }
    }
}

} // end of detail namespace

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DETAIL_DEVICE_HISTOGRAM_HPP_
