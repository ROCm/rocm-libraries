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

#ifndef HIPCUB_ROCPRIM_DEVICE_DEVICE_HISTOGRAM_HPP_
#define HIPCUB_ROCPRIM_DEVICE_DEVICE_HISTOGRAM_HPP_

#include "../../config.hpp"

#include "../util_type.hpp"

BEGIN_HIPCUB_NAMESPACE

struct DeviceHistogram
{
    template<
        typename SampleIteratorT,
        typename CounterT,
        typename LevelT,
        typename OffsetT
    >
    HIPCUB_RUNTIME_FUNCTION static
    hipError_t HistogramEven(void * d_temp_storage,
                             size_t& temp_storage_bytes,
                             SampleIteratorT d_samples,
                             CounterT * d_histogram,
                             int num_levels,
                             LevelT lower_level,
                             LevelT upper_level,
                             OffsetT num_samples,
                             hipStream_t stream = 0,
                             bool debug_synchronous = false)
    {
        return ::rocprim::histogram_even(
            d_temp_storage, temp_storage_bytes,
            d_samples, num_samples,
            d_histogram,
            num_levels, lower_level, upper_level,
            stream, debug_synchronous
        );
    }

    template<
        typename SampleIteratorT,
        typename CounterT,
        typename LevelT,
        typename OffsetT
    >
    HIPCUB_RUNTIME_FUNCTION static
    hipError_t HistogramEven(void * d_temp_storage,
                             size_t& temp_storage_bytes,
                             SampleIteratorT d_samples,
                             CounterT * d_histogram,
                             int num_levels,
                             LevelT lower_level,
                             LevelT upper_level,
                             OffsetT num_row_samples,
                             OffsetT num_rows,
                             size_t row_stride_bytes,
                             hipStream_t stream = 0,
                             bool debug_synchronous = false)
    {
        return ::rocprim::histogram_even(
            d_temp_storage, temp_storage_bytes,
            d_samples, num_row_samples, num_rows, row_stride_bytes,
            d_histogram,
            num_levels, lower_level, upper_level,
            stream, debug_synchronous
        );
    }

    template<
        int NUM_CHANNELS,
        int NUM_ACTIVE_CHANNELS,
        typename SampleIteratorT,
        typename CounterT,
        typename LevelT,
        typename OffsetT
    >
    HIPCUB_RUNTIME_FUNCTION static
    hipError_t MultiHistogramEven(void * d_temp_storage,
                                  size_t& temp_storage_bytes,
                                  SampleIteratorT d_samples,
                                  CounterT * d_histogram[NUM_ACTIVE_CHANNELS],
                                  int num_levels[NUM_ACTIVE_CHANNELS],
                                  LevelT lower_level[NUM_ACTIVE_CHANNELS],
                                  LevelT upper_level[NUM_ACTIVE_CHANNELS],
                                  OffsetT num_pixels,
                                  hipStream_t stream = 0,
                                  bool debug_synchronous = false)
    {
        unsigned int levels[NUM_ACTIVE_CHANNELS];
        for(unsigned int channel = 0; channel < NUM_ACTIVE_CHANNELS; channel++)
        {
            levels[channel] = num_levels[channel];
        }
        return ::rocprim::multi_histogram_even<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
            d_temp_storage, temp_storage_bytes,
            d_samples, num_pixels,
            d_histogram,
            levels, lower_level, upper_level,
            stream, debug_synchronous
        );
    }

    template<
        int NUM_CHANNELS,
        int NUM_ACTIVE_CHANNELS,
        typename SampleIteratorT,
        typename CounterT,
        typename LevelT,
        typename OffsetT
    >
    HIPCUB_RUNTIME_FUNCTION static
    hipError_t MultiHistogramEven(void * d_temp_storage,
                                  size_t& temp_storage_bytes,
                                  SampleIteratorT d_samples,
                                  CounterT * d_histogram[NUM_ACTIVE_CHANNELS],
                                  int num_levels[NUM_ACTIVE_CHANNELS],
                                  LevelT lower_level[NUM_ACTIVE_CHANNELS],
                                  LevelT upper_level[NUM_ACTIVE_CHANNELS],
                                  OffsetT num_row_pixels,
                                  OffsetT num_rows,
                                  size_t row_stride_bytes,
                                  hipStream_t stream = 0,
                                  bool debug_synchronous = false)
    {
        unsigned int levels[NUM_ACTIVE_CHANNELS];
        for(unsigned int channel = 0; channel < NUM_ACTIVE_CHANNELS; channel++)
        {
            levels[channel] = num_levels[channel];
        }
        return ::rocprim::multi_histogram_even<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
            d_temp_storage, temp_storage_bytes,
            d_samples, num_row_pixels, num_rows, row_stride_bytes,
            d_histogram,
            levels, lower_level, upper_level,
            stream, debug_synchronous
        );
    }

    template<
        typename SampleIteratorT,
        typename CounterT,
        typename LevelT,
        typename OffsetT
    >
    HIPCUB_RUNTIME_FUNCTION static
    hipError_t HistogramRange(void * d_temp_storage,
                              size_t& temp_storage_bytes,
                              SampleIteratorT d_samples,
                              CounterT * d_histogram,
                              int num_levels,
                              LevelT * d_levels,
                              OffsetT num_samples,
                              hipStream_t stream = 0,
                              bool debug_synchronous = false)
    {
        return ::rocprim::histogram_range(
            d_temp_storage, temp_storage_bytes,
            d_samples, num_samples,
            d_histogram,
            num_levels, d_levels,
            stream, debug_synchronous
        );
    }

    template<
        typename SampleIteratorT,
        typename CounterT,
        typename LevelT,
        typename OffsetT
    >
    HIPCUB_RUNTIME_FUNCTION static
    hipError_t HistogramRange(void * d_temp_storage,
                              size_t& temp_storage_bytes,
                              SampleIteratorT d_samples,
                              CounterT * d_histogram,
                              int num_levels,
                              LevelT * d_levels,
                              OffsetT num_row_samples,
                              OffsetT num_rows,
                              size_t row_stride_bytes,
                              hipStream_t stream = 0,
                              bool debug_synchronous = false)
    {
        return ::rocprim::histogram_range(
            d_temp_storage, temp_storage_bytes,
            d_samples, num_row_samples, num_rows, row_stride_bytes,
            d_histogram,
            num_levels, d_levels,
            stream, debug_synchronous
        );
    }

    template<
        int NUM_CHANNELS,
        int NUM_ACTIVE_CHANNELS,
        typename SampleIteratorT,
        typename CounterT,
        typename LevelT,
        typename OffsetT
    >
    HIPCUB_RUNTIME_FUNCTION static
    hipError_t MultiHistogramRange(void * d_temp_storage,
                                   size_t& temp_storage_bytes,
                                   SampleIteratorT d_samples,
                                   CounterT * d_histogram[NUM_ACTIVE_CHANNELS],
                                   int num_levels[NUM_ACTIVE_CHANNELS],
                                   LevelT * d_levels[NUM_ACTIVE_CHANNELS],
                                   OffsetT num_pixels,
                                   hipStream_t stream = 0,
                                   bool debug_synchronous = false)
    {
        unsigned int levels[NUM_ACTIVE_CHANNELS];
        for(unsigned int channel = 0; channel < NUM_ACTIVE_CHANNELS; channel++)
        {
            levels[channel] = num_levels[channel];
        }
        return ::rocprim::multi_histogram_range<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
            d_temp_storage, temp_storage_bytes,
            d_samples, num_pixels,
            d_histogram,
            levels, d_levels,
            stream, debug_synchronous
        );
    }

    template<
        int NUM_CHANNELS,
        int NUM_ACTIVE_CHANNELS,
        typename SampleIteratorT,
        typename CounterT,
        typename LevelT,
        typename OffsetT
    >
    HIPCUB_RUNTIME_FUNCTION static
    hipError_t MultiHistogramRange(void * d_temp_storage,
                                   size_t& temp_storage_bytes,
                                   SampleIteratorT d_samples,
                                   CounterT * d_histogram[NUM_ACTIVE_CHANNELS],
                                   int num_levels[NUM_ACTIVE_CHANNELS],
                                   LevelT * d_levels[NUM_ACTIVE_CHANNELS],
                                   OffsetT num_row_pixels,
                                   OffsetT num_rows,
                                   size_t row_stride_bytes,
                                   hipStream_t stream = 0,
                                   bool debug_synchronous = false)
    {
        unsigned int levels[NUM_ACTIVE_CHANNELS];
        for(unsigned int channel = 0; channel < NUM_ACTIVE_CHANNELS; channel++)
        {
            levels[channel] = num_levels[channel];
        }
        return ::rocprim::multi_histogram_range<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
            d_temp_storage, temp_storage_bytes,
            d_samples, num_row_pixels, num_rows, row_stride_bytes,
            d_histogram,
            levels, d_levels,
            stream, debug_synchronous
        );
    }
};

END_HIPCUB_NAMESPACE

#endif // HIPCUB_ROCPRIM_DEVICE_DEVICE_HISTOGRAM_HPP_
