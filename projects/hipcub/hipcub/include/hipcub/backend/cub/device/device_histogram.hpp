/******************************************************************************
 * Copyright (c) 2010-2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2017-2025, Advanced Micro Devices, Inc.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#ifndef HIPCUB_CUB_DEVICE_DEVICE_HISTOGRAM_HPP_
#define HIPCUB_CUB_DEVICE_DEVICE_HISTOGRAM_HPP_

#include "../../../config.hpp"
#include "../../../util_deprecated.hpp"

#include <cub/device/device_histogram.cuh> // IWYU pragma: export

BEGIN_HIPCUB_NAMESPACE

struct DeviceHistogram
{
    template<typename SampleIteratorT, typename CounterT, typename LevelT, typename OffsetT>
    HIPCUB_RUNTIME_FUNCTION static hipError_t HistogramEven(void*           d_temp_storage,
                                                            size_t&         temp_storage_bytes,
                                                            SampleIteratorT d_samples,
                                                            CounterT*       d_histogram,
                                                            int             num_levels,
                                                            LevelT          lower_level,
                                                            LevelT          upper_level,
                                                            OffsetT         num_samples,
                                                            hipStream_t     stream = 0)
    {
        return hipCUDAErrorTohipError(::cub::DeviceHistogram::HistogramEven(d_temp_storage,
                                                                            temp_storage_bytes,
                                                                            d_samples,
                                                                            d_histogram,
                                                                            num_levels,
                                                                            lower_level,
                                                                            upper_level,
                                                                            num_samples,
                                                                            stream));
    }

    template<typename SampleIteratorT, typename CounterT, typename LevelT, typename OffsetT>
    HIPCUB_RUNTIME_FUNCTION HIPCUB_DETAIL_DEPRECATED_DEBUG_SYNCHRONOUS static hipError_t
        HistogramEven(void*           d_temp_storage,
                      size_t&         temp_storage_bytes,
                      SampleIteratorT d_samples,
                      CounterT*       d_histogram,
                      int             num_levels,
                      LevelT          lower_level,
                      LevelT          upper_level,
                      OffsetT         num_samples,
                      hipStream_t     stream,
                      bool            debug_synchronous)
    {
        HIPCUB_DETAIL_RUNTIME_LOG_DEBUG_SYNCHRONOUS();
        return HistogramEven(d_temp_storage,
                             temp_storage_bytes,
                             d_samples,
                             d_histogram,
                             num_levels,
                             lower_level,
                             upper_level,
                             num_samples,
                             stream);
    }

    template<typename SampleIteratorT, typename CounterT, typename LevelT, typename OffsetT>
    HIPCUB_RUNTIME_FUNCTION static hipError_t HistogramEven(void*           d_temp_storage,
                                                            size_t&         temp_storage_bytes,
                                                            SampleIteratorT d_samples,
                                                            CounterT*       d_histogram,
                                                            int             num_levels,
                                                            LevelT          lower_level,
                                                            LevelT          upper_level,
                                                            OffsetT         num_row_samples,
                                                            OffsetT         num_rows,
                                                            size_t          row_stride_bytes,
                                                            hipStream_t     stream = 0)
    {
        return hipCUDAErrorTohipError(::cub::DeviceHistogram::HistogramEven(d_temp_storage,
                                                                            temp_storage_bytes,
                                                                            d_samples,
                                                                            d_histogram,
                                                                            num_levels,
                                                                            lower_level,
                                                                            upper_level,
                                                                            num_row_samples,
                                                                            num_rows,
                                                                            row_stride_bytes,
                                                                            stream));
    }

    template<typename SampleIteratorT, typename CounterT, typename LevelT, typename OffsetT>
    HIPCUB_RUNTIME_FUNCTION HIPCUB_DETAIL_DEPRECATED_DEBUG_SYNCHRONOUS static hipError_t
        HistogramEven(void*           d_temp_storage,
                      size_t&         temp_storage_bytes,
                      SampleIteratorT d_samples,
                      CounterT*       d_histogram,
                      int             num_levels,
                      LevelT          lower_level,
                      LevelT          upper_level,
                      OffsetT         num_row_samples,
                      OffsetT         num_rows,
                      size_t          row_stride_bytes,
                      hipStream_t     stream,
                      bool            debug_synchronous)
    {
        HIPCUB_DETAIL_RUNTIME_LOG_DEBUG_SYNCHRONOUS();
        return HistogramEven(d_temp_storage,
                             temp_storage_bytes,
                             d_samples,
                             d_histogram,
                             num_levels,
                             lower_level,
                             upper_level,
                             num_row_samples,
                             num_rows,
                             row_stride_bytes,
                             stream);
    }

    template<int NUM_CHANNELS,
             int NUM_ACTIVE_CHANNELS,
             typename SampleIteratorT,
             typename CounterT,
             typename LevelT,
             typename OffsetT>
    HIPCUB_RUNTIME_FUNCTION static hipError_t
        MultiHistogramEven(void*           d_temp_storage,
                           size_t&         temp_storage_bytes,
                           SampleIteratorT d_samples,
                           CounterT*       d_histogram[NUM_ACTIVE_CHANNELS],
                           int             num_levels[NUM_ACTIVE_CHANNELS],
                           LevelT          lower_level[NUM_ACTIVE_CHANNELS],
                           LevelT          upper_level[NUM_ACTIVE_CHANNELS],
                           OffsetT         num_pixels,
                           hipStream_t     stream = 0)
    {
        return hipCUDAErrorTohipError(
            ::cub::DeviceHistogram::MultiHistogramEven<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
                d_temp_storage,
                temp_storage_bytes,
                d_samples,
                d_histogram,
                num_levels,
                lower_level,
                upper_level,
                num_pixels,
                stream));
    }

    template<int NUM_CHANNELS,
             int NUM_ACTIVE_CHANNELS,
             typename SampleIteratorT,
             typename CounterT,
             typename LevelT,
             typename OffsetT>
    HIPCUB_RUNTIME_FUNCTION HIPCUB_DETAIL_DEPRECATED_DEBUG_SYNCHRONOUS static hipError_t
        MultiHistogramEven(void*           d_temp_storage,
                           size_t&         temp_storage_bytes,
                           SampleIteratorT d_samples,
                           CounterT*       d_histogram[NUM_ACTIVE_CHANNELS],
                           int             num_levels[NUM_ACTIVE_CHANNELS],
                           LevelT          lower_level[NUM_ACTIVE_CHANNELS],
                           LevelT          upper_level[NUM_ACTIVE_CHANNELS],
                           OffsetT         num_pixels,
                           hipStream_t     stream,
                           bool            debug_synchronous)
    {
        HIPCUB_DETAIL_RUNTIME_LOG_DEBUG_SYNCHRONOUS();
        return MultiHistogramEven<NUM_CHANNELS>(d_temp_storage,
                                                temp_storage_bytes,
                                                d_samples,
                                                d_histogram,
                                                num_levels,
                                                lower_level,
                                                upper_level,
                                                num_pixels,
                                                stream);
    }

    template<int NUM_CHANNELS,
             int NUM_ACTIVE_CHANNELS,
             typename SampleIteratorT,
             typename CounterT,
             typename LevelT,
             typename OffsetT>
    HIPCUB_RUNTIME_FUNCTION static hipError_t
        MultiHistogramEven(void*           d_temp_storage,
                           size_t&         temp_storage_bytes,
                           SampleIteratorT d_samples,
                           CounterT*       d_histogram[NUM_ACTIVE_CHANNELS],
                           int             num_levels[NUM_ACTIVE_CHANNELS],
                           LevelT          lower_level[NUM_ACTIVE_CHANNELS],
                           LevelT          upper_level[NUM_ACTIVE_CHANNELS],
                           OffsetT         num_row_pixels,
                           OffsetT         num_rows,
                           size_t          row_stride_bytes,
                           hipStream_t     stream = 0)
    {
        return hipCUDAErrorTohipError(
            ::cub::DeviceHistogram::MultiHistogramEven<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
                d_temp_storage,
                temp_storage_bytes,
                d_samples,
                d_histogram,
                num_levels,
                lower_level,
                upper_level,
                num_row_pixels,
                num_rows,
                row_stride_bytes,
                stream));
    }

    template<int NUM_CHANNELS,
             int NUM_ACTIVE_CHANNELS,
             typename SampleIteratorT,
             typename CounterT,
             typename LevelT,
             typename OffsetT>
    HIPCUB_RUNTIME_FUNCTION HIPCUB_DETAIL_DEPRECATED_DEBUG_SYNCHRONOUS static hipError_t
        MultiHistogramEven(void*           d_temp_storage,
                           size_t&         temp_storage_bytes,
                           SampleIteratorT d_samples,
                           CounterT*       d_histogram[NUM_ACTIVE_CHANNELS],
                           int             num_levels[NUM_ACTIVE_CHANNELS],
                           LevelT          lower_level[NUM_ACTIVE_CHANNELS],
                           LevelT          upper_level[NUM_ACTIVE_CHANNELS],
                           OffsetT         num_row_pixels,
                           OffsetT         num_rows,
                           size_t          row_stride_bytes,
                           hipStream_t     stream,
                           bool            debug_synchronous)
    {
        HIPCUB_DETAIL_RUNTIME_LOG_DEBUG_SYNCHRONOUS();
        return MultiHistogramEven<NUM_CHANNELS>(d_temp_storage,
                                                temp_storage_bytes,
                                                d_samples,
                                                d_histogram,
                                                num_levels,
                                                lower_level,
                                                upper_level,
                                                num_row_pixels,
                                                num_rows,
                                                row_stride_bytes,
                                                stream);
    }

    template<typename SampleIteratorT, typename CounterT, typename LevelT, typename OffsetT>
    HIPCUB_RUNTIME_FUNCTION static hipError_t HistogramRange(void*           d_temp_storage,
                                                             size_t&         temp_storage_bytes,
                                                             SampleIteratorT d_samples,
                                                             CounterT*       d_histogram,
                                                             int             num_levels,
                                                             LevelT*         d_levels,
                                                             OffsetT         num_samples,
                                                             hipStream_t     stream = 0)
    {
        return hipCUDAErrorTohipError(::cub::DeviceHistogram::HistogramRange(d_temp_storage,
                                                                             temp_storage_bytes,
                                                                             d_samples,
                                                                             d_histogram,
                                                                             num_levels,
                                                                             d_levels,
                                                                             num_samples,
                                                                             stream));
    }

    template<typename SampleIteratorT, typename CounterT, typename LevelT, typename OffsetT>
    HIPCUB_RUNTIME_FUNCTION HIPCUB_DETAIL_DEPRECATED_DEBUG_SYNCHRONOUS static hipError_t
        HistogramRange(void*           d_temp_storage,
                       size_t&         temp_storage_bytes,
                       SampleIteratorT d_samples,
                       CounterT*       d_histogram,
                       int             num_levels,
                       LevelT*         d_levels,
                       OffsetT         num_samples,
                       hipStream_t     stream,
                       bool            debug_synchronous)
    {
        HIPCUB_DETAIL_RUNTIME_LOG_DEBUG_SYNCHRONOUS();
        return HistogramRange(d_temp_storage,
                              temp_storage_bytes,
                              d_samples,
                              d_histogram,
                              num_levels,
                              d_levels,
                              num_samples,
                              stream);
    }

    template<typename SampleIteratorT, typename CounterT, typename LevelT, typename OffsetT>
    HIPCUB_RUNTIME_FUNCTION static hipError_t HistogramRange(void*           d_temp_storage,
                                                             size_t&         temp_storage_bytes,
                                                             SampleIteratorT d_samples,
                                                             CounterT*       d_histogram,
                                                             int             num_levels,
                                                             LevelT*         d_levels,
                                                             OffsetT         num_row_samples,
                                                             OffsetT         num_rows,
                                                             size_t          row_stride_bytes,
                                                             hipStream_t     stream = 0)
    {
        return hipCUDAErrorTohipError(::cub::DeviceHistogram::HistogramRange(d_temp_storage,
                                                                             temp_storage_bytes,
                                                                             d_samples,
                                                                             d_histogram,
                                                                             num_levels,
                                                                             d_levels,
                                                                             num_row_samples,
                                                                             num_rows,
                                                                             row_stride_bytes,
                                                                             stream));
    }

    template<typename SampleIteratorT, typename CounterT, typename LevelT, typename OffsetT>
    HIPCUB_RUNTIME_FUNCTION HIPCUB_DETAIL_DEPRECATED_DEBUG_SYNCHRONOUS static hipError_t
        HistogramRange(void*           d_temp_storage,
                       size_t&         temp_storage_bytes,
                       SampleIteratorT d_samples,
                       CounterT*       d_histogram,
                       int             num_levels,
                       LevelT*         d_levels,
                       OffsetT         num_row_samples,
                       OffsetT         num_rows,
                       size_t          row_stride_bytes,
                       hipStream_t     stream,
                       bool            debug_synchronous)
    {
        HIPCUB_DETAIL_RUNTIME_LOG_DEBUG_SYNCHRONOUS();
        return HistogramRange(d_temp_storage,
                              temp_storage_bytes,
                              d_samples,
                              d_histogram,
                              num_levels,
                              d_levels,
                              num_row_samples,
                              num_rows,
                              row_stride_bytes,
                              stream);
    }

    template<int NUM_CHANNELS,
             int NUM_ACTIVE_CHANNELS,
             typename SampleIteratorT,
             typename CounterT,
             typename LevelT,
             typename OffsetT>
    HIPCUB_RUNTIME_FUNCTION static hipError_t
        MultiHistogramRange(void*           d_temp_storage,
                            size_t&         temp_storage_bytes,
                            SampleIteratorT d_samples,
                            CounterT*       d_histogram[NUM_ACTIVE_CHANNELS],
                            int             num_levels[NUM_ACTIVE_CHANNELS],
                            LevelT*         d_levels[NUM_ACTIVE_CHANNELS],
                            OffsetT         num_pixels,
                            hipStream_t     stream = 0)
    {
        return hipCUDAErrorTohipError(
            ::cub::DeviceHistogram::MultiHistogramRange<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
                d_temp_storage,
                temp_storage_bytes,
                d_samples,
                d_histogram,
                num_levels,
                d_levels,
                num_pixels,
                stream));
    }

    template<int NUM_CHANNELS,
             int NUM_ACTIVE_CHANNELS,
             typename SampleIteratorT,
             typename CounterT,
             typename LevelT,
             typename OffsetT>
    HIPCUB_RUNTIME_FUNCTION HIPCUB_DETAIL_DEPRECATED_DEBUG_SYNCHRONOUS static hipError_t
        MultiHistogramRange(void*           d_temp_storage,
                            size_t&         temp_storage_bytes,
                            SampleIteratorT d_samples,
                            CounterT*       d_histogram[NUM_ACTIVE_CHANNELS],
                            int             num_levels[NUM_ACTIVE_CHANNELS],
                            LevelT*         d_levels[NUM_ACTIVE_CHANNELS],
                            OffsetT         num_pixels,
                            hipStream_t     stream,
                            bool            debug_synchronous)
    {
        HIPCUB_DETAIL_RUNTIME_LOG_DEBUG_SYNCHRONOUS();
        return MultiHistogramRange<NUM_CHANNELS>(d_temp_storage,
                                                 temp_storage_bytes,
                                                 d_samples,
                                                 d_histogram,
                                                 num_levels,
                                                 d_levels,
                                                 num_pixels,
                                                 stream);
    }

    template<int NUM_CHANNELS,
             int NUM_ACTIVE_CHANNELS,
             typename SampleIteratorT,
             typename CounterT,
             typename LevelT,
             typename OffsetT>
    HIPCUB_RUNTIME_FUNCTION static hipError_t
        MultiHistogramRange(void*           d_temp_storage,
                            size_t&         temp_storage_bytes,
                            SampleIteratorT d_samples,
                            CounterT*       d_histogram[NUM_ACTIVE_CHANNELS],
                            int             num_levels[NUM_ACTIVE_CHANNELS],
                            LevelT*         d_levels[NUM_ACTIVE_CHANNELS],
                            OffsetT         num_row_pixels,
                            OffsetT         num_rows,
                            size_t          row_stride_bytes,
                            hipStream_t     stream = 0)
    {
        return hipCUDAErrorTohipError(
            ::cub::DeviceHistogram::MultiHistogramRange<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
                d_temp_storage,
                temp_storage_bytes,
                d_samples,
                d_histogram,
                num_levels,
                d_levels,
                num_row_pixels,
                num_rows,
                row_stride_bytes,
                stream));
    }

    template<int NUM_CHANNELS,
             int NUM_ACTIVE_CHANNELS,
             typename SampleIteratorT,
             typename CounterT,
             typename LevelT,
             typename OffsetT>
    HIPCUB_RUNTIME_FUNCTION HIPCUB_DETAIL_DEPRECATED_DEBUG_SYNCHRONOUS static hipError_t
        MultiHistogramRange(void*           d_temp_storage,
                            size_t&         temp_storage_bytes,
                            SampleIteratorT d_samples,
                            CounterT*       d_histogram[NUM_ACTIVE_CHANNELS],
                            int             num_levels[NUM_ACTIVE_CHANNELS],
                            LevelT*         d_levels[NUM_ACTIVE_CHANNELS],
                            OffsetT         num_row_pixels,
                            OffsetT         num_rows,
                            size_t          row_stride_bytes,
                            hipStream_t     stream,
                            bool            debug_synchronous)
    {
        HIPCUB_DETAIL_RUNTIME_LOG_DEBUG_SYNCHRONOUS();
        return MultiHistogramRange<NUM_CHANNELS>(d_temp_storage,
                                                 temp_storage_bytes,
                                                 d_samples,
                                                 d_histogram,
                                                 num_levels,
                                                 d_levels,
                                                 num_row_pixels,
                                                 num_rows,
                                                 row_stride_bytes,
                                                 stream);
    }
};

END_HIPCUB_NAMESPACE

#endif // HIPCUB_CUB_DEVICE_DEVICE_HISTOGRAM_HPP_
