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

#ifndef HIPCUB_ROCPRIM_DEVICE_DEVICE_HISTOGRAM_HPP_
#define HIPCUB_ROCPRIM_DEVICE_DEVICE_HISTOGRAM_HPP_

#include "../../../config.hpp"
#include "../../../util_deprecated.hpp"

#include "../util_type.hpp"

#include <rocprim/device/device_histogram.hpp> // IWYU pragma: export

BEGIN_HIPCUB_NAMESPACE

namespace detail
{
template<typename IntArithmeticT, typename LevelT, typename CommonT>
    HIPCUB_HOST_DEVICE
HIPCUB_FORCEINLINE bool may_overflow(LevelT /* lower_level */,
                                     LevelT /* upper_level */,
                                     CommonT /* num_bins */,
                                     ::std::false_type /* is_integral */)
{
    return false;
}

// Returns true if the bin computation for a given combination of range (max_level - min_level)
// and number of bins may overflow.
template<typename IntArithmeticT, typename LevelT, typename CommonT>
    HIPCUB_HOST_DEVICE
HIPCUB_FORCEINLINE bool may_overflow(LevelT  lower_level,
                                     LevelT  upper_level,
                                     CommonT num_bins,
                                     ::std::true_type /* is_integral */)
{
    return static_cast<IntArithmeticT>(upper_level - lower_level)
           > (::std::numeric_limits<IntArithmeticT>::max() / static_cast<IntArithmeticT>(num_bins));
}

template<class SampleT, class CommonT>
struct int_arithmetic_t
{
    using type = ::std::conditional_t<
        sizeof(SampleT) + sizeof(CommonT) <= sizeof(uint32_t),
        uint32_t,
#if HIPCUB_IS_INT128_ENABLED
        ::std::conditional_t<(::std::is_same<CommonT, __int128_t>::value
                              || ::std::is_same<CommonT, __uint128_t>::value),
                             CommonT,
                             uint64_t>
#else
        uint64_t
#endif
        >;
};

// If potential overflow is detected, returns hipErrorInvalidValue, otherwise hipSuccess.
template<typename SampleIteratorT, typename LevelT>
HIPCUB_HOST_DEVICE
HIPCUB_FORCEINLINE hipError_t check_overflow(LevelT lower_level, LevelT upper_level, int num_levels)
{
    using sample_type      = typename std::iterator_traits<SampleIteratorT>::value_type;
    using common_type      = typename hipcub::common_type<LevelT, sample_type>::type;
    static_assert(std::is_convertible<common_type, int>::value,
                  "The common type of `LevelT` and `SampleT` must be "
                  "convertible to `int`.");
    static_assert(std::is_trivially_copyable<common_type>::value,
                  "The common type of `LevelT` and `SampleT` must be "
                  "trivially copyable.");
    using int_arithmetic_t = typename int_arithmetic_t<sample_type, common_type>::type;

    if(may_overflow<int_arithmetic_t>(lower_level,
                                      upper_level,
                                      static_cast<common_type>(num_levels - 1),
                                      ::std::is_integral<common_type>{}))
    {
        return hipErrorInvalidValue;
    }
    return hipSuccess;
}
} // namespace detail

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
        if(detail::check_overflow<SampleIteratorT>(lower_level, upper_level, num_levels)
           != hipSuccess)
        {
            return hipErrorInvalidValue;
        }
        return ::rocprim::histogram_even(d_temp_storage,
                                         temp_storage_bytes,
                                         d_samples,
                                         num_samples,
                                         d_histogram,
                                         num_levels,
                                         lower_level,
                                         upper_level,
                                         stream,
                                         HIPCUB_DETAIL_DEBUG_SYNC_VALUE);
    }

    template<typename SampleIteratorT, typename CounterT, typename LevelT, typename OffsetT>
    HIPCUB_DETAIL_DEPRECATED_DEBUG_SYNCHRONOUS HIPCUB_RUNTIME_FUNCTION static hipError_t
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
                             num_samples,
                             d_histogram,
                             num_levels,
                             lower_level,
                             upper_level,
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
        return ::rocprim::histogram_even(d_temp_storage,
                                         temp_storage_bytes,
                                         d_samples,
                                         num_row_samples,
                                         num_rows,
                                         row_stride_bytes,
                                         d_histogram,
                                         num_levels,
                                         lower_level,
                                         upper_level,
                                         stream,
                                         HIPCUB_DETAIL_DEBUG_SYNC_VALUE);
    }

    template<typename SampleIteratorT, typename CounterT, typename LevelT, typename OffsetT>
    HIPCUB_DETAIL_DEPRECATED_DEBUG_SYNCHRONOUS HIPCUB_RUNTIME_FUNCTION static hipError_t
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
                             num_row_samples,
                             num_rows,
                             row_stride_bytes,
                             d_histogram,
                             num_levels,
                             lower_level,
                             upper_level,
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
        unsigned int levels[NUM_ACTIVE_CHANNELS];
        for(unsigned int channel = 0; channel < NUM_ACTIVE_CHANNELS; channel++)
        {
            if(detail::check_overflow<SampleIteratorT>(lower_level[channel],
                                                       upper_level[channel],
                                                       num_levels[channel])
               != hipSuccess)
            {
                return hipErrorInvalidValue;
            }
            levels[channel] = num_levels[channel];
        }
        return ::rocprim::multi_histogram_even<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
            d_temp_storage,
            temp_storage_bytes,
            d_samples,
            num_pixels,
            d_histogram,
            levels,
            lower_level,
            upper_level,
            stream,
            HIPCUB_DETAIL_DEBUG_SYNC_VALUE);
    }

    template<int NUM_CHANNELS,
             int NUM_ACTIVE_CHANNELS,
             typename SampleIteratorT,
             typename CounterT,
             typename LevelT,
             typename OffsetT>
    HIPCUB_DETAIL_DEPRECATED_DEBUG_SYNCHRONOUS HIPCUB_RUNTIME_FUNCTION static hipError_t
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
        unsigned int levels[NUM_ACTIVE_CHANNELS];
        for(unsigned int channel = 0; channel < NUM_ACTIVE_CHANNELS; channel++)
        {
            levels[channel] = num_levels[channel];
        }
        return ::rocprim::multi_histogram_even<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
            d_temp_storage,
            temp_storage_bytes,
            d_samples,
            num_row_pixels,
            num_rows,
            row_stride_bytes,
            d_histogram,
            levels,
            lower_level,
            upper_level,
            stream,
            HIPCUB_DETAIL_DEBUG_SYNC_VALUE);
    }

    template<int NUM_CHANNELS,
             int NUM_ACTIVE_CHANNELS,
             typename SampleIteratorT,
             typename CounterT,
             typename LevelT,
             typename OffsetT>
    HIPCUB_DETAIL_DEPRECATED_DEBUG_SYNCHRONOUS HIPCUB_RUNTIME_FUNCTION static hipError_t
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
        return ::rocprim::histogram_range(d_temp_storage,
                                          temp_storage_bytes,
                                          d_samples,
                                          num_samples,
                                          d_histogram,
                                          num_levels,
                                          d_levels,
                                          stream,
                                          HIPCUB_DETAIL_DEBUG_SYNC_VALUE);
    }

    template<typename SampleIteratorT, typename CounterT, typename LevelT, typename OffsetT>
    HIPCUB_DETAIL_DEPRECATED_DEBUG_SYNCHRONOUS HIPCUB_RUNTIME_FUNCTION static hipError_t
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
        return ::rocprim::histogram_range(d_temp_storage,
                                          temp_storage_bytes,
                                          d_samples,
                                          num_row_samples,
                                          num_rows,
                                          row_stride_bytes,
                                          d_histogram,
                                          num_levels,
                                          d_levels,
                                          stream,
                                          HIPCUB_DETAIL_DEBUG_SYNC_VALUE);
    }

    template<typename SampleIteratorT, typename CounterT, typename LevelT, typename OffsetT>
    HIPCUB_DETAIL_DEPRECATED_DEBUG_SYNCHRONOUS HIPCUB_RUNTIME_FUNCTION static hipError_t
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
        unsigned int levels[NUM_ACTIVE_CHANNELS];
        for(unsigned int channel = 0; channel < NUM_ACTIVE_CHANNELS; channel++)
        {
            levels[channel] = num_levels[channel];
        }
        return ::rocprim::multi_histogram_range<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
            d_temp_storage,
            temp_storage_bytes,
            d_samples,
            num_pixels,
            d_histogram,
            levels,
            d_levels,
            stream,
            HIPCUB_DETAIL_DEBUG_SYNC_VALUE);
    }

    template<int NUM_CHANNELS,
             int NUM_ACTIVE_CHANNELS,
             typename SampleIteratorT,
             typename CounterT,
             typename LevelT,
             typename OffsetT>
    HIPCUB_DETAIL_DEPRECATED_DEBUG_SYNCHRONOUS HIPCUB_RUNTIME_FUNCTION static hipError_t
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
        unsigned int levels[NUM_ACTIVE_CHANNELS];
        for(unsigned int channel = 0; channel < NUM_ACTIVE_CHANNELS; channel++)
        {
            levels[channel] = num_levels[channel];
        }
        return ::rocprim::multi_histogram_range<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
            d_temp_storage,
            temp_storage_bytes,
            d_samples,
            num_row_pixels,
            num_rows,
            row_stride_bytes,
            d_histogram,
            levels,
            d_levels,
            stream,
            HIPCUB_DETAIL_DEBUG_SYNC_VALUE);
    }

    template<int NUM_CHANNELS,
             int NUM_ACTIVE_CHANNELS,
             typename SampleIteratorT,
             typename CounterT,
             typename LevelT,
             typename OffsetT>
    HIPCUB_DETAIL_DEPRECATED_DEBUG_SYNCHRONOUS HIPCUB_RUNTIME_FUNCTION static hipError_t
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

#endif // HIPCUB_ROCPRIM_DEVICE_DEVICE_HISTOGRAM_HPP_
