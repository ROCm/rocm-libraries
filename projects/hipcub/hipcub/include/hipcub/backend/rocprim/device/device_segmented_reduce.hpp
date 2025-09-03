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

#ifndef HIPCUB_ROCPRIM_DEVICE_DEVICE_SEGMENTED_REDUCE_HPP_
#define HIPCUB_ROCPRIM_DEVICE_DEVICE_SEGMENTED_REDUCE_HPP_

#include "../../../config.hpp"
#include "../../../util_deprecated.hpp"

#include "../iterator/arg_index_input_iterator.hpp"
#include "../thread/thread_operators.hpp"
#include "../util_sync.hpp"
#include "device_reduce.hpp"

#include <rocprim/device/device_segmented_reduce.hpp> // IWYU pragma: export
#include <rocprim/type_traits.hpp> // IWYU pragma: export

#include <chrono>
#include <iterator>
#include <limits>

BEGIN_HIPCUB_NAMESPACE

namespace detail
{

template<class Config,
         class InputIterator,
         class OutputIterator,
         class OffsetIterator,
         class ResultType,
         class BinaryFunction>
inline hipError_t launch_segmented_arg_minmax(::rocprim::detail::target_arch arch,
                                              InputIterator                  input,
                                              OutputIterator                 output,
                                              OffsetIterator                 begin_offsets,
                                              OffsetIterator                 end_offsets,
                                              BinaryFunction                 reduce_op,
                                              ResultType                     initial_value,
                                              ResultType                     empty_value,
                                              dim3                           grid,
                                              dim3                           block,
                                              size_t                         shmem,
                                              hipStream_t                    stream)
{
    auto kernel = [=](auto arch_config)
    {
        // each block processes one segment
        ::rocprim::detail::segmented_reduce<decltype(arch_config)>(input,
                                                                   output,
                                                                   begin_offsets,
                                                                   end_offsets,
                                                                   reduce_op,
                                                                   initial_value);
        // no synchronization is needed since thread 0 writes to output

        const unsigned int flat_id    = ::rocprim::detail::block_thread_id<0>();
        const unsigned int segment_id = ::rocprim::detail::block_id<0>();

        // Large indices need bigger offset type than unsigned int
        using offset_type = typename std::iterator_traits<OffsetIterator>::value_type;

        const offset_type begin_offset = begin_offsets[segment_id];
        const offset_type end_offset   = end_offsets[segment_id];

        // transform the segment output
        if(flat_id == 0)
        {
            if(begin_offset == end_offset)
            {
                output[segment_id] = empty_value;
            }
            else
            {
                output[segment_id].key -= begin_offset;
            }
        }
    };

    return ::rocprim::detail::execute_launch_plan<Config>(arch, kernel, grid, block, shmem, stream);
}

/// Dispatch function similar to \p rocprim::segmented_reduce but writes \p empty_value for empty
/// segments and writes a segment-relative index instead of an absolute one.
template<class Config = rocprim::default_config,
         class InputIterator,
         class OutputIterator,
         class OffsetIterator,
         class InitValueType,
         class BinaryFunction>
inline hipError_t segmented_arg_minmax(void*          temporary_storage,
                                       size_t&        storage_size,
                                       InputIterator  input,
                                       OutputIterator output,
                                       unsigned int   segments,
                                       OffsetIterator begin_offsets,
                                       OffsetIterator end_offsets,
                                       BinaryFunction reduce_op,
                                       InitValueType  initial_value,
                                       InitValueType  empty_value,
                                       hipStream_t    stream)
{
    using input_type  = typename std::iterator_traits<InputIterator>::value_type;
    using result_type = ::rocprim::accumulator_t<BinaryFunction, input_type>;

    using config = ::rocprim::detail::wrapped_reduce_config<Config, result_type>;

    ::rocprim::detail::target_arch target_arch;
    hipError_t                     result = host_target_arch(stream, target_arch);
    if(result != hipSuccess)
    {
        return result;
    }
    const ::rocprim::detail::reduce_config_params params
        = ::rocprim::detail::dispatch_target_arch<config, false>(target_arch);

    const unsigned int block_size = params.kernel_config.block_size;

    if(temporary_storage == nullptr)
    {
        // Make sure user won't try to allocate 0 bytes memory, because
        // hipMalloc will return nullptr when size is zero.
        storage_size = 4;
        return hipSuccess;
    }

    if(segments == 0u)
        return hipSuccess;

    std::chrono::high_resolution_clock::time_point start;

    if HIPCUB_IF_CONSTEXPR(HIPCUB_DETAIL_DEBUG_SYNC_VALUE)
    {
        start = std::chrono::high_resolution_clock::now();
    }
    ROCPRIM_RETURN_ON_ERROR(
        launch_segmented_arg_minmax<config>(target_arch,
                                            input,
                                            output,
                                            begin_offsets,
                                            end_offsets,
                                            reduce_op,
                                            static_cast<result_type>(initial_value),
                                            static_cast<result_type>(empty_value),
                                            dim3(segments),
                                            dim3(block_size),
                                            0,
                                            stream));
    HIPCUB_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("segmented_arg_minmax", segments, start);

    return hipSuccess;
}

} // namespace detail

struct DeviceSegmentedReduce
{
    template<typename InputIteratorT,
             typename OutputIteratorT,
             typename OffsetIteratorT,
             typename ReductionOp,
             typename T>
    HIPCUB_RUNTIME_FUNCTION
    static hipError_t Reduce(void*           d_temp_storage,
                             size_t&         temp_storage_bytes,
                             InputIteratorT  d_in,
                             OutputIteratorT d_out,
                             int             num_segments,
                             OffsetIteratorT d_begin_offsets,
                             OffsetIteratorT d_end_offsets,
                             ReductionOp     reduction_op,
                             T               initial_value,
                             hipStream_t     stream = 0)
    {
        return ::rocprim::segmented_reduce(
            d_temp_storage,
            temp_storage_bytes,
            d_in,
            d_out,
            num_segments,
            d_begin_offsets,
            d_end_offsets,
            ::hipcub::detail::convert_result_type<InputIteratorT, OutputIteratorT>(reduction_op),
            initial_value,
            stream);
    }

    template<typename InputIteratorT,
             typename OutputIteratorT,
             typename OffsetIteratorT,
             typename ReductionOp,
             typename T>
    HIPCUB_DETAIL_DEPRECATED_DEBUG_SYNCHRONOUS HIPCUB_RUNTIME_FUNCTION
    static hipError_t Reduce(void*           d_temp_storage,
                             size_t&         temp_storage_bytes,
                             InputIteratorT  d_in,
                             OutputIteratorT d_out,
                             int             num_segments,
                             OffsetIteratorT d_begin_offsets,
                             OffsetIteratorT d_end_offsets,
                             ReductionOp     reduction_op,
                             T               initial_value,
                             hipStream_t     stream,
                             bool            debug_synchronous)
    {
        HIPCUB_DETAIL_RUNTIME_LOG_DEBUG_SYNCHRONOUS();
        return Reduce(d_temp_storage,
                      temp_storage_bytes,
                      d_in,
                      d_out,
                      num_segments,
                      d_begin_offsets,
                      d_end_offsets,
                      reduction_op,
                      initial_value,
                      stream);
    }

    template<typename InputIteratorT, typename OutputIteratorT, typename OffsetIteratorT>
    HIPCUB_RUNTIME_FUNCTION
    static hipError_t Sum(void*           d_temp_storage,
                          size_t&         temp_storage_bytes,
                          InputIteratorT  d_in,
                          OutputIteratorT d_out,
                          int             num_segments,
                          OffsetIteratorT d_begin_offsets,
                          OffsetIteratorT d_end_offsets,
                          hipStream_t     stream = 0)
    {
        using input_type = typename std::iterator_traits<InputIteratorT>::value_type;

        return Reduce(d_temp_storage,
                      temp_storage_bytes,
                      d_in,
                      d_out,
                      num_segments,
                      d_begin_offsets,
                      d_end_offsets,
                      ::hipcub::Sum(),
                      input_type(),
                      stream);
    }

    template<typename InputIteratorT, typename OutputIteratorT, typename OffsetIteratorT>
    HIPCUB_DETAIL_DEPRECATED_DEBUG_SYNCHRONOUS HIPCUB_RUNTIME_FUNCTION
    static hipError_t Sum(void*           d_temp_storage,
                          size_t&         temp_storage_bytes,
                          InputIteratorT  d_in,
                          OutputIteratorT d_out,
                          int             num_segments,
                          OffsetIteratorT d_begin_offsets,
                          OffsetIteratorT d_end_offsets,
                          hipStream_t     stream,
                          bool            debug_synchronous)
    {
        HIPCUB_DETAIL_RUNTIME_LOG_DEBUG_SYNCHRONOUS();
        return Sum(d_temp_storage,
                   temp_storage_bytes,
                   d_in,
                   d_out,
                   num_segments,
                   d_begin_offsets,
                   d_end_offsets,
                   stream);
    }

    template<typename InputIteratorT, typename OutputIteratorT, typename OffsetIteratorT>
    HIPCUB_RUNTIME_FUNCTION
    static hipError_t Min(void*           d_temp_storage,
                          size_t&         temp_storage_bytes,
                          InputIteratorT  d_in,
                          OutputIteratorT d_out,
                          int             num_segments,
                          OffsetIteratorT d_begin_offsets,
                          OffsetIteratorT d_end_offsets,
                          hipStream_t     stream = 0)
    {
        using input_type = typename std::iterator_traits<InputIteratorT>::value_type;

        return Reduce(d_temp_storage,
                      temp_storage_bytes,
                      d_in,
                      d_out,
                      num_segments,
                      d_begin_offsets,
                      d_end_offsets,
                      ::hipcub::Min(),
                      std::numeric_limits<input_type>::max(),
                      stream);
    }

    template<typename InputIteratorT, typename OutputIteratorT, typename OffsetIteratorT>
    HIPCUB_DETAIL_DEPRECATED_DEBUG_SYNCHRONOUS HIPCUB_RUNTIME_FUNCTION
    static hipError_t Min(void*           d_temp_storage,
                          size_t&         temp_storage_bytes,
                          InputIteratorT  d_in,
                          OutputIteratorT d_out,
                          int             num_segments,
                          OffsetIteratorT d_begin_offsets,
                          OffsetIteratorT d_end_offsets,
                          hipStream_t     stream,
                          bool            debug_synchronous)
    {
        HIPCUB_DETAIL_RUNTIME_LOG_DEBUG_SYNCHRONOUS();
        return Min(d_temp_storage,
                   temp_storage_bytes,
                   d_in,
                   d_out,
                   num_segments,
                   d_begin_offsets,
                   d_end_offsets,
                   stream);
    }

    template<typename InputIteratorT, typename OutputIteratorT, typename OffsetIteratorT>
    HIPCUB_RUNTIME_FUNCTION
    static hipError_t ArgMin(void*           d_temp_storage,
                             size_t&         temp_storage_bytes,
                             InputIteratorT  d_in,
                             OutputIteratorT d_out,
                             int             num_segments,
                             OffsetIteratorT d_begin_offsets,
                             OffsetIteratorT d_end_offsets,
                             hipStream_t     stream = 0)
    {
        using OffsetT      = int;
        using T            = typename std::iterator_traits<InputIteratorT>::value_type;
        using O            = typename std::iterator_traits<OutputIteratorT>::value_type;
        using OutputTupleT = typename std::
            conditional<std::is_same<O, void>::value, KeyValuePair<OffsetT, T>, O>::type;

        using OutputValueT = typename OutputTupleT::Value;
        using IteratorT    = ArgIndexInputIterator<InputIteratorT, OffsetT, OutputValueT>;

        IteratorT d_indexed_in(d_in);
        // true maximum value of the full range
        // key is ::max because ArgMin finds the lowest value that has the lowest key
        const OutputTupleT init(std::numeric_limits<OffsetT>::max(),
                                detail::get_max_special_value<T>());
        // special value for empty segments
        const OutputTupleT empty_value(1, detail::get_max_value<T>());

        return detail::segmented_arg_minmax(d_temp_storage,
                                            temp_storage_bytes,
                                            d_indexed_in,
                                            d_out,
                                            num_segments,
                                            d_begin_offsets,
                                            d_end_offsets,
                                            ::hipcub::ArgMin(),
                                            init,
                                            empty_value,
                                            stream);
    }

    template<typename InputIteratorT, typename OutputIteratorT, typename OffsetIteratorT>
    HIPCUB_DETAIL_DEPRECATED_DEBUG_SYNCHRONOUS HIPCUB_RUNTIME_FUNCTION
    static hipError_t ArgMin(void*           d_temp_storage,
                             size_t&         temp_storage_bytes,
                             InputIteratorT  d_in,
                             OutputIteratorT d_out,
                             int             num_segments,
                             OffsetIteratorT d_begin_offsets,
                             OffsetIteratorT d_end_offsets,
                             hipStream_t     stream,
                             bool            debug_synchronous)
    {
        HIPCUB_DETAIL_RUNTIME_LOG_DEBUG_SYNCHRONOUS();
        return ArgMin(d_temp_storage,
                      temp_storage_bytes,
                      d_in,
                      d_out,
                      num_segments,
                      d_begin_offsets,
                      d_end_offsets,
                      stream);
    }

    template<typename InputIteratorT, typename OutputIteratorT, typename OffsetIteratorT>
    HIPCUB_RUNTIME_FUNCTION
    static hipError_t Max(void*           d_temp_storage,
                          size_t&         temp_storage_bytes,
                          InputIteratorT  d_in,
                          OutputIteratorT d_out,
                          int             num_segments,
                          OffsetIteratorT d_begin_offsets,
                          OffsetIteratorT d_end_offsets,
                          hipStream_t     stream = 0)
    {
        using input_type = typename std::iterator_traits<InputIteratorT>::value_type;

        return Reduce(d_temp_storage,
                      temp_storage_bytes,
                      d_in,
                      d_out,
                      num_segments,
                      d_begin_offsets,
                      d_end_offsets,
                      ::hipcub::Max(),
                      std::numeric_limits<input_type>::lowest(),
                      stream);
    }

    template<typename InputIteratorT, typename OutputIteratorT, typename OffsetIteratorT>
    HIPCUB_DETAIL_DEPRECATED_DEBUG_SYNCHRONOUS HIPCUB_RUNTIME_FUNCTION
    static hipError_t Max(void*           d_temp_storage,
                          size_t&         temp_storage_bytes,
                          InputIteratorT  d_in,
                          OutputIteratorT d_out,
                          int             num_segments,
                          OffsetIteratorT d_begin_offsets,
                          OffsetIteratorT d_end_offsets,
                          hipStream_t     stream,
                          bool            debug_synchronous)
    {
        HIPCUB_DETAIL_RUNTIME_LOG_DEBUG_SYNCHRONOUS();
        return Max(d_temp_storage,
                   temp_storage_bytes,
                   d_in,
                   d_out,
                   num_segments,
                   d_begin_offsets,
                   d_end_offsets,
                   stream);
    }

    template<typename InputIteratorT, typename OutputIteratorT, typename OffsetIteratorT>
    HIPCUB_RUNTIME_FUNCTION
    static hipError_t ArgMax(void*           d_temp_storage,
                             size_t&         temp_storage_bytes,
                             InputIteratorT  d_in,
                             OutputIteratorT d_out,
                             int             num_segments,
                             OffsetIteratorT d_begin_offsets,
                             OffsetIteratorT d_end_offsets,
                             hipStream_t     stream = 0)
    {
        using OffsetT      = int;
        using T            = typename std::iterator_traits<InputIteratorT>::value_type;
        using O            = typename std::iterator_traits<OutputIteratorT>::value_type;
        using OutputTupleT = typename std::
            conditional<std::is_same<O, void>::value, KeyValuePair<OffsetT, T>, O>::type;

        using OutputValueT = typename OutputTupleT::Value;
        using IteratorT    = ArgIndexInputIterator<InputIteratorT, OffsetT, OutputValueT>;

        IteratorT d_indexed_in(d_in);
        // true minimum value of the full range
        // key is ::max because ArgMax finds the highest value that has the lowest key
        const OutputTupleT init(std::numeric_limits<OffsetT>::max(),
                                detail::get_lowest_special_value<T>());
        // special value for empty segments
        const OutputTupleT empty_value(1, detail::get_lowest_value<T>());

        return detail::segmented_arg_minmax(d_temp_storage,
                                            temp_storage_bytes,
                                            d_indexed_in,
                                            d_out,
                                            num_segments,
                                            d_begin_offsets,
                                            d_end_offsets,
                                            ::hipcub::ArgMax(),
                                            init,
                                            empty_value,
                                            stream);
    }

    template<typename InputIteratorT, typename OutputIteratorT, typename OffsetIteratorT>
    HIPCUB_DETAIL_DEPRECATED_DEBUG_SYNCHRONOUS HIPCUB_RUNTIME_FUNCTION
    static hipError_t ArgMax(void*           d_temp_storage,
                             size_t&         temp_storage_bytes,
                             InputIteratorT  d_in,
                             OutputIteratorT d_out,
                             int             num_segments,
                             OffsetIteratorT d_begin_offsets,
                             OffsetIteratorT d_end_offsets,
                             hipStream_t     stream,
                             bool            debug_synchronous)
    {
        HIPCUB_DETAIL_RUNTIME_LOG_DEBUG_SYNCHRONOUS();
        return ArgMax(d_temp_storage,
                      temp_storage_bytes,
                      d_in,
                      d_out,
                      num_segments,
                      d_begin_offsets,
                      d_end_offsets,
                      stream);
    }
};

END_HIPCUB_NAMESPACE

#endif // HIPCUB_ROCPRIM_DEVICE_DEVICE_SEGMENTED_REDUCE_HPP_
