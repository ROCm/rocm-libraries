/******************************************************************************
 * Copyright (c) 2010-2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2017-2026, Advanced Micro Devices, Inc.  All rights reserved.
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

#include <rocprim/device/config_types.hpp>
#include <rocprim/device/device_segmented_reduce.hpp> // IWYU pragma: export
#include <rocprim/iterator/counting_iterator.hpp> // IWYU pragma: export
#include <rocprim/iterator/transform_iterator.hpp> // IWYU pragma: export
#include <rocprim/type_traits.hpp> // IWYU pragma: export

#include _HIPCUB_LIBCXX_INCLUDE(functional)
#include _HIPCUB_STD_INCLUDE(functional)
#include _HIPCUB_STD_INCLUDE(limits)

#include <chrono>
#include <iterator>

BEGIN_HIPCUB_NAMESPACE

namespace detail
{

template<class Config,
         class Selector,
         class InputIterator,
         class OutputIterator,
         class OffsetIterator,
         class ResultType,
         class BinaryFunction>
inline hipError_t launch_segmented_arg_minmax(::rocprim::detail::target current_target,
                                              InputIterator             input,
                                              OutputIterator            output,
                                              OffsetIterator            begin_offsets,
                                              OffsetIterator            end_offsets,
                                              BinaryFunction            reduce_op,
                                              ResultType                initial_value,
                                              ResultType                empty_value,
                                              dim3                      grid,
                                              dim3                      block,
                                              size_t                    shmem,
                                              hipStream_t               stream)
{
    auto kernel = [=](auto target_config)
    {
        // each block processes one segment
        ::rocprim::detail::segmented_reduce<decltype(target_config)>(input,
                                                                     output,
                                                                     begin_offsets,
                                                                     end_offsets,
                                                                     reduce_op,
                                                                     initial_value);
        // no synchronization is needed since thread 0 writes to output

        const unsigned int flat_id    = ::rocprim::detail::block_thread_id<0>();
        const unsigned int segment_id = ::rocprim::detail::block_id<0>();

        // Large indices need bigger offset type than unsigned int
        using offset_type = it_value_t<OffsetIterator>;

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

    return ::rocprim::detail::execute_launch_plan<Config, Selector>(current_target,
                                                                    kernel,
                                                                    grid,
                                                                    block,
                                                                    shmem,
                                                                    stream);
}

/// Dispatch function similar to \p rocprim::segmented_reduce but writes \p empty_value for empty
/// segments and writes a segment-relative index instead of an absolute one.
template<class Config,
         class InputIterator,
         class OutputIterator,
         class OffsetIterator,
         class InitValueType,
         class BinaryFunction>
struct segmented_arg_minmax
{
    constexpr hipError_t operator()(void*                temporary_storage,
                                    size_t&              storage_size,
                                    InputIterator        input,
                                    OutputIterator       output,
                                    _HIPCUB_STD::int64_t segments,
                                    OffsetIterator       begin_offsets,
                                    OffsetIterator       end_offsets,
                                    BinaryFunction       reduce_op,
                                    InitValueType        initial_value,
                                    InitValueType        empty_value,
                                    hipStream_t          stream)
    {
        using input_type  = detail::it_value_t<InputIterator>;
        using result_type = ::rocprim::accumulator_t<BinaryFunction, input_type>;

        using selector = ::rocprim::detail::segmented_reduce_config_selector<result_type>;

        const ::rocprim::detail::target current_target(stream);

        const auto params = ::rocprim::detail::get_config<selector>(Config{}, current_target);
        const unsigned int         block_size = params.kernel_config.block_size;
        const _HIPCUB_STD::int64_t segments_limit
            = static_cast<_HIPCUB_STD::int64_t>(params.kernel_config.size_limit);

        if(temporary_storage == nullptr)
        {
            // Make sure user won't try to allocate 0 bytes memory, because
            // hipMalloc will return nullptr when size is zero.
            storage_size = 4;
            return hipSuccess;
        }

        if(segments == _HIPCUB_STD::int64_t{0u})
        {
            return hipSuccess;
        }

        const _HIPCUB_STD::int64_t num_launch
            = ::rocprim::detail::ceiling_div(static_cast<uint64_t>(segments),
                                             static_cast<uint64_t>(segments_limit));

        for(_HIPCUB_STD::int64_t launch = 0, segments_offset = 0; launch < num_launch;
            ++launch, segments_offset += segments_limit)
        {
            const unsigned int current_segments = static_cast<unsigned int>(
                std::min<_HIPCUB_STD::int64_t>(segments - segments_offset, segments_limit));

            std::chrono::high_resolution_clock::time_point start;

            if constexpr(HIPCUB_DETAIL_DEBUG_SYNC_VALUE)
            {
                start = std::chrono::high_resolution_clock::now();
            }
            ROCPRIM_RETURN_ON_ERROR(launch_segmented_arg_minmax<Config, selector>(
                current_target,
                input,
                output + segments_offset,
                begin_offsets + segments_offset,
                end_offsets + segments_offset,
                reduce_op,
                static_cast<result_type>(initial_value),
                static_cast<result_type>(empty_value),
                dim3(current_segments),
                dim3(block_size),
                0,
                stream));
            HIPCUB_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("segmented_arg_minmax",
                                                       current_segments,
                                                       start);
        }

        return hipSuccess;
    }
};

template<typename OffsetT>
struct transform_idx_to_offset_op
{
    _HIPCUB_STD::int64_t segments;
    unsigned int         segment_length;

    HIPCUB_DEVICE
    inline OffsetT       operator()(_HIPCUB_STD::int64_t i) const
    {
        if(i < segments)
        {
            return static_cast<OffsetT>(segment_length * i);
        }
        else
        {
            return static_cast<OffsetT>(segment_length * segments);
        }
    }
};

template<class Config,
         class InputIterator,
         class OutputIterator,
         class InitValueType,
         class BinaryFunction>
struct segmented_arg_minmax_fixed_size_invoker
{
    using SizeT = _HIPCUB_STD::int64_t;

private:
    template<class OffsetIterator>
    using segmented_arg_minmax_t = segmented_arg_minmax<Config,
                                                        InputIterator,
                                                        OutputIterator,
                                                        OffsetIterator,
                                                        InitValueType,
                                                        BinaryFunction>;

    template<class OffsetT>
    using transform_idx_to_offset_op_t = transform_idx_to_offset_op<OffsetT>;

    template<class ActualSizeT>
    static inline constexpr auto in_range(const SizeT& size)
    {
        using common_t = std::common_type_t<SizeT, ActualSizeT>;
        return static_cast<common_t>(size)
               < static_cast<common_t>(std::numeric_limits<ActualSizeT>::max());
    }

    template<class ArgsPre, class ArgsPost, class OffsetIterator>
    struct invoke_impl
    {
        ArgsPre&             args_pre;
        ArgsPost&            args_post;
        _HIPCUB_STD::int64_t segments;
        OffsetIterator       begin_offsets;
        OffsetIterator       end_offsets;

        template<std::size_t... Ip, std::size_t... Iq>
        hipError_t operator()(std::index_sequence<Ip...>, std::index_sequence<Iq...>) const
        {
            return segmented_arg_minmax_t<OffsetIterator>{}(std::get<Ip>(args_pre)...,
                                                            segments,
                                                            begin_offsets,
                                                            end_offsets,
                                                            std::get<Iq>(args_post)...);
        }
    };

public:
    template<class ArgsPre, class ArgsPost>
    static inline hipError_t invoke(const _HIPCUB_STD::int64_t& segments,
                                    const unsigned int&         segment_length,
                                    ArgsPre&&                   args_pre,
                                    ArgsPost&&                  args_post)
    {
        const SizeT size = segments * segment_length;

        auto run_segmented_reduce = [&](auto transform_op) -> hipError_t
        {
            auto offsets = ::rocprim::make_transform_iterator(
                ::rocprim::make_counting_iterator<_HIPCUB_STD::int64_t>(0),
                transform_op);
            using OffsetIterator            = decltype(offsets);
            constexpr unsigned int PreSize  = std::tuple_size<std::decay_t<ArgsPre>>::value;
            constexpr unsigned int PostSize = std::tuple_size<std::decay_t<ArgsPost>>::value;

            auto impl = invoke_impl<std::decay_t<ArgsPre>, std::decay_t<ArgsPost>, OffsetIterator>{
                args_pre,
                args_post,
                segments,
                offsets,
                offsets + 1};

            return impl(std::make_index_sequence<PreSize>{}, std::make_index_sequence<PostSize>{});
        };

        if(in_range<_HIPCUB_STD::int32_t>(size))
        {
            auto transform_op
                = transform_idx_to_offset_op_t<_HIPCUB_STD::int32_t>{segments, segment_length};
            return run_segmented_reduce(transform_op);
        }
        else
        {
            auto transform_op
                = transform_idx_to_offset_op_t<_HIPCUB_STD::int64_t>{segments, segment_length};
            return run_segmented_reduce(transform_op);
        }
    }
};

} // namespace detail

struct DeviceSegmentedReduce
{
    template<typename InputIteratorT,
             typename OutputIteratorT,
             typename OffsetIteratorT,
             typename ReductionOp,
             typename T>
    HIPCUB_RUNTIME_FUNCTION
    static hipError_t Reduce(void*                d_temp_storage,
                             size_t&              temp_storage_bytes,
                             InputIteratorT       d_in,
                             OutputIteratorT      d_out,
                             _HIPCUB_STD::int64_t num_segments,
                             OffsetIteratorT      d_begin_offsets,
                             OffsetIteratorT      d_end_offsets,
                             ReductionOp          reduction_op,
                             T                    initial_value,
                             hipStream_t          stream = 0)
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
    static hipError_t Reduce(void*                d_temp_storage,
                             size_t&              temp_storage_bytes,
                             InputIteratorT       d_in,
                             OutputIteratorT      d_out,
                             _HIPCUB_STD::int64_t num_segments,
                             OffsetIteratorT      d_begin_offsets,
                             OffsetIteratorT      d_end_offsets,
                             ReductionOp          reduction_op,
                             T                    initial_value,
                             hipStream_t          stream,
                             bool                 debug_synchronous)
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

    template<typename InputIteratorT, typename OutputIteratorT, typename ReductionOp, typename T>
    HIPCUB_RUNTIME_FUNCTION
    static hipError_t Reduce(void*                d_temp_storage,
                             size_t&              temp_storage_bytes,
                             InputIteratorT       d_in,
                             OutputIteratorT      d_out,
                             _HIPCUB_STD::int64_t num_segments,
                             _HIPCUB_STD::int32_t segment_size,
                             ReductionOp          reduction_op,
                             T                    initial_value,
                             hipStream_t          stream = 0)
    {
        return ::rocprim::segmented_reduce(
            d_temp_storage,
            temp_storage_bytes,
            d_in,
            d_out,
            num_segments,
            segment_size,
            ::hipcub::detail::convert_result_type<InputIteratorT, OutputIteratorT>(reduction_op),
            initial_value,
            stream);
    }

    template<typename InputIteratorT, typename OutputIteratorT, typename ReductionOp, typename T>
    HIPCUB_DETAIL_DEPRECATED_DEBUG_SYNCHRONOUS HIPCUB_RUNTIME_FUNCTION
    static hipError_t Reduce(void*                d_temp_storage,
                             size_t&              temp_storage_bytes,
                             InputIteratorT       d_in,
                             OutputIteratorT      d_out,
                             _HIPCUB_STD::int64_t num_segments,
                             _HIPCUB_STD::int32_t segment_size,
                             ReductionOp          reduction_op,
                             T                    initial_value,
                             hipStream_t          stream,
                             bool                 debug_synchronous)
    {
        HIPCUB_DETAIL_RUNTIME_LOG_DEBUG_SYNCHRONOUS();
        return Reduce(d_temp_storage,
                      temp_storage_bytes,
                      d_in,
                      d_out,
                      num_segments,
                      segment_size,
                      reduction_op,
                      initial_value,
                      stream);
    }

    template<typename InputIteratorT, typename OutputIteratorT, typename OffsetIteratorT>
    HIPCUB_RUNTIME_FUNCTION
    static hipError_t Sum(void*                d_temp_storage,
                          size_t&              temp_storage_bytes,
                          InputIteratorT       d_in,
                          OutputIteratorT      d_out,
                          _HIPCUB_STD::int64_t num_segments,
                          OffsetIteratorT      d_begin_offsets,
                          OffsetIteratorT      d_end_offsets,
                          hipStream_t          stream = 0)
    {
        using input_type = detail::it_value_t<InputIteratorT>;

        return Reduce(d_temp_storage,
                      temp_storage_bytes,
                      d_in,
                      d_out,
                      num_segments,
                      d_begin_offsets,
                      d_end_offsets,
                      _HIPCUB_STD::plus<>{},
                      input_type(),
                      stream);
    }

    template<typename InputIteratorT, typename OutputIteratorT, typename OffsetIteratorT>
    HIPCUB_DETAIL_DEPRECATED_DEBUG_SYNCHRONOUS HIPCUB_RUNTIME_FUNCTION
    static hipError_t Sum(void*                d_temp_storage,
                          size_t&              temp_storage_bytes,
                          InputIteratorT       d_in,
                          OutputIteratorT      d_out,
                          _HIPCUB_STD::int64_t num_segments,
                          OffsetIteratorT      d_begin_offsets,
                          OffsetIteratorT      d_end_offsets,
                          hipStream_t          stream,
                          bool                 debug_synchronous)
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

    template<typename InputIteratorT, typename OutputIteratorT>
    HIPCUB_RUNTIME_FUNCTION
    static hipError_t Sum(void*                d_temp_storage,
                          size_t&              temp_storage_bytes,
                          InputIteratorT       d_in,
                          OutputIteratorT      d_out,
                          _HIPCUB_STD::int64_t num_segments,
                          _HIPCUB_STD::int32_t segment_size,
                          hipStream_t          stream = 0)
    {
        using input_type = detail::it_value_t<InputIteratorT>;

        return Reduce(d_temp_storage,
                      temp_storage_bytes,
                      d_in,
                      d_out,
                      num_segments,
                      segment_size,
                      _HIPCUB_STD::plus<>{},
                      input_type(),
                      stream);
    }

    template<typename InputIteratorT, typename OutputIteratorT>
    HIPCUB_DETAIL_DEPRECATED_DEBUG_SYNCHRONOUS HIPCUB_RUNTIME_FUNCTION
    static hipError_t Sum(void*                d_temp_storage,
                          size_t&              temp_storage_bytes,
                          InputIteratorT       d_in,
                          OutputIteratorT      d_out,
                          _HIPCUB_STD::int64_t num_segments,
                          _HIPCUB_STD::int32_t segment_size,
                          hipStream_t          stream,
                          bool                 debug_synchronous)
    {
        HIPCUB_DETAIL_RUNTIME_LOG_DEBUG_SYNCHRONOUS();
        return Sum(d_temp_storage,
                   temp_storage_bytes,
                   d_in,
                   d_out,
                   num_segments,
                   segment_size,
                   stream);
    }

    template<typename InputIteratorT, typename OutputIteratorT, typename OffsetIteratorT>
    HIPCUB_RUNTIME_FUNCTION
    static hipError_t Min(void*                d_temp_storage,
                          size_t&              temp_storage_bytes,
                          InputIteratorT       d_in,
                          OutputIteratorT      d_out,
                          _HIPCUB_STD::int64_t num_segments,
                          OffsetIteratorT      d_begin_offsets,
                          OffsetIteratorT      d_end_offsets,
                          hipStream_t          stream = 0)
    {
        using input_type = detail::it_value_t<InputIteratorT>;

        return Reduce(d_temp_storage,
                      temp_storage_bytes,
                      d_in,
                      d_out,
                      num_segments,
                      d_begin_offsets,
                      d_end_offsets,
#if _HIPCUB_HAS_DEVICE_SYSTEM_STD
                      _HIPCUB_LIBCXX::minimum<>{},
#else
                      [] (auto a, auto b) { return a > b ? b : a;},
#endif
                      _HIPCUB_STD::numeric_limits<input_type>::max(),
                      stream);
    }

    template<typename InputIteratorT, typename OutputIteratorT, typename OffsetIteratorT>
    HIPCUB_DETAIL_DEPRECATED_DEBUG_SYNCHRONOUS HIPCUB_RUNTIME_FUNCTION
    static hipError_t Min(void*                d_temp_storage,
                          size_t&              temp_storage_bytes,
                          InputIteratorT       d_in,
                          OutputIteratorT      d_out,
                          _HIPCUB_STD::int64_t num_segments,
                          OffsetIteratorT      d_begin_offsets,
                          OffsetIteratorT      d_end_offsets,
                          hipStream_t          stream,
                          bool                 debug_synchronous)
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

    template<typename InputIteratorT, typename OutputIteratorT>
    HIPCUB_RUNTIME_FUNCTION
    static hipError_t Min(void*                d_temp_storage,
                          size_t&              temp_storage_bytes,
                          InputIteratorT       d_in,
                          OutputIteratorT      d_out,
                          _HIPCUB_STD::int64_t num_segments,
                          _HIPCUB_STD::int32_t segment_size,
                          hipStream_t          stream = 0)
    {
        using input_type = detail::it_value_t<InputIteratorT>;

        return Reduce(d_temp_storage,
                      temp_storage_bytes,
                      d_in,
                      d_out,
                      num_segments,
                      segment_size,
#if _HIPCUB_HAS_DEVICE_SYSTEM_STD
                      _HIPCUB_LIBCXX::minimum<>{},
#else
                      [] (auto a, auto b) { return a > b ? b : a;},
#endif
                      _HIPCUB_STD::numeric_limits<input_type>::max(),
                      stream);
    }

    template<typename InputIteratorT, typename OutputIteratorT>
    HIPCUB_DETAIL_DEPRECATED_DEBUG_SYNCHRONOUS HIPCUB_RUNTIME_FUNCTION
    static hipError_t Min(void*                d_temp_storage,
                          size_t&              temp_storage_bytes,
                          InputIteratorT       d_in,
                          OutputIteratorT      d_out,
                          _HIPCUB_STD::int64_t num_segments,
                          _HIPCUB_STD::int32_t segment_size,
                          hipStream_t          stream,
                          bool                 debug_synchronous)
    {
        HIPCUB_DETAIL_RUNTIME_LOG_DEBUG_SYNCHRONOUS();
        return Min(d_temp_storage,
                   temp_storage_bytes,
                   d_in,
                   d_out,
                   num_segments,
                   segment_size,
                   stream);
    }

    template<typename InputIteratorT, typename OutputIteratorT, typename OffsetIteratorT>
    HIPCUB_RUNTIME_FUNCTION
    static hipError_t ArgMin(void*                d_temp_storage,
                             size_t&              temp_storage_bytes,
                             InputIteratorT       d_in,
                             OutputIteratorT      d_out,
                             _HIPCUB_STD::int64_t num_segments,
                             OffsetIteratorT      d_begin_offsets,
                             OffsetIteratorT      d_end_offsets,
                             hipStream_t          stream = 0)
    {
        using OffsetT      = int;
        using T            = hipcub::detail::it_value_t<InputIteratorT>;
        using O            = hipcub::detail::it_value_t<OutputIteratorT>;
        using OutputTupleT =
            typename std::conditional<std::is_same_v<O, void>, KeyValuePair<OffsetT, T>, O>::type;

        using OutputValueT = typename OutputTupleT::Value;
        using IteratorT    = ArgIndexInputIterator<InputIteratorT, OffsetT, OutputValueT>;

        using segmented_arg_minmax_t = detail::segmented_arg_minmax<rocprim::default_config,
                                                                    IteratorT,
                                                                    OutputIteratorT,
                                                                    OffsetIteratorT,
                                                                    OutputTupleT,
                                                                    ::hipcub::ArgMin>;

        IteratorT d_indexed_in(d_in);
        // true maximum value of the full range
        // key is ::max because ArgMin finds the lowest value that has the lowest key
        const OutputTupleT init(_HIPCUB_STD::numeric_limits<OffsetT>::max(),
                                detail::get_max_special_value<T>());
        // special value for empty segments
        const OutputTupleT empty_value(1, detail::get_max_value<T>());

        return segmented_arg_minmax_t{}(d_temp_storage,
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
    static hipError_t ArgMin(void*                d_temp_storage,
                             size_t&              temp_storage_bytes,
                             InputIteratorT       d_in,
                             OutputIteratorT      d_out,
                             _HIPCUB_STD::int64_t num_segments,
                             OffsetIteratorT      d_begin_offsets,
                             OffsetIteratorT      d_end_offsets,
                             hipStream_t          stream,
                             bool                 debug_synchronous)
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

    template<typename InputIteratorT, typename OutputIteratorT>
    HIPCUB_RUNTIME_FUNCTION
    static hipError_t ArgMin(void*                d_temp_storage,
                             size_t&              temp_storage_bytes,
                             InputIteratorT       d_in,
                             OutputIteratorT      d_out,
                             _HIPCUB_STD::int64_t num_segments,
                             _HIPCUB_STD::int32_t segment_size,
                             hipStream_t          stream = 0)
    {
        using OffsetT = int;
        using T       = hipcub::detail::it_value_t<InputIteratorT>;
        using O       = hipcub::detail::it_value_t<OutputIteratorT>;
        using OutputTupleT =
            typename std::conditional<std::is_same_v<O, void>, KeyValuePair<OffsetT, T>, O>::type;

        using OutputValueT = typename OutputTupleT::Value;
        using IteratorT    = ArgIndexInputIterator<InputIteratorT, OffsetT, OutputValueT>;
        using OpT          = ::hipcub::ArgMin;

        IteratorT d_indexed_in(d_in);
        // true maximum value of the full range
        // key is ::max because ArgMin finds the lowest value that has the lowest key
        const OutputTupleT init(_HIPCUB_STD::numeric_limits<OffsetT>::max(),
                                detail::get_max_special_value<T>());
        // special value for empty segments
        const OutputTupleT empty_value(1, detail::get_max_value<T>());

        OpT binary_op{};

        return detail::segmented_arg_minmax_fixed_size_invoker<
            rocprim::default_config,
            IteratorT,
            OutputIteratorT,
            OutputTupleT,
            OpT>::invoke(num_segments,
                         segment_size,
                         std::tie(d_temp_storage, temp_storage_bytes, d_indexed_in, d_out),
                         std::tie(binary_op, init, empty_value, stream));
    }

    template<typename InputIteratorT, typename OutputIteratorT>
    HIPCUB_DETAIL_DEPRECATED_DEBUG_SYNCHRONOUS HIPCUB_RUNTIME_FUNCTION
    static hipError_t ArgMin(void*                d_temp_storage,
                             size_t&              temp_storage_bytes,
                             InputIteratorT       d_in,
                             OutputIteratorT      d_out,
                             _HIPCUB_STD::int64_t num_segments,
                             _HIPCUB_STD::int32_t segment_size,
                             hipStream_t          stream,
                             bool                 debug_synchronous)
    {
        HIPCUB_DETAIL_RUNTIME_LOG_DEBUG_SYNCHRONOUS();
        return ArgMin(d_temp_storage,
                      temp_storage_bytes,
                      d_in,
                      d_out,
                      num_segments,
                      segment_size,
                      stream);
    }

    template<typename InputIteratorT, typename OutputIteratorT, typename OffsetIteratorT>
    HIPCUB_RUNTIME_FUNCTION
    static hipError_t Max(void*                d_temp_storage,
                          size_t&              temp_storage_bytes,
                          InputIteratorT       d_in,
                          OutputIteratorT      d_out,
                          _HIPCUB_STD::int64_t num_segments,
                          OffsetIteratorT      d_begin_offsets,
                          OffsetIteratorT      d_end_offsets,
                          hipStream_t          stream = 0)
    {
        using input_type = detail::it_value_t<InputIteratorT>;

        return Reduce(d_temp_storage,
                      temp_storage_bytes,
                      d_in,
                      d_out,
                      num_segments,
                      d_begin_offsets,
                      d_end_offsets,
#if _HIPCUB_HAS_DEVICE_SYSTEM_STD
                      _HIPCUB_LIBCXX::maximum<>{},
#else
                      [] (auto a, auto b) { return a > b ? a : b;},
#endif
                      _HIPCUB_STD::numeric_limits<input_type>::lowest(),
                      stream);
    }

    template<typename InputIteratorT, typename OutputIteratorT, typename OffsetIteratorT>
    HIPCUB_DETAIL_DEPRECATED_DEBUG_SYNCHRONOUS HIPCUB_RUNTIME_FUNCTION
    static hipError_t Max(void*                d_temp_storage,
                          size_t&              temp_storage_bytes,
                          InputIteratorT       d_in,
                          OutputIteratorT      d_out,
                          _HIPCUB_STD::int64_t num_segments,
                          OffsetIteratorT      d_begin_offsets,
                          OffsetIteratorT      d_end_offsets,
                          hipStream_t          stream,
                          bool                 debug_synchronous)
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

    template<typename InputIteratorT, typename OutputIteratorT>
    HIPCUB_RUNTIME_FUNCTION
    static hipError_t Max(void*                d_temp_storage,
                          size_t&              temp_storage_bytes,
                          InputIteratorT       d_in,
                          OutputIteratorT      d_out,
                          _HIPCUB_STD::int64_t num_segments,
                          _HIPCUB_STD::int32_t segment_size,
                          hipStream_t          stream = 0)
    {
        using input_type = detail::it_value_t<InputIteratorT>;

        return Reduce(d_temp_storage,
                      temp_storage_bytes,
                      d_in,
                      d_out,
                      num_segments,
                      segment_size,
#if _HIPCUB_HAS_DEVICE_SYSTEM_STD
                      _HIPCUB_LIBCXX::maximum<>{},
#else
                      [] (auto a, auto b) { return a > b ? a : b;},
#endif
                      _HIPCUB_STD::numeric_limits<input_type>::lowest(),
                      stream);
    }

    template<typename InputIteratorT, typename OutputIteratorT>
    HIPCUB_DETAIL_DEPRECATED_DEBUG_SYNCHRONOUS HIPCUB_RUNTIME_FUNCTION
    static hipError_t Max(void*                d_temp_storage,
                          size_t&              temp_storage_bytes,
                          InputIteratorT       d_in,
                          OutputIteratorT      d_out,
                          _HIPCUB_STD::int64_t num_segments,
                          _HIPCUB_STD::int32_t segment_size,
                          hipStream_t          stream,
                          bool                 debug_synchronous)
    {
        HIPCUB_DETAIL_RUNTIME_LOG_DEBUG_SYNCHRONOUS();
        return Max(d_temp_storage,
                   temp_storage_bytes,
                   d_in,
                   d_out,
                   num_segments,
                   segment_size,
                   stream);
    }

    template<typename InputIteratorT, typename OutputIteratorT, typename OffsetIteratorT>
    HIPCUB_RUNTIME_FUNCTION
    static hipError_t ArgMax(void*                d_temp_storage,
                             size_t&              temp_storage_bytes,
                             InputIteratorT       d_in,
                             OutputIteratorT      d_out,
                             _HIPCUB_STD::int64_t num_segments,
                             OffsetIteratorT      d_begin_offsets,
                             OffsetIteratorT      d_end_offsets,
                             hipStream_t          stream = 0)
    {
        using OffsetT      = int;
        using T            = hipcub::detail::it_value_t<InputIteratorT>;
        using O            = hipcub::detail::it_value_t<OutputIteratorT>;
        using OutputTupleT =
            typename std::conditional<std::is_same_v<O, void>, KeyValuePair<OffsetT, T>, O>::type;

        using OutputValueT = typename OutputTupleT::Value;
        using IteratorT    = ArgIndexInputIterator<InputIteratorT, OffsetT, OutputValueT>;

        using segmented_arg_minmax_t = detail::segmented_arg_minmax<rocprim::default_config,
                                                                    IteratorT,
                                                                    OutputIteratorT,
                                                                    OffsetIteratorT,
                                                                    OutputTupleT,
                                                                    ::hipcub::ArgMax>;

        IteratorT d_indexed_in(d_in);
        // true minimum value of the full range
        // key is ::max because ArgMax finds the highest value that has the lowest key
        const OutputTupleT init(_HIPCUB_STD::numeric_limits<OffsetT>::max(),
                                detail::get_lowest_special_value<T>());
        // special value for empty segments
        const OutputTupleT empty_value(1, detail::get_lowest_value<T>());

        return segmented_arg_minmax_t{}(d_temp_storage,
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
    static hipError_t ArgMax(void*                d_temp_storage,
                             size_t&              temp_storage_bytes,
                             InputIteratorT       d_in,
                             OutputIteratorT      d_out,
                             _HIPCUB_STD::int64_t num_segments,
                             OffsetIteratorT      d_begin_offsets,
                             OffsetIteratorT      d_end_offsets,
                             hipStream_t          stream,
                             bool                 debug_synchronous)
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

    template<typename InputIteratorT, typename OutputIteratorT>
    HIPCUB_RUNTIME_FUNCTION
    static hipError_t ArgMax(void*                d_temp_storage,
                             size_t&              temp_storage_bytes,
                             InputIteratorT       d_in,
                             OutputIteratorT      d_out,
                             _HIPCUB_STD::int64_t num_segments,
                             _HIPCUB_STD::int32_t segment_size,
                             hipStream_t          stream = 0)
    {
        using OffsetT = int;
        using T       = hipcub::detail::it_value_t<InputIteratorT>;
        using O       = hipcub::detail::it_value_t<OutputIteratorT>;
        using OutputTupleT =
            typename std::conditional<std::is_same_v<O, void>, KeyValuePair<OffsetT, T>, O>::type;

        using OutputValueT = typename OutputTupleT::Value;
        using IteratorT    = ArgIndexInputIterator<InputIteratorT, OffsetT, OutputValueT>;
        using OpT          = ::hipcub::ArgMax;

        IteratorT d_indexed_in(d_in);
        // true minimum value of the full range
        // key is ::max because ArgMax finds the highest value that has the lowest key
        const OutputTupleT init(_HIPCUB_STD::numeric_limits<OffsetT>::max(),
                                detail::get_lowest_special_value<T>());
        // special value for empty segments
        const OutputTupleT empty_value(1, detail::get_lowest_value<T>());

        OpT binary_op{};

        return detail::segmented_arg_minmax_fixed_size_invoker<
            rocprim::default_config,
            IteratorT,
            OutputIteratorT,
            OutputTupleT,
            OpT>::invoke(num_segments,
                         segment_size,
                         std::tie(d_temp_storage, temp_storage_bytes, d_indexed_in, d_out),
                         std::tie(binary_op, init, empty_value, stream));
    }

    template<typename InputIteratorT, typename OutputIteratorT>
    HIPCUB_DETAIL_DEPRECATED_DEBUG_SYNCHRONOUS HIPCUB_RUNTIME_FUNCTION
    static hipError_t ArgMax(void*                d_temp_storage,
                             size_t&              temp_storage_bytes,
                             InputIteratorT       d_in,
                             OutputIteratorT      d_out,
                             _HIPCUB_STD::int64_t num_segments,
                             _HIPCUB_STD::int32_t segment_size,
                             hipStream_t          stream,
                             bool                 debug_synchronous)
    {
        HIPCUB_DETAIL_RUNTIME_LOG_DEBUG_SYNCHRONOUS();
        return ArgMax(d_temp_storage,
                      temp_storage_bytes,
                      d_in,
                      d_out,
                      num_segments,
                      segment_size,
                      stream);
    }
};

END_HIPCUB_NAMESPACE

#endif // HIPCUB_ROCPRIM_DEVICE_DEVICE_SEGMENTED_REDUCE_HPP_
