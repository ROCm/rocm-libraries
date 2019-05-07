/******************************************************************************
 * Copyright (c) 2010-2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2019, Advanced Micro Devices, Inc.  All rights reserved.
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

#include <limits>
#include <iterator>

#include "../../config.hpp"

#include "../thread/thread_operators.hpp"

BEGIN_HIPCUB_NAMESPACE

struct DeviceSegmentedReduce
{
    template<
        typename InputIteratorT,
        typename OutputIteratorT,
        typename OffsetIteratorT,
        typename ReductionOp,
        typename T
    >
    HIPCUB_RUNTIME_FUNCTION static
    hipError_t Reduce(void * d_temp_storage,
                      size_t& temp_storage_bytes,
                      InputIteratorT d_in,
                      OutputIteratorT d_out,
                      int num_segments,
                      OffsetIteratorT d_begin_offsets,
                      OffsetIteratorT d_end_offsets,
                      ReductionOp reduction_op,
                      T initial_value,
                      hipStream_t stream = 0,
                      bool debug_synchronous = false)
    {
        return ::rocprim::segmented_reduce(
            d_temp_storage, temp_storage_bytes,
            d_in, d_out,
            num_segments, d_begin_offsets, d_end_offsets,
            ::hipcub::detail::convert_result_type<InputIteratorT, OutputIteratorT>(reduction_op),
            initial_value,
            stream, debug_synchronous
        );
    }

    template<
        typename InputIteratorT,
        typename OutputIteratorT,
        typename OffsetIteratorT
    >
    HIPCUB_RUNTIME_FUNCTION static
    hipError_t Sum(void * d_temp_storage,
                   size_t& temp_storage_bytes,
                   InputIteratorT d_in,
                   OutputIteratorT d_out,
                   int num_segments,
                   OffsetIteratorT d_begin_offsets,
                   OffsetIteratorT d_end_offsets,
                   hipStream_t stream = 0,
                   bool debug_synchronous = false)
    {
        using input_type = typename std::iterator_traits<InputIteratorT>::value_type;

        return Reduce(
            d_temp_storage, temp_storage_bytes,
            d_in, d_out,
            num_segments, d_begin_offsets, d_end_offsets,
            ::hipcub::Sum(), input_type(),
            stream, debug_synchronous
        );
    }

    template<
        typename InputIteratorT,
        typename OutputIteratorT,
        typename OffsetIteratorT
    >
    HIPCUB_RUNTIME_FUNCTION static
    hipError_t Min(void * d_temp_storage,
                   size_t& temp_storage_bytes,
                   InputIteratorT d_in,
                   OutputIteratorT d_out,
                   int num_segments,
                   OffsetIteratorT d_begin_offsets,
                   OffsetIteratorT d_end_offsets,
                   hipStream_t stream = 0,
                   bool debug_synchronous = false)
    {
        using input_type = typename std::iterator_traits<InputIteratorT>::value_type;

        return Reduce(
            d_temp_storage, temp_storage_bytes,
            d_in, d_out,
            num_segments, d_begin_offsets, d_end_offsets,
            ::hipcub::Min(), std::numeric_limits<input_type>::max(),
            stream, debug_synchronous
        );
    }

    template<
        typename InputIteratorT,
        typename OutputIteratorT,
        typename OffsetIteratorT
    >
    HIPCUB_RUNTIME_FUNCTION static
    hipError_t ArgMin(void * d_temp_storage,
                      size_t& temp_storage_bytes,
                      InputIteratorT d_in,
                      OutputIteratorT d_out,
                      int num_segments,
                      OffsetIteratorT d_begin_offsets,
                      OffsetIteratorT d_end_offsets,
                      hipStream_t stream = 0,
                      bool debug_synchronous = false)
    {
        using OffsetT = int;
        using T = typename std::iterator_traits<InputIteratorT>::value_type;
        using O = typename std::iterator_traits<OutputIteratorT>::value_type;
        using OutputTupleT = typename std::conditional<
                                 std::is_same<O, void>::value,
                                 KeyValuePair<OffsetT, T>,
                                 O
                             >::type;

        using OutputValueT = typename OutputTupleT::Value;
        using IteratorT = ArgIndexInputIterator<InputIteratorT, OffsetT, OutputValueT>;

        IteratorT d_indexed_in(d_in);
        const OutputTupleT init(1, std::numeric_limits<T>::max());

        return Reduce(
            d_temp_storage, temp_storage_bytes,
            d_indexed_in, d_out,
            num_segments, d_begin_offsets, d_end_offsets,
            ::hipcub::ArgMin(), init,
            stream, debug_synchronous
        );
    }

    template<
        typename InputIteratorT,
        typename OutputIteratorT,
        typename OffsetIteratorT
    >
    HIPCUB_RUNTIME_FUNCTION static
    hipError_t Max(void * d_temp_storage,
                   size_t& temp_storage_bytes,
                   InputIteratorT d_in,
                   OutputIteratorT d_out,
                   int num_segments,
                   OffsetIteratorT d_begin_offsets,
                   OffsetIteratorT d_end_offsets,
                   hipStream_t stream = 0,
                   bool debug_synchronous = false)
    {
        using input_type = typename std::iterator_traits<InputIteratorT>::value_type;

        return Reduce(
            d_temp_storage, temp_storage_bytes,
            d_in, d_out,
            num_segments, d_begin_offsets, d_end_offsets,
            ::hipcub::Max(), std::numeric_limits<input_type>::lowest(),
            stream, debug_synchronous
        );
    }

    template<
        typename InputIteratorT,
        typename OutputIteratorT,
        typename OffsetIteratorT
    >
    HIPCUB_RUNTIME_FUNCTION static
    hipError_t ArgMax(void * d_temp_storage,
                      size_t& temp_storage_bytes,
                      InputIteratorT d_in,
                      OutputIteratorT d_out,
                      int num_segments,
                      OffsetIteratorT d_begin_offsets,
                      OffsetIteratorT d_end_offsets,
                      hipStream_t stream = 0,
                      bool debug_synchronous = false)
    {
        using OffsetT = int;
        using T = typename std::iterator_traits<InputIteratorT>::value_type;
        using O = typename std::iterator_traits<OutputIteratorT>::value_type;
        using OutputTupleT = typename std::conditional<
                                 std::is_same<O, void>::value,
                                 KeyValuePair<OffsetT, T>,
                                 O
                             >::type;

        using OutputValueT = typename OutputTupleT::Value;
        using IteratorT = ArgIndexInputIterator<InputIteratorT, OffsetT, OutputValueT>;

        IteratorT d_indexed_in(d_in);
        const OutputTupleT init(1, std::numeric_limits<T>::lowest());

        return Reduce(
            d_temp_storage, temp_storage_bytes,
            d_indexed_in, d_out,
            num_segments, d_begin_offsets, d_end_offsets,
            ::hipcub::ArgMax(), init,
            stream, debug_synchronous
        );
    }
};

END_HIPCUB_NAMESPACE

#endif // HIPCUB_ROCPRIM_DEVICE_DEVICE_SEGMENTED_REDUCE_HPP_
