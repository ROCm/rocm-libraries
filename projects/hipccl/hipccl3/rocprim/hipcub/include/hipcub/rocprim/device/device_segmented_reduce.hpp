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
            reduction_op, initial_value,
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

        return ::rocprim::segmented_reduce(
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

        return ::rocprim::segmented_reduce(
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
        
        return ::rocprim::segmented_reduce(
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

        return ::rocprim::segmented_reduce(
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
        
        return ::rocprim::segmented_reduce(
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
