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

#ifndef HIPCUB_ROCPRIM_DEVICE_DEVICE_REDUCE_HPP_
#define HIPCUB_ROCPRIM_DEVICE_DEVICE_REDUCE_HPP_

#include <limits>
#include <iterator>

#include "../../config.hpp"

#include "../thread/thread_operators.hpp"

BEGIN_HIPCUB_NAMESPACE

class DeviceReduce
{
public:
    template <
        typename InputIteratorT,
        typename OutputIteratorT,
        typename ReduceOpT,
        typename InitValueT
    >
    HIPCUB_RUNTIME_FUNCTION static
    hipError_t Reduce(void *d_temp_storage,
                      size_t &temp_storage_bytes,
                      InputIteratorT d_in,
                      OutputIteratorT d_out,
                      ReduceOpT reduce_op,
                      InitValueT init_value,
                      int num_items,
                      hipStream_t stream = 0,
                      bool debug_synchronous = false)
    {
        return ::rocprim::reduce(
            d_temp_storage, temp_storage_bytes,
            d_in, d_out, init_value, num_items, reduce_op,
            stream, debug_synchronous
        );
    }
    
    template <
        typename InputIteratorT,
        typename OutputIteratorT
    >
    HIPCUB_RUNTIME_FUNCTION static
    hipError_t Sum(void *d_temp_storage,
                   size_t &temp_storage_bytes,
                   InputIteratorT d_in,
                   OutputIteratorT d_out,
                   int num_items,
                   hipStream_t stream = 0,
                   bool debug_synchronous = false)
    {
        using T = typename std::iterator_traits<InputIteratorT>::value_type;
        return ::rocprim::reduce(
            d_temp_storage, temp_storage_bytes,
            d_in, d_out, T(0), num_items, ::hipcub::Sum(),
            stream, debug_synchronous
        );
    }
    
    template <
        typename InputIteratorT,
        typename OutputIteratorT
    >
    HIPCUB_RUNTIME_FUNCTION static
    hipError_t Min(void *d_temp_storage,
                   size_t &temp_storage_bytes,
                   InputIteratorT d_in,
                   OutputIteratorT d_out,
                   int num_items,
                   hipStream_t stream = 0,
                   bool debug_synchronous = false)
    {
        using T = typename std::iterator_traits<InputIteratorT>::value_type;
        return ::rocprim::reduce(
            d_temp_storage, temp_storage_bytes,
            d_in, d_out, std::numeric_limits<T>::max(), num_items, ::hipcub::Min(),
            stream, debug_synchronous
        );
    }
    
    template <
        typename InputIteratorT,
        typename OutputIteratorT
    >
    HIPCUB_RUNTIME_FUNCTION static
    hipError_t ArgMin(void *d_temp_storage,
                      size_t &temp_storage_bytes,
                      InputIteratorT d_in,
                      OutputIteratorT d_out,
                      int num_items,
                      hipStream_t stream = 0,
                      bool debug_synchronous = false)
    {
        // TODO implement when ArgIndexInputIterator is ready
        (void) d_temp_storage;
        (void) temp_storage_bytes;
        (void) d_in;
        (void) d_out;
        (void) num_items;
        (void) stream;
        (void) debug_synchronous;
        return hipErrorUnknown;
    }
    
    template <
        typename InputIteratorT,
        typename OutputIteratorT
    >
    HIPCUB_RUNTIME_FUNCTION static
    hipError_t Max(void *d_temp_storage,
                   size_t &temp_storage_bytes,
                   InputIteratorT d_in,
                   OutputIteratorT d_out,
                   int num_items,
                   hipStream_t stream = 0,
                   bool debug_synchronous = false)
    {
        using T = typename std::iterator_traits<InputIteratorT>::value_type;
        return ::rocprim::reduce(
            d_temp_storage, temp_storage_bytes,
            d_in, d_out, std::numeric_limits<T>::min(), num_items, ::hipcub::Max(),
            stream, debug_synchronous
        );
    }
    
    template <
        typename InputIteratorT,
        typename OutputIteratorT
    >
    HIPCUB_RUNTIME_FUNCTION static
    hipError_t ArgMax(void *d_temp_storage,
                      size_t &temp_storage_bytes,
                      InputIteratorT d_in,
                      OutputIteratorT d_out,
                      int num_items,
                      hipStream_t stream = 0,
                      bool debug_synchronous = false)
    {
        // TODO implement when ArgIndexInputIterator is ready
        (void) d_temp_storage;
        (void) temp_storage_bytes;
        (void) d_in;
        (void) d_out;
        (void) num_items;
        (void) stream;
        (void) debug_synchronous;
        return hipErrorUnknown;
    }
};

END_HIPCUB_NAMESPACE

#endif // HIPCUB_ROCPRIM_DEVICE_DEVICE_REDUCE_HPP_
