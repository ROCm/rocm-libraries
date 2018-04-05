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

#ifndef HIPCUB_CUB_DEVICE_DEVICE_SELECT_HPP_
#define HIPCUB_CUB_DEVICE_DEVICE_SELECT_HPP_

#include "../../config.hpp"

#include <cub/device/device_select.cuh>

BEGIN_HIPCUB_NAMESPACE

class DeviceSelect
{
public:
    template <
        typename InputIteratorT,
        typename FlagIterator,
        typename OutputIteratorT,
        typename NumSelectedIteratorT
    >
    HIPCUB_RUNTIME_FUNCTION static
    hipError_t Flagged(void *d_temp_storage,
                       size_t &temp_storage_bytes,
                       InputIteratorT d_in,
                       FlagIterator d_flags,
                       OutputIteratorT d_out,
                       NumSelectedIteratorT d_num_selected_out,
                       int num_items,
                       hipStream_t stream = 0,
                       bool debug_synchronous = false)
    {
        return hipCUDAErrorTohipError(
            ::cub::DeviceSelect::Flagged(
                d_temp_storage, temp_storage_bytes,
                d_in, d_flags,
                d_out, d_num_selected_out, num_items,
                stream, debug_synchronous
            )
        );
    }

    template <
        typename InputIteratorT,
        typename OutputIteratorT,
        typename NumSelectedIteratorT,
        typename SelectOp
    >
    HIPCUB_RUNTIME_FUNCTION static
    hipError_t If(void *d_temp_storage,
                  size_t &temp_storage_bytes,
                  InputIteratorT d_in,
                  OutputIteratorT d_out,
                  NumSelectedIteratorT d_num_selected_out,
                  int num_items,
                  SelectOp select_op,
                  hipStream_t stream = 0,
                  bool debug_synchronous = false)
    {
        return hipCUDAErrorTohipError(
            ::cub::DeviceSelect::If(
                d_temp_storage, temp_storage_bytes,
                d_in, d_out, d_num_selected_out,
                num_items, select_op,
                stream, debug_synchronous
            )
        );
    }

    template <
        typename InputIteratorT,
        typename OutputIteratorT,
        typename NumSelectedIteratorT
    >
    HIPCUB_RUNTIME_FUNCTION static
    hipError_t Unique(void *d_temp_storage,
                      size_t &temp_storage_bytes,
                      InputIteratorT d_in,
                      OutputIteratorT d_out,
                      NumSelectedIteratorT d_num_selected_out,
                      int num_items,
                      hipStream_t stream = 0,
                      bool debug_synchronous = false)
    {
        return hipCUDAErrorTohipError(
            ::cub::DeviceSelect::Unique(
                d_temp_storage, temp_storage_bytes,
                d_in, d_out, d_num_selected_out, num_items,
                stream, debug_synchronous
            )
        );
    }
};

END_HIPCUB_NAMESPACE

#endif // HIPCUB_CUB_DEVICE_DEVICE_SELECT_HPP_
