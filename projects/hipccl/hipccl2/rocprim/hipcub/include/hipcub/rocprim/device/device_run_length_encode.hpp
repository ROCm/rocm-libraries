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

#ifndef HIPCUB_ROCPRIM_DEVICE_DEVICE_RUN_LENGTH_ENCODE_HPP_
#define HIPCUB_ROCPRIM_DEVICE_DEVICE_RUN_LENGTH_ENCODE_HPP_

#include "../../config.hpp"

BEGIN_HIPCUB_NAMESPACE

class DeviceRunLengthEncode
{
public:
    template<
        typename InputIteratorT,
        typename UniqueOutputIteratorT,
        typename LengthsOutputIteratorT,
        typename NumRunsOutputIteratorT
    >
    HIPCUB_RUNTIME_FUNCTION static
    hipError_t Encode(void * d_temp_storage,
                      size_t& temp_storage_bytes,
                      InputIteratorT d_in,
                      UniqueOutputIteratorT d_unique_out,
                      LengthsOutputIteratorT d_counts_out,
                      NumRunsOutputIteratorT d_num_runs_out,
                      int num_items,
                      hipStream_t stream = 0,
                      bool debug_synchronous = false)
    {
        return ::rocprim::run_length_encode(
            d_temp_storage, temp_storage_bytes,
            d_in, num_items,
            d_unique_out, d_counts_out, d_num_runs_out,
            stream, debug_synchronous
        );
    }

    template<
        typename InputIteratorT,
        typename OffsetsOutputIteratorT,
        typename LengthsOutputIteratorT,
        typename NumRunsOutputIteratorT
    >
    HIPCUB_RUNTIME_FUNCTION static
    hipError_t NonTrivialRuns(void * d_temp_storage,
                              size_t& temp_storage_bytes,
                              InputIteratorT d_in,
                              OffsetsOutputIteratorT d_offsets_out,
                              LengthsOutputIteratorT d_lengths_out,
                              NumRunsOutputIteratorT d_num_runs_out,
                              int num_items,
                              hipStream_t stream = 0,
                              bool debug_synchronous = false)
    {
        return ::rocprim::run_length_encode_non_trivial_runs(
            d_temp_storage, temp_storage_bytes,
            d_in, num_items,
            d_offsets_out, d_lengths_out, d_num_runs_out,
            stream, debug_synchronous
        );
    }
};

END_HIPCUB_NAMESPACE

#endif // HIPCUB_ROCPRIM_DEVICE_DEVICE_RUN_LENGTH_ENCODE_HPP_
