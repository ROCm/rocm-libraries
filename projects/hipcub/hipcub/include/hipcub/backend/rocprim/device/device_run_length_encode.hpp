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

#ifndef HIPCUB_ROCPRIM_DEVICE_DEVICE_RUN_LENGTH_ENCODE_HPP_
#define HIPCUB_ROCPRIM_DEVICE_DEVICE_RUN_LENGTH_ENCODE_HPP_

#include "../../../config.hpp"
#include "../../../util_deprecated.hpp"

#include <rocprim/device/device_run_length_encode.hpp> // IWYU pragma: export

BEGIN_HIPCUB_NAMESPACE

class DeviceRunLengthEncode
{
public:
    template<typename InputIteratorT,
             typename UniqueOutputIteratorT,
             typename LengthsOutputIteratorT,
             typename NumRunsOutputIteratorT>
    HIPCUB_RUNTIME_FUNCTION static hipError_t Encode(void*                  d_temp_storage,
                                                     size_t&                temp_storage_bytes,
                                                     InputIteratorT         d_in,
                                                     UniqueOutputIteratorT  d_unique_out,
                                                     LengthsOutputIteratorT d_counts_out,
                                                     NumRunsOutputIteratorT d_num_runs_out,
                                                     int                    num_items,
                                                     hipStream_t            stream = 0)
    {
        return ::rocprim::run_length_encode(d_temp_storage,
                                            temp_storage_bytes,
                                            d_in,
                                            num_items,
                                            d_unique_out,
                                            d_counts_out,
                                            d_num_runs_out,
                                            stream,
                                            HIPCUB_DETAIL_DEBUG_SYNC_VALUE);
    }

    template<typename InputIteratorT,
             typename UniqueOutputIteratorT,
             typename LengthsOutputIteratorT,
             typename NumRunsOutputIteratorT>
    HIPCUB_DETAIL_DEPRECATED_DEBUG_SYNCHRONOUS HIPCUB_RUNTIME_FUNCTION static hipError_t
        Encode(void*                  d_temp_storage,
               size_t&                temp_storage_bytes,
               InputIteratorT         d_in,
               UniqueOutputIteratorT  d_unique_out,
               LengthsOutputIteratorT d_counts_out,
               NumRunsOutputIteratorT d_num_runs_out,
               int                    num_items,
               hipStream_t            stream,
               bool                   debug_synchronous)
    {
        HIPCUB_DETAIL_RUNTIME_LOG_DEBUG_SYNCHRONOUS();
        return Encode(d_temp_storage,
                      temp_storage_bytes,
                      d_in,
                      d_unique_out,
                      d_counts_out,
                      d_num_runs_out,
                      num_items,
                      stream);
    }

    template<typename InputIteratorT,
             typename OffsetsOutputIteratorT,
             typename LengthsOutputIteratorT,
             typename NumRunsOutputIteratorT>
    HIPCUB_RUNTIME_FUNCTION static hipError_t NonTrivialRuns(void*          d_temp_storage,
                                                             size_t&        temp_storage_bytes,
                                                             InputIteratorT d_in,
                                                             OffsetsOutputIteratorT d_offsets_out,
                                                             LengthsOutputIteratorT d_lengths_out,
                                                             NumRunsOutputIteratorT d_num_runs_out,
                                                             int                    num_items,
                                                             hipStream_t            stream = 0)
    {
        return ::rocprim::run_length_encode_non_trivial_runs(d_temp_storage,
                                                             temp_storage_bytes,
                                                             d_in,
                                                             num_items,
                                                             d_offsets_out,
                                                             d_lengths_out,
                                                             d_num_runs_out,
                                                             stream,
                                                             HIPCUB_DETAIL_DEBUG_SYNC_VALUE);
    }

    template<typename InputIteratorT,
             typename OffsetsOutputIteratorT,
             typename LengthsOutputIteratorT,
             typename NumRunsOutputIteratorT>
    HIPCUB_DETAIL_DEPRECATED_DEBUG_SYNCHRONOUS HIPCUB_RUNTIME_FUNCTION static hipError_t
        NonTrivialRuns(void*                  d_temp_storage,
                       size_t&                temp_storage_bytes,
                       InputIteratorT         d_in,
                       OffsetsOutputIteratorT d_offsets_out,
                       LengthsOutputIteratorT d_lengths_out,
                       NumRunsOutputIteratorT d_num_runs_out,
                       int                    num_items,
                       hipStream_t            stream,
                       bool                   debug_synchronous)
    {
        HIPCUB_DETAIL_RUNTIME_LOG_DEBUG_SYNCHRONOUS();
        return NonTrivialRuns(d_temp_storage,
                              temp_storage_bytes,
                              d_in,
                              d_offsets_out,
                              d_lengths_out,
                              d_num_runs_out,
                              num_items,
                              stream);
    }
};

END_HIPCUB_NAMESPACE

#endif // HIPCUB_ROCPRIM_DEVICE_DEVICE_RUN_LENGTH_ENCODE_HPP_
