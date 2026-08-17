/******************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2025, Advanced Micro Devices, Inc.  All rights reserved.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#ifndef HIPCUB_ROCPRIM_DEVICE_DEVICE_MERGE_HPP_
#define HIPCUB_ROCPRIM_DEVICE_DEVICE_MERGE_HPP_

#include "../../../config.hpp"

#include <rocprim/device/device_merge.hpp> // IWYU pragma: export

BEGIN_HIPCUB_NAMESPACE

struct DeviceMerge
{

    template<typename KeyIteratorIn1,
             typename KeyIteratorIn2,
             typename KeyIteratorOut,
             typename CompareOp = ::rocprim::less<>>
    HIPCUB_RUNTIME_FUNCTION
    static hipError_t MergeKeys(void*          d_temp_storage,
                                std::size_t&   temp_storage_bytes,
                                KeyIteratorIn1 keys_in1,
                                int            num_keys1,
                                KeyIteratorIn2 keys_in2,
                                int            num_keys2,
                                KeyIteratorOut keys_out,
                                CompareOp      compare_op = {},
                                hipStream_t    stream     = 0)

    {
        return ::rocprim::merge(d_temp_storage,
                                temp_storage_bytes,
                                keys_in1,
                                keys_in2,
                                keys_out,
                                num_keys1,
                                num_keys2,
                                compare_op,
                                stream,
                                HIPCUB_DETAIL_DEBUG_SYNC_VALUE);
    }

    template<typename KeyIteratorIn1,
             typename ValueIteratorIn1,
             typename KeyIteratorIn2,
             typename ValueIteratorIn2,
             typename KeyIteratorOut,
             typename ValueIteratorOut,
             typename CompareOp = ::rocprim::less<>>
    HIPCUB_RUNTIME_FUNCTION
    static hipError_t MergePairs(void*            d_temp_storage,
                                 std::size_t&     temp_storage_bytes,
                                 KeyIteratorIn1   keys_in1,
                                 ValueIteratorIn1 values_in1,
                                 int              num_keys1,
                                 KeyIteratorIn2   keys_in2,
                                 ValueIteratorIn2 values_in2,
                                 int              num_keys2,
                                 KeyIteratorOut   keys_out,
                                 ValueIteratorOut values_out,
                                 CompareOp        compare_op = {},
                                 hipStream_t      stream     = 0)

    {
        return ::rocprim::merge(d_temp_storage,
                                temp_storage_bytes,
                                keys_in1,
                                keys_in2,
                                keys_out,
                                values_in1,
                                values_in2,
                                values_out,
                                num_keys1,
                                num_keys2,
                                compare_op,
                                stream,
                                HIPCUB_DETAIL_DEBUG_SYNC_VALUE);
    }
};

END_HIPCUB_NAMESPACE

#endif // HIPCUB_ROCPRIM_DEVICE_DEVICE_MERGE_HPP_
