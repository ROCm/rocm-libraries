/******************************************************************************
 * Copyright (c) 2011-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2024-2025, Advanced Micro Devices, Inc.  All rights reserved.
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

#ifndef HIPCUB_CUB_DEVICE_DEVICE_COPY_HPP_
#define HIPCUB_CUB_DEVICE_DEVICE_COPY_HPP_

#include "../../../config.hpp"

#include <cub/device/device_copy.cuh> // IWYU pragma: export

#include <hip/hip_runtime.h>

BEGIN_HIPCUB_NAMESPACE

struct DeviceCopy
{
    template<typename InputBufferIt, typename OutputBufferIt, typename BufferSizeIteratorT>
    static hipError_t Batched(void*               d_temp_storage,
                              size_t&             temp_storage_bytes,
                              InputBufferIt       input_buffer_it,
                              OutputBufferIt      output_buffer_it,
                              BufferSizeIteratorT buffer_sizes,
                              uint32_t            num_buffers,
                              hipStream_t         stream = 0)
    {
        return hipCUDAErrorTohipError(::cub::DeviceCopy::Batched(d_temp_storage,
                                                                 temp_storage_bytes,
                                                                 input_buffer_it,
                                                                 output_buffer_it,
                                                                 buffer_sizes,
                                                                 num_buffers,
                                                                 stream));
    }
};

END_HIPCUB_NAMESPACE

#endif // HIPCUB_CUB_DEVICE_DEVICE_COPY_HPP_
