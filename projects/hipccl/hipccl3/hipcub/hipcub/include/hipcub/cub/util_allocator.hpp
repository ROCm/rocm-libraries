/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
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

#ifndef HIPCUB_CUB_UTIL_ALLOCATOR_HPP_
#define HIPCUB_CUB_UTIL_ALLOCATOR_HPP_

#if defined(HIPCUB_STDERR) && !defined(CUB_STDERR)
    #define CUB_STDERR
#endif

#include "../config.hpp"

#include <cub/util_allocator.cuh>

BEGIN_HIPCUB_NAMESPACE

struct CachingDeviceAllocator : public ::cub::CachingDeviceAllocator
{
    hipError_t SetMaxCachedBytes(
        size_t max_cached_bytes)
    {
        return hipCUDAErrorTohipError(
            ::cub::CachingDeviceAllocator::SetMaxCachedBytes(max_cached_bytes)
        );
    }

    hipError_t DeviceAllocate(
        int             device,
        void            **d_ptr,
        size_t          bytes,
        hipStream_t     active_stream = 0)
    {
        return hipCUDAErrorTohipError(
            ::cub::CachingDeviceAllocator::DeviceAllocate(device, d_ptr, bytes, active_stream)
        );
    }

    hipError_t DeviceAllocate(
        void            **d_ptr,
        size_t          bytes,
        hipStream_t     active_stream = 0)
    {
        return hipCUDAErrorTohipError(
            ::cub::CachingDeviceAllocator::DeviceAllocate(d_ptr, bytes, active_stream)
        );
    }

    hipError_t DeviceFree(
        int             device,
        void*           d_ptr)
    {
        return hipCUDAErrorTohipError(
            ::cub::CachingDeviceAllocator::DeviceFree(device, d_ptr)
        );
    }

    hipError_t DeviceFree(
        void*           d_ptr)
    {
        return hipCUDAErrorTohipError(
            ::cub::CachingDeviceAllocator::DeviceFree(d_ptr)
        );
    }

    hipError_t FreeAllCached()
    {
        return hipCUDAErrorTohipError(
            ::cub::CachingDeviceAllocator::FreeAllCached()
        );
    }
};

END_HIPCUB_NAMESPACE

#endif // HIPCUB_CUB_UTIL_ALLOCATOR_HPP_
